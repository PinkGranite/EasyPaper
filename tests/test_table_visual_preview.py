"""
Tests for table visual preview pipeline and metadata fallback.
"""
import json
from pathlib import Path


def _load_meta_tables() -> list:
    """
    Load the metadata tables from the BLIP-2 sample meta file.
    - **Description**:
        - Reads the sample metadata JSON used by the user scenario.
        - Returns the list under the "tables" key.

    - **Args**:
        - None.

    - **Returns**:
        - `tables` (list): Table dictionaries from metadata.
    """
    worktree_root = Path(__file__).resolve().parents[1]
    primary_path = (
        worktree_root
        / "experiments"
        / "ai_track"
        / "metadatas"
        / "3f5b31c4f7350dc88002c121aecbdc82f86eb5bb"
        / "meta.json"
    )
    # In worktree mode, this metadata may only exist in the main workspace root.
    fallback_path = (
        Path(__file__).resolve().parents[3]
        / "experiments"
        / "ai_track"
        / "metadatas"
        / "3f5b31c4f7350dc88002c121aecbdc82f86eb5bb"
        / "meta.json"
    )
    meta_path = primary_path if primary_path.exists() else fallback_path
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    return payload.get("tables", [])


def test_read_table_content_falls_back_to_inline_when_file_missing(tmp_path):
    """
    _read_table_content should fallback to inline content when file_path is missing.
    """
    from src.agents.metadata_agent.models import TableSpec
    from src.agents.shared.table_converter import _read_table_content

    tables = _load_meta_tables()
    sample = dict(tables[0])
    # Keep file_path in metadata, but point base_path to a location where file does not exist.
    table = TableSpec(**sample)

    content = _read_table_content(table, base_path=str(tmp_path))
    assert content == sample["content"]


def test_build_table_preview_documents_from_meta_tables():
    """
    build_table_preview_documents should generate standalone preview tex for each table.
    """
    from src.agents.metadata_agent.models import TableSpec
    from src.agents.shared.table_converter import build_table_preview_documents

    table_dicts = _load_meta_tables()
    tables = [TableSpec(**item) for item in table_dicts]

    converted_tables = {}
    for table in tables:
        converted_tables[table.id] = (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            f"\\caption{{{table.caption}}}\n"
            f"\\label{{{table.id}}}\n"
            "\\begin{tabular}{lc}\n"
            "\\toprule\n"
            "A & B \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\end{table}"
        )

    previews = build_table_preview_documents(
        tables=tables,
        converted_tables=converted_tables,
        column_format="double",
    )
    assert len(previews) == len(tables)
    assert "tab:zero_shot_overview" in previews
    assert "\\documentclass[twocolumn]{article}" in previews["tab:zero_shot_overview"]
    assert "\\label{tab:zero_shot_overview}" in previews["tab:zero_shot_overview"]
