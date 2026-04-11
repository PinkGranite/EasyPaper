"""
Tests for table visual preview pipeline and metadata fallback.
"""
import json
import shutil
from pathlib import Path

import pytest


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


def test_build_table_preview_documents_uses_source_content_when_unconverted():
    """
    build_table_preview_documents should build structural table from metadata content.
    """
    from src.agents.metadata_agent.models import TableSpec
    from src.agents.shared.table_converter import build_table_preview_documents

    sample = dict(_load_meta_tables()[0])
    table = TableSpec(**sample)
    previews = build_table_preview_documents(
        tables=[table],
        converted_tables={},
        column_format="double",
    )
    tex = previews[table.id]
    assert "Flamingo" in tex
    assert "BLIP-2" in tex
    assert "\\begin{tabular}{lcccc}" in tex
    assert "Column & Value" not in tex


def test_build_table_preview_documents_raises_without_source_or_conversion():
    """
    build_table_preview_documents should raise if no converted table and no source content.
    """
    from src.agents.metadata_agent.models import TableSpec
    from src.agents.shared.table_converter import build_table_preview_documents

    table = TableSpec(
        id="tab:missing",
        caption="Missing",
        file_path="tables/does_not_exist.csv",
        content=None,
    )
    with pytest.raises(ValueError, match="Missing converted table and source content"):
        build_table_preview_documents(
            tables=[table],
            converted_tables={},
        )


def test_build_table_preview_documents_reads_relative_file_with_base_path(tmp_path):
    """
    build_table_preview_documents should resolve relative file_path via base_path.
    """
    from src.agents.metadata_agent.models import TableSpec
    from src.agents.shared.table_converter import build_table_preview_documents

    tables_dir = tmp_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / "sample.csv"
    csv_path.write_text(
        "Model,Score\nBLIP-2,65.0\n",
        encoding="utf-8",
    )

    table = TableSpec(
        id="tab:file_source",
        caption="File Source",
        file_path="tables/sample.csv",
        content=None,
    )
    previews = build_table_preview_documents(
        tables=[table],
        converted_tables={},
        base_path=str(tmp_path),
    )
    tex = previews["tab:file_source"]
    assert "BLIP-2" in tex
    assert "65.0" in tex


def test_compile_table_preview_documents_success(monkeypatch, tmp_path):
    """
    compile_table_preview_documents should report success and output artifact paths.
    """
    from src.agents.shared import table_converter as tc

    class _Proc:
        def __init__(self, returncode=0):
            self.returncode = returncode
            self.stdout = ""
            self.stderr = ""

    def _fake_run(cmd, capture_output, text, encoding, errors, timeout, cwd):
        out_dir = cmd[cmd.index("-output-directory") + 1]
        tex_name = Path(cmd[-1]).stem
        pdf_path = Path(out_dir) / f"{tex_name}.pdf"
        log_path = Path(out_dir) / f"{tex_name}.log"
        pdf_path.write_bytes(b"%PDF-1.4 mock")
        log_path.write_text("LaTeX Warning: mock warning", encoding="utf-8")
        return _Proc(returncode=0)

    monkeypatch.setattr(tc.subprocess, "run", _fake_run)

    preview_docs = {
        "tab:demo": (
            "\\documentclass{article}\n"
            "\\begin{document}\n"
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\caption{Demo}\\label{tab:demo}\n"
            "\\begin{tabular}{lc}\nA & B \\\\\n\\end{tabular}\n"
            "\\end{table}\n"
            "\\end{document}\n"
        )
    }

    results = tc.compile_table_preview_documents(
        preview_docs=preview_docs,
        output_dir=str(tmp_path),
    )

    assert "tab:demo" in results
    assert results["tab:demo"]["success"] is True
    assert Path(results["tab:demo"]["pdf_path"]).exists()
    assert Path(results["tab:demo"]["log_path"]).exists()
    assert results["tab:demo"]["warnings"] == ["mock warning"]


def test_compile_table_preview_documents_handles_missing_pdflatex(monkeypatch, tmp_path):
    """
    compile_table_preview_documents should fail gracefully when pdflatex is unavailable.
    """
    from src.agents.shared import table_converter as tc

    def _raise_not_found(*args, **kwargs):
        raise FileNotFoundError("pdflatex not found")

    monkeypatch.setattr(tc.subprocess, "run", _raise_not_found)

    results = tc.compile_table_preview_documents(
        preview_docs={"tab:demo": "\\documentclass{article}\\begin{document}x\\end{document}"},
        output_dir=str(tmp_path),
    )

    assert results["tab:demo"]["success"] is False
    assert "pdflatex not found" in " ".join(results["tab:demo"]["errors"]).lower()


def test_compile_table_preview_documents_uses_unique_artifact_paths(monkeypatch, tmp_path):
    """
    compile_table_preview_documents should avoid artifact collisions for similar ids.
    """
    from src.agents.shared import table_converter as tc

    class _Proc:
        def __init__(self, returncode=0):
            self.returncode = returncode
            self.stdout = ""
            self.stderr = ""

    def _fake_run(cmd, capture_output, text, encoding, errors, timeout, cwd):
        out_dir = Path(cmd[cmd.index("-output-directory") + 1])
        tex_name = Path(cmd[-1]).stem
        (out_dir / f"{tex_name}.pdf").write_bytes(b"%PDF-1.4 mock")
        (out_dir / f"{tex_name}.log").write_text("", encoding="utf-8")
        return _Proc(returncode=0)

    monkeypatch.setattr(tc.subprocess, "run", _fake_run)
    docs = {
        "tab:a/b": "\\documentclass{article}\\begin{document}x\\end{document}",
        "tab:a:b": "\\documentclass{article}\\begin{document}y\\end{document}",
    }
    results = tc.compile_table_preview_documents(docs, output_dir=str(tmp_path))
    path1 = results["tab:a/b"]["pdf_path"]
    path2 = results["tab:a:b"]["pdf_path"]
    assert path1 != path2


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")
def test_compile_table_preview_from_meta_single_table(tmp_path):
    """
    compile_table_preview_documents should compile one metadata table in a real TeX run.
    """
    from src.agents.metadata_agent.models import TableSpec
    from src.agents.shared.table_converter import (
        build_table_preview_documents,
        compile_table_preview_documents,
    )

    sample = dict(_load_meta_tables()[0])
    table = TableSpec(**sample)
    previews = build_table_preview_documents(
        tables=[table],
        converted_tables={
            table.id: (
                "\\begin{table}[htbp]\n"
                "\\centering\n"
                f"\\caption{{{table.caption}}}\n"
                f"\\label{{{table.id}}}\n"
                "\\begin{tabular}{lcccc}\n"
                "\\toprule\n"
                "Model & Trainable & VQA & NoCaps & Flickr \\\\\n"
                "\\midrule\n"
                "BLIP-2 & 188M & 65.0 & 121.6 & 97.6 \\\\\n"
                "\\bottomrule\n"
                "\\end{tabular}\n"
                "\\end{table}"
            )
        },
        column_format="double",
    )

    results = compile_table_preview_documents(
        preview_docs=previews,
        output_dir=str(tmp_path),
    )
    assert results[table.id]["success"] is True
    assert Path(results[table.id]["pdf_path"]).exists()
