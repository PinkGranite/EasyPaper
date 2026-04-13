"""
Asset curator: LLM-guided selection (D) plus dedupe / stable ordering (E).

When ``max_figures`` and/or ``max_tables`` are set, calls an LLM to pick the
minimum set of assets that support the synthesized prose, never exceeding
the caps. Dimensions without a cap keep the deduplicated full list.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from ..models import FigureSpec, PaperMetaData, TableSpec

logger = logging.getLogger(__name__)

CURATOR_SYSTEM_PROMPT = (
    "You are curating figures and tables for ONE academic paper from a large "
    "research artifact folder.\n"
    "You will receive:\n"
    "- The paper title and four prose fields (idea, method, data, experiments).\n"
    "- Candidate figures and tables with ids, relative paths, and short captions.\n\n"
    "Rules:\n"
    "- Select AS FEW assets as needed to support the narrative. Do NOT pad counts.\n"
    "- You may select ZERO figures and/or ZERO tables if none add value.\n"
    "- Never select an id that is not in the candidate list.\n"
    "- Remove redundancy: do not keep two assets that convey the same result.\n"
    "- Respect hard caps: at most MAX_FIGURES figure ids and MAX_TABLES table ids "
    "(if a cap is null, ignore that limit for counting — the user message will say "
    "unlimited for that axis).\n\n"
    "Return ONLY valid JSON with exactly these keys:\n"
    '- "selected_figure_ids": list of strings (subset of candidate figure ids)\n'
    '- "selected_table_ids": list of strings (subset of candidate table ids)\n'
    '- "rationale": object mapping id -> one-sentence string (optional but encouraged)\n'
)


def dedupe_figures(figures: List[FigureSpec]) -> List[FigureSpec]:
    """
    Drop duplicate figure entries by ``file_path`` (or by ``id`` if path missing).

    - **Args**:
        - `figures` (List[FigureSpec]): Raw figure list.

    - **Returns**:
        - `List[FigureSpec]`: Deduplicated list preserving first occurrence order.
    """
    seen: Set[str] = set()
    out: List[FigureSpec] = []
    for fig in figures:
        key = fig.file_path or fig.id
        if key in seen:
            continue
        seen.add(key)
        out.append(fig)
    return out


def dedupe_tables(tables: List[TableSpec]) -> List[TableSpec]:
    """
    Drop duplicate table entries by ``file_path`` when present, else by ``id``.

    - **Args**:
        - `tables` (List[TableSpec]): Raw table list.

    - **Returns**:
        - `List[TableSpec]`: Deduplicated list preserving first occurrence order.
    """
    seen: Set[str] = set()
    out: List[TableSpec] = []
    for tbl in tables:
        key = tbl.file_path or tbl.id
        if key in seen:
            continue
        seen.add(key)
        out.append(tbl)
    return out


def _score_asset_path(rel: str) -> int:
    """Heuristic priority for rule-based pre-truncation / fallback."""
    s = rel.lower()
    score = 0
    for kw in (
        "summary", "report", "main", "aggregate", "overview",
        "cooperation", "payoff", "reputation", "accuracy", "distribution",
        "hypothesis", "environment", "agent_state",
    ):
        if kw in s:
            score += 6
    if "/charts/" in s or "\\charts\\" in s:
        score += 4
    if "/assets/" in s or "\\assets\\" in s:
        score += 2
    return score


def _pre_rank_for_llm(
    items: List[Tuple[str, str, str]],
    budget: int,
) -> List[Tuple[str, str, str]]:
    """
    Keep top ``budget`` candidates by path score for LLM context limits.

    - **Args**:
        - `items` (list): Tuples ``(id, rel_path, caption)``.
        - `budget` (int): Max candidates to return.

    - **Returns**:
        - Sorted subset.
    """
    if len(items) <= budget:
        return sorted(items, key=lambda x: (-_score_asset_path(x[1]), x[1]))
    ranked = sorted(items, key=lambda x: (-_score_asset_path(x[1]), x[1]))
    return ranked[:budget]


def rule_fallback_select(
    figures: List[FigureSpec],
    tables: List[TableSpec],
    *,
    max_figures: Optional[int],
    max_tables: Optional[int],
) -> Tuple[List[FigureSpec], List[TableSpec]]:
    """
    Select assets by heuristic scores when LLM curation is unavailable.

    - **Args**:
        - `figures` (List[FigureSpec]): Deduped figures.
        - `tables` (List[TableSpec]): Deduped tables.
        - `max_figures` (int, optional): Hard cap; ``None`` means keep all.
        - `max_tables` (int, optional): Hard cap; ``None`` means keep all.

    - **Returns**:
        - `Tuple[List[FigureSpec], List[TableSpec]]`: Selected lists.
    """
    fig_scored = sorted(
        figures,
        key=lambda f: (-_score_asset_path(f.file_path or ""), f.file_path or f.id),
    )
    tab_scored = sorted(
        tables,
        key=lambda t: (-_score_asset_path(t.file_path or ""), t.file_path or t.id),
    )
    out_f = fig_scored if max_figures is None else fig_scored[: max(0, max_figures)]
    out_t = tab_scored if max_tables is None else tab_scored[: max(0, max_tables)]
    return out_f, out_t


def _pick_by_ids(
    ordered_ids: List[str],
    by_id: Dict[str, Any],
    cap: Optional[int],
) -> List[Any]:
    """
    Map ordered ids to objects, enforce optional hard cap, skip unknown ids.

    - **Args**:
        - `ordered_ids` (list): Ids in desired output order.
        - `by_id` (dict): Id -> spec object.
        - `cap` (int, optional): Maximum number of items to keep.

    - **Returns**:
        - `list`: Selected objects.
    """
    seen: Set[str] = set()
    out: List[Any] = []
    for i in ordered_ids:
        if not isinstance(i, str) or i not in by_id or i in seen:
            continue
        seen.add(i)
        out.append(by_id[i])
        if cap is not None and len(out) >= cap:
            break
    return out


async def curate_paper_assets(
    metadata: PaperMetaData,
    llm_client: Any,
    model_name: str,
    *,
    max_figures: Optional[int],
    max_tables: Optional[int],
    llm_candidate_budget_figures: int = 80,
    llm_candidate_budget_tables: int = 60,
) -> PaperMetaData:
    """
    Apply D (LLM selection under caps) and E (dedupe) to ``metadata.figures/tables``.

    - **Args**:
        - `metadata` (PaperMetaData): Metadata after prose synthesis.
        - `llm_client`: OpenAI-compatible client.
        - `model_name` (str): Chat model name.
        - `max_figures` (int, optional): Hard upper bound on figure count.
        - `max_tables` (int, optional): Hard upper bound on table count.
        - `llm_candidate_budget_*` (int): Max candidates sent to the LLM per axis.

    - **Returns**:
        - `PaperMetaData`: Same object with trimmed ``figures`` / ``tables``.
    """
    figures = dedupe_figures(list(metadata.figures))
    tables = dedupe_tables(list(metadata.tables))

    trim_figures = max_figures is not None and len(figures) > max_figures
    trim_tables = max_tables is not None and len(tables) > max_tables

    if not trim_figures and not trim_tables:
        metadata.figures = figures
        metadata.tables = tables
        return metadata

    fig_items = [(f.id, f.file_path or "", f.caption) for f in figures if f.file_path or f.id]
    tab_items = [(t.id, t.file_path or "", t.caption) for t in tables if t.file_path or t.id]

    fig_for_llm = _pre_rank_for_llm(fig_items, llm_candidate_budget_figures)
    tab_for_llm = _pre_rank_for_llm(tab_items, llm_candidate_budget_tables)

    user_payload = {
        "title": metadata.title,
        "idea_hypothesis": metadata.idea_hypothesis,
        "method": metadata.method,
        "data": metadata.data,
        "experiments": metadata.experiments,
        "caps": {"max_figures": max_figures, "max_tables": max_tables},
        "trim_figures": trim_figures,
        "trim_tables": trim_tables,
        "candidate_figures": [
            {"id": fid, "path": rel, "caption": cap}
            for fid, rel, cap in fig_for_llm
        ],
        "candidate_tables": [
            {"id": tid, "path": rel, "caption": cap}
            for tid, rel, cap in tab_for_llm
        ],
    }

    caps_line = (
        "If trim_figures is false, set selected_figure_ids to ALL candidate_figures ids "
        "in stable order. If trim_figures is true, pick the minimum set needed (<= max_figures). "
        "If trim_tables is false, set selected_table_ids to ALL candidate_tables ids. "
        "If trim_tables is true, pick the minimum set needed (<= max_tables). "
        "Never exceed caps when trim is true."
    )

    sel_f: List[str] = []
    sel_t: List[str] = []

    try:
        response = await llm_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": CURATOR_SYSTEM_PROMPT + "\n" + caps_line},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        parsed = json.loads(raw)
        sel_f = parsed.get("selected_figure_ids") or []
        sel_t = parsed.get("selected_table_ids") or []
        if not isinstance(sel_f, list):
            sel_f = []
        if not isinstance(sel_t, list):
            sel_t = []
    except Exception as e:
        logger.warning("LLM asset curation failed, using rule fallback: %s", e)

    fig_by_id = {f.id: f for f in figures}
    tab_by_id = {t.id: t for t in tables}

    if trim_figures:
        new_f = _pick_by_ids(sel_f, fig_by_id, max_figures)
        if not new_f:
            new_f, _ = rule_fallback_select(figures, [], max_figures=max_figures, max_tables=None)
    else:
        new_f = figures

    if trim_tables:
        new_t = _pick_by_ids(sel_t, tab_by_id, max_tables)
        if not new_t:
            _, new_t = rule_fallback_select([], tables, max_figures=None, max_tables=max_tables)
    else:
        new_t = tables

    metadata.figures = new_f
    metadata.tables = new_t
    return metadata
