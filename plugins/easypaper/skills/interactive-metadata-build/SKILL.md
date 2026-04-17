---
description: Interactively build EasyPaper PaperMetaData from a research-materials folder using Claude Code's own file-investigation tools and other installed skills. Output is a JSON file consumable by the `paper-from-metadata` skill.
---

Use this skill when the user has a folder of research materials (code, data, PDFs, images, notes, BibTeX) and wants Claude to **co-author** the metadata interactively, instead of running the one-shot SDK pipeline `ep.generate_metadata_from_folder(...)`.

The two paths are complementary, not redundant:

| Path | Driver | Interaction | Use when |
|------|--------|-------------|----------|
| SDK one-shot (`generate_metadata_from_folder`) | EasyPaper internal LLM | None | Batch / CI / regression eval (`run_metadata_e2e_eval.py`) |
| **This skill** | Claude Code (multi-turn) | High | Single high-value paper, atypical folder, ambiguity that needs user input |

Both paths must produce the **same `PaperMetaData` JSON shape** so the downstream `paper-from-metadata` skill can consume either output unchanged.

## Output contract

The final JSON object MUST match the `PaperMetaData` schema used by EasyPaper's SDK. Use the existing artifact at `experiments/universal_metadata_generator/reputation_game_metadata_e2e_*.json` as the canonical example. Required top-level keys:

- `title` (string, non-empty)
- `idea_hypothesis` (string, non-empty)
- `method` (string, non-empty)
- `data` (string, non-empty)
- `experiments` (string, non-empty)
- `references` (array; warning-only if empty)
- `materials_root` (absolute path string)
- `figures` (array of figure objects)
- `tables` (array of table objects)
- `template_path` / `style_guide` / `target_pages` (optional, may be `null`)

Each figure / table object follows the SDK shape:

```json
{
  "id": "fig:h<12hex>",
  "caption": "...",
  "description": "...",
  "section": "",
  "file_path": "relative/posix/path/to/asset.png",
  "wide": false,
  "auto_generate": false,
  "generation_prompt": null
}
```

Tables use the same shape with `id` prefix `tab:h`.

## Path-handling rule (different from `paper-from-metadata`!)

`paper-from-metadata` requires **absolute** paths because it accepts hand-written metadata. This skill writes folder-derived metadata, so it follows the SDK convention instead:

- `materials_root` → **absolute** path (resolved from user input).
- `figures[].file_path` and `tables[].file_path` → **relative POSIX** paths under `materials_root`. Downstream resolution is done via `materials_root + file_path`. Do NOT convert these to absolute paths.

This matches what `generate_metadata_from_folder` produces and keeps both paths interchangeable.

## ID generation (must match the SDK)

When operating in `cold` mode (Claude builds figure/table entries from scratch), IDs MUST be byte-identical to what the SDK would have produced for the same `materials_root`. The exact algorithm is in `src/agents/metadata_agent/metadata_generator/extractors/image_extractor.py:55` and `data_extractor.py:52`:

```python
import hashlib
from pathlib import Path

def figure_id(rel_posix_path: str) -> str:
    """
    Compute the SDK-compatible figure id for a file relative to materials_root.

    - **Description**:
        - Lowercases the relative POSIX path before hashing, matching
          `ImageExtractor._make_fragment` in the SDK.

    - **Args**:
        - `rel_posix_path` (str): Path relative to materials_root, POSIX form.

    - **Returns**:
        - `figure_id` (str): SDK-compatible id of the form `fig:h<12hex>`.
    """
    digest = hashlib.sha256(rel_posix_path.lower().encode("utf-8")).hexdigest()[:12]
    return f"fig:h{digest}"


def table_id(rel_posix_path: str) -> str:
    """
    Compute the SDK-compatible table id for a file relative to materials_root.

    - **Description**:
        - Hashes the relative POSIX path AS-IS (no lowercasing), matching
          `DataExtractor._stable_table_id` in the SDK.

    - **Args**:
        - `rel_posix_path` (str): Path relative to materials_root, POSIX form.

    - **Returns**:
        - `table_id` (str): SDK-compatible id of the form `tab:h<12hex>`.
    """
    digest = hashlib.sha256(rel_posix_path.encode("utf-8")).hexdigest()[:12]
    return f"tab:h{digest}"
```

Note the asymmetry: figures lowercase the path before hashing, tables do not. Preserve this exactly.

## Modes

There are three modes. **The very first thing this skill does is present these options to the user and wait for an explicit choice.** Do not proceed without one.

- **`cold`** — Claude alone walks the folder, drafts every field, and asks the user. No SDK call. Use only when the user explicitly wants no Python invocation, or when the materials are small and the user wants full transparency.
- **`warm-start` (recommended default)** — First call `ep.generate_metadata_from_folder(materials_root, max_figures=12, max_tables=12, vision_enrich_figures=False)` to obtain a draft `PaperMetaData`. Then walk the user through every field for review/edit. Best signal-to-effort ratio: SDK does the heavy lifting on file extraction and dedup, Claude does the judgment and conversation.
- **`refine`** — User already has a `metadata.json`. Load it, run `build_eval` (see Phase 5), surface failures/warnings, and walk the user through fixing each one.

## 6-Phase Workflow

### Phase 0 — Mode Selection (FIRST user interaction)

This is the **first** thing this skill does, before reading any file or asking for any path. Present the three modes verbatim and ask the user to pick one. Mark the recommended default but never auto-select.

Show this block (or an equivalent localized version if the user is conversing in another language) and wait for the user's reply:

```text
Choose how you want to build the metadata:

  [1] cold        — Claude walks the folder alone, no Python SDK call.
                    Best for small folders or when full transparency is required.

  [2] warm-start  — (recommended) Run ep.generate_metadata_from_folder() once
                    to get a draft, then refine each field with Claude.
                    Best signal-to-effort ratio.

  [3] refine      — Load an existing metadata.json and walk through fixing
                    failures and warnings reported by build_eval.

Reply with 1 / 2 / 3, or the mode name.
```

Only after the user answers, ask the next question, which depends on the mode:

- `cold` or `warm-start` → ask for `materials_root` (absolute path). Verify it exists and is a directory.
- `refine` → ask for the existing `metadata.json` path. Verify it exists.

If `warm-start`: invoke EasyPaper SDK once to obtain a draft. Persist the draft in memory; do NOT save yet.

```python
from easypaper import EasyPaper
from pathlib import Path

ep = EasyPaper(config_path=str(Path("configs/openrouter.yaml").resolve()))
draft = await ep.generate_metadata_from_folder(
    str(Path(materials_root).resolve()),
    max_figures=12,
    max_tables=12,
    vision_enrich_figures=False,
)
```

`vision_enrich_figures=False` here keeps cost predictable; the user can opt in later (Phase 3).

**Do not** proceed to Phase 1 until both the mode and the corresponding input path are confirmed.

### Phase 1 — Discovery

Use Claude Code's read-only tools (`Glob`, `Read`, `Grep`, `list_dir`) to build a mental map of the folder. Do NOT use any LLM call beyond what Claude itself runs.

1. List top-level entries; for each directory give a one-line purpose guess.
2. Auto-Read these high-signal files when present, top-level only:
   - `README*`, `*.md` at root
   - `pyproject.toml`, `requirements.txt`, `setup.py`
   - `config*.yaml|json|toml`, primary entrypoint script
   - All `*.bib`
3. Tag suspected sub-trees: `paper draft / experiments / data / figures / configs / refs`.
4. Output a concise "Folder Map" summary to the user before proceeding.

**Cross-skill delegation triggers** (invoke when the corresponding situation is detected):

| Detected | Delegate to |
|----------|-------------|
| `*.pdf` papers in folder | `pdf` skill, or `paperhub-read-paper` if PaperHub is configured |
| PaperHub workspace markers | `paperhub-search-paper`, `paperhub-extract-bib` |
| User asks "find related work" / "is this novel" | `exa-search` or `deep-research` |
| Methodological ambiguity needs reasoning | `sequential-thinking` |
| Cross-cutting code structure scan | `search-first` or `repo-scan` |

### Phase 2 — Field Drafting (per-field interaction)

For each of the five non-empty prose fields, in order: `title, idea_hypothesis, method, data, experiments`.

For each field:

1. Present a draft (from `warm-start` SDK output, or composed from Phase 1 in `cold` mode).
2. Cite specific evidence: `"based on README §2 and experiments/exp1/run.py:42"`.
3. Surface unresolved questions: `"I cannot tell whether this targets H1 or H4 — please confirm"`.
4. Apply user edits; if user says "I don't know", offer `sequential-thinking` for reasoning or `exa-search` for background.

The exact field names must match the SDK's `NONEMPTY_FIELDS` constant in `experiments/universal_metadata_generator/run_metadata_e2e_eval.py:39`. Any deviation will fail Phase 5 validation.

### Phase 3 — Asset Selection (figures and tables)

1. Glob all candidates:
   - Figures: `**/*.{png,jpg,jpeg,gif,svg,webp,bmp}`
   - Tables: `**/*.{csv,tsv}` (and any `*.tex` containing `\begin{table}`)
2. In `warm-start` mode, present the SDK's chosen subset and let the user accept / drop / add.
3. In `cold` mode, group candidates by relevance to the five fields, propose a subset of size ≤ `max_figures` (default 12) and ≤ `max_tables` (default 12), and have the user confirm.
4. For each retained asset:
   - Generate `id` using the exact algorithm above.
   - Set `file_path` to **relative POSIX** under `materials_root`.
   - Draft `caption` from the file stem (Title Case, underscores → spaces).
   - Draft `description` using whatever inspection Claude can do (read CSV header, summarize image filename + sibling caption file). Do NOT invent details about an image without evidence.
   - Optionally suggest `section` placement (e.g. `"Method"`, `"Experiments"`); leave empty string `""` if uncertain.
5. If the user explicitly wants vision-model enrichment for figure descriptions, recommend re-running `warm-start` with `vision_enrich_figures=True` and `max_vision_figures=N`, since Claude itself does not have multimodal vision in this skill context.

### Phase 4 — References

1. Parse every `*.bib` file under `materials_root` (Read + light regex on `@article`, `@inproceedings`, `@misc`, etc.). Append parsed entries to `references[]` as raw BibTeX strings, deduplicated by citation key.
2. If the user asks for additional references or coverage check, delegate to `exa-search` or `deep-research`. Only insert agreed-upon entries.
3. `build_eval` (Phase 5) treats empty `references[]` as a **warning**, not a failure (see `run_metadata_e2e_eval.py:92-97`). Surface the warning prominently but do not block the build.

### Phase 5 — Validate & Save

Reuse the existing validator. Import and run it against the in-memory metadata before writing anything to disk:

```python
import importlib.util
from pathlib import Path

spec_path = Path("experiments/universal_metadata_generator/run_metadata_e2e_eval.py").resolve()
spec = importlib.util.spec_from_file_location("run_metadata_e2e_eval", spec_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

report = mod.build_eval(metadata, max_figures=12, max_tables=12)
```

Inspect `report["checks"]`:

- `five_prose_nonempty` — must be `True`. If False, return to Phase 2 for the offending field.
- `materials_root_set` — must be `True`. Set `metadata.materials_root` to the absolute `materials_root`.
- `figure_paths_relative` / `table_paths_relative` — must be `True`. If False, fix the offending entries (do NOT use absolute paths here).
- `within_max_figures` / `within_max_tables` — must be `True`. If False, return to Phase 3 to drop assets.
- `figure_ids_prefixed` / `table_ids_prefixed` — must be `True`. If False, recompute IDs using the algorithm above.
- `has_references` — `False` is a warning only.

Only when `report["pass_core"]` is `True`, save to disk:

```python
import json
from pathlib import Path

out_path = Path(materials_root) / "easypaper_metadata.json"
out_path.write_text(
    json.dumps(metadata.model_dump(mode="json"), ensure_ascii=False, indent=2),
    encoding="utf-8",
)
```

Default save location: `<materials_root>/easypaper_metadata.json`. The user may override with an explicit path.

### Phase 6 — Handoff

Print a summary block:

- File written and its absolute path
- Eval report: `pass_core`, `pass_with_references`, list of warnings
- Two next-step options:
  - **SDK direct generation** —
    ```python
    from easypaper import EasyPaper
    ep = EasyPaper(config_path="configs/openrouter.yaml")
    result = await ep.generate(metadata, compile_pdf=True)
    ```
  - **Claude Code plugin** — `Run /paper-from-metadata and point it at <materials_root>/easypaper_metadata.json`.

## Best Practices

- **Stay in conversation.** This skill exists because the SDK path is silent. Every phase should produce a short, scannable user-facing block; do not dump full file contents.
- **Cite evidence.** Every field draft must point to specific files / line ranges Claude actually read.
- **Schema parity, always.** Never invent a key the SDK does not produce. When in doubt, diff against `reputation_game_metadata_e2e_*.json`.
- **No silent vision calls.** This skill does not invoke vision models on its own. Vision enrichment is opt-in via `warm-start` + `vision_enrich_figures=True`.
- **Idempotent re-runs.** Saving twice with the same materials must produce byte-identical figure/table IDs (the hash algorithm guarantees this). If IDs drift, the build is wrong.

## Error Handling

- Materials root missing → ask for correct absolute path; do not guess.
- `easypaper` package not importable → invoke the `setup-environment` skill first.
- `warm-start` SDK call fails → fall back to `cold` mode and inform the user.
- Validation fails after Phase 5 → loop back to the specific failing phase; never save a failing metadata.
- User aborts mid-phase → offer to save a partial draft to `<materials_root>/easypaper_metadata.draft.json` (clearly suffixed) so the next session can resume in `refine` mode.
