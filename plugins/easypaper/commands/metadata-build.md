Interactively build EasyPaper PaperMetaData from a research-materials folder, using Claude Code's own file-investigation tools and other installed skills (`paperhub-*`, `pdf`, `exa-search`, `sequential-thinking`, `deep-research`). Output is a JSON file consumable by `/paper-from-metadata`.

Use this command when the user wants Claude to **co-author** the metadata interactively, instead of running the SDK one-shot pipeline `ep.generate_metadata_from_folder(...)`.

## Execution contract

1. **FIRST**, present the three modes to the user and wait for an explicit choice (do not auto-select; mark `warm-start` as recommended):
   - `[1] cold` — Claude walks the folder alone, no Python SDK call. Best for small folders or full transparency.
   - `[2] warm-start` (recommended) — call `ep.generate_metadata_from_folder(materials_root, max_figures=12, max_tables=12, vision_enrich_figures=False)` once, then walk the user through every field for review/edit.
   - `[3] refine` — load an existing `metadata.json`, run `build_eval`, walk the user through fixing failures and warnings.
2. After the mode is confirmed, ask for the corresponding input:
   - `cold` / `warm-start` → ask for `materials_root` (absolute path); verify it is an existing directory.
   - `refine` → ask for the existing `metadata.json` path; verify it exists.
3. Hand off to the `interactive-metadata-build` skill — it owns the 6-phase workflow (Mode Selection → Discovery → Field Drafting → Asset Selection → References → Validate & Save → Handoff).
4. Validate the in-memory metadata using `experiments/universal_metadata_generator/run_metadata_e2e_eval.py:build_eval` before saving. Do not write to disk if `pass_core` is `False`.
5. Save to `<materials_root>/easypaper_metadata.json` (default) and tell the user to run `/paper-from-metadata` next.

## Output rules (must match SDK schema)

- `materials_root` is absolute; `figures[].file_path` and `tables[].file_path` are **relative POSIX** under `materials_root` (this differs from `/paper-from-metadata` which expects absolute paths for hand-written metadata).
- Figure IDs use `f"fig:h{sha256(rel.lower()).hexdigest()[:12]}"`.
- Table IDs use `f"tab:h{sha256(rel).hexdigest()[:12]}"` (no lowercasing).
- Field names in prose are exactly: `title, idea_hypothesis, method, data, experiments`.

## Fallbacks

- `easypaper` package not installed → use `setup-environment` skill first.
- `warm-start` SDK call fails → fall back to `cold` mode and inform the user.
- Validation fails → loop back to the failing phase; never save a failing metadata.

$ARGUMENTS
