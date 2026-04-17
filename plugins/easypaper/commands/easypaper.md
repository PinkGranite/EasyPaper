Run the EasyPaper end-to-end paper generation workflow with guided setup and metadata collection.

## Execution contract

### Phase 1: Environment Setup (First-time only)

1. **Check if environment is set up**:
   - Check if `.easypaper-env` directory exists
   - Check if `easypaper` package is importable: `python -c "import easypaper"`
   - Check if `pdflatex` command is available

2. **If environment is not ready**:
   - Use the `setup-environment` skill to automatically:
     - Create isolated virtual environment (prefer `uv`, fallback to `venv`)
     - Install easypaper package
     - Check and guide LaTeX installation
     - Verify all components are working

### Phase 2: Paper Generation

3. **Determine metadata source** — ask user which applies:
   - **(A) Complete metadata file / JSON** → proceed to step 4.
   - **(B) Research materials folder, one-shot SDK pipeline** (code, data, images, notes, PDFs) → use `generate_metadata_from_folder` first:
     ```python
     metadata = await ep.generate_metadata_from_folder(
         str(Path("path/to/materials").resolve()),
         max_figures=12, max_tables=12,
         vision_enrich_figures=True,
     )
     result = await ep.generate(metadata, compile_pdf=True)
     ```
     Key options: `max_figures`, `max_tables`, `vision_enrich_figures` (default True),
     `vision_model`, `max_vision_figures`.
     See `skills/paper-from-metadata/SKILL.md` § "Alternative: Generate Metadata from a Materials Folder" for full parameter table and cost-control guidance.
   - **(B-interactive) Research materials folder, Claude-driven interactive build** → recommend `/easypaper-metadata-build` (or invoke the `interactive-metadata-build` skill directly). Claude walks the folder with its own Read/Glob/Grep tools, asks the user to confirm each field, and saves a `PaperMetaData` JSON that this command can then consume in path (A). Use this when the folder is non-typical, when the user wants to be in the loop, or when other skills (`paperhub-*`, `pdf`, `exa-search`, `sequential-thinking`) should help interpret the materials.
   - **(C) No metadata yet** → collect interactively (step 4).

4. **Use the `paper-from-metadata` skill** which handles:
   - **Check for existing metadata**: Ask user if they have complete metadata file/JSON
   - **Collect metadata if needed**: If missing or incomplete, interactively collect all required fields:
     - Required: `title`, `idea_hypothesis`, `method`, `data`, `experiments`, `references`
     - Optional: `style_guide`, `target_pages`, `template_path` (absolute path), `compile_pdf`, `enable_review`, `max_review_iterations`
     - Advanced: `figures` (with absolute file_path), `tables`, `code_repository` (with absolute path if local_dir), `output_dir` (absolute path)
   - **Path handling**: Ensure all paths are absolute - convert relative paths to absolute using `pathlib.Path.resolve()`
   - **Review and confirm**: Display summary, allow edits, save to file, get confirmation
   - **Generate paper**: Use EasyPaper Python SDK directly. Prefer parsing metadata files as `PaperGenerationRequest`, then convert with `to_metadata()` + `to_generate_options()`.
     ```python
     from easypaper import EasyPaper, PaperGenerationRequest
     from pathlib import Path
     
     # Config path should be absolute
     config_path = Path("configs/openrouter.yaml").resolve()
     ep = EasyPaper(config_path=str(config_path))
     # If user has metadata.json (e.g. examples/meta.json)
     request = PaperGenerationRequest.model_validate_json_file("metadata.json")
     metadata = request.to_metadata()
     options = request.to_generate_options()
     result = await ep.generate(metadata, **options)
     ```
   - **Report results**: Show status, output files, absolute paths, summary.
   - **Final PDF selection rule**:
     1. Use `result.pdf_path` first.
     2. If missing, search `result.output_path` in order:
        - `iteration_*_final/**/*.pdf`
        - latest `iteration_*` directory PDF
        - root `paper.pdf`
     3. If still missing, report final PDF unavailable and include compile error summary.

## Path Requirements

**IMPORTANT**: All paths in metadata must be absolute paths:
- `template_path`: Absolute path to LaTeX template file/directory
- `figures[].file_path`: Absolute paths to figure image files
- `code_repository.path`: Absolute path (if type is `local_dir`)
- `output_dir`: Absolute path to output directory
- `config_path`: Absolute path to EasyPaper config file

The skill will automatically convert relative paths to absolute paths, but users should be encouraged to provide absolute paths.

## User Experience Guidelines

- **First-time users**: Automatically trigger environment setup without asking
- **Clear progress**: Show what step you're on (e.g., "Step 1/2: Setting up environment...")
- **Error handling**: If any step fails, explain clearly and provide next steps
- **Flexibility**: Allow users to provide complete metadata or collect interactively
- **Path conversion**: Automatically convert relative paths to absolute and inform user
- **Reference**: When users ask about structure, reference `examples/meta.json` as the template (note: paths should be absolute)
- **Direct import**: Use EasyPaper as Python SDK - no API server needed for generation.
- **Typesetter path**: PDF compilation prefers in-process Typesetter (self-contained SDK) and falls back to HTTP Typesetter endpoint when peer is unavailable.

$ARGUMENTS
