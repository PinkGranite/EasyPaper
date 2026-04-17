# AGENTS

Repository-level instructions for coding agents working on EasyPaper.

## Project Scope

- EasyPaper is a metadata-to-paper generation system with Python SDK (primary usage) and optional FastAPI API.
- This repository also publishes a Claude Code plugin marketplace.
- The installable plugin root is `plugins/easypaper/`.

## Plugin/Marketplace Layout

- Marketplace manifest: `.claude-plugin/marketplace.json`
- Plugin manifest: `plugins/easypaper/.claude-plugin/plugin.json`
- Plugin commands: `plugins/easypaper/commands/`
- Plugin skills: `plugins/easypaper/skills/`
- OpenCode/OpenClaw config: `.opencode/opencode.json`

## EasyPaper Workflow

For end-to-end paper generation, use the Python SDK directly:

1. **Use the `paper-from-metadata` skill** (in `plugins/easypaper/skills/paper-from-metadata/SKILL.md`):
   - Check if user has complete metadata (file or JSON)
   - If missing, collect interactively (title, idea_hypothesis, method, data, experiments, references)
   - Generate paper using EasyPaper SDK:
     ```python
     from easypaper import EasyPaper, PaperMetaData
     ep = EasyPaper(config_path="configs/openrouter.yaml")
     result = await ep.generate(metadata, **options)
     ```

2. **For Claude Code plugin usage**:
   - Use `/easypaper` command which handles environment setup and metadata collection automatically
   - No API server needed - uses Python SDK directly

3. **Optional FastAPI server** (for external integrations):
   - `uv run uvicorn src.main:app --reload --port 8000`
   - Endpoints: `POST /metadata/generate`, `POST /metadata/generate/section`

## Required Metadata Fields

- `title`
- `idea_hypothesis`
- `method`
- `data`
- `experiments`
- `references`

Optional fields include `style_guide`, `target_pages`, `template_path`, `compile_pdf`, and review options.

## Skills Source of Truth

- Backend YAML skills remain under `skills/` and are loaded by Python service config.
- Claude/OpenCode skill prompts live under `plugins/easypaper/skills/*/SKILL.md`.
- Main skills:
 - `paper-from-metadata`: Unified skill for metadata collection and paper generation
 - `interactive-metadata-build`: Claude-driven, conversational build of `PaperMetaData` from a research-materials folder (slash command `/easypaper-metadata-build`); complementary to the SDK one-shot `generate_metadata_from_folder` path
 - `setup-environment`: Automatic environment setup (Python, LaTeX)
 - `venue-selection`: Venue-specific formatting
 - `academic-writing-rules`: Academic writing conventions

## Validation Checklist

- Keep marketplace `source` pointing to `./plugins/easypaper`.
- Keep plugin version in `plugins/easypaper/.claude-plugin/plugin.json`.
- Keep README installation steps aligned with actual marketplace command syntax.
