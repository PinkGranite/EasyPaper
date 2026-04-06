# EasyPaper

EasyPaper is a multi-agent academic paper generation system. It turns a small set of metadata
(title, idea, method, data, experiments, references) into a structured LaTeX paper and optionally
compiles it into a PDF through a typesetting agent.

EasyPaper can be used in two modes:

- **SDK mode** — `pip install -e .` and call from Python directly (no server needed)
- **Server mode** — `pip install -e ".[server]"` and run as a FastAPI service

## Features

- Multi-agent pipeline: planning, writing, review, typesetting, and optional VLM review
- Python SDK for in-process paper generation (`from easypaper import EasyPaper`)
- Optional FastAPI service with health and agent discovery endpoints
- Streaming progress via `generate_stream()` (SDK) or SSE (server)
- CLI scripts for metadata-driven generation and paper assembly demos
- LaTeX output with citation validation, figure/table injection, and review loop

## Requirements

- Python 3.11+
- LaTeX toolchain (`pdflatex` + `bibtex`) for PDF compilation
- [Poppler](https://poppler.freedesktop.org/) — required by `pdf2image` for PDF-to-image conversion
  - macOS: `brew install poppler`
  - Ubuntu/Debian: `apt install poppler-utils`
- Model API keys configured in YAML (see [Config](#config))

## Quickstart (SDK mode)

1. Install core dependencies:

```bash
pip install -e .
```

2. Copy the example config and fill in your API keys:

```bash
cp examples/config.example.yaml configs/dev.yaml
# Edit configs/dev.yaml — replace YOUR_API_KEY with real keys
```

3. Set the config path (or create a `.env` file):

```bash
export AGENT_CONFIG_PATH=./configs/dev.yaml
```

4. Use from Python:

```python
import asyncio
from easypaper import EasyPaper, PaperMetaData

async def main():
    ep = EasyPaper(config_path="configs/dev.yaml")
    result = await ep.generate(PaperMetaData(
        title="My Paper",
        idea_hypothesis="...",
        method="...",
        data="...",
        experiments="...",
    ))
    print(f"Status: {result.status}, Words: {result.total_word_count}")

asyncio.run(main())
```

5. Or use streaming for progress updates:

```python
async for event in ep.generate_stream(metadata):
    print(f"{event.get('phase', '')}: {event.get('message', '')}")
```

See [`examples/sdk_demo.py`](examples/sdk_demo.py) for a complete working example.

## Server Mode

To run as a FastAPI service (for external integrations):

1. Install with server extras:

```bash
pip install -e ".[server]"
```

2. Start the server:

```bash
uvicorn src.main:app --reload --port 8000
```

3. Verify health:

```bash
curl http://localhost:8000/healthz
```

### Generate a Paper via API

```bash
curl -X POST http://localhost:8000/metadata/generate \
  -H "Content-Type: application/json" \
  -d @economist_example/metadata.json
```

### Generate via CLI

```bash
python scripts/generate_paper.py --input economist_example/metadata.json
```

## Optional Dependencies

```bash
pip install -e ".[dev]"    # pytest, ipython, etc.
pip install -e ".[vlm]"    # Claude VLM review support
pip install -e ".[server]" # FastAPI + uvicorn
```

## Config

The application loads configuration from `AGENT_CONFIG_PATH` (defaults to `./configs/dev.yaml`).
You can also set this variable in a `.env` file at the project root.

See `configs/example.yaml` for a fully commented configuration template. Each agent entry defines
its model and optional agent-specific settings.

Key fields per agent:
- `model_name` — LLM model identifier
- `api_key` — API key for the model provider
- `base_url` — API endpoint URL

Additional top-level sections:
- `skills` — skills system toggle and active skill list
- `tools` — ReAct tool configuration (citation validation, paper search, etc.)
- `vlm_service` — shared VLM provider for visual review (supports OpenAI-compatible and Claude)

## Service Endpoints (Server Mode)

- `GET /healthz` — health check
- `GET /config` — current app config
- `GET /list_agents` — list registered agents and endpoints
- Agent-specific routes are registered under `/agent/*` and `/metadata/*`

## Repository Layout

```
.
├── easypaper/          # Thin SDK package (public API)
│   ├── __init__.py     # Re-exports: EasyPaper, PaperMetaData, EventType, ...
│   └── client.py       # EasyPaper class: generate(), generate_stream()
├── src/                # Core implementation (agents, config, skills)
│   ├── main.py         # FastAPI app (server mode entrypoint)
│   ├── agents/         # Agent implementations (metadata, writer, reviewer, ...)
│   ├── config/         # YAML config loading and schema
│   └── skills/         # Skill loader, registry, and router
├── configs/            # YAML configs for agents and models
├── skills/             # Built-in YAML skill definitions (venues, writing, reviewing)
├── scripts/            # CLI utilities and demos
├── examples/           # SDK usage examples
├── plugins/            # Claude Code plugin assets
├── tests/              # Test suite
└── pyproject.toml      # Package metadata (name: easypaper)
```
