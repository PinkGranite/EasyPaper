# easypaper

AI-powered academic paper generation plugin for Claude Code.

## Description

Generate LaTeX academic papers from metadata interactively. The plugin provides a guided workflow to collect all necessary information and automatically sets up the required environment.

## Quick Start

After installing the plugin, simply run:

```
/easypaper
```

The plugin will:
1. **Automatically set up the environment** (first time only):
   - Create an isolated Python virtual environment
   - Install easypaper package and dependencies
   - Check and guide LaTeX installation

2. **Guide you through metadata collection**:
   - Check if you have complete metadata (file or JSON)
   - If not, collect all required fields interactively (title, hypothesis, method, data, experiments, references)
   - Ask about optional fields (venue, page count, review options, etc.)
   - Allow you to review and edit before generation

3. **Generate your paper**:
   - Use EasyPaper Python SDK directly (no API server needed)
   - Generate the paper from metadata
   - Provide output files (LaTeX source, references, PDF if compiled)

## Manual Setup (Optional)

If you prefer to set up manually or need to troubleshoot:

```bash
# Set up environment
/setup

# Or manually:
# Using uv (recommended)
uv venv .easypaper-env
source .easypaper-env/bin/activate
uv pip install easypaper

# Using standard Python
python -m venv .easypaper-env
source .easypaper-env/bin/activate
pip install easypaper
```

## Prerequisites

- **Python 3.11+** (automatically checked)
- **LaTeX toolchain** (automatically checked, installation instructions provided if missing)
  - macOS: `brew install --cask mactex` or `brew install basictex`
  - Linux: `sudo apt-get install texlive-full`
  - Windows: Install MiKTeX or TeX Live
- **API key for LLM provider** (configured via config file)

## Configuration

Create a YAML config file (see `configs/example.yaml` in the project) with your API keys:

```yaml
agents:
  - name: metadata
    model:
      model_name: claude-sonnet-4-20250514
      api_key: YOUR_API_KEY
      base_url: https://api.anthropic.com/v1
  # ... other agents
```

The config path will be requested when initializing EasyPaper.

## Metadata Structure

The plugin collects metadata following the structure in `examples/meta.json`:

**Required fields:**
- `title`: Paper title
- `idea_hypothesis`: Core research question or hypothesis
- `method`: Research methodology
- `data`: Data sources and collection process
- `experiments`: Experimental results and findings
- `references`: List of references (BibTeX or structured format)

**Optional fields:**
- `style_guide`: Venue name (NeurIPS, ICML, ICLR, ACL, AAAI, COLM, Nature)
- `target_pages`: Target page count (default: 20)
- `template_path`: Custom LaTeX template path
- `compile_pdf`: Whether to compile PDF (default: true)
- `enable_vlm_review`: Enable review loop (default: true)
- `max_review_iterations`: Max review iterations (default: 3)
- `figures`: Array of figure objects
- `tables`: Array of table objects
- `code_repository`: Code repository configuration
- `output_dir`: Output directory path

## Supported Venues

- NeurIPS
- ICML
- ICLR
- ACL
- AAAI
- COLM
- Nature

## Commands

- `/easypaper` - Main command: guided workflow from setup to paper generation
- `/setup` - Set up environment manually
- `/paper-from-metadata` - Generate paper directly from existing metadata JSON
