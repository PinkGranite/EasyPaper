# easypaper

Generate AI-powered academic papers from metadata interactively.

## allowed-tools

- Bash
- Read
- Glob
- Grep

## workflow

Follow these steps to generate an academic paper:

### 1. Check Installation

First, verify that `easypaper` is installed:

```
Bash: pip show easypaper
```

If not installed, instruct the user to install it:
```
pip install easypaper
```

### 2. Check API Key Configuration

Check for API key in environment variables:
```
Bash: echo $ANTHROPIC_API_KEY (or $OPENAI_API_KEY, $MOONSHOT_API_KEY)
```

If no API key is found, prompt the user to input their API key. Do NOT create or modify any config files. Simply ask them to provide the key.

### 3. Gather Paper Parameters

Ask the user to choose input method:

**Option A: Interactive Input**
Ask the user for these fields one by one:
- **Title**: Paper title
- **Idea/Hypothesis**: Research idea or hypothesis
- **Method**: Method description
- **Data**: Data or validation method description
- **Experiments**: Experiment design, results, findings
- **References**: BibTeX entries (optional, can be empty)
- **Target Venue**: NeurIPS, ICML, ICLR, ACL, AAAI, COLM, Nature, or custom

**Option B: YAML File**
If the user has an existing YAML/JSON file with paper parameters, ask for the file path:
- Read the file to extract parameters
- Validate required fields are present

### 4. Select Output Options

Ask the user for:
- **Output Format**: LaTeX only, or LaTeX + PDF (PDF requires LaTeX toolchain)
- **Output Directory**: Where to save files (default: `./output/`)
- **Config File Path**: Path to YAML config with API keys (or use default location)

### 5. Generate Paper

Use the following Python code to generate the paper:

```python
import asyncio
import os
import sys

# Try to import easypaper - use local path if available
try:
    import easypaper
except ImportError:
    # Try adding current directory to path
    sys.path.insert(0, os.getcwd())
    import easypaper

from easypaper import EasyPaper, PaperMetaData

# Set API key from user input
os.environ['ANTHROPIC_API_KEY'] = user_api_key  # or other provider

# Create EasyPaper instance with config
config_path = user_config_path or './configs/example.yaml'
ep = EasyPaper(config_path=config_path)

# Create metadata from user input
metadata = PaperMetaData(
    title=user_title,
    idea_hypothesis=user_idea,
    method=user_method,
    data=user_data,
    experiments=user_experiments,
    references=user_references or [],
    style_guide=user_venue,
)

# Generate paper
result = await ep.generate(
    metadata,
    compile_pdf=compile_pdf,
    output_dir=output_dir,
    enable_review=True,
)

# Print results
print(f"Status: {result.status}")
print(f"LaTeX: {result.output_path}")
if result.pdf_path:
    print(f"PDF: {result.pdf_path}")
```

### 6. Return Results

After generation, present the results to the user:

```
Paper generated successfully!

- Title: {paper_title}
- LaTeX output: {output_path}
- PDF: {pdf_path} (if generated)
- Word count: {total_word_count}
- Review iterations: {review_iterations}
```

If errors occurred, show the error messages and suggest fixes.

## notes

- This command requires the `easypaper` Python package to be installed
- API keys should be provided by the user through interactive input
- PDF generation requires LaTeX toolchain (pdflatex + bibtex)
- Supported venues: NeurIPS, ICML, ICLR, ACL, AAAI, COLM, Nature
- Generated papers are saved to the specified output directory
