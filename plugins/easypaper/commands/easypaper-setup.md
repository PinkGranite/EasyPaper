Set up EasyPaper environment including Python dependencies and LaTeX toolchain.

## Execution contract

1. Use the `setup-environment` skill to:
   - Create isolated virtual environment (prefer `uv`, fallback to `venv`)
   - Install easypaper package
   - Check LaTeX installation and provide installation instructions if missing
   - Verify all components are working

2. After setup, provide clear instructions on:
   - How to activate the environment
   - How to use EasyPaper as Python SDK: `from easypaper import EasyPaper, PaperMetaData`
   - Next steps for using the plugin (need config file with API keys)

$ARGUMENTS
