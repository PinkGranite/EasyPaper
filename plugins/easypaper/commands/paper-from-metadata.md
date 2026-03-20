Generate a paper directly from metadata using EasyPaper Python SDK.

## Execution contract

1. **Use the `paper-from-metadata` skill** which handles the complete workflow:
   - Check if user has complete metadata (file or JSON object)
   - If missing or incomplete, collect interactively
   - Validate metadata against `examples/meta.json` structure
   - **IMPORTANT**: Ensure all paths in metadata are absolute paths (template_path, figures[].file_path, code_repository.path, output_dir)
   - Generate paper using EasyPaper SDK directly

2. **The skill will**:
   - Import EasyPaper: `from easypaper import EasyPaper, PaperMetaData`
   - Initialize with config: `ep = EasyPaper(config_path="...")` (config path should be absolute)
   - Convert metadata to `PaperMetaData` object (with all paths converted to absolute)
   - Generate: `result = await ep.generate(metadata, **options)`
   - Report results with absolute file paths and summary

## Path Requirements

**All paths must be absolute**:
- `template_path`: Absolute path to LaTeX template
- `figures[].file_path`: Absolute paths to figure files
- `code_repository.path`: Absolute path (if type is local_dir)
- `output_dir`: Absolute path to output directory
- `config_path`: Absolute path to config file

If user provides relative paths, convert them to absolute paths using `pathlib.Path.resolve()` before use.

## Fallback behavior

- **Missing required fields**: Skill automatically collects missing information interactively
- **Invalid metadata format**: Show validation errors and guide user to correct format (reference `examples/meta.json`)
- **Relative paths detected**: Automatically convert to absolute paths and inform user
- **Package not installed**: Use `setup-environment` skill first
- **Config missing**: Ask user for config path or use default (ensure absolute path)

$ARGUMENTS
