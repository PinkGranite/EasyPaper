#!/usr/bin/env python3
"""
Verify external binaries used by EasyPaper (LaTeX + optional PDF helpers).

- **Description**:
    - Prints PASS/FAIL for each tool. Applies the same TeX ``PATH`` bootstrap as
      TypesetterAgent so MiKTeX/TeX Live installs are detected even when the
      shell was not restarted after installation.

- **Args**:
    - None (CLI flags may be added later).

- **Returns**:
    - Process exit code ``0`` if ``pdflatex`` and ``bibtex`` are available; ``1`` otherwise.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.agents.shared.tex_path_bootstrap import (  # noqa: E402
    _candidate_tex_bin_dirs,
    ensure_tex_bin_on_path,
)


def _run_version(cmd: list[str], timeout: float = 12.0) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        out = (p.stdout or "") + (p.stderr or "")
        first = out.strip().splitlines()[0] if out.strip() else "(no output)"
        return p.returncode, first[:200]
    except FileNotFoundError:
        return -1, "executable not found"
    except subprocess.TimeoutExpired:
        return -1, "timeout"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check EasyPaper external toolchain")
    parser.add_argument(
        "--strict-poppler",
        action="store_true",
        help="Treat Poppler (pdftoppm) as required (only if you rely on pdf2image without PyMuPDF)",
    )
    args = parser.parse_args()

    print("EasyPaper toolchain check")
    print(f"Repository: {_REPO_ROOT}")
    tex_override = os.environ.get("EASYPAPER_TEX_BIN", "").strip()
    if tex_override:
        print(f"EASYPAPER_TEX_BIN={tex_override}")
    print()

    candidates = _candidate_tex_bin_dirs()
    print("TeX search paths (before bootstrap):")
    for d in candidates[:12]:
        mark = "exists" if d.is_dir() else "missing"
        print(f"  [{mark}] {d}")
    extra = len(candidates) - 12
    if extra > 0:
        print(f"  ... and {extra} more")
    print()

    ensure_tex_bin_on_path()
    ok = True

    def line(name: str, required: bool, version_cmd: list[str]) -> None:
        nonlocal ok
        req = "required" if required else "optional"
        w = shutil.which(name)
        if not w:
            status = "FAIL" if required else "SKIP"
            if required:
                ok = False
            print(f"  [{status}] {name} ({req}): not on PATH")
            return
        code, hint = _run_version(version_cmd)
        status = "PASS" if code == 0 else "WARN"
        if required and code != 0:
            ok = False
        print(f"  [{status}] {name}: {w}")
        if hint and hint != "(no output)":
            print(f"         {hint}")

    print("LaTeX (required for PDF compilation):")
    line("pdflatex", True, ["pdflatex", "--version"])
    line("bibtex", True, ["bibtex", "--version"])
    print()
    print("LaTeX helpers (optional, some templates / figures):")
    line("kpsewhich", False, ["kpsewhich", "--version"])
    line("epstopdf", False, ["epstopdf", "--help"])
    print()
    print("PDF rasterization (optional; PyMuPDF is preferred and already a dependency):")
    line("pdftoppm", args.strict_poppler, ["pdftoppm", "-v"])
    line("pdfinfo", False, ["pdfinfo", "-h"])

    print()
    if ok:
        print("Summary: OK - pdflatex and bibtex are available for TypesetterAgent.")
        return 0
    print(
        "Summary: FAIL - install MiKTeX or TeX Live, enable "
        "'Install missing packages on-the-fly' in MiKTeX Console, "
        "add the TeX bin directory to your user PATH, then open a new terminal. "
        "Or set EASYPAPER_TEX_BIN to the folder that contains pdflatex.exe "
        "(e.g. ...\\MiKTeX\\miktex\\bin\\x64)."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
