"""
Bootstrap TeX binaries onto PATH when they are installed but not exposed.

- **Description**:
    - On Windows, MiKTeX / TeX Live often install under Program Files or
      LocalAppData while the user session PATH was never updated. This module
      prepends a discovered ``bin`` directory that contains ``pdflatex`` so
      ``subprocess`` calls in TypesetterAgent succeed without a manual PATH edit.

- **Args**:
    - None.

- **Returns**:
    - None (mutates ``os.environ["PATH"]`` in-process when a candidate is found).
"""
from __future__ import annotations

import glob
import os
import platform
import shutil
from pathlib import Path
from typing import Iterable, List

_LOGGED_PREPEND = False


def _candidate_tex_bin_dirs() -> List[Path]:
    """
    Return ordered search locations for TeX ``bin`` directories.
    - **Description**:
        - Covers MiKTeX (user + machine), TeX Live (Windows win32), macOS
          MacTeX/BasicTeX, and Linux texlive.
        - If ``EASYPAPER_TEX_BIN`` is set to a directory containing ``pdflatex``,
          that directory is tried first.

    - **Returns**:
        - ``paths`` (List[Path]): Candidate directories (may not exist).
    """
    paths: List[Path] = []
    override = os.environ.get("EASYPAPER_TEX_BIN", "").strip()
    if override:
        paths.append(Path(override))
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA", "")
        if local:
            paths.append(Path(local) / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64")
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        pfx86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        paths.append(Path(pf) / "MiKTeX" / "miktex" / "bin" / "x64")
        paths.append(Path(pfx86) / "MiKTeX" / "miktex" / "bin" / "x64")
        # TeX Live: pick newest year under Program Files or C:\texlive
        for root in (Path(pf), Path(r"C:\texlive")):
            if root.exists():
                for d in sorted(
                    glob.glob(str(root / "20*" / "bin" / "win32")),
                    reverse=True,
                ):
                    paths.append(Path(d))
        # Portable / custom installs (user-defined convention)
        for label in ("MIKTEX_INSTALL_DIR", "TEXLIVE_ROOT"):
            root = os.environ.get(label, "").strip()
            if root:
                r = Path(root)
                if (r / "miktex" / "bin" / "x64").is_dir():
                    paths.append(r / "miktex" / "bin" / "x64")
                if (r / "bin" / "win32").is_dir():
                    paths.append(r / "bin" / "win32")
    elif platform.system() == "Darwin":
        paths.extend(
            [
                Path("/Library/TeX/texbin"),
                Path("/usr/local/texlive/2025/bin/universal-darwin"),
                Path("/usr/local/texlive/2024/bin/universal-darwin"),
                Path("/usr/local/texlive/2023/bin/universal-darwin"),
            ]
        )
    else:
        paths.extend(
            [
                Path("/usr/local/bin"),
                Path("/usr/bin"),
                Path("/usr/local/texlive/2025/bin/x86_64-linux"),
                Path("/usr/local/texlive/2024/bin/x86_64-linux"),
            ]
        )
    seen: set[str] = set()
    unique: List[Path] = []
    for p in paths:
        key = os.path.normcase(os.path.normpath(str(p)))
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def _pdflatex_names() -> Iterable[str]:
    if os.name == "nt":
        return ("pdflatex.exe", "pdflatex.cmd", "pdflatex")
    return ("pdflatex",)


def _first_pdflatex_in(dir_path: Path) -> Path | None:
    for name in _pdflatex_names():
        candidate = dir_path / name
        if candidate.is_file():
            return candidate
    return None


def ensure_tex_bin_on_path(logger=None) -> bool:
    """
    Ensure ``pdflatex`` is on PATH for the current process.

    - **Description**:
        - If ``shutil.which("pdflatex")`` already succeeds, returns immediately.
        - Otherwise prepends the first candidate directory that contains a
          ``pdflatex`` executable.

    - **Args**:
        - ``logger`` (Optional[logging.Logger]): If set, logs prepended directory once.

    - **Returns**:
        - ``True`` if ``pdflatex`` is callable after bootstrap; ``False`` otherwise.
    """
    global _LOGGED_PREPEND
    if shutil.which("pdflatex"):
        return True
    for d in _candidate_tex_bin_dirs():
        if not d.is_dir():
            continue
        if _first_pdflatex_in(d) is None:
            continue
        sep = os.pathsep
        os.environ["PATH"] = str(d) + sep + os.environ.get("PATH", "")
        if logger and not _LOGGED_PREPEND:
            logger.info("typesetter.tex_path_bootstrap prepended=%s", d)
            _LOGGED_PREPEND = True
        return shutil.which("pdflatex") is not None
    return False
