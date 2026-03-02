"""
Code repository ingestion and section-aware context builder.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...metadata_agent.models import CodeRepositorySpec


DEFAULT_INCLUDE_GLOBS = [
    "**/*.py",
    "**/*.c",
    "**/*.cc",
    "**/*.cpp",
    "**/*.h",
    "**/*.hpp",
    "**/*.md",
    "**/*.markdown",
    "**/*.yaml",
    "**/*.yml",
    "**/*.json",
    "**/*.toml",
]

DEFAULT_EXCLUDE_GLOBS = [
    "**/.git/**",
    "**/node_modules/**",
    "**/venv/**",
    "**/.venv/**",
    "**/__pycache__/**",
    "**/build/**",
    "**/dist/**",
    "**/.idea/**",
    "**/.vscode/**",
]

METHOD_KEYWORDS = (
    "model", "algorithm", "architecture", "module", "class ", "def ",
    "forward", "inference", "encode", "decode", "optimizer",
)
EXPERIMENT_KEYWORDS = (
    "train", "eval", "experiment", "ablation", "metric", "dataset",
    "benchmark", "config", "seed", "reproduce", "hyperparameter",
)
RESULT_KEYWORDS = (
    "result", "analysis", "plot", "table", "figure", "report",
    "compare", "improvement", "error", "accuracy",
)


@dataclass
class FileSummary:
    """
    Summarized file record used for section evidence retrieval.
    """
    path: str
    extension: str
    size: int
    symbols: List[str]
    summary: str
    snippet: str
    lower_text: str
    method_score: int
    experiment_score: int
    result_score: int


def _score_by_keywords(text: str, keywords: Tuple[str, ...]) -> int:
    """
    Score relevance by simple keyword hit counting.
    """
    return sum(text.count(k) for k in keywords)


def _extract_symbols(text: str, ext: str, max_items: int = 12) -> List[str]:
    """
    Extract rough symbol names from code/doc text.
    """
    symbols: List[str] = []
    if ext in {".py"}:
        symbols.extend(re.findall(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text, flags=re.M))
        symbols.extend(re.findall(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*[\(:]", text, flags=re.M))
    elif ext in {".c", ".cc", ".cpp", ".h", ".hpp"}:
        symbols.extend(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*\{", text))
    elif ext in {".md", ".markdown"}:
        symbols.extend(re.findall(r"^\s*#+\s+(.+)$", text, flags=re.M))
    return symbols[:max_items]


def _build_summary(path: str, symbols: List[str], text: str) -> str:
    """
    Build a concise, deterministic per-file summary.
    """
    first_non_empty = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            first_non_empty = stripped
            break
    if symbols:
        return f"{path}: defines/contains {', '.join(symbols[:4])}"
    if first_non_empty:
        return f"{path}: {first_non_empty[:120]}"
    return f"{path}: content available"


def _safe_read_text(path: Path, max_bytes: int = 256_000) -> Optional[str]:
    """
    Read text safely and skip likely-binary content.
    """
    try:
        raw = path.read_bytes()
    except Exception:
        return None
    if b"\x00" in raw[:4096]:
        return None
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
    return raw.decode("utf-8", errors="ignore")


def _apply_glob_filters(
    rel_path: str,
    include_globs: List[str],
    exclude_globs: List[str],
) -> bool:
    """
    Return True if path should be included.
    """
    normalized = rel_path.replace("\\", "/")
    if include_globs and not any(fnmatch.fnmatch(normalized, g) for g in include_globs):
        return False
    if exclude_globs and any(fnmatch.fnmatch(normalized, g) for g in exclude_globs):
        return False
    return True


def _pick_top(files: List[FileSummary], attr: str, top_k: int = 8) -> List[FileSummary]:
    """
    Select top files by section score with stable fallback.
    """
    ranked = sorted(files, key=lambda x: (getattr(x, attr), x.size), reverse=True)
    return [f for f in ranked if getattr(f, attr) > 0][:top_k]


class CodeContextBuilder:
    """
    Build section-aware writing context from an optional code repository.
    """

    def __init__(self, workspace_root: Optional[str] = None):
        self.workspace_root = Path(workspace_root or os.getcwd())

    async def build(
        self,
        code_repo: CodeRepositorySpec,
        paper_title: str = "",
    ) -> Dict[str, Any]:
        """
        Build context packs and lightweight index from repository contents.
        """
        source_path, cleanup_dir = self._resolve_source(code_repo)
        try:
            scoped_path = source_path
            if code_repo.subdir:
                scoped_path = (source_path / code_repo.subdir).resolve()
                if not scoped_path.exists() or not scoped_path.is_dir():
                    raise ValueError(f"code_repository.subdir not found: {code_repo.subdir}")

            include_globs = code_repo.include_globs or list(DEFAULT_INCLUDE_GLOBS)
            exclude_globs = list(DEFAULT_EXCLUDE_GLOBS)
            if code_repo.exclude_globs:
                exclude_globs.extend(code_repo.exclude_globs)

            files, stats = self._scan_and_summarize(
                root=scoped_path,
                include_globs=include_globs,
                exclude_globs=exclude_globs,
                max_files=code_repo.max_files,
                max_total_bytes=code_repo.max_total_bytes,
            )

            method_files = _pick_top(files, "method_score")
            experiment_files = _pick_top(files, "experiment_score")
            result_files = _pick_top(files, "result_score")

            context = {
                "repository_info": {
                    "type": code_repo.type.value,
                    "source": str(scoped_path),
                    "ref": code_repo.ref,
                    "paper_title": paper_title,
                },
                "scan_stats": stats,
                "repo_overview": [f.summary for f in files[:16]],
                "method_pack": self._to_evidence_pack(method_files, "method"),
                "experiment_pack": self._to_evidence_pack(experiment_files, "experiment"),
                "result_pack": self._to_evidence_pack(result_files, "result"),
                "index": [
                    {
                        "path": f.path,
                        "summary": f.summary,
                        "symbols": f.symbols,
                        "snippet": f.snippet,
                        "lower_text": f.lower_text,
                    }
                    for f in files
                ],
            }
            return context
        finally:
            if cleanup_dir and cleanup_dir.exists():
                shutil.rmtree(cleanup_dir, ignore_errors=True)

    def retrieve_for_section(
        self,
        context: Dict[str, Any],
        section_type: str,
        query_bundle: List[str],
        top_k: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve additional evidence snippets by query for a section.
        """
        if not context or not context.get("index"):
            return []
        queries = [q.strip().lower() for q in query_bundle if q and q.strip()]
        if not queries:
            return []

        ranked: List[Tuple[int, Dict[str, Any]]] = []
        for item in context["index"]:
            text = f"{item.get('summary', '')}\n{item.get('lower_text', '')}"
            score = 0
            for q in queries:
                score += text.count(q)
            if section_type == "method":
                score += text.count("method")
                score += text.count("algorithm")
            elif section_type == "experiment":
                score += text.count("experiment")
                score += text.count("dataset")
                score += text.count("metric")
            elif section_type == "result":
                score += text.count("result")
                score += text.count("analysis")
            if score > 0:
                ranked.append((score, item))

        ranked.sort(key=lambda x: x[0], reverse=True)
        output: List[Dict[str, Any]] = []
        for score, item in ranked[:top_k]:
            output.append(
                {
                    "path": item.get("path", ""),
                    "symbol": ", ".join(item.get("symbols", [])[:3]),
                    "snippet": item.get("snippet", ""),
                    "why_relevant": f"Matched runtime query bundle for {section_type} (score={score})",
                    "confidence": round(min(0.95, 0.5 + 0.08 * score), 2),
                }
            )
        return output

    def _resolve_source(self, code_repo: CodeRepositorySpec) -> Tuple[Path, Optional[Path]]:
        """
        Resolve repository source and optional temporary cleanup path.
        """
        if code_repo.type.value == "local_dir":
            source = Path(code_repo.path or "").expanduser()
            if not source.is_absolute():
                source = (self.workspace_root / source).resolve()
            if not source.exists() or not source.is_dir():
                raise FileNotFoundError(f"Local code repository path not found: {source}")
            return source, None

        temp_dir = Path(tempfile.mkdtemp(prefix="easy_paper_repo_"))
        clone_cmd = ["git", "clone", "--depth", "1"]
        if code_repo.ref:
            clone_cmd.extend(["--branch", code_repo.ref])
        clone_cmd.extend([code_repo.url or "", str(temp_dir)])
        result = subprocess.run(clone_cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(f"Failed to clone git repo: {stderr}")
        return temp_dir, temp_dir

    def _scan_and_summarize(
        self,
        root: Path,
        include_globs: List[str],
        exclude_globs: List[str],
        max_files: int,
        max_total_bytes: int,
    ) -> Tuple[List[FileSummary], Dict[str, Any]]:
        """
        Scan and summarize matching files under resource limits.
        """
        files: List[FileSummary] = []
        total_bytes = 0
        skipped_binary = 0
        skipped_limits = 0
        scanned = 0

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            scanned += 1
            rel_path = path.relative_to(root).as_posix()
            if not _apply_glob_filters(rel_path, include_globs, exclude_globs):
                continue

            size = path.stat().st_size
            if len(files) >= max_files or total_bytes + size > max_total_bytes:
                skipped_limits += 1
                continue

            text = _safe_read_text(path)
            if not text:
                skipped_binary += 1
                continue

            ext = path.suffix.lower()
            symbols = _extract_symbols(text, ext)
            snippet = "\n".join(text.splitlines()[:16])[:1200]
            lower_text = text.lower()
            summary = _build_summary(rel_path, symbols, text)
            files.append(
                FileSummary(
                    path=rel_path,
                    extension=ext,
                    size=size,
                    symbols=symbols,
                    summary=summary,
                    snippet=snippet,
                    lower_text=lower_text,
                    method_score=_score_by_keywords(lower_text, METHOD_KEYWORDS),
                    experiment_score=_score_by_keywords(lower_text, EXPERIMENT_KEYWORDS),
                    result_score=_score_by_keywords(lower_text, RESULT_KEYWORDS),
                )
            )
            total_bytes += size

        files.sort(key=lambda x: x.size, reverse=True)
        stats = {
            "scanned_files": scanned,
            "indexed_files": len(files),
            "indexed_total_bytes": total_bytes,
            "skipped_binary_or_unreadable": skipped_binary,
            "skipped_by_limits": skipped_limits,
        }
        return files, stats

    def _to_evidence_pack(self, files: List[FileSummary], section_type: str) -> List[Dict[str, Any]]:
        """
        Convert file summaries to prompt-friendly evidence records.
        """
        output: List[Dict[str, Any]] = []
        for f in files:
            top_symbols = ", ".join(f.symbols[:3]) if f.symbols else ""
            output.append(
                {
                    "path": f.path,
                    "symbol": top_symbols,
                    "snippet": f.snippet,
                    "why_relevant": f"Likely supports {section_type} based on code/doc keyword coverage",
                    "confidence": 0.7 if top_symbols else 0.6,
                }
            )
        return output


def format_code_context_for_prompt(
    context: Optional[Dict[str, Any]],
    section_type: str,
    retrieved_evidence: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 6,
) -> str:
    """
    Format section-scoped code context for writer prompts.
    """
    if not context:
        return ""

    pack_key = f"{section_type}_pack"
    if pack_key not in context:
        # Map common section names to available packs.
        if section_type in {"related_work", "introduction"}:
            pack_key = "method_pack"
        elif section_type == "experiment":
            pack_key = "experiment_pack"
        elif section_type == "result":
            pack_key = "result_pack"
        else:
            pack_key = "method_pack"

    pack = list(context.get(pack_key, []))[:top_k]
    if retrieved_evidence:
        pack.extend(retrieved_evidence[:max(0, top_k - len(pack))])

    if not pack:
        return ""

    lines = [
        "## Code Repository Context",
        "Use the following repository-derived evidence where relevant.",
    ]
    for idx, ev in enumerate(pack, start=1):
        lines.append(f"- Evidence {idx}: {ev.get('path', '')}")
        symbol = ev.get("symbol", "")
        if symbol:
            lines.append(f"  - Symbols: {symbol}")
        why = ev.get("why_relevant", "")
        if why:
            lines.append(f"  - Relevance: {why}")
        snippet = (ev.get("snippet", "") or "").strip()
        if snippet:
            lines.append("  - Snippet:")
            lines.append("```text")
            lines.append(snippet[:800])
            lines.append("```")
    return "\n".join(lines)


def render_code_repository_summary_markdown(context: Dict[str, Any]) -> str:
    """
    Render a concise markdown summary for output directory export.
    """
    repo_info = context.get("repository_info", {})
    stats = context.get("scan_stats", {})
    overview = context.get("repo_overview", [])
    method_pack = context.get("method_pack", [])
    experiment_pack = context.get("experiment_pack", [])
    result_pack = context.get("result_pack", [])

    lines = [
        "# Code Repository Summary",
        "",
        "## Repository Info",
        f"- Type: `{repo_info.get('type', 'unknown')}`",
        f"- Source: `{repo_info.get('source', 'unknown')}`",
    ]
    if repo_info.get("ref"):
        lines.append(f"- Ref: `{repo_info.get('ref')}`")

    lines.extend(
        [
            "",
            "## Scan Stats",
            f"- Scanned files: {stats.get('scanned_files', 0)}",
            f"- Indexed files: {stats.get('indexed_files', 0)}",
            f"- Indexed bytes: {stats.get('indexed_total_bytes', 0)}",
            f"- Skipped binary/unreadable: {stats.get('skipped_binary_or_unreadable', 0)}",
            f"- Skipped by limits: {stats.get('skipped_by_limits', 0)}",
            "",
            "## Repository Overview",
        ]
    )
    for item in overview[:12]:
        lines.append(f"- {item}")

    def _add_pack(title: str, pack: List[Dict[str, Any]]) -> None:
        lines.append("")
        lines.append(f"## {title}")
        if not pack:
            lines.append("- No strong evidence extracted.")
            return
        for ev in pack[:8]:
            path = ev.get("path", "")
            symbol = ev.get("symbol", "")
            why = ev.get("why_relevant", "")
            lines.append(f"- `{path}`")
            if symbol:
                lines.append(f"  - Symbols: {symbol}")
            if why:
                lines.append(f"  - Why relevant: {why}")

    _add_pack("Method Evidence Index", method_pack)
    _add_pack("Experiment Evidence Index", experiment_pack)
    _add_pack("Result Evidence Index", result_pack)

    lines.extend(
        [
            "",
            "## Known Limitations",
            "- Evidence ranking currently uses heuristic keyword scoring.",
            "- Extremely large repositories may be partially indexed due to configured limits.",
            "- Binary assets are ignored during text understanding.",
        ]
    )

    return "\n".join(lines).strip() + "\n"
