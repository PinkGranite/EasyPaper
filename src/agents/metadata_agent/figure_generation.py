"""Figure preprocessing for AcademicDreamer-backed image generation."""

from __future__ import annotations

import inspect
import os
import re
import shutil
from hashlib import sha1
from pathlib import Path
from typing import Any, Awaitable, Callable

from .models import FigureSpec, PaperMetaData

TARGET_TYPES = (
    "infograph",
    "architecture_diagram",
    "flowchart",
    "timeline",
    "data_visualization",
)

_KNOWN_VENUES = {
    "cvpr": "cvpr",
    "computer vision and pattern recognition": "cvpr",
    "iclr": "iclr",
    "international conference on learning representations": "iclr",
    "neurips": "neurips",
    "nips": "neurips",
    "neural information processing systems": "neurips",
    "nature": "nature",
    "nature communications": "nature",
    "nature machine intelligence": "nature",
}

_FALLBACK_STYLE_PROMPTS = {
    "icml": (
        "ICML-style academic figure: clean machine learning systems diagram, "
        "precise labels, modular blocks, restrained blue/teal palette, white background."
    ),
    "acl": (
        "ACL-style NLP illustration: publication-ready vector schematic with sequence "
        "flows, text processing modules, clean typography, neutral palette."
    ),
    "emnlp": (
        "EMNLP-style language model figure: crisp vector diagram, token/data flow emphasis, "
        "readable annotations, minimal but technical conference-paper aesthetic."
    ),
    "kdd": (
        "KDD-style data mining visualization: polished academic infographic with analytic "
        "workflow structure, concise labels, subtle color coding, white background."
    ),
}

_TARGET_TYPE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("timeline", ("timeline", "chronology", "roadmap", "milestone", "phase")),
    (
        "data_visualization",
        (
            "plot",
            "chart",
            "graph",
            "histogram",
            "scatter",
            "distribution",
            "curve",
            "bar chart",
            "line chart",
            "metric",
            "results",
            "ablation",
        ),
    ),
    (
        "flowchart",
        (
            "workflow",
            "flowchart",
            "pipeline",
            "process",
            "procedure",
            "training loop",
            "inference loop",
            "step",
        ),
    ),
    (
        "architecture_diagram",
        (
            "architecture",
            "framework",
            "system",
            "module",
            "component",
            "network",
            "overview",
            "block diagram",
            "method",
            "model",
        ),
    ),
    (
        "infograph",
        (
            "overview figure",
            "comparison",
            "taxonomy",
            "summary",
            "teaser",
            "concept",
            "infographic",
        ),
    ),
)


class FigureGenerationError(RuntimeError):
    """Raised when figure preprocessing cannot produce a usable asset."""


def _slugify(value: str, *, default: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower()).strip("-")
    return slug or default


def _normalize_style_guide(style_guide: str | None) -> str | None:
    if not style_guide:
        return None
    normalized = style_guide.strip().lower()
    if normalized in _KNOWN_VENUES:
        return _KNOWN_VENUES[normalized]
    for key, canonical in _KNOWN_VENUES.items():
        if key in normalized:
            return canonical
    for key in _FALLBACK_STYLE_PROMPTS:
        if normalized == key or normalized.startswith(f"{key} ") or f" {key} " in normalized:
            return key
    return normalized


def derive_figure_style(fig: FigureSpec, metadata: PaperMetaData) -> str:
    """Derive the effective Dreamer style string for a figure."""
    if fig.style and fig.style.strip():
        return fig.style.strip()

    normalized = _normalize_style_guide(metadata.style_guide)
    if normalized in {"cvpr", "iclr", "neurips", "nature"}:
        return normalized
    if normalized in _FALLBACK_STYLE_PROMPTS:
        return _FALLBACK_STYLE_PROMPTS[normalized]
    if metadata.style_guide and metadata.style_guide.strip():
        return (
            f"Publication-ready academic illustration for {metadata.style_guide.strip()}: "
            "clean vector schematic, restrained palette, legible labels, white background."
        )
    return (
        "Professional academic illustration style: clean vector schematic, white background, "
        "precise annotations, restrained scientific color palette."
    )


def derive_figure_target_type(fig: FigureSpec, metadata: PaperMetaData) -> str:
    """Derive the effective Dreamer target type for a figure."""
    if fig.target_type and fig.target_type.strip():
        candidate = fig.target_type.strip()
        if candidate in TARGET_TYPES:
            return candidate
        lowered = candidate.lower().replace(" ", "_")
        if lowered in TARGET_TYPES:
            return lowered
        return candidate

    haystack = " ".join(
        part
        for part in (
            fig.caption,
            fig.description,
            fig.generation_prompt,
            metadata.method,
            metadata.experiments,
        )
        if part
    ).lower()
    scored: list[tuple[int, str]] = []
    for target_type, keywords in _TARGET_TYPE_KEYWORDS:
        score = sum(1 for keyword in keywords if keyword in haystack)
        if score:
            scored.append((score, target_type))
    if not scored:
        return "architecture_diagram"
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def derive_figure_idea(fig: FigureSpec, metadata: PaperMetaData) -> str:
    """Derive the effective Dreamer idea string for a figure."""
    if fig.generation_prompt and fig.generation_prompt.strip():
        return fig.generation_prompt.strip()
    if fig.description and fig.description.strip():
        return f"{fig.description.strip()} Caption: {fig.caption.strip()}"
    if fig.caption and fig.caption.strip():
        return fig.caption.strip()

    details = [
        f"Paper title: {metadata.title.strip()}." if metadata.title else "",
        f"Method: {metadata.method.strip()}." if metadata.method else "",
        f"Data: {metadata.data.strip()}." if metadata.data else "",
        f"Experiments: {metadata.experiments.strip()}." if metadata.experiments else "",
    ]
    joined = " ".join(part for part in details if part)
    if joined:
        return joined
    return "Academic paper figure illustrating the core method and experimental setup."


def build_figure_generation_request(fig: FigureSpec, metadata: PaperMetaData) -> dict[str, str]:
    """Build the fully resolved AcademicDreamer request payload."""
    return {
        "idea": derive_figure_idea(fig, metadata),
        "style": derive_figure_style(fig, metadata),
        "target_type": derive_figure_target_type(fig, metadata),
    }


def _resolve_generation_roots(
    metadata: PaperMetaData,
    *,
    output_dir: str | None,
    results_dir: str | Path | None,
) -> tuple[Path, Path]:
    if metadata.materials_root:
        materials_root = Path(metadata.materials_root).expanduser().resolve()
    elif output_dir:
        materials_root = Path(output_dir).expanduser().resolve()
        metadata.materials_root = str(materials_root)
    elif results_dir:
        materials_root = (
            Path(results_dir).expanduser().resolve()
            / "generated_materials"
            / _slugify(metadata.title or "untitled-paper", default="untitled-paper")
        )
        metadata.materials_root = str(materials_root)
    else:
        raise FigureGenerationError(
            "Figure generation could not resolve a writable materials root. "
            "Set metadata.materials_root or provide an output directory."
        )

    generated_dir = materials_root / "generated_figures"
    try:
        generated_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - defensive filesystem branch
        raise FigureGenerationError(
            f"Failed to prepare generated figure directory: {generated_dir}"
        ) from exc
    if not os.access(generated_dir, os.W_OK):
        raise FigureGenerationError(f"Generated figure directory is not writable: {generated_dir}")

    metadata.materials_root = str(materials_root)
    return materials_root, generated_dir


def resolve_figure_file_path(fig: FigureSpec, materials_root: str | Path | None) -> Path | None:
    """Resolve a figure file path against materials_root when needed."""
    if not fig.file_path:
        return None

    candidate = Path(fig.file_path).expanduser()
    if not candidate.is_absolute() and materials_root:
        candidate = Path(materials_root) / candidate
    return candidate.resolve()


def _normalize_path_for_metadata(path: Path, materials_root: Path) -> str:
    try:
        return path.resolve().relative_to(materials_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _load_academic_dreamer_generate() -> Callable[..., Awaitable[dict[str, Any]]]:
    try:
        from academic_dreamer import generate_academic_illustration
    except ImportError:
        try:
            from academic_dreamer.main import generate_academic_illustration
        except ImportError as exc:  # pragma: no cover - import boundary
            raise FigureGenerationError(
                "Figure generation requested but academic_dreamer is not installed. "
                "Install EasyPaper with the images extra: pip install easypaper[images]."
            ) from exc
    return generate_academic_illustration


def _select_generated_output_path(result: dict[str, Any]) -> Path:
    output_paths = result.get("output_paths")
    if isinstance(output_paths, dict):
        for key in ("png", "pdf", "jpg", "jpeg", "svg"):
            value = output_paths.get(key)
            if value:
                return Path(value).expanduser().resolve()
        for value in output_paths.values():
            if value:
                return Path(value).expanduser().resolve()
    elif isinstance(output_paths, (list, tuple)):
        for value in output_paths:
            if value:
                return Path(str(value)).expanduser().resolve()
    elif isinstance(output_paths, str):
        return Path(output_paths).expanduser().resolve()
    raise FigureGenerationError(
        "AcademicDreamer did not return a usable output path for the generated figure."
    )


async def _maybe_await(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result


async def preprocess_generated_figures(
    metadata: PaperMetaData,
    *,
    output_dir: str | None,
    results_dir: str | Path | None,
    generator: Callable[..., Awaitable[dict[str, Any]]] | None = None,
) -> list[FigureSpec]:
    """Resolve/generate auto-generated figures into ordinary file-backed assets."""
    if not metadata.figures:
        return []

    materials_root = (
        Path(metadata.materials_root).expanduser().resolve()
        if metadata.materials_root
        else None
    )
    generated_dir: Path | None = None
    generator_fn = generator

    for fig in metadata.figures:
        resolved_existing = resolve_figure_file_path(fig, materials_root)
        if resolved_existing and resolved_existing.exists():
            if fig.auto_generate:
                fig.auto_generate = False
            continue
        if not fig.auto_generate:
            continue

        request = build_figure_generation_request(fig, metadata)
        print(
            "[MetaDataAgent] Generating figure "
            f"{fig.id} target_type={request['target_type']} style={request['style']}"
        )
        if materials_root is None or generated_dir is None:
            materials_root, generated_dir = _resolve_generation_roots(
                metadata,
                output_dir=output_dir,
                results_dir=results_dir,
            )
        if generator_fn is None:
            generator_fn = _load_academic_dreamer_generate()

        result = await _maybe_await(
            generator_fn(
                idea=request["idea"],
                style=request["style"],
                target_type=request["target_type"],
                output_dir=generated_dir,
            )
        )
        if not isinstance(result, dict):
            raise FigureGenerationError(
                f"AcademicDreamer returned an unexpected response for figure {fig.id!r}."
            )
        if result.get("error"):
            raise FigureGenerationError(
                f"AcademicDreamer failed for figure {fig.id}: {result['error']}"
            )

        raw_output_path = _select_generated_output_path(result)
        if not raw_output_path.exists():
            raise FigureGenerationError(
                f"AcademicDreamer reported output for figure {fig.id} but the file was missing: "
                f"{raw_output_path}"
            )

        ext = raw_output_path.suffix or ".png"
        fingerprint = sha1(
            "|".join(
                [fig.id, request["idea"], request["style"], request["target_type"]]
            ).encode("utf-8")
        ).hexdigest()[:10]
        normalized_name = f"{_slugify(fig.id, default='figure')}-{fingerprint}{ext.lower()}"
        normalized_output = generated_dir / normalized_name
        if raw_output_path != normalized_output:
            shutil.copy2(raw_output_path, normalized_output)
        fig.file_path = _normalize_path_for_metadata(normalized_output, materials_root)
        fig.auto_generate = False

    return metadata.figures
