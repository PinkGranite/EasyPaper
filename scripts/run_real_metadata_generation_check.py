#!/usr/bin/env python3
"""Manual real-image verification for metadata figure generation.

This script is intentionally outside the pytest testpaths so it does not run as
part of the normal test routine. It exercises the real EasyPaper metadata
preprocessing path against the real ``academic_dreamer`` package.

Usage:
    uv run python scripts/run_real_metadata_generation_check.py \
        --metadata /abs/path/to/metadata.json \
        --output-dir /abs/path/to/output
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from src.agents.metadata_agent.figure_generation import preprocess_generated_figures
from src.agents.metadata_agent.models import PaperMetaData


@dataclass
class ImageInspection:
    figure_id: str
    caption: str
    metadata_file_path: str
    resolved_path: str
    exists: bool
    file_size_bytes: int | None
    sha256: str | None
    format: str | None
    mode: str | None
    width: int | None
    height: int | None
    mean_pixel_value: float | None
    non_blank_hint: bool | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real AcademicDreamer-backed metadata preprocessing and inspect the output images.",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Absolute path to a metadata JSON file compatible with PaperMetaData.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Absolute directory where generated assets and reports should be written.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Optional fallback results root if metadata.materials_root is absent.",
    )
    return parser.parse_args()


def _load_metadata(path: Path) -> PaperMetaData:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return PaperMetaData(**payload)


def _resolve_figure_path(file_path: str | None, materials_root: str | None) -> Path | None:
    if not file_path:
        return None
    candidate = Path(file_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if materials_root:
        return (Path(materials_root).expanduser().resolve() / candidate).resolve()
    return candidate.resolve()


def _inspect_image(figure: Any, materials_root: str | None) -> ImageInspection:
    resolved = _resolve_figure_path(getattr(figure, "file_path", None), materials_root)
    if resolved is None:
        return ImageInspection(
            figure_id=figure.id,
            caption=figure.caption,
            metadata_file_path=figure.file_path or "",
            resolved_path="",
            exists=False,
            file_size_bytes=None,
            sha256=None,
            format=None,
            mode=None,
            width=None,
            height=None,
            mean_pixel_value=None,
            non_blank_hint=None,
        )

    if not resolved.exists():
        return ImageInspection(
            figure_id=figure.id,
            caption=figure.caption,
            metadata_file_path=figure.file_path or "",
            resolved_path=str(resolved),
            exists=False,
            file_size_bytes=None,
            sha256=None,
            format=None,
            mode=None,
            width=None,
            height=None,
            mean_pixel_value=None,
            non_blank_hint=None,
        )

    digest = sha256(resolved.read_bytes()).hexdigest()
    file_size = resolved.stat().st_size

    with Image.open(resolved) as img:
        img.load()
        stat = ImageStat.Stat(img.convert("L"))
        mean_pixel_value = float(stat.mean[0]) if stat.mean else None
        width, height = img.size
        return ImageInspection(
            figure_id=figure.id,
            caption=figure.caption,
            metadata_file_path=figure.file_path or "",
            resolved_path=str(resolved),
            exists=True,
            file_size_bytes=file_size,
            sha256=digest,
            format=img.format,
            mode=img.mode,
            width=width,
            height=height,
            mean_pixel_value=mean_pixel_value,
            non_blank_hint=(mean_pixel_value is not None and 1.0 < mean_pixel_value < 254.0),
        )


def _build_report(
    *,
    metadata_input_path: Path,
    output_dir: Path,
    metadata_before: dict[str, Any],
    metadata_after: PaperMetaData,
    image_inspections: list[ImageInspection],
) -> dict[str, Any]:
    generated = []
    bypassed = []
    unresolved = []
    for figure in metadata_after.figures:
        record = {
            "id": figure.id,
            "caption": figure.caption,
            "auto_generate": figure.auto_generate,
            "file_path": figure.file_path,
            "style": getattr(figure, "style", None),
            "target_type": getattr(figure, "target_type", None),
            "generation_prompt": getattr(figure, "generation_prompt", None),
        }
        if figure.file_path and not figure.auto_generate:
            generated.append(record)
        elif figure.file_path:
            bypassed.append(record)
        else:
            unresolved.append(record)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata_input_path": str(metadata_input_path),
        "output_dir": str(output_dir),
        "materials_root": metadata_after.materials_root,
        "summary": {
            "figure_count": len(metadata_after.figures),
            "resolved_file_backed_figures": len(generated),
            "unresolved_figures": len(unresolved),
        },
        "metadata_before": metadata_before,
        "metadata_after": metadata_after.model_dump(),
        "resolved_figures": generated,
        "image_inspections": [asdict(item) for item in image_inspections],
        "review_hints": [
            "Check metadata_after.figures[*].file_path and auto_generate flags.",
            "Open the resolved image paths to visually inspect the generated figures.",
            "Use non_blank_hint only as a coarse sanity check, not as semantic validation.",
        ],
    }


async def _main() -> int:
    args = _parse_args()
    metadata_path = Path(args.metadata).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    results_dir = (
        Path(args.results_dir).expanduser().resolve() if args.results_dir else output_dir
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_metadata(metadata_path)
    metadata_before = metadata.model_dump()

    await preprocess_generated_figures(
        metadata,
        output_dir=str(output_dir),
        results_dir=str(results_dir),
    )

    inspections = [
        _inspect_image(figure, metadata.materials_root)
        for figure in metadata.figures
    ]

    report = _build_report(
        metadata_input_path=metadata_path,
        output_dir=output_dir,
        metadata_before=metadata_before,
        metadata_after=metadata,
        image_inspections=inspections,
    )

    normalized_metadata_path = output_dir / "normalized_metadata.json"
    report_path = output_dir / "generation_report.json"

    normalized_metadata_path.write_text(
        json.dumps(metadata.model_dump(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print(f"normalized_metadata: {normalized_metadata_path}")
    print(f"report: {report_path}")
    for inspection in inspections:
        print(
            json.dumps(
                {
                    "figure_id": inspection.figure_id,
                    "resolved_path": inspection.resolved_path,
                    "exists": inspection.exists,
                    "size": [inspection.width, inspection.height],
                    "format": inspection.format,
                    "non_blank_hint": inspection.non_blank_hint,
                },
                ensure_ascii=True,
            )
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(_main()))
    except KeyboardInterrupt:
        print("aborted", file=sys.stderr)
        raise SystemExit(130)
