"""
Vision-based figure description enrichment for metadata generation.

Runs only on figures already selected for ``PaperMetaData`` (after dedupe/curation)
to bound API cost. Uses downscaled raster images and optional disk cache.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, List, Optional

from ..models import PaperMetaData

logger = logging.getLogger(__name__)

VISION_SYSTEM = (
    "You help academic paper generation. You see one scientific figure at a time. "
    "Reply with plain text only (no JSON, no markdown fences)."
)

VISION_USER_TEMPLATE = (
    "Figure id: {fig_id}\n"
    "Caption (filename-derived, may be imperfect): {caption}\n\n"
    "Write 2–4 sentences in English for downstream writers and planners:\n"
    "- What the figure shows (plot type, axes or panels if visible, main comparisons)\n"
    "- The main empirical or qualitative takeaway\n"
    "- How it could support an Experiments or Results narrative\n\n"
    "Do not mention file paths or disk locations. If the image is unreadable, say so briefly."
)


def _default_cache_dir() -> Path:
    base = os.environ.get("EASYPAPER_CACHE_DIR", str(Path.home() / ".cache" / "easypaper"))
    return Path(base) / "figure_vision"


def _file_cache_key(abs_path: Path, rel: str) -> str:
    """
    Stable cache key from relative path and full file digest.

    - **Returns**:
        - Hex string used as cache filename stem.
    """
    h = hashlib.sha256()
    with abs_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return hashlib.sha256(f"{rel}\n{h.hexdigest()}".encode("utf-8")).hexdigest()[:40]


def _load_resize_jpeg(abs_path: Path, max_long_edge: int) -> tuple[bytes, str]:
    """
    Load an image file and return JPEG bytes plus data-URL media type.

    - **Raises**:
        - `ValueError`: If the format cannot be decoded.
    """
    from PIL import Image

    with Image.open(abs_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        m = max(w, h)
        if m > max_long_edge and m > 0:
            scale = max_long_edge / float(m)
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=82, optimize=True)
        return buf.getvalue(), "image/jpeg"


def _read_cached_description(cache_dir: Path, key: str) -> Optional[str]:
    path = cache_dir / f"{key}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        desc = data.get("description")
        return desc if isinstance(desc, str) and desc.strip() else None
    except Exception:
        return None


def _write_cache(cache_dir: Path, key: str, description: str) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{key}.json"
        path.write_text(
            json.dumps({"description": description}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logger.debug("Vision cache write skipped: %s", e)


async def _vision_one_figure(
    llm_client: Any,
    model: str,
    abs_path: Path,
    rel: str,
    fig_id: str,
    caption: str,
    *,
    max_long_edge: int,
    cache_dir: Path,
) -> str:
    key = _file_cache_key(abs_path, rel)
    cached = _read_cached_description(cache_dir, key)
    if cached:
        return cached

    try:
        jpeg_bytes, media_type = await asyncio.to_thread(
            _load_resize_jpeg, abs_path, max_long_edge
        )
    except Exception as e:
        logger.warning("Vision skip (decode/resize failed) %s: %s", rel, e)
        return ""

    b64 = base64.standard_b64encode(jpeg_bytes).decode("ascii")
    url = f"data:{media_type};base64,{b64}"

    messages = [
        {"role": "system", "content": VISION_SYSTEM},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": url, "detail": "low"},
                },
                {
                    "type": "text",
                    "text": VISION_USER_TEMPLATE.format(fig_id=fig_id, caption=caption or "(none)"),
                },
            ],
        },
    ]

    response = await llm_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=400,
    )
    raw = (response.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    if raw:
        _write_cache(cache_dir, key, raw)
    return raw


async def enrich_figure_descriptions_vision(
    metadata: PaperMetaData,
    llm_client: Any,
    vision_model: str,
    *,
    max_long_edge: int = 896,
    max_vision_figures: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> None:
    """
    Replace ``FigureSpec.description`` with vision-backed prose where possible.

    Only processes figures with ``file_path`` resolvable under ``metadata.materials_root``.
    Mutates ``metadata.figures`` in place.

    - **Args**:
        - `metadata` (PaperMetaData): Metadata after curation; must set ``materials_root``.
        - `llm_client`: OpenAI-compatible async client.
        - `vision_model` (str): Model id supporting image_url (e.g. gpt-4o).
        - `max_long_edge` (int): Downscale bound for the raster sent to the API.
        - `max_vision_figures` (int, optional): Max figures to call vision on; ``None`` means all.
        - `cache_dir` (Path, optional): Disk cache root; default under ``EASYPAPER_CACHE_DIR``.
    """
    root = getattr(metadata, "materials_root", None) or ""
    if not root or not llm_client:
        return

    root_path = Path(root)
    figures = list(metadata.figures)
    if not figures:
        return

    cdir = cache_dir or _default_cache_dir()
    limit = max_vision_figures if max_vision_figures is not None else len(figures)
    limit = max(0, min(limit, len(figures)))

    updated = []
    for fig in figures[:limit]:
        rel = fig.file_path
        if not rel:
            updated.append(fig)
            continue
        abs_path = (root_path / rel).resolve()
        if not abs_path.is_file():
            logger.warning("Vision skip (missing file): %s", rel)
            updated.append(fig)
            continue
        suf = abs_path.suffix.lower()
        if suf == ".svg":
            logger.warning("Vision skip (SVG not rasterized): %s", rel)
            updated.append(fig)
            continue
        try:
            text = await _vision_one_figure(
                llm_client,
                vision_model,
                abs_path,
                rel.replace("\\", "/"),
                fig.id,
                fig.caption,
                max_long_edge=max_long_edge,
                cache_dir=cdir,
            )
        except Exception as e:
            logger.warning("Vision enrichment failed for %s: %s", rel, e)
            updated.append(fig)
            continue
        if text:
            updated.append(fig.model_copy(update={"description": text}))
        else:
            updated.append(fig)

    # Append any figures beyond limit unchanged
    updated.extend(figures[limit:])
    metadata.figures = updated
