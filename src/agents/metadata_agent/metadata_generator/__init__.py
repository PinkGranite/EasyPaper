"""
Universal metadata generator: scan a folder of research materials and
synthesize a complete PaperMetaData object.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models import PaperMetaData
from .models import FileCategory, ExtractedFragment, FolderScanResult
from .scanner import FolderScanner
from .synthesizer import MetadataSynthesizer
from .structure_analyzer import StructureAnalyzer
from .asset_curator import (
    curate_paper_assets,
    dedupe_figures,
    dedupe_tables,
    rule_fallback_select,
)
from .extractors.bib_extractor import BibExtractor
from .extractors.image_extractor import ImageExtractor
from .extractors.data_extractor import DataExtractor
from .extractors.text_extractor import TextExtractor
from .extractors.code_extractor import CodeExtractor

logger = logging.getLogger(__name__)

_CATEGORY_EXTRACTORS = {
    FileCategory.BIB: BibExtractor,
    FileCategory.TEXT: TextExtractor,
    FileCategory.CODE: CodeExtractor,
    FileCategory.DATA: DataExtractor,
}


async def generate_metadata_from_folder(
    folder_path: str,
    llm_client: Any = None,
    model_name: str = "",
    include_globs: Optional[List[str]] = None,
    exclude_globs: Optional[List[str]] = None,
    max_figures: Optional[int] = None,
    max_tables: Optional[int] = None,
    vision_enrich_figures: bool = True,
    max_vision_figures: Optional[int] = None,
    vision_model: Optional[str] = None,
    max_vision_long_edge: int = 896,
    vision_cache_dir: Optional[str] = None,
    **overrides: Any,
) -> PaperMetaData:
    """
    Scan a folder of research materials and synthesize a PaperMetaData.

    - **Description**:
        - Walks *folder_path*, classifies every file by type, analyses folder
          structure to prioritise guidance and field-relevant files, then runs
          phased extraction and passes all fragments to an LLM synthesizer
          that produces coherent five-field prose.
        - BibTeX, images, CSV/JSON data, text, and code files are all
          handled automatically.  PDF extraction requires *llm_client*.

    - **Args**:
        - `folder_path` (str): Path to the research materials folder.
        - `llm_client` (optional): OpenAI-compatible client for LLM calls.
        - `model_name` (str): Model name for chat completions.
        - `include_globs` (list, optional): Glob patterns to include.
        - `exclude_globs` (list, optional): Glob patterns to exclude.
        - `max_figures` (int, optional): Hard cap on figure count; when the folder
          yields more, an LLM curator picks a minimal supporting subset (requires
          *llm_client*). If omitted, no cap is applied.
        - `max_tables` (int, optional): Hard cap on table count; same behaviour as
          ``max_figures``.
        - `vision_enrich_figures` (bool): If True and *llm_client* is set, run a vision
          model on each retained figure (after curation) to replace ``description`` with
          prose. Disable for tests or non-multimodal models.
        - `max_vision_figures` (int, optional): Max vision API calls (figures processed in
          list order). ``None`` means all retained figures.
        - `vision_model` (str, optional): Multimodal model name; defaults to *model_name*.
        - `max_vision_long_edge` (int): Longest image edge in pixels before JPEG encode.
        - `vision_cache_dir` (str, optional): Directory for vision description cache keys
          (content-hash based). Use a per-project or per-run path in tests to avoid
          cross-run cache hits. Defaults to ``EASYPAPER_CACHE_DIR`` / ``easypaper/figure_vision``.
        - `**overrides`: Fields to override in the final PaperMetaData
          (e.g. title, style_guide, template_path, target_pages).

    - **Returns**:
        - `PaperMetaData`: The fully populated metadata object.

    - **Raises**:
        - `FileNotFoundError`: If *folder_path* does not exist.
    """
    scanner = FolderScanner(
        include_globs=include_globs,
        exclude_globs=exclude_globs,
    )
    scan_result = scanner.scan(folder_path)
    logger.info(
        "Scanned %s: %d files in %d categories",
        folder_path,
        scan_result.total_files,
        len(scan_result.files_by_category),
    )

    root = os.path.abspath(scan_result.folder_path)
    all_fragments: List[ExtractedFragment] = []

    blueprint = StructureAnalyzer().analyze(scan_result)
    logger.info(
        "Structure blueprint: %d clusters, %d guidance files",
        len(blueprint.clusters),
        len(blueprint.guidance_files),
    )

    category_lookup = _build_category_lookup(scan_result)
    processed: set[str] = set()

    for phase, rel_paths in blueprint.phase_files.items():
        for rel in rel_paths:
            if rel in processed:
                continue
            category = category_lookup.get(rel, FileCategory.UNKNOWN)
            frags = await _extract_file_by_category(
                category=category,
                root=root,
                rel_path=rel,
                llm_client=llm_client,
                model_name=model_name,
                materials_root=root,
            )
            for frag in frags:
                frag.extra.setdefault("phase", phase)
                frag.extra.setdefault("cluster", blueprint.file_to_cluster.get(rel, "root"))
            all_fragments.extend(frags)
            processed.add(rel)

    for rel, category in category_lookup.items():
        if rel in processed:
            continue
        if category in (FileCategory.IMAGE, FileCategory.CONFIG, FileCategory.UNKNOWN):
            continue

        frags = await _extract_file_by_category(
            category=category,
            root=root,
            rel_path=rel,
            llm_client=llm_client,
            model_name=model_name,
            materials_root=root,
        )
        for frag in frags:
            frag.extra.setdefault("phase", "phase_3_remaining")
            frag.extra.setdefault("cluster", blueprint.file_to_cluster.get(rel, "root"))
        all_fragments.extend(frags)
        processed.add(rel)

    img_ext = ImageExtractor()
    image_frags = img_ext.extract_from_folder(root)
    for frag in image_frags:
        frag.extra.setdefault("phase", "phase_figures")
        frag.extra.setdefault(
            "cluster",
            blueprint.file_to_cluster.get(frag.source_file, _cluster_from_rel(frag.source_file)),
        )
    all_fragments.extend(image_frags)

    logger.info("Extracted %d fragments total", len(all_fragments))

    synthesizer = MetadataSynthesizer(llm_client=llm_client, model_name=model_name)
    metadata = await synthesizer.synthesize(all_fragments, overrides=overrides or None)
    metadata.materials_root = root

    figures_d = dedupe_figures(list(metadata.figures))
    tables_d = dedupe_tables(list(metadata.tables))
    trim_fig = max_figures is not None and len(figures_d) > max_figures
    trim_tab = max_tables is not None and len(tables_d) > max_tables
    metadata.figures = figures_d
    metadata.tables = tables_d

    if trim_fig or trim_tab:
        if llm_client and model_name:
            await curate_paper_assets(
                metadata,
                llm_client,
                model_name,
                max_figures=max_figures,
                max_tables=max_tables,
            )
        else:
            sel_f, sel_t = rule_fallback_select(
                metadata.figures,
                metadata.tables,
                max_figures=max_figures,
                max_tables=max_tables,
            )
            metadata.figures = sel_f
            metadata.tables = sel_t

    if vision_enrich_figures and llm_client and model_name and metadata.figures:
        from .figure_vision_enricher import enrich_figure_descriptions_vision

        await enrich_figure_descriptions_vision(
            metadata,
            llm_client,
            vision_model or model_name,
            max_long_edge=max_vision_long_edge,
            max_vision_figures=max_vision_figures,
            cache_dir=Path(vision_cache_dir).resolve() if vision_cache_dir else None,
        )

    return metadata


def _join(root: str, rel: str) -> str:
    return os.path.join(root, rel)


def _cluster_from_rel(rel_path: str) -> str:
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p]
    for p in parts:
        if p.lower().startswith("experiment_"):
            return p.lower()
    return parts[0].lower() if parts else "root"


def _build_category_lookup(scan_result: FolderScanResult) -> Dict[str, FileCategory]:
    mapping: Dict[str, FileCategory] = {}
    for category, rel_paths in scan_result.files_by_category.items():
        for rel in rel_paths:
            mapping[rel] = category
    return mapping


async def _extract_file_by_category(
    category: FileCategory,
    root: str,
    rel_path: str,
    llm_client: Any,
    model_name: str,
    materials_root: Optional[str] = None,
) -> List[ExtractedFragment]:
    full = _join(root, rel_path)

    if category == FileCategory.PDF:
        if not llm_client:
            return []
        from .extractors.pdf_extractor import PDFExtractor

        pdf_ext = PDFExtractor(llm_client=llm_client, model_name=model_name)
        try:
            return await pdf_ext.extract_async(full, materials_root=materials_root)
        except Exception as e:
            logger.warning("PDF extraction failed for %s: %s", rel_path, e)
            return []

    extractor_cls = _CATEGORY_EXTRACTORS.get(category)
    if extractor_cls is None:
        return []

    extractor = extractor_cls()
    try:
        return extractor.extract(full, materials_root=materials_root)
    except Exception as e:
        logger.warning("Extraction failed for %s: %s", rel_path, e)
        return []


__all__ = [
    "generate_metadata_from_folder",
    "StructureAnalyzer",
    "FileCategory",
    "ExtractedFragment",
    "FolderScanResult",
    "FolderScanner",
    "MetadataSynthesizer",
]
