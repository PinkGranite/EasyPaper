"""
Universal metadata generator: scan a folder of research materials and
synthesize a complete PaperMetaData object.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..models import PaperMetaData
from .models import FileCategory, ExtractedFragment, FolderScanResult
from .scanner import FolderScanner
from .synthesizer import MetadataSynthesizer
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
    **overrides: Any,
) -> PaperMetaData:
    """
    Scan a folder of research materials and synthesize a PaperMetaData.

    - **Description**:
        - Walks *folder_path*, classifies every file by type, runs the
          appropriate extractor for each, and passes all fragments to an
          LLM synthesizer that produces coherent five-field prose.
        - BibTeX, images, CSV/JSON data, text, and code files are all
          handled automatically.  PDF extraction requires *llm_client*.

    - **Args**:
        - `folder_path` (str): Path to the research materials folder.
        - `llm_client` (optional): OpenAI-compatible client for LLM calls.
        - `model_name` (str): Model name for chat completions.
        - `include_globs` (list, optional): Glob patterns to include.
        - `exclude_globs` (list, optional): Glob patterns to exclude.
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

    all_fragments: List[ExtractedFragment] = []
    root = scan_result.folder_path

    for category, rel_paths in scan_result.files_by_category.items():
        if category == FileCategory.IMAGE:
            img_ext = ImageExtractor()
            all_fragments.extend(img_ext.extract_from_folder(root))
            continue

        if category == FileCategory.PDF:
            if llm_client:
                from .extractors.pdf_extractor import PDFExtractor
                pdf_ext = PDFExtractor(llm_client=llm_client, model_name=model_name)
                for rel in rel_paths:
                    full = _join(root, rel)
                    try:
                        frags = await pdf_ext.extract_async(full)
                        all_fragments.extend(frags)
                    except Exception as e:
                        logger.warning("PDF extraction failed for %s: %s", rel, e)
            continue

        extractor_cls = _CATEGORY_EXTRACTORS.get(category)
        if extractor_cls is None:
            continue
        extractor = extractor_cls()
        for rel in rel_paths:
            full = _join(root, rel)
            try:
                all_fragments.extend(extractor.extract(full))
            except Exception as e:
                logger.warning("Extraction failed for %s: %s", rel, e)

    logger.info("Extracted %d fragments total", len(all_fragments))

    synthesizer = MetadataSynthesizer(llm_client=llm_client, model_name=model_name)
    return await synthesizer.synthesize(all_fragments, overrides=overrides or None)


def _join(root: str, rel: str) -> str:
    import os
    return os.path.join(root, rel)


__all__ = [
    "generate_metadata_from_folder",
    "FileCategory",
    "ExtractedFragment",
    "FolderScanResult",
    "FolderScanner",
    "MetadataSynthesizer",
]
