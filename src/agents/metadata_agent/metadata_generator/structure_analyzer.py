"""
Structure analyzer for complex research-material folders.

Builds a lightweight blueprint that identifies:
- semantic clusters (e.g. experiment_* buckets)
- guidance files (README/report/summary/meta)
- metadata-field candidates for phased extraction
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .models import FileCategory, FolderScanResult

FIELD_ORDER = ("idea_hypothesis", "method", "data", "experiments")

GUIDANCE_KEYWORDS = (
    "readme",
    "report",
    "summary",
    "meta",
    "overview",
    "introduction",
    "guide",
)

FIELD_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "idea_hypothesis": (
        "hypothesis",
        "idea",
        "motivation",
        "introduction",
        "background",
        "question",
        "problem",
    ),
    "method": (
        "method",
        "approach",
        "algorithm",
        "model",
        "pipeline",
        "train",
        "architecture",
    ),
    "data": (
        "data",
        "dataset",
        "schema",
        "source",
        "sample",
        "statistics",
    ),
    "experiments": (
        "experiment",
        "evaluation",
        "result",
        "ablation",
        "benchmark",
        "metric",
        "analysis",
    ),
}

FIELD_CATEGORY_BONUS: Dict[str, Tuple[FileCategory, ...]] = {
    "idea_hypothesis": (FileCategory.TEXT, FileCategory.PDF),
    "method": (FileCategory.TEXT, FileCategory.CODE, FileCategory.PDF),
    "data": (FileCategory.DATA, FileCategory.TEXT, FileCategory.PDF),
    "experiments": (FileCategory.DATA, FileCategory.TEXT, FileCategory.CODE, FileCategory.PDF),
}


@dataclass
class StructureBlueprint:
    """
    In-memory blueprint for phased metadata extraction.

    - **Description**:
      - Captures cluster assignment and extraction priorities.
      - Helps reduce noisy all-at-once extraction on large folders.
    """

    clusters: Dict[str, List[str]]
    file_to_cluster: Dict[str, str]
    guidance_files: List[str]
    field_candidates: Dict[str, List[str]]
    phase_files: Dict[str, List[str]]
    prioritized_files: List[str]


class StructureAnalyzer:
    """
    Analyze a FolderScanResult and construct a phased extraction blueprint.
    """

    def analyze(
        self,
        scan_result: FolderScanResult,
        max_guidance_files: int = 30,
        max_candidates_per_field: int = 120,
    ) -> StructureBlueprint:
        file_categories = _build_category_lookup(scan_result)
        all_files = sorted(file_categories.keys())

        clusters, file_to_cluster = self._build_clusters(all_files)
        guidance_files = self._pick_guidance_files(
            all_files,
            file_categories,
            max_items=max_guidance_files,
        )
        guidance_set = set(guidance_files)

        field_candidates: Dict[str, List[str]] = {}
        for field in FIELD_ORDER:
            field_candidates[field] = self._pick_field_candidates(
                files=all_files,
                file_categories=file_categories,
                field=field,
                guidance_set=guidance_set,
                max_items=max_candidates_per_field,
            )

        phase_files = {
            "phase_1_guidance": guidance_files,
            "phase_2_idea_hypothesis": field_candidates["idea_hypothesis"],
            "phase_2_method": field_candidates["method"],
            "phase_2_data": field_candidates["data"],
            "phase_2_experiments": field_candidates["experiments"],
        }

        prioritized_files: List[str] = []
        for phase in (
            "phase_1_guidance",
            "phase_2_idea_hypothesis",
            "phase_2_method",
            "phase_2_data",
            "phase_2_experiments",
        ):
            prioritized_files.extend(phase_files[phase])
        prioritized_files = _unique_keep_order(prioritized_files)

        return StructureBlueprint(
            clusters=clusters,
            file_to_cluster=file_to_cluster,
            guidance_files=guidance_files,
            field_candidates=field_candidates,
            phase_files=phase_files,
            prioritized_files=prioritized_files,
        )

    @staticmethod
    def _build_clusters(files: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        clusters: Dict[str, List[str]] = {}
        file_to_cluster: Dict[str, str] = {}

        for rel in files:
            cluster = _detect_cluster(rel)
            clusters.setdefault(cluster, []).append(rel)
            file_to_cluster[rel] = cluster

        return clusters, file_to_cluster

    @staticmethod
    def _pick_guidance_files(
        files: List[str],
        file_categories: Dict[str, FileCategory],
        max_items: int,
    ) -> List[str]:
        scored: List[Tuple[int, str]] = []
        for rel in files:
            category = file_categories.get(rel, FileCategory.UNKNOWN)
            if category in (FileCategory.IMAGE, FileCategory.UNKNOWN):
                continue

            basename = Path(rel).stem.lower()
            score = 0
            if basename == "readme":
                score += 120
            elif basename.startswith("readme"):
                score += 100

            for kw in GUIDANCE_KEYWORDS:
                if kw in basename:
                    score += 40

            if rel.lower().endswith(".md"):
                score += 15
            elif rel.lower().endswith((".txt", ".rst", ".json", ".jsonl")):
                score += 8

            if score > 0:
                scored.append((score, rel))

        scored.sort(key=lambda x: (-x[0], x[1]))
        return [rel for _, rel in scored[:max_items]]

    @staticmethod
    def _pick_field_candidates(
        files: List[str],
        file_categories: Dict[str, FileCategory],
        field: str,
        guidance_set: set[str],
        max_items: int,
    ) -> List[str]:
        keywords = FIELD_KEYWORDS[field]
        preferred_categories = FIELD_CATEGORY_BONUS[field]

        scored: List[Tuple[int, str]] = []
        for rel in files:
            rel_l = rel.lower()
            category = file_categories.get(rel, FileCategory.UNKNOWN)
            if category in (FileCategory.CONFIG, FileCategory.UNKNOWN, FileCategory.IMAGE):
                continue

            score = 0
            for kw in keywords:
                if kw in rel_l:
                    score += 22

            if category in preferred_categories:
                score += 8

            if rel in guidance_set:
                score += 12

            if score > 0:
                scored.append((score, rel))

        scored.sort(key=lambda x: (-x[0], x[1]))
        return [rel for _, rel in scored[:max_items]]


def _build_category_lookup(scan_result: FolderScanResult) -> Dict[str, FileCategory]:
    mapping: Dict[str, FileCategory] = {}
    for category, rel_paths in scan_result.files_by_category.items():
        for rel in rel_paths:
            mapping[rel] = category
    return mapping


def _detect_cluster(rel_path: str) -> str:
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p]
    lower_parts = [p.lower() for p in parts]

    for p in lower_parts:
        if p.startswith("experiment_"):
            return p

    if len(lower_parts) >= 2 and lower_parts[0] in {"analysis", "experiments"}:
        return "/".join(lower_parts[:2])
    if lower_parts:
        return lower_parts[0]
    return "root"


def _unique_keep_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    output: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output
