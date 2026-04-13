"""Tests for folder structure and guidance-first analysis."""
from src.agents.metadata_agent.metadata_generator.models import (
    FileCategory,
    FolderScanResult,
)
from src.agents.metadata_agent.metadata_generator.structure_analyzer import (
    StructureAnalyzer,
)


def _scan_result() -> FolderScanResult:
    return FolderScanResult(
        folder_path="/tmp/reputation_game",
        files_by_category={
            FileCategory.TEXT: [
                "analysis/single/experiment_1/README.md",
                "analysis/single/experiment_1/report.md",
                "analysis/single/experiment_2/report.md",
            ],
            FileCategory.DATA: [
                "analysis/single/experiment_1/data/analysis_summary.json",
                "analysis/single/experiment_2/charts/result_metrics.csv",
            ],
            FileCategory.CODE: [
                "experiments/experiment_1/src/train_model.py",
                "experiments/experiment_1/src/evaluation.py",
            ],
            FileCategory.IMAGE: [
                "analysis/single/experiment_1/assets/accuracy_curve.png",
            ],
        },
    )


class TestStructureAnalyzer:
    def test_detects_experiment_clusters(self) -> None:
        blueprint = StructureAnalyzer().analyze(_scan_result())

        assert "experiment_1" in blueprint.clusters
        assert "experiment_2" in blueprint.clusters
        assert any(
            p.endswith("README.md")
            for p in blueprint.clusters["experiment_1"]
        )

    def test_picks_guidance_files_first(self) -> None:
        blueprint = StructureAnalyzer().analyze(_scan_result())

        assert blueprint.guidance_files
        assert blueprint.guidance_files[0].endswith("README.md")
        assert any("analysis_summary.json" in p for p in blueprint.guidance_files)

    def test_builds_field_candidates(self) -> None:
        blueprint = StructureAnalyzer().analyze(_scan_result())

        assert any(
            p.endswith("train_model.py")
            for p in blueprint.field_candidates["method"]
        )
        assert any(
            p.endswith("result_metrics.csv")
            for p in blueprint.field_candidates["experiments"]
        )
        assert blueprint.phase_files["phase_1_guidance"]
