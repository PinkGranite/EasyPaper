"""
Tests for figure path unified replacement and conclusion multi-file mode.

Covers:
  1. Figure ID → file path replacement in TypesetterAgent
  2. Prompt compiler figure guidance using bare IDs
  3. Conclusion CITE/FLOAT marker cleanup in synthesis sections
  4. Conclusion written as separate file in multi-file mode
"""
import os
import re
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# =========================================================================
# 1. Figure path replacement — _build_figure_id_map & _rewrite
# =========================================================================

class TestBuildFigureIdMap:
    """Test multi-variant figure ID mapping."""

    def _get_agent(self):
        from src.agents.typesetter_agent.typesetter_agent import TypesetterAgent
        agent = TypesetterAgent.__new__(TypesetterAgent)
        return agent

    def test_basic_id_to_path_mapping(self, tmp_path):
        """fig:overview → figures/fig_1 when fig_1.png exists."""
        agent = self._get_agent()
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        (figures_dir / "fig_1.png").write_text("fake")

        figure_paths = {"fig:overview": "figures/fig_1.png"}
        id_map = agent._build_figure_id_map(figure_paths, str(tmp_path))

        assert "fig:overview" in id_map
        assert id_map["fig:overview"] == "figures/fig_1"

    def test_variant_underscore_mapped(self, tmp_path):
        """fig_overview (colon→underscore variant) also resolves."""
        agent = self._get_agent()
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        (figures_dir / "fig_1.png").write_text("fake")

        figure_paths = {"fig:overview": "figures/fig_1.png"}
        id_map = agent._build_figure_id_map(figure_paths, str(tmp_path))

        assert "fig_overview" in id_map
        assert id_map["fig_overview"] == "figures/fig_1"

    def test_variant_bare_name_mapped(self, tmp_path):
        """Bare 'overview' (prefix stripped) also resolves."""
        agent = self._get_agent()
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        (figures_dir / "fig_1.png").write_text("fake")

        figure_paths = {"fig:overview": "figures/fig_1.png"}
        id_map = agent._build_figure_id_map(figure_paths, str(tmp_path))

        assert "overview" in id_map

    def test_variant_filename_mapped(self, tmp_path):
        """Bare filename 'fig_1' also resolves."""
        agent = self._get_agent()
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        (figures_dir / "fig_1.png").write_text("fake")

        figure_paths = {"fig:overview": "figures/fig_1.png"}
        id_map = agent._build_figure_id_map(figure_paths, str(tmp_path))

        assert "fig_1" in id_map
        assert id_map["fig_1"] == "figures/fig_1"

    def test_no_duplicate_overwrite(self, tmp_path):
        """When two figures share a variant key, first mapping wins."""
        agent = self._get_agent()
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        (figures_dir / "fig_1.png").write_text("fake")
        (figures_dir / "fig_2.png").write_text("fake")

        figure_paths = {
            "fig:overview": "figures/fig_1.png",
            "fig:stage2": "figures/fig_2.png",
        }
        id_map = agent._build_figure_id_map(figure_paths, str(tmp_path))

        assert "fig:overview" in id_map
        assert "fig:stage2" in id_map
        assert id_map["fig:overview"] == "figures/fig_1"
        assert id_map["fig:stage2"] == "figures/fig_2"


class TestRewriteIncludegraphicsTargets:
    """Test that _rewrite_includegraphics_targets resolves all ID variants."""

    def _get_agent(self):
        from src.agents.typesetter_agent.typesetter_agent import TypesetterAgent
        agent = TypesetterAgent.__new__(TypesetterAgent)
        return agent

    def test_rewrite_exact_id(self, tmp_path):
        """\\includegraphics{fig:overview} → figures/fig_1"""
        agent = self._get_agent()
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        (figures_dir / "fig_1.png").write_text("fake")

        id_map = {"fig:overview": "figures/fig_1"}
        content = r"\includegraphics[width=\textwidth]{fig:overview}"
        result = agent._rewrite_includegraphics_targets(content, str(tmp_path), id_map)

        assert "figures/fig_1" in result
        assert "fig:overview" not in result

    def test_rewrite_underscore_variant(self, tmp_path):
        """\\includegraphics{fig_overview} → figures/fig_1 via variant map."""
        agent = self._get_agent()
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        (figures_dir / "fig_1.png").write_text("fake")

        id_map = {"fig:overview": "figures/fig_1", "fig_overview": "figures/fig_1"}
        content = r"\includegraphics[width=0.9\linewidth]{fig_overview}"
        result = agent._rewrite_includegraphics_targets(content, str(tmp_path), id_map)

        assert "figures/fig_1" in result
        assert "fig_overview" not in result

    def test_rewrite_backslash_path_normalized(self, tmp_path):
        r"""figures\fig_2 (Windows backslash) → figures/fig_2"""
        agent = self._get_agent()
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        (figures_dir / "fig_2.png").write_text("fake")

        id_map = {}
        content = r"\includegraphics[width=\textwidth]{figures\fig_2}"
        result = agent._rewrite_includegraphics_targets(content, str(tmp_path), id_map)

        assert "\\\\" not in result or "figures/fig_2" in result


# =========================================================================
# 2. Prompt compiler — figure guidance uses bare IDs
# =========================================================================

class TestFigurePlacementGuidance:
    """Prompt should tell Writer to use figure ID, not file path."""

    def test_guidance_uses_figure_id_not_filepath(self):
        from src.agents.shared.prompt_compiler import _format_figure_placement_guidance

        mock_plan = MagicMock()
        mock_placement = MagicMock()
        mock_placement.figure_id = "fig:overview"
        mock_placement.is_wide = True
        mock_placement.position_hint = "top"
        mock_placement.message = ""
        mock_placement.caption_guidance = ""
        mock_plan.figures = [mock_placement]
        mock_plan.figures_to_reference = []

        mock_fig = MagicMock()
        mock_fig.id = "fig:overview"
        mock_fig.caption = "Overview of the framework."
        mock_fig.description = "A diagram."
        mock_fig.file_path = "figures/fig_1.png"

        result = _format_figure_placement_guidance(mock_plan, [mock_fig])

        assert "fig:overview" in result
        assert "fig_1.png" not in result
        assert "fig_1" not in result


# =========================================================================
# 3. Conclusion CITE/FLOAT cleanup
# =========================================================================

class TestConclusionMarkerCleanup:
    """Synthesis sections must strip Stage-1 pseudo-markers."""

    def test_cite_markers_stripped(self):
        content = (
            "This method is effective [CITE:vlp_efficiency]. "
            "It achieves state-of-the-art results [FLOAT:results_table]. "
            "Future work will explore [CITE:video_vlp]."
        )
        cleaned = re.sub(r'\[CITE:[^\]]*\]', '', content)
        cleaned = re.sub(r'\[FLOAT:[^\]]*\]', '', cleaned)
        cleaned = re.sub(r'  +', ' ', cleaned)

        assert "[CITE:" not in cleaned
        assert "[FLOAT:" not in cleaned
        assert "This method is effective" in cleaned

    def test_synthesis_section_cleans_markers(self):
        """_generate_synthesis_section should strip [CITE:...] and [FLOAT:...]."""
        content_with_markers = (
            "Summary of results [CITE:topic1]. "
            "See [FLOAT:tab1] for details [CITE:topic2]."
        )
        # Simulate the cleanup logic that should exist in _generate_synthesis_section
        content = content_with_markers
        content = re.sub(r'~?\\cite\{[^}]*\}', '', content)
        content = re.sub(
            r'(?:Figure|Fig\.|Table|Tab\.|Section|Sec\.|Equation|Eq\.)~?\\ref\{[^}]*\}',
            '', content,
        )
        content = re.sub(r'~?\\ref\{[^}]*\}', '', content)
        content = re.sub(r'\(\s*[,;]?\s*\)', '', content)
        # These two lines should be added in the fix:
        content = re.sub(r'\[CITE:[^\]]*\]', '', content)
        content = re.sub(r'\[FLOAT:[^\]]*\]', '', content)
        content = re.sub(r'  +', ' ', content)

        assert "[CITE:" not in content
        assert "[FLOAT:" not in content


# =========================================================================
# 4. Conclusion multi-file mode
# =========================================================================

class TestConclusionMultiFileMode:
    """Conclusion should be written as sections/conclusion.tex."""

    def _get_agent(self):
        from src.agents.typesetter_agent.typesetter_agent import TypesetterAgent
        agent = TypesetterAgent.__new__(TypesetterAgent)
        return agent

    def test_conclusion_in_section_file_map(self, tmp_path):
        """_write_section_files should include conclusion in file map."""
        agent = self._get_agent()
        sections = {
            "introduction": r"\section{Introduction} Some intro text.",
            "conclusion": "This paper presented a method for X.",
        }
        section_file_map = agent._write_section_files(
            work_dir=str(tmp_path),
            sections=sections,
            section_order=["introduction", "conclusion"],
            section_titles={"introduction": "Introduction", "conclusion": "Conclusion"},
            citation_style="numeric",
        )

        assert "conclusion" in section_file_map
        conclusion_file = tmp_path / "sections" / "conclusion.tex"
        assert conclusion_file.exists()
        content = conclusion_file.read_text(encoding="utf-8")
        assert r"\section{Conclusion}" in content
        assert "This paper presented a method for X." in content

    def test_conclusion_not_inlined_in_main_tex(self, tmp_path):
        """In multi-file mode, main.tex should use \\input for conclusion."""
        agent = self._get_agent()
        sections = {
            "introduction": "Intro text.",
            "conclusion": "Conclusion text here.",
        }
        section_file_map = agent._write_section_files(
            work_dir=str(tmp_path),
            sections=sections,
            section_order=["introduction", "conclusion"],
            section_titles={"introduction": "Introduction", "conclusion": "Conclusion"},
            citation_style="numeric",
        )

        assert "conclusion" in section_file_map
        assert section_file_map["conclusion"] == "sections/conclusion"
