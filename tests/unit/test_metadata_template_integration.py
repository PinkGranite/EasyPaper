"""Tests for TemplateAnalyzer.analyze_zip used by MetaDataAgent pipeline."""
import pytest
import zipfile

from src.agents.shared.template_analyzer import TemplateAnalyzer, TemplateWriterGuide


class TestTemplateAnalyzerFromZip:
    def test_analyze_zip_extracts_packages(self, tmp_path):
        tex_content = r"""
\documentclass{article}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{natbib}
\begin{document}
Hello
\end{document}
"""
        zip_path = tmp_path / "template.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("main.tex", tex_content)

        guide = TemplateAnalyzer.analyze_zip(str(zip_path))
        assert isinstance(guide, TemplateWriterGuide)
        assert "amsmath" in guide.available_packages
        assert "booktabs" in guide.available_packages
        assert guide.citation_style == "citep"

    def test_analyze_zip_no_main_tex_returns_empty(self, tmp_path):
        zip_path = tmp_path / "empty.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("readme.txt", "No tex here")

        guide = TemplateAnalyzer.analyze_zip(str(zip_path))
        assert guide.available_packages == []

    def test_analyze_zip_nonexistent_returns_empty(self):
        guide = TemplateAnalyzer.analyze_zip("/nonexistent/path.zip")
        assert guide.available_packages == []

    def test_analyze_zip_nested_main_tex(self, tmp_path):
        tex_content = r"""
\documentclass[twocolumn]{article}
\usepackage{algorithm2e}
\begin{document}
\end{document}
"""
        zip_path = tmp_path / "nested.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("submission/main.tex", tex_content)

        guide = TemplateAnalyzer.analyze_zip(str(zip_path))
        assert "algorithm2e" in guide.available_packages
        assert guide.column_format == "double"

    def test_analyze_zip_prefers_main_tex_over_other(self, tmp_path):
        main_content = r"""
\documentclass{article}
\usepackage{booktabs}
\begin{document}
\end{document}
"""
        other_content = r"""
\documentclass{article}
\usepackage{xcolor}
\begin{document}
\end{document}
"""
        zip_path = tmp_path / "multi.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("sample.tex", other_content)
            zf.writestr("main.tex", main_content)

        guide = TemplateAnalyzer.analyze_zip(str(zip_path))
        assert "booktabs" in guide.available_packages

    def test_analyze_zip_skips_macosx(self, tmp_path):
        tex_content = r"""
\documentclass{article}
\usepackage{amsmath}
\begin{document}
\end{document}
"""
        zip_path = tmp_path / "mac.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("__MACOSX/main.tex", "junk")
            zf.writestr("main.tex", tex_content)

        guide = TemplateAnalyzer.analyze_zip(str(zip_path))
        assert "amsmath" in guide.available_packages
