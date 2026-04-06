"""Tests for auto-package detection and injection utilities."""
import pytest

from src.agents.shared.template_analyzer import (
    COMMAND_TO_PACKAGE,
    detect_missing_packages,
    inject_missing_packages,
)


class TestCommandToPackageMapping:
    def test_mapping_exists(self):
        assert isinstance(COMMAND_TO_PACKAGE, dict)
        assert "\\toprule" in COMMAND_TO_PACKAGE
        assert COMMAND_TO_PACKAGE["\\toprule"] == "booktabs"

    def test_common_mappings(self):
        m = COMMAND_TO_PACKAGE
        assert m.get("\\url") == "url"
        assert m.get("\\midrule") == "booktabs"
        assert m.get("\\bottomrule") == "booktabs"
        assert m.get("\\KwIn") == "algorithm2e"
        assert m.get("\\subfloat") == "subfig"


class TestDetectMissingPackages:
    def test_detect_booktabs_commands(self):
        preamble = r"\documentclass{article}" + "\n" + r"\usepackage{amsmath}"
        body = (
            r"\begin{tabular}{ll}"
            r"\toprule A & B \\ \midrule C & D \\ \bottomrule"
            r"\end{tabular}"
        )
        missing = detect_missing_packages(preamble, body)
        assert "booktabs" in missing

    def test_no_missing_when_package_loaded(self):
        preamble = r"\documentclass{article}" + "\n" + r"\usepackage{booktabs}"
        body = r"\toprule A \midrule B \bottomrule"
        missing = detect_missing_packages(preamble, body)
        assert "booktabs" not in missing

    def test_detect_url_missing(self):
        preamble = r"\documentclass{article}"
        body = r"Visit \url{https://example.com} for details."
        missing = detect_missing_packages(preamble, body)
        assert "url" in missing

    def test_url_not_missing_when_hyperref_loaded(self):
        preamble = (
            r"\documentclass{article}" + "\n"
            r"\usepackage{hyperref}"
        )
        body = r"Visit \url{https://example.com} for details."
        missing = detect_missing_packages(preamble, body)
        assert "url" not in missing

    def test_empty_body_no_missing(self):
        preamble = r"\documentclass{article}"
        missing = detect_missing_packages(preamble, "")
        assert missing == []


class TestInjectMissingPackages:
    def test_inject_before_begin_document(self):
        tex = (
            r"\documentclass{article}" + "\n"
            r"\usepackage{amsmath}" + "\n"
            r"\begin{document}" + "\n"
            r"\toprule" + "\n"
            r"\end{document}"
        )
        result = inject_missing_packages(tex, ["booktabs"])
        assert r"\usepackage{booktabs}" in result
        assert result.index(r"\usepackage{booktabs}") < result.index(
            r"\begin{document}"
        )

    def test_inject_no_duplicates(self):
        tex = (
            r"\documentclass{article}" + "\n"
            r"\usepackage{booktabs}" + "\n"
            r"\begin{document}" + "\n"
            r"\toprule" + "\n"
            r"\end{document}"
        )
        result = inject_missing_packages(tex, ["booktabs"])
        assert result.count(r"\usepackage{booktabs}") == 1

    def test_inject_empty_list_unchanged(self):
        tex = (
            r"\documentclass{article}" + "\n"
            r"\begin{document}" + "\n"
            "Hello" + "\n"
            r"\end{document}"
        )
        result = inject_missing_packages(tex, [])
        assert result == tex
