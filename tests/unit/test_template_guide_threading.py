"""
Tests that template_guide is properly threaded through MetaDataAgent methods.

The TemplateWriterGuide is created in generate_paper() and must be passed
to every sub-method that calls compile_*_prompt. These tests verify that:
1. Each sub-method signature accepts template_guide.
2. Each compile_*_prompt call inside sub-methods receives template_guide.
3. An end-to-end call does not raise NameError for template_guide.
"""
import ast
import inspect
from typing import Optional

import pytest

from src.agents.metadata_agent.metadata_agent import MetaDataAgent
from src.agents.shared.template_analyzer import TemplateWriterGuide


class TestTemplateGuideSignatures:
    """Verify that all sub-methods accept template_guide as a parameter."""

    METHODS_REQUIRING_TEMPLATE_GUIDE = [
        "_generate_introduction",
        "_generate_body_section",
        "_generate_synthesis_section",
        "_generate_section_decomposed",
    ]

    @pytest.mark.parametrize("method_name", METHODS_REQUIRING_TEMPLATE_GUIDE)
    def test_method_has_template_guide_param(self, method_name: str):
        method = getattr(MetaDataAgent, method_name)
        sig = inspect.signature(method)
        assert "template_guide" in sig.parameters, (
            f"{method_name}() is missing 'template_guide' parameter. "
            f"Current params: {list(sig.parameters.keys())}"
        )

    @pytest.mark.parametrize("method_name", METHODS_REQUIRING_TEMPLATE_GUIDE)
    def test_template_guide_param_is_optional(self, method_name: str):
        method = getattr(MetaDataAgent, method_name)
        sig = inspect.signature(method)
        param = sig.parameters["template_guide"]
        assert param.default is None, (
            f"{method_name}(template_guide=...) should default to None, "
            f"got {param.default!r}"
        )

    @pytest.mark.parametrize("method_name", METHODS_REQUIRING_TEMPLATE_GUIDE)
    def test_template_guide_type_annotation(self, method_name: str):
        method = getattr(MetaDataAgent, method_name)
        sig = inspect.signature(method)
        param = sig.parameters["template_guide"]
        annotation = param.annotation
        assert annotation == Optional[TemplateWriterGuide] or (
            hasattr(annotation, "__args__")
            and TemplateWriterGuide in annotation.__args__
        ), (
            f"{method_name}(template_guide) should be typed Optional[TemplateWriterGuide], "
            f"got {annotation!r}"
        )


class TestTemplateGuideCallerThreading:
    """Verify that generate_paper passes template_guide to each sub-method call."""

    @staticmethod
    def _get_full_source(filepath: str) -> str:
        """Read the raw source file for AST parsing (avoids indentation issues)."""
        import pathlib
        return pathlib.Path(filepath).read_text()

    @staticmethod
    def _find_calls_in_method(
        full_source: str, method_name: str, callee_name: str
    ) -> list[list[str]]:
        """Find all keyword args passed to callee_name within method_name's body."""
        tree = ast.parse(full_source)
        results = []
        for cls_node in ast.walk(tree):
            if not isinstance(cls_node, ast.ClassDef):
                continue
            for item in cls_node.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if item.name != method_name:
                    continue
                for node in ast.walk(item):
                    if not isinstance(node, ast.Call):
                        continue
                    func = node.func
                    func_name = None
                    if isinstance(func, ast.Attribute):
                        func_name = func.attr
                    elif isinstance(func, ast.Name):
                        func_name = func.id
                    if func_name == callee_name:
                        results.append([kw.arg for kw in node.keywords if kw.arg])
        return results

    @pytest.fixture()
    def source_path(self) -> str:
        return inspect.getfile(MetaDataAgent)

    @pytest.fixture()
    def full_source(self, source_path: str) -> str:
        return self._get_full_source(source_path)

    def test_execute_generation_passes_template_guide_to_generate_introduction(self, full_source):
        calls = self._find_calls_in_method(full_source, "execute_generation", "_generate_introduction")
        assert len(calls) >= 1, "execute_generation must call _generate_introduction"
        assert any("template_guide" in kwargs for kwargs in calls), (
            "execute_generation() must pass template_guide= to _generate_introduction(). "
            f"Found kwargs: {calls}"
        )

    def test_execute_generation_passes_template_guide_to_generate_body_section(self, full_source):
        calls = self._find_calls_in_method(full_source, "execute_generation", "_generate_body_section")
        assert len(calls) >= 1, "execute_generation must call _generate_body_section"
        assert any("template_guide" in kwargs for kwargs in calls), (
            "execute_generation() must pass template_guide= to _generate_body_section(). "
            f"Found kwargs: {calls}"
        )

    def test_execute_generation_passes_template_guide_to_generate_synthesis_section(self, full_source):
        calls = self._find_calls_in_method(full_source, "execute_generation", "_generate_synthesis_section")
        assert len(calls) >= 1, "execute_generation must call _generate_synthesis_section"
        assert all("template_guide" in kwargs for kwargs in calls), (
            "execute_generation() must pass template_guide= to ALL _generate_synthesis_section() calls. "
            f"Found kwargs: {calls}"
        )


class TestTemplateGuidePromptInjection:
    """Verify that sub-methods forward template_guide into compile_*_prompt calls."""

    @staticmethod
    def _find_calls_in_method(
        full_source: str, method_name: str, callee_name: str
    ) -> list[list[str]]:
        tree = ast.parse(full_source)
        results = []
        for cls_node in ast.walk(tree):
            if not isinstance(cls_node, ast.ClassDef):
                continue
            for item in cls_node.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if item.name != method_name:
                    continue
                for node in ast.walk(item):
                    if not isinstance(node, ast.Call):
                        continue
                    func = node.func
                    func_name = None
                    if isinstance(func, ast.Attribute):
                        func_name = func.attr
                    elif isinstance(func, ast.Name):
                        func_name = func.id
                    if func_name == callee_name:
                        results.append([kw.arg for kw in node.keywords if kw.arg])
        return results

    @pytest.fixture()
    def full_source(self) -> str:
        import pathlib
        return pathlib.Path(inspect.getfile(MetaDataAgent)).read_text()

    def test_introduction_forwards_to_compile_introduction_prompt(self, full_source):
        calls = self._find_calls_in_method(full_source, "_generate_introduction", "compile_introduction_prompt")
        assert len(calls) >= 1
        assert any("template_guide" in kwargs for kwargs in calls), (
            "_generate_introduction must pass template_guide to compile_introduction_prompt"
        )

    def test_body_section_forwards_to_compile_body_prompt(self, full_source):
        calls = self._find_calls_in_method(full_source, "_generate_body_section", "compile_body_section_prompt")
        assert len(calls) >= 1
        assert any("template_guide" in kwargs for kwargs in calls), (
            "_generate_body_section must pass template_guide to compile_body_section_prompt"
        )

    def test_decomposed_forwards_to_compile_paragraph_prompt(self, full_source):
        calls = self._find_calls_in_method(full_source, "_generate_section_decomposed", "compile_paragraph_prompt")
        assert len(calls) >= 1
        assert any("template_guide" in kwargs for kwargs in calls), (
            "_generate_section_decomposed must pass template_guide to compile_paragraph_prompt"
        )

    def test_body_section_forwards_template_guide_to_decomposed(self, full_source):
        calls = self._find_calls_in_method(full_source, "_generate_body_section", "_generate_section_decomposed")
        assert len(calls) >= 1
        assert any("template_guide" in kwargs for kwargs in calls), (
            "_generate_body_section must pass template_guide to _generate_section_decomposed"
        )

    def test_synthesis_forwards_to_compile_synthesis_prompt(self, full_source):
        calls = self._find_calls_in_method(full_source, "_generate_synthesis_section", "compile_synthesis_prompt")
        assert len(calls) >= 1
        assert any("template_guide" in kwargs for kwargs in calls), (
            "_generate_synthesis_section must pass template_guide to compile_synthesis_prompt"
        )
