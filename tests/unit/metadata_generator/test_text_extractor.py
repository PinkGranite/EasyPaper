"""Tests for TextExtractor."""
import pytest
from pathlib import Path

from src.agents.metadata_agent.metadata_generator.models import FileCategory, ExtractedFragment
from src.agents.metadata_agent.metadata_generator.extractors.text_extractor import TextExtractor


@pytest.fixture
def markdown_file(tmp_path: Path) -> Path:
    content = (
        "# Research Hypothesis\n"
        "We hypothesize that attention-based models outperform CNNs.\n\n"
        "# Method\n"
        "We use a transformer architecture with 12 layers.\n\n"
        "# Results\n"
        "Our model achieves 95% accuracy on CIFAR-10.\n"
    )
    f = tmp_path / "notes.md"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def plain_text_file(tmp_path: Path) -> Path:
    content = "This is a plain text research note about our experiment.\n"
    f = tmp_path / "ideas.txt"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def latex_file(tmp_path: Path) -> Path:
    content = (
        "\\section{Introduction}\n"
        "We propose a novel approach to image classification.\n\n"
        "\\section{Method}\n"
        "Our method uses self-attention mechanisms.\n"
    )
    f = tmp_path / "draft.tex"
    f.write_text(content, encoding="utf-8")
    return f


class TestTextExtractor:
    def test_markdown_extraction(self, markdown_file: Path):
        ext = TextExtractor()
        fragments = ext.extract(str(markdown_file))
        assert len(fragments) >= 1
        assert all(f.file_category == FileCategory.TEXT for f in fragments)
        full_text = " ".join(f.content for f in fragments)
        assert "hypothesize" in full_text or "attention" in full_text

    def test_markdown_sections_split(self, markdown_file: Path):
        ext = TextExtractor()
        fragments = ext.extract(str(markdown_file))
        # Should produce multiple fragments from different sections
        assert len(fragments) >= 2

    def test_plain_text_extraction(self, plain_text_file: Path):
        ext = TextExtractor()
        fragments = ext.extract(str(plain_text_file))
        assert len(fragments) == 1
        assert "experiment" in fragments[0].content

    def test_latex_extraction(self, latex_file: Path):
        ext = TextExtractor()
        fragments = ext.extract(str(latex_file))
        assert len(fragments) >= 1
        full_text = " ".join(f.content for f in fragments)
        assert "classification" in full_text or "self-attention" in full_text

    def test_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.md"
        f.write_text("", encoding="utf-8")
        ext = TextExtractor()
        fragments = ext.extract(str(f))
        assert fragments == []
