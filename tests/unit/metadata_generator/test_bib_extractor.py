"""Tests for BibExtractor."""
import pytest
from pathlib import Path

from src.agents.metadata_agent.metadata_generator.models import FileCategory, ExtractedFragment
from src.agents.metadata_agent.metadata_generator.extractors.bib_extractor import BibExtractor


@pytest.fixture
def standard_bib(tmp_path: Path) -> Path:
    content = (
        '@article{smith2024,\n'
        '  title={Deep Learning for Vision},\n'
        '  author={Smith, A and Doe, B},\n'
        '  journal={NeurIPS},\n'
        '  year={2024}\n'
        '}\n'
        '\n'
        '@inproceedings{jones2023,\n'
        '  title={Robust Training},\n'
        '  author={Jones, C},\n'
        '  booktitle={ICML},\n'
        '  year={2023}\n'
        '}\n'
    )
    f = tmp_path / "refs.bib"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def messy_bib(tmp_path: Path) -> Path:
    content = (
        '% A comment line\n'
        '@article{key1, title = {Title One},\n'
        'author={Auth},year={2020}}\n'
        '\n'
        '  @book{key2,\n'
        '    title = {Book Title},\n'
        '    author = {Book Author},\n'
        '    year = {2019}\n'
        '  }\n'
    )
    f = tmp_path / "messy.bib"
    f.write_text(content, encoding="utf-8")
    return f


class TestBibExtractor:
    def test_parse_standard_bib(self, standard_bib: Path):
        ext = BibExtractor()
        fragments = ext.extract(str(standard_bib))
        assert len(fragments) == 2
        assert all(f.file_category == FileCategory.BIB for f in fragments)
        assert all(f.metadata_field == "references" for f in fragments)
        assert "smith2024" in fragments[0].content
        assert "jones2023" in fragments[1].content

    def test_parse_messy_bib(self, messy_bib: Path):
        ext = BibExtractor()
        fragments = ext.extract(str(messy_bib))
        assert len(fragments) == 2
        assert any("key1" in f.content for f in fragments)
        assert any("key2" in f.content for f in fragments)

    def test_empty_bib(self, tmp_path: Path):
        f = tmp_path / "empty.bib"
        f.write_text("", encoding="utf-8")
        ext = BibExtractor()
        fragments = ext.extract(str(f))
        assert fragments == []

    def test_high_confidence(self, standard_bib: Path):
        ext = BibExtractor()
        fragments = ext.extract(str(standard_bib))
        for f in fragments:
            assert f.confidence >= 0.9
