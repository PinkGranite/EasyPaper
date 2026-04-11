"""Tests for CodeExtractor."""
import pytest
from pathlib import Path

from src.agents.metadata_agent.metadata_generator.models import FileCategory, ExtractedFragment
from src.agents.metadata_agent.metadata_generator.extractors.code_extractor import CodeExtractor


@pytest.fixture
def method_code(tmp_path: Path) -> Path:
    content = (
        'import torch\n'
        '\n'
        'class TransformerModel(torch.nn.Module):\n'
        '    """Main model architecture."""\n'
        '    def __init__(self, d_model=512):\n'
        '        super().__init__()\n'
        '        self.encoder = torch.nn.TransformerEncoder(...)\n'
        '\n'
        '    def forward(self, x):\n'
        '        return self.encoder(x)\n'
    )
    f = tmp_path / "model.py"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture
def experiment_code(tmp_path: Path) -> Path:
    content = (
        '# Training and evaluation experiment script\n'
        'DATASET = "cifar10"\n'
        'SEED = 42\n'
        'EPOCHS = 100\n'
        '\n'
        'def train(dataset, epochs=100, seed=42):\n'
        '    """Train and evaluate experiment."""\n'
        '    set_seed(seed)\n'
        '    for epoch in range(epochs):\n'
        '        train_loss = run_epoch(dataset)\n'
        '\n'
        'def evaluate(test_data):\n'
        '    accuracy = compute_metric(test_data)\n'
        '    f1 = compute_f1(test_data)\n'
        '    return {"accuracy": accuracy, "f1": f1}\n'
        '\n'
        'def ablation_study(config):\n'
        '    """Ablation experiment with different hyperparameter settings."""\n'
        '    results = benchmark(config)\n'
        '    return results\n'
    )
    f = tmp_path / "train.py"
    f.write_text(content, encoding="utf-8")
    return f


class TestCodeExtractor:
    def test_extract_method_code(self, method_code: Path):
        ext = CodeExtractor()
        fragments = ext.extract(str(method_code))
        assert len(fragments) >= 1
        assert all(f.file_category == FileCategory.CODE for f in fragments)
        method_frags = [f for f in fragments if f.metadata_field == "method"]
        assert len(method_frags) >= 1

    def test_extract_experiment_code(self, experiment_code: Path):
        ext = CodeExtractor()
        fragments = ext.extract(str(experiment_code))
        assert len(fragments) >= 1
        exp_frags = [f for f in fragments if f.metadata_field == "experiments"]
        assert len(exp_frags) >= 1

    def test_symbols_in_extra(self, method_code: Path):
        ext = CodeExtractor()
        fragments = ext.extract(str(method_code))
        for f in fragments:
            assert "symbols" in f.extra

    def test_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.py"
        f.write_text("", encoding="utf-8")
        ext = CodeExtractor()
        fragments = ext.extract(str(f))
        assert fragments == []
