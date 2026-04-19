"""Behavioral contract tests for metadata figure generation preprocessing."""

from __future__ import annotations

import importlib
import inspect
import sys
import types
from pathlib import Path

import pytest

from src.agents.metadata_agent.models import FigureSpec, PaperMetaData


def _load_figure_generation_module():
    return importlib.import_module("src.agents.metadata_agent.figure_generation")


def _get_preprocess_fn(module: object):
    for name in (
        "preprocess_generated_figures",
        "preprocess_figures",
        "materialize_generated_figures",
    ):
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    pytest.fail(
        "figure_generation module must expose a preprocessing entrypoint "
        "(preprocess_generated_figures/preprocess_figures/materialize_generated_figures)",
    )


async def _call_preprocess(fn, *, metadata: PaperMetaData, output_dir: str, results_dir: str):
    sig = inspect.signature(fn)
    kwargs = {}

    if "metadata" in sig.parameters:
        kwargs["metadata"] = metadata
    if "output_dir" in sig.parameters:
        kwargs["output_dir"] = output_dir
    if "results_dir" in sig.parameters:
        kwargs["results_dir"] = results_dir

    if "metadata" not in kwargs:
        result = fn(metadata, **kwargs)
    else:
        result = fn(**kwargs)

    if inspect.isawaitable(result):
        result = await result
    return result


class _DreamerResult(str):
    def __new__(cls, path: str) -> "_DreamerResult":
        return str.__new__(cls, path)

    def get(self, key: str, default: object = None) -> object:
        if key in {"output_path", "file_path", "path"}:
            return str(self)
        return default

    @property
    def output_path(self) -> str:
        return str(self)

    @property
    def file_path(self) -> str:
        return str(self)

    @property
    def path(self) -> str:
        return str(self)


@pytest.mark.asyncio
async def test_generated_figure_missing_dependency_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_figure_generation_module()
    preprocess = _get_preprocess_fn(module)

    metadata = PaperMetaData(
        title="Missing dependency",
        idea_hypothesis="Validate the optional dependency error.",
        method="Call preprocessing without academic_dreamer installed.",
        data="No external data required.",
        experiments="Expect a runtime failure.",
        references=[],
        figures=[
            FigureSpec(
                id="fig:auto",
                caption="Auto-generated figure.",
                auto_generate=True,
                generation_prompt="Create an academic diagram.",
                style="neurips",
                target_type="architecture_diagram",
            ),
        ],
    )

    monkeypatch.delitem(sys.modules, "academic_dreamer", raising=False)

    with pytest.raises(RuntimeError, match="academic_dreamer"):
        await _call_preprocess(
            preprocess,
            metadata=metadata,
            output_dir=str(tmp_path / "paper"),
            results_dir=str(tmp_path / "results"),
        )


@pytest.mark.asyncio
async def test_existing_figure_file_bypasses_generation_and_clears_auto_generate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_figure_generation_module()
    preprocess = _get_preprocess_fn(module)

    existing = tmp_path / "materials" / "figures" / "existing.png"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(b"existing image")

    calls = {"count": 0}

    async def _generate_academic_illustration(*args: object, **kwargs: object) -> _DreamerResult:
        calls["count"] += 1
        raise AssertionError("existing figure should bypass generation")

    monkeypatch.setitem(
        sys.modules,
        "academic_dreamer",
        types.SimpleNamespace(generate_academic_illustration=_generate_academic_illustration),
    )

    metadata = PaperMetaData(
        title="Existing figure bypass",
        idea_hypothesis="Validate bypass behavior.",
        method="Use a real existing file path.",
        data="No external data required.",
        experiments="Expect no dreamer calls.",
        references=[],
        materials_root=str(tmp_path / "materials"),
        figures=[
            FigureSpec(
                id="fig:existing",
                caption="Existing figure.",
                file_path="figures/existing.png",
                auto_generate=True,
                generation_prompt="Should be ignored because the file already exists.",
                style="cvpr",
                target_type="architecture_diagram",
            ),
        ],
    )

    await _call_preprocess(
        preprocess,
        metadata=metadata,
        output_dir=str(tmp_path / "paper"),
        results_dir=str(tmp_path / "results"),
    )

    assert calls["count"] == 0
    assert metadata.figures[0].file_path == "figures/existing.png"
    assert metadata.figures[0].auto_generate is False


@pytest.mark.asyncio
async def test_missing_generated_figure_uses_output_dir_fallback_and_materializes_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_figure_generation_module()
    preprocess = _get_preprocess_fn(module)

    async def _generate_academic_illustration(*args: object, **kwargs: object) -> _DreamerResult:
        explicit_output = kwargs.get("output_path")
        output_dir = kwargs.get("output_dir")
        if explicit_output:
            out_path = Path(str(explicit_output))
        elif output_dir:
            out_path = Path(str(output_dir)) / "generated_figures" / "dreamer-output.png"
        else:
            out_path = tmp_path / "generated_figures" / "dreamer-output.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"fake generated image")
        return _DreamerResult(str(out_path))

    monkeypatch.setitem(
        sys.modules,
        "academic_dreamer",
        types.SimpleNamespace(generate_academic_illustration=_generate_academic_illustration),
    )

    output_dir = tmp_path / "paper"
    results_dir = tmp_path / "results"

    metadata = PaperMetaData(
        title="Output-dir fallback",
        idea_hypothesis="Validate fallback-root assignment.",
        method="Generate a figure without materials_root.",
        data="No external data required.",
        experiments="Expect output_dir/generated_figures fallback.",
        references=[],
        figures=[
            FigureSpec(
                id="fig:generated",
                caption="Generated figure.",
                file_path="missing/original.png",
                auto_generate=True,
                generation_prompt="Create an architecture diagram.",
                style="icml",
                target_type="architecture_diagram",
            ),
        ],
    )

    await _call_preprocess(
        preprocess,
        metadata=metadata,
        output_dir=str(output_dir),
        results_dir=str(results_dir),
    )

    figure = metadata.figures[0]
    assert metadata.materials_root == str(output_dir)
    assert figure.auto_generate is False
    assert figure.file_path
    materialized_path = Path(metadata.materials_root) / figure.file_path
    assert materialized_path.is_file()
