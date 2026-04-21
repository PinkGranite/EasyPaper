"""Schema coverage for FigureSpec AcademicDreamer fields."""

from src.agents.metadata_agent.models import FigureSpec


def test_figure_spec_preserves_generation_fields() -> None:
    fig = FigureSpec(
        id="fig:architecture",
        caption="Model overview.",
        description="Block diagram of the proposed method.",
        auto_generate=True,
        generation_prompt="Illustrate the model pipeline.",
        style="ICML-style diagram",
        target_type="architecture_diagram",
    )

    payload = fig.model_dump()

    assert payload["generation_prompt"] == "Illustrate the model pipeline."
    assert payload["style"] == "ICML-style diagram"
    assert payload["target_type"] == "architecture_diagram"
