"""
EasyPaper SDK Usage Demo

Demonstrates both one-shot and streaming paper generation.
Before running, create a config.yaml with your API keys (see config.example.yaml).

Usage::

    python examples/sdk_demo.py
"""
import asyncio
from pathlib import Path

from easypaper import EasyPaper, PaperMetaData, EventType


async def demo_oneshot():
    """One-shot mode: wait for the full result."""

    config_path = Path(__file__).parent / "config.example.yaml"
    ep = EasyPaper(config_path=str(config_path))

    metadata = PaperMetaData(
        title="Attention Is All You Need: A Revisit",
        idea_hypothesis=(
            "Transformer architectures with pure self-attention mechanisms "
            "can replace recurrent and convolutional layers entirely, achieving "
            "superior performance on sequence-to-sequence tasks while being "
            "more parallelizable and requiring significantly less training time."
        ),
        method=(
            "We propose the Transformer, a model architecture eschewing recurrence "
            "and instead relying entirely on an attention mechanism to draw global "
            "dependencies between input and output."
        ),
        data=(
            "We train on the standard WMT 2014 English-German dataset consisting "
            "of about 4.5 million sentence pairs."
        ),
        experiments=(
            "On the WMT 2014 English-to-German translation task, the big transformer "
            "model outperforms the best previously reported models by more than 2.0 BLEU."
        ),
    )

    print("=" * 60)
    print("  EasyPaper — One-Shot Generation Demo")
    print("=" * 60)

    result = await ep.generate(metadata, compile_pdf=False)

    print(f"Status:      {result.status}")
    print(f"Title:       {result.paper_title}")
    print(f"Word count:  {result.total_word_count}")
    print(f"Sections:    {len(result.sections)}")
    if result.output_path:
        print(f"Output:      {result.output_path}")


async def demo_stream():
    """Streaming mode: observe every generation phase in real time."""

    config_path = Path(__file__).parent / "config.example.yaml"
    ep = EasyPaper(config_path=str(config_path))

    metadata = PaperMetaData(
        title="Attention Is All You Need: A Revisit",
        idea_hypothesis="Transformers replace RNNs with self-attention.",
        method="Pure attention mechanism for sequence-to-sequence.",
        data="WMT 2014 English-German dataset.",
        experiments="Outperforms prior models by 2+ BLEU.",
    )

    print("=" * 60)
    print("  EasyPaper — Streaming Generation Demo")
    print("=" * 60)
    print()

    async for event in ep.generate_stream(metadata, compile_pdf=False):
        etype = event.get("type", "")

        if etype == EventType.PHASE_START:
            print(f"  > [{event.get('phase', '')}] {event.get('message', '')}")

        elif etype == EventType.PHASE_COMPLETE:
            print(f"  * [{event.get('phase', '')}] {event.get('message', '')}")
            print()

        elif etype == EventType.SECTION_CONTENT:
            print(f"    Section done: {event.get('phase', '')}")

        elif etype == EventType.ERROR:
            print(f"  ! ERROR: {event.get('message', '')}")

        elif etype == EventType.COMPLETED:
            print("-" * 60)
            print("Generation complete.")


if __name__ == "__main__":
    asyncio.run(demo_stream())
