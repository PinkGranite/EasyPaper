#!/usr/bin/env python3
"""
End-to-end SDK test: load metadata JSON and run paper generation via package mode.

Usage:
    python -u e2e_sdk_test.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

from easypaper import EasyPaper, PaperMetaData, EventType


async def main():
    meta_path = Path("examples/nature_health_meta/meta_health.json")
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found")
        sys.exit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Build PaperMetaData from JSON (handle figures/tables if present)
    figures = raw.pop("figures", [])
    tables = raw.pop("tables", [])
    
    # Extract generation options (not part of PaperMetaData)
    enable_vlm_review = raw.pop("enable_vlm_review", False)
    max_review_iterations = raw.pop("max_review_iterations", 1)
    max_review_iterations = 1  # forced for quick E2E
    save_output = raw.pop("save_output", True)
    output_dir = raw.pop("output_dir", None)

    metadata = PaperMetaData(
        **raw,
        figures=figures,
        tables=tables,
    )

    print("=" * 70)
    print("  EasyPaper SDK E2E Test — Package Mode")
    print("=" * 70)
    print(f"  Title:    {metadata.title}")
    print(f"  Template: {metadata.template_path}")
    print(f"  Style:    {metadata.style_guide}")
    print(f"  Figures:  {len(metadata.figures)}")
    print(f"  Tables:   {len(metadata.tables)}")
    print("=" * 70)
    print()

    ep = EasyPaper()  # uses AGENT_CONFIG_PATH from .env

    print("[Streaming mode] Starting generation...\n")
    event_count = 0
    async for event in ep.generate_stream(
        metadata,
        compile_pdf=True,
        enable_review=True,
        max_review_iterations=max_review_iterations,
        enable_vlm_review=enable_vlm_review,
        save_output=save_output,
        output_dir=output_dir,
    ):
        event_count += 1
        etype = event.get("type", "")
        phase = event.get("phase", "")
        message = event.get("message", "")

        if etype == EventType.GENERATION_STARTED:
            print(f">>> GENERATION STARTED: {message}")
        elif etype == EventType.PHASE_START:
            print(f"  > [{phase}] {message}")
        elif etype == EventType.PHASE_COMPLETE:
            print(f"  * [{phase}] {message}")
            print()
        elif etype == EventType.SECTION_START:
            section = event.get("section", phase)
            print(f"    >> Section: {section} — {message}")
        elif etype == EventType.SECTION_CONTENT:
            wc = event.get("word_count", "?")
            print(f"    << Section done ({wc} words)")
        elif etype == EventType.THINKING:
            print(f"    ... thinking: {message[:80]}")
        elif etype == EventType.PLAN_CREATED:
            print(f"  * Plan created: {message}")
        elif etype == EventType.REVIEW_START:
            print(f"    >> Review: {message}")
        elif etype == EventType.REVIEW_RESULT:
            print(f"    << Review result: {message[:100]}")
        elif etype == EventType.COMPILE_START:
            print(f"  > Compiling PDF...")
        elif etype == EventType.COMPILE_COMPLETE:
            print(f"  * PDF compiled: {message}")
        elif etype == EventType.ERROR:
            print(f"  !! ERROR: {message}")
        elif etype == EventType.COMPLETED:
            print(f">>> COMPLETED: {message}")
        elif etype == EventType.LOG:
            pass  # skip verbose logs
        else:
            print(f"    [{etype}] {message[:80]}" if message else f"    [{etype}]")

    print()
    print("-" * 70)
    print(f"Total events received: {event_count}")
    print("E2E SDK test finished.")


if __name__ == "__main__":
    asyncio.run(main())
