"""
OpenRouter + Gemini reasoning scratch script.

Pytest imports test modules during collection; keep all network calls under __main__
so CI does not hit the API (or fail on missing/invalid keys).
"""
from __future__ import annotations

import os
import sys

from openai import OpenAI


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY to run this script.", file=sys.stderr)
        raise SystemExit(1)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model="google/gemini-3-pro-preview",
        messages=[
            {
                "role": "user",
                "content": "How many r's are in the word 'strawberry'?",
            }
        ],
        extra_body={"reasoning": {"enabled": True}},
    )

    message = response.choices[0].message
    print(message)

    messages = [
        {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
        {
            "role": "assistant",
            "content": message.content,
            "reasoning_details": message.reasoning_details,
        },
        {"role": "user", "content": "Are you sure? Think carefully."},
    ]

    response2 = client.chat.completions.create(
        model="google/gemini-3-pro-preview",
        messages=messages,
        extra_body={"reasoning": {"enabled": True}},
    )
    print(response2)


if __name__ == "__main__":
    main()
