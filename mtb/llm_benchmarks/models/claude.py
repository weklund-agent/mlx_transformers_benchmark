from typing import Any

from mtb.llm_benchmarks.models.base import ModelSpec

__all__ = [
    "Claude_Opus_4_6",
]


def format_claude_prompt(prompt: str) -> Any:
    """Standard system/user format for Claude models."""
    messages = [
        {
            "role": "system",
            "content": "You are Claude, made by Anthropic. You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    return messages


Claude_Opus_4_6 = ModelSpec(
    name="claude-opus-4-6",
    num_params=0,  # unknown / not applicable for API models
    prompt_formatter=format_claude_prompt,
    thinking=True,
    model_ids={
        "anthropic": {
            "api": "claude-opus-4-6",
        },
    },
)
