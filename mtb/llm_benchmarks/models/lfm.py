from typing import Any

from mtb.llm_benchmarks.models.base import ModelSpec

__all__ = [
    "LFM2_24B_A2B",
]


def format_lfm_prompt(prompt: str) -> Any:
    """LFM2 models use standard system/user prompt format."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return messages


LFM2_24B_A2B = ModelSpec(
    name="lfm2-24b-a2b",
    num_params=2e9,
    prompt_formatter=format_lfm_prompt,
    model_ids={
        "mlx": {
            "int4": "LiquidAI/LFM2-24B-A2B-MLX-4bit",
            "int8": "LiquidAI/LFM2-24B-A2B-MLX-8bit",
        },
    },
)
