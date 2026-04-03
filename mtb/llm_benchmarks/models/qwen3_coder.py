from typing import Any

from mtb.llm_benchmarks.models.base import ModelSpec

__all__ = [
    "Qwen3_Coder_30B_A3B",
]


def format_qwen3_coder_prompt(prompt: str) -> Any:
    """Qwen3-Coder uses the same prompt format as Qwen3."""
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    return messages


Qwen3_Coder_30B_A3B = ModelSpec(
    name="qwen3-coder-30b-a3b",
    num_params=3e9,
    prompt_formatter=format_qwen3_coder_prompt,
    thinking=True,
    model_ids={
        "mlx": {
            "int4": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
            "int8": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit",
        },
    },
)
