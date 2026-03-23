from typing import Any

from mtb.llm_benchmarks.models.base import ModelSpec

__all__ = [
    "Qwen3p5_0p8B",
    "Qwen3p5_2B",
    "Qwen3p5_4B",
    "Qwen3p5_9B",
    "Qwen3p5_27B",
    "Qwen3p5_35B_A3B",
    "Qwen3p5_27B_Claude_Opus_Distilled",
]


def format_qwen35_prompt(prompt: str) -> Any:
    """Qwen3.5 models use the same prompt format as Qwen3."""
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


Qwen3p5_0p8B = ModelSpec(
    name="qwen-3.5-0.8b",
    num_params=int(8e8),
    prompt_formatter=format_qwen35_prompt,
    thinking=True,
    model_ids={
        "mlx": {
            "int4": "models/Qwen3.5-0.8B-4bit",
            "int8": "models/Qwen3.5-0.8B-8bit",
        },
    },
)


Qwen3p5_2B = ModelSpec(
    name="qwen-3.5-2b",
    num_params=int(2e9),
    prompt_formatter=format_qwen35_prompt,
    thinking=True,
    model_ids={
        "mlx": {
            "int4": "models/Qwen3.5-2B-4bit",
            "int8": "models/Qwen3.5-2B-8bit",
            "bfloat16": "models/Qwen3.5-2B-bf16",
        },
    },
)


Qwen3p5_4B = ModelSpec(
    name="qwen-3.5-4b",
    num_params=int(4e9),
    prompt_formatter=format_qwen35_prompt,
    thinking=True,
    model_ids={
        "mlx": {
            "int4": "models/Qwen3.5-4B-4bit",
            "int8": "models/Qwen3.5-4B-8bit",
            "bfloat16": "models/Qwen3.5-4B-bf16",
        },
    },
)


Qwen3p5_9B = ModelSpec(
    name="qwen-3.5-9b",
    num_params=int(9e9),
    prompt_formatter=format_qwen35_prompt,
    thinking=True,
    model_ids={
        "mlx": {
            "int4": "models/Qwen3.5-9B-4bit",
            "int8": "models/Qwen3.5-9B-8bit",
            "bfloat16": "models/Qwen3.5-9B-bf16",
        },
    },
)


Qwen3p5_27B = ModelSpec(
    name="qwen-3.5-27b",
    num_params=int(27e9),
    prompt_formatter=format_qwen35_prompt,
    thinking=True,
    model_ids={
        "mlx": {
            "int4": "models/Qwen3.5-27B-4bit",
            "int8": "models/Qwen3.5-27B-8bit",
        },
    },
)


Qwen3p5_35B_A3B = ModelSpec(
    name="qwen-3.5-35b-a3b",
    num_params=int(35e9),
    prompt_formatter=format_qwen35_prompt,
    thinking=True,
    model_ids={
        "mlx": {
            "int4": "models/Qwen3.5-35B-A3B-4bit",
            "int8": "models/Qwen3.5-35B-A3B-8bit",
        },
    },
)


Qwen3p5_27B_Claude_Opus_Distilled = ModelSpec(
    name="qwen-3.5-27b-claude-opus-distilled",
    num_params=int(27e9),
    prompt_formatter=format_qwen35_prompt,
    thinking=True,
    model_ids={
        "mlx": {
            "int4": "models/Qwen3.5-27B-Claude-Opus-Distilled-4bit",
        },
    },
)
