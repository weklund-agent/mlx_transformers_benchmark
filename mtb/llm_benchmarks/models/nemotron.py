from typing import Any

from mtb.llm_benchmarks.models.base import ModelSpec

__all__ = [
    "Nemotron3_Nano_4B",
    "Nemotron_Nano_9B_v2",
    "Nemotron_Cascade2_30B_A3B",
]


def format_nemotron_prompt(prompt: str) -> Any:
    """Nemotron models use standard chat format with system message."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return messages


Nemotron3_Nano_4B = ModelSpec(
    name="nemotron-3-nano-4b",
    num_params=int(4e9),
    prompt_formatter=format_nemotron_prompt,
    thinking=False,
    model_ids={
        "mlx": {
            "int4": "models/Nemotron-3-Nano-4B-4bit",
            "int8": "models/Nemotron-3-Nano-4B-8bit",
        },
        "ollama": {
            "int4": "nemotron-3-nano:4b",
        },
    },
)


Nemotron_Nano_9B_v2 = ModelSpec(
    name="nemotron-nano-9b-v2",
    num_params=int(9e9),
    prompt_formatter=format_nemotron_prompt,
    thinking=True,
    model_ids={
        "mlx": {
            "int4": "mlx-community/NVIDIA-Nemotron-Nano-9B-v2-4bits",
            "int8": "mlx-community/NVIDIA-Nemotron-Nano-9B-v2-8bit",
        },
    },
)


Nemotron_Cascade2_30B_A3B = ModelSpec(
    name="nemotron-cascade-2-30b-a3b",
    num_params=int(30e9),
    prompt_formatter=format_nemotron_prompt,
    thinking=True,
    model_ids={
        "mlx": {
            "int4": "models/Nemotron-Cascade-2-30B-A3B-4bit",
            "int8": "models/Nemotron-Cascade-2-30B-A3B-8bit",
        },
    },
)
