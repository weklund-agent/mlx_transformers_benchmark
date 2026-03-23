from typing import Any

from mtb.llm_benchmarks.models.base import ModelSpec

__all__ = [
    "Deepseek_R1_Distill_Qwen_7B",
    "Deepseek_R1_0528_Qwen3_8B",
]


def format_deepseek_prompt(prompt: str) -> Any:
    """Deepseek models expect a regular system prompt."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return messages


Deepseek_R1_Distill_Qwen_7B = ModelSpec(
    name="Deepseek-R1-Distill-7B",
    num_params=int(7e9),
    prompt_formatter=format_deepseek_prompt,
    thinking=True,
    model_ids={
        "mlx": {
            "int4": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
        },
        "lmstudio": {
            "int4": "lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        },
        "lmstudio_mlx": {
            "int4": "deepseek-r1-distill-qwen-7b",
        },
        "ollama": {
            "int4": "deepseek-r1:7b",
        },
    },
)


Deepseek_R1_0528_Qwen3_8B = ModelSpec(
    name="Deepseek-R1-0528_Qwen3-8B",
    num_params=int(13e9),
    prompt_formatter=format_deepseek_prompt,
    thinking=True,
    model_ids={
        "mlx": {
            "int4": "mlx-community/DeepSeek-R1-0528-Qwen3-8B-4bit",
            "int8": "mlx-community/DeepSeek-R1-0528-Qwen3-8B-8bit",
            "bfloat16": "mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16",
        },
        "lmstudio": {
            "int4": "deepseek/deepseek-r1-0528-qwen3-8b:deepseek-r1-0528-qwen3-8b@q4_k_m",
            "int8": "deepseek/deepseek-r1-0528-qwen3-8b:deepseek-r1-0528-qwen3-8b@q8_0",
        },
        "lmstudio_mlx": {
            "int4": "deepseek-r1-0528-qwen3-8b-mlx@4bit",
            "int8": "deepseek-r1-0528-qwen3-8b-mlx@8bit",
        },
        "ollama": {
            "int4": "deepseek-r1:8b",
        },
        "torch": {
            "bfloat16": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        },
    },
)
