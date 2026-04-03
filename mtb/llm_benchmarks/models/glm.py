from typing import Any

from mtb.llm_benchmarks.models.base import ModelSpec

__all__ = [
    "GLM4p7_Flash",
]


def format_glm_prompt(prompt: str) -> Any:
    """GLM models use standard system/user prompt format."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return messages


GLM4p7_Flash = ModelSpec(
    name="glm-4.7-flash",
    num_params=3e9,
    prompt_formatter=format_glm_prompt,
    model_ids={
        "mlx": {
            "int4": "mlx-community/GLM-4.7-Flash-4bit",
            "int8": "mlx-community/GLM-4.7-Flash-8bit",
        },
    },
)
