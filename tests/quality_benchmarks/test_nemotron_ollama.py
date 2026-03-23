"""Diagnostic test: Run Nemotron-3-Nano-4B through Ollama to determine if the
quality issues we see are specific to the MLX conversion or inherent to the model.

If Ollama (which uses its own GGUF quantization) produces coherent output for
the same trivial prompts that fail via mlx-lm, the problem is mlx-lm's
nemotron_h architecture support, not the model itself.

Requires: ollama running locally (`ollama serve`)
"""

import pytest

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

MODEL_ID = "nemotron-3-nano:4b"

pytestmark = pytest.mark.skipif(
    not OLLAMA_AVAILABLE,
    reason="ollama package not installed",
)


@pytest.fixture(scope="module")
def ensure_model():
    """Pull the model if not already available."""
    try:
        ollama.show(MODEL_ID)
    except ollama.ResponseError:
        pytest.skip(f"Model {MODEL_ID} not available and pull skipped (run `ollama pull {MODEL_ID}` first)")


def chat(prompt: str, max_tokens: int = 256) -> str:
    response = ollama.chat(
        MODEL_ID,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        options=dict(temperature=0.0, num_predict=max_tokens),
    )
    return response.message.content


class TestOllamaTrivialPrompts:
    """Same trivial prompts we tested with mlx-lm. If these pass here,
    the problem is mlx-lm, not the model."""

    def test_say_hello(self, ensure_model):
        result = chat("Say hello.", max_tokens=64)
        print(f"\n[ollama] 'Say hello.' -> '{result}'")
        assert len(result.strip()) > 2, f"Response too short: {result!r}"
        assert "hello" in result.lower() or "hi" in result.lower(), (
            f"Expected a greeting, got: {result!r}"
        )

    def test_arithmetic(self, ensure_model):
        result = chat("What is 2+2? Reply with just the number.", max_tokens=32)
        print(f"\n[ollama] 'What is 2+2?' -> '{result}'")
        assert "4" in result, f"Expected '4', got: {result!r}"

    def test_capital_of_france(self, ensure_model):
        result = chat("What is the capital of France? Reply with just the city name.", max_tokens=32)
        print(f"\n[ollama] 'Capital of France?' -> '{result}'")
        assert "paris" in result.lower(), f"Expected 'Paris', got: {result!r}"

    def test_fizzbuzz(self, ensure_model):
        result = chat(
            "Write a Python function that prints FizzBuzz for numbers 1 to 100.",
            max_tokens=512,
        )
        print(f"\n[ollama] FizzBuzz -> '{result[:300]}'")
        result_lower = result.lower()
        has_fizzbuzz = "fizzbuzz" in result_lower or "fizz buzz" in result_lower
        has_mod = "%" in result or "mod" in result_lower or "divisible" in result_lower
        assert has_fizzbuzz or has_mod, f"No FizzBuzz logic found in: {result[:300]!r}"


class TestOllamaTokenLeakage:
    """Check that Ollama's output doesn't leak special tokens."""

    SPECIAL_TOKENS = ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]

    def test_no_special_tokens(self, ensure_model):
        result = chat("Explain what Python is in two sentences.")
        print(f"\n[ollama] 'Explain Python' -> '{result}'")
        for token in self.SPECIAL_TOKENS:
            assert token not in result, (
                f"Special token {token!r} leaked into output: {result!r}"
            )
