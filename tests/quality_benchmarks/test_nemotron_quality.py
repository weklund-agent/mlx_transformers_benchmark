"""Diagnostic test: Is Nemotron-3-Nano-4B failing quality benchmarks due to
config/template issues, or is the model simply not capable enough?

Strategy:
  1. Raw completion test — bypass chat template entirely, feed raw text to the
     model. If this produces coherent output but the chat-templated version
     doesn't, the problem is our template/config.
  2. Trivially easy chat-templated prompts — "What is 2+2?", "Say hello." —
     things any working chat model should handle regardless of quality.
  3. Check for <|im_end|> leaking into generated text, which would indicate a
     tokenizer/template misconfiguration.

If both raw and chat-templated paths produce incoherent / truncated output with
<|im_end|> tokens, the model itself is the problem, not our config.
"""

import pytest
import mlx_lm
import mlx.core as mx
from pathlib import Path

from mtb.llm_benchmarks.models.nemotron import (
    Nemotron3_Nano_4B,
    format_nemotron_prompt,
)

MODEL_ID = Nemotron3_Nano_4B.model_ids["mlx"]["int4"]

pytestmark = pytest.mark.skipif(
    not Path(MODEL_ID).exists(),
    reason=f"Model not downloaded: {MODEL_ID}",
)


@pytest.fixture(scope="module")
def model_and_tokenizer():
    model, tokenizer = mlx_lm.load(MODEL_ID)
    yield model, tokenizer
    del model, tokenizer
    mx.clear_cache()


def generate(model, tokenizer, prompt_tokens, max_tokens=256):
    """Generate text and return it."""
    text = ""
    for response in mlx_lm.stream_generate(
        model, tokenizer, max_tokens=max_tokens, prompt=prompt_tokens
    ):
        text += response.text
    return text


# ---------------------------------------------------------------------------
# 1. Raw completion (no chat template) — tests the model weights directly
# ---------------------------------------------------------------------------
class TestRawCompletion:
    """Bypass chat template. If this fails, the model weights themselves are bad."""

    def test_complete_sentence(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        prompt = "The capital of France is"
        tokens = mx.array(tokenizer.encode(prompt))
        result = generate(model, tokenizer, tokens, max_tokens=32)
        print(f"\n[raw] 'The capital of France is' -> '{result}'")
        # Should mention Paris somewhere
        assert "paris" in result.lower(), f"Expected 'Paris' in raw completion, got: {result!r}"

    def test_simple_arithmetic_raw(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        prompt = "2 + 2 ="
        tokens = mx.array(tokenizer.encode(prompt))
        result = generate(model, tokenizer, tokens, max_tokens=16)
        print(f"\n[raw] '2 + 2 =' -> '{result}'")
        assert "4" in result, f"Expected '4' in raw completion, got: {result!r}"


# ---------------------------------------------------------------------------
# 2. Chat-templated trivial prompts — tests our template config
# ---------------------------------------------------------------------------
class TestChatTemplate:
    """Use our standard chat template pipeline. If raw works but these fail,
    it's a template/config issue."""

    def _generate_chat(self, model, tokenizer, user_prompt, max_tokens=256):
        messages = format_nemotron_prompt(user_prompt)
        prompt_tokens = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="mlx"
        )
        if isinstance(prompt_tokens, mx.array):
            prompt_tokens = prompt_tokens[0]
        else:
            prompt_tokens = prompt_tokens.input_ids[0]
        return generate(model, tokenizer, prompt_tokens, max_tokens)

    def test_say_hello(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        result = self._generate_chat(model, tokenizer, "Say hello.", max_tokens=64)
        print(f"\n[chat] 'Say hello.' -> '{result}'")
        # Should contain some greeting
        assert len(result.strip()) > 2, f"Response too short: {result!r}"
        assert "hello" in result.lower() or "hi" in result.lower(), (
            f"Expected a greeting, got: {result!r}"
        )

    def test_arithmetic_chat(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        result = self._generate_chat(model, tokenizer, "What is 2+2? Reply with just the number.")
        print(f"\n[chat] 'What is 2+2?' -> '{result}'")
        assert "4" in result, f"Expected '4', got: {result!r}"

    def test_fizzbuzz_chat(self, model_and_tokenizer):
        """The simplest quality benchmark problem — FizzBuzz."""
        model, tokenizer = model_and_tokenizer
        result = self._generate_chat(
            model,
            tokenizer,
            "Write a Python function that prints FizzBuzz for numbers 1 to 100.",
            max_tokens=512,
        )
        print(f"\n[chat] FizzBuzz -> '{result[:300]}'")
        result_lower = result.lower()
        has_fizzbuzz = "fizzbuzz" in result_lower or "fizz buzz" in result_lower
        has_mod = "%" in result or "mod" in result_lower or "divisible" in result_lower
        assert has_fizzbuzz or has_mod, f"No FizzBuzz logic found in: {result[:300]!r}"


# ---------------------------------------------------------------------------
# 3. Check for special token leakage — a template/tokenizer misconfiguration
# ---------------------------------------------------------------------------
class TestTokenLeakage:
    """If <|im_end|> or other special tokens leak into output, that's a
    template issue we could potentially fix."""

    SPECIAL_TOKENS = ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]

    def _generate_chat(self, model, tokenizer, user_prompt, max_tokens=256):
        messages = format_nemotron_prompt(user_prompt)
        prompt_tokens = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="mlx"
        )
        if isinstance(prompt_tokens, mx.array):
            prompt_tokens = prompt_tokens[0]
        else:
            prompt_tokens = prompt_tokens.input_ids[0]
        return generate(model, tokenizer, prompt_tokens, max_tokens)

    def test_no_special_tokens_in_greeting(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        result = self._generate_chat(model, tokenizer, "Say hello.")
        print(f"\n[leak] 'Say hello.' -> '{result}'")
        for token in self.SPECIAL_TOKENS:
            assert token not in result, (
                f"Special token {token!r} leaked into output: {result!r}"
            )

    def test_no_special_tokens_in_arithmetic(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        result = self._generate_chat(model, tokenizer, "What is 10 * 5?")
        print(f"\n[leak] 'What is 10 * 5?' -> '{result}'")
        for token in self.SPECIAL_TOKENS:
            assert token not in result, (
                f"Special token {token!r} leaked into output: {result!r}"
            )
