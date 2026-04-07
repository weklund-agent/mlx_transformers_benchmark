import os
import time
from typing import Any, Iterable

import anthropic

from mtb.llm_benchmarks.base_llm_benchmark import BaseLLMBenchmark
from mtb.measurement import LlmBenchmarkMeasurement


class AnthropicLlmBenchmark(BaseLLMBenchmark):
    """LLM benchmark using the Anthropic Messages API.

    Calls the Anthropic API (e.g. Claude Opus 4.6) for inference.
    Speed metrics are end-to-end (includes network latency) and are NOT
    comparable to local inference benchmarks. Quality benchmarks are
    directly comparable.
    """

    framework = "anthropic"

    def setup(self):
        """Initialize the Anthropic client."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is required. "
                "Set it with: export ANTHROPIC_API_KEY=sk-ant-..."
            )
        self.client = anthropic.Anthropic(api_key=api_key)

    def format_and_tokenize_prompt(self, prompt: str) -> Iterable:
        """Format the prompt. Tokenization is not available via API."""
        return self.prompt_formatter(prompt)

    def run_once(self, prompt: Any) -> LlmBenchmarkMeasurement:
        """Run one inference call against the Anthropic API."""
        messages = self.prompt_formatter(prompt)

        # Separate system message from user messages (Anthropic API format)
        system_content = None
        api_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                api_messages.append(msg)

        # Build request kwargs
        kwargs = dict(
            model=self.model_id,
            max_tokens=self.max_num_tokens,
            messages=api_messages,
        )
        if system_content is not None:
            kwargs["system"] = system_content

        if self.thinking:
            kwargs["thinking"] = {"type": "adaptive"}

        # Time the API call
        start = time.perf_counter()
        response = self.client.messages.create(**kwargs)
        elapsed = time.perf_counter() - start

        # Extract text from response (skip thinking blocks)
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
        text = "\n".join(text_parts)

        # Extract token counts from usage
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        # We can only measure end-to-end; attribute all time to generation
        generation_tps = output_tokens / elapsed if elapsed > 0 else 0

        return LlmBenchmarkMeasurement(
            response=text,
            prompt_time_sec=0.0,
            prompt_tps=0.0,
            generation_time_sec=elapsed,
            generation_tps=generation_tps,
            num_prompt_tokens=input_tokens,
            num_generated_tokens=output_tokens,
            peak_memory_gib=0.0,
        )

    def teardown(self):
        """Cleanup."""
        self.client = None
