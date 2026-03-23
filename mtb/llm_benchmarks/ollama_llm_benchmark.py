import gc
from typing import Any

import ollama
from ollama import ChatResponse, StatusResponse

from mtb.llm_benchmarks.base_llm_benchmark import BaseLLMBenchmark
from mtb.measurement import LlmBenchmarkMeasurement
from mtb.ollama_utils import try_pull_ollama_model


class OllamaLlmBenchmark(BaseLLMBenchmark):
    """Base class for LLM benchmarks using Ollama."""

    framework = "ollama"

    def __init__(
        self,
        max_context_length: int = 5000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_context_length = max_context_length

    def format_and_tokenize_prompt(self, prompt: str) -> Any:
        """Format the given prompt for Ollama.

        Ollama does not (yet) provide a separate tokenizer, see
        https://github.com/ollama/ollama/issues/3582

        """
        raise NotImplementedError("ollama does not provide a tokenizer endpoint yet")

    def setup(self):
        """Set up the benchmark. Load the model, tokenizer."""
        try_pull_ollama_model(self.model_id)

        response: StatusResponse = ollama.create(
            model=self.model_id,
            from_=self.model_id,
            parameters=dict(
                # define the max context length through load options
                num_ctx=self.max_context_length,
            ),
        )
        assert response.status == "success", response

        # we use the model_id as the identifier
        self.model: str = self.model_id
        return

    def run_once(self, prompt: Any) -> LlmBenchmarkMeasurement:
        """Run the benchmark once. Return measurements."""

        prompt = self.prompt_formatter(prompt)

        response: ChatResponse = ollama.chat(
            self.model,
            messages=prompt,
            options=dict(
                temperature=0.0,
                num_predict=self.max_num_tokens,
            ),
        )

        # ollama returns time in nanoseconds; when the prompt is cached
        # from a previous call, prompt_eval_count/duration may be None
        prompt_time_sec = (response.prompt_eval_duration or 0) / 1e9
        generation_time_sec = (response.eval_duration or 0) / 1e9
        num_prompt_tokens = response.prompt_eval_count or 0
        num_generated_tokens = response.eval_count or 0

        print(
            prompt_time_sec,
            generation_time_sec,
            num_prompt_tokens,
            num_generated_tokens,
        )

        return LlmBenchmarkMeasurement(
            response=response.message.content,
            prompt_time_sec=prompt_time_sec,
            prompt_tps=num_prompt_tokens / prompt_time_sec if prompt_time_sec > 0 else 0,
            generation_time_sec=generation_time_sec,
            generation_tps=num_generated_tokens / generation_time_sec if generation_time_sec > 0 else 0,
            num_prompt_tokens=num_prompt_tokens,
            num_generated_tokens=num_generated_tokens,
            peak_memory_gib=0,  # placeholder, ollama does not provide this
        )

    def teardown(self):
        """Teardown the benchmark."""

        # Unload, delete references
        # response = ollama.delete(self.model)
        # assert response.status == "success", response
        self.model = None

        # Reset indicators
        self._backend = None
        self._dtype = None
        self._device = None

        gc.collect()
        return
