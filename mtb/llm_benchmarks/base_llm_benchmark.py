from typing import Any, Callable, Iterable

from mtb.measurement import LlmBenchmarkMeasurement
from mtb.system.memory import get_process_memory_gib


class BaseLLMBenchmark:
    """Base class for LLM benchmarks.

    Should implement:

      1. `setup`: Initialize the model and tokenizer. Prepare the prompt.
      2. `run_once`: Run the benchmark once.
      3. `teardown`: Cleanup.

    """

    # Name of the framework used for the benchmark
    framework = None

    def __init__(
        self,
        name: str,
        model_id: str,
        backend: str,
        dtype: str,
        prompt_formatter: Callable[[str], Any],
        max_num_tokens: int = 100,
        thinking: bool = False,
    ):
        self.name = name
        self.model_id = model_id
        self.backend = backend
        self.dtype = dtype
        self.prompt_formatter = prompt_formatter
        self.max_num_tokens = max_num_tokens
        self.thinking = thinking

        # track memory allocated by the process after this benchmark
        self.initial_process_memory_gib = get_process_memory_gib()

    def format_and_tokenize_prompt(self, prompt: str) -> Iterable:
        """Format and tokenize the prompt. Return a list, array or tensor of tokens."""
        raise NotImplementedError

    def get_num_prompt_tokens(self, user_prompt: str) -> int:
        """Get the number of tokens for a given user prompt."""
        tokens = self.tokenize(user_prompt)
        return len(tokens)

    def setup(self):
        """Set up the benchmark. Load the model, tokenizer."""
        raise NotImplementedError

    def run_once(self) -> LlmBenchmarkMeasurement:
        """Run the benchmark once. Return measurements."""
        raise NotImplementedError

    def teardown(self):
        """Teardown the benchmark."""
        raise NotImplementedError
