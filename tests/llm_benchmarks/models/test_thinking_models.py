"""Tests for thinking model support.

Thinking models (Nemotron, DeepSeek R1, Qwen3, Qwen3.5) have chat templates
that inject <think> blocks. These models must have thinking enabled so they
generate properly — disabling thinking causes degenerate output (e.g. repeated
<|im_end|> tokens). The thinking content is stripped during quality evaluation.

ModelSpec.thinking=True marks thinking-capable models, and
format_and_tokenize_prompt passes enable_thinking=True to apply_chat_template.
"""

import pytest

from mtb.llm_benchmarks.models.base import ModelSpec


class TestModelSpecThinkingField:
    """ModelSpec should have a thinking field to identify thinking models."""

    def test_model_spec_has_thinking_field(self):
        spec = ModelSpec(
            name="test",
            num_params=1e9,
            prompt_formatter=lambda x: x,
        )
        assert hasattr(spec, "thinking")

    def test_model_spec_thinking_defaults_false(self):
        spec = ModelSpec(
            name="test",
            num_params=1e9,
            prompt_formatter=lambda x: x,
        )
        assert spec.thinking is False

    def test_model_spec_thinking_can_be_set_true(self):
        spec = ModelSpec(
            name="test",
            num_params=1e9,
            prompt_formatter=lambda x: x,
            thinking=True,
        )
        assert spec.thinking is True


class TestThinkingModelSpecs:
    """Known thinking models should have thinking=True."""

    def test_nemotron_nano_4b_is_not_thinking(self):
        from mtb.llm_benchmarks.models.nemotron import Nemotron3_Nano_4B

        assert Nemotron3_Nano_4B.thinking is False

    def test_nemotron_cascade_is_thinking(self):
        from mtb.llm_benchmarks.models.nemotron import Nemotron_Cascade2_30B_A3B

        assert Nemotron_Cascade2_30B_A3B.thinking is True

    def test_deepseek_r1_distill_is_thinking(self):
        from mtb.llm_benchmarks.models.deepseek import Deepseek_R1_Distill_Qwen_7B

        assert Deepseek_R1_Distill_Qwen_7B.thinking is True

    def test_deepseek_r1_0528_is_thinking(self):
        from mtb.llm_benchmarks.models.deepseek import Deepseek_R1_0528_Qwen3_8B

        assert Deepseek_R1_0528_Qwen3_8B.thinking is True

    def test_qwen3_models_are_thinking(self):
        from mtb.llm_benchmarks.models.qwen import (
            Qwen3_0p6B_it,
            Qwen3_8B_it,
            Qwen3_14B_it,
        )

        assert Qwen3_0p6B_it.thinking is True
        assert Qwen3_8B_it.thinking is True
        assert Qwen3_14B_it.thinking is True

    def test_qwen35_models_are_thinking(self):
        from mtb.llm_benchmarks.models.qwen35 import (
            Qwen3p5_0p8B,
            Qwen3p5_4B,
            Qwen3p5_27B,
        )

        assert Qwen3p5_0p8B.thinking is True
        assert Qwen3p5_4B.thinking is True
        assert Qwen3p5_27B.thinking is True

    def test_non_thinking_models_are_not_thinking(self):
        from mtb.llm_benchmarks.models.gemma import Gemma3_1B_it, Gemma3_4B_it
        from mtb.llm_benchmarks.models.qwen import Qwen2p5_0p5B_it, Qwen2p5_3B_it

        assert Gemma3_1B_it.thinking is False
        assert Gemma3_4B_it.thinking is False
        assert Qwen2p5_0p5B_it.thinking is False
        assert Qwen2p5_3B_it.thinking is False


class TestBenchmarkThinkingSupport:
    """BaseLLMBenchmark should thread the thinking flag through."""

    def test_base_benchmark_stores_thinking_flag(self):
        from mtb.llm_benchmarks.base_llm_benchmark import BaseLLMBenchmark

        benchmark = BaseLLMBenchmark(
            name="test",
            model_id="test",
            backend="metal",
            dtype="int4",
            prompt_formatter=lambda x: x,
            thinking=True,
        )
        assert benchmark.thinking is True

    def test_base_benchmark_thinking_defaults_false(self):
        from mtb.llm_benchmarks.base_llm_benchmark import BaseLLMBenchmark

        benchmark = BaseLLMBenchmark(
            name="test",
            model_id="test",
            backend="metal",
            dtype="int4",
            prompt_formatter=lambda x: x,
        )
        assert benchmark.thinking is False

    def test_create_benchmark_passes_thinking_flag(self):
        from mtb.llm_benchmarks.models.nemotron import Nemotron_Nano_9B_v2
        from mtb.llm_benchmarks.run_llm_benchmark import create_benchmark

        benchmark = create_benchmark(
            model_spec=Nemotron_Nano_9B_v2,
            framework="mlx",
            backend="metal",
            dtype="int4",
            max_num_tokens=10,
        )
        assert benchmark.thinking is True

    def test_create_benchmark_non_thinking_model(self):
        from mtb.llm_benchmarks.models.qwen import Qwen2p5_0p5B_it
        from mtb.llm_benchmarks.run_llm_benchmark import create_benchmark

        benchmark = create_benchmark(
            model_spec=Qwen2p5_0p5B_it,
            framework="mlx",
            backend="metal",
            dtype="int4",
            max_num_tokens=10,
        )
        assert benchmark.thinking is False


class TestApplyChatTemplateThinking:
    """format_and_tokenize_prompt should enable thinking for thinking models."""

    @pytest.fixture(scope="class")
    def nemotron_benchmark(self):
        from mtb.llm_benchmarks.models.nemotron import Nemotron_Nano_9B_v2
        from mtb.llm_benchmarks.run_llm_benchmark import create_benchmark

        benchmark = create_benchmark(
            model_spec=Nemotron_Nano_9B_v2,
            framework="mlx",
            backend="metal",
            dtype="int4",
            max_num_tokens=10,
        )
        benchmark.setup()
        yield benchmark
        benchmark.teardown()

    def test_nemotron_prompt_has_open_think_tag(self, nemotron_benchmark):
        """The tokenized prompt should end with an open <think> tag.

        When thinking is properly enabled, the template inserts <think>\\n
        (open) so the model generates reasoning followed by its answer.
        """
        prompt_tokens = nemotron_benchmark.format_and_tokenize_prompt("Write fizzbuzz in Python.")
        decoded = nemotron_benchmark.tokenizer.decode(prompt_tokens.tolist())
        # Should end with an open <think> tag (thinking enabled)
        assert decoded.rstrip().endswith("<think>")
        # Should NOT have the closed <think></think> pattern
        assert "<think></think>" not in decoded
