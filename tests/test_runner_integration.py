"""Tests for quality benchmark runner integration with code execution and tool calling.

Verifies:
- Runner uses sandbox for problems with test_cases, check() for others
- Runner uses structured tool call parsing for tool_calling problems
- CSV output has execution detail columns for code-execution problems
- CSV output has parsed_tool_calls column for tool calling problems
- CLI preserves all existing flags
- Sandbox timeout produces failure row, not crash
- Majority vote logic correct for even/odd num_runs
- Mixed problem types in single run handled correctly
- Non-coding pattern-match problems produce identical results
- Summary prints pass counts per category
- Tool calling subcategory breakdowns in summary

Maps to validation contract assertions:
  VAL-RUNNER-001, -002, -003, -004, -005, -009, -010, -011, -012
  VAL-CROSS-001, -002, -007, -008
  VAL-TOOLCALL-904
  VAL-MODULE-005
"""

import math
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mtb.measurement import LlmBenchmarkMeasurement
from mtb.quality_benchmarks.eval_problem import EvalProblem
from mtb.quality_benchmarks.run_quality_benchmark import run_quality_benchmark
from mtb.quality_benchmarks.sandbox import SandboxResult

# Resolve repo root relative to this test file to avoid hardcoded absolute paths
_REPO_ROOT = Path(__file__).resolve().parent.parent

# The create_benchmark function is imported inside run_quality_benchmark() via:
#   from mtb.llm_benchmarks.run_llm_benchmark import create_benchmark
# To mock it, we patch the module-level import path.
_CREATE_BENCHMARK_PATCH = "mtb.llm_benchmarks.run_llm_benchmark.create_benchmark"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_measurement(response: str) -> LlmBenchmarkMeasurement:
    """Create a mock LlmBenchmarkMeasurement with the given response."""
    return LlmBenchmarkMeasurement(
        response=response,
        prompt_time_sec=0.1,
        prompt_tps=100.0,
        generation_time_sec=0.5,
        generation_tps=50.0,
        num_prompt_tokens=10,
        num_generated_tokens=25,
        peak_memory_gib=1.0,
    )


def _make_pattern_match_problem(name: str, category: str, passes: bool) -> EvalProblem:
    """Create a simple pattern-match EvalProblem (no test_cases)."""
    return EvalProblem(
        category=category,
        name=name,
        prompt=f"Test prompt for {name}",
        check=lambda r, _p=passes: _p,
        max_tokens=128,
    )


def _make_code_execution_problem(
    name: str, function_signature: str, test_cases: list, correct_code: str
) -> EvalProblem:
    """Create a coding EvalProblem with test_cases for code execution."""
    return EvalProblem(
        category="coding",
        name=name,
        prompt=f"Write a Python function: {function_signature}",
        check=lambda r: True,  # fallback check, should be bypassed by sandbox
        max_tokens=512,
        function_signature=function_signature,
        test_cases=test_cases,
    )


def _mock_benchmark_factory(responses):
    """Create a mock benchmark object that returns specified responses in order."""
    mock_benchmark = MagicMock()
    call_count = [0]

    def run_once(prompt):
        idx = call_count[0] % len(responses)
        call_count[0] += 1
        return _make_measurement(responses[idx])

    mock_benchmark.run_once = run_once
    mock_benchmark.max_num_tokens = 512
    return mock_benchmark


# ---------------------------------------------------------------------------
# VAL-RUNNER-011: Evaluation strategy dispatch mechanism
# ---------------------------------------------------------------------------


class TestEvaluationDispatch:
    """Problems with test_cases use sandbox; problems without use check()."""

    def test_pattern_match_problem_uses_check(self, tmp_path):
        """Pattern-match problem (no test_cases) evaluates via check()."""
        check_called = []

        def tracking_check(response):
            check_called.append(response)
            return True

        problem = EvalProblem(
            category="reasoning",
            name="test_reasoning",
            prompt="What is 2+2?",
            check=tracking_check,
            max_tokens=128,
        )

        mock_benchmark = _mock_benchmark_factory(["The answer is 4."])
        output_path = tmp_path / "results.csv"

        with patch(
            _CREATE_BENCHMARK_PATCH,
            return_value=mock_benchmark,
        ):
            df = run_quality_benchmark(
                model_spec=MagicMock(name="test-model"),
                framework="mlx",
                backend="metal",
                dtype="int4",
                output_path=output_path,
                problems=[problem],
                num_runs=1,
                cooldown_time_fraction=0.0,
            )

        assert (
            len(check_called) == 1
        ), "check() should have been called for pattern-match problem"
        assert df.iloc[0]["passed"] is True or df.iloc[0]["passed"] == True

    def test_code_execution_problem_uses_sandbox(self, tmp_path):
        """Code-execution problem (with test_cases) evaluates via sandbox."""
        problem = EvalProblem(
            category="coding",
            name="test_add",
            prompt="Write a Python function `add(a, b)` that returns a + b.",
            check=lambda r: False,  # This should NOT be used for code execution
            max_tokens=512,
            function_signature="def add(a, b):",
            test_cases=[
                {"input": "1, 2", "expected_output": 3},
                {"input": "0, 0", "expected_output": 0},
                {"input": "-1, 1", "expected_output": 0},
            ],
        )

        # Model returns correct code
        correct_response = "```python\ndef add(a, b):\n    return a + b\n```"
        mock_benchmark = _mock_benchmark_factory([correct_response])
        output_path = tmp_path / "results.csv"

        with patch(
            _CREATE_BENCHMARK_PATCH,
            return_value=mock_benchmark,
        ):
            df = run_quality_benchmark(
                model_spec=MagicMock(name="test-model"),
                framework="mlx",
                backend="metal",
                dtype="int4",
                output_path=output_path,
                problems=[problem],
                num_runs=1,
                cooldown_time_fraction=0.0,
            )

        assert len(df) == 1
        assert df.iloc[0]["passed"] is True or df.iloc[0]["passed"] == True

    def test_code_execution_failing_code(self, tmp_path):
        """Code-execution problem with incorrect code evaluates as failure."""
        problem = EvalProblem(
            category="coding",
            name="test_add",
            prompt="Write a Python function `add(a, b)` that returns a + b.",
            check=lambda r: True,  # This should NOT be used for code execution
            max_tokens=512,
            function_signature="def add(a, b):",
            test_cases=[
                {"input": "1, 2", "expected_output": 3},
                {"input": "0, 0", "expected_output": 0},
            ],
        )

        # Model returns wrong code
        wrong_response = "```python\ndef add(a, b):\n    return a - b\n```"
        mock_benchmark = _mock_benchmark_factory([wrong_response])
        output_path = tmp_path / "results.csv"

        with patch(
            _CREATE_BENCHMARK_PATCH,
            return_value=mock_benchmark,
        ):
            df = run_quality_benchmark(
                model_spec=MagicMock(name="test-model"),
                framework="mlx",
                backend="metal",
                dtype="int4",
                output_path=output_path,
                problems=[problem],
                num_runs=1,
                cooldown_time_fraction=0.0,
            )

        assert len(df) == 1
        assert df.iloc[0]["passed"] is False or df.iloc[0]["passed"] == False


# ---------------------------------------------------------------------------
# VAL-RUNNER-009: Sandbox timeout produces failure, not crash
# ---------------------------------------------------------------------------


class TestSandboxTimeout:
    """Sandbox timeout during code execution produces failure row, runner doesn't crash."""

    def test_timeout_produces_failure_not_crash(self, tmp_path):
        """Infinite loop code triggers timeout → failure row, not exception."""
        problem = EvalProblem(
            category="coding",
            name="test_infinite",
            prompt="Write a function.",
            check=lambda r: True,
            max_tokens=512,
            function_signature="def compute():",
            test_cases=[
                {"input": "", "expected_output": 42},
            ],
        )

        # Model returns infinite loop
        infinite_response = (
            "```python\ndef compute():\n    while True:\n        pass\n```"
        )
        mock_benchmark = _mock_benchmark_factory([infinite_response])
        output_path = tmp_path / "results.csv"

        with patch(
            _CREATE_BENCHMARK_PATCH,
            return_value=mock_benchmark,
        ):
            # Must complete without exception
            df = run_quality_benchmark(
                model_spec=MagicMock(name="test-model"),
                framework="mlx",
                backend="metal",
                dtype="int4",
                output_path=output_path,
                problems=[problem],
                num_runs=1,
                cooldown_time_fraction=0.0,
            )

        assert len(df) == 1
        assert df.iloc[0]["passed"] is False or df.iloc[0]["passed"] == False
        # Execution columns should be present
        assert "execution_exit_code" in df.columns


# ---------------------------------------------------------------------------
# VAL-RUNNER-004: CSV output contains all required columns
# ---------------------------------------------------------------------------


class TestCSVOutputColumns:
    """CSV output includes all required columns including execution detail columns."""

    def test_pattern_match_columns_present(self, tmp_path):
        """Pattern-match problem has all base columns + execution columns as NaN."""
        problem = _make_pattern_match_problem("test_reason", "reasoning", True)
        mock_benchmark = _mock_benchmark_factory(["answer"])
        output_path = tmp_path / "results.csv"

        with patch(
            _CREATE_BENCHMARK_PATCH,
            return_value=mock_benchmark,
        ):
            df = run_quality_benchmark(
                model_spec=MagicMock(name="test-model"),
                framework="mlx",
                backend="metal",
                dtype="int4",
                output_path=output_path,
                problems=[problem],
                num_runs=1,
                cooldown_time_fraction=0.0,
            )

        required_columns = {
            "model",
            "framework",
            "backend",
            "dtype",
            "category",
            "problem",
            "pass_count",
            "num_runs",
            "passed",
            "avg_generation_time_sec",
            "avg_generation_tps",
            "avg_tokens_generated",
            "sample_response",
            "execution_stdout",
            "execution_stderr",
            "execution_exit_code",
        }
        assert required_columns.issubset(
            set(df.columns)
        ), f"Missing columns: {required_columns - set(df.columns)}"

    def test_pattern_match_execution_columns_nan(self, tmp_path):
        """For non-code-execution problems, execution columns should be NaN."""
        problem = _make_pattern_match_problem("test_reason", "reasoning", True)
        mock_benchmark = _mock_benchmark_factory(["answer"])
        output_path = tmp_path / "results.csv"

        with patch(
            _CREATE_BENCHMARK_PATCH,
            return_value=mock_benchmark,
        ):
            df = run_quality_benchmark(
                model_spec=MagicMock(name="test-model"),
                framework="mlx",
                backend="metal",
                dtype="int4",
                output_path=output_path,
                problems=[problem],
                num_runs=1,
                cooldown_time_fraction=0.0,
            )

        row = df.iloc[0]
        assert pd.isna(row["execution_stdout"])
        assert pd.isna(row["execution_stderr"])
        assert pd.isna(row["execution_exit_code"])

    def test_code_execution_columns_populated(self, tmp_path):
        """For code-execution problems, execution columns should be populated."""
        problem = EvalProblem(
            category="coding",
            name="test_add",
            prompt="Write add(a, b).",
            check=lambda r: True,
            max_tokens=512,
            function_signature="def add(a, b):",
            test_cases=[
                {"input": "1, 2", "expected_output": 3},
            ],
        )

        correct_response = "```python\ndef add(a, b):\n    return a + b\n```"
        mock_benchmark = _mock_benchmark_factory([correct_response])
        output_path = tmp_path / "results.csv"

        with patch(
            _CREATE_BENCHMARK_PATCH,
            return_value=mock_benchmark,
        ):
            df = run_quality_benchmark(
                model_spec=MagicMock(name="test-model"),
                framework="mlx",
                backend="metal",
                dtype="int4",
                output_path=output_path,
                problems=[problem],
                num_runs=1,
                cooldown_time_fraction=0.0,
            )

        row = df.iloc[0]
        assert not pd.isna(row["execution_exit_code"])


# ---------------------------------------------------------------------------
# VAL-RUNNER-010: Majority vote logic
# ---------------------------------------------------------------------------


class TestMajorityVote:
    """Majority vote logic correct for even and odd num_runs."""

    def test_majority_3_runs_2_pass(self, tmp_path):
        """3 runs, 2 pass → majority pass (pass_count > num_runs / 2)."""
        problem = _make_pattern_match_problem("test_vote", "reasoning", True)
        # Mock check to pass 2 out of 3 times
        call_count = [0]

        def alternating_check(response):
            call_count[0] += 1
            return call_count[0] != 2  # fail on 2nd call

        problem.check = alternating_check
        mock_benchmark = _mock_benchmark_factory(["answer"])
        output_path = tmp_path / "results.csv"

        with patch(
            _CREATE_BENCHMARK_PATCH,
            return_value=mock_benchmark,
        ):
            df = run_quality_benchmark(
                model_spec=MagicMock(name="test-model"),
                framework="mlx",
                backend="metal",
                dtype="int4",
                output_path=output_path,
                problems=[problem],
                num_runs=3,
                cooldown_time_fraction=0.0,
            )

        assert df.iloc[0]["pass_count"] == 2
        assert df.iloc[0]["passed"] is True or df.iloc[0]["passed"] == True

    def test_majority_3_runs_1_pass(self, tmp_path):
        """3 runs, 1 pass → majority fail."""
        call_count = [0]

        def mostly_fail_check(response):
            call_count[0] += 1
            return call_count[0] == 1  # only pass on first call

        problem = _make_pattern_match_problem("test_vote_fail", "reasoning", True)
        problem.check = mostly_fail_check
        mock_benchmark = _mock_benchmark_factory(["answer"])
        output_path = tmp_path / "results.csv"

        with patch(
            _CREATE_BENCHMARK_PATCH,
            return_value=mock_benchmark,
        ):
            df = run_quality_benchmark(
                model_spec=MagicMock(name="test-model"),
                framework="mlx",
                backend="metal",
                dtype="int4",
                output_path=output_path,
                problems=[problem],
                num_runs=3,
                cooldown_time_fraction=0.0,
            )

        assert df.iloc[0]["pass_count"] == 1
        assert df.iloc[0]["passed"] is False or df.iloc[0]["passed"] == False

    def test_majority_4_runs_2_pass_is_false(self, tmp_path):
        """4 runs, 2 pass → NOT majority (2 is not > 4/2 = 2.0)."""
        call_count = [0]

        def half_pass(response):
            call_count[0] += 1
            return call_count[0] <= 2  # pass first 2, fail last 2

        problem = _make_pattern_match_problem("test_vote_even", "reasoning", True)
        problem.check = half_pass
        mock_benchmark = _mock_benchmark_factory(["answer"])
        output_path = tmp_path / "results.csv"

        with patch(
            _CREATE_BENCHMARK_PATCH,
            return_value=mock_benchmark,
        ):
            df = run_quality_benchmark(
                model_spec=MagicMock(name="test-model"),
                framework="mlx",
                backend="metal",
                dtype="int4",
                output_path=output_path,
                problems=[problem],
                num_runs=4,
                cooldown_time_fraction=0.0,
            )

        assert df.iloc[0]["pass_count"] == 2
        # 2 > 4/2 → 2 > 2.0 → False (strict majority)
        assert df.iloc[0]["passed"] is False or df.iloc[0]["passed"] == False

    def test_majority_5_runs_3_pass(self, tmp_path):
        """5 runs, 3 pass → majority pass."""
        call_count = [0]

        def three_pass(response):
            call_count[0] += 1
            return call_count[0] <= 3  # pass first 3, fail last 2

        problem = _make_pattern_match_problem("test_vote_5", "reasoning", True)
        problem.check = three_pass
        mock_benchmark = _mock_benchmark_factory(["answer"])
        output_path = tmp_path / "results.csv"

        with patch(
            _CREATE_BENCHMARK_PATCH,
            return_value=mock_benchmark,
        ):
            df = run_quality_benchmark(
                model_spec=MagicMock(name="test-model"),
                framework="mlx",
                backend="metal",
                dtype="int4",
                output_path=output_path,
                problems=[problem],
                num_runs=5,
                cooldown_time_fraction=0.0,
            )

        assert df.iloc[0]["pass_count"] == 3
        assert df.iloc[0]["passed"] is True or df.iloc[0]["passed"] == True


# ---------------------------------------------------------------------------
# VAL-RUNNER-012: Non-coding pattern-match problems unaffected
# ---------------------------------------------------------------------------


class TestPatternMatchUnaffected:
    """Non-coding problems (reasoning, math, writing, instruction) still use check()."""

    @pytest.mark.parametrize(
        "category", ["reasoning", "math", "writing", "instruction_following"]
    )
    def test_non_coding_categories_use_check(self, tmp_path, category):
        """Non-coding categories use pattern-match check function."""
        check_used = []

        def tracking_check(response):
            check_used.append(True)
            return True

        problem = EvalProblem(
            category=category,
            name=f"test_{category}",
            prompt=f"Test {category} prompt.",
            check=tracking_check,
            max_tokens=128,
        )
        mock_benchmark = _mock_benchmark_factory(["answer"])
        output_path = tmp_path / f"results_{category}.csv"

        with patch(
            _CREATE_BENCHMARK_PATCH,
            return_value=mock_benchmark,
        ):
            df = run_quality_benchmark(
                model_spec=MagicMock(name="test-model"),
                framework="mlx",
                backend="metal",
                dtype="int4",
                output_path=output_path,
                problems=[problem],
                num_runs=1,
                cooldown_time_fraction=0.0,
            )

        assert len(check_used) == 1, f"check() should have been called for {category}"
        assert df.iloc[0]["passed"] is True or df.iloc[0]["passed"] == True


# ---------------------------------------------------------------------------
# VAL-CROSS-007: Mixed problem types in single run
# ---------------------------------------------------------------------------


class TestMixedProblemTypes:
    """Runner handles heterogeneous problems: pattern-match + code execution in one run."""

    def test_mixed_problems_single_run(self, tmp_path):
        """Run with both pattern-match and code-execution problems succeeds."""
        pattern_problem = _make_pattern_match_problem(
            "test_reasoning", "reasoning", True
        )
        code_problem = EvalProblem(
            category="coding",
            name="test_add_mixed",
            prompt="Write add(a, b).",
            check=lambda r: True,
            max_tokens=512,
            function_signature="def add(a, b):",
            test_cases=[
                {"input": "1, 2", "expected_output": 3},
                {"input": "0, 0", "expected_output": 0},
            ],
        )

        # Model returns different responses per problem call
        responses = [
            "The answer is correct.",  # for pattern-match
            "```python\ndef add(a, b):\n    return a + b\n```",  # for code execution
        ]
        mock_benchmark = _mock_benchmark_factory(responses)
        output_path = tmp_path / "results.csv"

        with patch(
            _CREATE_BENCHMARK_PATCH,
            return_value=mock_benchmark,
        ):
            df = run_quality_benchmark(
                model_spec=MagicMock(name="test-model"),
                framework="mlx",
                backend="metal",
                dtype="int4",
                output_path=output_path,
                problems=[pattern_problem, code_problem],
                num_runs=1,
                cooldown_time_fraction=0.0,
            )

        assert len(df) == 2
        # Pattern-match problem
        pattern_row = df[df["problem"] == "test_reasoning"].iloc[0]
        assert pd.isna(pattern_row["execution_exit_code"])
        assert pattern_row["passed"] is True or pattern_row["passed"] == True

        # Code execution problem
        code_row = df[df["problem"] == "test_add_mixed"].iloc[0]
        assert not pd.isna(code_row["execution_exit_code"])


# ---------------------------------------------------------------------------
# VAL-RUNNER-005: CLI preserves all existing flags
# ---------------------------------------------------------------------------


class TestCLIFlags:
    """scripts/run_quality_benchmarks.py preserves all existing CLI flags."""

    def test_main_function_has_all_flags(self):
        """All expected flags are parameters of the main() function.

        Note: --help via fire.Fire is broken due to a pre-existing
        fire/IPython compatibility issue. We verify the function signature
        directly instead.
        """
        import inspect

        # Import the main function from the CLI script
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "run_quality_benchmarks",
            str(_REPO_ROOT / "scripts" / "run_quality_benchmarks.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        sig = inspect.signature(module.main)
        param_names = set(sig.parameters.keys())

        expected_flags = {
            "difficulty",
            "run_only_benchmarks",
            "dtypes",
            "num_runs",
            "cooldown_time_fraction",
            "output_root",
            "hf_cache_dir",
            "run_mlx_metal",
            "run_ollama_metal",
        }
        missing = expected_flags - param_names
        assert not missing, f"Missing CLI flags: {missing}"


# ---------------------------------------------------------------------------
# VAL-CROSS-008: Integration test exercises complete flow with mock model
# ---------------------------------------------------------------------------


class TestCompleteFlowMocked:
    """Complete flow: import problems, invoke runner with mock, verify CSV + summary."""

    def test_full_flow_with_mock(self, tmp_path):
        """Full flow: problems → runner → CSV output with correct columns and values."""
        problems = [
            _make_pattern_match_problem("reason_q1", "reasoning", True),
            _make_pattern_match_problem("reason_q2", "reasoning", False),
            EvalProblem(
                category="coding",
                name="code_q1",
                prompt="Write add(a, b).",
                check=lambda r: True,
                max_tokens=512,
                function_signature="def add(a, b):",
                test_cases=[{"input": "2, 3", "expected_output": 5}],
            ),
        ]

        responses = [
            "answer1",
            "answer2",
            "```python\ndef add(a, b):\n    return a + b\n```",
        ]
        mock_benchmark = _mock_benchmark_factory(responses)
        output_path = tmp_path / "results.csv"

        with patch(
            _CREATE_BENCHMARK_PATCH,
            return_value=mock_benchmark,
        ):
            df = run_quality_benchmark(
                model_spec=MagicMock(name="test-model"),
                framework="mlx",
                backend="metal",
                dtype="int4",
                output_path=output_path,
                problems=problems,
                num_runs=1,
                cooldown_time_fraction=0.0,
            )

        assert len(df) == 3

        # Verify categories are correct
        categories = df["category"].tolist()
        assert categories == ["reasoning", "reasoning", "coding"]

        # Verify passed values
        assert df.iloc[0]["passed"] is True or df.iloc[0]["passed"] == True
        assert df.iloc[1]["passed"] is False or df.iloc[1]["passed"] == False
        assert df.iloc[2]["passed"] is True or df.iloc[2]["passed"] == True

        # Verify CSV was saved
        assert output_path.exists()
        saved_df = pd.read_csv(output_path)
        assert len(saved_df) == 3

    def test_code_execution_majority_vote(self, tmp_path):
        """Code execution with majority vote: 2/3 pass → overall pass."""
        problem = EvalProblem(
            category="coding",
            name="test_vote_code",
            prompt="Write add(a, b).",
            check=lambda r: True,
            max_tokens=512,
            function_signature="def add(a, b):",
            test_cases=[{"input": "1, 2", "expected_output": 3}],
        )

        # 2 correct, 1 wrong
        responses = [
            "```python\ndef add(a, b):\n    return a + b\n```",
            "```python\ndef add(a, b):\n    return a - b\n```",
            "```python\ndef add(a, b):\n    return a + b\n```",
        ]
        mock_benchmark = _mock_benchmark_factory(responses)
        output_path = tmp_path / "results.csv"

        with patch(
            _CREATE_BENCHMARK_PATCH,
            return_value=mock_benchmark,
        ):
            df = run_quality_benchmark(
                model_spec=MagicMock(name="test-model"),
                framework="mlx",
                backend="metal",
                dtype="int4",
                output_path=output_path,
                problems=[problem],
                num_runs=3,
                cooldown_time_fraction=0.0,
            )

        assert df.iloc[0]["pass_count"] == 2
        assert df.iloc[0]["passed"] is True or df.iloc[0]["passed"] == True


# ---------------------------------------------------------------------------
# VAL-RUNNER-003: Tool-call problems use structured tool-call parser
# ---------------------------------------------------------------------------


class TestToolCallingEvaluation:
    """Runner handles tool_calling category problems via structured parsing."""

    def test_tool_calling_problem_uses_parser_and_check(self, tmp_path):
        """Tool calling problem is evaluated via parse_tool_calls + check()."""
        check_called = []

        def tracking_check(response):
            check_called.append(response)
            # Simulate a passing check
            return True

        problem = EvalProblem(
            category="tool_calling",
            name="ts_test_tool",
            prompt="Call the get_weather tool.",
            check=tracking_check,
            max_tokens=256,
        )

        # Model returns a valid tool call response
        tool_call_response = '{"name": "get_weather", "arguments": {"city": "NYC"}}'
        mock_benchmark = _mock_benchmark_factory([tool_call_response])
        output_path = tmp_path / "results.csv"

        with patch(
            _CREATE_BENCHMARK_PATCH,
            return_value=mock_benchmark,
        ):
            df = run_quality_benchmark(
                model_spec=MagicMock(name="test-model"),
                framework="mlx",
                backend="metal",
                dtype="int4",
                output_path=output_path,
                problems=[problem],
                num_runs=1,
                cooldown_time_fraction=0.0,
            )

        # check() should have been called
        assert (
            len(check_called) == 1
        ), "check() should be called for tool_calling problem"
        assert df.iloc[0]["passed"] is True or df.iloc[0]["passed"] == True

    def test_tool_calling_logs_parsed_calls(self, tmp_path):
        """Tool calling problems log parsed tool calls in CSV output."""
        problem = EvalProblem(
            category="tool_calling",
            name="ts_test_logging",
            prompt="Call the get_weather tool.",
            check=lambda r: True,
            max_tokens=256,
        )

        tool_call_response = '{"name": "get_weather", "arguments": {"city": "NYC"}}'
        mock_benchmark = _mock_benchmark_factory([tool_call_response])
        output_path = tmp_path / "results.csv"

        with patch(
            _CREATE_BENCHMARK_PATCH,
            return_value=mock_benchmark,
        ):
            df = run_quality_benchmark(
                model_spec=MagicMock(name="test-model"),
                framework="mlx",
                backend="metal",
                dtype="int4",
                output_path=output_path,
                problems=[problem],
                num_runs=1,
                cooldown_time_fraction=0.0,
            )

        # parsed_tool_calls column should be present and populated
        assert "parsed_tool_calls" in df.columns
        parsed = df.iloc[0]["parsed_tool_calls"]
        assert isinstance(parsed, str)
        assert "get_weather" in parsed
        assert "NYC" in parsed

    def test_tool_calling_no_tool_calls_logs_null(self, tmp_path):
        """When model response has no tool calls, parsed_tool_calls is 'null'."""
        problem = EvalProblem(
            category="tool_calling",
            name="ec_test_no_tool",
            prompt="What is 2 + 2?",
            check=lambda r: "4" in r,
            max_tokens=256,
        )

        # Model responds without any tool call
        plain_response = "The answer is 4."
        mock_benchmark = _mock_benchmark_factory([plain_response])
        output_path = tmp_path / "results.csv"

        with patch(
            _CREATE_BENCHMARK_PATCH,
            return_value=mock_benchmark,
        ):
            df = run_quality_benchmark(
                model_spec=MagicMock(name="test-model"),
                framework="mlx",
                backend="metal",
                dtype="int4",
                output_path=output_path,
                problems=[problem],
                num_runs=1,
                cooldown_time_fraction=0.0,
            )

        assert df.iloc[0]["parsed_tool_calls"] == "null"

    def test_tool_calling_failing_check(self, tmp_path):
        """Tool calling problem with failing check records failure."""
        problem = EvalProblem(
            category="tool_calling",
            name="ts_test_fail",
            prompt="Call search_flights.",
            check=lambda r: False,  # always fail
            max_tokens=256,
        )

        response = '{"name": "wrong_tool", "arguments": {}}'
        mock_benchmark = _mock_benchmark_factory([response])
        output_path = tmp_path / "results.csv"

        with patch(
            _CREATE_BENCHMARK_PATCH,
            return_value=mock_benchmark,
        ):
            df = run_quality_benchmark(
                model_spec=MagicMock(name="test-model"),
                framework="mlx",
                backend="metal",
                dtype="int4",
                output_path=output_path,
                problems=[problem],
                num_runs=1,
                cooldown_time_fraction=0.0,
            )

        assert df.iloc[0]["passed"] is False or df.iloc[0]["passed"] == False

    def test_tool_calling_execution_columns_nan(self, tmp_path):
        """Tool calling problems have NaN execution columns (not code execution)."""
        problem = EvalProblem(
            category="tool_calling",
            name="ts_test_exec_cols",
            prompt="Call get_weather.",
            check=lambda r: True,
            max_tokens=256,
        )

        response = '{"name": "get_weather", "arguments": {"city": "NYC"}}'
        mock_benchmark = _mock_benchmark_factory([response])
        output_path = tmp_path / "results.csv"

        with patch(
            _CREATE_BENCHMARK_PATCH,
            return_value=mock_benchmark,
        ):
            df = run_quality_benchmark(
                model_spec=MagicMock(name="test-model"),
                framework="mlx",
                backend="metal",
                dtype="int4",
                output_path=output_path,
                problems=[problem],
                num_runs=1,
                cooldown_time_fraction=0.0,
            )

        # Execution columns should be NaN for tool calling
        row = df.iloc[0]
        assert pd.isna(row["execution_stdout"])
        assert pd.isna(row["execution_stderr"])
        assert pd.isna(row["execution_exit_code"])
        # But parsed_tool_calls should be populated
        assert not pd.isna(row["parsed_tool_calls"])

    def test_tool_calling_exception_in_check_no_crash(self, tmp_path):
        """If check() raises, runner catches it and records failure."""

        def bad_check(response):
            raise RuntimeError("check exploded")

        problem = EvalProblem(
            category="tool_calling",
            name="ts_test_exception",
            prompt="Call get_weather.",
            check=bad_check,
            max_tokens=256,
        )

        response = '{"name": "get_weather", "arguments": {}}'
        mock_benchmark = _mock_benchmark_factory([response])
        output_path = tmp_path / "results.csv"

        with patch(
            _CREATE_BENCHMARK_PATCH,
            return_value=mock_benchmark,
        ):
            df = run_quality_benchmark(
                model_spec=MagicMock(name="test-model"),
                framework="mlx",
                backend="metal",
                dtype="int4",
                output_path=output_path,
                problems=[problem],
                num_runs=1,
                cooldown_time_fraction=0.0,
            )

        # Runner should not crash, should record failure
        assert len(df) == 1
        assert df.iloc[0]["passed"] is False or df.iloc[0]["passed"] == False
        assert df.iloc[0]["parsed_tool_calls"] == "null"


# ---------------------------------------------------------------------------
# VAL-CROSS-002: Tool-calling problem → parser → check → CSV end-to-end
# ---------------------------------------------------------------------------


class TestToolCallingEndToEnd:
    """End-to-end test: tool_calling → parser → check → CSV."""

    def test_tool_calling_full_flow(self, tmp_path):
        """Full pipeline: tool calling problem → parser extracts calls → check validates → CSV correct."""
        import json

        from mtb.quality_benchmarks.tool_call_parser import parse_tool_calls

        # Use a real-ish check function that uses the parser
        def check_get_weather(response):
            calls = parse_tool_calls(response)
            if not calls:
                return False
            return any(
                c.name == "get_weather" and c.arguments.get("city") == "NYC"
                for c in calls
            )

        problem = EvalProblem(
            category="tool_calling",
            name="ts_e2e_weather",
            prompt="Call get_weather for NYC.",
            check=check_get_weather,
            max_tokens=256,
        )

        # Correct tool call response
        correct_response = json.dumps(
            {"name": "get_weather", "arguments": {"city": "NYC"}}
        )
        mock_benchmark = _mock_benchmark_factory([correct_response])
        output_path = tmp_path / "results.csv"

        with patch(
            _CREATE_BENCHMARK_PATCH,
            return_value=mock_benchmark,
        ):
            df = run_quality_benchmark(
                model_spec=MagicMock(name="test-model"),
                framework="mlx",
                backend="metal",
                dtype="int4",
                output_path=output_path,
                problems=[problem],
                num_runs=1,
                cooldown_time_fraction=0.0,
            )

        assert len(df) == 1
        row = df.iloc[0]

        # Verify CSV columns
        assert row["category"] == "tool_calling"
        assert row["problem"] == "ts_e2e_weather"
        assert row["passed"] is True or row["passed"] == True

        # Verify parsed tool calls were logged
        parsed_tc = row["parsed_tool_calls"]
        assert isinstance(parsed_tc, str)
        parsed_list = json.loads(parsed_tc)
        assert len(parsed_list) == 1
        assert parsed_list[0]["name"] == "get_weather"
        assert parsed_list[0]["arguments"]["city"] == "NYC"

        # Verify CSV was saved to disk
        assert output_path.exists()
        saved_df = pd.read_csv(output_path)
        assert "parsed_tool_calls" in saved_df.columns

    def test_tool_calling_wrong_tool_fails(self, tmp_path):
        """Wrong tool call fails the check function in the full pipeline."""
        from mtb.quality_benchmarks.tool_call_parser import parse_tool_calls

        def check_search_flights(response):
            calls = parse_tool_calls(response)
            if not calls:
                return False
            return any(c.name == "search_flights" for c in calls)

        problem = EvalProblem(
            category="tool_calling",
            name="ts_e2e_wrong_tool",
            prompt="Search for flights.",
            check=check_search_flights,
            max_tokens=256,
        )

        # Wrong tool
        wrong_response = '{"name": "get_weather", "arguments": {"city": "NYC"}}'
        mock_benchmark = _mock_benchmark_factory([wrong_response])
        output_path = tmp_path / "results.csv"

        with patch(
            _CREATE_BENCHMARK_PATCH,
            return_value=mock_benchmark,
        ):
            df = run_quality_benchmark(
                model_spec=MagicMock(name="test-model"),
                framework="mlx",
                backend="metal",
                dtype="int4",
                output_path=output_path,
                problems=[problem],
                num_runs=1,
                cooldown_time_fraction=0.0,
            )

        assert df.iloc[0]["passed"] is False or df.iloc[0]["passed"] == False
        # But tool calls were still parsed and logged
        assert "get_weather" in df.iloc[0]["parsed_tool_calls"]


# ---------------------------------------------------------------------------
# VAL-TOOLCALL-904: update_readme_table.py includes tool calling in output
# ---------------------------------------------------------------------------


class TestReadmeToolCallingAnnotation:
    """update_readme_table.py references tool calling in quality annotation."""

    def test_problem_count_annotation_updated(self):
        """The problem count annotation reflects 81 problems, not 46."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "update_readme_table",
            str(_REPO_ROOT / "scripts" / "update_readme_table.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Call generate_tables to get the output text
        tables_text = module.generate_tables()

        # Should mention 81 problems (updated from 46)
        assert "81 problems" in tables_text

    def test_tool_calling_in_table_header(self):
        """The table generator includes 'Tool Calling' in column headers."""
        source_code = (_REPO_ROOT / "scripts" / "update_readme_table.py").read_text()
        assert "Tool Calling" in source_code


# ---------------------------------------------------------------------------
# VAL-MODULE-005: Total problem count is approximately 85 (within 75-95)
# ---------------------------------------------------------------------------


class TestTotalProblemCount:
    """Total problem count across all tiers is ~81 (within 75-95 range)."""

    def test_total_count_in_range(self):
        """Combined problem count is between 75 and 95."""
        from mtb.quality_benchmarks import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )

        total = (
            len(EVAL_PROBLEMS)
            + len(HARD_EVAL_PROBLEMS)
            + len(EXPERT_EVAL_PROBLEMS)
            + len(TOOL_CALLING_PROBLEMS)
        )
        assert 75 <= total <= 95, f"Total problem count {total} outside [75, 95] range"

    def test_exact_total_is_81(self):
        """Exact total: 15 easy + 10 hard + 16 expert + 40 tool_calling = 81."""
        from mtb.quality_benchmarks import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )

        assert len(EVAL_PROBLEMS) == 15
        assert len(HARD_EVAL_PROBLEMS) == 10
        assert len(EXPERT_EVAL_PROBLEMS) == 16
        assert len(TOOL_CALLING_PROBLEMS) == 40

    def test_difficulty_tool_calling_loads_all_40(self):
        """--difficulty tool_calling should load all 40 problems."""
        from mtb.quality_benchmarks import TOOL_CALLING_PROBLEMS

        assert len(TOOL_CALLING_PROBLEMS) == 40
        # All should have category "tool_calling"
        assert all(p.category == "tool_calling" for p in TOOL_CALLING_PROBLEMS)


# ---------------------------------------------------------------------------
# Mixed problem types including tool calling
# ---------------------------------------------------------------------------


class TestMixedWithToolCalling:
    """Runner handles mixed problem types: pattern-match + code execution + tool calling."""

    def test_all_three_evaluation_types(self, tmp_path):
        """Run with pattern-match, code-execution, and tool-calling problems."""
        import json

        from mtb.quality_benchmarks.tool_call_parser import parse_tool_calls

        pattern_problem = _make_pattern_match_problem("reason_q", "reasoning", True)

        code_problem = EvalProblem(
            category="coding",
            name="code_add",
            prompt="Write add(a, b).",
            check=lambda r: True,
            max_tokens=512,
            function_signature="def add(a, b):",
            test_cases=[{"input": "1, 2", "expected_output": 3}],
        )

        def check_tool(response):
            calls = parse_tool_calls(response)
            return calls is not None and any(c.name == "get_time" for c in calls)

        tool_problem = EvalProblem(
            category="tool_calling",
            name="ts_get_time",
            prompt="Get the current time.",
            check=check_tool,
            max_tokens=256,
        )

        responses = [
            "The answer is 42.",
            "```python\ndef add(a, b):\n    return a + b\n```",
            '{"name": "get_time", "arguments": {"timezone": "UTC"}}',
        ]
        mock_benchmark = _mock_benchmark_factory(responses)
        output_path = tmp_path / "results.csv"

        with patch(
            _CREATE_BENCHMARK_PATCH,
            return_value=mock_benchmark,
        ):
            df = run_quality_benchmark(
                model_spec=MagicMock(name="test-model"),
                framework="mlx",
                backend="metal",
                dtype="int4",
                output_path=output_path,
                problems=[pattern_problem, code_problem, tool_problem],
                num_runs=1,
                cooldown_time_fraction=0.0,
            )

        assert len(df) == 3

        # Check categories
        assert df.iloc[0]["category"] == "reasoning"
        assert df.iloc[1]["category"] == "coding"
        assert df.iloc[2]["category"] == "tool_calling"

        # Pattern match: no execution columns, no parsed_tool_calls
        assert pd.isna(df.iloc[0]["execution_exit_code"])
        assert pd.isna(df.iloc[0]["parsed_tool_calls"])

        # Code execution: has execution columns, no parsed_tool_calls
        assert not pd.isna(df.iloc[1]["execution_exit_code"])
        assert pd.isna(df.iloc[1]["parsed_tool_calls"])

        # Tool calling: no execution columns, has parsed_tool_calls
        assert pd.isna(df.iloc[2]["execution_exit_code"])
        assert not pd.isna(df.iloc[2]["parsed_tool_calls"])
        assert "get_time" in df.iloc[2]["parsed_tool_calls"]

    def test_csv_has_parsed_tool_calls_column(self, tmp_path):
        """CSV output includes parsed_tool_calls column for all problem types."""
        pattern_problem = _make_pattern_match_problem("reason_q2", "reasoning", True)

        tool_problem = EvalProblem(
            category="tool_calling",
            name="ts_csv_test",
            prompt="Call a tool.",
            check=lambda r: True,
            max_tokens=256,
        )

        responses = ["answer", '{"name": "test_tool", "arguments": {}}']
        mock_benchmark = _mock_benchmark_factory(responses)
        output_path = tmp_path / "results.csv"

        with patch(
            _CREATE_BENCHMARK_PATCH,
            return_value=mock_benchmark,
        ):
            df = run_quality_benchmark(
                model_spec=MagicMock(name="test-model"),
                framework="mlx",
                backend="metal",
                dtype="int4",
                output_path=output_path,
                problems=[pattern_problem, tool_problem],
                num_runs=1,
                cooldown_time_fraction=0.0,
            )

        # Column exists
        assert "parsed_tool_calls" in df.columns

        # Saved CSV also has the column
        saved_df = pd.read_csv(output_path)
        assert "parsed_tool_calls" in saved_df.columns
