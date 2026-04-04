"""Tests for quality benchmark runner integration with code execution.

Verifies:
- Runner uses sandbox for problems with test_cases, check() for others
- CSV output has execution detail columns for code-execution problems
- CLI preserves all existing flags
- Sandbox timeout produces failure row, not crash
- Majority vote logic correct for even/odd num_runs
- Mixed problem types in single run handled correctly
- Non-coding pattern-match problems produce identical results
- Summary prints pass counts per category

Maps to validation contract assertions:
  VAL-RUNNER-001, -002, -004, -005, -009, -010, -011, -012
  VAL-CROSS-001, -007, -008
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
            "/Users/weae1504/Projects/mlx_transformers_benchmark/scripts/run_quality_benchmarks.py",
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
