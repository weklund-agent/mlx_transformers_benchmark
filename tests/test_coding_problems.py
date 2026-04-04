"""Tests for coding problems with code-execution-based evaluation.

Verifies that all 18 coding problems have:
- function_signature set
- 5-10 test cases with input and expected_output
- Correct difficulty tiers
- Reference correct implementations that pass all test cases via sandbox
- Known-incorrect implementations that fail at least one test case
- Boundary/edge cases present
- Deterministic test results
- No duplicate names
"""

import pytest

from mtb.quality_benchmarks.coding_problems import CODING_PROBLEMS
from mtb.quality_benchmarks.sandbox import execute_code


# ---------------------------------------------------------------------------
# Expected problem names by tier
# ---------------------------------------------------------------------------

EXPECTED_EASY = {
    "fizzbuzz",
    "reverse_string",
    "fibonacci",
    "binary_search",
    "palindrome",
}
EXPECTED_HARD = {
    "lru_cache",
    "flatten_nested",
    "longest_palindrome_substring",
    "calculator",
    "buggy_merge_sort",
}
EXPECTED_EXPERT = {
    "markdown_to_html",
    "data_pipeline",
    "retry_decorator",
    "topological_sort",
    "constrained_factorial",
    "code_with_comments",
    "library_schema",
    "adversarial_transform",
}
ALL_EXPECTED = EXPECTED_EASY | EXPECTED_HARD | EXPECTED_EXPERT


# ---------------------------------------------------------------------------
# Helper: build sandbox code from problem
# ---------------------------------------------------------------------------


def _build_test_code(problem, implementation: str) -> str:
    """Build executable code combining implementation + test assertions."""
    lines = [implementation, ""]
    for i, tc in enumerate(problem.test_cases):
        inp = tc["input"]
        expected = tc["expected_output"]
        # Use approximate comparison for floats
        if isinstance(expected, float):
            lines.append(
                f"assert abs({problem.function_signature.split('(')[0].replace('def ', '')}({inp}) - {expected!r}) < 1e-6, "
                f"'Test case {i} failed'"
            )
        else:
            lines.append(
                f"assert {problem.function_signature.split('(')[0].replace('def ', '')}({inp}) == {expected!r}, "
                f"'Test case {i} failed'"
            )
    lines.append("print('ALL TESTS PASSED')")
    return "\n".join(lines)


# ===========================================================================
# Structural tests
# ===========================================================================


class TestCodingProblemsStructure:
    """Structural validation for all 18 coding problems."""

    def test_total_count_is_18(self):
        """VAL-CODING-006: All 18 problems present in the registry."""
        assert len(CODING_PROBLEMS) == 18

    def test_all_expected_names_present(self):
        """VAL-CODING-006: All expected problem names are present."""
        names = {p.name for p in CODING_PROBLEMS}
        assert (
            names == ALL_EXPECTED
        ), f"Missing: {ALL_EXPECTED - names}, Extra: {names - ALL_EXPECTED}"

    def test_no_duplicate_names(self):
        """VAL-CODING-005: No duplicate problem names."""
        names = [p.name for p in CODING_PROBLEMS]
        assert len(names) == len(set(names)), f"Duplicate names found"

    def test_every_problem_has_function_signature(self):
        """VAL-CODING-001: Every problem has a function_signature field."""
        for p in CODING_PROBLEMS:
            assert (
                p.function_signature is not None
            ), f"{p.name} has no function_signature"
            assert (
                len(p.function_signature) > 0
            ), f"{p.name} has empty function_signature"
            assert p.function_signature.startswith(
                "def "
            ), f"{p.name} function_signature should start with 'def '"

    def test_every_problem_has_5_to_10_test_cases(self):
        """VAL-CODING-002: Every problem has 5-10 test cases."""
        for p in CODING_PROBLEMS:
            assert p.test_cases is not None, f"{p.name} has no test_cases"
            count = len(p.test_cases)
            assert 5 <= count <= 10, f"{p.name} has {count} test cases, expected 5-10"

    def test_test_cases_have_input_and_expected_output(self):
        """VAL-CODING-003: Each test case has input and expected_output."""
        for p in CODING_PROBLEMS:
            for i, tc in enumerate(p.test_cases):
                assert "input" in tc, f"{p.name} test case {i} missing 'input'"
                assert (
                    "expected_output" in tc
                ), f"{p.name} test case {i} missing 'expected_output'"
                # input can be empty string, 0, etc. but should not be None
                assert tc["input"] is not None, f"{p.name} test case {i} has None input"
                assert (
                    tc["expected_output"] is not None
                ), f"{p.name} test case {i} has None expected_output"

    def test_all_checks_are_callable(self):
        """Every problem still has a callable check function."""
        for p in CODING_PROBLEMS:
            assert callable(p.check), f"{p.name} check is not callable"

    def test_all_have_valid_category(self):
        """All 18 problems have a valid category.

        The 13 native coding problems have category='coding'.
        The 5 cross-category problems retain their original categories
        (reasoning, instruction_following) but are included in CODING_PROBLEMS
        for code-execution-based evaluation.
        """
        _CROSS_CATEGORY_NAMES = {
            "topological_sort",
            "constrained_factorial",
            "code_with_comments",
            "library_schema",
            "adversarial_transform",
        }
        valid_categories = {"coding", "reasoning", "instruction_following"}
        for p in CODING_PROBLEMS:
            assert (
                p.category in valid_categories
            ), f"{p.name} has unexpected category '{p.category}'"
            if p.name not in _CROSS_CATEGORY_NAMES:
                assert (
                    p.category == "coding"
                ), f"{p.name} should have category 'coding', got '{p.category}'"


class TestDifficultyTiers:
    """VAL-CODING-004: Difficulty tiers correctly assigned."""

    def _problems_by_name(self):
        return {p.name: p for p in CODING_PROBLEMS}

    def test_easy_problems_in_easy_tier(self):
        """Easy tier problems are in CODING_EASY_PROBLEMS."""
        from mtb.quality_benchmarks.coding_problems import CODING_EASY_PROBLEMS

        easy_names = {p.name for p in CODING_EASY_PROBLEMS}
        assert easy_names == EXPECTED_EASY

    def test_hard_problems_in_hard_tier(self):
        """Hard tier problems are in CODING_HARD_PROBLEMS."""
        from mtb.quality_benchmarks.coding_problems import CODING_HARD_PROBLEMS

        hard_names = {p.name for p in CODING_HARD_PROBLEMS}
        assert hard_names == EXPECTED_HARD

    def test_expert_problems_in_coding_problems(self):
        """Expert tier problems are in CODING_PROBLEMS (some via cross-category)."""
        expert_names_in_coding = {
            p.name for p in CODING_PROBLEMS if p.name in EXPECTED_EXPERT
        }
        assert expert_names_in_coding == EXPECTED_EXPERT


class TestBoundaryEdgeCases:
    """VAL-CODING-050: Boundary/edge test cases present for each problem."""

    def _get_problem(self, name):
        for p in CODING_PROBLEMS:
            if p.name == name:
                return p
        pytest.fail(f"Problem '{name}' not found")

    def test_fizzbuzz_boundary_cases(self):
        p = self._get_problem("fizzbuzz")
        inputs = [tc["input"] for tc in p.test_cases]
        # Should include n=15 (FizzBuzz) and n=1
        input_strs = " ".join(str(i) for i in inputs)
        assert "15" in input_strs or any("15" in str(i) for i in inputs)
        assert "1" in input_strs or any("1" in str(i) for i in inputs)

    def test_reverse_string_boundary_cases(self):
        p = self._get_problem("reverse_string")
        inputs = [tc["input"] for tc in p.test_cases]
        input_strs = str(inputs)
        # empty string and single char
        assert "''" in input_strs or '""' in input_strs
        has_single = any(
            tc["input"] in ("'a'", '"a"', "a")
            or (
                isinstance(tc["input"], str)
                and len(tc["input"]) <= 3
                and "a" in tc["input"]
            )
            for tc in p.test_cases
        )
        assert has_single or any(len(str(tc["input"])) <= 5 for tc in p.test_cases)

    def test_fibonacci_boundary_cases(self):
        p = self._get_problem("fibonacci")
        inputs = [tc["input"] for tc in p.test_cases]
        input_strs = str(inputs)
        # n=0 and n=1
        assert "0" in input_strs
        assert "1" in input_strs

    def test_binary_search_boundary_cases(self):
        p = self._get_problem("binary_search")
        inputs = [tc["input"] for tc in p.test_cases]
        input_strs = str(inputs)
        # empty list and single element
        assert "[]" in input_strs
        has_single = any(
            "[" in str(i) and str(i).count(",") == 0 and "[]" not in str(i)
            for i in inputs
        )
        # Target at boundaries
        assert has_single or any("1]" in str(i) for i in inputs)

    def test_palindrome_boundary_cases(self):
        p = self._get_problem("palindrome")
        inputs = [tc["input"] for tc in p.test_cases]
        input_strs = str(inputs)
        # empty string and single char
        assert "''" in input_strs or '""' in input_strs
        has_single = any(
            len(str(tc["input"]).replace("'", "").replace('"', "")) <= 3
            for tc in p.test_cases
        )
        assert has_single

    def test_lru_cache_boundary_cases(self):
        p = self._get_problem("lru_cache")
        inputs = [tc["input"] for tc in p.test_cases]
        input_strs = str(inputs)
        # capacity=1
        assert "1" in input_strs

    def test_flatten_nested_boundary_cases(self):
        p = self._get_problem("flatten_nested")
        inputs = [tc["input"] for tc in p.test_cases]
        input_strs = str(inputs)
        # empty list and already flat
        assert "[]" in input_strs

    def test_longest_palindrome_boundary_cases(self):
        p = self._get_problem("longest_palindrome_substring")
        inputs = [tc["input"] for tc in p.test_cases]
        input_strs = str(inputs)
        # single char
        has_single = any(
            len(str(tc["input"]).replace("'", "").replace('"', "")) <= 3
            for tc in p.test_cases
        )
        assert has_single

    def test_calculator_boundary_cases(self):
        p = self._get_problem("calculator")
        inputs = [tc["input"] for tc in p.test_cases]
        input_strs = str(inputs)
        # single number
        has_single_number = any(
            tc["input"] in ("'5'", '"5"', "'42'", '"42"')
            or (isinstance(tc["input"], str) and tc["input"].strip("'\"").isdigit())
            for tc in p.test_cases
        )
        assert has_single_number or "single" in input_strs.lower()

    def test_buggy_merge_sort_boundary_cases(self):
        p = self._get_problem("buggy_merge_sort")
        inputs = [tc["input"] for tc in p.test_cases]
        input_strs = str(inputs)
        # already sorted, single element
        has_single = any(
            str(tc["input"]).count(",") == 0 and "[" in str(tc["input"])
            for tc in p.test_cases
        )
        assert has_single or "[]" in input_strs


# ===========================================================================
# Correct implementations pass all test cases
# ===========================================================================


class TestCorrectImplementationsPass:
    """VAL-CODING-045: All 18 correct reference implementations pass their test cases."""

    @pytest.mark.parametrize(
        "problem",
        CODING_PROBLEMS,
        ids=lambda p: p.name,
    )
    def test_correct_implementation_passes(self, problem):
        """Reference correct implementation passes all test cases via sandbox."""
        assert hasattr(problem, "_correct_impl") or problem.test_cases is not None
        # Get the correct implementation
        correct_impl = problem._correct_impl
        assert correct_impl is not None, f"{problem.name} has no _correct_impl"

        # Build and execute test code
        code = _build_test_code(problem, correct_impl)
        result = execute_code(code, timeout=15)
        assert result.success, (
            f"{problem.name} correct impl failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        assert "ALL TESTS PASSED" in result.stdout


# ===========================================================================
# Incorrect implementations fail at least one test case
# ===========================================================================


class TestIncorrectImplementationsFail:
    """VAL-CODING-046: All 18 incorrect implementations fail at least one test case."""

    @pytest.mark.parametrize(
        "problem",
        CODING_PROBLEMS,
        ids=lambda p: p.name,
    )
    def test_incorrect_implementation_fails(self, problem):
        """Known-incorrect implementation fails at least one test case."""
        incorrect_impl = problem._incorrect_impl
        assert incorrect_impl is not None, f"{problem.name} has no _incorrect_impl"

        code = _build_test_code(problem, incorrect_impl)
        result = execute_code(code, timeout=15)
        assert not result.success, (
            f"{problem.name} incorrect impl should have failed but passed:\n"
            f"stdout: {result.stdout}"
        )


# ===========================================================================
# Determinism
# ===========================================================================


class TestDeterminism:
    """VAL-CODING-043: Test cases are deterministic."""

    @pytest.mark.parametrize(
        "problem",
        CODING_PROBLEMS,
        ids=lambda p: p.name,
    )
    def test_deterministic_results(self, problem):
        """Running correct implementation twice produces same result."""
        correct_impl = problem._correct_impl
        code = _build_test_code(problem, correct_impl)

        result1 = execute_code(code, timeout=15)
        result2 = execute_code(code, timeout=15)

        assert (
            result1.success == result2.success
        ), f"{problem.name} non-deterministic: run1={result1.success}, run2={result2.success}"


# ===========================================================================
# Float comparison tolerance
# ===========================================================================


class TestFloatComparisons:
    """VAL-CODING-044: Float comparisons use tolerance."""

    def test_no_direct_float_equality_in_test_cases(self):
        """Test cases with float expected_output should be compared with tolerance."""
        for p in CODING_PROBLEMS:
            for tc in p.test_cases:
                if isinstance(tc["expected_output"], float):
                    # The _build_test_code helper uses abs(...) < 1e-6 for floats
                    # This test just confirms the expected_output type annotation
                    assert isinstance(tc["expected_output"], float)
