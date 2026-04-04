"""Tests for quality benchmarks module structure after the split.

Verifies:
- Category files exist and export problems
- eval_problems.py re-exports everything for backward compatibility
- EvalProblem backward compatibility with new optional fields
- _strip_thinking accessible from shared location
- No duplicate problem names across all lists
- Total problem count matches expected (46)
- __init__.py exports all four problem lists
"""

import pytest


# ---------------------------------------------------------------------------
# VAL-MODULE-001: __init__.py exports all four problem lists
# ---------------------------------------------------------------------------


class TestInitExports:
    def test_init_exports_eval_problems(self):
        from mtb.quality_benchmarks import EVAL_PROBLEMS

        assert isinstance(EVAL_PROBLEMS, list)
        assert len(EVAL_PROBLEMS) > 0

    def test_init_exports_hard_eval_problems(self):
        from mtb.quality_benchmarks import HARD_EVAL_PROBLEMS

        assert isinstance(HARD_EVAL_PROBLEMS, list)
        assert len(HARD_EVAL_PROBLEMS) > 0

    def test_init_exports_expert_eval_problems(self):
        from mtb.quality_benchmarks import EXPERT_EVAL_PROBLEMS

        assert isinstance(EXPERT_EVAL_PROBLEMS, list)
        assert len(EXPERT_EVAL_PROBLEMS) > 0

    def test_init_exports_tool_calling_problems(self):
        from mtb.quality_benchmarks import TOOL_CALLING_PROBLEMS

        assert isinstance(TOOL_CALLING_PROBLEMS, list)
        assert len(TOOL_CALLING_PROBLEMS) > 0

    def test_init_exports_run_quality_benchmark(self):
        from mtb.quality_benchmarks import run_quality_benchmark

        assert callable(run_quality_benchmark)


# ---------------------------------------------------------------------------
# VAL-MODULE-002: Backward-compatible imports from eval_problems
# ---------------------------------------------------------------------------


class TestEvalProblemsBackwardCompat:
    def test_import_eval_problem_class(self):
        from mtb.quality_benchmarks.eval_problems import EvalProblem

        assert EvalProblem is not None

    def test_import_problem_lists(self):
        from mtb.quality_benchmarks.eval_problems import (
            EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )

        assert len(EVAL_PROBLEMS) == 15
        assert len(HARD_EVAL_PROBLEMS) == 10
        assert len(EXPERT_EVAL_PROBLEMS) == 16
        assert len(TOOL_CALLING_PROBLEMS) == 5

    def test_import_check_functions(self):
        """All _check_* functions must be importable from eval_problems."""
        from mtb.quality_benchmarks.eval_problems import (
            _check_fizzbuzz,
            _check_reverse_string,
            _check_fibonacci,
            _check_binary_search,
            _check_palindrome,
            _check_train_problem,
            _check_coin_problem,
            _check_workers_problem,
            _check_age_problem,
            _check_sequence_problem,
            _check_json_output,
            _check_list_format,
            _check_word_constraint,
            _check_code_with_comments,
            _check_no_thinking,
            _check_lru_cache,
            _check_flatten_nested,
            _check_longest_palindrome_substring,
            _check_calculator,
            _check_buggy_merge_sort,
            _check_compound_interest,
            _check_circular_seating,
            _check_logic_puzzle,
            _check_bayes_theorem,
            _check_proof_bug,
            _check_modular_arithmetic,
            _check_inclusion_exclusion,
            _check_bouncing_ball,
            _check_markdown_to_html,
            _check_data_pipeline,
            _check_retry_decorator,
            _check_einstein_riddle,
            _check_three_urns,
            _check_topological_sort,
            _check_constrained_factorial,
            _check_library_schema,
            _check_adversarial_transform,
            _check_multi_doc_summary,
            _check_structured_meeting_notes,
            _check_tone_rewrite,
            _check_contradiction_detection,
            _check_tool_call_weather,
            _check_tool_call_calculator,
            _check_tool_call_multi_step,
            _check_tool_call_json_args,
            _check_tool_call_selection,
        )

        # Verify they are all callable
        all_checks = [
            _check_fizzbuzz,
            _check_reverse_string,
            _check_fibonacci,
            _check_binary_search,
            _check_palindrome,
            _check_train_problem,
            _check_coin_problem,
            _check_workers_problem,
            _check_age_problem,
            _check_sequence_problem,
            _check_json_output,
            _check_list_format,
            _check_word_constraint,
            _check_code_with_comments,
            _check_no_thinking,
            _check_lru_cache,
            _check_flatten_nested,
            _check_longest_palindrome_substring,
            _check_calculator,
            _check_buggy_merge_sort,
            _check_compound_interest,
            _check_circular_seating,
            _check_logic_puzzle,
            _check_bayes_theorem,
            _check_proof_bug,
            _check_modular_arithmetic,
            _check_inclusion_exclusion,
            _check_bouncing_ball,
            _check_markdown_to_html,
            _check_data_pipeline,
            _check_retry_decorator,
            _check_einstein_riddle,
            _check_three_urns,
            _check_topological_sort,
            _check_constrained_factorial,
            _check_library_schema,
            _check_adversarial_transform,
            _check_multi_doc_summary,
            _check_structured_meeting_notes,
            _check_tone_rewrite,
            _check_contradiction_detection,
            _check_tool_call_weather,
            _check_tool_call_calculator,
            _check_tool_call_multi_step,
            _check_tool_call_json_args,
            _check_tool_call_selection,
        ]
        for check in all_checks:
            assert callable(check)

    def test_import_helper_functions(self):
        """Helper functions must be importable from eval_problems."""
        from mtb.quality_benchmarks.eval_problems import (
            _strip_thinking,
            _contains_any,
            _extract_number,
        )

        assert callable(_strip_thinking)
        assert callable(_contains_any)
        assert callable(_extract_number)


# ---------------------------------------------------------------------------
# VAL-MODULE-003: Category-specific module files exist and export problems
# ---------------------------------------------------------------------------


class TestCategoryModules:
    def test_coding_problems_module_exists(self):
        from mtb.quality_benchmarks.coding_problems import (
            CODING_EASY_PROBLEMS,
            CODING_HARD_PROBLEMS,
            CODING_EXPERT_PROBLEMS,
        )

        assert len(CODING_EASY_PROBLEMS) == 5
        assert len(CODING_HARD_PROBLEMS) == 5
        assert len(CODING_EXPERT_PROBLEMS) == 3

    def test_reasoning_problems_module_exists(self):
        from mtb.quality_benchmarks.reasoning_problems import (
            REASONING_EASY_PROBLEMS,
            REASONING_HARD_PROBLEMS,
            REASONING_EXPERT_PROBLEMS,
        )

        assert len(REASONING_EASY_PROBLEMS) == 5
        assert len(REASONING_HARD_PROBLEMS) == 5
        assert len(REASONING_EXPERT_PROBLEMS) == 3

    def test_instruction_problems_module_exists(self):
        from mtb.quality_benchmarks.instruction_problems import (
            INSTRUCTION_EASY_PROBLEMS,
            INSTRUCTION_EXPERT_PROBLEMS,
        )

        assert len(INSTRUCTION_EASY_PROBLEMS) == 5
        assert len(INSTRUCTION_EXPERT_PROBLEMS) == 3

    def test_math_problems_module_exists(self):
        from mtb.quality_benchmarks.math_problems import MATH_EXPERT_PROBLEMS

        assert len(MATH_EXPERT_PROBLEMS) == 3

    def test_writing_problems_module_exists(self):
        from mtb.quality_benchmarks.writing_problems import WRITING_EXPERT_PROBLEMS

        assert len(WRITING_EXPERT_PROBLEMS) == 4

    def test_all_category_problems_are_eval_problem_instances(self):
        from mtb.quality_benchmarks.eval_problems import EvalProblem
        from mtb.quality_benchmarks.coding_problems import (
            CODING_EASY_PROBLEMS,
            CODING_HARD_PROBLEMS,
            CODING_EXPERT_PROBLEMS,
        )
        from mtb.quality_benchmarks.reasoning_problems import (
            REASONING_EASY_PROBLEMS,
            REASONING_HARD_PROBLEMS,
            REASONING_EXPERT_PROBLEMS,
        )
        from mtb.quality_benchmarks.instruction_problems import (
            INSTRUCTION_EASY_PROBLEMS,
            INSTRUCTION_EXPERT_PROBLEMS,
        )
        from mtb.quality_benchmarks.math_problems import MATH_EXPERT_PROBLEMS
        from mtb.quality_benchmarks.writing_problems import WRITING_EXPERT_PROBLEMS

        all_problems = (
            CODING_EASY_PROBLEMS
            + CODING_HARD_PROBLEMS
            + CODING_EXPERT_PROBLEMS
            + REASONING_EASY_PROBLEMS
            + REASONING_HARD_PROBLEMS
            + REASONING_EXPERT_PROBLEMS
            + INSTRUCTION_EASY_PROBLEMS
            + INSTRUCTION_EXPERT_PROBLEMS
            + MATH_EXPERT_PROBLEMS
            + WRITING_EXPERT_PROBLEMS
        )
        for p in all_problems:
            assert isinstance(p, EvalProblem), f"{p.name} is not an EvalProblem"


# ---------------------------------------------------------------------------
# VAL-MODULE-008: _strip_thinking accessible from shared location
# ---------------------------------------------------------------------------


class TestSharedUtils:
    def test_strip_thinking_from_utils(self):
        from mtb.quality_benchmarks.utils import _strip_thinking

        result = _strip_thinking("<think>ignore</think>Real answer")
        assert result == "Real answer"

    def test_contains_any_from_utils(self):
        from mtb.quality_benchmarks.utils import _contains_any

        assert _contains_any("hello world", ["hello"]) is True
        assert _contains_any("hello world", ["foo"]) is False

    def test_extract_number_from_utils(self):
        from mtb.quality_benchmarks.utils import _extract_number

        assert _extract_number("the answer is 42") == 42.0

    def test_strip_thinking_same_function_in_eval_problems_and_utils(self):
        """The function in eval_problems and utils should be the same object."""
        from mtb.quality_benchmarks.eval_problems import _strip_thinking as sp1
        from mtb.quality_benchmarks.utils import _strip_thinking as sp2

        # They should be the same function (re-exported, not copied)
        assert sp1 is sp2


# ---------------------------------------------------------------------------
# VAL-MODULE-011: EvalProblem backward compatibility with new optional fields
# ---------------------------------------------------------------------------


class TestEvalProblemBackwardCompat:
    def test_old_style_construction(self):
        """EvalProblem can be constructed with only original fields."""
        from mtb.quality_benchmarks.eval_problems import EvalProblem

        p = EvalProblem(
            category="test",
            name="test_problem",
            prompt="test prompt",
            check=lambda r: True,
        )
        assert p.category == "test"
        assert p.name == "test_problem"
        assert p.prompt == "test prompt"
        assert p.max_tokens == 512

    def test_new_fields_default_to_none(self):
        """New optional fields default to None."""
        from mtb.quality_benchmarks.eval_problems import EvalProblem

        p = EvalProblem(
            category="test",
            name="test_problem",
            prompt="test prompt",
            check=lambda r: True,
        )
        assert p.function_signature is None
        assert p.test_cases is None
        assert p.generate_variant is None

    def test_new_fields_can_be_set(self):
        """New optional fields can be provided."""
        from mtb.quality_benchmarks.eval_problems import EvalProblem

        p = EvalProblem(
            category="coding",
            name="test_coding",
            prompt="Write fizzbuzz",
            check=lambda r: True,
            function_signature="def fizzbuzz(n: int) -> str:",
            test_cases=[{"input": 15, "expected": "FizzBuzz"}],
            generate_variant=lambda: None,
        )
        assert p.function_signature == "def fizzbuzz(n: int) -> str:"
        assert len(p.test_cases) == 1
        assert callable(p.generate_variant)


# ---------------------------------------------------------------------------
# Problem count and uniqueness across all tiers
# ---------------------------------------------------------------------------


class TestProblemIntegrity:
    def test_total_problem_count_is_46(self):
        """Total problems across all tiers should be 46."""
        from mtb.quality_benchmarks.eval_problems import (
            EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )

        total = (
            len(EVAL_PROBLEMS)
            + len(HARD_EVAL_PROBLEMS)
            + len(EXPERT_EVAL_PROBLEMS)
            + len(TOOL_CALLING_PROBLEMS)
        )
        assert total == 46

    def test_no_duplicate_names_across_all_lists(self):
        """No duplicate problem names across all four problem lists."""
        from mtb.quality_benchmarks.eval_problems import (
            EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )

        all_problems = (
            EVAL_PROBLEMS
            + HARD_EVAL_PROBLEMS
            + EXPERT_EVAL_PROBLEMS
            + TOOL_CALLING_PROBLEMS
        )
        names = [p.name for p in all_problems]
        assert len(names) == len(set(names)), (
            f"Duplicate names found: " f"{[n for n in names if names.count(n) > 1]}"
        )

    def test_all_problems_have_required_fields(self):
        """Every problem has category, name, prompt, check, and max_tokens."""
        from mtb.quality_benchmarks.eval_problems import (
            EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )

        all_problems = (
            EVAL_PROBLEMS
            + HARD_EVAL_PROBLEMS
            + EXPERT_EVAL_PROBLEMS
            + TOOL_CALLING_PROBLEMS
        )
        for p in all_problems:
            assert p.category, f"{p.name} has empty category"
            assert p.name, f"Problem has empty name"
            assert p.prompt, f"{p.name} has empty prompt"
            assert callable(p.check), f"{p.name} check is not callable"
            assert (
                isinstance(p.max_tokens, int) and p.max_tokens > 0
            ), f"{p.name} has invalid max_tokens={p.max_tokens}"
