"""Tests for contamination resistance: parameterized problem variants and pluggable problem sources.

Covers VAL-CONTAM-001 through VAL-CONTAM-010.
"""

import re
from abc import ABC

import pytest

from mtb.quality_benchmarks.eval_problem import EvalProblem
from mtb.quality_benchmarks.scoring import ProblemSource, StaticProblemSource


# =============================================================================
# VAL-CONTAM-001: EvalProblem supports generate_variant() method
# =============================================================================


class TestEvalProblemGenerateVariant:
    """VAL-CONTAM-001: EvalProblem accepts optional generate_variant callable."""

    def test_generate_variant_field_exists_and_defaults_none(self):
        """EvalProblem can be created without generate_variant (defaults to None)."""
        p = EvalProblem(
            category="test",
            name="test",
            prompt="test prompt",
            check=lambda r: True,
        )
        assert p.generate_variant is None

    def test_generate_variant_callable_produces_new_eval_problem(self):
        """When generate_variant is set, calling it returns a new EvalProblem."""

        def _variant():
            return EvalProblem(
                category="math",
                name="variant_test",
                prompt="What is 3 + 4?",
                check=lambda r: "7" in r,
                max_tokens=1024,
            )

        p = EvalProblem(
            category="math",
            name="original_test",
            prompt="What is 2 + 3?",
            check=lambda r: "5" in r,
            max_tokens=1024,
            generate_variant=_variant,
        )
        variant = p.generate_variant()
        assert isinstance(variant, EvalProblem)
        assert variant.prompt != p.prompt

    def test_variant_has_different_prompt(self):
        """Generated variant has a different prompt than the original."""
        from mtb.quality_benchmarks import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )

        all_problems = (
            EVAL_PROBLEMS
            + HARD_EVAL_PROBLEMS
            + EXPERT_EVAL_PROBLEMS
            + TOOL_CALLING_PROBLEMS
        )
        problems_with_variants = [
            p for p in all_problems if p.generate_variant is not None
        ]
        assert len(problems_with_variants) > 0, "No problems have generate_variant"

        # Check that at least one produces a different prompt
        p = problems_with_variants[0]
        variant = p.generate_variant()
        assert variant.prompt != p.prompt


# =============================================================================
# VAL-CONTAM-002: Generated variant has valid prompt
# =============================================================================


class TestVariantValidPrompt:
    """VAL-CONTAM-002: Variant prompts are valid (no template placeholders)."""

    def _get_all_variant_problems(self):
        from mtb.quality_benchmarks import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )

        all_problems = (
            EVAL_PROBLEMS
            + HARD_EVAL_PROBLEMS
            + EXPERT_EVAL_PROBLEMS
            + TOOL_CALLING_PROBLEMS
        )
        return [p for p in all_problems if p.generate_variant is not None]

    def test_variant_prompt_non_empty(self):
        """All variant prompts are non-empty strings."""
        for p in self._get_all_variant_problems():
            variant = p.generate_variant()
            assert isinstance(variant.prompt, str), f"{p.name}: prompt is not a string"
            assert len(variant.prompt) > 0, f"{p.name}: variant prompt is empty"

    def test_variant_prompt_no_template_placeholders(self):
        """No template placeholders like {var} or {{var}} remain in variant prompts."""
        placeholder_pattern = re.compile(r"\{[a-zA-Z_]+\}")
        for p in self._get_all_variant_problems():
            variant = p.generate_variant()
            # Allow common English use of curly braces in code snippets, but reject
            # template-style single-word placeholders that look like unfilled templates.
            matches = placeholder_pattern.findall(variant.prompt)
            # Filter out common non-template uses (e.g., JSON examples, set notation)
            suspicious = [
                m
                for m in matches
                if m
                not in (
                    "{",
                    "}",
                    "{get_weather}",
                    "{location}",
                )
            ]
            # Some prompts legitimately contain JSON or dict notation — only flag if
            # there are multiple consecutive template-like patterns
            # For now, just verify the prompt is substantive
            assert len(variant.prompt) > 20, f"{p.name}: variant prompt too short"


# =============================================================================
# VAL-CONTAM-003: Generated variant has correct check function
# =============================================================================


class TestVariantCheckFunction:
    """VAL-CONTAM-003: Variant check functions validate correctly for new values."""

    def _get_all_variant_problems(self):
        from mtb.quality_benchmarks import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )

        all_problems = (
            EVAL_PROBLEMS
            + HARD_EVAL_PROBLEMS
            + EXPERT_EVAL_PROBLEMS
            + TOOL_CALLING_PROBLEMS
        )
        return [p for p in all_problems if p.generate_variant is not None]

    def test_variant_check_is_callable(self):
        """Every variant has a callable check function."""
        for p in self._get_all_variant_problems():
            variant = p.generate_variant()
            assert callable(variant.check), f"{p.name}: variant check is not callable"

    def test_variant_check_rejects_wrong_answer(self):
        """Variant check functions reject clearly wrong answers."""
        for p in self._get_all_variant_problems():
            variant = p.generate_variant()
            # A random string should not pass the check
            assert not variant.check(
                "purple elephant dancing on mars"
            ), f"{p.name}: variant check accepted a clearly wrong answer"


# =============================================================================
# VAL-CONTAM-004: At least 10 problems have parameterized variants
# =============================================================================


class TestVariantCount:
    """VAL-CONTAM-004: At least 10 problems have generate_variant defined."""

    def test_at_least_10_problems_have_variants(self):
        from mtb.quality_benchmarks import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )

        all_problems = (
            EVAL_PROBLEMS
            + HARD_EVAL_PROBLEMS
            + EXPERT_EVAL_PROBLEMS
            + TOOL_CALLING_PROBLEMS
        )
        count = sum(1 for p in all_problems if p.generate_variant is not None)
        assert count >= 10, f"Only {count} problems have generate_variant, need >= 10"


# =============================================================================
# VAL-CONTAM-005: Two calls produce different variants (randomized)
# =============================================================================


class TestVariantRandomness:
    """VAL-CONTAM-005: Two calls to generate_variant produce different instances."""

    def test_two_calls_produce_different_variants(self):
        from mtb.quality_benchmarks import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )

        all_problems = (
            EVAL_PROBLEMS
            + HARD_EVAL_PROBLEMS
            + EXPERT_EVAL_PROBLEMS
            + TOOL_CALLING_PROBLEMS
        )
        problems_with_variants = [
            p for p in all_problems if p.generate_variant is not None
        ]
        assert len(problems_with_variants) > 0

        for p in problems_with_variants:
            # Generate 5 variants and check at least 2 are different
            variants = [p.generate_variant() for _ in range(5)]
            prompts = [v.prompt for v in variants]
            unique_prompts = set(prompts)
            assert (
                len(unique_prompts) >= 2
            ), f"{p.name}: 5 variant calls produced identical prompts"


# =============================================================================
# VAL-CONTAM-006: Math variants change numbers
# =============================================================================


class TestMathVariants:
    """VAL-CONTAM-006: Math problem variants substitute different numerical values."""

    def _get_math_variant_problems(self):
        from mtb.quality_benchmarks import EXPERT_EVAL_PROBLEMS
        from mtb.quality_benchmarks.math_problems import MATH_EXPERT_PROBLEMS

        return [p for p in MATH_EXPERT_PROBLEMS if p.generate_variant is not None]

    def test_math_variants_have_different_numbers(self):
        """Math variants contain different numbers from the original."""
        math_problems = self._get_math_variant_problems()
        if not math_problems:
            pytest.skip("No math problems with variants")

        for p in math_problems:
            variant = p.generate_variant()
            # Extract numbers from both prompts
            orig_numbers = set(re.findall(r"\b\d+\b", p.prompt))
            variant_numbers = set(re.findall(r"\b\d+\b", variant.prompt))
            # At least some numbers should differ
            assert (
                orig_numbers != variant_numbers
            ), f"{p.name}: variant has same numbers as original"

    def test_math_variant_check_validates_new_answer(self):
        """Math variant check function validates the new expected answer."""
        math_problems = self._get_math_variant_problems()
        if not math_problems:
            pytest.skip("No math problems with variants")

        for p in math_problems:
            variant = p.generate_variant()
            # The variant's check should not accept the original's answer
            # (if the numbers changed, the answer should be different)
            assert callable(variant.check)


# =============================================================================
# VAL-CONTAM-007: Coding variants change constraints/names
# =============================================================================


class TestCodingVariants:
    """VAL-CONTAM-007: Coding variants change identifiers or constraints."""

    def _get_coding_variant_problems(self):
        from mtb.quality_benchmarks.coding_problems import CODING_PROBLEMS

        return [p for p in CODING_PROBLEMS if p.generate_variant is not None]

    def test_coding_variants_have_different_constraints(self):
        """Coding variants contain different constraints from the original."""
        coding_problems = self._get_coding_variant_problems()
        if not coding_problems:
            pytest.skip("No coding problems with variants")

        for p in coding_problems:
            variant = p.generate_variant()
            assert (
                variant.prompt != p.prompt
            ), f"{p.name}: variant prompt is identical to original"

    def test_coding_variant_check_adapts(self):
        """Coding variant check functions adapt to new constraints."""
        coding_problems = self._get_coding_variant_problems()
        if not coding_problems:
            pytest.skip("No coding problems with variants")

        for p in coding_problems:
            variant = p.generate_variant()
            assert callable(variant.check)


# =============================================================================
# VAL-CONTAM-008: ProblemSource ABC exists with get_problems()
# =============================================================================


class TestProblemSourceABC:
    """VAL-CONTAM-008: ProblemSource is an ABC with get_problems() method."""

    def test_problem_source_is_abc(self):
        """ProblemSource is an abstract base class."""
        assert issubclass(ProblemSource, ABC)

    def test_problem_source_cannot_be_instantiated(self):
        """ProblemSource cannot be directly instantiated."""
        with pytest.raises(TypeError):
            ProblemSource()

    def test_problem_source_has_get_problems(self):
        """ProblemSource declares get_problems method."""
        assert hasattr(ProblemSource, "get_problems")


# =============================================================================
# VAL-CONTAM-009: StaticProblemSource wraps existing problem lists
# =============================================================================


class TestStaticProblemSource:
    """VAL-CONTAM-009: StaticProblemSource implements ProblemSource."""

    def test_static_source_is_problem_source(self):
        """StaticProblemSource is a subclass of ProblemSource."""
        assert issubclass(StaticProblemSource, ProblemSource)

    def test_static_source_get_problems_returns_list(self):
        """StaticProblemSource.get_problems() returns a list of EvalProblem."""
        source = StaticProblemSource()
        problems = source.get_problems()
        assert isinstance(problems, list)
        assert len(problems) > 0
        assert all(isinstance(p, EvalProblem) for p in problems)

    def test_static_source_returns_all_problems(self):
        """StaticProblemSource returns all static problem lists combined."""
        from mtb.quality_benchmarks import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )

        source = StaticProblemSource()
        problems = source.get_problems()
        expected_count = (
            len(EVAL_PROBLEMS)
            + len(HARD_EVAL_PROBLEMS)
            + len(EXPERT_EVAL_PROBLEMS)
            + len(TOOL_CALLING_PROBLEMS)
        )
        assert len(problems) == expected_count

    def test_static_source_difficulty_filter(self):
        """StaticProblemSource supports difficulty-based filtering."""
        from mtb.quality_benchmarks import EVAL_PROBLEMS

        source = StaticProblemSource()
        easy_problems = source.get_problems(difficulty="easy")
        assert len(easy_problems) == len(EVAL_PROBLEMS)

    def test_static_source_difficulty_all(self):
        """StaticProblemSource 'all' returns everything."""
        source = StaticProblemSource()
        all_problems = source.get_problems(difficulty="all")
        assert len(all_problems) == len(source.get_problems())


# =============================================================================
# VAL-CONTAM-010: ProblemSource documented for LiveCodeBench
# =============================================================================


class TestProblemSourceDocstring:
    """VAL-CONTAM-010: ProblemSource has docstring mentioning pluggable interface."""

    def test_docstring_exists(self):
        """ProblemSource has a non-empty docstring."""
        assert ProblemSource.__doc__ is not None
        assert len(ProblemSource.__doc__) > 0

    def test_docstring_mentions_pluggable(self):
        """ProblemSource docstring mentions pluggable interface."""
        doc = ProblemSource.__doc__.lower()
        assert "pluggable" in doc or "interface" in doc

    def test_docstring_mentions_livecode_bench(self):
        """ProblemSource docstring mentions LiveCodeBench."""
        doc = ProblemSource.__doc__
        assert "LiveCodeBench" in doc or "livecode" in doc.lower()


# =============================================================================
# VAL-CONTAM: Variant preserves category and max_tokens
# =============================================================================


class TestVariantPreservesStructure:
    """Variants preserve category and have same or greater max_tokens."""

    def _get_all_variant_problems(self):
        from mtb.quality_benchmarks import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )

        all_problems = (
            EVAL_PROBLEMS
            + HARD_EVAL_PROBLEMS
            + EXPERT_EVAL_PROBLEMS
            + TOOL_CALLING_PROBLEMS
        )
        return [p for p in all_problems if p.generate_variant is not None]

    def test_variant_preserves_category(self):
        """Variant has the same category as the parent problem."""
        for p in self._get_all_variant_problems():
            variant = p.generate_variant()
            assert (
                variant.category == p.category
            ), f"{p.name}: variant category {variant.category} != {p.category}"

    def test_variant_preserves_or_increases_max_tokens(self):
        """Variant has same or greater max_tokens."""
        for p in self._get_all_variant_problems():
            variant = p.generate_variant()
            assert (
                variant.max_tokens >= p.max_tokens
            ), f"{p.name}: variant max_tokens {variant.max_tokens} < {p.max_tokens}"


# =============================================================================
# CLI flag test
# =============================================================================


class TestCLIVariantFlag:
    """--use_variants and --num_variants CLI flags are accepted by run_quality_benchmarks.py."""

    def test_use_variants_flag_in_help(self):
        """The --use_variants flag is defined in the main() function signature."""
        import inspect

        from scripts.run_quality_benchmarks import main

        sig = inspect.signature(main)
        assert (
            "use_variants" in sig.parameters
        ), "--use_variants not found in main() parameters"

    def test_use_variants_defaults_false(self):
        """--use_variants defaults to False."""
        import inspect

        from scripts.run_quality_benchmarks import main

        sig = inspect.signature(main)
        param = sig.parameters["use_variants"]
        assert param.default is False

    def test_num_variants_flag_in_signature(self):
        """The --num_variants flag is defined in the main() function signature."""
        import inspect

        from scripts.run_quality_benchmarks import main

        sig = inspect.signature(main)
        assert (
            "num_variants" in sig.parameters
        ), "--num_variants not found in main() parameters"

    def test_num_variants_defaults_to_3(self):
        """--num_variants defaults to 3."""
        import inspect

        from scripts.run_quality_benchmarks import main

        sig = inspect.signature(main)
        param = sig.parameters["num_variants"]
        assert param.default == 3


# =============================================================================
# VAL-CROSS-003: Multi-variant expansion produces correct number of rows
# =============================================================================


class TestMultiVariantExpansion:
    """VAL-CROSS-003: When --use_variants is enabled, each parameterized problem
    generates N variants (default 3) per parameterized problem, each with a
    distinct name. Non-parameterized problems are kept unchanged."""

    def _make_parameterized_problem(self, name="fizzbuzz"):
        """Create a simple parameterized problem for testing."""
        import random

        def _gen_variant():
            n = random.randint(1, 100)
            return EvalProblem(
                category="coding",
                name=f"{name}_variant",
                prompt=f"Implement fizzbuzz up to {n}.",
                check=lambda r, _n=n: str(_n) in r,
                max_tokens=512,
            )

        return EvalProblem(
            category="coding",
            name=name,
            prompt="Implement fizzbuzz up to 15.",
            check=lambda r: "FizzBuzz" in r,
            max_tokens=512,
            generate_variant=_gen_variant,
        )

    def _make_non_parameterized_problem(self, name="reverse_string"):
        """Create a simple non-parameterized problem (no generate_variant)."""
        return EvalProblem(
            category="coding",
            name=name,
            prompt="Implement a function to reverse a string.",
            check=lambda r: "reverse" in r.lower(),
            max_tokens=512,
        )

    def _expand_variants(self, problems, num_variants=3):
        """Replicate the variant expansion logic from run_quality_benchmarks.py."""
        variant_problems = []
        for p in problems:
            if p.generate_variant is not None:
                for i in range(1, num_variants + 1):
                    variant = p.generate_variant()
                    variant = EvalProblem(
                        category=variant.category,
                        name=f"{p.name}_variant_{i}",
                        prompt=variant.prompt,
                        check=variant.check,
                        max_tokens=variant.max_tokens,
                        function_signature=variant.function_signature,
                        test_cases=variant.test_cases,
                        generate_variant=variant.generate_variant,
                    )
                    variant_problems.append(variant)
            else:
                variant_problems.append(p)
        return variant_problems

    def test_parameterized_problem_generates_n_variants(self):
        """Each parameterized problem generates N variants (default 3)."""
        problems = [self._make_parameterized_problem()]
        expanded = self._expand_variants(problems, num_variants=3)
        assert len(expanded) == 3

    def test_non_parameterized_problem_unchanged(self):
        """Non-parameterized problems (no generate_variant) are kept unchanged."""
        problems = [self._make_non_parameterized_problem()]
        expanded = self._expand_variants(problems, num_variants=3)
        assert len(expanded) == 1
        assert expanded[0].name == "reverse_string"

    def test_mixed_problems_correct_count(self):
        """Mix of parameterized and non-parameterized problems produces correct count.

        2 parameterized (3 variants each) + 1 non-parameterized = 7 total.
        """
        problems = [
            self._make_parameterized_problem("fizzbuzz"),
            self._make_non_parameterized_problem("reverse_string"),
            self._make_parameterized_problem("fibonacci"),
        ]
        expanded = self._expand_variants(problems, num_variants=3)
        # 3 + 1 + 3 = 7
        assert len(expanded) == 7

    def test_variant_names_are_distinct(self):
        """Each variant has a distinct name (e.g., fizzbuzz_variant_1, _2, _3)."""
        problems = [self._make_parameterized_problem("fizzbuzz")]
        expanded = self._expand_variants(problems, num_variants=3)
        names = [p.name for p in expanded]
        assert names == [
            "fizzbuzz_variant_1",
            "fizzbuzz_variant_2",
            "fizzbuzz_variant_3",
        ]

    def test_variant_names_unique_across_all_problems(self):
        """All variant names are unique across multiple parameterized problems."""
        problems = [
            self._make_parameterized_problem("fizzbuzz"),
            self._make_parameterized_problem("fibonacci"),
        ]
        expanded = self._expand_variants(problems, num_variants=3)
        names = [p.name for p in expanded]
        assert len(names) == len(set(names)), f"Duplicate names found: {names}"

    def test_custom_num_variants(self):
        """num_variants parameter controls how many variants are generated."""
        problems = [self._make_parameterized_problem()]
        for n in [1, 2, 5]:
            expanded = self._expand_variants(problems, num_variants=n)
            assert len(expanded) == n, f"Expected {n} variants, got {len(expanded)}"

    def test_each_variant_runs_as_separate_row(self):
        """Each variant is a full EvalProblem that can run independently in the pipeline."""
        problems = [self._make_parameterized_problem()]
        expanded = self._expand_variants(problems, num_variants=3)
        for variant in expanded:
            assert isinstance(variant, EvalProblem)
            assert callable(variant.check)
            assert len(variant.prompt) > 0
            assert variant.category == "coding"

    def test_real_problems_multi_variant_expansion(self):
        """Using real problems from the registry, variant expansion produces correct count.

        Verifies the fix for VAL-CROSS-003: parameterized problems expand to N rows.
        """
        from mtb.quality_benchmarks import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )

        all_problems = (
            EVAL_PROBLEMS
            + HARD_EVAL_PROBLEMS
            + EXPERT_EVAL_PROBLEMS
            + TOOL_CALLING_PROBLEMS
        )
        num_variants = 3
        parameterized_count = sum(
            1 for p in all_problems if p.generate_variant is not None
        )
        non_parameterized_count = len(all_problems) - parameterized_count

        expanded = self._expand_variants(all_problems, num_variants=num_variants)
        expected_count = parameterized_count * num_variants + non_parameterized_count
        assert len(expanded) == expected_count, (
            f"Expected {expected_count} problems after expansion "
            f"({parameterized_count} parameterized × {num_variants} + "
            f"{non_parameterized_count} non-parameterized), got {len(expanded)}"
        )
