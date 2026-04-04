"""Tests for mtb/quality_benchmarks/scoring.py — weighted scoring system.

Covers all VAL-SCORING-001 through VAL-SCORING-010 assertions.
"""

import pytest

from mtb.quality_benchmarks.scoring import (
    TIER_WEIGHTS,
    compute_weighted_score,
)


# ---------------------------------------------------------------------------
# VAL-SCORING-008: Weight constants match specification
# ---------------------------------------------------------------------------
class TestTierWeights:
    def test_weight_constants_match_specification(self):
        """TIER_WEIGHTS must be: easy=1, hard=2, expert=3, tool_calling=3."""
        assert TIER_WEIGHTS == {
            "easy": 1,
            "hard": 2,
            "expert": 3,
            "tool_calling": 3,
        }

    def test_tier_weights_importable(self):
        """Weight constants must be importable from scoring module."""
        from mtb.quality_benchmarks.scoring import TIER_WEIGHTS as tw

        assert isinstance(tw, dict)
        assert len(tw) == 4


# ---------------------------------------------------------------------------
# VAL-SCORING-001: Correct weighted score for known mixed results
# ---------------------------------------------------------------------------
class TestKnownMixedResults:
    def test_known_mixed_weighted_score(self):
        """
        Given:
          Easy: 10/15 pass (weight 1x)
          Hard: 6/10 pass (weight 2x)
          Expert: 9/16 pass (weight 3x)
          Tool Calling: 25/40 pass (weight 3x)

        Expected weighted_score = (1*10 + 2*6 + 3*9 + 3*25) / (1*15 + 2*10 + 3*16 + 3*40)
                                = (10 + 12 + 27 + 75) / (15 + 20 + 48 + 120)
                                = 124 / 203 ≈ 0.6108
        """
        # Build results dict mapping problem_name -> passed (bool)
        # We need 15 easy problems, 10 hard, 16 expert, 40 tool_calling
        results = {}

        # Easy: 10 pass, 5 fail (use real problem names from the lists)
        from mtb.quality_benchmarks import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )

        for i, p in enumerate(EVAL_PROBLEMS):
            results[p.name] = i < 10  # first 10 pass

        for i, p in enumerate(HARD_EVAL_PROBLEMS):
            results[p.name] = i < 6  # first 6 pass

        for i, p in enumerate(EXPERT_EVAL_PROBLEMS):
            results[p.name] = i < 9  # first 9 pass

        for i, p in enumerate(TOOL_CALLING_PROBLEMS):
            results[p.name] = i < 25  # first 25 pass

        score = compute_weighted_score(results)

        expected_weighted = 124 / 203
        assert abs(score["weighted_score"] - expected_weighted) < 1e-9
        assert isinstance(score["weighted_score"], float)

    def test_known_mixed_raw_pass_rate(self):
        """Raw pass rate = total passed / total problems."""
        from mtb.quality_benchmarks import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )

        results = {}
        for i, p in enumerate(EVAL_PROBLEMS):
            results[p.name] = i < 10
        for i, p in enumerate(HARD_EVAL_PROBLEMS):
            results[p.name] = i < 6
        for i, p in enumerate(EXPERT_EVAL_PROBLEMS):
            results[p.name] = i < 9
        for i, p in enumerate(TOOL_CALLING_PROBLEMS):
            results[p.name] = i < 25

        score = compute_weighted_score(results)

        # raw_pass_rate = (10 + 6 + 9 + 25) / (15 + 10 + 16 + 40) = 50/81
        expected_raw = 50 / 81
        assert abs(score["raw_pass_rate"] - expected_raw) < 1e-9


# ---------------------------------------------------------------------------
# VAL-SCORING-002: All pass yields 100%
# ---------------------------------------------------------------------------
class TestAllPass:
    def test_all_pass_weighted_score(self):
        """All problems pass → weighted_score == 1.0."""
        from mtb.quality_benchmarks import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )

        results = {}
        for p in (
            EVAL_PROBLEMS
            + HARD_EVAL_PROBLEMS
            + EXPERT_EVAL_PROBLEMS
            + TOOL_CALLING_PROBLEMS
        ):
            results[p.name] = True

        score = compute_weighted_score(results)
        assert score["weighted_score"] == 1.0

    def test_all_pass_raw_pass_rate(self):
        """All problems pass → raw_pass_rate == 1.0."""
        from mtb.quality_benchmarks import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )

        results = {}
        for p in (
            EVAL_PROBLEMS
            + HARD_EVAL_PROBLEMS
            + EXPERT_EVAL_PROBLEMS
            + TOOL_CALLING_PROBLEMS
        ):
            results[p.name] = True

        score = compute_weighted_score(results)
        assert score["raw_pass_rate"] == 1.0


# ---------------------------------------------------------------------------
# VAL-SCORING-003: All fail yields 0%
# ---------------------------------------------------------------------------
class TestAllFail:
    def test_all_fail_weighted_score(self):
        """All problems fail → weighted_score == 0.0."""
        from mtb.quality_benchmarks import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )

        results = {}
        for p in (
            EVAL_PROBLEMS
            + HARD_EVAL_PROBLEMS
            + EXPERT_EVAL_PROBLEMS
            + TOOL_CALLING_PROBLEMS
        ):
            results[p.name] = False

        score = compute_weighted_score(results)
        assert score["weighted_score"] == 0.0

    def test_all_fail_raw_pass_rate(self):
        """All problems fail → raw_pass_rate == 0.0."""
        from mtb.quality_benchmarks import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )

        results = {}
        for p in (
            EVAL_PROBLEMS
            + HARD_EVAL_PROBLEMS
            + EXPERT_EVAL_PROBLEMS
            + TOOL_CALLING_PROBLEMS
        ):
            results[p.name] = False

        score = compute_weighted_score(results)
        assert score["raw_pass_rate"] == 0.0


# ---------------------------------------------------------------------------
# VAL-SCORING-004: Empty tier handled gracefully
# ---------------------------------------------------------------------------
class TestEmptyTier:
    def test_empty_tier_no_error(self):
        """If an entire tier has zero problems, no error or division by zero."""
        # Only provide results for easy tier problems (skip hard, expert, tool_calling)
        from mtb.quality_benchmarks import EVAL_PROBLEMS

        results = {}
        for p in EVAL_PROBLEMS:
            results[p.name] = True  # all easy pass

        # This should not raise, and should compute correctly from only easy tier
        score = compute_weighted_score(results)
        assert score["weighted_score"] == 1.0  # all provided pass
        assert score["raw_pass_rate"] == 1.0

    def test_skip_missing_tiers(self):
        """Only hard tier provided → weighted uses only hard weight."""
        from mtb.quality_benchmarks import HARD_EVAL_PROBLEMS

        results = {}
        for i, p in enumerate(HARD_EVAL_PROBLEMS):
            results[p.name] = i < 6  # 6/10 pass

        score = compute_weighted_score(results)
        # Only hard tier: weighted = (2*6) / (2*10) = 12/20 = 0.6
        assert abs(score["weighted_score"] - 0.6) < 1e-9
        assert abs(score["raw_pass_rate"] - 0.6) < 1e-9

    def test_completely_empty_results(self):
        """Empty results dict should not crash."""
        score = compute_weighted_score({})
        # With no problems, weighted_score should be 0.0 (no problems to score)
        assert score["weighted_score"] == 0.0
        assert score["raw_pass_rate"] == 0.0


# ---------------------------------------------------------------------------
# VAL-SCORING-005: Single problem in a single tier
# ---------------------------------------------------------------------------
class TestSingleProblem:
    def test_single_problem_pass(self):
        """One expert problem passes → weighted_score = (3*1)/(3*1) = 1.0."""
        from mtb.quality_benchmarks import EXPERT_EVAL_PROBLEMS

        results = {EXPERT_EVAL_PROBLEMS[0].name: True}
        score = compute_weighted_score(results)
        assert score["weighted_score"] == 1.0

    def test_single_problem_fail(self):
        """One expert problem fails → weighted_score = (3*0)/(3*1) = 0.0."""
        from mtb.quality_benchmarks import EXPERT_EVAL_PROBLEMS

        results = {EXPERT_EVAL_PROBLEMS[0].name: False}
        score = compute_weighted_score(results)
        assert score["weighted_score"] == 0.0

    def test_single_easy_problem(self):
        """One easy problem passes → weighted_score = (1*1)/(1*1) = 1.0."""
        from mtb.quality_benchmarks import EVAL_PROBLEMS

        results = {EVAL_PROBLEMS[0].name: True}
        score = compute_weighted_score(results)
        assert score["weighted_score"] == 1.0


# ---------------------------------------------------------------------------
# VAL-SCORING-006: Per-category weighted scores
# ---------------------------------------------------------------------------
class TestPerCategoryScores:
    def test_per_category_coding(self):
        """Per-category 'coding' weights easy=1x, hard=2x, expert=3x."""
        from mtb.quality_benchmarks import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
        )

        results = {}
        # Easy coding: 5 problems, pass 3
        easy_coding = [p for p in EVAL_PROBLEMS if p.category == "coding"]
        for i, p in enumerate(easy_coding):
            results[p.name] = i < 3  # 3/5 pass

        # Hard coding: 5 problems, pass 2
        hard_coding = [p for p in HARD_EVAL_PROBLEMS if p.category == "coding"]
        for i, p in enumerate(hard_coding):
            results[p.name] = i < 2  # 2/5 pass

        # Expert coding: 3 problems, pass 1
        expert_coding = [p for p in EXPERT_EVAL_PROBLEMS if p.category == "coding"]
        for i, p in enumerate(expert_coding):
            results[p.name] = i < 1  # 1/3 pass

        score = compute_weighted_score(results)
        # coding weighted: (1*3 + 2*2 + 3*1) / (1*5 + 2*5 + 3*3)
        #                = (3 + 4 + 3) / (5 + 10 + 9) = 10/24
        expected = 10 / 24
        assert "category_scores" in score
        assert "coding" in score["category_scores"]
        assert abs(score["category_scores"]["coding"] - expected) < 1e-9

    def test_per_category_tool_calling(self):
        """Tool calling category all at weight 3x."""
        from mtb.quality_benchmarks import TOOL_CALLING_PROBLEMS

        results = {}
        for i, p in enumerate(TOOL_CALLING_PROBLEMS):
            results[p.name] = i < 20  # 20/40 pass

        score = compute_weighted_score(results)
        # tool_calling: (3*20) / (3*40) = 60/120 = 0.5
        assert "category_scores" in score
        assert "tool_calling" in score["category_scores"]
        assert abs(score["category_scores"]["tool_calling"] - 0.5) < 1e-9

    def test_per_category_reasoning(self):
        """Per-category 'reasoning' weights easy=1x, hard=2x, expert=3x."""
        from mtb.quality_benchmarks import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
        )

        results = {}
        # Easy reasoning: 5 problems, all pass
        easy_reasoning = [p for p in EVAL_PROBLEMS if p.category == "reasoning"]
        for p in easy_reasoning:
            results[p.name] = True

        # Hard reasoning: 5 problems, all fail
        hard_reasoning = [p for p in HARD_EVAL_PROBLEMS if p.category == "reasoning"]
        for p in hard_reasoning:
            results[p.name] = False

        # Expert reasoning: 3 problems, 1 pass
        expert_reasoning = [
            p for p in EXPERT_EVAL_PROBLEMS if p.category == "reasoning"
        ]
        for i, p in enumerate(expert_reasoning):
            results[p.name] = i < 1

        score = compute_weighted_score(results)
        # reasoning: (1*5 + 2*0 + 3*1) / (1*5 + 2*5 + 3*3) = 8/24
        expected = 8 / 24
        assert abs(score["category_scores"]["reasoning"] - expected) < 1e-9

    def test_all_categories_present_when_all_problems_provided(self):
        """When all problems are provided, all categories appear in category_scores."""
        from mtb.quality_benchmarks import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )

        results = {}
        for p in (
            EVAL_PROBLEMS
            + HARD_EVAL_PROBLEMS
            + EXPERT_EVAL_PROBLEMS
            + TOOL_CALLING_PROBLEMS
        ):
            results[p.name] = True

        score = compute_weighted_score(results)
        categories = set(score["category_scores"].keys())
        expected_cats = {
            "coding",
            "reasoning",
            "instruction_following",
            "math",
            "writing",
            "tool_calling",
        }
        assert categories == expected_cats

    def test_per_category_only_provided_categories(self):
        """category_scores should only include categories that have results."""
        from mtb.quality_benchmarks import EVAL_PROBLEMS

        # Only provide easy problems (coding, reasoning, instruction_following)
        results = {}
        for p in EVAL_PROBLEMS:
            results[p.name] = True

        score = compute_weighted_score(results)
        # Should have coding, reasoning, instruction_following — but not math, writing, tool_calling
        cats = set(score["category_scores"].keys())
        assert "coding" in cats
        assert "reasoning" in cats
        assert "instruction_following" in cats
        # These should NOT be present since no problems in those categories were provided
        assert "math" not in cats
        assert "writing" not in cats
        assert "tool_calling" not in cats


# ---------------------------------------------------------------------------
# VAL-SCORING-007: Raw pass rate alongside weighted score
# ---------------------------------------------------------------------------
class TestRawPassRate:
    def test_both_fields_present(self):
        """Result must have both weighted_score and raw_pass_rate as distinct keys."""
        from mtb.quality_benchmarks import EVAL_PROBLEMS, HARD_EVAL_PROBLEMS

        results = {}
        # Easy: all pass, Hard: all fail → different weighted vs raw
        for p in EVAL_PROBLEMS:
            results[p.name] = True
        for p in HARD_EVAL_PROBLEMS:
            results[p.name] = False

        score = compute_weighted_score(results)
        assert "weighted_score" in score
        assert "raw_pass_rate" in score

    def test_values_differ_when_tier_rates_unequal(self):
        """weighted_score and raw_pass_rate should differ for asymmetric results."""
        from mtb.quality_benchmarks import EVAL_PROBLEMS, HARD_EVAL_PROBLEMS

        results = {}
        # Easy: all 15 pass, Hard: all 10 fail
        for p in EVAL_PROBLEMS:
            results[p.name] = True
        for p in HARD_EVAL_PROBLEMS:
            results[p.name] = False

        score = compute_weighted_score(results)
        # raw_pass_rate = 15/25 = 0.6
        # weighted = (1*15 + 2*0) / (1*15 + 2*10) = 15/35 ≈ 0.4286
        assert abs(score["raw_pass_rate"] - 15 / 25) < 1e-9
        assert abs(score["weighted_score"] - 15 / 35) < 1e-9
        assert score["weighted_score"] != score["raw_pass_rate"]


# ---------------------------------------------------------------------------
# VAL-SCORING-009: Problem-to-tier mapping covers all problems
# ---------------------------------------------------------------------------
class TestProblemToTierMapping:
    def test_all_problems_mappable(self):
        """Every problem in all lists must be mappable to its tier with no KeyError."""
        from mtb.quality_benchmarks import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )
        from mtb.quality_benchmarks.scoring import _build_problem_tier_map

        tier_map = _build_problem_tier_map()
        all_problems = (
            EVAL_PROBLEMS
            + HARD_EVAL_PROBLEMS
            + EXPERT_EVAL_PROBLEMS
            + TOOL_CALLING_PROBLEMS
        )

        for p in all_problems:
            assert p.name in tier_map, f"Problem '{p.name}' not mapped to any tier"

        assert len(tier_map) == len(all_problems)

    def test_tier_assignments_correct(self):
        """Verify tier assignments: easy→easy, hard→hard, expert→expert, tool→tool_calling."""
        from mtb.quality_benchmarks import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )
        from mtb.quality_benchmarks.scoring import _build_problem_tier_map

        tier_map = _build_problem_tier_map()

        for p in EVAL_PROBLEMS:
            assert tier_map[p.name] == "easy"
        for p in HARD_EVAL_PROBLEMS:
            assert tier_map[p.name] == "hard"
        for p in EXPERT_EVAL_PROBLEMS:
            assert tier_map[p.name] == "expert"
        for p in TOOL_CALLING_PROBLEMS:
            assert tier_map[p.name] == "tool_calling"


# ---------------------------------------------------------------------------
# VAL-SCORING-010: Weighted score is deterministic
# ---------------------------------------------------------------------------
class TestDeterminism:
    def test_deterministic_output(self):
        """Running compute_weighted_score twice with identical inputs → identical output."""
        from mtb.quality_benchmarks import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )

        results = {}
        for i, p in enumerate(EVAL_PROBLEMS):
            results[p.name] = i % 2 == 0
        for i, p in enumerate(HARD_EVAL_PROBLEMS):
            results[p.name] = i % 3 == 0
        for i, p in enumerate(EXPERT_EVAL_PROBLEMS):
            results[p.name] = i < 5
        for i, p in enumerate(TOOL_CALLING_PROBLEMS):
            results[p.name] = i < 30

        score1 = compute_weighted_score(results)
        score2 = compute_weighted_score(results)

        assert score1["weighted_score"] == score2["weighted_score"]
        assert score1["raw_pass_rate"] == score2["raw_pass_rate"]
        assert score1["category_scores"] == score2["category_scores"]


# ---------------------------------------------------------------------------
# VAL-MODULE-004: scoring.py importable
# ---------------------------------------------------------------------------
class TestModuleImportable:
    def test_scoring_module_importable(self):
        """scoring.py must be importable from mtb.quality_benchmarks.scoring."""
        from mtb.quality_benchmarks.scoring import (
            TIER_WEIGHTS,
            compute_weighted_score,
        )

        assert callable(compute_weighted_score)
        assert isinstance(TIER_WEIGHTS, dict)

    def test_scoring_importable_from_wildcard(self):
        """from mtb.quality_benchmarks.scoring import * should work."""
        # Just verify the module loads without error
        import importlib

        mod = importlib.import_module("mtb.quality_benchmarks.scoring")
        assert hasattr(mod, "compute_weighted_score")
        assert hasattr(mod, "TIER_WEIGHTS")


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_unknown_problem_names_ignored(self):
        """Results with problem names not in any tier should be skipped gracefully."""
        results = {
            "nonexistent_problem_xyz": True,
            "another_unknown": False,
        }
        score = compute_weighted_score(results)
        # No known problems → 0.0
        assert score["weighted_score"] == 0.0
        assert score["raw_pass_rate"] == 0.0

    def test_mixed_known_and_unknown_problems(self):
        """Known problems scored; unknown problems silently ignored."""
        from mtb.quality_benchmarks import EVAL_PROBLEMS

        results = {}
        for p in EVAL_PROBLEMS:
            results[p.name] = True
        results["nonexistent_problem_xyz"] = True

        score = compute_weighted_score(results)
        # Should score only the known easy problems (all pass)
        assert score["weighted_score"] == 1.0
        assert score["raw_pass_rate"] == 1.0

    def test_result_type_is_dict(self):
        """compute_weighted_score returns a dict."""
        from mtb.quality_benchmarks import EVAL_PROBLEMS

        results = {EVAL_PROBLEMS[0].name: True}
        score = compute_weighted_score(results)
        assert isinstance(score, dict)

    def test_score_values_are_floats(self):
        """weighted_score and raw_pass_rate must be floats."""
        from mtb.quality_benchmarks import EVAL_PROBLEMS

        results = {EVAL_PROBLEMS[0].name: True}
        score = compute_weighted_score(results)
        assert isinstance(score["weighted_score"], float)
        assert isinstance(score["raw_pass_rate"], float)

    def test_scores_bounded_0_to_1(self):
        """Scores should always be in [0.0, 1.0]."""
        from mtb.quality_benchmarks import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )

        # Test with various mixtures
        for pass_count in [0, 5, 15, 40]:
            results = {}
            all_probs = (
                EVAL_PROBLEMS
                + HARD_EVAL_PROBLEMS
                + EXPERT_EVAL_PROBLEMS
                + TOOL_CALLING_PROBLEMS
            )
            for i, p in enumerate(all_probs):
                results[p.name] = i < pass_count

            score = compute_weighted_score(results)
            assert 0.0 <= score["weighted_score"] <= 1.0
            assert 0.0 <= score["raw_pass_rate"] <= 1.0
