"""Weighted scoring system for quality benchmarks.

Computes weighted scores based on problem difficulty tiers:
  - Easy (EVAL_PROBLEMS): weight 1x
  - Hard (HARD_EVAL_PROBLEMS): weight 2x
  - Expert (EXPERT_EVAL_PROBLEMS): weight 3x
  - Tool Calling (TOOL_CALLING_PROBLEMS): weight 3x

Formula: weighted_score = sum(weight * passed_count) / sum(weight * total_count)

The scoring function takes pass/fail results per problem and returns both
a weighted score and a raw (unweighted) pass rate.

Also provides the pluggable ProblemSource interface and StaticProblemSource.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

from mtb.quality_benchmarks.eval_problem import EvalProblem


# ---------------------------------------------------------------------------
# Pluggable problem source interface
# ---------------------------------------------------------------------------


class ProblemSource(ABC):
    """Abstract base class for pluggable problem sources.

    Provides a unified interface for retrieving evaluation problems.
    Built-in implementations include ``StaticProblemSource`` which wraps
    the existing hardcoded problem lists. Future implementations may
    include dynamic sources such as LiveCodeBench for fresh, uncontaminated
    problems fetched from an online repository.

    Subclass this to create custom problem sources — for example, pulling
    problems from a database, an API, or generating them procedurally.
    """

    @abstractmethod
    def get_problems(self, difficulty: str = "all") -> List[EvalProblem]:
        """Return a list of evaluation problems.

        Args:
            difficulty: Filter by difficulty tier. One of ``"easy"``,
                ``"hard"``, ``"expert"``, ``"tool_calling"``, or ``"all"``
                (default). ``"all"`` returns every problem across all tiers.

        Returns:
            List of EvalProblem instances matching the requested difficulty.
        """
        ...


class StaticProblemSource(ProblemSource):
    """Problem source that wraps the existing static problem lists.

    Returns problems from EVAL_PROBLEMS, HARD_EVAL_PROBLEMS,
    EXPERT_EVAL_PROBLEMS, and TOOL_CALLING_PROBLEMS.
    """

    def get_problems(self, difficulty: str = "all") -> List[EvalProblem]:
        """Return static problems filtered by difficulty tier.

        Args:
            difficulty: One of ``"easy"``, ``"hard"``, ``"expert"``,
                ``"tool_calling"``, or ``"all"`` (default).

        Returns:
            List of EvalProblem instances for the requested tier.

        Raises:
            ValueError: If difficulty is not a recognized value.
        """
        # Import here to avoid circular imports
        from mtb.quality_benchmarks import (
            EVAL_PROBLEMS,
            EXPERT_EVAL_PROBLEMS,
            HARD_EVAL_PROBLEMS,
            TOOL_CALLING_PROBLEMS,
        )

        if difficulty == "easy":
            return list(EVAL_PROBLEMS)
        elif difficulty == "hard":
            return list(HARD_EVAL_PROBLEMS)
        elif difficulty == "expert":
            return list(EXPERT_EVAL_PROBLEMS)
        elif difficulty == "tool_calling":
            return list(TOOL_CALLING_PROBLEMS)
        elif difficulty == "all":
            return list(
                EVAL_PROBLEMS
                + HARD_EVAL_PROBLEMS
                + EXPERT_EVAL_PROBLEMS
                + TOOL_CALLING_PROBLEMS
            )
        else:
            raise ValueError(
                f"Unknown difficulty '{difficulty}', must be 'easy', 'hard', "
                f"'expert', 'tool_calling', or 'all'"
            )


# ---------------------------------------------------------------------------
# Weight constants (importable)
# ---------------------------------------------------------------------------
TIER_WEIGHTS: Dict[str, int] = {
    "easy": 1,
    "hard": 2,
    "expert": 3,
    "tool_calling": 3,
}


def _build_problem_tier_map() -> Dict[str, str]:
    """Build a mapping from problem name to its difficulty tier.

    Dynamically reads the four problem lists to ensure all problems are covered.
    Returns a dict mapping problem_name -> tier_name.
    """
    # Import here to avoid circular imports at module level
    from mtb.quality_benchmarks import (
        EVAL_PROBLEMS,
        EXPERT_EVAL_PROBLEMS,
        HARD_EVAL_PROBLEMS,
        TOOL_CALLING_PROBLEMS,
    )

    tier_map: Dict[str, str] = {}

    for problem in EVAL_PROBLEMS:
        tier_map[problem.name] = "easy"

    for problem in HARD_EVAL_PROBLEMS:
        tier_map[problem.name] = "hard"

    for problem in EXPERT_EVAL_PROBLEMS:
        tier_map[problem.name] = "expert"

    for problem in TOOL_CALLING_PROBLEMS:
        tier_map[problem.name] = "tool_calling"

    return tier_map


def compute_weighted_score(results: Dict[str, bool]) -> Dict:
    """Compute weighted and raw scores from pass/fail results per problem.

    Args:
        results: Dict mapping problem_name (str) -> passed (bool).
                 Problem names not found in any tier are silently ignored.

    Returns:
        Dict with keys:
          - "weighted_score": float in [0.0, 1.0]
              sum(weight * passed_count_per_tier) / sum(weight * total_count_per_tier)
          - "raw_pass_rate": float in [0.0, 1.0]
              total_passed / total_problems
          - "category_scores": Dict[str, float]
              Per-category weighted scores. Each category's score is computed by
              weighting that category's problems by their tier weight.
              Only categories with at least one result are included.
    """
    tier_map = _build_problem_tier_map()

    # Classify results by tier — only include problems we can map
    tier_passed: Dict[str, int] = {}
    tier_total: Dict[str, int] = {}

    # Also track per-category, per-tier stats for category_scores
    # key: (category, tier) -> (passed, total)
    cat_tier_passed: Dict[tuple, int] = {}
    cat_tier_total: Dict[tuple, int] = {}

    # We need the category for each problem. Build name -> category map.
    from mtb.quality_benchmarks import (
        EVAL_PROBLEMS,
        EXPERT_EVAL_PROBLEMS,
        HARD_EVAL_PROBLEMS,
        TOOL_CALLING_PROBLEMS,
    )

    name_to_category: Dict[str, str] = {}
    for p in (
        EVAL_PROBLEMS
        + HARD_EVAL_PROBLEMS
        + EXPERT_EVAL_PROBLEMS
        + TOOL_CALLING_PROBLEMS
    ):
        name_to_category[p.name] = p.category

    for problem_name, passed in results.items():
        tier = tier_map.get(problem_name)
        if tier is None:
            # Unknown problem — skip
            continue

        category = name_to_category[problem_name]

        # Update tier totals
        tier_total[tier] = tier_total.get(tier, 0) + 1
        tier_passed[tier] = tier_passed.get(tier, 0) + (1 if passed else 0)

        # Update category-tier totals
        key = (category, tier)
        cat_tier_total[key] = cat_tier_total.get(key, 0) + 1
        cat_tier_passed[key] = cat_tier_passed.get(key, 0) + (1 if passed else 0)

    # Compute overall weighted score
    weighted_numerator = 0.0
    weighted_denominator = 0.0
    total_passed = 0
    total_problems = 0

    for tier, total in tier_total.items():
        weight = TIER_WEIGHTS[tier]
        passed_count = tier_passed.get(tier, 0)
        weighted_numerator += weight * passed_count
        weighted_denominator += weight * total
        total_passed += passed_count
        total_problems += total

    if weighted_denominator == 0:
        weighted_score = 0.0
    else:
        weighted_score = weighted_numerator / weighted_denominator

    if total_problems == 0:
        raw_pass_rate = 0.0
    else:
        raw_pass_rate = total_passed / total_problems

    # Compute per-category weighted scores
    # Group by category, then weight each tier's contribution
    categories: Dict[str, bool] = {}
    for cat, _tier in cat_tier_total:
        categories[cat] = True

    category_scores: Dict[str, float] = {}
    for cat in categories:
        cat_w_num = 0.0
        cat_w_den = 0.0
        for tier in TIER_WEIGHTS:
            key = (cat, tier)
            if key in cat_tier_total:
                weight = TIER_WEIGHTS[tier]
                cat_w_num += weight * cat_tier_passed.get(key, 0)
                cat_w_den += weight * cat_tier_total[key]
        if cat_w_den > 0:
            category_scores[cat] = cat_w_num / cat_w_den

    return {
        "weighted_score": float(weighted_score),
        "raw_pass_rate": float(raw_pass_rate),
        "category_scores": category_scores,
    }
