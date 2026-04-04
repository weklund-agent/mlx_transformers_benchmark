"""Math evaluation problems (expert tier).

Contains check functions and EvalProblem instances for math problems:
- Expert (3): modular_arithmetic, inclusion_exclusion, bouncing_ball
"""

import re
from typing import List

from mtb.quality_benchmarks.utils import _contains_any, _strip_thinking
from mtb.quality_benchmarks.eval_problem import EvalProblem


# =============================================================================
# EXPERT MATH CHECK FUNCTIONS
# =============================================================================


def _check_modular_arithmetic(response: str) -> bool:
    """Find the remainder when 2^100 is divided by 7.
    2^1=2, 2^2=4, 2^3=1 (mod 7) — cycle of length 3.
    100 mod 3 = 1, so 2^100 mod 7 = 2^1 mod 7 = 2.
    """
    response = _strip_thinking(response)
    has_remainder_2 = bool(
        re.search(r"remainder\s+(is\s+|of\s+|=\s*)\b2\b", response, re.IGNORECASE)
    )
    has_congruence = bool(re.search(r"≡\s*2\s*\(?\s*mod\s*7\s*\)?", response))
    has_equals_mod = bool(re.search(r"=\s*2\s*\(?\s*mod\s*7\s*\)?", response))
    has_answer_2 = bool(
        re.search(
            r"(answer|result)\s+(is\s+|=\s*)\**2\**\.?\s*$",
            response,
            re.IGNORECASE | re.MULTILINE,
        )
    )
    return has_remainder_2 or has_congruence or has_equals_mod or has_answer_2


def _check_inclusion_exclusion(response: str) -> bool:
    """How many integers 1-1000 are divisible by 2, 3, or 5?
    |A∪B∪C| = |A|+|B|+|C| - |A∩B| - |A∩C| - |B∩C| + |A∩B∩C|
    = 500 + 333 + 200 - 166 - 100 - 66 + 33 = 734
    """
    response = _strip_thinking(response)
    return _contains_any(response, ["734"])


def _check_bouncing_ball(response: str) -> bool:
    """Ball drops from 10m, rebounds to 3/4 height. Total distance after infinite bounces.
    Down: 10 / (1 - 3/4) = 40
    Up: 7.5 / (1 - 3/4) = 30
    Total = 40 + 30 = 70 meters
    """
    response = _strip_thinking(response)
    return _contains_any(response, ["70 meter", "70m", "= 70", "**70**", "70 m"])


# =============================================================================
# PROBLEM LISTS BY TIER
# =============================================================================

MATH_EXPERT_PROBLEMS: List[EvalProblem] = [
    EvalProblem(
        category="math",
        name="modular_arithmetic",
        prompt=(
            "What is the remainder when 2^100 is divided by 7? "
            "Show your work step by step. Hint: look for a pattern in powers of 2 mod 7."
        ),
        check=_check_modular_arithmetic,
        max_tokens=2048,
    ),
    EvalProblem(
        category="math",
        name="inclusion_exclusion",
        prompt=(
            "How many integers from 1 to 1000 (inclusive) are divisible by 2, 3, or 5? "
            "Use the inclusion-exclusion principle and show each term."
        ),
        check=_check_inclusion_exclusion,
        max_tokens=2048,
    ),
    EvalProblem(
        category="math",
        name="bouncing_ball",
        prompt=(
            "A ball is dropped from a height of 10 meters. Each time it bounces, it "
            "rebounds to exactly 3/4 of the height from which it fell. What is the total "
            "distance the ball travels before coming to rest (i.e., after infinitely many "
            "bounces)? Show your work using geometric series."
        ),
        check=_check_bouncing_ball,
        max_tokens=2048,
    ),
]
