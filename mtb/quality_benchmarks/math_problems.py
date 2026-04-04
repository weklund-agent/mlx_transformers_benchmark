"""Math evaluation problems (expert tier).

Contains check functions and EvalProblem instances for math problems:
- Expert (3): modular_arithmetic, inclusion_exclusion, bouncing_ball

Each problem also defines a generate_variant() callable that produces
a new EvalProblem with different numerical parameters but the same
mathematical structure, for contamination resistance.
"""

import random
import re
from typing import List

from mtb.quality_benchmarks.eval_problem import EvalProblem
from mtb.quality_benchmarks.utils import _contains_any, _strip_thinking


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
# VARIANT GENERATORS
# =============================================================================


def _generate_modular_arithmetic_variant() -> EvalProblem:
    """Generate a modular arithmetic variant with different base, exponent, and modulus."""
    base = random.choice([2, 3, 5, 7, 11])
    exponent = random.randint(50, 200)
    modulus = random.choice([7, 11, 13, 17, 19])
    # Ensure we don't repeat the original (base=2, exp=100, mod=7)
    if base == 2 and exponent == 100 and modulus == 7:
        exponent = random.randint(101, 200)

    expected_answer = pow(base, exponent, modulus)

    def _check(response: str) -> bool:
        response = _strip_thinking(response)
        answer_str = str(expected_answer)
        has_remainder = bool(
            re.search(
                rf"remainder\s+(is\s+|of\s+|=\s*)\b{answer_str}\b",
                response,
                re.IGNORECASE,
            )
        )
        has_congruence = bool(
            re.search(rf"≡\s*{answer_str}\s*\(?\s*mod\s*{modulus}\s*\)?", response)
        )
        has_equals_mod = bool(
            re.search(rf"=\s*{answer_str}\s*\(?\s*mod\s*{modulus}\s*\)?", response)
        )
        has_answer = bool(
            re.search(
                rf"(answer|result)\s+(is\s+|=\s*)\**{answer_str}\**\.?\s*$",
                response,
                re.IGNORECASE | re.MULTILINE,
            )
        )
        return has_remainder or has_congruence or has_equals_mod or has_answer

    return EvalProblem(
        category="math",
        name="modular_arithmetic",
        prompt=(
            f"What is the remainder when {base}^{exponent} is divided by {modulus}? "
            f"Show your work step by step. Hint: look for a pattern in powers of {base} mod {modulus}."
        ),
        check=_check,
        max_tokens=2048,
    )


def _generate_inclusion_exclusion_variant() -> EvalProblem:
    """Generate an inclusion-exclusion variant with different range and divisors."""
    n = random.choice([500, 750, 1200, 1500, 2000])
    # Choose 3 distinct small primes
    divisors = sorted(random.sample([2, 3, 5, 7, 11], 3))
    a, b, c = divisors

    # Compute using inclusion-exclusion
    count_a = n // a
    count_b = n // b
    count_c = n // c
    count_ab = n // (a * b)
    count_ac = n // (a * c)
    count_bc = n // (b * c)
    count_abc = n // (a * b * c)
    expected = count_a + count_b + count_c - count_ab - count_ac - count_bc + count_abc

    def _check(response: str) -> bool:
        response = _strip_thinking(response)
        return _contains_any(response, [str(expected)])

    return EvalProblem(
        category="math",
        name="inclusion_exclusion",
        prompt=(
            f"How many integers from 1 to {n} (inclusive) are divisible by {a}, {b}, or {c}? "
            f"Use the inclusion-exclusion principle and show each term."
        ),
        check=_check,
        max_tokens=2048,
    )


def _generate_bouncing_ball_variant() -> EvalProblem:
    """Generate a bouncing ball variant with different height and rebound ratio."""
    heights = [8, 12, 15, 16, 20, 24, 25, 30]
    # Use fractions that produce integer answers
    ratios = [(1, 2), (2, 3), (3, 5), (1, 4), (3, 4), (2, 5)]
    height = random.choice(heights)
    num, den = random.choice(ratios)
    ratio_frac = f"{num}/{den}"

    # Avoid original (height=10, ratio=3/4)
    if height == 10 and num == 3 and den == 4:
        height = random.choice([h for h in heights if h != 10])

    r = num / den
    # Total = height / (1-r) + height*r / (1-r)
    # = height * (1 + r) / (1 - r)
    total_distance = height * (1 + r) / (1 - r)

    # Format: if integer, use int; otherwise use one decimal
    if total_distance == int(total_distance):
        answer_str = str(int(total_distance))
    else:
        answer_str = f"{total_distance:.1f}"

    def _check(response: str) -> bool:
        response = _strip_thinking(response)
        # Check for the numeric answer
        return _contains_any(
            response,
            [
                f"{answer_str} meter",
                f"{answer_str}m",
                f"= {answer_str}",
                f"**{answer_str}**",
                f"{answer_str} m",
            ],
        )

    return EvalProblem(
        category="math",
        name="bouncing_ball",
        prompt=(
            f"A ball is dropped from a height of {height} meters. Each time it bounces, it "
            f"rebounds to exactly {ratio_frac} of the height from which it fell. What is the total "
            f"distance the ball travels before coming to rest (i.e., after infinitely many "
            f"bounces)? Show your work using geometric series."
        ),
        check=_check,
        max_tokens=2048,
    )


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
        generate_variant=_generate_modular_arithmetic_variant,
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
        generate_variant=_generate_inclusion_exclusion_variant,
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
        generate_variant=_generate_bouncing_ball_variant,
    ),
]
