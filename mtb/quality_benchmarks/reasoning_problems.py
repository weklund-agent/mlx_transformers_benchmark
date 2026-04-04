"""Reasoning evaluation problems across all difficulty tiers.

Contains check functions and EvalProblem instances for reasoning problems:
- Easy (5): train_problem, coin_probability, workers_problem, age_problem, sequence_pattern
- Hard (5): compound_interest, circular_seating, logic_puzzle, bayes_theorem, proof_bug
- Expert (3): einstein_riddle, three_urns, topological_sort
"""

import re
from typing import List

from mtb.quality_benchmarks.utils import _contains_any, _strip_thinking
from mtb.quality_benchmarks.eval_problem import EvalProblem


# =============================================================================
# EASY REASONING CHECK FUNCTIONS
# =============================================================================


def _check_train_problem(response: str) -> bool:
    """Two trains 200km apart, 60km/h and 40km/h toward each other. When do they meet?"""
    # Answer: 200 / (60+40) = 2 hours
    response = _strip_thinking(response)
    return _contains_any(response, ["2 hour", "2.0 hour", "two hour", "= 2", "**2**"])


def _check_coin_problem(response: str) -> bool:
    """You flip a fair coin 3 times. P(at least 2 heads)?"""
    # Answer: 4/8 = 0.5 or 50%
    response = _strip_thinking(response)
    return _contains_any(response, ["0.5", "50%", "1/2", "4/8", "50 percent"])


def _check_workers_problem(response: str) -> bool:
    """If 5 workers build a wall in 10 days, how long for 10 workers?"""
    # Answer: 5 days
    response = _strip_thinking(response)
    return _contains_any(response, ["5 day", "five day", "= 5", "**5**"])


def _check_age_problem(response: str) -> bool:
    """Tom is twice as old as Jerry. In 5 years, Tom will be 1.5x Jerry's age. Ages now?"""
    # T=2J. T+5=1.5(J+5). 2J+5=1.5J+7.5. 0.5J=2.5. J=5, T=10.
    response = _strip_thinking(response)
    return _contains_any(
        response,
        [
            "jerry is 5",
            "jerry's age is 5",
            "jerry = 5",
            "tom is 10",
            "tom's age is 10",
            "tom = 10",
            "5 and 10",
            "5, tom",
            "10, jerry",
            "jerry: 5",
            "tom: 10",
            "jerry** is **5",
            "tom** is **10",
        ],
    )


def _check_sequence_problem(response: str) -> bool:
    """What's the next number: 2, 6, 12, 20, 30, ?"""
    # Pattern: n*(n+1): 1*2=2, 2*3=6, 3*4=12, 4*5=20, 5*6=30, 6*7=42
    response = _strip_thinking(response)
    return _contains_any(response, ["42", "**42**"])


# =============================================================================
# HARD REASONING CHECK FUNCTIONS
# =============================================================================


def _check_compound_interest(response: str) -> bool:
    """$10,000 at 5% annual compound interest for 3 years, compounded quarterly.
    A = 10000 * (1 + 0.05/4)^(4*3) = 10000 * (1.0125)^12 = 10000 * 1.16075... ≈ $11,607.55
    """
    response = _strip_thinking(response)
    return _contains_any(
        response,
        [
            "11607",
            "11,607",
            "11608",
            "11,608",
            "1607.55",
            "1,607.55",
            "1607.5",
            "1,607.5",
        ],
    )


def _check_circular_seating(response: str) -> bool:
    """How many ways to seat 6 people at a round table if 2 specific people must NOT sit next to each other?
    Total circular permutations = (6-1)! = 120
    Ways they ARE adjacent = 2 * (5-1)! = 2 * 24 = 48
    Answer = 120 - 48 = 72
    """
    response = _strip_thinking(response)
    return bool(re.search(r"\b72\b", response))


def _check_logic_puzzle(response: str) -> bool:
    """A says 'B is a liar.' B says 'C is a liar.' C says 'A and B are both liars.'
    Answer: B is truthful, A and C are liars.
    """
    response = _strip_thinking(response)
    has_b_truthful = _contains_any(
        response,
        [
            "b is truthful",
            "b is telling the truth",
            "b is honest",
            "b tells the truth",
            "only b",
            "b is a truth-teller",
            "b is the truth-teller",
            "b is truth-teller",
            "b=t",
            "b = t",
            "b: truth",
        ],
    )
    has_a_c_liars = _contains_any(
        response,
        [
            "a and c are liars",
            "a and c lie",
            "a is a liar",
            "c is a liar",
            "a lies",
            "c lies",
        ],
    )
    return has_b_truthful and has_a_c_liars


def _check_bayes_theorem(response: str) -> bool:
    """A disease affects 1% of the population. Test is 95% accurate.
    P(D|+) ≈ 0.161 or ~16.1%
    """
    response = _strip_thinking(response)
    return _contains_any(
        response,
        [
            "16.1%",
            "16.1 %",
            "0.161",
            "≈ 16%",
            "about 16%",
            "roughly 16%",
            "approximately 16",
            "~16%",
            "16 percent",
            "16.0%",
            "16.2%",
            "0.16",
            "16%",
            "19/118",
        ],
    )


def _check_proof_bug(response: str) -> bool:
    """Find the bug in this 'proof' that 1=2.
    The bug is dividing by (a-b) which equals zero since a=b.
    """
    response = _strip_thinking(response)
    return _contains_any(
        response,
        [
            "division by zero",
            "divide by zero",
            "dividing by zero",
            "a - b = 0",
            "a-b = 0",
            "a = b",
            "equals zero",
            "a minus b is zero",
            "a-b is 0",
            "(a-b) is 0",
            "factor is zero",
            "both sides by zero",
        ],
    )


# =============================================================================
# EXPERT REASONING CHECK FUNCTIONS
# =============================================================================


def _check_einstein_riddle(response: str) -> bool:
    """Simplified Einstein's riddle with 5 houses."""
    response = _strip_thinking(response)
    has_assignment = _contains_any(
        response,
        [
            "house 1",
            "house 2",
            "house 3",
            "first house",
            "second house",
            "position 1",
            "position 2",
        ],
    )
    has_nationalities = _contains_any(
        response,
        [
            "norwegian",
            "dane",
            "brit",
            "swede",
            "german",
        ],
    )
    has_key_deduction = _contains_any(
        response,
        [
            "norwegian.*first",
            "norwegian.*house 1",
            "blue.*house 2",
            "blue.*second",
        ],
    ) or (
        _contains_any(response, ["norwegian"])
        and _contains_any(response, ["first", "house 1", "position 1"])
    )
    return has_assignment and has_nationalities and has_key_deduction


def _check_three_urns(response: str) -> bool:
    """Three urns problem. P(second red) = 11/25 = 0.44"""
    response = _strip_thinking(response)
    return _contains_any(
        response,
        [
            "11/25",
            "0.44",
            "44%",
            "11 / 25",
            "= 0.44",
        ],
    )


def _check_topological_sort(response: str) -> bool:
    """Given dependency graph, find all valid topological orderings.
    Only 2 valid orderings: A,B,D,C,E and A,D,B,C,E.
    """
    response = _strip_thinking(response)
    has_orderings = _contains_any(
        response,
        [
            "A, B, D, C, E",
            "A,B,D,C,E",
            "ABDCE",
            "A, D, B, C, E",
            "A,D,B,C,E",
            "ADBCE",
            "A → B → D → C → E",
            "A → D → B → C → E",
        ],
    )
    has_both = _contains_any(
        response,
        ["A, B, D, C, E", "A,B,D,C,E", "ABDCE", "A → B → D → C → E", "A B D C E"],
    ) and _contains_any(
        response,
        ["A, D, B, C, E", "A,D,B,C,E", "ADBCE", "A → D → B → C → E", "A D B C E"],
    )
    has_count = _contains_any(
        response,
        [
            "2 valid",
            "two valid",
            "2 orderings",
            "two orderings",
            "exactly 2",
            "exactly two",
        ],
    )
    return has_both or (has_orderings and has_count)


# =============================================================================
# PROBLEM LISTS BY TIER
# =============================================================================

REASONING_EASY_PROBLEMS: List[EvalProblem] = [
    EvalProblem(
        category="reasoning",
        name="train_problem",
        prompt="Two trains are 200 km apart and moving toward each other. Train A moves at 60 km/h and Train B at 40 km/h. How long until they meet?",
        check=_check_train_problem,
        max_tokens=1024,
    ),
    EvalProblem(
        category="reasoning",
        name="coin_probability",
        prompt="You flip a fair coin 3 times. What is the probability of getting at least 2 heads?",
        check=_check_coin_problem,
        max_tokens=1024,
    ),
    EvalProblem(
        category="reasoning",
        name="workers_problem",
        prompt="If 5 workers can build a wall in 10 days, how long would it take 10 workers to build the same wall?",
        check=_check_workers_problem,
        max_tokens=1024,
    ),
    EvalProblem(
        category="reasoning",
        name="age_problem",
        prompt="Tom is twice as old as Jerry. In 5 years, Tom will be 1.5 times as old as Jerry. How old are Tom and Jerry now?",
        check=_check_age_problem,
        max_tokens=1024,
    ),
    EvalProblem(
        category="reasoning",
        name="sequence_pattern",
        prompt="What is the next number in this sequence: 2, 6, 12, 20, 30, ?",
        check=_check_sequence_problem,
        max_tokens=1024,
    ),
]

REASONING_HARD_PROBLEMS: List[EvalProblem] = [
    EvalProblem(
        category="reasoning",
        name="compound_interest",
        prompt=(
            "You invest $10,000 at 5% annual interest rate, compounded quarterly, "
            "for 3 years. What is the final amount? Show your work step by step."
        ),
        check=_check_compound_interest,
        max_tokens=2048,
    ),
    EvalProblem(
        category="reasoning",
        name="circular_seating",
        prompt=(
            "In how many ways can 6 people be seated around a circular table if "
            "2 specific people (Alice and Bob) must NOT sit next to each other? "
            "Show your reasoning."
        ),
        check=_check_circular_seating,
        max_tokens=2048,
    ),
    EvalProblem(
        category="reasoning",
        name="logic_puzzle",
        prompt=(
            "Three people — A, B, and C — are either truth-tellers (always tell the truth) "
            "or liars (always lie).\n"
            "  A says: 'B is a liar.'\n"
            "  B says: 'C is a liar.'\n"
            "  C says: 'A and B are both liars.'\n"
            "Who is telling the truth and who is lying? Prove your answer by checking "
            "all cases."
        ),
        check=_check_logic_puzzle,
        max_tokens=2048,
    ),
    EvalProblem(
        category="reasoning",
        name="bayes_theorem",
        prompt=(
            "A rare disease affects 1% of the population. A diagnostic test has a "
            "95% true positive rate (sensitivity) and a 95% true negative rate "
            "(specificity), meaning it has a 5% false positive rate. "
            "If a randomly selected person tests positive, what is the probability "
            "they actually have the disease? Show your work using Bayes' theorem."
        ),
        check=_check_bayes_theorem,
        max_tokens=2048,
    ),
    EvalProblem(
        category="reasoning",
        name="proof_bug",
        prompt=(
            "Find the error in this 'proof' that 1 = 2:\n\n"
            "Let a = b.\n"
            "Then a² = ab\n"
            "a² - b² = ab - b²\n"
            "(a + b)(a - b) = b(a - b)\n"
            "a + b = b\n"
            "Since a = b, we get b + b = b, so 2b = b, therefore 2 = 1.\n\n"
            "Where exactly is the error and why is it invalid?"
        ),
        check=_check_proof_bug,
        max_tokens=1024,
    ),
]

REASONING_EXPERT_PROBLEMS: List[EvalProblem] = [
    EvalProblem(
        category="reasoning",
        name="einstein_riddle",
        prompt=(
            "Solve this logic puzzle. There are 5 houses in a row, numbered 1-5 from left to right.\n\n"
            "Constraints:\n"
            "1. The Brit lives in the red house.\n"
            "2. The Swede drinks milk.\n"
            "3. The green house is immediately to the left of the white house.\n"
            "4. The Dane drinks tea.\n"
            "5. The green house owner drinks coffee.\n"
            "6. The Norwegian lives in house 1.\n"
            "7. The Norwegian lives next to the blue house.\n"
            "8. The German lives in house 3.\n\n"
            "Assign each person to a house and determine their house color and drink. "
            "Show your reasoning step by step."
        ),
        check=_check_einstein_riddle,
        max_tokens=4096,
    ),
    EvalProblem(
        category="reasoning",
        name="three_urns",
        prompt=(
            "There are three urns:\n"
            "  Urn A contains 3 red balls and 2 blue balls.\n"
            "  Urn B contains 1 red ball and 4 blue balls.\n"
            "  Urn C contains 4 red balls and 1 blue ball.\n\n"
            "You draw one ball from Urn A. If it is red, you then draw from Urn B. "
            "If it is blue, you then draw from Urn C.\n\n"
            "What is the probability that the second ball drawn is red? "
            "Show your work using the law of total probability."
        ),
        check=_check_three_urns,
        max_tokens=2048,
    ),
    EvalProblem(
        category="reasoning",
        name="topological_sort",
        prompt=(
            "Given this dependency graph:\n"
            "  A → B (A must come before B)\n"
            "  B → C\n"
            "  A → D\n"
            "  D → C\n"
            "  C → E\n\n"
            "List ALL valid topological orderings of these 5 nodes. "
            "Show your reasoning by tracking in-degrees at each step."
        ),
        check=_check_topological_sort,
        max_tokens=2048,
        function_signature="def topological_sort(graph: dict) -> list:",
        test_cases=[
            {
                "input": "{'A': ['B', 'C'], 'B': ['D'], 'C': ['D'], 'D': []}",
                "expected_output": ["A", "B", "C", "D"],
            },
            {
                "input": "{'A': [], 'B': [], 'C': []}",
                "expected_output": ["A", "B", "C"],
            },
            {
                "input": "{'A': ['B'], 'B': ['C'], 'C': []}",
                "expected_output": ["A", "B", "C"],
            },
            {
                "input": "{'X': ['Y'], 'Y': ['Z'], 'Z': [], 'W': ['X']}",
                "expected_output": ["W", "X", "Y", "Z"],
            },
            {
                "input": "{'A': []}",
                "expected_output": ["A"],
            },
            {
                "input": "{'C': ['A'], 'B': ['A'], 'A': []}",
                "expected_output": ["B", "C", "A"],
            },
        ],
        _correct_impl=(
            "def topological_sort(graph: dict) -> list:\n"
            "    in_degree = {node: 0 for node in graph}\n"
            "    for node in graph:\n"
            "        for neighbor in graph[node]:\n"
            "            if neighbor not in in_degree:\n"
            "                in_degree[neighbor] = 0\n"
            "            in_degree[neighbor] += 1\n"
            "    queue = sorted([n for n in in_degree if in_degree[n] == 0])\n"
            "    result = []\n"
            "    while queue:\n"
            "        node = queue.pop(0)\n"
            "        result.append(node)\n"
            "        for neighbor in sorted(graph.get(node, [])):\n"
            "            in_degree[neighbor] -= 1\n"
            "            if in_degree[neighbor] == 0:\n"
            "                queue.append(neighbor)\n"
            "                queue.sort()\n"
            "    return result\n"
        ),
        _incorrect_impl=(
            "def topological_sort(graph: dict) -> list:\n"
            "    return sorted(graph.keys())\n"
        ),
    ),
]
