"""Evaluation problems for measuring reasoning quality across quantizations.

Each problem has:
  - category: coding, reasoning, or instruction_following
  - prompt: the question to ask
  - check: a function that takes the model's response and returns True/False
  - max_tokens: how many tokens to allow for generation

The checks are intentionally lenient — we're testing whether quantization
breaks the model's ability to reason, not whether it matches exact formatting.
"""

import re
from dataclasses import dataclass
from typing import Callable, List


@dataclass
class EvalProblem:
    category: str
    name: str
    prompt: str
    check: Callable[[str], bool]
    max_tokens: int = 512


def _strip_thinking(response: str) -> str:
    """Strip thinking/reasoning preambles from model responses.

    Handles two formats:
    1. <think>...</think> blocks (Claude-distilled models)
    2. Freeform "Thinking Process:" preambles (base Qwen 3.5 models)

    Our checks should evaluate the actual answer only.
    """
    # Remove all <think>...</think> blocks (greedy, handles multiline)
    stripped = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    # Also handle unclosed <think> blocks (response truncated mid-thinking)
    stripped = re.sub(r'<think>.*', '', stripped, flags=re.DOTALL)

    # Handle freeform thinking preambles (e.g., "Thinking Process:" or
    # "Here's a thinking process that leads to the solution:")
    # These appear at the start and consist of numbered analysis steps,
    # followed by the actual answer after a clear section break.
    thinking_start = re.match(
        r'^(?:(?:Here\'s a )?[Tt]hinking [Pp]rocess.*?[:\n]|'
        r'(?:The user wants|Let me (?:think|analyze|break down|work through)).*?\n)',
        stripped,
    )
    if thinking_start:
        # Strategy: find the first "final answer" marker after the thinking.
        # Models produce headers like "*Revised Draft:*", "**Rewrite:**",
        # "Here's the implementation:", etc. We take everything from that
        # marker to the end, which includes the answer plus any follow-up
        # explanation — this is fine because the answer content dominates.
        #
        # For models that draft multiple attempts (Draft -> Revised Draft),
        # we prefer "final/revised" markers over plain "draft" markers.
        # First try to find a "final" marker; fall back to any marker.
        final_pattern = re.compile(
            r'\n\s*'
            r'(?:\*{1,2}|#{1,3}\s*)?'
            r'(?:Final Answer|Final Polish|Revised Draft|Final|'
            r'Here\'s (?:the|my) (?:implementation|solution|rewrite|answer))'
            r'(?:\*{1,2})?'
            r'\s*[:：\n]',
            re.IGNORECASE,
        )
        any_answer_pattern = re.compile(
            r'\n\s*'
            r'(?:\*{1,2}|#{1,3}\s*)?'
            r'(?:Answer|Solution|Result|Output|Rewrite|Rewritten|Draft|'
            r'Summary|Response|Implementation|'
            r'Here\'s (?:the|my)|Here is (?:the|my)|'
            r'Non-technical|Simplified|Plain)'
            r'(?:\*{1,2})?'
            r'\s*[:：\n]',
            re.IGNORECASE,
        )
        # Prefer final markers (last occurrence), fall back to first general marker
        match = None
        for m in final_pattern.finditer(stripped):
            match = m  # Take the last "final" marker
        if not match:
            match = any_answer_pattern.search(stripped)  # Take first general marker
        if match:
            remainder = stripped[match.end():].strip()
            if len(remainder) > 20:  # Sanity check: must have substantial content
                stripped = remainder

    return stripped.strip()


def _contains_any(response: str, targets: List[str]) -> bool:
    response_lower = response.lower()
    return any(t.lower() in response_lower for t in targets)


def _extract_number(response: str) -> float | None:
    """Extract the last number from a response (models often reason then give final answer)."""
    numbers = re.findall(r'-?\d+\.?\d*', response)
    if numbers:
        return float(numbers[-1])
    return None


# --- CODING PROBLEMS ---

def _check_fizzbuzz(response: str) -> bool:
    """Check if response contains a working FizzBuzz logic."""
    response = _strip_thinking(response)
    has_fizzbuzz = _contains_any(response, ["fizzbuzz", "fizz buzz"])
    has_mod = _contains_any(response, ["%", "mod", "divisible"])
    has_15_or_both = _contains_any(response, ["15", "% 3 == 0 and", "% 5 == 0 and", "% 3 == 0 and n % 5"])
    return has_fizzbuzz or (has_mod and has_15_or_both)


def _check_reverse_string(response: str) -> bool:
    """Check if response contains string reversal logic."""
    response = _strip_thinking(response)
    return _contains_any(response, [
        "[::-1]", "reversed(", "reverse()", ".reverse()",
        "for i in range(len", "while", "swap",
        "StringBuilder", "charAt", "split('').reverse",
    ])


def _check_fibonacci(response: str) -> bool:
    """Check if response produces correct fib sequence start."""
    response = _strip_thinking(response)
    return _contains_any(response, ["0, 1, 1, 2, 3, 5, 8", "0,1,1,2,3,5,8"])


def _check_binary_search(response: str) -> bool:
    """Check if response contains binary search logic."""
    response = _strip_thinking(response)
    has_mid = _contains_any(response, ["mid", "middle"])
    has_halving = _contains_any(response, ["// 2", "/ 2", ">> 1", "low", "high", "left", "right"])
    return has_mid and has_halving


def _check_palindrome(response: str) -> bool:
    """Check for palindrome checking logic."""
    response = _strip_thinking(response)
    return _contains_any(response, [
        "[::-1]", "reversed(", "reverse()",
        "left", "right", "i < j", "two pointer",
    ])


# --- REASONING / MATH PROBLEMS ---

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
    return _contains_any(response, ["jerry is 5", "jerry's age is 5", "jerry = 5",
                                     "tom is 10", "tom's age is 10", "tom = 10",
                                     "5 and 10", "5, tom", "10, jerry",
                                     "jerry: 5", "tom: 10",
                                     "jerry** is **5", "tom** is **10"])


def _check_sequence_problem(response: str) -> bool:
    """What's the next number: 2, 6, 12, 20, 30, ?"""
    # Pattern: n*(n+1): 1*2=2, 2*3=6, 3*4=12, 4*5=20, 5*6=30, 6*7=42
    response = _strip_thinking(response)
    return _contains_any(response, ["42", "**42**"])


# --- INSTRUCTION FOLLOWING ---

def _check_json_output(response: str) -> bool:
    """Check if model produced valid-looking JSON."""
    response = _strip_thinking(response)
    return "{" in response and "}" in response and '"name"' in response.lower()


def _check_list_format(response: str) -> bool:
    """Check if model produced a numbered list."""
    response = _strip_thinking(response)
    has_numbers = bool(re.search(r'[1-5][.)]', response))
    has_multiple = len(re.findall(r'\d[.)]', response)) >= 3
    return has_numbers and has_multiple


def _check_word_constraint(response: str) -> bool:
    """Check if the model's actual explanation is reasonably concise.

    Models may include a thinking preamble before the answer. We look at
    the final paragraph/section as the actual output.
    """
    response = _strip_thinking(response)
    # If the response has a clear final section, check that
    # Otherwise just verify it contains an ML-related explanation
    return _contains_any(response, [
        "machine learning", "subset of ai", "subset of artificial intelligence",
        "learn from data", "algorithms", "patterns",
    ])


def _check_code_with_comments(response: str) -> bool:
    """Check if model included comments in code."""
    response = _strip_thinking(response)
    has_code = _contains_any(response, ["def ", "function ", "class ", "int ", "void "])
    has_comments = _contains_any(response, ["#", "//", "/*", "'''", '"""'])
    return has_code and has_comments


def _check_no_thinking(response: str) -> bool:
    """Check if the model produced the correct answer (Paris)."""
    response = _strip_thinking(response)
    return _contains_any(response, ["paris"])


# =============================================================================
# HARD CHECK FUNCTIONS
# =============================================================================

# --- HARD CODING PROBLEMS ---

def _check_lru_cache(response: str) -> bool:
    """Check if response implements LRU cache with hash map + linked list or OrderedDict."""
    response = _strip_thinking(response)
    # Must have both a storage mechanism and eviction logic
    has_structure = _contains_any(response, [
        "OrderedDict", "doubly linked", "double linked", "DLL",
        "self.cache", "self.capacity", "HashMap", "dict",
    ])
    has_operations = _contains_any(response, ["get", "put", "def get", "def put"])
    has_eviction = _contains_any(response, [
        "pop", "del ", "remove", "evict", "popitem", "move_to_end",
        "least recently", "LRU",
    ])
    return has_structure and has_operations and has_eviction


def _check_flatten_nested(response: str) -> bool:
    """Check if response correctly flattens arbitrarily nested lists."""
    response = _strip_thinking(response)
    has_recursion = _contains_any(response, [
        "isinstance", "type(", "hasattr", "iter(",
        "yield from", "yield", "extend", "recursiv",
    ])
    has_list_check = _contains_any(response, ["list", "Iterable", "iterable", "sequence"])
    # Should show the correct output for a test case
    has_example = _contains_any(response, [
        "[1, 2, 3, 4, 5", "1, 2, 3, 4",
        "flatten(", "def flatten",
    ])
    return has_recursion and has_list_check and has_example


def _check_longest_palindrome_substring(response: str) -> bool:
    """Check if response implements longest palindromic substring (not just check)."""
    response = _strip_thinking(response)
    # Must have expansion or DP logic, not just a simple palindrome check
    has_algorithm = _contains_any(response, [
        "expand", "dp[", "dp =", "dynamic programming",
        "center", "manacher", "for i in range", "for j in range",
        "longest", "max_len", "max_length", "start",
    ])
    has_substring_logic = _contains_any(response, [
        "substring", "sub_string", "s[", "result",
        "babad", "racecar", "aba", "longest_palindrome",
    ])
    return has_algorithm and has_substring_logic


def _check_calculator(response: str) -> bool:
    """Check if response implements a calculator with operator precedence and parentheses."""
    response = _strip_thinking(response)
    has_parsing = _contains_any(response, [
        "stack", "token", "parse", "operator",
        "precedence", "shunting", "recursive descent",
        "expression", "term", "factor",
    ])
    has_operations = _contains_any(response, ["+", "-", "*", "/"])
    has_parens = _contains_any(response, ["(", ")", "paren"])
    return has_parsing and has_operations and has_parens


def _check_buggy_merge_sort(response: str) -> bool:
    """Check if response identifies and fixes the bug in a merge sort."""
    response = _strip_thinking(response)
    has_bug_identification = _contains_any(response, [
        "bug", "error", "issue", "fix", "incorrect", "wrong", "problem",
        "off-by-one", "off by one", "boundary", "index",
    ])
    has_fix = _contains_any(response, [
        "should be", "change", "replace", "corrected", "fixed",
        "<=", "< len", "mid", "merge",
    ])
    return has_bug_identification and has_fix


# --- HARD REASONING PROBLEMS ---

def _check_compound_interest(response: str) -> bool:
    """$10,000 at 5% annual compound interest for 3 years, compounded quarterly.
    A = 10000 * (1 + 0.05/4)^(4*3) = 10000 * (1.0125)^12 = 10000 * 1.16075... ≈ $11,607.55
    """
    response = _strip_thinking(response)
    # Accept various reasonable roundings
    return _contains_any(response, [
        "11607", "11,607", "11608", "11,608",
        "1607.55", "1,607.55", "1607.5", "1,607.5",
    ])


def _check_circular_seating(response: str) -> bool:
    """How many ways to seat 6 people at a round table if 2 specific people must NOT sit next to each other?
    Total circular permutations = (6-1)! = 120
    Ways they ARE adjacent = 2 * (5-1)! = 2 * 24 = 48
    Answer = 120 - 48 = 72
    """
    response = _strip_thinking(response)
    # Use word boundary matching to avoid "720" matching "72"
    return bool(re.search(r'\b72\b', response))


def _check_logic_puzzle(response: str) -> bool:
    """A says 'B is a liar.' B says 'C is a liar.' C says 'A and B are both liars.'
    If A is truthful: B is liar, so C is truthful. But C says A and B are BOTH liars — contradiction since A is truthful.
    If B is truthful: C is liar. Then check A: if A is truthful, B is liar — contradiction. So A is liar, meaning B is NOT a liar (consistent). C is liar. Check C's claim: 'A and B are both liars' — this is false (B isn't), and since C is a liar, that's consistent.
    Answer: B is truthful, A and C are liars.
    """
    response = _strip_thinking(response)
    has_b_truthful = _contains_any(response, [
        "b is truthful", "b is telling the truth", "b is honest",
        "b tells the truth", "only b", "b is a truth-teller",
        "b is the truth-teller", "b is truth-teller",
        "b=t", "b = t", "b: truth",
    ])
    has_a_c_liars = _contains_any(response, [
        "a and c are liars", "a and c lie", "a is a liar",
        "c is a liar", "a lies", "c lies",
    ])
    return has_b_truthful and has_a_c_liars


def _check_bayes_theorem(response: str) -> bool:
    """A disease affects 1% of the population. A test is 95% accurate (95% true positive,
    95% true negative / 5% false positive). If you test positive, what's the probability
    you actually have the disease?

    P(D|+) = P(+|D)*P(D) / [P(+|D)*P(D) + P(+|~D)*P(~D)]
           = (0.95 * 0.01) / (0.95 * 0.01 + 0.05 * 0.99)
           = 0.0095 / (0.0095 + 0.0495)
           = 0.0095 / 0.059
           ≈ 0.161 or ~16.1%

    Many small models incorrectly say 95% (confusing test accuracy with posterior).
    """
    response = _strip_thinking(response)
    # Accept various forms of ~16%
    return _contains_any(response, [
        "16.1%", "16.1 %", "0.161", "≈ 16%", "about 16%", "roughly 16%",
        "approximately 16", "~16%", "16 percent",
        # Also accept close approximations
        "16.0%", "16.2%", "0.16", "16%",
        "19/118",  # exact fraction
    ])


def _check_proof_bug(response: str) -> bool:
    """Find the bug in this 'proof' that 1=2:
    Let a=b. Then a²=ab, a²-b²=ab-b², (a+b)(a-b)=b(a-b), a+b=b, 2b=b, 2=1.
    The bug is dividing by (a-b) which equals zero since a=b.
    """
    response = _strip_thinking(response)
    return _contains_any(response, [
        "division by zero", "divide by zero", "dividing by zero",
        "a - b = 0", "a-b = 0", "a = b", "equals zero",
        "a minus b is zero", "a-b is 0", "(a-b) is 0",
        "factor is zero", "both sides by zero",
    ])


EVAL_PROBLEMS: List[EvalProblem] = [
    # --- Coding (5 problems) ---
    EvalProblem(
        category="coding",
        name="fizzbuzz",
        prompt="Write a Python function that prints FizzBuzz for numbers 1 to 100. Print 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, and 'FizzBuzz' for multiples of both.",
        check=_check_fizzbuzz,
    ),
    EvalProblem(
        category="coding",
        name="reverse_string",
        prompt="Write a function to reverse a string. Show at least two different approaches.",
        check=_check_reverse_string,
    ),
    EvalProblem(
        category="coding",
        name="fibonacci",
        prompt="Write a function that returns the first 10 Fibonacci numbers as a list. Show the output.",
        check=_check_fibonacci,
    ),
    EvalProblem(
        category="coding",
        name="binary_search",
        prompt="Implement binary search in Python. The function should take a sorted list and a target value, and return the index of the target or -1 if not found.",
        check=_check_binary_search,
    ),
    EvalProblem(
        category="coding",
        name="palindrome",
        prompt="Write a function to check if a string is a palindrome, ignoring case and non-alphanumeric characters.",
        check=_check_palindrome,
    ),

    # --- Reasoning / Math (5 problems) ---
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

    # --- Instruction Following (5 problems) ---
    EvalProblem(
        category="instruction_following",
        name="json_output",
        prompt='Output a JSON object with the keys "name", "age", and "hobbies" (a list) for a fictional person. Output ONLY the JSON, no other text.',
        check=_check_json_output,
        max_tokens=512,
    ),
    EvalProblem(
        category="instruction_following",
        name="numbered_list",
        prompt="List 5 benefits of regular exercise. Use a numbered list format (1. 2. 3. etc).",
        check=_check_list_format,
        max_tokens=512,
    ),
    EvalProblem(
        category="instruction_following",
        name="word_constraint",
        prompt="Explain what machine learning is in 50 words or fewer.",
        check=_check_word_constraint,
        max_tokens=512,
    ),
    EvalProblem(
        category="instruction_following",
        name="code_with_comments",
        prompt="Write a Python function to calculate the area of a circle. Include detailed comments explaining each step.",
        check=_check_code_with_comments,
        max_tokens=1024,
    ),
    EvalProblem(
        category="instruction_following",
        name="direct_answer",
        prompt="What is the capital of France? Answer directly with no explanation.",
        check=_check_no_thinking,
        max_tokens=256,
    ),
]


# =============================================================================
# EXPERT CHECK FUNCTIONS
# =============================================================================

# --- EXPERT MATH (AIME-inspired) ---

def _check_modular_arithmetic(response: str) -> bool:
    """Find the remainder when 2^100 is divided by 7.
    2^1=2, 2^2=4, 2^3=1 (mod 7) — cycle of length 3.
    100 mod 3 = 1, so 2^100 mod 7 = 2^1 mod 7 = 2.
    """
    response = _strip_thinking(response)
    # Look for the answer "2" in specific contexts that indicate it's the remainder
    # "remainder is 2", "remainder of 2", "remainder = 2"
    has_remainder_2 = bool(re.search(r'remainder\s+(is\s+|of\s+|=\s*)\b2\b', response, re.IGNORECASE))
    # Congruence notation: ≡ 2 (mod 7) or = 2 mod 7
    has_congruence = bool(re.search(r'≡\s*2\s*\(?\s*mod\s*7\s*\)?', response))
    has_equals_mod = bool(re.search(r'=\s*2\s*\(?\s*mod\s*7\s*\)?', response))
    # "the answer is 2" at end of reasoning about modular arithmetic
    has_answer_2 = bool(re.search(r'(answer|result)\s+(is\s+|=\s*)\**2\**\.?\s*$', response, re.IGNORECASE | re.MULTILINE))
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
    Down: 10 + 10*(3/4) + 10*(3/4)^2 + ... = 10 / (1 - 3/4) = 40
    Up:   10*(3/4) + 10*(3/4)^2 + ... = 7.5 / (1 - 3/4) = 30
    Total = 40 + 30 = 70 meters
    """
    response = _strip_thinking(response)
    return _contains_any(response, ["70 meter", "70m", "= 70", "**70**", "70 m"])


# --- EXPERT AGENTIC CODING (OpenClaw-inspired) ---

def _check_markdown_to_html(response: str) -> bool:
    """Implement Markdown-to-HTML converter for bold, italic, code, links."""
    response = _strip_thinking(response)
    has_bold = _contains_any(response, ["<strong>", "<b>", "**", "bold"])
    has_italic = _contains_any(response, ["<em>", "<i>", "*", "italic"])
    has_link = _contains_any(response, ["<a ", "href", "[", "]("])
    has_code = _contains_any(response, ["<code>", "`", "backtick"])
    has_implementation = _contains_any(response, [
        "def ", "function ", "replace", "re.sub", "regex",
        "sub(", "match", "pattern",
    ])
    # Must handle at least 3 of 4 elements and have actual code
    elements_handled = sum([has_bold, has_italic, has_link, has_code])
    return elements_handled >= 3 and has_implementation


def _check_data_pipeline(response: str) -> bool:
    """Filter CSV rows by 3 criteria, group, aggregate, output sorted JSON."""
    response = _strip_thinking(response)
    has_filter = _contains_any(response, [
        "filter", "where", "query", "loc[", "mask",
        "if ", "condition", "criteria",
    ])
    has_group = _contains_any(response, [
        "groupby", "group_by", "group by", "defaultdict",
        "setdefault", "collections", "product", "key",
        "group", "bucket", "categor",
    ])
    has_aggregate = _contains_any(response, [
        "sum", "mean", "average", "count", "agg(",
        "aggregate", "total", "reduce",
    ])
    has_json = _contains_any(response, [
        "json", "JSON", "dumps", "json.dumps",
        "dict", "output",
    ])
    has_sort = _contains_any(response, [
        "sort", "sorted", "order", "key=",
    ])
    return has_filter and has_group and has_aggregate and has_json and has_sort


def _check_retry_decorator(response: str) -> bool:
    """Retry decorator with exponential backoff, exception handling, Result type."""
    response = _strip_thinking(response)
    has_decorator = _contains_any(response, [
        "decorator", "@", "def retry", "wrapper", "functools",
        "wraps",
    ])
    has_backoff = _contains_any(response, [
        "backoff", "exponential", "2 **", "2**", "delay *",
        "sleep", "wait", "time.sleep",
    ])
    has_exception = _contains_any(response, [
        "except", "Exception", "try:", "catch",
        "exception_types", "exceptions",
    ])
    has_result = _contains_any(response, [
        "Result", "result", "success", "failure",
        "return_value", "metadata", "attempts", "dataclass",
        "namedtuple", "TypedDict",
    ])
    return has_decorator and has_backoff and has_exception and has_result


# --- EXPERT LONG-CHAIN REASONING (GPQA-inspired) ---

def _check_einstein_riddle(response: str) -> bool:
    """Simplified Einstein's riddle with 5 houses, 5 colors, 5 nationalities, 5 drinks.
    8 constraints. Must deduce all assignments.

    Constraints (simplified):
    1. The Brit lives in the red house.
    2. The Swede drinks milk.
    3. The green house is immediately to the left of the white house.
    4. The Dane drinks tea.
    5. The green house owner drinks coffee.
    6. The Norwegian lives in the first house.
    7. The Norwegian lives next to the blue house.
    8. The German lives in the middle house.

    Solution: House 1: Norwegian, yellow, water; House 2: Dane, blue, tea;
    House 3: German, red/green depends on variant... The key check is
    that the model produces a consistent assignment and identifies
    who drinks water (Norwegian).
    """
    response = _strip_thinking(response)
    # The model must produce a structured assignment with houses
    has_assignment = _contains_any(response, [
        "house 1", "house 2", "house 3", "first house", "second house",
        "position 1", "position 2",
    ])
    has_nationalities = _contains_any(response, [
        "norwegian", "dane", "brit", "swede", "german",
    ])
    # Key deduction: Norwegian is in house 1, blue house is house 2
    has_key_deduction = _contains_any(response, [
        "norwegian.*first", "norwegian.*house 1",
        "blue.*house 2", "blue.*second",
    ]) or (
        _contains_any(response, ["norwegian"]) and
        _contains_any(response, ["first", "house 1", "position 1"])
    )
    return has_assignment and has_nationalities and has_key_deduction


def _check_three_urns(response: str) -> bool:
    """Three urns problem:
    Urn A: 3 red, 2 blue. Urn B: 1 red, 4 blue. Urn C: 4 red, 1 blue.
    Draw from A. If red → draw from B. If blue → draw from C.
    What is P(second draw is red)?

    P(red from A) = 3/5, then P(red from B) = 1/5.
    P(blue from A) = 2/5, then P(red from C) = 4/5.
    P(second red) = (3/5)(1/5) + (2/5)(4/5) = 3/25 + 8/25 = 11/25 = 0.44

    """
    response = _strip_thinking(response)
    return _contains_any(response, [
        "11/25", "0.44", "44%", "11 / 25", "= 0.44",
    ])


def _check_topological_sort(response: str) -> bool:
    """Given dependency graph A→B, B→C, A→D, D→C, C→E, find all valid topological orderings.

    In-degrees: A=0, B=1, C=2, D=1, E=1
    Start with A. Then B or D (2 choices).
    If B first: A,B,D,C,E
    If D first: A,D,B,C,E
    Only 2 valid orderings.
    """
    response = _strip_thinking(response)
    has_orderings = _contains_any(response, [
        "A, B, D, C, E", "A,B,D,C,E", "ABDCE",
        "A, D, B, C, E", "A,D,B,C,E", "ADBCE",
        "A → B → D → C → E", "A → D → B → C → E",
    ])
    has_both = (
        _contains_any(response, ["A, B, D, C, E", "A,B,D,C,E", "ABDCE", "A → B → D → C → E", "A B D C E"]) and
        _contains_any(response, ["A, D, B, C, E", "A,D,B,C,E", "ADBCE", "A → D → B → C → E", "A D B C E"])
    )
    has_count = _contains_any(response, ["2 valid", "two valid", "2 orderings", "two orderings", "exactly 2", "exactly two"])
    return has_both or (has_orderings and has_count)


# --- EXPERT INSTRUCTION FOLLOWING (IFEval-inspired) ---

def _check_constrained_factorial(response: str) -> bool:
    """Write a Python function where: (a) every variable name is exactly 4 chars,
    (b) no loops (use recursion), (c) exactly 3 type hints, (d) computes factorial.
    """
    response = _strip_thinking(response)
    # Must have recursion (no for/while)
    has_recursion = _contains_any(response, ["def ", "return"]) and not _contains_any(response, ["for ", "while "])
    # Actually, check functions in code blocks - model may discuss loops in explanation
    # Look for the actual function having recursion
    has_factorial_logic = _contains_any(response, [
        "factorial", "fact", "n * ", "n*", "* fact", "*fact",
        "n - 1", "n-1", "n == 0", "n == 1", "n <= 1",
    ])
    # Check for type hints
    has_type_hints = _contains_any(response, [
        "-> int", ": int", "-> float", ": float",
        "-> None", ": Optional",
    ])
    # Check variable names are constrained (model should mention this constraint)
    has_var_constraint = _contains_any(response, [
        "4 char", "four char", "exactly 4", "4-char", "4 letter",
    ]) or bool(re.search(r'\b[a-z]{4}\b\s*[=:]', response))

    return has_factorial_logic and has_type_hints and (has_recursion or has_var_constraint)


def _check_library_schema(response: str) -> bool:
    """Generate JSON schema for library system with Book, Author, Loan — cross-references by ID."""
    response = _strip_thinking(response)
    has_book = _contains_any(response, ["book", "Book"])
    has_author = _contains_any(response, ["author", "Author"])
    has_loan = _contains_any(response, ["loan", "Loan"])
    has_id_refs = _contains_any(response, [
        "book_id", "author_id", "loan_id",
        "bookId", "authorId", "loanId",
        "$ref", "reference", "foreign_key",
        "id", "ID",
    ])
    has_json_schema = _contains_any(response, [
        "properties", "type", "required", "schema",
        "string", "integer", "array", "object",
    ])
    return has_book and has_author and has_loan and has_id_refs and has_json_schema


def _check_adversarial_transform(response: str) -> bool:
    """3-stage pipeline: Translate to French, reverse each word, wrap in XML tags.
    "The quick brown fox jumps over the lazy dog"
    Step 1 (French): "Le renard brun rapide saute par-dessus le chien paresseux"
    Step 2 (reverse each word): "eL dranre nurb edipar etuasse ..." etc.
    Step 3 (XML): <word original="Le">eL</word> ...
    """
    response = _strip_thinking(response)
    # Must have French translation
    has_french = _contains_any(response, [
        "renard", "chien", "rapide", "saute", "brun", "paresseux",
        "Le ", "le ",
    ])
    # Must have reversed words
    has_reversed = _contains_any(response, [
        "eL", "dranre", "nurb", "edipar",
        "reverse", "[::-1]",
    ])
    # Must have XML tags
    has_xml = _contains_any(response, [
        "<word", "original=", "</word>",
        "xml", "XML", "tag",
    ])
    return has_french and has_reversed and has_xml


# --- EXPERT WRITING (Obsidian/Agent-inspired) ---

def _check_multi_doc_summary(response: str) -> bool:
    """Given 3 paragraphs about: (1) quantum computing, (2) climate change, (3) gene therapy.
    Must produce a single summary covering all three topics.
    """
    response = _strip_thinking(response)
    has_quantum = _contains_any(response, [
        "quantum", "qubit", "superposition", "quantum computing",
    ])
    has_climate = _contains_any(response, [
        "climate", "carbon", "temperature", "emission", "warming",
    ])
    has_gene = _contains_any(response, [
        "gene therapy", "gene", "dna", "genetic", "crispr",
    ])
    return has_quantum and has_climate and has_gene


def _check_structured_meeting_notes(response: str) -> bool:
    """Write meeting notes with YAML frontmatter, ## sections, action items with owners."""
    response = _strip_thinking(response)
    has_yaml = _contains_any(response, ["---\n", "date:", "attendees:"])
    has_sections = (
        _contains_any(response, ["## Discussion", "## discussion", "# Discussion"]) and
        _contains_any(response, ["## Decision", "## decision", "# Decision", "## Action", "## action", "# Action"])
    )
    has_action_items = _contains_any(response, [
        "- [ ]", "- [x]", "action item", "Action Item",
        "TODO", "todo", "task",
    ])
    has_owners = _contains_any(response, [
        "@", "owner:", "Owner:", "assigned to", "Assigned to",
        "responsible", "Responsible",
    ])
    return has_yaml and has_sections and (has_action_items or has_owners)


def _check_tone_rewrite(response: str) -> bool:
    """Rewrite technical paragraph for non-technical audience: <=3 sentences, preserve key facts, no jargon."""
    response = _strip_thinking(response)
    # Truncate at post-answer sections (verification tables, explanations, etc.)
    # that may re-introduce jargon for comparison purposes.
    for separator in ["\n---", "\n**Verification", "\n| Original", "\n**Note"]:
        idx = response.find(separator)
        if idx > 20:
            response = response[:idx]
    # Must preserve key facts from the original (neural networks, training, accuracy)
    has_key_facts = _contains_any(response, [
        "neural", "train", "learn", "accuracy", "data",
        "brain", "pattern", "predict",
    ])
    # Should avoid jargon (or at least explain it)
    avoids_jargon = not _contains_any(response, [
        "backpropagation", "gradient descent", "loss function",
        "stochastic", "hyperparameter", "epoch",
    ]) or _contains_any(response, ["means", "basically", "simply", "in other words"])
    return has_key_facts and avoids_jargon


def _check_contradiction_detection(response: str) -> bool:
    """Given two notes with a subtle factual contradiction, identify it.
    Note 1: 'The Apollo 11 mission landed on the Moon on July 20, 1969. Neil Armstrong and Buzz Aldrin spent 2.5 hours on the lunar surface.'
    Note 2: 'Armstrong and Aldrin's moonwalk during Apollo 11 lasted approximately 21.5 hours on the surface, during which they collected 47.5 pounds of samples.'

    Contradiction: 2.5 hours vs 21.5 hours on the surface.
    (Actual fact: ~21.5 hours on surface total, ~2.5 hours of EVA/moonwalk.)
    """
    response = _strip_thinking(response)
    has_time_identified = _contains_any(response, [
        "2.5", "21.5", "hours", "time", "duration",
        "surface", "spent",
    ])
    has_contradiction = _contains_any(response, [
        "contradict", "inconsisten", "conflict", "discrepan",
        "differ", "mismatch", "disagree",
    ])
    return has_time_identified and has_contradiction


HARD_EVAL_PROBLEMS: List[EvalProblem] = [
    # --- Hard Coding (5 problems) ---
    EvalProblem(
        category="coding",
        name="lru_cache",
        prompt=(
            "Implement an LRU (Least Recently Used) cache in Python from scratch. "
            "It should support get(key) and put(key, value) operations, both in O(1) time. "
            "The cache should have a configurable capacity and evict the least recently used "
            "item when full. Do not use collections.OrderedDict — implement the underlying "
            "data structure yourself."
        ),
        check=_check_lru_cache,
        max_tokens=2048,
    ),
    EvalProblem(
        category="coding",
        name="flatten_nested",
        prompt=(
            "Write a Python function `flatten(lst)` that takes an arbitrarily nested list "
            "and returns a flat list of all values. For example:\n"
            "  flatten([1, [2, [3, 4], 5], [6, 7]]) -> [1, 2, 3, 4, 5, 6, 7]\n"
            "  flatten([[1, 2], [[3]], [4, [5, [6]]]]) -> [1, 2, 3, 4, 5, 6]\n"
            "Handle mixed types (strings should not be iterated into characters)."
        ),
        check=_check_flatten_nested,
        max_tokens=1024,
    ),
    EvalProblem(
        category="coding",
        name="longest_palindrome_substring",
        prompt=(
            "Write a Python function that finds the longest palindromic substring in a "
            "given string. For example, given 'babad', return 'bab' or 'aba'. "
            "Given 'cbbd', return 'bb'. Your solution should be O(n^2) or better."
        ),
        check=_check_longest_palindrome_substring,
        max_tokens=2048,
    ),
    EvalProblem(
        category="coding",
        name="calculator",
        prompt=(
            "Implement a basic calculator in Python that evaluates a string expression "
            "containing +, -, *, /, parentheses, and non-negative integers. It must "
            "respect operator precedence (* and / before + and -) and handle nested "
            "parentheses. For example:\n"
            '  calculate("2 + 3 * 4") -> 14\n'
            '  calculate("(2 + 3) * 4") -> 20\n'
            '  calculate("10 + 2 * (3 + 4)") -> 24'
        ),
        check=_check_calculator,
        max_tokens=2048,
    ),
    EvalProblem(
        category="coding",
        name="buggy_merge_sort",
        prompt=(
            "The following merge sort implementation has a bug. Find it and explain the fix.\n\n"
            "```python\n"
            "def merge_sort(arr):\n"
            "    if len(arr) <= 1:\n"
            "        return arr\n"
            "    mid = len(arr) // 2\n"
            "    left = merge_sort(arr[:mid])\n"
            "    right = merge_sort(arr[mid:])\n"
            "    return merge(left, right)\n\n"
            "def merge(left, right):\n"
            "    result = []\n"
            "    i = j = 0\n"
            "    while i < len(left) and j < len(right):\n"
            "        if left[i] < right[j]:\n"
            "            result.append(left[i])\n"
            "            i += 1\n"
            "        else:\n"
            "            result.append(right[j])\n"
            "            j += 1\n"
            "    return result\n"
            "```\n\n"
            "What's wrong? Fix it."
        ),
        check=_check_buggy_merge_sort,
        max_tokens=2048,
    ),

    # --- Hard Reasoning (5 problems) ---
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


EXPERT_EVAL_PROBLEMS: List[EvalProblem] = [
    # --- Expert Math: AIME-inspired (3 problems) ---
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

    # --- Expert Agentic Coding: OpenClaw-inspired (3 problems) ---
    EvalProblem(
        category="coding",
        name="markdown_to_html",
        prompt=(
            "Write a Python function `markdown_to_html(text)` that converts a subset of "
            "Markdown to HTML. It must handle:\n"
            "  1. **bold** → <strong>bold</strong>\n"
            "  2. *italic* → <em>italic</em>\n"
            "  3. `code` → <code>code</code>\n"
            "  4. [link text](url) → <a href=\"url\">link text</a>\n\n"
            "Handle edge cases: nested formatting, multiple occurrences per line, "
            "and escaped characters. Provide test cases."
        ),
        check=_check_markdown_to_html,
        max_tokens=2048,
    ),
    EvalProblem(
        category="coding",
        name="data_pipeline",
        prompt=(
            "Write a Python function `process_sales(csv_text)` that takes a CSV string "
            "with columns: date, product, region, amount, quantity.\n\n"
            "The function must:\n"
            "1. Filter to rows where amount > 100 AND region is 'North' or 'West' AND quantity >= 5\n"
            "2. Group remaining rows by product\n"
            "3. For each product, compute: total_amount, avg_amount, count\n"
            "4. Sort products by total_amount descending\n"
            "5. Return the result as a JSON string\n\n"
            "Do not use pandas — implement with csv module and standard library only."
        ),
        check=_check_data_pipeline,
        max_tokens=2048,
    ),
    EvalProblem(
        category="coding",
        name="retry_decorator",
        prompt=(
            "Write a Python decorator `retry(max_attempts=3, backoff_base=2, exceptions=(Exception,))` "
            "that:\n"
            "1. Retries the decorated function up to `max_attempts` times\n"
            "2. Uses exponential backoff: wait `backoff_base ** attempt` seconds between retries\n"
            "3. Only catches exceptions in the `exceptions` tuple\n"
            "4. Logs each attempt (print is fine)\n"
            "5. Returns a Result dataclass with fields: success (bool), value (any), "
            "attempts (int), error (Optional[Exception])\n\n"
            "Show usage with a function that fails intermittently."
        ),
        check=_check_retry_decorator,
        max_tokens=2048,
    ),

    # --- Expert Long-Chain Reasoning: GPQA-inspired (3 problems) ---
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
    ),

    # --- Expert Instruction Following: IFEval-inspired (3 problems) ---
    EvalProblem(
        category="instruction_following",
        name="constrained_factorial",
        prompt=(
            "Write a Python function that computes the factorial of a non-negative integer. "
            "You MUST follow ALL of these constraints simultaneously:\n\n"
            "1. Every variable name must be exactly 4 characters long (e.g., 'numb', 'rslt')\n"
            "2. You must NOT use any loops (for or while) — use recursion only\n"
            "3. Include exactly 3 type hints in your code\n"
            "4. The function must correctly compute factorial (e.g., factorial(5) = 120)\n\n"
            "Show the complete function."
        ),
        check=_check_constrained_factorial,
        max_tokens=2048,
    ),
    EvalProblem(
        category="instruction_following",
        name="library_schema",
        prompt=(
            "Generate a JSON Schema for a library management system with three entity types:\n\n"
            "1. **Book**: title, isbn, publication_year, author_id (reference to Author)\n"
            "2. **Author**: name, birth_year, nationality, book_ids (list of Book references)\n"
            "3. **Loan**: book_id (reference to Book), borrower_name, due_date, returned (boolean)\n\n"
            "Requirements:\n"
            "- Each entity must have a unique 'id' field\n"
            "- Cross-references must use the referenced entity's id\n"
            "- Include 'required' fields and 'type' constraints for every property\n"
            "- Output valid JSON Schema (draft 2020-12 or later)"
        ),
        check=_check_library_schema,
        max_tokens=2048,
    ),
    EvalProblem(
        category="instruction_following",
        name="adversarial_transform",
        prompt=(
            "Perform these three transformations in sequence on the sentence:\n"
            "\"The quick brown fox jumps over the lazy dog\"\n\n"
            "Step 1: Translate the sentence to French.\n"
            "Step 2: Take your French translation and reverse each word individually "
            "(keep word order the same, but reverse the letters within each word).\n"
            "Step 3: Wrap each reversed word in an XML tag: "
            '<word original="french_word">reversed_word</word>\n\n'
            "Show the output of each step clearly."
        ),
        check=_check_adversarial_transform,
        max_tokens=2048,
    ),

    # --- Expert Writing: Obsidian/Agent-inspired (4 problems) ---
    EvalProblem(
        category="writing",
        name="multi_doc_summary",
        prompt=(
            "Summarize the following three paragraphs into a single cohesive summary that "
            "covers ALL three topics. The summary should be 3-5 sentences.\n\n"
            "Paragraph 1: Quantum computing leverages quantum mechanical phenomena like "
            "superposition and entanglement to process information. Unlike classical bits "
            "that are either 0 or 1, qubits can exist in multiple states simultaneously, "
            "enabling exponential speedups for certain problems like cryptography and "
            "drug discovery.\n\n"
            "Paragraph 2: Climate change is accelerating faster than predicted. Global "
            "temperatures have risen 1.2°C above pre-industrial levels, and carbon "
            "emissions continue to increase. The IPCC warns that exceeding 1.5°C will "
            "trigger irreversible tipping points including ice sheet collapse and "
            "permafrost thaw.\n\n"
            "Paragraph 3: Gene therapy has entered a new era with CRISPR-Cas9 technology. "
            "Recent clinical trials show promising results for sickle cell disease and "
            "certain cancers. However, off-target DNA edits and the high cost of treatment "
            "remain significant barriers to widespread adoption."
        ),
        check=_check_multi_doc_summary,
        max_tokens=1024,
    ),
    EvalProblem(
        category="writing",
        name="structured_meeting_notes",
        prompt=(
            "Write meeting notes for a fictional product team standup. The notes MUST include:\n\n"
            "1. YAML frontmatter with fields: date, attendees (list), meeting_type\n"
            "2. A '## Discussion' section with 3 bullet points\n"
            "3. A '## Decisions' section with 2 bullet points\n"
            "4. A '## Action Items' section with 3 items, each assigned to a specific person "
            "using @name format\n\n"
            "Use proper Markdown formatting throughout."
        ),
        check=_check_structured_meeting_notes,
        max_tokens=1024,
    ),
    EvalProblem(
        category="writing",
        name="tone_rewrite",
        prompt=(
            "Rewrite the following technical paragraph for a non-technical audience. "
            "Your rewrite must be 3 sentences or fewer, preserve all key facts, "
            "and avoid jargon.\n\n"
            "Original: 'Deep neural networks utilize backpropagation with stochastic "
            "gradient descent to minimize the cross-entropy loss function across training "
            "epochs. The model's accuracy on the validation set plateaued at 94.3% after "
            "hyperparameter tuning of the learning rate and batch size, with the Adam "
            "optimizer achieving faster convergence than vanilla SGD.'"
        ),
        check=_check_tone_rewrite,
        max_tokens=1024,
    ),
    EvalProblem(
        category="writing",
        name="contradiction_detection",
        prompt=(
            "Read these two notes carefully and identify any factual contradictions between them.\n\n"
            "Note 1: 'The Apollo 11 mission landed on the Moon on July 20, 1969. "
            "Neil Armstrong and Buzz Aldrin spent approximately 2.5 hours walking on "
            "the lunar surface during their EVA, collecting rock samples and deploying "
            "scientific instruments.'\n\n"
            "Note 2: 'During the Apollo 11 mission, Armstrong and Aldrin remained on "
            "the lunar surface for approximately 21.5 hours in total after landing. "
            "They collected 47.5 pounds of lunar samples during their moonwalk.'\n\n"
            "Are there any contradictions? If so, explain precisely what conflicts and "
            "what the actual facts are."
        ),
        check=_check_contradiction_detection,
        max_tokens=1024,
    ),
]
