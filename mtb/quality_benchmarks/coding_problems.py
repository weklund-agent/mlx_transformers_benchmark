"""Coding evaluation problems across all difficulty tiers.

Contains check functions and EvalProblem instances for coding problems:
- Easy (5): fizzbuzz, reverse_string, fibonacci, binary_search, palindrome
- Hard (5): lru_cache, flatten_nested, longest_palindrome_substring, calculator, buggy_merge_sort
- Expert (3): markdown_to_html, data_pipeline, retry_decorator
"""

from typing import List

from mtb.quality_benchmarks.utils import _contains_any, _strip_thinking

# Import EvalProblem from the canonical location to avoid circular imports.
# The eval_problems module will import from here after defining EvalProblem.
from mtb.quality_benchmarks.eval_problem import EvalProblem


# =============================================================================
# EASY CODING CHECK FUNCTIONS
# =============================================================================


def _check_fizzbuzz(response: str) -> bool:
    """Check if response contains a working FizzBuzz logic."""
    response = _strip_thinking(response)
    has_fizzbuzz = _contains_any(response, ["fizzbuzz", "fizz buzz"])
    has_mod = _contains_any(response, ["%", "mod", "divisible"])
    has_15_or_both = _contains_any(
        response, ["15", "% 3 == 0 and", "% 5 == 0 and", "% 3 == 0 and n % 5"]
    )
    return has_fizzbuzz or (has_mod and has_15_or_both)


def _check_reverse_string(response: str) -> bool:
    """Check if response contains string reversal logic."""
    response = _strip_thinking(response)
    return _contains_any(
        response,
        [
            "[::-1]",
            "reversed(",
            "reverse()",
            ".reverse()",
            "for i in range(len",
            "while",
            "swap",
            "StringBuilder",
            "charAt",
            "split('').reverse",
        ],
    )


def _check_fibonacci(response: str) -> bool:
    """Check if response produces correct fib sequence start."""
    response = _strip_thinking(response)
    return _contains_any(response, ["0, 1, 1, 2, 3, 5, 8", "0,1,1,2,3,5,8"])


def _check_binary_search(response: str) -> bool:
    """Check if response contains binary search logic."""
    response = _strip_thinking(response)
    has_mid = _contains_any(response, ["mid", "middle"])
    has_halving = _contains_any(
        response, ["// 2", "/ 2", ">> 1", "low", "high", "left", "right"]
    )
    return has_mid and has_halving


def _check_palindrome(response: str) -> bool:
    """Check for palindrome checking logic."""
    response = _strip_thinking(response)
    return _contains_any(
        response,
        [
            "[::-1]",
            "reversed(",
            "reverse()",
            "left",
            "right",
            "i < j",
            "two pointer",
        ],
    )


# =============================================================================
# HARD CODING CHECK FUNCTIONS
# =============================================================================


def _check_lru_cache(response: str) -> bool:
    """Check if response implements LRU cache with hash map + linked list or OrderedDict."""
    response = _strip_thinking(response)
    has_structure = _contains_any(
        response,
        [
            "OrderedDict",
            "doubly linked",
            "double linked",
            "DLL",
            "self.cache",
            "self.capacity",
            "HashMap",
            "dict",
        ],
    )
    has_operations = _contains_any(response, ["get", "put", "def get", "def put"])
    has_eviction = _contains_any(
        response,
        [
            "pop",
            "del ",
            "remove",
            "evict",
            "popitem",
            "move_to_end",
            "least recently",
            "LRU",
        ],
    )
    return has_structure and has_operations and has_eviction


def _check_flatten_nested(response: str) -> bool:
    """Check if response correctly flattens arbitrarily nested lists."""
    response = _strip_thinking(response)
    has_recursion = _contains_any(
        response,
        [
            "isinstance",
            "type(",
            "hasattr",
            "iter(",
            "yield from",
            "yield",
            "extend",
            "recursiv",
        ],
    )
    has_list_check = _contains_any(
        response, ["list", "Iterable", "iterable", "sequence"]
    )
    has_example = _contains_any(
        response,
        [
            "[1, 2, 3, 4, 5",
            "1, 2, 3, 4",
            "flatten(",
            "def flatten",
        ],
    )
    return has_recursion and has_list_check and has_example


def _check_longest_palindrome_substring(response: str) -> bool:
    """Check if response implements longest palindromic substring (not just check)."""
    response = _strip_thinking(response)
    has_algorithm = _contains_any(
        response,
        [
            "expand",
            "dp[",
            "dp =",
            "dynamic programming",
            "center",
            "manacher",
            "for i in range",
            "for j in range",
            "longest",
            "max_len",
            "max_length",
            "start",
        ],
    )
    has_substring_logic = _contains_any(
        response,
        [
            "substring",
            "sub_string",
            "s[",
            "result",
            "babad",
            "racecar",
            "aba",
            "longest_palindrome",
        ],
    )
    return has_algorithm and has_substring_logic


def _check_calculator(response: str) -> bool:
    """Check if response implements a calculator with operator precedence and parentheses."""
    response = _strip_thinking(response)
    has_parsing = _contains_any(
        response,
        [
            "stack",
            "token",
            "parse",
            "operator",
            "precedence",
            "shunting",
            "recursive descent",
            "expression",
            "term",
            "factor",
        ],
    )
    has_operations = _contains_any(response, ["+", "-", "*", "/"])
    has_parens = _contains_any(response, ["(", ")", "paren"])
    return has_parsing and has_operations and has_parens


def _check_buggy_merge_sort(response: str) -> bool:
    """Check if response identifies and fixes the bug in a merge sort."""
    response = _strip_thinking(response)
    has_bug_identification = _contains_any(
        response,
        [
            "bug",
            "error",
            "issue",
            "fix",
            "incorrect",
            "wrong",
            "problem",
            "off-by-one",
            "off by one",
            "boundary",
            "index",
        ],
    )
    has_fix = _contains_any(
        response,
        [
            "should be",
            "change",
            "replace",
            "corrected",
            "fixed",
            "<=",
            "< len",
            "mid",
            "merge",
        ],
    )
    return has_bug_identification and has_fix


# =============================================================================
# EXPERT CODING CHECK FUNCTIONS
# =============================================================================


def _check_markdown_to_html(response: str) -> bool:
    """Implement Markdown-to-HTML converter for bold, italic, code, links."""
    response = _strip_thinking(response)
    has_bold = _contains_any(response, ["<strong>", "<b>", "**", "bold"])
    has_italic = _contains_any(response, ["<em>", "<i>", "*", "italic"])
    has_link = _contains_any(response, ["<a ", "href", "[", "]("])
    has_code = _contains_any(response, ["<code>", "`", "backtick"])
    has_implementation = _contains_any(
        response,
        [
            "def ",
            "function ",
            "replace",
            "re.sub",
            "regex",
            "sub(",
            "match",
            "pattern",
        ],
    )
    elements_handled = sum([has_bold, has_italic, has_link, has_code])
    return elements_handled >= 3 and has_implementation


def _check_data_pipeline(response: str) -> bool:
    """Filter CSV rows by 3 criteria, group, aggregate, output sorted JSON."""
    response = _strip_thinking(response)
    has_filter = _contains_any(
        response,
        [
            "filter",
            "where",
            "query",
            "loc[",
            "mask",
            "if ",
            "condition",
            "criteria",
        ],
    )
    has_group = _contains_any(
        response,
        [
            "groupby",
            "group_by",
            "group by",
            "defaultdict",
            "setdefault",
            "collections",
            "product",
            "key",
            "group",
            "bucket",
            "categor",
        ],
    )
    has_aggregate = _contains_any(
        response,
        [
            "sum",
            "mean",
            "average",
            "count",
            "agg(",
            "aggregate",
            "total",
            "reduce",
        ],
    )
    has_json = _contains_any(
        response,
        [
            "json",
            "JSON",
            "dumps",
            "json.dumps",
            "dict",
            "output",
        ],
    )
    has_sort = _contains_any(
        response,
        [
            "sort",
            "sorted",
            "order",
            "key=",
        ],
    )
    return has_filter and has_group and has_aggregate and has_json and has_sort


def _check_retry_decorator(response: str) -> bool:
    """Retry decorator with exponential backoff, exception handling, Result type."""
    response = _strip_thinking(response)
    has_decorator = _contains_any(
        response,
        [
            "decorator",
            "@",
            "def retry",
            "wrapper",
            "functools",
            "wraps",
        ],
    )
    has_backoff = _contains_any(
        response,
        [
            "backoff",
            "exponential",
            "2 **",
            "2**",
            "delay *",
            "sleep",
            "wait",
            "time.sleep",
        ],
    )
    has_exception = _contains_any(
        response,
        [
            "except",
            "Exception",
            "try:",
            "catch",
            "exception_types",
            "exceptions",
        ],
    )
    has_result = _contains_any(
        response,
        [
            "Result",
            "result",
            "success",
            "failure",
            "return_value",
            "metadata",
            "attempts",
            "dataclass",
            "namedtuple",
            "TypedDict",
        ],
    )
    return has_decorator and has_backoff and has_exception and has_result


# =============================================================================
# PROBLEM LISTS BY TIER
# =============================================================================

CODING_EASY_PROBLEMS: List[EvalProblem] = [
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
]

CODING_HARD_PROBLEMS: List[EvalProblem] = [
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
]

CODING_EXPERT_PROBLEMS: List[EvalProblem] = [
    EvalProblem(
        category="coding",
        name="markdown_to_html",
        prompt=(
            "Write a Python function `markdown_to_html(text)` that converts a subset of "
            "Markdown to HTML. It must handle:\n"
            "  1. **bold** → <strong>bold</strong>\n"
            "  2. *italic* → <em>italic</em>\n"
            "  3. `code` → <code>code</code>\n"
            '  4. [link text](url) → <a href="url">link text</a>\n\n'
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
]
