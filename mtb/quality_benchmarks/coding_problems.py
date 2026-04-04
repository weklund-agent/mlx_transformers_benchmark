"""Coding evaluation problems across all difficulty tiers.

Contains check functions and EvalProblem instances for coding problems:
- Easy (5): fizzbuzz, reverse_string, fibonacci, binary_search, palindrome
- Hard (5): lru_cache, flatten_nested, longest_palindrome_substring, calculator, buggy_merge_sort
- Expert (8): markdown_to_html, data_pipeline, retry_decorator, topological_sort,
              constrained_factorial, code_with_comments, library_schema, adversarial_transform

Each problem defines:
- function_signature: the signature the model must implement
- test_cases: 5-10 deterministic (input, expected_output) pairs with boundary cases
- _correct_impl: reference implementation that passes all test cases
- _incorrect_impl: deliberately wrong implementation that fails at least one test case
"""

from typing import List

from mtb.quality_benchmarks.utils import _contains_any, _strip_thinking

# Import EvalProblem from the canonical location to avoid circular imports.
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


def _check_topological_sort(response: str) -> bool:
    """Check if response implements topological sort correctly."""
    response = _strip_thinking(response)
    has_algorithm = _contains_any(
        response,
        [
            "topological",
            "kahn",
            "dfs",
            "in_degree",
            "in-degree",
            "indegree",
            "visited",
            "stack",
            "queue",
        ],
    )
    has_graph = _contains_any(
        response,
        [
            "graph",
            "node",
            "edge",
            "adjacen",
            "neighbor",
            "depend",
        ],
    )
    return has_algorithm and has_graph


def _check_constrained_factorial(response: str) -> bool:
    """Check if response implements constrained factorial correctly."""
    response = _strip_thinking(response)
    has_recursion = _contains_any(response, ["def ", "return"]) and not _contains_any(
        response, ["for ", "while "]
    )
    has_factorial_logic = _contains_any(
        response,
        [
            "factorial",
            "fact",
            "n * ",
            "n*",
            "* fact",
            "*fact",
            "n - 1",
            "n-1",
            "n == 0",
            "n == 1",
            "n <= 1",
        ],
    )
    return has_recursion and has_factorial_logic


def _check_code_with_comments(response: str) -> bool:
    """Check if response contains code with comments."""
    response = _strip_thinking(response)
    has_code = _contains_any(response, ["def ", "function ", "class ", "int ", "void "])
    has_comments = _contains_any(response, ["#", "//", "/*", "'''", '"""'])
    return has_code and has_comments


def _check_library_schema(response: str) -> bool:
    """Check if response defines a library schema with correct structure."""
    response = _strip_thinking(response)
    has_book = _contains_any(response, ["book", "Book"])
    has_author = _contains_any(response, ["author", "Author"])
    has_schema = _contains_any(
        response,
        [
            "properties",
            "type",
            "required",
            "schema",
            "class ",
            "def ",
        ],
    )
    return has_book and has_author and has_schema


def _check_adversarial_transform(response: str) -> bool:
    """Check if response implements the adversarial text transformation."""
    response = _strip_thinking(response)
    has_transform = _contains_any(
        response,
        [
            "transform",
            "reverse",
            "[::-1]",
            "upper",
            "lower",
            "replace",
            "def ",
        ],
    )
    has_pipeline = _contains_any(
        response,
        [
            "step",
            "stage",
            "pipeline",
            "chain",
            "sequence",
            "then",
            "def ",
        ],
    )
    return has_transform and has_pipeline


# =============================================================================
# REFERENCE IMPLEMENTATIONS
# =============================================================================

# --- Easy ---

_FIZZBUZZ_CORRECT = """\
def fizzbuzz(n: int) -> str:
    if n % 15 == 0:
        return "FizzBuzz"
    elif n % 3 == 0:
        return "Fizz"
    elif n % 5 == 0:
        return "Buzz"
    else:
        return str(n)
"""

_FIZZBUZZ_INCORRECT = """\
def fizzbuzz(n: int) -> str:
    if n % 3 == 0:
        return "Fizz"
    elif n % 5 == 0:
        return "Buzz"
    else:
        return str(n)
"""

_REVERSE_STRING_CORRECT = """\
def reverse_string(s: str) -> str:
    return s[::-1]
"""

_REVERSE_STRING_INCORRECT = """\
def reverse_string(s: str) -> str:
    return s
"""

_FIBONACCI_CORRECT = """\
def fibonacci(n: int) -> list:
    if n <= 0:
        return []
    if n == 1:
        return [0]
    result = [0, 1]
    for i in range(2, n):
        result.append(result[-1] + result[-2])
    return result
"""

_FIBONACCI_INCORRECT = """\
def fibonacci(n: int) -> list:
    if n <= 0:
        return []
    if n == 1:
        return [1]
    result = [1, 1]
    for i in range(2, n):
        result.append(result[-1] + result[-2])
    return result
"""

_BINARY_SEARCH_CORRECT = """\
def binary_search(arr: list, target: int) -> int:
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
"""

_BINARY_SEARCH_INCORRECT = """\
def binary_search(arr: list, target: int) -> int:
    low, high = 0, len(arr) - 1
    while low < high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
"""

_PALINDROME_CORRECT = """\
def is_palindrome(s: str) -> bool:
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]
"""

_PALINDROME_INCORRECT = """\
def is_palindrome(s: str) -> bool:
    return s == s[::-1]
"""

# --- Hard ---

_LRU_CACHE_CORRECT = """\
def lru_cache_ops(capacity: int, operations: list) -> list:
    from collections import OrderedDict
    cache = OrderedDict()
    results = []
    for op in operations:
        if op[0] == "get":
            key = op[1]
            if key in cache:
                cache.move_to_end(key)
                results.append(cache[key])
            else:
                results.append(-1)
        elif op[0] == "put":
            key, value = op[1], op[2]
            if key in cache:
                cache.move_to_end(key)
            cache[key] = value
            if len(cache) > capacity:
                cache.popitem(last=False)
            results.append(None)
    return results
"""

_LRU_CACHE_INCORRECT = """\
def lru_cache_ops(capacity: int, operations: list) -> list:
    cache = {}
    results = []
    for op in operations:
        if op[0] == "get":
            key = op[1]
            results.append(cache.get(key, -1))
        elif op[0] == "put":
            key, value = op[1], op[2]
            cache[key] = value
            results.append(None)
    return results
"""

_FLATTEN_NESTED_CORRECT = """\
def flatten(lst: list) -> list:
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
"""

_FLATTEN_NESTED_INCORRECT = """\
def flatten(lst: list) -> list:
    result = []
    for item in lst:
        if isinstance(item, list):
            for sub in item:
                result.append(sub)
        else:
            result.append(item)
    return result
"""

_LONGEST_PALINDROME_CORRECT = """\
def longest_palindrome(s: str) -> str:
    if not s:
        return ""
    start, max_len = 0, 1
    def expand(left, right):
        nonlocal start, max_len
        while left >= 0 and right < len(s) and s[left] == s[right]:
            if right - left + 1 > max_len:
                start = left
                max_len = right - left + 1
            left -= 1
            right += 1
    for i in range(len(s)):
        expand(i, i)
        expand(i, i + 1)
    return s[start:start + max_len]
"""

_LONGEST_PALINDROME_INCORRECT = """\
def longest_palindrome(s: str) -> str:
    if not s:
        return ""
    # Bug: returns first char instead of longest palindromic substring
    return s[0]
"""

_CALCULATOR_CORRECT = """\
def calculate(expression: str) -> float:
    tokens = []
    i = 0
    expr = expression.replace(" ", "")
    while i < len(expr):
        if expr[i].isdigit() or (expr[i] == '.' and i + 1 < len(expr) and expr[i+1].isdigit()):
            j = i
            while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                j += 1
            tokens.append(float(expr[i:j]))
            i = j
        else:
            tokens.append(expr[i])
            i += 1
    pos = [0]
    def parse_expr():
        result = parse_term()
        while pos[0] < len(tokens) and tokens[pos[0]] in ('+', '-'):
            op = tokens[pos[0]]
            pos[0] += 1
            right = parse_term()
            if op == '+':
                result += right
            else:
                result -= right
        return result
    def parse_term():
        result = parse_factor()
        while pos[0] < len(tokens) and tokens[pos[0]] in ('*', '/'):
            op = tokens[pos[0]]
            pos[0] += 1
            right = parse_factor()
            if op == '*':
                result *= right
            else:
                result /= right
        return result
    def parse_factor():
        if tokens[pos[0]] == '(':
            pos[0] += 1
            result = parse_expr()
            pos[0] += 1  # skip ')'
            return result
        else:
            val = tokens[pos[0]]
            pos[0] += 1
            return val
    return parse_expr()
"""

_CALCULATOR_INCORRECT = """\
def calculate(expression: str) -> float:
    tokens = expression.replace(" ", "")
    result = 0.0
    current = 0.0
    op = '+'
    i = 0
    while i < len(tokens):
        if tokens[i].isdigit() or tokens[i] == '.':
            j = i
            while j < len(tokens) and (tokens[j].isdigit() or tokens[j] == '.'):
                j += 1
            num = float(tokens[i:j])
            if op == '+':
                result += num
            elif op == '-':
                result -= num
            elif op == '*':
                result *= num
            elif op == '/':
                result /= num
            i = j
        elif tokens[i] in '+-*/':
            op = tokens[i]
            i += 1
        else:
            i += 1
    return result
"""

_BUGGY_MERGE_SORT_CORRECT = """\
def merge_sort(arr: list) -> list:
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left: list, right: list) -> list:
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
"""

_BUGGY_MERGE_SORT_INCORRECT = """\
def merge_sort(arr: list) -> list:
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left: list, right: list) -> list:
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    return result
"""

# --- Expert ---

_MARKDOWN_TO_HTML_CORRECT = """\
import re

def markdown_to_html(text: str) -> str:
    # Bold: **text** -> <strong>text</strong>
    text = re.sub(r'\\*\\*(.+?)\\*\\*', r'<strong>\\1</strong>', text)
    # Italic: *text* -> <em>text</em>
    text = re.sub(r'\\*(.+?)\\*', r'<em>\\1</em>', text)
    # Code: `text` -> <code>text</code>
    text = re.sub(r'`(.+?)`', r'<code>\\1</code>', text)
    # Links: [text](url) -> <a href="url">text</a>
    text = re.sub(r'\\[(.+?)\\]\\((.+?)\\)', r'<a href="\\2">\\1</a>', text)
    return text
"""

_MARKDOWN_TO_HTML_INCORRECT = """\
import re

def markdown_to_html(text: str) -> str:
    # Only handles bold
    text = re.sub(r'\\*\\*(.+?)\\*\\*', r'<strong>\\1</strong>', text)
    return text
"""

_DATA_PIPELINE_CORRECT = """\
import csv
import json
from io import StringIO
from collections import defaultdict

def process_sales(csv_text: str) -> str:
    reader = csv.DictReader(StringIO(csv_text))
    groups = defaultdict(list)
    for row in reader:
        amount = float(row['amount'])
        quantity = int(row['quantity'])
        region = row['region']
        if amount > 100 and region in ('North', 'West') and quantity >= 5:
            groups[row['product']].append(amount)
    result = []
    for product, amounts in groups.items():
        result.append({
            'product': product,
            'total_amount': sum(amounts),
            'avg_amount': sum(amounts) / len(amounts),
            'count': len(amounts)
        })
    result.sort(key=lambda x: x['total_amount'], reverse=True)
    return json.dumps(result)
"""

_DATA_PIPELINE_INCORRECT = """\
import csv
import json
from io import StringIO

def process_sales(csv_text: str) -> str:
    reader = csv.DictReader(StringIO(csv_text))
    result = []
    for row in reader:
        result.append({'product': row['product'], 'amount': float(row['amount'])})
    return json.dumps(result)
"""

_RETRY_DECORATOR_CORRECT = """\
def retry_call(func, max_attempts: int, args: tuple) -> dict:
    attempts = 0
    last_error = None
    for i in range(max_attempts):
        attempts += 1
        try:
            value = func(*args)
            return {"success": True, "value": value, "attempts": attempts, "error": None}
        except Exception as e:
            last_error = str(e)
    return {"success": False, "value": None, "attempts": attempts, "error": last_error}
"""

_RETRY_DECORATOR_INCORRECT = """\
def retry_call(func, max_attempts: int, args: tuple) -> dict:
    try:
        value = func(*args)
        return {"success": True, "value": value, "attempts": 1, "error": None}
    except Exception as e:
        return {"success": False, "value": None, "attempts": 1, "error": str(e)}
"""

_TOPOLOGICAL_SORT_CORRECT = """\
def topological_sort(graph: dict) -> list:
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            if neighbor not in in_degree:
                in_degree[neighbor] = 0
            in_degree[neighbor] += 1
    queue = sorted([n for n in in_degree if in_degree[n] == 0])
    result = []
    while queue:
        node = queue.pop(0)
        result.append(node)
        for neighbor in sorted(graph.get(node, [])):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
                queue.sort()
    return result
"""

_TOPOLOGICAL_SORT_INCORRECT = """\
def topological_sort(graph: dict) -> list:
    return sorted(graph.keys())
"""

_CONSTRAINED_FACTORIAL_CORRECT = """\
def constrained_factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * constrained_factorial(n - 1)
"""

_CONSTRAINED_FACTORIAL_INCORRECT = """\
def constrained_factorial(n: int) -> int:
    if n <= 1:
        return 0
    return n * constrained_factorial(n - 1)
"""

_CODE_WITH_COMMENTS_CORRECT = """\
import math

def circle_area(radius: float) -> float:
    # Calculate the area of a circle using A = pi * r^2
    # Validate input
    if radius < 0:
        raise ValueError("Radius cannot be negative")
    # Compute and return area
    area = math.pi * radius ** 2
    return round(area, 6)
"""

_CODE_WITH_COMMENTS_INCORRECT = """\
import math

def circle_area(radius: float) -> float:
    return math.pi * radius
"""

_LIBRARY_SCHEMA_CORRECT = """\
def library_schema() -> dict:
    return {
        "Book": {
            "fields": {
                "id": "int",
                "title": "str",
                "isbn": "str",
                "author_id": "int",
                "publication_year": "int"
            },
            "required": ["id", "title", "author_id"]
        },
        "Author": {
            "fields": {
                "id": "int",
                "name": "str",
                "birth_year": "int",
                "nationality": "str"
            },
            "required": ["id", "name"]
        },
        "Loan": {
            "fields": {
                "id": "int",
                "book_id": "int",
                "borrower_name": "str",
                "due_date": "str",
                "returned": "bool"
            },
            "required": ["id", "book_id", "borrower_name"]
        }
    }
"""

_LIBRARY_SCHEMA_INCORRECT = """\
def library_schema() -> dict:
    return {
        "Book": {
            "fields": {"title": "str"},
            "required": ["title"]
        }
    }
"""

_ADVERSARIAL_TRANSFORM_CORRECT = """\
def adversarial_transform(text: str) -> str:
    words = text.split()
    transformed = []
    for word in words:
        reversed_word = word[::-1]
        swapped = reversed_word.swapcase()
        transformed.append(swapped)
    return ' '.join(transformed)
"""

_ADVERSARIAL_TRANSFORM_INCORRECT = """\
def adversarial_transform(text: str) -> str:
    return text.upper()
"""


# =============================================================================
# TEST CASES
# =============================================================================

_FIZZBUZZ_TEST_CASES = [
    {"input": "1", "expected_output": "1"},
    {"input": "3", "expected_output": "Fizz"},
    {"input": "5", "expected_output": "Buzz"},
    {"input": "15", "expected_output": "FizzBuzz"},
    {"input": "30", "expected_output": "FizzBuzz"},
    {"input": "7", "expected_output": "7"},
    {"input": "9", "expected_output": "Fizz"},
    {"input": "20", "expected_output": "Buzz"},
]

_REVERSE_STRING_TEST_CASES = [
    {"input": "''", "expected_output": ""},
    {"input": "'a'", "expected_output": "a"},
    {"input": "'hello'", "expected_output": "olleh"},
    {"input": "'racecar'", "expected_output": "racecar"},
    {"input": "'ab cd'", "expected_output": "dc ba"},
    {"input": "'12345'", "expected_output": "54321"},
]

_FIBONACCI_TEST_CASES = [
    {"input": "0", "expected_output": []},
    {"input": "1", "expected_output": [0]},
    {"input": "2", "expected_output": [0, 1]},
    {"input": "5", "expected_output": [0, 1, 1, 2, 3]},
    {"input": "8", "expected_output": [0, 1, 1, 2, 3, 5, 8, 13]},
    {"input": "10", "expected_output": [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]},
]

_BINARY_SEARCH_TEST_CASES = [
    {"input": "[], 5", "expected_output": -1},
    {"input": "[1], 1", "expected_output": 0},
    {"input": "[1], 2", "expected_output": -1},
    {"input": "[1, 3, 5, 7, 9], 5", "expected_output": 2},
    {"input": "[1, 3, 5, 7, 9], 1", "expected_output": 0},
    {"input": "[1, 3, 5, 7, 9], 9", "expected_output": 4},
    {"input": "[1, 3, 5, 7, 9], 4", "expected_output": -1},
    {"input": "[2, 4, 6, 8, 10, 12], 8", "expected_output": 3},
]

_PALINDROME_TEST_CASES = [
    {"input": "''", "expected_output": True},
    {"input": "'a'", "expected_output": True},
    {"input": "'racecar'", "expected_output": True},
    {"input": "'hello'", "expected_output": False},
    {"input": "'A man a plan a canal Panama'", "expected_output": True},
    {"input": "'Was it a car or a cat I saw'", "expected_output": True},
    {"input": "'No lemon, no melon'", "expected_output": True},
    {"input": "'abc'", "expected_output": False},
]

_LRU_CACHE_TEST_CASES = [
    {
        "input": "1, [('put', 1, 10), ('get', 1)]",
        "expected_output": [None, 10],
    },
    {
        "input": "2, [('put', 1, 1), ('put', 2, 2), ('get', 1), ('put', 3, 3), ('get', 2)]",
        "expected_output": [None, None, 1, None, -1],
    },
    {
        "input": "2, [('put', 1, 1), ('put', 2, 2), ('put', 3, 3), ('get', 1)]",
        "expected_output": [None, None, None, -1],
    },
    {
        "input": "1, [('put', 1, 1), ('put', 2, 2), ('get', 1)]",
        "expected_output": [None, None, -1],
    },
    {
        "input": "3, [('put', 1, 1), ('put', 2, 2), ('put', 3, 3), ('get', 1), ('get', 2), ('get', 3)]",
        "expected_output": [None, None, None, 1, 2, 3],
    },
    {
        "input": "2, [('put', 1, 1), ('put', 2, 2), ('get', 1), ('put', 3, 3), ('get', 2), ('get', 3)]",
        "expected_output": [None, None, 1, None, -1, 3],
    },
]

_FLATTEN_NESTED_TEST_CASES = [
    {"input": "[]", "expected_output": []},
    {"input": "[1, 2, 3]", "expected_output": [1, 2, 3]},
    {"input": "[1, [2, 3], 4]", "expected_output": [1, 2, 3, 4]},
    {"input": "[1, [2, [3, 4], 5], [6, 7]]", "expected_output": [1, 2, 3, 4, 5, 6, 7]},
    {"input": "[[1, 2], [[3]], [4, [5, [6]]]]", "expected_output": [1, 2, 3, 4, 5, 6]},
    {"input": "[[], [1], []]", "expected_output": [1]},
    {"input": "[[[[1]]]]", "expected_output": [1]},
]

_LONGEST_PALINDROME_TEST_CASES = [
    {"input": "'a'", "expected_output": "a"},
    {"input": "'bb'", "expected_output": "bb"},
    {"input": "'babad'", "expected_output": "bab"},
    {"input": "'cbbd'", "expected_output": "bb"},
    {"input": "'racecar'", "expected_output": "racecar"},
    {"input": "'abc'", "expected_output": "a"},
    {"input": "''", "expected_output": ""},
]

_CALCULATOR_TEST_CASES = [
    {"input": "'5'", "expected_output": 5.0},
    {"input": "'2 + 3'", "expected_output": 5.0},
    {"input": "'2 + 3 * 4'", "expected_output": 14.0},
    {"input": "'(2 + 3) * 4'", "expected_output": 20.0},
    {"input": "'10 + 2 * (3 + 4)'", "expected_output": 24.0},
    {"input": "'100 / 10'", "expected_output": 10.0},
    {"input": "'2 * 3 + 4 * 5'", "expected_output": 26.0},
]

_BUGGY_MERGE_SORT_TEST_CASES = [
    {"input": "[]", "expected_output": []},
    {"input": "[1]", "expected_output": [1]},
    {"input": "[3, 1, 2]", "expected_output": [1, 2, 3]},
    {"input": "[5, 4, 3, 2, 1]", "expected_output": [1, 2, 3, 4, 5]},
    {"input": "[1, 2, 3, 4, 5]", "expected_output": [1, 2, 3, 4, 5]},
    {"input": "[4, 2, 7, 1, 9, 3]", "expected_output": [1, 2, 3, 4, 7, 9]},
    {"input": "[1, 1, 1]", "expected_output": [1, 1, 1]},
]

_MARKDOWN_TO_HTML_TEST_CASES = [
    {"input": "'**bold**'", "expected_output": "<strong>bold</strong>"},
    {"input": "'*italic*'", "expected_output": "<em>italic</em>"},
    {"input": "'`code`'", "expected_output": "<code>code</code>"},
    {
        "input": "'[link](http://example.com)'",
        "expected_output": '<a href="http://example.com">link</a>',
    },
    {
        "input": "'**bold** and *italic*'",
        "expected_output": "<strong>bold</strong> and <em>italic</em>",
    },
    {"input": "'plain text'", "expected_output": "plain text"},
]

_DATA_PIPELINE_CSV = (
    "date,product,region,amount,quantity\n"
    "2024-01-01,Widget,North,150,10\n"
    "2024-01-02,Widget,South,200,8\n"
    "2024-01-03,Gadget,West,120,5\n"
    "2024-01-04,Widget,North,80,3\n"
    "2024-01-05,Gadget,North,250,7\n"
    "2024-01-06,Widget,West,300,6\n"
)

_DATA_PIPELINE_TEST_CASES = [
    {
        "input": repr(_DATA_PIPELINE_CSV),
        "expected_output": '[{"product": "Widget", "total_amount": 450.0, "avg_amount": 225.0, "count": 2}, {"product": "Gadget", "total_amount": 370.0, "avg_amount": 185.0, "count": 2}]',
    },
    {
        "input": repr("date,product,region,amount,quantity\n2024-01-01,A,South,50,2\n"),
        "expected_output": "[]",
    },
    {
        "input": repr(
            "date,product,region,amount,quantity\n2024-01-01,A,North,200,10\n"
        ),
        "expected_output": '[{"product": "A", "total_amount": 200.0, "avg_amount": 200.0, "count": 1}]',
    },
    {
        "input": repr(
            "date,product,region,amount,quantity\n2024-01-01,A,North,200,3\n"
        ),
        "expected_output": "[]",
    },
    {
        "input": repr(
            "date,product,region,amount,quantity\n2024-01-01,A,North,50,10\n"
        ),
        "expected_output": "[]",
    },
]

_RETRY_TEST_CASES = [
    {
        "input": "lambda: 42, 3, ()",
        "expected_output": {"success": True, "value": 42, "attempts": 1, "error": None},
    },
    {
        "input": "lambda: (_ for _ in ()).throw(ValueError('fail')), 1, ()",
        "expected_output": {
            "success": False,
            "value": None,
            "attempts": 1,
            "error": "fail",
        },
    },
    {
        "input": "lambda: 99, 5, ()",
        "expected_output": {"success": True, "value": 99, "attempts": 1, "error": None},
    },
    {
        "input": "lambda: (_ for _ in ()).throw(RuntimeError('err')), 3, ()",
        "expected_output": {
            "success": False,
            "value": None,
            "attempts": 3,
            "error": "err",
        },
    },
    {
        "input": "lambda: 'hello', 2, ()",
        "expected_output": {
            "success": True,
            "value": "hello",
            "attempts": 1,
            "error": None,
        },
    },
]

_TOPOLOGICAL_SORT_TEST_CASES = [
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
]

_CONSTRAINED_FACTORIAL_TEST_CASES = [
    {"input": "0", "expected_output": 1},
    {"input": "1", "expected_output": 1},
    {"input": "5", "expected_output": 120},
    {"input": "10", "expected_output": 3628800},
    {"input": "3", "expected_output": 6},
    {"input": "7", "expected_output": 5040},
]

_CODE_WITH_COMMENTS_TEST_CASES = [
    {"input": "0.0", "expected_output": 0.0},
    {"input": "1.0", "expected_output": 3.141593},
    {"input": "2.0", "expected_output": 12.566371},
    {"input": "0.5", "expected_output": 0.785398},
    {"input": "10.0", "expected_output": 314.159265},
    {"input": "100.0", "expected_output": 31415.926536},
]

_LIBRARY_SCHEMA_TEST_CASES = [
    {
        "input": "",
        "expected_output": {
            "Book": {
                "fields": {
                    "id": "int",
                    "title": "str",
                    "isbn": "str",
                    "author_id": "int",
                    "publication_year": "int",
                },
                "required": ["id", "title", "author_id"],
            },
            "Author": {
                "fields": {
                    "id": "int",
                    "name": "str",
                    "birth_year": "int",
                    "nationality": "str",
                },
                "required": ["id", "name"],
            },
            "Loan": {
                "fields": {
                    "id": "int",
                    "book_id": "int",
                    "borrower_name": "str",
                    "due_date": "str",
                    "returned": "bool",
                },
                "required": ["id", "book_id", "borrower_name"],
            },
        },
    },
    {
        "input": "",
        "expected_output": {
            "Book": {
                "fields": {
                    "id": "int",
                    "title": "str",
                    "isbn": "str",
                    "author_id": "int",
                    "publication_year": "int",
                },
                "required": ["id", "title", "author_id"],
            },
            "Author": {
                "fields": {
                    "id": "int",
                    "name": "str",
                    "birth_year": "int",
                    "nationality": "str",
                },
                "required": ["id", "name"],
            },
            "Loan": {
                "fields": {
                    "id": "int",
                    "book_id": "int",
                    "borrower_name": "str",
                    "due_date": "str",
                    "returned": "bool",
                },
                "required": ["id", "book_id", "borrower_name"],
            },
        },
    },
    {
        "input": "",
        "expected_output": {
            "Book": {
                "fields": {
                    "id": "int",
                    "title": "str",
                    "isbn": "str",
                    "author_id": "int",
                    "publication_year": "int",
                },
                "required": ["id", "title", "author_id"],
            },
            "Author": {
                "fields": {
                    "id": "int",
                    "name": "str",
                    "birth_year": "int",
                    "nationality": "str",
                },
                "required": ["id", "name"],
            },
            "Loan": {
                "fields": {
                    "id": "int",
                    "book_id": "int",
                    "borrower_name": "str",
                    "due_date": "str",
                    "returned": "bool",
                },
                "required": ["id", "book_id", "borrower_name"],
            },
        },
    },
    {
        "input": "",
        "expected_output": {
            "Book": {
                "fields": {
                    "id": "int",
                    "title": "str",
                    "isbn": "str",
                    "author_id": "int",
                    "publication_year": "int",
                },
                "required": ["id", "title", "author_id"],
            },
            "Author": {
                "fields": {
                    "id": "int",
                    "name": "str",
                    "birth_year": "int",
                    "nationality": "str",
                },
                "required": ["id", "name"],
            },
            "Loan": {
                "fields": {
                    "id": "int",
                    "book_id": "int",
                    "borrower_name": "str",
                    "due_date": "str",
                    "returned": "bool",
                },
                "required": ["id", "book_id", "borrower_name"],
            },
        },
    },
    {
        "input": "",
        "expected_output": {
            "Book": {
                "fields": {
                    "id": "int",
                    "title": "str",
                    "isbn": "str",
                    "author_id": "int",
                    "publication_year": "int",
                },
                "required": ["id", "title", "author_id"],
            },
            "Author": {
                "fields": {
                    "id": "int",
                    "name": "str",
                    "birth_year": "int",
                    "nationality": "str",
                },
                "required": ["id", "name"],
            },
            "Loan": {
                "fields": {
                    "id": "int",
                    "book_id": "int",
                    "borrower_name": "str",
                    "due_date": "str",
                    "returned": "bool",
                },
                "required": ["id", "book_id", "borrower_name"],
            },
        },
    },
]

_ADVERSARIAL_TRANSFORM_TEST_CASES = [
    {"input": "'hello'", "expected_output": "OLLEH"},
    {"input": "'Hello World'", "expected_output": "OLLEh DLROw"},
    {"input": "''", "expected_output": ""},
    {"input": "'a'", "expected_output": "A"},
    {"input": "'ABC'", "expected_output": "cba"},
    {"input": "'Test Case'", "expected_output": "TSEt ESAc"},
]


# =============================================================================
# PROBLEM LISTS BY TIER
# =============================================================================

CODING_EASY_PROBLEMS: List[EvalProblem] = [
    EvalProblem(
        category="coding",
        name="fizzbuzz",
        prompt="Write a Python function `fizzbuzz(n: int) -> str` that returns 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for multiples of both, or the number as a string otherwise.",
        check=_check_fizzbuzz,
        function_signature="def fizzbuzz(n: int) -> str:",
        test_cases=_FIZZBUZZ_TEST_CASES,
        _correct_impl=_FIZZBUZZ_CORRECT,
        _incorrect_impl=_FIZZBUZZ_INCORRECT,
    ),
    EvalProblem(
        category="coding",
        name="reverse_string",
        prompt="Write a Python function `reverse_string(s: str) -> str` that reverses the input string.",
        check=_check_reverse_string,
        function_signature="def reverse_string(s: str) -> str:",
        test_cases=_REVERSE_STRING_TEST_CASES,
        _correct_impl=_REVERSE_STRING_CORRECT,
        _incorrect_impl=_REVERSE_STRING_INCORRECT,
    ),
    EvalProblem(
        category="coding",
        name="fibonacci",
        prompt="Write a Python function `fibonacci(n: int) -> list` that returns the first n Fibonacci numbers as a list starting with 0.",
        check=_check_fibonacci,
        function_signature="def fibonacci(n: int) -> list:",
        test_cases=_FIBONACCI_TEST_CASES,
        _correct_impl=_FIBONACCI_CORRECT,
        _incorrect_impl=_FIBONACCI_INCORRECT,
    ),
    EvalProblem(
        category="coding",
        name="binary_search",
        prompt="Write a Python function `binary_search(arr: list, target: int) -> int` that returns the index of target in a sorted list, or -1 if not found.",
        check=_check_binary_search,
        function_signature="def binary_search(arr: list, target: int) -> int:",
        test_cases=_BINARY_SEARCH_TEST_CASES,
        _correct_impl=_BINARY_SEARCH_CORRECT,
        _incorrect_impl=_BINARY_SEARCH_INCORRECT,
    ),
    EvalProblem(
        category="coding",
        name="palindrome",
        prompt="Write a Python function `is_palindrome(s: str) -> bool` that checks if a string is a palindrome, ignoring case and non-alphanumeric characters.",
        check=_check_palindrome,
        function_signature="def is_palindrome(s: str) -> bool:",
        test_cases=_PALINDROME_TEST_CASES,
        _correct_impl=_PALINDROME_CORRECT,
        _incorrect_impl=_PALINDROME_INCORRECT,
    ),
]

CODING_HARD_PROBLEMS: List[EvalProblem] = [
    EvalProblem(
        category="coding",
        name="lru_cache",
        prompt=(
            "Implement a Python function `lru_cache_ops(capacity: int, operations: list) -> list` "
            "that simulates an LRU cache. Operations are tuples: ('get', key) or ('put', key, value). "
            "Return a list of results (get returns value or -1, put returns None)."
        ),
        check=_check_lru_cache,
        max_tokens=2048,
        function_signature="def lru_cache_ops(capacity: int, operations: list) -> list:",
        test_cases=_LRU_CACHE_TEST_CASES,
        _correct_impl=_LRU_CACHE_CORRECT,
        _incorrect_impl=_LRU_CACHE_INCORRECT,
    ),
    EvalProblem(
        category="coding",
        name="flatten_nested",
        prompt=(
            "Write a Python function `flatten(lst: list) -> list` that takes an arbitrarily nested list "
            "and returns a flat list of all values."
        ),
        check=_check_flatten_nested,
        max_tokens=1024,
        function_signature="def flatten(lst: list) -> list:",
        test_cases=_FLATTEN_NESTED_TEST_CASES,
        _correct_impl=_FLATTEN_NESTED_CORRECT,
        _incorrect_impl=_FLATTEN_NESTED_INCORRECT,
    ),
    EvalProblem(
        category="coding",
        name="longest_palindrome_substring",
        prompt=(
            "Write a Python function `longest_palindrome(s: str) -> str` that finds the longest "
            "palindromic substring in a given string."
        ),
        check=_check_longest_palindrome_substring,
        max_tokens=2048,
        function_signature="def longest_palindrome(s: str) -> str:",
        test_cases=_LONGEST_PALINDROME_TEST_CASES,
        _correct_impl=_LONGEST_PALINDROME_CORRECT,
        _incorrect_impl=_LONGEST_PALINDROME_INCORRECT,
    ),
    EvalProblem(
        category="coding",
        name="calculator",
        prompt=(
            "Implement a Python function `calculate(expression: str) -> float` that evaluates "
            "a math expression with +, -, *, /, and parentheses, respecting operator precedence."
        ),
        check=_check_calculator,
        max_tokens=2048,
        function_signature="def calculate(expression: str) -> float:",
        test_cases=_CALCULATOR_TEST_CASES,
        _correct_impl=_CALCULATOR_CORRECT,
        _incorrect_impl=_CALCULATOR_INCORRECT,
    ),
    EvalProblem(
        category="coding",
        name="buggy_merge_sort",
        prompt=(
            "Implement a correct Python function `merge_sort(arr: list) -> list` that sorts a list "
            "using merge sort. The function should handle empty lists, single elements, and duplicates."
        ),
        check=_check_buggy_merge_sort,
        max_tokens=2048,
        function_signature="def merge_sort(arr: list) -> list:",
        test_cases=_BUGGY_MERGE_SORT_TEST_CASES,
        _correct_impl=_BUGGY_MERGE_SORT_CORRECT,
        _incorrect_impl=_BUGGY_MERGE_SORT_INCORRECT,
    ),
]

CODING_EXPERT_PROBLEMS: List[EvalProblem] = [
    EvalProblem(
        category="coding",
        name="markdown_to_html",
        prompt=(
            "Write a Python function `markdown_to_html(text: str) -> str` that converts "
            "Markdown to HTML. Handle **bold**, *italic*, `code`, and [link](url)."
        ),
        check=_check_markdown_to_html,
        max_tokens=2048,
        function_signature="def markdown_to_html(text: str) -> str:",
        test_cases=_MARKDOWN_TO_HTML_TEST_CASES,
        _correct_impl=_MARKDOWN_TO_HTML_CORRECT,
        _incorrect_impl=_MARKDOWN_TO_HTML_INCORRECT,
    ),
    EvalProblem(
        category="coding",
        name="data_pipeline",
        prompt=(
            "Write a Python function `process_sales(csv_text: str) -> str` that processes CSV data: "
            "filter (amount>100, region in North/West, quantity>=5), group by product, "
            "aggregate (total, avg, count), sort by total descending, return JSON string."
        ),
        check=_check_data_pipeline,
        max_tokens=2048,
        function_signature="def process_sales(csv_text: str) -> str:",
        test_cases=_DATA_PIPELINE_TEST_CASES,
        _correct_impl=_DATA_PIPELINE_CORRECT,
        _incorrect_impl=_DATA_PIPELINE_INCORRECT,
    ),
    EvalProblem(
        category="coding",
        name="retry_decorator",
        prompt=(
            "Write a Python function `retry_call(func, max_attempts: int, args: tuple) -> dict` "
            "that calls func(*args) up to max_attempts times, returning a dict with keys: "
            "success (bool), value (any), attempts (int), error (str or None)."
        ),
        check=_check_retry_decorator,
        max_tokens=2048,
        function_signature="def retry_call(func, max_attempts: int, args: tuple) -> dict:",
        test_cases=_RETRY_TEST_CASES,
        _correct_impl=_RETRY_DECORATOR_CORRECT,
        _incorrect_impl=_RETRY_DECORATOR_INCORRECT,
    ),
]

# =============================================================================
# EXPERT CODING PROBLEMS FROM OTHER CATEGORIES
# These problems originate in reasoning_problems.py and instruction_problems.py
# but are also coding problems that can be evaluated via code execution.
# They are NOT added to CODING_EXPERT_PROBLEMS (which feeds into
# EXPERT_EVAL_PROBLEMS) to avoid double-counting. They are only included
# in the CODING_PROBLEMS aggregate list for code-execution-based evaluation.
# =============================================================================

_CROSS_CATEGORY_EXPERT_CODING: List[EvalProblem] = [
    EvalProblem(
        category="coding",
        name="topological_sort",
        prompt=(
            "Write a Python function `topological_sort(graph: dict) -> list` that performs "
            "topological sort on a directed acyclic graph represented as adjacency list. "
            "Return nodes in topologically sorted order (alphabetical tie-breaking)."
        ),
        check=_check_topological_sort,
        max_tokens=2048,
        function_signature="def topological_sort(graph: dict) -> list:",
        test_cases=_TOPOLOGICAL_SORT_TEST_CASES,
        _correct_impl=_TOPOLOGICAL_SORT_CORRECT,
        _incorrect_impl=_TOPOLOGICAL_SORT_INCORRECT,
    ),
    EvalProblem(
        category="coding",
        name="constrained_factorial",
        prompt=(
            "Write a Python function `constrained_factorial(n: int) -> int` that computes "
            "the factorial of n using recursion (no loops). factorial(0) = 1."
        ),
        check=_check_constrained_factorial,
        max_tokens=2048,
        function_signature="def constrained_factorial(n: int) -> int:",
        test_cases=_CONSTRAINED_FACTORIAL_TEST_CASES,
        _correct_impl=_CONSTRAINED_FACTORIAL_CORRECT,
        _incorrect_impl=_CONSTRAINED_FACTORIAL_INCORRECT,
    ),
    EvalProblem(
        category="coding",
        name="code_with_comments",
        prompt=(
            "Write a Python function `circle_area(radius: float) -> float` that calculates "
            "the area of a circle. Include comments. Return value rounded to 6 decimal places. "
            "Raise ValueError for negative radius."
        ),
        check=_check_code_with_comments,
        max_tokens=1024,
        function_signature="def circle_area(radius: float) -> float:",
        test_cases=_CODE_WITH_COMMENTS_TEST_CASES,
        _correct_impl=_CODE_WITH_COMMENTS_CORRECT,
        _incorrect_impl=_CODE_WITH_COMMENTS_INCORRECT,
    ),
    EvalProblem(
        category="coding",
        name="library_schema",
        prompt=(
            "Write a Python function `library_schema() -> dict` that returns a schema dict "
            "with keys 'Book', 'Author', 'Loan'. Each has 'fields' (name->type mapping) "
            "and 'required' (list of required field names)."
        ),
        check=_check_library_schema,
        max_tokens=2048,
        function_signature="def library_schema() -> dict:",
        test_cases=_LIBRARY_SCHEMA_TEST_CASES,
        _correct_impl=_LIBRARY_SCHEMA_CORRECT,
        _incorrect_impl=_LIBRARY_SCHEMA_INCORRECT,
    ),
    EvalProblem(
        category="coding",
        name="adversarial_transform",
        prompt=(
            "Write a Python function `adversarial_transform(text: str) -> str` that: "
            "1) splits text into words, 2) reverses each word, 3) swaps case of each character, "
            "4) joins back with spaces."
        ),
        check=_check_adversarial_transform,
        max_tokens=2048,
        function_signature="def adversarial_transform(text: str) -> str:",
        test_cases=_ADVERSARIAL_TRANSFORM_TEST_CASES,
        _correct_impl=_ADVERSARIAL_TRANSFORM_CORRECT,
        _incorrect_impl=_ADVERSARIAL_TRANSFORM_INCORRECT,
    ),
]

# =============================================================================
# COMBINED LIST — all 18 coding problems for code-execution evaluation
# =============================================================================

CODING_PROBLEMS: List[EvalProblem] = (
    CODING_EASY_PROBLEMS
    + CODING_HARD_PROBLEMS
    + CODING_EXPERT_PROBLEMS
    + _CROSS_CATEGORY_EXPERT_CODING
)
