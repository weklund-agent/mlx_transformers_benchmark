"""Instruction following evaluation problems across all difficulty tiers.

Contains check functions and EvalProblem instances for instruction following problems:
- Easy (5): json_output, numbered_list, word_constraint, code_with_comments, direct_answer
- Expert (3): constrained_factorial, library_schema, adversarial_transform
"""

from typing import List

from mtb.quality_benchmarks.utils import _contains_any, _strip_thinking
from mtb.quality_benchmarks.eval_problem import EvalProblem

import re


# =============================================================================
# EASY INSTRUCTION FOLLOWING CHECK FUNCTIONS
# =============================================================================


def _check_json_output(response: str) -> bool:
    """Check if model produced valid-looking JSON."""
    response = _strip_thinking(response)
    return "{" in response and "}" in response and '"name"' in response.lower()


def _check_list_format(response: str) -> bool:
    """Check if model produced a numbered list."""
    response = _strip_thinking(response)
    has_numbers = bool(re.search(r"[1-5][.)]", response))
    has_multiple = len(re.findall(r"\d[.)]", response)) >= 3
    return has_numbers and has_multiple


def _check_word_constraint(response: str) -> bool:
    """Check if the model's actual explanation is reasonably concise."""
    response = _strip_thinking(response)
    return _contains_any(
        response,
        [
            "machine learning",
            "subset of ai",
            "subset of artificial intelligence",
            "learn from data",
            "algorithms",
            "patterns",
        ],
    )


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
# EXPERT INSTRUCTION FOLLOWING CHECK FUNCTIONS
# =============================================================================


def _check_constrained_factorial(response: str) -> bool:
    """Write a Python function where: (a) every variable name is exactly 4 chars,
    (b) no loops (use recursion), (c) exactly 3 type hints, (d) computes factorial.
    """
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
    has_type_hints = _contains_any(
        response,
        [
            "-> int",
            ": int",
            "-> float",
            ": float",
            "-> None",
            ": Optional",
        ],
    )
    has_var_constraint = _contains_any(
        response,
        [
            "4 char",
            "four char",
            "exactly 4",
            "4-char",
            "4 letter",
        ],
    ) or bool(re.search(r"\b[a-z]{4}\b\s*[=:]", response))

    return (
        has_factorial_logic and has_type_hints and (has_recursion or has_var_constraint)
    )


def _check_library_schema(response: str) -> bool:
    """Generate JSON schema for library system with Book, Author, Loan — cross-references by ID."""
    response = _strip_thinking(response)
    has_book = _contains_any(response, ["book", "Book"])
    has_author = _contains_any(response, ["author", "Author"])
    has_loan = _contains_any(response, ["loan", "Loan"])
    has_id_refs = _contains_any(
        response,
        [
            "book_id",
            "author_id",
            "loan_id",
            "bookId",
            "authorId",
            "loanId",
            "$ref",
            "reference",
            "foreign_key",
            "id",
            "ID",
        ],
    )
    has_json_schema = _contains_any(
        response,
        [
            "properties",
            "type",
            "required",
            "schema",
            "string",
            "integer",
            "array",
            "object",
        ],
    )
    return has_book and has_author and has_loan and has_id_refs and has_json_schema


def _check_adversarial_transform(response: str) -> bool:
    """3-stage pipeline: Translate to French, reverse each word, wrap in XML tags."""
    response = _strip_thinking(response)
    has_french = _contains_any(
        response,
        [
            "renard",
            "chien",
            "rapide",
            "saute",
            "brun",
            "paresseux",
            "Le ",
            "le ",
        ],
    )
    has_reversed = _contains_any(
        response,
        [
            "eL",
            "dranre",
            "nurb",
            "edipar",
            "reverse",
            "[::-1]",
        ],
    )
    has_xml = _contains_any(
        response,
        [
            "<word",
            "original=",
            "</word>",
            "xml",
            "XML",
            "tag",
        ],
    )
    return has_french and has_reversed and has_xml


# =============================================================================
# PROBLEM LISTS BY TIER
# =============================================================================

INSTRUCTION_EASY_PROBLEMS: List[EvalProblem] = [
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

INSTRUCTION_EXPERT_PROBLEMS: List[EvalProblem] = [
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
            '"The quick brown fox jumps over the lazy dog"\n\n'
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
]
