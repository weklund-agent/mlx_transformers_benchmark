"""Evaluation problems for measuring reasoning quality across quantizations.

Each problem has:
  - category: coding, reasoning, or instruction_following
  - prompt: the question to ask
  - check: a function that takes the model's response and returns True/False
  - max_tokens: how many tokens to allow for generation

The checks are intentionally lenient — we're testing whether quantization
breaks the model's ability to reason, not whether it matches exact formatting.

This module re-exports all problems and check functions from category-specific
modules for backward compatibility. All imports that previously worked from
this module will continue to work.
"""

from typing import Callable, List

# Re-export EvalProblem from its canonical location
from mtb.quality_benchmarks.eval_problem import EvalProblem

# Re-export shared utility functions
from mtb.quality_benchmarks.utils import _contains_any, _extract_number, _strip_thinking

# --- Coding problems and check functions ---
from mtb.quality_benchmarks.coding_problems import (
    CODING_EASY_PROBLEMS,
    CODING_EXPERT_PROBLEMS,
    CODING_HARD_PROBLEMS,
    _check_binary_search,
    _check_buggy_merge_sort,
    _check_calculator,
    _check_data_pipeline,
    _check_fibonacci,
    _check_fizzbuzz,
    _check_flatten_nested,
    _check_longest_palindrome_substring,
    _check_lru_cache,
    _check_markdown_to_html,
    _check_palindrome,
    _check_retry_decorator,
    _check_reverse_string,
)

# --- Reasoning problems and check functions ---
from mtb.quality_benchmarks.reasoning_problems import (
    REASONING_EASY_PROBLEMS,
    REASONING_EXPERT_PROBLEMS,
    REASONING_HARD_PROBLEMS,
    _check_age_problem,
    _check_bayes_theorem,
    _check_circular_seating,
    _check_coin_problem,
    _check_compound_interest,
    _check_einstein_riddle,
    _check_logic_puzzle,
    _check_proof_bug,
    _check_sequence_problem,
    _check_three_urns,
    _check_topological_sort,
    _check_train_problem,
    _check_workers_problem,
)

# --- Instruction following problems and check functions ---
from mtb.quality_benchmarks.instruction_problems import (
    INSTRUCTION_EASY_PROBLEMS,
    INSTRUCTION_EXPERT_PROBLEMS,
    _check_adversarial_transform,
    _check_code_with_comments,
    _check_constrained_factorial,
    _check_json_output,
    _check_library_schema,
    _check_list_format,
    _check_no_thinking,
    _check_word_constraint,
)

# --- Math problems and check functions ---
from mtb.quality_benchmarks.math_problems import (
    MATH_EXPERT_PROBLEMS,
    _check_bouncing_ball,
    _check_inclusion_exclusion,
    _check_modular_arithmetic,
)

# --- Writing problems and check functions ---
from mtb.quality_benchmarks.writing_problems import (
    WRITING_EXPERT_PROBLEMS,
    _check_contradiction_detection,
    _check_multi_doc_summary,
    _check_structured_meeting_notes,
    _check_tone_rewrite,
)

# --- Tool calling problems and check functions ---
# Tool calling problems stay in this file for now (milestone 2 will split them out)
from mtb.quality_benchmarks.utils import _contains_any as __contains_any
from mtb.quality_benchmarks.utils import _strip_thinking as __strip_thinking


def _check_tool_call_weather(response: str) -> bool:
    """Check if response attempts to call a weather tool with a location."""
    response = __strip_thinking(response)
    has_tool_ref = __contains_any(
        response,
        [
            "get_weather",
            "tool_call",
            "tool_use",
            "function_call",
            '"name"',
            "'name'",
        ],
    )
    has_location = __contains_any(
        response,
        [
            "san francisco",
            "sf",
            '"location"',
            "'location'",
        ],
    )
    return has_tool_ref and has_location


def _check_tool_call_calculator(response: str) -> bool:
    """Check if response calls a calculator tool with the correct expression."""
    response = __strip_thinking(response)
    has_tool_ref = __contains_any(
        response,
        [
            "calculate",
            "calculator",
            "tool_call",
            "tool_use",
            "function_call",
            '"name"',
        ],
    )
    has_expression = __contains_any(
        response,
        [
            "347 * 823",
            "347*823",
            "285481",
            "expression",
        ],
    )
    return has_tool_ref and has_expression


def _check_tool_call_multi_step(response: str) -> bool:
    """Check if response uses multiple tools in sequence."""
    response = __strip_thinking(response)
    has_search = __contains_any(
        response,
        [
            "search_web",
            "web_search",
            "search",
        ],
    )
    has_summarize = __contains_any(
        response,
        [
            "summarize",
            "summary",
            "summarize_text",
        ],
    )
    return has_search and has_summarize


def _check_tool_call_json_args(response: str) -> bool:
    """Check if response produces valid JSON-like arguments for a tool call."""
    response = __strip_thinking(response)
    has_json_structure = __contains_any(
        response,
        [
            '{"',
            "{'",
            '"arguments"',
            '"input"',
            '"parameters"',
        ],
    )
    has_tool_name = __contains_any(
        response,
        [
            "create_file",
            "tool_call",
            "tool_use",
            "function_call",
            '"name"',
        ],
    )
    has_filename = __contains_any(
        response,
        [
            "hello.py",
            "hello",
            "filename",
            "path",
        ],
    )
    return has_json_structure and (has_tool_name or has_filename)


def _check_tool_call_selection(response: str) -> bool:
    """Check if model selects the right tool from multiple options."""
    response = __strip_thinking(response)
    has_correct_tool = __contains_any(response, ["send_email"])
    has_recipient = __contains_any(
        response,
        [
            "alice",
            "recipient",
            "to",
        ],
    )
    return has_correct_tool and has_recipient


# =============================================================================
# ASSEMBLED PROBLEM LISTS BY TIER (backward compatible)
# =============================================================================

EVAL_PROBLEMS: List[EvalProblem] = (
    CODING_EASY_PROBLEMS + REASONING_EASY_PROBLEMS + INSTRUCTION_EASY_PROBLEMS
)

HARD_EVAL_PROBLEMS: List[EvalProblem] = CODING_HARD_PROBLEMS + REASONING_HARD_PROBLEMS

EXPERT_EVAL_PROBLEMS: List[EvalProblem] = (
    MATH_EXPERT_PROBLEMS
    + CODING_EXPERT_PROBLEMS
    + REASONING_EXPERT_PROBLEMS
    + INSTRUCTION_EXPERT_PROBLEMS
    + WRITING_EXPERT_PROBLEMS
)

TOOL_CALLING_PROBLEMS: List[EvalProblem] = [
    EvalProblem(
        category="tool_calling",
        name="simple_tool_call",
        prompt=(
            "You have access to the following tool:\n\n"
            "Tool: get_weather\n"
            "Description: Get the current weather for a city\n"
            "Parameters:\n"
            "  - location (string, required): The city name\n\n"
            "Use this tool to get the weather in San Francisco. "
            "Output a tool call with the function name and arguments."
        ),
        check=_check_tool_call_weather,
        max_tokens=256,
    ),
    EvalProblem(
        category="tool_calling",
        name="calculator_tool",
        prompt=(
            "You have access to the following tool:\n\n"
            "Tool: calculate\n"
            "Description: Evaluate a mathematical expression\n"
            "Parameters:\n"
            "  - expression (string, required): The math expression to evaluate\n\n"
            "Use this tool to compute 347 * 823. "
            "Output a tool call with the function name and arguments."
        ),
        check=_check_tool_call_calculator,
        max_tokens=256,
    ),
    EvalProblem(
        category="tool_calling",
        name="tool_with_json_args",
        prompt=(
            "You have access to the following tool:\n\n"
            "```json\n"
            '{"type": "function", "function": {"name": "create_file", '
            '"description": "Create a new file with the given content", '
            '"parameters": {"type": "object", "properties": {'
            '"filename": {"type": "string", "description": "Name of the file"}, '
            '"content": {"type": "string", "description": "File content"}}, '
            '"required": ["filename", "content"]}}}\n'
            "```\n\n"
            "Call this tool to create a file called 'hello.py' containing "
            "'print(\"Hello, World!\")'. Output the tool call as JSON."
        ),
        check=_check_tool_call_json_args,
        max_tokens=512,
    ),
    EvalProblem(
        category="tool_calling",
        name="tool_selection",
        prompt=(
            "You have access to the following tools:\n\n"
            "1. send_email(recipient: str, subject: str, body: str) - Send an email\n"
            "2. create_calendar_event(title: str, date: str, time: str) - Create a calendar event\n"
            "3. set_reminder(message: str, time: str) - Set a reminder\n\n"
            "The user says: 'Send Alice a message about the project deadline tomorrow.'\n\n"
            "Which tool should you call, and with what arguments? "
            "Output the tool call."
        ),
        check=_check_tool_call_selection,
        max_tokens=512,
    ),
    EvalProblem(
        category="tool_calling",
        name="multi_step_tool_use",
        prompt=(
            "You have access to the following tools:\n\n"
            "1. search_web(query: str) - Search the web for information\n"
            "2. summarize_text(text: str) - Summarize a piece of text\n\n"
            "The user wants a summary of the latest news about AI regulation. "
            "Plan which tools to call and in what order. "
            "Output the sequence of tool calls you would make."
        ),
        check=_check_tool_call_multi_step,
        max_tokens=512,
    ),
]
