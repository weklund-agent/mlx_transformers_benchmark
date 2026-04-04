"""EvalProblem dataclass definition.

Separated into its own module to avoid circular imports between
category problem files and eval_problems.py.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class EvalProblem:
    """A single evaluation problem for measuring model quality.

    Fields:
        category: Problem category (coding, reasoning, math, instruction_following,
                  writing, tool_calling)
        name: Unique problem identifier
        prompt: The question/instruction to send to the model
        check: Function that takes model response and returns True/False
        max_tokens: Maximum tokens to allow for generation (default 512)
        function_signature: Optional function signature for code execution problems
        test_cases: Optional list of test cases for code execution problems
        generate_variant: Optional callable that returns a variant EvalProblem
    """

    category: str
    name: str
    prompt: str
    check: Callable[[str], bool]
    max_tokens: int = 512
    function_signature: Optional[str] = None
    test_cases: Optional[list] = field(default=None)
    generate_variant: Optional[Callable] = None
