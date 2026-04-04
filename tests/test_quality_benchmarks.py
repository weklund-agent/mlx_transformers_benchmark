"""Tests for quality benchmark evaluation logic (pure functions only)."""

import pytest

from mtb.quality_benchmarks.eval_problems import (
    EVAL_PROBLEMS,
    EXPERT_EVAL_PROBLEMS,
    HARD_EVAL_PROBLEMS,
    TOOL_CALLING_PROBLEMS,
    EvalProblem,
    _contains_any,
    _strip_thinking,
    _check_fizzbuzz,
    _check_reverse_string,
    _check_fibonacci,
    _check_binary_search,
    _check_palindrome,
    _check_train_problem,
    _check_coin_problem,
    _check_workers_problem,
    _check_age_problem,
    _check_sequence_problem,
    _check_json_output,
    _check_list_format,
    _check_word_constraint,
    _check_code_with_comments,
    _check_no_thinking,
    _check_lru_cache,
    _check_flatten_nested,
    _check_longest_palindrome_substring,
    _check_calculator,
    _check_buggy_merge_sort,
    _check_compound_interest,
    _check_circular_seating,
    _check_logic_puzzle,
    _check_bayes_theorem,
    _check_proof_bug,
    _check_modular_arithmetic,
    _check_inclusion_exclusion,
    _check_bouncing_ball,
    _check_markdown_to_html,
    _check_data_pipeline,
    _check_retry_decorator,
    _check_einstein_riddle,
    _check_three_urns,
    _check_topological_sort,
    _check_constrained_factorial,
    _check_library_schema,
    _check_adversarial_transform,
    _check_multi_doc_summary,
    _check_structured_meeting_notes,
    _check_tone_rewrite,
    _check_contradiction_detection,
    _check_tool_call_weather,
    _check_tool_call_calculator,
    _check_tool_call_multi_step,
    _check_tool_call_json_args,
    _check_tool_call_selection,
)


# ---------------------------------------------------------------------------
# _strip_thinking
# ---------------------------------------------------------------------------


class TestStripThinking:
    def test_strips_closed_think_block(self):
        response = "<think>Let me reason about this.</think>The answer is 42."
        assert _strip_thinking(response) == "The answer is 42."

    def test_strips_multiline_closed_think_block(self):
        response = (
            "<think>\nStep 1: Consider X\nStep 2: Therefore Y\n</think>\n"
            "The final answer is Paris."
        )
        assert "think" not in _strip_thinking(response).lower()
        assert "Paris" in _strip_thinking(response)

    def test_strips_unclosed_think_block(self):
        response = "<think>I'm still thinking and the response was truncated"
        assert _strip_thinking(response) == ""

    def test_strips_unclosed_think_block_with_prior_content(self):
        response = "Preamble.\n<think>Still going on and on..."
        result = _strip_thinking(response)
        assert "Preamble." in result
        assert "think" not in result.lower()

    def test_no_think_blocks_unchanged(self):
        response = "Just a plain answer with no thinking tags."
        assert _strip_thinking(response) == response

    def test_multiple_think_blocks(self):
        response = (
            "<think>First thought</think>Answer part 1. "
            "<think>Second thought</think>Answer part 2."
        )
        result = _strip_thinking(response)
        assert "First thought" not in result
        assert "Second thought" not in result
        assert "Answer part 1." in result
        assert "Answer part 2." in result

    def test_empty_string(self):
        assert _strip_thinking("") == ""

    def test_empty_think_block(self):
        response = "<think></think>Result."
        assert _strip_thinking(response) == "Result."

    def test_strips_freeform_thinking_preamble_with_answer_header(self):
        response = (
            "Thinking Process:\n\n"
            "1. **Analyze the Request:**\n"
            "   The user wants a rewrite using backpropagation terms.\n\n"
            "2. **Plan:** Simplify the jargon.\n\n"
            "**Rewrite:**\n"
            "The AI learns from data like a brain learns from practice."
        )
        result = _strip_thinking(response)
        assert "backpropagation" not in result
        assert "AI learns" in result

    def test_strips_freeform_thinking_with_solution_header(self):
        response = (
            "Here's a thinking process that leads to the solution:\n\n"
            "1. Parse the CSV data.\n"
            "2. Filter rows.\n\n"
            "**Solution:**\n"
            "```python\ndef process_sales(csv_text):\n    pass\n```"
        )
        result = _strip_thinking(response)
        assert "Parse the CSV" not in result
        assert "process_sales" in result

    def test_strips_freeform_thinking_uses_last_answer_header(self):
        """When multiple draft headers exist, use the last one (final answer)."""
        response = (
            "Thinking Process:\n\n"
            "1. Analyze jargon like backpropagation.\n\n"
            "*Draft:*\n"
            "First attempt with gradient descent mentioned.\n\n"
            "*Revised Draft:*\n"
            "The AI system learned by adjusting its settings to reduce mistakes."
        )
        result = _strip_thinking(response)
        assert "backpropagation" not in result
        assert "gradient descent" not in result
        assert "adjusting its settings" in result

    def test_freeform_thinking_without_answer_header_unchanged(self):
        """If no answer header is found, the response is returned as-is."""
        response = (
            "Thinking Process:\n\n"
            "1. Do step one.\n"
            "2. Do step two.\n\n"
            "The answer is 42."
        )
        result = _strip_thinking(response)
        # No recognized answer header, so full text remains
        assert "answer is 42" in result


# ---------------------------------------------------------------------------
# _contains_any
# ---------------------------------------------------------------------------


class TestContainsAny:
    def test_matches_present_target(self):
        assert _contains_any("hello world", ["hello"]) is True

    def test_no_match(self):
        assert _contains_any("hello world", ["foo", "bar"]) is False

    def test_case_insensitive(self):
        assert _contains_any("Hello World", ["hello"]) is True
        assert _contains_any("hello world", ["HELLO"]) is True

    def test_multiple_targets_any_match(self):
        assert _contains_any("the answer is 42", ["99", "42"]) is True

    def test_empty_targets(self):
        assert _contains_any("anything", []) is False

    def test_empty_response(self):
        assert _contains_any("", ["something"]) is False


# ---------------------------------------------------------------------------
# Check functions – passing responses
# ---------------------------------------------------------------------------


class TestCheckFunctionsPass:
    """Each check function should return True for a realistic correct response."""

    def test_fizzbuzz_pass(self):
        response = (
            "def fizzbuzz():\n"
            "    for i in range(1, 101):\n"
            "        if i % 15 == 0:\n"
            "            print('FizzBuzz')\n"
            "        elif i % 3 == 0:\n"
            "            print('Fizz')\n"
            "        elif i % 5 == 0:\n"
            "            print('Buzz')\n"
        )
        assert _check_fizzbuzz(response) is True

    def test_reverse_string_pass(self):
        response = (
            "Approach 1: return s[::-1]\n" "Approach 2: return ''.join(reversed(s))"
        )
        assert _check_reverse_string(response) is True

    def test_fibonacci_pass(self):
        response = (
            "def fibonacci(n):\n"
            "    ...\n"
            "Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]"
        )
        assert _check_fibonacci(response) is True

    def test_binary_search_pass(self):
        response = (
            "def binary_search(arr, target):\n"
            "    low, high = 0, len(arr) - 1\n"
            "    while low <= high:\n"
            "        mid = (low + high) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
        )
        assert _check_binary_search(response) is True

    def test_palindrome_pass(self):
        response = (
            "def is_palindrome(s):\n"
            "    s = ''.join(c.lower() for c in s if c.isalnum())\n"
            "    return s == s[::-1]"
        )
        assert _check_palindrome(response) is True

    def test_train_problem_pass(self):
        response = "The combined speed is 100 km/h. 200/100 = 2 hours."
        assert _check_train_problem(response) is True

    def test_coin_problem_pass(self):
        response = "P(at least 2 heads) = 4/8 = 0.5 or 50%."
        assert _check_coin_problem(response) is True

    def test_workers_problem_pass(self):
        response = "Work = 5*10 = 50 worker-days. 50/10 = 5 days."
        assert _check_workers_problem(response) is True

    def test_age_problem_pass(self):
        response = "Jerry is 5 years old and Tom is 10 years old."
        assert _check_age_problem(response) is True

    def test_sequence_problem_pass(self):
        response = "The pattern is n*(n+1). Next is 6*7 = 42."
        assert _check_sequence_problem(response) is True

    def test_json_output_pass(self):
        response = '{"name": "Alice", "age": 30, "hobbies": ["reading"]}'
        assert _check_json_output(response) is True

    def test_list_format_pass(self):
        response = (
            "1. Improves cardiovascular health\n"
            "2. Reduces stress\n"
            "3. Builds muscle strength\n"
            "4. Enhances flexibility\n"
            "5. Boosts mood"
        )
        assert _check_list_format(response) is True

    def test_word_constraint_pass(self):
        response = (
            "Machine learning is a subset of AI that enables computers "
            "to learn from data and improve without explicit programming."
        )
        assert _check_word_constraint(response) is True

    def test_code_with_comments_pass(self):
        response = (
            "import math\n\n"
            "def area_of_circle(radius):\n"
            '    """Calculate the area of a circle."""\n'
            "    # Use the formula: A = pi * r^2\n"
            "    return math.pi * radius ** 2\n"
        )
        assert _check_code_with_comments(response) is True

    def test_no_thinking_pass(self):
        response = "Paris."
        assert _check_no_thinking(response) is True


# ---------------------------------------------------------------------------
# Check functions – failing responses
# ---------------------------------------------------------------------------


class TestCheckFunctionsFail:
    """Each check function should return False for a clearly wrong response."""

    def test_fizzbuzz_fail(self):
        response = "Here is a hello world program:\nprint('hello world')"
        assert _check_fizzbuzz(response) is False

    def test_reverse_string_fail(self):
        response = "To sort a list, use sorted()."
        assert _check_reverse_string(response) is False

    def test_fibonacci_fail(self):
        response = "The Fibonacci sequence is 1, 2, 3, 4, 5, 6, 7, 8, 9, 10."
        assert _check_fibonacci(response) is False

    def test_binary_search_fail(self):
        response = "def linear_search(arr, target):\n    for item in arr:\n        if item == target: return True"
        assert _check_binary_search(response) is False

    def test_palindrome_fail(self):
        response = "A palindrome is a word that reads the same forwards and backwards, like racecar."
        assert _check_palindrome(response) is False

    def test_train_problem_fail(self):
        response = "They will meet in 10 hours."
        assert _check_train_problem(response) is False

    def test_coin_problem_fail(self):
        response = "The probability is 0.75 or 75%."
        assert _check_coin_problem(response) is False

    def test_workers_problem_fail(self):
        response = "It would take 20 days for 10 workers."
        assert _check_workers_problem(response) is False

    def test_age_problem_fail(self):
        response = "Tom is 20 and Jerry is 15."
        assert _check_age_problem(response) is False

    def test_sequence_problem_fail(self):
        response = "The next number is 36."
        assert _check_sequence_problem(response) is False

    def test_json_output_fail(self):
        response = "Name: Alice, Age: 30, Hobbies: reading"
        assert _check_json_output(response) is False

    def test_list_format_fail(self):
        response = (
            "Exercise is great because it helps your heart, "
            "reduces stress, and builds muscle."
        )
        assert _check_list_format(response) is False

    def test_word_constraint_fail(self):
        response = "The sky is blue because of Rayleigh scattering."
        assert _check_word_constraint(response) is False

    def test_code_with_comments_fail(self):
        # Code but no comments
        response = "def area(r):\n    return 3.14 * r * r"
        assert _check_code_with_comments(response) is False

    def test_code_with_comments_fail_no_code(self):
        # Comments but no code keywords
        response = "# This is just a comment with no actual code"
        assert _check_code_with_comments(response) is False

    def test_no_thinking_fail(self):
        response = "The capital of Germany is Berlin."
        assert _check_no_thinking(response) is False


# ---------------------------------------------------------------------------
# Check functions – think block stripping
# ---------------------------------------------------------------------------


class TestCheckFunctionsWithThinkBlocks:
    """Verify that check functions strip <think> blocks before evaluation."""

    def test_fizzbuzz_with_think_block(self):
        response = (
            "<think>I need to write FizzBuzz. Let me use modulo.</think>"
            "def fizzbuzz():\n"
            "    for i in range(1, 101):\n"
            "        if i % 15 == 0:\n"
            "            print('FizzBuzz')\n"
        )
        assert _check_fizzbuzz(response) is True

    def test_train_problem_with_think_block(self):
        response = (
            "<think>Combined speed = 60 + 40 = 100 km/h.\n"
            "Time = 200/100 = 2 hours.</think>\n"
            "The trains will meet in 2 hours."
        )
        assert _check_train_problem(response) is True

    def test_answer_only_in_think_block_fails(self):
        """If the answer only appears inside <think>, the check should fail."""
        response = (
            "<think>The answer is 42, which is 6*7.</think>\n"
            "The next number in the sequence follows the pattern."
        )
        assert _check_sequence_problem(response) is False

    def test_json_output_with_think_block(self):
        response = (
            "<think>I need to produce valid JSON.</think>\n"
            '{"name": "Bob", "age": 25, "hobbies": ["hiking"]}'
        )
        assert _check_json_output(response) is True

    def test_no_thinking_with_think_block(self):
        response = "<think>The user wants the capital of France.</think>" "Paris"
        assert _check_no_thinking(response) is True


# ---------------------------------------------------------------------------
# Check functions – Qwen-style "Thinking Process:" preamble
# ---------------------------------------------------------------------------


class TestCheckFunctionsWithThinkingPreamble:
    """Qwen 3.5 and similar models emit a freeform 'Thinking Process:' preamble.

    _strip_thinking removes these preambles when a recognized answer header is
    found. For checks that search the full response text, the preamble keywords
    still work even if not stripped.
    """

    def test_fizzbuzz_with_thinking_preamble(self):
        response = (
            "Thinking Process:\n"
            "The user wants FizzBuzz. I need to use modulo operators.\n\n"
            "def fizzbuzz():\n"
            "    for i in range(1, 101):\n"
            "        if i % 15 == 0:\n"
            "            print('FizzBuzz')\n"
        )
        assert _check_fizzbuzz(response) is True

    def test_coin_problem_with_thinking_preamble(self):
        response = (
            "Thinking Process:\n"
            "3 coin flips, sample space = 8.\n"
            "Favorable outcomes: HHH, HHT, HTH, THH = 4.\n\n"
            "The probability is 4/8 = 0.5 or 50%."
        )
        assert _check_coin_problem(response) is True

    def test_sequence_with_thinking_preamble(self):
        response = (
            "Thinking Process:\n"
            "Pattern: n(n+1). 1*2=2, 2*3=6, ..., 6*7=42\n\n"
            "The next number is 42."
        )
        assert _check_sequence_problem(response) is True

    def test_word_constraint_with_thinking_preamble(self):
        response = (
            "Thinking Process:\n"
            "I need to explain ML concisely.\n\n"
            "Machine learning is a subset of AI where algorithms "
            "learn from data to make predictions."
        )
        assert _check_word_constraint(response) is True


# ---------------------------------------------------------------------------
# EvalProblem dataclass and EVAL_PROBLEMS list
# ---------------------------------------------------------------------------


class TestEvalProblems:
    def test_eval_problems_count(self):
        assert len(EVAL_PROBLEMS) == 15

    def test_five_per_category(self):
        categories = {}
        for p in EVAL_PROBLEMS:
            categories.setdefault(p.category, 0)
            categories[p.category] += 1

        assert categories == {
            "coding": 5,
            "reasoning": 5,
            "instruction_following": 5,
        }

    def test_all_checks_are_callable(self):
        for p in EVAL_PROBLEMS:
            assert callable(p.check), f"{p.name} check is not callable"

    def test_all_have_nonempty_prompts(self):
        for p in EVAL_PROBLEMS:
            assert len(p.prompt) > 0, f"{p.name} has empty prompt"

    def test_all_have_nonempty_names(self):
        for p in EVAL_PROBLEMS:
            assert len(p.name) > 0, f"Problem has empty name"

    def test_unique_names(self):
        names = [p.name for p in EVAL_PROBLEMS]
        assert len(names) == len(set(names)), "Duplicate problem names found"

    def test_default_max_tokens(self):
        """EvalProblem defaults to max_tokens=512."""
        p = EvalProblem(
            category="test",
            name="test_problem",
            prompt="test",
            check=lambda r: True,
        )
        assert p.max_tokens == 512


# ---------------------------------------------------------------------------
# Additional edge cases for specific check functions
# ---------------------------------------------------------------------------


class TestCheckFunctionEdgeCases:
    def test_fizzbuzz_passes_with_mod_and_15(self):
        """FizzBuzz check has an OR: fizzbuzz keyword or (mod + 15)."""
        response = "Use modulo operator: if n % 15 == 0 print both"
        assert _check_fizzbuzz(response) is True

    def test_binary_search_needs_both_mid_and_halving(self):
        """Binary search requires both 'mid' AND a halving indicator."""
        assert _check_binary_search("mid") is False
        assert _check_binary_search("low high // 2") is False
        assert _check_binary_search("mid = (low + high) // 2") is True

    def test_list_format_needs_at_least_3_numbered_items(self):
        response = "1. First\n2. Second"
        assert _check_list_format(response) is False

        response = "1. First\n2. Second\n3. Third"
        assert _check_list_format(response) is True

    def test_json_output_needs_braces_and_name_key(self):
        # Has braces but no "name" key
        response = '{"age": 30}'
        assert _check_json_output(response) is False

    def test_age_problem_alternate_formats(self):
        assert _check_age_problem("Tom: 10, Jerry: 5") is True
        assert _check_age_problem("Jerry = 5, Tom = 10") is True

    def test_train_problem_alternate_formats(self):
        assert _check_train_problem("They meet in two hours.") is True
        assert _check_train_problem("The answer is **2** hours.") is True

    def test_workers_problem_alternate_formats(self):
        assert _check_workers_problem("It takes five days.") is True

    def test_coin_problem_alternate_formats(self):
        assert _check_coin_problem("The probability is 1/2.") is True
        assert _check_coin_problem("50 percent chance.") is True

    def test_palindrome_two_pointer(self):
        response = "Use a two pointer approach with left and right indices."
        assert _check_palindrome(response) is True

    def test_reverse_string_javascript(self):
        response = "return str.split('').reverse().join('');"
        assert _check_reverse_string(response) is True

    def test_fibonacci_comma_no_spaces(self):
        response = "Output: [0,1,1,2,3,5,8,13,21,34]"
        assert _check_fibonacci(response) is True

    def test_word_constraint_various_phrasings(self):
        assert (
            _check_word_constraint("ML uses algorithms to find patterns in data.")
            is True
        )
        assert (
            _check_word_constraint("It is a subset of artificial intelligence.") is True
        )

    def test_sequence_bold_markdown(self):
        assert _check_sequence_problem("The answer is **42**.") is True


# ---------------------------------------------------------------------------
# Hard problem structure validation
# ---------------------------------------------------------------------------


class TestHardEvalProblems:
    def test_hard_eval_problems_count(self):
        assert len(HARD_EVAL_PROBLEMS) == 10

    def test_five_coding_five_reasoning(self):
        categories = [p.category for p in HARD_EVAL_PROBLEMS]
        assert categories.count("coding") == 5
        assert categories.count("reasoning") == 5

    def test_all_checks_are_callable(self):
        for problem in HARD_EVAL_PROBLEMS:
            assert callable(problem.check), f"{problem.name} check is not callable"

    def test_unique_names(self):
        names = [p.name for p in HARD_EVAL_PROBLEMS]
        assert len(names) == len(set(names))

    def test_no_name_overlap_with_easy(self):
        easy_names = {p.name for p in EVAL_PROBLEMS}
        hard_names = {p.name for p in HARD_EVAL_PROBLEMS}
        overlap = easy_names & hard_names
        assert len(overlap) == 0, f"Overlapping names: {overlap}"

    def test_hard_problems_have_sufficient_max_tokens(self):
        for problem in HARD_EVAL_PROBLEMS:
            assert problem.max_tokens >= 1024, (
                f"{problem.name} has max_tokens={problem.max_tokens}, "
                "hard problems need >= 1024 for thinking models"
            )


# ---------------------------------------------------------------------------
# Hard check functions - passing cases
# ---------------------------------------------------------------------------


class TestHardCheckFunctionsPass:
    def test_lru_cache_pass(self):
        response = (
            "class LRUCache:\n"
            "    def __init__(self, capacity):\n"
            "        self.capacity = capacity\n"
            "        self.cache = {}\n"
            "        self.head = Node()\n"
            "        self.tail = Node()\n"
            "    def get(self, key):\n"
            "        if key in self.cache:\n"
            "            self._move_to_end(node)\n"
            "            return node.value\n"
            "        return -1\n"
            "    def put(self, key, value):\n"
            "        if len(self.cache) >= self.capacity:\n"
            "            lru = self.head.next\n"
            "            self._remove(lru)\n"
            "            del self.cache[lru.key]\n"
        )
        assert _check_lru_cache(response) is True

    def test_lru_cache_with_ordereddict_mention(self):
        response = (
            "Instead of OrderedDict, we use a doubly linked list.\n"
            "def get(self, key): ...\n"
            "def put(self, key, value): ...\n"
            "We evict the least recently used item with pop.\n"
        )
        assert _check_lru_cache(response) is True

    def test_flatten_nested_pass(self):
        response = (
            "def flatten(lst):\n"
            "    result = []\n"
            "    for item in lst:\n"
            "        if isinstance(item, list):\n"
            "            result.extend(flatten(item))\n"
            "        else:\n"
            "            result.append(item)\n"
            "    return result\n"
        )
        assert _check_flatten_nested(response) is True

    def test_longest_palindrome_substring_pass(self):
        response = (
            "def longest_palindrome(s):\n"
            "    result = ''\n"
            "    for i in range(len(s)):\n"
            "        # expand around center for odd length\n"
            "        sub = expand(s, i, i)\n"
            "        if len(sub) > max_length:\n"
            "            result = sub\n"
            "    # Test: 'babad' -> 'bab'\n"
            "    return result\n"
        )
        assert _check_longest_palindrome_substring(response) is True

    def test_calculator_pass(self):
        response = (
            "def calculate(expression):\n"
            "    tokens = tokenize(expression)\n"
            "    stack = []\n"
            "    # Handle operator precedence: * and / before + and -\n"
            "    # Handle parentheses ( ) by recursive descent\n"
            "    return evaluate(tokens)\n"
        )
        assert _check_calculator(response) is True

    def test_buggy_merge_sort_pass(self):
        response = (
            "The bug is that the merge function doesn't append remaining elements. "
            "After the while loop, any leftover elements in left or right are lost. "
            "The fix: add `result.extend(left[i:])` and `result.extend(right[j:])` "
            "after the while loop. This is an off-by-one boundary issue where the "
            "remaining elements should be appended to the result."
        )
        assert _check_buggy_merge_sort(response) is True

    def test_compound_interest_pass(self):
        response = (
            "A = P(1 + r/n)^(nt)\n"
            "A = 10000(1 + 0.05/4)^(4*3)\n"
            "A = 10000(1.0125)^12\n"
            "A = 10000 * 1.16075...\n"
            "A = $11,607.55"
        )
        assert _check_compound_interest(response) is True

    def test_circular_seating_pass(self):
        response = (
            "Total circular permutations = (6-1)! = 120\n"
            "Ways Alice and Bob ARE adjacent = 2 * (5-1)! = 48\n"
            "Answer = 120 - 48 = 72"
        )
        assert _check_circular_seating(response) is True

    def test_logic_puzzle_pass(self):
        response = (
            "Testing all cases:\n"
            "If A is truthful, B is a liar, so C is truthful. "
            "But C says A and B are both liars - contradiction.\n"
            "If B is truthful, C is a liar. A must be a liar (otherwise B is liar). "
            "C's claim 'A and B are both liars' is false (B isn't), consistent with C being a liar.\n"
            "Therefore B is truthful, A and C are liars."
        )
        assert _check_logic_puzzle(response) is True

    def test_logic_puzzle_pass_truth_teller_phrasing(self):
        response = (
            "## Summary\n"
            "| Person | Type |\n"
            "| A | **Liar** |\n"
            "| B | **Truth-teller** |\n"
            "| C | **Liar** |\n"
            "B is a truth-teller. A is a liar and C is a liar."
        )
        assert _check_logic_puzzle(response) is True

    def test_bayes_theorem_pass(self):
        response = (
            "Using Bayes' theorem:\n"
            "P(D|+) = P(+|D)*P(D) / [P(+|D)*P(D) + P(+|~D)*P(~D)]\n"
            "= (0.95 * 0.01) / (0.95 * 0.01 + 0.05 * 0.99)\n"
            "= 0.0095 / 0.059\n"
            "≈ 0.161 or about 16.1%"
        )
        assert _check_bayes_theorem(response) is True

    def test_proof_bug_pass(self):
        response = (
            "The error is in step 4 where we divide both sides by (a-b). "
            "Since a = b, we have a - b = 0, so this is division by zero, "
            "which is undefined and invalidates the proof."
        )
        assert _check_proof_bug(response) is True


# ---------------------------------------------------------------------------
# Hard check functions - failing cases
# ---------------------------------------------------------------------------


class TestHardCheckFunctionsFail:
    def test_lru_cache_fail(self):
        assert _check_lru_cache("Here's a simple dictionary cache.") is False

    def test_flatten_nested_fail(self):
        assert _check_flatten_nested("Just use a for loop to iterate.") is False

    def test_longest_palindrome_substring_fail(self):
        assert (
            _check_longest_palindrome_substring("Check if a string is a palindrome.")
            is False
        )

    def test_calculator_fail(self):
        assert _check_calculator("Use eval() to evaluate the expression.") is False

    def test_buggy_merge_sort_fail(self):
        assert _check_buggy_merge_sort("The code looks correct to me.") is False

    def test_compound_interest_fail_simple_interest(self):
        # Common mistake: using simple interest instead of compound
        # 10000 * 0.05 * 3 = 1500, total = 11500
        assert _check_compound_interest("The final amount is $11,500.") is False

    def test_circular_seating_fail(self):
        # Common mistake: just doing 6! = 720
        assert _check_circular_seating("The answer is 720.") is False

    def test_logic_puzzle_fail_says_a_truthful(self):
        assert _check_logic_puzzle("A is truthful, B and C are liars.") is False

    def test_bayes_theorem_fail_says_95(self):
        # The classic mistake: confusing test accuracy with posterior
        assert _check_bayes_theorem("The probability is 95%.") is False

    def test_proof_bug_fail_generic(self):
        assert _check_proof_bug("The algebra is wrong somewhere.") is False


# ---------------------------------------------------------------------------
# Hard check functions - with think blocks
# ---------------------------------------------------------------------------


class TestHardCheckFunctionsWithThinkBlocks:
    def test_lru_cache_with_think_block(self):
        response = (
            "<think>I need to implement LRU cache with O(1) operations...</think>\n"
            "class LRUCache:\n"
            "    def __init__(self, capacity):\n"
            "        self.cache = {}\n"
            "        self.capacity = capacity\n"
            "    def get(self, key): return self.cache.get(key)\n"
            "    def put(self, key, val):\n"
            "        if len(self.cache) >= self.capacity:\n"
            "            self.cache.pop(next(iter(self.cache)))\n"
        )
        assert _check_lru_cache(response) is True

    def test_bayes_with_think_block(self):
        response = (
            "<think>Let me apply Bayes' theorem step by step...</think>\n"
            "The probability is approximately 16.1%."
        )
        assert _check_bayes_theorem(response) is True

    def test_proof_bug_with_think_block(self):
        response = (
            "<think>Let me trace through each step...</think>\n"
            "The error is division by zero since a - b = 0."
        )
        assert _check_proof_bug(response) is True


# ===========================================================================
# Expert problem structure validation
# ===========================================================================


class TestExpertEvalProblems:
    def test_expert_eval_problems_count(self):
        assert len(EXPERT_EVAL_PROBLEMS) == 16

    def test_category_distribution(self):
        categories = {}
        for p in EXPERT_EVAL_PROBLEMS:
            categories.setdefault(p.category, 0)
            categories[p.category] += 1
        assert categories == {
            "math": 3,
            "coding": 3,
            "reasoning": 3,
            "instruction_following": 3,
            "writing": 4,
        }

    def test_all_checks_are_callable(self):
        for problem in EXPERT_EVAL_PROBLEMS:
            assert callable(problem.check), f"{problem.name} check is not callable"

    def test_unique_names(self):
        names = [p.name for p in EXPERT_EVAL_PROBLEMS]
        assert len(names) == len(set(names))

    def test_no_name_overlap_with_easy_or_hard(self):
        easy_names = {p.name for p in EVAL_PROBLEMS}
        hard_names = {p.name for p in HARD_EVAL_PROBLEMS}
        expert_names = {p.name for p in EXPERT_EVAL_PROBLEMS}
        overlap = expert_names & (easy_names | hard_names)
        assert len(overlap) == 0, f"Overlapping names: {overlap}"

    def test_expert_problems_have_sufficient_max_tokens(self):
        for problem in EXPERT_EVAL_PROBLEMS:
            assert problem.max_tokens >= 1024, (
                f"{problem.name} has max_tokens={problem.max_tokens}, "
                "expert problems need >= 1024 for thinking models"
            )


# ===========================================================================
# Expert check functions - passing cases
# ===========================================================================


class TestExpertCheckFunctionsPass:
    # --- Math ---
    def test_modular_arithmetic_pass(self):
        response = (
            "Powers of 2 mod 7: 2^1=2, 2^2=4, 2^3=1, 2^4=2, ... cycle of length 3.\n"
            "100 mod 3 = 1, so 2^100 ≡ 2 (mod 7).\n"
            "The remainder is 2."
        )
        assert _check_modular_arithmetic(response) is True

    def test_modular_arithmetic_pass_congruence_notation(self):
        response = "Since the pattern repeats every 3, and 100 = 33*3 + 1, we get 2^100 ≡ 2 (mod 7)."
        assert _check_modular_arithmetic(response) is True

    def test_inclusion_exclusion_pass(self):
        response = (
            "|A| = 500, |B| = 333, |C| = 200\n"
            "|A∩B| = 166, |A∩C| = 100, |B∩C| = 66\n"
            "|A∩B∩C| = 33\n"
            "Answer = 500 + 333 + 200 - 166 - 100 - 66 + 33 = 734"
        )
        assert _check_inclusion_exclusion(response) is True

    def test_bouncing_ball_pass(self):
        response = (
            "Down series: 10 + 7.5 + 5.625 + ... = 10/(1-3/4) = 40\n"
            "Up series: 7.5 + 5.625 + ... = 7.5/(1-3/4) = 30\n"
            "Total distance = 40 + 30 = 70 meters."
        )
        assert _check_bouncing_ball(response) is True

    # --- Agentic Coding ---
    def test_markdown_to_html_pass(self):
        response = (
            "def markdown_to_html(text):\n"
            "    # Handle bold: **text** -> <strong>text</strong>\n"
            "    text = re.sub(r'\\*\\*(.+?)\\*\\*', r'<strong>\\1</strong>', text)\n"
            "    # Handle italic: *text* -> <em>text</em>\n"
            "    text = re.sub(r'\\*(.+?)\\*', r'<em>\\1</em>', text)\n"
            "    # Handle code: `text` -> <code>text</code>\n"
            "    text = re.sub(r'`(.+?)`', r'<code>\\1</code>', text)\n"
            "    # Handle links: [text](url) -> <a href='url'>text</a>\n"
            "    text = re.sub(r'\\[(.+?)\\]\\((.+?)\\)', r'<a href=\"\\2\">\\1</a>', text)\n"
            "    return text\n"
        )
        assert _check_markdown_to_html(response) is True

    def test_data_pipeline_pass(self):
        response = (
            "import csv, json\n"
            "def process_sales(csv_text):\n"
            "    reader = csv.DictReader(csv_text.splitlines())\n"
            "    # Filter rows\n"
            "    filtered = [r for r in reader if float(r['amount']) > 100 "
            "and r['region'] in ('North', 'West') and int(r['quantity']) >= 5]\n"
            "    # Group by product\n"
            "    groups = defaultdict(list)\n"
            "    for row in filtered:\n"
            "        groups[row['product']].append(float(row['amount']))\n"
            "    # Aggregate\n"
            "    result = []\n"
            "    for product, amounts in groups.items():\n"
            "        result.append({'product': product, 'total_amount': sum(amounts), "
            "'avg_amount': sum(amounts)/len(amounts), 'count': len(amounts)})\n"
            "    # Sort descending\n"
            "    result = sorted(result, key=lambda x: x['total_amount'], reverse=True)\n"
            "    return json.dumps(result)\n"
        )
        assert _check_data_pipeline(response) is True

    def test_retry_decorator_pass(self):
        response = (
            "from dataclasses import dataclass\n"
            "import time, functools\n\n"
            "@dataclass\n"
            "class Result:\n"
            "    success: bool\n"
            "    value: any\n"
            "    attempts: int\n"
            "    error: Exception = None\n\n"
            "def retry(max_attempts=3, backoff_base=2, exceptions=(Exception,)):\n"
            "    def decorator(func):\n"
            "        @functools.wraps(func)\n"
            "        def wrapper(*args, **kwargs):\n"
            "            for attempt in range(max_attempts):\n"
            "                try:\n"
            "                    value = func(*args, **kwargs)\n"
            "                    return Result(success=True, value=value, attempts=attempt+1)\n"
            "                except exceptions as e:\n"
            "                    print(f'Attempt {attempt+1} failed: {e}')\n"
            "                    time.sleep(backoff_base ** attempt)\n"
            "            return Result(success=False, value=None, attempts=max_attempts, error=e)\n"
            "        return wrapper\n"
            "    return decorator\n"
        )
        assert _check_retry_decorator(response) is True

    # --- Long-Chain Reasoning ---
    def test_einstein_riddle_pass(self):
        response = (
            "From constraint 6: Norwegian lives in house 1.\n"
            "From constraint 7: Norwegian is next to blue, so house 2 is blue.\n"
            "From constraint 8: German lives in house 3.\n"
            "From constraint 1: Brit lives in red house.\n"
            "From constraint 3: Green is left of white.\n"
            "House 1: Norwegian, yellow, water\n"
            "House 2: Dane, blue, tea\n"
            "House 3: German, red, milk (wait, Swede drinks milk...)\n"
            "Let me reconsider...\n"
            "House 3: German, green, coffee (from constraint 5)\n"
            "House 4: white (from constraint 3, green=3 so white=4)\n"
        )
        assert _check_einstein_riddle(response) is True

    def test_three_urns_pass(self):
        response = (
            "P(red from A) = 3/5, then draw from B: P(red from B) = 1/5\n"
            "P(blue from A) = 2/5, then draw from C: P(red from C) = 4/5\n"
            "P(2nd red) = (3/5)(1/5) + (2/5)(4/5) = 3/25 + 8/25 = 11/25 = 0.44"
        )
        assert _check_three_urns(response) is True

    def test_topological_sort_pass(self):
        response = (
            "In-degrees: A=0, B=1, D=1, C=2, E=1\n"
            "Step 1: Only A has in-degree 0. Pick A. Update: B=0, D=0\n"
            "Step 2: Both B and D have in-degree 0.\n"
            "  Path 1: Pick B, then D, then C, then E → A, B, D, C, E\n"
            "  Path 2: Pick D, then B, then C, then E → A, D, B, C, E\n"
            "There are exactly 2 valid topological orderings."
        )
        assert _check_topological_sort(response) is True

    # --- Instruction Following ---
    def test_constrained_factorial_pass(self):
        response = (
            "Here's the function with all 4 constraints:\n\n"
            "Every variable name is exactly 4 characters long.\n\n"
            "```python\n"
            "def fact(numb: int) -> int:\n"
            "    rslt: int = 1\n"
            "    if numb <= 1:\n"
            "        return rslt\n"
            "    return numb * fact(numb - 1)\n"
            "```\n"
        )
        assert _check_constrained_factorial(response) is True

    def test_library_schema_pass(self):
        response = (
            "{\n"
            '  "$schema": "https://json-schema.org/draft/2020-12/schema",\n'
            '  "definitions": {\n'
            '    "Book": {\n'
            '      "type": "object",\n'
            '      "properties": {\n'
            '        "id": {"type": "string"},\n'
            '        "title": {"type": "string"},\n'
            '        "author_id": {"type": "string"}\n'
            "      },\n"
            '      "required": ["id", "title", "author_id"]\n'
            "    },\n"
            '    "Author": {\n'
            '      "type": "object",\n'
            '      "properties": {\n'
            '        "id": {"type": "string"},\n'
            '        "name": {"type": "string"},\n'
            '        "book_ids": {"type": "array"}\n'
            "      },\n"
            '      "required": ["id", "name"]\n'
            "    },\n"
            '    "Loan": {\n'
            '      "type": "object",\n'
            '      "properties": {\n'
            '        "id": {"type": "string"},\n'
            '        "book_id": {"type": "string"},\n'
            '        "returned": {"type": "boolean"}\n'
            "      },\n"
            '      "required": ["id", "book_id"]\n'
            "    }\n"
            "  }\n"
            "}\n"
        )
        assert _check_library_schema(response) is True

    def test_adversarial_transform_pass(self):
        response = (
            "Step 1 (French): Le renard brun rapide saute par-dessus le chien paresseux\n\n"
            "Step 2 (Reverse each word): eL dranre nurb edipar etuase ...\n\n"
            'Step 3 (XML): <word original="Le">eL</word> '
            '<word original="renard">dranre</word> ...\n'
        )
        assert _check_adversarial_transform(response) is True

    # --- Writing ---
    def test_multi_doc_summary_pass(self):
        response = (
            "Three major scientific frontiers are advancing rapidly. Quantum computing "
            "harnesses superposition and entanglement to solve problems beyond classical "
            "computers' reach. Meanwhile, climate change continues to accelerate with "
            "global temperatures rising 1.2°C and carbon emissions still increasing. "
            "In medicine, gene therapy using CRISPR shows promise for diseases like "
            "sickle cell, though cost and off-target DNA edits remain challenges."
        )
        assert _check_multi_doc_summary(response) is True

    def test_structured_meeting_notes_pass(self):
        response = (
            "---\n"
            "date: 2026-03-07\n"
            "attendees:\n"
            "  - Alice\n"
            "  - Bob\n"
            "  - Charlie\n"
            "meeting_type: standup\n"
            "---\n\n"
            "## Discussion\n\n"
            "- Sprint progress is on track\n"
            "- API migration needs review\n"
            "- Customer feedback on new UI\n\n"
            "## Decisions\n\n"
            "- Postpone v2 launch to April\n"
            "- Hire contractor for API work\n\n"
            "## Action Items\n\n"
            "- [ ] @Alice: Review API migration PR by Friday\n"
            "- [ ] @Bob: Update sprint board\n"
            "- [ ] @Charlie: Schedule customer call\n"
        )
        assert _check_structured_meeting_notes(response) is True

    def test_tone_rewrite_pass(self):
        response = (
            "The AI system learns by repeatedly adjusting itself based on training data, "
            "similar to how a brain forms patterns through practice. After fine-tuning, "
            "it achieved 94.3% accuracy on test data."
        )
        assert _check_tone_rewrite(response) is True

    def test_contradiction_detection_pass(self):
        response = (
            "There is a contradiction between the two notes regarding time spent on "
            "the lunar surface. Note 1 states they spent 2.5 hours on the surface, "
            "while Note 2 says they were on the surface for 21.5 hours. These durations "
            "are inconsistent. In reality, they spent about 21.5 hours total on the "
            "surface, but only about 2.5 hours of that was outside during the EVA."
        )
        assert _check_contradiction_detection(response) is True


# ===========================================================================
# Expert check functions - failing cases
# ===========================================================================


class TestExpertCheckFunctionsFail:
    # --- Math ---
    def test_modular_arithmetic_fail_wrong_answer(self):
        response = "2^100 divided by 7 gives remainder 4."
        assert _check_modular_arithmetic(response) is False

    def test_inclusion_exclusion_fail_wrong_count(self):
        response = "There are 700 integers divisible by 2, 3, or 5."
        assert _check_inclusion_exclusion(response) is False

    def test_bouncing_ball_fail_only_down(self):
        # Only counts downward distance = 40
        response = "The total distance is 40 meters."
        assert _check_bouncing_ball(response) is False

    # --- Agentic Coding ---
    def test_markdown_to_html_fail_no_implementation(self):
        response = "Markdown uses **bold** and *italic* and `code` and [links](url)."
        assert _check_markdown_to_html(response) is False

    def test_data_pipeline_fail_missing_steps(self):
        response = "def process(data):\n    return json.dumps(data)"
        assert _check_data_pipeline(response) is False

    def test_retry_decorator_fail_no_backoff(self):
        response = (
            "def retry(func):\n"
            "    def wrapper(*args):\n"
            "        try:\n"
            "            return func(*args)\n"
            "        except Exception:\n"
            "            return func(*args)\n"
            "    return wrapper\n"
        )
        assert _check_retry_decorator(response) is False

    # --- Long-Chain Reasoning ---
    def test_einstein_riddle_fail_no_deduction(self):
        response = "This is a hard puzzle. I think the Brit lives in house 3."
        assert _check_einstein_riddle(response) is False

    def test_three_urns_fail_wrong_answer(self):
        response = "The probability is 3/5 = 0.6 or 60%."
        assert _check_three_urns(response) is False

    def test_topological_sort_fail_only_one_ordering(self):
        response = "The valid topological ordering is: A, B, D, C, E."
        assert _check_topological_sort(response) is False

    # --- Instruction Following ---
    def test_constrained_factorial_fail_uses_loop(self):
        response = (
            "def factorial(n):\n"
            "    result = 1\n"
            "    for i in range(1, n+1):\n"
            "        result *= i\n"
            "    return result\n"
        )
        assert _check_constrained_factorial(response) is False

    def test_library_schema_fail_missing_entity(self):
        response = (
            '{"Book": {"type": "object", "properties": {"title": {"type": "string"}}}, '
            '"Author": {"type": "object", "properties": {"name": {"type": "string"}}}}'
        )
        assert _check_library_schema(response) is False

    def test_adversarial_transform_fail_no_reverse(self):
        response = (
            "French: Le renard brun rapide saute par-dessus le chien paresseux\n"
            '<word original="Le">Le</word>\n'
        )
        assert _check_adversarial_transform(response) is False

    # --- Writing ---
    def test_multi_doc_summary_fail_missing_topic(self):
        response = (
            "Quantum computing uses qubits for exponential speedups. "
            "Climate change is raising temperatures globally."
        )
        assert _check_multi_doc_summary(response) is False

    def test_structured_meeting_notes_fail_no_yaml(self):
        response = (
            "## Discussion\n- Topic 1\n## Decisions\n- Decision 1\n"
            "## Action Items\n- @Alice: do thing\n"
        )
        assert _check_structured_meeting_notes(response) is False

    def test_tone_rewrite_fail_keeps_jargon(self):
        response = (
            "The model uses backpropagation with stochastic gradient descent "
            "to minimize the loss function across epochs."
        )
        assert _check_tone_rewrite(response) is False

    def test_contradiction_detection_fail_no_contradiction_found(self):
        response = "Both notes describe the Apollo 11 mission accurately."
        assert _check_contradiction_detection(response) is False


# ===========================================================================
# Expert check functions - with think blocks
# ===========================================================================


class TestExpertCheckFunctionsWithThinkBlocks:
    def test_inclusion_exclusion_with_think_block(self):
        response = (
            "<think>Let me apply inclusion-exclusion...</think>\n"
            "Using inclusion-exclusion: 500+333+200-166-100-66+33 = 734"
        )
        assert _check_inclusion_exclusion(response) is True

    def test_three_urns_with_think_block(self):
        response = (
            "<think>P(red from A)=3/5, then B gives 1/5 red. "
            "P(blue from A)=2/5, then C gives 4/5 red.</think>\n"
            "P(second red) = 3/25 + 8/25 = 11/25 = 0.44"
        )
        assert _check_three_urns(response) is True

    def test_contradiction_detection_with_think_block(self):
        response = (
            "<think>Note 1 says 2.5 hours, Note 2 says 21.5 hours...</think>\n"
            "The notes contradict each other on the time spent on the lunar surface: "
            "2.5 hours vs 21.5 hours."
        )
        assert _check_contradiction_detection(response) is True

    def test_multi_doc_summary_with_think_block(self):
        response = (
            "<think>I need to cover quantum, climate, and gene therapy.</think>\n"
            "Quantum computing, climate change, and gene therapy represent three "
            "cutting-edge fields. Quantum computers use qubits and superposition. "
            "Global warming threatens with rising carbon emissions. CRISPR gene "
            "therapy shows promise for genetic diseases."
        )
        assert _check_multi_doc_summary(response) is True

    def test_tone_rewrite_with_freeform_thinking_preamble(self):
        """27B models discuss jargon in thinking preamble; check should still pass."""
        response = (
            "Thinking Process:\n\n"
            "1. **Analyze the Request:**\n"
            "   Rewrite for non-technical audience. The original mentions "
            "backpropagation with stochastic gradient descent and "
            "cross-entropy loss function across training epochs.\n\n"
            "2. **Simplify:** Remove jargon, keep facts.\n\n"
            "**Rewrite:**\n"
            "The AI system learns by adjusting itself based on training data, "
            "similar to how a brain forms patterns. After fine-tuning, "
            "it achieved 94.3% accuracy on test data."
        )
        assert _check_tone_rewrite(response) is True

    def test_tone_rewrite_with_think_tags_and_jargon(self):
        """Distilled models discuss jargon inside <think> tags."""
        response = (
            "<think>The original uses backpropagation, gradient descent, "
            "loss function, hyperparameter tuning. I need to simplify.</think>\n"
            "The AI learns from data by repeatedly adjusting itself, much like "
            "how practice helps a brain improve. It reached 94.3% accuracy "
            "after the team optimized its settings."
        )
        assert _check_tone_rewrite(response) is True


# ===========================================================================
# Tool calling problem structure validation
# ===========================================================================


class TestToolCallingEvalProblems:
    def test_tool_calling_problems_count(self):
        assert len(TOOL_CALLING_PROBLEMS) == 40

    def test_all_category_is_tool_calling(self):
        for p in TOOL_CALLING_PROBLEMS:
            assert p.category == "tool_calling", f"{p.name} has category {p.category}"

    def test_all_checks_are_callable(self):
        for problem in TOOL_CALLING_PROBLEMS:
            assert callable(problem.check), f"{problem.name} check is not callable"

    def test_unique_names(self):
        names = [p.name for p in TOOL_CALLING_PROBLEMS]
        assert len(names) == len(set(names))

    def test_no_name_overlap_with_other_tiers(self):
        easy_names = {p.name for p in EVAL_PROBLEMS}
        hard_names = {p.name for p in HARD_EVAL_PROBLEMS}
        expert_names = {p.name for p in EXPERT_EVAL_PROBLEMS}
        tool_names = {p.name for p in TOOL_CALLING_PROBLEMS}
        overlap = tool_names & (easy_names | hard_names | expert_names)
        assert len(overlap) == 0, f"Overlapping names: {overlap}"


# ===========================================================================
# Tool calling check functions - passing cases
# ===========================================================================


class TestToolCallingCheckFunctionsPass:
    def test_weather_structured_json(self):
        response = '{"name": "get_weather", "arguments": {"location": "San Francisco"}}'
        assert _check_tool_call_weather(response) is True

    def test_weather_natural_language(self):
        response = (
            "I'll call the get_weather function with the location "
            'parameter set to "San Francisco".'
        )
        assert _check_tool_call_weather(response) is True

    def test_weather_tool_use_format(self):
        response = (
            "<tool_use>\n" 'get_weather(location="San Francisco")\n' "</tool_use>"
        )
        assert _check_tool_call_weather(response) is True

    def test_calculator_structured_json(self):
        response = '{"name": "calculate", "arguments": {"expression": "347 * 823"}}'
        assert _check_tool_call_calculator(response) is True

    def test_calculator_with_result(self):
        response = 'function_call: calculate(expression="347*823")\n' "Result: 285481"
        assert _check_tool_call_calculator(response) is True

    def test_multi_step_plan(self):
        response = (
            'Step 1: Call search_web(query="AI regulation news 2026")\n'
            "Step 2: Call summarize_text(text=<results from step 1>)"
        )
        assert _check_tool_call_multi_step(response) is True

    def test_multi_step_with_summary(self):
        response = (
            "I would first use search_web to find articles, then "
            "use summarize to create a summary of the findings."
        )
        assert _check_tool_call_multi_step(response) is True

    def test_json_args_structured(self):
        response = (
            '{"name": "create_file", "arguments": '
            '{"filename": "hello.py", "content": "print(\\"Hello, World!\\")"}}'
        )
        assert _check_tool_call_json_args(response) is True

    def test_json_args_tool_use_block(self):
        response = (
            "tool_call:\n"
            '{"filename": "hello.py", "content": "print(\'Hello, World!\')"}'
        )
        assert _check_tool_call_json_args(response) is True

    def test_selection_picks_send_email(self):
        response = (
            "I should use send_email because the user wants to send a message.\n"
            'send_email(recipient="alice", subject="Project Deadline", '
            'body="The project deadline is tomorrow.")'
        )
        assert _check_tool_call_selection(response) is True

    def test_selection_json_format(self):
        response = (
            '{"name": "send_email", "arguments": '
            '{"recipient": "Alice", "subject": "Deadline", "body": "Tomorrow"}}'
        )
        assert _check_tool_call_selection(response) is True


# ===========================================================================
# Tool calling check functions - failing cases
# ===========================================================================


class TestToolCallingCheckFunctionsFail:
    def test_weather_no_location(self):
        response = "I'll call get_weather to check the conditions."
        assert _check_tool_call_weather(response) is False

    def test_weather_no_tool_ref(self):
        response = "The weather in San Francisco is sunny and 72°F."
        assert _check_tool_call_weather(response) is False

    def test_calculator_wrong_expression(self):
        response = 'function_call: calculate(input="2 + 2")'
        assert _check_tool_call_calculator(response) is False

    def test_calculator_no_tool_ref(self):
        response = "347 * 823 = 285481"
        assert _check_tool_call_calculator(response) is False

    def test_multi_step_only_search(self):
        response = 'I would call search_web(query="AI regulation") to find articles.'
        assert _check_tool_call_multi_step(response) is False

    def test_multi_step_only_summarize(self):
        response = "I would use summarize_text to create a summary."
        assert _check_tool_call_multi_step(response) is False

    def test_json_args_no_json_structure(self):
        response = "Call create_file with filename hello.py and content print hello."
        assert _check_tool_call_json_args(response) is False

    def test_selection_picks_wrong_tool(self):
        response = (
            "I should use create_calendar_event to schedule a meeting "
            "with Alice about the deadline."
        )
        assert _check_tool_call_selection(response) is False

    def test_selection_picks_reminder(self):
        response = 'set_reminder(message="Tell Alice about deadline", time="tomorrow")'
        assert _check_tool_call_selection(response) is False


# ===========================================================================
# Tool calling check functions - with think blocks
# ===========================================================================


class TestToolCallingCheckFunctionsWithThinkBlocks:
    def test_weather_with_think_block(self):
        response = (
            "<think>The user wants weather in SF. I should call get_weather.</think>\n"
            '{"name": "get_weather", "arguments": {"location": "San Francisco"}}'
        )
        assert _check_tool_call_weather(response) is True

    def test_calculator_with_think_block(self):
        response = (
            "<think>I need to compute 347 * 823. Let me use the calculator.</think>\n"
            'function_call: calculate(expression="347 * 823")'
        )
        assert _check_tool_call_calculator(response) is True

    def test_selection_with_think_block(self):
        response = (
            "<think>The user wants to send a message to Alice. "
            "send_email is the right tool, not calendar or reminder.</think>\n"
            'send_email(recipient="Alice", subject="Deadline", body="Tomorrow")'
        )
        assert _check_tool_call_selection(response) is True

    def test_json_args_with_think_block(self):
        response = (
            "<think>I need to create hello.py with a print statement.</think>\n"
            '{"name": "create_file", "arguments": '
            '{"filename": "hello.py", "content": "print(\'Hello, World!\')"}}'
        )
        assert _check_tool_call_json_args(response) is True

    def test_multi_step_with_think_block(self):
        response = (
            "<think>First search for AI regulation news, then summarize.</think>\n"
            '1. search_web(query="AI regulation 2026")\n'
            "2. summarize_text(text=search_results)"
        )
        assert _check_tool_call_multi_step(response) is True
