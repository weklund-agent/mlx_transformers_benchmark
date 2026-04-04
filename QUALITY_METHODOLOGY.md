# Quality Benchmark Methodology

## Overview

The quality benchmark system measures how well locally-run LLMs handle **agentic coding tasks** — the kinds of work a coding assistant needs to perform: writing functions, solving logic puzzles, following complex instructions, and calling tools with the right arguments.

This matters because speed alone doesn't determine the best model for local inference. A 400 tok/s model that can't write correct code is less useful than a 50 tok/s model that can. The quality benchmarks provide the other half of the picture, letting you compare models on a standardized set of problems that reflect real-world agentic usage.

All evaluation runs locally — no API calls, no external services. Models are loaded via MLX on Apple Silicon and prompted with each problem. Responses are evaluated through three approaches depending on the problem type: **code execution**, **tool call parsing**, or **pattern matching**.

---

## Problem Categories

The suite covers **6 categories** testing distinct capabilities:

### 1. Coding (13 problems)

Tests the model's ability to write correct, executable Python functions. Problems range from simple algorithm implementation (FizzBuzz, Fibonacci) to complex data structures (LRU cache, topological sort) and real-world tasks (Markdown-to-HTML conversion, data pipelines with method chaining).

**Evaluation:** Code execution — the model's output is run in a sandbox against test cases with concrete input/output assertions.

### 2. Reasoning (13 problems)

Tests logical deduction, multi-step inference, and mathematical reasoning. Problems include classic logic puzzles (Einstein's riddle), probability (Bayes' theorem, urn problems), and constraint satisfaction (worker scheduling, circular seating arrangements).

**Evaluation:** Pattern matching — checks for the correct answer value (number, name, or conclusion) in the response.

### 3. Math (3 problems)

Tests applied mathematical reasoning: modular arithmetic, inclusion-exclusion principle, and physics-style problems (bouncing ball trajectories). These require precise numerical computation, not just conceptual understanding.

**Evaluation:** Pattern matching — extracts the numerical answer and compares against the expected value.

### 4. Instruction Following (8 problems)

Tests the model's ability to follow specific formatting and structural constraints. Problems include generating valid JSON output, maintaining word count limits, writing code with inline comments, producing library API schemas, and adversarial text transformation tasks.

**Evaluation:** Mixed — 4 problems use code execution (code_with_comments, constrained_factorial, library_schema, adversarial_transform) and 4 use pattern matching (JSON format checking, list format, word constraints, no-thinking compliance).

### 5. Writing (4 problems)

Tests document comprehension and generation: summarizing multiple documents coherently, rewriting text in a different tone, detecting logical contradictions, and converting meeting transcripts into structured notes.

**Evaluation:** Pattern matching — checks for required structural elements, keywords, and semantic content.

### 6. Tool Calling (40 problems)

Tests the model's ability to produce structured tool/function calls in response to natural language requests. Covers selecting the right tool, formatting arguments correctly, handling multi-tool sequences, recognizing edge cases, and producing valid JSON output. See the [Tool Calling Deep Dive](#tool-calling-deep-dive) section for full details.

**Evaluation:** Structured tool call parsing — responses are parsed for JSON/Hermes-format tool calls, then validated against expected tool names and argument values.

---

## Category Distribution Table

| Category              | Easy | Hard | Expert | Tool Calling | Total |
|-----------------------|-----:|-----:|-------:|-------------:|------:|
| Coding                |    5 |    5 |      3 |            — |    13 |
| Reasoning             |    5 |    5 |      3 |            — |    13 |
| Math                  |    — |    — |      3 |            — |     3 |
| Instruction Following |    5 |    — |      3 |            — |     8 |
| Writing               |    — |    — |      4 |            — |     4 |
| Tool Calling          |    — |    — |      — |           40 |    40 |
| **Total**             | **15** | **10** | **16** | **40** | **81** |

---

## Difficulty Tiers

### Easy (15 problems)

Foundational tasks that any capable model should handle. Easy coding problems test basic algorithms (FizzBuzz, reverse string, Fibonacci, binary search, palindrome detection). Easy reasoning problems involve straightforward logic puzzles and sequence problems. Easy instruction following covers basic format compliance (JSON output, list formatting, word constraints).

A model that fails Easy problems likely has fundamental reasoning deficits at its quantization level.

### Hard (10 problems)

Intermediate tasks requiring deeper reasoning or more complex implementation. Hard coding problems involve data structures and algorithms (LRU cache, flatten nested lists, longest palindrome substring, expression calculator, identifying bugs in merge sort). Hard reasoning problems require multi-step logical deduction (compound interest, Einstein's riddle, Bayes' theorem, worker scheduling, three urns).

### Expert (16 problems)

Challenging tasks that test the upper bounds of model capability. Expert coding problems involve system-level tasks (Markdown-to-HTML parser, data pipeline with method chaining, retry decorator with exponential backoff). Expert reasoning includes graph algorithms (topological sort) and advanced probability (circular seating). Expert math requires precise computation (modular arithmetic, inclusion-exclusion, bouncing ball physics). Expert instruction following tests adversarial robustness. Expert writing tests complex document comprehension and generation.

### Tool Calling (40 problems)

A dedicated tier for structured tool use evaluation. All 40 problems are in the tool_calling category and are weighted at the Expert level (3×) because correct tool use is critical for agentic coding workflows. See [Tool Calling Deep Dive](#tool-calling-deep-dive) for subcategory details.

---

## Weighted Scoring Formula

Problems are weighted by difficulty tier to ensure harder problems have more impact on the final score:

| Tier          | Weight | Problem Count | Max Weighted Points |
|---------------|-------:|--------------:|--------------------:|
| Easy          |     1× |            15 |                  15 |
| Hard          |     2× |            10 |                  20 |
| Expert        |     3× |            16 |                  48 |
| Tool Calling  |     3× |            40 |                 120 |
| **Total**     |        |        **81** |             **203** |

### Formula

```
weighted_score = sum(weight × passed_count_per_tier) / sum(weight × total_count_per_tier)
```

Where:
- `weight` is the tier weight (1, 2, 3, or 3)
- `passed_count_per_tier` is the number of problems passed in that tier
- `total_count_per_tier` is the total number of problems in that tier

The maximum possible denominator is **203**.

### Worked Example

Suppose a model achieves:
- Easy: 10 of 15 passed
- Hard: 6 of 10 passed
- Expert: 9 of 16 passed
- Tool Calling: 25 of 40 passed

```
weighted_score = (1×10 + 2×6 + 3×9 + 3×25) / (1×15 + 2×10 + 3×16 + 3×40)
               = (10 + 12 + 27 + 75) / (15 + 20 + 48 + 120)
               = 124 / 203
               ≈ 61.1%
```

The raw (unweighted) pass rate would be `(10+6+9+25) / (15+10+16+40) = 50/81 ≈ 61.7%`. In this case the two values are close, but they diverge when a model is strong at Easy problems but weak at Expert/Tool Calling (weighted score will be lower) or vice versa.

Both the weighted score and raw pass rate are reported. The weighted score is used as the primary "Quality" percentage in the README benchmark table.

---

## Evaluation Approaches

### 1. Code Execution (18 problems)

Coding problems are evaluated by actually **running the model's generated code** in a subprocess sandbox, not by pattern matching keywords.

**How it works:**

1. Each coding problem defines a `function_signature` (e.g., `def fizzbuzz(n: int) -> str:`) and a set of `test_cases` (5–10 per problem) with input/expected output pairs.
2. The model generates a Python function in response to the prompt.
3. The runner extracts the code from the model's response (stripping markdown fences, explanatory text, and thinking blocks).
4. The extracted code is combined with test case assertions (e.g., `assert fizzbuzz(15) == "FizzBuzz"`) and executed in a **subprocess sandbox**.
5. If all assertions pass (exit code 0), the problem passes. If any assertion fails or the code errors/times out, it fails.

**Sandbox details (`mtb/quality_benchmarks/sandbox.py`):**
- Executes code via `subprocess.run()` in an isolated process
- Default timeout: 10 seconds (configurable per problem)
- Markdown code fences (`\`\`\`python ... \`\`\``) are stripped automatically
- Captures stdout, stderr, and exit code separately
- Classifies errors as `timeout`, `syntax_error`, or `runtime_error`
- Temp files are always cleaned up (success, failure, or timeout)
- Thread-safe: concurrent executions use separate temp files

**Why code execution?** Pattern matching (checking if "FizzBuzz" appears in the response) can pass models that explain the algorithm without implementing it correctly. Code execution verifies functional correctness — the code must actually work.

### 2. Tool Call Parsing (40 problems)

Tool calling problems are evaluated by **parsing structured tool calls** from the model's response and validating them against expected tool names and argument values.

**How it works:**

1. Each problem defines available tools (with names, descriptions, and parameter schemas) and a user request.
2. The model generates a response containing one or more tool calls.
3. The **multi-format parser** (`mtb/quality_benchmarks/tool_call_parser.py`) extracts structured `ToolCall` objects (name + arguments dict) from the response.
4. The problem's `check` function validates that the correct tool was called with the correct arguments.

**Parser priority chain:**
1. **Hermes format**: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` — XML-style tags wrapping JSON
2. **JSON code blocks**: `` ```json {"name": "...", "arguments": {...}} ``` `` — Markdown fenced code blocks
3. **Raw JSON**: Bare JSON objects with a `name` or `function` key embedded in prose

The parser always strips `<think>...</think>` blocks before extracting tool calls and never raises exceptions — it returns `None` if no valid tool calls are found.

**Recognized patterns:**
- `{"name": "...", "arguments": {...}}` — standard format
- `{"function": "...", "arguments": {...}}` — function key variant
- `{"name": "...", "parameters": {...}}` — parameters variant
- `{"function": {"name": "...", "arguments": {...}}}` — nested function object
- Arrays of the above for multi-tool calls
- Single-quoted JSON (best-effort conversion to double quotes)

### 3. Pattern Matching (23 remaining problems)

Reasoning, math, writing, and non-code instruction following problems use **keyword and semantic checks** via each problem's `check()` function.

**How it works:**

1. The model generates a free-form response.
2. `<think>...</think>` blocks are stripped (models with reasoning/thinking mode produce these).
3. The `check()` function searches for expected answer indicators — correct numbers, keywords, structural elements, or semantic content.

**Design philosophy:** Pattern matching is intentionally lenient. The goal is to detect whether quantization breaks a model's reasoning ability, not to enforce exact formatting. For example, a math problem checking for the answer "42" will accept "The answer is 42", "42.", "= 42", etc.

Common check utilities:
- `_strip_thinking(response)` — removes `<think>` blocks and freeform thinking preambles
- `_contains_any(response, targets)` — case-insensitive check for any target string
- `_extract_number(response)` — extracts the last number from the response (models often reason before giving a final answer)

---

## Tool Calling Deep Dive

The 40 tool calling problems are divided into **5 subcategories** of 8 problems each, testing progressively more nuanced aspects of tool use:

### 1. Tool Selection (8 problems, prefix: `ts_`)

Tests whether the model can pick the right tool from a set of options.

| Problem | Description |
|---------|-------------|
| `ts_correct_tool` | Select the correct tool from 5+ options given a clear request |
| `ts_ambiguous_request` | Handle a request that could match multiple tools; pick the best fit |
| `ts_none_selection` | Recognize when no available tool matches the request |
| `ts_similar_names` | Distinguish between tools with similar names but different purposes |
| `ts_parameter_based` | Select based on which tool accepts the right parameter types |
| `ts_nested_descriptions` | Parse verbose, nested tool descriptions to find the right match |
| `ts_specialized_vs_general` | Prefer a specialized tool over a general-purpose one when appropriate |
| `ts_multiple_valid` | Handle cases where multiple tools are valid; select any correct one |

### 2. Argument Accuracy (8 problems, prefix: `aa_`)

Tests whether the model extracts and formats arguments correctly from natural language.

| Problem | Description |
|---------|-------------|
| `aa_date_iso_format` | Parse natural language dates into ISO format (YYYY-MM-DD) |
| `aa_email_extraction` | Extract email addresses from conversational text |
| `aa_enum_values` | Map natural language to valid enum parameter values |
| `aa_nested_objects` | Construct nested JSON argument objects from flat descriptions |
| `aa_numeric_coercion` | Convert string numbers to numeric types in arguments |
| `aa_boolean_natural_lang` | Interpret "yes"/"no"/"true"/"false" as boolean argument values |
| `aa_all_required_args` | Include all required parameters (no omissions) |
| `aa_preserve_exact_strings` | Pass string arguments exactly as specified (no paraphrasing) |

### 3. Multi-Tool (8 problems, prefix: `mt_`)

Tests handling of scenarios requiring multiple tool calls.

| Problem | Description |
|---------|-------------|
| `mt_parallel_independent` | Make two independent tool calls that can run in parallel |
| `mt_sequential_dependent` | Chain two calls where the second depends on the first's result |
| `mt_mixed` | Combine parallel and sequential tool calls in one response |
| `mt_three_parallel` | Make three independent tool calls simultaneously |
| `mt_multi_turn_with_result` | Use a tool, receive a result, then use another tool with that result |
| `mt_chain_of_three` | Chain three dependent tool calls in sequence |
| `mt_parallel_different` | Make parallel calls to different tools (not the same tool twice) |
| `mt_conditional_planning` | Plan conditional tool use (if tool A returns X, then call tool B) |

### 4. Edge Cases (8 problems, prefix: `ec_`)

Tests handling of unusual or adversarial scenarios.

| Problem | Description |
|---------|-------------|
| `ec_refuse_trivial` | Refuse to use a tool when the question is trivially answerable (e.g., "what is 2+2?") |
| `ec_missing_required_params` | Handle requests where required parameters can't be determined |
| `ec_handle_tool_error` | Respond appropriately when told a previous tool call failed |
| `ec_no_matching_tool` | Recognize when no tool can fulfill the request |
| `ec_optional_params` | Correctly handle optional parameters (include or omit as appropriate) |
| `ec_reject_harmful` | Refuse to call tools for harmful or unsafe requests |
| `ec_deprecated_param` | Avoid using deprecated parameters when alternatives exist |
| `ec_idempotency` | Avoid making duplicate tool calls for the same operation |

### 5. Format Compliance (8 problems, prefix: `fc_`)

Tests whether tool calls are properly structured JSON.

| Problem | Description |
|---------|-------------|
| `fc_valid_json_format` | Produce a syntactically valid JSON tool call |
| `fc_array_params` | Correctly format array-type parameters |
| `fc_optional_included` | Include optional parameters when the user specifies them |
| `fc_multiple_tools_single_response` | Format multiple tool calls in a single response |
| `fc_null_argument` | Use null/None for explicitly unset optional parameters |
| `fc_empty_string` | Pass empty strings when appropriate (not null, not omitted) |
| `fc_consistent_format` | Maintain consistent formatting across multiple tool calls |
| `fc_type_matching` | Ensure argument types match the schema (string vs number vs boolean) |

---

## Contamination Resistance

LLMs may memorize benchmark problems from their training data. The contamination resistance system mitigates this by generating **parameterized variants** with different concrete values while preserving the same problem structure and difficulty.

### Parameterized Variant System

12 problems define a `generate_variant()` method that produces a new `EvalProblem` with randomized values:

| Problem | Category | What Changes |
|---------|----------|-------------|
| `fizzbuzz` | coding | Divisor numbers and replacement words |
| `fibonacci` | coding | Target index and sequence starting values |
| `binary_search` | coding | Array contents and search target |
| `flatten_nested` | coding | Nesting structure and leaf values |
| `train_problem` | reasoning | Train speeds, distances, and departure times |
| `workers_problem` | reasoning | Number of workers, task completion times |
| `age_problem` | reasoning | Ages, relationships, and time offsets |
| `compound_interest` | reasoning | Principal, rate, and compounding period |
| `circular_seating` | reasoning | Number of people, seating constraints |
| `modular_arithmetic` | math | Modulus, base, and exponent values |
| `inclusion_exclusion` | math | Set sizes and overlap counts |
| `bouncing_ball` | math | Drop height, bounce ratio, and target bounce |

Each call to `generate_variant()` produces a new problem with:
- A modified prompt containing the new concrete values
- An updated `check()` function that validates the new expected answer
- Updated `test_cases` (for coding problems) with new input/output pairs

### Usage

```bash
# Enable variants (each parameterized problem generates 3 variants by default)
uv run python scripts/run_quality_benchmarks.py \
    --difficulty all --use_variants --num_variants 3 \
    --run_only_benchmarks '["model-name"]' \
    --dtypes '["int4"]'
```

When `--use_variants` is enabled, each parameterized problem is replaced by `num_variants` freshly generated variants (e.g., `fizzbuzz_variant_1`, `fizzbuzz_variant_2`, `fizzbuzz_variant_3`). Non-parameterized problems are kept unchanged.

### ProblemSource Interface

The `ProblemSource` abstract base class (`mtb/quality_benchmarks/scoring.py`) provides a pluggable interface for problem sources:

```python
from abc import ABC, abstractmethod

class ProblemSource(ABC):
    """Abstract base class for pluggable problem sources.

    Provides a unified interface for retrieving evaluation problems.
    Built-in implementations include StaticProblemSource which wraps
    the existing hardcoded problem lists. Future implementations may
    include dynamic sources such as LiveCodeBench for fresh, uncontaminated
    problems fetched from an online repository.
    """

    @abstractmethod
    def get_problems(self, difficulty: str = "all") -> List[EvalProblem]:
        ...
```

The built-in `StaticProblemSource` wraps the four static problem lists (`EVAL_PROBLEMS`, `HARD_EVAL_PROBLEMS`, `EXPERT_EVAL_PROBLEMS`, `TOOL_CALLING_PROBLEMS`). Future implementations could pull fresh problems from external sources like LiveCodeBench to eliminate memorization effects entirely.

---

## Majority Voting

Each problem is run `num_runs` times (default: 3) to reduce noise from non-deterministic generation. A problem passes only if a **strict majority** of runs pass:

```
passed = pass_count > num_runs / 2
```

Examples:
- `num_runs=3`: requires ≥2 passes (2/3 ✓, 1/3 ✗)
- `num_runs=5`: requires ≥3 passes (3/5 ✓, 2/5 ✗)
- `num_runs=4`: requires ≥3 passes (strict majority, not tie — 2/4 ✗)

This smooths out cases where a model occasionally generates an incorrect response due to sampling randomness while ensuring that consistently failing models aren't given false credit.

---

## Running Benchmarks

### CLI Flags Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--difficulty` | `"easy"` | Problem tier: `"easy"`, `"hard"`, `"expert"`, `"tool_calling"`, or `"all"` |
| `--dtypes` | `("int4", "int8", "bfloat16")` | Quantization types to evaluate (JSON array string) |
| `--num_runs` | `3` | Number of runs per problem for majority voting |
| `--run_only_benchmarks` | `None` | Filter to specific model names (JSON array string) |
| `--run_mlx_metal` | `True` | Run MLX with Metal backend |
| `--run_ollama_metal` | `False` | Run Ollama with Metal backend |
| `--use_variants` | `False` | Enable parameterized variants for contamination resistance |
| `--num_variants` | `3` | Number of variants per parameterized problem |
| `--cooldown_time_fraction` | `0.05` | Cooldown between problems (fraction of generation time) |
| `--output_root` | `measurements/quality_benchmarks` | Output directory for results |
| `--hf_cache_dir` | System default | HuggingFace model cache directory |

### Example Commands

```bash
# Quick test: Easy problems only, single model, int4
uv run python scripts/run_quality_benchmarks.py \
    --difficulty easy \
    --run_only_benchmarks '["gemma-4-e2b-it"]' \
    --dtypes '["int4"]'

# Full evaluation: All 81 problems, single model
uv run python scripts/run_quality_benchmarks.py \
    --difficulty all \
    --run_only_benchmarks '["qwen-3.5-9b"]' \
    --dtypes '["int4"]' \
    --num_runs 3

# With contamination-resistant variants
uv run python scripts/run_quality_benchmarks.py \
    --difficulty all --use_variants --num_variants 5 \
    --run_only_benchmarks '["qwen-3.5-9b"]' \
    --dtypes '["int4"]'

# Multiple models, multiple quantizations
uv run python scripts/run_quality_benchmarks.py \
    --difficulty all \
    --dtypes '["int4","int8"]' \
    --num_runs 5

# Integration tests (requires Gemma 4 E2B-it int4 model)
uv run pytest tests/ -m integration
```

### Output

Results are saved as CSV files in `measurements/quality_benchmarks/` with columns including model, dtype, category, problem name, pass count, number of runs, passed (majority vote), generation time, tokens per second, and a sample response.

After running benchmarks, update the README table:

```bash
uv run python scripts/update_readme_table.py        # Update README in place
uv run python scripts/update_readme_table.py --dry-run  # Preview without modifying
```

---

## Extending the Suite

### Adding a New Problem

1. **Choose the category file:** Problems live in `mtb/quality_benchmarks/<category>_problems.py` (e.g., `coding_problems.py`, `reasoning_problems.py`).

2. **Create an `EvalProblem`:**

```python
EvalProblem(
    category="coding",          # One of: coding, reasoning, math, instruction_following, writing, tool_calling
    name="my_new_problem",      # Unique lowercase identifier with underscores
    prompt="Write a function that...",  # The prompt sent to the model
    check=_check_my_new_problem,        # Validation function: (str) -> bool
    max_tokens=512,             # Token budget for generation
    # For code execution problems:
    function_signature="def my_func(x: int) -> int:",  # Optional
    test_cases=[...],           # Optional list of test cases
    # For contamination resistance:
    generate_variant=_generate_my_new_variant,  # Optional variant generator
)
```

3. **Write the check function:** For pattern matching, write a `_check_my_new_problem(response: str) -> bool` function. For code execution, provide `test_cases` instead.

4. **Add to the appropriate list:** Append to `CODING_EASY_PROBLEMS`, `REASONING_HARD_PROBLEMS`, etc. in the category file.

5. **Verify backward compatibility:**
```bash
uv run python -c "from mtb.quality_benchmarks import EVAL_PROBLEMS; print(len(EVAL_PROBLEMS))"
uv run pytest tests/test_quality_benchmarks.py -v --tb=short
```

### Adding a New Category

1. Create `mtb/quality_benchmarks/<category>_problems.py` with problem lists.
2. Import and re-export from `mtb/quality_benchmarks/eval_problems.py`.
3. Add to the appropriate tier lists (`EVAL_PROBLEMS`, `HARD_EVAL_PROBLEMS`, etc.).
4. Update the `scoring.py` tier mapping if the new category introduces a new tier.

### Adding a New Problem Source

Implement the `ProblemSource` ABC:

```python
from mtb.quality_benchmarks.scoring import ProblemSource

class LiveCodeBenchSource(ProblemSource):
    """Fetch fresh problems from LiveCodeBench API."""

    def get_problems(self, difficulty: str = "all") -> list[EvalProblem]:
        # Fetch and convert problems from external source
        ...
```

The runner can then be configured to use different problem sources, enabling evaluation against problems the model has never seen in training.
