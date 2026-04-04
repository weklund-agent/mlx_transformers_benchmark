---
name: python-worker
description: Python module worker for quality benchmark refactoring — implements modules, tests, and scripts
---

# Python Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Features involving Python module creation/modification, test writing, script updates, and infrastructure work for the quality benchmarks system. This includes:
- Building new modules (sandbox, parser, scoring, problem files)
- Converting existing problems to new evaluation modes
- Writing unit tests
- Updating the runner, CLI, and README table generator
- Creating integration tests

## Required Skills

None — all work is done with standard file editing and shell execution tools.

## Work Procedure

### Step 1: Understand the Feature

1. Read the feature description, preconditions, expectedBehavior, and verificationSteps carefully.
2. Read `mission.md` for overall context and `.factory/library/architecture.md` for system design.
3. Read `AGENTS.md` for boundaries and conventions.
4. Identify which existing files need modification and which new files need creation.

### Step 2: Read Existing Code

1. Read ALL files that your feature touches or depends on. Don't assume — read them.
2. For module work: read the current `mtb/quality_benchmarks/__init__.py` and related modules.
3. For test work: read `tests/test_quality_benchmarks.py` to understand existing test patterns.
4. For script work: read the relevant script in `scripts/`.

### Step 3: Write Tests First (TDD)

1. Create or update test files BEFORE implementation.
2. Write failing tests that verify the feature's expectedBehavior.
3. Run the tests to confirm they fail: `uv run pytest tests/<test_file>.py -v --tb=short`
4. For new modules: create test file in `tests/` following existing naming convention (`test_<module>.py`).
5. For existing modules: add test cases to existing test files where appropriate.

### Step 4: Implement

1. Write the implementation to make tests pass.
2. Follow existing code patterns:
   - Use `@dataclass` for data classes (match `EvalProblem` style)
   - Use `_` prefix for internal helper functions
   - Use type hints
   - Keep functions focused and testable
3. Maintain backward compatibility:
   - `eval_problems.py` must re-export everything it currently exports
   - New fields on `EvalProblem` must have defaults
   - Existing check functions must not change behavior
4. When splitting modules: use imports in `__init__.py` and `eval_problems.py` to maintain backward compatibility.

### Step 5: Run Tests and Fix

1. Run the specific tests: `uv run pytest tests/<test_file>.py -v --tb=short`
2. Run the full unit test suite: `uv run pytest tests/ --ignore=tests/quality_benchmarks --ignore=tests/llm_benchmarks --ignore=tests/system -v --tb=short -q`
3. Fix any failures.
4. Run the existing quality benchmark tests to verify backward compatibility: `uv run pytest tests/test_quality_benchmarks.py -v --tb=short`
5. All tests must pass before proceeding.

### Step 6: Format Code

1. Run: `uv run black mtb/ tests/ scripts/`
2. Fix any formatting issues.

### Step 7: Manual Verification

1. For new modules: verify imports work: `uv run python -c "from mtb.quality_benchmarks.<module> import <key_export>; print('OK')"`
2. For problems: verify problem counts: `uv run python -c "from mtb.quality_benchmarks import EVAL_PROBLEMS, HARD_EVAL_PROBLEMS, EXPERT_EVAL_PROBLEMS, TOOL_CALLING_PROBLEMS; print(len(EVAL_PROBLEMS), len(HARD_EVAL_PROBLEMS), len(EXPERT_EVAL_PROBLEMS), len(TOOL_CALLING_PROBLEMS))"`
3. For runner changes: do a smoke test with `--help` flag
4. For README changes: run `uv run python scripts/update_readme_table.py --dry-run 2>&1 | head -30`

### Step 8: Commit

1. Stage all changes.
2. Commit with a descriptive message referencing the feature.

## Example Handoff

```json
{
  "salientSummary": "Built the code execution sandbox module (mtb/quality_benchmarks/sandbox.py) with subprocess-based execution, timeout enforcement, temp file cleanup, and markdown fence stripping. Wrote 22 tests covering valid execution, timeout, errors, cleanup, and concurrent calls. All tests pass (22/22), existing quality benchmark tests unaffected (184/184 pass).",
  "whatWasImplemented": "Created sandbox.py with SandboxResult dataclass and execute_code() function. Supports configurable timeout (default 10s), captures stdout/stderr/exit_code, strips markdown code fences, cleans up temp files on success/failure/timeout. Thread-safe for concurrent use.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "uv run pytest tests/test_sandbox.py -v --tb=short", "exitCode": 0, "observation": "22 passed in 12.3s (timeout tests take ~10s each)"},
      {"command": "uv run pytest tests/test_quality_benchmarks.py -v --tb=short", "exitCode": 0, "observation": "184 passed, no regressions"},
      {"command": "uv run pytest tests/ --ignore=tests/quality_benchmarks --ignore=tests/llm_benchmarks --ignore=tests/system -v --tb=short -q", "exitCode": 0, "observation": "297 passed, 1 skipped"},
      {"command": "uv run black mtb/ tests/ scripts/", "exitCode": 0, "observation": "All files formatted"},
      {"command": "uv run python -c \"from mtb.quality_benchmarks.sandbox import execute_code, SandboxResult; print('OK')\"", "exitCode": 0, "observation": "Import successful"}
    ],
    "interactiveChecks": []
  },
  "tests": {
    "added": [
      {"file": "tests/test_sandbox.py", "cases": [
        {"name": "test_valid_code_stdout", "verifies": "Valid code produces captured stdout"},
        {"name": "test_timeout_default", "verifies": "10s timeout kills long-running code"},
        {"name": "test_timeout_custom", "verifies": "Custom timeout value respected"},
        {"name": "test_syntax_error", "verifies": "Syntax errors reported cleanly"},
        {"name": "test_runtime_errors", "verifies": "Runtime errors (TypeError, NameError, etc.) reported"},
        {"name": "test_temp_cleanup", "verifies": "Temp files cleaned up in all scenarios"},
        {"name": "test_concurrent_calls", "verifies": "Parallel sandbox calls don't interfere"}
      ]}
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Feature depends on a module or interface that doesn't exist yet
- Existing tests break in ways unrelated to this feature
- The feature's preconditions are not met (required modules missing)
- Ambiguity in requirements that affects implementation direction
- A dependency (e.g., mlx-lm) has a bug that blocks the feature
