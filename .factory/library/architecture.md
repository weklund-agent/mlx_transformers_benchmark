# Quality Benchmarks Architecture

Reference doc for the refactored quality benchmarks system.

---

## Module Structure

### Before (current)

```
mtb/quality_benchmarks/
  __init__.py              # Exports EVAL_PROBLEMS, HARD_EVAL_PROBLEMS, EXPERT_EVAL_PROBLEMS, TOOL_CALLING_PROBLEMS, run_quality_benchmark
  eval_problems.py         # ~1400 lines: EvalProblem dataclass, all check functions, all problem lists (4 lists, 46 problems)
  run_quality_benchmark.py # Runner: loads model, iterates problems, majority-vote scoring, writes CSV
scripts/
  run_quality_benchmarks.py  # CLI: selects difficulty, filters models, invokes runner
  update_readme_table.py     # Reads CSV, generates markdown tables between README markers
```

### After (refactored)

```
mtb/quality_benchmarks/
  __init__.py                 # Re-exports: EVAL_PROBLEMS, run_quality_benchmark, ProblemSource, EvalProblem, CodeExecutionProblem
  problems/
    __init__.py               # Aggregates all problem lists; enforces CATEGORY_ALLOWLIST
    coding.py                 # Coding problems + check functions (easy/hard/expert)
    reasoning.py              # Reasoning + math problems + check functions
    instruction.py            # Instruction-following problems + check functions
    math.py                   # Standalone math category (expert-level)
    writing.py                # Writing problems + check functions
    tool_calling.py           # Tool-calling problems + check functions
  sandbox.py                  # Subprocess-based code execution with timeout/resource limits
  tool_call_parser.py         # Multi-format tool call extraction (JSON, XML, Hermes, plain-text)
  scoring.py                  # Weighted scoring: per-category weights, composite score calculation
  run_quality_benchmark.py    # Updated runner: dispatches to 3 eval modes, writes enriched CSV
  problem_source.py           # ProblemSource interface + parameterized variant support
scripts/
  run_quality_benchmarks.py   # CLI (unchanged interface, updated imports)
  update_readme_table.py      # Updated to read new CSV columns (weighted scores, per-category breakdowns)
```

---

## Key Interfaces

### EvalProblem (existing, preserved)

```python
@dataclass
class EvalProblem:
    category: str              # Must be in CATEGORY_ALLOWLIST
    name: str                  # Unique within difficulty tier
    prompt: str
    check: Callable[[str], bool]   # Pattern-match evaluator
    max_tokens: int = 512
```

### CodeExecutionProblem (new, extends EvalProblem)

```python
@dataclass
class CodeExecutionProblem(EvalProblem):
    eval_mode: str = "code_execution"  # Override default "pattern_match"
    test_code: str = ""                # Python code to validate model output
    expected_output: Optional[str] = None
    timeout_sec: float = 10.0
```

The `check` field on CodeExecutionProblem delegates to sandbox execution — the runner detects `eval_mode` and routes accordingly.

### ProblemSource (new)

```python
class ProblemSource(Protocol):
    def get_problems(self, difficulty: str) -> List[EvalProblem]: ...
```

Built-in source: `StaticProblemSource` (returns the hardcoded lists). Future: `ParameterizedProblemSource` generates variants by substituting values into prompt templates.

### Sandbox Result

```python
@dataclass
class SandboxResult:
    passed: bool
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_sec: float
```

Returned by `sandbox.run(code: str, timeout: float) -> SandboxResult`. Executes in a subprocess with no network access and a wall-clock timeout.

---

## Evaluation Modes

The runner dispatches each problem based on its `eval_mode` field (default: `"pattern_match"`):

| Mode | Trigger | How it works |
|---|---|---|
| `pattern_match` | `EvalProblem` (default) | Calls `problem.check(response)` — regex/substring matching on model output |
| `code_execution` | `CodeExecutionProblem` | Extracts code from response, runs via `sandbox.run()`, checks exit code / stdout |
| `tool_call_parse` | `eval_mode="tool_call_parse"` | Extracts tool calls via `tool_call_parser.parse()`, validates name + args against expected schema |

All three modes produce a `bool` pass/fail that feeds into the same majority-vote aggregation.

---

## Data Flow

```
Problem Definition (problems/*.py or ProblemSource)
        │
        ▼
  run_quality_benchmark()
    ├─ load model via create_benchmark()
    ├─ for each problem:
    │   ├─ run num_runs times (model.run_once)
    │   ├─ dispatch to eval mode → bool per run
    │   ├─ majority vote → final pass/fail
    │   └─ collect speed metrics (gen_tps, gen_time)
    │
    ▼
  CSV (measurements/quality_benchmarks/<hw>/<run>/quality_results.csv)
    │  Columns: model, framework, backend, dtype, category, problem,
    │           pass_count, num_runs, passed, avg_generation_tps, ...,
    │           weighted_score (NEW), eval_mode (NEW)
    │
    ▼
  update_readme_table.py
    ├─ load_speed_data() from llm_benchmarks CSVs
    ├─ load_quality_data() from quality_benchmarks CSVs
    ├─ compute_quality_summary(): per-model overall %, per-category breakdowns
    ├─ generate_hardware_table(): markdown table per hardware profile
    └─ write between <!-- BEGIN/END BENCHMARK TABLE --> markers in README.md
```

---

## Scoring (scoring.py)

Weighted composite score replaces simple pass-count percentage:

```python
CATEGORY_WEIGHTS = {
    "coding": 0.30,
    "reasoning": 0.20,
    "tool_calling": 0.20,
    "math": 0.10,
    "instruction_following": 0.10,
    "writing": 0.10,
}
```

Per-category score = (problems passed / total problems) in that category. Composite = weighted sum. The CSV gains a `weighted_score` column; `update_readme_table.py` uses it for the Quality % column.

---

## Category Allowlist

```python
CATEGORY_ALLOWLIST = {"coding", "reasoning", "tool_calling", "math", "instruction_following", "writing"}
```

Enforced at problem registration time. Any `EvalProblem.category` not in this set raises `ValueError`. This prevents drift and keeps CSV/README categories consistent.

---

## Backward Compatibility Invariants

1. **CLI interface unchanged**: `scripts/run_quality_benchmarks.py` keeps the same `--difficulty`, `--dtypes`, `--run_only_benchmarks`, `--num_runs` flags.
2. **CSV superset**: New columns (`weighted_score`, `eval_mode`) are additive. Existing columns preserved with same semantics.
3. **README markers**: `<!-- BEGIN/END BENCHMARK TABLE -->` protocol unchanged. `update_readme_table.py` remains the single writer.
4. **EvalProblem backward-compat**: Existing `EvalProblem` instances with `check` callables continue to work as `pattern_match` eval mode (the default).
5. **Problem names stable**: Problem `name` fields are identifiers used in CSV rows. Renaming breaks historical comparison — avoid it.
6. **Difficulty tiers preserved**: `"easy"`, `"hard"`, `"expert"`, `"tool_calling"`, `"all"` map to the same problem sets. New problems append, don't replace.

---

## Naming Conventions

- Model names: lowercase with hyphens (`gemma-4-e2b-it`)
- Problem names: lowercase with underscores (`fizzbuzz`, `lru_cache`)
- Category names: lowercase with underscores, must be in `CATEGORY_ALLOWLIST`
- Module files: lowercase, match category name where possible (`coding.py`, `reasoning.py`)
- CSV filenames: `quality_results.csv` (unchanged)
