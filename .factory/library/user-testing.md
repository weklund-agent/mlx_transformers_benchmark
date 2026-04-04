# User Testing

Testing surface, required testing skills/tools, and resource cost classification.

---

## Validation Surface

This is a pure Python library/CLI project with no web UI. All validation is through:

1. **pytest** — Unit tests verify component behavior (sandbox, parser, scoring, problems)
2. **CLI scripts** — Integration verification via `scripts/run_quality_benchmarks.py` and `scripts/update_readme_table.py`
3. **Shell commands** — Import checks, module structure verification

### Testing Tools
- `pytest` with `pytest-cov` and `pytest-mock`
- Shell commands (uv run python -c "...")
- No browser, TUI, or API testing needed

### Test Categories
- **Fast (unit)**: ~5s total, runs in default `pytest`
- **Slow (integration)**: ~10-20 min, requires `@pytest.mark.integration`, loads real model (Gemma 4 E2B-it int4)

## Validation Concurrency

**Surface: pytest (unit tests)**
- Each validator instance: ~100MB RAM
- Max concurrent: **5** (trivial resource usage)
- Rationale: Unit tests are fast and lightweight, no GPU needed

**Surface: pytest (integration tests)**
- Each validator instance: ~4-8GB RAM (model loading)
- Max concurrent: **1** (only one model should be loaded at a time to avoid GPU memory issues)
- Rationale: MLX models use unified memory; concurrent model loads would compete for GPU memory
