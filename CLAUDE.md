# CLAUDE.md

## Project Overview

This repo benchmarks LLM inference on Apple Silicon (MLX Metal) for agentic coding use cases. It measures speed (tok/s, memory) and quality (coding, reasoning, tool calling) across models, then publishes results to the README and exports data for [mlx-stack](https://github.com/weklund/mlx-stack).

## Common Workflows

### 1. Add a new model

Models live in `mtb/llm_benchmarks/models/`. Each model family gets its own file.

**Steps:**
1. Create or update a model file in `mtb/llm_benchmarks/models/` with a `ModelSpec`
2. Add a prompt formatter function (most models use standard `system`/`user` format)
3. Register the model in `mtb/llm_benchmarks/__init__.py` (import + add to `MODEL_SPECS`)
4. Verify: `uv run python -c "from mtb.llm_benchmarks import MODEL_SPECS; print(len(MODEL_SPECS))"`

**ModelSpec fields:**
- `name`: lowercase identifier used in CLI args (e.g. `"gemma-4-e2b-it"`)
- `num_params`: active parameters as float (e.g. `2.3e9`). For MoE, use active params, not total.
- `prompt_formatter`: function returning messages list
- `model_ids`: dict of `{framework: {dtype: model_id}}`. MLX models from mlx-community use `"mlx"` key with `"int4"`, `"int8"`, `"bfloat16"` dtypes.
- `thinking`: set `True` for models with reasoning/thinking mode

**Prompt format:** Most models use standard system/user messages. Check the model family — Gemma uses `content: [{"type": "text", "text": ...}]` format while most others use plain strings.

**Common issue:** If `mlx-lm` doesn't support the model architecture yet, you'll get `"Model type X not supported"`. Check for open PRs on [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm). You can temporarily pin `mlx-lm` to a branch in `pyproject.toml` (requires `[tool.hatch.metadata] allow-direct-references = true`).

### 2. Run benchmarks

**Speed:**
```bash
uv run python scripts/run_llm_benchmarks.py \
    --num_iterations 3 \
    --run_only_benchmarks '["model-name"]' \
    --dtypes '["int4","int8"]'
```

**Quality:**
```bash
uv run python scripts/run_quality_benchmarks.py \
    --difficulty all \
    --run_only_benchmarks '["model-name"]' \
    --dtypes '["int4"]' \
    --num_runs 3
```

Quality benchmarks take much longer than speed benchmarks (~40 min per MoE model, hours for large dense models).

### 3. Update README table

```bash
uv run python scripts/update_readme_table.py
```

This reads all benchmark CSVs and regenerates the table between `<!-- BEGIN BENCHMARK TABLE -->` and `<!-- END BENCHMARK TABLE -->` markers in README.md. Use `--dry-run` to preview.

## Project Structure

```
mtb/llm_benchmarks/models/     # Model definitions (one file per family)
mtb/llm_benchmarks/__init__.py # MODEL_SPECS registry
mtb/quality_benchmarks/        # Quality evaluation problems and runner
scripts/run_llm_benchmarks.py  # Speed benchmark script
scripts/run_quality_benchmarks.py # Quality benchmark script
scripts/update_readme_table.py # README table generator
measurements/                  # All benchmark results (CSV)
```

## Key Dependencies

- `mlx-lm`: Model loading and inference. Must support the model architecture.
- `uv`: Package manager. Always use `uv run python` to run scripts.
- Benchmarks run on MLX Metal backend by default.

## Conventions

- Model names are lowercase with hyphens (e.g. `gemma-4-e2b-it`)
- MoE models use active param count for `num_params`, not total
- int4 is the primary quantization for README tables (what most people run)
- Speed measured at 1024 prompt tokens, quality uses all difficulty levels
- The `<!-- BEGIN/END BENCHMARK TABLE -->` markers in README.md are auto-managed by `update_readme_table.py` — don't hand-edit between them
