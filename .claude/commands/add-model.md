Add a new model to the benchmark suite.

The user will provide model name(s) or HuggingFace model IDs. For each model:

1. **Research the model**: Search mlx-community on HuggingFace for available quantizations (int4, int8, bf16). Determine the model architecture (dense vs MoE), active parameter count, and whether it supports thinking/reasoning mode.

2. **Check mlx-lm support**: Verify the model architecture is supported by running:
   ```
   uv run python -c "import importlib; importlib.import_module('mlx_lm.models.<model_type>')"
   ```
   If not supported, search for open PRs on ml-explore/mlx-lm and inform the user.

3. **Create the model spec**: Add to an existing or new file in `mtb/llm_benchmarks/models/`. Follow the existing pattern:
   - Prompt formatter function (check the model family's chat template format)
   - ModelSpec with name, num_params (active params for MoE), prompt_formatter, model_ids, and thinking flag
   - Use lowercase-hyphenated names (e.g. `gemma-4-e2b-it`)

4. **Register in __init__.py**: Import the model and add to MODEL_SPECS list in `mtb/llm_benchmarks/__init__.py`.

5. **Verify**: Run `uv run python -c "from mtb.llm_benchmarks import MODEL_SPECS; [print(m.name) for m in MODEL_SPECS if '<model>' in m.name]"` to confirm.

Model to add: $ARGUMENTS
