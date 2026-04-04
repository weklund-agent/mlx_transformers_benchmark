Run speed and quality benchmarks for specified models, then update the README table.

The user will provide model name(s) that are already registered in MODEL_SPECS. If no models are specified, ask which models to benchmark.

## Steps

1. **Verify models exist** in MODEL_SPECS:
   ```
   uv run python -c "from mtb.llm_benchmarks import MODEL_SPECS; names = [m.name for m in MODEL_SPECS]; print([n for n in names if '<model>' in n])"
   ```

2. **Run speed benchmarks** (int4 and int8):
   ```
   uv run python scripts/run_llm_benchmarks.py \
       --num_iterations 3 \
       --run_only_benchmarks '["model-1","model-2"]' \
       --dtypes '["int4","int8"]'
   ```
   This can take 5-15 min per model depending on size. Run in background.

3. **Run quality benchmarks** (int4, all difficulties):
   ```
   uv run python scripts/run_quality_benchmarks.py \
       --difficulty all \
       --run_only_benchmarks '["model-1","model-2"]' \
       --dtypes '["int4"]' \
       --num_runs 3
   ```
   This takes much longer — ~40 min per MoE model, hours for large dense models. Run in background.

4. **Update README table**:
   ```
   uv run python scripts/update_readme_table.py
   ```

5. **Show summary**: Display the updated rankings and highlight where the new models placed relative to existing ones.

## Notes
- Speed benchmarks measure at prompt lengths [64, 256, 1024, 4096] — the README uses 1024
- Quality benchmarks run 46 problems x 3 runs each with majority voting
- If a model fails to load, check if mlx-lm supports its architecture

Models to benchmark: $ARGUMENTS
