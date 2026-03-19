## Targets for virtual environments

# Create a virtual environment using uv
setup:
	uv sync --group=dev

# Activate the virtual environment
activate-venv:
	@echo "We recommend using `uv run python <your-script>` to use the venv without activating it."
	@echo "If you insist, to activate the virtual environment, run: "
	@echo "source .venv/bin/activate"

## Targets for running benchmarks
run-llm-benchmarks:
	uv run python scripts/run_llm_benchmarks.py --num_iterations 3

run-qwen35-benchmarks:
	uv run python scripts/run_llm_benchmarks.py --num_iterations 20 --run_only_benchmarks '["qwen-3.5-0.8b","qwen-3.5-2b","qwen-3.5-4b","qwen-3.5-9b","qwen-3.5-27b","qwen-3.5-35b-a3b"]'

run-quality-benchmarks:
	uv run python scripts/run_quality_benchmarks.py --run_only_benchmarks '["qwen-3.5-0.8b","qwen-3.5-2b","qwen-3.5-4b","qwen-3.5-9b","qwen-3.5-27b","qwen-3.5-35b-a3b"]' --num_runs 3

run-quality-benchmarks-hard:
	uv run python scripts/run_quality_benchmarks.py --difficulty hard --run_only_benchmarks '["qwen-3.5-4b","qwen-3.5-9b","qwen-3.5-27b","qwen-3.5-35b-a3b","qwen-3.5-27b-claude-opus-distilled"]' --dtypes '["int4"]' --num_runs 3

run-quality-benchmarks-expert:
	uv run python scripts/run_quality_benchmarks.py --difficulty expert --run_only_benchmarks '["qwen-3.5-4b","qwen-3.5-27b","qwen-3.5-27b-claude-opus-distilled"]' --dtypes '["int4"]' --num_runs 3

run-layer-benchmarks:
	uv run python scripts/run_layer_benchmarks.py --num_iterations 30

show-llm-benchmarks:
	uv run python scripts/visualize_llm_benchmarks.py --show_all_measurements
	open visualizations/index.html 

show-layer-benchmarks:
	uv run python scripts/visualize_layer_benchmarks.py --show_all_measurements
	open visualizations/index.html 

test:
	uv run pytest --cov --cov-report=term-missing --cov-report=html --disable-warnings -v
