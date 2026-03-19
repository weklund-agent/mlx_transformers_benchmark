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

## Targets for downloading and converting models

# HuggingFace source models for Qwen 3.5
HF_QWEN35_MODELS = \
	Qwen/Qwen3.5-0.8B \
	Qwen/Qwen3.5-2B \
	Qwen/Qwen3.5-4B \
	Qwen/Qwen3.5-9B \
	Qwen/Qwen3.5-27B \
	Qwen/Qwen3.5-35B-A3B \
	Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled

# Download all source models from HuggingFace
download-models:
	@echo "Downloading Qwen 3.5 source models from HuggingFace..."
	@for model in $(HF_QWEN35_MODELS); do \
		echo "\n=== Downloading $$model ==="; \
		huggingface-cli download $$model || echo "WARNING: Failed to download $$model"; \
	done
	@echo "\nAll downloads complete. Run 'make convert-models' to convert to MLX."

# Convert all downloaded models to MLX quantized formats
# Creates models/ directory with int4, int8, and bf16 variants
convert-models: convert-qwen35-small convert-qwen35-medium convert-qwen35-large convert-qwen35-distilled
	@echo "\nAll conversions complete. Models are in ./models/"

# Small models (0.8B): int4 + int8 only
convert-qwen35-small:
	@echo "Converting Qwen3.5-0.8B..."
	uv run python -m mlx_lm.convert --hf-path Qwen/Qwen3.5-0.8B -q --q-bits 4 --mlx-path models/Qwen3.5-0.8B-4bit
	uv run python -m mlx_lm.convert --hf-path Qwen/Qwen3.5-0.8B -q --q-bits 8 --mlx-path models/Qwen3.5-0.8B-8bit

# Medium models (2B, 4B, 9B): int4 + int8 + bf16
convert-qwen35-medium:
	@echo "Converting Qwen3.5-2B..."
	uv run python -m mlx_lm.convert --hf-path Qwen/Qwen3.5-2B -q --q-bits 4 --mlx-path models/Qwen3.5-2B-4bit
	uv run python -m mlx_lm.convert --hf-path Qwen/Qwen3.5-2B -q --q-bits 8 --mlx-path models/Qwen3.5-2B-8bit
	uv run python -m mlx_lm.convert --hf-path Qwen/Qwen3.5-2B --mlx-path models/Qwen3.5-2B-bf16
	@echo "Converting Qwen3.5-4B..."
	uv run python -m mlx_lm.convert --hf-path Qwen/Qwen3.5-4B -q --q-bits 4 --mlx-path models/Qwen3.5-4B-4bit
	uv run python -m mlx_lm.convert --hf-path Qwen/Qwen3.5-4B -q --q-bits 8 --mlx-path models/Qwen3.5-4B-8bit
	uv run python -m mlx_lm.convert --hf-path Qwen/Qwen3.5-4B --mlx-path models/Qwen3.5-4B-bf16
	@echo "Converting Qwen3.5-9B..."
	uv run python -m mlx_lm.convert --hf-path Qwen/Qwen3.5-9B -q --q-bits 4 --mlx-path models/Qwen3.5-9B-4bit
	uv run python -m mlx_lm.convert --hf-path Qwen/Qwen3.5-9B -q --q-bits 8 --mlx-path models/Qwen3.5-9B-8bit
	uv run python -m mlx_lm.convert --hf-path Qwen/Qwen3.5-9B --mlx-path models/Qwen3.5-9B-bf16

# Large models (27B, 35B-A3B): int4 + int8 only
convert-qwen35-large:
	@echo "Converting Qwen3.5-27B..."
	uv run python -m mlx_lm.convert --hf-path Qwen/Qwen3.5-27B -q --q-bits 4 --mlx-path models/Qwen3.5-27B-4bit
	uv run python -m mlx_lm.convert --hf-path Qwen/Qwen3.5-27B -q --q-bits 8 --mlx-path models/Qwen3.5-27B-8bit
	@echo "Converting Qwen3.5-35B-A3B..."
	uv run python -m mlx_lm.convert --hf-path Qwen/Qwen3.5-35B-A3B -q --q-bits 4 --mlx-path models/Qwen3.5-35B-A3B-4bit
	uv run python -m mlx_lm.convert --hf-path Qwen/Qwen3.5-35B-A3B -q --q-bits 8 --mlx-path models/Qwen3.5-35B-A3B-8bit

# Distilled model: int4 only
convert-qwen35-distilled:
	@echo "Converting Qwen3.5-27B-Claude-Opus-Distilled..."
	uv run python -m mlx_lm.convert --hf-path Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled -q --q-bits 4 --mlx-path models/Qwen3.5-27B-Claude-Opus-Distilled-4bit
