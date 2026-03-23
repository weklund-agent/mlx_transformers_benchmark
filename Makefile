.DEFAULT_GOAL := help

# Show all available targets with descriptions
help:
	@awk '/^## / { \
		section = substr($$0, 4); \
		printf "\n\033[1m%s\033[0m\n", section; \
		next \
	} \
	/^# / { \
		desc = substr($$0, 3); \
		next \
	} \
	/^[a-zA-Z0-9_-]+:/ && desc { \
		target = $$1; \
		sub(/:.*/, "", target); \
		printf "  \033[36m%-42s\033[0m %s\n", target, desc; \
		desc = "" \
	}' $(MAKEFILE_LIST)
	@echo ""

# ---------------------------------------------------------------------------
# Model groups
# ---------------------------------------------------------------------------
# Models that fit on any hardware (Mac Mini 64GB, M5 Max 128GB, etc.)
MODELS_QWEN35            := '["qwen-3.5-0.8b","qwen-3.5-2b","qwen-3.5-4b","qwen-3.5-9b","qwen-3.5-27b","qwen-3.5-35b-a3b"]'
MODELS_QWEN35_LARGE      := '["qwen-3.5-4b","qwen-3.5-9b","qwen-3.5-27b","qwen-3.5-35b-a3b","qwen-3.5-27b-claude-opus-distilled"]'

# Models that require >=128GB unified memory (M5 Max 128GB, etc.)
# Will NOT fit on Mac Mini M4 Pro 64GB or similar.
MODELS_128GB             := '["qwen-3.5-27b","qwen-3.5-35b-a3b","qwen-3.5-27b-claude-opus-distilled","gemma-3-27b-it","qwen-3-32B-it","llama-3.3-70b-it"]'

# ---------------------------------------------------------------------------

## Setup

# Create a virtual environment using uv
setup:
	uv sync --group=dev

# Activate the virtual environment
activate-venv:
	@echo "We recommend using 'uv run python <your-script>' to use the venv without activating it."
	@echo "If you insist, to activate the virtual environment, run:"
	@echo "  source .venv/bin/activate"

## Speed benchmarks — all hardware

# Run all LLM speed benchmarks (3 iterations)
run-llm-benchmarks:
	uv run python scripts/run_llm_benchmarks.py --num_iterations 3

# Run Qwen 3.5 speed benchmarks (20 iterations)
run-llm-benchmarks-qwen35:
	uv run python scripts/run_llm_benchmarks.py \
		--num_iterations 20 \
		--run_only_benchmarks $(MODELS_QWEN35)

# Run Nemotron speed benchmarks (20 iterations)
run-llm-benchmarks-nemotron:
	uv run python scripts/run_llm_benchmarks.py \
		--num_iterations 20 \
		--run_only_benchmarks '["nemotron-3-nano-4b"]'

# Run layer-level benchmarks (30 iterations)
run-layer-benchmarks:
	uv run python scripts/run_layer_benchmarks.py --num_iterations 30

## Speed benchmarks — 128GB+ only (e.g. M5 Max 128GB)

# Run 128GB-only models, all dtypes (3 iterations)
run-llm-benchmarks-128gb:
	uv run python scripts/run_llm_benchmarks.py \
		--num_iterations 3 \
		--run_only_benchmarks $(MODELS_128GB)

# Run Qwen 3.5 27B/35B-A3B bf16 only (3 iterations, 128GB+)
run-llm-benchmarks-128gb-qwen35-bf16:
	uv run python scripts/run_llm_benchmarks.py \
		--num_iterations 3 \
		--dtypes '["bfloat16"]' \
		--run_only_benchmarks '["qwen-3.5-27b","qwen-3.5-35b-a3b"]'

## Quality benchmarks — all hardware

# Run quality benchmarks, easy difficulty (Qwen 3.5, all dtypes)
run-quality-benchmarks:
	uv run python scripts/run_quality_benchmarks.py \
		--num_runs 3 \
		--run_only_benchmarks $(MODELS_QWEN35)

# Run quality benchmarks, hard difficulty (Qwen 3.5 4B+, all dtypes)
run-quality-benchmarks-hard:
	uv run python scripts/run_quality_benchmarks.py \
		--difficulty hard \
		--num_runs 3 \
		--run_only_benchmarks $(MODELS_QWEN35_LARGE)

# Run quality benchmarks, expert difficulty (Qwen 3.5 4B+, all dtypes)
run-quality-benchmarks-expert:
	uv run python scripts/run_quality_benchmarks.py \
		--difficulty expert \
		--num_runs 3 \
		--run_only_benchmarks $(MODELS_QWEN35_LARGE)

## Quality benchmarks — 128GB+ only (e.g. M5 Max 128GB)

# Run quality benchmarks, easy difficulty (128GB-only models, all dtypes)
run-quality-benchmarks-128gb:
	uv run python scripts/run_quality_benchmarks.py \
		--num_runs 3 \
		--run_only_benchmarks $(MODELS_128GB)

# Run quality benchmarks, hard difficulty (128GB-only models, all dtypes)
run-quality-benchmarks-128gb-hard:
	uv run python scripts/run_quality_benchmarks.py \
		--difficulty hard \
		--num_runs 3 \
		--run_only_benchmarks $(MODELS_128GB)

# Run quality benchmarks, expert difficulty (128GB-only models, all dtypes)
run-quality-benchmarks-128gb-expert:
	uv run python scripts/run_quality_benchmarks.py \
		--difficulty expert \
		--num_runs 3 \
		--run_only_benchmarks $(MODELS_128GB)

## Visualization

# Visualize LLM benchmark results in browser
show-llm:
	uv run python scripts/visualize_llm_benchmarks.py --show_all_measurements
	open visualizations/index.html

# Visualize layer benchmark results in browser
show-layer:
	uv run python scripts/visualize_layer_benchmarks.py --show_all_measurements
	open visualizations/index.html

## Testing

# Run unit tests with coverage
test:
	uv run pytest --cov --cov-report=term-missing --cov-report=html --disable-warnings -v

## Downloading and converting models

# HuggingFace source models for Nemotron
HF_NEMOTRON_MODELS = \
	nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
	nvidia/Nemotron-Cascade-2-30B-A3B

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
		uv run hf download $$model || echo "WARNING: Failed to download $$model"; \
	done
	@echo "Downloading Nemotron source models from HuggingFace..."
	@for model in $(HF_NEMOTRON_MODELS); do \
		echo "\n=== Downloading $$model ==="; \
		uv run hf download $$model || echo "WARNING: Failed to download $$model"; \
	done
	@echo "\nAll downloads complete. Run 'make convert-models' to convert to MLX."

# Helper: convert a model, skipping if the output directory already exists.
# Usage: $(call convert_model,HF_PATH,MLX_PATH,EXTRA_ARGS)
define convert_model
	@if [ -d "$(2)" ]; then \
		echo "  SKIP $(2) (already exists)"; \
	else \
		echo "  CONVERT $(2)"; \
		uv run mlx_lm.convert --hf-path $(1) --mlx-path $(2) $(3); \
	fi
endef

# Convert all downloaded models to MLX quantized formats
# Creates models/ directory with int4, int8, and bf16 variants
convert-models: convert-qwen35-small convert-qwen35-medium convert-qwen35-large convert-qwen35-distilled convert-nemotron
	@echo "\nAll conversions complete. Models are in ./models/"

# Small models (0.8B): int4 + int8 only
convert-qwen35-small:
	@echo "Converting Qwen3.5-0.8B..."
	$(call convert_model,Qwen/Qwen3.5-0.8B,models/Qwen3.5-0.8B-4bit,-q --q-bits 4)
	$(call convert_model,Qwen/Qwen3.5-0.8B,models/Qwen3.5-0.8B-8bit,-q --q-bits 8)

# Medium models (2B, 4B, 9B): int4 + int8 + bf16
convert-qwen35-medium:
	@echo "Converting Qwen3.5-2B..."
	$(call convert_model,Qwen/Qwen3.5-2B,models/Qwen3.5-2B-4bit,-q --q-bits 4)
	$(call convert_model,Qwen/Qwen3.5-2B,models/Qwen3.5-2B-8bit,-q --q-bits 8)
	$(call convert_model,Qwen/Qwen3.5-2B,models/Qwen3.5-2B-bf16,)
	@echo "Converting Qwen3.5-4B..."
	$(call convert_model,Qwen/Qwen3.5-4B,models/Qwen3.5-4B-4bit,-q --q-bits 4)
	$(call convert_model,Qwen/Qwen3.5-4B,models/Qwen3.5-4B-8bit,-q --q-bits 8)
	$(call convert_model,Qwen/Qwen3.5-4B,models/Qwen3.5-4B-bf16,)
	@echo "Converting Qwen3.5-9B..."
	$(call convert_model,Qwen/Qwen3.5-9B,models/Qwen3.5-9B-4bit,-q --q-bits 4)
	$(call convert_model,Qwen/Qwen3.5-9B,models/Qwen3.5-9B-8bit,-q --q-bits 8)
	$(call convert_model,Qwen/Qwen3.5-9B,models/Qwen3.5-9B-bf16,)

# Large models (27B, 35B-A3B): int4 + int8 only
convert-qwen35-large:
	@echo "Converting Qwen3.5-27B..."
	$(call convert_model,Qwen/Qwen3.5-27B,models/Qwen3.5-27B-4bit,-q --q-bits 4)
	$(call convert_model,Qwen/Qwen3.5-27B,models/Qwen3.5-27B-8bit,-q --q-bits 8)
	@echo "Converting Qwen3.5-35B-A3B..."
	$(call convert_model,Qwen/Qwen3.5-35B-A3B,models/Qwen3.5-35B-A3B-4bit,-q --q-bits 4)
	$(call convert_model,Qwen/Qwen3.5-35B-A3B,models/Qwen3.5-35B-A3B-8bit,-q --q-bits 8)

# Nemotron models
convert-nemotron:
	@echo "Converting Nemotron-3-Nano-4B..."
	$(call convert_model,nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16,models/Nemotron-3-Nano-4B-4bit,-q --q-bits 4)
	$(call convert_model,nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16,models/Nemotron-3-Nano-4B-8bit,-q --q-bits 8)
	@echo "Converting Nemotron-Cascade-2-30B-A3B..."
	$(call convert_model,nvidia/Nemotron-Cascade-2-30B-A3B,models/Nemotron-Cascade-2-30B-A3B-4bit,-q --q-bits 4)
	$(call convert_model,nvidia/Nemotron-Cascade-2-30B-A3B,models/Nemotron-Cascade-2-30B-A3B-8bit,-q --q-bits 8)

# Distilled model: int4 + int8
convert-qwen35-distilled:
	@echo "Converting Qwen3.5-27B-Claude-Opus-Distilled..."
	$(call convert_model,Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled,models/Qwen3.5-27B-Claude-Opus-Distilled-4bit,-q --q-bits 4)
	$(call convert_model,Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled,models/Qwen3.5-27B-Claude-Opus-Distilled-8bit,-q --q-bits 8)

## Downloading and converting 128GB+ models

# Convert Qwen 3.5 27B/35B-A3B to bf16 (128GB+ only)
convert-models-128gb: convert-qwen35-large-bf16
	@echo "\n128GB+ model conversions complete. Models are in ./models/"

# Qwen 3.5 27B + 35B-A3B bf16 variants (128GB+ only)
convert-qwen35-large-bf16:
	@echo "Converting Qwen3.5-27B and 35B-A3B to bf16 (128GB+ only)..."
	$(call convert_model,Qwen/Qwen3.5-27B,models/Qwen3.5-27B-bf16,)
	$(call convert_model,Qwen/Qwen3.5-35B-A3B,models/Qwen3.5-35B-A3B-bf16,)
