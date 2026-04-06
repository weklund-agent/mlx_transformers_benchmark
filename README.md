# MLX LLM Benchmarks for Agentic Coding on Apple Silicon

[![tests-Mac](https://github.com/weklund-agent/mlx_transformers_benchmark/actions/workflows/tests-mac.yaml/badge.svg)](https://github.com/weklund-agent/mlx_transformers_benchmark/actions/workflows/tests-mac.yaml)

Which local LLM runs best for coding on your Mac? Speed and quality benchmarks for MLX models, tested on real Apple hardware.

<!-- BEGIN BENCHMARK TABLE -->

> MLX Metal | int4 quantization | April 2026
> Speed: 1024 prompt tokens, 100 generated tokens
> Quality: 81 problems across coding, reasoning, tool calling, math, writing (3 runs each, majority vote)

### Best Models by Hardware

| Hardware | Best Overall | Best Fast | Best Coder |
|---|---|---|---|
| **M4 Pro 64GB** | Gemma 4 31B-it (13 tok/s, 72.2%) | Qwen 2.5-Coder-0.5B (420 tok/s) | Qwen 3.5-27B (12 tok/s, 68.6%) |
| **M5 Max 128GB** | Qwen 3.5-27B (25 tok/s, 100.0%) | Gemma 4 E2B-it (205 tok/s, 96.4%) | Gemma 4 E2B-it (205 tok/s, 96.4%) |

### M4 Pro 64GB

| Model | Arch | Gen tok/s | Quality | Coding | Tool Calling | Reasoning | Memory | Min HW |
|---|---|---:|---:|---|---|---|---:|---|
| Qwen 2.5-Coder-0.5B | 0.5B dense | **420** | -- | -- | -- | -- | 1.1 GiB | Any Mac |
| Qwen 2.5-0.5B-it | 0.5B dense | **380** | -- | -- | -- | -- | 1.1 GiB | Any Mac |
| Qwen 3-0.6B-it | 0.6B dense | **334** | -- | -- | -- | -- | 1.3 GiB | Any Mac |
| Qwen 3.5-0.8B | 0.8B dense | 275 | 28.6% (raw 34.3%) | 10/21 | 12/40 | 7/23 | 2.5 GiB | Any Mac |
| Gemma 3-1B-it | 1B dense | 252 | -- | -- | -- | -- | 1.6 GiB | Any Mac |
| Gemma 3-1B-it QAT | 1B dense | 250 | -- | -- | -- | -- | 1.6 GiB | Any Mac |
| Qwen 3.5-2B | 2B dense | 142 | 41.2% (raw 50.5%) | 13/21 | 9/40 | 18/23 | 3.3 GiB | Any Mac |
| Gemma 4 E2B-it | 2.3B dense | 121 | 65.3% (raw 70.5%) | 18/21 | 25/40 | 20/23 | 3.5 GiB | Any Mac |
| Qwen 2.5-Coder-3B | 3B dense | 117 | 46.9% (raw 55.2%) | 17/21 | 18/40 | 16/23 | 2.6 GiB | Any Mac |
| LFM2-24B-A2B | 2B MoE | 117 | 67.3% (raw 73.3%) | 17/21 | 30/40 | 20/23 | 14.2 GiB | 24GB+ |
| Qwen 2.5-3B-it | 3B dense | 111 | 37.6% (raw 47.6%) | 15/21 | 11/40 | 17/23 | 2.6 GiB | Any Mac |
| Nemotron-3-Nano-4B | 4B dense | 102 | -- | -- | -- | -- | 4.5 GiB | Any Mac |
| Gemma 3-4B-it | 4B dense | 88 | 44.5% (raw 55.2%) | 18/21 | 8/40 | 18/23 | 3.5 GiB | Any Mac |
| Gemma 3-4B-it QAT | 4B dense | 88 | 45.7% (raw 53.3%) | 17/21 | 10/40 | 16/23 | 3.5 GiB | Any Mac |
| Qwen3-Coder-30B-A3B | 3B MoE | 80 | 65.7% (raw 71.4%) | 19/21 | 26/40 | 19/23 | 17.8 GiB | 24GB+ |
| Gemma 4 E4B-it | 4.5B dense | 69 | 50.6% (raw 58.1%) | 18/21 | 14/40 | 18/23 | 5.0 GiB | Any Mac |
| Qwen 3.5-4B | 4B dense | 67 | 50.6% (raw 58.1%) | 16/21 | 14/40 | 19/23 | 4.9 GiB | Any Mac |
| Gemma 4 26B-A4B-it | 3.8B MoE | 65 | 64.1% (raw 68.6%) | 12/21 | 29/40 | 21/23 | 15.3 GiB | 24GB+ |
| GLM-4.7-Flash | 3B MoE | 62 | 45.3% (raw 49.5%) | 14/21 | 21/40 | 9/23 | 17.6 GiB | 24GB+ |
| DeepSeek-R1-Distill-7B | 7B dense | 56 | -- | -- | -- | -- | 5.1 GiB | Any Mac |
| Qwen 3-8B-it | 8B dense | 51 | -- | -- | -- | -- | 5.4 GiB | Any Mac |
| DeepSeek-R1-0528-Qwen3-8B | 8B dense | 51 | 17.1% (raw 21.0%) | 8/21 | 5/40 | 3/23 | 5.4 GiB | Any Mac |
| Nemotron-Nano-9B-v2 | 9B dense | 47 | 42.9% (raw 51.4%) | 12/21 | 9/40 | 20/23 | 8.1 GiB | 16GB+ |
| Qwen 3.5-9B | 9B dense | 39 | 61.6% (raw 68.6%) | 17/21 | 21/40 | 21/23 | 7.3 GiB | Any Mac |
| Gemma 3-12B-it QAT | 12B dense | 32 | -- | -- | -- | -- | 8.2 GiB | 16GB+ |
| Qwen 3-14B-it | 14B dense | 29 | -- | -- | -- | -- | 9.1 GiB | 16GB+ |
| Qwen 3.5-35B-A3B | 3B MoE | 26 | 66.9% (raw 72.4%) | 17/21 | 25/40 | 21/23 | 21.9 GiB | 24GB+ |
| Qwen 3.5-27B Opus Distilled | 27B dense | 16 | -- | -- | -- | -- | 16.9 GiB | 24GB+ |
| Gemma 4 31B-it | 31B dense | 13 | **72.2%** (raw 76.2%) | 15/21 | 32/40 | 22/23 | 18.9 GiB | 24GB+ |
| Qwen 3.5-27B | 27B dense | 12 | 68.6% (raw 74.3%) | 19/21 | 28/40 | 20/23 | 18.8 GiB | 24GB+ |

<details>
<summary>int8 speed results</summary>

| Model | Arch | Gen tok/s | Prefill tok/s | Memory | Min HW |
|---|---|---:|---:|---:|---|
| Qwen 2.5-Coder-0.5B | 0.5B dense | **272.1** | 7032 | 1.3 GiB | Any Mac |
| Qwen 2.5-0.5B-it | 0.5B dense | **272.0** | 6790 | 1.3 GiB | Any Mac |
| Qwen 3-0.6B-it | 0.6B dense | **237.8** | 5183 | 1.6 GiB | Any Mac |
| Qwen 3.5-0.8B | 0.8B dense | **195.0** | 3494 | 2.9 GiB | Any Mac |
| Gemma 3-1B-it QAT | 1B dense | 170.8 | 3722 | 2.2 GiB | Any Mac |
| Gemma 3-1B-it | 1B dense | 168.8 | 3560 | 2.2 GiB | Any Mac |
| Qwen 3.5-2B | 2B dense | 93.0 | 1668 | 4.0 GiB | Any Mac |
| Gemma 4 E2B-it | 2.3B dense | 77.9 | 4265 | 5.8 GiB | Any Mac |
| LFM2-24B-A2B | 2B MoE | 75.4 | 1186 | 25.9 GiB | 36GB+ |
| Qwen 2.5-Coder-3B | 3B dense | 71.8 | 1093 | 4.0 GiB | Any Mac |
| Qwen 2.5-3B-it | 3B dense | 67.6 | 1135 | 4.0 GiB | Any Mac |
| Nemotron-3-Nano-4B | 4B dense | 57.9 | 824 | 6.5 GiB | Any Mac |
| Qwen3-Coder-30B-A3B | 3B MoE | 54.4 | 865 | 33.1 GiB | 48GB+ |
| Gemma 3-4B-it QAT | 4B dense | 52.4 | 911 | 5.6 GiB | Any Mac |
| Gemma 3-4B-it | 4B dense | 51.9 | 887 | 5.6 GiB | Any Mac |
| Gemma 4 26B-A4B-it | 3.8B MoE | 45.5 | 785 | 27.7 GiB | 36GB+ |
| Qwen 3.5-4B | 4B dense | 43.1 | 684 | 6.8 GiB | Any Mac |
| GLM-4.7-Flash | 3B MoE | 42.8 | 674 | 32.5 GiB | 48GB+ |
| Gemma 4 E4B-it | 4.5B dense | 42.2 | 1229 | 8.7 GiB | 16GB+ |
| Qwen 3-8B-it | 8B dense | 29.7 | 454 | 9.5 GiB | 16GB+ |
| DeepSeek-R1-0528-Qwen3-8B | 8B dense | 29.4 | 450 | 9.5 GiB | 16GB+ |
| Qwen 3.5-9B | 9B dense | 24.4 | 383 | 11.7 GiB | 16GB+ |
| Qwen 3.5-35B-A3B | 3B MoE | 22.2 | 716 | 39.2 GiB | 48GB+ |
| Gemma 3-12B-it QAT | 12B dense | 18.5 | 280 | 14.6 GiB | 24GB+ |
| Qwen 3-14B-it | 14B dense | 15.9 | 237 | 16.4 GiB | 24GB+ |
| Qwen 3.5-27B | 27B dense | 7.3 | 108 | 32.0 GiB | 48GB+ |
| Gemma 4 31B-it | 31B dense | 7.0 | 99 | 34.1 GiB | 48GB+ |

</details>

### M5 Max 128GB

| Model | Arch | Gen tok/s | Quality | Coding | Tool Calling | Reasoning | Memory | Min HW |
|---|---|---:|---:|---|---|---|---:|---|
| Qwen 2.5-Coder-0.5B | 0.5B dense | **611** | -- | -- | -- | -- | 1.1 GiB | Any Mac |
| Qwen 2.5-0.5B-it | 0.5B dense | **542** | -- | -- | -- | -- | 1.1 GiB | Any Mac |
| Qwen 3-0.6B-it | 0.6B dense | **527** | -- | -- | -- | -- | 1.3 GiB | Any Mac |
| Qwen 3.5-0.8B | 0.8B dense | 409 | -- | -- | -- | -- | 2.5 GiB | Any Mac |
| Gemma 3-1B-it QAT | 1B dense | 373 | -- | -- | -- | -- | 1.6 GiB | Any Mac |
| Gemma 3-1B-it | 1B dense | 371 | -- | -- | -- | -- | 1.6 GiB | Any Mac |
| Qwen 3.5-2B | 2B dense | 235 | -- | -- | -- | -- | 3.3 GiB | Any Mac |
| Qwen 2.5-Coder-3B | 3B dense | 226 | -- | -- | -- | -- | 2.6 GiB | Any Mac |
| Qwen 2.5-3B-it | 3B dense | 209 | -- | -- | -- | -- | 2.6 GiB | Any Mac |
| Gemma 4 E2B-it | 2.3B dense | 205 | 96.4% (raw 95.7%) | 13/13 | 4/5 | 13/13 | 3.5 GiB | Any Mac |
| LFM2-24B-A2B | 2B MoE | 180 | 89.2% (raw 93.5%) | 13/13 | 5/5 | 12/13 | 14.2 GiB | 24GB+ |
| Gemma 3-4B-it QAT | 4B dense | 172 | -- | -- | -- | -- | 3.5 GiB | Any Mac |
| Gemma 3-4B-it | 4B dense | 171 | -- | -- | -- | -- | 3.5 GiB | Any Mac |
| Qwen 3.5-4B | 4B dense | 131 | -- | -- | -- | -- | 4.9 GiB | Any Mac |
| Gemma 4 E4B-it | 4.5B dense | 130 | 92.8% (raw 95.7%) | 13/13 | 5/5 | 12/13 | 5.0 GiB | Any Mac |
| Qwen3-Coder-30B-A3B | 3B MoE | 129 | 89.2% (raw 91.3%) | 13/13 | 5/5 | 11/13 | 17.8 GiB | 24GB+ |
| DeepSeek-R1-Distill-7B | 7B dense | 114 | -- | -- | -- | -- | 5.1 GiB | Any Mac |
| Gemma 4 26B-A4B-it | 3.8B MoE | 110 | 92.8% (raw 95.7%) | 13/13 | 5/5 | 13/13 | 15.3 GiB | 24GB+ |
| DeepSeek-R1-0528-Qwen3-8B | 8B dense | 104 | -- | -- | -- | -- | 5.4 GiB | Any Mac |
| Qwen 3-8B-it | 8B dense | 103 | -- | -- | -- | -- | 5.4 GiB | Any Mac |
| GLM-4.7-Flash | 3B MoE | 96 | 68.7% (raw 78.3%) | 13/13 | 5/5 | 8/13 | 17.6 GiB | 24GB+ |
| Qwen 3.5-9B | 9B dense | 79 | -- | -- | -- | -- | 7.3 GiB | Any Mac |
| Nemotron-Nano-9B-v2 | 9B dense | 67 | -- | -- | -- | -- | 8.1 GiB | 16GB+ |
| Qwen 3-14B-it | 14B dense | 60 | -- | -- | -- | -- | 9.1 GiB | 16GB+ |
| Gemma 3-12B-it QAT | 12B dense | 48 | -- | -- | -- | -- | 8.2 GiB | 16GB+ |
| Qwen 3.5-35B-A3B | 3B MoE | 44 | 87.5% | 2/3 | -- | 3/3 | 21.9 GiB | 24GB+ |
| Qwen 3.5-27B Opus Distilled | 27B dense | 28 | -- | -- | -- | -- | 16.9 GiB | 24GB+ |
| Qwen 3.5-27B | 27B dense | 25 | **100.0%** | 3/3 | -- | 3/3 | 18.8 GiB | 24GB+ |
| Gemma 4 31B-it | 31B dense | 17 | 91.6% (raw 93.5%) | 13/13 | 5/5 | 12/13 | 18.9 GiB | 24GB+ |

<details>
<summary>int8 speed results</summary>

| Model | Arch | Gen tok/s | Prefill tok/s | Memory | Min HW |
|---|---|---:|---:|---:|---|
| Qwen 3-0.6B-it | 0.6B dense | **422.1** | 16697 | 1.6 GiB | Any Mac |
| Qwen 2.5-Coder-0.5B | 0.5B dense | **416.4** | 20203 | 1.3 GiB | Any Mac |
| Qwen 2.5-0.5B-it | 0.5B dense | **410.5** | 18370 | 1.3 GiB | Any Mac |
| Qwen 3.5-0.8B | 0.8B dense | 290.5 | 9616 | 2.9 GiB | Any Mac |
| Gemma 3-1B-it | 1B dense | 277.4 | 14773 | 2.2 GiB | Any Mac |
| Gemma 3-1B-it QAT | 1B dense | 276.3 | 14428 | 2.2 GiB | Any Mac |
| Qwen 3.5-2B | 2B dense | 150.9 | 5177 | 4.0 GiB | Any Mac |
| Gemma 4 E2B-it | 2.3B dense | 146.9 | 14893 | 5.8 GiB | Any Mac |
| Qwen 2.5-Coder-3B | 3B dense | 143.9 | 6504 | 4.0 GiB | Any Mac |
| Qwen 2.5-3B-it | 3B dense | 133.6 | 6342 | 4.0 GiB | Any Mac |
| LFM2-24B-A2B | 2B MoE | 132.0 | 5297 | 25.9 GiB | 36GB+ |
| Gemma 3-4B-it QAT | 4B dense | 104.1 | 5157 | 5.6 GiB | Any Mac |
| Gemma 3-4B-it | 4B dense | 103.5 | 5087 | 5.6 GiB | Any Mac |
| Qwen3-Coder-30B-A3B | 3B MoE | 96.1 | 3353 | 33.1 GiB | 48GB+ |
| Gemma 4 26B-A4B-it | 3.8B MoE | 85.3 | 3208 | 27.7 GiB | 36GB+ |
| Qwen 3.5-4B | 4B dense | 85.3 | 2584 | 6.8 GiB | Any Mac |
| Gemma 4 E4B-it | 4.5B dense | 84.7 | 6374 | 8.7 GiB | 16GB+ |
| GLM-4.7-Flash | 3B MoE | 71.7 | 2565 | 32.5 GiB | 48GB+ |
| Qwen 3-8B-it | 8B dense | 62.6 | 2818 | 9.5 GiB | 16GB+ |
| DeepSeek-R1-0528-Qwen3-8B | 8B dense | 61.7 | 2719 | 9.5 GiB | 16GB+ |
| Qwen 3.5-9B | 9B dense | 48.8 | 1511 | 11.7 GiB | 16GB+ |
| Qwen 3.5-35B-A3B | 3B MoE | 44.6 | 2151 | 39.2 GiB | 48GB+ |
| Qwen 3-14B-it | 14B dense | 33.6 | 1426 | 16.4 GiB | 24GB+ |
| Gemma 3-12B-it QAT | 12B dense | 32.2 | 1060 | 14.6 GiB | 24GB+ |
| Qwen 3.5-27B Opus Distilled | 27B dense | 17.4 | 434 | 30.3 GiB | 48GB+ |
| Qwen 3.5-27B | 27B dense | 14.5 | 480 | 32.0 GiB | 48GB+ |
| Gemma 4 31B-it | 31B dense | 9.4 | 544 | 34.1 GiB | 48GB+ |

</details>

<details>
<summary><h3>M4 Pro 24GB (legacy)</h3></summary>

| Model | Arch | Gen tok/s | Prefill tok/s | Memory | Min HW |
|---|---|---:|---:|---:|---|
| Qwen 2.5-0.5B-it | 0.5B dense | **376** | 5530 | 1.4 GiB | Any Mac |
| Qwen 2.5-Coder-0.5B | 0.5B dense | **371** | 5464 | 1.4 GiB | Any Mac |
| Qwen 3-0.6B-it | 0.6B dense | **328** | 4277 | 1.6 GiB | Any Mac |
| Gemma 3-1B-it QAT | 1B dense | 222 | 2858 | 2.0 GiB | Any Mac |
| Gemma 3-1B-it | 1B dense | 222 | 2866 | 2.0 GiB | Any Mac |
| Qwen 2.5-Coder-3B | 3B dense | 118 | 1050 | 2.9 GiB | Any Mac |
| Qwen 2.5-3B-it | 3B dense | 112 | 1051 | 2.9 GiB | Any Mac |
| Gemma 3-4B-it QAT | 4B dense | 84 | 756 | 3.9 GiB | Any Mac |
| Gemma 3-4B-it | 4B dense | 81 | 739 | 3.9 GiB | Any Mac |
| DeepSeek-R1-Distill-7B | 7B dense | 57 | 476 | 5.3 GiB | Any Mac |
| Qwen 3-8B-it | 8B dense | 52 | 414 | 5.6 GiB | Any Mac |
| DeepSeek-R1-0528-Qwen3-8B | 8B dense | 52 | 416 | 5.6 GiB | Any Mac |
| Gemma 3-12B-it QAT | 12B dense | 31 | 256 | 8.6 GiB | 16GB+ |
| Qwen 3-14B-it | 14B dense | 29 | 224 | 9.3 GiB | 16GB+ |

<details>
<summary>int8 speed results</summary>

| Model | Arch | Gen tok/s | Prefill tok/s | Memory | Min HW |
|---|---|---:|---:|---:|---|
| Qwen 2.5-0.5B-it | 0.5B dense | **272.6** | 5745 | 1.6 GiB | Any Mac |
| Qwen 2.5-Coder-0.5B | 0.5B dense | **270.9** | 5717 | 1.6 GiB | Any Mac |
| Qwen 3-0.6B-it | 0.6B dense | **239.1** | 4323 | 1.8 GiB | Any Mac |
| Gemma 3-1B-it QAT | 1B dense | 156.1 | 2861 | 2.7 GiB | Any Mac |
| Gemma 3-1B-it | 1B dense | 155.4 | 2850 | 2.7 GiB | Any Mac |
| Qwen 2.5-Coder-3B | 3B dense | 72.0 | 1052 | 4.2 GiB | Any Mac |
| Qwen 2.5-3B-it | 3B dense | 67.2 | 1046 | 4.2 GiB | Any Mac |
| Gemma 3-4B-it QAT | 4B dense | 50.8 | 808 | 6.1 GiB | Any Mac |
| Gemma 3-4B-it | 4B dense | 50.5 | 744 | 6.1 GiB | Any Mac |
| Qwen 3-8B-it | 8B dense | 29.7 | 413 | 9.7 GiB | 16GB+ |
| DeepSeek-R1-0528-Qwen3-8B | 8B dense | 29.6 | 413 | 9.7 GiB | 16GB+ |
| Gemma 3-12B-it QAT | 12B dense | 18.2 | 254 | 15.0 GiB | 24GB+ |

</details>

</details>


<!-- END BENCHMARK TABLE -->

## Reading the Table

**Speed** is measured as generation tokens/sec at int4 quantization with a 1024-token prompt. **Bold** tok/s values are within 70% of the fastest model on that hardware.

**Quality** is a weighted score across 81 problems designed for agentic coding. Harder problems count more: Easy (1x), Hard (2x), Expert (3x), Tool Calling (3x). When you see `67.5% (raw 73.0%)`, the first number is the weighted score and the parenthetical is the flat pass rate. Each problem runs 3 times with majority vote. **Coding**, **Tool Calling**, and **Reasoning** columns show raw pass counts for those categories.

**Min HW** is the minimum RAM needed to run the model at int4 without swapping.

For full scoring methodology, see [QUALITY_METHODOLOGY.md](QUALITY_METHODOLOGY.md).


## Quick Start

```bash
git clone git@github.com:weklund-agent/mlx_transformers_benchmark.git
cd mlx_transformers_benchmark
make setup

# Run speed benchmarks
uv run python scripts/run_llm_benchmarks.py \
    --run_only_benchmarks '["gemma-4-e2b-it"]' \
    --dtypes '["int4"]' --num_iterations 3

# Run quality benchmarks
uv run python scripts/run_quality_benchmarks.py \
    --difficulty all \
    --run_only_benchmarks '["gemma-4-e2b-it"]' \
    --dtypes '["int4"]' --num_runs 3

# Update the README table
uv run python scripts/update_readme_table.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full setup instructions and how to submit results.


## Related

- **[mlx-stack](https://github.com/weklund/mlx-stack)** -- runs multiple MLX models behind a single OpenAI-compatible endpoint. This benchmark suite feeds its model catalog.
- **[QUALITY_METHODOLOGY.md](QUALITY_METHODOLOGY.md)** -- detailed breakdown of the quality benchmark suite, scoring formula, and problem categories.
- **[CONTRIBUTING.md](CONTRIBUTING.md)** -- how to add models, run benchmarks, and submit measurements.
