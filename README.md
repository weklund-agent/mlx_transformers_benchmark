# Benchmarking transformer operators on Apple silicon

[![tests-Mac](https://github.com/weklund-agent/mlx_transformers_benchmark/actions/workflows/tests-mac.yaml/badge.svg)](https://github.com/weklund-agent/mlx_transformers_benchmark/actions/workflows/tests-mac.yaml)

Let's say you're interested in performing LLM inference on Apple hardware. You care about speed, but don't know which model or framework to pick.

Do you:
- use [PyTorch with the Metal Performance Shaders backend](https://pytorch.org/docs/stable/notes/mps.html),
- use Apple's [MLX, built directly for Metal](https://github.com/ml-explore/mlx),
- use `LM Studio` and its `llama.cpp` engine for Metal, 
- use `Ollama`,
- or use `llama.cpp` directly?

We aim to help you make this choice, by benchmarking inference for a few common models and operators. 
Results can be found at 
[https://weklund-agent.github.io/mlx_transformers_benchmark/](https://weklund-agent.github.io/mlx_transformers_benchmark/).


## Agentic Coding Model Benchmarks (MLX on Apple Silicon)

<!-- BEGIN BENCHMARK TABLE -->

> MLX Metal | int4 quantization | April 2026
> Speed: 1024 prompt tokens, 100 generated tokens
> Quality: 46 problems across coding, reasoning, tool calling, math, writing (3 runs each, majority vote)

### Best Models by Hardware

| Hardware | Best Overall | Best Fast | Best Coder |
|---|---|---|---|
| **M4 Pro 64GB** | Qwen 3.5-35B-A3B (26 tok/s, 100.0%) | Gemma 4 E2B-it (121 tok/s, 95.7%) | Qwen 3.5-27B (12 tok/s, 100.0%) |
| **M5 Max 128GB** | Qwen 3.5-27B Opus Distilled (28 tok/s, 100.0%) | Qwen 3.5-27B (25 tok/s, 100.0%) | Qwen 3.5-27B (25 tok/s, 100.0%) |

### M4 Pro 64GB

| Model | Arch | Gen tok/s | Quality | Coding | Tool Calling | Reasoning | Memory | Min HW |
|---|---|---:|---:|---|---|---|---:|---|
| Qwen 2.5-Coder-0.5B | 0.5B dense | **420** | -- | -- | -- | -- | 1.1 GiB | Any Mac |
| Qwen 2.5-0.5B-it | 0.5B dense | **380** | -- | -- | -- | -- | 1.1 GiB | Any Mac |
| Qwen 3-0.6B-it | 0.6B dense | **334** | -- | -- | -- | -- | 1.3 GiB | Any Mac |
| Qwen 3.5-0.8B | 0.8B dense | 275 | 70.7% | 11/13 | -- | 8/13 | 2.5 GiB | Any Mac |
| Gemma 3-1B-it | 1B dense | 252 | -- | -- | -- | -- | 1.6 GiB | Any Mac |
| Gemma 3-1B-it QAT | 1B dense | 250 | -- | -- | -- | -- | 1.6 GiB | Any Mac |
| Qwen 3.5-2B | 2B dense | 142 | 85.4% | 13/13 | -- | 9/13 | 3.3 GiB | Any Mac |
| Gemma 4 E2B-it | 2.3B dense | 121 | 95.7% | 13/13 | 4/5 | 13/13 | 3.5 GiB | Any Mac |
| Qwen 2.5-Coder-3B | 3B dense | 117 | -- | -- | -- | -- | 2.6 GiB | Any Mac |
| LFM2-24B-A2B | 2B MoE | 117 | 95.7% | 13/13 | 5/5 | 12/13 | 14.2 GiB | 24GB+ |
| Qwen 2.5-3B-it | 3B dense | 111 | 80.5% | 13/13 | -- | 8/13 | 2.6 GiB | Any Mac |
| Nemotron-3-Nano-4B | 4B dense | 102 | 58.5% | 8/13 | -- | 5/13 | 4.5 GiB | Any Mac |
| Gemma 3-4B-it | 4B dense | 88 | 92.7% | 13/13 | -- | 11/13 | 3.5 GiB | Any Mac |
| Gemma 3-4B-it QAT | 4B dense | 88 | 97.6% | 13/13 | -- | 12/13 | 3.5 GiB | Any Mac |
| Qwen3-Coder-30B-A3B | 3B MoE | 80 | 89.1% | 13/13 | 5/5 | 10/13 | 17.8 GiB | 24GB+ |
| Gemma 4 E4B-it | 4.5B dense | 69 | 95.7% | 13/13 | 5/5 | 12/13 | 5.0 GiB | Any Mac |
| Qwen 3.5-4B | 4B dense | 67 | 97.6% | 12/13 | -- | 13/13 | 4.9 GiB | Any Mac |
| Gemma 4 26B-A4B-it | 3.8B MoE | 65 | 95.7% | 13/13 | 5/5 | 13/13 | 15.3 GiB | 24GB+ |
| GLM-4.7-Flash | 3B MoE | 62 | 73.9% | 12/13 | 5/5 | 7/13 | 17.6 GiB | 24GB+ |
| DeepSeek-R1-Distill-7B | 7B dense | 56 | -- | -- | -- | -- | 5.1 GiB | Any Mac |
| Qwen 3-8B-it | 8B dense | 51 | -- | -- | -- | -- | 5.4 GiB | Any Mac |
| DeepSeek-R1-0528-Qwen3-8B | 8B dense | 51 | -- | -- | -- | -- | 5.4 GiB | Any Mac |
| Nemotron-Nano-9B-v2 | 9B dense | 47 | 97.6% | 13/13 | -- | 12/13 | 8.1 GiB | 16GB+ |
| Qwen 3.5-9B | 9B dense | 39 | 97.6% | 13/13 | -- | 12/13 | 7.3 GiB | Any Mac |
| Gemma 3-12B-it QAT | 12B dense | 32 | -- | -- | -- | -- | 8.2 GiB | 16GB+ |
| Qwen 3-14B-it | 14B dense | 29 | -- | -- | -- | -- | 9.1 GiB | 16GB+ |
| Qwen 3.5-35B-A3B | 3B MoE | 26 | **100.0%** | 13/13 | -- | 13/13 | 21.9 GiB | 24GB+ |
| Qwen 3.5-27B Opus Distilled | 27B dense | 16 | 97.6% | 12/13 | -- | 13/13 | 16.9 GiB | 24GB+ |
| Gemma 4 31B-it | 31B dense | 13 | 95.7% | 13/13 | 5/5 | 12/13 | 18.9 GiB | 24GB+ |
| Qwen 3.5-27B | 27B dense | 12 | **100.0%** | 13/13 | -- | 13/13 | 18.8 GiB | 24GB+ |

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

| Model | Arch | Gen tok/s | Quality | Memory | Min HW |
|---|---|---:|---:|---:|---|
| Qwen 2.5-Coder-0.5B | 0.5B dense | **611** | -- | 1.1 GiB | Any Mac |
| Qwen 2.5-0.5B-it | 0.5B dense | **542** | -- | 1.1 GiB | Any Mac |
| Qwen 3-0.6B-it | 0.6B dense | **527** | -- | 1.3 GiB | Any Mac |
| Qwen 3.5-0.8B | 0.8B dense | 409 | -- | 2.5 GiB | Any Mac |
| Gemma 3-1B-it QAT | 1B dense | 373 | -- | 1.6 GiB | Any Mac |
| Gemma 3-1B-it | 1B dense | 371 | -- | 1.6 GiB | Any Mac |
| Qwen 3.5-2B | 2B dense | 235 | -- | 3.3 GiB | Any Mac |
| Qwen 2.5-Coder-3B | 3B dense | 226 | -- | 2.6 GiB | Any Mac |
| Qwen 2.5-3B-it | 3B dense | 209 | -- | 2.6 GiB | Any Mac |
| Gemma 3-4B-it QAT | 4B dense | 172 | -- | 3.5 GiB | Any Mac |
| Gemma 3-4B-it | 4B dense | 171 | -- | 3.5 GiB | Any Mac |
| Qwen 3.5-4B | 4B dense | 131 | -- | 4.9 GiB | Any Mac |
| DeepSeek-R1-Distill-7B | 7B dense | 114 | -- | 5.1 GiB | Any Mac |
| DeepSeek-R1-0528-Qwen3-8B | 8B dense | 104 | -- | 5.4 GiB | Any Mac |
| Qwen 3-8B-it | 8B dense | 103 | -- | 5.4 GiB | Any Mac |
| Qwen 3.5-9B | 9B dense | 79 | -- | 7.3 GiB | Any Mac |
| Nemotron-Nano-9B-v2 | 9B dense | 67 | -- | 8.1 GiB | 16GB+ |
| Qwen 3-14B-it | 14B dense | 60 | -- | 9.1 GiB | 16GB+ |
| Gemma 3-12B-it QAT | 12B dense | 48 | -- | 8.2 GiB | 16GB+ |
| Qwen 3.5-35B-A3B | 3B MoE | 44 | 87.5% | 21.9 GiB | 24GB+ |
| Qwen 3.5-27B Opus Distilled | 27B dense | 28 | **100.0%** | 16.9 GiB | 24GB+ |
| Qwen 3.5-27B | 27B dense | 25 | **100.0%** | 18.8 GiB | 24GB+ |

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
| Qwen 2.5-Coder-3B | 3B dense | 143.9 | 6504 | 4.0 GiB | Any Mac |
| Qwen 2.5-3B-it | 3B dense | 133.6 | 6342 | 4.0 GiB | Any Mac |
| Gemma 3-4B-it QAT | 4B dense | 104.1 | 5157 | 5.6 GiB | Any Mac |
| Gemma 3-4B-it | 4B dense | 103.5 | 5087 | 5.6 GiB | Any Mac |
| Qwen 3.5-4B | 4B dense | 85.3 | 2584 | 6.8 GiB | Any Mac |
| Qwen 3-8B-it | 8B dense | 62.6 | 2818 | 9.5 GiB | 16GB+ |
| DeepSeek-R1-0528-Qwen3-8B | 8B dense | 61.7 | 2719 | 9.5 GiB | 16GB+ |
| Qwen 3.5-9B | 9B dense | 48.8 | 1511 | 11.7 GiB | 16GB+ |
| Qwen 3.5-35B-A3B | 3B MoE | 44.6 | 2151 | 39.2 GiB | 48GB+ |
| Qwen 3-14B-it | 14B dense | 33.6 | 1426 | 16.4 GiB | 24GB+ |
| Gemma 3-12B-it QAT | 12B dense | 32.2 | 1060 | 14.6 GiB | 24GB+ |
| Qwen 3.5-27B Opus Distilled | 27B dense | 17.4 | 434 | 30.3 GiB | 48GB+ |
| Qwen 3.5-27B | 27B dense | 14.5 | 480 | 32.0 GiB | 48GB+ |

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


## Installation

Before you start, you will need:
 - [`uv`](https://github.com/astral-sh/uv) to manage dependencies, available as [homebrew](https://formulae.brew.sh/formula/uv)

 To (optionally) benchmark `Metal+llama.cpp` models in common interfaces, you may also need:
 - [`LM Studio`](https://lmstudio.ai/)
 - [`Ollama`](https://ollama.com/)

To get started:

1. Clone the repo:
   ```
   git clone git@github.com:weklund-agent/mlx_transformers_benchmark.git
   cd mlx_transformers_benchmark
   ```

2. Set up a python3.11 virtual environment using 
   [`uv`](https://github.com/astral-sh/uv):
   ```
   make setup
   ```

3. For good measure, run the tests. This also tells you whether we can use the GPU.
   ```
   make test
   ```

3. Run benchmarking, here for the 0.5B parameter `Qwen2.5` model:
   ```
   uv run python scripts/run_llm_benchmarks.py \
      --run_only_benchmarks qwen-2.5-0.5b-it \
      --dtypes \["int4","int8"\] \
      --num_iterations 3 
   ```
   This creates a new result in the `measurements` folder.

   Optionally, to run a full benchmark for `bfloat16`, `int8`, `int4` datatypes, you can use:
   ``` 
   make run-llm-benchmarks
   ```
   This will take a longer time however, so make sure you aren't busy!

4. To create a HTML report of all available measurements and open the index page:
   ```
   make show-llm
   ```
   This should open a page similar to 
   [https://weklund-agent.github.io/mlx_transformers_benchmark/](https://weklund-agent.github.io/mlx_transformers_benchmark/).


## Relationship to mlx-stack

This benchmark suite is the **data source** for [mlx-stack](https://github.com/weklund/mlx-stack), a tool that runs multiple LLMs simultaneously on Apple Silicon behind a single OpenAI-compatible endpoint.

### How the two repos work together

```
mlx_transformers_benchmark          mlx-stack
┌─────────────────────────┐         ┌──────────────────────────┐
│  Speed benchmarks       │         │  Model catalog           │
│  (generation_tps,       │  JSON   │  (catalog/*.yaml)        │
│   prompt_tps, memory)   │───────▶ │                          │
│                         │         │  Recommendation engine   │
│  Quality benchmarks     │         │  (scoring.py)            │
│  (pass rates by         │         │                          │
│   category + difficulty)│         │  Hardware-aware model    │
│                         │         │  selection for init/setup │
└─────────────────────────┘         └──────────────────────────┘
```

1. **Benchmark** models here using `make run-llm-benchmarks` and `make run-quality-benchmarks`
2. **Export** aggregated results with `python scripts/export_for_mlx_stack.py`
3. **Import** into mlx-stack's catalog, where the recommendation engine uses generation speed, memory usage, and quality pass rates to decide which models to assign to each tier

### Export script

```bash
# Generate benchmark_data.json from all measurements
python scripts/export_for_mlx_stack.py

# Generate and copy directly into the mlx-stack data directory
python scripts/export_for_mlx_stack.py --copy-to ../mlx-stack/src/mlx_stack/data/
```

The export produces a single JSON keyed by HuggingFace repo ID (e.g. `mlx-community/Qwen3.5-9B-4bit`) containing speed metrics per hardware profile and quality pass rates per category.


## Contributing

If you have an Apple device, additional measurements are always welcome! 
The easiest way to contribute is to 
[fork the repo](https://github.com/weklund-agent/mlx_transformers_benchmark/fork), 
and run benchmarks for common LLMs and/or operators. 

See [CONTRIBUTING.md](./CONTRIBUTING.md) for more info.


### On reproducibility

As Apple machines share memory with other background processes, these benchmarks are not exact, certainly not for Macbooks. 
Still, the numbers should give a decent idea of the performance to expect. 

Although the default parameters do not result in thermal throttling for a Macbook M4 Pro, older
machines may have trouble with the heavier models and operators. We do try to skip large models,
but you may still have too little RAM and fall back on swap space. If you see huge memory pressure 
or outlier measurements, do take a closer look!

> [!NOTE] 
> For a large number of iterations, the GPU will certainly heat up. If needed, you can 
increase the cooldown period using the `cooldown_time_fraction` argument. Monitoring GPU 
temperature programatically requires admin privileges, but you can use third-party apps like 
[stats](https://github.com/exelban/stats), also available as 
[homebrew](https://formulae.brew.sh/cask/stats).


### Notes

Apple silicon is fairly cost-effective for LLM inference due to its unified memory architecture.
As LLM inference is mostly memory-bound for low batch sizes, devices with high memory bandwidth 
typically obtain 
[high tokens/sec in inference benchmarks](https://github.com/ggml-org/llama.cpp/discussions/4167).

This benchmark focuses on the inference time of easy-to-run LLMs and unquantized transformer ops, primarily 
useful when running inference locally, or when finetuning custom models for (or on!) Apple devices. 

You may also be interested in:

- Tristan Bilot's comprehensive benchmark for fundamental operators for `mlx`, 
  `torch+mps`, and `torch+cuda` ([link](https://github.com/TristanBilot/mlx-benchmark)). Placing both `mlx` 
  and `torch` functions in a single benchmark class makes it easy to see the differences between the 
  two, and we adopt the same strategy here.

- [The work of Feng et al.](https://arxiv.org/pdf/2501.14925) comparing training on Nvidia cards vs Apple Silicon. 
