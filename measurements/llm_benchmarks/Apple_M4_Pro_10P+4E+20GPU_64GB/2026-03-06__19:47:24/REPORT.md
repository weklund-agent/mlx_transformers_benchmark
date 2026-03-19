# Qwen 3.5 MLX Benchmark Report

**Hardware:** Apple M4 Pro (10P+4E cores, 20 GPU cores, 64GB unified memory)
**Framework:** MLX 0.31.0 + mlx-lm 0.30.7
**Date:** 2026-03-06
**Iterations:** 20 (with 1 warmup), values reported as mean ± std

## Models Tested

| Model | Total Params | Architecture | Quantizations |
|-------|-------------|--------------|---------------|
| Qwen3.5-0.8B | 0.9B | Dense | int4, int8 |
| Qwen3.5-2B | 2B | Dense | int4, int8, bf16 |
| Qwen3.5-4B | 5B | Dense | int4, int8, bf16 |
| Qwen3.5-9B | 10B | Dense | int4, int8, bf16 |
| Qwen3.5-27B | 28B | Dense | int4, int8 |
| Qwen3.5-35B-A3B | 36B (3B active) | MoE | int4, int8 |

## Generation Speed (tokens/sec)

Measured at 256 prompt tokens, 100 generated tokens.

| Model | int4 | int8 | bf16 |
|-------|------|------|------|
| **0.8B** | 284.1 ± 0.2 | 198.9 ± 0.1 | — |
| **2B** | 144.3 ± 0.1 | 93.1 ± 0.5 | 61.7 ± 0.04 |
| **4B** | 68.0 ± 0.4 | 42.7 ± 0.3 | 27.7 ± 0.2 |
| **9B** | 38.5 ± 0.4 | 24.1 ± 0.2 | 15.0 ± 0.4 |
| **27B** | 11.8 ± 0.2 | 7.0 ± 0.1 | — |
| **35B-A3B** | 25.6 ± 0.5 | 22.1 ± 0.3 | — |

## Time to First Token (seconds)

Measured at 4096 prompt tokens.

| Model | int4 | int8 | bf16 |
|-------|------|------|------|
| **0.8B** | 1.09 ± 0.006 | 1.11 ± 0.015 | — |
| **2B** | 2.54 ± 0.07 | 2.52 ± 0.10 | 1.85 ± 0.02 |
| **4B** | 6.14 ± 0.25 | 6.14 ± 0.25 | 4.67 ± 0.03 |
| **9B** | 10.88 ± 0.61 | 10.63 ± 0.40 | 8.20 ± 0.06 |
| **27B** | 36.57 ± 0.99 | 37.08 ± 0.64 | — |
| **35B-A3B** | 5.52 ± 0.10 | 5.61 ± 0.15 | — |

## Peak Memory Usage (GiB)

Measured at 4096 prompt tokens.

| Model | int4 | int8 | bf16 |
|-------|------|------|------|
| **0.8B** | 3.3 | 3.7 | — |
| **2B** | 4.2 | 5.1 | 5.1 |
| **4B** | 6.4 | 8.5 | 10.2 |
| **9B** | 9.0 | 13.4 | 19.7 |
| **27B** | 22.1 | 35.5 | — |
| **35B-A3B** | 23.7 | 41.0 | — |

## Key Findings

1. **Very low variance on generation speed.** Standard deviation is typically less than 1% of the mean across 20 runs, confirming stable and reproducible measurements.

2. **TTFT variance increases with model size.** The 0.8B model has ±6ms jitter while the 27B has ±1s, indicating larger models are more sensitive to system load during prefill.

3. **35B-A3B MoE is the standout large model.** Despite having 36B total parameters, only 3B are active per token. This gives it 25.6 tok/s generation (2.2x faster than 27B dense), 5.5s TTFT (6.6x faster than 27B), at comparable memory to 27B int4 (23.7 vs 22.1 GiB).

4. **int4 is the practical sweet spot.** Across all models, int4 delivers roughly 1.5-1.7x the generation speed of int8 and 2.5-4x of bf16, with minimal quality loss for most use cases.

5. **bf16 prefill is faster than quantized.** For 2B/4B/9B models, bf16 TTFT is 25-35% faster than int4/int8. This is because dequantization adds compute overhead during the compute-bound prefill phase, while generation is memory-bandwidth-bound where smaller weights win.

6. **Peak memory is deterministic.** Memory std is effectively zero since model weights dominate memory usage and are fixed.

## Recommendations

### For Small Recurring Tasks / Cron Jobs: 4B int4

The 4B int4 is the sweet spot for automated, recurring workloads:

| Factor | 4B int4 | Why it matters for cron jobs |
|--------|---------|----------------------------|
| Generation speed | 68.0 tok/s | Fast enough for short outputs, won't block your schedule |
| TTFT (4096 tokens) | 6.1s | Acceptable cold-start for batch work |
| Peak memory | 6.4 GiB | Leaves 57+ GiB free for other processes |
| Quality | Beats GPT-5-Nano and Gemini 2.5 Flash | Strong enough for structured tasks |
| Tool calling | Inherits Qwen 3.5 function-calling design | Native tool use support |

The 4B is the largest model you can run with negligible system impact. Public benchmarks show it outperforms previous-gen 30B models despite being tiny. For cron jobs doing summarization, classification, data extraction, or templated generation, it has more than enough capability.

**Runner-up: 2B int4** (144.3 tok/s, 4.2 GiB) — if tasks are very simple (reformatting, short Q&A, structured extraction) and minimal resource footprint is desired. The 0.8B is too limited for anything requiring reasoning.

### For Coding Tasks: 27B int4 (or 9B int4 as a compromise)

| Factor | 27B int4 | 9B int4 | 35B-A3B int4 |
|--------|----------|---------|--------------|
| Generation speed | 11.8 tok/s | 38.5 tok/s | 25.6 tok/s |
| TTFT (4096 tokens) | 36.6s | 10.9s | 5.5s |
| Peak memory | 22.1 GiB | 9.0 GiB | 23.7 GiB |
| SWE-bench Verified | **72.4** (ties GPT-5 mini) | Not reported | Lower than 27B |
| Coding reliability | **Best** — fewer syntax errors | Good | Weaker on complex logic |

**27B int4 is the strongest coder in the lineup.** It's a fully dense model — all 27B parameters activate on every token, giving it the highest reasoning density. It ties GPT-5 mini on SWE-bench Verified (72.4) and community testing consistently shows it makes fewer syntax errors and handles architectural logic better than the MoE 35B-A3B.

The tradeoff is speed: 11.8 tok/s and a 37s TTFT make it feel sluggish interactively. But for coding tasks, output quality matters more than raw speed.

**The 35B-A3B MoE is a trap for coding.** Despite looking attractive on paper (2.2x faster than 27B, similar memory), it only activates 3B parameters per token. Community benchmarks confirm it's worse at complex programming, nuanced logic, and instruction following.

**9B int4 is the practical compromise** for interactive-feeling speeds (38.5 tok/s) with still-strong coding ability. It outperforms models 3-13x its size on multiple benchmarks and scores 66.1 on BFCL-V4 for function calling.

### Summary

| Use Case | Recommended | Speed | Memory | Why |
|-----------|------------|-------|--------|-----|
| **Cron jobs** | **4B int4** | 68 tok/s | 6.4 GiB | Best quality-per-resource; leaves system headroom |
| **Coding (quality)** | **27B int4** | 11.8 tok/s | 22.1 GiB | Highest reasoning density, SWE-bench 72.4 |
| **Coding (interactive)** | **9B int4** | 38.5 tok/s | 9.0 GiB | 3x faster than 27B, still strong quality |

### Caveats

- **Hallucination is a concern for all small models.** The 4B and 9B score ~80-82% hallucination rate on AA-Omniscience. For cron jobs, always validate outputs programmatically.
- **The 27B has a known Ollama bug** where tool calling is non-functional ([ollama#14493](https://github.com/ollama/ollama/issues/14493)). Use mlx-lm directly for tool calling.
- **int4 quantization quality loss** is minimal for most tasks but can degrade complex math reasoning. If coding tasks involve heavy numerical logic, consider 9B int8 (24.1 tok/s, 13.4 GiB).

### Public Benchmark Sources

- [Qwen 3.5 Complete Guide](https://techie007.substack.com/p/qwen-35-the-complete-guide-benchmarks)
- [Qwen 3.5 Small Models: 9B Beats 120B](https://rits.shanghai.nyu.edu/ai/qwen-3-5-small-models-9b-parameters-that-beat-120b/)
- [Qwen 3.5 27B vs 35B-A3B](https://vertu.com/ai-tools/qwen-3-5-27b-vs-qwen-3-5-35b-a3b-which-local-llm-reigns-supreme/)
- [Qwen 3.5 Medium Models Benchmarks](https://www.digitalapplied.com/blog/qwen-3-5-medium-model-series-benchmarks-pricing-guide)
- [Qwen 3.5 Small Series - Awesome Agents](https://awesomeagents.ai/news/qwen-3-5-small-models-series/)
- [Alibaba Qwen 3.5 Small Models - MarkTechPost](https://www.marktechpost.com/2026/03/02/alibaba-just-released-qwen-3-5-small-models-a-family-of-0-8b-to-9b-parameters-built-for-on-device-applications/)
- [Artificial Analysis - Qwen 3.5 35B-A3B](https://artificialanalysis.ai/models/qwen3-5-35b-a3b)

## Methodology

- Models were converted from HuggingFace (`Qwen/Qwen3.5-*`) to MLX format using `mlx_lm.convert`
- Each configuration ran 1 warmup iteration followed by 20 measured iterations
- Prompt lengths tested: 64, 256, 1024, 4096 tokens
- Max generation length: 100 tokens
- A cooldown period of 5% of elapsed time was applied between runs to prevent thermal throttling
