# M5 Max vs M4 Pro — MLX Benchmark Comparison

## Hardware Specs

| Spec | Mac mini M4 Pro | MacBook Pro M5 Max |
|---|---|---|
| CPU Cores | 10P + 4E (14 total) | ~18 total (XP+XE) |
| GPU Cores | 20 | 40 |
| Unified Memory | 64 GB | 128 GB |
| Memory Bandwidth | ~273 GB/s | ~546 GB/s |

## Software Environment

Both systems ran identical software stacks:
- MLX 0.31.0 / mlx-lm 0.30.7
- Python 3.11.11, PyTorch 2.6.0
- macOS 26.x (arm64)

## Benchmark Dates

- M4 Pro: March 8, 2026
- M5 Max: March 19-20, 2026

---

## LLM Benchmarks — Prompt Processing (tokens/sec, int4, 4096 prompt tokens)

| Model | M4 Pro | M5 Max | Speedup |
|---|---|---|---|
| Qwen 3.5 0.8B | 3,727 | 13,067 | 3.5x |
| Qwen 3.5 2B | 1,641 | 7,765 | 4.7x |
| Qwen 3.5 4B | 661 | 2,785 | 4.2x |
| Qwen 3.5 9B | 375 | 1,740 | 4.6x |
| Qwen 3.5 27B | 108 | 476 | 4.4x |
| Qwen 3.5 35B-A3B (MoE) | 777 | 2,725 | 3.5x |

Prompt processing scales **3.5-4.7x** across models. This exceeds the 2x GPU core ratio, driven by the combined effect of double the GPU cores and double the memory bandwidth.

## LLM Benchmarks — Generation (tokens/sec, int4, 4096 prompt tokens)

| Model | M4 Pro | M5 Max | Speedup |
|---|---|---|---|
| Qwen 3.5 0.8B | 249 | 394 | 1.6x |
| Qwen 3.5 2B | 133 | 201 | 1.5x |
| Qwen 3.5 4B | 62 | 126 | 2.0x |
| Qwen 3.5 9B | 36 | 78 | 2.2x |
| Qwen 3.5 27B | 11 | 22 | 2.0x |
| Qwen 3.5 35B-A3B (MoE) | 25 | 51 | 2.1x |

Generation scales **1.5-2.2x**. This is memory-bandwidth bound (autoregressive decoding reads the full model weights per token), so the 2x bandwidth advantage is the primary factor. Smaller models (0.8B, 2B) benefit less because they are not fully bandwidth-constrained.

## Memory Usage

Peak memory is **identical** across both chips for the same model and quantization. Memory usage is determined by model weights and KV-cache, not hardware.

## Quality Benchmarks

Quality results are **hardware-independent** as expected — the same model produces the same pass/fail outcomes regardless of chip. The M4 Pro run confirmed that small models (0.8B) fail expert-difficulty math tasks, while 27B+ models pass consistently.

---

## Key Takeaways

1. **Prompt processing is compute-bound** — scales super-linearly with GPU cores + bandwidth (3.5-4.7x for 2x cores)
2. **Generation is bandwidth-bound** — scales roughly linearly with memory bandwidth (~2x for 2x bandwidth)
3. **Small model generation doesn't scale as well** — overhead and compute saturation limit gains at 0.8B-2B sizes
4. **MoE models (35B-A3B) show similar scaling** to dense models despite only activating 3B params, because the full expert weights still reside in memory
5. **128GB unlocks larger models** — Llama 3.3 70B and Qwen 27B bfloat16 only fit on the M5 Max

---

## Recommended Additional Benchmarks

### Models to run on both chips (for direct comparison)

| Model | Available | Why |
|---|---|---|
| Gemma 3 1B, 4B (int4/int8) | Defined | Cross-architecture comparison (Gemma vs Qwen at similar sizes) |
| Gemma 3 1B, 4B, 12B QAT (int4) | Defined | QAT vs standard quantization scaling behavior |
| DeepSeek R1 Distill 7B (int4) | Defined | Popular reasoning model, fits on both chips |
| DeepSeek R1-0528 Qwen3-8B (int4/int8/bf16) | Defined | Reasoning model with multiple dtype support |
| Qwen 3.5 27B Claude Opus Distilled (int4) | Defined | Run on M4 Pro but missing from M5 Max LLM benchmarks |

### M5 Max-only models (128GB required)

| Model | Available | Why |
|---|---|---|
| Llama 3.3 70B (int4) | Defined | Largest dense model available; shows M5 Max ceiling |
| Gemma 3 27B (int4/int8/bf16) | Defined | Large Gemma model, compare against Qwen 27B |
| Qwen 3.5 27B (bfloat16) | Defined | Already run on M5; confirm it doesn't fit on M4 Pro 64GB |

### Benchmark configuration gaps

| Gap | Recommendation |
|---|---|
| Longer generation | Run with 500-1000 generated tokens to test sustained throughput and KV-cache growth impact |
| Larger prompts on M5 Max | Test 8k and 16k prompt lengths for models that fit, showing prefill scaling beyond 4k |
| Quality across dtypes on M4 Pro | Run quality benchmarks with int8 on M4 Pro (currently only int4 was tested) |
| Quality for mid-size models | Run quality benchmarks for 4B-9B range to find the minimum viable model size for expert tasks |
| Layer benchmarks on M4 Pro 64GB | The latest M4 Pro layer benchmarks are from the 24GB config; re-run on 64GB for apples-to-apples comparison |
