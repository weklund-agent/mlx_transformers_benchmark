# TurboQuant & Sub-4-bit Quantization — Research and Next Steps

**Date:** 2026-03-31
**Hardware targets:** Mac Mini M4 Pro 64GB, MacBook Pro M5 Max 128GB
**Goal:** Evaluate whether TurboQuant KV cache compression and sub-4-bit weight quantization can fit larger models (70B dense, 122B MoE) on our hardware alongside existing benchmarks.

---

## 1. TurboQuant Overview

**Paper:** "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
- arXiv: 2504.19874 (April 28, 2025), accepted at ICLR 2026
- Authors: Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni (Google DeepMind)

### What it is

TurboQuant compresses the **KV cache** (key/value activations stored during inference), NOT model weights. It is **complementary** to weight quantization (int4/int8/GPTQ/AWQ). You use both together: GPTQ/int4 for weights + TurboQuant for KV cache.

| Aspect | Weight Quantization (int4/AWQ/GPTQ) | TurboQuant |
|---|---|---|
| Compresses | Model weights (on disk + memory) | KV cache (runtime activations) |
| When applied | Offline, once | Online, every forward pass |
| Primary benefit | Smaller model footprint | Longer context at same memory |
| Calibration data | GPTQ/AWQ need it; basic int4 doesn't | None (data-oblivious) |

### How it works

1. **Random rotation** via Walsh-Hadamard Transform (WHT) "Gaussianizes" KV vectors, spreading outliers uniformly across dimensions. Reduces kurtosis from ~900 to ~2.9 on real Qwen3 KV tensors.
2. **Scalar quantization** applied per-coordinate after rotation. Simple and near-optimal due to the Gaussianized distribution.
3. **1-bit QJL residual correction** removes inner-product estimation bias at negligible memory cost.

Key design choices:
- **Asymmetric K/V compression** — K precision matters far more than V (K controls softmax attention routing). Practical configs: q8_0-K + turbo3-V or turbo2-V.
- **Boundary layer protection** — First 2 + last 2 transformer layers kept at higher precision (recovers 37-91% of quality gap).
- **Sparse V dequant** — Skip V dequantization where attention weight < 1e-6 for +22.8% decode speedup at 32K context.

### Published results

| Format | Bits/value | Compression | Wikitext-2 PPL | vs q8_0 baseline |
|---|---|---|---|---|
| f16 | 16.0 | 1.0x | 6.121 | -0.16% |
| q8_0 | 8.5 | 1.9x | 6.111 | baseline |
| **turbo4** | **4.25** | **3.8x** | **6.125** | **+0.23%** |
| q4_0 | 4.5 | 3.6x | 6.142 | +0.52% |
| **turbo3** | **3.5** | **4.6-5.1x** | **6.176** | **+1.06%** |
| **turbo2** | **2.5** | **6.4x** | **6.507** | **+6.48%** |

Highlight: Command-R+ 104B at 128K context on a single M5 Max 128GB with turbo3 KV: 74 GB peak memory, PPL 6.415 (+3.6%).

### Qwen compatibility caveat

Qwen2.5 with Q4_K_M weights is **sensitive to symmetric TurboQuant on K** (catastrophic perplexity). Safe config: asymmetric with `q8_0` on K and `turbo3`/`turbo4` on V. Qwen with Q8_0 weights tolerates symmetric turbo. Qwen 3.5 models (9B, 27B, 35B-A3B) have been validated by the community.

### Nemotron compatibility

No specific TurboQuant validation for Nemotron yet. Standard decoder-only transformer attention, so should work. Validate empirically starting with the safe asymmetric config (q8_0 K, turbo4 V).

---

## 2. Sub-4-bit Weight Quantization in MLX

MLX already supports aggressive weight quantization that we haven't benchmarked yet.

### Native capabilities (mlx-lm 0.30.7+)

| Method | Description | Bit widths |
|---|---|---|
| **Affine** (default) | Scale+bias per group | 2, 3, 4, 5, 6, 8 |
| **AWQ** | Activation-aware weight scaling | 4 (default), supports qwen2/qwen3 |
| **GPTQ** | Hessian-based optimal rounding | 2, 4, 8 |
| **DWQ** | Knowledge distillation refinement | Any (best at 2-4 bit) |
| **Dynamic** | Gradient-based per-layer sensitivity | Mixed bit-widths targeting a BPW budget |

### Mixed-precision recipes (built-in)

`mlx_lm.convert` supports four predefined mixed-precision recipes via `--quant-predicate`:

| Recipe | Base bits | Sensitive layer bits | Sensitive layers |
|---|---|---|---|
| `mixed_2_6` | 2 | 6 | First/last 1/8 of layers + every 3rd in between (v_proj, down_proj, lm_head) |
| `mixed_3_4` | 3 | 4 | Same selection |
| `mixed_3_6` | 3 | 6 | Same selection |
| `mixed_4_6` | 4 | 6 | Same selection |

### What this means for model sizes

Estimated weight memory for models at various quantizations:

| Model | bf16 | int8 | int4 | int3 | mixed_2_6 | int2 |
|---|---|---|---|---|---|---|
| Qwen 3.5 27B | 50.1 GB | 26.6 GB | 14.1 GB | ~10.6 GB | ~8-10 GB | ~7 GB |
| Qwen 3.5 35B-A3B | 64.6 GB | 34.3 GB | 18.2 GB | ~13.7 GB | ~10-13 GB | ~9 GB |
| Llama 3.3 70B | ~141 GB | ~71 GB | ~35 GB | ~26 GB | ~20-25 GB | ~18 GB |
| Qwen 3.5 122B-A10B | ~244 GB | ~122 GB | ~61 GB | ~46 GB | ~35-40 GB | ~31 GB |

---

## 3. What Fits on Our Hardware

### Mac Mini M4 Pro 64GB

| Model | int4 | int3 | mixed_2_6 | Notes |
|---|---|---|---|---|
| Qwen 3.5 27B | 22 GB -- YES | ~14 GB -- YES | ~12 GB -- YES | Already benchmarked at int4/int8 |
| Qwen 3.5 35B-A3B | 24 GB -- YES | ~17 GB -- YES | ~14 GB -- YES | Already benchmarked at int4/int8 |
| Llama 3.3 70B | ~40 GB -- TIGHT | ~30 GB -- YES | ~25 GB -- YES | int4 feasible but limited context (~2-4K) |
| Qwen 3.5 122B-A10B | ~70 GB -- NO | ~52 GB -- NO | ~40 GB -- TIGHT | mixed_2_6 marginal, limited context |

**Key unlock for M4 Pro:** int3 or mixed_2_6 makes the 70B class viable with reasonable context headroom.

### MacBook Pro M5 Max 128GB

| Model | int4 | int3 | mixed_2_6 | Notes |
|---|---|---|---|---|
| Llama 3.3 70B | ~40 GB -- YES | ~30 GB -- YES | ~25 GB -- YES | Already defined, not yet benchmarked |
| Qwen 3.5 122B-A10B | ~70 GB -- TIGHT | ~52 GB -- YES | ~40 GB -- YES | Fits with int3 or mixed_2_6, generous context |
| Nemotron 70B | ~40 GB -- YES | ~30 GB -- YES | ~25 GB -- YES | Same arch as Llama 3.3 70B |

**Key unlock for M5 Max:** int3/mixed_2_6 makes the 122B-A10B MoE viable. TurboQuant KV cache extends context to 128K+ on the 70B models.

### KV cache overhead at 4096 tokens (fp16 KV cache)

| Model | KV cache at 4K | KV cache at 32K | Notes |
|---|---|---|---|
| Llama 3.3 70B / Nemotron 70B | ~1.3 GB | ~10.7 GB | Standard GQA |
| Qwen 3.5 27B | ~0.5-0.8 GB | ~4-6 GB | Hybrid linear+full attention |
| Qwen 3.5 35B-A3B | ~0.1-0.3 GB | ~1-2 GB | Very efficient (sparse attention + Mamba) |
| Qwen 3.5 122B-A10B | ~0.5-1 GB | ~4-8 GB | MoE with hybrid attention |

With TurboQuant turbo3 on KV cache, these numbers drop 4-5x (e.g., 70B at 32K: 10.7 GB -> ~2.1 GB).

---

## 4. Implementations Available

### TurboQuant KV cache compression

| Implementation | Framework | Stars | Notes |
|---|---|---|---|
| [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) | Python reference | 3,685 | Leading implementation, includes Sparse V + boundary layer protection |
| [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) | llama.cpp + Metal | 270 | Full Metal kernel support. Usage: `llama-server -m model.gguf -ctk turbo3 -ctv turbo3 -fa 1` |
| [arozanov/turboquant-mlx](https://github.com/arozanov/turboquant-mlx) | MLX + Metal | 52 | Fused Metal kernels, 4.6x compression at 98% fp16 speed claimed |
| [alicankiraz1/Qwen3.5-TurboQuant-MLX-LM](https://github.com/alicankiraz1/Qwen3.5-TurboQuant-MLX-LM) | MLX | 68 | Qwen 3.5-focused |
| Upstream llama.cpp PR [#21089](https://github.com/ggml-org/llama.cpp/pull/21089) | llama.cpp | -- | CPU-only TBQ3_0/TBQ4_0 types, in progress |

### Sub-4-bit weight quantization

Already built into `mlx_lm.convert`:
```
mlx_lm.convert --hf-path MODEL --mlx-path OUTPUT -q --q-bits 3
mlx_lm.convert --hf-path MODEL --mlx-path OUTPUT -q --q-bits 2
mlx_lm.convert --hf-path MODEL --mlx-path OUTPUT --quant-predicate mixed_2_6
mlx_lm.convert --hf-path MODEL --mlx-path OUTPUT --quant-predicate mixed_3_6
```

No new dependencies required.

### Related community projects (for reference)

| Project | What it does |
|---|---|
| [mlx-optiq](https://mlx-optiq.pages.dev/) | KL-divergence per-layer sensitivity + greedy knapsack optimization for mixed-precision |
| [JANG-Q](https://github.com/jjang-ai/jangq) | Adaptive mixed-precision "GGUF for MLX" with per-tensor-type bit depths |

---

## 5. Recommended Next Steps

### Phase 1: Sub-4-bit weight quantization benchmarks (low effort, high value)

This requires only Makefile + ModelSpec changes. No new framework code.

**5.1 Add int3 and mixed_2_6 conversion targets**

Add to `Makefile`:
- `convert-qwen35-3bit` — Convert Qwen 3.5 family to 3-bit
- `convert-qwen35-mixed26` — Convert Qwen 3.5 family to mixed_2_6
- `convert-70b-3bit` — Convert Llama 3.3 70B to 3-bit
- `convert-70b-mixed26` — Convert Llama 3.3 70B to mixed_2_6

Add to `ModelSpec` definitions:
- `"int3"` and `"mixed_2_6"` model_ids for Qwen 3.5 (0.8B through 35B-A3B) and Llama 3.3 70B

**5.2 Run speed benchmarks on both machines**

| Machine | Models | Dtypes | Iterations |
|---|---|---|---|
| M4 Pro 64GB | Qwen 3.5 (0.8B-35B-A3B), Llama 3.3 70B | int3, mixed_2_6 | 20 |
| M5 Max 128GB | Same + Qwen 3.5 122B-A10B | int3, mixed_2_6 | 20 |

Key questions to answer:
- What is the speed vs quality tradeoff at int3 compared to int4?
- Does mixed_2_6 preserve quality better than uniform int3 at similar memory footprint?
- Does 70B int3 on M4 Pro 64GB actually work with usable context windows?
- Can 122B-A10B mixed_2_6 run on M5 Max 128GB?

**5.3 Run quality benchmarks**

Run all three difficulty tiers (easy/hard/expert) at int3 and mixed_2_6 alongside existing int4/int8/bf16 results. This is the critical comparison — does sub-4-bit quantization destroy reasoning quality?

Priority models for quality testing:
- Qwen 3.5 9B (int3 vs int4 vs int8 — quality inflection point?)
- Qwen 3.5 27B (int3 vs int4 — can we afford to go lower on the largest dense model?)
- Llama 3.3 70B int3 on M5 Max (does the larger model compensate for lower quantization?)

### Phase 2: TurboQuant KV cache benchmarks (medium effort, context-length focused)

This requires integrating a community TurboQuant implementation. Two paths:

**Path A: llama.cpp fork (easiest to start)**

Use [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) via our existing LM Studio / llama.cpp benchmark infrastructure.

Test matrix:
| KV config | Weight quant | Model | Context lengths |
|---|---|---|---|
| q8_0 K + q8_0 V (baseline) | Q4_K_M | Qwen 3.5 27B | 4K, 16K, 32K |
| q8_0 K + turbo4 V | Q4_K_M | Qwen 3.5 27B | 4K, 16K, 32K, 64K |
| q8_0 K + turbo3 V | Q4_K_M | Qwen 3.5 27B | 4K, 16K, 32K, 64K, 128K |
| q8_0 K + turbo3 V | Q4_K_M | Llama 3.3 70B | 4K, 16K, 32K |

Metrics: generation tok/s, peak memory, perplexity (wikitext-2), NIAH retrieval accuracy.

**Path B: MLX native (better long-term fit)**

Use [arozanov/turboquant-mlx](https://github.com/arozanov/turboquant-mlx) or [alicankiraz1/Qwen3.5-TurboQuant-MLX-LM](https://github.com/alicankiraz1/Qwen3.5-TurboQuant-MLX-LM). Integrate as a new `kv_quant` parameter in `ModelSpec` / `run_benchmark`.

This path requires more work (new benchmark code) but produces results directly comparable to our existing MLX benchmarks.

**Recommended:** Start with Path A for fast validation, then integrate Path B if TurboQuant shows clear wins.

### Phase 3: Combined weight + KV cache quantization (validation)

Once both are individually validated, test the combined configuration:
- Qwen 3.5 27B: int3 weights + turbo3 KV — peak memory? quality? context ceiling?
- Llama 3.3 70B: int4 weights + turbo3 KV — does 128K context work on M5 Max?
- Llama 3.3 70B: int3 weights + turbo3 KV — does this fit on M4 Pro 64GB with usable context?

---

## 6. Relevance to Local AI Infrastructure

For the vllm-mlx + LiteLLM + OpenClaw stack:

- **Mac Mini M4 Pro 64GB (always-on agent worker):** Sub-4-bit quantization could upgrade from Qwen 3.5 27B int4 to 70B int3 — a significant quality jump for autonomous agent tasks, at the cost of lower quantization. Quality benchmarks in Phase 1 will determine if this is a net win.

- **MacBook Pro M5 Max 128GB (deep work):** TurboQuant KV cache unlocks 128K+ context on 70B models, enabling document-scale reasoning. Sub-4-bit weight quant opens the door to 122B-A10B MoE, which activates only ~10B params per token (fast generation) while having 122B total knowledge.

- **Model serving consideration:** vllm-mlx may not yet support TurboQuant KV cache types. This needs investigation before Phase 2 can feed into the serving stack.
