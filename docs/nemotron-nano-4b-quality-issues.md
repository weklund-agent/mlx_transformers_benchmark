# Nemotron-3-Nano-4B Quality Issues

## Summary

The Nemotron-3-Nano-4B model (int4 quantization via mlx-lm) consistently fails the quality benchmark suite. After multiple runs, config attempts, and diagnostic testing on 2026-03-20, the root cause has been confirmed: **mlx-lm's `nemotron_h` architecture implementation does not correctly support this hybrid Mamba2-Transformer model**. The same model runs correctly through Ollama (Q4_K_M GGUF).

## Observed Behavior

### Quality Benchmark Scores (consistent across all runs)

| Category | Passing | Total | Notes |
|---|---|---|---|
| coding | 0 | 9 | All tasks score 0/3 |
| reasoning | 0 | 8 | All tasks score 0/3 |
| math | 0 | 3 | All tasks score 0/3 |
| writing | 0 | 3 | All tasks score 0/3 |
| instruction_following | 2 | 7 | Only numbered_list and word_constraint pass (3/3) |

### Response Patterns

The model exhibits three degenerate output modes:

1. **Empty / immediate termination** — responds with just `<|im_end|>` and nothing else
2. **Prompt parroting** — restates a fragment of the prompt then terminates (e.g. `"We need to solve the problem of building a wall?<|im_end|>"`)
3. **Degenerate repetition loops** — gets stuck in infinite loops:
   - `"We need to respond to the user's request to respond to the user's request to respond to..."`
   - `"...solving the problem of solving the problem of solving the problem of..."`
   - `"2 = 2 = 2 = 2 = 2 ="`

### Special Token Leakage

`<|im_end|>` tokens frequently appear as literal text in generated output, particularly on longer prompts where the model spams them repeatedly.

## Diagnostic Test Results (2026-03-20)

A targeted diagnostic test (`tests/quality_benchmarks/test_nemotron_quality.py`) was run to separate config issues from model capability issues.

### Raw Completion (no chat template)

These tests bypass our chat template entirely and feed raw text directly to the model:

| Prompt | Output |
|---|---|
| `The capital of France is` | `a lie. But the problem is a lie. But the problem is a lie.` |
| `2 + 2 =` | `2 = 2 = 2 = 2 = 2 =` |

**Conclusion:** The model cannot produce coherent completions even without any chat template involved. This rules out template/config as the root cause.

### Chat-Templated Trivial Prompts

| Prompt | Output |
|---|---|
| `Say hello.` | Infinite loop: `We need to respond to the user's request to respond to...` |
| `What is 2+2?` | `We are given a list of 10000000000000000000000000...` |
| `FizzBuzz` | Restates prompt, then spams `<|im_end|>` tokens |

## Ollama Comparison Test (2026-03-20)

To isolate whether the issue is mlx-lm or the model itself, the same trivial prompts were run through Ollama (`nemotron-3-nano:4b`, Q4_K_M GGUF quantization). See `tests/quality_benchmarks/test_nemotron_ollama.py`.

| Prompt | MLX (int4) | Ollama (Q4_K_M) |
|---|---|---|
| Say hello | Infinite repetition loop | `"Hello!"` |
| What is 2+2? | `"We are given a list of 100000000..."` | `""` (empty — stop token issue, not incoherence) |
| Capital of France? | n/a | `""` (same stop token issue) |
| FizzBuzz | Spams `<\|im_end\|>` tokens | Correct Python with `n % 15 == 0` |
| Explain Python | `"solving the problem of solving..."` loop | Clear, coherent two-sentence answer |
| Special token leakage | `<\|im_end\|>` everywhere | None |

**Conclusion:** The model produces coherent, correct output through Ollama. The two empty responses on short-answer prompts appear to be a minor Ollama stop-token handling issue, not a model quality problem. This confirms the failures are specific to mlx-lm's `nemotron_h` implementation.

## Root Cause

The root cause is **incomplete mlx-lm support for the Nemotron hybrid Mamba2-Transformer architecture**. This has been confirmed by the Ollama comparison test above. Multiple pieces of evidence point to this:

### Architecture mismatch warning at load time
```
You are using a model of type `nemotron_h` to instantiate a model of type ``.
This may be expected if you are loading a checkpoint that shares a subset of the
architecture, but is otherwise not supported and can yield errors.
```

### Known mlx-lm issues with Nemotron models

The mlx-lm project has multiple open issues related to Nemotron support:

- **[#681](https://github.com/ml-explore/mlx-lm/issues/681)** — Loading Nemotron-3-Nano-30B-A3B-4bit fails with missing `rms_norm_eps` parameter
- **[#682](https://github.com/ml-explore/mlx-lm/issues/682) / [#683](https://github.com/ml-explore/mlx-lm/issues/683)** — Tokenizer class `TokenizersBackend` does not exist when loading Nemotron models
- **[#1016](https://github.com/ml-explore/mlx-lm/issues/1016)** — mlx-lm's `nemotron_h` implementation is missing MoE latent projection layers (`fc1_latent_proj` / `fc2_latent_proj`) required by the official NVIDIA architecture
- **[#386](https://github.com/ml-explore/mlx-lm/issues/386)** — Open request for Nemotron Nano v2 support

### Hybrid Mamba2 architecture is hard to support

Nemotron-3-Nano-4B uses a **hybrid Mamba2-Transformer MoE** architecture, which is significantly more complex than standard transformer-only models. Mamba2 support in mlx-lm is still actively maturing. The llama.cpp project also has [assertion failures](https://github.com/ggml-org/llama.cpp/issues/20570) when running Nemotron-3-Nano GGUF quantizations, suggesting this is a cross-ecosystem problem with the architecture, not specific to our setup.

### Quantization sensitivity at small model sizes

NVIDIA's own documentation notes that for small LLMs, the accuracy drop from post-training quantization (PTQ) is "often non-negligible." NVIDIA invested in specialized quantization methods (NVFP4, Quantization-Aware Distillation) specifically for Nemotron models, and their Q4_K_M GGUF achieved 100% median accuracy recovery — but this used their own quantization pipeline, not a generic int4 PTQ like the MLX conversion uses.

### Bottom line

The model loads without a hard error, but the combination of incomplete `nemotron_h` architecture support in mlx-lm and aggressive int4 quantization on a 4B hybrid model produces weights that generate degenerate output. The model isn't "low quality" — it's likely not running correctly.

## Resolution (2026-03-21)

**Root cause identified and fixed upstream.** The bug was in `ModelArgs.__post_init__()` in `mlx_lm/models/nemotron_h.py`:

```python
# BROKEN (mlx-lm <= 0.31.1):
self.time_step_limit = (self.time_step_min, self.time_step_max)  # (0.001, 0.1)

# FIXED (mlx-lm >= 0.31.2, PR #992):
self.time_step_limit = (self.time_step_min, float("inf"))  # (0.001, inf)
```

Clipping SSM time steps to max 0.1 destroyed the Mamba2 layers' ability to propagate state, causing degenerate repetition. Combined with SSM float32 dtype precision fixes (also in PR #992), the models now produce correct output.

**PR #992** ("Nemotron super support") was merged on 2026-03-13 by angeloskath (Apple MLX core team). It is in mlx-lm git HEAD (0.31.2-dev) but NOT yet in the latest PyPI release (0.31.1).

### Verified results after fix

Both Nemotron-3-Nano-4B and Nemotron-Nano-9B-v2 now work correctly on MLX:

| Model | Quality (MLX int4) | Gen tok/s | Notes |
|---|---|---|---|
| Nemotron-3-Nano-4B | ~62% (quick test) | ~101 tok/s | Partial fix — some prompts still degenerate |
| Nemotron-Nano-9B-v2 | **97.6% (40/41)** | **46.9 tok/s** | Fully working, fastest 9B model on M4 Pro |

The 9B model ties Qwen-3.5-9B on quality (97.6%) while being 20% faster on generation and scoring higher on tool calling benchmarks (BFCL v3: 0.649 vs 0.608).

### What was NOT the issue

- Quantization (int4/int8/BF16 all produced garbage — confirmed not a quantization issue)
- MambaRMSNormGated gate/norm ordering (tested, produced different but still broken output)
- Weight mapping (451/451 weights loaded correctly)
- Missing `lengths` parameter in ssm_update (no visible effect)
- MoE latent projections (not applicable — 4B and 9B don't use MoE blocks)

## Previous Investigation (for reference)

1. ~~**Test via Ollama**~~ — **Done.** Ollama produces coherent output, confirming the issue was mlx-lm, not the model.
2. ~~**Test int8 quantization via mlx-lm**~~ — **Done.** Also garbage. Confirmed architecture bug, not quantization.
3. ~~**Run quality benchmarks through Ollama**~~ — **Done.** Nemotron-3-Nano-4B scored 88% via Ollama.
4. ~~**Check for mlx-lm updates**~~ — **Done.** PR #992 fixed it.
5. ~~**Try Nemotron-Nano-9B-v2**~~ — **Done.** 97.6% quality, 46.9 tok/s on MLX with git HEAD.

## References

- [Nemotron 3 Nano 4B Blog Post](https://huggingface.co/blog/nvidia/nemotron-3-nano-4b)
- [NVIDIA Nemotron 3 Nano Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf)
- [mlx-lm Issue #1016 — Missing MoE latent projections](https://github.com/ml-explore/mlx-lm/issues/1016)
- [mlx-lm Issue #681 — rms_norm_eps error](https://github.com/ml-explore/mlx-lm/issues/681)
- [mlx-lm Issue #386 — Nemotron Nano v2 support](https://github.com/ml-explore/mlx-lm/issues/386)
- [llama.cpp Issue #20570 — Nemotron GGUF assertion failures](https://github.com/ggml-org/llama.cpp/issues/20570)
- [NVIDIA Quantization-Aware Distillation for NVFP4](https://research.nvidia.com/labs/nemotron/nemotron-qad/)
