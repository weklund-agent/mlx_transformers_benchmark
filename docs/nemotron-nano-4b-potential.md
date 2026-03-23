# Nemotron-3-Nano-4B Could Be the Best Small Model on Apple Silicon — If MLX Can Catch Up

We almost wrote off NVIDIA's Nemotron-3-Nano-4B. After weeks of failing quality benchmarks, we assumed the model just wasn't good enough. Every run: 0/3 on coding, 0/3 on reasoning, 0/3 on math. Degenerate repetition loops. Special tokens leaking into output. We kept tweaking configs, templates, prompts — nothing worked.

Turns out the model was never the problem.

## The real issue: mlx-lm doesn't fully support this architecture

Nemotron-3-Nano-4B isn't a standard transformer. It's a **hybrid Mamba2-Transformer MoE** — a fundamentally different architecture that mlx-lm's `nemotron_h` implementation doesn't correctly handle yet. The model loads, generates tokens, and reports reasonable tok/s numbers. But the output is garbage:

```
Prompt: "What is 2+2?"
MLX:    "We are given a list of 100000000000000000000000..."

Prompt: "Say hello."
MLX:    "We need to respond to the user's request to respond
         to the user's request to respond to the user's request..."
```

There are multiple open issues on the mlx-lm repo — missing MoE latent projections, tokenizer errors, architecture mismatch warnings. The model silently loads broken.

## The proof: same model, same prompts, through Ollama

We ran the exact same prompts through Ollama's Q4_K_M GGUF quantization:

```
Prompt: "Say hello."
Ollama: "Hello!"

Prompt: "Write FizzBuzz in Python."
Ollama: *correct, working code with n % 15 == 0*
```

Night and day.

## The benchmarks tell the story

We ran our full quality benchmark suite (41 problems across easy, hard, and expert difficulty) through Ollama. The results:

| Difficulty | Score |
|---|---|
| Easy | 14/15 (93%) |
| Hard | 9/10 (90%) |
| Expert | 13/16 (81%) |
| **Total** | **36/41 (88%)** |

Perfect scores on coding at every difficulty level. Perfect on writing. Perfect on instruction following. The only weaknesses: abstract logic puzzles and exact math — expected limitations for a 4B model.

## How it stacks up against the competition

Here's how Nemotron-3-Nano-4B compares to every other model in the 1-4B parameter class on our benchmark suite, all at int4 quantization:

| Model | Params | Quality | Gen tok/s |
|---|---|---|---|
| Gemma-3-4B-it-QAT | 4B | 97.6% | 84.6 |
| Gemma-3-4B-it | 4B | 92.7% | 84.6 |
| Qwen-3.5-4B | 4B | 91.1% | 66.9 |
| **Nemotron-3-Nano-4B** | **4B** | **87.8%** | **50.2*** |
| Qwen-3.5-35B-A3B | 35B (3B active) | 87.3% | 25.4 |
| Qwen-3.5-2B | 2B | 85.4% | 141.8 |
| Qwen-2.5-3B-it | 3B | 80.5% | 111.9 |
| Qwen-3.5-0.8B | 0.8B | 70.7% | 274.5 |

*\*Running through Ollama, not native MLX. See below.*

At 87.8% quality, Nemotron beats everything below the Qwen-3.5-4B tier — and it edges out the much larger Qwen-3.5-35B-A3B MoE model on both quality and speed.

## Update: MLX support is here (2026-03-21)

**The fix landed.** [PR #992](https://github.com/ml-explore/mlx-lm/pull/992) by the Apple MLX team fixed the `nemotron_h` implementation. The bug was `time_step_limit` clipping SSM time steps to max 0.1, destroying state propagation across Mamba2 layers. Available in mlx-lm git HEAD (0.31.2-dev), pending PyPI release.

## Nemotron-Nano-9B-v2: the real story

With MLX working, we benchmarked the bigger sibling — **Nemotron-Nano-9B-v2** — and it's the fastest 9B-class model on Apple Silicon:

| Model | Params | Quality | Gen tok/s | Memory |
|---|---|---|---|---|
| **Nemotron-Nano-9B-v2** | **9B** | **97.6%** | **46.9** | **6.4 GB** |
| Qwen-3.5-9B | 9B | 97.6% | 39.0 | 6.3 GB |
| Gemma-3-12B-it-QAT | 12B | — | 33.6 | 7.7 GB |

Same quality as Qwen-3.5-9B (40/41 on our suite). **20% faster generation.** And the Mamba2 architecture advantage grows at long contexts — consistent ~47 tok/s regardless of prompt length, while transformers slow down as context grows.

### Quality breakdown (Nemotron-Nano-9B-v2, int4, MLX)

| Difficulty | Score |
|---|---|
| Easy | 15/15 (100%) |
| Hard | 12/13 (92%) |
| Expert | 13/13 (100%) |
| **Total** | **40/41 (97.6%)** |

Perfect scores on coding, math, instruction following, and writing. Only failure: compound_interest.

### Tool calling

Nemotron-Nano-9B-v2 also beats Qwen on tool calling benchmarks:
- **BFCL v3**: Nemotron 0.649 vs Qwen3-8B 0.608
- Native function calling support, trained with NVIDIA's WorkBench multi-step tool-calling environment

### Speed comparison across prompt lengths

| Prompt Length | Nemotron-Nano-9B-v2 | Qwen-3.5-9B | Advantage |
|---|---|---|---|
| 64 tokens | 46.5 tok/s | 39.1 tok/s | +19% |
| 256 tokens | 46.9 tok/s | 39.0 tok/s | +20% |
| 1024 tokens | 46.9 tok/s | 38.6 tok/s | +21% |
| 4096 tokens | 46.3 tok/s | 36.2 tok/s | **+28%** |

The Mamba2 advantage compounds with context length. At 4K tokens, Nemotron is 28% faster.

## The original 4B story still holds

The 4B model also works now on MLX (~101 tok/s generation), though quality is lower than the Ollama results (~62% via MLX vs 88% via Ollama). The 9B model is the clear winner for quality-critical applications.

## The takeaway

Don't trust silent failures. We spent weeks assuming a model was bad because it produced bad output. A simple A/B test against Ollama proved the model was fine and the framework was broken. If we'd given up, we'd have missed one of the most promising model families for on-device inference.

Nemotron on Apple Silicon: the quality is here, the speed is here, and it's the fastest 9B model in its class.
