# Running Every Qwen 3.5 Model Locally: A Quality and Speed Benchmark on Apple Silicon

> **TL;DR:** We benchmarked all seven Qwen 3.5 models (0.8B to 35B) on an M4 Pro Mac for both inference speed and reasoning quality across 41 problems of increasing difficulty. The 4B model is the sweet spot for most local AI agent tasks — it scores within 1 problem of the 27B while running 6x faster. The Claude-distilled 27B is the best option when you need maximum reliability on complex multi-step workflows.

## Why This Benchmark Exists

Off-the-shelf benchmarks like MMLU and HumanEval tell you how a model performs on academic tasks, but they don't answer the question that matters for local deployment: **which model should I actually run on my hardware for my use case?**

We built this benchmark to make that decision for [OpenClaw](https://github.com/AidfulAI/openclaw) autonomous agents running on a Mac Mini — cron jobs that summarize Obsidian notes, multi-step coding tasks, data pipeline automation, and structured document generation. The problems are designed to mirror real agent workloads, not textbook exercises.

This report covers:
- **Inference speed** across all 7 models (tokens/sec, memory footprint, prompt processing)
- **Reasoning quality** across 41 problems at 3 difficulty tiers (easy, hard, expert)
- **Practical recommendations** mapping models to specific use cases

## Test Environment

| Component | Details |
|-----------|---------|
| **Hardware** | Mac Mini, Apple M4 Pro (10P + 4E CPU cores, 20 GPU cores) |
| **Memory** | 64 GB unified memory (273 GB/s bandwidth) |
| **OS** | macOS 25.2.0 (Darwin) |
| **Framework** | MLX 0.31.0 + mlx-lm 0.30.7 |
| **Quantization** | int4 (4-bit) for all models |
| **Date** | March 2026 |

All models were converted from HuggingFace to MLX format using `mlx_lm.convert`. Benchmarks ran with no other GPU-intensive processes active.

## Models Tested

| Model | Parameters | Architecture | Memory (int4) | Notes |
|-------|-----------|--------------|---------------|-------|
| Qwen3.5-0.8B | 0.8B | Dense | 0.6 GiB | Smallest in the family |
| Qwen3.5-2B | 2B | Dense | 1.3 GiB | |
| Qwen3.5-4B | 5B | Dense | 2.8 GiB | |
| Qwen3.5-9B | 9B | Dense | 5.7 GiB | |
| Qwen3.5-27B | 28B | Dense | 16.0 GiB | Largest dense model |
| Qwen3.5-35B-A3B | 35B (3B active) | Mixture of Experts | 20.2 GiB | MoE — only 3B params active per token |
| Qwen3.5-27B-Claude-Distilled | 28B | Dense (LoRA fine-tune) | 15.4 GiB | CoT reasoning distilled from Claude Opus 4.6 |

The Claude-distilled model is [Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled](https://huggingface.co/Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled), a LoRA fine-tune (rank 64) trained on chain-of-thought traces from Claude Opus 4.6.

## Part 1: Inference Speed

Speed benchmarks used a standardized 100-token generation pass with 20 iterations per configuration. We tested two prompt lengths: 64 tokens (short prompt, approximating chat) and 4,096 tokens (long prompt, approximating document processing).

### Generation Speed (tokens/sec, int4)

| Model | Short Prompt (64 tok) | Long Prompt (4K tok) | Relative to 27B |
|-------|----------------------|---------------------|-----------------|
| **0.8B** | 281 tps | 249 tps | 22x faster |
| **2B** | 141 tps | 133 tps | 12x faster |
| **4B** | 67 tps | 62 tps | **6x faster** |
| **9B** | 37 tps | 36 tps | 3x faster |
| **35B-A3B** | 25 tps | 25 tps | 2.2x faster |
| **27B-Distilled** | 15 tps | 15 tps | 1.35x faster |
| **27B** | 11 tps | 11 tps | baseline |

### Prompt Processing Speed (tokens/sec, int4)

| Model | Short Prompt (64 tok) | Long Prompt (4K tok) |
|-------|----------------------|---------------------|
| **0.8B** | 2,009 tps | 3,727 tps |
| **2B** | 1,133 tps | 1,641 tps |
| **4B** | 497 tps | 661 tps |
| **9B** | 280 tps | 375 tps |
| **35B-A3B** | 249 tps | 777 tps |
| **27B-Distilled** | 94 tps | 124 tps |
| **27B** | 80 tps | 108 tps |

### Peak Memory Usage (GiB, int4)

| Model | Short Prompt (64 tok) | Long Prompt (4K tok) |
|-------|----------------------|---------------------|
| **0.8B** | 0.6 | 3.3 |
| **2B** | 1.3 | 4.2 |
| **4B** | 2.8 | 6.4 |
| **9B** | 5.7 | 9.0 |
| **35B-A3B** | 20.2 | 23.7 |
| **27B-Distilled** | 15.4 | 18.6 |
| **27B** | 16.0 | 22.1 |

### Speed Observations

**Generation speed is memory-bandwidth bound.** On Apple Silicon, token generation is bottlenecked by how fast weights can be read from unified memory, not by compute. This is why generation speed scales almost linearly with model size — halving the parameters roughly doubles the tokens/sec.

**The 35B MoE model punches above its weight.** Despite having 35B total parameters, it only activates 3B per token. This gives it 2.2x the generation speed of the 27B dense model while using similar memory (the full 35B of weights must still be loaded).

**The distilled model is faster than base 27B.** At 15 tps vs 11 tps, the Claude-distilled variant is 35% faster. It also uses less memory (15.4 vs 16.0 GiB at short context). The LoRA fine-tuning appears to have produced a more efficiently quantized model.

**Prompt processing scales differently.** Unlike generation, prompt processing is compute-bound and benefits from batched matrix operations. The 0.8B processes 4K-token prompts at 3,727 tps — fast enough to ingest a full document in about a second.

## Part 2: Reasoning Quality

### Methodology

We designed 41 evaluation problems across three difficulty tiers, targeting the kinds of tasks that local AI agents actually perform:

- **Easy (15 problems):** Baseline coding, arithmetic reasoning, and instruction following at GSM8K / HumanEval difficulty
- **Hard (10 problems):** Competition-level coding (LRU cache, calculator with operator precedence) and multi-step reasoning (Bayes' theorem, combinatorics)
- **Expert (16 problems):** Agent-level tasks inspired by real OpenClaw workloads — multi-step data pipelines, constraint satisfaction, structured document generation, and Obsidian note-taking workflows

Each problem has a deterministic check function that evaluates correctness. Every problem was run 3 times per model, and a problem passes if the majority of runs (2/3+) succeed. This majority-vote approach accounts for sampling variance in temperature-based generation.

Models emit chain-of-thought reasoning in different formats: base Qwen 3.5 models use freeform "Thinking Process:" preambles, while the Claude-distilled model uses `<think>` tags. Our check functions strip both formats before evaluating the actual answer.

### Overall Results

| Model | Easy (15) | Hard (10) | Expert (16) | Total (41) |
|-------|-----------|-----------|-------------|------------|
| **0.8B** | 12/15 | 7/10 | 10/16 | 29/41 (71%) |
| **2B** | 14/15 | 8/10 | 13/16 | 35/41 (85%) |
| **4B** | 15/15 | 10/10 | 15/16 | 40/41 (98%) |
| **9B** | 15/15 | 9/10 | 16/16 | 40/41 (98%) |
| **27B** | 15/15 | 10/10 | 16/16 | 41/41 (100%) |
| **35B-A3B** | 15/15 | 10/10 | 16/16 | 41/41 (100%) |
| **27B-Distilled** | 14/15 | 10/10 | 16/16 | 40/41 (98%) |

### Expert Results by Category

| Category | 0.8B | 2B | 4B | 9B | 27B | 35B-A3B | Distilled |
|----------|------|-----|-----|-----|------|---------|-----------|
| Math (3) | 0/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| Coding (3) | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| Reasoning (3) | 2/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| Instruction Following (3) | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| Writing (4) | 3/4 | 3/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |

### Generation Speed During Quality Benchmarks

Unlike synthetic speed benchmarks that generate exactly 100 tokens, quality benchmarks produce variable-length responses (500-2000+ tokens) with real reasoning content. This gives us a more realistic picture of throughput during actual use.

| Model | Avg Tokens/sec | Avg Generation Time | Avg Tokens Generated |
|-------|---------------|-------------------|---------------------|
| **0.8B** | 279.9 tps | 2.4s | 653 |
| **2B** | 143.3 tps | 6.0s | 822 |
| **4B** | 65.4 tps | 19.5s | 1,274 |
| **9B** | 36.6 tps | 37.5s | 1,375 |
| **27B** | 11.2 tps | 115.7s | 1,290 |
| **35B-A3B** | 25.1 tps | 51.9s | 1,294 |
| **27B-Distilled** | 15.1 tps | 67.1s | 1,012 |

### Problem Catalog

#### Easy Problems (15)

| # | Category | Problem | What It Tests |
|---|----------|---------|---------------|
| 1 | Coding | FizzBuzz | Basic control flow and modulo |
| 2 | Coding | Reverse string | Two approaches to string manipulation |
| 3 | Coding | Fibonacci | Sequence generation |
| 4 | Coding | Binary search | Algorithm implementation with edge cases |
| 5 | Coding | Palindrome | String processing with case/character handling |
| 6 | Reasoning | Train problem | Relative speed calculation (answer: 2 hours) |
| 7 | Reasoning | Coin probability | P(>=2 heads in 3 flips) = 50% |
| 8 | Reasoning | Workers problem | Inverse proportion (answer: 5 days) |
| 9 | Reasoning | Age problem | Simultaneous equations |
| 10 | Reasoning | Sequence pattern | Pattern recognition: n*(n+1), answer: 42 |
| 11 | Instruction | JSON output | Produce valid JSON with required keys |
| 12 | Instruction | Numbered list | Correct list formatting |
| 13 | Instruction | Word constraint | Concise ML explanation |
| 14 | Instruction | Code with comments | Code documentation |
| 15 | Instruction | Direct answer | Follow "no explanation" constraint |

#### Hard Problems (10)

| # | Category | Problem | Why It's Hard |
|---|----------|---------|---------------|
| 1 | Coding | LRU cache | Hash map + doubly linked list for O(1) operations |
| 2 | Coding | Flatten nested | Recursive with mixed types (don't iterate strings) |
| 3 | Coding | Longest palindrome | Expand-around-center or DP, not copy-paste |
| 4 | Coding | Calculator | Parsing + stack + operator precedence + parentheses |
| 5 | Coding | Buggy merge sort | Read comprehension + identify missing append |
| 6 | Reasoning | Compound interest | Multi-step: A = P(1 + r/n)^(nt), answer: ~$11,607.55 |
| 7 | Reasoning | Circular seating | Complementary counting: (6-1)! - 2*(5-1)! = 72 |
| 8 | Reasoning | Logic puzzle | Exhaustive case analysis over 8 truth/liar combos |
| 9 | Reasoning | Bayes' theorem | P(D\|+) ~ 16.1% — trap for models that say 95% |
| 10 | Reasoning | Proof bug | Identify division by zero when a = b |

#### Expert Problems (16)

These problems are specifically designed to mirror autonomous agent workloads:

| # | Category | Problem | Real-World Analog |
|---|----------|---------|-------------------|
| 1 | Math | Modular arithmetic | Pattern recognition in cyclic computations |
| 2 | Math | Inclusion-exclusion | 7-term formula with error compounding |
| 3 | Math | Bouncing ball | Infinite geometric series |
| 4 | Coding | Markdown to HTML | Regex ordering, nested formatting, edge cases |
| 5 | Coding | CSV data pipeline | Parse, filter (3 criteria), group, aggregate, sort, JSON output |
| 6 | Coding | Retry decorator | Decorator factory + dataclass + exponential backoff |
| 7 | Reasoning | Einstein riddle | Constraint propagation across 5 houses |
| 8 | Reasoning | Three urns | Conditional probability with branching |
| 9 | Reasoning | Topological sort | Enumerate ALL valid orderings |
| 10 | Instruction | Constrained factorial | 4 simultaneous code constraints |
| 11 | Instruction | Library JSON schema | Cross-references, required fields, types |
| 12 | Instruction | Adversarial transform | 3 sequential transformations |
| 13 | Writing | Multi-doc summary | Synthesize 3 documents into 3-5 sentences |
| 14 | Writing | Structured meeting notes | YAML frontmatter + exact section counts |
| 15 | Writing | Tone rewrite | Simplify jargon while preserving facts |
| 16 | Writing | Contradiction detection | Find subtle inconsistency between notes |

## Part 3: Key Findings

### 1. The 4B is the sweet spot for local agents

Scoring 40/41 (98%), the 4B matches the 9B and Claude-distilled 27B while running **6x faster** and using **1/4 the memory**. It falls just 1 problem short of the 27B and 35B-A3B (both 41/41). For the vast majority of autonomous agent tasks — cron jobs, note summarization, routine coding — the speed advantage vastly outweighs the marginal quality gap.

At 62-67 tokens/sec, the 4B generates responses fast enough to feel interactive. A typical 500-token response completes in ~8 seconds. The 27B takes ~45 seconds for the same output.

### 2. Model size only matters for multi-step composition

The only expert problem where the 4B fails is `data_pipeline` — a task requiring CSV parsing, 3-criteria filtering, grouping by key, aggregation, sorting, and JSON serialization all composed in a single function. This is exactly the kind of "chain 5+ operations coherently" task where larger models have an advantage.

For simpler tasks — even complex ones like Einstein's riddle, Bayes' theorem, or topological sort — the 4B handles them just as well as the 27B.

### 3. The Claude-distilled 27B offers the best quality-speed tradeoff at the top end

The distilled model matches the base 27B on quality (16/16 expert) while running 35% faster (15 vs 11 tps) and using less memory (15.4 vs 16.0 GiB). It produces shorter, more structured reasoning — averaging 1,307 tokens per expert problem vs 1,821 for the base 27B — because it uses concise `<think>` tags rather than verbose freeform preambles.

This makes it both faster (fewer tokens to generate) and easier to parse programmatically (clean tag-delimited reasoning).

### 4. The 35B MoE model achieves perfect quality at 2.2x the speed of 27B

The 35B-A3B scores 41/41 — matching the 27B dense model — while running at 25 tps (2.2x faster). Its Mixture of Experts architecture activates only 3B parameters per token, giving it generation speed closer to the 9B despite having 35B total parameters. The tradeoff is memory: it still needs 20.2 GiB to hold all expert weights, compared to 16.0 GiB for the 27B.

### 5. Quantization doesn't degrade reasoning

From our earlier int4 vs int8 comparisons on easy problems, both quantization levels produce identical quality scores. int4 runs 1.5-1.7x faster with half the memory. There is no reason to use int8 for these models on Apple Silicon.

### 6. Writing tasks differentiate only the smallest models

All models 4B and above pass all 4 writing problems — multi-document summarization, structured meeting notes, tone rewriting, and contradiction detection. The 0.8B and 2B each miss 1 of 4 writing tasks (tone rewrite). The Qwen 3.5 family has strong writing capabilities from 4B up, which is good news for Obsidian note-taking use cases.

### 7. The Bayes' theorem trap works — but not on Qwen 3.5

The Bayes' theorem problem is designed to catch models that confuse test accuracy (95%) with posterior probability (~16.1%). All Qwen 3.5 models, including the 4B, correctly apply Bayes' theorem. This family has unusually strong probabilistic reasoning across all sizes.

## Part 4: Recommendations for OpenClaw Agents

| Use Case | Recommended Model | Speed | Memory | Why |
|-----------|------------------|-------|--------|-----|
| **Cron jobs / automation** | 4B int4 | 62 tps | 6.4 GiB | Near-perfect quality, 6x faster than 27B |
| **Obsidian note summarization** | 4B int4 | 62 tps | 6.4 GiB | Perfect writing scores, fastest practical model |
| **Obsidian note writing** | 4B int4 | 62 tps | 6.4 GiB | Handles structured output and tone control |
| **Simple coding tasks** | 4B int4 | 62 tps | 6.4 GiB | Passes all hard coding problems |
| **Complex data pipelines** | 27B-Distilled int4 | 15 tps | 18.6 GiB | Only reliable option for multi-step composition |
| **Multi-step agentic workflows** | 27B-Distilled int4 | 15 tps | 18.6 GiB | Best quality-speed tradeoff for complex chains |
| **Bulk processing (speed critical)** | 0.8B int4 | 249 tps | 0.6 GiB | When you need volume over depth |

### Memory Budget Guide

| Available RAM | Best Model | Generation Speed |
|--------------|-----------|-----------------|
| 8 GB | 4B int4 | 62 tps |
| 16 GB | 9B int4 | 36 tps |
| 32 GB | 27B-Distilled int4 | 15 tps |
| 64 GB | 27B-Distilled int4 (with headroom) | 15 tps |

## Limitations and Caveats

- **Single-user benchmarks.** All tests ran sequentially with one model loaded at a time. Concurrent inference or multiple models loaded simultaneously would change memory and speed characteristics.
- **int4 only for quality tests.** We validated int4 vs int8 parity on easy problems but ran hard and expert exclusively with int4. If you're concerned about quantization effects on harder tasks, an int8 comparison would be worth running.
- **Keyword-based evaluation.** Our check functions use keyword matching and regex, not LLM-as-judge. This is deterministic and reproducible but can miss valid answers that use unexpected phrasing. We iteratively improved checks where we found false negatives (see Methodology Evolution below).
- **Majority vote with 3 runs.** A problem that passes 2/3 times is marked as passed. More runs would give higher confidence, but 3 strikes a practical balance for overnight benchmarking.
- **Fixed prompts, no system prompts.** All models received identical prompts with no system message tuning. Performance could likely be improved with model-specific prompt engineering.

## Methodology Evolution

Building reliable automated evaluation turned out to be harder than building the benchmarks themselves. Our check functions went through several iterations:

1. **Initial version:** `_strip_thinking()` only removed `<think>` tags. This worked for Claude-distilled models but failed on base Qwen 3.5 models, which use freeform "Thinking Process:" preambles that quote the original jargon/terms while reasoning about how to simplify them. The check functions would see jargon in the thinking and incorrectly fail the response.

2. **Preamble stripping:** We enhanced `_strip_thinking()` to detect freeform thinking headers and find answer section markers (like `**Rewrite:**`, `*Revised Draft:*`, `**Solution:**`). It uses the last "final answer" marker to extract only the actual answer, discarding the reasoning preamble.

3. **Post-answer cleanup:** Some models append verification tables or explanations after their answer that re-introduce the original terms for comparison. We added truncation at common post-answer markers (`---`, `**Verification**`, markdown tables).

4. **Keyword expansion:** The `logic_puzzle` check required "B is truthful" but models consistently said "B is a truth-teller." The `data_pipeline` check required specific grouping keywords (`groupby`, `defaultdict`) but valid solutions used plain dict iteration.

Each fix was validated with unit tests (154 total) to prevent regressions. The lesson: **automated LLM evaluation requires as much engineering as the benchmarks themselves.**

## Reproducibility

All code, check functions, and raw measurement CSVs are available in this repository:

- **Speed benchmarks:** `scripts/run_llm_benchmarks.py`
- **Quality benchmarks:** `scripts/run_quality_benchmarks.py`
- **Check functions:** `mtb/quality_benchmarks/eval_problems.py`
- **Tests:** `tests/test_quality_benchmarks.py` (154 tests)
- **Raw data:** `measurements/` directory, organized by hardware and timestamp

To reproduce:
```bash
# Speed benchmarks (all models, 20 iterations)
uv run python scripts/run_llm_benchmarks.py --num_iterations 20 \
  --run_only_benchmarks '["qwen-3.5-0.8b","qwen-3.5-2b","qwen-3.5-4b","qwen-3.5-9b","qwen-3.5-27b","qwen-3.5-35b-a3b","qwen-3.5-27b-claude-opus-distilled"]'

# Quality benchmarks (expert difficulty, 3 runs per problem)
uv run python scripts/run_quality_benchmarks.py --difficulty expert \
  --run_only_benchmarks '["qwen-3.5-4b","qwen-3.5-27b","qwen-3.5-27b-claude-opus-distilled"]' \
  --dtypes '["int4"]' --num_runs 3
```
