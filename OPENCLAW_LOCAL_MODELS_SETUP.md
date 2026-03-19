# OpenClaw Local Models Setup — MLX on Apple Silicon

> M4 Pro 64GB | OpenClaw 2026.3.2 | mlx-lm | Qwen 3.5

---

## Pre-Flight Checklist

- [ ] OpenClaw gateway is running and healthy (`openclaw health`)
- [ ] Gateway config has `gateway.mode: "local"` set in `~/.openclaw/openclaw.json`
- [ ] Gateway token is pasted into Control UI settings (browser dashboard)
- [ ] `mlx-lm` is installed (`pip install mlx-lm`)
- [ ] Models are downloaded (see Model Selection below)
- [ ] Sufficient free RAM for chosen model combo (check with `memory_pressure` or `top -l 1`)
- [ ] No stale benchmark processes hogging memory (`ps aux | grep python | grep mlx`)

---

## Model Selection

Speed benchmarked on M4 Pro 64GB, 2026-03-06 (20 iterations). Quality benchmarked 2026-03-08 (41 problems, 3 runs each):

### Recommended Dual-Model Setup

| Role | Model | Quality | Gen TPS | TTFT (4k) | RAM | Port |
|------|-------|---------|---------|-----------|-----|------|
| **Primary** (chat, sessions, coding) | Qwen3.5-35B-A3B-int4 | 41/41 (100%) | 25.2 | 5.5s | 21.6 GiB | 8800 |
| **Fast** (cron, automation, message ack) | Qwen3.5-4B-int4 | 40/41 (98%) | 66.6 | 6.1s | 4.4 GiB | 8801 |
| **Combined** | | | | | **~26 GiB** | |

Leaves ~38 GiB free on a 64GB machine.

### Alternative Primary Models

| Model | Quality | Gen TPS | TTFT (4k) | RAM | Trade-off |
|-------|---------|---------|-----------|-----|-----------|
| Qwen3.5-27B-int4 | 41/41 (100%) | 11.7 | 36.6s | 18.4 GiB | Ties 35B-A3B on quality, but 2.2x slower generation and 6.6x slower TTFT |
| Qwen3.5-27B-Claude-Opus-Distilled-int4 | 40/41 (98%) | 15.1 | ~36.6s | 15.4 GiB | Shorter, more structured reasoning (`<think>` tags). Misses 1 easy problem |
| Qwen3.5-9B-int4 | 40/41 (98%) | 38.1 | 10.9s | 7.1 GiB | Same quality as distilled at 2.5x the speed and half the RAM |

### Alternative Fast Models

| Model | Quality | Gen TPS | RAM | Trade-off |
|-------|---------|---------|-----|-----------|
| Qwen3.5-2B-int4 | 35/41 (85%) | 141.8 | 2.7 GiB | 2x faster than 4B, but drops 5 problems (math, reasoning, writing) |
| Qwen3.5-0.8B-int4 | 29/41 (71%) | 276.0 | 1.9 GiB | 4x faster than 4B, but fails 12 problems. Only for trivial tasks |

### Key Benchmark Insights

- **35B-A3B + 4B is the optimal pairing**: 100% + 98% quality, both fast enough for interactive use
- **int4 is the sweet spot**: 1.5-1.7x faster than int8, 2.5-4x faster than bf16, no quality degradation
- **35B-A3B MoE** has 6.6x faster TTFT than 27B dense (5.5s vs 36.6s) with similar memory
- **4B matches 9B on quality** (both 40/41) at nearly double the speed — no reason to use 9B as the fast model
- **Quality tested on 41 problems** across 3 difficulty tiers (easy/hard/expert) with majority voting (3 runs each). See `measurements/quality_benchmarks/` for raw data and `REPORT.md` for full analysis
- **Ollama has a tool-calling bug with 27B** — use mlx-lm.server directly
- bf16 has faster prefill (25-35%) but much slower generation
- Peak memory is deterministic (zero variance) — safe to plan around

---

## Step 1: Download Models

If not already cached from benchmarking:

```bash
# Primary
huggingface-cli download mlx-community/Qwen3.5-35B-A3B-4bit

# Fast
huggingface-cli download mlx-community/Qwen3.5-4B-4bit
```

Or use local paths if already downloaded to the benchmark directory:
```
~/Projects/mlx_transformers_benchmark/models/Qwen3.5-35B-A3B/int4
~/Projects/mlx_transformers_benchmark/models/Qwen3.5-4B/int4
```

---

## Step 2: Start MLX Servers

### Manual (two terminals)

```bash
# Terminal 1 — Primary (interactive)
mlx_lm.server \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --port 8800 \
  --host 127.0.0.1

# Terminal 2 — Fast (automation)
mlx_lm.server \
  --model mlx-community/Qwen3.5-4B-4bit \
  --port 8801 \
  --host 127.0.0.1
```

### Using local model paths

```bash
mlx_lm.server \
  --model ~/Projects/mlx_transformers_benchmark/models/Qwen3.5-35B-A3B/int4 \
  --port 8800 \
  --host 127.0.0.1

mlx_lm.server \
  --model ~/Projects/mlx_transformers_benchmark/models/Qwen3.5-4B/int4 \
  --port 8801 \
  --host 127.0.0.1
```

### Verify servers are running

```bash
curl http://127.0.0.1:8800/v1/models
curl http://127.0.0.1:8801/v1/models
```

---

## Step 3: Configure OpenClaw

Edit `~/.openclaw/openclaw.json` and add/merge the `models` and update the `agents` sections:

```json
{
  "models": {
    "providers": {
      "mlx-primary": {
        "baseUrl": "http://127.0.0.1:8800/v1",
        "apiKey": "mlx-local",
        "api": "openai-completions",
        "models": [{
          "id": "Qwen3.5-35B-A3B-4bit",
          "name": "Qwen 3.5 35B A3B (MLX int4)",
          "contextWindow": 32768,
          "maxTokens": 8192,
          "reasoning": false,
          "input": ["text"],
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }]
      },
      "mlx-fast": {
        "baseUrl": "http://127.0.0.1:8801/v1",
        "apiKey": "mlx-local",
        "api": "openai-completions",
        "models": [{
          "id": "Qwen3.5-4B-4bit",
          "name": "Qwen 3.5 4B (MLX int4)",
          "contextWindow": 32768,
          "maxTokens": 4096,
          "reasoning": false,
          "input": ["text"],
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": {
        "primary": "mlx-primary/Qwen3.5-35B-A3B-4bit",
        "fallbacks": ["mlx-fast/Qwen3.5-4B-4bit"]
      }
    }
  }
}
```

Then set the default and verify:

```bash
openclaw models set mlx-primary/Qwen3.5-35B-A3B-4bit
openclaw models status
openclaw models list
```

### Route cron/automation to the fast model

```bash
openclaw config set agents.defaults.cron.model "mlx-fast/Qwen3.5-4B-4bit"
```

Or create a dedicated automation agent:

```bash
openclaw agents create --name auto --model "mlx-fast/Qwen3.5-4B-4bit"
```

---

## Step 4: Restart Gateway and Test

```bash
openclaw daemon restart
openclaw health
```

Test a quick inference:

```bash
openclaw agent --message "What is 2+2?" --deliver
```

---

## Troubleshooting

### Gateway won't start
- Error: `Gateway start blocked: set gateway.mode=local`
- Fix: `openclaw config set gateway.mode local`

### "unauthorized: gateway token missing" in Control UI
- The browser dashboard stores the token separately from server config
- Run `openclaw dashboard`, paste the token from `gateway.auth.token` in settings
- Token location: `~/.openclaw/openclaw.json` → `gateway.auth.token`

### Model server not responding
```bash
# Check if ports are in use
lsof -i :8800
lsof -i :8801

# Check server logs in the terminal where mlx_lm.server is running
```

### Out of memory
```bash
# Check what's using RAM
top -l 1 -o mem -n 10 -stats pid,command,mem
memory_pressure

# Kill stale benchmark processes
ps aux | grep python | grep -E 'benchmark|mlx'
```

### Ollama tool-calling issues
The 27B model has known tool-calling bugs in Ollama. Use `mlx_lm.server` directly instead — OpenClaw connects via the same OpenAI-compatible `/v1` endpoint.

---

## RAM Budget (64GB Machine)

| Component | RAM |
|-----------|-----|
| macOS + system processes | ~8 GiB |
| OpenClaw gateway (node) | ~0.4 GiB |
| MLX primary (35B-A3B-int4) | 21.6 GiB |
| MLX fast (4B-int4) | 4.4 GiB |
| **Total** | **~34.4 GiB** |
| **Free** | **~29.6 GiB** |

Do not run benchmark processes simultaneously with served models — a single 27B benchmark run consumes ~15 GiB.

---

## File Locations

| What | Path |
|------|------|
| OpenClaw config | `~/.openclaw/openclaw.json` |
| Gateway logs | `~/.openclaw/logs/gateway.log` |
| Gateway error logs | `~/.openclaw/logs/gateway.err.log` |
| LaunchAgent plist | `~/Library/LaunchAgents/ai.openclaw.gateway.plist` |
| Benchmark models | `~/Projects/mlx_transformers_benchmark/models/` |
| Benchmark results | `~/Projects/mlx_transformers_benchmark/measurements/` |
| Speed benchmark report | `~/Projects/mlx_transformers_benchmark/measurements/llm_benchmarks/Apple_M4_Pro_10P+4E+20GPU_64GB/2026-03-06__19:47:24/REPORT.md` |
| Quality benchmark report | `~/Projects/mlx_transformers_benchmark/measurements/quality_benchmarks/Apple_M4_Pro_10P+4E+20GPU_64GB/REPORT.md` |
