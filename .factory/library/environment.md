# Environment

Environment variables, external dependencies, and setup notes.

**What belongs here:** Required env vars, external API keys/services, dependency quirks, platform-specific notes.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

---

## Python Version
- Python 3.11.11 (pinned in pyproject.toml, managed by uv)
- System Python is 3.14.3 but uv uses its own managed interpreter

## Package Manager
- `uv` for all Python operations
- Always use `uv run python` to execute scripts
- `uv sync` to install dependencies

## Key Dependencies
- `mlx == 0.31.0` (pinned)
- `mlx-lm` from git branch `pc/add-gemma-4` (Gemma 4 support)
- `torch == 2.6.0` (pinned)
- `fire >= 0.7.0` (CLI argument parsing)
- `pandas >= 2.2.3` (data handling)
- `pytest >= 8.3.5` with `pytest-cov` and `pytest-mock`

## Hardware
- Apple Silicon (M-series) with MLX Metal backend
- 128GB RAM, 18 CPU cores on current machine

## No External Services Required
- This is a pure Python library/CLI project
- No databases, APIs, or running services needed for development
- Model files are cached by HuggingFace in the default cache directory
