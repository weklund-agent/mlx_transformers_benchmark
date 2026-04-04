"""Regenerate the agentic coding benchmark tables in README.md.

Reads all speed and quality benchmark CSVs, computes the latest results
for each model at int4 and int8, and rewrites the benchmark section of
the README between the markers.

Usage:
    uv run python scripts/update_readme_table.py
    uv run python scripts/update_readme_table.py --models '["gemma-4-e2b-it","lfm2-24b-a2b"]'
    uv run python scripts/update_readme_table.py --dry-run
"""

import glob
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import fire
import pandas as pd

import mtb

REPO_ROOT = mtb.REPO_ROOT
README_PATH = REPO_ROOT / "README.md"
SPEED_ROOT = REPO_ROOT / "measurements" / "llm_benchmarks"
QUALITY_ROOT = REPO_ROOT / "measurements" / "quality_benchmarks"

# Markers in README that delimit the auto-generated section
BEGIN_MARKER = "<!-- BEGIN BENCHMARK TABLE -->"
END_MARKER = "<!-- END BENCHMARK TABLE -->"

# Model metadata not available from CSVs
MODEL_META = {
    "gemma-4-e2b-it": {"arch": "2.3B dense", "min_hw": "Any Mac"},
    "gemma-4-e4b-it": {"arch": "4.5B dense", "min_hw": "Any Mac"},
    "gemma-4-26b-a4b-it": {"arch": "3.8B MoE", "min_hw": "24GB+"},
    "gemma-4-31b-it": {"arch": "31B dense", "min_hw": "24GB+"},
    "qwen3-coder-30b-a3b": {"arch": "3B MoE", "min_hw": "24GB+"},
    "glm-4.7-flash": {"arch": "3B MoE", "min_hw": "24GB+"},
    "lfm2-24b-a2b": {"arch": "2B MoE", "min_hw": "16GB+"},
    "qwen-3.5-35b-a3b": {"arch": "3B MoE", "min_hw": "24GB+"},
    "qwen-3.5-27b": {"arch": "27B dense", "min_hw": "36GB+"},
    "qwen-3.5-9b": {"arch": "9B dense", "min_hw": "16GB+"},
    "qwen-3.5-4b": {"arch": "4B dense", "min_hw": "Any Mac"},
    "qwen-3.5-2b": {"arch": "2B dense", "min_hw": "Any Mac"},
    "qwen-3.5-0.8b": {"arch": "0.8B dense", "min_hw": "Any Mac"},
    "nemotron-nano-9b-v2": {"arch": "9B dense", "min_hw": "16GB+"},
    "nemotron-3-nano-4b": {"arch": "4B dense", "min_hw": "Any Mac"},
}


def _min_hw_from_memory(mem_gib: float) -> str:
    """Estimate minimum hardware from peak memory."""
    if mem_gib <= 8:
        return "Any Mac"
    elif mem_gib <= 14:
        return "16GB+"
    elif mem_gib <= 22:
        return "24GB+"
    elif mem_gib <= 30:
        return "36GB+"
    elif mem_gib <= 44:
        return "48GB+"
    elif mem_gib <= 58:
        return "64GB+"
    else:
        return "128GB+"


def load_speed_data(models: Optional[List[str]] = None) -> pd.DataFrame:
    """Load latest speed benchmark for each model/dtype, at 1024 prompt tokens."""
    files = glob.glob(str(SPEED_ROOT / "**" / "benchmark_results.csv"), recursive=True)
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Extract timestamp from directory name for recency
        dirname = Path(f).parent.name
        df["_source_dir"] = dirname
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    # Filter to 1024 prompt tokens, MLX metal
    mask = (all_df["num_prompt_tokens"] == 1024) & (all_df["framework"] == "mlx")
    all_df = all_df[mask].copy()

    if models:
        all_df = all_df[all_df["name"].isin(models)]

    # Keep latest measurement per model/dtype (by directory name = timestamp)
    all_df = all_df.sort_values("_source_dir").groupby(["name", "dtype"]).last().reset_index()

    return all_df[["name", "dtype", "generation_tps", "prompt_tps", "peak_memory_gib"]]


def load_quality_data(models: Optional[List[str]] = None) -> pd.DataFrame:
    """Load latest quality results, compute overall % and per-category scores."""
    files = glob.glob(str(QUALITY_ROOT / "**" / "quality_results.csv"), recursive=True)
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["_source_dir"] = Path(f).parent.name
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    if models:
        all_df = all_df[all_df["model"].isin(models)]

    # Keep latest per model/dtype/problem
    all_df = all_df.sort_values("_source_dir").groupby(["model", "dtype", "category", "problem"]).last().reset_index()

    return all_df


def compute_quality_summary(quality_df: pd.DataFrame, dtype: str = "int4") -> pd.DataFrame:
    """Compute overall % and per-category pass rates."""
    df = quality_df[quality_df["dtype"] == dtype].copy()
    if df.empty:
        return pd.DataFrame()

    # Overall
    overall = df.groupby("model").agg(
        passed=("passed", "sum"), total=("passed", "count")
    ).reset_index()
    overall["quality_pct"] = (overall["passed"] / overall["total"] * 100).round(1)

    # Per category
    cats = {}
    for cat in ["coding", "tool_calling", "reasoning"]:
        cat_df = df[df["category"] == cat]
        cat_sum = cat_df.groupby("model").agg(
            p=("passed", "sum"), t=("passed", "count")
        ).reset_index()
        cat_sum[cat] = cat_sum.apply(lambda r: f"{int(r['p'])}/{int(r['t'])}", axis=1)
        cats[cat] = cat_sum[["model", cat]]

    result = overall[["model", "quality_pct"]].copy()
    for cat, cat_df in cats.items():
        result = result.merge(cat_df, on="model", how="left")

    return result


def pick_quick_picks(combined: pd.DataFrame) -> List[dict]:
    """Select quick pick recommendations from the combined data."""
    picks = []

    if combined.empty:
        return picks

    # Best overall: highest quality, then fastest
    best = combined.sort_values(["quality_pct", "generation_tps"], ascending=[False, False]).iloc[0]
    picks.append({"use_case": "Best overall", "row": best})

    # Best MoE: fastest MoE model
    moe = combined[combined["arch"].str.contains("MoE")]
    if not moe.empty:
        best_moe = moe.sort_values(["quality_pct", "generation_tps"], ascending=[False, False]).iloc[0]
        if best_moe["name"] != best["name"]:
            picks.append({"use_case": "Best MoE", "row": best_moe})

    # Best coder: highest coding score, then fastest
    if "coding" in combined.columns:
        coders = combined.sort_values(["coding", "generation_tps"], ascending=[False, False])
        for _, row in coders.iterrows():
            if row["name"] not in [p["row"]["name"] for p in picks]:
                picks.append({"use_case": "Best coder", "row": row})
                break

    # Best reasoning: highest quality (which includes reasoning), then fastest, excluding already picked
    if "quality_pct" in combined.columns:
        reasoners = combined.sort_values(["quality_pct", "generation_tps"], ascending=[False, False])
        for _, row in reasoners.iterrows():
            if row["name"] not in [p["row"]["name"] for p in picks]:
                picks.append({"use_case": "Best reasoning", "row": row})
                break

    return picks


def format_model_name(name: str) -> str:
    """Convert internal model name to display name."""
    display_names = {
        "gemma-4-e2b-it": "Gemma 4 E2B-it",
        "gemma-4-e4b-it": "Gemma 4 E4B-it",
        "gemma-4-26b-a4b-it": "Gemma 4 26B-A4B-it",
        "gemma-4-31b-it": "Gemma 4 31B-it",
        "qwen3-coder-30b-a3b": "Qwen3-Coder-30B-A3B",
        "glm-4.7-flash": "GLM-4.7-Flash",
        "lfm2-24b-a2b": "LFM2-24B-A2B",
    }
    if name in display_names:
        return display_names[name]
    # Fallback: capitalize segments
    return name.replace("qwen-3.5-", "Qwen 3.5-").replace(
        "qwen-", "Qwen-"
    ).replace("nemotron-", "Nemotron-").replace(
        "gemma-", "Gemma-"
    ).replace("deepseek-", "DeepSeek-")


def generate_tables(
    models: Optional[List[str]] = None,
) -> str:
    """Generate the full benchmark section markdown."""
    speed_df = load_speed_data(models)
    quality_df = load_quality_data(models)

    if speed_df.empty:
        return "No benchmark data found.\n"

    # int4 speed
    int4_speed = speed_df[speed_df["dtype"] == "int4"].copy()
    int8_speed = speed_df[speed_df["dtype"] == "int8"].copy()

    # Quality summary (int4)
    quality_summary = compute_quality_summary(quality_df, "int4")

    # Combine speed + quality
    combined = int4_speed.rename(columns={"name": "model"}).copy()
    if not quality_summary.empty:
        combined = combined.merge(quality_summary, on="model", how="left")
    else:
        combined["quality_pct"] = None
        combined["coding"] = None
        combined["tool_calling"] = None
        combined["reasoning"] = None

    # Add metadata
    combined["arch"] = combined["model"].map(
        lambda m: MODEL_META.get(m, {}).get("arch", "unknown")
    )
    combined["min_hw"] = combined.apply(
        lambda r: MODEL_META.get(r["model"], {}).get("min_hw", _min_hw_from_memory(r["peak_memory_gib"])),
        axis=1,
    )
    # Rename for output
    combined = combined.rename(columns={"model": "name"})

    # Sort by generation speed descending
    combined = combined.sort_values("generation_tps", ascending=False)

    today = datetime.now().strftime("%B %Y")
    lines = []
    lines.append(f"> M4 Pro 64GB | MLX Metal | int4 quantization | {today}")
    lines.append("> Speed: 1024 prompt tokens, 100 generated tokens, 3 iterations")
    lines.append("> Quality: 46 problems across coding, reasoning, tool calling, math, writing (3 runs each, majority vote)")
    lines.append("")

    # Quick Picks
    has_quality = combined["quality_pct"].notna().any()
    if has_quality:
        picks = pick_quick_picks(combined)
        if picks:
            lines.append("### Quick Picks")
            lines.append("")
            lines.append("| Use Case | Model | Speed | Quality | Memory |")
            lines.append("|---|---|---|---|---|")
            for p in picks:
                r = p["row"]
                name = format_model_name(r["name"])
                qual = f"{r['quality_pct']}%" if pd.notna(r.get("quality_pct")) else "N/A"
                lines.append(
                    f"| **{p['use_case']}** | {name} | {r['generation_tps']:.0f} tok/s | {qual} | {r['peak_memory_gib']:.0f} GiB |"
                )
            lines.append("")

    # All Models table
    lines.append("### All Models")
    lines.append("")
    if has_quality:
        lines.append("| Model | Arch | Gen tok/s | Quality | Coding | Tool Calling | Reasoning | Memory | Min HW |")
        lines.append("|---|---|---:|---:|---|---|---|---:|---|")
    else:
        lines.append("| Model | Arch | Gen tok/s | Prefill tok/s | Memory | Min HW |")
        lines.append("|---|---|---:|---:|---:|---|")

    max_tps = combined["generation_tps"].max()
    top_quality = combined["quality_pct"].max() if has_quality else None

    for _, r in combined.iterrows():
        name = format_model_name(r["name"])
        tps = r["generation_tps"]
        tps_str = f"**{tps:.0f}**" if tps >= max_tps * 0.7 else f"{tps:.0f}"

        if has_quality:
            qual = r.get("quality_pct")
            if pd.notna(qual):
                qual_str = f"**{qual}%**" if qual >= top_quality - 0.1 else f"{qual}%"
            else:
                qual_str = "N/A"
            coding = r.get("coding", "N/A") or "N/A"
            tool_calling = r.get("tool_calling", "N/A") or "N/A"
            reasoning = r.get("reasoning", "N/A") or "N/A"
            lines.append(
                f"| {name} | {r['arch']} | {tps_str} | {qual_str} | {coding} | {tool_calling} | {reasoning} | {r['peak_memory_gib']:.1f} GiB | {r['min_hw']} |"
            )
        else:
            lines.append(
                f"| {name} | {r['arch']} | {tps_str} | {r['prompt_tps']:.0f} | {r['peak_memory_gib']:.1f} GiB | {r['min_hw']} |"
            )

    # int8 collapsible
    if not int8_speed.empty:
        int8_speed = int8_speed.sort_values("generation_tps", ascending=False)
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>int8 speed results</summary>")
        lines.append("")
        lines.append("| Model | Arch | Gen tok/s | Prefill tok/s | Memory | Min HW |")
        lines.append("|---|---|---:|---:|---:|---|")
        for _, r in int8_speed.iterrows():
            name = format_model_name(r["name"])
            arch = MODEL_META.get(r["name"], {}).get("arch", "unknown")
            min_hw = _min_hw_from_memory(r["peak_memory_gib"])
            tps_str = f"**{r['generation_tps']:.1f}**" if r["generation_tps"] >= int8_speed["generation_tps"].max() * 0.7 else f"{r['generation_tps']:.1f}"
            lines.append(
                f"| {name} | {arch} | {tps_str} | {r['prompt_tps']:.0f} | {r['peak_memory_gib']:.1f} GiB | {min_hw} |"
            )
        lines.append("")
        lines.append("</details>")

    return "\n".join(lines)


def update_readme(
    models: Optional[List[str]] = None,
    dry_run: bool = False,
):
    """Update the README.md benchmark section between markers."""
    readme = README_PATH.read_text()

    table_content = generate_tables(models)

    if BEGIN_MARKER in readme and END_MARKER in readme:
        # Replace between markers
        pattern = re.compile(
            re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER),
            re.DOTALL,
        )
        new_readme = pattern.sub(
            f"{BEGIN_MARKER}\n\n{table_content}\n\n{END_MARKER}",
            readme,
        )
    else:
        # Insert markers around existing section
        old_section_start = "## Agentic Coding Model Benchmarks"
        if old_section_start in readme:
            # Find the section and next ## heading
            start_idx = readme.index(old_section_start)
            heading_after = readme.find("\n## ", start_idx + len(old_section_start))
            if heading_after == -1:
                heading_after = len(readme)

            new_readme = (
                readme[:start_idx]
                + f"## Agentic Coding Model Benchmarks (MLX on Apple Silicon)\n\n"
                + f"{BEGIN_MARKER}\n\n{table_content}\n\n{END_MARKER}\n\n"
                + readme[heading_after + 1:]
            )
        else:
            print("WARNING: Could not find benchmark section in README. Appending.")
            new_readme = readme + f"\n\n## Agentic Coding Model Benchmarks (MLX on Apple Silicon)\n\n{BEGIN_MARKER}\n\n{table_content}\n\n{END_MARKER}\n"

    if dry_run:
        print("=== DRY RUN — would write: ===")
        # Just show the table section
        print(table_content)
    else:
        README_PATH.write_text(new_readme)
        print(f"Updated {README_PATH}")


if __name__ == "__main__":
    fire.Fire(update_readme)
