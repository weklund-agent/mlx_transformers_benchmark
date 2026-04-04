"""Regenerate the agentic coding benchmark tables in README.md.

Reads all speed and quality benchmark CSVs, computes the latest results
for each model at int4 and int8 per hardware profile, and rewrites the
benchmark section of the README between the markers.

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

BEGIN_MARKER = "<!-- BEGIN BENCHMARK TABLE -->"
END_MARKER = "<!-- END BENCHMARK TABLE -->"

# Hardware display names
HARDWARE_DISPLAY = {
    "Apple_M4_Pro_10P+4E+20GPU_64GB": "M4 Pro 64GB",
    "Apple_M4_Pro_10P+4E+20GPU_24GB": "M4 Pro 24GB",
    "Apple_M5_Max_XP+XE+40GPU_128GB": "M5 Max 128GB",
}

# Model display names and metadata
MODEL_META = {
    "gemma-4-e2b-it": {"display": "Gemma 4 E2B-it", "arch": "2.3B dense"},
    "gemma-4-e4b-it": {"display": "Gemma 4 E4B-it", "arch": "4.5B dense"},
    "gemma-4-26b-a4b-it": {"display": "Gemma 4 26B-A4B-it", "arch": "3.8B MoE"},
    "gemma-4-31b-it": {"display": "Gemma 4 31B-it", "arch": "31B dense"},
    "qwen3-coder-30b-a3b": {"display": "Qwen3-Coder-30B-A3B", "arch": "3B MoE"},
    "glm-4.7-flash": {"display": "GLM-4.7-Flash", "arch": "3B MoE"},
    "lfm2-24b-a2b": {"display": "LFM2-24B-A2B", "arch": "2B MoE"},
    "qwen-3.5-35b-a3b": {"display": "Qwen 3.5-35B-A3B", "arch": "3B MoE"},
    "qwen-3.5-27b": {"display": "Qwen 3.5-27B", "arch": "27B dense"},
    "qwen-3.5-27b-claude-opus-distilled": {
        "display": "Qwen 3.5-27B Opus Distilled",
        "arch": "27B dense",
    },
    "qwen-3.5-9b": {"display": "Qwen 3.5-9B", "arch": "9B dense"},
    "qwen-3.5-4b": {"display": "Qwen 3.5-4B", "arch": "4B dense"},
    "qwen-3.5-2b": {"display": "Qwen 3.5-2B", "arch": "2B dense"},
    "qwen-3.5-0.8b": {"display": "Qwen 3.5-0.8B", "arch": "0.8B dense"},
    "nemotron-nano-9b-v2": {"display": "Nemotron-Nano-9B-v2", "arch": "9B dense"},
    "nemotron-3-nano-4b": {"display": "Nemotron-3-Nano-4B", "arch": "4B dense"},
    "nemotron-cascade-2-30b-a3b": {
        "display": "Nemotron-Cascade-2-30B-A3B",
        "arch": "3B MoE",
    },
    "Deepseek-R1-0528_Qwen3-8B": {
        "display": "DeepSeek-R1-0528-Qwen3-8B",
        "arch": "8B dense",
    },
    "Deepseek-R1-Distill-7B": {"display": "DeepSeek-R1-Distill-7B", "arch": "7B dense"},
    "gemma-3-4b-it": {"display": "Gemma 3-4B-it", "arch": "4B dense"},
    "gemma-3-4b-it-qat": {"display": "Gemma 3-4B-it QAT", "arch": "4B dense"},
    "gemma-3-1b-it": {"display": "Gemma 3-1B-it", "arch": "1B dense"},
    "gemma-3-1b-it-qat": {"display": "Gemma 3-1B-it QAT", "arch": "1B dense"},
    "gemma-3-12b-it-qat": {"display": "Gemma 3-12B-it QAT", "arch": "12B dense"},
    "gemma-3-27b-it": {"display": "Gemma 3-27B-it", "arch": "27B dense"},
    "qwen-2.5-0.5b-it": {"display": "Qwen 2.5-0.5B-it", "arch": "0.5B dense"},
    "qwen-2.5-3b-it": {"display": "Qwen 2.5-3B-it", "arch": "3B dense"},
    "qwen-2.5-coder-0.5b-it": {"display": "Qwen 2.5-Coder-0.5B", "arch": "0.5B dense"},
    "qwen-2.5-coder-3b-it": {"display": "Qwen 2.5-Coder-3B", "arch": "3B dense"},
    "qwen-3-0.6b-it": {"display": "Qwen 3-0.6B-it", "arch": "0.6B dense"},
    "qwen-3-8B-it": {"display": "Qwen 3-8B-it", "arch": "8B dense"},
    "qwen-3-14B-it": {"display": "Qwen 3-14B-it", "arch": "14B dense"},
    "qwen-3-32B-it": {"display": "Qwen 3-32B-it", "arch": "32B dense"},
    "llama-3.3-70b-it": {"display": "Llama 3.3-70B-it", "arch": "70B dense"},
}


def _min_hw_from_memory(mem_gib: float) -> str:
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


def format_model_name(name: str) -> str:
    meta = MODEL_META.get(name)
    if meta:
        return meta["display"]
    return name


def get_arch(name: str) -> str:
    meta = MODEL_META.get(name)
    if meta:
        return meta["arch"]
    return "unknown"


def load_speed_data(
    models: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load speed data with hardware profile info."""
    files = glob.glob(str(SPEED_ROOT / "**" / "benchmark_results.csv"), recursive=True)
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        p = Path(f)
        df["_source_dir"] = p.parent.name
        df["hardware"] = p.parts[-3]  # hardware directory name
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    # Filter to 1024 prompt tokens, MLX metal
    mask = (all_df["num_prompt_tokens"] == 1024) & (all_df["framework"] == "mlx")
    all_df = all_df[mask].copy()

    if models:
        all_df = all_df[all_df["name"].isin(models)]

    # Keep latest per model/dtype/hardware
    all_df = (
        all_df.sort_values("_source_dir")
        .groupby(["hardware", "name", "dtype"])
        .last()
        .reset_index()
    )

    return all_df[
        ["hardware", "name", "dtype", "generation_tps", "prompt_tps", "peak_memory_gib"]
    ]


def load_quality_data(
    models: Optional[List[str]] = None,
) -> pd.DataFrame:
    files = glob.glob(str(QUALITY_ROOT / "**" / "quality_results.csv"), recursive=True)
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        p = Path(f)
        df["_source_dir"] = p.parent.name
        df["hardware"] = p.parts[-3]
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    if models:
        all_df = all_df[all_df["model"].isin(models)]

    # Keep latest per model/dtype/hardware/problem
    all_df = (
        all_df.sort_values("_source_dir")
        .groupby(["hardware", "model", "dtype", "category", "problem"])
        .last()
        .reset_index()
    )

    return all_df


def compute_quality_summary(
    quality_df: pd.DataFrame, hardware: str, dtype: str = "int4"
) -> pd.DataFrame:
    if quality_df.empty or "dtype" not in quality_df.columns:
        return pd.DataFrame()

    df = quality_df[
        (quality_df["dtype"] == dtype) & (quality_df["hardware"] == hardware)
    ].copy()
    if df.empty:
        return pd.DataFrame()

    # Raw (flat) pass rate
    overall = (
        df.groupby("model")
        .agg(passed=("passed", "sum"), total=("passed", "count"))
        .reset_index()
    )
    overall["quality_pct"] = (overall["passed"] / overall["total"] * 100).round(1)

    # Weighted score via scoring module
    from mtb.quality_benchmarks.scoring import compute_weighted_score

    weighted_scores = []
    for model_name in overall["model"]:
        model_df = df[df["model"] == model_name]
        results = {
            row["problem"]: bool(row["passed"]) for _, row in model_df.iterrows()
        }
        score = compute_weighted_score(results)
        weighted_scores.append(round(score["weighted_score"] * 100, 1))
    overall["weighted_pct"] = weighted_scores

    # Per-category breakdowns
    cats = {}
    for cat in ["coding", "tool_calling", "reasoning"]:
        cat_df = df[df["category"] == cat]
        if cat_df.empty:
            continue
        cat_sum = (
            cat_df.groupby("model")
            .agg(p=("passed", "sum"), t=("passed", "count"))
            .reset_index()
        )
        cat_sum[cat] = cat_sum.apply(lambda r: f"{int(r['p'])}/{int(r['t'])}", axis=1)
        cats[cat] = cat_sum[["model", cat]]

    result = overall[["model", "weighted_pct", "quality_pct"]].copy()
    for cat, cat_df in cats.items():
        result = result.merge(cat_df, on="model", how="left")

    return result


def pick_quick_picks(combined: pd.DataFrame) -> List[dict]:
    picks = []
    if combined.empty:
        return picks

    # Use weighted_pct as primary quality metric, fall back to quality_pct
    qual_col = (
        "weighted_pct"
        if "weighted_pct" in combined.columns and combined["weighted_pct"].notna().any()
        else "quality_pct"
    )
    has_quality = qual_col in combined.columns and combined[qual_col].notna().any()

    if has_quality:
        # Best overall: highest quality, then fastest
        with_quality = combined[combined[qual_col].notna()]
        if not with_quality.empty:
            best = with_quality.sort_values(
                [qual_col, "generation_tps"], ascending=[False, False]
            ).iloc[0]
            picks.append({"use_case": "Best overall", "row": best})

            # Best MoE
            moe = with_quality[with_quality["arch"].str.contains("MoE")]
            if not moe.empty:
                best_moe = moe.sort_values(
                    [qual_col, "generation_tps"], ascending=[False, False]
                ).iloc[0]
                if best_moe["name"] != best["name"]:
                    picks.append({"use_case": "Best MoE", "row": best_moe})

            # Best coder: not yet picked, highest quality then fastest
            for _, row in with_quality.sort_values(
                [qual_col, "generation_tps"], ascending=[False, False]
            ).iterrows():
                if row["name"] not in [p["row"]["name"] for p in picks]:
                    picks.append({"use_case": "Best coder", "row": row})
                    break

            # Best reasoning: not yet picked
            for _, row in with_quality.sort_values(
                [qual_col, "generation_tps"], ascending=[False, False]
            ).iterrows():
                if row["name"] not in [p["row"]["name"] for p in picks]:
                    picks.append({"use_case": "Best reasoning", "row": row})
                    break
    else:
        # No quality data — just pick fastest
        best = combined.sort_values("generation_tps", ascending=False).iloc[0]
        picks.append({"use_case": "Fastest", "row": best})

    return picks


def _format_quality_cell(row, top_quality: float, has_weighted: bool) -> str:
    """Format the Quality column cell.

    Shows weighted score as primary percentage.  When a raw pass rate
    (quality_pct) is also available and differs from the weighted score,
    it is shown as a parenthetical like ``(raw 81/81)``.
    """
    weighted = row.get("weighted_pct") if has_weighted else None
    raw = row.get("quality_pct")

    if pd.notna(weighted):
        primary = weighted
    elif pd.notna(raw):
        primary = raw
    else:
        return "--"

    # Build raw pass-rate footnote when we have both weighted and raw
    raw_note = ""
    if has_weighted and pd.notna(weighted) and pd.notna(raw):
        # Compute passed/total from raw pct stored in quality_pct
        # quality_pct = passed/total * 100, but we don't have total here.
        # Instead just show raw % in parenthetical when it differs from weighted
        if abs(weighted - raw) > 0.05:
            raw_note = f" (raw {raw}%)"

    is_top = top_quality is not None and primary >= top_quality - 0.1
    if is_top:
        return f"**{primary}%**{raw_note}"
    return f"{primary}%{raw_note}"


def generate_hardware_table(
    speed_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    hardware: str,
) -> str:
    """Generate tables for a single hardware profile."""
    hw_speed = speed_df[speed_df["hardware"] == hardware].copy()
    if hw_speed.empty:
        return ""

    int4_speed = hw_speed[hw_speed["dtype"] == "int4"].copy()
    int8_speed = hw_speed[hw_speed["dtype"] == "int8"].copy()

    if int4_speed.empty:
        return ""

    # Quality
    quality_summary = compute_quality_summary(quality_df, hardware, "int4")

    # Combine
    combined = int4_speed.rename(columns={"name": "model"}).copy()
    if not quality_summary.empty:
        combined = combined.merge(quality_summary, on="model", how="left")
    else:
        combined["weighted_pct"] = None
        combined["quality_pct"] = None

    combined["arch"] = combined["model"].map(get_arch)
    combined["min_hw"] = combined.apply(
        lambda r: _min_hw_from_memory(r["peak_memory_gib"]), axis=1
    )
    combined = combined.rename(columns={"model": "name"})
    combined = combined.sort_values("generation_tps", ascending=False)

    has_weighted = (
        "weighted_pct" in combined.columns and combined["weighted_pct"].notna().any()
    )
    has_quality = (
        "quality_pct" in combined.columns and combined["quality_pct"].notna().any()
    )
    has_coding = "coding" in combined.columns
    has_tool = "tool_calling" in combined.columns
    has_reasoning = "reasoning" in combined.columns

    lines = []

    if (has_weighted or has_quality) and has_coding and has_tool and has_reasoning:
        lines.append(
            "| Model | Arch | Gen tok/s | Quality | Coding | Tool Calling | Reasoning | Memory | Min HW |"
        )
        lines.append("|---|---|---:|---:|---|---|---|---:|---|")
    elif has_weighted or has_quality:
        lines.append("| Model | Arch | Gen tok/s | Quality | Memory | Min HW |")
        lines.append("|---|---|---:|---:|---:|---|")
    else:
        lines.append("| Model | Arch | Gen tok/s | Prefill tok/s | Memory | Min HW |")
        lines.append("|---|---|---:|---:|---:|---|")

    max_tps = combined["generation_tps"].max()
    # Top quality uses weighted_pct for bolding when available
    if has_weighted:
        top_quality = combined["weighted_pct"].max()
    elif has_quality:
        top_quality = combined["quality_pct"].max()
    else:
        top_quality = None

    for _, r in combined.iterrows():
        name = format_model_name(r["name"])
        tps = r["generation_tps"]
        tps_str = f"**{tps:.0f}**" if tps >= max_tps * 0.7 else f"{tps:.0f}"

        if (has_weighted or has_quality) and has_coding and has_tool and has_reasoning:
            qual_str = _format_quality_cell(r, top_quality, has_weighted)
            coding = r.get("coding", "--") if pd.notna(r.get("coding")) else "--"
            tool_calling = (
                r.get("tool_calling", "--") if pd.notna(r.get("tool_calling")) else "--"
            )
            reasoning = (
                r.get("reasoning", "--") if pd.notna(r.get("reasoning")) else "--"
            )
            lines.append(
                f"| {name} | {r['arch']} | {tps_str} | {qual_str} | {coding} | {tool_calling} | {reasoning} | {r['peak_memory_gib']:.1f} GiB | {r['min_hw']} |"
            )
        elif has_weighted or has_quality:
            qual_str = _format_quality_cell(r, top_quality, has_weighted)
            lines.append(
                f"| {name} | {r['arch']} | {tps_str} | {qual_str} | {r['peak_memory_gib']:.1f} GiB | {r['min_hw']} |"
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
            arch = get_arch(r["name"])
            min_hw = _min_hw_from_memory(r["peak_memory_gib"])
            tps_str = (
                f"**{r['generation_tps']:.1f}**"
                if r["generation_tps"] >= int8_speed["generation_tps"].max() * 0.7
                else f"{r['generation_tps']:.1f}"
            )
            lines.append(
                f"| {name} | {arch} | {tps_str} | {r['prompt_tps']:.0f} | {r['peak_memory_gib']:.1f} GiB | {min_hw} |"
            )
        lines.append("")
        lines.append("</details>")

    return "\n".join(lines)


def _get_combined_for_hardware(
    speed_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    hardware: str,
) -> pd.DataFrame:
    """Build combined speed+quality dataframe for a hardware profile."""
    hw_speed = speed_df[speed_df["hardware"] == hardware].copy()
    int4_speed = hw_speed[hw_speed["dtype"] == "int4"].copy()
    if int4_speed.empty:
        return pd.DataFrame()

    quality_summary = compute_quality_summary(quality_df, hardware, "int4")
    combined = int4_speed.rename(columns={"name": "model"}).copy()
    if not quality_summary.empty:
        combined = combined.merge(quality_summary, on="model", how="left")
    else:
        combined["weighted_pct"] = None
        combined["quality_pct"] = None

    combined["arch"] = combined["model"].map(get_arch)
    combined["min_hw"] = combined.apply(
        lambda r: _min_hw_from_memory(r["peak_memory_gib"]), axis=1
    )
    combined = combined.rename(columns={"model": "name"})
    combined = combined.sort_values("generation_tps", ascending=False)
    return combined


def generate_cross_hardware_summary(
    speed_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    hardware_profiles: List[str],
) -> str:
    """Generate a cross-hardware Best Models summary table."""
    lines = []
    lines.append("### Best Models by Hardware")
    lines.append("")
    lines.append("| Hardware | Best Overall | Best Fast | Best Coder |")
    lines.append("|---|---|---|---|")

    for hw in hardware_profiles:
        hw_display = HARDWARE_DISPLAY.get(hw, hw)
        combined = _get_combined_for_hardware(speed_df, quality_df, hw)
        if combined.empty:
            continue

        # Use weighted_pct as primary quality metric, fall back to quality_pct
        qual_col = (
            "weighted_pct"
            if "weighted_pct" in combined.columns
            and combined["weighted_pct"].notna().any()
            else "quality_pct"
        )
        has_quality = qual_col in combined.columns and combined[qual_col].notna().any()

        # Best overall: highest quality then fastest (or just fastest if no quality)
        if has_quality:
            with_quality = combined[combined[qual_col].notna()]
            if not with_quality.empty:
                best = with_quality.sort_values(
                    [qual_col, "generation_tps"], ascending=[False, False]
                ).iloc[0]
                best_str = f"{format_model_name(best['name'])} ({best['generation_tps']:.0f} tok/s, {best[qual_col]}%)"
            else:
                best = combined.iloc[0]
                best_str = f"{format_model_name(best['name'])} ({best['generation_tps']:.0f} tok/s)"
        else:
            best = combined.iloc[0]
            best_str = f"{format_model_name(best['name'])} ({best['generation_tps']:.0f} tok/s)"

        # Best fast: fastest model with quality >= 90% if available, else just fastest
        # Must be different from best overall
        if has_quality:
            fast_candidates = combined[combined[qual_col] >= 90.0]
            if fast_candidates.empty:
                fast_candidates = combined
        else:
            fast_candidates = combined
        fastest = fast_candidates.sort_values("generation_tps", ascending=False).iloc[0]
        # Try to pick a different model than best overall
        if fastest["name"] == best["name"] and len(fast_candidates) > 1:
            fastest = fast_candidates.sort_values(
                "generation_tps", ascending=False
            ).iloc[1]
        if has_quality and pd.notna(fastest.get(qual_col)):
            fast_str = f"{format_model_name(fastest['name'])} ({fastest['generation_tps']:.0f} tok/s, {fastest[qual_col]}%)"
        else:
            fast_str = f"{format_model_name(fastest['name'])} ({fastest['generation_tps']:.0f} tok/s)"

        # Best coder: highest quality with perfect coding, then fastest
        if (
            has_quality
            and "coding" in combined.columns
            and combined["coding"].notna().any()
        ):
            with_quality = combined[combined[qual_col].notna()]
            coders = with_quality.sort_values(
                [qual_col, "generation_tps"], ascending=[False, False]
            )
            # Pick first one not already used as best overall
            coder = coders.iloc[0]
            for _, c in coders.iterrows():
                if c["name"] != best["name"]:
                    coder = c
                    break
            coder_str = f"{format_model_name(coder['name'])} ({coder['generation_tps']:.0f} tok/s, {coder[qual_col]}%)"
        else:
            coder_str = "--"

        lines.append(f"| **{hw_display}** | {best_str} | {fast_str} | {coder_str} |")

    return "\n".join(lines)


def generate_tables(
    models: Optional[List[str]] = None,
) -> str:
    """Generate benchmark tables for all hardware profiles."""
    speed_df = load_speed_data(models)
    quality_df = load_quality_data(models)

    if speed_df.empty:
        return "No benchmark data found.\n"

    # Determine hardware profiles with data, sorted by preference
    hardware_order = [
        "Apple_M4_Pro_10P+4E+20GPU_64GB",
        "Apple_M5_Max_XP+XE+40GPU_128GB",
        "Apple_M4_Pro_10P+4E+20GPU_24GB",
    ]
    available_hw = speed_df["hardware"].unique()
    hardware_profiles = [h for h in hardware_order if h in available_hw]
    for h in sorted(available_hw):
        if h not in hardware_profiles:
            hardware_profiles.append(h)

    today = datetime.now().strftime("%B %Y")
    lines = []
    lines.append(f"> MLX Metal | int4 quantization | {today}")
    lines.append("> Speed: 1024 prompt tokens, 100 generated tokens")
    lines.append(
        "> Quality: 81 problems across coding, reasoning, tool calling, math, writing (3 runs each, majority vote)"
    )
    lines.append("")

    # Cross-hardware summary at top
    # Only include primary hardware profiles (skip older/legacy)
    primary_hw = [h for h in hardware_profiles if "24GB" not in h]
    if len(primary_hw) >= 1:
        lines.append(generate_cross_hardware_summary(speed_df, quality_df, primary_hw))
        lines.append("")

    # Per-hardware detailed tables
    for hw in hardware_profiles:
        hw_display = HARDWARE_DISPLAY.get(hw, hw)
        is_legacy = "24GB" in hw

        if is_legacy:
            lines.append(f"<details>")
            lines.append(f"<summary><h3>{hw_display} (legacy)</h3></summary>")
            lines.append("")
        else:
            lines.append(f"### {hw_display}")
            lines.append("")

        table = generate_hardware_table(speed_df, quality_df, hw)
        if table:
            lines.append(table)
        else:
            lines.append("No data available.")

        if is_legacy:
            lines.append("")
            lines.append("</details>")

        lines.append("")

    return "\n".join(lines)


def update_readme(
    models: Optional[List[str]] = None,
    dry_run: bool = False,
):
    """Update the README.md benchmark section between markers."""
    readme = README_PATH.read_text()

    table_content = generate_tables(models)

    if BEGIN_MARKER in readme and END_MARKER in readme:
        pattern = re.compile(
            re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER),
            re.DOTALL,
        )
        new_readme = pattern.sub(
            f"{BEGIN_MARKER}\n\n{table_content}\n\n{END_MARKER}",
            readme,
        )
    else:
        old_section_start = "## Agentic Coding Model Benchmarks"
        if old_section_start in readme:
            start_idx = readme.index(old_section_start)
            heading_after = readme.find("\n## ", start_idx + len(old_section_start))
            if heading_after == -1:
                heading_after = len(readme)
            new_readme = (
                readme[:start_idx]
                + f"## Agentic Coding Model Benchmarks (MLX on Apple Silicon)\n\n"
                + f"{BEGIN_MARKER}\n\n{table_content}\n\n{END_MARKER}\n\n"
                + readme[heading_after + 1 :]
            )
        else:
            new_readme = (
                readme
                + f"\n\n## Agentic Coding Model Benchmarks (MLX on Apple Silicon)\n\n{BEGIN_MARKER}\n\n{table_content}\n\n{END_MARKER}\n"
            )

    if dry_run:
        print("=== DRY RUN — would write: ===")
        print(table_content)
    else:
        README_PATH.write_text(new_readme)
        print(f"Updated {README_PATH}")


if __name__ == "__main__":
    fire.Fire(update_readme)
