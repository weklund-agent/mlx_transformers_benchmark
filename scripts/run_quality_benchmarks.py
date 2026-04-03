"""Run quality evaluation benchmarks across models and quantizations.

Usage:
    uv run python scripts/run_quality_benchmarks.py
    uv run python scripts/run_quality_benchmarks.py --difficulty hard
    uv run python scripts/run_quality_benchmarks.py --difficulty all
    uv run python scripts/run_quality_benchmarks.py --run_only_benchmarks '["qwen-3.5-4b"]'
    uv run python scripts/run_quality_benchmarks.py --dtypes '["int4","int8"]' --num_runs 5
"""

from pathlib import Path
from typing import Iterable, List, Optional, Union

import fire
import pandas as pd
from tqdm import tqdm

import mtb
import mtb.llm_benchmarks
from mtb.file_io import create_benchmark_output_dir
from mtb.llm_benchmarks.models.base import ModelSpec
from mtb.quality_benchmarks import EVAL_PROBLEMS, EXPERT_EVAL_PROBLEMS, HARD_EVAL_PROBLEMS, TOOL_CALLING_PROBLEMS, run_quality_benchmark
from mtb.select_benchmarks import filter_llm_benchmarks

DEFAULT_OUTPUT_ROOT = mtb.REPO_ROOT / "measurements" / "quality_benchmarks"


def main(
    output_root: Union[str, Path] = DEFAULT_OUTPUT_ROOT,
    difficulty: str = "easy",
    dtypes: Iterable[str] = ("int4", "int8", "bfloat16"),
    num_runs: int = 3,
    cooldown_time_fraction: float = 0.05,
    hf_cache_dir: Optional[str] = mtb.DEFAULT_HF_HOME,
    *,
    run_only_benchmarks: Optional[Iterable[str]] = None,
    run_mlx_metal: bool = True,
    run_ollama_metal: bool = False,
):
    """Run quality evaluation benchmarks.

    Args:
        output_root: Root directory for output.
        difficulty: Problem difficulty - "easy", "hard", or "all".
        dtypes: Data types to evaluate.
        num_runs: Number of runs per problem for majority voting.
        cooldown_time_fraction: Cooldown between problems.
        hf_cache_dir: HuggingFace cache directory.
        run_only_benchmarks: Optional list of model names to run.
        run_mlx_metal: Whether to run MLX with Metal backend.

    """
    from mtb.hf_utils import set_hf_home

    set_hf_home(path=hf_cache_dir, enable_hf_progressbar=False)

    # Select problem set based on difficulty
    if difficulty == "easy":
        problems = EVAL_PROBLEMS
    elif difficulty == "hard":
        problems = HARD_EVAL_PROBLEMS
    elif difficulty == "expert":
        problems = EXPERT_EVAL_PROBLEMS
    elif difficulty == "tool_calling":
        problems = TOOL_CALLING_PROBLEMS
    elif difficulty == "all":
        problems = EVAL_PROBLEMS + HARD_EVAL_PROBLEMS + EXPERT_EVAL_PROBLEMS + TOOL_CALLING_PROBLEMS
    else:
        raise ValueError(f"Unknown difficulty '{difficulty}', must be 'easy', 'hard', 'expert', 'tool_calling', or 'all'")

    model_specs: List[ModelSpec] = list(mtb.llm_benchmarks.MODEL_SPECS)

    benchmarks_to_run = filter_llm_benchmarks(
        model_specs=model_specs,
        dtypes=dtypes,
        run_only_benchmarks=run_only_benchmarks,
        run_mlx_metal=run_mlx_metal,
        run_torch_mps=False,
        run_torch_cpu=False,
        run_torch_cuda=False,
        run_mlx_cpu=False,
        run_lmstudio_metal=False,
        run_lmstudio_mlx=False,
        run_ollama_metal=run_ollama_metal,
    )

    output_dir = create_benchmark_output_dir(
        output_root=output_root,
        benchmark_settings=dict(
            difficulty=difficulty,
            num_runs=num_runs,
            dtypes=list(dtypes),
            run_only_benchmarks=run_only_benchmarks,
        ),
    )
    output_path = output_dir / "quality_results.csv"
    print(f"\nOutput directory: '{output_dir}'")
    print(f"Difficulty: {difficulty} ({len(problems)} problems x {num_runs} runs each)")
    print(f"Problems: {[p.name for p in problems]}\n")

    with tqdm(benchmarks_to_run, position=0) as iterator:
        for config in iterator:
            model_spec = config["model_spec"]
            iterator.set_description(
                f"{model_spec.name}, {config['framework']}+{config['backend']}, "
                f"dtype={config['dtype']}"
            )

            run_quality_benchmark(
                model_spec=model_spec,
                framework=config["framework"],
                backend=config["backend"],
                dtype=config["dtype"],
                output_path=output_path,
                problems=problems,
                num_runs=num_runs,
                cooldown_time_fraction=cooldown_time_fraction,
            )

    # Print summary
    if output_path.exists():
        df = pd.read_csv(output_path)
        print("\n" + "=" * 70)
        print("QUALITY BENCHMARK RESULTS")
        print("=" * 70)

        # Summary by model and dtype
        summary = (
            df.groupby(["model", "dtype"])
            .agg(
                total=("passed", "count"),
                passed=("passed", "sum"),
            )
            .reset_index()
        )
        summary["score"] = summary.apply(
            lambda r: f"{int(r['passed'])}/{int(r['total'])}", axis=1
        )

        # Pivot for easy comparison
        pivot = summary.pivot(index="model", columns="dtype", values="score")
        print("\nOverall Pass Rate (majority vote):")
        print(pivot.to_string())

        # By category
        for category in df["category"].unique():
            cat_df = df[df["category"] == category]
            cat_summary = (
                cat_df.groupby(["model", "dtype"])
                .agg(passed=("passed", "sum"), total=("passed", "count"))
                .reset_index()
            )
            cat_summary["score"] = cat_summary.apply(
                lambda r: f"{int(r['passed'])}/{int(r['total'])}", axis=1
            )
            cat_pivot = cat_summary.pivot(index="model", columns="dtype", values="score")
            print(f"\n{category.replace('_', ' ').title()}:")
            print(cat_pivot.to_string())

        # Show any failures
        failures = df[~df["passed"]]
        if not failures.empty:
            print(f"\nFailed problems ({len(failures)}):")
            for _, row in failures.iterrows():
                print(f"  {row['model']} {row['dtype']}: {row['problem']} "
                      f"({row['pass_count']}/{row['num_runs']} runs)")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
