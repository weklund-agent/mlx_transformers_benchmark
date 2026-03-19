"""Run quality evaluation benchmarks to measure reasoning accuracy across quantizations."""

import gc
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from mtb.llm_benchmarks.models.base import ModelSpec
from mtb.quality_benchmarks.eval_problems import EVAL_PROBLEMS, EvalProblem


def run_quality_benchmark(
    model_spec: ModelSpec,
    framework: str,
    backend: str,
    dtype: str,
    output_path: Path,
    problems: Optional[List[EvalProblem]] = None,
    num_runs: int = 3,
    cooldown_time_fraction: float = 0.05,
) -> pd.DataFrame:
    """Run quality evaluation for a single model+dtype configuration.

    Each problem is run num_runs times to account for sampling variance.
    A problem "passes" if it passes in the majority of runs.

    Args:
        model_spec: The model specification.
        framework: Framework identifier (e.g. "mlx").
        backend: Backend identifier (e.g. "metal").
        dtype: Data type identifier (e.g. "int4", "int8", "bfloat16").
        output_path: Path to save results CSV.
        problems: List of eval problems. Defaults to EVAL_PROBLEMS.
        num_runs: Number of times to run each problem (for majority vote).
        cooldown_time_fraction: Fraction of elapsed time to sleep between problems.

    Returns:
        DataFrame with results.

    """
    if problems is None:
        problems = EVAL_PROBLEMS

    from mtb.llm_benchmarks.run_llm_benchmark import create_benchmark

    benchmark = create_benchmark(
        model_spec=model_spec,
        framework=framework,
        backend=backend,
        dtype=dtype,
        max_num_tokens=512,  # will be overridden per problem
    )

    try:
        benchmark.setup()

        results = []
        for problem in tqdm(problems, desc=f"{model_spec.name} {dtype}", position=1, leave=False):
            benchmark.max_num_tokens = problem.max_tokens

            prompt = problem.prompt
            run_passes = []
            run_responses = []
            run_gen_times = []
            run_gen_tps = []
            run_gen_tokens = []

            start_time = time.perf_counter()
            for run_idx in range(num_runs):
                measurement = benchmark.run_once(prompt=prompt)
                response = measurement.response
                passed = problem.check(response)
                run_passes.append(passed)
                run_responses.append(response)
                run_gen_times.append(measurement.generation_time_sec)
                run_gen_tps.append(measurement.generation_tps)
                run_gen_tokens.append(measurement.num_generated_tokens)

            elapsed = time.perf_counter() - start_time
            time.sleep(cooldown_time_fraction * elapsed)

            # Majority vote
            pass_count = sum(run_passes)
            majority_pass = pass_count > num_runs / 2

            # Average speed metrics across runs
            avg_gen_time = sum(run_gen_times) / len(run_gen_times)
            avg_gen_tps = sum(run_gen_tps) / len(run_gen_tps)
            avg_gen_tokens = sum(run_gen_tokens) / len(run_gen_tokens)

            result = dict(
                model=model_spec.name,
                framework=framework,
                backend=backend,
                dtype=dtype,
                category=problem.category,
                problem=problem.name,
                pass_count=pass_count,
                num_runs=num_runs,
                passed=majority_pass,
                avg_generation_time_sec=round(avg_gen_time, 2),
                avg_generation_tps=round(avg_gen_tps, 1),
                avg_tokens_generated=round(avg_gen_tokens, 0),
                # Store the first response for debugging
                sample_response=run_responses[0][:500],
            )
            results.append(result)

        benchmark.teardown()

    except Exception as e:
        print(f"\n  Exception for '{model_spec.name}' {dtype}: {e}")
        return pd.DataFrame()

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(
        output_path,
        index=False,
        mode="a",
        header=(not output_path.exists()),
    )

    return df
