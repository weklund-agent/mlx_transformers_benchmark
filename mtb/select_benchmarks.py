from typing import Dict, List, Type, Union

from mtb.layer_benchmarks.base_layer_benchmark import BaseLayerBenchmark
from mtb.llm_benchmarks.models.base import ModelSpec
from mtb.system.memory import estimate_model_size, get_available_ram_gib


def benchmark_name_to_benchmark_class(
    benchmark_name: str,
) -> Type:
    """Get a benchmark class from a vague string identifier.

    Args:
        benchmark_name: String identifier for a benchmark.

    """
    original_benchmark_name = benchmark_name
    benchmark_name = benchmark_name.lower().replace("_", "")

    from mtb.layer_benchmarks import LAYER_BENCHMARKS as layer_benchmarks
    from mtb.llm_benchmarks import MODEL_SPECS as model_specs

    name_to_benchmark_class = dict()
    for layer_benchmark in layer_benchmarks:
        name = layer_benchmark.__name__.replace("Benchmark", "")
        name_to_benchmark_class[name.lower()] = layer_benchmark
    for model_spec in model_specs:
        name = model_spec.name
        name_to_benchmark_class[name.lower()] = model_spec

    try:
        benchmark_class = name_to_benchmark_class[benchmark_name]
    except KeyError:
        raise ValueError(
            f"Could not find benchmark class for name '{original_benchmark_name}', "
            f"must be one of {list(name_to_benchmark_class.keys())}."
        )

    return benchmark_class


def filter_benchmarks(
    benchmarks: List[BaseLayerBenchmark],
    run_only_benchmarks: Union[str, List[str]],
) -> List[BaseLayerBenchmark]:
    """Filter given benchmarks, return only the ones meeting the criterion.

    Args:
        benchmarks: List of benchmarks to filter.
        run_only_benchmarks: List of benchmark names to include.

    """
    if isinstance(run_only_benchmarks, str):
        run_only_benchmarks = [run_only_benchmarks]

    run_only_benchmarks = [
        benchmark_name.lower().replace("_", "")
        for benchmark_name in run_only_benchmarks
    ]

    benchmarks_to_keep = []
    for benchmark in benchmarks:
        benchmark_name = benchmark.name.lower().replace("_", "")

        for match in run_only_benchmarks:
            if benchmark_name.startswith(match):
                benchmarks_to_keep.append(benchmark)
                print(f"  keeping benchmark '{benchmark.name}', matched '{match}'")
                break

    if len(benchmarks_to_keep) == 0:
        raise ValueError(
            f"No benchmarks to run after filtering! "
            f"Check the filter criterion '{run_only_benchmarks}' "
            f"and benchmarks {[b.name for b in benchmarks]}."
        )

    return benchmarks_to_keep


def filter_llm_benchmarks(
    model_specs: List[ModelSpec],
    dtypes: List[str],
    run_only_benchmarks: Union[str, List[str]],
    run_mlx_metal: bool = True,
    run_torch_mps: bool = False,
    run_torch_cpu: bool = False,
    run_torch_cuda: bool = False,
    run_mlx_cpu: bool = False,
    run_lmstudio_metal: bool = False,
    run_lmstudio_mlx: bool = False,
    run_ollama_metal: bool = False,
    run_anthropic_api: bool = False,
    verbose: bool = True,
) -> List[Dict]:
    """Determine which LLM benchmarks to run based on boolean flags."""

    def print_or_not(message: str, **kwargs):
        if verbose:
            print(message, **kwargs)

    print_or_not("Filtering model_specs:")

    if run_only_benchmarks is not None:
        model_specs = filter_benchmarks(
            model_specs,
            run_only_benchmarks=run_only_benchmarks,
        )

    flag_to_framework_backend = [
        (run_mlx_metal, dict(framework="mlx", backend="metal")),
        (run_mlx_cpu, dict(framework="mlx", backend="cpu")),
        (run_torch_mps, dict(framework="torch", backend="mps")),
        (run_torch_cpu, dict(framework="torch", backend="cpu")),
        (run_torch_cuda, dict(framework="torch", backend="cuda")),
        (run_lmstudio_metal, dict(framework="lmstudio", backend="metal+llama.cpp")),
        (run_lmstudio_mlx, dict(framework="lmstudio_mlx", backend="mlx")),
        (run_ollama_metal, dict(framework="ollama", backend="metal+llama.cpp")),
        (run_anthropic_api, dict(framework="anthropic", backend="api")),
    ]

    available_memory = get_available_ram_gib()

    print_or_not("Filtering benchmarks by framework:")
    benchmark_to_run: List[Dict] = []
    for model_spec in model_specs:
        print_or_not(f"  Model {model_spec.name}:")

        for dtype in dtypes:
            # Skip memory check for API models (num_params=0 or unknown dtype)
            if model_spec.num_params > 0 and dtype in (
                "float32", "bfloat16", "float16", "int8", "int6", "int4", "int3",
            ):
                memory_needed_gib = estimate_model_size(
                    num_params=model_spec.num_params,
                    dtype=dtype,
                )
            else:
                memory_needed_gib = 0

            if memory_needed_gib > available_memory:
                # model too large to load (probably)
                print_or_not(
                    f"    skipping {dtype:>10}: "
                    f"it needs {memory_needed_gib:.3f} GiB memory to load parameters, "
                    f"but only {available_memory:.3f} GiB is available."
                )
                continue

            for flag, framework_backend in flag_to_framework_backend:
                if not flag:
                    # user disabled
                    continue

                framework = framework_backend["framework"]
                backend = framework_backend["backend"]
                framework_backend = f"{framework}, {backend}, {dtype}"
                print_or_not(f"    {framework_backend:<40}- ", end="")

                spec_has_model_id = model_spec.has_model_id(
                    framework=framework,
                    dtype=dtype,
                )
                if not spec_has_model_id:
                    # no model_id available
                    print_or_not(f"skipping, no known model id.")
                    continue

                # if we made it here, we have a model_id, and the model will fit
                benchmark_settings = dict(
                    model_spec=model_spec,
                    framework=framework,
                    backend=backend,
                    dtype=dtype,
                )
                model_id = model_spec.model_ids[framework][dtype]
                print_or_not(f"running, model_id='{model_id}'")
                benchmark_to_run.append(benchmark_settings)

    return benchmark_to_run
