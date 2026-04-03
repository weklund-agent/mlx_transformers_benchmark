# Contributing to `mtb`

There are two easy ways to contribute: adding a new measurement, or adding a new benchmark task.

You will need:
 - [`uv`](https://github.com/astral-sh/uv) to manage dependencies, available as [homebrew](https://formulae.brew.sh/formula/uv)

First, [fork the repo](https://github.com/weklund-agent/mlx_transformers_benchmark/fork) and set up a local environment using `uv`:
```
git clone git@github.com:<your-username>/mlx_transformers_benchmark.git
cd mlx_transformers_benchmark
make setup
```

You can check if installation was successful by running the tests:
```
make test
```

### Adding a measurement

1. To run a specific llm benchmark:
   ```
   python scripts/run_llm_benchmarks.py \
       --run_only_benchmarks qwen-2.5-0.5b-it \
       --dtypes \["int4","int8"\] \
       --num_iterations 3 \
   ```
   This will create a new measurement file in the `measurements` folder.
   By default, each measurement stores device information and software versions.

2. Check that the new measurements look sensible by visualizing results:
   ```
   make show-llm
   ```
   This will open a browser window and show individual measurements. Do check for outliers!

3. Optionally, add your github username to the `settings.json` file for tracking purposes:
   ```
   - "contributor": "",
   + "contributor": "aukejw",
   ```

4. Add the new files, and commit the changes:
   ```
   git add measurements/
   git commit -am "Adding a new measurement"
   git push
   ```

5. Submit a PR from your fork.


Alternatively, you can run *all* llm benchmarks using:
```
make run-llm-benchmarks
```
This will take more than 40 minutes excluding download times, and the larger 
`bfloat16` models will certainly take up memory. We will automatically skip models 
that do not fit, but make sure you aren't busy with other tasks!


### Adding a benchmark task

All benchmarks are located in `mtb/llm_benchmarks` and `mtb/layer_benchmarks`. 

#### LLM benchmarks 

To create a new LLM benchmarks, you must define at least the huggingface `model_id` for each dtype
(e.g. `mlx-community/gemma-3-1b-it-4bit`), the number of parameters, and the benchmark name.
For a Gemma 3 model with 1 billion parameters, this could look something like the below:

```
class Gemma3_1B_it_Benchmark(GemmaBenchmark):
    dtype_to_model_id = {
        torch.bfloat16: "google/gemma-3-1b-it",
        mx.bfloat16: "mlx-community/gemma-3-1b-it-bf16",
        mx.int8: "mlx-community/gemma-3-1b-it-8bit",
    }
    name = "gemma-3-1b-it"
    num_params = 1e9
```

For more examples, see 
[the gemma benchmark](https://github.com/weklund-agent/mlx_transformers_benchmark/blob/main/mtb/llm_benchmarks/gemma.py).

#### Layer benchmarks

Layer benchmarks will run operators in torch and mlx, and therefore require you to define 
`setup_torch`, `setup_mlx` and `run_torch`, `run_mlx` functions. For an example, see 
[the mhsa benchmark](https://github.com/weklund-agent/mlx_transformers_benchmark/blob/main/mtb/layer_benchmarks/mhsa.py).
