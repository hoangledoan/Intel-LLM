from optimum_benchmark import Benchmark, BenchmarkConfig, ProcessConfig, InferenceConfig, IPEXConfig, PyTorchConfig
from optimum_benchmark.logging_utils import setup_logging

setup_logging(level="INFO")

if __name__ == "__main__":
    launcher_config = ProcessConfig(
        numactl=True,
        numactl_kwargs={
            # "physcpubind": "0-159",
            "cpunodebind": "0",
            "membind": "0"
        }
    )

    scenario_config = InferenceConfig(
        latency=True,
        memory=True,
        # warmup_runs=10,
        # iterations=10,
        # duration=10,
        input_shapes={
            "batch_size": 16,
            "sequence_length": 256
        },
        generate_kwargs={
            "max_new_tokens": 32,
            "min_new_tokens": 32
        }
    )

    # backend_config = PyTorchConfig(model="meta-llama/Llama-3.2-3B", device="cpu", no_weights=True, torch_dtype="bfloat16")

    # Backend configuration for IPEX
    backend_config = IPEXConfig(
        model="meta-llama/Llama-3.2-3B",
        device="cpu",
        export=True,
        no_weights=False,
        torch_dtype="bfloat16"
    )

    benchmark_config = BenchmarkConfig(
        name="cpu_ipex_llama",
        scenario=scenario_config,
        launcher=launcher_config,
        backend=backend_config,
    )

    benchmark_report = Benchmark.launch(benchmark_config)

    benchmark_report.save_json("benchmark_report.json")
