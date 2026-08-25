from __future__ import annotations

import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import triton

from .cost_model import MODEL_VERSION, DeviceProfile, estimate, shared_bytes_per_cta
from .profile import make_device_profile, parse_primitive_output, profile_as_dict
from .report import write_report
from .triton_kernels import TRITON_LAUNCHERS, allocate_outputs
from .workloads import WORKLOADS, WorkloadSpec


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CUDA_DIR = PACKAGE_ROOT / "cuda"
WORKLOAD_SOURCE_DIR = CUDA_DIR / "workloads"
DEFAULT_N_VALUES = (4096, 8192, 16384, 32768, 65536)
DEFAULT_CLUSTER_SIZES = (2, 4, 8)
DEFAULT_TRITON_CONFIGS = ((256, 4), (512, 4), (1024, 8))


@dataclass(frozen=True)
class RunConfig:
    workloads: tuple[str, ...] = tuple(WORKLOADS)
    n_values: tuple[int, ...] = DEFAULT_N_VALUES
    cluster_sizes: tuple[int, ...] = DEFAULT_CLUSTER_SIZES
    warmup: int = 20
    iterations: int = 100
    trials: int = 5
    quick: bool = False
    profile_only: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunConfig":
        config = cls(
            workloads=tuple(raw.get("workloads", WORKLOADS)),
            n_values=tuple(int(value) for value in raw.get("n_values", DEFAULT_N_VALUES)),
            cluster_sizes=tuple(
                int(value) for value in raw.get("cluster_sizes", DEFAULT_CLUSTER_SIZES)
            ),
            warmup=int(raw.get("warmup", 20)),
            iterations=int(raw.get("iterations", 100)),
            trials=int(raw.get("trials", 5)),
            quick=bool(raw.get("quick", False)),
            profile_only=bool(raw.get("profile_only", False)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        unknown = sorted(set(self.workloads) - set(WORKLOADS))
        if unknown:
            raise ValueError(f"unknown workloads: {', '.join(unknown)}")
        if not self.workloads and not self.profile_only:
            raise ValueError("at least one workload is required")
        if not self.n_values or any(value <= 0 for value in self.n_values):
            raise ValueError("n_values must contain positive integers")
        if not self.cluster_sizes or any(
            value not in (2, 4, 8) for value in self.cluster_sizes
        ):
            raise ValueError("cluster_sizes must be selected from 2, 4, and 8")
        if min(self.warmup, self.iterations, self.trials) <= 0:
            raise ValueError("warmup, iterations, and trials must be positive")

    def effective(self) -> "RunConfig":
        if not self.quick:
            return self
        return RunConfig(
            workloads=self.workloads,
            n_values=self.n_values[:2],
            cluster_sizes=(8,),
            warmup=2,
            iterations=5,
            trials=1,
            quick=True,
            profile_only=self.profile_only,
        )


def _command_text(command: list[str]) -> str:
    return " ".join(command)


def run_checked(
    command: list[str],
    *,
    cwd: Path = PACKAGE_ROOT,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {_command_text(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(samples: Iterable[float], fraction: float) -> float:
    ordered = sorted(samples)
    if not ordered:
        return math.nan
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(samples: Iterable[float]) -> dict[str, float]:
    finite = [float(value) for value in samples if math.isfinite(float(value))]
    if not finite:
        return {"median": math.nan, "p20": math.nan, "p80": math.nan}
    return {
        "median": statistics.median(finite),
        "p20": percentile(finite, 0.2),
        "p80": percentile(finite, 0.8),
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        return math.nan
    return numerator / denominator


def normalize_outputs(value: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def benchmark_callable(
    function: Callable[[], object], warmup: int, iterations: int, trials: int
) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) / iterations)
    return samples


def assert_outputs_close(
    name: str,
    outputs: tuple[torch.Tensor, ...],
    references: tuple[torch.Tensor, ...],
    atol: float,
    rtol: float,
) -> list[dict[str, float | int]]:
    if len(outputs) != len(references):
        raise RuntimeError(
            f"{name}: output count {len(outputs)} != reference count {len(references)}"
        )
    diagnostics: list[dict[str, float | int]] = []
    for index, (output, reference) in enumerate(zip(outputs, references)):
        if output.dtype == torch.int8:
            difference = torch.abs(output.to(torch.int16) - reference.to(torch.int16))
            mismatch = int(torch.count_nonzero(difference).item())
            maximum = int(torch.max(difference).item())
            mismatch_rate = mismatch / output.numel()
            if maximum > 1 or mismatch_rate > 1.0e-5:
                raise RuntimeError(
                    f"{name} output {index}: {mismatch} int8 mismatches "
                    f"({mismatch_rate:.3e}), maximum difference {maximum}"
                )
            diagnostics.append(
                {
                    "mismatch_count": mismatch,
                    "mismatch_rate": mismatch_rate,
                    "max_integer_difference": maximum,
                }
            )
            continue
        max_error = float(torch.max(torch.abs(output - reference)).item())
        diagnostics.append({"max_abs_error": max_error})
        if not torch.allclose(output, reference, atol=atol, rtol=rtol):
            raise RuntimeError(
                f"{name} output {index}: max error {max_error} exceeds "
                f"atol={atol}, rtol={rtol}"
            )
    return diagnostics


def nvidia_smi_snapshot() -> str:
    return run_checked(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,power.limit,"
            "clocks.max.sm,clocks.current.sm,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    ).stdout.strip()


def compile_cuda_sources(
    architecture: str, build_dir: Path, selected_names: tuple[str, ...]
) -> list[dict[str, Any]]:
    binary_dir = build_dir / "bin"
    binary_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        ("primitive_benchmark", CUDA_DIR / "primitive_benchmark.cu"),
        ("cluster_control", CUDA_DIR / "cluster_control.cu"),
    ]
    sources.extend(
        (WORKLOADS[name].cuda_binary, WORKLOAD_SOURCE_DIR / WORKLOADS[name].cuda_source)
        for name in selected_names
    )
    records: list[dict[str, Any]] = []
    for binary_name, source in sources:
        binary = binary_dir / binary_name
        command = [
            "nvcc",
            "-std=c++17",
            "-O3",
            "-lineinfo",
            "-Xcompiler",
            "-Wall",
            f"-arch={architecture}",
            str(source),
            "-o",
            str(binary),
        ]
        started = time.time()
        result = run_checked(command, timeout=15 * 60)
        records.append(
            {
                "name": binary_name,
                "source": str(source.relative_to(PACKAGE_ROOT)),
                "binary": str(binary),
                "command": command,
                "elapsed_seconds": time.time() - started,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
    return records


def run_primitive_profile(
    build_dir: Path,
) -> tuple[dict[str, Any], DeviceProfile, str]:
    result = run_checked([str(build_dir / "bin/primitive_benchmark")], timeout=20 * 60)
    parsed = parse_primitive_output(result.stdout)
    properties = torch.cuda.get_device_properties(0)
    profile = make_device_profile(parsed, properties)
    return parsed, profile, result.stdout


def parse_cuda_output(output: str) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.startswith("RESULT,"):
            continue
        fields = next(csv.reader([line]))
        parsed.append(
            {
                "cols": int(fields[1]),
                "rows": int(fields[2]),
                "block_variant": fields[3],
                "block_ms": math.nan if fields[4] == "n/a" else float(fields[4]),
                "cluster_variant": fields[6],
                "cluster_ms": math.nan if fields[7] == "n/a" else float(fields[7]),
                "cluster_size": int(fields[9]),
            }
        )
    if not parsed:
        raise RuntimeError(f"CUDA workload produced no RESULT rows:\n{output}")
    return parsed


def run_cuda_trials(
    spec: WorkloadSpec,
    build_dir: Path,
    rows: int,
    n_values: tuple[int, ...],
    cluster_size: int,
    warmup: int,
    iterations: int,
    trials: int,
) -> list[dict[str, Any]]:
    command = [
        str(build_dir / "bin" / spec.cuda_binary),
        "--csv",
        "--rows",
        str(rows),
        "--n-values",
        ",".join(map(str, n_values)),
        "--warmup",
        str(warmup),
        "--iters",
        str(iterations),
        "--cluster-size",
        str(cluster_size),
    ]
    all_rows: list[dict[str, Any]] = []
    for trial in range(trials):
        output = run_checked(command, timeout=30 * 60).stdout
        for row in parse_cuda_output(output):
            row["trial"] = trial
            all_rows.append(row)
    return all_rows


def run_control_trials(
    build_dir: Path,
    rows: int,
    cluster_size: int,
    barriers: int,
    shared_bytes: int,
    warmup: int,
    iterations: int,
    trials: int,
) -> list[float]:
    command = [
        str(build_dir / "bin/cluster_control"),
        "--rows",
        str(rows),
        "--cluster-size",
        str(cluster_size),
        "--barriers",
        str(barriers),
        "--shared-bytes",
        str(shared_bytes),
        "--warmup",
        str(warmup),
        "--iters",
        str(iterations),
    ]
    samples: list[float] = []
    for _ in range(trials):
        output = run_checked(command, timeout=10 * 60).stdout
        control_line = next(
            (line for line in output.splitlines() if line.startswith("CONTROL,")), None
        )
        if control_line is None:
            return []
        samples.append(float(control_line.split(",")[7]))
    return samples


def benchmark_frameworks(
    spec: WorkloadSpec,
    rows: int,
    cols: int,
    warmup: int,
    iterations: int,
    trials: int,
) -> dict[str, Any]:
    torch.manual_seed(1234 + cols)
    inputs = spec.input_factory(rows, cols, torch.device("cuda"))
    launcher = TRITON_LAUNCHERS[spec.name]
    triton_outputs = allocate_outputs(spec.name, inputs)

    with torch.no_grad():
        references = normalize_outputs(spec.torch_function(*inputs))
        launcher(inputs, triton_outputs, 512, 4)
        torch.cuda.synchronize()
        correctness = assert_outputs_close(
            f"{spec.name}/triton", triton_outputs, references, spec.atol, spec.rtol
        )
        del references

    config_times: dict[tuple[int, int], float] = {}
    config_errors: dict[str, str] = {}
    for block_size, num_warps in DEFAULT_TRITON_CONFIGS:
        try:
            samples = benchmark_callable(
                lambda b=block_size, w=num_warps: launcher(
                    inputs, triton_outputs, b, w
                ),
                max(3, warmup // 4),
                max(20, iterations // 4),
                2,
            )
            config_times[(block_size, num_warps)] = statistics.median(samples)
        except Exception as error:
            config_errors[f"{block_size}/{num_warps}"] = str(error)
            print(
                f"Triton config BLOCK={block_size}, warps={num_warps} failed: {error}",
                file=sys.stderr,
                flush=True,
            )
    if not config_times:
        raise RuntimeError(f"no Triton configuration compiled for {spec.name}")
    best_config = min(config_times, key=config_times.get)
    triton_samples = benchmark_callable(
        lambda: launcher(inputs, triton_outputs, *best_config),
        warmup,
        iterations,
        trials,
    )

    torch._dynamo.reset()
    compiled = torch.compile(
        spec.torch_function,
        fullgraph=True,
        dynamic=False,
        mode="max-autotune-no-cudagraphs",
    )
    with torch.no_grad():
        compiled_outputs = normalize_outputs(compiled(*inputs))
        torch.cuda.synchronize()
        references = normalize_outputs(spec.torch_function(*inputs))
        compiled_correctness = assert_outputs_close(
            f"{spec.name}/torch.compile",
            compiled_outputs,
            references,
            spec.atol,
            spec.rtol,
        )
        del compiled_outputs, references
    torch_samples = benchmark_callable(
        lambda: compiled(*inputs), warmup, iterations, trials
    )

    result = {
        "torch_samples_ms": torch_samples,
        "torch_correctness": compiled_correctness,
        "triton_samples_ms": triton_samples,
        "triton_block_size": best_config[0],
        "triton_num_warps": best_config[1],
        "triton_config_medians_ms": {
            f"{block}/{warps}": value
            for (block, warps), value in config_times.items()
        },
        "triton_config_errors": config_errors,
        "triton_correctness": correctness,
    }
    del compiled, triton_outputs, inputs
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    return result


def build_summary(
    selected_names: tuple[str, ...],
    config: RunConfig,
    profile: DeviceProfile,
    raw_cuda: list[dict[str, Any]],
    control_samples: dict[tuple[str, int, int], list[float]],
    framework_results: dict[tuple[str, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cuda_groups: dict[tuple[str, int, str, int], list[float]] = defaultdict(list)
    block_variants: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in raw_cuda:
        name = str(row["workload"])
        cols = int(row["cols"])
        cluster_size = int(row["cluster_size"])
        block_ms = float(row["block_ms"])
        cluster_ms = float(row["cluster_ms"])
        if math.isfinite(block_ms):
            cuda_groups[(name, cols, "block", cluster_size)].append(block_ms)
            block_variants[(name, cols)].add(str(row["block_variant"]))
        if math.isfinite(cluster_ms):
            cuda_groups[(name, cols, "cluster", cluster_size)].append(cluster_ms)

    summary_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    block_trial_cluster_size = max(config.cluster_sizes)
    for name in selected_names:
        spec = WORKLOADS[name]
        rows = min(spec.rows, 128) if config.quick else spec.rows
        for cols in config.n_values:
            block = summarize(
                cuda_groups[(name, cols, "block", block_trial_cluster_size)]
            )
            framework = framework_results.get((name, cols), {})
            torch_stats = summarize(framework.get("torch_samples_ms", []))
            triton_stats = summarize(framework.get("triton_samples_ms", []))
            measured_candidates: list[tuple[float, int]] = []
            model_candidates: list[tuple[float, int, Any]] = []
            for cluster_size in config.cluster_sizes:
                cluster = summarize(
                    cuda_groups[(name, cols, "cluster", cluster_size)]
                )
                control = summarize(
                    control_samples.get((name, cols, cluster_size), [])
                )
                if not math.isfinite(cluster["median"]):
                    continue
                measured_candidates.append((cluster["median"], cluster_size))
                if math.isfinite(control["median"]):
                    model = estimate(
                        spec,
                        cols,
                        cluster_size,
                        control["median"],
                        profile,
                        baseline_ms=block["median"],
                        rows=rows,
                    )
                    model_candidates.append(
                        (model.predicted_delta_ms, cluster_size, model)
                    )
                    cluster_rows.append(
                        {
                            "workload": name,
                            "category": spec.category,
                            "rows": rows,
                            "cols": cols,
                            "cluster_size": cluster_size,
                            "shared_bytes_per_cta": shared_bytes_per_cta(
                                spec, cols, cluster_size
                            ),
                            "cluster_ms": cluster["median"],
                            "cluster_p20_ms": cluster["p20"],
                            "cluster_p80_ms": cluster["p80"],
                            "control_overhead_ms": control["median"],
                            "model_version": MODEL_VERSION,
                            "effective_bandwidth_gbps": model.effective_bandwidth_gbps,
                            "saved_reread_ms": model.saved_reread_ms,
                            "saved_hbm_ms": model.saved_hbm_ms,
                            "local_replay_ms": model.local_replay_ms,
                            "local_store_residual_ms": model.local_store_residual_ms,
                            "local_staging_ms": model.local_staging_ms,
                            "dsmem_store_ms": model.dsmem_store_ms,
                            "predicted_delta_ms": model.predicted_delta_ms,
                            "predicted_profitable": model.predicted_profitable,
                            "measured_delta_vs_block_ms": block["median"]
                            - cluster["median"],
                            "measured_profitable_vs_block": cluster["median"]
                            < block["median"],
                        }
                    )

            if not measured_candidates:
                continue
            best_cluster_ms, best_cluster_size = min(measured_candidates)
            best_cluster_stats = summarize(
                cuda_groups[(name, cols, "cluster", best_cluster_size)]
            )
            predicted_delta, predicted_cluster_size, predicted_model = max(
                model_candidates,
                default=(math.nan, 0, None),
                key=lambda item: item[0],
            )
            baseline_values = {
                "torch.compile": torch_stats["median"],
                "triton": triton_stats["median"],
                "cuda_nodsmem": block["median"],
            }
            finite_baselines = {
                key: value
                for key, value in baseline_values.items()
                if math.isfinite(value)
            }
            if not finite_baselines:
                continue
            best_baseline_name = min(finite_baselines, key=finite_baselines.get)
            best_baseline_ms = finite_baselines[best_baseline_name]
            baseline_stats = {
                "torch.compile": torch_stats,
                "triton": triton_stats,
                "cuda_nodsmem": block,
            }[best_baseline_name]
            measured_profitable = best_cluster_ms < block["median"]
            summary_rows.append(
                {
                    "workload": name,
                    "label": spec.label,
                    "category": spec.category,
                    "rows": rows,
                    "cols": cols,
                    "block_variant": "+".join(
                        sorted(block_variants[(name, cols)])
                    ),
                    "cuda_nodsmem_ms": block["median"],
                    "cuda_nodsmem_p20_ms": block["p20"],
                    "cuda_nodsmem_p80_ms": block["p80"],
                    "torch_compile_ms": torch_stats["median"],
                    "torch_compile_p20_ms": torch_stats["p20"],
                    "torch_compile_p80_ms": torch_stats["p80"],
                    "triton_ms": triton_stats["median"],
                    "triton_p20_ms": triton_stats["p20"],
                    "triton_p80_ms": triton_stats["p80"],
                    "triton_block_size": framework.get("triton_block_size", ""),
                    "triton_num_warps": framework.get("triton_num_warps", ""),
                    "dsmem_ms": best_cluster_ms,
                    "dsmem_p20_ms": best_cluster_stats["p20"],
                    "dsmem_p80_ms": best_cluster_stats["p80"],
                    "measured_cluster_size": best_cluster_size,
                    "best_baseline": best_baseline_name,
                    "best_baseline_ms": best_baseline_ms,
                    "dsmem_vs_torch": safe_ratio(
                        torch_stats["median"], best_cluster_ms
                    ),
                    "dsmem_vs_triton": safe_ratio(
                        triton_stats["median"], best_cluster_ms
                    ),
                    "dsmem_vs_cuda_nodsmem": safe_ratio(
                        block["median"], best_cluster_ms
                    ),
                    "dsmem_vs_best": safe_ratio(best_baseline_ms, best_cluster_ms),
                    "dsmem_vs_best_p20": safe_ratio(
                        baseline_stats["p20"], best_cluster_stats["p80"]
                    ),
                    "dsmem_vs_best_p80": safe_ratio(
                        baseline_stats["p80"], best_cluster_stats["p20"]
                    ),
                    "predicted_cluster_size": predicted_cluster_size,
                    "model_version": MODEL_VERSION,
                    "predicted_delta_ms": predicted_delta,
                    "predicted_profitable": bool(
                        predicted_model and predicted_model.predicted_profitable
                    ),
                    "measured_profitable_vs_cuda": measured_profitable,
                    "model_correct": bool(predicted_model)
                    and predicted_model.predicted_profitable == measured_profitable,
                }
            )
    return summary_rows, cluster_rows


def run_suite(
    raw_config: dict[str, Any], output_dir: Path, requested_gpu: str
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    config = RunConfig.from_dict(raw_config).effective()
    output_dir.mkdir(parents=True, exist_ok=True)
    build_dir = output_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    properties = torch.cuda.get_device_properties(0)
    architecture = f"sm_{properties.major}{properties.minor}"
    device_name = torch.cuda.get_device_name(0)
    initial_metadata = {
        "status": "running",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "requested_gpu": requested_gpu,
        "device": device_name,
        "compute_capability": [properties.major, properties.minor],
        "compile_architecture": architecture,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "triton_version": triton.__version__,
        "cost_model": {
            "version": MODEL_VERSION,
            "uses_dsmem_workload_timing": False,
            "baseline_calibration": "non-DSMEM CUDA time at the same shape",
        },
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "nvidia_smi_before": nvidia_smi_snapshot(),
        "config": asdict(config),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(initial_metadata, indent=2) + "\n", encoding="utf-8"
    )

    compile_started = time.time()
    compile_records = compile_cuda_sources(
        architecture, build_dir, config.workloads if not config.profile_only else ()
    )
    compile_seconds = time.time() - compile_started
    (output_dir / "compile.json").write_text(
        json.dumps(compile_records, indent=2) + "\n", encoding="utf-8"
    )
    primitive, profile, primitive_stdout = run_primitive_profile(build_dir)
    (output_dir / "primitive_stdout.txt").write_text(
        primitive_stdout, encoding="utf-8"
    )
    (output_dir / "primitive_profile.json").write_text(
        json.dumps(
            {"measurements": primitive, "cost_model_profile": profile_as_dict(profile)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if config.profile_only:
        finished = time.time()
        metadata = {
            **initial_metadata,
            "status": "complete",
            "profile": profile_as_dict(profile),
            "compile_seconds": compile_seconds,
            "benchmark_seconds": finished - started - compile_seconds,
            "elapsed_seconds": finished - started,
            "nvidia_smi_after": nvidia_smi_snapshot(),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        return metadata

    raw_cuda: list[dict[str, Any]] = []
    control_samples: dict[tuple[str, int, int], list[float]] = {}
    for name in config.workloads:
        spec = WORKLOADS[name]
        rows = min(spec.rows, 128) if config.quick else spec.rows
        for cluster_size in config.cluster_sizes:
            print(f"CUDA {name}: P={cluster_size}", flush=True)
            workload_rows = run_cuda_trials(
                spec,
                build_dir,
                rows,
                config.n_values,
                cluster_size,
                config.warmup,
                config.iterations,
                config.trials,
            )
            for row in workload_rows:
                row["workload"] = name
                raw_cuda.append(row)
            write_csv(output_dir / "cuda_raw.csv", raw_cuda)

            for cols in config.n_values:
                shared_bytes = shared_bytes_per_cta(spec, cols, cluster_size)
                if shared_bytes > profile.max_shared_bytes_per_cta:
                    continue
                print(f"  control N={cols}, smem={shared_bytes}", flush=True)
                control_samples[(name, cols, cluster_size)] = run_control_trials(
                    build_dir,
                    rows,
                    cluster_size,
                    spec.barrier_count,
                    shared_bytes,
                    config.warmup,
                    config.iterations,
                    config.trials,
                )

    control_rows = [
        {
            "workload": name,
            "cols": cols,
            "cluster_size": cluster_size,
            "trial": trial,
            "overhead_ms": value,
        }
        for (name, cols, cluster_size), values in control_samples.items()
        for trial, value in enumerate(values)
    ]
    write_csv(output_dir / "control_raw.csv", control_rows)

    framework_results: dict[tuple[str, int], dict[str, Any]] = {}
    framework_path = output_dir / "framework_raw.json"
    for name in config.workloads:
        spec = WORKLOADS[name]
        rows = min(spec.rows, 128) if config.quick else spec.rows
        for cols in config.n_values:
            print(f"Frameworks {name}: N={cols}", flush=True)
            framework_results[(name, cols)] = benchmark_frameworks(
                spec,
                rows,
                cols,
                config.warmup,
                config.iterations,
                config.trials,
            )
            framework_path.write_text(
                json.dumps(
                    {
                        f"{saved_name}:{saved_cols}": value
                        for (saved_name, saved_cols), value in framework_results.items()
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    summary_rows, cluster_rows = build_summary(
        config.workloads,
        config,
        profile,
        raw_cuda,
        control_samples,
        framework_results,
    )
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "cluster_sweep.csv", cluster_rows)

    finished = time.time()
    metadata = {
        **initial_metadata,
        "status": "complete",
        "profile": profile_as_dict(profile),
        "compile_seconds": compile_seconds,
        "benchmark_seconds": finished - started - compile_seconds,
        "elapsed_seconds": finished - started,
        "nvidia_smi_after": nvidia_smi_snapshot(),
    }
    aggregate = write_report(output_dir, summary_rows, metadata)
    metadata["aggregate"] = aggregate
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata
