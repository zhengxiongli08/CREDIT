#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import torch
import triton

from cost_model import MODEL_VERSION, RTX5090_PROFILE, estimate, shared_bytes_per_cta
from triton_kernels import TRITON_LAUNCHERS, allocate_outputs
from workloads import WORKLOADS, WorkloadSpec


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONTROL_SOURCE = HERE / "cluster_control.cu"
CONTROL_BINARY = HERE / "bin/cluster_control"
DEFAULT_N_VALUES = (4096, 8192, 16384, 32768, 65536)
DEFAULT_CONFIGS = ((256, 4), (512, 4), (1024, 8))


def parse_int_list(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified RTX 5090 DSMEM, CUDA, Triton, and torch.compile evaluation."
    )
    parser.add_argument(
        "--workloads",
        default=",".join(WORKLOADS),
        help="comma-separated workload names",
    )
    parser.add_argument(
        "--n-values", type=parse_int_list, default=list(DEFAULT_N_VALUES)
    )
    parser.add_argument(
        "--cluster-sizes", type=parse_int_list, default=[2, 4, 8]
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-cuda", action="store_true")
    parser.add_argument("--skip-frameworks", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=HERE / "results/rtx5090"
    )
    return parser.parse_args()


def normalize_outputs(value) -> tuple[torch.Tensor, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


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


def summarize(samples: list[float]) -> dict[str, float]:
    finite = [value for value in samples if math.isfinite(value)]
    if not finite:
        return {"median": math.nan, "p20": math.nan, "p80": math.nan}
    return {
        "median": statistics.median(finite),
        "p20": percentile(finite, 0.2),
        "p80": percentile(finite, 0.8),
    }


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
            max_difference = int(torch.max(difference).item())
            mismatch_rate = mismatch / output.numel()
            if max_difference > 1 or mismatch_rate > 1.0e-5:
                raise RuntimeError(
                    f"{name} output {index}: {mismatch} int8 mismatches "
                    f"({mismatch_rate:.3e}), maximum difference {max_difference}"
                )
            diagnostics.append(
                {
                    "mismatch_count": mismatch,
                    "mismatch_rate": mismatch_rate,
                    "max_integer_difference": max_difference,
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


def parse_cuda_output(output: str) -> list[dict[str, object]]:
    parsed: list[dict[str, object]] = []
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
        raise RuntimeError(f"CUDA benchmark produced no RESULT rows:\n{output}")
    return parsed


def run_checked(command: list[str], env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def run_cuda_trials(
    spec: WorkloadSpec,
    rows: int,
    n_values: list[int],
    cluster_size: int,
    warmup: int,
    iterations: int,
    trials: int,
) -> list[dict[str, object]]:
    arguments = [
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
    environment = os.environ.copy()
    environment["ARCH"] = "sm_120"
    all_rows: list[dict[str, object]] = []
    for trial in range(trials):
        if trial == 0 or not spec.binary.exists():
            command = ["bash", str(spec.run_script), *arguments]
        else:
            command = [str(spec.binary), *arguments]
        output = run_checked(command, env=environment)
        for row in parse_cuda_output(output):
            row["trial"] = trial
            all_rows.append(row)
    return all_rows


def compile_control() -> None:
    CONTROL_BINARY.parent.mkdir(parents=True, exist_ok=True)
    if (
        CONTROL_BINARY.exists()
        and CONTROL_BINARY.stat().st_mtime >= CONTROL_SOURCE.stat().st_mtime
    ):
        return
    run_checked(
        [
            "nvcc",
            "-std=c++17",
            "-O3",
            "-lineinfo",
            "-arch=sm_120",
            str(CONTROL_SOURCE),
            "-o",
            str(CONTROL_BINARY),
        ]
    )


def run_control_trials(
    rows: int,
    cluster_size: int,
    barriers: int,
    shared_bytes: int,
    warmup: int,
    iterations: int,
    trials: int,
) -> list[float]:
    command = [
        str(CONTROL_BINARY),
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
        output = run_checked(command)
        control_line = next(
            (line for line in output.splitlines() if line.startswith("CONTROL,")),
            None,
        )
        if control_line is None:
            return []
        fields = control_line.split(",")
        samples.append(float(fields[7]))
    return samples


def benchmark_frameworks(
    spec: WorkloadSpec,
    rows: int,
    cols: int,
    warmup: int,
    iterations: int,
    trials: int,
) -> dict[str, object]:
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
    for block_size, num_warps in DEFAULT_CONFIGS:
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
            print(
                f"  Triton config BLOCK={block_size}, warps={num_warps} failed: {error}",
                file=sys.stderr,
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
        assert_outputs_close(
            f"{spec.name}/torch.compile",
            compiled_outputs,
            normalize_outputs(spec.torch_function(*inputs)),
            spec.atol,
            spec.rtol,
        )
        del compiled_outputs
    torch_samples = benchmark_callable(
        lambda: compiled(*inputs), warmup, iterations, trials
    )

    result = {
        "torch_samples_ms": torch_samples,
        "triton_samples_ms": triton_samples,
        "triton_block_size": best_config[0],
        "triton_num_warps": best_config[1],
        "triton_config_medians_ms": {
            f"{block}/{warps}": value
            for (block, warps), value in config_times.items()
        },
        "triton_correctness": correctness,
    }
    del compiled, triton_outputs, inputs
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    selected_names = [name.strip() for name in args.workloads.split(",") if name.strip()]
    unknown = sorted(set(selected_names) - set(WORKLOADS))
    if unknown:
        raise SystemExit(f"unknown workloads: {', '.join(unknown)}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device_name = torch.cuda.get_device_name(0)
    if "RTX 5090" not in device_name:
        raise SystemExit(f"expected RTX 5090, found {device_name}")

    n_values = list(args.n_values)
    cluster_sizes = list(args.cluster_sizes)
    warmup = args.warmup
    iterations = args.iters
    trials = args.trials
    if args.quick:
        n_values = n_values[:2]
        cluster_sizes = [8]
        warmup = 2
        iterations = 5
        trials = 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cuda_checkpoint = args.output_dir / "cuda_raw.csv"
    control_checkpoint = args.output_dir / "control_raw.csv"
    framework_checkpoint = args.output_dir / "framework_raw.json"
    metadata_path = args.output_dir / "metadata.json"
    previous_metadata: dict[str, object] = {}
    if metadata_path.exists():
        try:
            previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_metadata = {}
    started = time.time()
    raw_cuda: list[dict[str, object]] = []
    control_samples: dict[tuple[str, int, int], list[float]] = {}
    framework_results: dict[tuple[str, int], dict[str, object]] = {}

    if args.skip_cuda:
        if not cuda_checkpoint.exists() or not control_checkpoint.exists():
            raise SystemExit(
                "--skip-cuda requires cuda_raw.csv and control_raw.csv in the output directory"
            )
        with cuda_checkpoint.open(newline="", encoding="utf-8") as source:
            raw_cuda = list(csv.DictReader(source))
        with control_checkpoint.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                key = (
                    str(row["workload"]),
                    int(row["cols"]),
                    int(row["cluster_size"]),
                )
                control_samples.setdefault(key, []).append(float(row["overhead_ms"]))

    if framework_checkpoint.exists():
        saved_frameworks = json.loads(framework_checkpoint.read_text(encoding="utf-8"))
        for key, value in saved_frameworks.items():
            name, cols = key.rsplit(":", 1)
            framework_results[(name, int(cols))] = value

    if not args.skip_cuda:
        compile_control()
        for name in selected_names:
            spec = WORKLOADS[name]
            rows = min(spec.rows, 128) if args.quick else spec.rows
            for cluster_size in cluster_sizes:
                print(f"CUDA {name}: P={cluster_size}", flush=True)
                rows_from_run = run_cuda_trials(
                    spec,
                    rows,
                    n_values,
                    cluster_size,
                    warmup,
                    iterations,
                    trials,
                )
                for row in rows_from_run:
                    row["workload"] = name
                    raw_cuda.append(row)
                for cols in n_values:
                    shared_bytes = shared_bytes_per_cta(spec, cols, cluster_size)
                    if shared_bytes > RTX5090_PROFILE.max_shared_bytes_per_cta:
                        continue
                    print(
                        f"  control N={cols}, smem={shared_bytes}", flush=True
                    )
                    control_samples[(name, cols, cluster_size)] = run_control_trials(
                        rows,
                        cluster_size,
                        spec.barrier_count,
                        shared_bytes,
                        warmup,
                        iterations,
                        trials,
                    )

        write_csv(cuda_checkpoint, raw_cuda)
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
        write_csv(control_checkpoint, control_rows)

    if not args.skip_frameworks:
        for name in selected_names:
            spec = WORKLOADS[name]
            rows = min(spec.rows, 128) if args.quick else spec.rows
            for cols in n_values:
                if (name, cols) in framework_results:
                    print(f"Frameworks {name}: N={cols} (checkpoint)", flush=True)
                    continue
                print(f"Frameworks {name}: N={cols}", flush=True)
                framework_results[(name, cols)] = benchmark_frameworks(
                    spec, rows, cols, warmup, iterations, trials
                )
                framework_checkpoint.write_text(
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

    summary_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    for name in selected_names:
        spec = WORKLOADS[name]
        rows = min(spec.rows, 128) if args.quick else spec.rows
        block_trial_cluster_size = max(cluster_sizes)
        for cols in n_values:
            block = summarize(
                cuda_groups[(name, cols, "block", block_trial_cluster_size)]
            )
            framework = framework_results.get((name, cols), {})
            torch_stats = summarize(list(framework.get("torch_samples_ms", [])))
            triton_stats = summarize(list(framework.get("triton_samples_ms", [])))
            measured_candidates: list[tuple[float, int]] = []
            model_candidates: list[tuple[float, int, object]] = []
            for cluster_size in cluster_sizes:
                cluster = summarize(
                    cuda_groups[(name, cols, "cluster", cluster_size)]
                )
                control = summarize(
                    control_samples.get((name, cols, cluster_size), [])
                )
                if not math.isfinite(cluster["median"]):
                    continue
                measured_candidates.append((cluster["median"], cluster_size))
                model = estimate(
                    spec,
                    cols,
                    cluster_size,
                    control["median"],
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
            best_cluster_ms, best_cluster_size = min(
                measured_candidates, default=(math.nan, 0)
            )
            best_cluster_stats = summarize(
                cuda_groups[(name, cols, "cluster", best_cluster_size)]
            )
            predicted_delta, predicted_cluster_size, predicted_model = max(
                model_candidates, default=(math.nan, 0, None), key=lambda item: item[0]
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
            best_baseline_name = min(
                finite_baselines, key=finite_baselines.get, default=""
            )
            best_baseline_ms = finite_baselines.get(best_baseline_name, math.nan)
            baseline_stats = {
                "torch.compile": torch_stats,
                "triton": triton_stats,
                "cuda_nodsmem": block,
            }.get(best_baseline_name, summarize([]))
            summary_rows.append(
                {
                    "workload": name,
                    "label": spec.label,
                    "category": spec.category,
                    "rows": rows,
                    "cols": cols,
                    "block_variant": "+".join(sorted(block_variants[(name, cols)])),
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
                    "dsmem_vs_torch": torch_stats["median"] / best_cluster_ms,
                    "dsmem_vs_triton": triton_stats["median"] / best_cluster_ms,
                    "dsmem_vs_cuda_nodsmem": block["median"] / best_cluster_ms,
                    "dsmem_vs_best": best_baseline_ms / best_cluster_ms,
                    "dsmem_vs_best_p20": baseline_stats["p20"]
                    / best_cluster_stats["p80"],
                    "dsmem_vs_best_p80": baseline_stats["p80"]
                    / best_cluster_stats["p20"],
                    "predicted_cluster_size": predicted_cluster_size,
                    "model_version": MODEL_VERSION,
                    "predicted_delta_ms": predicted_delta,
                    "predicted_profitable": bool(
                        predicted_model and predicted_model.predicted_profitable
                    ),
                    "measured_profitable_vs_cuda": best_cluster_ms
                    < block["median"],
                    "model_correct": bool(predicted_model)
                    and predicted_model.predicted_profitable
                    == (best_cluster_ms < block["median"]),
                }
            )

    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "cluster_sweep.csv", cluster_rows)
    write_csv(args.output_dir / "cuda_raw.csv", raw_cuda)
    finished = time.time()
    preserve_benchmark_provenance = args.skip_cuda and bool(previous_metadata)
    metadata = {
        "timestamp_unix": previous_metadata.get("timestamp_unix", finished)
        if preserve_benchmark_provenance
        else finished,
        "elapsed_seconds": previous_metadata.get(
            "elapsed_seconds", finished - started
        )
        if preserve_benchmark_provenance
        else finished - started,
        "summary_timestamp_unix": finished,
        "summary_elapsed_seconds": finished - started,
        "summary_rebuilt_from_checkpoints": args.skip_cuda,
        "device": device_name,
        "compute_capability": torch.cuda.get_device_capability(0),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "profile": RTX5090_PROFILE.__dict__,
        "cost_model": {
            "version": MODEL_VERSION,
            "uses_dsmem_workload_timing": False,
            "baseline_calibration": "non-DSMEM CUDA time at the same shape",
        },
        "warmup": warmup,
        "iterations": iterations,
        "trials": trials,
        "n_values": n_values,
        "cluster_sizes": cluster_sizes,
        "workloads": selected_names,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_dir / 'summary.csv'}")
    print(f"Wrote {args.output_dir / 'cluster_sweep.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
