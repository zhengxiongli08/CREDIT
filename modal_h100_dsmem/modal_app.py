from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import modal


HERE = Path(__file__).resolve().parent
REMOTE_SOURCE = "/root/benchmark.cu"
REMOTE_BINARY = "/tmp/dsmem_benchmark"

image = modal.Image.from_registry(
    "nvidia/cuda:12.8.1-devel-ubuntu22.04",
    add_python="3.11",
).add_local_file(HERE / "benchmark.cu", REMOTE_SOURCE)

app = modal.App("h100-dsmem-profiler")


METRIC_NAMES = (
    "Cluster sync latency",
    "Local SMEM read latency",
    "DSMEM read latency",
    "Local SMEM read throughput",
    "DSMEM read throughput",
    "Local SMEM store throughput",
    "DSMEM store issue throughput",
    "Store visibility round trip",
    "One-way visibility estimate",
)


def parse_benchmark_output(output: str) -> dict[str, object]:
    metrics: dict[str, dict[str, object]] = {}
    placements: dict[str, dict[str, int]] = {}
    device: dict[str, object] = {}

    metric_pattern = re.compile(
        rf"^({'|'.join(re.escape(name) for name in METRIC_NAMES)})\s*:\s*"
        r"([0-9.]+)\s+([^()]+?)\s+\(min\s+([0-9.]+),\s+max\s+([0-9.]+)\)$"
    )
    placement_pattern = re.compile(
        r"^(cluster sync|read latency|throughput|visibility)\s*:\s*"
        r"SM\s+(\d+)\s+->\s+SM\s+(\d+)$"
    )

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("GPU Model:"):
            device["name"] = line.split(":", 1)[1].strip()
        elif line.startswith("Clock frequency:"):
            device["reported_clock_ghz"] = float(
                line.split(":", 1)[1].strip().split()[0]
            )
        elif line.startswith("SM count:"):
            device["sm_count"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("Dynamic shared memory/CTA"):
            match = re.search(r":\s*(\d+)\s+bytes", line)
            if match:
                device["dynamic_shared_bytes_per_cta"] = int(match.group(1))

        metric_match = metric_pattern.match(line)
        if metric_match:
            name, median, unit, minimum, maximum = metric_match.groups()
            metrics[name] = {
                "median": float(median),
                "minimum": float(minimum),
                "maximum": float(maximum),
                "unit": unit.strip(),
            }

        placement_match = placement_pattern.match(line)
        if placement_match:
            name, requester, provider = placement_match.groups()
            placements[name] = {
                "requester_sm": int(requester),
                "provider_sm": int(provider),
            }

    return {
        "device": device,
        "metrics": metrics,
        "placements": placements,
    }


@app.function(
    image=image,
    gpu="H100!",
    cpu=2,
    memory=4096,
    timeout=20 * 60,
)
def profile_h100(run_sanitizers: bool = False) -> dict[str, object]:
    import subprocess

    def run(command: list[str], timeout: int = 10 * 60) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    gpu_query = run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,clocks.max.sm",
            "--format=csv,noheader,nounits",
        ]
    ).stdout.strip()
    nvcc_version = run(["nvcc", "--version"]).stdout.strip()

    compile_result = run(
        [
            "nvcc",
            "-O3",
            "-lineinfo",
            "-std=c++20",
            "-arch=sm_90",
            "-Xptxas=-v",
            REMOTE_SOURCE,
            "-o",
            REMOTE_BINARY,
        ]
    )

    benchmark_result = run([REMOTE_BINARY])
    parsed = parse_benchmark_output(benchmark_result.stdout)

    throughput_symbol = "_Z22dsmem_throughput_benchP12DsmemResultsPi"
    sass_result = run(
        [
            "cuobjdump",
            "--dump-sass",
            "--function",
            throughput_symbol,
            REMOTE_BINARY,
        ]
    )
    sass_counts = {
        "LD.E": len(re.findall(r"\bLD\.E(?:\.|\b)", sass_result.stdout)),
        "ST.E": len(re.findall(r"\bST\.E(?:\.|\b)", sass_result.stdout)),
        "LDS": len(re.findall(r"\bLDS(?:\.|\b)", sass_result.stdout)),
        "STS": len(re.findall(r"\bSTS(?:\.|\b)", sass_result.stdout)),
    }

    sanitizer_results: dict[str, str] = {}
    if run_sanitizers:
        for tool in ("memcheck", "synccheck"):
            result = run(
                [
                    "compute-sanitizer",
                    "--tool",
                    tool,
                    "--kernel-name",
                    "regex=dsmem_",
                    REMOTE_BINARY,
                ],
                timeout=15 * 60,
            )
            sanitizer_results[tool] = result.stdout + result.stderr

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "modal_gpu_request": "H100!",
        "compile_architecture": "sm_90",
        "gpu_query": gpu_query,
        "nvcc_version": nvcc_version,
        "ptxas_output": compile_result.stderr.strip(),
        "benchmark_stdout": benchmark_result.stdout,
        "benchmark_stderr": benchmark_result.stderr,
        "parsed": parsed,
        "throughput_kernel_sass_counts": sass_counts,
        "sanitizers": sanitizer_results,
    }


@app.local_entrypoint()
def main(
    output: str = "h100_dsmem_results.json",
    run_sanitizers: bool = False,
) -> None:
    result = profile_h100.remote(run_sanitizers=run_sanitizers)
    print(result["benchmark_stdout"], end="")

    output_path = Path(output).expanduser().resolve()
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Structured results: {output_path}")
