#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Callable

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    print(
        "error: torch is required for this script. "
        "Run it through ./reduction_test/run_compare_torch.sh",
        file=sys.stderr,
    )
    raise

warnings.filterwarnings(
    "ignore",
    message=r"[\s\S]*Online softmax is disabled on the fly[\s\S]*",
    category=UserWarning,
)


COLS_TO_TEST = [
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
]
RMS_EPS = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare custom CUDA workload benchmarks against torch.compile."
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--target-mib", type=int, default=64)
    parser.add_argument("--cluster-size", type=int, default=4)
    parser.add_argument(
        "--workload",
        choices=["all", "rmsnorm", "softmax", "cross_entropy"],
        default="all",
    )
    parser.add_argument("--no-verify", action="store_true")
    return parser.parse_args()


def repo_paths() -> tuple[Path, Path]:
    script_path = Path(__file__).resolve()
    folder = script_path.parent
    return folder, folder / "run_workloads.sh"


def parse_cpp_results(output: str) -> dict[str, list[dict[str, float]]]:
    results: dict[str, list[dict[str, float]]] = {}
    for line in output.splitlines():
        if not line.startswith("RESULT,"):
            continue
        row = next(csv.reader([line]))
        workload = row[1]
        results.setdefault(workload, []).append(
            {
                "cols": int(row[2]),
                "rows": int(row[3]),
                "input_mib": float(row[4]),
                "block_gibs": float("nan") if row[5] == "n/a" else float(row[5]),
                "cluster_gibs": float("nan") if row[6] == "n/a" else float(row[6]),
                "cluster_vs_block": float("nan") if row[7] == "n/a" else float(row[7]),
            }
        )
    return results


def run_cpp_benchmark(args: argparse.Namespace) -> dict[str, list[dict[str, float]]]:
    _, runner = repo_paths()
    cmd = [
        str(runner),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--target-mib",
        str(args.target_mib),
        "--cluster-size",
        str(args.cluster_size),
        "--csv",
        "--workload",
        args.workload,
    ]
    if args.no_verify:
        cmd.append("--no-verify")
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return parse_cpp_results(proc.stdout)


def make_logits(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    total = rows * cols
    values = torch.arange(total, device=device, dtype=torch.int64)
    logits = ((values % 251) - 125).to(torch.float32) * 0.03125
    return logits.view(rows, cols)


def make_weight(cols: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(cols, device=device, dtype=torch.int64)
    return 0.75 + (values % 31).to(torch.float32) * 0.015625


def make_targets(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(rows, device=device, dtype=torch.int64)
    return ((values * 8191 + 17) % cols).to(torch.int64)


def modeled_bytes(workload: str, elements: int, rows: int) -> int:
    tensor_bytes = elements * 4
    if workload == "rmsnorm":
        return tensor_bytes * 4
    if workload == "softmax":
        return tensor_bytes * 4
    if workload == "cross_entropy":
        return tensor_bytes * 2 + rows * (4 + 4)
    raise ValueError(f"unknown workload {workload}")


def throughput_gibs(workload: str, elements: int, rows: int, avg_ms: float) -> float:
    gib = modeled_bytes(workload, elements, rows) / (1024.0 * 1024.0 * 1024.0)
    return gib / (avg_ms / 1000.0)


def time_callable(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


@torch.no_grad()
def benchmark_torch_compile(
    workload: str, rows: int, cols: int, warmup: int, iters: int, device: torch.device
) -> float:
    x = make_logits(rows, cols, device)

    if workload == "rmsnorm":
        w = make_weight(cols, device)

        def eager_fn(inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return inp * torch.rsqrt(inp.square().mean(dim=-1, keepdim=True) + RMS_EPS) * weight

        compiled = torch.compile(eager_fn)
        compiled(x, w)
        torch.cuda.synchronize()
        avg_ms = time_callable(lambda: compiled(x, w), warmup, iters)
    elif workload == "softmax":

        def eager_fn(inp: torch.Tensor) -> torch.Tensor:
            return torch.softmax(inp, dim=-1)

        compiled = torch.compile(eager_fn)
        compiled(x)
        torch.cuda.synchronize()
        avg_ms = time_callable(lambda: compiled(x), warmup, iters)
    elif workload == "cross_entropy":
        targets = make_targets(rows, cols, device)

        def eager_fn(inp: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
            return F.cross_entropy(inp, tgt, reduction="none")

        compiled = torch.compile(eager_fn)
        compiled(x, targets)
        torch.cuda.synchronize()
        avg_ms = time_callable(lambda: compiled(x, targets), warmup, iters)
    else:
        raise ValueError(f"unknown workload {workload}")

    return throughput_gibs(workload, rows * cols, rows, avg_ms)


def fmt(value: float, precision: int = 2) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.{precision}f}"


def print_workload_table(workload: str, custom_rows: list[dict[str, float]], torch_rows: list[dict[str, float]]) -> None:
    torch_by_cols = {row["cols"]: row for row in torch_rows}
    print(f"Workload: {workload}")
    print(
        f"{'cols':<9}{'rows':<9}{'input_MiB':<12}"
        f"{'block_GiB/s':<13}{'cluster_GiB/s':<15}{'torch_GiB/s':<13}"
        f"{'block/torch':<13}{'cluster/torch':<14}"
    )
    for row in custom_rows:
        torch_row = torch_by_cols[row["cols"]]
        torch_gibs = torch_row["torch_gibs"]
        block_vs_torch = row["block_gibs"] / torch_gibs
        cluster_vs_torch = row["cluster_gibs"] / torch_gibs
        print(
            f"{int(row['cols']):<9}{int(row['rows']):<9}{row['input_mib']:<12.1f}"
            f"{fmt(row['block_gibs']):<13}{fmt(row['cluster_gibs']):<15}"
            f"{fmt(torch_gibs):<13}{fmt(block_vs_torch):<13}{fmt(cluster_vs_torch):<14}"
        )

    block_wins = 0
    cluster_wins = 0
    large_block = []
    large_cluster = []
    best_cluster_vs_torch = 0.0
    best_cols = 0
    for row in custom_rows:
        torch_gibs = torch_by_cols[row["cols"]]["torch_gibs"]
        block_vs_torch = row["block_gibs"] / torch_gibs
        cluster_vs_torch = row["cluster_gibs"] / torch_gibs
        if block_vs_torch > 1.0:
            block_wins += 1
        if cluster_vs_torch > 1.0:
            cluster_wins += 1
        if row["cols"] >= 65536:
            large_block.append(block_vs_torch)
            large_cluster.append(cluster_vs_torch)
        if cluster_vs_torch > best_cluster_vs_torch:
            best_cluster_vs_torch = cluster_vs_torch
            best_cols = int(row["cols"])

    avg_large_block = sum(large_block) / len(large_block)
    avg_large_cluster = sum(large_cluster) / len(large_cluster)
    print(
        f"Summary: block beat torch.compile on {block_wins}/{len(custom_rows)} sizes, "
        f"cluster beat torch.compile on {cluster_wins}/{len(custom_rows)} sizes. "
        f"Best cluster/torch speedup was {best_cluster_vs_torch:.2f}x at cols={best_cols}. "
        f"Average block/torch for cols >= 65536 was {avg_large_block:.2f}x; "
        f"average cluster/torch was {avg_large_cluster:.2f}x."
    )
    print()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("error: CUDA is not available in this torch environment", file=sys.stderr)
        return 1

    torch.set_grad_enabled(False)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    custom_results = run_cpp_benchmark(args)
    workloads = custom_results.keys() if args.workload == "all" else [args.workload]

    print(
        f"torch.compile baseline: torch {torch.__version__} on "
        f"{torch.cuda.get_device_name(0)}"
    )
    print(
        f"Using default torch.compile mode, warmup={args.warmup}, "
        f"iters={args.iters}, target_mib={args.target_mib}, cluster_size={args.cluster_size}"
    )
    print()

    for workload in workloads:
        torch_results = []
        for cols in COLS_TO_TEST:
            matching = next(row for row in custom_results[workload] if int(row["cols"]) == cols)
            rows = int(matching["rows"])
            torch._dynamo.reset()
            torch_gibs = benchmark_torch_compile(
                workload, rows, cols, args.warmup, args.iters, device
            )
            torch_results.append({"cols": cols, "torch_gibs": torch_gibs})
        print_workload_table(workload, custom_results[workload], torch_results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
