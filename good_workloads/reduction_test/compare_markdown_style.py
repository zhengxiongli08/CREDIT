#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path
from typing import Callable

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    print(
        "error: torch is required for this script. "
        "Run it through ./reduction_test/run_markdown_compare.sh",
        file=sys.stderr,
    )
    raise


RMS_EPS = 1e-6
DEFAULT_N_VALUES = [4096, 8192, 16384, 32768, 65536, 131072]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Markdown-style comparison between our CUDA binary and torch.compile."
        )
    )
    parser.add_argument(
        "--workload",
        choices=["softmax", "rmsnorm", "cross_entropy", "all"],
        default="all",
    )
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument(
        "--n-values",
        type=str,
        default=",".join(map(str, DEFAULT_N_VALUES)),
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--cluster-size", type=int, default=8)
    parser.add_argument("--thread-per-row-values", type=str, default="16,32,256")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path for a combined CUDA-vs-torch CSV summary.",
    )
    return parser.parse_args()


def parse_n_values(spec: str) -> list[int]:
    values = [int(part) for part in spec.split(",") if part.strip()]
    if not values:
        raise ValueError("n-values cannot be empty")
    return values


def repo_paths() -> tuple[Path, Path]:
    script_path = Path(__file__).resolve()
    folder = script_path.parent
    return folder, folder / "run_markdown_workloads.sh"


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
                "batch_size": int(row[3]),
                "block_ms": float("nan") if row[4] == "n/a" else float(row[4]),
                "block_gbps": float("nan") if row[5] == "n/a" else float(row[5]),
                "block_tpr": int(row[6]),
                "cluster_ms": float("nan") if row[7] == "n/a" else float(row[7]),
                "cluster_gbps": float("nan") if row[8] == "n/a" else float(row[8]),
                "cluster_tpr": int(row[9]),
                "cluster_size": int(row[10]),
                "cluster_vs_block": float("nan") if row[11] == "n/a" else float(row[11]),
            }
        )
    return results


def run_cpp_benchmark(args: argparse.Namespace) -> dict[str, list[dict[str, float]]]:
    _, runner = repo_paths()
    cmd = [
        str(runner),
        "--csv",
        "--workload",
        args.workload,
        "--batch-size",
        str(args.batch_size),
        "--n-values",
        args.n_values,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--cluster-size",
        str(args.cluster_size),
        "--thread-per-row-values",
        args.thread_per_row_values,
    ]
    if args.no_verify:
        cmd.append("--no-verify")
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return parse_cpp_results(proc.stdout)


def make_logits(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    logits = torch.empty((rows, cols), device=device, dtype=torch.float32)
    logits.uniform_(-4.0, 4.0)
    return logits


def make_weight(cols: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(cols, device=device, dtype=torch.float32)
    return 0.75 + torch.remainder(values, 31.0) * 0.015625


def make_targets(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(rows, device=device, dtype=torch.int64)
    return ((values * 8191 + 17) % cols).to(torch.int64)


def modeled_bytes(workload: str, rows: int, cols: int) -> int:
    elements = rows * cols
    if workload == "softmax":
        return elements * 4 * 4
    if workload == "rmsnorm":
        return elements * 4 * 4
    if workload == "cross_entropy":
        return elements * 4 * 2 + rows * 4
    raise ValueError(f"unknown workload {workload}")


def bandwidth_gbps(workload: str, rows: int, cols: int, avg_ms: float) -> float:
    return modeled_bytes(workload, rows, cols) / (avg_ms / 1000.0) / 1.0e9


def time_callable(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        out = fn()
        del out
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
        del out
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


@torch.no_grad()
def benchmark_torch_compile(
    workload: str, rows: int, cols: int, warmup: int, iters: int, device: torch.device
) -> dict[str, float]:
    x = make_logits(rows, cols, device)
    torch._dynamo.reset()
    if workload == "softmax":
        compiled = torch.compile(lambda inp: F.softmax(inp, dim=-1))
        compiled(x)
        torch.cuda.synchronize()
        avg_ms = time_callable(lambda: compiled(x), warmup, iters)
    elif workload == "rmsnorm":
        w = make_weight(cols, device)
        compiled = torch.compile(
            lambda inp, weight: inp
            * torch.rsqrt(inp.float().pow(2).mean(-1, keepdim=True) + RMS_EPS)
            * weight
        )
        compiled(x, w)
        torch.cuda.synchronize()
        avg_ms = time_callable(lambda: compiled(x, w), warmup, iters)
        del w
    elif workload == "cross_entropy":
        targets = make_targets(rows, cols, device)
        compiled = torch.compile(
            lambda inp, tgt: F.cross_entropy(inp, tgt, reduction="none")
        )
        compiled(x, targets)
        torch.cuda.synchronize()
        avg_ms = time_callable(lambda: compiled(x, targets), warmup, iters)
        del targets
    else:
        raise ValueError(f"unknown workload {workload}")
    del x
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    return {
        "cols": cols,
        "torch_ms": avg_ms,
        "torch_gbps": bandwidth_gbps(workload, rows, cols, avg_ms),
    }


def fmt(value: float, precision: int = 2) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.{precision}f}"


def print_workload_table(
    workload: str,
    custom_rows: list[dict[str, float]],
    torch_rows: list[dict[str, float]],
) -> None:
    torch_by_cols = {row["cols"]: row for row in torch_rows}
    batch_size = int(custom_rows[0]["batch_size"]) if custom_rows else 0
    print(f"Workload: {workload}")
    print(f"Batch size: {batch_size}")
    print(
        f"{'N':<10}{'block_GB/s':<12}{'block_tpr':<11}"
        f"{'cluster_GB/s':<14}{'cluster_tpr':<13}{'torch_GB/s':<12}"
        f"{'block/torch':<13}{'cluster/torch':<14}"
    )
    for row in custom_rows:
        torch_gbps = torch_by_cols[row["cols"]]["torch_gbps"]
        block_vs_torch = row["block_gbps"] / torch_gbps
        cluster_vs_torch = row["cluster_gbps"] / torch_gbps
        print(
            f"{int(row['cols']):<10}{fmt(row['block_gbps']):<12}{int(row['block_tpr']):<11}"
            f"{fmt(row['cluster_gbps']):<14}{int(row['cluster_tpr']):<13}"
            f"{fmt(torch_gbps):<12}{fmt(block_vs_torch):<13}{fmt(cluster_vs_torch):<14}"
        )

    block_wins = 0
    cluster_wins = 0
    best_cluster = 0.0
    best_cols = 0
    large_cluster = []
    for row in custom_rows:
        torch_gbps = torch_by_cols[row["cols"]]["torch_gbps"]
        block_vs_torch = row["block_gbps"] / torch_gbps
        cluster_vs_torch = row["cluster_gbps"] / torch_gbps
        if block_vs_torch > 1.0:
            block_wins += 1
        if cluster_vs_torch > 1.0:
            cluster_wins += 1
        if row["cols"] >= 65536:
            large_cluster.append(cluster_vs_torch)
        if cluster_vs_torch > best_cluster:
            best_cluster = cluster_vs_torch
            best_cols = int(row["cols"])

    avg_large_cluster = sum(large_cluster) / len(large_cluster) if large_cluster else float("nan")
    print(
        f"Summary: block beat torch.compile on {block_wins}/{len(custom_rows)} sizes, "
        f"cluster beat torch.compile on {cluster_wins}/{len(custom_rows)} sizes. "
        f"Best cluster/torch speedup was {best_cluster:.2f}x at N={best_cols}. "
        f"Average cluster/torch for N >= 65536 was {fmt(avg_large_cluster)}x."
    )
    print()


def combined_rows(
    workload: str,
    custom_rows: list[dict[str, float]],
    torch_rows: list[dict[str, float]],
) -> list[dict[str, float | int | str]]:
    torch_by_cols = {row["cols"]: row for row in torch_rows}
    rows: list[dict[str, float | int | str]] = []
    for row in custom_rows:
        torch_row = torch_by_cols[row["cols"]]
        torch_gbps = torch_row["torch_gbps"]
        rows.append(
            {
                "workload": workload,
                "cols": int(row["cols"]),
                "batch_size": int(row["batch_size"]),
                "block_ms": row["block_ms"],
                "block_gbps": row["block_gbps"],
                "block_tpr": int(row["block_tpr"]),
                "cluster_ms": row["cluster_ms"],
                "cluster_gbps": row["cluster_gbps"],
                "cluster_tpr": int(row["cluster_tpr"]),
                "cluster_size": int(row["cluster_size"]),
                "torch_ms": torch_row["torch_ms"],
                "torch_gbps": torch_gbps,
                "block_vs_torch": row["block_gbps"] / torch_gbps,
                "cluster_vs_torch": row["cluster_gbps"] / torch_gbps,
                "cluster_vs_block": row["cluster_vs_block"],
            }
        )
    return rows


def write_combined_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "workload",
        "cols",
        "batch_size",
        "block_ms",
        "block_gbps",
        "block_tpr",
        "cluster_ms",
        "cluster_gbps",
        "cluster_tpr",
        "cluster_size",
        "torch_ms",
        "torch_gbps",
        "block_vs_torch",
        "cluster_vs_torch",
        "cluster_vs_block",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("error: CUDA is not available in this torch environment", file=sys.stderr)
        return 1

    custom_results = run_cpp_benchmark(args)
    workloads = (
        ["softmax", "rmsnorm", "cross_entropy"]
        if args.workload == "all"
        else [args.workload]
    )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    n_values = parse_n_values(args.n_values)

    print(
        f"torch.compile baseline: torch {torch.__version__} on "
        f"{torch.cuda.get_device_name(0)}"
    )
    print(
        f"Markdown-style comparison using our CUDA binary, "
        f"warmup={args.warmup}, iters={args.iters}, batch_size={args.batch_size}, "
        f"cluster_size={args.cluster_size}, thread_per_row_values={args.thread_per_row_values}"
    )
    print()

    all_rows: list[dict[str, float | int | str]] = []
    for workload in workloads:
        torch_rows = []
        for cols in n_values:
            torch_row = benchmark_torch_compile(
                workload, args.batch_size, cols, args.warmup, args.iters, device
            )
            torch_rows.append(torch_row)
        print_workload_table(workload, custom_results[workload], torch_rows)
        all_rows.extend(combined_rows(workload, custom_results[workload], torch_rows))

    if args.output_csv is not None:
        write_combined_csv(args.output_csv, all_rows)
        print(f"Wrote combined CSV: {args.output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
