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
except Exception:
    print("error: torch is required; run through conda env 'cluster'", file=sys.stderr)
    raise


ROOT = Path(__file__).resolve().parent
DEFAULT_N_VALUES = [4096, 8192, 16384, 32768, 65536]
EPS = 1.0e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare row-wise L1 normalization DSMEM kernels with torch.compile."
    )
    parser.add_argument("--rows", "--batch-size", type=int, default=4096)
    parser.add_argument(
        "--n-values", type=str, default=",".join(map(str, DEFAULT_N_VALUES))
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--cluster-size", type=int, default=8)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "results" / "l1norm_torch_compare.csv",
    )
    return parser.parse_args()


def parse_n_values(spec: str) -> list[int]:
    values = [int(part) for part in spec.split(",") if part.strip()]
    if not values:
        raise ValueError("n-values cannot be empty")
    return values


def parse_cuda_results(output: str) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for line in output.splitlines():
        if not line.startswith("RESULT,"):
            continue
        row = next(csv.reader([line]))
        rows.append(
            {
                "cols": int(row[1]),
                "rows": int(row[2]),
                "block_variant": row[3],
                "block_ms": float("nan") if row[4] == "n/a" else float(row[4]),
                "block_gbps": float("nan") if row[5] == "n/a" else float(row[5]),
                "cluster_variant": row[6],
                "cluster_ms": float("nan") if row[7] == "n/a" else float(row[7]),
                "cluster_gbps": float("nan") if row[8] == "n/a" else float(row[8]),
                "cluster_size": int(row[9]),
                "cluster_vs_block": float("nan")
                if row[10] == "n/a"
                else float(row[10]),
            }
        )
    return rows


def run_cuda(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    cmd = [
        "bash",
        str(ROOT / "run_l1norm.sh"),
        "--csv",
        "--rows",
        str(args.rows),
        "--n-values",
        args.n_values,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--cluster-size",
        str(args.cluster_size),
    ]
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return parse_cuda_results(proc.stdout)


def make_input(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    x = torch.empty((rows, cols), device=device, dtype=torch.float32)
    x.uniform_(-2.0, 2.0)
    return x


def modeled_bytes(rows: int, cols: int) -> int:
    return rows * cols * 4 * 3


def bandwidth_gbps(rows: int, cols: int, avg_ms: float) -> float:
    return modeled_bytes(rows, cols) / (avg_ms / 1000.0) / 1.0e9


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


def l1norm(x: torch.Tensor) -> torch.Tensor:
    denom = torch.sum(torch.abs(x), dim=-1, keepdim=True) + EPS
    return x / denom


@torch.no_grad()
def benchmark_torch_compile(
    rows: int, cols: int, warmup: int, iters: int, device: torch.device
) -> dict[str, float | int]:
    x = make_input(rows, cols, device)
    torch._dynamo.reset()
    compiled = torch.compile(l1norm)
    compiled(x)
    torch.cuda.synchronize()
    avg_ms = time_callable(lambda: compiled(x), warmup, iters)
    del x
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    return {
        "cols": cols,
        "torch_ms": avg_ms,
        "torch_gbps": bandwidth_gbps(rows, cols, avg_ms),
    }


def fmt(value: float, precision: int = 2) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.{precision}f}"


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cols",
        "rows",
        "block_variant",
        "block_ms",
        "block_gbps",
        "cluster_variant",
        "cluster_ms",
        "cluster_gbps",
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

    cuda_rows = run_cuda(args)
    cuda_by_cols = {int(row["cols"]): row for row in cuda_rows}
    n_values = parse_n_values(args.n_values)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print(
        f"torch.compile baseline: torch {torch.__version__} on "
        f"{torch.cuda.get_device_name(0)}"
    )
    print(
        f"Row-wise L1 normalization, rows={args.rows}, "
        f"warmup={args.warmup}, iters={args.iters}, "
        f"cluster_size={args.cluster_size}"
    )
    print()
    print(
        f"{'N':<10}{'block':<13}{'block_GB/s':<12}"
        f"{'cluster_GB/s':<14}{'torch_GB/s':<12}"
        f"{'cluster/block':<15}{'cluster/torch':<15}"
    )

    combined: list[dict[str, float | int | str]] = []
    for cols in n_values:
        torch_row = benchmark_torch_compile(
            args.rows, cols, args.warmup, args.iters, device
        )
        cuda_row = cuda_by_cols[cols]
        torch_gbps = float(torch_row["torch_gbps"])
        block_vs_torch = float(cuda_row["block_gbps"]) / torch_gbps
        cluster_vs_torch = float(cuda_row["cluster_gbps"]) / torch_gbps
        combined_row: dict[str, float | int | str] = {
            **cuda_row,
            "torch_ms": torch_row["torch_ms"],
            "torch_gbps": torch_gbps,
            "block_vs_torch": block_vs_torch,
            "cluster_vs_torch": cluster_vs_torch,
        }
        combined.append(combined_row)
        print(
            f"{cols:<10}{str(cuda_row['block_variant']):<13}"
            f"{fmt(float(cuda_row['block_gbps'])):<12}"
            f"{fmt(float(cuda_row['cluster_gbps'])):<14}"
            f"{fmt(torch_gbps):<12}"
            f"{fmt(float(cuda_row['cluster_vs_block']), 3):<15}"
            f"{fmt(cluster_vs_torch, 3):<15}"
        )

    write_csv(args.output_csv, combined)
    best = max(combined, key=lambda row: float(row["cluster_vs_torch"]))
    wins = sum(1 for row in combined if float(row["cluster_vs_torch"]) > 1.0)
    print()
    print(f"Wrote {args.output_csv}")
    print(
        f"Cluster beat torch.compile on {wins}/{len(combined)} shapes. "
        f"Best cluster/torch speedup: {float(best['cluster_vs_torch']):.3f}x "
        f"at N={best['cols']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
