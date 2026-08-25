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
except Exception:
    print("error: torch is required; run through conda env 'cluster'", file=sys.stderr)
    raise


ROOT = Path(__file__).resolve().parent
DEFAULT_N_VALUES = [4096, 8192, 16384, 32768, 65536]
EPS = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare LayerNorm CUDA DSMEM kernels against torch.compile."
    )
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--n-values", type=str, default=",".join(map(str, DEFAULT_N_VALUES))
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--cluster-size", type=int, default=8)
    parser.add_argument("--thread-per-row-values", type=str, default="16,32,256")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "results" / "layernorm_torch_compare.csv",
    )
    return parser.parse_args()


def parse_n_values(spec: str) -> list[int]:
    values = [int(part) for part in spec.split(",") if part.strip()]
    if not values:
        raise ValueError("n-values cannot be empty")
    return values


def parse_cuda_results(output: str) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for line in output.splitlines():
        if not line.startswith("RESULT,"):
            continue
        row = next(csv.reader([line]))
        rows.append(
            {
                "cols": int(row[1]),
                "batch_size": int(row[2]),
                "block_ms": float("nan") if row[3] == "n/a" else float(row[3]),
                "block_gbps": float("nan") if row[4] == "n/a" else float(row[4]),
                "block_tpr": int(row[5]),
                "cluster_ms": float("nan") if row[6] == "n/a" else float(row[6]),
                "cluster_gbps": float("nan") if row[7] == "n/a" else float(row[7]),
                "cluster_tpr": int(row[8]),
                "cluster_size": int(row[9]),
                "cluster_vs_block": float("nan")
                if row[10] == "n/a"
                else float(row[10]),
            }
        )
    return rows


def run_cuda(args: argparse.Namespace) -> list[dict[str, float | int]]:
    cmd = [
        "bash",
        str(ROOT / "run_layernorm.sh"),
        "--csv",
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
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return parse_cuda_results(proc.stdout)


def make_input(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    x = torch.empty((rows, cols), device=device, dtype=torch.float32)
    x.uniform_(-1.0, 1.0)
    return x


def make_affine(cols: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.arange(cols, device=device, dtype=torch.float32)
    gamma = 0.75 + torch.remainder(values, 31.0) * 0.015625
    beta = (torch.remainder(values, 17.0) - 8.0) * 0.00390625
    return gamma, beta


def modeled_bytes(rows: int, cols: int) -> int:
    return rows * cols * 4 * 4


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


@torch.no_grad()
def benchmark_torch_compile(
    rows: int, cols: int, warmup: int, iters: int, device: torch.device
) -> dict[str, float | int]:
    x = make_input(rows, cols, device)
    gamma, beta = make_affine(cols, device)
    torch._dynamo.reset()
    compiled = torch.compile(
        lambda inp, g, b: F.layer_norm(inp, (cols,), g, b, EPS)
    )
    compiled(x, gamma, beta)
    torch.cuda.synchronize()
    avg_ms = time_callable(lambda: compiled(x, gamma, beta), warmup, iters)
    del x, gamma, beta
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


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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

    cuda_rows = run_cuda(args)
    cuda_by_cols = {int(row["cols"]): row for row in cuda_rows}
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    n_values = parse_n_values(args.n_values)

    print(
        f"torch.compile baseline: torch {torch.__version__} on "
        f"{torch.cuda.get_device_name(0)}"
    )
    print(
        f"LayerNorm comparison, warmup={args.warmup}, iters={args.iters}, "
        f"batch_size={args.batch_size}, cluster_size={args.cluster_size}, "
        f"thread_per_row_values={args.thread_per_row_values}"
    )
    print()
    print(
        f"{'N':<10}{'block_GB/s':<12}{'block_tpr':<11}"
        f"{'cluster_GB/s':<14}{'torch_GB/s':<12}"
        f"{'block/torch':<13}{'cluster/torch':<14}{'cluster/block':<14}"
    )

    combined: list[dict[str, float | int]] = []
    for cols in n_values:
        torch_row = benchmark_torch_compile(
            args.batch_size, cols, args.warmup, args.iters, device
        )
        cuda_row = cuda_by_cols[cols]
        torch_gbps = float(torch_row["torch_gbps"])
        block_vs_torch = float(cuda_row["block_gbps"]) / torch_gbps
        cluster_vs_torch = float(cuda_row["cluster_gbps"]) / torch_gbps
        combined_row = {
            **cuda_row,
            "torch_ms": torch_row["torch_ms"],
            "torch_gbps": torch_row["torch_gbps"],
            "block_vs_torch": block_vs_torch,
            "cluster_vs_torch": cluster_vs_torch,
        }
        combined.append(combined_row)
        print(
            f"{cols:<10}{fmt(float(cuda_row['block_gbps'])):<12}"
            f"{int(cuda_row['block_tpr']):<11}"
            f"{fmt(float(cuda_row['cluster_gbps'])):<14}"
            f"{fmt(torch_gbps):<12}"
            f"{fmt(block_vs_torch):<13}{fmt(cluster_vs_torch):<14}"
            f"{fmt(float(cuda_row['cluster_vs_block'])):<14}"
        )

    wins = sum(float(row["cluster_vs_torch"]) > 1.0 for row in combined)
    best = max(combined, key=lambda row: float(row["cluster_vs_torch"]))
    print()
    print(
        f"Summary: cluster beat torch.compile on {wins}/{len(combined)} sizes. "
        f"Best cluster/torch speedup was {float(best['cluster_vs_torch']):.2f}x "
        f"at N={int(best['cols'])}."
    )
    write_csv(args.output_csv, combined)
    print(f"Wrote combined CSV: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
