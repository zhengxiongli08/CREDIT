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
EPS = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare BatchNorm backward dx DSMEM kernels with torch.compile."
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument(
        "--n-values", "--elems-values", type=str, default=",".join(map(str, DEFAULT_N_VALUES))
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--cluster-size", type=int, default=8)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "results" / "batchnorm_backward_torch_compare.csv",
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
                "elems_per_channel": int(row[1]),
                "batch_size": int(row[2]),
                "channels": int(row[3]),
                "spatial": int(row[4]),
                "block_variant": row[5],
                "block_ms": float("nan") if row[6] == "n/a" else float(row[6]),
                "block_gbps": float("nan") if row[7] == "n/a" else float(row[7]),
                "cluster_variant": row[8],
                "cluster_ms": float("nan") if row[9] == "n/a" else float(row[9]),
                "cluster_gbps": float("nan") if row[10] == "n/a" else float(row[10]),
                "cluster_size": int(row[11]),
                "cluster_vs_block": float("nan")
                if row[12] == "n/a"
                else float(row[12]),
            }
        )
    return rows


def run_cuda(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    cmd = [
        "bash",
        str(ROOT / "run_batchnorm_backward.sh"),
        "--csv",
        "--batch-size",
        str(args.batch_size),
        "--channels",
        str(args.channels),
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


def make_inputs(
    batch_size: int, channels: int, spatial: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.empty((batch_size, channels, spatial), device=device, dtype=torch.float32)
    dy = torch.empty((batch_size, channels, spatial), device=device, dtype=torch.float32)
    gamma = torch.empty((channels,), device=device, dtype=torch.float32)
    x.uniform_(-1.0, 1.0)
    dy.uniform_(-1.0, 1.0)
    values = torch.arange(channels, device=device, dtype=torch.float32)
    gamma.copy_(0.75 + torch.remainder(values, 31.0) * 0.015625)
    return x, dy, gamma


def modeled_bytes(channels: int, elems_per_channel: int) -> int:
    return channels * elems_per_channel * 4 * 6


def bandwidth_gbps(channels: int, elems_per_channel: int, avg_ms: float) -> float:
    return modeled_bytes(channels, elems_per_channel) / (avg_ms / 1000.0) / 1.0e9


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


def batchnorm_bwd_dx(
    x: torch.Tensor, dy: torch.Tensor, gamma: torch.Tensor
) -> torch.Tensor:
    mean = x.mean(dim=(0, 2), keepdim=True)
    xmu = x - mean
    inv_std = torch.rsqrt((xmu * xmu).mean(dim=(0, 2), keepdim=True) + EPS)
    xhat = xmu * inv_std
    dyg = dy * gamma.view(1, -1, 1)
    mean_dyg = dyg.mean(dim=(0, 2), keepdim=True)
    mean_dyg_xhat = (dyg * xhat).mean(dim=(0, 2), keepdim=True)
    return (dyg - mean_dyg - xhat * mean_dyg_xhat) * inv_std


@torch.no_grad()
def benchmark_torch_compile(
    batch_size: int,
    channels: int,
    elems_per_channel: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, float | int]:
    if elems_per_channel % batch_size != 0:
        raise ValueError("elems_per_channel must be divisible by batch_size")
    spatial = elems_per_channel // batch_size
    x, dy, gamma = make_inputs(batch_size, channels, spatial, device)
    torch._dynamo.reset()
    compiled = torch.compile(batchnorm_bwd_dx)
    compiled(x, dy, gamma)
    torch.cuda.synchronize()
    avg_ms = time_callable(lambda: compiled(x, dy, gamma), warmup, iters)
    del x, dy, gamma
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    return {
        "elems_per_channel": elems_per_channel,
        "torch_ms": avg_ms,
        "torch_gbps": bandwidth_gbps(channels, elems_per_channel, avg_ms),
    }


def fmt(value: float, precision: int = 2) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.{precision}f}"


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "elems_per_channel",
        "batch_size",
        "channels",
        "spatial",
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

    n_values = parse_n_values(args.n_values)
    for value in n_values:
        if value % args.batch_size != 0:
            print(
                f"error: n-value {value} is not divisible by batch_size {args.batch_size}",
                file=sys.stderr,
            )
            return 1

    cuda_rows = run_cuda(args)
    cuda_by_elems = {int(row["elems_per_channel"]): row for row in cuda_rows}
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print(
        f"torch.compile baseline: torch {torch.__version__} on "
        f"{torch.cuda.get_device_name(0)}"
    )
    print(
        f"BatchNorm backward dx, batch={args.batch_size}, "
        f"channels={args.channels}, warmup={args.warmup}, iters={args.iters}, "
        f"cluster_size={args.cluster_size}"
    )
    print()
    print(
        f"{'elems/C':<10}{'block':<13}{'block_GB/s':<12}"
        f"{'cluster_GB/s':<14}{'torch_GB/s':<12}"
        f"{'cluster/block':<15}{'cluster/torch':<15}"
    )

    combined: list[dict[str, float | int | str]] = []
    for elems_per_channel in n_values:
        torch_row = benchmark_torch_compile(
            args.batch_size,
            args.channels,
            elems_per_channel,
            args.warmup,
            args.iters,
            device,
        )
        cuda_row = cuda_by_elems[elems_per_channel]
        torch_gbps = float(torch_row["torch_gbps"])
        block_gbps = float(cuda_row["block_gbps"])
        cluster_gbps = float(cuda_row["cluster_gbps"])
        block_vs_torch = block_gbps / torch_gbps
        cluster_vs_torch = cluster_gbps / torch_gbps
        combined_row = {
            **cuda_row,
            "torch_ms": torch_row["torch_ms"],
            "torch_gbps": torch_row["torch_gbps"],
            "block_vs_torch": block_vs_torch,
            "cluster_vs_torch": cluster_vs_torch,
        }
        combined.append(combined_row)
        print(
            f"{elems_per_channel:<10}{str(cuda_row['block_variant']):<13}"
            f"{fmt(block_gbps):<12}{fmt(cluster_gbps):<14}"
            f"{fmt(torch_gbps):<12}"
            f"{fmt(float(cuda_row['cluster_vs_block'])):<15}"
            f"{fmt(cluster_vs_torch):<15}"
        )

    write_csv(args.output_csv, combined)
    print()
    print(f"wrote {args.output_csv}")
    wins = [row for row in combined if float(row["cluster_vs_torch"]) > 1.0]
    if wins:
        best = max(wins, key=lambda row: float(row["cluster_vs_torch"]))
        print(
            f"Cluster beat torch.compile on {len(wins)}/{len(combined)} shapes. "
            f"Best cluster/torch speedup: {float(best['cluster_vs_torch']):.3f}x "
            f"at elems/channel={best['elems_per_channel']}."
        )
    else:
        print("Cluster did not beat torch.compile on these shapes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
