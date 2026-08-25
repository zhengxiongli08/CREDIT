#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot row-wise weighted variance normalization DSMEM results.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=ROOT / "results" / "weighted_var_norm_torch_compare.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "plots")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda row: int(row["cols"]))
    return rows


def save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=180)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def plot_speedup(rows: list[dict[str, str]], out_dir: Path) -> None:
    xs = [str(int(row["cols"])) for row in rows]
    cluster_vs_torch = [float(row["cluster_vs_torch"]) for row in rows]
    cluster_vs_block = [float(row["cluster_vs_block"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(xs, cluster_vs_torch, marker="o", linewidth=2,
            label="DSMEM / torch.compile")
    ax.plot(xs, cluster_vs_block, marker="o", linewidth=2,
            label="DSMEM / best noDSMEM")
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    for x, y in zip(xs, cluster_vs_torch):
        ax.annotate(f"{y:.2f}x", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_title("Row-wise Weighted Variance Normalization DSMEM Speedup")
    ax.set_xlabel("N")
    ax.set_ylabel("throughput ratio")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save(fig, out_dir, "weighted_var_norm_speedup")


def plot_runtime(rows: list[dict[str, str]], out_dir: Path) -> None:
    xs = [str(int(row["cols"])) for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(xs, [float(row["block_ms"]) for row in rows],
            marker="o", label="best noDSMEM")
    ax.plot(xs, [float(row["cluster_ms"]) for row in rows],
            marker="o", label="DSMEM cluster")
    ax.plot(xs, [float(row["torch_ms"]) for row in rows],
            marker="o", label="torch.compile")
    ax.set_title("Row-wise Weighted Variance Normalization Runtime")
    ax.set_xlabel("N")
    ax.set_ylabel("runtime (ms)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save(fig, out_dir, "weighted_var_norm_runtime_ms")


def plot_gbps(rows: list[dict[str, str]], out_dir: Path) -> None:
    xs = [str(int(row["cols"])) for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(xs, [float(row["block_gbps"]) for row in rows],
            marker="o", label="best noDSMEM")
    ax.plot(xs, [float(row["cluster_gbps"]) for row in rows],
            marker="o", label="DSMEM cluster")
    ax.plot(xs, [float(row["torch_gbps"]) for row in rows],
            marker="o", label="torch.compile")
    ax.set_title("Row-wise Weighted Variance Normalization Modeled Throughput")
    ax.set_xlabel("N")
    ax.set_ylabel("modeled GB/s")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save(fig, out_dir, "weighted_var_norm_gbps")


def plot_ncu(out_dir: Path) -> None:
    path = ROOT / "results" / "weighted_var_norm_ncu_summary.csv"
    if not path.exists():
        return
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    labels = [row["variant"] for row in rows]
    colors = ["#4c78a8" if row["variant"] == "block" else "#f58518"
              for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.0))
    axes[0].bar(labels, [float(row["time_us"]) for row in rows], color=colors)
    axes[0].set_title("NCU Time")
    axes[0].set_ylabel("us")
    axes[1].bar(labels, [float(row["l2_bytes_per_element"]) for row in rows],
                color=colors)
    axes[1].set_title("L2 Traffic")
    axes[1].set_ylabel("B / element")
    axes[2].bar(labels, [float(row["dram_bytes_per_element"]) for row in rows],
                color=colors)
    axes[2].set_title("DRAM Traffic")
    axes[2].set_ylabel("B / element")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
    save(fig, out_dir, "weighted_var_norm_ncu_metrics")


def main() -> int:
    args = parse_args()
    rows = read_rows(args.input_csv)
    plot_speedup(rows, args.output_dir)
    plot_runtime(rows, args.output_dir)
    plot_gbps(rows, args.output_dir)
    plot_ncu(args.output_dir)
    print(f"Wrote plots to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
