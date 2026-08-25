#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "plots" / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "plots" / ".cache"))

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CSV = ROOT / "results" / "logsoftmax_torch_compare.csv"
DEFAULT_NCU_CSV = ROOT / "results" / "logsoftmax_ncu_summary.csv"
DEFAULT_OUT = ROOT / "plots"

COLORS = {
    "block": "#4c78a8",
    "cluster": "#e45756",
    "torch.compile": "#54a24b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot log_softmax DSMEM results.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--ncu-csv", type=Path, default=DEFAULT_NCU_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def label_points(ax, xs, ys, labels, dy=4):
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(0, dy),
            ha="center",
            fontsize=8,
        )


def plot_speedup(rows: list[dict[str, str]], out_dir: Path) -> None:
    rows = sorted(rows, key=lambda row: int(row["cols"]))
    labels = [row["cols"] for row in rows]
    x = np.arange(len(rows))
    width = 0.34
    block = [float(row["block_vs_torch"]) for row in rows]
    cluster = [float(row["cluster_vs_torch"]) for row in rows]

    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    block_bars = ax.bar(
        x - width / 2,
        block,
        width,
        label="block / torch.compile",
        color=COLORS["block"],
        edgecolor="black",
        linewidth=0.4,
    )
    cluster_bars = ax.bar(
        x + width / 2,
        cluster,
        width,
        label="cluster DSMEM / torch.compile",
        color=COLORS["cluster"],
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar_label(block_bars, labels=[f"{v:.2f}x" for v in block], fontsize=8)
    ax.bar_label(cluster_bars, labels=[f"{v:.2f}x" for v in cluster], fontsize=8)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="break-even")
    ax.set_title("log_softmax: speedup relative to torch.compile")
    ax.set_ylabel("speedup")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("N")
    ax.set_ylim(0, max(max(block), max(cluster), 1.0) * 1.25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=3, fontsize=9)
    fig.savefig(out_dir / "logsoftmax_speedup_vs_torch.png", dpi=220)
    fig.savefig(out_dir / "logsoftmax_speedup_vs_torch.pdf")
    plt.close(fig)


def plot_lines(rows: list[dict[str, str]], out_dir: Path, metric: str, ylabel: str, filename: str) -> None:
    rows = sorted(rows, key=lambda row: int(row["cols"]))
    labels = [row["cols"] for row in rows]
    x = np.arange(len(rows))
    series = {
        "block": [float(row[f"block_{metric}"]) for row in rows],
        "cluster": [float(row[f"cluster_{metric}"]) for row in rows],
        "torch.compile": [float(row[f"torch_{metric}"]) for row in rows],
    }

    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    for name, values in series.items():
        ax.plot(x, values, marker="o", linewidth=2.0, color=COLORS[name], label=name)
        fmt = "{:.0f}" if metric == "gbps" else "{:.3f}"
        label_points(ax, x, values, [fmt.format(v) for v in values])
    ax.set_title(f"log_softmax: {ylabel}")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("N")
    ax.grid(alpha=0.25)
    ax.legend(ncols=3, fontsize=9)
    fig.savefig(out_dir / filename, dpi=220)
    fig.savefig(out_dir / filename.replace(".png", ".pdf"))
    plt.close(fig)


def plot_ncu(path: Path, out_dir: Path) -> None:
    if not path.exists():
        return
    rows = read_csv(path)
    if not rows:
        return
    by_variant = {row["variant"]: row for row in rows}
    variants = ["block", "cluster"]
    metrics = [
        ("time_us", "NCU kernel time (us)"),
        ("l2_bytes_per_element", "L2 bytes / element"),
        ("dram_bytes_per_element", "DRAM bytes / element"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4.0), constrained_layout=True)
    for ax, (metric, title) in zip(axes, metrics):
        values = [float(by_variant[variant][metric]) for variant in variants]
        bars = ax.bar(
            variants,
            values,
            color=[COLORS["block"], COLORS["cluster"]],
            edgecolor="black",
            linewidth=0.4,
        )
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_dir / "logsoftmax_ncu_metrics.png", dpi=220)
    fig.savefig(out_dir / "logsoftmax_ncu_metrics.pdf")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.csv)
    plot_speedup(rows, args.out_dir)
    plot_lines(rows, args.out_dir, "gbps", "GB/s, higher is better", "logsoftmax_gbps.png")
    plot_lines(rows, args.out_dir, "ms", "ms, lower is better", "logsoftmax_runtime_ms.png")
    plot_ncu(args.ncu_csv, args.out_dir)
    print(f"Wrote plots to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
