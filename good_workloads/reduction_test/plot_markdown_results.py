#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "plots" / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "plots" / ".cache"))

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CSV = ROOT / "results" / "markdown_torch_compare.csv"
DEFAULT_NCU_CSV = ROOT / "results" / "markdown_ncu_summary.csv"
DEFAULT_OUT = ROOT / "plots"

COLORS = {
    "block": "#4c78a8",
    "cluster": "#e45756",
    "torch.compile": "#54a24b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot markdown-style reduction results.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--ncu-csv", type=Path, default=DEFAULT_NCU_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def group_by_workload(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["workload"]].append(row)
    for workload in grouped:
        grouped[workload].sort(key=lambda row: int(row["cols"]))
    return grouped


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
    grouped = group_by_workload(rows)
    fig, axes = plt.subplots(
        len(grouped), 1, figsize=(12, 4.2 * len(grouped)), constrained_layout=True
    )
    if len(grouped) == 1:
        axes = [axes]

    for ax, (workload, items) in zip(axes, grouped.items()):
        labels = [row["cols"] for row in items]
        x = np.arange(len(labels))
        width = 0.34
        block = [float(row["block_vs_torch"]) for row in items]
        cluster = [float(row["cluster_vs_torch"]) for row in items]

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
        ax.set_title(f"{workload}: speedup relative to torch.compile")
        ax.set_ylabel("speedup")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("N")
        ax.set_ylim(0, max(max(block), max(cluster), 1.0) * 1.25)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncols=3, fontsize=9)

    fig.savefig(out_dir / "markdown_speedup_vs_torch.png", dpi=220)
    fig.savefig(out_dir / "markdown_speedup_vs_torch.pdf")
    plt.close(fig)


def plot_gbps(rows: list[dict[str, str]], out_dir: Path) -> None:
    grouped = group_by_workload(rows)
    fig, axes = plt.subplots(
        len(grouped), 1, figsize=(12, 4.2 * len(grouped)), constrained_layout=True
    )
    if len(grouped) == 1:
        axes = [axes]

    for ax, (workload, items) in zip(axes, grouped.items()):
        x = np.arange(len(items))
        labels = [row["cols"] for row in items]
        series = {
            "block": [float(row["block_gbps"]) for row in items],
            "cluster": [float(row["cluster_gbps"]) for row in items],
            "torch.compile": [float(row["torch_gbps"]) for row in items],
        }
        for name, values in series.items():
            ax.plot(
                x,
                values,
                marker="o",
                linewidth=2.0,
                label=name,
                color=COLORS[name],
            )
            label_points(ax, x, values, [f"{v:.0f}" for v in values])
        ax.set_title(f"{workload}: modeled bandwidth")
        ax.set_ylabel("GB/s, higher is better")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("N")
        ax.grid(alpha=0.25)
        ax.legend(ncols=3, fontsize=9)

    fig.savefig(out_dir / "markdown_gbps.png", dpi=220)
    fig.savefig(out_dir / "markdown_gbps.pdf")
    plt.close(fig)


def plot_runtime(rows: list[dict[str, str]], out_dir: Path) -> None:
    grouped = group_by_workload(rows)
    fig, axes = plt.subplots(
        len(grouped), 1, figsize=(12, 4.2 * len(grouped)), constrained_layout=True
    )
    if len(grouped) == 1:
        axes = [axes]

    for ax, (workload, items) in zip(axes, grouped.items()):
        x = np.arange(len(items))
        labels = [row["cols"] for row in items]
        series = {
            "block": [float(row["block_ms"]) for row in items],
            "cluster": [float(row["cluster_ms"]) for row in items],
            "torch.compile": [float(row["torch_ms"]) for row in items],
        }
        for name, values in series.items():
            ax.plot(
                x,
                values,
                marker="o",
                linewidth=2.0,
                label=name,
                color=COLORS[name],
            )
            label_points(ax, x, values, [f"{v:.3f}" for v in values])
        ax.set_title(f"{workload}: runtime")
        ax.set_ylabel("ms, lower is better")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("N")
        ax.grid(alpha=0.25)
        ax.legend(ncols=3, fontsize=9)

    fig.savefig(out_dir / "markdown_runtime_ms.png", dpi=220)
    fig.savefig(out_dir / "markdown_runtime_ms.pdf")
    plt.close(fig)


def plot_ncu(path: Path, out_dir: Path) -> None:
    if not path.exists():
        return
    rows = read_csv(path)
    if not rows:
        return

    preferred = ["softmax", "rmsnorm", "cross_entropy"]
    present = {row["workload"] for row in rows}
    workloads = [workload for workload in preferred if workload in present]
    variants = ["block", "cluster"]
    metrics = [
        ("time_us", "NCU kernel time (us)"),
        ("l2_bytes_per_element", "L2 bytes / element"),
        ("dram_bytes_per_element", "DRAM bytes / element"),
    ]

    fig, axes = plt.subplots(
        len(metrics),
        len(workloads),
        figsize=(5.0 * len(workloads), 4.0 * len(metrics)),
        constrained_layout=True,
    )
    if len(workloads) == 1:
        axes = np.array(axes).reshape(len(metrics), 1)

    row_by_key = {(row["workload"], row["variant"]): row for row in rows}
    for col, workload in enumerate(workloads):
        for metric_idx, (metric, title) in enumerate(metrics):
            ax = axes[metric_idx][col]
            values = [
                float(row_by_key[(workload, variant)][metric])
                if (workload, variant) in row_by_key
                else np.nan
                for variant in variants
            ]
            bars = ax.bar(
                variants,
                values,
                color=[COLORS["block"], COLORS["cluster"]],
                edgecolor="black",
                linewidth=0.4,
            )
            ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], fontsize=8)
            ax.set_title(f"{workload}: {title}")
            ax.grid(axis="y", alpha=0.25)

    fig.savefig(out_dir / "markdown_ncu_metrics.png", dpi=220)
    fig.savefig(out_dir / "markdown_ncu_metrics.pdf")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.csv)
    plot_speedup(rows, args.out_dir)
    plot_gbps(rows, args.out_dir)
    plot_runtime(rows, args.out_dir)
    plot_ncu(args.ncu_csv, args.out_dir)
    print(f"Wrote plots to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
