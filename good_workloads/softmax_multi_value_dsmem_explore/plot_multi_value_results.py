#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot multi-value DSMEM results.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=ROOT / "results" / "multi_value_torch_compare.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "plots")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def group_by_channels(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(int(row["channels"]), []).append(row)
    for channel_rows in grouped.values():
        channel_rows.sort(key=lambda row: int(row["cols"]))
    return grouped


def save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=180)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def plot_speedup(rows: list[dict[str, str]], out_dir: Path) -> None:
    grouped = group_by_channels(rows)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for channels, channel_rows in grouped.items():
        xs = [str(int(row["cols"])) for row in channel_rows]
        ys = [float(row["cluster_vs_torch"]) for row in channel_rows]
        ax.plot(xs, ys, marker="o", linewidth=2, label=f"D={channels}")
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2f}x", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_title("DSMEM Cluster Speedup vs torch.compile")
    ax.set_xlabel("N")
    ax.set_ylabel("cluster / torch.compile throughput")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Value channels")
    save(fig, out_dir, "multi_value_speedup_vs_torch")


def plot_runtime(rows: list[dict[str, str]], out_dir: Path) -> None:
    grouped = group_by_channels(rows)
    fig, axes = plt.subplots(1, len(grouped), figsize=(13.5, 4.5), sharey=False)
    if len(grouped) == 1:
        axes = [axes]
    for ax, (channels, channel_rows) in zip(axes, sorted(grouped.items())):
        xs = [str(int(row["cols"])) for row in channel_rows]
        ax.plot(xs, [float(row["block_ms"]) for row in channel_rows],
                marker="o", label="best noDSMEM")
        ax.plot(xs, [float(row["cluster_ms"]) for row in channel_rows],
                marker="o", label="DSMEM cluster")
        ax.plot(xs, [float(row["torch_ms"]) for row in channel_rows],
                marker="o", label="torch.compile")
        ax.set_title(f"D={channels}")
        ax.set_xlabel("N")
        ax.set_ylabel("runtime (ms)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    save(fig, out_dir, "multi_value_runtime_ms")


def plot_gbps(rows: list[dict[str, str]], out_dir: Path) -> None:
    grouped = group_by_channels(rows)
    fig, axes = plt.subplots(1, len(grouped), figsize=(13.5, 4.5), sharey=True)
    if len(grouped) == 1:
        axes = [axes]
    for ax, (channels, channel_rows) in zip(axes, sorted(grouped.items())):
        xs = [str(int(row["cols"])) for row in channel_rows]
        ax.plot(xs, [float(row["block_gbps"]) for row in channel_rows],
                marker="o", label="best noDSMEM")
        ax.plot(xs, [float(row["cluster_gbps"]) for row in channel_rows],
                marker="o", label="DSMEM cluster")
        ax.plot(xs, [float(row["torch_gbps"]) for row in channel_rows],
                marker="o", label="torch.compile")
        ax.set_title(f"D={channels}")
        ax.set_xlabel("N")
        ax.set_ylabel("modeled GB/s")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    save(fig, out_dir, "multi_value_gbps")


def plot_ncu(out_dir: Path) -> None:
    path = ROOT / "results" / "multi_value_ncu_summary.csv"
    if not path.exists():
        return
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    labels = [f"D={row['channels']} {row['variant']}" for row in rows]
    time_us = [float(row["time_us"]) for row in rows]
    l2_bpe = [float(row["l2_bytes_per_element"]) for row in rows]
    dram_bpe = [float(row["dram_bytes_per_element"]) for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.2))
    colors = ["#4c78a8" if row["variant"] == "block" else "#f58518"
              for row in rows]
    axes[0].bar(labels, time_us, color=colors)
    axes[0].set_title("NCU Time")
    axes[0].set_ylabel("us")
    axes[1].bar(labels, l2_bpe, color=colors)
    axes[1].set_title("L2 Traffic")
    axes[1].set_ylabel("B / score element")
    axes[2].bar(labels, dram_bpe, color=colors)
    axes[2].set_title("DRAM Traffic")
    axes[2].set_ylabel("B / score element")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=25)
    save(fig, out_dir, "multi_value_ncu_metrics")


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
