#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _common import (
    DEVICE_COLORS,
    WORKLOAD_ORDER,
    WORKLOAD_SHORT,
    configure_matplotlib,
    load_paper_runs,
    save_figure,
)


def truth(value: str) -> bool:
    return value.strip().lower() == "true"


def first_crossover(rows: list[dict[str, str]], workload: str, field: str) -> int:
    selected = sorted(
        (row for row in rows if row["workload"] == workload),
        key=lambda row: int(row["cols"]),
    )
    return next(int(row["cols"]) for row in selected if truth(row[field]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the paper cost-model validation figure."
    )
    parser.add_argument(
        "runs",
        nargs=2,
        type=Path,
        metavar="RESULT_DIR",
        help="one complete RTX 5090 bundle and one complete H100 bundle",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/figures")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = load_paper_runs(args.runs)
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    widths = [4096, 8192, 16384, 32768, 65536]
    width_position = {width: index for index, width in enumerate(widths)}
    fig, axis = plt.subplots(figsize=(3.67, 2.05))

    for device_index, (device, rows) in enumerate(runs):
        offset = -0.11 if device_index == 0 else 0.11
        color = DEVICE_COLORS[device]
        for workload_index, workload in enumerate(WORKLOAD_ORDER):
            predicted = width_position[
                first_crossover(rows, workload, "predicted_profitable")
            ]
            measured = width_position[
                first_crossover(rows, workload, "measured_profitable_vs_cuda")
            ]
            x = workload_index + offset
            axis.plot([x, x], [predicted, measured], color=color, linewidth=1.0)
            axis.plot(
                x,
                measured,
                marker="o",
                markersize=4.5,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.1,
                linestyle="none",
            )
            axis.plot(
                x,
                predicted,
                marker="x",
                markersize=4.7,
                markeredgewidth=1.1,
                color=color,
                linestyle="none",
            )

    device_handles = [
        Line2D([0], [0], color=DEVICE_COLORS[device], linewidth=1.8, label=device)
        for device, _ in runs
    ]
    meaning_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="white",
            markeredgecolor="#333333",
            markeredgewidth=1.0,
            markersize=4.8,
            linestyle="none",
            label="Measured",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="#333333",
            markeredgewidth=1.0,
            markersize=4.8,
            linestyle="none",
            label="Predicted",
        ),
    ]
    handles = [*device_handles, *meaning_handles]
    axis.legend(
        handles,
        [handle.get_label() for handle in handles],
        ncol=2,
        loc="upper left",
        frameon=False,
        columnspacing=1.0,
        handlelength=1.8,
        handletextpad=0.4,
        borderaxespad=0.4,
    )
    axis.set_title("Predicted vs. measured", loc="left", fontweight="bold", pad=4.0)
    axis.set_ylabel("CUDA crossover width")
    axis.set_xticks(
        range(len(WORKLOAD_ORDER)),
        [WORKLOAD_SHORT[name] for name in WORKLOAD_ORDER],
        rotation=30,
        ha="right",
    )
    axis.set_yticks(range(len(widths)), [f"{width // 1024}K" for width in widths])
    axis.set_ylim(-0.38, 4.35)
    axis.grid(axis="y", color="#D0D0D0", linewidth=0.45)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    fig.subplots_adjust(left=0.185, right=0.985, bottom=0.225, top=0.80)
    save_figure(fig, args.output_dir, "model_validation")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
