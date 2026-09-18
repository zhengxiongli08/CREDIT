#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _common import (
    DEVICE_COLORS,
    DEVICE_MARKERS,
    WORKLOAD_ORDER,
    configure_matplotlib,
    load_paper_runs,
    save_figure,
)


TITLES = {
    "layernorm_backward": "LayerNorm backward",
    "weighted_var_backward": "Weighted variance backward",
    "pearson_backward": "Pearson backward",
    "softmax_logits_backward": "Softmax-logits backward",
    "lars_momentum": "LARS momentum",
    "rowwise_quant": "Row-wise quantization",
}
Y_AXIS = {
    "layernorm_backward": ((0.45, 2.08), [0.5, 1.0, 1.5, 2.0]),
    "weighted_var_backward": ((0.65, 2.55), [0.75, 1.0, 1.5, 2.0, 2.5]),
    "pearson_backward": ((0.85, 1.62), [0.9, 1.0, 1.2, 1.4, 1.6]),
    "softmax_logits_backward": ((0.68, 1.52), [0.75, 1.0, 1.25, 1.5]),
    "lars_momentum": ((0.94, 1.055), [0.95, 1.0, 1.05]),
    "rowwise_quant": ((0.2, 1.28), [0.25, 0.5, 0.75, 1.0, 1.25]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the paper workload-scaling figure."
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

    fig, axes = plt.subplots(2, 3, figsize=(7.08, 2.73))
    legend_handles = []
    legend_labels = []
    for index, (axis, workload) in enumerate(zip(axes.flat, WORKLOAD_ORDER)):
        for device, rows in runs:
            selected = sorted(
                (row for row in rows if row["workload"] == workload),
                key=lambda row: int(row["cols"]),
            )
            line = axis.plot(
                range(len(selected)),
                [float(row["dsmem_vs_best"]) for row in selected],
                color=DEVICE_COLORS[device],
                marker=DEVICE_MARKERS[device],
                linewidth=1.35,
                markersize=4.0,
                markeredgewidth=0.0,
                label=device,
            )[0]
            if index == 0:
                legend_handles.append(line)
                legend_labels.append(device)

        axis.axhline(1.0, color="#222222", linewidth=0.7, linestyle="--")
        axis.set_xticks(
            range(len(selected)),
            [f"{int(row['cols']) // 1024}K" for row in selected],
        )
        limits, ticks = Y_AXIS[workload]
        axis.set_ylim(*limits)
        axis.set_yticks(ticks)
        axis.grid(axis="y", color="#D0D0D0", linewidth=0.45)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.set_title(
            f"({chr(ord('a') + index)}) {TITLES[workload]}",
            loc="left",
            fontweight="bold",
            pad=4.0,
        )
        if index % 3 == 0:
            axis.set_ylabel("Speedup")
        if index >= 3:
            axis.set_xlabel("Row width, $N$")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.015),
        ncol=2,
        frameon=False,
        handlelength=2.1,
        columnspacing=1.7,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.995,
        bottom=0.16,
        top=0.89,
        wspace=0.20,
        hspace=0.58,
    )
    save_figure(fig, args.output_dir, "workload_scaling")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
