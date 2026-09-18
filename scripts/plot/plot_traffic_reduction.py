#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _common import WORKLOAD_ORDER, WORKLOAD_SHORT, configure_matplotlib, save_figure


# Static DRAM accounting used in the camera-ready paper figure.
NO_DSMEM_BYTES = [24.0, 40.0, 24.0, 28.0, 32.0, 9.0]
CREDIT_BYTES = [12.0, 16.0, 16.0, 12.0, 20.0, 5.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the paper DRAM-traffic figure."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/figures")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(WORKLOAD_ORDER))
    width = 0.36
    fig, axis = plt.subplots(figsize=(3.823, 1.93))
    axis.bar(
        x - width / 2,
        NO_DSMEM_BYTES,
        width,
        color="#6B6B6B",
        label="No DSMEM",
    )
    axis.bar(
        x + width / 2,
        CREDIT_BYTES,
        width,
        color="#009E73",
        label="CREDIT",
    )

    for index, (before, after) in enumerate(zip(NO_DSMEM_BYTES, CREDIT_BYTES)):
        arrow_x = index + width / 2
        axis.plot(
            [index - width / 2, arrow_x],
            [before, before],
            color="#333333",
            linewidth=0.65,
        )
        axis.annotate(
            "",
            xy=(arrow_x, before),
            xytext=(arrow_x, after),
            arrowprops={
                "arrowstyle": "<->",
                "color": "#333333",
                "linewidth": 0.75,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
            },
        )
        reduction = round(100.0 * (before - after) / before)
        axis.text(
            arrow_x,
            (before + after) / 2,
            f"{reduction:.0f}%",
            ha="center",
            va="center",
            fontsize=6.0,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
        )

    axis.set_title("RTX 5090 at $N=64$K", loc="left", fontweight="bold", pad=3.5)
    axis.set_ylabel("DRAM traffic (B/element)")
    axis.set_xticks(
        x,
        [WORKLOAD_SHORT[name] for name in WORKLOAD_ORDER],
        rotation=30,
        ha="right",
    )
    axis.set_ylim(0, 44.5)
    axis.set_yticks([0, 10, 20, 30, 40])
    axis.grid(axis="y", color="#D0D0D0", linewidth=0.45)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper right",
        bbox_to_anchor=(0.995, 1.0),
        frameon=False,
        handlelength=1.6,
        borderaxespad=0.3,
    )
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.27, top=0.86)
    save_figure(fig, args.output_dir, "traffic_reduction")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
