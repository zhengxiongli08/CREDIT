#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


WORKLOAD_ORDER = [
    "layernorm_backward",
    "weighted_var_backward",
    "pearson_backward",
    "softmax_logits_backward",
    "lars_momentum",
    "rowwise_quant",
]
WORKLOAD_SHORT = {
    "layernorm_backward": "LayerNorm",
    "weighted_var_backward": "WVar",
    "pearson_backward": "Pearson",
    "softmax_logits_backward": "SM-logits",
    "lars_momentum": "LARS",
    "rowwise_quant": "Quant",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate DSMEM result bundles across selected GPU devices."
    )
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/comparison"))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def geometric_mean(values: Iterable[float]) -> float:
    positive = [value for value in values if value > 0 and math.isfinite(value)]
    if not positive:
        return math.nan
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def short_device_name(metadata: dict[str, Any]) -> str:
    requested = str(metadata.get("requested_gpu", ""))
    device = str(metadata.get("device", "unknown"))
    if "B200" in requested or "B200" in device:
        return "B200"
    if "H100" in requested or "H100" in device:
        return "H100"
    if "5090" in device:
        return "RTX 5090"
    return device


def load_run(path: Path) -> dict[str, Any]:
    metadata_path = path / "metadata.json"
    summary_path = path / "summary.csv"
    if not metadata_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            f"{path} must contain metadata.json and summary.csv"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = read_csv(summary_path)
    primitive_path = path / "primitive_profile.json"
    primitive = (
        json.loads(primitive_path.read_text(encoding="utf-8"))
        if primitive_path.exists()
        else {}
    )
    return {
        "path": path,
        "label": short_device_name(metadata),
        "metadata": metadata,
        "rows": rows,
        "primitive": primitive,
    }


def build_report(
    runs: list[dict[str, Any]], common_width: int, model_rows: list[dict[str, Any]]
) -> str:
    lines = [
        "# Cross-GPU DSMEM Evaluation",
        "",
        f"Largest width shared by every result bundle: N={common_width}.",
        "",
        "## Performance",
        "",
        "| Device | Wins/all points | Geomean at common N |",
        "|---|---:|---:|",
    ]
    for run in runs:
        rows = run["rows"]
        wins = sum(float(row["dsmem_vs_best"]) > 1.0 for row in rows)
        common = [row for row in rows if int(row["cols"]) == common_width]
        geomean = geometric_mean(float(row["dsmem_vs_best"]) for row in common)
        lines.append(
            f"| {run['label']} | {wins}/{len(rows)} | {geomean:.3f}x |"
        )

    lines.extend(
        [
            "",
            "## Cost Model",
            "",
            "| Device | Peak/additive | Revised | Revised accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in model_rows:
        lines.append(
            f"| {row['device']} | {row['previous_correct']}/{row['points']} "
            f"| {row['correct']}/{row['points']} "
            f"| {100.0 * row['accuracy']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Device Calibration",
            "",
            "| Device | HBM GB/s | L2 GB/s | Local store B/cycle/CTA | DSMEM store B/cycle/CTA | Max SMEM/CTA |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for run in runs:
        profile = run["metadata"].get("profile", {})
        lines.append(
            f"| {run['label']} | {float(profile.get('hbm_bandwidth_gbps', math.nan)):.1f} "
            f"| {float(profile.get('l2_bandwidth_gbps', math.nan)):.1f} "
            f"| {float(profile.get('local_smem_store_bytes_per_cycle_cta', math.nan)):.2f} "
            f"| {float(profile.get('dsmem_store_bytes_per_cycle_cta', math.nan)):.2f} "
            f"| {int(profile.get('max_shared_bytes_per_cta', 0))} |"
        )
    return "\n".join(lines) + "\n"


def plot_results(
    runs: list[dict[str, Any]],
    common_width: int,
    model_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7.5,
            "axes.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
        }
    )
    colors = ["#666666", "#0072B2", "#D55E00"]
    markers = ["o", "s", "^"]
    fig, (ax_width, ax_workload, ax_model) = plt.subplots(
        1, 3, figsize=(7.05, 2.42), gridspec_kw={"width_ratios": [1.05, 1.55, 0.75]}
    )

    all_widths = sorted({int(row["cols"]) for run in runs for row in run["rows"]})
    width_positions = {width: index for index, width in enumerate(all_widths)}
    for run, color, marker in zip(runs, colors, markers):
        by_width: dict[int, list[float]] = defaultdict(list)
        for row in run["rows"]:
            by_width[int(row["cols"])].append(float(row["dsmem_vs_best"]))
        widths = sorted(by_width)
        speedups = [geometric_mean(by_width[width]) for width in widths]
        ax_width.plot(
            [width_positions[width] for width in widths],
            speedups,
            marker=marker,
            color=color,
            linewidth=1.4,
            markersize=4,
            label=run["label"],
        )
    ax_width.set_xticks(
        np.arange(len(all_widths)), [f"{width // 1024}K" for width in all_widths]
    )
    ax_width.axhline(1.0, color="#333333", linewidth=0.8, linestyle="--")
    ax_width.set_xlabel("Row width, $N$")
    ax_width.set_ylabel("Geomean D/best")
    ax_width.legend(frameon=False)
    ax_width.text(0.02, 0.96, "(a)", transform=ax_width.transAxes, va="top", fontweight="bold")

    x = np.arange(len(WORKLOAD_ORDER))
    width = 0.78 / len(runs)
    for index, (run, color) in enumerate(zip(runs, colors)):
        by_name = {
            row["workload"]: float(row["dsmem_vs_best"])
            for row in run["rows"]
            if int(row["cols"]) == common_width
        }
        values = [by_name.get(name, math.nan) for name in WORKLOAD_ORDER]
        offset = (index - (len(runs) - 1) / 2) * width
        ax_workload.bar(x + offset, values, width, color=color, label=run["label"])
    ax_workload.axhline(1.0, color="#333333", linewidth=0.8, linestyle="--")
    ax_workload.set_xticks(
        x,
        [WORKLOAD_SHORT[name] for name in WORKLOAD_ORDER],
        rotation=35,
        ha="right",
    )
    ax_workload.set_ylabel(f"D/best at {common_width // 1024}K")
    ax_workload.legend(frameon=False, ncol=min(3, len(runs)), loc="upper left")
    ax_workload.text(
        0.98,
        0.96,
        "(b)",
        transform=ax_workload.transAxes,
        ha="right",
        va="top",
        fontweight="bold",
    )

    model_x = np.arange(len(model_rows))
    model_values = [100.0 * row["accuracy"] for row in model_rows]
    ax_model.bar(model_x, model_values, color=colors[: len(model_rows)], width=0.65)
    ax_model.set_xticks(model_x, [row["device"] for row in model_rows], rotation=25, ha="right")
    ax_model.set_ylim(0, 105)
    ax_model.set_yticks([0, 25, 50, 75, 100])
    ax_model.set_ylabel("Model accuracy (%)")
    ax_model.text(0.03, 0.96, "(c)", transform=ax_model.transAxes, va="top", fontweight="bold")

    for axis in (ax_width, ax_workload, ax_model):
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_axisbelow(True)
    fig.tight_layout(pad=0.45, w_pad=0.8)
    fig.savefig(output_dir / "cross_gpu_summary.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "cross_gpu_summary.png", dpi=300, bbox_inches="tight")


def main() -> int:
    args = parse_args()
    runs = [load_run(path.resolve()) for path in args.runs]
    labels = [run["label"] for run in runs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"device labels must be unique, got {labels}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    common_widths = set.intersection(
        *[
            {int(row["cols"]) for row in run["rows"]}
            for run in runs
        ]
    )
    if not common_widths:
        raise ValueError("the result bundles have no common row width")
    common_width = max(common_widths)

    point_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for run in runs:
        by_width: dict[int, list[float]] = defaultdict(list)
        for row in run["rows"]:
            copied = {"device": run["label"], **row}
            point_rows.append(copied)
            by_width[int(row["cols"])].append(float(row["dsmem_vs_best"]))
        for width, values in sorted(by_width.items()):
            aggregate_rows.append(
                {
                    "device": run["label"],
                    "cols": width,
                    "points": len(values),
                    "wins_over_best": sum(value > 1.0 for value in values),
                    "geomean_dsmem_vs_best": geometric_mean(values),
                }
            )
        correct = sum(truth(row["model_correct"]) for row in run["rows"])
        points = len(run["rows"])
        previous_correct = int(
            run["metadata"].get("cost_model", {}).get("previous_correct", correct)
        )
        model_rows.append(
            {
                "device": run["label"],
                "previous_correct": previous_correct,
                "correct": correct,
                "points": points,
                "accuracy": correct / points,
            }
        )

    write_csv(args.output_dir / "cross_gpu_points.csv", point_rows)
    write_csv(args.output_dir / "cross_gpu_aggregate.csv", aggregate_rows)
    write_csv(args.output_dir / "cross_gpu_model.csv", model_rows)
    (args.output_dir / "CROSS_GPU_REPORT.md").write_text(
        build_report(runs, common_width, model_rows), encoding="utf-8"
    )
    plot_results(runs, common_width, model_rows, args.output_dir)
    print(f"Wrote cross-GPU artifacts to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
