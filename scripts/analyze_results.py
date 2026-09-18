#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


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
    print(f"Wrote cross-GPU tables and report to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
