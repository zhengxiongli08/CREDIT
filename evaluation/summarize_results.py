#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_SUMMARY = HERE / "results/rtx5090/summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the RTX 5090 evaluation.")
    parser.add_argument("--input", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> int:
    args = parse_args()
    with args.input.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["workload"]].append(row)
    for workload_rows in grouped.values():
        workload_rows.sort(key=lambda row: int(row["cols"]))

    best_wins = sum(float(row["dsmem_vs_best"]) > 1.0 for row in rows)
    model_correct = sum(as_bool(row["model_correct"]) for row in rows)
    widest = [row for row in rows if int(row["cols"]) == 65536]
    widest_geomean = geometric_mean(
        [float(row["dsmem_vs_best"]) for row in widest]
    )

    lines = [
        "# RTX 5090 Evaluation Summary",
        "",
        f"- DSMEM beats the fastest baseline on {best_wins}/{len(rows)} workload-size points.",
        f"- The model predicts {model_correct}/{len(rows)} CUDA profitability signs correctly.",
        f"- At N=65536, DSMEM wins on {len(widest)}/{len(widest)} workloads with a {widest_geomean:.3f}x geometric-mean speedup over the best baseline.",
        "",
        "## Results at N=65536",
        "",
        "| Workload | D/torch.compile | D/Triton | D/CUDA | D/best | Best baseline | P |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in widest:
        lines.append(
            f"| {row['label']} | {float(row['dsmem_vs_torch']):.3f}x "
            f"| {float(row['dsmem_vs_triton']):.3f}x "
            f"| {float(row['dsmem_vs_cuda_nodsmem']):.3f}x "
            f"| {float(row['dsmem_vs_best']):.3f}x "
            f"| {row['best_baseline']} | {row['measured_cluster_size']} |"
        )

    lines.extend(
        [
            "",
            "## Crossover Validation",
            "",
            "The crossover below is relative to the best non-DSMEM CUDA variant, which is the mechanism ablation used by the cost model.",
            "",
            "| Workload | Predicted | Measured | Distance |",
            "|---|---:|---:|---:|",
        ]
    )
    widths = [4096, 8192, 16384, 32768, 65536]
    for workload_rows in grouped.values():
        predicted = next(
            int(row["cols"])
            for row in workload_rows
            if as_bool(row["predicted_profitable"])
        )
        measured = next(
            int(row["cols"])
            for row in workload_rows
            if as_bool(row["measured_profitable_vs_cuda"])
        )
        distance = abs(widths.index(predicted) - widths.index(measured))
        lines.append(
            f"| {workload_rows[0]['label']} | {predicted // 1024}K "
            f"| {measured // 1024}K | {distance} tested interval(s) |"
        )

    mismatches = [row for row in rows if not as_bool(row["model_correct"])]
    lines.extend(
        [
            "",
            "## Model Mismatches",
            "",
            "| Workload | N | D/CUDA | Predicted | Measured |",
            "|---|---:|---:|---|---|",
        ]
    )
    for row in mismatches:
        lines.append(
            f"| {row['label']} | {int(row['cols']) // 1024}K "
            f"| {float(row['dsmem_vs_cuda_nodsmem']):.3f}x "
            f"| {row['predicted_profitable']} | {row['measured_profitable_vs_cuda']} |"
        )

    report = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
