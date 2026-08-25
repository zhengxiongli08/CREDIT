from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def geometric_mean(values: Iterable[float]) -> float:
    positive = [value for value in values if value > 0.0 and math.isfinite(value)]
    if not positive:
        return math.nan
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def build_report(
    summary_rows: list[dict[str, Any]], metadata: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    valid_rows = [row for row in summary_rows if _finite(row.get("dsmem_vs_best"))]
    wins = sum(float(row["dsmem_vs_best"]) > 1.0 for row in valid_rows)
    model_rows = [row for row in summary_rows if "model_correct" in row]
    model_correct = sum(_truth(row["model_correct"]) for row in model_rows)
    largest_width = max((int(row["cols"]) for row in valid_rows), default=0)
    largest_rows = [row for row in valid_rows if int(row["cols"]) == largest_width]
    largest_geomean = geometric_mean(
        float(row["dsmem_vs_best"]) for row in largest_rows
    )

    by_workload: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_workload[str(row["workload"])].append(row)
    crossovers: list[dict[str, Any]] = []
    for rows in by_workload.values():
        ordered = sorted(rows, key=lambda row: int(row["cols"]))
        measured = next(
            (
                int(row["cols"])
                for row in ordered
                if _truth(row.get("measured_profitable_vs_cuda", False))
            ),
            None,
        )
        predicted = next(
            (
                int(row["cols"])
                for row in ordered
                if _truth(row.get("predicted_profitable", False))
            ),
            None,
        )
        crossovers.append(
            {
                "workload": ordered[0]["workload"],
                "label": ordered[0]["label"],
                "predicted": predicted,
                "measured": measured,
            }
        )

    device = str(metadata["device"])
    lines = [
        f"# {device} DSMEM Evaluation",
        "",
        f"- DSMEM beats the fastest baseline on {wins}/{len(valid_rows)} workload-size points.",
        f"- The calibrated model predicts {model_correct}/{len(model_rows)} CUDA profitability signs correctly.",
    ]
    if largest_width:
        lines.append(
            f"- At N={largest_width}, DSMEM has a {largest_geomean:.3f}x geometric-mean speedup over the best baseline."
        )

    lines.extend(
        [
            "",
            f"## Results at N={largest_width}",
            "",
            "| Workload | D/torch.compile | D/Triton | D/CUDA | D/best | Best baseline | P |",
            "|---|---:|---:|---:|---:|---|---:|",
        ]
    )
    for row in largest_rows:
        lines.append(
            f"| {row['label']} | {float(row['dsmem_vs_torch']):.3f}x "
            f"| {float(row['dsmem_vs_triton']):.3f}x "
            f"| {float(row['dsmem_vs_cuda_nodsmem']):.3f}x "
            f"| {float(row['dsmem_vs_best']):.3f}x | {row['best_baseline']} "
            f"| {row['measured_cluster_size']} |"
        )

    lines.extend(
        [
            "",
            "## Crossover Validation",
            "",
            "Crossovers are relative to the best non-DSMEM CUDA path.",
            "",
            "| Workload | Predicted | Measured |",
            "|---|---:|---:|",
        ]
    )
    for row in crossovers:
        predicted = "none" if row["predicted"] is None else str(row["predicted"])
        measured = "none" if row["measured"] is None else str(row["measured"])
        lines.append(f"| {row['label']} | {predicted} | {measured} |")

    aggregate = {
        "device": device,
        "points": len(valid_rows),
        "wins_over_best": wins,
        "model_points": len(model_rows),
        "model_correct": model_correct,
        "largest_width": largest_width,
        "largest_width_geomean_speedup": largest_geomean,
        "crossovers": crossovers,
    }
    return "\n".join(lines) + "\n", aggregate


def write_report(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    report, aggregate = build_report(summary_rows, metadata)
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    return aggregate

