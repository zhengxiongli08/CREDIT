#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from dsmem_eval.cost_model import MODEL_VERSION, DeviceProfile, estimate
from dsmem_eval.report import build_report
from dsmem_eval.workloads import WORKLOADS


RTX5090_LOCAL_STORE_BYTES_PER_CYCLE_CTA = 33.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rescore existing result bundles with the revised cost model."
    )
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument(
        "--write",
        action="store_true",
        help="replace summary/model columns after preserving *.peak_model.csv backups",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def profile_from_bundle(run: Path, metadata: dict[str, Any]) -> DeviceProfile:
    raw = metadata["profile"]
    primitive_path = run / "primitive_profile.json"
    measurements: dict[str, Any] = {}
    if primitive_path.exists():
        primitive = json.loads(primitive_path.read_text(encoding="utf-8"))
        measurements = primitive.get("measurements", {})

    local_store = raw.get("local_smem_store_bytes_per_cycle_cta")
    if local_store is None:
        local_store = measurements.get("local_smem_store_bytes_per_cycle_cta")
    if local_store is None and "5090" in str(metadata.get("device", "")):
        local_store = RTX5090_LOCAL_STORE_BYTES_PER_CYCLE_CTA
    if local_store is None:
        raise ValueError(f"{run}: missing local-SMEM store throughput")

    return DeviceProfile(
        name=str(raw["name"]),
        sm_count=int(raw["sm_count"]),
        clock_ghz=float(raw["clock_ghz"]),
        hbm_bandwidth_gbps=float(raw["hbm_bandwidth_gbps"]),
        l2_capacity_bytes=int(raw["l2_capacity_bytes"]),
        l2_bandwidth_gbps=float(raw["l2_bandwidth_gbps"]),
        local_smem_bytes_per_cycle_cta=float(
            raw["local_smem_bytes_per_cycle_cta"]
        ),
        local_smem_store_bytes_per_cycle_cta=float(local_store),
        dsmem_store_bytes_per_cycle_cta=float(
            raw["dsmem_store_bytes_per_cycle_cta"]
        ),
        max_shared_bytes_per_cta=int(raw["max_shared_bytes_per_cta"]),
    )


def rescore(run: Path, write: bool) -> dict[str, Any]:
    run = run.resolve()
    summary_path = run / "summary.csv"
    cluster_path = run / "cluster_sweep.csv"
    metadata_path = run / "metadata.json"
    if not all(path.exists() for path in (summary_path, cluster_path, metadata_path)):
        raise FileNotFoundError(
            f"{run} must contain summary.csv, cluster_sweep.csv, and metadata.json"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    profile = profile_from_bundle(run, metadata)
    summary_rows: list[dict[str, Any]] = read_csv(summary_path)
    cluster_rows: list[dict[str, Any]] = read_csv(cluster_path)
    summary_by_point = {
        (str(row["workload"]), int(row["cols"])): row for row in summary_rows
    }
    estimates_by_point: dict[tuple[str, int], list[tuple[float, int, bool]]] = (
        defaultdict(list)
    )

    current_correct = sum(
        truth(row.get("model_correct", False)) for row in summary_rows
    )
    previous_correct = metadata.get("cost_model", {}).get("previous_correct")
    summary_backup = run / "summary.peak_model.csv"
    if previous_correct is None and summary_backup.exists():
        previous_correct = sum(
            truth(row.get("model_correct", False))
            for row in read_csv(summary_backup)
        )
    if previous_correct is None:
        previous_correct = current_correct
    for row in cluster_rows:
        name = str(row["workload"])
        cols = int(row["cols"])
        point = summary_by_point[(name, cols)]
        model = estimate(
            WORKLOADS[name],
            cols,
            int(row["cluster_size"]),
            float(row["control_overhead_ms"]),
            profile,
            baseline_ms=float(point["cuda_nodsmem_ms"]),
            rows=int(row["rows"]),
        )
        row.update(
            {
                "model_version": MODEL_VERSION,
                "effective_bandwidth_gbps": model.effective_bandwidth_gbps,
                "saved_reread_ms": model.saved_reread_ms,
                "saved_hbm_ms": model.saved_hbm_ms,
                "local_replay_ms": model.local_replay_ms,
                "local_store_residual_ms": model.local_store_residual_ms,
                "local_staging_ms": model.local_staging_ms,
                "dsmem_store_ms": model.dsmem_store_ms,
                "predicted_delta_ms": model.predicted_delta_ms,
                "predicted_profitable": model.predicted_profitable,
            }
        )
        estimates_by_point[(name, cols)].append(
            (
                model.predicted_delta_ms,
                int(row["cluster_size"]),
                model.predicted_profitable,
            )
        )

    for row in summary_rows:
        key = (str(row["workload"]), int(row["cols"]))
        predicted_delta, predicted_cluster_size, predicted_profitable = max(
            estimates_by_point[key], key=lambda item: item[0]
        )
        measured_profitable = truth(row["measured_profitable_vs_cuda"])
        row.update(
            {
                "model_version": MODEL_VERSION,
                "predicted_cluster_size": predicted_cluster_size,
                "predicted_delta_ms": predicted_delta,
                "predicted_profitable": predicted_profitable,
                "model_correct": predicted_profitable == measured_profitable,
            }
        )

    new_correct = sum(truth(row["model_correct"]) for row in summary_rows)
    result = {
        "device": str(metadata["device"]),
        "points": len(summary_rows),
        "old_correct": int(previous_correct),
        "new_correct": new_correct,
    }
    if not write:
        return result

    cluster_backup = run / "cluster_sweep.peak_model.csv"
    if not summary_backup.exists():
        shutil.copy2(summary_path, summary_backup)
    if not cluster_backup.exists():
        shutil.copy2(cluster_path, cluster_backup)
    write_csv(summary_path, summary_rows)
    write_csv(cluster_path, cluster_rows)

    report, aggregate = build_report(summary_rows, metadata)
    (run / "REPORT.md").write_text(report, encoding="utf-8")
    metadata["aggregate"] = aggregate
    metadata["profile"]["local_smem_store_bytes_per_cycle_cta"] = (
        profile.local_smem_store_bytes_per_cycle_cta
    )
    metadata["cost_model"] = {
        "version": MODEL_VERSION,
        "uses_dsmem_workload_timing": False,
        "baseline_calibration": "non-DSMEM CUDA time at the same shape",
        "previous_correct": int(previous_correct),
        "revised_correct": new_correct,
        "points": len(summary_rows),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    args = parse_args()
    for run in args.runs:
        result = rescore(run, args.write)
        print(
            f"{result['device']}: {result['old_correct']}/{result['points']} -> "
            f"{result['new_correct']}/{result['points']}"
        )
    if not args.write:
        print("Dry run only; pass --write to update the bundles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
