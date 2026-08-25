#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_WORKLOADS = {
    "layernorm_backward",
    "weighted_var_backward",
    "pearson_backward",
    "softmax_logits_backward",
    "lars_momentum",
    "rowwise_quant",
}
EXPECTED_WIDTHS = {4096, 8192, 16384, 32768, 65536}
REQUIRED_RESULT_FILES = {
    "REPORT.md",
    "cluster_sweep.csv",
    "control_raw.csv",
    "cuda_raw.csv",
    "framework_raw.json",
    "metadata.json",
    "summary.csv",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources() -> None:
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        path = ROOT / record["path"]
        if not path.is_file():
            raise AssertionError(f"missing source listed in manifest: {record['path']}")
        actual = sha256(path)
        if actual != record["sha256"]:
            raise AssertionError(
                f"source hash mismatch for {record['path']}: {actual}"
            )


def verify_result_bundle(directory: Path, device_fragment: str) -> None:
    missing = sorted(
        name for name in REQUIRED_RESULT_FILES if not (directory / name).is_file()
    )
    if missing:
        raise AssertionError(f"{directory}: missing {', '.join(missing)}")

    metadata: dict[str, Any] = json.loads(
        (directory / "metadata.json").read_text(encoding="utf-8")
    )
    if device_fragment not in str(metadata["device"]):
        raise AssertionError(f"{directory}: unexpected device {metadata['device']!r}")

    rows = read_csv(directory / "summary.csv")
    points = {(row["workload"], int(row["cols"])) for row in rows}
    expected = {
        (workload, width)
        for workload in EXPECTED_WORKLOADS
        for width in EXPECTED_WIDTHS
    }
    if points != expected:
        raise AssertionError(f"{directory}: summary does not contain the 30 expected points")

    largest_width = [row for row in rows if int(row["cols"]) == 65536]
    if len(largest_width) != len(EXPECTED_WORKLOADS):
        raise AssertionError(f"{directory}: incomplete 64K result set")
    for row in largest_width:
        speedup = float(row["dsmem_vs_best"])
        if not math.isfinite(speedup) or speedup <= 1.0:
            raise AssertionError(
                f"{directory}: {row['workload']} is not a DSMEM win at 64K"
            )


def verify_comparison() -> None:
    rows = read_csv(ROOT / "results/comparison/cross_gpu_aggregate.csv")
    devices = {row["device"] for row in rows}
    if devices != {"RTX 5090", "H100"}:
        raise AssertionError(f"unexpected aggregate devices: {sorted(devices)}")
    if len(rows) != 2 * len(EXPECTED_WIDTHS):
        raise AssertionError("cross-GPU aggregate has an unexpected number of rows")


def main() -> int:
    verify_sources()
    verify_result_bundle(ROOT / "results/rtx5090", "RTX 5090")
    verify_result_bundle(ROOT / "results/h100", "H100")
    verify_comparison()
    print("Artifact verification passed: sources, result bundles, and aggregates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
