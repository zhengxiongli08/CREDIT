#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ROWS = int(os.environ.get("ROWS", "2048"))
COLS = int(os.environ.get("COLS", "65536"))
CHANNELS = int(os.environ.get("CHANNELS", "8"))


def read_ncu_rows(path: Path) -> list[dict[str, str]]:
    lines = path.read_text().splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith('"ID",'))
    except StopIteration:
        return []
    rows = list(csv.DictReader(lines[start:]))
    return [row for row in rows if row.get("ID", "").isdigit()]


def to_float(row: dict[str, str], name: str) -> float:
    value = row.get(name, "")
    return float(value) if value else 0.0


def modeled_bytes(channels: int) -> float:
    return ROWS * COLS * 4.0 * (channels + 2.0) + ROWS * channels * 4.0


def summarize(
    variant: str, path: Path, channels: int
) -> dict[str, float | int | str] | None:
    rows = read_ncu_rows(path)
    if not rows:
        return None
    score_elements = ROWS * COLS
    time_ns = sum(to_float(row, "gpu__time_duration.sum") for row in rows)
    l2_bytes = sum(to_float(row, "lts__t_bytes.sum") for row in rows)
    dram_read = sum(to_float(row, "dram__bytes_op_read.sum") for row in rows)
    dram_write = sum(to_float(row, "dram__bytes_op_write.sum") for row in rows)
    return {
        "variant": variant,
        "cols": COLS,
        "channels": channels,
        "rows": ROWS,
        "profiled_launches": len(rows),
        "time_us": time_ns / 1.0e3,
        "modeled_gbps": modeled_bytes(channels) / (time_ns / 1.0e9) / 1.0e9,
        "l2_bytes_per_element": l2_bytes / score_elements,
        "dram_bytes_per_element": (dram_read + dram_write) / score_elements,
        "dram_read_bytes_per_element": dram_read / score_elements,
        "dram_write_bytes_per_element": dram_write / score_elements,
    }


def discover_cases() -> list[tuple[int, str, Path]]:
    cases: list[tuple[int, str, Path]] = []
    pattern = re.compile(r"ncu_multi_value_d(\d+)_(block|cluster)\.csv$")
    for path in sorted(RESULTS.glob("ncu_multi_value_d*_*.csv")):
        match = pattern.match(path.name)
        if match is None:
            continue
        cases.append((int(match.group(1)), match.group(2), path))
    if cases:
        return cases

    fallback = [
        (CHANNELS, "block", RESULTS / "ncu_multi_value_block.csv"),
        (CHANNELS, "cluster", RESULTS / "ncu_multi_value_cluster.csv"),
    ]
    return [(channels, variant, path) for channels, variant, path in fallback
            if path.exists()]


def main() -> int:
    rows = []
    for channels, variant, path in discover_cases():
        if not path.exists():
            continue
        summary = summarize(variant, path, channels)
        if summary is not None:
            rows.append(summary)
    rows.sort(key=lambda row: (int(row["channels"]), str(row["variant"])))

    out = RESULTS / "multi_value_ncu_summary.csv"
    fieldnames = [
        "variant",
        "cols",
        "channels",
        "rows",
        "profiled_launches",
        "time_us",
        "modeled_gbps",
        "l2_bytes_per_element",
        "dram_bytes_per_element",
        "dram_read_bytes_per_element",
        "dram_write_bytes_per_element",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out}")
    for row in rows:
        print(
            f"{row['variant']:<8} time_us={row['time_us']:.3f} "
            f"l2_B/score_elem={row['l2_bytes_per_element']:.3f} "
            f"dram_B/score_elem={row['dram_bytes_per_element']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
