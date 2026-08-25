#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ROWS = int(os.environ.get("ROWS", "4096"))
COLS = int(os.environ.get("COLS", "65536"))


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


def summarize(variant: str, path: Path) -> dict[str, float | int | str] | None:
    rows = read_ncu_rows(path)
    if not rows:
        return None
    elements = ROWS * COLS
    time_ns = sum(to_float(row, "gpu__time_duration.sum") for row in rows)
    l2_bytes = sum(to_float(row, "lts__t_bytes.sum") for row in rows)
    dram_read = sum(to_float(row, "dram__bytes_op_read.sum") for row in rows)
    dram_write = sum(to_float(row, "dram__bytes_op_write.sum") for row in rows)
    modeled_bytes = elements * 12.0
    return {
        "variant": variant,
        "cols": COLS,
        "rows": ROWS,
        "profiled_launches": len(rows),
        "time_us": time_ns / 1.0e3,
        "modeled_gbps": modeled_bytes / (time_ns / 1.0e9) / 1.0e9,
        "l2_bytes_per_element": l2_bytes / elements,
        "dram_bytes_per_element": (dram_read + dram_write) / elements,
        "dram_read_bytes_per_element": dram_read / elements,
        "dram_write_bytes_per_element": dram_write / elements,
    }


def main() -> int:
    cases = [
        ("block", RESULTS / "ncu_minmax_norm_block.csv"),
        ("cluster", RESULTS / "ncu_minmax_norm_cluster.csv"),
    ]
    rows = []
    for variant, path in cases:
        if not path.exists():
            continue
        summary = summarize(variant, path)
        if summary is not None:
            rows.append(summary)

    out = RESULTS / "minmax_norm_ncu_summary.csv"
    fieldnames = [
        "variant",
        "cols",
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
            f"l2_B/elem={row['l2_bytes_per_element']:.3f} "
            f"dram_B/elem={row['dram_bytes_per_element']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
