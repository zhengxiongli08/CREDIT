#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AVAILABLE_WORKLOADS = (
    "layernorm_backward",
    "weighted_var_backward",
    "pearson_backward",
    "softmax_logits_backward",
    "lars_momentum",
    "rowwise_quant",
)
DEFAULT_N_VALUES = (4096, 8192, 16384, 32768, 65536)
DEFAULT_CLUSTER_SIZES = (2, 4, 8)


def comma_separated_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of integers"
        ) from error
    if not values:
        raise argparse.ArgumentTypeError("the list cannot be empty")
    return values


def workload_names(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return AVAILABLE_WORKLOADS
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(values) - set(AVAILABLE_WORKLOADS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown workload(s): {', '.join(unknown)}"
        )
    if not values:
        raise argparse.ArgumentTypeError("at least one workload is required")
    return values


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description="Run the CREDIT DSMEM evaluation on the local NVIDIA GPU."
    )
    parser.add_argument(
        "--workloads", type=workload_names, default=AVAILABLE_WORKLOADS
    )
    parser.add_argument(
        "--n-values",
        type=comma_separated_ints,
        default=DEFAULT_N_VALUES,
        help="comma-separated row widths",
    )
    parser.add_argument(
        "--cluster-sizes",
        type=comma_separated_ints,
        default=DEFAULT_CLUSTER_SIZES,
        help="comma-separated cluster sizes selected from 2, 4, and 8",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "runs" / f"local_{timestamp}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from dsmem_eval.runner import run_suite

    config = {
        "workloads": args.workloads,
        "n_values": args.n_values,
        "cluster_sizes": args.cluster_sizes,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "trials": args.trials,
        "quick": args.quick,
        "profile_only": args.profile_only,
    }
    output_dir = args.output_dir.resolve()
    metadata = run_suite(config, output_dir, "local")
    print(f"Status: {metadata['status']}")
    print(f"Results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
