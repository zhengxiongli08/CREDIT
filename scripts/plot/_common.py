from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


WORKLOAD_ORDER = (
    "layernorm_backward",
    "weighted_var_backward",
    "pearson_backward",
    "softmax_logits_backward",
    "lars_momentum",
    "rowwise_quant",
)
WORKLOAD_SHORT = {
    "layernorm_backward": "LayerNorm",
    "weighted_var_backward": "WVar",
    "pearson_backward": "Pearson",
    "softmax_logits_backward": "SM-logits",
    "lars_momentum": "LARS",
    "rowwise_quant": "Quant",
}
DEVICE_COLORS = {"RTX 5090": "#5F5F5F", "H100": "#0072B2"}
DEVICE_MARKERS = {"RTX 5090": "o", "H100": "s"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def device_label(metadata: dict[str, Any]) -> str:
    requested = str(metadata.get("requested_gpu", ""))
    device = str(metadata.get("device", "unknown"))
    if "H100" in requested or "H100" in device:
        return "H100"
    if "5090" in device:
        return "RTX 5090"
    return device


def load_run(path: Path) -> tuple[str, list[dict[str, str]]]:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    return device_label(metadata), read_csv(path / "summary.csv")


def load_paper_runs(paths: list[Path]) -> list[tuple[str, list[dict[str, str]]]]:
    if len(paths) != 2:
        raise ValueError("provide exactly two result bundles: RTX 5090 and H100")
    by_device = {device: rows for device, rows in map(load_run, paths)}
    expected = ("RTX 5090", "H100")
    if set(by_device) != set(expected):
        raise ValueError(
            f"expected one RTX 5090 and one H100 bundle, got {sorted(by_device)}"
        )
    return [(device, by_device[device]) for device in expected]


def configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.linewidth": 0.8,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
        }
    )


def save_figure(figure: Any, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
