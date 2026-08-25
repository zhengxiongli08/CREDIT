from __future__ import annotations

import io
import json
import tarfile
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


HERE = Path(__file__).resolve().parent
REMOTE_ROOT = "/opt/credit"
CUDA_IMAGE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"
TORCH_VERSION = "2.11.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
    .entrypoint([])
    .apt_install("build-essential")
    .pip_install("numpy==2.2.6")
    .pip_install(f"torch=={TORCH_VERSION}", index_url=TORCH_INDEX)
    .env(
        {
            "PYTHONPATH": REMOTE_ROOT,
            "CUDA_MODULE_LOADING": "EAGER",
            "TORCHINDUCTOR_CACHE_DIR": "/tmp/torchinductor",
        }
    )
    .add_local_dir(
        HERE / "dsmem_eval",
        f"{REMOTE_ROOT}/dsmem_eval",
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_dir(HERE / "cuda", f"{REMOTE_ROOT}/cuda")
)

app = modal.App("credit-h100-evaluation")


def _archive_directory(directory: Path) -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_dir() or "build" in path.relative_to(directory).parts:
                continue
            archive.add(path, arcname=path.relative_to(directory))
    return payload.getvalue()


def _remote_run(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    from dsmem_eval.runner import run_suite

    device_name = torch.cuda.get_device_name(0)
    if "H100" not in device_name:
        raise RuntimeError(
            f"Modal allocated {device_name!r}; refusing to record it as H100"
        )

    with tempfile.TemporaryDirectory(prefix="dsmem-eval-") as temporary:
        output_dir = Path(temporary) / "run"
        output_dir.mkdir(parents=True)
        try:
            metadata = run_suite(config, output_dir, "H100!")
        except Exception as error:
            failure = {
                "status": "failed",
                "requested_gpu": "H100!",
                "device": device_name,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            (output_dir / "failure.json").write_text(
                json.dumps(failure, indent=2) + "\n", encoding="utf-8"
            )
            metadata = failure

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = f"h100_{timestamp}"
        return {
            "slug": slug,
            "metadata": metadata,
            "archive": _archive_directory(output_dir),
        }


@app.function(
    image=image,
    gpu="H100!",
    cpu=4,
    memory=16_384,
    timeout=2 * 60 * 60,
)
def run_h100(config: dict[str, Any]) -> dict[str, Any]:
    return _remote_run(config)


def _parse_ints(value: str, name: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated integer list") from error
    if not parsed:
        raise ValueError(f"{name} cannot be empty")
    return parsed


def _save_result(result: dict[str, Any], output_root: Path) -> Path:
    target = output_root / str(result["slug"])
    target.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(result["archive"]), mode="r:gz") as archive:
        archive.extractall(target, filter="data")
    return target


@app.local_entrypoint()
def main(
    gpu: str = "h100",
    output_dir: str = "results",
    workloads: str = "all",
    n_values: str = "4096,8192,16384,32768,65536",
    cluster_sizes: str = "2,4,8",
    warmup: int = 20,
    iterations: int = 100,
    trials: int = 5,
    quick: bool = False,
    profile_only: bool = False,
) -> None:
    gpu = gpu.lower().strip()
    if gpu != "h100":
        raise ValueError("this release evaluates H100 only; --gpu must be h100")
    workload_names = (
        [
            "layernorm_backward",
            "weighted_var_backward",
            "pearson_backward",
            "softmax_logits_backward",
            "lars_momentum",
            "rowwise_quant",
        ]
        if workloads.strip().lower() == "all"
        else [item.strip() for item in workloads.split(",") if item.strip()]
    )
    config = {
        "workloads": workload_names,
        "n_values": _parse_ints(n_values, "n_values"),
        "cluster_sizes": _parse_ints(cluster_sizes, "cluster_sizes"),
        "warmup": warmup,
        "iterations": iterations,
        "trials": trials,
        "quick": quick,
        "profile_only": profile_only,
    }
    output_root = (HERE / output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print("Starting H100 evaluation", flush=True)
    result = run_h100.remote(config)
    target = _save_result(result, output_root)
    status = result["metadata"].get("status", "unknown")
    print(f"H100 status: {status}")
    print(f"Results: {target}")
    if status != "complete":
        print(f"Failure details: {target / 'failure.json'}")
