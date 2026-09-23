from __future__ import annotations

import io
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/workspace"

# Build the environment, copy the CUDA sources, and compile for H100.
image = (
    modal.Image.from_dockerfile(ROOT / "Dockerfile", context_dir=ROOT)
    .add_local_file(ROOT / "Makefile", f"{REMOTE_ROOT}/Makefile", copy=True)
    .add_local_dir(ROOT / "src", f"{REMOTE_ROOT}/src", copy=True)
    .run_commands(f"make -C {REMOTE_ROOT} CUDA_ARCH=sm_90 all")
    .add_local_file(
        ROOT / "scripts/run_local.py", f"{REMOTE_ROOT}/scripts/run_local.py"
    )
    .add_local_dir(ROOT / "dsmem_eval", f"{REMOTE_ROOT}/dsmem_eval")
)

app = modal.App("credit-h100-evaluation")


@app.function(image=image, gpu="H100!", cpu=4, memory=16_384, timeout=2 * 60 * 60)
def run_h100(arguments: list[str]) -> tuple[str, bytes]:
    """Run the local evaluator on Modal and return its result directory."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"h100_{timestamp}"
    result_dir = Path("/tmp") / name
    command = [
        "python",
        f"{REMOTE_ROOT}/scripts/run_local.py",
        "--output-dir",
        str(result_dir),
        *arguments,
    ]

    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=REMOTE_ROOT, check=True)

    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        archive.add(result_dir, arcname=name)
    return name, payload.getvalue()


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
    if gpu.lower() != "h100":
        raise ValueError("--gpu must be h100")

    arguments = [
        "--workloads", workloads,
        "--n-values", n_values,
        "--cluster-sizes", cluster_sizes,
        "--warmup", str(warmup),
        "--iterations", str(iterations),
        "--trials", str(trials),
    ]
    if quick:
        arguments.append("--quick")
    if profile_only:
        arguments.append("--profile-only")

    name, archive_bytes = run_h100.remote(arguments)
    destination = (ROOT / output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        archive.extractall(destination, filter="data")
    print(f"Results: {destination / name}")
