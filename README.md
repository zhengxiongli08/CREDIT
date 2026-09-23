# CREDIT: Cost-guided Reduction-reuse with Efficient DSMEM Inter-CTA Tiling

Starting from Hopper architecture, NVIDIA introduced distributed shared memory (DSMEM) as a new programming hierarchy in CUDA. CREDIT is designed for deciding when DSMEM is profitable for wide, row-wise GPU workloads. For more details, please refer to our paper, [CREDIT: Cost-guided Reduction-reuse with Efficient DSMEM Inter-CTA Tiling](https://arxiv.org/abs/2609.01864).

This repository contains the code, analytical model and plotting scripts to facilitate the reproduction.

## Environment Setup
It's recommended to use the Docker environment. But you can also install it as a local environment.

### Docker Setup (Recommended)
The included Dockerfile pins CUDA 13.0.1, Python 3.11, and PyTorch 2.11.0. The host needs an NVIDIA driver compatible with CUDA 13 and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Build the environment from the repository root:

```
docker build -t credit .
```

Confirm that the container can access the GPU:

```
docker run --rm --gpus all credit nvidia-smi
```

Start an interactive container with the current repository mounted at `/workspace`:

```bash
./scripts/run_docker.sh
```

The bind mount exposes the host checkout directly inside the container. Now you have finished the setup using Dockerfile. You are free to run commands in the build and evaluation sections below interactively within the container.

### Local Setup

Please make sure you have nvidia driver that supports CUDA 13.0 installed in your local machine. You also need `nvcc` v13.0 to compile the project.

Then, install the dependencies:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install matplotlib numpy
```

## Build & Evaluate

Compile for the attached GPU and run the evaluation:

```bash
make all
python scripts/run_local.py
```

The generated `build/` and timestamped `results/` directories appear directly in the host checkout.

## Reproduce the Figures

The aggregate tables and plots can be regenerated:

```bash
python scripts/analyze_results.py \
  results/local_<rtx-timestamp> \
  results/h100_<h100-timestamp> \
  --output-dir results/reproduced
```

Generate the three camera-ready paper figures:

```bash
python scripts/plot/plot_workload_scaling.py \
  results/local_<rtx-timestamp> results/h100_<h100-timestamp>
python scripts/plot/plot_model_validation.py \
  results/local_<rtx-timestamp> results/h100_<h100-timestamp>
python scripts/plot/plot_traffic_reduction.py
```

Each command validates that exactly one bundle comes from an RTX 5090 and one from an H100, then orders them consistently in the figure.

## Repository Layout

```
.
|-- src/
|   |-- primitive_benchmark.cu    DSMEM and memory-hierarchy microbenchmarks
|   |-- cluster_control.cu        Work-free cluster overhead calibration
|   `-- workloads/                Six standalone CUDA workload implementations
|-- dsmem_eval/
|   |-- runner.py                 Correctness, timing, and CSV collection pipeline
|   |-- workloads.py              Workload definitions and PyTorch references
|   |-- triton_kernels.py         Handwritten Triton baselines
|   |-- cost_model.py             Architecture-calibrated profitability model
|   |-- profile.py                Primitive benchmark parser and calibration
|   `-- report.py                 Per-device report generation
|-- scripts/
|   |-- run_local.py              Local data-collection entry point
|   |-- run_modal.py              Modal H100 data-collection entry point
|   |-- analyze_results.py        Cross-GPU tables and report
|   |-- rescore_cost_model.py     Re-evaluate saved runs with the current model
|   |-- verify_artifact.py        Validate tracked result bundles
|   `-- plot/                     Reproducible paper-figure scripts
|-- results/
|   |-- figures/                  Generated paper figures
|   |-- local_<timestamp>/        New local evaluation bundles
|   |-- h100_<timestamp>/         New Modal H100 evaluation bundles
|   `-- comparison/               Optional derived cross-GPU tables and report
|-- figs_reference/               Camera-ready figure references
`-- Makefile                      CUDA build interface
```

## Citation
```
@article{li2026credit,
  title={CREDIT: Cost-guided Reduction-reuse with Efficient DSMEM Inter-CTA Tiling},
  author={Li, Zhengxiong and Huang, Tsung-Wei and Ogras, Umit},
  journal={arXiv preprint arXiv:2609.01864},
  year={2026}
}
```
