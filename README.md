# CREDIT: Cost-Guided DSMEM Tiling

CREDIT is a research artifact for deciding when NVIDIA Distributed Shared Memory (DSMEM) is profitable for wide, row-wise GPU workloads. The implementation keeps bulk tensor slices in each CTA's local shared memory, exchanges only compact reduction statistics through DSMEM, and uses an independently calibrated cost model to screen candidate transformations.

This repository contains the code and measurements used for the RTX 5090 and H100 evaluation. It intentionally does not contain the manuscript, abandoned prototypes, generated binaries, or B200 measurements.

## Main Results

The included evaluation compares DSMEM kernels against the fastest of `torch.compile`, handwritten Triton, and non-DSMEM CUDA baselines.

| GPU | DSMEM wins over best baseline | Geomean speedup at N=65,536 | Cost-model accuracy |
|---|---:|---:|---:|
| NVIDIA GeForce RTX 5090 | 22/30 points | 1.466x | 27/30 (90.0%) |
| NVIDIA H100 80GB HBM3 | 9/30 points | 1.318x | 28/30 (93.3%) |

At `N=65,536`, DSMEM beats the best baseline on all six evaluated workloads on both GPUs. See [the result guide](docs/RESULTS.md) for the workload-level data and interpretation.

## Repository Layout

```text
.
|-- cuda/
|   |-- primitive_benchmark.cu    DSMEM and memory-hierarchy microbenchmarks
|   |-- cluster_control.cu        Work-free cluster overhead calibration
|   `-- workloads/                Six standalone CUDA workload implementations
|-- dsmem_eval/
|   |-- runner.py                 Build, correctness, timing, and CSV pipeline
|   |-- workloads.py              Workload definitions and PyTorch references
|   |-- triton_kernels.py         Handwritten Triton baselines
|   |-- cost_model.py             Architecture-calibrated profitability model
|   |-- profile.py                Primitive benchmark parser and calibration
|   `-- report.py                 Per-device report generation
|-- scripts/
|   |-- aggregate.py              Cross-GPU tables and plots
|   `-- rescore_cost_model.py     Re-evaluate saved runs with the current model
|-- tests/                        Cost-model unit tests
|-- results/
|   |-- rtx5090/                  Canonical RTX 5090 result bundle
|   |-- h100/                     Canonical H100 result bundle
|   `-- comparison/               Cross-GPU tables and figure
|-- run_local.py                  Local GPU entry point
`-- modal_app.py                  Reproducible Modal H100 entry point
```

## Requirements

The recorded environment used Python 3.11, CUDA 13.0, PyTorch 2.11.0 with CUDA 13.0, and Triton 3.6.0. The CUDA kernels require thread-block clusters and are intended for Hopper or newer NVIDIA GPUs. The checked configurations are H100 (`sm_90`) and RTX 5090 (`sm_120`).

Install the host-side tools:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a local GPU run, also install a CUDA-enabled PyTorch build. The exact paper environment used:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
```

`nvcc` and `nvidia-smi` must be available on `PATH`. CUDA 13.0 is recommended because it supports both evaluated architectures.

## Local Evaluation

First check compilation, allocation, and primitive profiling without running the workload suite:

```bash
python run_local.py --profile-only
```

Run an inexpensive end-to-end check:

```bash
python run_local.py --quick --workloads layernorm_backward
```

Run the complete protocol:

```bash
python run_local.py
```

By default, new bundles are written below `results/runs/`. Use `--output-dir`, `--n-values`, `--cluster-sizes`, `--warmup`, `--iterations`, or `--trials` to override the protocol. Do not mix protocols when constructing one aggregate comparison.

## H100 on Modal

Authenticate once with `modal setup`, then use the same staged sequence:

```bash
modal run modal_app.py --gpu h100 --profile-only
modal run modal_app.py --gpu h100 --quick --workloads layernorm_backward
modal run modal_app.py --gpu h100
```

The Modal image pins the CUDA and PyTorch versions, requests an exact `H100!`, compiles the CUDA sources inside the allocation, and downloads a timestamped result bundle into `results/`.

## Reproduce the Cross-GPU Summary

The aggregate tables and plots can be regenerated without a GPU:

```bash
python scripts/aggregate.py \
  results/rtx5090 \
  results/h100 \
  --output-dir results/reproduced
```

Run the cost-model unit tests with:

```bash
python -m unittest discover -s tests -v
python scripts/verify_artifact.py
```

## Measurement Scope

Each timing point uses 20 warmup launches, 100 timed launches per trial, and five trials. Compilation and Triton configuration search are excluded. Framework outputs are checked against eager PyTorch; the standalone CUDA binaries also perform their own correctness checks. The cost model consumes one same-shape non-DSMEM CUDA timing, independent primitive measurements, and static workload traffic counts. It does not consume a DSMEM workload timing.

The kernels are research implementations for reproducing the study, not a production operator library. Performance depends on GPU architecture, clocks, software versions, and launch configuration; reruns should retain the generated metadata and raw trial files.
