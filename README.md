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

## Build

The Makefile is the only compilation interface. Build everything for the local GPU with:

```bash
make all
```

The narrower targets are `make benchmark`, `make workloads`, and individual workload names such as `make layernorm_backward`. Binaries are written directly to `build/`; `make clean` removes them. Set `CUDA_ARCH=sm_90` or another explicit architecture when cross-compiling.

## Local Evaluation

After `make all`, check allocation and primitive profiling without running the workload suite:

```bash
python scripts/run_local.py --profile-only
```

Run an inexpensive end-to-end check:

```bash
python scripts/run_local.py --quick --workloads layernorm_backward
```

Run the complete protocol:

```bash
python scripts/run_local.py
```

By default, each new bundle is written directly to `results/local_<timestamp>/`. Use `--output-dir`, `--n-values`, `--cluster-sizes`, `--warmup`, `--iterations`, or `--trials` to override the protocol. Do not mix protocols when constructing one aggregate comparison.

## H100 on Modal

Authenticate once with `modal setup`, then use the same staged sequence:

```bash
modal run scripts/run_modal.py --gpu h100 --profile-only
modal run scripts/run_modal.py --gpu h100 --quick --workloads layernorm_backward
modal run scripts/run_modal.py --gpu h100
```

The Modal image pins the CUDA and PyTorch versions, builds the Makefile targets for `sm_90`, requests an exact `H100!`, and downloads a timestamped result bundle into `results/`.

## Reproduce the Cross-GPU Summary

The aggregate tables and plots can be regenerated without a GPU:

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

The two data-driven plotting commands require explicit result paths; they never guess which timestamps to use. Each command validates that exactly one bundle comes from an RTX 5090 and one from an H100, then orders them consistently in the figure. This prevents an incomplete, quick, or newer unrelated run from silently replacing the intended paper data.

`plot_traffic_reduction.py` uses the paper's static traffic accounting and does not read an evaluation bundle. Each plotting script writes one PDF file to `results/figures/` by default. The scripts reproduce the layout, labels, colors, markers, and dimensions of the corresponding PDFs in `figs_reference/`.

Verify the packaged result files with:

```bash
python scripts/verify_artifact.py \
  results/local_<rtx-timestamp> results/h100_<h100-timestamp>
```

## Measurement Scope

Each timing point uses 20 warmup launches, 100 timed launches per trial, and five trials. Compilation and Triton configuration search are excluded. Framework outputs are checked against eager PyTorch; the standalone CUDA binaries also perform their own correctness checks. The cost model consumes one same-shape non-DSMEM CUDA timing, independent primitive measurements, and static workload traffic counts. It does not consume a DSMEM workload timing.

The kernels are research implementations for reproducing the study, not a production operator library. Performance depends on GPU architecture, clocks, software versions, and launch configuration; reruns should retain the generated metadata and raw trial files.
