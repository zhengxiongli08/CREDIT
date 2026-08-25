# H100 DSMEM Evaluation on Modal

This directory contains the self-contained H100 version of the RTX 5090 paper evaluation. It does not import from or write into `good_workloads`. The six validated CUDA kernels are snapshotted under `cuda/workloads`, while the PyTorch expressions, handwritten Triton kernels, calibrated cost model, runner, and reporting code live under `dsmem_eval`. B200 support remains in the runner but is outside the current paper scope.

The full run compares four implementations under one protocol:

- `torch.compile` with full-graph, static-shape TorchInductor and `max-autotune-no-cudagraphs`.
- Handwritten streaming Triton kernels tuned over `(BLOCK_SIZE, num_warps)` configurations `(256,4)`, `(512,4)`, and `(1024,8)`.
- The best applicable non-DSMEM CUDA path in each standalone workload binary.
- DSMEM CUDA with legal cluster sizes selected from 2, 4, and 8.

Before workload timing, each allocation runs the portable primitive benchmark. The revised model combines local-SMEM and DSMEM rates with one same-shape non-DSMEM CUDA timing and a work-free cluster control. It never consumes a DSMEM workload time or fits a coefficient to those timings.

## Directory Layout

```text
modal_dsmem_evaluation/
├── modal_app.py                 Modal image, H100/B200 functions, local CLI
├── dsmem_eval/
│   ├── runner.py                Compilation, correctness, timing, and CSV pipeline
│   ├── workloads.py             Workload metadata and PyTorch expressions
│   ├── triton_kernels.py        Handwritten Triton baselines
│   ├── profile.py               Primitive-output parser and model calibration
│   ├── cost_model.py            Architecture-calibrated screening model
│   └── report.py                Per-device report generation
├── cuda/
│   ├── primitive_benchmark.cu   Memory hierarchy and DSMEM microbenchmark
│   ├── cluster_control.cu       Work-free cluster-control calibration
│   └── workloads/               Six standalone workload snapshots
├── scripts/aggregate.py         RTX 5090/H100 CSV, report, and paper figure
├── SOURCE_MANIFEST.json         Snapshot provenance and source hashes
└── results/                     Downloaded result bundles; ignored by Git
```

## One-Time Setup

From this directory:

```bash
cd /home/zli2793/projects/fuser/modal_dsmem_evaluation
conda create -n modal-dsmem python=3.11 -y
conda activate modal-dsmem
python -m pip install -r requirements.txt
modal setup
```

The remote image is pinned to CUDA 13.0.1 and PyTorch 2.11.0 with the CUDA 13.0 wheel. The runner detects compute capability and compiles the CUDA sources as `sm_90` on H100 and `sm_100` on B200. `H100!` is used deliberately so Modal does not substitute an H200; `B200` requests an exact B200.

## Recommended Run Order

First verify image construction, allocation, compilation, and primitive profiling without running the workload suite:

```bash
modal run modal_app.py --gpu h100 --profile-only
modal run modal_app.py --gpu b200 --profile-only
```

Then run one inexpensive end-to-end smoke test per architecture:

```bash
modal run modal_app.py --gpu h100 --quick --workloads layernorm_backward
modal run modal_app.py --gpu b200 --quick --workloads layernorm_backward
```

Run the full directly comparable paper sweep:

```bash
modal run modal_app.py --gpu h100
modal run modal_app.py --gpu b200
```

The two jobs can also be requested sequentially from one command:

```bash
modal run modal_app.py --gpu all
```

Separate commands are preferable while debugging because a failure or capacity delay on one GPU does not hold up the other result.

## Enriching the Width Sweep

The default widths exactly match the RTX 5090 paper data: 4K, 8K, 16K, 32K, and 64K. H100 and B200 have enough memory to add wider rows. A useful first extension is 128K:

```bash
modal run modal_app.py \
  --gpu h100 \
  --n-values 4096,8192,16384,32768,65536,131072

modal run modal_app.py \
  --gpu b200 \
  --n-values 4096,8192,16384,32768,65536,131072
```

Run selected workloads with a comma-separated list:

```bash
modal run modal_app.py \
  --gpu b200 \
  --workloads layernorm_backward,weighted_var_backward,softmax_logits_backward
```

Available workload names are:

```text
layernorm_backward
weighted_var_backward
pearson_backward
softmax_logits_backward
lars_momentum
rowwise_quant
```

You can override the measurement protocol when doing sensitivity studies:

```bash
modal run modal_app.py \
  --gpu h100 \
  --cluster-sizes 2,4,8 \
  --warmup 20 \
  --iterations 100 \
  --trials 5
```

Do not mix different warmup/iteration/trial settings in the primary cross-GPU figure.

## Result Bundle

Each command downloads a timestamped directory such as `results/h100_20260713T180000Z` containing:

```text
REPORT.md                 concise performance and crossover summary
summary.csv               one row per workload and width
cluster_sweep.csv         timing and model terms for every legal P
cuda_raw.csv              five raw CUDA trials per configuration
control_raw.csv           work-free cluster-control trials
framework_raw.json        torch.compile/Triton trials and correctness diagnostics
primitive_profile.json    memory/DSMEM measurements and calibrated model profile
primitive_stdout.txt      unmodified primitive benchmark output
compile.json              nvcc commands, architecture, timings, and diagnostics
metadata.json             software, device, clocks, protocol, and aggregate results
failure.json              traceback when a remote run returns partial results
```

CUDA binaries and TorchInductor caches are not downloaded. Compilation and Triton configuration search are excluded from reported kernel times.

## Cross-GPU Paper Artifacts

After obtaining the H100 directory, combine it with the existing RTX 5090 bundle:

```bash
python scripts/aggregate.py \
  ../evaluation/results/rtx5090 \
  results/h100_YYYYMMDDTHHMMSSZ \
  --output-dir results/comparison
```

This writes:

- `cross_gpu_points.csv`: every workload-size measurement with a device column.
- `cross_gpu_aggregate.csv`: wins and geometric-mean speedup by device and width.
- `cross_gpu_model.csv`: cost-model classification accuracy by device.
- `CROSS_GPU_REPORT.md`: compact tables for drafting Section 4.
- `cross_gpu_summary.pdf`: a paper-ready three-panel vector figure.
- `cross_gpu_summary.png`: a preview of the same figure.

The model comparison is against non-DSMEM CUDA because it tests mechanism selection. Practical DSMEM speedup is separately computed against the fastest of `torch.compile`, handwritten Triton, and non-DSMEM CUDA.

Existing RTX 5090 and H100 bundles can be rescored without rerunning a GPU. The command preserves the original peak-bandwidth model columns in `*.peak_model.csv` backups:

```bash
python scripts/rescore_cost_model.py \
  ../evaluation/results/rtx5090 \
  results/h100_YYYYMMDDTHHMMSSZ \
  --write
```

## Measurement Practice

For paper numbers, run the complete command at least three times on independent Modal allocations and retain every timestamped bundle. Report the median of run-level medians or show allocation-to-allocation variation. Modal does not expose privileged clock locking here, so `metadata.json` records `nvidia-smi` clock, power, and temperature information before and after each run.

The package does not run Nsight Compute on Modal. The timing, primitive, and model results are portable, while the existing RTX 5090 Nsight Compute traffic experiment remains the mechanism-level traffic evidence unless profiler access is separately validated on Modal.

Modal currently documents exact `H100!` and `B200` requests in its [GPU guide](https://modal.com/docs/guide/gpu), and its [CUDA guide](https://modal.com/docs/guide/cuda) recommends NVIDIA development images when `nvcc` is required.
