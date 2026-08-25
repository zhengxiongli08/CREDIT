# DSMEM Paper Evaluation

This directory contains the shared evaluation framework used for the RTX 5090 results in Section 4. It compares four implementation classes under one protocol:

- `torch.compile` with full-graph, static-shape, max-autotune TorchInductor compilation and CUDA graphs disabled.
- Handwritten streaming Triton kernels, autotuned over block sizes and warp counts.
- The fastest applicable single-CTA CUDA implementation from each workload directory.
- DSMEM CUDA kernels swept over legal cluster sizes 2, 4, and 8.

The initial representative suite covers normalization, softmax, pairwise statistics, optimizer, and quantization workloads. New workloads are added through `workloads.py` and `triton_kernels.py`; the runner and result schema remain unchanged.

## RTX 5090 Run

Run the full benchmark in the existing `cluster` environment:

```bash
conda run -n cluster --no-capture-output \
  python evaluation/run_rtx5090.py \
  --output-dir evaluation/results/rtx5090
```

The default protocol uses 20 warmup launches, 100 timed launches per trial, and five trials. Compilation and Triton configuration search are excluded from reported times. Raw CUDA, control, and framework measurements are checkpointed before summary generation.

For a smoke test:

```bash
conda run -n cluster --no-capture-output \
  python evaluation/run_rtx5090.py \
  --quick --workloads layernorm_backward
```

Regenerate a summary from checkpoints without rerunning CUDA:

```bash
conda run -n cluster --no-capture-output \
  python evaluation/run_rtx5090.py \
  --skip-cuda --output-dir evaluation/results/rtx5090
```

## Cost Model

`cluster_control.cu` independently measures launch, scheduling, shared-memory-footprint, and barrier overhead for each `(rows, N, P)` configuration. `cost_model.py` combines that calibration with:

- One same-shape non-DSMEM CUDA timing to estimate achieved source bandwidth.
- The measured local-SMEM read/store and DSMEM-store issue rates.
- The workload's eliminated reread bytes, staged bytes, and reduction stages.
- An overlap term that charges only local-store issue time not hidden by the compulsory input stream.

The model predicts profitability relative to the selected non-DSMEM CUDA implementation. It does not use the measured DSMEM workload time as an input and has no coefficient fitted to those timings.

## Artifacts

- `results/rtx5090/summary.csv`: one row per workload and width.
- `results/rtx5090/cluster_sweep.csv`: per-cluster-size timing and model terms.
- `results/rtx5090/framework_raw.json`: raw framework timings and Triton correctness diagnostics.
- `results/rtx5090/REPORT.md`: concise aggregate findings.
- `HPEC-Draft/scripts/plot_results_summary.py`: paper-figure generator.
