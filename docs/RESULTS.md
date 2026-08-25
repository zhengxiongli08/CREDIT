# Included Results

The repository includes the canonical result bundles used for the RTX 5090 and H100 comparison. No B200 measurements are included or used.

## Summary

| Device | All-point wins | 64K geomean speedup | Model decisions correct |
|---|---:|---:|---:|
| RTX 5090 | 22/30 | 1.466x | 27/30 |
| H100 | 9/30 | 1.318x | 28/30 |

"Win" means that the selected DSMEM implementation is faster than the fastest measured `torch.compile`, handwritten Triton, or non-DSMEM CUDA implementation at the same shape. The all-point count spans six workloads and five row widths from 4K through 64K. At 64K, all six workloads win on both devices.

## 64K Workload Results

| Workload | RTX 5090 | H100 |
|---|---:|---:|
| LayerNorm backward | 1.961x | 1.452x |
| Weighted variance backward | 2.401x | 1.339x |
| Pearson backward | 1.540x | 1.457x |
| Softmax-logits backward | 1.308x | 1.459x |
| LARS momentum | 1.014x | 1.045x |
| Row-wise int8 quantization | 1.029x | 1.216x |

These values are `dsmem_vs_best` from each bundle's `summary.csv`. Values above 1.0 favor DSMEM.

## Bundle Contents

Each device directory contains:

- `summary.csv`: one row per workload and row width.
- `cluster_sweep.csv`: timing and model terms for every legal cluster size.
- `cuda_raw.csv`: raw standalone CUDA trials.
- `control_raw.csv`: work-free cluster-control trials.
- `framework_raw.json`: raw PyTorch and Triton trials plus correctness diagnostics.
- `metadata.json`: device, software, protocol, and aggregate information.
- `REPORT.md`: generated per-device summary.
- `primitive_profile.json` and `primitive_stdout.txt`: included for the H100 run, where profiling and evaluation used one self-contained pipeline.
- `compile.json`: CUDA compilation commands and diagnostics for the H100 run.

The `*.peak_model.csv` files preserve the earlier cost-model outputs used for the ablation. The unqualified CSV files contain the revised model.

## Rebuilding Aggregates

Run:

```bash
python scripts/aggregate.py \
  results/rtx5090 \
  results/h100 \
  --output-dir results/reproduced
```

This produces the cross-GPU point table, width aggregates, model-accuracy table, Markdown report, PDF figure, and PNG preview. The expected compact report is available at `results/comparison/CROSS_GPU_REPORT.md`.
