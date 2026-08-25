# DSMEM for LayerNorm Affine Full Backward Partials

## Goal

This experiment tests whether DSMEM helps a fuller affine LayerNorm backward
surface that writes per-element partial parameter gradients in addition to
`dx`:

```text
dyg             = dy * gamma
xhat            = (x - mean(x)) * inv_std(x)
dx              = (dyg - mean(dyg) - xhat * mean(dyg * xhat)) * inv_std(x)
dgamma_partial  = dy * xhat
dbeta_partial   = dy
```

The default external baseline is `torch.compile` running the equivalent PyTorch
expression. A non-cluster CUDA block kernel is kept as an internal noDSMEM
baseline.

## Why This Workload Is a DSMEM Candidate

This workload keeps the LayerNorm backward reductions, but adds two full-row
outputs. It needs row statistics for `x`, row statistics for `dy * gamma`, then
reuses `x`, `dy`, and `gamma` to write `dx`, `dgamma_partial`, and
`dbeta_partial`.

The DSMEM cluster implementation splits one row across an 8-CTA cluster. Each
CTA stages contiguous `x`, `dy`, and `gamma` slices in local shared memory,
computes local scalar partials, uses DSMEM only for cross-CTA reductions, and
writes its own output slices. The one-CTA block-staged baseline also stages all
three streams when the row fits in local shared memory.

## Files

Folder: `layernorm_affine_full_backward_dsmem_explore`

- `layernorm_affine_full_backward_bench.cu`: CUDA block and DSMEM cluster
  kernels.
- `run_layernorm_affine_full_backward.sh`: builds and runs the CUDA benchmark.
- `compare_layernorm_affine_full_backward_torch.py`: compares CUDA results
  against `torch.compile`.
- `run_compare.sh`: wrapper using the `cluster` conda environment.
- `ncu_layernorm_affine_full_backward_metrics.sh`: focused Nsight Compute
  profile for `N=65536`.
- `parse_layernorm_affine_full_backward_ncu.py`: converts raw NCU CSV into a
  compact summary.
- `plot_layernorm_affine_full_backward_results.py`: generates timing and
  profiler plots.

The CUDA benchmark performs sample-based correctness checks for all three
outputs by default. The NCU script passes `--no-verify` so profiler captures
only the target kernels.

## Timing Method

Command:

```bash
bash layernorm_affine_full_backward_dsmem_explore/run_compare.sh \
  --rows 2048 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv layernorm_affine_full_backward_dsmem_explore/results/layernorm_affine_full_backward_torch_compare.csv
```

Environment:

- GPU: NVIDIA GeForce RTX 5090
- PyTorch: `2.11.0+cu130`
- CUDA arch: `sm_120`
- rows: `2048`
- cluster size: `8`
- block candidates: `block_read` and `block_smem`

The modeled bandwidth denominator counts the natural noDSMEM logical traffic:
three input streams reused across statistics and output phases plus three output
writes, or `40 bytes/element`. Since all variants use the same denominator,
speedup is equivalent to runtime speedup.

## Timing Results

| N | block ms | block variant | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---:|---|---:|---:|---:|---:|
| 4096 | 0.1134 | block_smem | 0.1953 | 0.1163 | 0.581x | 0.596x |
| 8192 | 0.2272 | block_smem | 0.2184 | 0.2456 | 1.040x | 1.124x |
| 16384 | 0.7202 | block_read | 0.4476 | 0.7874 | 1.609x | 1.759x |
| 32768 | 1.4742 | block_read | 0.8939 | 1.6810 | 1.649x | 1.881x |
| 65536 | 2.9801 | block_read | 1.8047 | 3.4174 | 1.651x | 1.894x |

DSMEM beats `torch.compile` on 4 of 5 shapes. The strongest measured case is
`N=65536`:

```text
torch.compile: 3.4174 ms
DSMEM cluster: 1.8047 ms
speedup:       1.894x
```

## NCU Metrics

Command:

```bash
bash layernorm_affine_full_backward_dsmem_explore/ncu_layernorm_affine_full_backward_metrics.sh
python3 layernorm_affine_full_backward_dsmem_explore/parse_layernorm_affine_full_backward_ncu.py
```

Profiled case: `rows=2048`, `N=65536`, one captured launch per variant.

| variant | time us | modeled GB/s | L2 B/elem | DRAM B/elem | DRAM read B/elem | DRAM write B/elem |
|---|---:|---:|---:|---:|---:|---:|
| block | 2982.208 | 1800.246 | 37.755 | 31.973 | 20.005 | 11.969 |
| cluster | 1784.640 | 3008.287 | 24.190 | 19.953 | 8.001 | 11.952 |

The profiler confirms that DSMEM reduces input rereads while output traffic
dominates more than in dx-only LayerNorm backward. DRAM read traffic drops from
about `20.0 B/element` to `8.0 B/element`; the three output writes stay near
`12.0 B/element`. Total DRAM falls from about `32.0` to `20.0 B/element`.

## Interpretation

This is a positive output-heavy normalization case. Adding
`dgamma_partial` and `dbeta_partial` makes the write traffic unavoidable, so the
traffic reduction is smaller than in dx-only LayerNorm backward. Even so, DSMEM
still wins once one CTA can no longer stage all three streams locally.

The crossover is visible at `N=8192`: the local block-staged path still fits and
is roughly tied with DSMEM. From `N=16384` onward, the one-CTA staged variant is
not viable, and the DSMEM cluster path beats both the block-read CUDA baseline
and `torch.compile`.
