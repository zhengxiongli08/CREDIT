# DSMEM for Softmax Backward from Logits

## Goal

This experiment tests whether DSMEM helps the row-wise softmax backward formula
when the softmax probabilities are not precomputed:

```text
p   = softmax(logits)
dot = sum(p * dy)
dx  = p * (dy - dot)
```

The default external baseline is `torch.compile` running the equivalent PyTorch
expression. A non-cluster CUDA block kernel is kept as an internal noDSMEM
baseline.

## Why This Workload Is a DSMEM Candidate

Softmax backward from logits needs row max, denominator, and softmax-dot
reductions, then writes one gradient per element. A simple block implementation
reads logits once for the max, again for the denominator, reads logits and `dy`
for the dot product, then rereads logits and `dy` to write `dx`.

The DSMEM cluster implementation splits one row across an 8-CTA cluster. Each
CTA stages contiguous logits and `dy` slices in local shared memory, computes
local max, sum, and dot partials, uses DSMEM only for cross-CTA scalar
reductions, and writes its own `dx` slice.

## Files

Folder: `softmax_logits_backward_dsmem_explore`

- `softmax_logits_backward_bench.cu`: CUDA block and DSMEM cluster kernels.
- `run_softmax_logits_backward.sh`: builds and runs the CUDA benchmark.
- `compare_softmax_logits_backward_torch.py`: compares CUDA results against
  `torch.compile`.
- `run_compare.sh`: wrapper using the `cluster` conda environment.
- `ncu_softmax_logits_backward_metrics.sh`: focused Nsight Compute profile for
  `N=65536`.
- `parse_softmax_logits_backward_ncu.py`: converts raw NCU CSV into a compact
  summary.
- `plot_softmax_logits_backward_results.py`: generates timing and profiler
  plots.

The CUDA benchmark performs sample-based correctness checks by default. The NCU
script passes `--no-verify` so profiler captures only the target kernels.

## Timing Method

Command:

```bash
bash softmax_logits_backward_dsmem_explore/run_compare.sh \
  --rows 4096 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv softmax_logits_backward_dsmem_explore/results/softmax_logits_backward_torch_compare.csv
```

Environment:

- GPU: NVIDIA GeForce RTX 5090
- PyTorch: `2.11.0+cu130`
- CUDA arch: `sm_120`
- rows: `4096`
- cluster size: `8`
- block candidates: `block_read` and `block_smem`

The bandwidth model counts the natural noDSMEM traffic: four logits reads, two
`dy` reads, and one `dx` write, or `28 bytes/element`. Since all variants use
the same model, speedup is equivalent to runtime speedup.

## Timing Results

| N | block ms | block variant | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---:|---|---:|---:|---:|---:|
| 4096 | 0.1182 | block_smem | 0.3998 | 0.1213 | 0.296x | 0.304x |
| 8192 | 0.2542 | block_smem | 0.3881 | 0.2631 | 0.655x | 0.678x |
| 16384 | 1.0211 | block_read | 0.5017 | 0.5178 | 2.035x | 1.032x |
| 32768 | 2.3358 | block_read | 1.0199 | 1.0594 | 2.290x | 1.039x |
| 65536 | 4.7531 | block_read | 2.0080 | 3.9651 | 2.367x | 1.975x |

DSMEM beats `torch.compile` on 3 of 5 shapes. The strongest measured case is
`N=65536`:

```text
torch.compile: 3.9651 ms
DSMEM cluster: 2.0080 ms
speedup:       1.975x
```

## NCU Metrics

Command:

```bash
bash softmax_logits_backward_dsmem_explore/ncu_softmax_logits_backward_metrics.sh
python3 softmax_logits_backward_dsmem_explore/parse_softmax_logits_backward_ncu.py
```

Profiled case: `rows=4096`, `N=65536`, one captured launch per variant.

| variant | time us | modeled GB/s | L2 B/elem | DRAM B/elem | DRAM read B/elem | DRAM write B/elem |
|---|---:|---:|---:|---:|---:|---:|
| block | 4762.880 | 1578.077 | 28.005 | 28.002 | 24.002 | 4.000 |
| cluster | 2003.904 | 3750.775 | 12.183 | 11.984 | 8.000 | 3.983 |

The profiler confirms the intended traffic reduction. The noDSMEM block path
uses about `28.0 B/element` of DRAM traffic, while the DSMEM path stages logits
and `dy` once and uses about `12.0 B/element`. Read traffic drops from about
`24.0 B/element` to `8.0 B/element`; the `dx` write is unchanged.

## Interpretation

This is a strong positive softmax-family case. Compared with cross-entropy
backward from logits, the extra upstream-gradient stream and softmax-dot
reduction create more input reuse, so DSMEM removes more reread traffic. At
`N=65536`, the cluster path is `2.367x` faster than the best noDSMEM CUDA path
and `1.975x` faster than `torch.compile`.

For smaller rows, one CTA can stage the whole row and avoid some rereads without
cluster synchronization, so DSMEM only crosses over once the row is too wide for
the local-SMEM path.
