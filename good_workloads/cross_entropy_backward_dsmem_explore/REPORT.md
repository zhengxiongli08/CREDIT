# DSMEM for Cross-Entropy Backward from Logits

## Goal

This experiment tests whether DSMEM helps the row-wise gradient of cross entropy
with respect to logits:

```text
p  = softmax(logits)
dx = p - one_hot(target)
```

The default external baseline is `torch.compile` running the equivalent PyTorch
expression. A non-cluster CUDA block kernel is kept as an internal noDSMEM
baseline.

## Why This Workload Is a DSMEM Candidate

Cross-entropy backward from logits needs row max and denominator reductions,
then writes one gradient per input element. A simple block implementation reads
the logits once for the max, again for the denominator, and again to write
`dx`.

The DSMEM cluster implementation splits one row across an 8-CTA cluster. Each
CTA stages a contiguous logits slice in local shared memory, computes local max
and sum partials, uses DSMEM only for cross-CTA scalar reductions, and writes
its own `dx` slice. Target-label traffic is one scalar per row and is ignored in
the per-element bandwidth model.

## Files

Folder: `cross_entropy_backward_dsmem_explore`

- `cross_entropy_backward_bench.cu`: CUDA block and DSMEM cluster kernels.
- `run_cross_entropy_backward.sh`: builds and runs the CUDA benchmark.
- `compare_cross_entropy_backward_torch.py`: compares CUDA results against
  `torch.compile`.
- `run_compare.sh`: wrapper using the `cluster` conda environment.
- `ncu_cross_entropy_backward_metrics.sh`: focused Nsight Compute profile for
  `N=65536`.
- `parse_cross_entropy_backward_ncu.py`: converts raw NCU CSV into a compact
  summary.
- `plot_cross_entropy_backward_results.py`: generates timing and profiler
  plots.

The CUDA benchmark performs sample-based correctness checks by default,
including target columns. The NCU script passes `--no-verify` so profiler
captures only the target kernels.

## Timing Method

Command:

```bash
bash cross_entropy_backward_dsmem_explore/run_compare.sh \
  --rows 4096 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv cross_entropy_backward_dsmem_explore/results/cross_entropy_backward_torch_compare.csv
```

Environment:

- GPU: NVIDIA GeForce RTX 5090
- PyTorch: `2.11.0+cu130`
- CUDA arch: `sm_120`
- rows: `4096`
- cluster size: `8`
- block candidates: `block_read` and `block_smem`

The bandwidth model counts the natural noDSMEM traffic: read logits for max,
read logits for denominator, reread logits for output, then write `dx`, or
`16 bytes/element`. Since all variants use the same model, speedup is
equivalent to runtime speedup.

## Timing Results

| N | block ms | block variant | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---:|---|---:|---:|---:|---:|
| 4096 | 0.0704 | block_read | 0.3267 | 0.0767 | 0.215x | 0.235x |
| 8192 | 0.1771 | block_smem | 0.3079 | 0.1853 | 0.575x | 0.602x |
| 16384 | 0.3516 | block_smem | 0.3783 | 0.3599 | 0.929x | 0.951x |
| 32768 | 1.3671 | block_read | 0.6955 | 0.7185 | 1.966x | 1.033x |
| 65536 | 2.7732 | block_read | 1.4037 | 1.9861 | 1.976x | 1.415x |

DSMEM beats `torch.compile` on 2 of 5 shapes. The strongest measured case is
`N=65536`:

```text
torch.compile: 1.9861 ms
DSMEM cluster: 1.4037 ms
speedup:       1.415x
```

## NCU Metrics

Command:

```bash
bash cross_entropy_backward_dsmem_explore/ncu_cross_entropy_backward_metrics.sh
python3 cross_entropy_backward_dsmem_explore/parse_cross_entropy_backward_ncu.py
```

Profiled case: `rows=4096`, `N=65536`, one captured launch per variant.

| variant | time us | modeled GB/s | L2 B/elem | DRAM B/elem | DRAM read B/elem | DRAM write B/elem |
|---|---:|---:|---:|---:|---:|---:|
| block | 2780.352 | 1544.757 | 16.003 | 15.979 | 11.989 | 3.990 |
| cluster | 1385.696 | 3099.502 | 8.226 | 7.983 | 4.000 | 3.983 |

The profiler confirms the intended traffic reduction. The noDSMEM block path
uses about `16.0 B/element` of DRAM traffic, while the DSMEM path stages logits
once and uses about `8.0 B/element`. Read traffic drops from about
`12.0 B/element` to `4.0 B/element`; the `dx` write is unchanged.

## Interpretation

This is a positive but thresholded DSMEM case. The kernel has the profitable
shape: row reductions followed by a full-row output that reuses the staged
input. NCU shows the memory-system benefit clearly, and at `N=32768` and
`N=65536` the cluster path beats both the noDSMEM block path and
`torch.compile`.

For smaller rows, cluster launch and synchronization overhead outweigh the
saved rereads. The one-CTA `block_smem` variant can stage the whole row at
`N=8192` and `N=16384`, so DSMEM has no capacity advantage there. This makes
cross-entropy backward a useful contrast with scalar cross-entropy forward:
the forward loss improved the custom block path but did not beat PyTorch,
whereas the backward gradient has enough full-row reuse to cross over on wide
rows.
