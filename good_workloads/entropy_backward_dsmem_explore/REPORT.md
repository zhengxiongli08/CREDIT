# DSMEM for Entropy Backward from Logits

## Goal

This experiment tests whether DSMEM helps the row-wise gradient of softmax
entropy with respect to logits:

```text
p      = softmax(logits)
H      = -sum(p * log(p))
dx     = -p * (log(p) + H)
```

Equivalently, using shifted logits `s = logits - max(logits)`:

```text
mean_s = sum(p * s)
dx     = p * (mean_s - s)
```

The default external baseline is `torch.compile` running the equivalent PyTorch
expression. A non-cluster CUDA block kernel is kept as an internal noDSMEM
baseline.

## Why This Workload Is a DSMEM Candidate

Entropy backward from logits needs row max, softmax denominator, and weighted
shifted-logit reductions, then writes one gradient per element. A simple block
implementation reads logits for max, denominator, weighted numerator, and
output.

The DSMEM cluster implementation splits one row across an 8-CTA cluster. Each
CTA stages a contiguous logits slice in local shared memory, computes local max,
sum, and weighted-shift partials, uses DSMEM only for cross-CTA scalar
reductions, and writes its own `dx` slice.

## Files

Folder: `entropy_backward_dsmem_explore`

- `entropy_backward_bench.cu`: CUDA block and DSMEM cluster kernels.
- `run_entropy_backward.sh`: builds and runs the CUDA benchmark.
- `compare_entropy_backward_torch.py`: compares CUDA results against
  `torch.compile`.
- `run_compare.sh`: wrapper using the `cluster` conda environment.
- `ncu_entropy_backward_metrics.sh`: focused Nsight Compute profile for
  `N=65536`.
- `parse_entropy_backward_ncu.py`: converts raw NCU CSV into a compact summary.
- `plot_entropy_backward_results.py`: generates timing and profiler plots.

The CUDA benchmark performs sample-based correctness checks by default. The NCU
script passes `--no-verify` so profiler captures only the target kernels.

## Timing Method

Command:

```bash
bash entropy_backward_dsmem_explore/run_compare.sh \
  --rows 4096 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv entropy_backward_dsmem_explore/results/entropy_backward_torch_compare.csv
```

Environment:

- GPU: NVIDIA GeForce RTX 5090
- PyTorch: `2.11.0+cu130`
- CUDA arch: `sm_120`
- rows: `4096`
- cluster size: `8`
- block candidates: `block_read` and `block_smem`

The bandwidth model counts the natural noDSMEM traffic: four logits reads and
one `dx` write, or `20 bytes/element`. Since all variants use the same model,
speedup is equivalent to runtime speedup.

## Timing Results

| N | block ms | block variant | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---:|---|---:|---:|---:|---:|
| 4096 | 0.0735 | block_smem | 0.3857 | 0.0746 | 0.191x | 0.193x |
| 8192 | 0.1780 | block_smem | 0.3657 | 0.1824 | 0.487x | 0.499x |
| 16384 | 0.3509 | block_smem | 0.4216 | 0.3574 | 0.832x | 0.848x |
| 32768 | 1.6863 | block_read | 0.6959 | 0.7171 | 2.423x | 1.030x |
| 65536 | 3.4315 | block_read | 1.4011 | 2.5646 | 2.449x | 1.830x |

DSMEM beats `torch.compile` on 2 of 5 shapes. The strongest measured case is
`N=65536`:

```text
torch.compile: 2.5646 ms
DSMEM cluster: 1.4011 ms
speedup:       1.830x
```

## NCU Metrics

Command:

```bash
bash entropy_backward_dsmem_explore/ncu_entropy_backward_metrics.sh
python3 entropy_backward_dsmem_explore/parse_entropy_backward_ncu.py
```

Profiled case: `rows=4096`, `N=65536`, one captured launch per variant.

| variant | time us | modeled GB/s | L2 B/elem | DRAM B/elem | DRAM read B/elem | DRAM write B/elem |
|---|---:|---:|---:|---:|---:|---:|
| block | 3445.920 | 1557.990 | 20.004 | 19.969 | 15.978 | 3.991 |
| cluster | 1390.240 | 3861.714 | 8.252 | 7.943 | 4.000 | 3.943 |

The profiler confirms the intended traffic reduction. The noDSMEM block path
uses about `20.0 B/element` of DRAM traffic, while the DSMEM path stages logits
once and uses about `8.0 B/element`. Read traffic drops from about
`16.0 B/element` to `4.0 B/element`; the `dx` write is unchanged.

## Interpretation

Entropy backward is a positive softmax-derived case with one more reduction
than cross-entropy backward. The extra weighted-shift reduction increases input
reuse: the block-read path rereads logits for the numerator and output phases,
while DSMEM keeps each CTA's row slice local across all phases.

For smaller rows, the one-CTA `block_smem` variant can stage the whole row and
the cluster path is not worthwhile. Once rows are too wide for whole-row local
staging, the DSMEM path beats both the noDSMEM block-read baseline and
`torch.compile`.
