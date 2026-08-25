# Row-wise Affine Standardization DSMEM Exploration

## Workload

This folder benchmarks row-wise standardization with per-element affine
parameters:

```text
mean = mean(x)
var = mean(x*x) - mean*mean
y = (x - mean) * rsqrt(var + eps) * gamma + beta
```

The default sweep uses `rows=4096` and
`N in {4096, 8192, 16384, 32768, 65536}`.

## Implementations

- `block_read`: one CTA per row. It reads `x` once to reduce mean/variance,
  then rereads `x` and reads `gamma`/`beta` to write the affine output.
- `block_smem`: one CTA per row with local shared-memory staging of `x`,
  `gamma`, and `beta` when the full row fits.
- `cluster_smem`: one 8-CTA cluster per row. Each CTA stages local slices of
  `x`, `gamma`, and `beta`, uses DSMEM only for the scalar mean/variance
  reductions, then writes its local output slice.
- `torch.compile`: compiled PyTorch expression for the same three-input
  formula.

Throughput uses a 20 B/element model for the logical block-read path:
one float `x` read for statistics, one `x` reread, one `gamma` read, one
`beta` read, and one float output write.

## Timing Results

Command:

```bash
bash affine_standardize_dsmem_explore/run_compare.sh \
  --rows 4096 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv affine_standardize_dsmem_explore/results/affine_standardize_torch_compare.csv
```

| N | block variant | block ms | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | block_smem | 0.1598 | 0.2838 | 0.1634 | 0.563x | 0.576x |
| 8192 | block_smem | 0.3341 | 0.3301 | 0.3455 | 1.012x | 1.047x |
| 16384 | block_read | 0.8511 | 0.6704 | 0.6808 | 1.270x | 1.016x |
| 32768 | block_read | 1.7219 | 1.3417 | 1.3541 | 1.283x | 1.009x |
| 65536 | block_read | 3.4469 | 2.6514 | 3.2806 | 1.300x | 1.237x |

The DSMEM cluster kernel beats `torch.compile` on 4 of 5 shapes. The best
`torch.compile` speedup is `1.237x` at `N=65536`.

## Nsight Compute

Command:

```bash
bash affine_standardize_dsmem_explore/ncu_affine_standardize_metrics.sh
python3 affine_standardize_dsmem_explore/parse_affine_standardize_ncu.py
```

Profiled shape: `rows=4096`, `N=65536`.

| variant | NCU time (us) | modeled GB/s | L2 B/element | DRAM B/element | DRAM read B/element | DRAM write B/element |
|---|---:|---:|---:|---:|---:|---:|
| block | 3444.128 | 1558.801 | 20.001 | 20.001 | 16.000 | 4.001 |
| cluster | 2651.040 | 2025.133 | 16.316 | 15.991 | 12.004 | 3.988 |

The DSMEM path reduces DRAM traffic from `20.0` to `16.0 B/element` and reduces
NCU kernel time by `1.299x` on the profiled shape. Output traffic and the
single reads of `gamma` and `beta` are unchanged; DSMEM removes the second
global read of `x`.

## Verdict

Row-wise affine standardization is a positive DSMEM case for wide rows, but the
speedup is capped by output-only affine traffic. It extends the partial-reuse
story beyond the sigmoid-gated case: adding two non-reused input streams still
allows DSMEM to beat `torch.compile` once rows are wide enough, but the
best-case gain drops to about `1.24x`.

Artifacts:

- `results/affine_standardize_torch_compare.csv`
- `results/affine_standardize_ncu_summary.csv`
- `plots/affine_standardize_speedup.png`
- `plots/affine_standardize_runtime_ms.png`
- `plots/affine_standardize_gbps.png`
- `plots/affine_standardize_ncu_metrics.png`
