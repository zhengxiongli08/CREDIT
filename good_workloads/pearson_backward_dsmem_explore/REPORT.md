# Pearson Correlation Backward DSMEM Exploration

## Workload

This folder benchmarks backward gradients for row-wise Pearson correlation
between two vectors:

```text
xc = x - mean(x)
yc = y - mean(y)
corr = sum(xc * yc) / sqrt(sum(xc * xc) + eps) / sqrt(sum(yc * yc) + eps)
dx = grad * (yc * inv_x * inv_y - xc * cov * inv_x^3 * inv_y)
dy = grad * (xc * inv_x * inv_y - yc * cov * inv_x * inv_y^3)
```

The default sweep uses `rows=4096` and
`N in {4096, 8192, 16384, 32768, 65536}`.

## Implementations

- `block_read`: one CTA per row. It reads `x` and `y` once to reduce
  `sum(x)`, `sum(y)`, `sum(x*x)`, `sum(y*y)`, and `sum(x*y)`, then rereads
  `x` and `y` to write both `dx` and `dy`.
- `block_smem`: one CTA per row with local shared-memory staging of both `x`
  and `y` when the whole row fits.
- `cluster_smem`: one 8-CTA cluster per row. Each CTA stages a row slice of
  `x` and `y`, uses DSMEM only for the five scalar reductions, then writes
  its local `dx` and `dy` output slices.
- `torch.compile`: compiled PyTorch expression for the same gradient formula.

Throughput uses a 24 B/element model for the logical no-DSMEM read path:
one `x/y` read pass for reductions, then one `x/y` reread plus two output
writes.

## Timing Results

Command:

```bash
bash pearson_backward_dsmem_explore/run_compare.sh \
  --rows 4096 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv pearson_backward_dsmem_explore/results/pearson_backward_torch_compare.csv
```

| N | block variant | block ms | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | block_smem | 0.176730 | 0.309235 | 0.185210 | 0.572x | 0.599x |
| 8192 | block_smem | 0.352109 | 0.351136 | 0.450477 | 1.003x | 1.283x |
| 16384 | block_read | 1.070270 | 0.701517 | 1.186176 | 1.526x | 1.691x |
| 32768 | block_read | 2.151710 | 1.409270 | 2.767155 | 1.527x | 1.964x |
| 65536 | block_read | 4.341060 | 2.816570 | 6.050470 | 1.541x | 2.148x |

The DSMEM cluster kernel beats `torch.compile` on 4 of 5 shapes. It is
especially favorable at wide rows because the PyTorch expression performs
several centered reductions, while the DSMEM kernel reduces five raw sums in
one input pass and reuses staged `x` and `y` for both outputs.

## Nsight Compute

Command:

```bash
bash pearson_backward_dsmem_explore/ncu_pearson_backward_metrics.sh
python3 pearson_backward_dsmem_explore/parse_pearson_backward_ncu.py
```

Profiled shape: `rows=4096`, `N=65536`.

| variant | NCU time (us) | L2 B/element | DRAM B/element | DRAM read B/element | DRAM write B/element |
|---|---:|---:|---:|---:|---:|
| block | 4362.272 | 24.002 | 23.990 | 16.000 | 7.990 |
| cluster | 2809.536 | 16.192 | 15.957 | 8.000 | 7.956 |

The DSMEM path reduces DRAM traffic from `24.0` to `16.0 B/element` and
reduces NCU kernel time by `1.553x` on the profiled shape. As in the other
two-output pairwise workloads, output traffic is unchanged; DSMEM removes the
second global read of `x` and `y`.

## Verdict

Pearson correlation backward is a strong positive DSMEM case. It extends the
pairwise-vector pattern to centered statistics: more scalar reductions do not
hurt because DSMEM only exchanges compact partials, while the element data
stays staged locally for reuse during the gradient write.

Artifacts:

- `results/pearson_backward_torch_compare.csv`
- `results/pearson_backward_ncu_summary.csv`
- `plots/pearson_backward_speedup.png`
- `plots/pearson_backward_runtime_ms.png`
- `plots/pearson_backward_gbps.png`
- `plots/pearson_backward_ncu_metrics.png`
