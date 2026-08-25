# Row-wise Weighted Variance Normalization Backward DSMEM Exploration

## Workload

This folder benchmarks the `dx` path for row-wise weighted variance
normalization:

```text
sum_w = sum(w)
mean = sum(w * x) / sum_w
var = sum(w * (x - mean)^2) / sum_w
inv_std = rsqrt(var + eps)
sum_dy = sum(dy)
sum_dy_centered = sum(dy * (x - mean))
dx = inv_std * (
    dy
    - w * sum_dy / sum_w
    - w * (x - mean) * inv_std^2 * sum_dy_centered / sum_w
)
```

The default sweep uses `rows=4096` and
`N in {4096, 8192, 16384, 32768, 65536}`.

## Implementations

- `block_read`: one CTA per row. It reads `x,w` for the weighted mean, rereads
  `x,w` for the weighted variance, reads `x,dy` for gradient statistics, then
  rereads `x,w,dy` to write `dx`.
- `block_smem`: one CTA per row with local shared-memory staging of `x`, `w`,
  and `dy` when the full row fits.
- `cluster_smem`: one 8-CTA cluster per row. Each CTA stages local slices of
  `x`, `w`, and `dy`, uses DSMEM for the scalar weighted-statistics and
  gradient reductions, then writes its local `dx` slice.
- `torch.compile`: compiled PyTorch expression for the same three-input
  backward formula.

Throughput uses a 40 B/element model for the logical block-read path: `x,w`
for the mean, `x,w` for the variance, `x,dy` for gradient statistics, and
`x,w,dy` plus one `dx` write for output.

## Timing Results

Command:

```bash
bash weighted_var_norm_backward_dsmem_explore/run_compare.sh \
  --rows 4096 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv weighted_var_norm_backward_dsmem_explore/results/weighted_var_norm_backward_torch_compare.csv
```

| N | block variant | block ms | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | block_smem | 0.1614 | 0.4819 | 0.1773 | 0.335x | 0.368x |
| 8192 | block_smem | 0.3354 | 0.4787 | 0.5831 | 0.701x | 1.218x |
| 16384 | block_read | 1.6163 | 0.6693 | 1.5641 | 2.415x | 2.337x |
| 32768 | block_read | 3.5107 | 1.3376 | 3.2692 | 2.625x | 2.444x |
| 65536 | block_read | 7.0255 | 2.7124 | 6.6109 | 2.590x | 2.437x |

The DSMEM cluster kernel beats `torch.compile` on 4 of 5 shapes. The best
`torch.compile` speedup is `2.444x` at `N=32768`.

## Nsight Compute

Command:

```bash
bash weighted_var_norm_backward_dsmem_explore/ncu_weighted_var_norm_backward_metrics.sh
python3 weighted_var_norm_backward_dsmem_explore/parse_weighted_var_norm_backward_ncu.py
```

Profiled shape: `rows=4096`, `N=65536`.

| variant | NCU time (us) | modeled GB/s | L2 B/element | DRAM B/element | DRAM read B/element | DRAM write B/element |
|---|---:|---:|---:|---:|---:|---:|
| block | 7030.912 | 1527.173 | 40.002 | 39.957 | 35.947 | 4.010 |
| cluster | 2724.864 | 3940.534 | 16.290 | 16.011 | 12.004 | 4.007 |

The DSMEM path reduces DRAM traffic from `40.0` to `16.0 B/element` and reduces
NCU kernel time by `2.580x` on the profiled shape. Output traffic is unchanged;
DSMEM removes repeated global reads of `x`, `w`, and `dy` across the dependent
statistics and output phases.

## Verdict

Weighted variance normalization backward is a strong positive DSMEM case. It
extends the weighted forward result into a three-stream backward formula:
staging `x`, weights, and `dy` once lets DSMEM reuse them across weighted mean,
weighted variance, gradient-statistic, and output phases. The cluster path is
slower than one-CTA local shared-memory staging while the full row fits, then
crosses over sharply once rows require block-read rereads.

Artifacts:

- `results/weighted_var_norm_backward_torch_compare.csv`
- `results/weighted_var_norm_backward_ncu_summary.csv`
- `plots/weighted_var_norm_backward_speedup.png`
- `plots/weighted_var_norm_backward_runtime_ms.png`
- `plots/weighted_var_norm_backward_gbps.png`
- `plots/weighted_var_norm_backward_ncu_metrics.png`
