# Row-wise Weighted Variance Normalization DSMEM Exploration

## Workload

This folder benchmarks row-wise weighted variance normalization:

```text
sum_w = sum(w)
mean = sum(w * x) / sum_w
var = sum(w * (x - mean)^2) / sum_w
y = (x - mean) * rsqrt(var + eps)
```

The default sweep uses `rows=4096` and
`N in {4096, 8192, 16384, 32768, 65536}`.

## Implementations

- `block_read`: one CTA per row. It reads `x` and `w` to reduce the weighted
  mean, rereads both for the weighted variance, then rereads `x` to write the
  normalized output.
- `block_smem`: one CTA per row with local shared-memory staging of `x` and
  `w` when the full row fits.
- `cluster_smem`: one 8-CTA cluster per row. Each CTA stages local slices of
  `x` and `w`, uses DSMEM for the scalar weighted-mean and weighted-variance
  reductions, then writes its local output slice.
- `torch.compile`: compiled PyTorch expression for the same two-input formula.

Throughput uses a 24 B/element model for the logical block-read path: one
`x,w` read pair for the weighted mean, one `x,w` reread pair for the weighted
variance, one `x` reread for the output, and one float output write.

## Timing Results

Command:

```bash
bash weighted_var_norm_dsmem_explore/run_compare.sh \
  --rows 4096 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv weighted_var_norm_dsmem_explore/results/weighted_var_norm_torch_compare.csv
```

| N | block variant | block ms | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | block_smem | 0.1186 | 0.3720 | 0.1227 | 0.319x | 0.330x |
| 8192 | block_smem | 0.2538 | 0.3728 | 0.2697 | 0.681x | 0.723x |
| 16384 | block_read | 0.9629 | 0.4991 | 0.5177 | 1.929x | 1.037x |
| 32768 | block_read | 2.0270 | 1.0203 | 1.0611 | 1.987x | 1.040x |
| 65536 | block_read | 4.1178 | 2.0093 | 3.9491 | 2.049x | 1.965x |

The DSMEM cluster kernel beats `torch.compile` on 3 of 5 shapes. The best
`torch.compile` speedup is `1.965x` at `N=65536`.

## Nsight Compute

Command:

```bash
bash weighted_var_norm_dsmem_explore/ncu_weighted_var_norm_metrics.sh
python3 weighted_var_norm_dsmem_explore/parse_weighted_var_norm_ncu.py
```

Profiled shape: `rows=4096`, `N=65536`.

| variant | NCU time (us) | modeled GB/s | L2 B/element | DRAM B/element | DRAM read B/element | DRAM write B/element |
|---|---:|---:|---:|---:|---:|---:|
| block | 4102.304 | 1570.447 | 24.002 | 23.927 | 19.938 | 3.989 |
| cluster | 2012.928 | 3200.537 | 12.292 | 11.984 | 8.001 | 3.983 |

The DSMEM path reduces DRAM traffic from `23.9` to `12.0 B/element` and reduces
NCU kernel time by `2.037x` on the profiled shape. Output traffic is unchanged;
DSMEM removes the repeated global reads of both `x` and `w` for the second
reduction and the extra `x` read for the output phase.

## Verdict

Row-wise weighted variance normalization is a strong positive DSMEM case once
rows are too wide for one-CTA local shared-memory staging. It extends the
dependent-reduction result from mean absolute deviation to a two-input formula:
`x` and weights both participate in reductions, and staging both streams once
nearly halves measured DRAM traffic.

Artifacts:

- `results/weighted_var_norm_torch_compare.csv`
- `results/weighted_var_norm_ncu_summary.csv`
- `plots/weighted_var_norm_speedup.png`
- `plots/weighted_var_norm_runtime_ms.png`
- `plots/weighted_var_norm_gbps.png`
- `plots/weighted_var_norm_ncu_metrics.png`
