# Row-wise Variance Clipping DSMEM Exploration

## Workload

This folder benchmarks row-wise variance normalization with clipping:

```text
mean = mean(x)
var = mean(x * x) - mean * mean
y = clamp((x - mean) / sqrt(var + eps), -3, 3)
```

The default sweep uses `rows=4096` and
`N in {4096, 8192, 16384, 32768, 65536}`.

## Implementations

- `block_read`: one CTA per row. It reads `x` once to reduce `sum(x)` and
  `sum(x*x)`, then rereads `x` to write clipped normalized output.
- `block_smem`: one CTA per row with local shared-memory staging of the whole
  row when it fits.
- `cluster_smem`: one 8-CTA cluster per row. Each CTA stages a row slice of
  `x`, uses DSMEM only for the two scalar reductions, then writes its local
  output slice.
- `torch.compile`: compiled PyTorch expression for the same mean/variance,
  normalization, and clamp formula.

Throughput uses a 12 B/element model for the logical no-DSMEM read path:
one float input read for reductions, one input reread, and one float output
write.

## Timing Results

Command:

```bash
bash varclip_norm_dsmem_explore/run_compare.sh \
  --rows 4096 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv varclip_norm_dsmem_explore/results/varclip_norm_torch_compare.csv
```

| N | block variant | block ms | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | block_read | 0.067539 | 0.260896 | 0.072365 | 0.259x | 0.277x |
| 8192 | block_smem | 0.177184 | 0.266061 | 0.183226 | 0.666x | 0.689x |
| 16384 | block_smem | 0.352653 | 0.339136 | 0.359136 | 1.040x | 1.059x |
| 32768 | block_read | 1.057240 | 0.695827 | 0.710419 | 1.519x | 1.021x |
| 65536 | block_read | 2.119400 | 1.401080 | 1.955910 | 1.513x | 1.396x |

The DSMEM cluster kernel beats `torch.compile` on 3 of 5 shapes. The smallest
rows are dominated by cluster overhead, while the wide rows benefit once a
single CTA can no longer stage the full row efficiently.

## Nsight Compute

Command:

```bash
bash varclip_norm_dsmem_explore/ncu_varclip_norm_metrics.sh
python3 varclip_norm_dsmem_explore/parse_varclip_norm_ncu.py
```

Profiled shape: `rows=4096`, `N=65536`.

| variant | NCU time (us) | L2 B/element | DRAM B/element | DRAM read B/element | DRAM write B/element |
|---|---:|---:|---:|---:|---:|
| block | 2149.440 | 12.006 | 11.989 | 8.000 | 3.989 |
| cluster | 1388.480 | 8.290 | 7.940 | 4.000 | 3.939 |

The DSMEM path reduces DRAM traffic from `12.0` to `7.9 B/element` and reduces
NCU kernel time by `1.548x` on the profiled shape. Output traffic is unchanged;
DSMEM removes the second global read of the input row.

## Verdict

Row-wise variance clipping is a positive DSMEM case for wide rows. The branchy
clamp does not change the main condition: DSMEM helps when compact scalar
partials are exchanged across CTAs and staged row data is reused for a full-row
output.

Artifacts:

- `results/varclip_norm_torch_compare.csv`
- `results/varclip_norm_ncu_summary.csv`
- `plots/varclip_norm_speedup.png`
- `plots/varclip_norm_runtime_ms.png`
- `plots/varclip_norm_gbps.png`
- `plots/varclip_norm_ncu_metrics.png`
