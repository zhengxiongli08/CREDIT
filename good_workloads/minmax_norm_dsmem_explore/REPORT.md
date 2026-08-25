# Row-wise Min-Max Normalization DSMEM Exploration

## Workload

This folder benchmarks row-wise min-max normalization:

```text
lo = min(x)
hi = max(x)
y = (x - lo) / (hi - lo + eps)
```

The default sweep uses `rows=4096` and
`N in {4096, 8192, 16384, 32768, 65536}`.

## Implementations

- `block_read`: one CTA per row. It reads `x` once to reduce row min/max, then
  rereads `x` to write normalized output.
- `block_smem`: one CTA per row with local shared-memory staging of the whole
  row when it fits.
- `cluster_smem`: one 8-CTA cluster per row. Each CTA stages a row slice of
  `x`, uses DSMEM only for the scalar min/max reductions, then writes its local
  output slice.
- `torch.compile`: compiled PyTorch expression using `torch.amin`,
  `torch.amax`, and the normalization expression.

Throughput uses a 12 B/element model for the logical no-DSMEM read path:
one float input read for the min/max reductions, one input reread, and one
float output write.

## Timing Results

Command:

```bash
bash minmax_norm_dsmem_explore/run_compare.sh \
  --rows 4096 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv minmax_norm_dsmem_explore/results/minmax_norm_torch_compare.csv
```

| N | block variant | block ms | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | block_read | 0.072448 | 0.264762 | 0.072262 | 0.274x | 0.273x |
| 8192 | block_smem | 0.177869 | 0.261747 | 0.182989 | 0.680x | 0.699x |
| 16384 | block_smem | 0.351981 | 0.339322 | 0.359194 | 1.037x | 1.059x |
| 32768 | block_read | 1.045280 | 0.694413 | 0.715898 | 1.505x | 1.031x |
| 65536 | block_read | 2.107510 | 1.400400 | 1.962899 | 1.505x | 1.402x |

The DSMEM cluster kernel beats `torch.compile` on 3 of 5 shapes. As with
dynamic quantization, the cluster path becomes useful once local whole-row
staging is no longer the best no-DSMEM option.

## Nsight Compute

Command:

```bash
bash minmax_norm_dsmem_explore/ncu_minmax_norm_metrics.sh
python3 minmax_norm_dsmem_explore/parse_minmax_norm_ncu.py
```

Profiled shape: `rows=4096`, `N=65536`.

| variant | NCU time (us) | L2 B/element | DRAM B/element | DRAM read B/element | DRAM write B/element |
|---|---:|---:|---:|---:|---:|
| block | 2106.016 | 12.001 | 11.985 | 7.995 | 3.990 |
| cluster | 1384.672 | 8.253 | 7.983 | 4.001 | 3.982 |

The DSMEM path reduces DRAM traffic from `12.0` to `8.0 B/element` and reduces
NCU kernel time by `1.521x` on the profiled shape. Output traffic is unchanged;
DSMEM removes the second global read of the input row.

## Verdict

Row-wise min-max normalization is a positive DSMEM case for wide rows. It
extends the profitable staging/reuse pattern to non-sum reductions: DSMEM only
exchanges compact min/max partials while the staged element data is reused for
the output pass.

Artifacts:

- `results/minmax_norm_torch_compare.csv`
- `results/minmax_norm_ncu_summary.csv`
- `plots/minmax_norm_speedup.png`
- `plots/minmax_norm_runtime_ms.png`
- `plots/minmax_norm_gbps.png`
- `plots/minmax_norm_ncu_metrics.png`
