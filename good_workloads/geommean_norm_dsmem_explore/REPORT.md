# Row-wise Geometric-Mean Normalization DSMEM Exploration

## Workload

This folder benchmarks row-wise geometric-mean normalization:

```text
mean_log = mean(log(abs(x) + eps))
scale = exp(mean_log)
y = x / scale
```

The default sweep uses `rows=4096` and
`N in {4096, 8192, 16384, 32768, 65536}`.

## Implementations

- `block_read`: one CTA per row. It reads `x` once to reduce
  `sum(log(abs(x) + eps))`, then rereads `x` to write normalized output.
- `block_smem`: one CTA per row with local shared-memory staging of the whole
  row when it fits.
- `cluster_smem`: one 8-CTA cluster per row. Each CTA stages a row slice of
  `x`, uses DSMEM only for the scalar log-sum reduction, then writes its local
  output slice using the global geometric mean.
- `torch.compile`: compiled PyTorch expression using
  `torch.mean(torch.log(torch.abs(x) + eps))` and the normalization expression.

Throughput uses a 12 B/element model for the logical no-DSMEM read path:
one float input read for the log reduction, one input reread, and one float
output write.

## Timing Results

Command:

```bash
bash geommean_norm_dsmem_explore/run_compare.sh \
  --rows 4096 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv geommean_norm_dsmem_explore/results/geommean_norm_torch_compare.csv
```

| N | block variant | block ms | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | block_smem | 0.0722 | 0.2567 | 0.0739 | 0.281x | 0.288x |
| 8192 | block_smem | 0.1777 | 0.2611 | 0.1873 | 0.681x | 0.717x |
| 16384 | block_smem | 0.3515 | 0.3409 | 0.3567 | 1.031x | 1.046x |
| 32768 | block_read | 1.0445 | 0.6941 | 0.7139 | 1.505x | 1.028x |
| 65536 | block_read | 2.1186 | 1.4012 | 1.9810 | 1.512x | 1.414x |

The DSMEM cluster kernel beats `torch.compile` on 3 of 5 shapes. The small-row
cases are better handled by a single CTA or local shared-memory staging; DSMEM
wins once rows become too wide for whole-row local staging. The best
`torch.compile` speedup is `1.414x` at `N=65536`.

## Nsight Compute

Command:

```bash
bash geommean_norm_dsmem_explore/ncu_geommean_norm_metrics.sh
python3 geommean_norm_dsmem_explore/parse_geommean_norm_ncu.py
```

Profiled shape: `rows=4096`, `N=65536`.

| variant | NCU time (us) | modeled GB/s | L2 B/element | DRAM B/element | DRAM read B/element | DRAM write B/element |
|---|---:|---:|---:|---:|---:|---:|
| block | 2111.680 | 1525.433 | 12.001 | 11.989 | 8.000 | 3.989 |
| cluster | 1385.376 | 2325.163 | 8.267 | 7.975 | 4.000 | 3.975 |

The DSMEM path reduces DRAM traffic from `12.0` to `8.0 B/element` and reduces
NCU kernel time by `1.524x` on the profiled shape. Output traffic is unchanged;
DSMEM removes the second global read of the input row.

## Verdict

Row-wise geometric-mean normalization is a positive DSMEM case for wide rows.
It confirms that even a single scalar log reduction can benefit when the row
data is reused for a full-row output and the row is too wide for one-CTA local
shared-memory staging. The extra `log` and `exp` math reduces the margin
relative to simpler memory-bound reductions, but it does not break the
staging/reuse pattern.

Artifacts:

- `results/geommean_norm_torch_compare.csv`
- `results/geommean_norm_ncu_summary.csv`
- `plots/geommean_norm_speedup.png`
- `plots/geommean_norm_runtime_ms.png`
- `plots/geommean_norm_gbps.png`
- `plots/geommean_norm_ncu_metrics.png`
