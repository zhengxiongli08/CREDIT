# Row-wise L1 Normalization DSMEM Exploration

## Workload

This folder benchmarks row-wise L1 normalization:

```text
denom = sum(abs(x)) + eps
y = x / denom
```

The default sweep uses `rows=4096` and
`N in {4096, 8192, 16384, 32768, 65536}`.

## Implementations

- `block_read`: one CTA per row. It reads `x` once to reduce `sum(abs(x))`,
  then rereads `x` to write normalized output.
- `block_smem`: one CTA per row with local shared-memory staging of the whole
  row when it fits.
- `cluster_smem`: one 8-CTA cluster per row. Each CTA stages a row slice of
  `x`, uses DSMEM only for the scalar L1 reduction, then writes its local
  output slice.
- `torch.compile`: compiled PyTorch expression using `torch.sum(torch.abs(x))`
  and the normalization expression.

Throughput uses a 12 B/element model for the logical no-DSMEM read path:
one float input read for the L1 reduction, one input reread, and one float
output write.

## Timing Results

Command:

```bash
bash l1norm_dsmem_explore/run_compare.sh \
  --rows 4096 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv l1norm_dsmem_explore/results/l1norm_torch_compare.csv
```

| N | block variant | block ms | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | block_read | 0.071693 | 0.250298 | 0.076525 | 0.286x | 0.306x |
| 8192 | block_smem | 0.177862 | 0.249933 | 0.183443 | 0.712x | 0.734x |
| 16384 | block_smem | 0.351629 | 0.337741 | 0.361542 | 1.041x | 1.070x |
| 32768 | block_read | 1.044470 | 0.693312 | 0.711942 | 1.506x | 1.027x |
| 65536 | block_read | 2.110790 | 1.401400 | 1.948646 | 1.506x | 1.390x |

The DSMEM cluster kernel beats `torch.compile` on 3 of 5 shapes. The small-row
cases are better handled by a single CTA or local shared-memory staging; DSMEM
wins once rows become too wide for whole-row local staging.

## Nsight Compute

Command:

```bash
bash l1norm_dsmem_explore/ncu_l1norm_metrics.sh
python3 l1norm_dsmem_explore/parse_l1norm_ncu.py
```

Profiled shape: `rows=4096`, `N=65536`.

| variant | NCU time (us) | L2 B/element | DRAM B/element | DRAM read B/element | DRAM write B/element |
|---|---:|---:|---:|---:|---:|
| block | 2114.656 | 12.001 | 11.987 | 7.998 | 3.989 |
| cluster | 1383.392 | 8.258 | 7.952 | 4.000 | 3.951 |

The DSMEM path reduces DRAM traffic from `12.0` to `8.0 B/element` and reduces
NCU kernel time by `1.529x` on the profiled shape. Output traffic is unchanged;
DSMEM removes the second global read of the input row.

## Verdict

Row-wise L1 normalization is a positive DSMEM case for wide rows. It confirms
that even a single scalar reduction can benefit when the row data is reused for
a full-row output and the row is too wide for one-CTA local shared-memory
staging.

Artifacts:

- `results/l1norm_torch_compare.csv`
- `results/l1norm_ncu_summary.csv`
- `plots/l1norm_speedup.png`
- `plots/l1norm_runtime_ms.png`
- `plots/l1norm_gbps.png`
- `plots/l1norm_ncu_metrics.png`
