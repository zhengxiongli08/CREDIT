# Mean Absolute Deviation Normalization DSMEM Exploration

## Workload

This folder benchmarks row-wise mean absolute deviation (MAD) normalization:

```text
mean = mean(x)
mad = mean(abs(x - mean))
y = (x - mean) / (mad + eps)
```

The default sweep uses `rows=4096` and
`N in {4096, 8192, 16384, 32768, 65536}`.

## Implementations

- `block_read`: one CTA per row. It reads `x` once to reduce `sum(x)`, rereads
  `x` to reduce `sum(abs(x - mean))`, then rereads `x` again to write output.
- `block_smem`: one CTA per row with local shared-memory staging of the whole
  row when it fits. It computes both dependent reductions and the output from
  the staged row.
- `cluster_smem`: one 8-CTA cluster per row. Each CTA stages a row slice of
  `x`, uses DSMEM for the mean reduction, reuses the staged slice for the MAD
  reduction, uses DSMEM again, then writes its local output slice.
- `torch.compile`: compiled PyTorch expression for the same mean, MAD, and
  normalization formula.

Throughput uses a 16 B/element model for the logical no-DSMEM read path:
three float input reads plus one float output write.

## Timing Results

Command:

```bash
bash mad_norm_dsmem_explore/run_compare.sh \
  --rows 4096 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv mad_norm_dsmem_explore/results/mad_norm_torch_compare.csv
```

| N | block variant | block ms | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | block_read | 0.072461 | 0.337555 | 0.072442 | 0.215x | 0.215x |
| 8192 | block_smem | 0.177242 | 0.335034 | 0.183424 | 0.529x | 0.547x |
| 16384 | block_smem | 0.351334 | 0.377594 | 0.357299 | 0.930x | 0.946x |
| 32768 | block_read | 1.303010 | 0.691635 | 0.706714 | 1.884x | 1.022x |
| 65536 | block_read | 2.776200 | 1.397790 | 2.503821 | 1.986x | 1.791x |

The DSMEM cluster kernel beats `torch.compile` on 2 of 5 shapes, with a large
win at the widest row. Smaller shapes are still better served by a single CTA
with whole-row local staging.

## Nsight Compute

Command:

```bash
bash mad_norm_dsmem_explore/ncu_mad_norm_metrics.sh
python3 mad_norm_dsmem_explore/parse_mad_norm_ncu.py
```

Profiled shape: `rows=4096`, `N=65536`.

| variant | NCU time (us) | L2 B/element | DRAM B/element | DRAM read B/element | DRAM write B/element |
|---|---:|---:|---:|---:|---:|
| block | 2794.560 | 16.001 | 15.971 | 11.980 | 3.991 |
| cluster | 1374.656 | 8.315 | 7.962 | 4.001 | 3.961 |

The DSMEM path reduces DRAM traffic from `16.0` to `8.0 B/element` and reduces
NCU kernel time by `2.033x` on the profiled shape. Output traffic is unchanged;
DSMEM removes two global rereads of the input row by retaining each CTA's slice
in local shared memory across both dependent reductions.

## Verdict

Mean absolute deviation normalization is a strong positive DSMEM case for wide
rows. It is useful because the second reduction depends on the result of the
first; DSMEM still works well when the element data remains staged locally and
only scalar partials cross CTAs.

Artifacts:

- `results/mad_norm_torch_compare.csv`
- `results/mad_norm_ncu_summary.csv`
- `plots/mad_norm_speedup.png`
- `plots/mad_norm_runtime_ms.png`
- `plots/mad_norm_gbps.png`
- `plots/mad_norm_ncu_metrics.png`
