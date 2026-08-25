# Row-wise Gated Standardization DSMEM Exploration

## Workload

This folder benchmarks row-wise standardization gated by a second input:

```text
mean = mean(x)
var = mean(x*x) - mean*mean
y = (x - mean) * rsqrt(var + eps) * sigmoid(gate)
```

The default sweep uses `rows=4096` and
`N in {4096, 8192, 16384, 32768, 65536}`.

## Implementations

- `block_read`: one CTA per row. It reads `x` once to reduce mean/variance,
  then rereads `x` and reads `gate` to write the gated standardized output.
- `block_smem`: one CTA per row with local shared-memory staging of both `x`
  and `gate` when the full row fits.
- `cluster_smem`: one 8-CTA cluster per row. Each CTA stages local slices of
  `x` and `gate`, uses DSMEM only for the scalar mean/variance reductions, then
  writes its local output slice.
- `torch.compile`: compiled PyTorch expression for the same two-input formula.

Throughput uses a 16 B/element model for the logical block-read path:
one float `x` read for statistics, one `x` reread, one `gate` read, and one
float output write.

## Timing Results

Command:

```bash
bash gated_standardize_dsmem_explore/run_compare.sh \
  --rows 4096 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv gated_standardize_dsmem_explore/results/gated_standardize_torch_compare.csv
```

| N | block variant | block ms | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | block_smem | 0.1179 | 0.2797 | 0.1179 | 0.422x | 0.422x |
| 8192 | block_smem | 0.2543 | 0.2870 | 0.2644 | 0.886x | 0.921x |
| 16384 | block_read | 0.6605 | 0.5087 | 0.5177 | 1.298x | 1.018x |
| 32768 | block_read | 1.3683 | 1.0183 | 1.0332 | 1.344x | 1.015x |
| 65536 | block_read | 2.7741 | 2.0091 | 2.6132 | 1.381x | 1.301x |

The DSMEM cluster kernel beats `torch.compile` on 3 of 5 shapes. The best
`torch.compile` speedup is `1.301x` at `N=65536`.

## Nsight Compute

Command:

```bash
bash gated_standardize_dsmem_explore/ncu_gated_standardize_metrics.sh
python3 gated_standardize_dsmem_explore/parse_gated_standardize_ncu.py
```

Profiled shape: `rows=4096`, `N=65536`.

| variant | NCU time (us) | modeled GB/s | L2 B/element | DRAM B/element | DRAM read B/element | DRAM write B/element |
|---|---:|---:|---:|---:|---:|---:|
| block | 2761.888 | 1555.084 | 16.001 | 16.001 | 12.000 | 4.001 |
| cluster | 2008.000 | 2138.928 | 12.275 | 11.988 | 8.001 | 3.987 |

The DSMEM path reduces DRAM traffic from `16.0` to `12.0 B/element` and reduces
NCU kernel time by `1.375x` on the profiled shape. Output traffic and the
single `gate` read are unchanged; DSMEM removes the second global read of `x`.

## Verdict

Row-wise gated standardization is a positive DSMEM case for wide rows, but the
margin is smaller than in one-input normalization workloads. Only the `x`
reread is reusable; `gate` is read once in both paths and the sigmoid/output
math remains. This makes it a useful partial-reuse case: DSMEM still wins once
the row is too wide for one-CTA staging, but the extra non-reused traffic caps
the speedup.

Artifacts:

- `results/gated_standardize_torch_compare.csv`
- `results/gated_standardize_ncu_summary.csv`
- `plots/gated_standardize_speedup.png`
- `plots/gated_standardize_runtime_ms.png`
- `plots/gated_standardize_gbps.png`
- `plots/gated_standardize_ncu_metrics.png`
