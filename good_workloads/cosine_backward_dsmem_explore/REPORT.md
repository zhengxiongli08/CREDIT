# Cosine Similarity Backward DSMEM Exploration

## Workload

This folder benchmarks backward gradients for row-wise cosine similarity
between two vectors:

```text
cos = sum(x * y) / sqrt(sum(x * x) + eps) / sqrt(sum(y * y) + eps)
dx = grad * (y * inv_x * inv_y - x * dot * inv_x^3 * inv_y)
dy = grad * (x * inv_x * inv_y - y * dot * inv_x * inv_y^3)
```

The default sweep uses `rows=4096` and
`N in {4096, 8192, 16384, 32768, 65536}`.

## Implementations

- `block_read`: one CTA per row. It reads `x` and `y` once to compute
  `sum(x*x)`, `sum(y*y)`, and `sum(x*y)`, then rereads `x` and `y` to write
  both `dx` and `dy`.
- `block_smem`: one CTA per row with local shared-memory staging of both `x`
  and `y` when the whole row fits.
- `cluster_smem`: one 8-CTA cluster per row. Each CTA stages a row slice of
  `x` and `y`, uses DSMEM only for the three scalar reductions, then writes
  its local `dx` and `dy` output slices.
- `torch.compile`: compiled PyTorch expression for the same gradient formula.

Throughput uses a 24 B/element model for the logical no-DSMEM read path:
one `x/y` read pass for reductions, then one `x/y` reread plus two output
writes.

## Timing Results

Command:

```bash
bash cosine_backward_dsmem_explore/run_compare.sh \
  --rows 4096 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv cosine_backward_dsmem_explore/results/cosine_backward_torch_compare.csv
```

| N | block variant | block ms | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | block_smem | 0.176390 | 0.288173 | 0.183942 | 0.612x | 0.638x |
| 8192 | block_smem | 0.352000 | 0.344314 | 0.360262 | 1.022x | 1.046x |
| 16384 | block_read | 1.070240 | 0.701254 | 0.709677 | 1.526x | 1.012x |
| 32768 | block_read | 2.146380 | 1.410080 | 1.479270 | 1.522x | 1.049x |
| 65536 | block_read | 4.367170 | 2.817650 | 4.111591 | 1.550x | 1.459x |

The DSMEM cluster kernel beats `torch.compile` on 4 of 5 shapes. It loses at
the smallest shape where a single CTA can stage the row, then wins once rows
are wide enough for the cluster split to amortize synchronization.

## Nsight Compute

Command:

```bash
bash cosine_backward_dsmem_explore/ncu_cosine_backward_metrics.sh
python3 cosine_backward_dsmem_explore/parse_cosine_backward_ncu.py
```

Profiled shape: `rows=4096`, `N=65536`.

| variant | NCU time (us) | L2 B/element | DRAM B/element | DRAM read B/element | DRAM write B/element |
|---|---:|---:|---:|---:|---:|
| block | 4369.984 | 24.003 | 23.991 | 16.001 | 7.990 |
| cluster | 2808.992 | 16.192 | 15.947 | 8.000 | 7.947 |

The DSMEM path reduces DRAM traffic from `24.0` to `15.9 B/element` and
reduces NCU kernel time by `1.556x` on the profiled shape. The write traffic
is unchanged because both variants produce `dx` and `dy`; the win comes from
avoiding the second global read of `x` and `y`.

## Verdict

Cosine similarity backward is a positive DSMEM case. It extends the staged
row-reuse pattern from single-input normalization to a pairwise vector
workload with three scalar reductions and two full-row outputs.

Artifacts:

- `results/cosine_backward_torch_compare.csv`
- `results/cosine_backward_ncu_summary.csv`
- `plots/cosine_backward_speedup.png`
- `plots/cosine_backward_runtime_ms.png`
- `plots/cosine_backward_gbps.png`
- `plots/cosine_backward_ncu_metrics.png`
