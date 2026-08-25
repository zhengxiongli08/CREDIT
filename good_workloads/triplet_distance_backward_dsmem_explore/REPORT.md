# Triplet Distance Backward DSMEM Exploration

## Workload

This folder benchmarks backward gradients for a row-wise triplet-style distance
difference:

```text
ap = x - y
an = x - z
dx = grad * (ap / sqrt(sum(ap * ap) + eps) - an / sqrt(sum(an * an) + eps))
dy = -grad * ap / sqrt(sum(ap * ap) + eps)
dz = grad * an / sqrt(sum(an * an) + eps)
```

The default sweep uses `rows=2048` and
`N in {4096, 8192, 16384, 32768, 65536}`.

## Implementations

- `block_read`: one CTA per row. It reads `x`, `y`, and `z` once to reduce the
  anchor-positive and anchor-negative squared distances, then rereads all three
  vectors to write `dx`, `dy`, and `dz`.
- `block_smem`: one CTA per row with local shared-memory staging of `x`, `y`,
  and `z` when the whole row fits.
- `cluster_smem`: one 8-CTA cluster per row. Each CTA stages a row slice of
  `x`, `y`, and `z`, uses DSMEM only for the two scalar distance reductions,
  then writes its local `dx`, `dy`, and `dz` output slices.
- `torch.compile`: compiled PyTorch expression for the same gradient formula.

Throughput uses a 36 B/element model for the logical no-DSMEM read path:
one `x/y/z` read pass for reductions, then one `x/y/z` reread plus three output
writes.

## Timing Results

Command:

```bash
bash triplet_distance_backward_dsmem_explore/run_compare.sh \
  --rows 2048 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv triplet_distance_backward_dsmem_explore/results/triplet_distance_backward_torch_compare.csv
```

| N | block variant | block ms | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | block_smem | 0.133376 | 0.163309 | 0.142086 | 0.817x | 0.870x |
| 8192 | block_smem | 0.265312 | 0.261485 | 0.273094 | 1.015x | 1.044x |
| 16384 | block_read | 0.811507 | 0.528051 | 0.709491 | 1.537x | 1.344x |
| 32768 | block_read | 1.632740 | 1.058480 | 1.535373 | 1.543x | 1.451x |
| 65536 | block_read | 3.286850 | 2.116400 | 3.103290 | 1.553x | 1.466x |

The DSMEM cluster kernel beats `torch.compile` on 4 of 5 shapes. The smallest
shape remains better served by a single CTA with local shared-memory staging;
the cluster path wins once a row is too wide for that staging strategy.

## Nsight Compute

Command:

```bash
bash triplet_distance_backward_dsmem_explore/ncu_triplet_distance_backward_metrics.sh
python3 triplet_distance_backward_dsmem_explore/parse_triplet_distance_backward_ncu.py
```

Profiled shape: `rows=2048`, `N=65536`.

| variant | NCU time (us) | L2 B/element | DRAM B/element | DRAM read B/element | DRAM write B/element |
|---|---:|---:|---:|---:|---:|
| block | 3302.624 | 36.003 | 35.979 | 24.001 | 11.978 |
| cluster | 2110.720 | 24.194 | 23.933 | 12.001 | 11.933 |

The DSMEM path reduces DRAM traffic from `36.0` to `23.9 B/element` and
reduces NCU kernel time by `1.565x` on the profiled shape. Output traffic is
unchanged because both variants write `dx`, `dy`, and `dz`; DSMEM removes the
second global read of `x`, `y`, and `z`.

## Verdict

Triplet distance backward is a positive DSMEM case. It extends the
pairwise-vector pattern to three staged inputs, two scalar reductions, and
three full-row gradient outputs. The larger output traffic does not prevent a
speedup because the cluster path still removes a full reread of all staged
input vectors.

Artifacts:

- `results/triplet_distance_backward_torch_compare.csv`
- `results/triplet_distance_backward_ncu_summary.csv`
- `plots/triplet_distance_backward_speedup.png`
- `plots/triplet_distance_backward_runtime_ms.png`
- `plots/triplet_distance_backward_gbps.png`
- `plots/triplet_distance_backward_ncu_metrics.png`
