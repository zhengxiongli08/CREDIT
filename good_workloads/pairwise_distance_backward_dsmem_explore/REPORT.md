# Pairwise Distance Backward DSMEM Exploration

## Workload

This folder benchmarks backward gradients for row-wise Euclidean pairwise
distance between two vectors:

```text
dist = sqrt(sum((x - y) * (x - y)) + eps)
dx = grad * (x - y) / dist
dy = -dx
```

The default sweep uses `rows=4096` and
`N in {4096, 8192, 16384, 32768, 65536}`.

## Implementations

- `block_read`: one CTA per row. It reads `x` and `y` once to reduce
  `sum((x-y)^2)`, then rereads `x` and `y` to write both `dx` and `dy`.
- `block_smem`: one CTA per row with local shared-memory staging of both `x`
  and `y` when the whole row fits.
- `cluster_smem`: one 8-CTA cluster per row. Each CTA stages a row slice of
  `x` and `y`, uses DSMEM only for the scalar distance reduction, then writes
  its local `dx` and `dy` output slices.
- `torch.compile`: compiled PyTorch expression for the same gradient formula.

Throughput uses a 24 B/element model for the logical no-DSMEM read path:
one `x/y` read pass for the reduction, then one `x/y` reread plus two output
writes.

## Timing Results

Command:

```bash
bash pairwise_distance_backward_dsmem_explore/run_compare.sh \
  --rows 4096 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv pairwise_distance_backward_dsmem_explore/results/pairwise_distance_backward_torch_compare.csv
```

| N | block variant | block ms | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | block_smem | 0.177440 | 0.272448 | 0.184435 | 0.651x | 0.677x |
| 8192 | block_smem | 0.351232 | 0.344038 | 0.360960 | 1.021x | 1.049x |
| 16384 | block_read | 1.065900 | 0.701741 | 0.711706 | 1.519x | 1.014x |
| 32768 | block_read | 2.159900 | 1.409540 | 1.421811 | 1.532x | 1.009x |
| 65536 | block_read | 4.367520 | 2.815320 | 4.115097 | 1.551x | 1.462x |

The DSMEM cluster kernel beats `torch.compile` on 4 of 5 shapes. The smallest
shape is better served by a single CTA with local shared-memory staging; the
cluster path wins once a row is too wide for that staging strategy.

## Nsight Compute

Command:

```bash
bash pairwise_distance_backward_dsmem_explore/ncu_pairwise_distance_backward_metrics.sh
python3 pairwise_distance_backward_dsmem_explore/parse_pairwise_distance_backward_ncu.py
```

Profiled shape: `rows=4096`, `N=65536`.

| variant | NCU time (us) | L2 B/element | DRAM B/element | DRAM read B/element | DRAM write B/element |
|---|---:|---:|---:|---:|---:|
| block | 4364.032 | 24.004 | 23.993 | 16.002 | 7.991 |
| cluster | 2810.624 | 16.196 | 15.955 | 8.004 | 7.951 |

The DSMEM path reduces DRAM traffic from `24.0` to `16.0 B/element` and
reduces NCU kernel time by `1.553x` on the profiled shape. Output traffic is
unchanged because both variants write `dx` and `dy`; DSMEM removes the second
global read of `x` and `y`.

## Verdict

Pairwise distance backward is a positive DSMEM case. It is useful because it
has only one scalar reduction, unlike cosine backward, yet still benefits from
staging two input vectors once and reusing them for two full-row gradient
outputs.

Artifacts:

- `results/pairwise_distance_backward_torch_compare.csv`
- `results/pairwise_distance_backward_ncu_summary.csv`
- `plots/pairwise_distance_backward_speedup.png`
- `plots/pairwise_distance_backward_runtime_ms.png`
- `plots/pairwise_distance_backward_gbps.png`
- `plots/pairwise_distance_backward_ncu_metrics.png`
