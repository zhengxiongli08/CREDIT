# RMSNorm Gated Backward-Dx DSMEM Exploration

## Workload

This folder benchmarks the input-gradient for a row-wise gated RMSNorm-style
operation:

```text
inv_rms = rsqrt(mean(x * x) + eps)
gated_dy = dy * gate
dx = gated_dy * inv_rms - x * sum(gated_dy * x) * inv_rms^3 / N
```

The `gate` tensor has the same shape as `x` and `dy`, so this case tests a
third per-element input stream, not a cached per-column affine parameter. The
default sweep uses `rows=4096` and
`N in {4096, 8192, 16384, 32768, 65536}`.

## Implementations

- `block_read`: one CTA per row. It reads `x`, `dy`, and `gate` once to
  compute `sum(x*x)` and `sum(dy*gate*x)`, then rereads all three inputs to
  write `dx`.
- `block_smem`: one CTA per row with local shared-memory staging of `x`, `dy`,
  and `gate` when the whole row fits.
- `cluster_smem`: one 8-CTA cluster per row. Each CTA stages a row slice of
  all three inputs in local shared memory, uses DSMEM only for the two scalar
  reductions, and writes its local `dx` slice.
- `torch.compile`: compiled PyTorch expression for the same `dx` formula.

Throughput uses a 28 B/element model for the logical no-DSMEM read path:
one `x/dy/gate` read pass for the two reductions, then one `x/dy/gate` read
plus one `dx` write pass.

## Timing Results

Command:

```bash
bash rmsnorm_gated_backward_dsmem_explore/run_compare.sh \
  --rows 4096 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv rmsnorm_gated_backward_dsmem_explore/results/rmsnorm_gated_backward_torch_compare.csv
```

| N | block variant | block ms | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | block_smem | 0.159053 | 0.259674 | 0.168486 | 0.613x | 0.649x |
| 8192 | block_smem | 0.334656 | 0.327507 | 0.346374 | 1.022x | 1.058x |
| 16384 | block_read | 1.210060 | 0.669568 | 0.685875 | 1.807x | 1.024x |
| 32768 | block_read | 2.451240 | 1.339540 | 1.543078 | 1.830x | 1.152x |
| 65536 | block_read | 4.904260 | 2.654220 | 4.654771 | 1.848x | 1.754x |

The DSMEM cluster kernel beats `torch.compile` on 4 of 5 shapes. It loses at
the smallest shape where a single CTA can stage the whole row, then wins once
the full three-input row no longer fits comfortably in local shared memory.

## Nsight Compute

Command:

```bash
bash rmsnorm_gated_backward_dsmem_explore/ncu_rmsnorm_gated_backward_metrics.sh
python3 rmsnorm_gated_backward_dsmem_explore/parse_rmsnorm_gated_backward_ncu.py
```

Profiled shape: `rows=4096`, `N=65536`.

| variant | NCU time (us) | L2 B/element | DRAM B/element | DRAM read B/element | DRAM write B/element |
|---|---:|---:|---:|---:|---:|
| block | 4903.968 | 28.009 | 28.013 | 24.005 | 4.008 |
| cluster | 2652.128 | 16.190 | 15.986 | 12.002 | 3.985 |

The DSMEM path reduces DRAM traffic from `28.0` to `16.0 B/element` and
reduces NCU kernel time by `1.849x` on the profiled shape. This confirms that
DSMEM still works when the reusable row data is three input streams: it stages
the element data locally and exchanges only two scalar partials across CTAs.

## Verdict

Gated RMSNorm backward-dx is a strong positive DSMEM case. It generalizes the
L2/RMS-style backward pattern from two staged streams to three staged streams,
and the traffic reduction is larger because DSMEM removes a full reread of
`x`, `dy`, and `gate`.

Artifacts:

- `results/rmsnorm_gated_backward_torch_compare.csv`
- `results/rmsnorm_gated_backward_ncu_summary.csv`
- `plots/rmsnorm_gated_backward_speedup.png`
- `plots/rmsnorm_gated_backward_runtime_ms.png`
- `plots/rmsnorm_gated_backward_gbps.png`
- `plots/rmsnorm_gated_backward_ncu_metrics.png`
