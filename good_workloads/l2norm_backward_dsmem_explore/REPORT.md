# L2 Normalization Backward-Dx DSMEM Exploration

## Workload

This folder benchmarks the input-gradient for row-wise L2 normalization:

```text
y = x / sqrt(sum(x * x) + eps)
dx = dy * inv_norm - x * sum(dy * x) * inv_norm^3
```

The default sweep uses `rows=4096` and
`N in {4096, 8192, 16384, 32768, 65536}`.

## Implementations

- `block_read`: one CTA per row. It combines the independent `sum(x*x)` and
  `sum(dy*x)` reductions in one global read pass, then rereads `x` and `dy` to
  write `dx`.
- `block_smem`: one CTA per row with local shared-memory staging of both `x`
  and `dy` when the whole row fits.
- `cluster_smem`: one 8-CTA cluster per row. Each CTA stages a row slice of
  `x` and `dy` in local shared memory, uses DSMEM only for the two scalar
  reductions, and writes its local output slice.
- `torch.compile`: compiled PyTorch expression for the same `dx` formula.

Throughput uses a 20 B/element model for the logical no-DSMEM read path:
one `x/dy` read pass for the two reductions, then one `x/dy` read plus one
`dx` write pass.

## Timing Results

Command:

```bash
bash l2norm_backward_dsmem_explore/run_compare.sh \
  --rows 4096 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv l2norm_backward_dsmem_explore/results/l2norm_backward_torch_compare.csv
```

| N | block variant | block ms | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | block_smem | 0.119040 | 0.241683 | 0.120083 | 0.493x | 0.497x |
| 8192 | block_smem | 0.254317 | 0.262355 | 0.263725 | 0.969x | 1.005x |
| 16384 | block_read | 0.840768 | 0.509402 | 0.520154 | 1.651x | 1.021x |
| 32768 | block_read | 1.707920 | 1.022390 | 1.056493 | 1.671x | 1.033x |
| 65536 | block_read | 3.442230 | 2.049450 | 3.349376 | 1.680x | 1.634x |

The DSMEM cluster kernel beats `torch.compile` on 4 of 5 shapes. It loses at
the smallest shape where a single CTA can stage the whole row cheaply, then
wins once rows are too wide for local shared-memory staging.

## Nsight Compute

Command:

```bash
bash l2norm_backward_dsmem_explore/ncu_l2norm_backward_metrics.sh
python3 l2norm_backward_dsmem_explore/parse_l2norm_backward_ncu.py
```

Profiled shape: `rows=4096`, `N=65536`.

| variant | NCU time (us) | L2 B/element | DRAM B/element | DRAM read B/element | DRAM write B/element |
|---|---:|---:|---:|---:|---:|
| block | 3434.336 | 20.005 | 19.999 | 16.002 | 3.998 |
| cluster | 2013.792 | 12.185 | 11.984 | 8.001 | 3.984 |

The DSMEM path reduces DRAM traffic from `20.0` to `12.0 B/element` and
reduces NCU kernel time by `1.705x` on the profiled shape. This is the intended
pattern: stage element data once locally, exchange only compact scalar partials
through DSMEM, and avoid the final global reread of `x` and `dy`.

## Verdict

L2 normalization backward is a positive DSMEM case. It is a smaller formula
than LayerNorm backward because the two reductions can be computed in the same
first pass, but DSMEM still gives a clear win on wide rows by keeping both
`x` and `dy` staged across the output pass.

Artifacts:

- `results/l2norm_backward_torch_compare.csv`
- `results/l2norm_backward_ncu_summary.csv`
- `plots/l2norm_backward_speedup.png`
- `plots/l2norm_backward_runtime_ms.png`
- `plots/l2norm_backward_gbps.png`
- `plots/l2norm_backward_ncu_metrics.png`
