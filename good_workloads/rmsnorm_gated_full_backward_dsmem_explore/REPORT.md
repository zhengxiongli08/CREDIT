# RMSNorm Gated Full Backward DSMEM Exploration

## Workload

This folder benchmarks the full input/gate backward pass for a row-wise gated
RMSNorm-style operation:

```text
inv_rms = rsqrt(mean(x * x) + eps)
gated_dy = dy * gate
dx = gated_dy * inv_rms - x * sum(gated_dy * x) * inv_rms^3 / N
dgate = dy * x * inv_rms
```

The `gate` tensor has the same shape as `x` and `dy`. The default sweep uses
`rows=4096` and `N in {4096, 8192, 16384, 32768, 65536}`.

## Implementations

- `block_read`: one CTA per row. It reads `x`, `dy`, and `gate` once to
  compute `sum(x*x)` and `sum(dy*gate*x)`, then rereads all three inputs to
  write both `dx` and `dgate`.
- `block_smem`: one CTA per row with local shared-memory staging of `x`, `dy`,
  and `gate` when the whole row fits.
- `cluster_smem`: one 8-CTA cluster per row. Each CTA stages a row slice of
  all three inputs in local shared memory, uses DSMEM only for the two scalar
  reductions, and writes its local `dx` and `dgate` slices.
- `torch.compile`: compiled PyTorch expression returning the same `(dx, dgate)`
  tuple.

Throughput uses a 32 B/element model for the logical no-DSMEM read path:
one `x/dy/gate` read pass for the two reductions, then one `x/dy/gate` read
plus `dx/dgate` writes.

## Timing Results

Command:

```bash
bash rmsnorm_gated_full_backward_dsmem_explore/run_compare.sh \
  --rows 4096 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv rmsnorm_gated_full_backward_dsmem_explore/results/rmsnorm_gated_full_backward_torch_compare.csv
```

| N | block variant | block ms | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | block_smem | 0.215309 | 0.280019 | 0.224800 | 0.769x | 0.803x |
| 8192 | block_smem | 0.431597 | 0.431648 | 0.442579 | 1.000x | 1.025x |
| 16384 | block_read | 1.386570 | 0.864154 | 0.873325 | 1.605x | 1.011x |
| 32768 | block_read | 2.801640 | 1.728170 | 1.884589 | 1.621x | 1.091x |
| 65536 | block_read | 5.612490 | 3.454300 | 5.431091 | 1.625x | 1.572x |

The DSMEM cluster kernel beats `torch.compile` on 4 of 5 shapes. The best
result is smaller than the dx-only gated RMSNorm case because the second output
write is unavoidable traffic that DSMEM cannot remove.

## Nsight Compute

Command:

```bash
bash rmsnorm_gated_full_backward_dsmem_explore/ncu_rmsnorm_gated_full_backward_metrics.sh
python3 rmsnorm_gated_full_backward_dsmem_explore/parse_rmsnorm_gated_full_backward_ncu.py
```

Profiled shape: `rows=4096`, `N=65536`.

| variant | NCU time (us) | L2 B/element | DRAM B/element | DRAM read B/element | DRAM write B/element |
|---|---:|---:|---:|---:|---:|
| block | 5650.368 | 32.006 | 31.998 | 24.004 | 7.993 |
| cluster | 3446.048 | 20.189 | 19.981 | 12.001 | 7.981 |

The DSMEM path reduces DRAM traffic from `32.0` to `20.0 B/element` and
reduces NCU kernel time by `1.640x` on the profiled shape. DSMEM removes the
second read of `x`, `dy`, and `gate`; the `dx/dgate` output writes remain.

## Verdict

Gated RMSNorm full backward is a positive DSMEM case. It confirms that the
three-stream backward pattern still works when the kernel writes two full-row
outputs, but also shows the output-heavy boundary: extra writes reduce the
benefit compared with the dx-only case.

Artifacts:

- `results/rmsnorm_gated_full_backward_torch_compare.csv`
- `results/rmsnorm_gated_full_backward_ncu_summary.csv`
- `plots/rmsnorm_gated_full_backward_speedup.png`
- `plots/rmsnorm_gated_full_backward_runtime_ms.png`
- `plots/rmsnorm_gated_full_backward_gbps.png`
- `plots/rmsnorm_gated_full_backward_ncu_metrics.png`
