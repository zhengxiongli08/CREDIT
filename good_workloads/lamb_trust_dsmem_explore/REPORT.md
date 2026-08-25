# LAMB Trust-Ratio DSMEM Exploration

## Workload

This folder benchmarks a row-wise LAMB/AdamW-style trust-ratio optimizer
update:

```text
first_out = beta1 * first + (1 - beta1) * grad
second_out = beta2 * second + (1 - beta2) * grad * grad
update = first_out / (sqrt(second_out) + eps) + weight_decay * weight
weight_norm = sqrt(sum(weight * weight))
update_norm = sqrt(sum(update * update))
trust = trust_coeff * weight_norm / (update_norm + eps)
weight_out = weight - learning_rate * trust * update
```

The default sweep uses `rows=4096` and
`N in {4096, 8192, 16384, 32768, 65536}`.

## Implementations

- `block_read`: one CTA per row. It reads `weight`, `grad`, `first`, and
  `second` to reduce `weight_norm` and `update_norm`, then rereads all four
  streams to write `weight_out`, `first_out`, and `second_out`.
- `block_smem`: one CTA per row with local shared-memory staging of all four
  input streams when the full row fits.
- `cluster_smem`: one 8-CTA cluster per row. Each CTA stages a row slice of
  `weight`, `grad`, `first`, and `second`, uses DSMEM for the two scalar norm
  reductions, then writes its local output slice.
- `torch.compile`: compiled PyTorch expression returning the same
  `(weight_out, first_out, second_out)` tuple.

Throughput uses a 44 B/element model for the logical block-read path: one
four-stream read pass for the two norm reductions, then one four-stream read
pass plus three output writes.

## Timing Results

Command:

```bash
bash lamb_trust_dsmem_explore/run_compare.sh \
  --rows 4096 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv lamb_trust_dsmem_explore/results/lamb_trust_torch_compare.csv
```

| N | block variant | block ms | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | block_smem | 0.3033 | 0.3379 | 0.3170 | 0.898x | 0.938x |
| 8192 | block_read | 0.9520 | 0.6111 | 0.6226 | 1.558x | 1.019x |
| 16384 | block_read | 1.9284 | 1.2195 | 1.2307 | 1.581x | 1.009x |
| 32768 | block_read | 3.8934 | 2.4330 | 2.7338 | 1.600x | 1.124x |
| 65536 | block_read | 7.8463 | n/a | 6.8311 | n/a | n/a |

The DSMEM cluster kernel beats `torch.compile` on 3 of the 4 shapes where the
cluster path is valid. The best `torch.compile` speedup is `1.124x` at
`N=32768`. At `N=65536`, four staged arrays exceed the per-block dynamic
shared-memory limit for an 8-CTA cluster on this device, so no DSMEM result is
reported.

## Nsight Compute

Command:

```bash
bash lamb_trust_dsmem_explore/ncu_lamb_trust_metrics.sh
python3 lamb_trust_dsmem_explore/parse_lamb_trust_ncu.py
```

Profiled shape: `rows=4096`, `N=32768`.

| variant | NCU time (us) | modeled GB/s | L2 B/element | DRAM B/element | DRAM read B/element | DRAM write B/element |
|---|---:|---:|---:|---:|---:|---:|
| block | 3909.856 | 1510.434 | 44.007 | 43.991 | 32.004 | 11.987 |
| cluster | 2414.528 | 2445.853 | 28.377 | 27.963 | 16.001 | 11.963 |

The DSMEM path reduces DRAM traffic from `44.0` to `28.0 B/element` and reduces
NCU kernel time by `1.620x` on the profiled shape. DSMEM removes the second
read of the four input streams; the three output writes are unchanged.

## Verdict

LAMB trust-ratio update is a positive but capacity-limited optimizer-style
DSMEM case. It extends the optimizer update pattern to four input streams and
three outputs: DSMEM still improves the valid wide-row shapes, but local
shared-memory demand now caps the maximum row width that an 8-CTA cluster can
stage.

Artifacts:

- `results/lamb_trust_torch_compare.csv`
- `results/lamb_trust_ncu_summary.csv`
- `plots/lamb_trust_speedup.png`
- `plots/lamb_trust_runtime_ms.png`
- `plots/lamb_trust_gbps.png`
- `plots/lamb_trust_ncu_metrics.png`
