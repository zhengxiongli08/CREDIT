# LARS Momentum DSMEM Exploration

## Workload

This folder benchmarks a row-wise LARS-style momentum optimizer update:

```text
update = momentum_coeff * momentum + grad + weight_decay * weight
weight_norm = sqrt(sum(weight * weight))
update_norm = sqrt(sum(update * update))
trust = trust_coeff * weight_norm / (update_norm + eps)
weight_out = weight - learning_rate * trust * update
momentum_out = update
```

The default sweep uses `rows=4096` and
`N in {4096, 8192, 16384, 32768, 65536}`.

## Implementations

- `block_read`: one CTA per row. It reads `weight`, `grad`, and `momentum` to
  reduce `weight_norm` and `update_norm`, then rereads all three streams to
  write `weight_out` and `momentum_out`.
- `block_smem`: one CTA per row with local shared-memory staging of the three
  input streams when the full row fits.
- `cluster_smem`: one 8-CTA cluster per row. Each CTA stages a row slice of
  `weight`, `grad`, and `momentum`, uses DSMEM for the two scalar norm
  reductions, then writes its local output slice.
- `torch.compile`: compiled PyTorch expression returning the same
  `(weight_out, momentum_out)` tuple.

Throughput uses a 32 B/element model for the logical block-read path: one
three-stream read pass for the two norm reductions, then one three-stream read
pass plus two output writes.

## Timing Results

Command:

```bash
bash lars_momentum_dsmem_explore/run_compare.sh \
  --rows 4096 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv lars_momentum_dsmem_explore/results/lars_momentum_torch_compare.csv
```

| N | block variant | block ms | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | block_smem | 0.2155 | 0.2814 | 0.2271 | 0.766x | 0.807x |
| 8192 | block_smem | 0.4328 | 0.4331 | 0.4447 | 0.999x | 1.027x |
| 16384 | block_read | 1.3826 | 0.8636 | 0.8896 | 1.601x | 1.030x |
| 32768 | block_read | 2.7905 | 1.7283 | 1.7781 | 1.615x | 1.029x |
| 65536 | block_read | 5.6026 | 3.4558 | 4.5967 | 1.621x | 1.330x |

The DSMEM cluster kernel beats `torch.compile` on 4 of 5 shapes. The best
`torch.compile` speedup is `1.330x` at `N=65536`.

## Nsight Compute

Command:

```bash
bash lars_momentum_dsmem_explore/ncu_lars_momentum_metrics.sh
python3 lars_momentum_dsmem_explore/parse_lars_momentum_ncu.py
```

Profiled shape: `rows=4096`, `N=65536`.

| variant | NCU time (us) | modeled GB/s | L2 B/element | DRAM B/element | DRAM read B/element | DRAM write B/element |
|---|---:|---:|---:|---:|---:|---:|
| block | 5626.208 | 1526.772 | 32.006 | 32.002 | 24.004 | 7.998 |
| cluster | 3449.280 | 2490.356 | 20.189 | 19.982 | 12.000 | 7.982 |

The DSMEM path reduces DRAM traffic from `32.0` to `20.0 B/element` and reduces
NCU kernel time by `1.631x` on the profiled shape. DSMEM removes the second
read of `weight`, `grad`, and `momentum`; the two output writes are unchanged.

## Verdict

LARS momentum is a positive optimizer-style DSMEM case. It is not a
normalization or softmax derivative, but it has the same profitable shape:
wide row-wise scalar reductions followed by a full-row update that reuses all
staged inputs. The two output streams cap the gain, but the cluster path still
beats `torch.compile` for wide rows.

Artifacts:

- `results/lars_momentum_torch_compare.csv`
- `results/lars_momentum_ncu_summary.csv`
- `plots/lars_momentum_speedup.png`
- `plots/lars_momentum_runtime_ms.png`
- `plots/lars_momentum_gbps.png`
- `plots/lars_momentum_ncu_metrics.png`
