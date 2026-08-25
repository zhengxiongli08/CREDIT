# Gradient-Centralized Clipped Momentum DSMEM Exploration

## Workload

This folder benchmarks a row-wise optimizer-style update with dependent
gradient reductions:

```text
grad_mean = mean(grad)
centered = grad - grad_mean
norm = sqrt(sum(centered * centered))
scale = min(1, clip_threshold / (norm + eps))
update = momentum_coeff * momentum + scale * centered + weight_decay * weight
weight_out = weight - learning_rate * update
momentum_out = update
```

The noDSMEM read baseline reads `grad` for the mean, rereads `grad` for the
centered norm, then rereads `weight`, `grad`, and `momentum` for the output
phase. Its modeled traffic is `7 floats/element`: 5 input reads plus 2 output
writes.

The DSMEM cluster path splits each wide row across 8 CTAs. Each CTA stages its
slice of `weight`, `grad`, and `momentum` in local shared memory, uses DSMEM
only for the scalar row reductions, and writes its local output slice. Its
ideal element traffic is `5 floats/element`: 3 input reads plus 2 output
writes.

## Timing

Command:

```bash
bash grad_centralized_clip_dsmem_explore/run_compare.sh \
  --rows 4096 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv grad_centralized_clip_dsmem_explore/results/grad_centralized_clip_torch_compare.csv
```

Baseline: `torch.compile` on torch `2.11.0+cu130`, NVIDIA GeForce RTX 5090.

| N | Best noDSMEM | DSMEM cluster | torch.compile | DSMEM / torch | DSMEM / noDSMEM |
|---:|---:|---:|---:|---:|---:|
| 4096 | 2177.21 GB/s | 1296.64 GB/s | 2074.33 GB/s | 0.625x | 0.596x |
| 8192 | 2176.69 GB/s | 2213.85 GB/s | 2114.40 GB/s | 1.047x | 1.017x |
| 16384 | 1593.46 GB/s | 2170.90 GB/s | 2144.81 GB/s | 1.012x | 1.362x |
| 32768 | 1513.18 GB/s | 2172.63 GB/s | 2160.47 GB/s | 1.006x | 1.436x |
| 65536 | 1499.00 GB/s | 2173.93 GB/s | 1615.54 GB/s | 1.346x | 1.450x |

The cluster kernel beats `torch.compile` on 4 of 5 tested shapes. The best
speedup is `1.346x` at `N=65536`. The small `N=4096` shape is still better with
one-block local shared-memory staging because cluster launch and synchronization
overhead dominates before the row is wide enough.

## Nsight Compute

Command:

```bash
bash grad_centralized_clip_dsmem_explore/ncu_grad_centralized_clip_metrics.sh
python3 grad_centralized_clip_dsmem_explore/parse_grad_centralized_clip_ncu.py
```

Profiled shape: `rows=4096`, `N=65536`, one measured launch per variant.

| Variant | Time | Modeled GB/s | L2 B/elem | DRAM B/elem | DRAM read B/elem | DRAM write B/elem |
|---|---:|---:|---:|---:|---:|---:|
| block read | 5018.528 us | 1497.69 | 28.007 | 27.998 | 20.004 | 7.994 |
| DSMEM cluster | 3452.224 us | 2177.20 | 20.189 | 19.985 | 12.000 | 7.985 |

NCU matches the traffic model closely. The block-read path lands at about
`28 B/element`: three input read phases plus two full output writes. The DSMEM
path lands at about `20 B/element`: staged reads of `weight`, `grad`, and
`momentum`, plus unchanged output writes.

## Verdict

Gradient-centralized clipped momentum is a positive optimizer-style DSMEM case.
It makes the second reduction depend on the first reduction result
(`centered = grad - mean`), but DSMEM still helps because the same staged
`grad` slice is reused across both reductions and the output pass.

The case also shows the same practical crossover as other row-wise DSMEM
experiments: one-block shared-memory staging wins on small rows, but DSMEM wins
once the row is too wide for a single CTA to stage the whole row.

## Artifacts

- `grad_centralized_clip_bench.cu`
- `compare_grad_centralized_clip_torch.py`
- `results/grad_centralized_clip_torch_compare.csv`
- `results/grad_centralized_clip_ncu_summary.csv`
- `plots/grad_centralized_clip_speedup.png`
- `plots/grad_centralized_clip_runtime_ms.png`
- `plots/grad_centralized_clip_gbps.png`
- `plots/grad_centralized_clip_ncu_metrics.png`
