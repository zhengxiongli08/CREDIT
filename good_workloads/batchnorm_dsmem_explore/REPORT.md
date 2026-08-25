# BatchNorm Training Forward DSMEM Exploration

## Workload

This folder benchmarks BatchNorm-style training forward on an NCHW-like tensor
stored as `[batch, channels, spatial]`. For each channel, the kernel reduces
over `batch * spatial` elements, then writes:

```text
y = (x - mean(x_c)) * rsqrt(var(x_c) + eps) * gamma_c + beta_c
```

The default sweep uses `batch=32`, `channels=256`, and
`elems_per_channel in {4096, 8192, 16384, 32768, 65536}`.

## Implementations

- `block_read`: one CTA per channel. It reads `x` once for `sum/sumsq`, then
  rereads `x` to write the normalized output.
- `block_smem`: one CTA per channel with local shared-memory staging when the
  whole channel slice fits. This was not the fastest default-sweep variant.
- `cluster_smem`: one 8-CTA cluster per channel. Each CTA stages its channel
  slice in local shared memory, DSMEM exchanges only two scalar partials
  (`sum`, `sumsq`), and each CTA writes its local output slice.
- `torch.compile`: compiled PyTorch expression reducing `x.mean(dim=(0, 2))`
  and applying affine normalization.

Throughput uses a 12 B/element model for the non-DSMEM logical traffic:
two `x` reads plus one output write. This keeps speedup ratios equivalent to
runtime ratios.

## Timing Results

Command:

```bash
bash batchnorm_dsmem_explore/run_compare.sh \
  --batch-size 32 \
  --channels 256 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv batchnorm_dsmem_explore/results/batchnorm_torch_compare.csv
```

| elems/channel | block variant | block ms | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | block_read | 0.006803 | 0.016749 | 0.055962 | 0.406x | 3.341x |
| 8192 | block_read | 0.012288 | 0.020390 | 0.042112 | 0.603x | 2.065x |
| 16384 | block_read | 0.022579 | 0.031002 | 0.091763 | 0.728x | 2.960x |
| 32768 | block_read | 0.041786 | 0.051718 | 0.072026 | 0.808x | 1.393x |
| 65536 | block_read | 0.134406 | 0.104794 | 0.138298 | 1.283x | 1.320x |

The custom CUDA kernels beat `torch.compile` on all five shapes. The DSMEM
cluster kernel beats the non-DSMEM block kernel only at the widest shape.

## Nsight Compute

Command:

```bash
bash batchnorm_dsmem_explore/ncu_batchnorm_metrics.sh
python3 batchnorm_dsmem_explore/parse_batchnorm_ncu.py
```

Profiled shape: `batch=32`, `channels=256`, `elems/channel=65536`.

| variant | NCU time (us) | L2 B/element | DRAM B/element | DRAM read B/element | DRAM write B/element |
|---|---:|---:|---:|---:|---:|
| block | 148.576 | 12.028 | 7.264 | 3.787 | 3.478 |
| cluster | 109.600 | 19.396 | 7.786 | 4.005 | 3.781 |

The DSMEM path is faster on the profiled shape, but this is not a DRAM-traffic
reduction result. The block reread is mostly served from cache, while the
cluster path has higher L2 traffic and slightly higher DRAM bytes per element.

## Verdict

BatchNorm training forward is a conditional positive DSMEM case. DSMEM helps at
the widest per-channel reduction because the channel is split across a cluster,
so each CTA owns less serial per-channel work and the cross-CTA reduction is
only two scalars. Profiling shows this is a parallelism/latency benefit rather
than the clean memory-traffic reduction seen in LayerNorm backward.

Artifacts:

- `results/batchnorm_torch_compare.csv`
- `results/batchnorm_ncu_summary.csv`
- `plots/batchnorm_speedup_vs_torch.png`
- `plots/batchnorm_runtime_ms.png`
- `plots/batchnorm_gbps.png`
- `plots/batchnorm_ncu_metrics.png`
