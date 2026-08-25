# BatchNorm Backward-Dx DSMEM Exploration

## Workload

This folder benchmarks BatchNorm-style backward input-gradient (`dx`) on an
NCHW-like tensor stored as `[batch, channels, spatial]`. For each channel, the
kernel reduces over `batch * spatial` elements:

```text
mean = mean(x_c)
inv_std = rsqrt(var(x_c) + eps)
xhat = (x - mean) * inv_std
dyg = dy * gamma_c
dx = (dyg - mean(dyg) - xhat * mean(dyg * xhat)) * inv_std
```

The default sweep uses `batch=32`, `channels=256`, and
`elems_per_channel in {4096, 8192, 16384, 32768, 65536}`.

## Implementations

- `block_read`: one CTA per channel. It reads `x` for channel statistics,
  rereads `x` and `dy` for gradient statistics, then rereads them again to
  write `dx`.
- `block_smem`: one CTA per channel with local shared-memory staging of both
  `x` and `dy` when the whole channel slice fits.
- `cluster_smem`: one 8-CTA cluster per channel. Each CTA stages a slice of
  `x` and `dy` in local shared memory, uses DSMEM only for scalar reductions
  (`sum/sumsq` and `sum(dyg)/sum(dyg*xhat)`), and writes its local output
  slice.
- `torch.compile`: compiled PyTorch expression for the same `dx` formula.

Throughput uses a 24 B/element model: three logical `x` reads, two logical
`dy` reads, and one `dx` write. Gamma traffic is one scalar per channel and is
not counted per element.

## Timing Results

Command:

```bash
bash batchnorm_backward_dsmem_explore/run_compare.sh \
  --batch-size 32 \
  --channels 256 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv batchnorm_backward_dsmem_explore/results/batchnorm_backward_torch_compare.csv
```

| elems/channel | block variant | block ms | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | block_smem | 0.006406 | 0.026560 | 0.058592 | 0.241x | 2.206x |
| 8192 | block_smem | 0.012320 | 0.030752 | 0.045331 | 0.401x | 1.474x |
| 16384 | block_read | 0.039226 | 0.043757 | 0.095706 | 0.896x | 2.187x |
| 32768 | block_read | 0.076256 | 0.074298 | 0.097395 | 1.026x | 1.311x |
| 65536 | block_read | 0.306176 | 0.209952 | 0.278451 | 1.458x | 1.326x |

The cluster DSMEM kernel beats `torch.compile` on all tested shapes. It beats
the best non-DSMEM CUDA block path once the channel reduction is wide enough
that a single CTA becomes the bottleneck.

## Nsight Compute

Command:

```bash
bash batchnorm_backward_dsmem_explore/ncu_batchnorm_backward_metrics.sh
python3 batchnorm_backward_dsmem_explore/parse_batchnorm_backward_ncu.py
```

Profiled shape: `batch=32`, `channels=256`, `elems/channel=65536`.

| variant | NCU time (us) | L2 B/element | DRAM B/element | DRAM read B/element | DRAM write B/element |
|---|---:|---:|---:|---:|---:|
| block | 311.968 | 24.045 | 20.711 | 17.067 | 3.644 |
| cluster | 212.928 | 22.988 | 12.086 | 8.005 | 4.081 |

The DSMEM path cuts DRAM traffic from `20.7` to `12.1 B/element` and reduces
NCU kernel time by `1.465x` on the profiled shape. This matches the intended
DSMEM pattern: stage element data once locally, exchange only compact scalar
partials through distributed shared memory, and avoid later global rereads.

## Verdict

BatchNorm backward dx is a positive DSMEM case. It extends the LayerNorm
backward result from row-wise contiguous reductions to channel-wise reductions
over `[batch, spatial]`. The benefit appears only once each channel has enough
elements; for smaller channel reductions, the local shared-memory block variant
is faster because it avoids cluster launch and synchronization overhead.

Artifacts:

- `results/batchnorm_backward_torch_compare.csv`
- `results/batchnorm_backward_ncu_summary.csv`
- `plots/batchnorm_backward_speedup_vs_torch.png`
- `plots/batchnorm_backward_runtime_ms.png`
- `plots/batchnorm_backward_gbps.png`
- `plots/batchnorm_backward_ncu_metrics.png`
