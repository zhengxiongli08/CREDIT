# Soft-Target Cross-Entropy Backward DSMEM Exploration

## Workload

This folder benchmarks the row-wise gradient for a soft-target cross-entropy
loss from logits:

```text
p = softmax(logits)
dx = p - target_probs
```

The default sweep uses `rows=4096` and
`N in {4096, 8192, 16384, 32768, 65536}`. `target_probs` is a dense input
stream initialized to a uniform row distribution; the kernel still reads it as
a normal input tensor.

## Implementations

- `block_read`: one CTA per row. It reads logits once for row max, rereads
  logits for the softmax denominator, then rereads logits and reads
  `target_probs` to write `dx`.
- `block_smem`: one CTA per row with local shared-memory staging of logits and
  `target_probs` when the full row fits.
- `cluster_smem`: one 8-CTA cluster per row. Each CTA stages local slices of
  logits and `target_probs`, uses DSMEM only for row max and denominator
  reductions, then writes its local `dx` slice.
- `torch.compile`: compiled PyTorch expression for
  `torch.softmax(logits, dim=-1) - target_probs`.

Throughput uses a 20 B/element model for the logical block-read path:
three float logits reads, one `target_probs` read, and one float output write.

## Timing Results

Command:

```bash
bash soft_target_ce_backward_dsmem_explore/run_compare.sh \
  --rows 4096 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv soft_target_ce_backward_dsmem_explore/results/soft_target_ce_backward_torch_compare.csv
```

| N | block variant | block ms | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | block_smem | 0.1191 | 0.3190 | 0.1201 | 0.373x | 0.376x |
| 8192 | block_smem | 0.2542 | 0.3082 | 0.2640 | 0.825x | 0.856x |
| 16384 | block_read | 0.7334 | 0.5067 | 0.5207 | 1.447x | 1.028x |
| 32768 | block_read | 1.7112 | 1.0196 | 1.0325 | 1.678x | 1.013x |
| 65536 | block_read | 3.4539 | 2.0180 | 2.6652 | 1.712x | 1.321x |

The DSMEM cluster kernel beats `torch.compile` on 3 of 5 shapes. The best
`torch.compile` speedup is `1.321x` at `N=65536`.

## Nsight Compute

Command:

```bash
bash soft_target_ce_backward_dsmem_explore/ncu_soft_target_ce_backward_metrics.sh
python3 soft_target_ce_backward_dsmem_explore/parse_soft_target_ce_backward_ncu.py
```

Profiled shape: `rows=4096`, `N=65536`.

| variant | NCU time (us) | modeled GB/s | L2 B/element | DRAM B/element | DRAM read B/element | DRAM write B/element |
|---|---:|---:|---:|---:|---:|---:|
| block | 3459.168 | 1552.023 | 20.004 | 19.999 | 15.999 | 3.999 |
| cluster | 2011.008 | 2669.661 | 12.184 | 11.990 | 8.001 | 3.989 |

The DSMEM path reduces DRAM traffic from `20.0` to `12.0 B/element` and reduces
NCU kernel time by `1.720x` on the profiled shape. Output traffic and the
single `target_probs` read are unchanged; DSMEM removes two extra global reads
of logits across the max, denominator, and output phases.

## Verdict

Soft-target cross-entropy backward is a positive DSMEM case for wide rows. It
extends the hard-label cross-entropy backward result to dense target
distributions: the target stream adds unavoidable traffic, but staging logits
once still removes enough reread traffic for the cluster path to beat
`torch.compile` at wide `N`.

Artifacts:

- `results/soft_target_ce_backward_torch_compare.csv`
- `results/soft_target_ce_backward_ncu_summary.csv`
- `plots/soft_target_ce_backward_speedup.png`
- `plots/soft_target_ce_backward_runtime_ms.png`
- `plots/soft_target_ce_backward_gbps.png`
- `plots/soft_target_ce_backward_ncu_metrics.png`
