# Softmax Weighted Sum Backward DSMEM Exploration

## Workload

This folder benchmarks the full backward pass for a scalar softmax weighted
sum:

```text
p = softmax(scores)
out = sum(p * values)
dscores = dout * p * (values - out)
dvalues = dout * p
```

The default sweep uses `rows=4096` and
`N in {4096, 8192, 16384, 32768, 65536}`.

## Implementations

- `block_read`: one CTA per row. It reads `scores` for the max reduction,
  rereads `scores` and `values` for the denominator and weighted numerator,
  then rereads both inputs again to write `dscores` and `dvalues`.
- `block_smem`: one CTA per row with local shared-memory staging of `scores`
  and `values` when the whole row fits.
- `cluster_smem`: one 8-CTA cluster per row. Each CTA stages a row slice of
  `scores` and `values`, uses DSMEM only for the row max and `(denom,
  numerator)` reductions, and writes its local output slices.
- `torch.compile`: compiled PyTorch expression returning the same
  `(dscores, dvalues)` tuple.

Throughput uses a 28 B/element model for the logical no-DSMEM read path:
one `scores` read, then a `scores/values` read, then another `scores/values`
read plus `dscores/dvalues` writes.

## Timing Results

Command:

```bash
bash softmax_weighted_sum_backward_dsmem_explore/run_compare.sh \
  --rows 4096 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv softmax_weighted_sum_backward_dsmem_explore/results/weighted_sum_backward_torch_compare.csv
```

| N | block variant | block ms | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | block_smem | 0.176685 | 0.362086 | 0.184896 | 0.488x | 0.511x |
| 8192 | block_smem | 0.351994 | 0.381094 | 0.362976 | 0.924x | 0.952x |
| 16384 | block_read | 1.198670 | 0.700826 | 0.796090 | 1.710x | 1.136x |
| 32768 | block_read | 2.454710 | 1.405910 | 2.220410 | 1.746x | 1.579x |
| 65536 | block_read | 4.946490 | 2.817170 | 4.741273 | 1.756x | 1.683x |

The DSMEM cluster kernel beats `torch.compile` on 3 of 5 shapes. It loses
while a single CTA can stage the row locally, then wins once row width forces
the no-DSMEM block path to reread global memory.

## Nsight Compute

Command:

```bash
ROWS=4096 bash softmax_weighted_sum_backward_dsmem_explore/ncu_weighted_sum_backward_metrics.sh
ROWS=4096 python3 softmax_weighted_sum_backward_dsmem_explore/parse_weighted_sum_backward_ncu.py
```

Profiled shape: `rows=4096`, `N=65536`.

| variant | NCU time (us) | L2 B/element | DRAM B/element | DRAM read B/element | DRAM write B/element |
|---|---:|---:|---:|---:|---:|
| block | 4959.392 | 28.005 | 27.993 | 20.001 | 7.992 |
| cluster | 2796.640 | 16.192 | 15.954 | 8.001 | 7.954 |

The DSMEM path reduces DRAM traffic from `28.0` to `16.0 B/element` and
reduces NCU kernel time by `1.773x` on the profiled shape. Output traffic is
unchanged; DSMEM removes repeated input reads by keeping `scores` and `values`
staged locally across all reduction and output phases.

## Verdict

Softmax weighted-sum backward is a positive DSMEM case. It is more useful than
the forward scalar weighted-sum case because it writes full-row gradients and
therefore reuses the staged row slices for meaningful output work.

Artifacts:

- `results/weighted_sum_backward_torch_compare.csv`
- `results/weighted_sum_backward_ncu_summary.csv`
- `plots/weighted_sum_backward_speedup.png`
- `plots/weighted_sum_backward_runtime_ms.png`
- `plots/weighted_sum_backward_gbps.png`
- `plots/weighted_sum_backward_ncu_metrics.png`
