# Row-Wise Top-4 DSMEM Exploration

## Workload

This folder benchmarks row-wise `topk` for `k=4`:

```text
values[row, :], indices[row, :] = topk(x[row, :], k=4)
```

Outputs are sorted by descending value with a lowest-index tie break. The
default sweep uses `rows=8192` and
`N in {4096, 8192, 16384, 32768, 65536}`.

## Implementations

- `block_top4`: one CTA per row. Each thread keeps a local top-4, warp-level
  reductions merge candidates, and one CTA writes four values plus four
  indices.
- `cluster_top4`: one 8-CTA cluster per row. Each CTA scans one row slice and
  keeps a local top-4; DSMEM exchanges `4 * cluster_size` compact partial
  pairs for the final merge.
- `torch.compile`: compiled PyTorch `torch.topk(x, 4, dim=-1, sorted=True)`.

Throughput uses a model of one input read plus the compact top-4 value/index
output.

## Timing Results

Command:

```bash
bash topk_dsmem_explore/run_compare.sh \
  --rows 8192 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv topk_dsmem_explore/results/topk_torch_compare.csv
```

| N | block ms | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---:|---:|---:|---:|---:|
| 4096 | 0.378214 | 2.518413 | 0.615686 | 0.150x | 0.244x |
| 8192 | 0.444224 | 2.790771 | 1.061517 | 0.159x | 0.380x |
| 16384 | 0.547014 | 3.164525 | 1.929459 | 0.173x | 0.610x |
| 32768 | 0.813286 | 3.632698 | 3.633888 | 0.224x | 1.000x |
| 65536 | 1.489146 | 4.571085 | 7.070605 | 0.326x | 1.547x |

The DSMEM cluster kernel beats `torch.compile` on the two largest rows, but it
is much slower than the non-DSMEM custom block implementation across the entire
sweep.

## Nsight Compute

Command:

```bash
bash topk_dsmem_explore/ncu_topk_metrics.sh
python3 topk_dsmem_explore/parse_topk_ncu.py
```

Profiled shape: `rows=8192`, `N=65536`.

| variant | NCU time (us) | L2 B/element | DRAM B/element | DRAM read B/element | DRAM write B/element |
|---|---:|---:|---:|---:|---:|
| block | 1489.248 | 7.812 | 4.304 | 4.050 | 0.254 |
| cluster | 4599.744 | 26.302 | 4.015 | 4.001 | 0.014 |

The DSMEM path is close to the ideal one input read in DRAM traffic, but its L2
traffic and runtime are much higher. The block path writes more compact output
traffic and shows more L2 activity than a pure streaming read, but it is still
about `3.1x` faster than the cluster kernel in NCU time.

## Verdict

Row-wise top-4 is a mixed result. Relative to the requested PyTorch baseline,
the DSMEM cluster kernel is a positive case for very wide rows, reaching
`1.547x` over `torch.compile` at `N=65536`. Relative to an internal custom CUDA
baseline, DSMEM is not the right implementation: the local block top-4 is
substantially faster.

This strengthens the boundary from argmax. Exchanging a few compact selection
partials through DSMEM can beat PyTorch's generic `topk` for large rows, but it
does not prove DSMEM is the source of the best implementation. For selection
workloads without repeated row-data reuse, an efficient single-CTA streaming
kernel can be superior.

Artifacts:

- `results/topk_torch_compare.csv`
- `results/topk_ncu_summary.csv`
- `plots/topk_speedup_vs_torch.png`
- `plots/topk_runtime_ms.png`
- `plots/topk_gbps.png`
- `plots/topk_ncu_metrics.png`
