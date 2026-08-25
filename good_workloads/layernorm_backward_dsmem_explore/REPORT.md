# DSMEM for Large Row-wise LayerNorm Backward

## Goal

This experiment tests whether DSMEM helps the `dx` part of LayerNorm backward on
very wide rows:

```text
dyg = dy * gamma
xhat = (x - mean(x)) * inv_std(x)
dx = (dyg - mean(dyg) - xhat * mean(dyg * xhat)) * inv_std(x)
```

The default external baseline is `torch.compile` running the equivalent PyTorch
expression. A non-cluster CUDA block kernel is kept as an internal noDSMEM
baseline.

## Why This Workload Is a DSMEM Candidate

LayerNorm backward needs four row-level reductions before writing the full row:

1. `sum(x)`
2. `sum(x * x)`
3. `sum(dy * gamma)`
4. `sum(dy * gamma * xhat)`

The block implementation rereads the row for the statistics and output phases.
The DSMEM cluster implementation splits one row across an 8-CTA cluster. Each
CTA stages its contiguous `x` slice in local shared memory, computes local
partials, uses DSMEM for compact cross-CTA scalar reductions, and writes only
its own `dx` slice.

This is the same coarse-grained DSMEM pattern that worked for the positive
softmax/RMSNorm-style cases: element traffic stays local, while DSMEM carries
only scalar partials.

## Files

Folder: `layernorm_backward_dsmem_explore`

- `layernorm_backward_bench.cu`: CUDA block and DSMEM cluster kernels.
- `run_layernorm_backward.sh`: builds and runs the CUDA benchmark.
- `compare_layernorm_backward_torch.py`: compares CUDA results against
  `torch.compile`.
- `run_compare.sh`: wrapper using the `cluster` conda environment.
- `ncu_layernorm_backward_metrics.sh`: focused Nsight Compute profile for
  `N=65536`.
- `parse_layernorm_backward_ncu.py`: converts raw NCU CSV into a compact
  summary.
- `plot_layernorm_backward_results.py`: generates timing and profiler plots.

The CUDA benchmark performs sample-based correctness checks by default. The NCU
script passes `--no-verify` so profiler captures only the target kernels.

## Timing Method

Command:

```bash
bash layernorm_backward_dsmem_explore/run_compare.sh \
  --rows 2048 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv layernorm_backward_dsmem_explore/results/layernorm_backward_torch_compare.csv
```

Environment:

- GPU: NVIDIA GeForce RTX 5090
- PyTorch: `2.11.0+cu130`
- CUDA arch: `sm_120`
- rows: `2048`
- cluster size: `8`
- block candidates: `block_read` and `block_smem`

The bandwidth model counts four logical row reads plus `dy`, `gamma`, and `dx`
traffic, or `32 bytes/element`. Since all variants use the same model, speedup
is equivalent to runtime speedup.

## Timing Results

| N | block ms | block variant | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---:|---|---:|---:|---:|---:|
| 4096 | 0.0293 | block_read | 0.1838 | 0.0557 | 0.160x | 0.303x |
| 8192 | 0.1349 | block_smem | 0.2059 | 0.1358 | 0.655x | 0.660x |
| 16384 | 0.4178 | block_smem | 0.2847 | 0.5523 | 1.468x | 1.940x |
| 32768 | 1.0163 | block_read | 0.5044 | 1.2993 | 2.015x | 2.576x |
| 65536 | 2.0492 | block_read | 1.0208 | 2.6280 | 2.007x | 2.574x |

DSMEM wins once the row is wide enough. It beats `torch.compile` on 3 of 5
shapes, with the best measured speedup at `N=32768`:

```text
torch.compile: 1.2993 ms
DSMEM cluster: 0.5044 ms
speedup:       2.576x
```

## NCU Metrics

Command:

```bash
bash layernorm_backward_dsmem_explore/ncu_layernorm_backward_metrics.sh
python3 layernorm_backward_dsmem_explore/parse_layernorm_backward_ncu.py
```

Profiled case: `rows=2048`, `N=65536`, one captured launch per variant.

| variant | time us | modeled GB/s | L2 B/elem | DRAM B/elem | DRAM read B/elem | DRAM write B/elem |
|---|---:|---:|---:|---:|---:|---:|
| block | 2049.056 | 2096.071 | 29.629 | 23.996 | 19.998 | 3.998 |
| cluster | 1006.144 | 4268.740 | 24.222 | 11.981 | 8.001 | 3.981 |

The profiler explains the speedup: the DSMEM path cuts DRAM traffic from about
`24.0 B/element` to about `12.0 B/element`, mainly by reducing read traffic from
about `20.0 B/element` to about `8.0 B/element`. The output write traffic is
unchanged, as expected.

## Interpretation

LayerNorm backward is a stronger positive DSMEM case than forward LayerNorm in
this implementation. Forward LayerNorm needs two row statistics and one full-row
write; backward needs four row reductions and can reuse staged `x` across more
phases. The wider rows amortize cluster launch/synchronization overhead and make
the avoided global rereads large enough to beat both the noDSMEM CUDA baseline
and `torch.compile`.

The boundary is also visible: at `N=4096` and `N=8192`, cluster overhead is still
too large. DSMEM only becomes attractive when the row is wide enough for the
traffic reduction to dominate.
