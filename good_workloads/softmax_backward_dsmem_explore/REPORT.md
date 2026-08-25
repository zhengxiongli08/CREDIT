# DSMEM for Large Row-wise Softmax Backward

## Goal

This experiment tests whether DSMEM helps the `dx` part of softmax backward
when the saved softmax probabilities are already available:

```text
dot = sum(y * dy)
dx  = y * (dy - dot)
```

The default external baseline is `torch.compile` running the equivalent PyTorch
expression. A non-cluster CUDA block kernel is kept as an internal noDSMEM
baseline.

## Why This Workload Is a DSMEM Candidate

Softmax backward needs one row-level reduction and then writes one output per
element. A simple block implementation reads `y` and `dy` once to compute
`sum(y * dy)`, then rereads both inputs to write `dx`.

The DSMEM cluster implementation splits one row across an 8-CTA cluster. Each
CTA stages a contiguous slice of `y` and `dy` in local shared memory, computes a
local dot-product partial, uses DSMEM for the cross-CTA scalar reduction, and
writes its own `dx` slice. DSMEM is used only for scalar partials, not for
fine-grained remote element reads.

## Files

Folder: `softmax_backward_dsmem_explore`

- `softmax_backward_bench.cu`: CUDA block and DSMEM cluster kernels.
- `run_softmax_backward.sh`: builds and runs the CUDA benchmark.
- `compare_softmax_backward_torch.py`: compares CUDA results against
  `torch.compile`.
- `run_compare.sh`: wrapper using the `cluster` conda environment.
- `ncu_softmax_backward_metrics.sh`: focused Nsight Compute profile for
  `N=65536`.
- `parse_softmax_backward_ncu.py`: converts raw NCU CSV into a compact summary.
- `plot_softmax_backward_results.py`: generates timing and profiler plots.

The CUDA benchmark performs sample-based correctness checks by default. The NCU
script passes `--no-verify` so profiler captures only the target kernels.

## Timing Method

Command:

```bash
bash softmax_backward_dsmem_explore/run_compare.sh \
  --rows 8192 \
  --n-values 4096,8192,16384,32768,65536 \
  --warmup 2 \
  --iters 5 \
  --cluster-size 8 \
  --output-csv softmax_backward_dsmem_explore/results/softmax_backward_torch_compare.csv
```

Environment:

- GPU: NVIDIA GeForce RTX 5090
- PyTorch: `2.11.0+cu130`
- CUDA arch: `sm_120`
- rows: `8192`
- cluster size: `8`
- block candidates: `block_read` and `block_smem`

The bandwidth model counts the natural noDSMEM traffic: read `y` and `dy` for
the dot product, reread `y` and `dy` for output, then write `dx`, or
`20 bytes/element`. Since all variants use the same model, speedup is equivalent
to runtime speedup.

## Timing Results

| N | block ms | block variant | cluster ms | torch.compile ms | cluster/block | cluster/torch |
|---:|---:|---|---:|---:|---:|---:|
| 4096 | 0.2550 | block_smem | 0.4656 | 0.2636 | 0.548x | 0.566x |
| 8192 | 0.5090 | block_smem | 0.5097 | 0.5247 | 0.999x | 1.029x |
| 16384 | 1.6782 | block_read | 1.0156 | 1.0293 | 1.652x | 1.013x |
| 32768 | 3.4349 | block_read | 2.0399 | 2.0841 | 1.684x | 1.022x |
| 65536 | 6.8951 | block_read | 4.0838 | 6.4121 | 1.688x | 1.570x |

DSMEM beats `torch.compile` on 4 of 5 shapes. The strongest measured case is
`N=65536`:

```text
torch.compile: 6.4121 ms
DSMEM cluster: 4.0838 ms
speedup:       1.570x
```

## NCU Metrics

Command:

```bash
bash softmax_backward_dsmem_explore/ncu_softmax_backward_metrics.sh
python3 softmax_backward_dsmem_explore/parse_softmax_backward_ncu.py
```

Profiled case: `rows=8192`, `N=65536`, one captured launch per variant.

| variant | time us | modeled GB/s | L2 B/elem | DRAM B/elem | DRAM read B/elem | DRAM write B/elem |
|---|---:|---:|---:|---:|---:|---:|
| block | 6889.568 | 1558.504 | 20.002 | 19.999 | 16.001 | 3.998 |
| cluster | 4092.800 | 2623.490 | 12.181 | 11.998 | 8.000 | 3.998 |

The profiler confirms the intended traffic reduction. The noDSMEM block path
rereads both inputs and uses about `20.0 B/element` of DRAM traffic. The DSMEM
path stages each CTA's local slice once and uses about `12.0 B/element`, with
read traffic dropping from about `16.0 B/element` to `8.0 B/element`.

## Interpretation

Softmax backward is a useful middle case. It has fewer reductions than
LayerNorm backward, so the maximum speedup is smaller, but it still has enough
row reuse to make DSMEM profitable for wide rows. The break-even point is around
`N=8192`; below that, cluster overhead dominates, while at `N=65536` the avoided
global rereads are large enough to beat both the noDSMEM CUDA path and
`torch.compile`.
