# Reduction Scope Benchmark

This folder now contains three main CUDA benchmark entrypoints:

- `benchmark.cu`: simple row-sum reduction scope comparison
- `workload_bench.cu`: wider workloads that use reduction as a subroutine
- `markdown_workload_bench.cu`: a markdown-style CUDA benchmark that adapts the QuACK ideas into our own kernels
- `compare_torch_compile.py`: compares the custom workload kernels against `torch.compile`
- `compare_markdown_style.py`: compares `markdown_workload_bench.cu` against `torch.compile`

The simple reduction benchmark compares four row-wise reduction strategies on the same input data:

- `thread`: one thread reduces one row by itself
- `warp`: one warp reduces one row with shuffle instructions
- `block`: one thread block reduces one row with shared-memory aggregation
- `cluster`: one thread-block cluster reduces one row, combining block partials through DSMEM

This is meant to isolate the effect of the reduction scope itself. It is not a fully tuned softmax benchmark like the QuACK writeup.

The workload benchmark focuses on kernels closer to the markdown note:

- `rmsnorm`
- `softmax`
- `cross_entropy`

For these workloads, the comparison is `block` versus `cluster`, since the interesting question is whether DSMEM helps once reduction becomes only part of a larger memory-bound kernel.

## Why this benchmark

The markdown note in `quack/media/2025-07-10-membound-sol.md` discusses reduction moving through the hardware hierarchy:

- thread/register
- warp/register shuffle
- block/shared memory
- cluster/distributed shared memory

This benchmark follows that same progression for a simpler operation: row-wise sum.

## Build and run

From the repo root:

```bash
./reduction_test/run.sh
```

To run the workload suite:

```bash
./reduction_test/run_workloads.sh
```

To compare the custom workload kernels against `torch.compile`:

```bash
./reduction_test/run_compare_torch.sh
```

To run the markdown-style CUDA benchmark directly:

```bash
./reduction_test/run_markdown_workloads.sh --workload softmax --batch-size 16384
```

To compare the markdown-style CUDA benchmark against `torch.compile`:

```bash
./reduction_test/run_markdown_compare.sh --workload softmax --batch-size 16384
```

Useful overrides:

```bash
./reduction_test/run.sh --iters 40 --warmup 8
./reduction_test/run.sh --target-mib 256
./reduction_test/run.sh --cluster-size 8
./reduction_test/run.sh --no-verify
```

Useful workload overrides:

```bash
./reduction_test/run_workloads.sh --iters 16 --warmup 4
./reduction_test/run_workloads.sh --workload softmax
./reduction_test/run_workloads.sh --cluster-size 4
./reduction_test/run_workloads.sh --no-verify
```

Useful `torch.compile` comparison overrides:

```bash
./reduction_test/run_compare_torch.sh --workload softmax
./reduction_test/run_compare_torch.sh --cluster-size 8
TORCH_ENV=quack ./reduction_test/run_compare_torch.sh
```

Useful markdown-style overrides:

```bash
./reduction_test/run_markdown_compare.sh --workload softmax --batch-size 16384
./reduction_test/run_markdown_compare.sh --workload rmsnorm --batch-size 16384
./reduction_test/run_markdown_compare.sh --workload all --batch-size 8192
./reduction_test/run_markdown_compare.sh --n-values 32768,65536,131072
./reduction_test/run_markdown_compare.sh --cluster-size 8 --thread-per-row-values 16,32,256
```

The script defaults to `ARCH=sm_120`, which matches an RTX 5090. If you want to override that:

```bash
ARCH=sm_120 ./reduction_test/run.sh
```

## Output

The simple reduction benchmark prints:

- device and cluster support info
- a table of effective bandwidth for each reduction scope across row lengths from `256` to `262144`
- `cluster/block`, which is the direct speedup ratio for DSMEM-backed cluster reduction relative to block reduction
- a final verdict explaining whether DSMEM helped on this workload

The workload benchmark prints the same style of table for each workload, with modeled throughput for:

- `block`
- `cluster`
- `cluster/block`

The `torch.compile` comparison prints:

- `block`
- `cluster`
- `torch.compile`
- `block/torch`
- `cluster/torch`

The markdown-style benchmark and comparison use our own CUDA kernels, not the QuACK implementation directly. The tuned path borrows the main ideas from the markdown:

- vectorized `float4` loads and stores
- explicit `thread_per_row` tuning
- explicit `cluster_n` tuning
- cluster-partitioned long rows
- DSMEM only for scalar cross-block reduction
- shared-memory staging so a cluster block can reuse its local slice instead of reloading from GMEM

The markdown-style comparison prints:

- `block`
- `cluster`
- `torch.compile`
- chosen `thread_per_row`
- explicit `batch-size`

## Interpreting the result

If `cluster` beats `block` at large row sizes, DSMEM is helping this simple reduction on your GPU.

If `cluster` is flat or slower, that does not mean DSMEM is useless in general. For a plain sum reduction, every element is already read exactly once from global memory, so cluster reduction mostly changes how block partials are combined and how much parallelism is available per row. The QuACK blog's bigger wins come from more complex multi-stage reductions like softmax, where avoiding extra global-memory passes or register spilling matters much more.

For the markdown-style CUDA path, the strongest results currently come from `softmax` and `rmsnorm`. Those kernels use a staged DSMEM-aware path when the per-block slice fits in shared memory. `cross_entropy` also benefits relative to our block kernel, but on this 5090 setup it still trails `torch.compile`.
