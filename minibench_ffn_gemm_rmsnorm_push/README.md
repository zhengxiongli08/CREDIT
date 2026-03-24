# minibench_ffn_gemm_rmsnorm_push

This benchmark is a closer approximation to an FFN-style pipeline than `minibench_ffn_push`.

## Intended interpretation

For each token-group, it models:

1. an **up projection** over a real reduction dimension `input_k`,
2. a **gate projection** over the same `input_k`,
3. a **SiLU gate** to form the hidden activation,
4. an **RMSNorm** that needs the full hidden vector,
5. a **down projection** over several output rows.

The key change versus `minibench_ffn_push` is that the producer cost now scales with a realistic GEMM/GEMV-like `K` dimension rather than an abstract `producer_iters` loop.

## Why this is useful

This benchmark preserves the win mechanism already observed in `minibench_push_norm` and `minibench_ffn_push`:

- the consumer has a **full-hidden-vector dependency** because of RMSNorm,
- a single CTA local fusion path must **chunk** the hidden tile,
- that chunked path must **recompute producer work** after normalization to finish the down projection,
- the cluster path computes the hidden tile once and **pushes** it into the consumer CTA via DSMEM writes.

If cluster fusion wins here, the claim is easier to defend in a paper because the producer scaling is tied to a real projection `K` dimension.

## Modes

- `baseline`: producer writes the hidden vector to global memory, cache is flushed, consumer reloads it.
- `fused_local`: one CTA loads the input vector once, computes hidden chunks for RMS statistics, then recomputes hidden chunks for each output row.
- `fused_cluster`: rank 0 computes the hidden vector once and pushes it into rank 1 shared memory; rank 1 performs RMSNorm and the down projection locally.

## Run

```bash
cd minibench_ffn_gemm_rmsnorm_push
./run.sh --groups 256 --input-k 128 --hidden 12288 --out-rows 8
```

## Sweep

```bash
./sweep.sh
```

The expectation is:

- increasing `input_k` makes producer work more expensive, so cluster fusion should look better relative to `fused_local`,
- increasing `out_rows` increases reuse after the handoff, which also favors the cluster path.

## Observed on this RTX 5090

The first validation run already shows a strong win region:

- `--groups 128 --input-k 128 --hidden 12288 --out-rows 8`
- `baseline = 0.583 ms`
- `fused_local = 0.775 ms`
- `fused_cluster = 0.172 ms`
- `cluster vs baseline = 3.40x`
- `cluster vs local = 4.51x`

The sweep also shows the expected trend:

- at fixed `input_k`, `cluster_vs_local` grows strongly with `out_rows`, because the normalized hidden tile is reused more after the handoff,
- at fixed `out_rows`, increasing `input_k` makes the producer more expensive, so the local chunked path is hurt more by recomputation,
- `cluster_vs_baseline` shrinks as `input_k` grows, which is also expected, because both `baseline` and `fused_cluster` compute the producer only once and the saved global-memory round-trip becomes a smaller fraction of total time.

## Mainstream framework comparison

Plain framework baselines were added in the same folder:

- `ffn_pytorch_compile.py`: plain `torch.compile` on an FFN-like module,
- `ffn_tensorrt.py`: plain TensorRT engine built from ONNX export,
- `compare_frameworks.sh`: runs the custom CUDA cluster kernel plus both framework baselines on the same shape.

First comparison point tested:

- shape: `groups=128, input_k=128, hidden=12288, out_rows=8`
- custom cluster CUDA: `0.171 ms`
- PyTorch `torch.compile`: `0.138 ms`
- TensorRT: `0.031 ms`

So for this dense FFN-like workload, the current custom cluster kernel does **not** beat the mainstream baselines.

That is still an informative result: it suggests the current kernel is competitive against synthetic/local baselines because it avoids the producer-consumer handoff and recomputation, but dense FFN execution is still dominated by highly optimized framework GEMMs and Tensor Core-backed library kernels.

One caveat is that the custom CUDA microbenchmark still uses synthetic on-the-fly weight generation inside the kernel, while the framework baselines use ordinary stored weights. That means the comparison is useful as a mainstream reference point, but not yet a perfectly apples-to-apples kernel-level evaluation.
