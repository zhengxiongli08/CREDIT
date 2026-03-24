# minibench_ffn_push

This benchmark is a more paper-ready version of the winning `minibench_push_norm` pattern.

## Intended interpretation

It approximates a small FFN pipeline for one token group:

1. **Producer CTA** computes a hidden activation tile that behaves like a gated up-projection.
2. The consumer needs the **full hidden tile** to compute an RMS-like normalization.
3. The consumer then performs a down-projection over multiple output rows.

The three execution modes are:

- `baseline`: stage the hidden tile in global memory, flush cache, then consume it.
- `fused_local`: use one CTA, but chunk the hidden tile and recompute producer work because the whole hidden vector is not kept cheaply for both passes.
- `fused_cluster`: rank 0 produces the hidden tile once and **pushes** it into rank 1 shared memory through DSMEM; rank 1 then performs normalization and down-projection locally.

## Why this matters

This benchmark is closer to the story you want for a paper:

- the intermediate is FFN-like rather than a generic toy tile,
- the consumer has a real full-vector dependency,
- the cluster handoff uses write-heavy DSMEM behavior, which matches the measured hardware asymmetry.

## Mainstream framework comparison

Plain framework baselines were added in the same folder:

- `ffn_push_pytorch_compile.py`: plain `torch.compile`
- `ffn_push_tensorrt.py`: plain TensorRT from ONNX export
- `compare_frameworks.sh`: runs the cluster CUDA kernel and both framework baselines on the same shape

At the current winning point:

- shape: `groups=512, hidden=12288, out_rows=8, producer_iters=32`
- cluster CUDA: `0.143 ms`
- PyTorch `torch.compile`: `0.147 ms`
- TensorRT: `0.164 ms`

So this workload gives a useful result for the paper: the inter-SM cluster-fused kernel is slightly faster than plain `torch.compile` and clearly faster than plain TensorRT.

The likely reason is that this benchmark is not dominated by a pair of highly tuned dense GEMMs. Instead, it is dominated by a full-hidden-tile handoff, RMS-style full-vector dependency, and producer recomputation pressure in the local path — exactly the regime where cluster fusion is designed to help.

## More convincing win region

The strongest effect so far comes from increasing `out_rows`, which increases reuse of the same normalized full hidden tile after the producer-consumer handoff.

With `groups=512`, `hidden=12288`, and `producer_iters=32`:

- `out_rows=16`
	- cluster CUDA: `0.178 ms`
	- PyTorch `torch.compile`: `0.268 ms` (`1.51x` slower than cluster)
	- TensorRT: `0.291 ms` (`1.64x` slower than cluster)
- `out_rows=32`
	- cluster CUDA: `0.248 ms`
	- PyTorch `torch.compile`: `0.506 ms` (`2.04x` slower than cluster)
	- TensorRT: `0.541 ms` (`2.18x` slower than cluster)
- `out_rows=64`
	- cluster CUDA: `0.389 ms`
	- PyTorch `torch.compile`: `0.979 ms` (`2.52x` slower than cluster)
	- TensorRT: `1.026 ms` (`2.64x` slower than cluster)

This is the most convincing evidence in the folder so far that the idea has real value: when one produced hidden tile feeds many downstream outputs after a full-vector dependency, keeping the handoff on-chip via DSMEM becomes much more attractive than routing the intermediate through the ordinary framework path.

In other words, the best paper claim is not “cluster fusion beats PyTorch/TensorRT for dense FFNs in general”. It is closer to: cluster fusion wins in the regime where a large intermediate must be materialized, normalized or globally reduced, and then reused enough times that on-chip handoff amortizes the cluster overhead.

## Run

```bash
cd minibench_ffn_push
./run.sh --groups 512 --hidden 12288 --out-rows 8 --producer-iters 32
```

Validated winning point on this RTX 5090:

- `baseline = 0.577 ms`
- `fused_local = 0.454 ms`
- `fused_cluster = 0.143 ms`
- `cluster vs baseline = 4.04x`
- `cluster vs local = 3.18x`

This point is intentionally in the regime where:

- the hidden vector is large,
- the consumer needs the full hidden tile for normalization,
- single-CTA fusion pays chunking and recomputation,
- cluster fusion pays the handoff once and then reuses the full hidden tile locally.

## Sweep

```bash
./sweep.sh
```

## Profile

```bash
./profile.sh
```
