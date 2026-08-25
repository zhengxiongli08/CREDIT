# Reproducibility Notes

## Evaluation Contract

For every `(device, workload, width)` point, the runner measures four implementation classes under one timing protocol:

1. `torch.compile` with a full static graph and `max-autotune-no-cudagraphs`.
2. A handwritten streaming Triton kernel selected from the tested block/warp configurations.
3. The applicable non-DSMEM CUDA implementation.
4. DSMEM CUDA at every legal cluster size in the configured set.

Compilation, TorchInductor compilation, and Triton configuration search occur before reported timing. Framework and CUDA outputs are validated before performance results are accepted.

## Cost-Model Inputs

The model predicts DSMEM profitability relative to non-DSMEM CUDA. It uses:

- one non-DSMEM CUDA time at the same workload shape;
- independently measured local-shared-memory and DSMEM primitive rates;
- a work-free cluster-control measurement with matching cluster size, barrier count, row count, and shared-memory footprint;
- static workload counts for modeled bytes, eliminated reread bytes, staged bytes, and scalar partials.

The measured DSMEM workload time is not a model input. It is used only after prediction to score the decision.

## Repeatability

GPU timing is sensitive to clocks, temperature, power state, driver version, and co-tenancy. Preserve every generated result bundle. For additional paper-quality runs, use the full protocol on at least three independent allocations and report run-to-run variation or a median over allocation-level medians.

The Modal job records an `nvidia-smi` snapshot before and after each run. Privileged clock locking is not assumed. Local experiments should use an otherwise idle GPU and record the same environmental information.

## Adding a Workload

A new evaluated workload requires four coordinated pieces:

1. A standalone source under `cuda/workloads/` that emits the runner's `RESULT,...` CSV records and validates its outputs.
2. A `WorkloadSpec` and eager PyTorch reference in `dsmem_eval/workloads.py`.
3. A launcher and output allocator in `dsmem_eval/triton_kernels.py`.
4. Static model fields for total bytes, avoidable reread bytes, staged bytes, and reduction partial counts.

Start with `--quick --workloads NAME`, inspect correctness diagnostics, and only then run the complete protocol.
