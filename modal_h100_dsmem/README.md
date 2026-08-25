# H100 DSMEM Profiler on Modal

This folder runs the same CUDA microbenchmarks used on the local RTX 5090 on a strict Modal H100 allocation. Python manages the cloud job, compilation, metadata collection, parsing, and result download; the latency-sensitive operations remain in CUDA.

## Metrics

- Cluster-wide synchronization latency.
- Local shared-memory and remote DSMEM dependent-read latency.
- Local shared-memory and remote DSMEM read throughput.
- Local shared-memory store and DSMEM store-issue throughput.
- DSMEM store-visibility request/acknowledgment round trip and an explicitly labeled one-way estimate.
- L1, L2, and DRAM latency/bandwidth context from the original benchmark.

The DSMEM configuration uses one two-CTA cluster with 256 threads per CTA. Dynamic shared-memory allocation forces at most one benchmark CTA per SM when the H100 resource limits allow it, and every kernel records the requester and provider SM IDs.

## Setup

```bash
cd /home/zli2793/projects/fuser/modal_h100_dsmem
python -m pip install -r requirements.txt
modal setup
```

`gpu="H100!"` is intentional. Modal may upgrade an ordinary `H100` request to H200, while `H100!` requests an actual H100 for reproducible benchmarking. Modal documents this behavior at <https://modal.com/docs/guide/gpu>.

## Run

```bash
modal run modal_app.py
```

The benchmark output is printed locally and the full result is written to `h100_dsmem_results.json`. Select another output path with:

```bash
modal run modal_app.py --output results/h100_dsmem.json
```

Run Compute Sanitizer memcheck and synccheck after the normal measurement with:

```bash
modal run modal_app.py --run-sanitizers
```

Sanitizer execution is optional because instrumentation changes timing. The JSON always contains the uninstrumented benchmark results; sanitizer logs are stored separately under `sanitizers`.

## Output Interpretation

Throughput is reported in bytes per SM cycle per requester CTA. Under the reported clock assumption,

```text
GB/s per requester CTA = B/cycle/CTA * SM clock in GHz
```

This is the achieved throughput of the fixed eight-warp, single-requester configuration, not peak aggregate DSMEM-fabric bandwidth. Use the identical configuration when comparing H100 and RTX 5090.

The one-way store-visibility value is half of a measured request/acknowledgment round trip. It is an estimate, not a directly timestamped one-way store-completion latency.

