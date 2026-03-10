"""
Multi-stream FFN benchmark.

Each CUDA stream runs an independent FFN workload concurrently.
This simulates multi-tenant inference on a single GPU and increases
memory pressure relative to the single-stream baseline.

Usage examples:
    python gemm_pytorch_multi_stream.py --num-streams 1   # baseline
    python gemm_pytorch_multi_stream.py --num-streams 2
    python gemm_pytorch_multi_stream.py --num-streams 4
"""

import torch
import torch.nn as nn
import argparse

DEFAULT_M, DEFAULT_K, DEFAULT_N = 128, 4096, 16384


class FFN(nn.Module):
    def __init__(self, k, n):
        super().__init__()
        self.fc1 = nn.Linear(k, n, bias=False)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(n, k, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup",      type=int, default=20)
    parser.add_argument("--iters",       type=int, default=100)
    parser.add_argument("--m",           type=int, default=DEFAULT_M)
    parser.add_argument("--k",           type=int, default=DEFAULT_K)
    parser.add_argument("--n",           type=int, default=DEFAULT_N)
    parser.add_argument("--num-streams", type=int, default=2,
                        help="Number of concurrent FFN workloads / CUDA streams")
    args = parser.parse_args()
    m, k, n, S = args.m, args.k, args.n, args.num_streams

    device = "cuda"
    dtype  = torch.float16

    # One independent model + input per stream so there is no shared state.
    # Weights are distinct → S× the weight traffic compared to 1-stream baseline.
    models = [FFN(k, n).to(device=device, dtype=dtype).eval() for _ in range(S)]
    inputs = [torch.randn(m, k, device=device, dtype=dtype)   for _ in range(S)]
    streams = [torch.cuda.Stream() for _ in range(S)]

    # Warmup
    print(f"Warming up ({S} stream(s))...")
    for _ in range(args.warmup):
        for i in range(S):
            with torch.cuda.stream(streams[i]):
                _ = models[i](inputs[i])
    torch.cuda.synchronize()

    # --- Benchmark: wall-clock time from first launch to last completion ---
    # One start event on stream 0, one end event per stream.
    start_event = torch.cuda.Event(enable_timing=True)
    end_events  = [torch.cuda.Event(enable_timing=True) for _ in range(S)]

    # Record start on the default stream so all work streams see it completed.
    start_event.record(torch.cuda.current_stream())

    print(f"Benchmarking {S}-stream FFN ({m}x{k} -> {n} -> {k})...")
    for _ in range(args.iters):
        for i in range(S):
            with torch.cuda.stream(streams[i]):
                _ = models[i](inputs[i])

    # Record end on each stream, then wait for all.
    for i in range(S):
        end_events[i].record(streams[i])

    torch.cuda.synchronize()

    # Wall-clock latency = time from start to the last stream finishing.
    wall_ms = max(start_event.elapsed_time(e) for e in end_events)
    avg_latency = wall_ms / args.iters

    print(f"Run {args.iters} iterations, {S} concurrent stream(s).")
    print(f"Wall-clock Average Latency : {avg_latency:.3f} ms")
    print(f"Aggregate FFN Throughput   : {S / (avg_latency * 1e-3):.1f} FFN/s")


if __name__ == "__main__":
    main()
