import torch
import torch.nn as nn
import argparse

DEFAULT_M, DEFAULT_K, DEFAULT_N = 128, 4096, 16384

# Define the GEMM Chain (Standard FFN)
class FFN(nn.Module):
    def __init__(self, k, n):
        super().__init__()
        # GEMM 1: Project Up (K -> N)
        self.fc1 = nn.Linear(k, n, bias=False)
        self.act = nn.ReLU()
        # GEMM 2: Project Down (N -> K)
        self.fc2 = nn.Linear(n, k, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--m", type=int, default=DEFAULT_M)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    args = parser.parse_args()
    m, k, n = args.m, args.k, args.n

    # Setup device and precision (FP16 is standard for H100 benchmarks)
    device = "cuda"
    dtype = torch.float32

    model = FFN(k, n).to(device=device, dtype=dtype)
    dummy_inputs = torch.randn(m, k, device=device, dtype=dtype)

    # --- BASELINE OPTIMIZATION: torch.compile ---
    model = torch.compile(model)

    # Warmup to compile kernels and stabilize GPU clock
    print("Warming up...")
    for _ in range(args.warmup):
        _ = model(dummy_inputs)
    torch.cuda.synchronize()

    # --- BENCHMARKING ---
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(f"Benchmarking PyTorch (Standard FFN, {m}x{k} -> {n} -> {k})...")

    start_event.record()
    for _ in range(args.iters):
        _ = model(dummy_inputs)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_latency = elapsed_time_ms / args.iters

    print(f"Run {args.iters} iterations.")
    print(f"Average Latency: {avg_latency:.3f} ms")


if __name__ == "__main__":
    main()
