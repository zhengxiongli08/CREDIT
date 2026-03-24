import argparse
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1.0e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_square + self.eps)
        return x * inv_rms * self.weight


class FFNGemmRMSNorm(nn.Module):
    def __init__(self, input_k: int, hidden: int, out_rows: int):
        super().__init__()
        self.up_proj = nn.Linear(input_k, hidden, bias=False)
        self.gate_proj = nn.Linear(input_k, hidden, bias=False)
        self.norm = RMSNorm(hidden)
        self.down_proj = nn.Linear(hidden, out_rows, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.up_proj(x)
        gate = self.gate_proj(x)
        hidden = F.silu(gate) * up
        hidden = self.norm(hidden)
        return self.down_proj(hidden)


def benchmark(model: Callable[[torch.Tensor], torch.Tensor], inputs: torch.Tensor, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        model(inputs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        model(inputs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, default=128)
    parser.add_argument("--input-k", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=12288)
    parser.add_argument("--out-rows", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float32
    model = FFNGemmRMSNorm(args.input_k, args.hidden, args.out_rows).to(device=device, dtype=dtype).eval()
    inputs = torch.randn(args.groups, args.input_k, device=device, dtype=dtype)

    with torch.inference_mode():
        compiled_model = torch.compile(model)
        for _ in range(3):
            compiled_model(inputs)
        torch.cuda.synchronize()
        avg_ms = benchmark(compiled_model, inputs, args.warmup, args.iters)

    print(f"FRAMEWORK=pytorch_compile")
    print(f"GROUPS={args.groups}")
    print(f"INPUT_K={args.input_k}")
    print(f"HIDDEN={args.hidden}")
    print(f"OUT_ROWS={args.out_rows}")
    print(f"AVG_MS={avg_ms:.3f}")


if __name__ == "__main__":
    main()
