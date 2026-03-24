import argparse
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def tensor_attr(value: object) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1.0e-5):
        super().__init__()
        self.register_buffer('weight', torch.ones(hidden_size, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_square + self.eps)
        return x * inv_rms * tensor_attr(self.weight)


class FFNPushModule(nn.Module):
    def __init__(self, groups: int, hidden: int, out_rows: int, producer_iters: int):
        super().__init__()
        self.groups = groups
        self.hidden = hidden
        self.out_rows = out_rows
        self.producer_iters = producer_iters

        hidden_idx = torch.arange(hidden, dtype=torch.float32)
        group_idx = torch.arange(groups, dtype=torch.float32).unsqueeze(1)
        row_idx = torch.arange(out_rows, dtype=torch.float32).view(1, out_rows, 1)

        gate = 0.0009765625 * (((group_idx * 17.0 + hidden_idx * 5.0) % 512.0) - 256.0)
        for iter_idx in range(producer_iters):
            gate = gate * 1.00390625 + 0.000244140625 * float((iter_idx & 7) - 3)

        down = 0.001953125 * (((group_idx.view(groups, 1, 1) * 131.0 + row_idx * 29.0 + hidden_idx.view(1, 1, hidden) * 7.0) % 255.99998474121094).floor() - 128.0)

        self.register_buffer('gate_proj', gate)
        self.register_buffer('down_weight', down)
        self.norm = RMSNorm(hidden)

    def forward(self, qweights: torch.Tensor) -> torch.Tensor:
        hidden_idx = torch.arange(self.hidden, device=qweights.device, dtype=torch.float32)
        up = qweights.to(torch.float32) * 0.03125
        bias = 0.001953125 * ((torch.remainder(hidden_idx, 64.0)) - 32.0)
        for iter_idx in range(self.producer_iters):
            up = torch.maximum(up * (1.005859375 + 0.001953125 * float(iter_idx & 3)) + bias, torch.zeros((), device=qweights.device, dtype=torch.float32))
        hidden = F.silu(up) * tensor_attr(self.gate_proj)
        hidden = self.norm(hidden)
        return torch.sum(hidden.unsqueeze(1) * tensor_attr(self.down_weight), dim=-1)


def benchmark(model: Callable[[torch.Tensor], torch.Tensor], qweights: torch.Tensor, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        model(qweights)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        model(qweights)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def init_qweights(groups: int, hidden: int, device: str) -> torch.Tensor:
    idx = torch.arange(groups * hidden, device=device, dtype=torch.int32)
    pattern = torch.bitwise_and(idx * 37 + 11, 127)
    return (pattern - 64).to(torch.int8).view(groups, hidden)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups', type=int, default=512)
    parser.add_argument('--hidden', type=int, default=12288)
    parser.add_argument('--out-rows', type=int, default=8)
    parser.add_argument('--producer-iters', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--ncu-profile', action='store_true')
    parser.add_argument('--profile-iters', type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = 'cuda'
    qweights = init_qweights(args.groups, args.hidden, device)
    model = FFNPushModule(args.groups, args.hidden, args.out_rows, args.producer_iters).to(device).eval()

    with torch.inference_mode():
        compiled_model = torch.compile(model)
        for _ in range(3):
            compiled_model(qweights)
        torch.cuda.synchronize()

        if args.ncu_profile:
            for _ in range(args.warmup):
                compiled_model(qweights)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            for _ in range(args.profile_iters):
                compiled_model(qweights)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            print('FRAMEWORK=pytorch_compile')
            print(f'GROUPS={args.groups}')
            print(f'HIDDEN={args.hidden}')
            print(f'OUT_ROWS={args.out_rows}')
            print(f'PRODUCER_ITERS={args.producer_iters}')
            print(f'PROFILED_ITERS={args.profile_iters}')
            return

        avg_ms = benchmark(compiled_model, qweights, args.warmup, args.iters)

    print('FRAMEWORK=pytorch_compile')
    print(f'GROUPS={args.groups}')
    print(f'HIDDEN={args.hidden}')
    print(f'OUT_ROWS={args.out_rows}')
    print(f'PRODUCER_ITERS={args.producer_iters}')
    print(f'AVG_MS={avg_ms:.3f}')


if __name__ == '__main__':
    main()
