import argparse
import tempfile
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorrt as trt


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
        self.hidden = hidden
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
        zero = torch.zeros((), device=qweights.device, dtype=torch.float32)
        for iter_idx in range(self.producer_iters):
            up = torch.maximum(up * (1.005859375 + 0.001953125 * float(iter_idx & 3)) + bias, zero)
        hidden = F.silu(up) * tensor_attr(self.gate_proj)
        hidden = self.norm(hidden)
        return torch.sum(hidden.unsqueeze(1) * tensor_attr(self.down_weight), dim=-1)


def export_onnx(groups: int, hidden: int, out_rows: int, producer_iters: int) -> bytes:
    model = FFNPushModule(groups, hidden, out_rows, producer_iters).cuda().eval()
    sample = init_qweights(groups, hidden, 'cuda')
    with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp:
        torch.onnx.export(
            model,
            (sample,),
            tmp.name,
            input_names=['qweights'],
            output_names=['output'],
            opset_version=18,
            dynamo=False,
        )
        tmp.seek(0)
        return tmp.read()


def build_engine(onnx_bytes: bytes):
    trt_mod: Any = trt
    logger = trt_mod.Logger(trt_mod.Logger.WARNING)
    builder = trt_mod.Builder(logger)
    network = builder.create_network(1 << int(trt_mod.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt_mod.OnnxParser(network, logger)
    config = builder.create_builder_config()
    if not parser.parse(onnx_bytes):
        for idx in range(parser.num_errors):
            print(parser.get_error(idx))
        raise RuntimeError('TensorRT ONNX parsing failed')
    for idx in range(network.num_inputs):
        tensor = network.get_input(idx)
        if tensor.dtype == trt_mod.DataType.INT8:
            tensor.dynamic_range = (-64.0, 63.0)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TensorRT failed to build serialized engine')
    runtime = trt_mod.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError('TensorRT failed to deserialize engine')
    return engine


def allocate_buffers(engine, groups: int, hidden: int):
    trt_mod: Any = trt
    type_map = {
        trt_mod.DataType.FLOAT: torch.float32,
        trt_mod.DataType.HALF: torch.float16,
        trt_mod.DataType.INT32: torch.int32,
        trt_mod.DataType.INT8: torch.int8,
        trt_mod.DataType.BOOL: torch.bool,
    }
    buffers = {}
    for idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(idx)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = type_map[engine.get_tensor_dtype(name)]
        if engine.get_tensor_mode(name) == trt_mod.TensorIOMode.INPUT:
            tensor = init_qweights(groups, hidden, 'cuda').to(dtype=dtype)
        else:
            tensor = torch.empty(shape, device='cuda', dtype=dtype)
        buffers[name] = tensor
    return buffers


def init_qweights(groups: int, hidden: int, device: str) -> torch.Tensor:
    idx = torch.arange(groups * hidden, device=device, dtype=torch.int32)
    pattern = torch.bitwise_and(idx * 37 + 11, 127)
    return (pattern - 64).to(torch.int8).view(groups, hidden)


def benchmark(context, stream, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        for _ in range(iters):
            ok = context.execute_async_v3(stream_handle=stream.cuda_stream)
            if not ok:
                raise RuntimeError('TensorRT execute_async_v3 failed')
        end.record(stream)
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


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

    onnx_bytes = export_onnx(args.groups, args.hidden, args.out_rows, args.producer_iters)
    engine = build_engine(onnx_bytes)
    context = engine.create_execution_context()
    buffers = allocate_buffers(engine, args.groups, args.hidden)
    for name, tensor in buffers.items():
        context.set_tensor_address(name, tensor.data_ptr())
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(args.warmup):
            ok = context.execute_async_v3(stream_handle=stream.cuda_stream)
            if not ok:
                raise RuntimeError('TensorRT execute_async_v3 failed during warmup')
    torch.cuda.synchronize()

    if args.ncu_profile:
        torch.cuda.cudart().cudaProfilerStart()
        with torch.cuda.stream(stream):
            for _ in range(args.profile_iters):
                ok = context.execute_async_v3(stream_handle=stream.cuda_stream)
                if not ok:
                    raise RuntimeError('TensorRT execute_async_v3 failed during profiling')
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        print('FRAMEWORK=tensorrt')
        print(f'GROUPS={args.groups}')
        print(f'HIDDEN={args.hidden}')
        print(f'OUT_ROWS={args.out_rows}')
        print(f'PRODUCER_ITERS={args.producer_iters}')
        print(f'PROFILED_ITERS={args.profile_iters}')
        return

    avg_ms = benchmark(context, stream, args.iters)

    print('FRAMEWORK=tensorrt')
    print(f'GROUPS={args.groups}')
    print(f'HIDDEN={args.hidden}')
    print(f'OUT_ROWS={args.out_rows}')
    print(f'PRODUCER_ITERS={args.producer_iters}')
    print(f'AVG_MS={avg_ms:.3f}')


if __name__ == '__main__':
    main()
