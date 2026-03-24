import argparse
import tempfile
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorrt as trt


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


def export_onnx(groups: int, input_k: int, hidden: int, out_rows: int) -> bytes:
    model = FFNGemmRMSNorm(input_k, hidden, out_rows).cuda().eval()
    sample = torch.randn(groups, input_k, device="cuda", dtype=torch.float32)
    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
        torch.onnx.export(
            model,
            (sample,),
            tmp.name,
            input_names=["input"],
            output_names=["output"],
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
        raise RuntimeError("TensorRT ONNX parsing failed")

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT failed to build serialized engine")

    runtime = trt_mod.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if engine is None:
        raise RuntimeError("TensorRT failed to deserialize engine")
    return engine


def allocate_buffers(engine):
    trt_mod: Any = trt
    trt_to_torch = {
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
        dtype = trt_to_torch[engine.get_tensor_dtype(name)]
        if engine.get_tensor_mode(name) == trt_mod.TensorIOMode.INPUT:
            tensor = torch.randn(shape, device="cuda", dtype=dtype)
        else:
            tensor = torch.empty(shape, device="cuda", dtype=dtype)
        buffers[name] = tensor
    return buffers


def benchmark(context, stream, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        for _ in range(iters):
            ok = context.execute_async_v3(stream_handle=stream.cuda_stream)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 failed")
        end.record(stream)
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
    onnx_bytes = export_onnx(args.groups, args.input_k, args.hidden, args.out_rows)
    engine = build_engine(onnx_bytes)
    context = engine.create_execution_context()
    buffers = allocate_buffers(engine)
    for name, tensor in buffers.items():
        context.set_tensor_address(name, tensor.data_ptr())

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(args.warmup):
            ok = context.execute_async_v3(stream_handle=stream.cuda_stream)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 failed during warmup")
    torch.cuda.synchronize()

    avg_ms = benchmark(context, stream, args.iters)

    print(f"FRAMEWORK=tensorrt")
    print(f"GROUPS={args.groups}")
    print(f"INPUT_K={args.input_k}")
    print(f"HIDDEN={args.hidden}")
    print(f"OUT_ROWS={args.out_rows}")
    print(f"AVG_MS={avg_ms:.3f}")


if __name__ == "__main__":
    main()
