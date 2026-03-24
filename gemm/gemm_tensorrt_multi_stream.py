import argparse
import io

import torch
import torch.nn as nn
import tensorrt as trt

DEFAULT_M, DEFAULT_K, DEFAULT_N = 128, 4096, 16384


class FFN(nn.Module):
    def __init__(self, k, n):
        super().__init__()
        self.fc1 = nn.Linear(k, n, bias=False)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(n, k, bias=False)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def export_onnx(m, k, n):
    model = FFN(k, n).cuda().half().eval()
    dummy_input = torch.randn(m, k, device="cuda", dtype=torch.float16)
    buf = io.BytesIO()
    torch.onnx.export(
        model,
        dummy_input,
        buf,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
    )
    return buf.getvalue()


def build_engine(onnx_bytes):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)

    if not parser.parse(onnx_bytes):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("TensorRT ONNX parsing failed")

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT failed to build serialized engine")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if engine is None:
        raise RuntimeError("TensorRT failed to deserialize engine")
    return engine


def allocate_io_buffers(engine):
    trt_to_torch_dtype = {
        trt.DataType.HALF: torch.float16,
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT8: torch.int8,
        trt.DataType.BOOL: torch.bool,
    }
    buffers = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = trt_to_torch_dtype[engine.get_tensor_dtype(name)]
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            t = torch.randn(shape, device="cuda", dtype=dtype)
        else:
            t = torch.empty(shape, device="cuda", dtype=dtype)
        buffers[name] = t
    return buffers


def main():
    parser = argparse.ArgumentParser(description="TensorRT FFN benchmark with multiple CUDA streams")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--m", type=int, default=DEFAULT_M)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--num-streams", type=int, default=2)
    args = parser.parse_args()

    m, k, n, S = args.m, args.k, args.n, args.num_streams
    if S < 1:
        raise ValueError("--num-streams must be >= 1")

    print(f"Building TensorRT engine for FFN ({m}x{k} -> {n} -> {k})...")
    onnx_bytes = export_onnx(m, k, n)
    engine = build_engine(onnx_bytes)

    streams = [torch.cuda.Stream() for _ in range(S)]
    contexts = [engine.create_execution_context() for _ in range(S)]
    all_buffers = [allocate_io_buffers(engine) for _ in range(S)]

    for i in range(S):
        for name, tensor in all_buffers[i].items():
            contexts[i].set_tensor_address(name, tensor.data_ptr())

    print(f"Warming up ({S} stream(s))...")
    for _ in range(args.warmup):
        for i in range(S):
            with torch.cuda.stream(streams[i]):
                ok = contexts[i].execute_async_v3(stream_handle=streams[i].cuda_stream)
                if not ok:
                    raise RuntimeError(f"TensorRT execute_async_v3 failed on stream {i}")
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(S)]

    start_event.record(torch.cuda.current_stream())
    for _ in range(args.iters):
        for i in range(S):
            with torch.cuda.stream(streams[i]):
                ok = contexts[i].execute_async_v3(stream_handle=streams[i].cuda_stream)
                if not ok:
                    raise RuntimeError(f"TensorRT execute_async_v3 failed on stream {i}")

    for i in range(S):
        end_events[i].record(streams[i])

    torch.cuda.synchronize()

    wall_ms = max(start_event.elapsed_time(e) for e in end_events)
    avg_latency = wall_ms / args.iters

    print(f"Run {args.iters} iterations, {S} concurrent stream(s).")
    print(f"Wall-clock Average Latency : {avg_latency:.3f} ms")
    print(f"Aggregate FFN Throughput   : {S / (avg_latency * 1e-3):.1f} FFN/s")


if __name__ == "__main__":
    main()
