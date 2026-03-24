import io
import torch
import torch.nn as nn
import tensorrt as trt
import argparse

DEFAULT_M, DEFAULT_K, DEFAULT_N = 128, 4096, 16384


# ---------------------------------------------------------------------------
# Step 1: Define the model and export to ONNX
# ---------------------------------------------------------------------------

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


def export_onnx(m, k, n):
    model = FFN(k, n).cuda().eval()
    dummy_input = torch.randn(m, k, device="cuda", dtype=torch.float32)
    buf = io.BytesIO()
    torch.onnx.export(
        model, dummy_input, buf,
        input_names=["input"], output_names=["output"],
        opset_version=18,
    )
    print("Exported ONNX model to in-memory buffer")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Step 2: Build a TensorRT engine from the ONNX file
# ---------------------------------------------------------------------------

def build_engine(onnx_bytes):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    # Use FP32 kernels for lower arithmetic intensity
    # FP32 has 2x the data size of FP16, reducing arithmetic intensity

    if not parser.parse(onnx_bytes):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("TensorRT ONNX parsing failed")

    serialized_engine = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    print("TensorRT engine built successfully")
    return engine


# ---------------------------------------------------------------------------
# Step 3: Allocate I/O buffers as PyTorch tensors
#         TensorRT accepts raw device pointers, so we just pass .data_ptr()
# ---------------------------------------------------------------------------

def allocate_buffers(engine):
    """Return a {tensor_name: torch.Tensor} dict for all I/O tensors."""
    TRT_TO_TORCH_DTYPE = {
        trt.DataType.HALF:  torch.float16,
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT8:  torch.int8,
        trt.DataType.BOOL:  torch.bool,
    }
    buffers = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        dtype = TRT_TO_TORCH_DTYPE[engine.get_tensor_dtype(name)]
        buffers[name] = torch.zeros(shape, dtype=dtype, device="cuda")
    return buffers


# ---------------------------------------------------------------------------
# Step 4: Benchmark
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters",  type=int, default=1000)
    parser.add_argument("--m", type=int, default=DEFAULT_M)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    args = parser.parse_args()
    m, k, n = args.m, args.k, args.n

    # Build engine entirely in memory — no files written to disk
    onnx_bytes = export_onnx(m, k, n)
    engine     = build_engine(onnx_bytes)
    context = engine.create_execution_context()

    # Allocate buffers and bind them to the execution context
    buffers = allocate_buffers(engine)
    for name, tensor in buffers.items():
        context.set_tensor_address(name, tensor.data_ptr())

    # Use a dedicated CUDA stream so TRT work is isolated
    stream = torch.cuda.Stream()

    # Warmup
    print("Warming up...")
    with torch.cuda.stream(stream):
        for _ in range(args.warmup):
            context.execute_async_v3(stream_handle=stream.cuda_stream)
    torch.cuda.synchronize()

    # Benchmark with CUDA events (same pattern as gemm_pytorch.py)
    print(f"Benchmarking TensorRT FP16 (Standard FFN, {m}x{k} -> {n} -> {k})...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(stream):
        start_event.record(stream)
        for _ in range(args.iters):
            context.execute_async_v3(stream_handle=stream.cuda_stream)
        end_event.record(stream)

    torch.cuda.synchronize()

    elapsed_ms  = start_event.elapsed_time(end_event)
    avg_latency = elapsed_ms / args.iters

    print(f"Run {args.iters} iterations.")
    print(f"Average Latency: {avg_latency:.3f} ms")


if __name__ == "__main__":
    main()
