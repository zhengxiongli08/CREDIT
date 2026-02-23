import torch
import torch.nn as nn
import tensorrt as trt
import cuda.cudart as cudart
import numpy as np
import argparse

# --- Configuration ---
DEFAULT_M, DEFAULT_K, DEFAULT_N = 128, 4096, 16384

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--m", type=int, default=DEFAULT_M)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    args = parser.parse_args()
    m, k, n = args.m, args.k, args.n
    onnx_file = f"ffn_baseline_{m}_{k}_{n}.onnx"

    # 1. Export ONNX (Same as before)
    class FFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(k, n, bias=False)
            self.act = nn.ReLU()
            self.fc2 = nn.Linear(n, k, bias=False)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    model = FFN().cuda().half().eval()
    dummy_input = torch.randn(m, k, device='cuda', dtype=torch.float16)

    torch.onnx.export(
        model, dummy_input, onnx_file,
        input_names=['input'], output_names=['output'],
        opset_version=17
    )

    # 2. Build Engine
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser_trt = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)

    with open(onnx_file, 'rb') as f:
        parser_trt.parse(f.read())

    plan = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()

    # 3. Allocation
    def allocate_buffers(local_engine):
        inputs, outputs, bindings_map = [], [], {}
        for i in range(local_engine.num_io_tensors):
            tensor_name = local_engine.get_tensor_name(i)
            shape = local_engine.get_tensor_shape(tensor_name)
            dtype = trt.nptype(local_engine.get_tensor_dtype(tensor_name))
            size = trt.volume(shape) * np.dtype(dtype).itemsize

            ptr = cudart.cudaMalloc(size)[1]
            bindings_map[tensor_name] = ptr

            if local_engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append({'name': tensor_name, 'shape': shape})
            else:
                outputs.append({'name': tensor_name, 'shape': shape})

        return inputs, outputs, bindings_map

    _, _, bindings_map = allocate_buffers(engine)

    for name, ptr in bindings_map.items():
        context.set_tensor_address(name, ptr)

    # 4. Benchmarking
    print("Benchmarking TensorRT (Standard FFN)...")
    stream = cudart.cudaStreamCreate()[1]

    for _ in range(args.warmup):
        context.execute_async_v3(stream_handle=stream)
    cudart.cudaStreamSynchronize(stream)

    start_event = cudart.cudaEventCreate()[1]
    end_event = cudart.cudaEventCreate()[1]

    cudart.cudaEventRecord(start_event, stream)
    for _ in range(args.iters):
        context.execute_async_v3(stream_handle=stream)
    cudart.cudaEventRecord(end_event, stream)

    cudart.cudaEventSynchronize(end_event)
    elapsed_ms = cudart.cudaEventElapsedTime(start_event, end_event)[1]

    avg = elapsed_ms / args.iters
    print(f"Correct Average Latency: {avg:.3f} ms")
    print(f"LATENCY_MS={avg:.6f}")

    cudart.cudaStreamDestroy(stream)


if __name__ == "__main__":
    main()