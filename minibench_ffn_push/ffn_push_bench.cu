#include <cooperative_groups.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace cg = cooperative_groups;

namespace {

constexpr int kThreads = 256;
constexpr int kClusterSize = 2;
constexpr const char* kDefaultMode = "all";

#define CUDA_CHECK(command)                                                     \
    do {                                                                        \
        cudaError_t error__ = (command);                                        \
        if (error__ != cudaSuccess) {                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(error__)         \
                      << " at line " << __LINE__ << std::endl;                 \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

struct Options {
    int groups = 512;
    int hidden = 12288;
    int out_rows = 8;
    int producer_iters = 32;
    int warmup = 5;
    int iters = 20;
    int profile_iters = 1;
    bool verify = true;
    bool profile_range = false;
    std::string mode = kDefaultMode;
};

struct ModeResult {
    std::string name;
    float avg_ms = 0.0f;
    double global_read_bytes = 0.0;
    double global_write_bytes = 0.0;
    double remote_shared_write_bytes = 0.0;
};

__host__ __device__ inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

__host__ __device__ inline float up_proj_value(int8_t packed_weight, int h_idx, int producer_iters) {
    float value = 0.03125f * static_cast<float>(packed_weight);
    float bias = 0.001953125f * static_cast<float>((h_idx & 63) - 32);
    for (int iter = 0; iter < producer_iters; ++iter) {
        value = fmaf(value, 1.005859375f + 0.001953125f * static_cast<float>(iter & 3), bias);
    }
    return value;
}

__host__ __device__ inline float gate_proj_value(int group_idx, int h_idx, int producer_iters) {
    float value = 0.0009765625f * static_cast<float>(((group_idx * 17 + h_idx * 5) & 511) - 256);
    for (int iter = 0; iter < producer_iters; ++iter) {
        value = fmaf(value, 1.00390625f, 0.000244140625f * static_cast<float>((iter & 7) - 3));
    }
    return value;
}

__host__ __device__ inline float down_weight_value(int group_idx, int row_idx, int h_idx) {
    int mixed = group_idx * 131 + row_idx * 29 + h_idx * 7;
    return 0.001953125f * static_cast<float>((mixed & 255) - 128);
}

__inline__ __device__ float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__inline__ __device__ float block_reduce_sum(float value) {
    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    value = warp_reduce_sum(value);
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();
    value = (threadIdx.x < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
    if (warp == 0) value = warp_reduce_sum(value);
    return value;
}

__global__ void init_qweights_kernel(int8_t* qweights, int total_items) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_items) {
        int pattern = (idx * 37 + 11) & 127;
        qweights[idx] = static_cast<int8_t>(pattern - 64);
    }
}

__global__ void flush_l2_kernel(float* buffer, int elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < elements; i += stride) {
        buffer[i] = buffer[i] + 1.0f;
    }
}

__device__ inline float hidden_value(const int8_t* qweights, int base, int group_idx, int h_idx, int producer_iters) {
    float up = up_proj_value(qweights[base + h_idx], h_idx, producer_iters);
    float gate = gate_proj_value(group_idx, h_idx, producer_iters);
    return silu(up) * gate;
}

__global__ void baseline_ffn_producer_kernel(const int8_t* __restrict__ qweights,
                                             float* __restrict__ hidden,
                                             int hidden_size,
                                             int producer_iters) {
    int group_idx = blockIdx.x;
    int base = group_idx * hidden_size;
    for (int h_idx = threadIdx.x; h_idx < hidden_size; h_idx += blockDim.x) {
        hidden[base + h_idx] = hidden_value(qweights, base, group_idx, h_idx, producer_iters);
    }
}

__global__ void baseline_ffn_consumer_kernel(const float* __restrict__ hidden,
                                             float* __restrict__ outputs,
                                             int hidden_size,
                                             int out_rows) {
    extern __shared__ float tile[];
    int group_idx = blockIdx.x;
    int base = group_idx * hidden_size;
    int out_base = group_idx * out_rows;

    for (int h_idx = threadIdx.x; h_idx < hidden_size; h_idx += blockDim.x) {
        tile[h_idx] = hidden[base + h_idx];
    }
    __syncthreads();

    float sumsq_partial = 0.0f;
    for (int h_idx = threadIdx.x; h_idx < hidden_size; h_idx += blockDim.x) {
        sumsq_partial = fmaf(tile[h_idx], tile[h_idx], sumsq_partial);
    }
    float sumsq = block_reduce_sum(sumsq_partial);
    __shared__ float inv_rms;
    if (threadIdx.x == 0) inv_rms = rsqrtf(sumsq / static_cast<float>(hidden_size) + 1.0e-5f);
    __syncthreads();

    for (int row_idx = 0; row_idx < out_rows; ++row_idx) {
        float partial = 0.0f;
        for (int h_idx = threadIdx.x; h_idx < hidden_size; h_idx += blockDim.x) {
            partial = fmaf(tile[h_idx] * inv_rms, down_weight_value(group_idx, row_idx, h_idx), partial);
        }
        float reduced = block_reduce_sum(partial);
        if (threadIdx.x == 0) outputs[out_base + row_idx] = reduced;
        __syncthreads();
    }
}

__global__ void fused_local_ffn_chunked_kernel(const int8_t* __restrict__ qweights,
                                               float* __restrict__ outputs,
                                               int hidden_size,
                                               int out_rows,
                                               int producer_iters,
                                               int chunk_h) {
    extern __shared__ float chunk[];
    int group_idx = blockIdx.x;
    int base = group_idx * hidden_size;
    int out_base = group_idx * out_rows;

    float sumsq_partial = 0.0f;
    for (int chunk_base = 0; chunk_base < hidden_size; chunk_base += chunk_h) {
        for (int local_h = threadIdx.x; local_h < chunk_h; local_h += blockDim.x) {
            int h_idx = chunk_base + local_h;
            float value = hidden_value(qweights, base, group_idx, h_idx, producer_iters);
            chunk[local_h] = value;
            sumsq_partial = fmaf(value, value, sumsq_partial);
        }
        __syncthreads();
    }
    float sumsq = block_reduce_sum(sumsq_partial);
    __shared__ float inv_rms;
    if (threadIdx.x == 0) inv_rms = rsqrtf(sumsq / static_cast<float>(hidden_size) + 1.0e-5f);
    __syncthreads();

    for (int row_idx = 0; row_idx < out_rows; ++row_idx) {
        float row_partial = 0.0f;
        for (int chunk_base = 0; chunk_base < hidden_size; chunk_base += chunk_h) {
            for (int local_h = threadIdx.x; local_h < chunk_h; local_h += blockDim.x) {
                int h_idx = chunk_base + local_h;
                chunk[local_h] = hidden_value(qweights, base, group_idx, h_idx, producer_iters);
            }
            __syncthreads();
            float partial = 0.0f;
            for (int local_h = threadIdx.x; local_h < chunk_h; local_h += blockDim.x) {
                int h_idx = chunk_base + local_h;
                partial = fmaf(chunk[local_h] * inv_rms, down_weight_value(group_idx, row_idx, h_idx), partial);
            }
            row_partial += block_reduce_sum(partial);
            __syncthreads();
        }
        if (threadIdx.x == 0) outputs[out_base + row_idx] = row_partial;
        __syncthreads();
    }
}

__global__ __cluster_dims__(2, 1, 1)
void fused_cluster_ffn_push_kernel(const int8_t* __restrict__ qweights,
                                   float* __restrict__ outputs,
                                   int hidden_size,
                                   int out_rows,
                                   int producer_iters) {
    extern __shared__ float hidden_tile[];
    cg::cluster_group cluster = cg::this_cluster();
    int rank = cluster.block_rank();
    int group_idx = blockIdx.x / kClusterSize;
    int base = group_idx * hidden_size;
    int out_base = group_idx * out_rows;

    if (rank == 0) {
        float* consumer_hidden = cluster.map_shared_rank(hidden_tile, 1);
        for (int h_idx = threadIdx.x; h_idx < hidden_size; h_idx += blockDim.x) {
            consumer_hidden[h_idx] = hidden_value(qweights, base, group_idx, h_idx, producer_iters);
        }
    }
    cluster.sync();

    if (rank == 1) {
        float sumsq_partial = 0.0f;
        for (int h_idx = threadIdx.x; h_idx < hidden_size; h_idx += blockDim.x) {
            sumsq_partial = fmaf(hidden_tile[h_idx], hidden_tile[h_idx], sumsq_partial);
        }
        float sumsq = block_reduce_sum(sumsq_partial);
        __shared__ float inv_rms;
        if (threadIdx.x == 0) inv_rms = rsqrtf(sumsq / static_cast<float>(hidden_size) + 1.0e-5f);
        __syncthreads();

        for (int row_idx = 0; row_idx < out_rows; ++row_idx) {
            float partial = 0.0f;
            for (int h_idx = threadIdx.x; h_idx < hidden_size; h_idx += blockDim.x) {
                partial = fmaf(hidden_tile[h_idx] * inv_rms, down_weight_value(group_idx, row_idx, h_idx), partial);
            }
            float reduced = block_reduce_sum(partial);
            if (threadIdx.x == 0) outputs[out_base + row_idx] = reduced;
            __syncthreads();
        }
    }
    cluster.sync();
}

bool is_mode_selected(const std::string& selected_mode, const std::string& mode_name) {
    return selected_mode == "all" || selected_mode == mode_name;
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int arg_idx = 1; arg_idx < argc; ++arg_idx) {
        std::string argument = argv[arg_idx];
        auto require_value = [&](const std::string& flag_name) -> const char* {
            if (arg_idx + 1 >= argc) {
                std::cerr << "Missing value for " << flag_name << std::endl;
                std::exit(1);
            }
            return argv[++arg_idx];
        };
        if (argument == "--mode") options.mode = require_value(argument);
        else if (argument == "--groups") options.groups = std::atoi(require_value(argument));
        else if (argument == "--hidden") options.hidden = std::atoi(require_value(argument));
        else if (argument == "--out-rows") options.out_rows = std::atoi(require_value(argument));
        else if (argument == "--producer-iters") options.producer_iters = std::atoi(require_value(argument));
        else if (argument == "--warmup") options.warmup = std::atoi(require_value(argument));
        else if (argument == "--iters") options.iters = std::atoi(require_value(argument));
        else if (argument == "--profile-iters") options.profile_iters = std::atoi(require_value(argument));
        else if (argument == "--profile-range") options.profile_range = true;
        else if (argument == "--no-verify") options.verify = false;
        else if (argument == "--help") {
            std::cout << "Usage: ./ffn_push_bench [--mode all|baseline|fused_local|fused_cluster]\n"
                      << "                       [--groups N] [--hidden N] [--out-rows N] [--producer-iters N]\n"
                      << "                       [--warmup N] [--iters N] [--profile-range] [--profile-iters N] [--no-verify]\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown argument: " << argument << std::endl;
            std::exit(1);
        }
    }
    return options;
}

double to_megabytes(double bytes) { return bytes / (1024.0 * 1024.0); }

ModeResult estimate_traffic(const std::string& mode_name, const Options& options, int chunk_h) {
    double qweight_bytes = static_cast<double>(options.groups) * options.hidden * sizeof(int8_t);
    double hidden_bytes = static_cast<double>(options.groups) * options.hidden * sizeof(float);
    double output_bytes = static_cast<double>(options.groups) * options.out_rows * sizeof(float);
    ModeResult result;
    result.name = mode_name;
    if (mode_name == "baseline") {
        result.global_read_bytes = qweight_bytes + hidden_bytes;
        result.global_write_bytes = hidden_bytes + output_bytes;
    } else if (mode_name == "fused_local") {
        int chunks = options.hidden / chunk_h;
        result.global_read_bytes = static_cast<double>(chunks + 1) * qweight_bytes;
        result.global_write_bytes = output_bytes;
    } else if (mode_name == "fused_cluster") {
        result.global_read_bytes = qweight_bytes;
        result.global_write_bytes = output_bytes;
        result.remote_shared_write_bytes = hidden_bytes;
    }
    return result;
}

void print_mode_result(const ModeResult& result) {
    std::cout << std::fixed << std::setprecision(3)
              << "MODE=" << result.name << '\n'
              << "AVG_MS=" << result.avg_ms << '\n'
              << "EST_GLOBAL_READ_MB=" << to_megabytes(result.global_read_bytes) << '\n'
              << "EST_GLOBAL_WRITE_MB=" << to_megabytes(result.global_write_bytes) << '\n'
              << "EST_REMOTE_SHARED_WRITE_MB=" << to_megabytes(result.remote_shared_write_bytes) << '\n';
}

float benchmark_launch(const std::function<void()>& launch_fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) launch_fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t t0 = nullptr, t1 = nullptr;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) launch_fn();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return elapsed / static_cast<float>(iters);
}

float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
    return max_diff;
}

}  // namespace

int main(int argc, char** argv) {
    Options options = parse_args(argc, argv);
    CUDA_CHECK(cudaSetDevice(0));

    int cluster_supported = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&cluster_supported, cudaDevAttrClusterLaunch, 0));
    int max_dynamic_shared_bytes = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_dynamic_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));

    size_t full_shared_bytes = static_cast<size_t>(options.hidden) * sizeof(float);
    int chunk_h = options.hidden / 8;
    size_t chunk_shared_bytes = static_cast<size_t>(chunk_h) * sizeof(float);
    if (full_shared_bytes > static_cast<size_t>(max_dynamic_shared_bytes)) {
        std::cerr << "hidden too large for shared memory" << std::endl;
        return 1;
    }

    CUDA_CHECK(cudaFuncSetAttribute(baseline_ffn_consumer_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(full_shared_bytes)));
    CUDA_CHECK(cudaFuncSetAttribute(fused_local_ffn_chunked_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(chunk_shared_bytes)));
    CUDA_CHECK(cudaFuncSetAttribute(fused_cluster_ffn_push_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(full_shared_bytes)));
    CUDA_CHECK(cudaFuncSetAttribute(fused_cluster_ffn_push_kernel,
                                    cudaFuncAttributeNonPortableClusterSizeAllowed,
                                    1));

    size_t qweight_items = static_cast<size_t>(options.groups) * options.hidden;
    size_t output_items = static_cast<size_t>(options.groups) * options.out_rows;

    int8_t* device_qweights = nullptr;
    float* device_hidden = nullptr;
    float* device_outputs = nullptr;
    float* device_flush = nullptr;
    CUDA_CHECK(cudaMalloc(&device_qweights, qweight_items * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&device_hidden, qweight_items * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_outputs, output_items * sizeof(float)));
    const int flush_elements = (256 * 1024 * 1024) / static_cast<int>(sizeof(float));
    CUDA_CHECK(cudaMalloc(&device_flush, static_cast<size_t>(flush_elements) * sizeof(float)));
    CUDA_CHECK(cudaMemset(device_flush, 0, static_cast<size_t>(flush_elements) * sizeof(float)));

    int init_blocks = static_cast<int>((qweight_items + kThreads - 1) / kThreads);
    init_qweights_kernel<<<init_blocks, kThreads>>>(device_qweights, static_cast<int>(qweight_items));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto zero_outputs = [&]() { CUDA_CHECK(cudaMemset(device_outputs, 0, output_items * sizeof(float))); };
    auto launch_baseline = [&]() {
        zero_outputs();
        baseline_ffn_producer_kernel<<<options.groups, kThreads>>>(device_qweights, device_hidden, options.hidden, options.producer_iters);
        flush_l2_kernel<<<256, kThreads>>>(device_flush, flush_elements);
        baseline_ffn_consumer_kernel<<<options.groups, kThreads, full_shared_bytes>>>(device_hidden, device_outputs, options.hidden, options.out_rows);
        CUDA_CHECK(cudaGetLastError());
    };
    auto launch_fused_local = [&]() {
        zero_outputs();
        fused_local_ffn_chunked_kernel<<<options.groups, kThreads, chunk_shared_bytes>>>(device_qweights, device_outputs, options.hidden, options.out_rows, options.producer_iters, chunk_h);
        CUDA_CHECK(cudaGetLastError());
    };

    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute attr[1];
    cfg.gridDim = dim3(options.groups * kClusterSize, 1, 1);
    cfg.blockDim = dim3(kThreads, 1, 1);
    cfg.dynamicSmemBytes = full_shared_bytes;
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = kClusterSize;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    auto launch_fused_cluster = [&]() {
        zero_outputs();
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, fused_cluster_ffn_push_kernel, device_qweights, device_outputs, options.hidden, options.out_rows, options.producer_iters));
    };

    std::cout << "GROUPS=" << options.groups << '\n'
              << "HIDDEN=" << options.hidden << '\n'
              << "OUT_ROWS=" << options.out_rows << '\n'
              << "PRODUCER_ITERS=" << options.producer_iters << '\n'
              << "LOCAL_CHUNK_H=" << chunk_h << '\n';

    if (options.profile_range) {
        for (int i = 0; i < options.warmup; ++i) {
            if (is_mode_selected(options.mode, "baseline")) launch_baseline();
            if (is_mode_selected(options.mode, "fused_local")) launch_fused_local();
            if (is_mode_selected(options.mode, "fused_cluster")) launch_fused_cluster();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaProfilerStart());
        for (int i = 0; i < options.profile_iters; ++i) {
            if (is_mode_selected(options.mode, "baseline")) launch_baseline();
            if (is_mode_selected(options.mode, "fused_local")) launch_fused_local();
            if (is_mode_selected(options.mode, "fused_cluster")) launch_fused_cluster();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaProfilerStop());
        std::cout << "PROFILE_RANGE_COMPLETED=1\n";
        CUDA_CHECK(cudaFree(device_qweights));
        CUDA_CHECK(cudaFree(device_hidden));
        CUDA_CHECK(cudaFree(device_outputs));
        CUDA_CHECK(cudaFree(device_flush));
        return 0;
    }

    if (options.verify) {
        std::vector<float> baseline_out(output_items), cmp_out(output_items);
        launch_baseline();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(baseline_out.data(), device_outputs, output_items * sizeof(float), cudaMemcpyDeviceToHost));
        if (is_mode_selected(options.mode, "fused_local")) {
            launch_fused_local();
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(cmp_out.data(), device_outputs, output_items * sizeof(float), cudaMemcpyDeviceToHost));
            std::cout << "VERIFY_FUSED_LOCAL_MAX_ABS_DIFF=" << max_abs_diff(baseline_out, cmp_out) << '\n';
        }
        if (is_mode_selected(options.mode, "fused_cluster")) {
            launch_fused_cluster();
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(cmp_out.data(), device_outputs, output_items * sizeof(float), cudaMemcpyDeviceToHost));
            std::cout << "VERIFY_FUSED_CLUSTER_MAX_ABS_DIFF=" << max_abs_diff(baseline_out, cmp_out) << '\n';
        }
    }

    std::vector<ModeResult> results;
    if (is_mode_selected(options.mode, "baseline")) {
        auto result = estimate_traffic("baseline", options, chunk_h);
        result.avg_ms = benchmark_launch(launch_baseline, options.warmup, options.iters);
        print_mode_result(result);
        results.push_back(result);
    }
    if (is_mode_selected(options.mode, "fused_local")) {
        auto result = estimate_traffic("fused_local", options, chunk_h);
        result.avg_ms = benchmark_launch(launch_fused_local, options.warmup, options.iters);
        print_mode_result(result);
        results.push_back(result);
    }
    if (is_mode_selected(options.mode, "fused_cluster")) {
        auto result = estimate_traffic("fused_cluster", options, chunk_h);
        result.avg_ms = benchmark_launch(launch_fused_cluster, options.warmup, options.iters);
        print_mode_result(result);
        results.push_back(result);
    }

    float baseline_ms = 0.0f, local_ms = 0.0f, cluster_ms = 0.0f;
    for (const auto& result : results) {
        if (result.name == "baseline") baseline_ms = result.avg_ms;
        if (result.name == "fused_local") local_ms = result.avg_ms;
        if (result.name == "fused_cluster") cluster_ms = result.avg_ms;
    }
    std::cout << std::fixed << std::setprecision(3);
    if (baseline_ms > 0.0f && local_ms > 0.0f) std::cout << "SPEEDUP_FUSED_LOCAL_VS_BASELINE=" << (baseline_ms / local_ms) << '\n';
    if (baseline_ms > 0.0f && cluster_ms > 0.0f) std::cout << "SPEEDUP_FUSED_CLUSTER_VS_BASELINE=" << (baseline_ms / cluster_ms) << '\n';
    if (local_ms > 0.0f && cluster_ms > 0.0f) std::cout << "SPEEDUP_FUSED_CLUSTER_VS_FUSED_LOCAL=" << (local_ms / cluster_ms) << '\n';

    CUDA_CHECK(cudaFree(device_qweights));
    CUDA_CHECK(cudaFree(device_hidden));
    CUDA_CHECK(cudaFree(device_outputs));
    CUDA_CHECK(cudaFree(device_flush));
    return 0;
}
