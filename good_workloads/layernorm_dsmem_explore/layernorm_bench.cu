#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cg = cooperative_groups;

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t err__ = (expr);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << ": " << cudaGetErrorString(err__) << std::endl;        \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

namespace {

constexpr int kBlockThreads = 256;
constexpr int kVecSize = 4;
constexpr int kMaxClusterSize = 8;
constexpr float kEps = 1e-5f;
constexpr std::array<int, 5> kDefaultNValues = {
    4096, 8192, 16384, 32768, 65536,
};
constexpr std::array<int, 3> kDefaultThreadPerRow = {16, 32, 256};

struct Options {
    int warmup = 2;
    int iters = 5;
    int batch_size = 8192;
    int requested_cluster_size = 8;
    bool csv = false;
    bool verify = true;
    std::vector<int> n_values = {
        kDefaultNValues.begin(), kDefaultNValues.end(),
    };
    std::vector<int> thread_per_row_values = {
        kDefaultThreadPerRow.begin(), kDefaultThreadPerRow.end(),
    };
};

struct DeviceInfo {
    std::string name;
    int major = 0;
    int minor = 0;
    bool cluster_launch = false;
    int max_cluster_size = 0;
    int chosen_cluster_size = 0;
    int max_dynamic_smem_per_block = 0;
};

struct VariantResult {
    bool valid = false;
    int thread_per_row = 0;
    double avg_ms = std::numeric_limits<double>::quiet_NaN();
    double gbps = std::numeric_limits<double>::quiet_NaN();
};

struct Pair {
    float sum;
    float sumsq;
};

std::vector<int> parse_int_list(const std::string& spec) {
    std::vector<int> values;
    std::stringstream ss(spec);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (!part.empty()) {
            values.push_back(std::stoi(part));
        }
    }
    if (values.empty()) {
        throw std::runtime_error("integer list cannot be empty");
    }
    return values;
}

Options parse_options(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_int = [&](const char* name) -> int {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return std::stoi(argv[++i]);
        };
        auto require_string = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--warmup") {
            opts.warmup = require_int("--warmup");
        } else if (arg == "--iters") {
            opts.iters = require_int("--iters");
        } else if (arg == "--batch-size") {
            opts.batch_size = require_int("--batch-size");
        } else if (arg == "--cluster-size") {
            opts.requested_cluster_size = require_int("--cluster-size");
        } else if (arg == "--n-values") {
            opts.n_values = parse_int_list(require_string("--n-values"));
        } else if (arg == "--thread-per-row-values") {
            opts.thread_per_row_values = parse_int_list(require_string(arg.c_str()));
        } else if (arg == "--csv") {
            opts.csv = true;
        } else if (arg == "--no-verify") {
            opts.verify = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: layernorm_bench [--csv] [--warmup N] [--iters N] "
                << "[--batch-size N] [--cluster-size N] [--n-values list] "
                << "[--thread-per-row-values list] [--no-verify]\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (opts.warmup < 0 || opts.iters <= 0 || opts.batch_size <= 0 ||
        opts.requested_cluster_size < 1) {
        throw std::runtime_error("numeric arguments must be positive");
    }
    for (int value : opts.thread_per_row_values) {
        if ((value != 16 && value != 32 && value != 256) ||
            (kBlockThreads % value) != 0) {
            throw std::runtime_error(
                "thread_per_row values must be one of 16, 32, or 256");
        }
    }
    return opts;
}

double modeled_bytes(int batch_size, int cols) {
    return static_cast<double>(batch_size) * cols * sizeof(float) * 4.0;
}

double throughput_gbps(int batch_size, int cols, double avg_ms) {
    return modeled_bytes(batch_size, cols) / (avg_ms / 1000.0) / 1.0e9;
}

template <typename LaunchFn>
double time_kernel(LaunchFn&& launch, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) {
        launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return static_cast<double>(elapsed_ms) / iters;
}

__device__ __forceinline__ float4 load_float4_or_zero(const float* ptr,
                                                       int remaining) {
    if (remaining >= kVecSize) {
        return *reinterpret_cast<const float4*>(ptr);
    }
    float4 out{0.0f, 0.0f, 0.0f, 0.0f};
    if (remaining > 0) {
        out.x = ptr[0];
    }
    if (remaining > 1) {
        out.y = ptr[1];
    }
    if (remaining > 2) {
        out.z = ptr[2];
    }
    return out;
}

__device__ __forceinline__ void store_float4_or_tail(float* ptr, float4 value,
                                                      int remaining) {
    if (remaining >= kVecSize) {
        *reinterpret_cast<float4*>(ptr) = value;
        return;
    }
    if (remaining > 0) {
        ptr[0] = value.x;
    }
    if (remaining > 1) {
        ptr[1] = value.y;
    }
    if (remaining > 2) {
        ptr[2] = value.z;
    }
}

__device__ __forceinline__ float sum4(float4 value, int remaining) {
    float out = 0.0f;
    if (remaining > 0) {
        out += value.x;
    }
    if (remaining > 1) {
        out += value.y;
    }
    if (remaining > 2) {
        out += value.z;
    }
    if (remaining > 3) {
        out += value.w;
    }
    return out;
}

__device__ __forceinline__ float sumsq4(float4 value, int remaining) {
    float out = 0.0f;
    if (remaining > 0) {
        out += value.x * value.x;
    }
    if (remaining > 1) {
        out += value.y * value.y;
    }
    if (remaining > 2) {
        out += value.z * value.z;
    }
    if (remaining > 3) {
        out += value.w * value.w;
    }
    return out;
}

template <int Width>
__device__ __forceinline__ float subwarp_reduce_sum(float value) {
    static_assert(Width == 16 || Width == 32, "unsupported reduction width");
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset, Width);
    }
    return value;
}

template <int BlockThreads>
__device__ __forceinline__ Pair block_reduce_pair(Pair value) {
    constexpr int Warps = BlockThreads / 32;
    __shared__ float warp_sum[Warps];
    __shared__ float warp_sumsq[Warps];
    __shared__ Pair broadcast;

    value.sum = subwarp_reduce_sum<32>(value.sum);
    value.sumsq = subwarp_reduce_sum<32>(value.sumsq);
    if ((threadIdx.x & 31) == 0) {
        const int warp = threadIdx.x >> 5;
        warp_sum[warp] = value.sum;
        warp_sumsq[warp] = value.sumsq;
    }
    __syncthreads();

    Pair reduced{0.0f, 0.0f};
    if (threadIdx.x < Warps) {
        reduced.sum = warp_sum[threadIdx.x];
        reduced.sumsq = warp_sumsq[threadIdx.x];
    }
    if (threadIdx.x < 32) {
        reduced.sum = subwarp_reduce_sum<32>(reduced.sum);
        reduced.sumsq = subwarp_reduce_sum<32>(reduced.sumsq);
    }
    if (threadIdx.x == 0) {
        broadcast = reduced;
    }
    __syncthreads();
    return broadcast;
}

template <int BlockThreads, int MaxClusterSize>
__device__ __forceinline__ Pair cluster_reduce_pair(cg::cluster_group cluster,
                                                     Pair block_value) {
    __shared__ float cluster_sum[MaxClusterSize];
    __shared__ float cluster_sumsq[MaxClusterSize];
    __shared__ Pair broadcast;
    const int block_rank = cluster.block_rank();
    const int blocks_per_cluster = cluster.dim_blocks().x;

    if (threadIdx.x < blocks_per_cluster) {
        float* remote_sum =
            cluster.map_shared_rank(&cluster_sum[block_rank], threadIdx.x);
        float* remote_sumsq =
            cluster.map_shared_rank(&cluster_sumsq[block_rank], threadIdx.x);
        *remote_sum = block_value.sum;
        *remote_sumsq = block_value.sumsq;
    }
    cluster.sync();

    Pair reduced{0.0f, 0.0f};
    if (threadIdx.x < blocks_per_cluster) {
        reduced.sum = cluster_sum[threadIdx.x];
        reduced.sumsq = cluster_sumsq[threadIdx.x];
    }
    if (threadIdx.x < 32) {
        reduced.sum = subwarp_reduce_sum<32>(reduced.sum);
        reduced.sumsq = subwarp_reduce_sum<32>(reduced.sumsq);
    }
    if (threadIdx.x == 0) {
        broadcast = reduced;
    }
    __syncthreads();
    return broadcast;
}

__global__ void init_input_kernel(float* data, std::size_t elements) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }
    const unsigned v = static_cast<unsigned>((idx * 1103515245ull + 12345ull) >> 16);
    data[idx] = static_cast<float>(v & 2047u) * (1.0f / 1024.0f) - 1.0f;
}

__global__ void init_affine_kernel(float* gamma, float* beta, int cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) {
        return;
    }
    gamma[col] = 0.75f + static_cast<float>(col % 31) * 0.015625f;
    beta[col] = static_cast<float>((col % 17) - 8) * 0.00390625f;
}

template <int ThreadPerRow>
__global__ void block_layernorm_kernel(const float* __restrict__ input,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta,
                                       float* __restrict__ output,
                                       int rows,
                                       int cols,
                                       float eps) {
    constexpr int RowsPerBlock = kBlockThreads / ThreadPerRow;
    const int row_local = threadIdx.x / ThreadPerRow;
    const int lane = threadIdx.x % ThreadPerRow;
    const int row = blockIdx.x * RowsPerBlock + row_local;
    const bool row_valid = row < rows;
    const float* row_in =
        row_valid ? input + static_cast<std::size_t>(row) * cols : input;
    float* row_out =
        row_valid ? output + static_cast<std::size_t>(row) * cols : output;

    Pair local{0.0f, 0.0f};
    for (int col = lane * kVecSize; col < cols;
         col += ThreadPerRow * kVecSize) {
        const int remaining = cols - col;
        const float4 x = row_valid ? load_float4_or_zero(row_in + col, remaining)
                                   : float4{0.0f, 0.0f, 0.0f, 0.0f};
        local.sum += sum4(x, remaining);
        local.sumsq += sumsq4(x, remaining);
    }
    const float row_sum = subwarp_reduce_sum<ThreadPerRow>(local.sum);
    const float row_sumsq = subwarp_reduce_sum<ThreadPerRow>(local.sumsq);
    const float mean = row_sum / cols;
    const float variance = fmaxf(row_sumsq / cols - mean * mean, 0.0f);
    const float inv_std = rsqrtf(variance + eps);

    if (!row_valid) {
        return;
    }
    for (int col = lane * kVecSize; col < cols;
         col += ThreadPerRow * kVecSize) {
        const int remaining = cols - col;
        const float4 x = load_float4_or_zero(row_in + col, remaining);
        const float4 g = load_float4_or_zero(gamma + col, remaining);
        const float4 b = load_float4_or_zero(beta + col, remaining);
        float4 y{
            remaining > 0 ? (x.x - mean) * inv_std * g.x + b.x : 0.0f,
            remaining > 1 ? (x.y - mean) * inv_std * g.y + b.y : 0.0f,
            remaining > 2 ? (x.z - mean) * inv_std * g.z + b.z : 0.0f,
            remaining > 3 ? (x.w - mean) * inv_std * g.w + b.w : 0.0f,
        };
        store_float4_or_tail(row_out + col, y, remaining);
    }
}

template <int BlockThreads>
__global__ void block_layernorm_staged_kernel(const float* __restrict__ input,
                                              const float* __restrict__ gamma,
                                              const float* __restrict__ beta,
                                              float* __restrict__ output,
                                              int rows,
                                              int cols,
                                              float eps) {
    extern __shared__ float tile[];
    const int row = blockIdx.x;
    const bool row_valid = row < rows;
    const float* row_in =
        row_valid ? input + static_cast<std::size_t>(row) * cols : input;
    float* row_out =
        row_valid ? output + static_cast<std::size_t>(row) * cols : output;

    for (int col = threadIdx.x * kVecSize; col < cols;
         col += BlockThreads * kVecSize) {
        const int remaining = cols - col;
        const float4 x = row_valid ? load_float4_or_zero(row_in + col, remaining)
                                   : float4{0.0f, 0.0f, 0.0f, 0.0f};
        store_float4_or_tail(tile + col, x, remaining);
    }
    __syncthreads();

    Pair local{0.0f, 0.0f};
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        const float x = row_valid ? tile[col] : 0.0f;
        local.sum += x;
        local.sumsq += x * x;
    }
    const Pair total = block_reduce_pair<BlockThreads>(local);
    const float mean = total.sum / cols;
    const float variance = fmaxf(total.sumsq / cols - mean * mean, 0.0f);
    const float inv_std = rsqrtf(variance + eps);

    if (!row_valid) {
        return;
    }
    for (int col = threadIdx.x * kVecSize; col < cols;
         col += BlockThreads * kVecSize) {
        const int remaining = cols - col;
        const float4 x = load_float4_or_zero(tile + col, remaining);
        const float4 g = load_float4_or_zero(gamma + col, remaining);
        const float4 b = load_float4_or_zero(beta + col, remaining);
        float4 y{
            remaining > 0 ? (x.x - mean) * inv_std * g.x + b.x : 0.0f,
            remaining > 1 ? (x.y - mean) * inv_std * g.y + b.y : 0.0f,
            remaining > 2 ? (x.z - mean) * inv_std * g.z + b.z : 0.0f,
            remaining > 3 ? (x.w - mean) * inv_std * g.w + b.w : 0.0f,
        };
        store_float4_or_tail(row_out + col, y, remaining);
    }
}

template <int BlockThreads, int MaxClusterSize>
__global__ void cluster_layernorm_staged_kernel(const float* __restrict__ input,
                                                const float* __restrict__ gamma,
                                                const float* __restrict__ beta,
                                                float* __restrict__ output,
                                                int rows,
                                                int cols,
                                                float eps,
                                                int slice_elems) {
    extern __shared__ float tile[];
    cg::cluster_group cluster = cg::this_cluster();
    const int blocks_per_cluster = cluster.dim_blocks().x;
    const int block_rank = cluster.block_rank();
    const int row = blockIdx.x / blocks_per_cluster;
    const bool row_valid = row < rows;

    const int slice_start = block_rank * slice_elems;
    const int local_cols =
        row_valid ? max(0, min(slice_elems, cols - slice_start)) : 0;

    const float* row_in =
        row_valid ? input + static_cast<std::size_t>(row) * cols : input;
    float* row_out =
        row_valid ? output + static_cast<std::size_t>(row) * cols : output;

    for (int col = threadIdx.x * kVecSize; col < local_cols;
         col += BlockThreads * kVecSize) {
        const int remaining = local_cols - col;
        const float4 x = load_float4_or_zero(row_in + slice_start + col, remaining);
        store_float4_or_tail(tile + col, x, remaining);
    }
    __syncthreads();

    Pair local{0.0f, 0.0f};
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        const float x = tile[col];
        local.sum += x;
        local.sumsq += x * x;
    }
    const Pair block_total = block_reduce_pair<BlockThreads>(local);
    const Pair row_total =
        cluster_reduce_pair<BlockThreads, MaxClusterSize>(cluster, block_total);
    const float mean = row_total.sum / cols;
    const float variance = fmaxf(row_total.sumsq / cols - mean * mean, 0.0f);
    const float inv_std = rsqrtf(variance + eps);

    if (!row_valid) {
        return;
    }
    for (int col = threadIdx.x * kVecSize; col < local_cols;
         col += BlockThreads * kVecSize) {
        const int remaining = local_cols - col;
        const float4 x = load_float4_or_zero(tile + col, remaining);
        const float4 g = load_float4_or_zero(gamma + slice_start + col, remaining);
        const float4 b = load_float4_or_zero(beta + slice_start + col, remaining);
        float4 y{
            remaining > 0 ? (x.x - mean) * inv_std * g.x + b.x : 0.0f,
            remaining > 1 ? (x.y - mean) * inv_std * g.y + b.y : 0.0f,
            remaining > 2 ? (x.z - mean) * inv_std * g.z + b.z : 0.0f,
            remaining > 3 ? (x.w - mean) * inv_std * g.w + b.w : 0.0f,
        };
        store_float4_or_tail(row_out + slice_start + col, y, remaining);
    }
}

template <int ThreadPerRow>
void launch_block_tpr(const float* d_input, const float* d_gamma,
                      const float* d_beta, float* d_output, int rows, int cols) {
    constexpr int RowsPerBlock = kBlockThreads / ThreadPerRow;
    const dim3 grid((rows + RowsPerBlock - 1) / RowsPerBlock);
    block_layernorm_kernel<ThreadPerRow>
        <<<grid, kBlockThreads>>>(d_input, d_gamma, d_beta, d_output, rows, cols,
                                  kEps);
}

void dispatch_block_tpr(int thread_per_row, const float* d_input,
                        const float* d_gamma, const float* d_beta,
                        float* d_output, int rows, int cols) {
    switch (thread_per_row) {
        case 16:
            launch_block_tpr<16>(d_input, d_gamma, d_beta, d_output, rows, cols);
            break;
        case 32:
            launch_block_tpr<32>(d_input, d_gamma, d_beta, d_output, rows, cols);
            break;
        default:
            throw std::runtime_error("unsupported block thread_per_row");
    }
}

DeviceInfo query_device_info(const Options& options) {
    DeviceInfo info;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    info.name = prop.name;
    info.major = prop.major;
    info.minor = prop.minor;
    info.max_dynamic_smem_per_block = prop.sharedMemPerBlockOptin;

    int cluster_launch = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch, 0));
    info.cluster_launch = cluster_launch != 0;
    if (info.cluster_launch) {
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(options.requested_cluster_size);
        config.blockDim = dim3(kBlockThreads);
        config.dynamicSmemBytes = 0;
        int max_cluster_size = 0;
        CUDA_CHECK(cudaOccupancyMaxPotentialClusterSize(
            &max_cluster_size,
            cluster_layernorm_staged_kernel<kBlockThreads, kMaxClusterSize>,
            &config));
        info.max_cluster_size = max_cluster_size;
        info.chosen_cluster_size =
            std::min({options.requested_cluster_size, max_cluster_size,
                      kMaxClusterSize});
    }
    return info;
}

VariantResult benchmark_block_variant(int thread_per_row, const Options& options,
                                      const DeviceInfo& device,
                                      const float* d_input, const float* d_gamma,
                                      const float* d_beta, float* d_output,
                                      int rows, int cols) {
    VariantResult result;
    result.thread_per_row = thread_per_row;
    if (thread_per_row == 256) {
        const std::size_t smem_bytes = static_cast<std::size_t>(cols) * sizeof(float);
        if (smem_bytes > static_cast<std::size_t>(device.max_dynamic_smem_per_block)) {
            return result;
        }
        CUDA_CHECK(cudaFuncSetAttribute(
            block_layernorm_staged_kernel<kBlockThreads>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)));
        auto launch = [&]() {
            block_layernorm_staged_kernel<kBlockThreads>
                <<<rows, kBlockThreads, smem_bytes>>>(
                    d_input, d_gamma, d_beta, d_output, rows, cols, kEps);
        };
        result.valid = true;
        result.avg_ms = time_kernel(launch, options.warmup, options.iters);
        result.gbps = throughput_gbps(rows, cols, result.avg_ms);
        return result;
    }

    auto launch = [&]() {
        dispatch_block_tpr(thread_per_row, d_input, d_gamma, d_beta, d_output,
                           rows, cols);
    };
    result.valid = true;
    result.avg_ms = time_kernel(launch, options.warmup, options.iters);
    result.gbps = throughput_gbps(rows, cols, result.avg_ms);
    return result;
}

VariantResult benchmark_cluster_variant(const Options& options,
                                        const DeviceInfo& device,
                                        const float* d_input,
                                        const float* d_gamma,
                                        const float* d_beta,
                                        float* d_output,
                                        int rows,
                                        int cols) {
    VariantResult result;
    result.thread_per_row = 256;
    if (device.chosen_cluster_size < 2) {
        return result;
    }
    const int slice_elems =
        (cols + device.chosen_cluster_size - 1) / device.chosen_cluster_size;
    const std::size_t smem_bytes =
        static_cast<std::size_t>(slice_elems) * sizeof(float);
    if (smem_bytes > static_cast<std::size_t>(device.max_dynamic_smem_per_block)) {
        return result;
    }
    CUDA_CHECK(cudaFuncSetAttribute(
        cluster_layernorm_staged_kernel<kBlockThreads, kMaxClusterSize>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
    auto launch = [&]() {
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(rows * device.chosen_cluster_size);
        config.blockDim = dim3(kBlockThreads);
        config.dynamicSmemBytes = static_cast<std::uint32_t>(smem_bytes);

        cudaLaunchAttribute attr{};
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim.x = device.chosen_cluster_size;
        attr.val.clusterDim.y = 1;
        attr.val.clusterDim.z = 1;
        config.attrs = &attr;
        config.numAttrs = 1;

        CUDA_CHECK(cudaLaunchKernelEx(
            &config, cluster_layernorm_staged_kernel<kBlockThreads, kMaxClusterSize>,
            d_input, d_gamma, d_beta, d_output, rows, cols, kEps, slice_elems));
    };
    result.valid = true;
    result.avg_ms = time_kernel(launch, options.warmup, options.iters);
    result.gbps = throughput_gbps(rows, cols, result.avg_ms);
    return result;
}

VariantResult pick_better(const VariantResult& lhs, const VariantResult& rhs) {
    if (!lhs.valid) {
        return rhs;
    }
    if (!rhs.valid) {
        return lhs;
    }
    return rhs.avg_ms < lhs.avg_ms ? rhs : lhs;
}

std::vector<int> unique_samples(std::initializer_list<int> values, int limit) {
    std::vector<int> out;
    for (int value : values) {
        if (value < 0 || value >= limit) {
            continue;
        }
        if (std::find(out.begin(), out.end(), value) == out.end()) {
            out.push_back(value);
        }
    }
    return out;
}

void verify_output_samples(const float* d_input, const float* d_gamma,
                           const float* d_beta, const float* d_output,
                           int rows, int cols, const char* label) {
    const auto sample_rows = unique_samples({0, rows / 2, rows - 1}, rows);
    const auto sample_cols =
        unique_samples({0, cols / 7, cols / 2, cols - 1}, cols);
    std::vector<float> h_row(cols);
    std::vector<float> h_gamma(cols);
    std::vector<float> h_beta(cols);
    CUDA_CHECK(cudaMemcpy(h_gamma.data(), d_gamma,
                          static_cast<std::size_t>(cols) * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_beta.data(), d_beta,
                          static_cast<std::size_t>(cols) * sizeof(float),
                          cudaMemcpyDeviceToHost));

    for (int row : sample_rows) {
        CUDA_CHECK(cudaMemcpy(h_row.data(),
                              d_input + static_cast<std::size_t>(row) * cols,
                              static_cast<std::size_t>(cols) * sizeof(float),
                              cudaMemcpyDeviceToHost));
        double sum = 0.0;
        double sumsq = 0.0;
        for (float value : h_row) {
            sum += value;
            sumsq += static_cast<double>(value) * value;
        }
        const double mean = sum / cols;
        const double variance = std::max(0.0, sumsq / cols - mean * mean);
        const double inv_std = 1.0 / std::sqrt(variance + kEps);
        for (int col : sample_cols) {
            float got = 0.0f;
            CUDA_CHECK(cudaMemcpy(&got,
                                  d_output +
                                      static_cast<std::size_t>(row) * cols + col,
                                  sizeof(float), cudaMemcpyDeviceToHost));
            const double ref =
                (static_cast<double>(h_row[col]) - mean) * inv_std *
                    static_cast<double>(h_gamma[col]) +
                static_cast<double>(h_beta[col]);
            const double atol = 5e-3;
            const double rtol = 5e-3;
            const double err = std::abs(static_cast<double>(got) - ref);
            if (err > atol + rtol * std::abs(ref)) {
                std::ostringstream oss;
                oss << label << " verification failed at row=" << row
                    << " col=" << col << " got=" << got << " ref=" << ref
                    << " err=" << err;
                throw std::runtime_error(oss.str());
            }
        }
    }
}

void verify_selected_variants(const Options& options, const DeviceInfo& device,
                              const VariantResult& block,
                              const VariantResult& cluster,
                              const float* d_input, const float* d_gamma,
                              const float* d_beta, float* d_output,
                              int rows, int cols) {
    if (!options.verify) {
        return;
    }
    Options one_run = options;
    one_run.warmup = 0;
    one_run.iters = 1;
    if (block.valid) {
        benchmark_block_variant(block.thread_per_row, one_run, device, d_input,
                                d_gamma, d_beta, d_output, rows, cols);
        verify_output_samples(d_input, d_gamma, d_beta, d_output, rows, cols,
                              "block");
    }
    if (cluster.valid) {
        benchmark_cluster_variant(one_run, device, d_input, d_gamma, d_beta,
                                  d_output, rows, cols);
        verify_output_samples(d_input, d_gamma, d_beta, d_output, rows, cols,
                              "cluster");
    }
}

void print_device_header(const Options& options, const DeviceInfo& device) {
    if (options.csv) {
        std::cout << "META,device_name," << device.name << "\n";
        std::cout << "META,sm," << device.major << "." << device.minor << "\n";
        std::cout << "META,batch_size," << options.batch_size << "\n";
        std::cout << "META,warmup," << options.warmup << "\n";
        std::cout << "META,iters," << options.iters << "\n";
        std::cout << "META,cluster_launch," << (device.cluster_launch ? 1 : 0)
                  << "\n";
        std::cout << "META,requested_cluster_size,"
                  << options.requested_cluster_size << "\n";
        std::cout << "META,max_cluster_size," << device.max_cluster_size << "\n";
        std::cout << "META,chosen_cluster_size," << device.chosen_cluster_size
                  << "\n";
        return;
    }
    std::cout << "Device: " << device.name << " (sm_" << device.major
              << device.minor << ")\n";
    std::cout << "Batch size: " << options.batch_size << "\n";
    std::cout << "Warmup: " << options.warmup << ", iters: " << options.iters
              << "\n";
    std::cout << "Cluster launch: " << (device.cluster_launch ? "yes" : "no")
              << ", chosen cluster size: " << device.chosen_cluster_size << "\n";
    std::cout << "Tuning block thread_per_row over:";
    for (int value : options.thread_per_row_values) {
        std::cout << " " << value;
    }
    std::cout << "\n\n";
}

void print_result(int cols, int rows, const VariantResult& block,
                  const VariantResult& cluster, const DeviceInfo& device,
                  bool csv) {
    const double cluster_vs_block =
        (block.valid && cluster.valid) ? block.avg_ms / cluster.avg_ms
                                       : std::numeric_limits<double>::quiet_NaN();
    if (csv) {
        std::cout << "RESULT," << cols << "," << rows << ",";
        if (block.valid) {
            std::cout << block.avg_ms << "," << block.gbps << ","
                      << block.thread_per_row << ",";
        } else {
            std::cout << "n/a,n/a,-1,";
        }
        if (cluster.valid) {
            std::cout << cluster.avg_ms << "," << cluster.gbps << ","
                      << cluster.thread_per_row << "," << device.chosen_cluster_size
                      << ",";
        } else {
            std::cout << "n/a,n/a,-1," << device.chosen_cluster_size << ",";
        }
        if (std::isnan(cluster_vs_block)) {
            std::cout << "n/a\n";
        } else {
            std::cout << cluster_vs_block << "\n";
        }
        return;
    }

    std::cout << "N=" << cols << "\n";
    std::cout << "  block:   " << std::fixed << std::setprecision(4)
              << block.avg_ms << " ms, " << std::setprecision(2) << block.gbps
              << " GB/s, tpr=" << block.thread_per_row << "\n";
    std::cout << "  cluster: " << std::fixed << std::setprecision(4)
              << cluster.avg_ms << " ms, " << std::setprecision(2)
              << cluster.gbps << " GB/s, cluster=" << device.chosen_cluster_size
              << "\n";
    std::cout << "  cluster/block speedup: " << std::setprecision(3)
              << cluster_vs_block << "x\n\n";
}

void run_shape(int cols, const Options& options, const DeviceInfo& device) {
    const int rows = options.batch_size;
    const std::size_t elements = static_cast<std::size_t>(rows) * cols;
    const std::size_t data_bytes = elements * sizeof(float);
    const std::size_t affine_bytes = static_cast<std::size_t>(cols) * sizeof(float);

    float* d_input = nullptr;
    float* d_gamma = nullptr;
    float* d_beta = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_gamma, affine_bytes));
    CUDA_CHECK(cudaMalloc(&d_beta, affine_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, data_bytes));

    const int init_threads = 256;
    init_input_kernel<<<(elements + init_threads - 1) / init_threads, init_threads>>>(
        d_input, elements);
    init_affine_kernel<<<(cols + init_threads - 1) / init_threads, init_threads>>>(
        d_gamma, d_beta, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    VariantResult best_block;
    for (int thread_per_row : options.thread_per_row_values) {
        best_block = pick_better(
            best_block,
            benchmark_block_variant(thread_per_row, options, device, d_input,
                                    d_gamma, d_beta, d_output, rows, cols));
    }
    const VariantResult cluster = benchmark_cluster_variant(
        options, device, d_input, d_gamma, d_beta, d_output, rows, cols);
    verify_selected_variants(options, device, best_block, cluster, d_input,
                             d_gamma, d_beta, d_output, rows, cols);
    print_result(cols, rows, best_block, cluster, device, options.csv);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_output));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const DeviceInfo device = query_device_info(options);
        print_device_header(options, device);
        for (int cols : options.n_values) {
            run_shape(cols, options, device);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
