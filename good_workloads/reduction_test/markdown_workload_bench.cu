#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <math_constants.h>

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
constexpr int kDefaultWarmup = 3;
constexpr int kDefaultIters = 10;
constexpr int kDefaultBatchSize = 16384;
constexpr int kDefaultClusterSize = 8;
constexpr float kRmsEps = 1e-6f;
constexpr std::array<int, 6> kDefaultNValues = {
    4096, 8192, 16384, 32768, 65536, 131072,
};
constexpr std::array<int, 3> kDefaultThreadPerRow = {16, 32, 256};

enum class Workload {
    Softmax,
    RmsNorm,
    CrossEntropy,
};

struct Options {
    int warmup = kDefaultWarmup;
    int iters = kDefaultIters;
    int batch_size = kDefaultBatchSize;
    int requested_cluster_size = kDefaultClusterSize;
    bool csv = false;
    bool verify = true;
    std::vector<int> n_values = {
        kDefaultNValues.begin(), kDefaultNValues.end(),
    };
    std::vector<int> thread_per_row_values = {
        kDefaultThreadPerRow.begin(), kDefaultThreadPerRow.end(),
    };
    std::vector<Workload> workloads = {
        Workload::Softmax,
        Workload::RmsNorm,
        Workload::CrossEntropy,
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

struct SizeResult {
    int cols = 0;
    int batch_size = 0;
    VariantResult block;
    VariantResult cluster;
    double cluster_vs_block = std::numeric_limits<double>::quiet_NaN();
};

struct WorkloadReport {
    Workload workload;
    std::vector<SizeResult> results;
};

Workload parse_workload_name(const std::string& name) {
    if (name == "softmax") {
        return Workload::Softmax;
    }
    if (name == "rmsnorm") {
        return Workload::RmsNorm;
    }
    if (name == "cross_entropy") {
        return Workload::CrossEntropy;
    }
    throw std::runtime_error("unknown workload: " + name);
}

std::string workload_name(Workload workload) {
    switch (workload) {
        case Workload::Softmax:
            return "softmax";
        case Workload::RmsNorm:
            return "rmsnorm";
        case Workload::CrossEntropy:
            return "cross_entropy";
    }
    return "unknown";
}

std::vector<int> parse_int_list(const std::string& spec) {
    std::vector<int> values;
    std::stringstream ss(spec);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) {
            continue;
        }
        values.push_back(std::stoi(part));
    }
    if (values.empty()) {
        throw std::runtime_error("list cannot be empty");
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
            opts.thread_per_row_values =
                parse_int_list(require_string("--thread-per-row-values"));
        } else if (arg == "--workload") {
            const std::string value = require_string("--workload");
            if (value == "all") {
                opts.workloads = {
                    Workload::Softmax,
                    Workload::RmsNorm,
                    Workload::CrossEntropy,
                };
            } else {
                opts.workloads = {parse_workload_name(value)};
            }
        } else if (arg == "--csv") {
            opts.csv = true;
        } else if (arg == "--no-verify") {
            opts.verify = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: markdown_workload_bench [--warmup N] [--iters N] "
                << "[--batch-size N] [--cluster-size N] [--n-values list] "
                << "[--thread-per-row-values list] "
                << "[--workload all|softmax|rmsnorm|cross_entropy] [--csv] "
                << "[--no-verify]\n";
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
        if ((value != 8 && value != 16 && value != 32 && value != 256) ||
            (kBlockThreads % value) != 0) {
            throw std::runtime_error(
                "thread_per_row values must be one of 8, 16, 32, or 256");
        }
    }
    for (int value : opts.n_values) {
        if (value <= 0) {
            throw std::runtime_error("n-values must be positive");
        }
    }
    return opts;
}

std::string fmt_double(double value, int precision = 2) {
    if (std::isnan(value)) {
        return "n/a";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

double modeled_bytes(Workload workload, int batch_size, int cols) {
    const double elements = static_cast<double>(batch_size) * cols;
    switch (workload) {
        case Workload::Softmax:
            return elements * sizeof(float) * 4.0;
        case Workload::RmsNorm:
            return elements * sizeof(float) * 4.0;
        case Workload::CrossEntropy:
            return elements * sizeof(float) * 2.0 +
                   static_cast<double>(batch_size) * sizeof(float);
    }
    return 0.0;
}

double throughput_gbps(Workload workload, int batch_size, int cols, double avg_ms) {
    return modeled_bytes(workload, batch_size, cols) / (avg_ms / 1000.0) / 1.0e9;
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

__device__ __forceinline__ float reduce_max4(float4 v) {
    float m = fmaxf(v.x, v.y);
    m = fmaxf(m, v.z);
    m = fmaxf(m, v.w);
    return m;
}

__device__ __forceinline__ float reduce_sumsq4(float4 v) {
    return v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
}

template <int Width>
__device__ __forceinline__ float subwarp_reduce_sum(float value) {
    static_assert(Width == 8 || Width == 16 || Width == 32);
    #pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset, Width);
    }
    return value;
}

template <int Width>
__device__ __forceinline__ float subwarp_reduce_max(float value) {
    static_assert(Width == 8 || Width == 16 || Width == 32);
    #pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset, Width));
    }
    return value;
}

template <int BlockThreads>
__device__ __forceinline__ float block_reduce_sum(float value) {
    __shared__ float warp_sums[BlockThreads / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    value = subwarp_reduce_sum<32>(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (threadIdx.x < BlockThreads / 32) {
        block_sum = warp_sums[lane];
    }
    if (warp == 0) {
        block_sum = subwarp_reduce_sum<32>(block_sum);
    }
    __shared__ float broadcast;
    if (threadIdx.x == 0) {
        broadcast = block_sum;
    }
    __syncthreads();
    return broadcast;
}

template <int BlockThreads>
__device__ __forceinline__ float block_reduce_max(float value) {
    __shared__ float warp_max[BlockThreads / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    value = subwarp_reduce_max<32>(value);
    if (lane == 0) {
        warp_max[warp] = value;
    }
    __syncthreads();

    float block_max = -CUDART_INF_F;
    if (threadIdx.x < BlockThreads / 32) {
        block_max = warp_max[lane];
    }
    if (warp == 0) {
        block_max = subwarp_reduce_max<32>(block_max);
    }
    __shared__ float broadcast;
    if (threadIdx.x == 0) {
        broadcast = block_max;
    }
    __syncthreads();
    return broadcast;
}

template <int ThreadPerRow>
__device__ __forceinline__ float cluster_allreduce_sum(cg::cluster_group cluster,
                                                       float local_value,
                                                       float* row_slots,
                                                       int lane_in_row) {
    const int block_rank = cluster.block_rank();
    const int blocks_per_cluster = cluster.dim_blocks().x;

    if (lane_in_row < blocks_per_cluster) {
        float* remote_slot =
            cluster.map_shared_rank(row_slots + block_rank, lane_in_row);
        *remote_slot = local_value;
    }
    cluster.sync();

    float reduced = 0.0f;
    if (lane_in_row < blocks_per_cluster) {
        reduced = row_slots[lane_in_row];
    }
    reduced = subwarp_reduce_sum<ThreadPerRow>(reduced);
    cluster.sync();
    return reduced;
}

template <int ThreadPerRow>
__device__ __forceinline__ float cluster_allreduce_max(cg::cluster_group cluster,
                                                       float local_value,
                                                       float* row_slots,
                                                       int lane_in_row) {
    const int block_rank = cluster.block_rank();
    const int blocks_per_cluster = cluster.dim_blocks().x;

    if (lane_in_row < blocks_per_cluster) {
        float* remote_slot =
            cluster.map_shared_rank(row_slots + block_rank, lane_in_row);
        *remote_slot = local_value;
    }
    cluster.sync();

    float reduced = -CUDART_INF_F;
    if (lane_in_row < blocks_per_cluster) {
        reduced = row_slots[lane_in_row];
    }
    reduced = subwarp_reduce_max<ThreadPerRow>(reduced);
    cluster.sync();
    return reduced;
}

__device__ __forceinline__ float4 load_float4_or_zero(const float* ptr, int remaining) {
    if (remaining >= kVecSize && ((reinterpret_cast<std::uintptr_t>(ptr) & 0xfu) == 0u)) {
        return *reinterpret_cast<const float4*>(ptr);
    }
    float4 v{0.0f, 0.0f, 0.0f, 0.0f};
    if (remaining > 0) {
        v.x = ptr[0];
    }
    if (remaining > 1) {
        v.y = ptr[1];
    }
    if (remaining > 2) {
        v.z = ptr[2];
    }
    if (remaining > 3) {
        v.w = ptr[3];
    }
    return v;
}

__device__ __forceinline__ void store_float4_or_tail(float* ptr, float4 v,
                                                     int remaining) {
    if (remaining >= kVecSize && ((reinterpret_cast<std::uintptr_t>(ptr) & 0xfu) == 0u)) {
        *reinterpret_cast<float4*>(ptr) = v;
        return;
    }
    if (remaining > 0) {
        ptr[0] = v.x;
    }
    if (remaining > 1) {
        ptr[1] = v.y;
    }
    if (remaining > 2) {
        ptr[2] = v.z;
    }
    if (remaining > 3) {
        ptr[3] = v.w;
    }
}

template <int BlockThreads>
__global__ void block_softmax_staged_kernel(const float* __restrict__ input,
                                            float* __restrict__ output,
                                            int rows,
                                            int cols) {
    extern __shared__ float tile[];
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float* row_in = input + static_cast<std::size_t>(row) * cols;
    float* row_out = output + static_cast<std::size_t>(row) * cols;

    for (int col = threadIdx.x * kVecSize; col < cols; col += BlockThreads * kVecSize) {
        const int remaining = cols - col;
        float4 x = load_float4_or_zero(row_in + col, remaining);
        store_float4_or_tail(tile + col, x, remaining);
    }
    __syncthreads();

    float local_max = -CUDART_INF_F;
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        local_max = fmaxf(local_max, tile[col]);
    }
    const float row_max = block_reduce_max<BlockThreads>(local_max);

    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        local_sum += __expf(tile[col] - row_max);
    }
    const float row_sum = block_reduce_sum<BlockThreads>(local_sum);

    for (int col = threadIdx.x * kVecSize; col < cols; col += BlockThreads * kVecSize) {
        const int remaining = cols - col;
        float4 x = load_float4_or_zero(tile + col, remaining);
        float4 y{
            remaining > 0 ? __expf(x.x - row_max) / row_sum : 0.0f,
            remaining > 1 ? __expf(x.y - row_max) / row_sum : 0.0f,
            remaining > 2 ? __expf(x.z - row_max) / row_sum : 0.0f,
            remaining > 3 ? __expf(x.w - row_max) / row_sum : 0.0f,
        };
        store_float4_or_tail(row_out + col, y, remaining);
    }
}

template <int BlockThreads>
__global__ void block_rmsnorm_staged_kernel(const float* __restrict__ input,
                                            const float* __restrict__ weight,
                                            float* __restrict__ output,
                                            int rows,
                                            int cols,
                                            float eps) {
    extern __shared__ float tile[];
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float* row_in = input + static_cast<std::size_t>(row) * cols;
    float* row_out = output + static_cast<std::size_t>(row) * cols;

    for (int col = threadIdx.x * kVecSize; col < cols; col += BlockThreads * kVecSize) {
        const int remaining = cols - col;
        float4 x = load_float4_or_zero(row_in + col, remaining);
        store_float4_or_tail(tile + col, x, remaining);
    }
    __syncthreads();

    float local_sumsq = 0.0f;
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        const float x = tile[col];
        local_sumsq += x * x;
    }
    const float row_sumsq = block_reduce_sum<BlockThreads>(local_sumsq);
    const float inv_rms = rsqrtf(row_sumsq / cols + eps);

    for (int col = threadIdx.x * kVecSize; col < cols; col += BlockThreads * kVecSize) {
        const int remaining = cols - col;
        float4 x = load_float4_or_zero(tile + col, remaining);
        float4 w = load_float4_or_zero(weight + col, remaining);
        float4 y{
            remaining > 0 ? x.x * inv_rms * w.x : 0.0f,
            remaining > 1 ? x.y * inv_rms * w.y : 0.0f,
            remaining > 2 ? x.z * inv_rms * w.z : 0.0f,
            remaining > 3 ? x.w * inv_rms * w.w : 0.0f,
        };
        store_float4_or_tail(row_out + col, y, remaining);
    }
}

template <int BlockThreads>
__global__ void block_cross_entropy_staged_kernel(const float* __restrict__ logits,
                                                  const int* __restrict__ targets,
                                                  float* __restrict__ losses,
                                                  int rows,
                                                  int cols) {
    extern __shared__ float tile[];
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float* row_logits = logits + static_cast<std::size_t>(row) * cols;
    const int target = targets[row];

    for (int col = threadIdx.x * kVecSize; col < cols; col += BlockThreads * kVecSize) {
        const int remaining = cols - col;
        float4 x = load_float4_or_zero(row_logits + col, remaining);
        store_float4_or_tail(tile + col, x, remaining);
    }
    __syncthreads();

    float local_max = -CUDART_INF_F;
    float local_target = -CUDART_INF_F;
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        const float x = tile[col];
        local_max = fmaxf(local_max, x);
        if (col == target) {
            local_target = x;
        }
    }
    const float row_max = block_reduce_max<BlockThreads>(local_max);
    const float target_logit = block_reduce_max<BlockThreads>(local_target);

    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        local_sum += __expf(tile[col] - row_max);
    }
    const float row_sum = block_reduce_sum<BlockThreads>(local_sum);

    if (threadIdx.x == 0) {
        losses[row] = logf(row_sum) - (target_logit - row_max);
    }
}

template <int BlockThreads, int MaxClusterSize>
__device__ __forceinline__ float cluster_block_reduce_sum(cg::cluster_group cluster,
                                                          float block_value) {
    __shared__ float cluster_slots[MaxClusterSize];
    __shared__ float broadcast;
    const int block_rank = cluster.block_rank();
    const int blocks_per_cluster = cluster.dim_blocks().x;

    if (threadIdx.x < blocks_per_cluster) {
        float* remote_slot =
            cluster.map_shared_rank(&cluster_slots[block_rank], threadIdx.x);
        *remote_slot = block_value;
    }
    cluster.sync();

    float reduced = 0.0f;
    if (threadIdx.x < blocks_per_cluster) {
        reduced = cluster_slots[threadIdx.x];
    }
    if (threadIdx.x < 32) {
        reduced = subwarp_reduce_sum<32>(reduced);
    }
    if (threadIdx.x == 0) {
        broadcast = reduced;
    }
    __syncthreads();
    cluster.sync();
    return broadcast;
}

template <int BlockThreads, int MaxClusterSize>
__device__ __forceinline__ float cluster_block_reduce_max(cg::cluster_group cluster,
                                                          float block_value) {
    __shared__ float cluster_slots[MaxClusterSize];
    __shared__ float broadcast;
    const int block_rank = cluster.block_rank();
    const int blocks_per_cluster = cluster.dim_blocks().x;

    if (threadIdx.x < blocks_per_cluster) {
        float* remote_slot =
            cluster.map_shared_rank(&cluster_slots[block_rank], threadIdx.x);
        *remote_slot = block_value;
    }
    cluster.sync();

    float reduced = -CUDART_INF_F;
    if (threadIdx.x < blocks_per_cluster) {
        reduced = cluster_slots[threadIdx.x];
    }
    if (threadIdx.x < 32) {
        reduced = subwarp_reduce_max<32>(reduced);
    }
    if (threadIdx.x == 0) {
        broadcast = reduced;
    }
    __syncthreads();
    cluster.sync();
    return broadcast;
}

template <int BlockThreads, int MaxClusterSize>
__global__ void cluster_softmax_staged_kernel(const float* __restrict__ input,
                                              float* __restrict__ output,
                                              int rows,
                                              int cols,
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

    const float* row_in = row_valid ? input + static_cast<std::size_t>(row) * cols : input;
    float* row_out = row_valid ? output + static_cast<std::size_t>(row) * cols : output;

    for (int col = threadIdx.x * kVecSize; col < local_cols;
         col += BlockThreads * kVecSize) {
        const int remaining = local_cols - col;
        float4 x = load_float4_or_zero(row_in + slice_start + col, remaining);
        store_float4_or_tail(tile + col, x, remaining);
    }
    __syncthreads();

    float local_max = -CUDART_INF_F;
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        local_max = fmaxf(local_max, tile[col]);
    }
    const float block_max = block_reduce_max<BlockThreads>(local_max);
    const float row_max =
        cluster_block_reduce_max<BlockThreads, MaxClusterSize>(cluster, block_max);

    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        local_sum += __expf(tile[col] - row_max);
    }
    const float block_sum = block_reduce_sum<BlockThreads>(local_sum);
    const float row_sum =
        cluster_block_reduce_sum<BlockThreads, MaxClusterSize>(cluster, block_sum);

    if (!row_valid) {
        return;
    }
    for (int col = threadIdx.x * kVecSize; col < local_cols;
         col += BlockThreads * kVecSize) {
        const int remaining = local_cols - col;
        float4 x = load_float4_or_zero(tile + col, remaining);
        float4 y{
            remaining > 0 ? __expf(x.x - row_max) / row_sum : 0.0f,
            remaining > 1 ? __expf(x.y - row_max) / row_sum : 0.0f,
            remaining > 2 ? __expf(x.z - row_max) / row_sum : 0.0f,
            remaining > 3 ? __expf(x.w - row_max) / row_sum : 0.0f,
        };
        store_float4_or_tail(row_out + slice_start + col, y, remaining);
    }
}

template <int BlockThreads, int MaxClusterSize>
__global__ void cluster_rmsnorm_staged_kernel(const float* __restrict__ input,
                                              const float* __restrict__ weight,
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

    const float* row_in = row_valid ? input + static_cast<std::size_t>(row) * cols : input;
    float* row_out = row_valid ? output + static_cast<std::size_t>(row) * cols : output;

    for (int col = threadIdx.x * kVecSize; col < local_cols;
         col += BlockThreads * kVecSize) {
        const int remaining = local_cols - col;
        float4 x = load_float4_or_zero(row_in + slice_start + col, remaining);
        store_float4_or_tail(tile + col, x, remaining);
    }
    __syncthreads();

    float local_sumsq = 0.0f;
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        const float x = tile[col];
        local_sumsq += x * x;
    }
    const float block_sumsq = block_reduce_sum<BlockThreads>(local_sumsq);
    const float row_sumsq =
        cluster_block_reduce_sum<BlockThreads, MaxClusterSize>(cluster, block_sumsq);
    const float inv_rms = rsqrtf(row_sumsq / cols + eps);

    if (!row_valid) {
        return;
    }
    for (int col = threadIdx.x * kVecSize; col < local_cols;
         col += BlockThreads * kVecSize) {
        const int remaining = local_cols - col;
        float4 x = load_float4_or_zero(tile + col, remaining);
        float4 w = load_float4_or_zero(weight + slice_start + col, remaining);
        float4 y{
            remaining > 0 ? x.x * inv_rms * w.x : 0.0f,
            remaining > 1 ? x.y * inv_rms * w.y : 0.0f,
            remaining > 2 ? x.z * inv_rms * w.z : 0.0f,
            remaining > 3 ? x.w * inv_rms * w.w : 0.0f,
        };
        store_float4_or_tail(row_out + slice_start + col, y, remaining);
    }
}

template <int BlockThreads, int MaxClusterSize>
__global__ void cluster_cross_entropy_staged_kernel(
    const float* __restrict__ logits,
    const int* __restrict__ targets,
    float* __restrict__ losses,
    int rows,
    int cols,
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

    const float* row_logits =
        row_valid ? logits + static_cast<std::size_t>(row) * cols : logits;
    const int target = row_valid ? targets[row] : 0;

    for (int col = threadIdx.x * kVecSize; col < local_cols;
         col += BlockThreads * kVecSize) {
        const int remaining = local_cols - col;
        float4 x = load_float4_or_zero(row_logits + slice_start + col, remaining);
        store_float4_or_tail(tile + col, x, remaining);
    }
    __syncthreads();

    float local_max = -CUDART_INF_F;
    float local_target = -CUDART_INF_F;
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        const float x = tile[col];
        local_max = fmaxf(local_max, x);
        if (row_valid && slice_start + col == target) {
            local_target = x;
        }
    }
    const float block_max = block_reduce_max<BlockThreads>(local_max);
    const float block_target = block_reduce_max<BlockThreads>(local_target);
    const float row_max =
        cluster_block_reduce_max<BlockThreads, MaxClusterSize>(cluster, block_max);
    const float target_logit =
        cluster_block_reduce_max<BlockThreads, MaxClusterSize>(cluster, block_target);

    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        local_sum += __expf(tile[col] - row_max);
    }
    const float block_sum = block_reduce_sum<BlockThreads>(local_sum);
    const float row_sum =
        cluster_block_reduce_sum<BlockThreads, MaxClusterSize>(cluster, block_sum);

    if (row_valid && block_rank == 0 && threadIdx.x == 0) {
        losses[row] = logf(row_sum) - (target_logit - row_max);
    }
}

template <int ThreadPerRow>
__global__ void block_softmax_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int rows,
                                     int cols) {
    constexpr int RowsPerBlock = kBlockThreads / ThreadPerRow;
    const int row_local = threadIdx.x / ThreadPerRow;
    const int lane_in_row = threadIdx.x % ThreadPerRow;
    const int row = blockIdx.x * RowsPerBlock + row_local;
    const bool row_valid = row < rows;

    const float* row_in = row_valid ? input + static_cast<std::size_t>(row) * cols : input;
    float* row_out = row_valid ? output + static_cast<std::size_t>(row) * cols : output;

    float local_max = -CUDART_INF_F;
    for (int col = lane_in_row * kVecSize; col < cols; col += ThreadPerRow * kVecSize) {
        const int remaining = cols - col;
        float4 x = row_valid ? load_float4_or_zero(row_in + col, remaining)
                             : float4{0.0f, 0.0f, 0.0f, 0.0f};
        local_max = fmaxf(local_max, reduce_max4(x));
    }
    const float row_max = subwarp_reduce_max<ThreadPerRow>(local_max);

    float local_sum = 0.0f;
    for (int col = lane_in_row * kVecSize; col < cols; col += ThreadPerRow * kVecSize) {
        const int remaining = cols - col;
        float4 x = row_valid ? load_float4_or_zero(row_in + col, remaining)
                             : float4{0.0f, 0.0f, 0.0f, 0.0f};
        if (remaining > 0) {
            local_sum += __expf(x.x - row_max);
        }
        if (remaining > 1) {
            local_sum += __expf(x.y - row_max);
        }
        if (remaining > 2) {
            local_sum += __expf(x.z - row_max);
        }
        if (remaining > 3) {
            local_sum += __expf(x.w - row_max);
        }
    }
    const float row_sum = subwarp_reduce_sum<ThreadPerRow>(local_sum);

    if (!row_valid) {
        return;
    }
    for (int col = lane_in_row * kVecSize; col < cols; col += ThreadPerRow * kVecSize) {
        const int remaining = cols - col;
        float4 x = load_float4_or_zero(row_in + col, remaining);
        float4 y{
            remaining > 0 ? __expf(x.x - row_max) / row_sum : 0.0f,
            remaining > 1 ? __expf(x.y - row_max) / row_sum : 0.0f,
            remaining > 2 ? __expf(x.z - row_max) / row_sum : 0.0f,
            remaining > 3 ? __expf(x.w - row_max) / row_sum : 0.0f,
        };
        store_float4_or_tail(row_out + col, y, remaining);
    }
}

template <int ThreadPerRow>
__global__ void block_rmsnorm_kernel(const float* __restrict__ input,
                                     const float* __restrict__ weight,
                                     float* __restrict__ output,
                                     int rows,
                                     int cols,
                                     float eps) {
    constexpr int RowsPerBlock = kBlockThreads / ThreadPerRow;
    const int row_local = threadIdx.x / ThreadPerRow;
    const int lane_in_row = threadIdx.x % ThreadPerRow;
    const int row = blockIdx.x * RowsPerBlock + row_local;
    const bool row_valid = row < rows;

    const float* row_in = row_valid ? input + static_cast<std::size_t>(row) * cols : input;
    float* row_out = row_valid ? output + static_cast<std::size_t>(row) * cols : output;

    float local_sumsq = 0.0f;
    for (int col = lane_in_row * kVecSize; col < cols; col += ThreadPerRow * kVecSize) {
        const int remaining = cols - col;
        float4 x = row_valid ? load_float4_or_zero(row_in + col, remaining)
                             : float4{0.0f, 0.0f, 0.0f, 0.0f};
        if (remaining >= kVecSize) {
            local_sumsq += reduce_sumsq4(x);
        } else {
            if (remaining > 0) {
                local_sumsq += x.x * x.x;
            }
            if (remaining > 1) {
                local_sumsq += x.y * x.y;
            }
            if (remaining > 2) {
                local_sumsq += x.z * x.z;
            }
        }
    }
    const float row_sumsq = subwarp_reduce_sum<ThreadPerRow>(local_sumsq);
    const float inv_rms = rsqrtf(row_sumsq / cols + eps);

    if (!row_valid) {
        return;
    }
    for (int col = lane_in_row * kVecSize; col < cols; col += ThreadPerRow * kVecSize) {
        const int remaining = cols - col;
        float4 x = load_float4_or_zero(row_in + col, remaining);
        float4 w = load_float4_or_zero(weight + col, remaining);
        float4 y{
            remaining > 0 ? x.x * inv_rms * w.x : 0.0f,
            remaining > 1 ? x.y * inv_rms * w.y : 0.0f,
            remaining > 2 ? x.z * inv_rms * w.z : 0.0f,
            remaining > 3 ? x.w * inv_rms * w.w : 0.0f,
        };
        store_float4_or_tail(row_out + col, y, remaining);
    }
}

template <int ThreadPerRow>
__global__ void block_cross_entropy_kernel(const float* __restrict__ logits,
                                           const int* __restrict__ targets,
                                           float* __restrict__ losses,
                                           int rows,
                                           int cols) {
    constexpr int RowsPerBlock = kBlockThreads / ThreadPerRow;
    const int row_local = threadIdx.x / ThreadPerRow;
    const int lane_in_row = threadIdx.x % ThreadPerRow;
    const int row = blockIdx.x * RowsPerBlock + row_local;
    const bool row_valid = row < rows;

    const float* row_logits =
        row_valid ? logits + static_cast<std::size_t>(row) * cols : logits;
    const int target = row_valid ? targets[row] : 0;

    float local_max = -CUDART_INF_F;
    float local_target = -CUDART_INF_F;
    for (int col = lane_in_row * kVecSize; col < cols; col += ThreadPerRow * kVecSize) {
        const int remaining = cols - col;
        float4 x = row_valid ? load_float4_or_zero(row_logits + col, remaining)
                             : float4{0.0f, 0.0f, 0.0f, 0.0f};
        local_max = fmaxf(local_max, reduce_max4(x));
        if (row_valid) {
            if (remaining > 0 && target == col + 0) {
                local_target = x.x;
            }
            if (remaining > 1 && target == col + 1) {
                local_target = x.y;
            }
            if (remaining > 2 && target == col + 2) {
                local_target = x.z;
            }
            if (remaining > 3 && target == col + 3) {
                local_target = x.w;
            }
        }
    }
    const float row_max = subwarp_reduce_max<ThreadPerRow>(local_max);
    const float target_logit = subwarp_reduce_max<ThreadPerRow>(local_target);

    float local_sum = 0.0f;
    for (int col = lane_in_row * kVecSize; col < cols; col += ThreadPerRow * kVecSize) {
        const int remaining = cols - col;
        float4 x = row_valid ? load_float4_or_zero(row_logits + col, remaining)
                             : float4{0.0f, 0.0f, 0.0f, 0.0f};
        if (remaining > 0) {
            local_sum += __expf(x.x - row_max);
        }
        if (remaining > 1) {
            local_sum += __expf(x.y - row_max);
        }
        if (remaining > 2) {
            local_sum += __expf(x.z - row_max);
        }
        if (remaining > 3) {
            local_sum += __expf(x.w - row_max);
        }
    }
    const float row_sum = subwarp_reduce_sum<ThreadPerRow>(local_sum);

    if (row_valid && lane_in_row == 0) {
        losses[row] = logf(row_sum) - (target_logit - row_max);
    }
}

template <int ThreadPerRow, int MaxClusterSize>
__global__ void cluster_softmax_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int rows,
                                       int cols) {
    constexpr int RowsPerBlock = kBlockThreads / ThreadPerRow;
    __shared__ float reduction_buffer[RowsPerBlock][MaxClusterSize];

    cg::cluster_group cluster = cg::this_cluster();
    const int blocks_per_cluster = cluster.dim_blocks().x;
    const int block_rank = cluster.block_rank();

    const int row_local = threadIdx.x / ThreadPerRow;
    const int lane_in_row = threadIdx.x % ThreadPerRow;
    const int cluster_row_block = blockIdx.x / blocks_per_cluster;
    const int row = cluster_row_block * RowsPerBlock + row_local;
    const bool row_valid = row < rows;

    const float* row_in = row_valid ? input + static_cast<std::size_t>(row) * cols : input;
    float* row_out = row_valid ? output + static_cast<std::size_t>(row) * cols : output;
    float* row_slots = reduction_buffer[row_local];

    float local_max = -CUDART_INF_F;
    for (int col = (block_rank * ThreadPerRow + lane_in_row) * kVecSize;
         col < cols;
         col += blocks_per_cluster * ThreadPerRow * kVecSize) {
        const int remaining = cols - col;
        float4 x = row_valid ? load_float4_or_zero(row_in + col, remaining)
                             : float4{0.0f, 0.0f, 0.0f, 0.0f};
        local_max = fmaxf(local_max, reduce_max4(x));
    }
    local_max = subwarp_reduce_max<ThreadPerRow>(local_max);
    const float row_max = cluster_allreduce_max<ThreadPerRow>(
        cluster, local_max, row_slots, lane_in_row);

    float local_sum = 0.0f;
    for (int col = (block_rank * ThreadPerRow + lane_in_row) * kVecSize;
         col < cols;
         col += blocks_per_cluster * ThreadPerRow * kVecSize) {
        const int remaining = cols - col;
        float4 x = row_valid ? load_float4_or_zero(row_in + col, remaining)
                             : float4{0.0f, 0.0f, 0.0f, 0.0f};
        if (remaining > 0) {
            local_sum += __expf(x.x - row_max);
        }
        if (remaining > 1) {
            local_sum += __expf(x.y - row_max);
        }
        if (remaining > 2) {
            local_sum += __expf(x.z - row_max);
        }
        if (remaining > 3) {
            local_sum += __expf(x.w - row_max);
        }
    }
    local_sum = subwarp_reduce_sum<ThreadPerRow>(local_sum);
    const float row_sum = cluster_allreduce_sum<ThreadPerRow>(
        cluster, local_sum, row_slots, lane_in_row);

    if (!row_valid) {
        return;
    }
    for (int col = (block_rank * ThreadPerRow + lane_in_row) * kVecSize;
         col < cols;
         col += blocks_per_cluster * ThreadPerRow * kVecSize) {
        const int remaining = cols - col;
        float4 x = load_float4_or_zero(row_in + col, remaining);
        float4 y{
            remaining > 0 ? __expf(x.x - row_max) / row_sum : 0.0f,
            remaining > 1 ? __expf(x.y - row_max) / row_sum : 0.0f,
            remaining > 2 ? __expf(x.z - row_max) / row_sum : 0.0f,
            remaining > 3 ? __expf(x.w - row_max) / row_sum : 0.0f,
        };
        store_float4_or_tail(row_out + col, y, remaining);
    }
}

template <int ThreadPerRow, int MaxClusterSize>
__global__ void cluster_rmsnorm_kernel(const float* __restrict__ input,
                                       const float* __restrict__ weight,
                                       float* __restrict__ output,
                                       int rows,
                                       int cols,
                                       float eps) {
    constexpr int RowsPerBlock = kBlockThreads / ThreadPerRow;
    __shared__ float reduction_buffer[RowsPerBlock][MaxClusterSize];

    cg::cluster_group cluster = cg::this_cluster();
    const int blocks_per_cluster = cluster.dim_blocks().x;
    const int block_rank = cluster.block_rank();

    const int row_local = threadIdx.x / ThreadPerRow;
    const int lane_in_row = threadIdx.x % ThreadPerRow;
    const int cluster_row_block = blockIdx.x / blocks_per_cluster;
    const int row = cluster_row_block * RowsPerBlock + row_local;
    const bool row_valid = row < rows;

    const float* row_in = row_valid ? input + static_cast<std::size_t>(row) * cols : input;
    float* row_out = row_valid ? output + static_cast<std::size_t>(row) * cols : output;
    float* row_slots = reduction_buffer[row_local];

    float local_sumsq = 0.0f;
    for (int col = (block_rank * ThreadPerRow + lane_in_row) * kVecSize;
         col < cols;
         col += blocks_per_cluster * ThreadPerRow * kVecSize) {
        const int remaining = cols - col;
        float4 x = row_valid ? load_float4_or_zero(row_in + col, remaining)
                             : float4{0.0f, 0.0f, 0.0f, 0.0f};
        if (remaining >= kVecSize) {
            local_sumsq += reduce_sumsq4(x);
        } else {
            if (remaining > 0) {
                local_sumsq += x.x * x.x;
            }
            if (remaining > 1) {
                local_sumsq += x.y * x.y;
            }
            if (remaining > 2) {
                local_sumsq += x.z * x.z;
            }
        }
    }
    local_sumsq = subwarp_reduce_sum<ThreadPerRow>(local_sumsq);
    const float row_sumsq = cluster_allreduce_sum<ThreadPerRow>(
        cluster, local_sumsq, row_slots, lane_in_row);
    const float inv_rms = rsqrtf(row_sumsq / cols + eps);

    if (!row_valid) {
        return;
    }
    for (int col = (block_rank * ThreadPerRow + lane_in_row) * kVecSize;
         col < cols;
         col += blocks_per_cluster * ThreadPerRow * kVecSize) {
        const int remaining = cols - col;
        float4 x = load_float4_or_zero(row_in + col, remaining);
        float4 w = load_float4_or_zero(weight + col, remaining);
        float4 y{
            remaining > 0 ? x.x * inv_rms * w.x : 0.0f,
            remaining > 1 ? x.y * inv_rms * w.y : 0.0f,
            remaining > 2 ? x.z * inv_rms * w.z : 0.0f,
            remaining > 3 ? x.w * inv_rms * w.w : 0.0f,
        };
        store_float4_or_tail(row_out + col, y, remaining);
    }
}

template <int ThreadPerRow, int MaxClusterSize>
__global__ void cluster_cross_entropy_kernel(const float* __restrict__ logits,
                                             const int* __restrict__ targets,
                                             float* __restrict__ losses,
                                             int rows,
                                             int cols) {
    constexpr int RowsPerBlock = kBlockThreads / ThreadPerRow;
    __shared__ float reduction_buffer[RowsPerBlock][MaxClusterSize];

    cg::cluster_group cluster = cg::this_cluster();
    const int blocks_per_cluster = cluster.dim_blocks().x;
    const int block_rank = cluster.block_rank();

    const int row_local = threadIdx.x / ThreadPerRow;
    const int lane_in_row = threadIdx.x % ThreadPerRow;
    const int cluster_row_block = blockIdx.x / blocks_per_cluster;
    const int row = cluster_row_block * RowsPerBlock + row_local;
    const bool row_valid = row < rows;

    const float* row_logits =
        row_valid ? logits + static_cast<std::size_t>(row) * cols : logits;
    const int target = row_valid ? targets[row] : 0;
    float* row_slots = reduction_buffer[row_local];

    float local_max = -CUDART_INF_F;
    float local_target = -CUDART_INF_F;
    for (int col = (block_rank * ThreadPerRow + lane_in_row) * kVecSize;
         col < cols;
         col += blocks_per_cluster * ThreadPerRow * kVecSize) {
        const int remaining = cols - col;
        float4 x = row_valid ? load_float4_or_zero(row_logits + col, remaining)
                             : float4{0.0f, 0.0f, 0.0f, 0.0f};
        local_max = fmaxf(local_max, reduce_max4(x));
        if (row_valid) {
            if (remaining > 0 && target == col + 0) {
                local_target = x.x;
            }
            if (remaining > 1 && target == col + 1) {
                local_target = x.y;
            }
            if (remaining > 2 && target == col + 2) {
                local_target = x.z;
            }
            if (remaining > 3 && target == col + 3) {
                local_target = x.w;
            }
        }
    }
    local_max = subwarp_reduce_max<ThreadPerRow>(local_max);
    local_target = subwarp_reduce_max<ThreadPerRow>(local_target);
    const float row_max = cluster_allreduce_max<ThreadPerRow>(
        cluster, local_max, row_slots, lane_in_row);
    const float target_logit = cluster_allreduce_max<ThreadPerRow>(
        cluster, local_target, row_slots, lane_in_row);

    float local_sum = 0.0f;
    for (int col = (block_rank * ThreadPerRow + lane_in_row) * kVecSize;
         col < cols;
         col += blocks_per_cluster * ThreadPerRow * kVecSize) {
        const int remaining = cols - col;
        float4 x = row_valid ? load_float4_or_zero(row_logits + col, remaining)
                             : float4{0.0f, 0.0f, 0.0f, 0.0f};
        if (remaining > 0) {
            local_sum += __expf(x.x - row_max);
        }
        if (remaining > 1) {
            local_sum += __expf(x.y - row_max);
        }
        if (remaining > 2) {
            local_sum += __expf(x.z - row_max);
        }
        if (remaining > 3) {
            local_sum += __expf(x.w - row_max);
        }
    }
    local_sum = subwarp_reduce_sum<ThreadPerRow>(local_sum);
    const float row_sum = cluster_allreduce_sum<ThreadPerRow>(
        cluster, local_sum, row_slots, lane_in_row);

    if (row_valid && lane_in_row == 0 && block_rank == 0) {
        losses[row] = logf(row_sum) - (target_logit - row_max);
    }
}

__global__ void init_input_kernel(float* data, std::size_t elements) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                            threadIdx.x;
    if (idx >= elements) {
        return;
    }
    const int value = static_cast<int>(idx % 251u) - 125;
    data[idx] = static_cast<float>(value) * 0.03125f;
}

__global__ void init_weight_kernel(float* weight, int cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cols) {
        return;
    }
    weight[idx] = 0.75f + static_cast<float>(idx % 31) * 0.015625f;
}

__global__ void init_targets_kernel(int* targets, int rows, int cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows) {
        return;
    }
    targets[idx] = static_cast<int>((static_cast<long long>(idx) * 8191 + 17) % cols);
}

float host_input_value(int row, int col, int cols) {
    const std::size_t idx = static_cast<std::size_t>(row) * cols + col;
    return static_cast<float>(static_cast<int>(idx % 251u) - 125) * 0.03125f;
}

float host_weight_value(int col) {
    return 0.75f + static_cast<float>(col % 31) * 0.015625f;
}

int host_target_value(int row, int cols) {
    return static_cast<int>((static_cast<long long>(row) * 8191 + 17) % cols);
}

std::vector<int> select_sample_rows(int rows) {
    std::vector<int> samples;
    if (rows <= 0) {
        return samples;
    }
    const std::array<int, 4> candidates = {0, rows / 3, (2 * rows) / 3, rows - 1};
    for (int row : candidates) {
        if (row >= 0 && row < rows &&
            std::find(samples.begin(), samples.end(), row) == samples.end()) {
            samples.push_back(row);
        }
    }
    return samples;
}

void copy_sample_rows(float* d_output, int cols, const std::vector<int>& sample_rows,
                      std::vector<float>& host_samples) {
    host_samples.resize(static_cast<std::size_t>(sample_rows.size()) * cols);
    for (std::size_t i = 0; i < sample_rows.size(); ++i) {
        CUDA_CHECK(cudaMemcpy(host_samples.data() + i * cols,
                              d_output + static_cast<std::size_t>(sample_rows[i]) * cols,
                              static_cast<std::size_t>(cols) * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
}

void copy_sample_values(float* d_output, const std::vector<int>& sample_rows,
                        std::vector<float>& host_values) {
    host_values.resize(sample_rows.size());
    for (std::size_t i = 0; i < sample_rows.size(); ++i) {
        CUDA_CHECK(cudaMemcpy(&host_values[i], d_output + sample_rows[i], sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
}

void check_close(double got, double want, double atol, double rtol,
                 const std::string& workload, int row, int col) {
    const double tol = atol + rtol * std::abs(want);
    if (std::abs(got - want) > tol) {
        std::cerr << "verification failed for " << workload << " at row=" << row;
        if (col >= 0) {
            std::cerr << ", col=" << col;
        }
        std::cerr << ": got=" << got << ", want=" << want
                  << ", tol=" << tol << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void verify_softmax_samples(const std::vector<int>& sample_rows,
                            const std::vector<float>& host_samples, int cols) {
    for (std::size_t sample_idx = 0; sample_idx < sample_rows.size(); ++sample_idx) {
        const int row = sample_rows[sample_idx];
        double row_max = -std::numeric_limits<double>::infinity();
        for (int col = 0; col < cols; ++col) {
            row_max = std::max(row_max, static_cast<double>(host_input_value(row, col, cols)));
        }
        double row_sum = 0.0;
        for (int col = 0; col < cols; ++col) {
            row_sum += std::exp(static_cast<double>(host_input_value(row, col, cols)) -
                                row_max);
        }
        for (int col = 0; col < cols; ++col) {
            const double want =
                std::exp(static_cast<double>(host_input_value(row, col, cols)) - row_max) /
                row_sum;
            const double got = host_samples[sample_idx * cols + col];
            check_close(got, want, 2e-5, 5e-4, "softmax", row, col);
        }
    }
}

void verify_rmsnorm_samples(const std::vector<int>& sample_rows,
                            const std::vector<float>& host_samples, int cols) {
    for (std::size_t sample_idx = 0; sample_idx < sample_rows.size(); ++sample_idx) {
        const int row = sample_rows[sample_idx];
        double sumsq = 0.0;
        for (int col = 0; col < cols; ++col) {
            const double x = host_input_value(row, col, cols);
            sumsq += x * x;
        }
        const double inv_rms = 1.0 / std::sqrt(sumsq / cols + kRmsEps);
        for (int col = 0; col < cols; ++col) {
            const double want =
                host_input_value(row, col, cols) * inv_rms * host_weight_value(col);
            const double got = host_samples[sample_idx * cols + col];
            check_close(got, want, 2e-4, 2e-4, "rmsnorm", row, col);
        }
    }
}

void verify_cross_entropy_samples(const std::vector<int>& sample_rows,
                                  const std::vector<float>& host_values, int cols) {
    for (std::size_t sample_idx = 0; sample_idx < sample_rows.size(); ++sample_idx) {
        const int row = sample_rows[sample_idx];
        const int target = host_target_value(row, cols);
        double row_max = -std::numeric_limits<double>::infinity();
        for (int col = 0; col < cols; ++col) {
            row_max = std::max(row_max, static_cast<double>(host_input_value(row, col, cols)));
        }
        double row_sum = 0.0;
        for (int col = 0; col < cols; ++col) {
            row_sum += std::exp(static_cast<double>(host_input_value(row, col, cols)) -
                                row_max);
        }
        const double want = std::log(row_sum) -
                            (static_cast<double>(host_input_value(row, target, cols)) -
                             row_max);
        const double got = host_values[sample_idx];
        check_close(got, want, 2e-4, 2e-4, "cross_entropy", row, -1);
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
    info.cluster_launch = (cluster_launch != 0);
    if (info.cluster_launch) {
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(options.requested_cluster_size);
        config.blockDim = dim3(kBlockThreads);
        config.dynamicSmemBytes = 0;

        int max_cluster_size = 0;
        CUDA_CHECK(cudaOccupancyMaxPotentialClusterSize(
            &max_cluster_size, cluster_softmax_kernel<32, kMaxClusterSize>, &config));
        info.max_cluster_size = max_cluster_size;
        info.chosen_cluster_size =
            std::min({options.requested_cluster_size, max_cluster_size, kMaxClusterSize});
    }
    return info;
}

void print_device_header(const Options& options, const DeviceInfo& device) {
    if (options.csv) {
        std::cout << "META,device_name," << device.name << "\n";
        std::cout << "META,sm," << device.major << "." << device.minor << "\n";
        std::cout << "META,batch_size," << options.batch_size << "\n";
        std::cout << "META,warmup," << options.warmup << "\n";
        std::cout << "META,iters," << options.iters << "\n";
        std::cout << "META,cluster_launch," << (device.cluster_launch ? 1 : 0) << "\n";
        std::cout << "META,requested_cluster_size," << options.requested_cluster_size
                  << "\n";
        std::cout << "META,max_cluster_size," << device.max_cluster_size << "\n";
        std::cout << "META,chosen_cluster_size," << device.chosen_cluster_size << "\n";
        return;
    }

    std::cout << "Device: " << device.name << " (sm_" << device.major << device.minor
              << ")\n";
    std::cout << "Batch size: " << options.batch_size << "\n";
    std::cout << "Warmup iters: " << options.warmup
              << ", timing iters: " << options.iters << "\n";
    std::cout << "Cluster launch supported: "
              << (device.cluster_launch ? "yes" : "no") << "\n";
    if (device.cluster_launch) {
        std::cout << "Requested cluster size: " << options.requested_cluster_size
                  << ", max potential cluster size: " << device.max_cluster_size
                  << ", chosen cluster size: " << device.chosen_cluster_size << "\n";
    }
    std::cout << "Tuning thread_per_row over:";
    for (int value : options.thread_per_row_values) {
        std::cout << " " << value;
    }
    std::cout << "\n\n";
}

void init_active_inputs(float* d_input, float* d_weight, int* d_targets, int batch_size,
                        int cols) {
    const std::size_t elements = static_cast<std::size_t>(batch_size) * cols;
    const int threads = 256;
    const int input_blocks = static_cast<int>((elements + threads - 1) / threads);
    const int row_blocks = (batch_size + threads - 1) / threads;
    const int weight_blocks = (cols + threads - 1) / threads;
    init_input_kernel<<<input_blocks, threads>>>(d_input, elements);
    init_weight_kernel<<<weight_blocks, threads>>>(d_weight, cols);
    init_targets_kernel<<<row_blocks, threads>>>(d_targets, batch_size, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <int ThreadPerRow>
void launch_block_kernel(Workload workload, const float* d_input, const float* d_weight,
                         const int* d_targets, float* d_output, int batch_size,
                         int cols) {
    constexpr int RowsPerBlock = kBlockThreads / ThreadPerRow;
    const int grid = (batch_size + RowsPerBlock - 1) / RowsPerBlock;
    switch (workload) {
        case Workload::Softmax:
            block_softmax_kernel<ThreadPerRow>
                <<<grid, kBlockThreads>>>(d_input, d_output, batch_size, cols);
            break;
        case Workload::RmsNorm:
            block_rmsnorm_kernel<ThreadPerRow><<<grid, kBlockThreads>>>(
                d_input, d_weight, d_output, batch_size, cols, kRmsEps);
            break;
        case Workload::CrossEntropy:
            block_cross_entropy_kernel<ThreadPerRow><<<grid, kBlockThreads>>>(
                d_input, d_targets, d_output, batch_size, cols);
            break;
    }
}

template <int ThreadPerRow>
void launch_cluster_kernel(Workload workload, const float* d_input, const float* d_weight,
                           const int* d_targets, float* d_output, int batch_size,
                           int cols, int cluster_size) {
    constexpr int RowsPerBlock = kBlockThreads / ThreadPerRow;
    const int clusters = (batch_size + RowsPerBlock - 1) / RowsPerBlock;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(clusters * cluster_size);
    config.blockDim = dim3(kBlockThreads);
    config.dynamicSmemBytes = 0;

    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = cluster_size;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;
    config.attrs = &attr;
    config.numAttrs = 1;

    switch (workload) {
        case Workload::Softmax:
            CUDA_CHECK(cudaLaunchKernelEx(
                &config, cluster_softmax_kernel<ThreadPerRow, kMaxClusterSize>, d_input,
                d_output, batch_size, cols));
            break;
        case Workload::RmsNorm:
            CUDA_CHECK(cudaLaunchKernelEx(
                &config, cluster_rmsnorm_kernel<ThreadPerRow, kMaxClusterSize>, d_input,
                d_weight, d_output, batch_size, cols, kRmsEps));
            break;
        case Workload::CrossEntropy:
            CUDA_CHECK(cudaLaunchKernelEx(
                &config, cluster_cross_entropy_kernel<ThreadPerRow, kMaxClusterSize>,
                d_input, d_targets, d_output, batch_size, cols));
            break;
    }
}

void dispatch_block_kernel(int thread_per_row, Workload workload, const float* d_input,
                           const float* d_weight, const int* d_targets,
                           float* d_output, int batch_size, int cols) {
    switch (thread_per_row) {
        case 8:
            launch_block_kernel<8>(workload, d_input, d_weight, d_targets, d_output,
                                   batch_size, cols);
            break;
        case 16:
            launch_block_kernel<16>(workload, d_input, d_weight, d_targets, d_output,
                                    batch_size, cols);
            break;
        case 32:
            launch_block_kernel<32>(workload, d_input, d_weight, d_targets, d_output,
                                    batch_size, cols);
            break;
        default:
            throw std::runtime_error("unsupported thread_per_row");
    }
}

void dispatch_cluster_kernel(int thread_per_row, Workload workload, const float* d_input,
                             const float* d_weight, const int* d_targets,
                             float* d_output, int batch_size, int cols,
                             int cluster_size) {
    switch (thread_per_row) {
        case 8:
            launch_cluster_kernel<8>(workload, d_input, d_weight, d_targets, d_output,
                                     batch_size, cols, cluster_size);
            break;
        case 16:
            launch_cluster_kernel<16>(workload, d_input, d_weight, d_targets, d_output,
                                      batch_size, cols, cluster_size);
            break;
        case 32:
            launch_cluster_kernel<32>(workload, d_input, d_weight, d_targets, d_output,
                                      batch_size, cols, cluster_size);
            break;
        default:
            throw std::runtime_error("unsupported thread_per_row");
    }
}

VariantResult benchmark_block_variant(Workload workload, int thread_per_row,
                                      const Options& options,
                                      const DeviceInfo& device,
                                      const float* d_input, const float* d_weight,
                                      const int* d_targets, float* d_output,
                                      int batch_size, int cols) {
    VariantResult result;
    result.thread_per_row = thread_per_row;
    if (thread_per_row == 256) {
        if (workload != Workload::Softmax && workload != Workload::RmsNorm &&
            workload != Workload::CrossEntropy) {
            return result;
        }
        const std::size_t smem_bytes = static_cast<std::size_t>(cols) * sizeof(float);
        if (smem_bytes > static_cast<std::size_t>(device.max_dynamic_smem_per_block)) {
            return result;
        }
        std::function<void()> launch;
        if (workload == Workload::Softmax) {
            CUDA_CHECK(cudaFuncSetAttribute(
                block_softmax_staged_kernel<kBlockThreads>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_bytes)));
            launch = [&]() {
                block_softmax_staged_kernel<kBlockThreads>
                    <<<batch_size, kBlockThreads, smem_bytes>>>(d_input, d_output,
                                                                batch_size, cols);
            };
        } else if (workload == Workload::RmsNorm) {
            CUDA_CHECK(cudaFuncSetAttribute(
                block_rmsnorm_staged_kernel<kBlockThreads>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_bytes)));
            launch = [&]() {
                block_rmsnorm_staged_kernel<kBlockThreads>
                    <<<batch_size, kBlockThreads, smem_bytes>>>(d_input, d_weight,
                                                                d_output, batch_size,
                                                                cols, kRmsEps);
            };
        } else {
            CUDA_CHECK(cudaFuncSetAttribute(
                block_cross_entropy_staged_kernel<kBlockThreads>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_bytes)));
            launch = [&]() {
                block_cross_entropy_staged_kernel<kBlockThreads>
                    <<<batch_size, kBlockThreads, smem_bytes>>>(d_input, d_targets,
                                                                d_output, batch_size,
                                                                cols);
            };
        }
        result.valid = true;
        result.avg_ms = time_kernel(launch, options.warmup, options.iters);
        result.gbps = throughput_gbps(workload, batch_size, cols, result.avg_ms);
        return result;
    } else {
        auto launch = [&]() {
            dispatch_block_kernel(thread_per_row, workload, d_input, d_weight,
                                  d_targets, d_output, batch_size, cols);
        };
        result.valid = true;
        result.avg_ms = time_kernel(launch, options.warmup, options.iters);
        result.gbps = throughput_gbps(workload, batch_size, cols, result.avg_ms);
        return result;
    }
}

VariantResult benchmark_cluster_variant(Workload workload, int thread_per_row,
                                        const Options& options,
                                        const DeviceInfo& device,
                                        const float* d_input, const float* d_weight,
                                        const int* d_targets, float* d_output,
                                        int batch_size, int cols) {
    VariantResult result;
    if (device.chosen_cluster_size < 2 || thread_per_row < device.chosen_cluster_size) {
        return result;
    }
    result.thread_per_row = thread_per_row;
    if (thread_per_row == 256) {
        if (workload != Workload::Softmax && workload != Workload::RmsNorm &&
            workload != Workload::CrossEntropy) {
            return result;
        }
        const int slice_elems =
            (cols + device.chosen_cluster_size - 1) / device.chosen_cluster_size;
        const std::size_t smem_bytes =
            static_cast<std::size_t>(slice_elems) * sizeof(float);
        if (smem_bytes > static_cast<std::size_t>(device.max_dynamic_smem_per_block)) {
            return result;
        }
        std::function<void()> launch;
        if (workload == Workload::Softmax) {
            CUDA_CHECK(cudaFuncSetAttribute(
                cluster_softmax_staged_kernel<kBlockThreads, kMaxClusterSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_bytes)));
            launch = [&]() {
                cudaLaunchConfig_t config{};
                config.gridDim = dim3(batch_size * device.chosen_cluster_size);
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
                    &config,
                    cluster_softmax_staged_kernel<kBlockThreads, kMaxClusterSize>,
                    d_input, d_output, batch_size, cols, slice_elems));
            };
        } else if (workload == Workload::RmsNorm) {
            CUDA_CHECK(cudaFuncSetAttribute(
                cluster_rmsnorm_staged_kernel<kBlockThreads, kMaxClusterSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_bytes)));
            launch = [&]() {
                cudaLaunchConfig_t config{};
                config.gridDim = dim3(batch_size * device.chosen_cluster_size);
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
                    &config,
                    cluster_rmsnorm_staged_kernel<kBlockThreads, kMaxClusterSize>,
                    d_input, d_weight, d_output, batch_size, cols, kRmsEps,
                    slice_elems));
            };
        } else {
            CUDA_CHECK(cudaFuncSetAttribute(
                cluster_cross_entropy_staged_kernel<kBlockThreads, kMaxClusterSize>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_bytes)));
            launch = [&]() {
                cudaLaunchConfig_t config{};
                config.gridDim = dim3(batch_size * device.chosen_cluster_size);
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
                    &config,
                    cluster_cross_entropy_staged_kernel<kBlockThreads, kMaxClusterSize>,
                    d_input, d_targets, d_output, batch_size, cols, slice_elems));
            };
        }
        result.valid = true;
        result.avg_ms = time_kernel(launch, options.warmup, options.iters);
        result.gbps = throughput_gbps(workload, batch_size, cols, result.avg_ms);
        return result;
    } else {
        auto launch = [&]() {
            dispatch_cluster_kernel(thread_per_row, workload, d_input, d_weight,
                                    d_targets, d_output, batch_size, cols,
                                    device.chosen_cluster_size);
        };
        result.valid = true;
        result.avg_ms = time_kernel(launch, options.warmup, options.iters);
        result.gbps = throughput_gbps(workload, batch_size, cols, result.avg_ms);
        return result;
    }
}

VariantResult pick_better(const VariantResult& lhs, const VariantResult& rhs) {
    if (!lhs.valid) {
        return rhs;
    }
    if (!rhs.valid) {
        return lhs;
    }
    return (rhs.gbps > lhs.gbps) ? rhs : lhs;
}

void verify_best_variant(Workload workload, bool cluster_variant, int thread_per_row,
                         const DeviceInfo& device, const float* d_input,
                         const float* d_weight, const int* d_targets, float* d_output,
                         int batch_size, int cols) {
    if (thread_per_row == 256 &&
        (workload == Workload::Softmax || workload == Workload::RmsNorm ||
         workload == Workload::CrossEntropy)) {
        if (cluster_variant) {
            const int slice_elems =
                (cols + device.chosen_cluster_size - 1) / device.chosen_cluster_size;
            const std::size_t smem_bytes =
                static_cast<std::size_t>(slice_elems) * sizeof(float);
            cudaLaunchConfig_t config{};
            config.gridDim = dim3(batch_size * device.chosen_cluster_size);
            config.blockDim = dim3(kBlockThreads);
            config.dynamicSmemBytes = static_cast<std::uint32_t>(smem_bytes);

            cudaLaunchAttribute attr{};
            attr.id = cudaLaunchAttributeClusterDimension;
            attr.val.clusterDim.x = device.chosen_cluster_size;
            attr.val.clusterDim.y = 1;
            attr.val.clusterDim.z = 1;
            config.attrs = &attr;
            config.numAttrs = 1;
            if (workload == Workload::Softmax) {
                CUDA_CHECK(cudaFuncSetAttribute(
                    cluster_softmax_staged_kernel<kBlockThreads, kMaxClusterSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(smem_bytes)));
                CUDA_CHECK(cudaLaunchKernelEx(
                    &config,
                    cluster_softmax_staged_kernel<kBlockThreads, kMaxClusterSize>,
                    d_input, d_output, batch_size, cols, slice_elems));
            } else if (workload == Workload::RmsNorm) {
                CUDA_CHECK(cudaFuncSetAttribute(
                    cluster_rmsnorm_staged_kernel<kBlockThreads, kMaxClusterSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(smem_bytes)));
                CUDA_CHECK(cudaLaunchKernelEx(
                    &config,
                    cluster_rmsnorm_staged_kernel<kBlockThreads, kMaxClusterSize>,
                    d_input, d_weight, d_output, batch_size, cols, kRmsEps,
                    slice_elems));
            } else {
                CUDA_CHECK(cudaFuncSetAttribute(
                    cluster_cross_entropy_staged_kernel<kBlockThreads, kMaxClusterSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(smem_bytes)));
                CUDA_CHECK(cudaLaunchKernelEx(
                    &config,
                    cluster_cross_entropy_staged_kernel<kBlockThreads, kMaxClusterSize>,
                    d_input, d_targets, d_output, batch_size, cols, slice_elems));
            }
        } else {
            const std::size_t smem_bytes = static_cast<std::size_t>(cols) * sizeof(float);
            if (workload == Workload::Softmax) {
                CUDA_CHECK(cudaFuncSetAttribute(
                    block_softmax_staged_kernel<kBlockThreads>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(smem_bytes)));
                block_softmax_staged_kernel<kBlockThreads>
                    <<<batch_size, kBlockThreads, smem_bytes>>>(d_input, d_output,
                                                                batch_size, cols);
            } else if (workload == Workload::RmsNorm) {
                CUDA_CHECK(cudaFuncSetAttribute(
                    block_rmsnorm_staged_kernel<kBlockThreads>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(smem_bytes)));
                block_rmsnorm_staged_kernel<kBlockThreads>
                    <<<batch_size, kBlockThreads, smem_bytes>>>(d_input, d_weight,
                                                                d_output, batch_size,
                                                                cols, kRmsEps);
            } else {
                CUDA_CHECK(cudaFuncSetAttribute(
                    block_cross_entropy_staged_kernel<kBlockThreads>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(smem_bytes)));
                block_cross_entropy_staged_kernel<kBlockThreads>
                    <<<batch_size, kBlockThreads, smem_bytes>>>(d_input, d_targets,
                                                                d_output, batch_size,
                                                                cols);
            }
        }
    } else if (cluster_variant) {
        dispatch_cluster_kernel(thread_per_row, workload, d_input, d_weight,
                                d_targets, d_output, batch_size, cols,
                                device.chosen_cluster_size);
    } else {
        dispatch_block_kernel(thread_per_row, workload, d_input, d_weight, d_targets,
                              d_output, batch_size, cols);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<int> sample_rows = select_sample_rows(batch_size);
    std::vector<float> host_samples;
    std::vector<float> host_values;
    if (workload == Workload::CrossEntropy) {
        copy_sample_values(d_output, sample_rows, host_values);
        verify_cross_entropy_samples(sample_rows, host_values, cols);
    } else if (workload == Workload::Softmax) {
        copy_sample_rows(d_output, cols, sample_rows, host_samples);
        verify_softmax_samples(sample_rows, host_samples, cols);
    } else {
        copy_sample_rows(d_output, cols, sample_rows, host_samples);
        verify_rmsnorm_samples(sample_rows, host_samples, cols);
    }
}

WorkloadReport benchmark_workload(Workload workload, const Options& options,
                                  const DeviceInfo& device, float* d_input,
                                  float* d_weight, int* d_targets, float* d_output) {
    WorkloadReport report{workload, {}};
    report.results.reserve(options.n_values.size());

    for (int cols : options.n_values) {
        init_active_inputs(d_input, d_weight, d_targets, options.batch_size, cols);

        VariantResult best_block;
        VariantResult best_cluster;
        for (int thread_per_row : options.thread_per_row_values) {
            best_block = pick_better(best_block, benchmark_block_variant(
                                                     workload, thread_per_row, options,
                                                     device, d_input, d_weight,
                                                     d_targets, d_output,
                                                     options.batch_size, cols));
            best_cluster = pick_better(best_cluster, benchmark_cluster_variant(
                                                       workload, thread_per_row, options,
                                                       device, d_input, d_weight,
                                                       d_targets, d_output,
                                                       options.batch_size, cols));
        }

        if (options.verify && best_block.valid) {
            verify_best_variant(workload, false, best_block.thread_per_row, device,
                                d_input, d_weight, d_targets, d_output,
                                options.batch_size, cols);
        }
        if (options.verify && best_cluster.valid) {
            verify_best_variant(workload, true, best_cluster.thread_per_row, device,
                                d_input, d_weight, d_targets, d_output,
                                options.batch_size, cols);
        }

        SizeResult result;
        result.cols = cols;
        result.batch_size = options.batch_size;
        result.block = best_block;
        result.cluster = best_cluster;
        if (best_block.valid && best_cluster.valid) {
            result.cluster_vs_block = best_cluster.gbps / best_block.gbps;
        }
        report.results.push_back(result);
    }

    return report;
}

void print_report(const WorkloadReport& report, int chosen_cluster_size, bool csv) {
    if (csv) {
        for (const SizeResult& result : report.results) {
            std::cout << "RESULT," << workload_name(report.workload) << ","
                      << result.cols << "," << result.batch_size << ","
                      << fmt_double(result.block.avg_ms, 6) << ","
                      << fmt_double(result.block.gbps, 6) << ","
                      << result.block.thread_per_row << ","
                      << fmt_double(result.cluster.avg_ms, 6) << ","
                      << fmt_double(result.cluster.gbps, 6) << ","
                      << result.cluster.thread_per_row << ","
                      << chosen_cluster_size << ","
                      << fmt_double(result.cluster_vs_block, 6) << "\n";
        }
        return;
    }

    std::cout << "Workload: " << workload_name(report.workload) << "\n";
    std::cout << std::left << std::setw(10) << "N" << std::setw(12) << "block_GB/s"
              << std::setw(10) << "block_tpr" << std::setw(14) << "cluster_GB/s"
              << std::setw(12) << "cluster_tpr" << std::setw(14)
              << "cluster/block" << "\n";
    for (const SizeResult& result : report.results) {
        std::cout << std::left << std::setw(10) << result.cols
                  << std::setw(12) << fmt_double(result.block.gbps, 2)
                  << std::setw(10) << result.block.thread_per_row
                  << std::setw(14) << fmt_double(result.cluster.gbps, 2)
                  << std::setw(12) << result.cluster.thread_per_row
                  << std::setw(14) << fmt_double(result.cluster_vs_block, 2) << "\n";
    }

    int wins = 0;
    double best_speedup = 0.0;
    int best_cols = 0;
    double large_avg = 0.0;
    int large_count = 0;
    for (const SizeResult& result : report.results) {
        if (!result.cluster.valid || !result.block.valid) {
            continue;
        }
        if (result.cluster_vs_block > 1.0) {
            ++wins;
        }
        if (result.cluster_vs_block > best_speedup) {
            best_speedup = result.cluster_vs_block;
            best_cols = result.cols;
        }
        if (result.cols >= 65536) {
            large_avg += result.cluster_vs_block;
            ++large_count;
        }
    }
    if (large_count > 0) {
        large_avg /= large_count;
    }
    std::cout << "Summary: cluster beat block on " << wins << "/"
              << report.results.size() << " tested sizes. Best cluster/block was "
              << fmt_double(best_speedup, 2) << "x at N=" << best_cols
              << ". Average cluster/block for N >= 65536 was "
              << fmt_double(large_avg, 2) << "x.\n\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        CUDA_CHECK(cudaSetDevice(0));
        const DeviceInfo device = query_device_info(options);

        const int max_n =
            *std::max_element(options.n_values.begin(), options.n_values.end());
        const std::size_t max_elements =
            static_cast<std::size_t>(options.batch_size) * max_n;

        float* d_input = nullptr;
        float* d_weight = nullptr;
        int* d_targets = nullptr;
        float* d_output = nullptr;
        CUDA_CHECK(cudaMalloc(&d_input, max_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weight, static_cast<std::size_t>(max_n) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_targets,
                              static_cast<std::size_t>(options.batch_size) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_output, max_elements * sizeof(float)));

        print_device_header(options, device);

        std::vector<WorkloadReport> reports;
        reports.reserve(options.workloads.size());
        for (Workload workload : options.workloads) {
            reports.push_back(benchmark_workload(workload, options, device, d_input,
                                                 d_weight, d_targets, d_output));
        }

        for (const WorkloadReport& report : reports) {
            print_report(report, device.chosen_cluster_size, options.csv);
        }

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_weight));
        CUDA_CHECK(cudaFree(d_targets));
        CUDA_CHECK(cudaFree(d_output));
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}
