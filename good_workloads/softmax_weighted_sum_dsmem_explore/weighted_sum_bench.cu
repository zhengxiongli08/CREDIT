#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
    float weighted;
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
            opts.thread_per_row_values =
                parse_int_list(require_string("--thread-per-row-values"));
        } else if (arg == "--csv") {
            opts.csv = true;
        } else if (arg == "--no-verify") {
            opts.verify = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: weighted_sum_bench [--csv] [--warmup N] [--iters N] "
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
    const double elements = static_cast<double>(batch_size) * cols;
    return elements * sizeof(float) * 3.0 +
           static_cast<double>(batch_size) * sizeof(float);
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

__device__ __forceinline__ float reduce_max4(float4 value, int remaining) {
    float out = -CUDART_INF_F;
    if (remaining > 0) {
        out = fmaxf(out, value.x);
    }
    if (remaining > 1) {
        out = fmaxf(out, value.y);
    }
    if (remaining > 2) {
        out = fmaxf(out, value.z);
    }
    if (remaining > 3) {
        out = fmaxf(out, value.w);
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

template <int Width>
__device__ __forceinline__ float subwarp_reduce_max(float value) {
    static_assert(Width == 16 || Width == 32, "unsupported reduction width");
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset, Width));
    }
    return value;
}

template <int BlockThreads>
__device__ __forceinline__ float block_reduce_max(float value) {
    constexpr int Warps = BlockThreads / 32;
    __shared__ float warp_values[Warps];
    __shared__ float broadcast;
    value = subwarp_reduce_max<32>(value);
    if ((threadIdx.x & 31) == 0) {
        warp_values[threadIdx.x >> 5] = value;
    }
    __syncthreads();
    float reduced = -CUDART_INF_F;
    if (threadIdx.x < Warps) {
        reduced = warp_values[threadIdx.x];
    }
    if (threadIdx.x < 32) {
        reduced = subwarp_reduce_max<32>(reduced);
    }
    if (threadIdx.x == 0) {
        broadcast = reduced;
    }
    __syncthreads();
    return broadcast;
}

template <int BlockThreads>
__device__ __forceinline__ Pair block_reduce_pair(Pair value) {
    constexpr int Warps = BlockThreads / 32;
    __shared__ float warp_sum[Warps];
    __shared__ float warp_weighted[Warps];
    __shared__ Pair broadcast;

    value.sum = subwarp_reduce_sum<32>(value.sum);
    value.weighted = subwarp_reduce_sum<32>(value.weighted);
    if ((threadIdx.x & 31) == 0) {
        const int warp = threadIdx.x >> 5;
        warp_sum[warp] = value.sum;
        warp_weighted[warp] = value.weighted;
    }
    __syncthreads();

    Pair reduced{0.0f, 0.0f};
    if (threadIdx.x < Warps) {
        reduced.sum = warp_sum[threadIdx.x];
        reduced.weighted = warp_weighted[threadIdx.x];
    }
    if (threadIdx.x < 32) {
        reduced.sum = subwarp_reduce_sum<32>(reduced.sum);
        reduced.weighted = subwarp_reduce_sum<32>(reduced.weighted);
    }
    if (threadIdx.x == 0) {
        broadcast = reduced;
    }
    __syncthreads();
    return broadcast;
}

template <int BlockThreads, int MaxClusterSize>
__device__ __forceinline__ float cluster_reduce_max(cg::cluster_group cluster,
                                                     float block_value) {
    __shared__ float slots[MaxClusterSize];
    __shared__ float broadcast;
    const int block_rank = cluster.block_rank();
    const int blocks_per_cluster = cluster.dim_blocks().x;
    if (threadIdx.x < blocks_per_cluster) {
        float* remote = cluster.map_shared_rank(&slots[block_rank], threadIdx.x);
        *remote = block_value;
    }
    cluster.sync();
    float reduced = -CUDART_INF_F;
    if (threadIdx.x < blocks_per_cluster) {
        reduced = slots[threadIdx.x];
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
__device__ __forceinline__ Pair cluster_reduce_pair(cg::cluster_group cluster,
                                                     Pair block_value) {
    __shared__ float cluster_sum[MaxClusterSize];
    __shared__ float cluster_weighted[MaxClusterSize];
    __shared__ Pair broadcast;
    const int block_rank = cluster.block_rank();
    const int blocks_per_cluster = cluster.dim_blocks().x;

    if (threadIdx.x < blocks_per_cluster) {
        float* remote_sum =
            cluster.map_shared_rank(&cluster_sum[block_rank], threadIdx.x);
        float* remote_weighted =
            cluster.map_shared_rank(&cluster_weighted[block_rank], threadIdx.x);
        *remote_sum = block_value.sum;
        *remote_weighted = block_value.weighted;
    }
    cluster.sync();

    Pair reduced{0.0f, 0.0f};
    if (threadIdx.x < blocks_per_cluster) {
        reduced.sum = cluster_sum[threadIdx.x];
        reduced.weighted = cluster_weighted[threadIdx.x];
    }
    if (threadIdx.x < 32) {
        reduced.sum = subwarp_reduce_sum<32>(reduced.sum);
        reduced.weighted = subwarp_reduce_sum<32>(reduced.weighted);
    }
    if (threadIdx.x == 0) {
        broadcast = reduced;
    }
    __syncthreads();
    cluster.sync();
    return broadcast;
}

__global__ void init_input_kernel(float* scores, float* values,
                                  std::size_t elements) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }
    const unsigned a = static_cast<unsigned>((idx * 1103515245ull + 12345ull) >> 16);
    const unsigned b = static_cast<unsigned>((idx * 1664525ull + 1013904223ull) >> 16);
    scores[idx] = static_cast<float>(a & 4095u) * (1.0f / 512.0f) - 4.0f;
    values[idx] = static_cast<float>(b & 2047u) * (1.0f / 1024.0f) - 1.0f;
}

template <int ThreadPerRow>
__global__ void block_weighted_kernel(const float* __restrict__ scores,
                                      const float* __restrict__ values,
                                      float* __restrict__ output,
                                      int rows,
                                      int cols) {
    constexpr int RowsPerBlock = kBlockThreads / ThreadPerRow;
    const int row_local = threadIdx.x / ThreadPerRow;
    const int lane = threadIdx.x % ThreadPerRow;
    const int row = blockIdx.x * RowsPerBlock + row_local;
    const bool row_valid = row < rows;
    const float* row_scores =
        row_valid ? scores + static_cast<std::size_t>(row) * cols : scores;
    const float* row_values =
        row_valid ? values + static_cast<std::size_t>(row) * cols : values;

    float local_max = -CUDART_INF_F;
    for (int col = lane * kVecSize; col < cols;
         col += ThreadPerRow * kVecSize) {
        const int remaining = cols - col;
        const float4 x = row_valid ? load_float4_or_zero(row_scores + col, remaining)
                                   : float4{0.0f, 0.0f, 0.0f, 0.0f};
        local_max = fmaxf(local_max, reduce_max4(x, remaining));
    }
    const float row_max = subwarp_reduce_max<ThreadPerRow>(local_max);

    Pair local{0.0f, 0.0f};
    for (int col = lane * kVecSize; col < cols;
         col += ThreadPerRow * kVecSize) {
        const int remaining = cols - col;
        const float4 x = row_valid ? load_float4_or_zero(row_scores + col, remaining)
                                   : float4{0.0f, 0.0f, 0.0f, 0.0f};
        const float4 v = row_valid ? load_float4_or_zero(row_values + col, remaining)
                                   : float4{0.0f, 0.0f, 0.0f, 0.0f};
        if (remaining > 0) {
            const float e = __expf(x.x - row_max);
            local.sum += e;
            local.weighted += e * v.x;
        }
        if (remaining > 1) {
            const float e = __expf(x.y - row_max);
            local.sum += e;
            local.weighted += e * v.y;
        }
        if (remaining > 2) {
            const float e = __expf(x.z - row_max);
            local.sum += e;
            local.weighted += e * v.z;
        }
        if (remaining > 3) {
            const float e = __expf(x.w - row_max);
            local.sum += e;
            local.weighted += e * v.w;
        }
    }
    local.sum = subwarp_reduce_sum<ThreadPerRow>(local.sum);
    local.weighted = subwarp_reduce_sum<ThreadPerRow>(local.weighted);
    if (row_valid && lane == 0) {
        output[row] = local.weighted / local.sum;
    }
}

template <int BlockThreads>
__global__ void block_weighted_256_kernel(const float* __restrict__ scores,
                                          const float* __restrict__ values,
                                          float* __restrict__ output,
                                          int rows,
                                          int cols) {
    const int row = blockIdx.x;
    const bool row_valid = row < rows;
    const float* row_scores =
        row_valid ? scores + static_cast<std::size_t>(row) * cols : scores;
    const float* row_values =
        row_valid ? values + static_cast<std::size_t>(row) * cols : values;

    float local_max = -CUDART_INF_F;
    for (int col = threadIdx.x * kVecSize; col < cols;
         col += BlockThreads * kVecSize) {
        const int remaining = cols - col;
        const float4 x = row_valid ? load_float4_or_zero(row_scores + col, remaining)
                                   : float4{0.0f, 0.0f, 0.0f, 0.0f};
        local_max = fmaxf(local_max, reduce_max4(x, remaining));
    }
    const float row_max = block_reduce_max<BlockThreads>(local_max);

    Pair local{0.0f, 0.0f};
    for (int col = threadIdx.x * kVecSize; col < cols;
         col += BlockThreads * kVecSize) {
        const int remaining = cols - col;
        const float4 x = row_valid ? load_float4_or_zero(row_scores + col, remaining)
                                   : float4{0.0f, 0.0f, 0.0f, 0.0f};
        const float4 v = row_valid ? load_float4_or_zero(row_values + col, remaining)
                                   : float4{0.0f, 0.0f, 0.0f, 0.0f};
        if (remaining > 0) {
            const float e = __expf(x.x - row_max);
            local.sum += e;
            local.weighted += e * v.x;
        }
        if (remaining > 1) {
            const float e = __expf(x.y - row_max);
            local.sum += e;
            local.weighted += e * v.y;
        }
        if (remaining > 2) {
            const float e = __expf(x.z - row_max);
            local.sum += e;
            local.weighted += e * v.z;
        }
        if (remaining > 3) {
            const float e = __expf(x.w - row_max);
            local.sum += e;
            local.weighted += e * v.w;
        }
    }
    const Pair total = block_reduce_pair<BlockThreads>(local);
    if (row_valid && threadIdx.x == 0) {
        output[row] = total.weighted / total.sum;
    }
}

template <int BlockThreads>
__global__ void block_weighted_staged_kernel(const float* __restrict__ scores,
                                             const float* __restrict__ values,
                                             float* __restrict__ output,
                                             int rows,
                                             int cols) {
    extern __shared__ float tile[];
    const int row = blockIdx.x;
    const bool row_valid = row < rows;
    const float* row_scores =
        row_valid ? scores + static_cast<std::size_t>(row) * cols : scores;
    const float* row_values =
        row_valid ? values + static_cast<std::size_t>(row) * cols : values;

    for (int col = threadIdx.x * kVecSize; col < cols;
         col += BlockThreads * kVecSize) {
        const int remaining = cols - col;
        const float4 x = row_valid ? load_float4_or_zero(row_scores + col, remaining)
                                   : float4{0.0f, 0.0f, 0.0f, 0.0f};
        store_float4_or_tail(tile + col, x, remaining);
    }
    __syncthreads();

    float local_max = -CUDART_INF_F;
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        local_max = fmaxf(local_max, row_valid ? tile[col] : -CUDART_INF_F);
    }
    const float row_max = block_reduce_max<BlockThreads>(local_max);

    Pair local{0.0f, 0.0f};
    for (int col = threadIdx.x * kVecSize; col < cols;
         col += BlockThreads * kVecSize) {
        const int remaining = cols - col;
        const float4 x = load_float4_or_zero(tile + col, remaining);
        const float4 v = row_valid ? load_float4_or_zero(row_values + col, remaining)
                                   : float4{0.0f, 0.0f, 0.0f, 0.0f};
        if (remaining > 0) {
            const float e = __expf(x.x - row_max);
            local.sum += e;
            local.weighted += e * v.x;
        }
        if (remaining > 1) {
            const float e = __expf(x.y - row_max);
            local.sum += e;
            local.weighted += e * v.y;
        }
        if (remaining > 2) {
            const float e = __expf(x.z - row_max);
            local.sum += e;
            local.weighted += e * v.z;
        }
        if (remaining > 3) {
            const float e = __expf(x.w - row_max);
            local.sum += e;
            local.weighted += e * v.w;
        }
    }
    const Pair total = block_reduce_pair<BlockThreads>(local);
    if (row_valid && threadIdx.x == 0) {
        output[row] = total.weighted / total.sum;
    }
}

template <int BlockThreads, int MaxClusterSize>
__global__ void cluster_weighted_staged_kernel(const float* __restrict__ scores,
                                               const float* __restrict__ values,
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
    const float* row_scores =
        row_valid ? scores + static_cast<std::size_t>(row) * cols : scores;
    const float* row_values =
        row_valid ? values + static_cast<std::size_t>(row) * cols : values;

    for (int col = threadIdx.x * kVecSize; col < local_cols;
         col += BlockThreads * kVecSize) {
        const int remaining = local_cols - col;
        const float4 x =
            load_float4_or_zero(row_scores + slice_start + col, remaining);
        store_float4_or_tail(tile + col, x, remaining);
    }
    __syncthreads();

    float local_max = -CUDART_INF_F;
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        local_max = fmaxf(local_max, tile[col]);
    }
    const float block_max = block_reduce_max<BlockThreads>(local_max);
    const float row_max =
        cluster_reduce_max<BlockThreads, MaxClusterSize>(cluster, block_max);

    Pair local{0.0f, 0.0f};
    for (int col = threadIdx.x * kVecSize; col < local_cols;
         col += BlockThreads * kVecSize) {
        const int remaining = local_cols - col;
        const float4 x = load_float4_or_zero(tile + col, remaining);
        const float4 v =
            load_float4_or_zero(row_values + slice_start + col, remaining);
        if (remaining > 0) {
            const float e = __expf(x.x - row_max);
            local.sum += e;
            local.weighted += e * v.x;
        }
        if (remaining > 1) {
            const float e = __expf(x.y - row_max);
            local.sum += e;
            local.weighted += e * v.y;
        }
        if (remaining > 2) {
            const float e = __expf(x.z - row_max);
            local.sum += e;
            local.weighted += e * v.z;
        }
        if (remaining > 3) {
            const float e = __expf(x.w - row_max);
            local.sum += e;
            local.weighted += e * v.w;
        }
    }
    const Pair block_total = block_reduce_pair<BlockThreads>(local);
    const Pair row_total =
        cluster_reduce_pair<BlockThreads, MaxClusterSize>(cluster, block_total);
    if (row_valid && block_rank == 0 && threadIdx.x == 0) {
        output[row] = row_total.weighted / row_total.sum;
    }
}

template <int ThreadPerRow>
void launch_block_tpr(const float* d_scores, const float* d_values,
                      float* d_output, int rows, int cols) {
    constexpr int RowsPerBlock = kBlockThreads / ThreadPerRow;
    const dim3 grid((rows + RowsPerBlock - 1) / RowsPerBlock);
    block_weighted_kernel<ThreadPerRow>
        <<<grid, kBlockThreads>>>(d_scores, d_values, d_output, rows, cols);
}

void dispatch_block_tpr(int thread_per_row, const float* d_scores,
                        const float* d_values, float* d_output, int rows,
                        int cols) {
    switch (thread_per_row) {
        case 16:
            launch_block_tpr<16>(d_scores, d_values, d_output, rows, cols);
            break;
        case 32:
            launch_block_tpr<32>(d_scores, d_values, d_output, rows, cols);
            break;
        case 256:
            block_weighted_256_kernel<kBlockThreads>
                <<<rows, kBlockThreads>>>(d_scores, d_values, d_output, rows,
                                           cols);
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
            cluster_weighted_staged_kernel<kBlockThreads, kMaxClusterSize>,
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
                                      const float* d_scores,
                                      const float* d_values,
                                      float* d_output,
                                      int rows, int cols) {
    VariantResult result;
    result.thread_per_row = thread_per_row;
    if (thread_per_row == 256) {
        const std::size_t smem_bytes = static_cast<std::size_t>(cols) * sizeof(float);
        if (smem_bytes <= static_cast<std::size_t>(device.max_dynamic_smem_per_block)) {
            CUDA_CHECK(cudaFuncSetAttribute(
                block_weighted_staged_kernel<kBlockThreads>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_bytes)));
            auto launch = [&]() {
                block_weighted_staged_kernel<kBlockThreads>
                    <<<rows, kBlockThreads, smem_bytes>>>(
                        d_scores, d_values, d_output, rows, cols);
            };
            result.valid = true;
            result.avg_ms = time_kernel(launch, options.warmup, options.iters);
            result.gbps = throughput_gbps(rows, cols, result.avg_ms);
            return result;
        }
    }

    auto launch = [&]() {
        dispatch_block_tpr(thread_per_row, d_scores, d_values, d_output, rows,
                           cols);
    };
    result.valid = true;
    result.avg_ms = time_kernel(launch, options.warmup, options.iters);
    result.gbps = throughput_gbps(rows, cols, result.avg_ms);
    return result;
}

VariantResult benchmark_cluster_variant(const Options& options,
                                        const DeviceInfo& device,
                                        const float* d_scores,
                                        const float* d_values,
                                        float* d_output,
                                        int rows, int cols) {
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
        cluster_weighted_staged_kernel<kBlockThreads, kMaxClusterSize>,
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
            &config, cluster_weighted_staged_kernel<kBlockThreads, kMaxClusterSize>,
            d_scores, d_values, d_output, rows, cols, slice_elems));
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

void verify_output_samples(const float* d_scores, const float* d_values,
                           const float* d_output, int rows, int cols,
                           const char* label) {
    const auto sample_rows = unique_samples({0, rows / 2, rows - 1}, rows);
    std::vector<float> h_scores(cols);
    std::vector<float> h_values(cols);
    for (int row : sample_rows) {
        CUDA_CHECK(cudaMemcpy(h_scores.data(),
                              d_scores + static_cast<std::size_t>(row) * cols,
                              static_cast<std::size_t>(cols) * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_values.data(),
                              d_values + static_cast<std::size_t>(row) * cols,
                              static_cast<std::size_t>(cols) * sizeof(float),
                              cudaMemcpyDeviceToHost));
        double max_value = -std::numeric_limits<double>::infinity();
        for (float value : h_scores) {
            max_value = std::max(max_value, static_cast<double>(value));
        }
        double sum = 0.0;
        double weighted = 0.0;
        for (int col = 0; col < cols; ++col) {
            const double e = std::exp(static_cast<double>(h_scores[col]) - max_value);
            sum += e;
            weighted += e * static_cast<double>(h_values[col]);
        }
        const double ref = weighted / sum;
        float got = 0.0f;
        CUDA_CHECK(cudaMemcpy(&got, d_output + row, sizeof(float),
                              cudaMemcpyDeviceToHost));
        const double err = std::abs(static_cast<double>(got) - ref);
        if (err > 5e-3 + 5e-3 * std::abs(ref)) {
            std::ostringstream oss;
            oss << label << " verification failed at row=" << row
                << " got=" << got << " ref=" << ref << " err=" << err;
            throw std::runtime_error(oss.str());
        }
    }
}

void verify_selected_variants(const Options& options, const DeviceInfo& device,
                              const VariantResult& block,
                              const VariantResult& cluster,
                              const float* d_scores, const float* d_values,
                              float* d_output, int rows, int cols) {
    if (!options.verify) {
        return;
    }
    Options one_run = options;
    one_run.warmup = 0;
    one_run.iters = 1;
    if (block.valid) {
        benchmark_block_variant(block.thread_per_row, one_run, device, d_scores,
                                d_values, d_output, rows, cols);
        verify_output_samples(d_scores, d_values, d_output, rows, cols, "block");
    }
    if (cluster.valid) {
        benchmark_cluster_variant(one_run, device, d_scores, d_values, d_output,
                                  rows, cols);
        verify_output_samples(d_scores, d_values, d_output, rows, cols, "cluster");
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
}

void run_shape(int cols, const Options& options, const DeviceInfo& device) {
    const int rows = options.batch_size;
    const std::size_t elements = static_cast<std::size_t>(rows) * cols;
    const std::size_t data_bytes = elements * sizeof(float);
    const std::size_t output_bytes = static_cast<std::size_t>(rows) * sizeof(float);

    float* d_scores = nullptr;
    float* d_values = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scores, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_values, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));

    const int init_threads = 256;
    init_input_kernel<<<(elements + init_threads - 1) / init_threads, init_threads>>>(
        d_scores, d_values, elements);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    VariantResult best_block;
    for (int thread_per_row : options.thread_per_row_values) {
        best_block = pick_better(
            best_block,
            benchmark_block_variant(thread_per_row, options, device, d_scores,
                                    d_values, d_output, rows, cols));
    }
    const VariantResult cluster =
        benchmark_cluster_variant(options, device, d_scores, d_values, d_output,
                                  rows, cols);
    verify_selected_variants(options, device, best_block, cluster, d_scores,
                             d_values, d_output, rows, cols);
    print_result(cols, rows, best_block, cluster, device, options.csv);

    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_values));
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
