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
constexpr int kMaxChannels = 16;
constexpr int kMaxClusterSize = 8;
constexpr std::array<int, 4> kDefaultNValues = {8192, 16384, 32768, 65536};
constexpr std::array<int, 3> kDefaultChannelValues = {4, 8, 16};

struct Options {
    int warmup = 2;
    int iters = 5;
    int rows = 2048;
    int requested_cluster_size = 8;
    bool csv = false;
    bool verify = true;
    std::string profile_variant = "all";
    std::vector<int> n_values{kDefaultNValues.begin(), kDefaultNValues.end()};
    std::vector<int> channel_values{kDefaultChannelValues.begin(),
                                    kDefaultChannelValues.end()};
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
    std::string name = "n/a";
    double avg_ms = std::numeric_limits<double>::quiet_NaN();
    double gbps = std::numeric_limits<double>::quiet_NaN();
};

struct MultiAccum {
    float denom;
    float weighted[kMaxChannels];
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
        } else if (arg == "--rows" || arg == "--batch-size") {
            opts.rows = require_int(arg.c_str());
        } else if (arg == "--cluster-size") {
            opts.requested_cluster_size = require_int("--cluster-size");
        } else if (arg == "--n-values") {
            opts.n_values = parse_int_list(require_string("--n-values"));
        } else if (arg == "--channels-values") {
            opts.channel_values = parse_int_list(require_string("--channels-values"));
        } else if (arg == "--profile-variant") {
            opts.profile_variant = require_string("--profile-variant");
        } else if (arg == "--csv") {
            opts.csv = true;
        } else if (arg == "--no-verify") {
            opts.verify = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: multi_value_bench [--csv] [--warmup N] [--iters N] "
                << "[--rows N] [--cluster-size N] [--n-values list] "
                << "[--channels-values list] [--profile-variant all|block|cluster] "
                << "[--no-verify]\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (opts.warmup < 0 || opts.iters <= 0 || opts.rows <= 0 ||
        opts.requested_cluster_size < 1) {
        throw std::runtime_error("numeric arguments must be positive");
    }
    if (opts.profile_variant != "all" && opts.profile_variant != "block" &&
        opts.profile_variant != "cluster") {
        throw std::runtime_error("--profile-variant must be all, block, or cluster");
    }
    for (int channels : opts.channel_values) {
        if (channels != 4 && channels != 8 && channels != 16) {
            throw std::runtime_error("channels must be one of 4, 8, or 16");
        }
    }
    return opts;
}

double modeled_bytes(int rows, int cols, int channels) {
    const double elements = static_cast<double>(rows) * cols;
    return elements * sizeof(float) * (2.0 + static_cast<double>(channels)) +
           static_cast<double>(rows) * channels * sizeof(float);
}

double throughput_gbps(int rows, int cols, int channels, double avg_ms) {
    return modeled_bytes(rows, cols, channels) / (avg_ms / 1000.0) / 1.0e9;
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

template <int Width>
__device__ __forceinline__ float subwarp_reduce_sum(float value) {
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset, Width);
    }
    return value;
}

template <int Width>
__device__ __forceinline__ float subwarp_reduce_max(float value) {
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

template <int Channels, int BlockThreads>
__device__ __forceinline__ MultiAccum block_reduce_multi(MultiAccum value) {
    constexpr int Warps = BlockThreads / 32;
    __shared__ float warp_denom[Warps];
    __shared__ float warp_weighted[Warps][kMaxChannels];
    __shared__ MultiAccum broadcast;

    value.denom = subwarp_reduce_sum<32>(value.denom);
    #pragma unroll
    for (int ch = 0; ch < Channels; ++ch) {
        value.weighted[ch] = subwarp_reduce_sum<32>(value.weighted[ch]);
    }

    if ((threadIdx.x & 31) == 0) {
        const int warp = threadIdx.x >> 5;
        warp_denom[warp] = value.denom;
        #pragma unroll
        for (int ch = 0; ch < Channels; ++ch) {
            warp_weighted[warp][ch] = value.weighted[ch];
        }
    }
    __syncthreads();

    MultiAccum reduced{};
    reduced.denom = 0.0f;
    #pragma unroll
    for (int ch = 0; ch < Channels; ++ch) {
        reduced.weighted[ch] = 0.0f;
    }
    if (threadIdx.x < Warps) {
        reduced.denom = warp_denom[threadIdx.x];
        #pragma unroll
        for (int ch = 0; ch < Channels; ++ch) {
            reduced.weighted[ch] = warp_weighted[threadIdx.x][ch];
        }
    }
    if (threadIdx.x < 32) {
        reduced.denom = subwarp_reduce_sum<32>(reduced.denom);
        #pragma unroll
        for (int ch = 0; ch < Channels; ++ch) {
            reduced.weighted[ch] =
                subwarp_reduce_sum<32>(reduced.weighted[ch]);
        }
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

template <int Channels, int BlockThreads, int MaxClusterSize>
__device__ __forceinline__ MultiAccum
cluster_reduce_multi(cg::cluster_group cluster, MultiAccum block_value) {
    __shared__ float cluster_denom[MaxClusterSize];
    __shared__ float cluster_weighted[MaxClusterSize][kMaxChannels];
    __shared__ MultiAccum broadcast;
    const int block_rank = cluster.block_rank();
    const int blocks_per_cluster = cluster.dim_blocks().x;

    if (threadIdx.x < blocks_per_cluster) {
        const int target_rank = threadIdx.x;
        float* remote_denom =
            cluster.map_shared_rank(&cluster_denom[block_rank], target_rank);
        *remote_denom = block_value.denom;
        #pragma unroll
        for (int ch = 0; ch < Channels; ++ch) {
            float* remote_weighted = cluster.map_shared_rank(
                &cluster_weighted[block_rank][ch], target_rank);
            *remote_weighted = block_value.weighted[ch];
        }
    }
    cluster.sync();

    MultiAccum reduced{};
    reduced.denom = 0.0f;
    #pragma unroll
    for (int ch = 0; ch < Channels; ++ch) {
        reduced.weighted[ch] = 0.0f;
    }
    if (threadIdx.x < blocks_per_cluster) {
        reduced.denom = cluster_denom[threadIdx.x];
        #pragma unroll
        for (int ch = 0; ch < Channels; ++ch) {
            reduced.weighted[ch] = cluster_weighted[threadIdx.x][ch];
        }
    }
    if (threadIdx.x < 32) {
        reduced.denom = subwarp_reduce_sum<32>(reduced.denom);
        #pragma unroll
        for (int ch = 0; ch < Channels; ++ch) {
            reduced.weighted[ch] =
                subwarp_reduce_sum<32>(reduced.weighted[ch]);
        }
    }
    if (threadIdx.x == 0) {
        broadcast = reduced;
    }
    __syncthreads();
    cluster.sync();
    return broadcast;
}

__global__ void init_input_kernel(float* scores, float* values,
                                  std::size_t score_elements,
                                  std::size_t value_elements) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < score_elements) {
        const unsigned x =
            static_cast<unsigned>((idx * 1103515245ull + 12345ull) >> 16);
        scores[idx] = static_cast<float>(x & 4095u) * (1.0f / 512.0f) - 4.0f;
    }
    if (idx < value_elements) {
        const unsigned y =
            static_cast<unsigned>((idx * 1664525ull + 1013904223ull) >> 16);
        values[idx] = static_cast<float>(y & 2047u) * (1.0f / 1024.0f) - 1.0f;
    }
}

template <int Channels, int BlockThreads>
__global__ void block_multi_kernel(const float* __restrict__ scores,
                                   const float* __restrict__ values,
                                   float* __restrict__ output,
                                   int rows,
                                   int cols) {
    const int row = blockIdx.x;
    const bool row_valid = row < rows;
    const std::size_t row_score_base =
        static_cast<std::size_t>(row) * cols;
    const std::size_t row_value_base =
        static_cast<std::size_t>(row) * cols * Channels;

    float local_max = -CUDART_INF_F;
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        const float x = row_valid ? scores[row_score_base + col] : -CUDART_INF_F;
        local_max = fmaxf(local_max, x);
    }
    const float row_max = block_reduce_max<BlockThreads>(local_max);

    MultiAccum local{};
    local.denom = 0.0f;
    #pragma unroll
    for (int ch = 0; ch < Channels; ++ch) {
        local.weighted[ch] = 0.0f;
    }
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        if (!row_valid) {
            continue;
        }
        const float e = __expf(scores[row_score_base + col] - row_max);
        local.denom += e;
        const std::size_t value_base =
            row_value_base + static_cast<std::size_t>(col) * Channels;
        #pragma unroll
        for (int ch = 0; ch < Channels; ++ch) {
            local.weighted[ch] += e * values[value_base + ch];
        }
    }
    const MultiAccum total = block_reduce_multi<Channels, BlockThreads>(local);
    if (row_valid && threadIdx.x == 0) {
        #pragma unroll
        for (int ch = 0; ch < Channels; ++ch) {
            output[static_cast<std::size_t>(row) * Channels + ch] =
                total.weighted[ch] / total.denom;
        }
    }
}

template <int Channels, int BlockThreads>
__global__ void block_multi_staged_kernel(const float* __restrict__ scores,
                                          const float* __restrict__ values,
                                          float* __restrict__ output,
                                          int rows,
                                          int cols) {
    extern __shared__ float tile[];
    const int row = blockIdx.x;
    const bool row_valid = row < rows;
    const std::size_t row_score_base =
        static_cast<std::size_t>(row) * cols;
    const std::size_t row_value_base =
        static_cast<std::size_t>(row) * cols * Channels;

    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        tile[col] = row_valid ? scores[row_score_base + col] : -CUDART_INF_F;
    }
    __syncthreads();

    float local_max = -CUDART_INF_F;
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        local_max = fmaxf(local_max, tile[col]);
    }
    const float row_max = block_reduce_max<BlockThreads>(local_max);

    MultiAccum local{};
    local.denom = 0.0f;
    #pragma unroll
    for (int ch = 0; ch < Channels; ++ch) {
        local.weighted[ch] = 0.0f;
    }
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        if (!row_valid) {
            continue;
        }
        const float e = __expf(tile[col] - row_max);
        local.denom += e;
        const std::size_t value_base =
            row_value_base + static_cast<std::size_t>(col) * Channels;
        #pragma unroll
        for (int ch = 0; ch < Channels; ++ch) {
            local.weighted[ch] += e * values[value_base + ch];
        }
    }
    const MultiAccum total = block_reduce_multi<Channels, BlockThreads>(local);
    if (row_valid && threadIdx.x == 0) {
        #pragma unroll
        for (int ch = 0; ch < Channels; ++ch) {
            output[static_cast<std::size_t>(row) * Channels + ch] =
                total.weighted[ch] / total.denom;
        }
    }
}

template <int Channels, int BlockThreads, int MaxClusterSize>
__global__ void cluster_multi_staged_kernel(const float* __restrict__ scores,
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
    const std::size_t row_score_base =
        static_cast<std::size_t>(row) * cols;
    const std::size_t row_value_base =
        static_cast<std::size_t>(row) * cols * Channels;

    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        tile[col] = scores[row_score_base + slice_start + col];
    }
    __syncthreads();

    float local_max = -CUDART_INF_F;
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        local_max = fmaxf(local_max, tile[col]);
    }
    const float block_max = block_reduce_max<BlockThreads>(local_max);
    const float row_max =
        cluster_reduce_max<BlockThreads, MaxClusterSize>(cluster, block_max);

    MultiAccum local{};
    local.denom = 0.0f;
    #pragma unroll
    for (int ch = 0; ch < Channels; ++ch) {
        local.weighted[ch] = 0.0f;
    }
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        const float e = __expf(tile[col] - row_max);
        local.denom += e;
        const std::size_t value_base =
            row_value_base + static_cast<std::size_t>(slice_start + col) * Channels;
        #pragma unroll
        for (int ch = 0; ch < Channels; ++ch) {
            local.weighted[ch] += e * values[value_base + ch];
        }
    }
    const MultiAccum block_total =
        block_reduce_multi<Channels, BlockThreads>(local);
    const MultiAccum row_total =
        cluster_reduce_multi<Channels, BlockThreads, MaxClusterSize>(
            cluster, block_total);
    if (row_valid && block_rank == 0 && threadIdx.x == 0) {
        #pragma unroll
        for (int ch = 0; ch < Channels; ++ch) {
            output[static_cast<std::size_t>(row) * Channels + ch] =
                row_total.weighted[ch] / row_total.denom;
        }
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
            cluster_multi_staged_kernel<4, kBlockThreads, kMaxClusterSize>,
            &config));
        info.max_cluster_size = max_cluster_size;
        info.chosen_cluster_size =
            std::min({options.requested_cluster_size, max_cluster_size,
                      kMaxClusterSize});
    }
    return info;
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

template <int Channels>
VariantResult benchmark_block_read2(const Options& options,
                                    const float* d_scores,
                                    const float* d_values,
                                    float* d_output,
                                    int rows,
                                    int cols) {
    VariantResult result;
    result.valid = true;
    result.name = "block_read2";
    auto launch = [&]() {
        block_multi_kernel<Channels, kBlockThreads>
            <<<rows, kBlockThreads>>>(d_scores, d_values, d_output, rows, cols);
    };
    result.avg_ms = time_kernel(launch, options.warmup, options.iters);
    result.gbps = throughput_gbps(rows, cols, Channels, result.avg_ms);
    return result;
}

template <int Channels>
VariantResult benchmark_block_staged(const Options& options,
                                     const DeviceInfo& device,
                                     const float* d_scores,
                                     const float* d_values,
                                     float* d_output,
                                     int rows,
                                     int cols) {
    VariantResult result;
    const std::size_t smem_bytes = static_cast<std::size_t>(cols) * sizeof(float);
    if (smem_bytes > static_cast<std::size_t>(device.max_dynamic_smem_per_block)) {
        return result;
    }
    CUDA_CHECK(cudaFuncSetAttribute(
        block_multi_staged_kernel<Channels, kBlockThreads>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
    result.valid = true;
    result.name = "block_smem";
    auto launch = [&]() {
        block_multi_staged_kernel<Channels, kBlockThreads>
            <<<rows, kBlockThreads, smem_bytes>>>(d_scores, d_values, d_output,
                                                  rows, cols);
    };
    result.avg_ms = time_kernel(launch, options.warmup, options.iters);
    result.gbps = throughput_gbps(rows, cols, Channels, result.avg_ms);
    return result;
}

template <int Channels>
VariantResult benchmark_best_block(const Options& options,
                                   const DeviceInfo& device,
                                   const float* d_scores,
                                   const float* d_values,
                                   float* d_output,
                                   int rows,
                                   int cols) {
    VariantResult best;
    best = pick_better(best, benchmark_block_read2<Channels>(
                                 options, d_scores, d_values, d_output, rows, cols));
    best = pick_better(best, benchmark_block_staged<Channels>(
                                 options, device, d_scores, d_values, d_output,
                                 rows, cols));
    return best;
}

template <int Channels>
VariantResult benchmark_cluster(const Options& options,
                                const DeviceInfo& device,
                                const float* d_scores,
                                const float* d_values,
                                float* d_output,
                                int rows,
                                int cols) {
    VariantResult result;
    result.name = "cluster_smem";
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
        cluster_multi_staged_kernel<Channels, kBlockThreads, kMaxClusterSize>,
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
            &config,
            cluster_multi_staged_kernel<Channels, kBlockThreads, kMaxClusterSize>,
            d_scores, d_values, d_output, rows, cols, slice_elems));
    };
    result.valid = true;
    result.avg_ms = time_kernel(launch, options.warmup, options.iters);
    result.gbps = throughput_gbps(rows, cols, Channels, result.avg_ms);
    return result;
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

template <int Channels>
void verify_output_samples(const float* d_scores,
                           const float* d_values,
                           const float* d_output,
                           int rows,
                           int cols,
                           const char* label) {
    const auto sample_rows = unique_samples({0, rows / 2, rows - 1}, rows);
    std::vector<float> h_scores(cols);
    std::vector<float> h_values(static_cast<std::size_t>(cols) * Channels);
    std::vector<float> h_output(Channels);
    for (int row : sample_rows) {
        CUDA_CHECK(cudaMemcpy(h_scores.data(),
                              d_scores + static_cast<std::size_t>(row) * cols,
                              static_cast<std::size_t>(cols) * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            h_values.data(),
            d_values + static_cast<std::size_t>(row) * cols * Channels,
            static_cast<std::size_t>(cols) * Channels * sizeof(float),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_output.data(),
                              d_output + static_cast<std::size_t>(row) * Channels,
                              Channels * sizeof(float),
                              cudaMemcpyDeviceToHost));

        double max_value = -std::numeric_limits<double>::infinity();
        for (float value : h_scores) {
            max_value = std::max(max_value, static_cast<double>(value));
        }
        double denom = 0.0;
        std::array<double, kMaxChannels> weighted{};
        for (int col = 0; col < cols; ++col) {
            const double e =
                std::exp(static_cast<double>(h_scores[col]) - max_value);
            denom += e;
            for (int ch = 0; ch < Channels; ++ch) {
                weighted[ch] +=
                    e * static_cast<double>(
                            h_values[static_cast<std::size_t>(col) * Channels + ch]);
            }
        }
        for (int ch = 0; ch < Channels; ++ch) {
            const double ref = weighted[ch] / denom;
            const double got = static_cast<double>(h_output[ch]);
            const double err = std::abs(got - ref);
            if (err > 5e-3 + 5e-3 * std::abs(ref)) {
                std::ostringstream oss;
                oss << label << " verification failed row=" << row
                    << " channel=" << ch << " got=" << got << " ref=" << ref
                    << " err=" << err;
                throw std::runtime_error(oss.str());
            }
        }
    }
}

template <int Channels>
void verify_selected(const Options& options,
                     const DeviceInfo& device,
                     const VariantResult& block,
                     const VariantResult& cluster,
                     const float* d_scores,
                     const float* d_values,
                     float* d_output,
                     int rows,
                     int cols) {
    if (!options.verify) {
        return;
    }
    Options one_run = options;
    one_run.warmup = 0;
    one_run.iters = 1;
    if (block.valid && options.profile_variant != "cluster") {
        if (block.name == "block_smem") {
            benchmark_block_staged<Channels>(one_run, device, d_scores, d_values,
                                             d_output, rows, cols);
        } else {
            benchmark_block_read2<Channels>(one_run, d_scores, d_values,
                                            d_output, rows, cols);
        }
        verify_output_samples<Channels>(d_scores, d_values, d_output, rows, cols,
                                        "block");
    }
    if (cluster.valid && options.profile_variant != "block") {
        benchmark_cluster<Channels>(one_run, device, d_scores, d_values, d_output,
                                    rows, cols);
        verify_output_samples<Channels>(d_scores, d_values, d_output, rows, cols,
                                        "cluster");
    }
}

void print_device_header(const Options& options, const DeviceInfo& device) {
    if (!options.csv) {
        std::cout << "Device: " << device.name << " sm_" << device.major
                  << device.minor << "\n";
        std::cout << "Rows: " << options.rows << ", warmup: " << options.warmup
                  << ", iters: " << options.iters
                  << ", cluster size: " << device.chosen_cluster_size << "\n";
        return;
    }
    std::cout << "META,device_name," << device.name << "\n";
    std::cout << "META,sm," << device.major << "." << device.minor << "\n";
    std::cout << "META,rows," << options.rows << "\n";
    std::cout << "META,warmup," << options.warmup << "\n";
    std::cout << "META,iters," << options.iters << "\n";
    std::cout << "META,cluster_launch," << (device.cluster_launch ? 1 : 0)
              << "\n";
    std::cout << "META,requested_cluster_size," << options.requested_cluster_size
              << "\n";
    std::cout << "META,max_cluster_size," << device.max_cluster_size << "\n";
    std::cout << "META,chosen_cluster_size," << device.chosen_cluster_size
              << "\n";
}

void print_result(int cols,
                  int channels,
                  int rows,
                  const VariantResult& block,
                  const VariantResult& cluster,
                  const DeviceInfo& device,
                  bool csv) {
    const double cluster_vs_block =
        (block.valid && cluster.valid) ? block.avg_ms / cluster.avg_ms
                                       : std::numeric_limits<double>::quiet_NaN();
    if (csv) {
        std::cout << "RESULT," << cols << "," << channels << "," << rows << ",";
        if (block.valid) {
            std::cout << block.name << "," << block.avg_ms << "," << block.gbps
                      << ",";
        } else {
            std::cout << "n/a,n/a,n/a,";
        }
        if (cluster.valid) {
            std::cout << cluster.name << "," << cluster.avg_ms << ","
                      << cluster.gbps << "," << device.chosen_cluster_size << ",";
        } else {
            std::cout << "n/a,n/a,n/a," << device.chosen_cluster_size << ",";
        }
        if (std::isnan(cluster_vs_block)) {
            std::cout << "n/a\n";
        } else {
            std::cout << cluster_vs_block << "\n";
        }
        return;
    }

    std::cout << std::setw(8) << cols << std::setw(6) << channels
              << std::setw(14) << block.name << std::setw(12) << block.avg_ms
              << std::setw(14) << cluster.name << std::setw(12)
              << cluster.avg_ms << std::setw(12) << cluster_vs_block << "\n";
}

template <int Channels>
void run_shape_channels(int cols, const Options& options, const DeviceInfo& device) {
    const int rows = options.rows;
    const std::size_t score_elements = static_cast<std::size_t>(rows) * cols;
    const std::size_t value_elements = score_elements * Channels;
    const std::size_t output_elements = static_cast<std::size_t>(rows) * Channels;

    float* d_scores = nullptr;
    float* d_values = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scores, score_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_values, value_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, output_elements * sizeof(float)));

    const std::size_t init_elements = std::max(score_elements, value_elements);
    const int init_threads = 256;
    init_input_kernel<<<(init_elements + init_threads - 1) / init_threads,
                        init_threads>>>(d_scores, d_values, score_elements,
                                        value_elements);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    VariantResult block;
    VariantResult cluster;
    if (options.profile_variant != "cluster") {
        block = benchmark_best_block<Channels>(options, device, d_scores, d_values,
                                               d_output, rows, cols);
    }
    if (options.profile_variant != "block") {
        cluster = benchmark_cluster<Channels>(options, device, d_scores, d_values,
                                             d_output, rows, cols);
    }
    verify_selected<Channels>(options, device, block, cluster, d_scores, d_values,
                              d_output, rows, cols);
    print_result(cols, Channels, rows, block, cluster, device, options.csv);

    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_output));
}

void run_shape(int cols, int channels, const Options& options,
               const DeviceInfo& device) {
    switch (channels) {
        case 4:
            run_shape_channels<4>(cols, options, device);
            break;
        case 8:
            run_shape_channels<8>(cols, options, device);
            break;
        case 16:
            run_shape_channels<16>(cols, options, device);
            break;
        default:
            throw std::runtime_error("unsupported channel count");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const DeviceInfo device = query_device_info(options);
        print_device_header(options, device);
        if (!options.csv) {
            std::cout << std::setw(8) << "N" << std::setw(6) << "D"
                      << std::setw(14) << "block" << std::setw(12) << "ms"
                      << std::setw(14) << "cluster" << std::setw(12) << "ms"
                      << std::setw(12) << "c/b" << "\n";
        }
        for (int channels : options.channel_values) {
            for (int cols : options.n_values) {
                run_shape(cols, channels, options, device);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
