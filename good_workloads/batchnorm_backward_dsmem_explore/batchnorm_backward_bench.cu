#include <cooperative_groups.h>
#include <cuda_runtime.h>

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
constexpr float kEps = 1e-5f;
constexpr std::array<int, 5> kDefaultNValues = {
    4096, 8192, 16384, 32768, 65536,
};

struct Options {
    int warmup = 2;
    int iters = 5;
    int batch_size = 32;
    int channels = 256;
    int requested_cluster_size = 8;
    bool csv = false;
    bool verify = true;
    std::string profile_variant = "all";
    std::vector<int> n_values{kDefaultNValues.begin(), kDefaultNValues.end()};
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

struct Pair {
    float a;
    float b;
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
        } else if (arg == "--channels") {
            opts.channels = require_int("--channels");
        } else if (arg == "--cluster-size") {
            opts.requested_cluster_size = require_int("--cluster-size");
        } else if (arg == "--n-values" || arg == "--elems-values") {
            opts.n_values = parse_int_list(require_string(arg.c_str()));
        } else if (arg == "--profile-variant") {
            opts.profile_variant = require_string("--profile-variant");
        } else if (arg == "--csv") {
            opts.csv = true;
        } else if (arg == "--no-verify") {
            opts.verify = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: batchnorm_backward_bench [--csv] [--warmup N] "
                << "[--iters N] [--batch-size N] [--channels N] "
                << "[--cluster-size N] [--n-values list] "
                << "[--profile-variant all|block|cluster] [--no-verify]\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (opts.warmup < 0 || opts.iters <= 0 || opts.batch_size <= 0 ||
        opts.channels <= 0 || opts.requested_cluster_size < 1) {
        throw std::runtime_error("numeric arguments must be positive");
    }
    if (opts.profile_variant != "all" && opts.profile_variant != "block" &&
        opts.profile_variant != "cluster") {
        throw std::runtime_error("--profile-variant must be all, block, or cluster");
    }
    for (int value : opts.n_values) {
        if (value <= 0 || value % opts.batch_size != 0) {
            throw std::runtime_error(
                "all n-values must be positive multiples of --batch-size");
        }
    }
    return opts;
}

double modeled_bytes(int channels, int elems_per_channel) {
    const double elements = static_cast<double>(channels) * elems_per_channel;
    return elements * sizeof(float) * 6.0;
}

double throughput_gbps(int channels, int elems_per_channel, double avg_ms) {
    return modeled_bytes(channels, elems_per_channel) / (avg_ms / 1000.0) / 1.0e9;
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

__host__ __device__ __forceinline__ std::size_t nchw1d_index(int elem,
                                                             int channel,
                                                             int channels,
                                                             int spatial) {
    const int n = elem / spatial;
    const int s = elem - n * spatial;
    return (static_cast<std::size_t>(n) * channels + channel) * spatial + s;
}

template <int Width>
__device__ __forceinline__ float subwarp_reduce_sum(float value) {
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset, Width);
    }
    return value;
}

template <int BlockThreads>
__device__ __forceinline__ Pair block_reduce_pair(Pair value) {
    constexpr int Warps = BlockThreads / 32;
    __shared__ float warp_a[Warps];
    __shared__ float warp_b[Warps];
    __shared__ Pair broadcast;

    value.a = subwarp_reduce_sum<32>(value.a);
    value.b = subwarp_reduce_sum<32>(value.b);
    if ((threadIdx.x & 31) == 0) {
        const int warp = threadIdx.x >> 5;
        warp_a[warp] = value.a;
        warp_b[warp] = value.b;
    }
    __syncthreads();

    Pair reduced{0.0f, 0.0f};
    if (threadIdx.x < Warps) {
        reduced.a = warp_a[threadIdx.x];
        reduced.b = warp_b[threadIdx.x];
    }
    if (threadIdx.x < 32) {
        reduced.a = subwarp_reduce_sum<32>(reduced.a);
        reduced.b = subwarp_reduce_sum<32>(reduced.b);
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
    __shared__ float cluster_a[MaxClusterSize];
    __shared__ float cluster_b[MaxClusterSize];
    __shared__ Pair broadcast;
    const int block_rank = cluster.block_rank();
    const int blocks_per_cluster = cluster.dim_blocks().x;

    if (threadIdx.x < blocks_per_cluster) {
        float* remote_a =
            cluster.map_shared_rank(&cluster_a[block_rank], threadIdx.x);
        float* remote_b =
            cluster.map_shared_rank(&cluster_b[block_rank], threadIdx.x);
        *remote_a = block_value.a;
        *remote_b = block_value.b;
    }
    cluster.sync();

    Pair reduced{0.0f, 0.0f};
    if (threadIdx.x < blocks_per_cluster) {
        reduced.a = cluster_a[threadIdx.x];
        reduced.b = cluster_b[threadIdx.x];
    }
    if (threadIdx.x < 32) {
        reduced.a = subwarp_reduce_sum<32>(reduced.a);
        reduced.b = subwarp_reduce_sum<32>(reduced.b);
    }
    if (threadIdx.x == 0) {
        broadcast = reduced;
    }
    __syncthreads();
    cluster.sync();
    return broadcast;
}

__global__ void init_x_dy_kernel(float* x, float* dy, std::size_t elements) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }
    const unsigned a =
        static_cast<unsigned>((idx * 1103515245ull + 12345ull) >> 16);
    const unsigned b =
        static_cast<unsigned>((idx * 1664525ull + 1013904223ull) >> 16);
    x[idx] = static_cast<float>(a & 2047u) * (1.0f / 1024.0f) - 1.0f;
    dy[idx] = static_cast<float>(b & 2047u) * (1.0f / 1024.0f) - 1.0f;
}

__global__ void init_gamma_kernel(float* gamma, int channels) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) {
        return;
    }
    gamma[channel] = 0.75f + static_cast<float>(channel % 31) * 0.015625f;
}

template <int BlockThreads>
__global__ void block_bn_bwd_read_kernel(const float* __restrict__ x,
                                         const float* __restrict__ dy,
                                         const float* __restrict__ gamma,
                                         float* __restrict__ dx,
                                         int batch_size,
                                         int channels,
                                         int spatial,
                                         float eps) {
    const int channel = blockIdx.x;
    const bool channel_valid = channel < channels;
    const int elems_per_channel = batch_size * spatial;

    Pair x_stats{0.0f, 0.0f};
    for (int elem = threadIdx.x; elem < elems_per_channel; elem += BlockThreads) {
        const std::size_t idx =
            nchw1d_index(elem, channel, channels, spatial);
        const float xv = channel_valid ? x[idx] : 0.0f;
        x_stats.a += xv;
        x_stats.b += xv * xv;
    }
    const Pair x_total = block_reduce_pair<BlockThreads>(x_stats);
    const float mean = x_total.a / elems_per_channel;
    const float var =
        fmaxf(x_total.b / elems_per_channel - mean * mean, 0.0f);
    const float inv_std = rsqrtf(var + eps);
    const float g = channel_valid ? gamma[channel] : 0.0f;

    Pair grad_stats{0.0f, 0.0f};
    for (int elem = threadIdx.x; elem < elems_per_channel; elem += BlockThreads) {
        if (!channel_valid) {
            continue;
        }
        const std::size_t idx =
            nchw1d_index(elem, channel, channels, spatial);
        const float xv = x[idx];
        const float xhat = (xv - mean) * inv_std;
        const float dyg = dy[idx] * g;
        grad_stats.a += dyg;
        grad_stats.b += dyg * xhat;
    }
    const Pair grad_total = block_reduce_pair<BlockThreads>(grad_stats);
    const float mean_dyg = grad_total.a / elems_per_channel;
    const float mean_dyg_xhat = grad_total.b / elems_per_channel;

    if (!channel_valid) {
        return;
    }
    for (int elem = threadIdx.x; elem < elems_per_channel; elem += BlockThreads) {
        const std::size_t idx =
            nchw1d_index(elem, channel, channels, spatial);
        const float xv = x[idx];
        const float xhat = (xv - mean) * inv_std;
        const float dyg = dy[idx] * g;
        dx[idx] = (dyg - mean_dyg - xhat * mean_dyg_xhat) * inv_std;
    }
}

template <int BlockThreads>
__global__ void block_bn_bwd_staged_kernel(const float* __restrict__ x,
                                           const float* __restrict__ dy,
                                           const float* __restrict__ gamma,
                                           float* __restrict__ dx,
                                           int batch_size,
                                           int channels,
                                           int spatial,
                                           float eps) {
    extern __shared__ float smem[];
    float* tile_x = smem;
    float* tile_dy = smem + batch_size * spatial;
    const int channel = blockIdx.x;
    const bool channel_valid = channel < channels;
    const int elems_per_channel = batch_size * spatial;

    for (int elem = threadIdx.x; elem < elems_per_channel; elem += BlockThreads) {
        const std::size_t idx =
            nchw1d_index(elem, channel, channels, spatial);
        tile_x[elem] = channel_valid ? x[idx] : 0.0f;
        tile_dy[elem] = channel_valid ? dy[idx] : 0.0f;
    }
    __syncthreads();

    Pair x_stats{0.0f, 0.0f};
    for (int elem = threadIdx.x; elem < elems_per_channel; elem += BlockThreads) {
        const float xv = tile_x[elem];
        x_stats.a += xv;
        x_stats.b += xv * xv;
    }
    const Pair x_total = block_reduce_pair<BlockThreads>(x_stats);
    const float mean = x_total.a / elems_per_channel;
    const float var =
        fmaxf(x_total.b / elems_per_channel - mean * mean, 0.0f);
    const float inv_std = rsqrtf(var + eps);
    const float g = channel_valid ? gamma[channel] : 0.0f;

    Pair grad_stats{0.0f, 0.0f};
    for (int elem = threadIdx.x; elem < elems_per_channel; elem += BlockThreads) {
        const float xv = tile_x[elem];
        const float xhat = (xv - mean) * inv_std;
        const float dyg = tile_dy[elem] * g;
        grad_stats.a += dyg;
        grad_stats.b += dyg * xhat;
    }
    const Pair grad_total = block_reduce_pair<BlockThreads>(grad_stats);
    const float mean_dyg = grad_total.a / elems_per_channel;
    const float mean_dyg_xhat = grad_total.b / elems_per_channel;

    if (!channel_valid) {
        return;
    }
    for (int elem = threadIdx.x; elem < elems_per_channel; elem += BlockThreads) {
        const std::size_t idx =
            nchw1d_index(elem, channel, channels, spatial);
        const float xv = tile_x[elem];
        const float xhat = (xv - mean) * inv_std;
        const float dyg = tile_dy[elem] * g;
        dx[idx] = (dyg - mean_dyg - xhat * mean_dyg_xhat) * inv_std;
    }
}

template <int BlockThreads, int MaxClusterSize>
__global__ void cluster_bn_bwd_staged_kernel(const float* __restrict__ x,
                                             const float* __restrict__ dy,
                                             const float* __restrict__ gamma,
                                             float* __restrict__ dx,
                                             int batch_size,
                                             int channels,
                                             int spatial,
                                             float eps,
                                             int slice_elems) {
    extern __shared__ float smem[];
    float* tile_x = smem;
    float* tile_dy = smem + slice_elems;
    cg::cluster_group cluster = cg::this_cluster();
    const int blocks_per_cluster = cluster.dim_blocks().x;
    const int block_rank = cluster.block_rank();
    const int channel = blockIdx.x / blocks_per_cluster;
    const bool channel_valid = channel < channels;
    const int elems_per_channel = batch_size * spatial;
    const int slice_start = block_rank * slice_elems;
    const int local_elems =
        channel_valid ? max(0, min(slice_elems, elems_per_channel - slice_start)) : 0;

    for (int elem = threadIdx.x * kVecSize; elem < local_elems;
         elem += BlockThreads * kVecSize) {
        const int remaining = local_elems - elem;
        #pragma unroll
        for (int i = 0; i < kVecSize; ++i) {
            if (i < remaining) {
                const int global_elem = slice_start + elem + i;
                const std::size_t idx =
                    nchw1d_index(global_elem, channel, channels, spatial);
                tile_x[elem + i] = x[idx];
                tile_dy[elem + i] = dy[idx];
            }
        }
    }
    __syncthreads();

    Pair x_stats{0.0f, 0.0f};
    for (int elem = threadIdx.x; elem < local_elems; elem += BlockThreads) {
        const float xv = tile_x[elem];
        x_stats.a += xv;
        x_stats.b += xv * xv;
    }
    const Pair block_x = block_reduce_pair<BlockThreads>(x_stats);
    const Pair channel_x =
        cluster_reduce_pair<BlockThreads, MaxClusterSize>(cluster, block_x);
    const float mean = channel_x.a / elems_per_channel;
    const float var =
        fmaxf(channel_x.b / elems_per_channel - mean * mean, 0.0f);
    const float inv_std = rsqrtf(var + eps);
    const float g = channel_valid ? gamma[channel] : 0.0f;

    Pair grad_stats{0.0f, 0.0f};
    for (int elem = threadIdx.x; elem < local_elems; elem += BlockThreads) {
        const float xv = tile_x[elem];
        const float xhat = (xv - mean) * inv_std;
        const float dyg = tile_dy[elem] * g;
        grad_stats.a += dyg;
        grad_stats.b += dyg * xhat;
    }
    const Pair block_grad = block_reduce_pair<BlockThreads>(grad_stats);
    const Pair channel_grad =
        cluster_reduce_pair<BlockThreads, MaxClusterSize>(cluster, block_grad);
    const float mean_dyg = channel_grad.a / elems_per_channel;
    const float mean_dyg_xhat = channel_grad.b / elems_per_channel;

    if (!channel_valid) {
        return;
    }
    for (int elem = threadIdx.x * kVecSize; elem < local_elems;
         elem += BlockThreads * kVecSize) {
        const int remaining = local_elems - elem;
        #pragma unroll
        for (int i = 0; i < kVecSize; ++i) {
            if (i < remaining) {
                const int global_elem = slice_start + elem + i;
                const float xv = tile_x[elem + i];
                const float xhat = (xv - mean) * inv_std;
                const float dyg = tile_dy[elem + i] * g;
                const std::size_t idx =
                    nchw1d_index(global_elem, channel, channels, spatial);
                dx[idx] = (dyg - mean_dyg - xhat * mean_dyg_xhat) * inv_std;
            }
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
            cluster_bn_bwd_staged_kernel<kBlockThreads, kMaxClusterSize>,
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

VariantResult benchmark_block_read(const Options& options,
                                   const float* d_x,
                                   const float* d_dy,
                                   const float* d_gamma,
                                   float* d_dx,
                                   int elems_per_channel) {
    VariantResult result;
    result.valid = true;
    result.name = "block_read";
    const int spatial = elems_per_channel / options.batch_size;
    auto launch = [&]() {
        block_bn_bwd_read_kernel<kBlockThreads>
            <<<options.channels, kBlockThreads>>>(d_x, d_dy, d_gamma, d_dx,
                                                  options.batch_size,
                                                  options.channels, spatial, kEps);
    };
    result.avg_ms = time_kernel(launch, options.warmup, options.iters);
    result.gbps = throughput_gbps(options.channels, elems_per_channel,
                                  result.avg_ms);
    return result;
}

VariantResult benchmark_block_staged(const Options& options,
                                     const DeviceInfo& device,
                                     const float* d_x,
                                     const float* d_dy,
                                     const float* d_gamma,
                                     float* d_dx,
                                     int elems_per_channel) {
    VariantResult result;
    const std::size_t smem_bytes =
        static_cast<std::size_t>(elems_per_channel) * sizeof(float) * 2;
    if (smem_bytes > static_cast<std::size_t>(device.max_dynamic_smem_per_block)) {
        return result;
    }
    CUDA_CHECK(cudaFuncSetAttribute(
        block_bn_bwd_staged_kernel<kBlockThreads>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
    const int spatial = elems_per_channel / options.batch_size;
    auto launch = [&]() {
        block_bn_bwd_staged_kernel<kBlockThreads>
            <<<options.channels, kBlockThreads, smem_bytes>>>(
                d_x, d_dy, d_gamma, d_dx, options.batch_size, options.channels,
                spatial, kEps);
    };
    result.valid = true;
    result.name = "block_smem";
    result.avg_ms = time_kernel(launch, options.warmup, options.iters);
    result.gbps = throughput_gbps(options.channels, elems_per_channel,
                                  result.avg_ms);
    return result;
}

VariantResult benchmark_best_block(const Options& options,
                                   const DeviceInfo& device,
                                   const float* d_x,
                                   const float* d_dy,
                                   const float* d_gamma,
                                   float* d_dx,
                                   int elems_per_channel) {
    VariantResult best;
    best = pick_better(best,
                       benchmark_block_read(options, d_x, d_dy, d_gamma, d_dx,
                                            elems_per_channel));
    best = pick_better(best,
                       benchmark_block_staged(options, device, d_x, d_dy,
                                              d_gamma, d_dx, elems_per_channel));
    return best;
}

VariantResult benchmark_cluster(const Options& options,
                                const DeviceInfo& device,
                                const float* d_x,
                                const float* d_dy,
                                const float* d_gamma,
                                float* d_dx,
                                int elems_per_channel) {
    VariantResult result;
    result.name = "cluster_smem";
    if (device.chosen_cluster_size < 2) {
        return result;
    }
    const int slice_elems =
        (elems_per_channel + device.chosen_cluster_size - 1) /
        device.chosen_cluster_size;
    const std::size_t smem_bytes =
        static_cast<std::size_t>(slice_elems) * sizeof(float) * 2;
    if (smem_bytes > static_cast<std::size_t>(device.max_dynamic_smem_per_block)) {
        return result;
    }
    CUDA_CHECK(cudaFuncSetAttribute(
        cluster_bn_bwd_staged_kernel<kBlockThreads, kMaxClusterSize>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
    const int spatial = elems_per_channel / options.batch_size;
    auto launch = [&]() {
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(options.channels * device.chosen_cluster_size);
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
            cluster_bn_bwd_staged_kernel<kBlockThreads, kMaxClusterSize>,
            d_x, d_dy, d_gamma, d_dx, options.batch_size, options.channels,
            spatial, kEps, slice_elems));
    };
    result.valid = true;
    result.avg_ms = time_kernel(launch, options.warmup, options.iters);
    result.gbps = throughput_gbps(options.channels, elems_per_channel,
                                  result.avg_ms);
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

void verify_output_samples(const float* d_x,
                           const float* d_dy,
                           const float* d_gamma,
                           const float* d_dx,
                           const Options& options,
                           int elems_per_channel,
                           const char* label) {
    const int spatial = elems_per_channel / options.batch_size;
    const std::size_t total_elements =
        static_cast<std::size_t>(options.batch_size) * options.channels * spatial;
    std::vector<float> h_x(total_elements);
    std::vector<float> h_dy(total_elements);
    std::vector<float> h_dx(total_elements);
    std::vector<float> h_gamma(options.channels);
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, total_elements * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dy.data(), d_dy, total_elements * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dx.data(), d_dx, total_elements * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_gamma.data(), d_gamma,
                          options.channels * sizeof(float),
                          cudaMemcpyDeviceToHost));

    const auto sample_channels =
        unique_samples({0, options.channels / 2, options.channels - 1},
                       options.channels);
    const auto sample_elems =
        unique_samples({0, elems_per_channel / 7, elems_per_channel / 2,
                        elems_per_channel - 1},
                       elems_per_channel);
    double max_abs = 0.0;
    for (int channel : sample_channels) {
        long double sum = 0.0L;
        long double sumsq = 0.0L;
        for (int elem = 0; elem < elems_per_channel; ++elem) {
            const std::size_t idx =
                nchw1d_index(elem, channel, options.channels, spatial);
            const long double x = h_x[idx];
            sum += x;
            sumsq += x * x;
        }
        const long double mean = sum / elems_per_channel;
        const long double variance =
            std::max<long double>(sumsq / elems_per_channel - mean * mean, 0.0L);
        const long double inv_std = 1.0L / std::sqrt(variance + kEps);

        long double grad_sum = 0.0L;
        long double grad_xhat_sum = 0.0L;
        for (int elem = 0; elem < elems_per_channel; ++elem) {
            const std::size_t idx =
                nchw1d_index(elem, channel, options.channels, spatial);
            const long double xhat =
                (static_cast<long double>(h_x[idx]) - mean) * inv_std;
            const long double dyg =
                static_cast<long double>(h_dy[idx]) * h_gamma[channel];
            grad_sum += dyg;
            grad_xhat_sum += dyg * xhat;
        }
        const long double mean_dyg = grad_sum / elems_per_channel;
        const long double mean_dyg_xhat = grad_xhat_sum / elems_per_channel;
        for (int elem : sample_elems) {
            const std::size_t idx =
                nchw1d_index(elem, channel, options.channels, spatial);
            const long double xhat =
                (static_cast<long double>(h_x[idx]) - mean) * inv_std;
            const long double dyg =
                static_cast<long double>(h_dy[idx]) * h_gamma[channel];
            const long double expected =
                (dyg - mean_dyg - xhat * mean_dyg_xhat) * inv_std;
            const double abs_err =
                std::abs(static_cast<double>(h_dx[idx] - expected));
            max_abs = std::max(max_abs, abs_err);
        }
    }
    if (max_abs > 1.0e-3) {
        std::cerr << "verification failed for " << label
                  << ": max_abs=" << max_abs << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

std::string fmt_result(double value, int precision = 6) {
    if (std::isnan(value)) {
        return "n/a";
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(precision) << value;
    return ss.str();
}

void print_human_row(int elems_per_channel,
                     int batch_size,
                     int channels,
                     int spatial,
                     const VariantResult& block,
                     const VariantResult& cluster,
                     int cluster_size) {
    std::cout << "elems/channel=" << elems_per_channel
              << " batch=" << batch_size
              << " channels=" << channels
              << " spatial=" << spatial << "\n";
    std::cout << "  block(" << block.name << "):   "
              << fmt_result(block.avg_ms, 4) << " ms, "
              << fmt_result(block.gbps, 2) << " GB/s\n";
    std::cout << "  cluster(" << cluster.name << ", size=" << cluster_size
              << "): " << fmt_result(cluster.avg_ms, 4) << " ms, "
              << fmt_result(cluster.gbps, 2) << " GB/s\n";
    if (block.valid && cluster.valid) {
        std::cout << "  cluster/block speedup: "
                  << fmt_result(block.avg_ms / cluster.avg_ms, 3) << "x\n";
    }
}

void print_csv_row(int elems_per_channel,
                   int batch_size,
                   int channels,
                   int spatial,
                   const VariantResult& block,
                   const VariantResult& cluster,
                   int cluster_size) {
    std::cout << "RESULT," << elems_per_channel << "," << batch_size << ","
              << channels << "," << spatial << "," << block.name << ","
              << fmt_result(block.avg_ms) << "," << fmt_result(block.gbps) << ","
              << cluster.name << "," << fmt_result(cluster.avg_ms) << ","
              << fmt_result(cluster.gbps) << "," << cluster_size << ",";
    if (block.valid && cluster.valid) {
        std::cout << fmt_result(block.avg_ms / cluster.avg_ms);
    } else {
        std::cout << "n/a";
    }
    std::cout << "\n";
}

void print_device_summary(const DeviceInfo& device) {
    std::cout << "Device: " << device.name << " (SM " << device.major << "."
              << device.minor << ")\n";
    std::cout << "Cluster launch: " << (device.cluster_launch ? "yes" : "no")
              << ", max cluster size: " << device.max_cluster_size
              << ", chosen cluster size: " << device.chosen_cluster_size
              << ", max opt-in dynamic smem/block: "
              << device.max_dynamic_smem_per_block << " bytes\n";
}

int run(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    const DeviceInfo device = query_device_info(options);

    if (!options.csv) {
        print_device_summary(device);
        std::cout << "BatchNorm backward dx over NCHW-like [B,C,S] data\n";
    }

    for (int elems_per_channel : options.n_values) {
        const int spatial = elems_per_channel / options.batch_size;
        const std::size_t total_elements =
            static_cast<std::size_t>(options.batch_size) * options.channels *
            spatial;
        float* d_x = nullptr;
        float* d_dy = nullptr;
        float* d_gamma = nullptr;
        float* d_dx = nullptr;
        CUDA_CHECK(cudaMalloc(&d_x, total_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dy, total_elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gamma, options.channels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dx, total_elements * sizeof(float)));

        const int init_threads = 256;
        const int init_blocks =
            static_cast<int>((total_elements + init_threads - 1) / init_threads);
        init_x_dy_kernel<<<init_blocks, init_threads>>>(d_x, d_dy, total_elements);
        const int gamma_blocks =
            (options.channels + init_threads - 1) / init_threads;
        init_gamma_kernel<<<gamma_blocks, init_threads>>>(d_gamma, options.channels);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        VariantResult block;
        VariantResult cluster;
        if (options.profile_variant != "cluster") {
            block = benchmark_best_block(options, device, d_x, d_dy, d_gamma,
                                         d_dx, elems_per_channel);
            if (options.verify && block.valid) {
                verify_output_samples(d_x, d_dy, d_gamma, d_dx, options,
                                      elems_per_channel, block.name.c_str());
            }
        }
        if (options.profile_variant != "block") {
            cluster = benchmark_cluster(options, device, d_x, d_dy, d_gamma,
                                        d_dx, elems_per_channel);
            if (options.verify && cluster.valid) {
                verify_output_samples(d_x, d_dy, d_gamma, d_dx, options,
                                      elems_per_channel, cluster.name.c_str());
            }
        }

        if (options.csv) {
            print_csv_row(elems_per_channel, options.batch_size, options.channels,
                          spatial, block, cluster, device.chosen_cluster_size);
        } else {
            print_human_row(elems_per_channel, options.batch_size,
                            options.channels, spatial, block, cluster,
                            device.chosen_cluster_size);
        }

        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_dy));
        CUDA_CHECK(cudaFree(d_gamma));
        CUDA_CHECK(cudaFree(d_dx));
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}
