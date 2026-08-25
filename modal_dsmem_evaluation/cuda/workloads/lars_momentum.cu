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
constexpr int kMaxClusterSize = 8;
constexpr float kEps = 1e-6f;
constexpr float kMomentumCoeff = 0.9f;
constexpr float kWeightDecay = 0.01f;
constexpr float kLearningRate = 1.0e-3f;
constexpr float kTrustCoeff = 0.02f;
constexpr std::array<int, 5> kDefaultNValues = {
    4096, 8192, 16384, 32768, 65536,
};

struct Options {
    int warmup = 2;
    int iters = 5;
    int rows = 4096;
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
        } else if (arg == "--rows" || arg == "--batch-size") {
            opts.rows = require_int(arg.c_str());
        } else if (arg == "--cluster-size") {
            opts.requested_cluster_size = require_int("--cluster-size");
        } else if (arg == "--n-values") {
            opts.n_values = parse_int_list(require_string("--n-values"));
        } else if (arg == "--profile-variant") {
            opts.profile_variant = require_string("--profile-variant");
        } else if (arg == "--csv") {
            opts.csv = true;
        } else if (arg == "--no-verify") {
            opts.verify = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: lars_momentum_bench [--csv] [--warmup N] "
                << "[--iters N] [--rows N] [--cluster-size N] "
                << "[--n-values list] [--profile-variant all|block|cluster] "
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
    return opts;
}

double modeled_bytes(int rows, int cols) {
    return static_cast<double>(rows) * cols * sizeof(float) * 8.0;
}

double throughput_gbps(int rows, int cols, double avg_ms) {
    return modeled_bytes(rows, cols) / (avg_ms / 1000.0) / 1.0e9;
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

__global__ void init_inputs_kernel(float* weight,
                                   float* grad,
                                   float* momentum,
                                   std::size_t elements) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }
    const unsigned a =
        static_cast<unsigned>((idx * 1103515245ull + 12345ull) >> 16);
    const unsigned b =
        static_cast<unsigned>((idx * 1664525ull + 1013904223ull) >> 16);
    const unsigned c =
        static_cast<unsigned>((idx * 22695477ull + 1ull) >> 16);
    weight[idx] = static_cast<float>(a & 2047u) * (1.0f / 1024.0f) - 1.0f;
    grad[idx] = static_cast<float>(b & 2047u) * (1.0f / 1024.0f) - 1.0f;
    momentum[idx] = static_cast<float>(c & 2047u) * (1.0f / 1024.0f) - 1.0f;
}

__host__ __device__ __forceinline__ float update_value(float weight,
                                                       float grad,
                                                       float momentum) {
    return kMomentumCoeff * momentum + grad + kWeightDecay * weight;
}

template <int BlockThreads>
__global__ void block_lars_read_kernel(const float* __restrict__ weight,
                                       const float* __restrict__ grad,
                                       const float* __restrict__ momentum,
                                       float* __restrict__ weight_out,
                                       float* __restrict__ momentum_out,
                                       int rows,
                                       int cols) {
    const int row = blockIdx.x;
    const bool row_valid = row < rows;
    const std::size_t row_base = static_cast<std::size_t>(row) * cols;

    Pair stats{0.0f, 0.0f};
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        const float wv = row_valid ? weight[row_base + col] : 0.0f;
        const float gv = row_valid ? grad[row_base + col] : 0.0f;
        const float mv = row_valid ? momentum[row_base + col] : 0.0f;
        const float update = update_value(wv, gv, mv);
        stats.a += wv * wv;
        stats.b += update * update;
    }
    const Pair row_stats = block_reduce_pair<BlockThreads>(stats);
    const float trust =
        kTrustCoeff * sqrtf(row_stats.a) / (sqrtf(row_stats.b) + kEps);
    const float step = kLearningRate * trust;

    if (!row_valid) {
        return;
    }
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        const float wv = weight[row_base + col];
        const float update =
            update_value(wv, grad[row_base + col], momentum[row_base + col]);
        weight_out[row_base + col] = wv - step * update;
        momentum_out[row_base + col] = update;
    }
}

template <int BlockThreads>
__global__ void block_lars_staged_kernel(const float* __restrict__ weight,
                                         const float* __restrict__ grad,
                                         const float* __restrict__ momentum,
                                         float* __restrict__ weight_out,
                                         float* __restrict__ momentum_out,
                                         int rows,
                                         int cols) {
    extern __shared__ float tile[];
    float* tile_w = tile;
    float* tile_g = tile + cols;
    float* tile_m = tile + 2 * cols;
    const int row = blockIdx.x;
    const bool row_valid = row < rows;
    const std::size_t row_base = static_cast<std::size_t>(row) * cols;

    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        tile_w[col] = row_valid ? weight[row_base + col] : 0.0f;
        tile_g[col] = row_valid ? grad[row_base + col] : 0.0f;
        tile_m[col] = row_valid ? momentum[row_base + col] : 0.0f;
    }
    __syncthreads();

    Pair stats{0.0f, 0.0f};
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        const float update = update_value(tile_w[col], tile_g[col], tile_m[col]);
        stats.a += tile_w[col] * tile_w[col];
        stats.b += update * update;
    }
    const Pair row_stats = block_reduce_pair<BlockThreads>(stats);
    const float trust =
        kTrustCoeff * sqrtf(row_stats.a) / (sqrtf(row_stats.b) + kEps);
    const float step = kLearningRate * trust;

    if (!row_valid) {
        return;
    }
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        const float update = update_value(tile_w[col], tile_g[col], tile_m[col]);
        weight_out[row_base + col] = tile_w[col] - step * update;
        momentum_out[row_base + col] = update;
    }
}

template <int BlockThreads, int MaxClusterSize>
__global__ void cluster_lars_staged_kernel(const float* __restrict__ weight,
                                           const float* __restrict__ grad,
                                           const float* __restrict__ momentum,
                                           float* __restrict__ weight_out,
                                           float* __restrict__ momentum_out,
                                           int rows,
                                           int cols,
                                           int slice_elems) {
    extern __shared__ float tile[];
    float* tile_w = tile;
    float* tile_g = tile + slice_elems;
    float* tile_m = tile + 2 * slice_elems;
    cg::cluster_group cluster = cg::this_cluster();
    const int blocks_per_cluster = cluster.dim_blocks().x;
    const int block_rank = cluster.block_rank();
    const int row = blockIdx.x / blocks_per_cluster;
    const bool row_valid = row < rows;
    const int slice_start = block_rank * slice_elems;
    const int local_cols =
        row_valid ? max(0, min(slice_elems, cols - slice_start)) : 0;
    const std::size_t row_base = static_cast<std::size_t>(row) * cols;

    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        tile_w[col] = weight[row_base + slice_start + col];
        tile_g[col] = grad[row_base + slice_start + col];
        tile_m[col] = momentum[row_base + slice_start + col];
    }
    __syncthreads();

    Pair stats{0.0f, 0.0f};
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        const float update = update_value(tile_w[col], tile_g[col], tile_m[col]);
        stats.a += tile_w[col] * tile_w[col];
        stats.b += update * update;
    }
    const Pair block_stats = block_reduce_pair<BlockThreads>(stats);
    const Pair row_stats =
        cluster_reduce_pair<BlockThreads, MaxClusterSize>(cluster, block_stats);
    const float trust =
        kTrustCoeff * sqrtf(row_stats.a) / (sqrtf(row_stats.b) + kEps);
    const float step = kLearningRate * trust;

    if (!row_valid) {
        return;
    }
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        const int global_col = slice_start + col;
        const float update = update_value(tile_w[col], tile_g[col], tile_m[col]);
        weight_out[row_base + global_col] = tile_w[col] - step * update;
        momentum_out[row_base + global_col] = update;
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
            cluster_lars_staged_kernel<kBlockThreads, kMaxClusterSize>,
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
                                   const float* d_weight,
                                   const float* d_grad,
                                   const float* d_momentum,
                                   float* d_weight_out,
                                   float* d_momentum_out,
                                   int rows,
                                   int cols) {
    VariantResult result;
    result.valid = true;
    result.name = "block_read";
    auto launch = [&]() {
        block_lars_read_kernel<kBlockThreads>
            <<<rows, kBlockThreads>>>(d_weight, d_grad, d_momentum,
                                      d_weight_out, d_momentum_out, rows, cols);
    };
    result.avg_ms = time_kernel(launch, options.warmup, options.iters);
    result.gbps = throughput_gbps(rows, cols, result.avg_ms);
    return result;
}

VariantResult benchmark_block_staged(const Options& options,
                                     const DeviceInfo& device,
                                     const float* d_weight,
                                     const float* d_grad,
                                     const float* d_momentum,
                                     float* d_weight_out,
                                     float* d_momentum_out,
                                     int rows,
                                     int cols) {
    VariantResult result;
    const std::size_t smem_bytes =
        static_cast<std::size_t>(cols) * sizeof(float) * 3;
    if (smem_bytes > static_cast<std::size_t>(device.max_dynamic_smem_per_block)) {
        return result;
    }
    CUDA_CHECK(cudaFuncSetAttribute(
        block_lars_staged_kernel<kBlockThreads>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
    result.valid = true;
    result.name = "block_smem";
    auto launch = [&]() {
        block_lars_staged_kernel<kBlockThreads>
            <<<rows, kBlockThreads, smem_bytes>>>(d_weight, d_grad, d_momentum,
                                                  d_weight_out, d_momentum_out,
                                                  rows, cols);
    };
    result.avg_ms = time_kernel(launch, options.warmup, options.iters);
    result.gbps = throughput_gbps(rows, cols, result.avg_ms);
    return result;
}

VariantResult benchmark_best_block(const Options& options,
                                   const DeviceInfo& device,
                                   const float* d_weight,
                                   const float* d_grad,
                                   const float* d_momentum,
                                   float* d_weight_out,
                                   float* d_momentum_out,
                                   int rows,
                                   int cols) {
    VariantResult best;
    best = pick_better(best,
                       benchmark_block_read(options, d_weight, d_grad,
                                            d_momentum, d_weight_out,
                                            d_momentum_out, rows, cols));
    best = pick_better(best,
                       benchmark_block_staged(options, device, d_weight, d_grad,
                                              d_momentum, d_weight_out,
                                              d_momentum_out, rows, cols));
    return best;
}

VariantResult benchmark_cluster(const Options& options,
                                const DeviceInfo& device,
                                const float* d_weight,
                                const float* d_grad,
                                const float* d_momentum,
                                float* d_weight_out,
                                float* d_momentum_out,
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
        static_cast<std::size_t>(slice_elems) * sizeof(float) * 3;
    if (smem_bytes > static_cast<std::size_t>(device.max_dynamic_smem_per_block)) {
        return result;
    }
    CUDA_CHECK(cudaFuncSetAttribute(
        cluster_lars_staged_kernel<kBlockThreads, kMaxClusterSize>,
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
            cluster_lars_staged_kernel<kBlockThreads, kMaxClusterSize>,
            d_weight, d_grad, d_momentum, d_weight_out, d_momentum_out, rows,
            cols, slice_elems));
    };
    result.valid = true;
    result.avg_ms = time_kernel(launch, options.warmup, options.iters);
    result.gbps = throughput_gbps(rows, cols, result.avg_ms);
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

void verify_output_samples(const float* d_weight,
                           const float* d_grad,
                           const float* d_momentum,
                           const float* d_weight_out,
                           const float* d_momentum_out,
                           int rows,
                           int cols,
                           const char* label) {
    const auto sample_rows = unique_samples({0, rows / 2, rows - 1}, rows);
    const auto sample_cols = unique_samples({0, cols / 7, cols / 2, cols - 1}, cols);
    std::vector<float> h_weight(cols);
    std::vector<float> h_grad(cols);
    std::vector<float> h_momentum(cols);

    for (int row : sample_rows) {
        const std::size_t row_base = static_cast<std::size_t>(row) * cols;
        CUDA_CHECK(cudaMemcpy(h_weight.data(), d_weight + row_base,
                              static_cast<std::size_t>(cols) * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_grad.data(), d_grad + row_base,
                              static_cast<std::size_t>(cols) * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_momentum.data(), d_momentum + row_base,
                              static_cast<std::size_t>(cols) * sizeof(float),
                              cudaMemcpyDeviceToHost));

        double weight_sumsq = 0.0;
        double update_sumsq = 0.0;
        for (int col = 0; col < cols; ++col) {
            const double wv = static_cast<double>(h_weight[col]);
            const double update =
                static_cast<double>(kMomentumCoeff) *
                    static_cast<double>(h_momentum[col]) +
                static_cast<double>(h_grad[col]) +
                static_cast<double>(kWeightDecay) * wv;
            weight_sumsq += wv * wv;
            update_sumsq += update * update;
        }
        const double trust = static_cast<double>(kTrustCoeff) *
                             std::sqrt(weight_sumsq) /
                             (std::sqrt(update_sumsq) + kEps);
        const double step = static_cast<double>(kLearningRate) * trust;

        for (int col : sample_cols) {
            const double wv = static_cast<double>(h_weight[col]);
            const double ref_momentum =
                static_cast<double>(kMomentumCoeff) *
                    static_cast<double>(h_momentum[col]) +
                static_cast<double>(h_grad[col]) +
                static_cast<double>(kWeightDecay) * wv;
            const double ref_weight = wv - step * ref_momentum;
            float got = 0.0f;
            CUDA_CHECK(cudaMemcpy(&got, d_weight_out + row_base + col,
                                  sizeof(float),
                                  cudaMemcpyDeviceToHost));
            const double err = std::abs(static_cast<double>(got) - ref_weight);
            if (err > 2e-4 + 2e-4 * std::abs(ref_weight)) {
                std::ostringstream oss;
                oss << label << " weight_out verification failed row=" << row
                    << " col=" << col << " got=" << got
                    << " ref=" << ref_weight << " err=" << err;
                throw std::runtime_error(oss.str());
            }
            CUDA_CHECK(cudaMemcpy(&got, d_momentum_out + row_base + col,
                                  sizeof(float),
                                  cudaMemcpyDeviceToHost));
            const double momentum_err =
                std::abs(static_cast<double>(got) - ref_momentum);
            if (momentum_err > 2e-4 + 2e-4 * std::abs(ref_momentum)) {
                std::ostringstream oss;
                oss << label << " momentum_out verification failed row=" << row
                    << " col=" << col << " got=" << got
                    << " ref=" << ref_momentum << " err=" << momentum_err;
                throw std::runtime_error(oss.str());
            }
        }
    }
}

void verify_selected(const Options& options,
                     const DeviceInfo& device,
                     const VariantResult& block,
                     const VariantResult& cluster,
                     const float* d_weight,
                     const float* d_grad,
                     const float* d_momentum,
                     float* d_weight_out,
                     float* d_momentum_out,
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
            benchmark_block_staged(one_run, device, d_weight, d_grad,
                                   d_momentum, d_weight_out, d_momentum_out,
                                   rows, cols);
        } else {
            benchmark_block_read(one_run, d_weight, d_grad, d_momentum,
                                 d_weight_out, d_momentum_out, rows, cols);
        }
        verify_output_samples(d_weight, d_grad, d_momentum, d_weight_out,
                              d_momentum_out, rows, cols, "block");
    }
    if (cluster.valid && options.profile_variant != "block") {
        benchmark_cluster(one_run, device, d_weight, d_grad, d_momentum,
                          d_weight_out, d_momentum_out, rows, cols);
        verify_output_samples(d_weight, d_grad, d_momentum, d_weight_out,
                              d_momentum_out, rows, cols, "cluster");
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
                  int rows,
                  const VariantResult& block,
                  const VariantResult& cluster,
                  const DeviceInfo& device,
                  bool csv) {
    const double cluster_vs_block =
        (block.valid && cluster.valid) ? block.avg_ms / cluster.avg_ms
                                       : std::numeric_limits<double>::quiet_NaN();
    if (csv) {
        std::cout << "RESULT," << cols << "," << rows << ",";
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

    std::cout << std::setw(8) << cols << std::setw(14) << block.name
              << std::setw(12) << block.avg_ms << std::setw(14) << cluster.name
              << std::setw(12) << cluster.avg_ms << std::setw(12)
              << cluster_vs_block << "\n";
}

void run_shape(int cols, const Options& options, const DeviceInfo& device) {
    const int rows = options.rows;
    const std::size_t elements = static_cast<std::size_t>(rows) * cols;
    float* d_weight = nullptr;
    float* d_grad = nullptr;
    float* d_momentum = nullptr;
    float* d_weight_out = nullptr;
    float* d_momentum_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_weight, elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad, elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_momentum, elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weight_out, elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_momentum_out, elements * sizeof(float)));

    const int init_threads = 256;
    init_inputs_kernel<<<(elements + init_threads - 1) / init_threads,
                         init_threads>>>(d_weight, d_grad, d_momentum,
                                         elements);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    VariantResult block;
    VariantResult cluster;
    if (options.profile_variant != "cluster") {
        block = benchmark_best_block(options, device, d_weight, d_grad,
                                     d_momentum, d_weight_out, d_momentum_out,
                                     rows, cols);
    }
    if (options.profile_variant != "block") {
        cluster = benchmark_cluster(options, device, d_weight, d_grad,
                                    d_momentum, d_weight_out, d_momentum_out,
                                    rows, cols);
    }
    verify_selected(options, device, block, cluster, d_weight, d_grad,
                    d_momentum, d_weight_out, d_momentum_out, rows, cols);
    print_result(cols, rows, block, cluster, device, options.csv);

    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_grad));
    CUDA_CHECK(cudaFree(d_momentum));
    CUDA_CHECK(cudaFree(d_weight_out));
    CUDA_CHECK(cudaFree(d_momentum_out));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const DeviceInfo device = query_device_info(options);
        print_device_header(options, device);
        if (!options.csv) {
            std::cout << std::setw(8) << "N" << std::setw(14) << "block"
                      << std::setw(12) << "ms" << std::setw(14) << "cluster"
                      << std::setw(12) << "ms" << std::setw(12) << "c/b"
                      << "\n";
        }
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
