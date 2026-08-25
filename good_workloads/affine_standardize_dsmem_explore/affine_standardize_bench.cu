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
constexpr float kEps = 1.0e-6f;
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
                << "Usage: affine_standardize_bench [--csv] [--warmup N] "
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
    return static_cast<double>(rows) * cols * sizeof(float) * 5.0;
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
    cluster.sync();
    return broadcast;
}

__global__ void init_input_kernel(float* data, std::size_t elements) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }
    const unsigned a =
        static_cast<unsigned>((idx * 1103515245ull + 12345ull) >> 16);
    const unsigned b =
        static_cast<unsigned>((idx * 1664525ull + 1013904223ull) >> 16);
    const float base = static_cast<float>(a & 4095u) * (1.0f / 1024.0f) - 2.0f;
    const float tiny = static_cast<float>(b & 255u) * 1.0e-7f;
    data[idx] = base + tiny;
}

__host__ __device__ __forceinline__ float normalize_value(float value,
                                                          float gamma,
                                                          float beta,
                                                          Pair row,
                                                          int cols) {
    const float inv_n = 1.0f / static_cast<float>(cols);
    const float mean = row.sum * inv_n;
    const float var = fmaxf(row.sumsq * inv_n - mean * mean, 0.0f);
    const float norm = (value - mean) * rsqrtf(var + kEps);
    return norm * gamma + beta;
}

template <int BlockThreads>
__global__ void block_norm_read_kernel(const float* __restrict__ input,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta,
                                       float* __restrict__ output,
                                       int rows,
                                       int cols) {
    const int row = blockIdx.x;
    const bool row_valid = row < rows;
    const std::size_t row_base = static_cast<std::size_t>(row) * cols;

    Pair local{0.0f, 0.0f};
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        const float value = row_valid ? input[row_base + col] : 0.0f;
        local.sum += value;
        local.sumsq += value * value;
    }
    const Pair row_stats = block_reduce_pair<BlockThreads>(local);
    if (!row_valid) {
        return;
    }
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        output[row_base + col] =
            normalize_value(input[row_base + col], gamma[row_base + col],
                            beta[row_base + col], row_stats, cols);
    }
}

template <int BlockThreads>
__global__ void block_norm_staged_kernel(const float* __restrict__ input,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ beta,
                                         float* __restrict__ output,
                                         int rows,
                                         int cols) {
    extern __shared__ float tile[];
    float* tile_x = tile;
    float* tile_gamma = tile + cols;
    float* tile_beta = tile + 2 * cols;
    const int row = blockIdx.x;
    const bool row_valid = row < rows;
    const std::size_t row_base = static_cast<std::size_t>(row) * cols;

    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        tile_x[col] = row_valid ? input[row_base + col] : 0.0f;
        tile_gamma[col] = row_valid ? gamma[row_base + col] : 0.0f;
        tile_beta[col] = row_valid ? beta[row_base + col] : 0.0f;
    }
    __syncthreads();

    Pair local{0.0f, 0.0f};
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        const float value = tile_x[col];
        local.sum += value;
        local.sumsq += value * value;
    }
    const Pair row_stats = block_reduce_pair<BlockThreads>(local);
    if (!row_valid) {
        return;
    }
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        output[row_base + col] =
            normalize_value(tile_x[col], tile_gamma[col], tile_beta[col],
                            row_stats, cols);
    }
}

template <int BlockThreads, int MaxClusterSize>
__global__ void cluster_norm_staged_kernel(const float* __restrict__ input,
                                           const float* __restrict__ gamma,
                                           const float* __restrict__ beta,
                                           float* __restrict__ output,
                                           int rows,
                                           int cols,
                                           int slice_elems) {
    extern __shared__ float tile[];
    float* tile_x = tile;
    float* tile_gamma = tile + slice_elems;
    float* tile_beta = tile + 2 * slice_elems;
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
        const std::size_t idx = row_base + slice_start + col;
        tile_x[col] = input[idx];
        tile_gamma[col] = gamma[idx];
        tile_beta[col] = beta[idx];
    }
    __syncthreads();

    Pair local{0.0f, 0.0f};
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        const float value = tile_x[col];
        local.sum += value;
        local.sumsq += value * value;
    }
    const Pair block_stats = block_reduce_pair<BlockThreads>(local);
    const Pair row_stats =
        cluster_reduce_pair<BlockThreads, MaxClusterSize>(cluster, block_stats);
    if (!row_valid) {
        return;
    }
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        output[row_base + slice_start + col] =
            normalize_value(tile_x[col], tile_gamma[col], tile_beta[col],
                            row_stats, cols);
    }
    cluster.sync();
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
            cluster_norm_staged_kernel<kBlockThreads, kMaxClusterSize>,
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
                                   const float* d_input,
                                   const float* d_gamma,
                                   const float* d_beta,
                                   float* d_output,
                                   int rows,
                                   int cols) {
    VariantResult result;
    result.valid = true;
    result.name = "block_read";
    auto launch = [&]() {
        block_norm_read_kernel<kBlockThreads>
            <<<rows, kBlockThreads>>>(d_input, d_gamma, d_beta, d_output, rows,
                                      cols);
    };
    result.avg_ms = time_kernel(launch, options.warmup, options.iters);
    result.gbps = throughput_gbps(rows, cols, result.avg_ms);
    return result;
}

VariantResult benchmark_block_staged(const Options& options,
                                     const DeviceInfo& device,
                                     const float* d_input,
                                     const float* d_gamma,
                                     const float* d_beta,
                                     float* d_output,
                                     int rows,
                                     int cols) {
    VariantResult result;
    const std::size_t smem_bytes =
        static_cast<std::size_t>(cols) * sizeof(float) * 3;
    if (smem_bytes > static_cast<std::size_t>(device.max_dynamic_smem_per_block)) {
        return result;
    }
    CUDA_CHECK(cudaFuncSetAttribute(
        block_norm_staged_kernel<kBlockThreads>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
    result.valid = true;
    result.name = "block_smem";
    auto launch = [&]() {
        block_norm_staged_kernel<kBlockThreads>
            <<<rows, kBlockThreads, smem_bytes>>>(d_input, d_gamma, d_beta,
                                                  d_output, rows, cols);
    };
    result.avg_ms = time_kernel(launch, options.warmup, options.iters);
    result.gbps = throughput_gbps(rows, cols, result.avg_ms);
    return result;
}

VariantResult benchmark_best_block(const Options& options,
                                   const DeviceInfo& device,
                                   const float* d_input,
                                   const float* d_gamma,
                                   const float* d_beta,
                                   float* d_output,
                                   int rows,
                                   int cols) {
    VariantResult best;
    best = pick_better(best,
                       benchmark_block_read(options, d_input, d_gamma, d_beta,
                                            d_output, rows, cols));
    best = pick_better(best,
                       benchmark_block_staged(options, device, d_input,
                                              d_gamma, d_beta, d_output, rows,
                                              cols));
    return best;
}

VariantResult benchmark_cluster(const Options& options,
                                const DeviceInfo& device,
                                const float* d_input,
                                const float* d_gamma,
                                const float* d_beta,
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
        static_cast<std::size_t>(slice_elems) * sizeof(float) * 3;
    if (smem_bytes > static_cast<std::size_t>(device.max_dynamic_smem_per_block)) {
        return result;
    }
    CUDA_CHECK(cudaFuncSetAttribute(
        cluster_norm_staged_kernel<kBlockThreads, kMaxClusterSize>,
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
            cluster_norm_staged_kernel<kBlockThreads, kMaxClusterSize>,
            d_input, d_gamma, d_beta, d_output, rows, cols, slice_elems));
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

void verify_output_samples(const float* d_input,
                           const float* d_gamma,
                           const float* d_beta,
                           const float* d_output,
                           int rows,
                           int cols,
                           const char* label) {
    const auto sample_rows = unique_samples({0, rows / 2, rows - 1}, rows);
    const auto sample_cols = unique_samples({0, cols / 7, cols / 2, cols - 1}, cols);
    std::vector<float> h_input(cols);
    std::vector<float> h_gamma(cols);
    std::vector<float> h_beta(cols);
    std::vector<float> h_output(cols);

    for (int row : sample_rows) {
        const std::size_t row_base = static_cast<std::size_t>(row) * cols;
        CUDA_CHECK(cudaMemcpy(h_input.data(), d_input + row_base,
                              static_cast<std::size_t>(cols) * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_gamma.data(), d_gamma + row_base,
                              static_cast<std::size_t>(cols) * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_beta.data(), d_beta + row_base,
                              static_cast<std::size_t>(cols) * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output + row_base,
                              static_cast<std::size_t>(cols) * sizeof(float),
                              cudaMemcpyDeviceToHost));
        double sum = 0.0;
        double sumsq = 0.0;
        for (int col = 0; col < cols; ++col) {
            const double value = static_cast<double>(h_input[col]);
            sum += value;
            sumsq += value * value;
        }
        const double mean = sum / cols;
        const double var = std::max(sumsq / cols - mean * mean, 0.0);
        const double inv_std = 1.0 / std::sqrt(var + kEps);
        for (int col : sample_cols) {
            const double norm =
                (static_cast<double>(h_input[col]) - mean) * inv_std;
            const double affine =
                norm * static_cast<double>(h_gamma[col]) +
                static_cast<double>(h_beta[col]);
            const float expected = static_cast<float>(affine);
            const float err = std::fabs(h_output[col] - expected);
            if (err > 2.0e-5f + 2.0e-5f * std::fabs(expected)) {
                std::ostringstream oss;
                oss << label << " verification failed row=" << row
                    << " col=" << col << " got=" << h_output[col]
                    << " ref=" << expected << " err=" << err;
                throw std::runtime_error(oss.str());
            }
        }
    }
}

void verify_selected(const Options& options,
                     const DeviceInfo& device,
                     const VariantResult& block,
                     const VariantResult& cluster,
                     const float* d_input,
                     const float* d_gamma,
                     const float* d_beta,
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
            benchmark_block_staged(one_run, device, d_input, d_gamma, d_beta,
                                   d_output, rows, cols);
        } else {
            benchmark_block_read(one_run, d_input, d_gamma, d_beta, d_output,
                                 rows, cols);
        }
        verify_output_samples(d_input, d_gamma, d_beta, d_output, rows, cols,
                              "block");
    }
    if (cluster.valid && options.profile_variant != "block") {
        benchmark_cluster(one_run, device, d_input, d_gamma, d_beta, d_output,
                          rows, cols);
        verify_output_samples(d_input, d_gamma, d_beta, d_output, rows, cols,
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
    float* d_input = nullptr;
    float* d_gamma = nullptr;
    float* d_beta = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, elements * sizeof(float)));

    const int init_threads = 256;
    init_input_kernel<<<(elements + init_threads - 1) / init_threads,
                        init_threads>>>(d_input, elements);
    init_input_kernel<<<(elements + init_threads - 1) / init_threads,
                        init_threads>>>(d_gamma, elements);
    init_input_kernel<<<(elements + init_threads - 1) / init_threads,
                        init_threads>>>(d_beta, elements);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    VariantResult block;
    VariantResult cluster;
    if (options.profile_variant != "cluster") {
        block = benchmark_best_block(options, device, d_input, d_gamma, d_beta,
                                     d_output, rows, cols);
    }
    if (options.profile_variant != "block") {
        cluster =
            benchmark_cluster(options, device, d_input, d_gamma, d_beta,
                              d_output, rows, cols);
    }
    verify_selected(options, device, block, cluster, d_input, d_gamma, d_beta,
                    d_output, rows, cols);
    print_result(cols, rows, block, cluster, device, options.csv);

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
