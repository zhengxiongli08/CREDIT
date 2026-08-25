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
constexpr float kEps = 1e-5f;
constexpr std::array<int, 5> kDefaultNValues = {
    4096, 8192, 16384, 32768, 65536,
};

struct Options {
    int warmup = 2;
    int iters = 5;
    int rows = 2048;
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
                << "Usage: triplet_distance_backward_bench [--csv] [--warmup N] "
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
    return static_cast<double>(rows) * cols * sizeof(float) * 9.0;
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

__global__ void init_triplet_kernel(float* x,
                                    float* y,
                                    float* z,
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
    x[idx] = static_cast<float>(a & 2047u) * (1.0f / 1024.0f) - 1.0f;
    y[idx] = static_cast<float>(b & 2047u) * (1.0f / 1024.0f) - 1.0f;
    z[idx] = static_cast<float>(c & 2047u) * (1.0f / 1024.0f) - 1.0f;
}

__global__ void init_grad_kernel(float* grad, int rows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    grad[row] = 0.75f + static_cast<float>(row % 17) * 0.03125f;
}

template <int BlockThreads>
__global__ void block_bwd_read_kernel(const float* __restrict__ x,
                                      const float* __restrict__ y,
                                      const float* __restrict__ z,
                                      const float* __restrict__ grad,
                                      float* __restrict__ dx,
                                      float* __restrict__ dy,
                                      float* __restrict__ dz,
                                      int rows,
                                      int cols,
                                      float eps) {
    const int row = blockIdx.x;
    const bool row_valid = row < rows;
    const std::size_t row_base = static_cast<std::size_t>(row) * cols;

    Pair sums{0.0f, 0.0f};
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        const float xv = row_valid ? x[row_base + col] : 0.0f;
        const float yv = row_valid ? y[row_base + col] : 0.0f;
        const float zv = row_valid ? z[row_base + col] : 0.0f;
        const float diff_ap = xv - yv;
        const float diff_an = xv - zv;
        sums.a += diff_ap * diff_ap;
        sums.b += diff_an * diff_an;
    }
    const Pair row_sums = block_reduce_pair<BlockThreads>(sums);
    const float inv_ap = rsqrtf(row_sums.a + eps);
    const float inv_an = rsqrtf(row_sums.b + eps);
    const float grad_row = row_valid ? grad[row] : 0.0f;

    if (!row_valid) {
        return;
    }
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        const float xv = x[row_base + col];
        const float yv = y[row_base + col];
        const float zv = z[row_base + col];
        const float gap = (xv - yv) * inv_ap;
        const float gan = (xv - zv) * inv_an;
        dx[row_base + col] = grad_row * (gap - gan);
        dy[row_base + col] = -grad_row * gap;
        dz[row_base + col] = grad_row * gan;
    }
}

template <int BlockThreads>
__global__ void block_bwd_staged_kernel(const float* __restrict__ x,
                                        const float* __restrict__ y,
                                        const float* __restrict__ z,
                                        const float* __restrict__ grad,
                                        float* __restrict__ dx,
                                        float* __restrict__ dy,
                                        float* __restrict__ dz,
                                        int rows,
                                        int cols,
                                        float eps) {
    extern __shared__ float tile[];
    float* tile_x = tile;
    float* tile_y = tile + cols;
    float* tile_z = tile + 2 * cols;
    const int row = blockIdx.x;
    const bool row_valid = row < rows;
    const std::size_t row_base = static_cast<std::size_t>(row) * cols;

    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        tile_x[col] = row_valid ? x[row_base + col] : 0.0f;
        tile_y[col] = row_valid ? y[row_base + col] : 0.0f;
        tile_z[col] = row_valid ? z[row_base + col] : 0.0f;
    }
    __syncthreads();

    Pair sums{0.0f, 0.0f};
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        const float xv = tile_x[col];
        const float yv = tile_y[col];
        const float zv = tile_z[col];
        const float diff_ap = xv - yv;
        const float diff_an = xv - zv;
        sums.a += diff_ap * diff_ap;
        sums.b += diff_an * diff_an;
    }
    const Pair row_sums = block_reduce_pair<BlockThreads>(sums);
    const float inv_ap = rsqrtf(row_sums.a + eps);
    const float inv_an = rsqrtf(row_sums.b + eps);
    const float grad_row = row_valid ? grad[row] : 0.0f;

    if (!row_valid) {
        return;
    }
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        const float gap = (tile_x[col] - tile_y[col]) * inv_ap;
        const float gan = (tile_x[col] - tile_z[col]) * inv_an;
        dx[row_base + col] = grad_row * (gap - gan);
        dy[row_base + col] = -grad_row * gap;
        dz[row_base + col] = grad_row * gan;
    }
}

template <int BlockThreads, int MaxClusterSize>
__global__ void cluster_bwd_staged_kernel(const float* __restrict__ x,
                                          const float* __restrict__ y,
                                          const float* __restrict__ z,
                                          const float* __restrict__ grad,
                                          float* __restrict__ dx,
                                          float* __restrict__ dy,
                                          float* __restrict__ dz,
                                          int rows,
                                          int cols,
                                          float eps,
                                          int slice_elems) {
    extern __shared__ float tile[];
    float* tile_x = tile;
    float* tile_y = tile + slice_elems;
    float* tile_z = tile + 2 * slice_elems;
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
        tile_x[col] = x[row_base + slice_start + col];
        tile_y[col] = y[row_base + slice_start + col];
        tile_z[col] = z[row_base + slice_start + col];
    }
    __syncthreads();

    Pair sums{0.0f, 0.0f};
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        const float xv = tile_x[col];
        const float yv = tile_y[col];
        const float zv = tile_z[col];
        const float diff_ap = xv - yv;
        const float diff_an = xv - zv;
        sums.a += diff_ap * diff_ap;
        sums.b += diff_an * diff_an;
    }
    const Pair block_sums = block_reduce_pair<BlockThreads>(sums);
    const Pair row_sums =
        cluster_reduce_pair<BlockThreads, MaxClusterSize>(cluster, block_sums);
    const float inv_ap = rsqrtf(row_sums.a + eps);
    const float inv_an = rsqrtf(row_sums.b + eps);
    const float grad_row = row_valid ? grad[row] : 0.0f;

    if (!row_valid) {
        return;
    }
    for (int col = threadIdx.x; col < local_cols; col += BlockThreads) {
        const int global_col = slice_start + col;
        const float gap = (tile_x[col] - tile_y[col]) * inv_ap;
        const float gan = (tile_x[col] - tile_z[col]) * inv_an;
        dx[row_base + global_col] = grad_row * (gap - gan);
        dy[row_base + global_col] = -grad_row * gap;
        dz[row_base + global_col] = grad_row * gan;
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
            cluster_bwd_staged_kernel<kBlockThreads, kMaxClusterSize>,
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
                                   const float* d_y,
                                   const float* d_z,
                                   const float* d_grad,
                                   float* d_dx,
                                   float* d_dy,
                                   float* d_dz,
                                   int rows,
                                   int cols) {
    VariantResult result;
    result.valid = true;
    result.name = "block_read";
    auto launch = [&]() {
        block_bwd_read_kernel<kBlockThreads>
            <<<rows, kBlockThreads>>>(d_x, d_y, d_z, d_grad, d_dx, d_dy, d_dz,
                                      rows, cols, kEps);
    };
    result.avg_ms = time_kernel(launch, options.warmup, options.iters);
    result.gbps = throughput_gbps(rows, cols, result.avg_ms);
    return result;
}

VariantResult benchmark_block_staged(const Options& options,
                                     const DeviceInfo& device,
                                     const float* d_x,
                                     const float* d_y,
                                     const float* d_z,
                                     const float* d_grad,
                                     float* d_dx,
                                     float* d_dy,
                                     float* d_dz,
                                     int rows,
                                     int cols) {
    VariantResult result;
    const std::size_t smem_bytes =
        static_cast<std::size_t>(cols) * sizeof(float) * 3;
    if (smem_bytes > static_cast<std::size_t>(device.max_dynamic_smem_per_block)) {
        return result;
    }
    CUDA_CHECK(cudaFuncSetAttribute(
        block_bwd_staged_kernel<kBlockThreads>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
    result.valid = true;
    result.name = "block_smem";
    auto launch = [&]() {
        block_bwd_staged_kernel<kBlockThreads>
            <<<rows, kBlockThreads, smem_bytes>>>(d_x, d_y, d_z, d_grad, d_dx,
                                                  d_dy, d_dz, rows, cols, kEps);
    };
    result.avg_ms = time_kernel(launch, options.warmup, options.iters);
    result.gbps = throughput_gbps(rows, cols, result.avg_ms);
    return result;
}

VariantResult benchmark_best_block(const Options& options,
                                   const DeviceInfo& device,
                                   const float* d_x,
                                   const float* d_y,
                                   const float* d_z,
                                   const float* d_grad,
                                   float* d_dx,
                                   float* d_dy,
                                   float* d_dz,
                                   int rows,
                                   int cols) {
    VariantResult best;
    best = pick_better(best,
                       benchmark_block_read(options, d_x, d_y, d_z, d_grad,
                                            d_dx, d_dy, d_dz, rows, cols));
    best = pick_better(best,
                       benchmark_block_staged(options, device, d_x, d_y,
                                              d_z, d_grad, d_dx, d_dy, d_dz,
                                              rows, cols));
    return best;
}

VariantResult benchmark_cluster(const Options& options,
                                const DeviceInfo& device,
                                const float* d_x,
                                const float* d_y,
                                const float* d_z,
                                const float* d_grad,
                                float* d_dx,
                                float* d_dy,
                                float* d_dz,
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
        cluster_bwd_staged_kernel<kBlockThreads, kMaxClusterSize>,
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
            cluster_bwd_staged_kernel<kBlockThreads, kMaxClusterSize>,
            d_x, d_y, d_z, d_grad, d_dx, d_dy, d_dz, rows, cols, kEps,
            slice_elems));
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

void verify_output_samples(const float* d_x,
                           const float* d_y,
                           const float* d_z,
                           const float* d_grad,
                           const float* d_dx,
                           const float* d_dy,
                           const float* d_dz,
                           int rows,
                           int cols,
                           const char* label) {
    const auto sample_rows = unique_samples({0, rows / 2, rows - 1}, rows);
    const auto sample_cols = unique_samples({0, cols / 7, cols / 2, cols - 1}, cols);
    std::vector<float> h_x(cols);
    std::vector<float> h_y(cols);
    std::vector<float> h_z(cols);
    std::vector<float> h_grad(rows);
    CUDA_CHECK(cudaMemcpy(h_grad.data(), d_grad,
                          static_cast<std::size_t>(rows) * sizeof(float),
                          cudaMemcpyDeviceToHost));

    for (int row : sample_rows) {
        const std::size_t row_base = static_cast<std::size_t>(row) * cols;
        CUDA_CHECK(cudaMemcpy(h_x.data(), d_x + row_base,
                              static_cast<std::size_t>(cols) * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y + row_base,
                              static_cast<std::size_t>(cols) * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_z.data(), d_z + row_base,
                              static_cast<std::size_t>(cols) * sizeof(float),
                              cudaMemcpyDeviceToHost));

        double sum_ap = 0.0;
        double sum_an = 0.0;
        for (int col = 0; col < cols; ++col) {
            const double xv = static_cast<double>(h_x[col]);
            const double yv = static_cast<double>(h_y[col]);
            const double zv = static_cast<double>(h_z[col]);
            const double diff_ap = xv - yv;
            const double diff_an = xv - zv;
            sum_ap += diff_ap * diff_ap;
            sum_an += diff_an * diff_an;
        }
        const double inv_ap = 1.0 / std::sqrt(sum_ap + kEps);
        const double inv_an = 1.0 / std::sqrt(sum_an + kEps);
        const double grad = static_cast<double>(h_grad[row]);

        for (int col : sample_cols) {
            const double xv = static_cast<double>(h_x[col]);
            const double yv = static_cast<double>(h_y[col]);
            const double zv = static_cast<double>(h_z[col]);
            const double gap = (xv - yv) * inv_ap;
            const double gan = (xv - zv) * inv_an;
            const double ref_dx = grad * (gap - gan);
            const double ref_dy = -grad * gap;
            const double ref_dz = grad * gan;
            float got_dx = 0.0f;
            float got_dy = 0.0f;
            float got_dz = 0.0f;
            CUDA_CHECK(cudaMemcpy(&got_dx, d_dx + row_base + col, sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&got_dy, d_dy + row_base + col, sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&got_dz, d_dz + row_base + col, sizeof(float),
                                  cudaMemcpyDeviceToHost));
            const double err_dx = std::abs(static_cast<double>(got_dx) - ref_dx);
            const double err_dy = std::abs(static_cast<double>(got_dy) - ref_dy);
            const double err_dz = std::abs(static_cast<double>(got_dz) - ref_dz);
            if (err_dx > 2e-2 + 2e-2 * std::abs(ref_dx) ||
                err_dy > 2e-2 + 2e-2 * std::abs(ref_dy) ||
                err_dz > 2e-2 + 2e-2 * std::abs(ref_dz)) {
                std::ostringstream oss;
                oss << label << " verification failed row=" << row
                    << " col=" << col << " got_dx=" << got_dx
                    << " ref_dx=" << ref_dx << " err_dx=" << err_dx
                    << " got_dy=" << got_dy << " ref_dy=" << ref_dy
                    << " err_dy=" << err_dy << " got_dz=" << got_dz
                    << " ref_dz=" << ref_dz << " err_dz=" << err_dz;
                throw std::runtime_error(oss.str());
            }
        }
    }
}

void verify_selected(const Options& options,
                     const DeviceInfo& device,
                     const VariantResult& block,
                     const VariantResult& cluster,
                     const float* d_x,
                     const float* d_y,
                     const float* d_z,
                     const float* d_grad,
                     float* d_dx,
                     float* d_out_dy,
                     float* d_out_dz,
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
            benchmark_block_staged(one_run, device, d_x, d_y, d_z, d_grad,
                                   d_dx, d_out_dy, d_out_dz, rows, cols);
        } else {
            benchmark_block_read(one_run, d_x, d_y, d_z, d_grad, d_dx,
                                 d_out_dy, d_out_dz, rows, cols);
        }
        verify_output_samples(d_x, d_y, d_z, d_grad, d_dx, d_out_dy, d_out_dz,
                              rows, cols, "block");
    }
    if (cluster.valid && options.profile_variant != "block") {
        benchmark_cluster(one_run, device, d_x, d_y, d_z, d_grad, d_dx,
                          d_out_dy, d_out_dz, rows, cols);
        verify_output_samples(d_x, d_y, d_z, d_grad, d_dx, d_out_dy, d_out_dz,
                              rows, cols, "cluster");
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
    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_z = nullptr;
    float* d_grad = nullptr;
    float* d_dy = nullptr;
    float* d_dx = nullptr;
    float* d_dz = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z, elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad, static_cast<std::size_t>(rows) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dy, elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dx, elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dz, elements * sizeof(float)));

    const int init_threads = 256;
    init_triplet_kernel<<<(elements + init_threads - 1) / init_threads,
                          init_threads>>>(d_x, d_y, d_z, elements);
    init_grad_kernel<<<(rows + init_threads - 1) / init_threads, init_threads>>>(
        d_grad, rows);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    VariantResult block;
    VariantResult cluster;
    if (options.profile_variant != "cluster") {
        block = benchmark_best_block(options, device, d_x, d_y, d_z, d_grad,
                                     d_dx, d_dy, d_dz, rows, cols);
    }
    if (options.profile_variant != "block") {
        cluster = benchmark_cluster(options, device, d_x, d_y, d_z, d_grad,
                                    d_dx, d_dy, d_dz, rows, cols);
    }
    verify_selected(options, device, block, cluster, d_x, d_y, d_z, d_grad,
                    d_dx, d_dy, d_dz, rows, cols);
    print_result(cols, rows, block, cluster, device, options.csv);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_grad));
    CUDA_CHECK(cudaFree(d_dy));
    CUDA_CHECK(cudaFree(d_dx));
    CUDA_CHECK(cudaFree(d_dz));
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
