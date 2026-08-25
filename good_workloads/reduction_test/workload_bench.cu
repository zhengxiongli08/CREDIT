#include <cooperative_groups.h>
#include <math_constants.h>
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
#include <numeric>
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
                      << ": " << cudaGetErrorString(err__) << std::endl;       \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

namespace {

constexpr int kBlockThreads = 256;
constexpr int kDefaultWarmup = 3;
constexpr int kDefaultIters = 10;
constexpr int kDefaultTargetMiB = 128;
constexpr int kDefaultClusterSize = 4;
constexpr int kMaxClusterSize = 8;
constexpr float kRmsEps = 1e-5f;
constexpr std::array<int, 11> kColsToTest = {
    256,  512,   1024,   2048,   4096,   8192,
    16384, 32768, 65536, 131072, 262144,
};

enum class Workload {
    RmsNorm,
    Softmax,
    CrossEntropy,
};

struct Options {
    int warmup = kDefaultWarmup;
    int iters = kDefaultIters;
    int target_mib = kDefaultTargetMiB;
    int requested_cluster_size = kDefaultClusterSize;
    bool csv = false;
    bool verify = true;
    std::vector<Workload> workloads = {
        Workload::RmsNorm,
        Workload::Softmax,
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
};

struct SizeResult {
    int cols = 0;
    int rows = 0;
    double input_mib = 0.0;
    double block_gibs = std::numeric_limits<double>::quiet_NaN();
    double cluster_gibs = std::numeric_limits<double>::quiet_NaN();
    double cluster_vs_block = std::numeric_limits<double>::quiet_NaN();
};

struct WorkloadReport {
    Workload workload;
    std::vector<SizeResult> results;
};

Workload parse_workload_name(const std::string& name) {
    if (name == "rmsnorm") {
        return Workload::RmsNorm;
    }
    if (name == "softmax") {
        return Workload::Softmax;
    }
    if (name == "cross_entropy") {
        return Workload::CrossEntropy;
    }
    throw std::runtime_error("unknown workload: " + name);
}

std::string workload_name(Workload workload) {
    switch (workload) {
        case Workload::RmsNorm:
            return "rmsnorm";
        case Workload::Softmax:
            return "softmax";
        case Workload::CrossEntropy:
            return "cross_entropy";
    }
    return "unknown";
}

Options parse_options(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> int {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return std::stoi(argv[++i]);
        };
        if (arg == "--warmup") {
            opts.warmup = require_value("--warmup");
        } else if (arg == "--iters") {
            opts.iters = require_value("--iters");
        } else if (arg == "--target-mib") {
            opts.target_mib = require_value("--target-mib");
        } else if (arg == "--cluster-size") {
            opts.requested_cluster_size = require_value("--cluster-size");
        } else if (arg == "--csv") {
            opts.csv = true;
        } else if (arg == "--workload") {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for --workload");
            }
            const std::string value = argv[++i];
            if (value == "all") {
                opts.workloads = {
                    Workload::RmsNorm,
                    Workload::Softmax,
                    Workload::CrossEntropy,
                };
            } else {
                opts.workloads = {parse_workload_name(value)};
            }
        } else if (arg == "--no-verify") {
            opts.verify = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: workload_bench [--warmup N] [--iters N] "
                << "[--target-mib N] [--cluster-size N] "
                << "[--workload all|rmsnorm|softmax|cross_entropy] "
                << "[--csv] "
                << "[--no-verify]\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (opts.warmup < 0 || opts.iters <= 0 || opts.target_mib <= 0 ||
        opts.requested_cluster_size < 1) {
        throw std::runtime_error("all numeric arguments must be positive");
    }
    return opts;
}

double bytes_to_gib(std::size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

double bytes_to_mib(std::size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

std::string fmt_double(double value, int precision = 2) {
    if (std::isnan(value)) {
        return "n/a";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

__device__ inline float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ inline float warp_reduce_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

template <int BlockThreads>
__device__ float block_reduce_sum(float value) {
    __shared__ float warp_sums[BlockThreads / 32];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    value = warp_reduce_sum(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (threadIdx.x < BlockThreads / 32) {
        block_sum = warp_sums[lane];
    }
    if (warp == 0) {
        block_sum = warp_reduce_sum(block_sum);
    }
    return block_sum;
}

template <int BlockThreads>
__device__ float block_reduce_max(float value) {
    __shared__ float warp_max[BlockThreads / 32];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    value = warp_reduce_max(value);
    if (lane == 0) {
        warp_max[warp] = value;
    }
    __syncthreads();

    float block_max = -CUDART_INF_F;
    if (threadIdx.x < BlockThreads / 32) {
        block_max = warp_max[lane];
    }
    if (warp == 0) {
        block_max = warp_reduce_max(block_max);
    }
    return block_max;
}

template <int MaxClusterSize>
__device__ float cluster_allreduce_sum(cg::cluster_group cluster, float local_value) {
    __shared__ float slots[MaxClusterSize];
    __shared__ float reduced;

    const int block_rank = cluster.block_rank();
    const int blocks_per_cluster = cluster.dim_blocks().x;

    if (threadIdx.x == 0) {
        slots[block_rank] = local_value;
    }
    cluster.sync();

    if (threadIdx.x == 0) {
        for (int peer = 0; peer < blocks_per_cluster; ++peer) {
            if (peer == block_rank) {
                continue;
            }
            float* remote_slot = cluster.map_shared_rank(&slots[block_rank], peer);
            *remote_slot = local_value;
        }
    }
    cluster.sync();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < blocks_per_cluster; ++i) {
            total += slots[i];
        }
        reduced = total;
    }
    __syncthreads();
    return reduced;
}

template <int MaxClusterSize>
__device__ float cluster_allreduce_max(cg::cluster_group cluster, float local_value) {
    __shared__ float slots[MaxClusterSize];
    __shared__ float reduced;

    const int block_rank = cluster.block_rank();
    const int blocks_per_cluster = cluster.dim_blocks().x;

    if (threadIdx.x == 0) {
        slots[block_rank] = local_value;
    }
    cluster.sync();

    if (threadIdx.x == 0) {
        for (int peer = 0; peer < blocks_per_cluster; ++peer) {
            if (peer == block_rank) {
                continue;
            }
            float* remote_slot = cluster.map_shared_rank(&slots[block_rank], peer);
            *remote_slot = local_value;
        }
    }
    cluster.sync();

    if (threadIdx.x == 0) {
        float total = -CUDART_INF_F;
        for (int i = 0; i < blocks_per_cluster; ++i) {
            total = fmaxf(total, slots[i]);
        }
        reduced = total;
    }
    __syncthreads();
    return reduced;
}

template <int BlockThreads>
__global__ void block_rmsnorm_kernel(const float* input, const float* weight,
                                     float* output, int rows, int cols, float eps) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    __shared__ float inv_rms;
    const float* row_in = input + static_cast<std::size_t>(row) * cols;
    float* row_out = output + static_cast<std::size_t>(row) * cols;

    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        const float x = row_in[col];
        local_sum += x * x;
    }

    const float row_sum = block_reduce_sum<BlockThreads>(local_sum);
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(row_sum / cols + eps);
    }
    __syncthreads();

    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        row_out[col] = row_in[col] * inv_rms * weight[col];
    }
}

template <int BlockThreads, int MaxClusterSize>
__global__ void cluster_rmsnorm_kernel(const float* input, const float* weight,
                                       float* output, int rows, int cols,
                                       float eps) {
    cg::cluster_group cluster = cg::this_cluster();
    const int blocks_per_cluster = cluster.dim_blocks().x;
    const int block_rank = cluster.block_rank();
    const int row = blockIdx.x / blocks_per_cluster;
    if (row >= rows) {
        return;
    }

    __shared__ float inv_rms;
    const float* row_in = input + static_cast<std::size_t>(row) * cols;
    float* row_out = output + static_cast<std::size_t>(row) * cols;

    float local_sum = 0.0f;
    for (int col = block_rank * BlockThreads + threadIdx.x;
         col < cols; col += blocks_per_cluster * BlockThreads) {
        const float x = row_in[col];
        local_sum += x * x;
    }

    const float block_sum = block_reduce_sum<BlockThreads>(local_sum);
    const float row_sum = cluster_allreduce_sum<MaxClusterSize>(cluster, block_sum);

    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(row_sum / cols + eps);
    }
    __syncthreads();

    for (int col = block_rank * BlockThreads + threadIdx.x;
         col < cols; col += blocks_per_cluster * BlockThreads) {
        row_out[col] = row_in[col] * inv_rms * weight[col];
    }
}

template <int BlockThreads>
__global__ void block_softmax_kernel(const float* input, float* output, int rows,
                                     int cols) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    __shared__ float row_max;
    __shared__ float row_sum;

    const float* row_in = input + static_cast<std::size_t>(row) * cols;
    float* row_out = output + static_cast<std::size_t>(row) * cols;

    float local_max = -CUDART_INF_F;
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        local_max = fmaxf(local_max, row_in[col]);
    }
    const float block_max = block_reduce_max<BlockThreads>(local_max);
    if (threadIdx.x == 0) {
        row_max = block_max;
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        local_sum += __expf(row_in[col] - row_max);
    }
    const float block_sum = block_reduce_sum<BlockThreads>(local_sum);
    if (threadIdx.x == 0) {
        row_sum = block_sum;
    }
    __syncthreads();

    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        row_out[col] = __expf(row_in[col] - row_max) / row_sum;
    }
}

template <int BlockThreads, int MaxClusterSize>
__global__ void cluster_softmax_kernel(const float* input, float* output, int rows,
                                       int cols) {
    cg::cluster_group cluster = cg::this_cluster();
    const int blocks_per_cluster = cluster.dim_blocks().x;
    const int block_rank = cluster.block_rank();
    const int row = blockIdx.x / blocks_per_cluster;
    if (row >= rows) {
        return;
    }

    __shared__ float row_max;
    __shared__ float row_sum;

    const float* row_in = input + static_cast<std::size_t>(row) * cols;
    float* row_out = output + static_cast<std::size_t>(row) * cols;

    float local_max = -CUDART_INF_F;
    for (int col = block_rank * BlockThreads + threadIdx.x;
         col < cols; col += blocks_per_cluster * BlockThreads) {
        local_max = fmaxf(local_max, row_in[col]);
    }
    const float block_max = block_reduce_max<BlockThreads>(local_max);
    const float reduced_max = cluster_allreduce_max<MaxClusterSize>(cluster, block_max);
    if (threadIdx.x == 0) {
        row_max = reduced_max;
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int col = block_rank * BlockThreads + threadIdx.x;
         col < cols; col += blocks_per_cluster * BlockThreads) {
        local_sum += __expf(row_in[col] - row_max);
    }
    const float block_sum = block_reduce_sum<BlockThreads>(local_sum);
    const float reduced_sum = cluster_allreduce_sum<MaxClusterSize>(cluster, block_sum);
    if (threadIdx.x == 0) {
        row_sum = reduced_sum;
    }
    __syncthreads();

    for (int col = block_rank * BlockThreads + threadIdx.x;
         col < cols; col += blocks_per_cluster * BlockThreads) {
        row_out[col] = __expf(row_in[col] - row_max) / row_sum;
    }
}

template <int BlockThreads>
__global__ void block_cross_entropy_kernel(const float* logits, const int* targets,
                                           float* losses, int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    __shared__ float row_max;
    __shared__ float row_sum;
    __shared__ float target_logit;

    const float* row_logits = logits + static_cast<std::size_t>(row) * cols;
    const int target = targets[row] % cols;

    float local_max = -CUDART_INF_F;
    float local_target = -CUDART_INF_F;
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        const float value = row_logits[col];
        local_max = fmaxf(local_max, value);
        if (col == target) {
            local_target = value;
        }
    }

    const float block_max = block_reduce_max<BlockThreads>(local_max);
    const float block_target = block_reduce_max<BlockThreads>(local_target);
    if (threadIdx.x == 0) {
        row_max = block_max;
        target_logit = block_target;
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        local_sum += __expf(row_logits[col] - row_max);
    }

    const float block_sum = block_reduce_sum<BlockThreads>(local_sum);
    if (threadIdx.x == 0) {
        row_sum = block_sum;
        losses[row] = logf(row_sum) - (target_logit - row_max);
    }
}

template <int BlockThreads, int MaxClusterSize>
__global__ void cluster_cross_entropy_kernel(const float* logits, const int* targets,
                                             float* losses, int rows, int cols) {
    cg::cluster_group cluster = cg::this_cluster();
    const int blocks_per_cluster = cluster.dim_blocks().x;
    const int block_rank = cluster.block_rank();
    const int row = blockIdx.x / blocks_per_cluster;
    if (row >= rows) {
        return;
    }

    __shared__ float row_max;
    __shared__ float row_sum;
    __shared__ float global_target_logit;

    const float* row_logits = logits + static_cast<std::size_t>(row) * cols;
    const int target = targets[row] % cols;

    float local_max = -CUDART_INF_F;
    float local_target = -CUDART_INF_F;
    for (int col = block_rank * BlockThreads + threadIdx.x;
         col < cols; col += blocks_per_cluster * BlockThreads) {
        const float value = row_logits[col];
        local_max = fmaxf(local_max, value);
        if (col == target) {
            local_target = value;
        }
    }

    const float block_max = block_reduce_max<BlockThreads>(local_max);
    const float block_target = block_reduce_max<BlockThreads>(local_target);
    const float reduced_max = cluster_allreduce_max<MaxClusterSize>(cluster, block_max);
    const float reduced_target =
        cluster_allreduce_max<MaxClusterSize>(cluster, block_target);
    if (threadIdx.x == 0) {
        row_max = reduced_max;
        global_target_logit = reduced_target;
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int col = block_rank * BlockThreads + threadIdx.x;
         col < cols; col += blocks_per_cluster * BlockThreads) {
        local_sum += __expf(row_logits[col] - row_max);
    }

    const float block_sum = block_reduce_sum<BlockThreads>(local_sum);
    const float reduced_sum = cluster_allreduce_sum<MaxClusterSize>(cluster, block_sum);
    if (threadIdx.x == 0) {
        row_sum = reduced_sum;
        if (block_rank == 0) {
            losses[row] = logf(row_sum) - (global_target_logit - row_max);
        }
    }
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

void fill_logits(std::vector<float>& logits) {
    for (std::size_t i = 0; i < logits.size(); ++i) {
        logits[i] = static_cast<float>((static_cast<int>(i % 251) - 125) * 0.03125f);
    }
}

void fill_weights(std::vector<float>& weight) {
    for (std::size_t i = 0; i < weight.size(); ++i) {
        weight[i] = 0.75f + static_cast<float>(i % 31) * 0.015625f;
    }
}

void fill_targets(std::vector<int>& targets) {
    for (std::size_t i = 0; i < targets.size(); ++i) {
        targets[i] = static_cast<int>(i * 8191 + 17);
    }
}

std::vector<int> select_sample_rows(int rows) {
    std::vector<int> samples;
    if (rows <= 0) {
        return samples;
    }
    const std::array<int, 4> candidates = {0, rows / 3, (2 * rows) / 3, rows - 1};
    for (const int row : candidates) {
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
        const std::size_t row = static_cast<std::size_t>(sample_rows[i]);
        CUDA_CHECK(cudaMemcpy(host_samples.data() + i * cols,
                              d_output + row * cols,
                              static_cast<std::size_t>(cols) * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
}

void copy_sample_values(float* d_output, const std::vector<int>& sample_rows,
                        std::vector<float>& host_values) {
    host_values.resize(sample_rows.size());
    for (std::size_t i = 0; i < sample_rows.size(); ++i) {
        const std::size_t row = static_cast<std::size_t>(sample_rows[i]);
        CUDA_CHECK(cudaMemcpy(&host_values[i], d_output + row, sizeof(float),
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

void verify_rmsnorm_samples(const std::vector<float>& logits,
                            const std::vector<float>& weight,
                            const std::vector<int>& sample_rows,
                            const std::vector<float>& host_samples, int cols) {
    for (std::size_t sample_idx = 0; sample_idx < sample_rows.size(); ++sample_idx) {
        const int row = sample_rows[sample_idx];
        const std::size_t base = static_cast<std::size_t>(row) * cols;
        double sumsq = 0.0;
        for (int col = 0; col < cols; ++col) {
            const double x = logits[base + col];
            sumsq += x * x;
        }
        const double inv_rms = 1.0 / std::sqrt(sumsq / cols + kRmsEps);
        for (int col = 0; col < cols; ++col) {
            const double want = logits[base + col] * inv_rms * weight[col];
            const double got = host_samples[sample_idx * cols + col];
            check_close(got, want, 2e-4, 2e-4, "rmsnorm", row, col);
        }
    }
}

void verify_softmax_samples(const std::vector<float>& logits,
                            const std::vector<int>& sample_rows,
                            const std::vector<float>& host_samples, int cols) {
    for (std::size_t sample_idx = 0; sample_idx < sample_rows.size(); ++sample_idx) {
        const int row = sample_rows[sample_idx];
        const std::size_t base = static_cast<std::size_t>(row) * cols;
        double row_max = -std::numeric_limits<double>::infinity();
        for (int col = 0; col < cols; ++col) {
            row_max = std::max(row_max, static_cast<double>(logits[base + col]));
        }
        double row_sum = 0.0;
        for (int col = 0; col < cols; ++col) {
            row_sum += std::exp(static_cast<double>(logits[base + col]) - row_max);
        }
        for (int col = 0; col < cols; ++col) {
            const double want =
                std::exp(static_cast<double>(logits[base + col]) - row_max) / row_sum;
            const double got = host_samples[sample_idx * cols + col];
            check_close(got, want, 2e-5, 5e-4, "softmax", row, col);
        }
    }
}

void verify_cross_entropy_samples(const std::vector<float>& logits,
                                  const std::vector<int>& targets,
                                  const std::vector<int>& sample_rows,
                                  const std::vector<float>& host_values, int cols) {
    for (std::size_t sample_idx = 0; sample_idx < sample_rows.size(); ++sample_idx) {
        const int row = sample_rows[sample_idx];
        const std::size_t base = static_cast<std::size_t>(row) * cols;
        const int target = targets[row] % cols;
        double row_max = -std::numeric_limits<double>::infinity();
        for (int col = 0; col < cols; ++col) {
            row_max = std::max(row_max, static_cast<double>(logits[base + col]));
        }
        double row_sum = 0.0;
        for (int col = 0; col < cols; ++col) {
            row_sum += std::exp(static_cast<double>(logits[base + col]) - row_max);
        }
        const double want =
            std::log(row_sum) - (static_cast<double>(logits[base + target]) - row_max);
        const double got = host_values[sample_idx];
        check_close(got, want, 2e-4, 2e-4, "cross_entropy", row, -1);
    }
}

std::size_t modeled_bytes(Workload workload, std::size_t elements, int rows) {
    const std::size_t tensor_bytes = elements * sizeof(float);
    switch (workload) {
        case Workload::RmsNorm:
            return tensor_bytes * 4;
        case Workload::Softmax:
            return tensor_bytes * 4;
        case Workload::CrossEntropy:
            return tensor_bytes * 2 + static_cast<std::size_t>(rows) *
                                          (sizeof(float) + sizeof(int));
    }
    return 0;
}

double modeled_gibs(Workload workload, std::size_t elements, int rows, double avg_ms) {
    return bytes_to_gib(modeled_bytes(workload, elements, rows)) / (avg_ms / 1000.0);
}

DeviceInfo query_device_info(const Options& options) {
    DeviceInfo info;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    info.name = prop.name;
    info.major = prop.major;
    info.minor = prop.minor;

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
            &max_cluster_size, cluster_softmax_kernel<kBlockThreads, kMaxClusterSize>,
            &config));
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
        std::cout << "META,target_mib," << options.target_mib << "\n";
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

    std::cout << "Device: " << device.name << " (sm_" << device.major << device.minor
              << ")\n";
    std::cout << "Target input footprint per size: " << options.target_mib << " MiB\n";
    std::cout << "Warmup iters: " << options.warmup
              << ", timing iters: " << options.iters << "\n";
    std::cout << "Cluster launch supported: "
              << (device.cluster_launch ? "yes" : "no") << "\n";
    if (device.cluster_launch) {
        std::cout << "Requested cluster size: " << options.requested_cluster_size
                  << ", max potential cluster size: " << device.max_cluster_size
                  << ", chosen cluster size: " << device.chosen_cluster_size << "\n";
    }
    std::cout << "\n";
}

WorkloadReport benchmark_workload(
    Workload workload, const Options& options, const DeviceInfo& device,
    const std::vector<float>& host_logits, const std::vector<float>& host_weight,
    const std::vector<int>& host_targets, float* d_logits, float* d_weight,
    int* d_targets, float* d_output, std::size_t target_elements, int max_rows) {
    WorkloadReport report{workload, {}};
    report.results.reserve(kColsToTest.size());

    std::vector<float> host_samples;
    std::vector<float> host_values;

    for (const int cols : kColsToTest) {
        const int rows = std::max<int>(1, static_cast<int>(target_elements / cols));
        const std::size_t elements = static_cast<std::size_t>(rows) * cols;
        const std::vector<int> sample_rows =
            options.verify ? select_sample_rows(rows) : std::vector<int>{};

        SizeResult result;
        result.cols = cols;
        result.rows = rows;
        result.input_mib = bytes_to_mib(elements * sizeof(float));

        if (workload == Workload::RmsNorm) {
            auto launch_block = [&]() {
                block_rmsnorm_kernel<kBlockThreads>
                    <<<rows, kBlockThreads>>>(d_logits, d_weight, d_output, rows,
                                              cols, kRmsEps);
            };
            const double block_ms =
                time_kernel(launch_block, options.warmup, options.iters);
            if (options.verify) {
                copy_sample_rows(d_output, cols, sample_rows, host_samples);
                verify_rmsnorm_samples(host_logits, host_weight, sample_rows,
                                       host_samples, cols);
            }
            result.block_gibs = modeled_gibs(workload, elements, rows, block_ms);

            if (device.chosen_cluster_size >= 2) {
                auto launch_cluster = [&]() {
                    cudaLaunchConfig_t config{};
                    config.gridDim = dim3(rows * device.chosen_cluster_size);
                    config.blockDim = dim3(kBlockThreads);
                    config.dynamicSmemBytes = 0;

                    cudaLaunchAttribute attr{};
                    attr.id = cudaLaunchAttributeClusterDimension;
                    attr.val.clusterDim.x = device.chosen_cluster_size;
                    attr.val.clusterDim.y = 1;
                    attr.val.clusterDim.z = 1;
                    config.attrs = &attr;
                    config.numAttrs = 1;

                    CUDA_CHECK(cudaLaunchKernelEx(
                        &config, cluster_rmsnorm_kernel<kBlockThreads, kMaxClusterSize>,
                        d_logits, d_weight, d_output, rows, cols, kRmsEps));
                };
                const double cluster_ms =
                    time_kernel(launch_cluster, options.warmup, options.iters);
                if (options.verify) {
                    copy_sample_rows(d_output, cols, sample_rows, host_samples);
                    verify_rmsnorm_samples(host_logits, host_weight, sample_rows,
                                           host_samples, cols);
                }
                result.cluster_gibs =
                    modeled_gibs(workload, elements, rows, cluster_ms);
                result.cluster_vs_block = result.cluster_gibs / result.block_gibs;
            }
        } else if (workload == Workload::Softmax) {
            auto launch_block = [&]() {
                block_softmax_kernel<kBlockThreads>
                    <<<rows, kBlockThreads>>>(d_logits, d_output, rows, cols);
            };
            const double block_ms =
                time_kernel(launch_block, options.warmup, options.iters);
            if (options.verify) {
                copy_sample_rows(d_output, cols, sample_rows, host_samples);
                verify_softmax_samples(host_logits, sample_rows, host_samples, cols);
            }
            result.block_gibs = modeled_gibs(workload, elements, rows, block_ms);

            if (device.chosen_cluster_size >= 2) {
                auto launch_cluster = [&]() {
                    cudaLaunchConfig_t config{};
                    config.gridDim = dim3(rows * device.chosen_cluster_size);
                    config.blockDim = dim3(kBlockThreads);
                    config.dynamicSmemBytes = 0;

                    cudaLaunchAttribute attr{};
                    attr.id = cudaLaunchAttributeClusterDimension;
                    attr.val.clusterDim.x = device.chosen_cluster_size;
                    attr.val.clusterDim.y = 1;
                    attr.val.clusterDim.z = 1;
                    config.attrs = &attr;
                    config.numAttrs = 1;

                    CUDA_CHECK(cudaLaunchKernelEx(
                        &config, cluster_softmax_kernel<kBlockThreads, kMaxClusterSize>,
                        d_logits, d_output, rows, cols));
                };
                const double cluster_ms =
                    time_kernel(launch_cluster, options.warmup, options.iters);
                if (options.verify) {
                    copy_sample_rows(d_output, cols, sample_rows, host_samples);
                    verify_softmax_samples(host_logits, sample_rows, host_samples,
                                           cols);
                }
                result.cluster_gibs =
                    modeled_gibs(workload, elements, rows, cluster_ms);
                result.cluster_vs_block = result.cluster_gibs / result.block_gibs;
            }
        } else {
            auto launch_block = [&]() {
                block_cross_entropy_kernel<kBlockThreads>
                    <<<rows, kBlockThreads>>>(d_logits, d_targets, d_output, rows,
                                              cols);
            };
            const double block_ms =
                time_kernel(launch_block, options.warmup, options.iters);
            if (options.verify) {
                copy_sample_values(d_output, sample_rows, host_values);
                verify_cross_entropy_samples(host_logits, host_targets, sample_rows,
                                             host_values, cols);
            }
            result.block_gibs = modeled_gibs(workload, elements, rows, block_ms);

            if (device.chosen_cluster_size >= 2) {
                auto launch_cluster = [&]() {
                    cudaLaunchConfig_t config{};
                    config.gridDim = dim3(rows * device.chosen_cluster_size);
                    config.blockDim = dim3(kBlockThreads);
                    config.dynamicSmemBytes = 0;

                    cudaLaunchAttribute attr{};
                    attr.id = cudaLaunchAttributeClusterDimension;
                    attr.val.clusterDim.x = device.chosen_cluster_size;
                    attr.val.clusterDim.y = 1;
                    attr.val.clusterDim.z = 1;
                    config.attrs = &attr;
                    config.numAttrs = 1;

                    CUDA_CHECK(cudaLaunchKernelEx(
                        &config,
                        cluster_cross_entropy_kernel<kBlockThreads, kMaxClusterSize>,
                        d_logits, d_targets, d_output, rows, cols));
                };
                const double cluster_ms =
                    time_kernel(launch_cluster, options.warmup, options.iters);
                if (options.verify) {
                    copy_sample_values(d_output, sample_rows, host_values);
                    verify_cross_entropy_samples(host_logits, host_targets,
                                                 sample_rows, host_values, cols);
                }
                result.cluster_gibs =
                    modeled_gibs(workload, elements, rows, cluster_ms);
                result.cluster_vs_block = result.cluster_gibs / result.block_gibs;
            }
        }

        report.results.push_back(result);
    }

    return report;
}

void print_report(const WorkloadReport& report, bool cluster_supported, bool csv) {
    if (csv) {
        for (const SizeResult& result : report.results) {
            std::cout << "RESULT," << workload_name(report.workload) << ","
                      << result.cols << "," << result.rows << ","
                      << fmt_double(result.input_mib, 1) << ","
                      << fmt_double(result.block_gibs, 6) << ","
                      << fmt_double(result.cluster_gibs, 6) << ","
                      << fmt_double(result.cluster_vs_block, 6) << "\n";
        }
        return;
    }

    std::cout << "Workload: " << workload_name(report.workload) << "\n";
    std::cout << std::left << std::setw(9) << "cols" << std::setw(9) << "rows"
              << std::setw(12) << "input_MiB" << std::setw(13) << "block_GiB/s"
              << std::setw(15) << "cluster_GiB/s" << std::setw(16)
              << "cluster/block" << "\n";
    for (const SizeResult& result : report.results) {
        std::cout << std::left << std::setw(9) << result.cols << std::setw(9)
                  << result.rows << std::setw(12) << fmt_double(result.input_mib, 1)
                  << std::setw(13) << fmt_double(result.block_gibs, 2)
                  << std::setw(15) << fmt_double(result.cluster_gibs, 2)
                  << std::setw(16) << fmt_double(result.cluster_vs_block, 2) << "\n";
    }

    if (!cluster_supported) {
        std::cout << "Summary: cluster launch is unavailable, so DSMEM was not "
                     "benchmarked.\n\n";
        return;
    }

    int wins = 0;
    int large_wins = 0;
    int large_count = 0;
    double best_speedup = 0.0;
    int best_cols = 0;
    double large_avg_speedup = 0.0;

    for (const SizeResult& result : report.results) {
        if (std::isnan(result.cluster_vs_block)) {
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
            large_avg_speedup += result.cluster_vs_block;
            ++large_count;
            if (result.cluster_vs_block > 1.0) {
                ++large_wins;
            }
        }
    }

    if (large_count > 0) {
        large_avg_speedup /= large_count;
    }

    std::cout << "Summary: cluster beat block on " << wins << "/"
              << report.results.size() << " tested sizes. Best speedup was "
              << fmt_double(best_speedup, 2) << "x at cols=" << best_cols
              << ". Average cluster/block speedup for cols >= 65536 was "
              << fmt_double(large_avg_speedup, 2) << "x.\n";

    if (large_avg_speedup > 1.05) {
        std::cout << "Verdict: DSMEM helped for this workload once the reduced "
                     "dimension became large.\n\n";
    } else if (best_speedup > 1.05) {
        std::cout << "Verdict: DSMEM helped at select sizes, but not consistently "
                     "across the large-row regime.\n\n";
    } else {
        std::cout << "Verdict: DSMEM did not provide a clear benefit for this "
                     "workload in the tested configuration.\n\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

        CUDA_CHECK(cudaSetDevice(0));
        const DeviceInfo device = query_device_info(options);

        const std::size_t target_elements = std::max<std::size_t>(
            static_cast<std::size_t>(options.target_mib) * 1024 * 1024 / sizeof(float),
            static_cast<std::size_t>(kColsToTest.back()));
        const int max_rows = static_cast<int>(target_elements / kColsToTest.front());
        const int max_cols = kColsToTest.back();

        std::vector<float> host_logits(target_elements);
        std::vector<float> host_weight(max_cols);
        std::vector<int> host_targets(max_rows);
        fill_logits(host_logits);
        fill_weights(host_weight);
        fill_targets(host_targets);

        float* d_logits = nullptr;
        float* d_weight = nullptr;
        int* d_targets = nullptr;
        float* d_output = nullptr;

        CUDA_CHECK(cudaMalloc(&d_logits, host_logits.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weight, host_weight.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_targets, host_targets.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_output, host_logits.size() * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_logits, host_logits.data(),
                              host_logits.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weight, host_weight.data(),
                              host_weight.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_targets, host_targets.data(),
                              host_targets.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

        print_device_header(options, device);

        std::vector<WorkloadReport> reports;
        reports.reserve(options.workloads.size());
        for (const Workload workload : options.workloads) {
            reports.push_back(benchmark_workload(
                workload, options, device, host_logits, host_weight, host_targets,
                d_logits, d_weight, d_targets, d_output, target_elements, max_rows));
        }

        for (const WorkloadReport& report : reports) {
            print_report(report, device.chosen_cluster_size >= 2, options.csv);
        }

        CUDA_CHECK(cudaFree(d_logits));
        CUDA_CHECK(cudaFree(d_weight));
        CUDA_CHECK(cudaFree(d_targets));
        CUDA_CHECK(cudaFree(d_output));
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}
