#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
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
constexpr int kWarpsPerBlock = kBlockThreads / 32;
constexpr int kDefaultWarmup = 5;
constexpr int kDefaultIters = 25;
constexpr int kDefaultTargetMiB = 128;
constexpr int kDefaultClusterSize = 4;
constexpr std::array<int, 11> kColsToTest = {
    256,  512,   1024,   2048,   4096,   8192,
    16384, 32768, 65536, 131072, 262144,
};

struct Options {
    int warmup = kDefaultWarmup;
    int iters = kDefaultIters;
    int target_mib = kDefaultTargetMiB;
    int requested_cluster_size = kDefaultClusterSize;
    bool verify = true;
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
    double thread_gbps = std::numeric_limits<double>::quiet_NaN();
    double warp_gbps = std::numeric_limits<double>::quiet_NaN();
    double block_gbps = std::numeric_limits<double>::quiet_NaN();
    double cluster_gbps = std::numeric_limits<double>::quiet_NaN();
    double cluster_vs_block = std::numeric_limits<double>::quiet_NaN();
};

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
        } else if (arg == "--no-verify") {
            opts.verify = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: reduction_bench [--warmup N] [--iters N] "
                << "[--target-mib N] [--cluster-size N] [--no-verify]\n";
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

int round_down_power_of_two(int value) {
    int result = 1;
    while ((result << 1) <= value) {
        result <<= 1;
    }
    return result;
}

double bytes_to_gib(std::size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

double bytes_to_mib(std::size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

__device__ inline float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

template <int BlockThreads>
__device__ inline float block_reduce_sum(float value) {
    static_assert(BlockThreads % 32 == 0, "BlockThreads must be a multiple of 32");
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

__global__ void thread_row_sum_kernel(const float* input, float* output, int rows,
                                      int cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const float* row_ptr = input + static_cast<std::size_t>(row) * cols;
    float sum = 0.0f;
    for (int col = 0; col < cols; ++col) {
        sum += row_ptr[col];
    }
    output[row] = sum;
}

template <int WarpsPerBlock>
__global__ void warp_row_sum_kernel(const float* input, float* output, int rows,
                                    int cols) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * WarpsPerBlock + warp;
    if (row >= rows) {
        return;
    }

    const float* row_ptr = input + static_cast<std::size_t>(row) * cols;
    float sum = 0.0f;
    for (int col = lane; col < cols; col += 32) {
        sum += row_ptr[col];
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        output[row] = sum;
    }
}

template <int BlockThreads>
__global__ void block_row_sum_kernel(const float* input, float* output, int rows,
                                     int cols) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float* row_ptr = input + static_cast<std::size_t>(row) * cols;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        sum += row_ptr[col];
    }

    const float block_sum = block_reduce_sum<BlockThreads>(sum);
    if (threadIdx.x == 0) {
        output[row] = block_sum;
    }
}

template <int BlockThreads>
__global__ void cluster_row_sum_kernel(const float* input, float* output, int rows,
                                       int cols) {
    cg::cluster_group cluster = cg::this_cluster();
    __shared__ float partial_sum;

    const int blocks_per_cluster = cluster.dim_blocks().x;
    const int block_rank = cluster.block_rank();
    const int row = blockIdx.x / blocks_per_cluster;
    if (row >= rows) {
        return;
    }

    const float* row_ptr = input + static_cast<std::size_t>(row) * cols;
    float sum = 0.0f;
    for (int col = block_rank * BlockThreads + threadIdx.x;
         col < cols; col += blocks_per_cluster * BlockThreads) {
        sum += row_ptr[col];
    }

    const float block_sum = block_reduce_sum<BlockThreads>(sum);
    if (threadIdx.x == 0) {
        partial_sum = block_sum;
    }

    cluster.sync();

    if (block_rank == 0 && threadIdx.x == 0) {
        float total = 0.0f;
        for (int peer = 0; peer < blocks_per_cluster; ++peer) {
            float* peer_partial = cluster.map_shared_rank(&partial_sum, peer);
            total += *peer_partial;
        }
        output[row] = total;
    }

    cluster.sync();
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

void fill_input(std::vector<float>& values) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = 1.0f + static_cast<float>(i % 97) * 0.001f;
    }
}

void compute_reference(const std::vector<float>& input, int rows, int cols,
                       std::vector<double>& reference) {
    reference.assign(rows, 0.0);
    for (int row = 0; row < rows; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * cols;
        double sum = 0.0;
        for (int col = 0; col < cols; ++col) {
            sum += static_cast<double>(input[base + col]);
        }
        reference[row] = sum;
    }
}

void verify_output(const std::vector<float>& output,
                   const std::vector<double>& reference, int rows,
                   const std::string& label, int cols) {
    for (int row = 0; row < rows; ++row) {
        const double got = static_cast<double>(output[row]);
        const double want = reference[row];
        const double tol = 1e-4 * std::abs(want) + 1e-2;
        if (std::abs(got - want) > tol) {
            std::cerr << "verification failed for " << label << " at cols="
                      << cols << ", row=" << row << ": got=" << got
                      << ", want=" << want << ", tol=" << tol << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
}

double copy_and_verify(float* d_output, int rows, const std::vector<double>& reference,
                       std::vector<float>& host_output, const std::string& label,
                       int cols, bool verify) {
    host_output.resize(rows);
    CUDA_CHECK(cudaMemcpy(host_output.data(), d_output, rows * sizeof(float),
                          cudaMemcpyDeviceToHost));
    if (verify) {
        verify_output(host_output, reference, rows, label, cols);
    }
    return std::accumulate(host_output.begin(), host_output.end(), 0.0);
}

double effective_gbps(int rows, int cols, double avg_ms) {
    const std::size_t bytes =
        static_cast<std::size_t>(rows) * cols * sizeof(float) +
        static_cast<std::size_t>(rows) * sizeof(float);
    const double gib = bytes_to_gib(bytes);
    return gib / (avg_ms / 1000.0);
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
            &max_cluster_size, cluster_row_sum_kernel<kBlockThreads>, &config));
        info.max_cluster_size = max_cluster_size;

        const int capped = std::min(options.requested_cluster_size, max_cluster_size);
        info.chosen_cluster_size = (capped >= 2) ? round_down_power_of_two(capped) : 0;
    }

    return info;
}

std::string fmt_double(double value, int precision = 2) {
    if (std::isnan(value)) {
        return "n/a";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
    const Options options = parse_options(argc, argv);

    CUDA_CHECK(cudaSetDevice(0));
    const DeviceInfo device = query_device_info(options);

    const std::size_t target_elements = std::max<std::size_t>(
        static_cast<std::size_t>(options.target_mib) * 1024 * 1024 / sizeof(float),
        static_cast<std::size_t>(kColsToTest.back()));
    const int max_rows = static_cast<int>(target_elements / kColsToTest.front());

    std::vector<float> host_input(target_elements);
    std::vector<float> host_output;
    std::vector<double> reference;
    fill_input(host_input);

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, host_input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, static_cast<std::size_t>(max_rows) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, host_input.data(),
                            host_input.size() * sizeof(float),
                            cudaMemcpyHostToDevice));

    std::cout << "Device: " << device.name << " (sm_" << device.major
                << device.minor << ")\n";
    std::cout << "Target input footprint per size: " << options.target_mib
                << " MiB\n";
    std::cout << "Warmup iters: " << options.warmup
                << ", timing iters: " << options.iters << "\n";
    std::cout << "Cluster launch supported: "
                << (device.cluster_launch ? "yes" : "no") << "\n";
    if (device.cluster_launch) {
        std::cout << "Requested cluster size: " << options.requested_cluster_size
                    << ", max potential cluster size: " << device.max_cluster_size
                    << ", chosen cluster size: " << device.chosen_cluster_size
                    << "\n";
    }
    std::cout << "\n";

    std::vector<SizeResult> results;
    results.reserve(kColsToTest.size());

    for (const int cols : kColsToTest) {
        const int rows = std::max<int>(1, static_cast<int>(target_elements / cols));
        const std::size_t used_elements = static_cast<std::size_t>(rows) * cols;
        compute_reference(host_input, rows, cols, reference);

        SizeResult result;
        result.cols = cols;
        result.rows = rows;
        result.input_mib = bytes_to_mib(used_elements * sizeof(float));

        {
            auto launch = [&]() {
                const int grid = (rows + kBlockThreads - 1) / kBlockThreads;
                thread_row_sum_kernel<<<grid, kBlockThreads>>>(d_input, d_output,
                                                                rows, cols);
            };
            const double avg_ms =
                time_kernel(launch, options.warmup, options.iters);
            copy_and_verify(d_output, rows, reference, host_output, "thread",
                            cols, options.verify);
            result.thread_gbps = effective_gbps(rows, cols, avg_ms);
        }

        {
            auto launch = [&]() {
                const int grid =
                    (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
                warp_row_sum_kernel<kWarpsPerBlock>
                    <<<grid, kBlockThreads>>>(d_input, d_output, rows, cols);
            };
            const double avg_ms =
                time_kernel(launch, options.warmup, options.iters);
            copy_and_verify(d_output, rows, reference, host_output, "warp",
                            cols, options.verify);
            result.warp_gbps = effective_gbps(rows, cols, avg_ms);
        }

        {
            auto launch = [&]() {
                block_row_sum_kernel<kBlockThreads>
                    <<<rows, kBlockThreads>>>(d_input, d_output, rows, cols);
            };
            const double avg_ms =
                time_kernel(launch, options.warmup, options.iters);
            copy_and_verify(d_output, rows, reference, host_output, "block",
                            cols, options.verify);
            result.block_gbps = effective_gbps(rows, cols, avg_ms);
        }

        if (device.chosen_cluster_size >= 2) {
            auto launch = [&]() {
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
                    &config, cluster_row_sum_kernel<kBlockThreads>, d_input,
                    d_output, rows, cols));
            };
            const double avg_ms =
                time_kernel(launch, options.warmup, options.iters);
            copy_and_verify(d_output, rows, reference, host_output, "cluster",
                            cols, options.verify);
            result.cluster_gbps = effective_gbps(rows, cols, avg_ms);
            result.cluster_vs_block = result.cluster_gbps / result.block_gbps;
        }

        results.push_back(result);
    }

    std::cout << std::left << std::setw(9) << "cols" << std::setw(9) << "rows"
                << std::setw(12) << "input_MiB" << std::setw(13)
                << "thread_GiB/s" << std::setw(13) << "warp_GiB/s"
                << std::setw(13) << "block_GiB/s" << std::setw(15)
                << "cluster_GiB/s" << std::setw(16) << "cluster/block"
                << "\n";

    for (const SizeResult& result : results) {
        std::cout << std::left << std::setw(9) << result.cols
                    << std::setw(9) << result.rows << std::setw(12)
                    << fmt_double(result.input_mib, 1) << std::setw(13)
                    << fmt_double(result.thread_gbps, 2) << std::setw(13)
                    << fmt_double(result.warp_gbps, 2) << std::setw(13)
                    << fmt_double(result.block_gbps, 2) << std::setw(15)
                    << fmt_double(result.cluster_gbps, 2) << std::setw(16)
                    << fmt_double(result.cluster_vs_block, 2) << "\n";
    }

    std::cout << "\n";
    if (device.chosen_cluster_size < 2) {
        std::cout << "Summary: cluster launch is unavailable, so DSMEM was not "
                        "benchmarked.\n";
    } else {
        double best_speedup = 0.0;
        int best_cols = 0;
        int wins = 0;
        int large_wins = 0;
        double large_speedup_sum = 0.0;
        int large_count = 0;

        for (const SizeResult& result : results) {
            if (std::isnan(result.cluster_vs_block)) {
                continue;
            }
            if (result.cluster_vs_block > best_speedup) {
                best_speedup = result.cluster_vs_block;
                best_cols = result.cols;
            }
            if (result.cluster_vs_block > 1.0) {
                ++wins;
            }
            if (result.cols >= 65536) {
                large_speedup_sum += result.cluster_vs_block;
                ++large_count;
                if (result.cluster_vs_block > 1.0) {
                    ++large_wins;
                }
            }
        }

        const double large_avg_speedup =
            large_count > 0 ? large_speedup_sum / large_count : 0.0;

        std::cout << "Summary: cluster beat block on " << wins << "/"
                    << results.size() << " tested sizes. Best speedup was "
                    << fmt_double(best_speedup, 2) << "x at cols=" << best_cols
                    << ". Average cluster/block speedup for cols >= 65536 was "
                    << fmt_double(large_avg_speedup, 2) << "x.\n";

        if (large_avg_speedup > 1.05) {
            std::cout << "Verdict: DSMEM helped for the large-row cases in this "
                            "simple row-sum benchmark on this GPU.\n";
        } else if (best_speedup > 1.05) {
            std::cout << "Verdict: DSMEM only helped at select sizes here; the "
                            "benefit was not consistent across large rows.\n";
        } else {
            std::cout << "Verdict: DSMEM did not provide a clear benefit for this "
                            "simple one-pass sum reduction. That is consistent with "
                            "the fact that sum already needs only one global-memory "
                            "pass, unlike softmax-style multi-stage reductions.\n";
        }
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return 0;
}
