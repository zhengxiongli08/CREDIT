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
constexpr int kTopK = 4;
constexpr int kInvalidIndex = 0x7fffffff;
constexpr std::array<int, 5> kDefaultNValues = {
    4096, 8192, 16384, 32768, 65536,
};

struct Options {
    int warmup = 2;
    int iters = 5;
    int rows = 8192;
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
};

struct VariantResult {
    bool valid = false;
    std::string name = "n/a";
    double avg_ms = std::numeric_limits<double>::quiet_NaN();
    double gbps = std::numeric_limits<double>::quiet_NaN();
};

struct TopK {
    float value[kTopK];
    int index[kTopK];
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
        } else if (arg == "--rows") {
            opts.rows = require_int("--rows");
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
                << "Usage: topk_bench [--csv] [--warmup N] [--iters N] "
                << "[--rows N] [--cluster-size N] [--n-values list] "
                << "[--profile-variant all|block|cluster] [--no-verify]\n";
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
    return static_cast<double>(rows) * cols * sizeof(float) +
           static_cast<double>(rows) * kTopK * (sizeof(float) + sizeof(int));
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

__host__ __device__ __forceinline__ bool candidate_better(float a_value,
                                                          int a_index,
                                                          float b_value,
                                                          int b_index) {
    return a_value > b_value || (a_value == b_value && a_index < b_index);
}

__host__ __device__ __forceinline__ TopK empty_topk() {
    TopK out;
    for (int i = 0; i < kTopK; ++i) {
        out.value[i] = -INFINITY;
        out.index[i] = kInvalidIndex;
    }
    return out;
}

__host__ __device__ __forceinline__ void insert_candidate(TopK& top,
                                                          float value,
                                                          int index) {
    if (!candidate_better(value, index, top.value[kTopK - 1],
                          top.index[kTopK - 1])) {
        return;
    }
    int pos = kTopK - 1;
    while (pos > 0 &&
           candidate_better(value, index, top.value[pos - 1], top.index[pos - 1])) {
        top.value[pos] = top.value[pos - 1];
        top.index[pos] = top.index[pos - 1];
        --pos;
    }
    top.value[pos] = value;
    top.index[pos] = index;
}

__device__ __forceinline__ TopK warp_reduce_topk(TopK top) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        TopK other;
        #pragma unroll
        for (int k = 0; k < kTopK; ++k) {
            other.value[k] =
                __shfl_xor_sync(0xffffffffu, top.value[k], offset, 32);
            other.index[k] =
                __shfl_xor_sync(0xffffffffu, top.index[k], offset, 32);
        }
        #pragma unroll
        for (int k = 0; k < kTopK; ++k) {
            insert_candidate(top, other.value[k], other.index[k]);
        }
    }
    return top;
}

template <int BlockThreads>
__device__ __forceinline__ TopK block_reduce_topk(TopK top) {
    constexpr int Warps = BlockThreads / 32;
    __shared__ float warp_values[Warps * kTopK];
    __shared__ int warp_indices[Warps * kTopK];
    __shared__ TopK broadcast;

    top = warp_reduce_topk(top);
    if ((threadIdx.x & 31) == 0) {
        const int warp = threadIdx.x >> 5;
        #pragma unroll
        for (int k = 0; k < kTopK; ++k) {
            warp_values[warp * kTopK + k] = top.value[k];
            warp_indices[warp * kTopK + k] = top.index[k];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        TopK reduced = empty_topk();
        for (int warp = 0; warp < Warps; ++warp) {
            #pragma unroll
            for (int k = 0; k < kTopK; ++k) {
                insert_candidate(reduced, warp_values[warp * kTopK + k],
                                 warp_indices[warp * kTopK + k]);
            }
        }
        broadcast = reduced;
    }
    __syncthreads();
    return broadcast;
}

template <int BlockThreads, int MaxClusterSize>
__device__ __forceinline__ TopK cluster_reduce_topk(cg::cluster_group cluster,
                                                     TopK block_top) {
    __shared__ float cluster_values[MaxClusterSize * kTopK];
    __shared__ int cluster_indices[MaxClusterSize * kTopK];
    __shared__ TopK broadcast;
    const int block_rank = cluster.block_rank();
    const int blocks_per_cluster = cluster.dim_blocks().x;

    if (threadIdx.x < blocks_per_cluster * kTopK) {
        const int target_rank = threadIdx.x / kTopK;
        const int k = threadIdx.x - target_rank * kTopK;
        const int slot = block_rank * kTopK + k;
        float* remote_value =
            cluster.map_shared_rank(&cluster_values[slot], target_rank);
        int* remote_index =
            cluster.map_shared_rank(&cluster_indices[slot], target_rank);
        *remote_value = block_top.value[k];
        *remote_index = block_top.index[k];
    }
    cluster.sync();

    if (threadIdx.x == 0) {
        TopK reduced = empty_topk();
        for (int rank = 0; rank < blocks_per_cluster; ++rank) {
            #pragma unroll
            for (int k = 0; k < kTopK; ++k) {
                insert_candidate(reduced, cluster_values[rank * kTopK + k],
                                 cluster_indices[rank * kTopK + k]);
            }
        }
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
    const float a = sinf(static_cast<float>(idx) * 0.00137f);
    const float b = cosf(static_cast<float>(idx) * 0.00029f);
    const float tiny = static_cast<float>(idx & 1023u) * 1.0e-7f;
    data[idx] = a + b + tiny;
}

template <int BlockThreads>
__global__ void block_topk_kernel(const float* __restrict__ input,
                                  float* __restrict__ out_values,
                                  int* __restrict__ out_indices,
                                  int rows,
                                  int cols) {
    const int row = blockIdx.x;
    const bool row_valid = row < rows;
    const std::size_t row_base = static_cast<std::size_t>(row) * cols;

    TopK local = empty_topk();
    for (int col = threadIdx.x; col < cols; col += BlockThreads) {
        if (!row_valid) {
            continue;
        }
        insert_candidate(local, input[row_base + col], col);
    }
    const TopK total = block_reduce_topk<BlockThreads>(local);
    if (row_valid && threadIdx.x == 0) {
        #pragma unroll
        for (int k = 0; k < kTopK; ++k) {
            out_values[row * kTopK + k] = total.value[k];
            out_indices[row * kTopK + k] = total.index[k];
        }
    }
}

template <int BlockThreads, int MaxClusterSize>
__global__ void cluster_topk_kernel(const float* __restrict__ input,
                                    float* __restrict__ out_values,
                                    int* __restrict__ out_indices,
                                    int rows,
                                    int cols,
                                    int slice_elems) {
    cg::cluster_group cluster = cg::this_cluster();
    const int blocks_per_cluster = cluster.dim_blocks().x;
    const int block_rank = cluster.block_rank();
    const int row = blockIdx.x / blocks_per_cluster;
    const bool row_valid = row < rows;
    const int slice_start = block_rank * slice_elems;
    const int local_cols =
        row_valid ? max(0, min(slice_elems, cols - slice_start)) : 0;
    const std::size_t row_base = static_cast<std::size_t>(row) * cols;

    TopK local = empty_topk();
    for (int local_col = threadIdx.x; local_col < local_cols;
         local_col += BlockThreads) {
        const int col = slice_start + local_col;
        insert_candidate(local, input[row_base + col], col);
    }
    const TopK block_top = block_reduce_topk<BlockThreads>(local);
    const TopK row_top =
        cluster_reduce_topk<BlockThreads, MaxClusterSize>(cluster, block_top);

    if (row_valid && block_rank == 0 && threadIdx.x == 0) {
        #pragma unroll
        for (int k = 0; k < kTopK; ++k) {
            out_values[row * kTopK + k] = row_top.value[k];
            out_indices[row * kTopK + k] = row_top.index[k];
        }
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
            cluster_topk_kernel<kBlockThreads, kMaxClusterSize>,
            &config));
        info.max_cluster_size = max_cluster_size;
        info.chosen_cluster_size =
            std::min({options.requested_cluster_size, max_cluster_size,
                      kMaxClusterSize});
    }
    return info;
}

VariantResult benchmark_block(const Options& options,
                              const float* d_input,
                              float* d_values,
                              int* d_indices,
                              int cols) {
    VariantResult result;
    result.valid = true;
    result.name = "block_top4";
    auto launch = [&]() {
        block_topk_kernel<kBlockThreads>
            <<<options.rows, kBlockThreads>>>(d_input, d_values, d_indices,
                                              options.rows, cols);
    };
    result.avg_ms = time_kernel(launch, options.warmup, options.iters);
    result.gbps = throughput_gbps(options.rows, cols, result.avg_ms);
    return result;
}

VariantResult benchmark_cluster(const Options& options,
                                const DeviceInfo& device,
                                const float* d_input,
                                float* d_values,
                                int* d_indices,
                                int cols) {
    VariantResult result;
    result.name = "cluster_top4";
    if (device.chosen_cluster_size < 2) {
        return result;
    }
    const int slice_elems =
        (cols + device.chosen_cluster_size - 1) / device.chosen_cluster_size;
    auto launch = [&]() {
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(options.rows * device.chosen_cluster_size);
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
            cluster_topk_kernel<kBlockThreads, kMaxClusterSize>,
            d_input, d_values, d_indices, options.rows, cols, slice_elems));
    };
    result.valid = true;
    result.avg_ms = time_kernel(launch, options.warmup, options.iters);
    result.gbps = throughput_gbps(options.rows, cols, result.avg_ms);
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
                           const float* d_values,
                           const int* d_indices,
                           int rows,
                           int cols,
                           const char* label) {
    const auto sample_rows = unique_samples({0, rows / 2, rows - 1}, rows);
    std::vector<float> h_input(cols);
    std::vector<float> h_values(rows * kTopK);
    std::vector<int> h_indices(rows * kTopK);
    CUDA_CHECK(cudaMemcpy(h_values.data(), d_values,
                          rows * kTopK * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_indices.data(), d_indices,
                          rows * kTopK * sizeof(int),
                          cudaMemcpyDeviceToHost));

    for (int row : sample_rows) {
        const std::size_t row_base = static_cast<std::size_t>(row) * cols;
        CUDA_CHECK(cudaMemcpy(h_input.data(), d_input + row_base,
                              cols * sizeof(float), cudaMemcpyDeviceToHost));
        TopK expected = empty_topk();
        for (int col = 0; col < cols; ++col) {
            insert_candidate(expected, h_input[col], col);
        }
        for (int k = 0; k < kTopK; ++k) {
            const int out_pos = row * kTopK + k;
            const double abs_err = std::abs(h_values[out_pos] - expected.value[k]);
            if (h_indices[out_pos] != expected.index[k] || abs_err > 0.0) {
                std::cerr << "verification failed for " << label
                          << ": row=" << row << " k=" << k
                          << " expected=(" << expected.value[k] << ","
                          << expected.index[k] << ") got=("
                          << h_values[out_pos] << "," << h_indices[out_pos]
                          << ")\n";
                std::exit(EXIT_FAILURE);
            }
        }
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

void print_csv_row(int cols,
                   int rows,
                   const VariantResult& block,
                   const VariantResult& cluster,
                   int cluster_size) {
    std::cout << "RESULT," << cols << "," << rows << "," << block.name << ","
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

void print_human_row(int cols,
                     int rows,
                     const VariantResult& block,
                     const VariantResult& cluster,
                     int cluster_size) {
    std::cout << "cols=" << cols << " rows=" << rows << "\n";
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

void print_device_summary(const DeviceInfo& device) {
    std::cout << "Device: " << device.name << " (SM " << device.major << "."
              << device.minor << ")\n";
    std::cout << "Cluster launch: " << (device.cluster_launch ? "yes" : "no")
              << ", max cluster size: " << device.max_cluster_size
              << ", chosen cluster size: " << device.chosen_cluster_size << "\n";
}

int run(int argc, char** argv) {
    const Options options = parse_options(argc, argv);
    const DeviceInfo device = query_device_info(options);

    if (!options.csv) {
        print_device_summary(device);
        std::cout << "Row-wise top-4 values/indices\n";
    }

    for (int cols : options.n_values) {
        const std::size_t elements = static_cast<std::size_t>(options.rows) * cols;
        float* d_input = nullptr;
        float* d_values = nullptr;
        int* d_indices = nullptr;
        CUDA_CHECK(cudaMalloc(&d_input, elements * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_values,
                              options.rows * kTopK * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_indices,
                              options.rows * kTopK * sizeof(int)));
        const int init_threads = 256;
        const int init_blocks =
            static_cast<int>((elements + init_threads - 1) / init_threads);
        init_input_kernel<<<init_blocks, init_threads>>>(d_input, elements);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        VariantResult block;
        VariantResult cluster;
        if (options.profile_variant != "cluster") {
            block = benchmark_block(options, d_input, d_values, d_indices, cols);
            if (options.verify && block.valid) {
                verify_output_samples(d_input, d_values, d_indices, options.rows,
                                      cols, block.name.c_str());
            }
        }
        if (options.profile_variant != "block") {
            cluster = benchmark_cluster(options, device, d_input, d_values,
                                        d_indices, cols);
            if (options.verify && cluster.valid) {
                verify_output_samples(d_input, d_values, d_indices, options.rows,
                                      cols, cluster.name.c_str());
            }
        }

        if (options.csv) {
            print_csv_row(cols, options.rows, block, cluster,
                          device.chosen_cluster_size);
        } else {
            print_human_row(cols, options.rows, block, cluster,
                            device.chosen_cluster_size);
        }

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_values));
        CUDA_CHECK(cudaFree(d_indices));
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
