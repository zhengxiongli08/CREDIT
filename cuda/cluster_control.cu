#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error__ = (call);                                            \
        if (error__ != cudaSuccess) {                                            \
            throw std::runtime_error(std::string(#call) + ": " +               \
                                     cudaGetErrorString(error__));              \
        }                                                                       \
    } while (0)

struct Options {
    int rows = 4096;
    int cluster_size = 8;
    int barriers = 4;
    int shared_bytes = 16384;
    int warmup = 20;
    int iterations = 100;
};

int parse_positive(const char* flag, const char* value) {
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0 || parsed > (1L << 30)) {
        throw std::runtime_error(std::string("invalid ") + flag + ": " + value);
    }
    return static_cast<int>(parsed);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc) {
            throw std::runtime_error(std::string("missing value for ") + argv[i]);
        }
        const std::string flag = argv[i++];
        const int value = parse_positive(flag.c_str(), argv[i]);
        if (flag == "--rows") {
            options.rows = value;
        } else if (flag == "--cluster-size") {
            options.cluster_size = value;
        } else if (flag == "--barriers") {
            options.barriers = value;
        } else if (flag == "--shared-bytes") {
            options.shared_bytes = value;
        } else if (flag == "--warmup") {
            options.warmup = value;
        } else if (flag == "--iters") {
            options.iterations = value;
        } else {
            throw std::runtime_error("unknown option: " + flag);
        }
    }
    if (options.cluster_size != 2 && options.cluster_size != 4 &&
        options.cluster_size != 8) {
        throw std::runtime_error("cluster size must be 2, 4, or 8");
    }
    return options;
}

__global__ void block_control_kernel(unsigned int* sink) {
    if (threadIdx.x == 0) {
        sink[blockIdx.x] = static_cast<unsigned int>(blockIdx.x);
    }
}

__global__ void cluster_control_kernel(
    unsigned int* sink, int barrier_count, int shared_bytes) {
    extern __shared__ unsigned char shared[];
    cg::cluster_group cluster = cg::this_cluster();
    if (threadIdx.x == 0 && shared_bytes > 0) {
        shared[0] = static_cast<unsigned char>(cluster.block_rank());
    }
    for (int i = 0; i < barrier_count; ++i) {
        cluster.sync();
    }
    if (threadIdx.x == 0) {
        sink[blockIdx.x] = static_cast<unsigned int>(shared[0]) +
                           static_cast<unsigned int>(barrier_count);
    }
}

template <typename Launch>
double time_launch(Launch&& launch, int warmup, int iterations) {
    for (int i = 0; i < warmup; ++i) {
        launch();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return static_cast<double>(elapsed_ms) / iterations;
}

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        if (!properties.clusterLaunch) {
            throw std::runtime_error("device does not support cluster launch");
        }
        if (options.shared_bytes > properties.sharedMemPerBlockOptin) {
            std::cout << "UNSUPPORTED," << options.rows << ','
                      << options.cluster_size << ',' << options.barriers << ','
                      << options.shared_bytes << ",shared-memory-limit\n";
            return 0;
        }

        CUDA_CHECK(cudaFuncSetAttribute(
            cluster_control_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            options.shared_bytes));
        const int cluster_blocks = options.rows * options.cluster_size;
        unsigned int* sink = nullptr;
        CUDA_CHECK(cudaMalloc(
            &sink, static_cast<size_t>(cluster_blocks) * sizeof(unsigned int)));

        const auto launch_block = [&]() {
            block_control_kernel<<<options.rows, 256>>>(sink);
            CUDA_CHECK(cudaGetLastError());
        };
        const auto launch_cluster = [&]() {
            cudaLaunchConfig_t config{};
            config.gridDim = dim3(cluster_blocks);
            config.blockDim = dim3(256);
            config.dynamicSmemBytes = static_cast<unsigned int>(options.shared_bytes);
            cudaLaunchAttribute attribute{};
            attribute.id = cudaLaunchAttributeClusterDimension;
            attribute.val.clusterDim.x = options.cluster_size;
            attribute.val.clusterDim.y = 1;
            attribute.val.clusterDim.z = 1;
            config.attrs = &attribute;
            config.numAttrs = 1;
            CUDA_CHECK(cudaLaunchKernelEx(
                &config,
                cluster_control_kernel,
                sink,
                options.barriers,
                options.shared_bytes));
        };

        const double block_ms = time_launch(
            launch_block, options.warmup, options.iterations);
        const double cluster_ms = time_launch(
            launch_cluster, options.warmup, options.iterations);
        CUDA_CHECK(cudaFree(sink));

        std::cout << "CONTROL," << options.rows << ',' << options.cluster_size
                  << ',' << options.barriers << ',' << options.shared_bytes
                  << ',' << block_ms << ',' << cluster_ms << ','
                  << (cluster_ms - block_ms) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
