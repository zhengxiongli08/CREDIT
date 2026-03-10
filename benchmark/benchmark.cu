/*
RTX 5090 Official Specs
L1 cache: 128KB per SM
L2 cache: 96MB (L2 is shared across all SMs)
Global memory: 32GB
Global memory bandwidth: 1.79TB/s
*/

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK(x)                                                         \
    do {                                                                      \
        cudaError_t _e = (x);                                                 \
        if (_e != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d : %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_e));              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Explicit PTX loads let us control which cache hierarchy is targeted.
__device__ __forceinline__ int4 ld_global_ca_int4(const int4* addr) {
    int4 value;
    asm volatile(
        "ld.global.ca.v4.s32 {%0, %1, %2, %3}, [%4];"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "l"(addr)
        : "memory");
    return value;
}

__device__ __forceinline__ int4 ld_global_cg_int4(const int4* addr) {
    int4 value;
    asm volatile(
        "ld.global.cg.v4.s32 {%0, %1, %2, %3}, [%4];"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "l"(addr)
        : "memory");
    return value;
}

// Build a single cycle through the array for pointer-chasing latency tests.
void init_stride_cycle(int* data, int elements, int stride) {
    int cur = 0;
    for (int i = 0; i < elements; ++i) {
        data[cur] = (cur + stride) % elements;
        cur = data[cur];
    }
}

// Convert CUDA event time to device clock cycles.
double elapsed_cycles(float elapsed_ms, int clock_rate_khz) {
    return static_cast<double>(elapsed_ms) * static_cast<double>(clock_rate_khz);
}

// Pointer-chasing through an L1-resident array to measure serialized hit latency.
__global__ void read_latency_kernel(
    const int* __restrict__ data,
    int iterations,
    long long* __restrict__ result,
    int* __restrict__ sink)
{
    int ptr = 0;
    long long start = clock64();

    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile("ld.global.ca.s32 %0, [%1];" : "=r"(ptr) : "l"(&data[ptr]) : "memory");
    }

    long long end = clock64();
    *result = end - start;
    *sink = ptr;
}

// Pointer-chasing with .cg loads to bypass L1 and measure L2 hit latency.
__global__ void l2_read_latency_kernel(
    const int* __restrict__ data,
    int iterations,
    long long* __restrict__ result,
    int* __restrict__ sink)
{
    int ptr = 0;
    long long start = clock64();

    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(ptr) : "l"(&data[ptr]) : "memory");
    }

    long long end = clock64();
    *result = end - start;
    *sink = ptr;
}

// Pointer-chasing over a working set larger than L2 to approximate DRAM latency.
__global__ void dram_read_latency_kernel(
    const int* __restrict__ data,
    int iterations,
    long long* __restrict__ result,
    int* __restrict__ sink)
{
    int ptr = 0;
    long long start = clock64();

    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(ptr) : "l"(&data[ptr]) : "memory");
    }

    long long end = clock64();
    *result = end - start;
    *sink = ptr;
}

// Each block repeatedly scans a private L1-sized region to estimate per-SM L1 bandwidth.
__global__ void l1_bandwidth_kernel(
    const int4* __restrict__ data,
    int vectors_per_block,
    int repeats,
    int* __restrict__ sink)
{
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int4* block_base = data + static_cast<long long>(blockIdx.x) * vectors_per_block;
    int accum = 0;

    for (int rep = 0; rep < repeats; ++rep) {
        for (int idx = threadIdx.x; idx < vectors_per_block; idx += blockDim.x) {
            int4 value = ld_global_ca_int4(block_base + idx);
            accum += value.x + value.y + value.z + value.w;
        }
    }

    sink[global_tid] = accum;
}

// Stream through an L2-sized array with .cg loads to estimate aggregate L2 bandwidth.
__global__ void l2_bandwidth_kernel(
    const int4* __restrict__ data,
    long long total_vectors,
    int repeats,
    int* __restrict__ sink)
{
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int accum = 0;

    for (int rep = 0; rep < repeats; ++rep) {
        for (long long idx = global_tid; idx < total_vectors; idx += stride) {
            int4 value = ld_global_cg_int4(data + idx);
            accum += value.x + value.y + value.z + value.w;
        }
    }

    sink[global_tid] = accum;
}

// Simple read+write copy kernel to estimate sustained DRAM bandwidth.
__global__ void dram_bandwidth_kernel(
    const int4* __restrict__ src,
    int4* __restrict__ dst,
    long long total_vectors)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(gridDim.x) * blockDim.x;

    for (long long i = idx; i < total_vectors; i += stride) {
        dst[i] = src[i];
    }
}

// Inter-SM DSMEM microbenchmark for a fixed 2-block cluster.
//
// Block rank 0 acts as the requester and block rank 1 provides the remote
// shared memory backing store. The kernel measures four behaviors:
// 1) cluster.sync() overhead,
// 2) remote shared-memory writes (push bandwidth),
// 3) remote shared-memory reads (pull bandwidth),
// 4) serialized remote pointer-chase reads (pull latency).
//
// Note that remote write latency is intentionally not measured. DSMEM stores
// are fire-and-forget, so observing an individual store's completion requires
// an acknowledgment path or barrier, which would change the metric into a
// round-trip synchronization cost rather than a pure write latency.
__global__ void __cluster_dims__(2, 1, 1) cluster_bench(int* dump_sink) {
    // Form a two-block cluster and identify which block is local vs remote.
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int rank = cluster.block_rank();

    // payload_ints is the amount of DSMEM traffic per loop iteration.
    // bandwidth_iters increases the total bytes moved so timing noise is small.
    // latency_iters keeps the pointer chase long enough to average the result.
    constexpr int payload_ints = 1024;
    constexpr int bandwidth_iters = 1000;
    constexpr int latency_iters = 100000;

    // Each block exposes its local shared memory to the cluster. Rank 0 maps
    // rank 1's shared memory and rank 1 maps rank 0's shared memory, although
    // only rank 0 actively uses the remote mapping for measurement here.
    extern __shared__ int local_smem[];
    int* remote_smem = cluster.map_shared_rank(local_smem, 1 - rank);

    // Measure the cost of a single cluster-wide barrier.
    if (threadIdx.x == 0 && rank == 0) {
        long long start = clock64();
        cluster.sync();
        long long end = clock64();
        printf("----- Inter-SM DSMEM (clusterDim=2) -----\n");
        printf("Cluster sync overhead   : %lld cycles\n", end - start);
    } else {
        cluster.sync();
    }

    // Push benchmark: rank 0 writes directly into rank 1's shared memory.
    // The measured time includes the transfer plus the final cluster barrier
    // that ensures all remote writes are visible before reporting bandwidth.
    if (rank == 0) {
        long long start = clock64();

        #pragma unroll
        for (int iter = 0; iter < bandwidth_iters; ++iter) {
            if (threadIdx.x < payload_ints) {
                remote_smem[threadIdx.x] = iter;
            }
        }
        cluster.sync();
        long long end = clock64();

        if (threadIdx.x == 0) {
            double bytes = static_cast<double>(bandwidth_iters) * payload_ints * sizeof(int);
            printf("DSMEM write bandwidth   : %.2f B/cycle\n", bytes / static_cast<double>(end - start));
        }
    } else {
        // Rank 1 only participates in the barrier and keeps a side effect so
        // the compiler cannot treat the remote stores as dead.
        cluster.sync();
        if (threadIdx.x == 0) {
            dump_sink[0] = local_smem[0];
        }
    }

    // Prepare rank 1's local shared memory so rank 0 can measure remote reads.
    if (rank == 1 && threadIdx.x < payload_ints) {
        local_smem[threadIdx.x] = threadIdx.x;
    }
    cluster.sync();

    // Pull benchmark: rank 0 repeatedly reads rank 1's shared memory.
    // This measures sustained DSMEM read bandwidth rather than single-access
    // latency because many threads issue reads in parallel.
    if (rank == 0) {
        int thread_sum = 0;
        long long start = clock64();

        #pragma unroll
        for (int iter = 0; iter < bandwidth_iters; ++iter) {
            if (threadIdx.x < payload_ints) {
                thread_sum += remote_smem[threadIdx.x];
            }
        }

        cluster.sync();
        long long end = clock64();

        if (threadIdx.x == 0) {
            dump_sink[0] = thread_sum;
            double bytes = static_cast<double>(bandwidth_iters) * payload_ints * sizeof(int);
            printf("DSMEM read bandwidth    : %.2f B/cycle\n", bytes / static_cast<double>(end - start));
        }
    } else {
        cluster.sync();
    }

    // Set up a dependent pointer chain in remote shared memory. Each load
    // determines the next address, so the accesses cannot be pipelined and the
    // average cycles per iteration approximate remote DSMEM read latency.
    if (threadIdx.x == 0) {
        local_smem[0] = 0;
    }
    cluster.sync();

    if (rank == 0 && threadIdx.x == 0) {
        int ptr = 0;
        long long start = clock64();

        #pragma unroll 1
        for (int iter = 0; iter < latency_iters; ++iter) {
            ptr = remote_smem[ptr];
        }

        long long end = clock64();
        dump_sink[0] = ptr;
        printf("DSMEM read latency      : %.2f cycles\n", static_cast<double>(end - start) / latency_iters);
    } else {
        cluster.sync();
    }
}

int main() {
    // Working-set sizes are chosen to target L1, L2, and DRAM respectively.
    constexpr int L1_ELEMS = 1 << 13;
    constexpr int L2_ELEMS = 1 << 21;
    constexpr int DRAM_ELEMS = 1 << 26;
    constexpr int DRAM_STRIDE = L2_ELEMS + 1;
    constexpr int LATENCY_ITERS = 100000;
    constexpr int L1_BW_REPEATS = 4096;
    constexpr int L2_BW_REPEATS = 1024;
    constexpr int BW_BLOCK_SIZE = 256;
    constexpr int DSMEM_SHARED_BYTES = 1024 * sizeof(int);

    // Query device properties once so the same clock rate is reused for all reports.
    constexpr int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int clockRateKHz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, device));
    float clockGHz = clockRateKHz / 1e6f;

    printf("GPU Model: %s\n", prop.name);
    printf("Clock frequency: %.2f GHz\n", clockGHz);
    printf("SM count: %d\n", prop.multiProcessorCount);
    printf("===== Profiling ======\n");

    // Allocate and initialize the L1-resident pointer-chase array.
    const size_t l1Bytes = static_cast<size_t>(L1_ELEMS) * sizeof(int);
    const size_t l2Bytes = static_cast<size_t>(L2_ELEMS) * sizeof(int);
    const size_t dramBytes = static_cast<size_t>(DRAM_ELEMS) * sizeof(int);

    int* d_data = nullptr;
    int* d_sink = nullptr;
    long long* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, l1Bytes));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(int)));

    int* h_data = static_cast<int*>(malloc(l1Bytes));
    init_stride_cycle(h_data, L1_ELEMS, 2);
    CUDA_CHECK(cudaMemcpy(d_data, h_data, l1Bytes, cudaMemcpyHostToDevice));
    free(h_data);

    long long h_result = 0;

    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(long long)));
    CUDA_CHECK(cudaMemset(d_sink, 0, sizeof(int)));
    read_latency_kernel<<<1, 1>>>(d_data, LATENCY_ITERS, d_result, d_sink);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(long long), cudaMemcpyDeviceToHost));
    double l1Latency = static_cast<double>(h_result) / LATENCY_ITERS;

    // Repeat the same pattern with a larger array to force L2 hits.
    int* d_l2_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_l2_data, l2Bytes));
    int* h_l2 = static_cast<int*>(malloc(l2Bytes));
    init_stride_cycle(h_l2, L2_ELEMS, 2);
    CUDA_CHECK(cudaMemcpy(d_l2_data, h_l2, l2Bytes, cudaMemcpyHostToDevice));
    free(h_l2);

    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(long long)));
    CUDA_CHECK(cudaMemset(d_sink, 0, sizeof(int)));
    l2_read_latency_kernel<<<1, 1>>>(d_l2_data, LATENCY_ITERS, d_result, d_sink);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(long long), cudaMemcpyDeviceToHost));
    double l2Latency = static_cast<double>(h_result) / LATENCY_ITERS;

    // Use a working set larger than L2, plus an explicit cache flush, to bias
    // the pointer chase toward DRAM service.
    int* d_dram_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dram_data, dramBytes));
    int* h_dram = static_cast<int*>(malloc(dramBytes));
    init_stride_cycle(h_dram, DRAM_ELEMS, DRAM_STRIDE);
    CUDA_CHECK(cudaMemcpy(d_dram_data, h_dram, dramBytes, cudaMemcpyHostToDevice));
    free(h_dram);

    {
        constexpr size_t FLUSH_BYTES = 200ULL * 1024 * 1024;
        int* d_flush = nullptr;
        CUDA_CHECK(cudaMalloc(&d_flush, FLUSH_BYTES));
        CUDA_CHECK(cudaMemset(d_flush, 0, FLUSH_BYTES));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_flush));
    }

    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(long long)));
    CUDA_CHECK(cudaMemset(d_sink, 0, sizeof(int)));
    dram_read_latency_kernel<<<1, 1>>>(d_dram_data, LATENCY_ITERS, d_result, d_sink);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(long long), cudaMemcpyDeviceToHost));
    double dramLatency = static_cast<double>(h_result) / LATENCY_ITERS;

    printf("L1 read latency   : %.2f cycles (%.2f ns)\n", l1Latency, l1Latency / clockGHz);
    printf("L2 read latency   : %.2f cycles (%.2f ns)\n", l2Latency, l2Latency / clockGHz);
    printf("DRAM read latency : %.2f cycles (%.2f ns)\n", dramLatency, dramLatency / clockGHz);

    // Launch one block per SM, each block reusing its own small region so the
    // traffic mostly stays in the local L1 and can be normalized per SM.
    const int l1BwBlocks = prop.multiProcessorCount;
    const size_t l1BwTotalBytes = static_cast<size_t>(l1BwBlocks) * l1Bytes;
    const int l1VectorsPerBlock = L1_ELEMS / 4;
    int4* d_l1_bw_data = nullptr;
    int* d_l1_bw_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_l1_bw_data, l1BwTotalBytes));
    CUDA_CHECK(cudaMalloc(&d_l1_bw_sink, static_cast<size_t>(l1BwBlocks) * BW_BLOCK_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_l1_bw_data, 1, l1BwTotalBytes));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    l1_bandwidth_kernel<<<l1BwBlocks, BW_BLOCK_SIZE>>>(d_l1_bw_data, l1VectorsPerBlock, 8, d_l1_bw_sink);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    l1_bandwidth_kernel<<<l1BwBlocks, BW_BLOCK_SIZE>>>(
        d_l1_bw_data,
        l1VectorsPerBlock,
        L1_BW_REPEATS,
        d_l1_bw_sink);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float l1BwMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&l1BwMs, start, stop));
    double l1BwCycles = elapsed_cycles(l1BwMs, clockRateKHz);
    double l1Bandwidth =
        (static_cast<double>(l1BwTotalBytes) * L1_BW_REPEATS) /
        (l1BwCycles * l1BwBlocks);

    // Sweep a larger shared working set with .cg loads to estimate aggregate L2 throughput.
    const int l2BwBlocks = prop.multiProcessorCount * 8;
    const long long l2BwVectors = static_cast<long long>(l2Bytes / sizeof(int4));
    int4* d_l2_bw_data = nullptr;
    int* d_l2_bw_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_l2_bw_data, l2Bytes));
    CUDA_CHECK(cudaMalloc(&d_l2_bw_sink, static_cast<size_t>(l2BwBlocks) * BW_BLOCK_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_l2_bw_data, 2, l2Bytes));

    l2_bandwidth_kernel<<<l2BwBlocks, BW_BLOCK_SIZE>>>(d_l2_bw_data, l2BwVectors, 8, d_l2_bw_sink);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    l2_bandwidth_kernel<<<l2BwBlocks, BW_BLOCK_SIZE>>>(d_l2_bw_data, l2BwVectors, L2_BW_REPEATS, d_l2_bw_sink);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float l2BwMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&l2BwMs, start, stop));
    double l2BwCycles = elapsed_cycles(l2BwMs, clockRateKHz);
    double l2Bandwidth = (static_cast<double>(l2Bytes) * L2_BW_REPEATS) / l2BwCycles;

    // Use a read+write copy over a DRAM-sized buffer for sustained memory bandwidth.
    int4* d_dram_src = nullptr;
    int4* d_dram_dst = nullptr;
    const long long dramBwVectors = static_cast<long long>(dramBytes / sizeof(int4));
    const int dramBwBlocks = prop.multiProcessorCount * 8;
    CUDA_CHECK(cudaMalloc(&d_dram_src, dramBytes));
    CUDA_CHECK(cudaMalloc(&d_dram_dst, dramBytes));
    CUDA_CHECK(cudaMemset(d_dram_src, 3, dramBytes));

    dram_bandwidth_kernel<<<dramBwBlocks, BW_BLOCK_SIZE>>>(d_dram_src, d_dram_dst, dramBwVectors);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    dram_bandwidth_kernel<<<dramBwBlocks, BW_BLOCK_SIZE>>>(d_dram_src, d_dram_dst, dramBwVectors);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float dramBwMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&dramBwMs, start, stop));
    double dramBandwidth = (2.0 * static_cast<double>(dramBytes)) / (dramBwMs * 1e6);

    printf("L1 cache bandwidth      : %.2f B/cycle/SM\n", l1Bandwidth);
    printf("L2 cache bandwidth      : %.2f B/cycle\n", l2Bandwidth);
    printf("Global memory bandwidth : %.2f GB/s\n", dramBandwidth);

    // Launch a fixed 2-block cluster only on architectures that support DSMEM.
    // The kernel itself prints the inter-SM metrics because the measurements are
    // collected directly with clock64() inside the cluster execution context.
    if (prop.major >= 9) {
        CUDA_CHECK(cudaMemset(d_sink, 0, sizeof(int)));

        cudaLaunchConfig_t config = {};
        config.gridDim = dim3(2, 1, 1);
        config.blockDim = dim3(1024, 1, 1);
        config.dynamicSmemBytes = DSMEM_SHARED_BYTES;

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        config.attrs = attribute;
        config.numAttrs = 1;

        printf("DSMEM write latency     : not reported (remote stores are fire-and-forget, so a pure per-store latency needs a completion handshake that changes the metric)\n");
        CUDA_CHECK(cudaLaunchKernelEx(&config, cluster_bench, d_sink));
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        printf("Inter-SM DSMEM benchmarks skipped (requires Compute Capability 9.0+).\n");
    }

    printf("==================== SUMMARY ====================\n");
    printf("L1   read latency       : %8.2f cycles\n", l1Latency);
    printf("L2   read latency       : %8.2f cycles\n", l2Latency);
    printf("DRAM read latency       : %8.2f cycles\n", dramLatency);
    printf("L1   bandwidth          : %8.2f B/cycle/SM\n", l1Bandwidth);
    printf("L2   bandwidth          : %8.2f B/cycle\n", l2Bandwidth);
    printf("DRAM bandwidth          : %8.2f GB/s\n", dramBandwidth);
    printf("=================================================\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_sink));
    CUDA_CHECK(cudaFree(d_l2_data));
    CUDA_CHECK(cudaFree(d_dram_data));
    CUDA_CHECK(cudaFree(d_l1_bw_data));
    CUDA_CHECK(cudaFree(d_l1_bw_sink));
    CUDA_CHECK(cudaFree(d_l2_bw_data));
    CUDA_CHECK(cudaFree(d_l2_bw_sink));
    CUDA_CHECK(cudaFree(d_dram_src));
    CUDA_CHECK(cudaFree(d_dram_dst));

    return 0;
}
