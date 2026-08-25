// Portable L1/L2/DRAM and DSMEM microbenchmarks for Compute Capability 9.0+.
// The Modal launcher compiles this source for sm_90 and executes it on an H100.

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

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

constexpr int DSMEM_BLOCK_SIZE = 256;
constexpr int DSMEM_PAYLOAD_WORDS = 4096;
constexpr int DSMEM_TRIALS = 15;
constexpr int DSMEM_SYNC_ITERS = 1000;
constexpr int DSMEM_LATENCY_ITERS = 20000;
constexpr int DSMEM_LATENCY_WARMUP = 4096;
constexpr int DSMEM_THROUGHPUT_ITERS = 4096;
constexpr int DSMEM_VISIBILITY_ITERS = 1000;
constexpr int DSMEM_VISIBILITY_WARMUP = 100;
constexpr int DSMEM_WARPS = DSMEM_BLOCK_SIZE / 32;

enum DsmemMetric {
    DSMEM_METRIC_SYNC = 0,
    DSMEM_METRIC_READ_LATENCY,
    DSMEM_METRIC_THROUGHPUT,
    DSMEM_METRIC_VISIBILITY,
    DSMEM_METRIC_COUNT
};

struct DsmemResults {
    unsigned long long sync_cycles[DSMEM_TRIALS];
    unsigned long long local_read_latency_cycles[DSMEM_TRIALS];
    unsigned long long remote_read_latency_cycles[DSMEM_TRIALS];
    unsigned long long local_read_throughput_cycles[DSMEM_TRIALS];
    unsigned long long remote_read_throughput_cycles[DSMEM_TRIALS];
    unsigned long long local_store_throughput_cycles[DSMEM_TRIALS];
    unsigned long long remote_store_throughput_cycles[DSMEM_TRIALS];
    unsigned long long visibility_roundtrip_cycles[DSMEM_TRIALS];
    unsigned int sm_id[DSMEM_METRIC_COUNT][2];
};

__device__ __forceinline__ unsigned int current_smid() {
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

__device__ __forceinline__ int dsmem_load(const int* address) {
    int value;
    asm volatile("ld.u32 %0, [%1];"
                 : "=r"(value)
                 : "l"(address)
                 : "memory");
    return value;
}

__device__ __forceinline__ void dsmem_store(int* address, int value) {
    asm volatile("st.u32 [%0], %1;"
                 :
                 : "l"(address), "r"(value)
                 : "memory");
}

__device__ __forceinline__ void record_cluster_placement(
    DsmemResults* results,
    int metric,
    unsigned int rank)
{
    if (threadIdx.x == 0) {
        results->sm_id[metric][rank] = current_smid();
    }
}

__global__ void __cluster_dims__(2, 1, 1) dsmem_sync_bench(DsmemResults* results) {
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int rank = cluster.block_rank();
    extern __shared__ int local_smem[];

    if (threadIdx.x == 0) {
        local_smem[0] = 0;
    }
    record_cluster_placement(results, DSMEM_METRIC_SYNC, rank);

    #pragma unroll 1
    for (int i = 0; i < 100; ++i) {
        cluster.sync();
    }

    for (int trial = 0; trial < DSMEM_TRIALS; ++trial) {
        cluster.sync();
        unsigned long long start = 0;
        if (rank == 0 && threadIdx.x == 0) {
            start = clock64();
        }

        #pragma unroll 1
        for (int i = 0; i < DSMEM_SYNC_ITERS; ++i) {
            cluster.sync();
        }

        if (rank == 0 && threadIdx.x == 0) {
            results->sync_cycles[trial] = clock64() - start;
        }
    }
}

__global__ void __cluster_dims__(2, 1, 1) dsmem_read_latency_bench(
    DsmemResults* results,
    int* dump_sink)
{
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int rank = cluster.block_rank();
    extern __shared__ int local_storage[];
    volatile int* local_smem = local_storage;
    int* remote_smem = cluster.map_shared_rank(local_storage, 1 - rank);

    for (int i = threadIdx.x; i < DSMEM_PAYLOAD_WORDS; i += blockDim.x) {
        local_storage[i] = (i + 257) & (DSMEM_PAYLOAD_WORDS - 1);
    }
    record_cluster_placement(results, DSMEM_METRIC_READ_LATENCY, rank);
    cluster.sync();

    if (rank == 0 && threadIdx.x == 0) {
        int ptr = 0;
        #pragma unroll 1
        for (int i = 0; i < DSMEM_LATENCY_WARMUP; ++i) {
            ptr = local_smem[ptr];
        }
        dump_sink[0] = ptr;
    }
    cluster.sync();

    for (int trial = 0; trial < DSMEM_TRIALS; ++trial) {
        if (rank == 0 && threadIdx.x == 0) {
            int ptr = 0;
            unsigned long long start = clock64();
            #pragma unroll 1
            for (int i = 0; i < DSMEM_LATENCY_ITERS; ++i) {
                ptr = local_smem[ptr];
            }
            results->local_read_latency_cycles[trial] = clock64() - start;
            dump_sink[0] = ptr;
        }
        cluster.sync();
    }

    if (rank == 0 && threadIdx.x == 0) {
        int ptr = 0;
        #pragma unroll 1
        for (int i = 0; i < DSMEM_LATENCY_WARMUP; ++i) {
            ptr = dsmem_load(remote_smem + ptr);
        }
        dump_sink[0] = ptr;
    }
    cluster.sync();

    for (int trial = 0; trial < DSMEM_TRIALS; ++trial) {
        if (rank == 0 && threadIdx.x == 0) {
            int ptr = 0;
            unsigned long long start = clock64();
            #pragma unroll 1
            for (int i = 0; i < DSMEM_LATENCY_ITERS; ++i) {
                ptr = dsmem_load(remote_smem + ptr);
            }
            results->remote_read_latency_cycles[trial] = clock64() - start;
            dump_sink[0] = ptr;
        }
        cluster.sync();
    }
}

template <bool Remote>
__device__ __forceinline__ unsigned long long time_read_throughput(
    int* data,
    unsigned long long* warp_start,
    unsigned long long* warp_end,
    int* thread_sum)
{
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int index = threadIdx.x;
    int sum = 0;
    unsigned long long start = 0;

    __syncthreads();
    if (lane == 0) {
        start = clock64();
    }

    #pragma unroll 1
    for (int i = 0; i < DSMEM_THROUGHPUT_ITERS; ++i) {
        if constexpr (Remote) {
            sum += dsmem_load(data + index);
        } else {
            sum += reinterpret_cast<volatile int*>(data)[index];
        }
        index += DSMEM_BLOCK_SIZE;
        if (index >= DSMEM_PAYLOAD_WORDS) {
            index = threadIdx.x;
        }
    }

    if (lane == 0) {
        warp_start[warp] = start;
        warp_end[warp] = clock64();
    }
    __syncthreads();

    unsigned long long elapsed = 0;
    if (threadIdx.x == 0) {
        unsigned long long first = warp_start[0];
        unsigned long long last = warp_end[0];
        for (int warp_id = 1; warp_id < DSMEM_WARPS; ++warp_id) {
            first = min(first, warp_start[warp_id]);
            last = max(last, warp_end[warp_id]);
        }
        elapsed = last - first;
    }
    *thread_sum = sum;
    return elapsed;
}

template <bool Remote>
__device__ __forceinline__ unsigned long long time_store_throughput(
    int* data,
    unsigned long long* warp_start,
    unsigned long long* warp_end)
{
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int index = threadIdx.x;
    unsigned long long start = 0;

    __syncthreads();
    if (lane == 0) {
        start = clock64();
    }

    #pragma unroll 1
    for (int i = 0; i < DSMEM_THROUGHPUT_ITERS; ++i) {
        if constexpr (Remote) {
            dsmem_store(data + index, i + threadIdx.x);
        } else {
            reinterpret_cast<volatile int*>(data)[index] = i + threadIdx.x;
        }
        index += DSMEM_BLOCK_SIZE;
        if (index >= DSMEM_PAYLOAD_WORDS) {
            index = threadIdx.x;
        }
    }

    if (lane == 0) {
        warp_start[warp] = start;
        warp_end[warp] = clock64();
    }
    __syncthreads();

    unsigned long long elapsed = 0;
    if (threadIdx.x == 0) {
        unsigned long long first = warp_start[0];
        unsigned long long last = warp_end[0];
        for (int warp_id = 1; warp_id < DSMEM_WARPS; ++warp_id) {
            first = min(first, warp_start[warp_id]);
            last = max(last, warp_end[warp_id]);
        }
        elapsed = last - first;
    }
    return elapsed;
}

__global__ void __cluster_dims__(2, 1, 1) dsmem_throughput_bench(
    DsmemResults* results,
    int* dump_sink)
{
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int rank = cluster.block_rank();
    extern __shared__ int local_storage[];
    int* remote_smem = cluster.map_shared_rank(local_storage, 1 - rank);
    __shared__ unsigned long long warp_start[DSMEM_WARPS];
    __shared__ unsigned long long warp_end[DSMEM_WARPS];

    for (int i = threadIdx.x; i < DSMEM_PAYLOAD_WORDS; i += blockDim.x) {
        local_storage[i] = i + 1;
    }
    record_cluster_placement(results, DSMEM_METRIC_THROUGHPUT, rank);
    cluster.sync();

    for (int trial = 0; trial < DSMEM_TRIALS; ++trial) {
        if (rank == 0) {
            int sum = 0;
            unsigned long long elapsed = time_read_throughput<false>(
                local_storage, warp_start, warp_end, &sum);
            dump_sink[threadIdx.x] = sum;
            if (threadIdx.x == 0) {
                results->local_read_throughput_cycles[trial] = elapsed;
            }
        }
        cluster.sync();
    }

    for (int trial = 0; trial < DSMEM_TRIALS; ++trial) {
        if (rank == 0) {
            int sum = 0;
            unsigned long long elapsed = time_read_throughput<true>(
                remote_smem, warp_start, warp_end, &sum);
            dump_sink[threadIdx.x] = sum;
            if (threadIdx.x == 0) {
                results->remote_read_throughput_cycles[trial] = elapsed;
            }
        }
        cluster.sync();
    }

    for (int trial = 0; trial < DSMEM_TRIALS; ++trial) {
        if (rank == 0) {
            unsigned long long elapsed = time_store_throughput<false>(
                local_storage, warp_start, warp_end);
            if (threadIdx.x == 0) {
                results->local_store_throughput_cycles[trial] = elapsed;
            }
        }
        cluster.sync();
    }

    for (int trial = 0; trial < DSMEM_TRIALS; ++trial) {
        if (rank == 0) {
            unsigned long long elapsed = time_store_throughput<true>(
                remote_smem, warp_start, warp_end);
            if (threadIdx.x == 0) {
                results->remote_store_throughput_cycles[trial] = elapsed;
            }
        }
        cluster.sync();
    }

    if (rank == 1) {
        dump_sink[DSMEM_BLOCK_SIZE + threadIdx.x] = local_storage[threadIdx.x];
    }
    cluster.sync();
}

__global__ void __cluster_dims__(2, 1, 1) dsmem_visibility_bench(
    DsmemResults* results,
    int* dump_sink)
{
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int rank = cluster.block_rank();
    extern __shared__ int local_storage[];
    volatile int* local_smem = local_storage;
    int* remote_smem = cluster.map_shared_rank(local_storage, 1 - rank);
    constexpr int request_index = 0;
    constexpr int acknowledge_index = 1;

    if (threadIdx.x < 2) {
        local_storage[threadIdx.x] = 0;
    }
    record_cluster_placement(results, DSMEM_METRIC_VISIBILITY, rank);
    cluster.sync();

    if (threadIdx.x == 0) {
        for (int i = 1; i <= DSMEM_VISIBILITY_WARMUP; ++i) {
            if (rank == 0) {
                dsmem_store(remote_smem + request_index, i);
                while (local_smem[acknowledge_index] != i) {
                }
            } else {
                while (local_smem[request_index] != i) {
                }
                dsmem_store(remote_smem + acknowledge_index, i);
            }
        }
    }
    cluster.sync();

    for (int trial = 0; trial < DSMEM_TRIALS; ++trial) {
        int first_token = DSMEM_VISIBILITY_WARMUP + trial * DSMEM_VISIBILITY_ITERS + 1;
        if (threadIdx.x == 0) {
            unsigned long long start = 0;
            if (rank == 0) {
                start = clock64();
            }

            #pragma unroll 1
            for (int i = 0; i < DSMEM_VISIBILITY_ITERS; ++i) {
                int token = first_token + i;
                if (rank == 0) {
                    dsmem_store(remote_smem + request_index, token);
                    while (local_smem[acknowledge_index] != token) {
                    }
                } else {
                    while (local_smem[request_index] != token) {
                    }
                    dsmem_store(remote_smem + acknowledge_index, token);
                }
            }

            if (rank == 0) {
                results->visibility_roundtrip_cycles[trial] = clock64() - start;
                dump_sink[0] = local_smem[acknowledge_index];
            }
        }
        cluster.sync();
    }
}

struct MetricSummary {
    double minimum;
    double median;
    double maximum;
};

MetricSummary summarize_scaled(
    const unsigned long long* samples,
    int count,
    double scale)
{
    std::vector<double> values(count);
    for (int i = 0; i < count; ++i) {
        values[i] = static_cast<double>(samples[i]) * scale;
    }
    std::sort(values.begin(), values.end());
    return {values.front(), values[count / 2], values.back()};
}

MetricSummary summarize_throughput(
    const unsigned long long* samples,
    int count,
    double bytes)
{
    std::vector<double> values(count);
    for (int i = 0; i < count; ++i) {
        values[i] = bytes / static_cast<double>(samples[i]);
    }
    std::sort(values.begin(), values.end());
    return {values.front(), values[count / 2], values.back()};
}

void print_metric(const char* label, const char* unit, const MetricSummary& summary) {
    printf("%-34s : %9.2f %s  (min %.2f, max %.2f)\n",
           label, summary.median, unit, summary.minimum, summary.maximum);
}

int main() {
    // Working-set sizes are chosen to target L1, L2, and DRAM respectively.
    constexpr int L1_ELEMS = 1 << 13;
    constexpr int L2_ELEMS = 1 << 21;
    constexpr int DRAM_ELEMS = 1 << 26;
    constexpr int CACHE_LATENCY_STRIDE = 257;
    constexpr int DRAM_STRIDE = L2_ELEMS + 1;
    constexpr int LATENCY_ITERS = 100000;
    constexpr int L1_BW_REPEATS = 4096;
    constexpr int L2_BW_REPEATS = 1024;
    constexpr int BW_BLOCK_SIZE = 256;

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
    init_stride_cycle(h_data, L1_ELEMS, CACHE_LATENCY_STRIDE);
    CUDA_CHECK(cudaMemcpy(d_data, h_data, l1Bytes, cudaMemcpyHostToDevice));
    free(h_data);

    long long h_result = 0;

    read_latency_kernel<<<1, 1>>>(
        d_data, L1_ELEMS, d_result, d_sink);
    CUDA_CHECK(cudaDeviceSynchronize());
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
    init_stride_cycle(h_l2, L2_ELEMS, CACHE_LATENCY_STRIDE);
    CUDA_CHECK(cudaMemcpy(d_l2_data, h_l2, l2Bytes, cudaMemcpyHostToDevice));
    free(h_l2);

    l2_read_latency_kernel<<<1, 1>>>(
        d_l2_data, L2_ELEMS, d_result, d_sink);
    CUDA_CHECK(cudaDeviceSynchronize());
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

    int clusterLaunchSupported = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &clusterLaunchSupported, cudaDevAttrClusterLaunch, device));

    if (clusterLaunchSupported) {
        const size_t payloadSharedBytes = DSMEM_PAYLOAD_WORDS * sizeof(int);
        const size_t halfSmSharedBytes =
            static_cast<size_t>(prop.sharedMemPerMultiprocessor) / 2;
        const size_t forceOneBlockBytes =
            ((halfSmSharedBytes + 256) / 256) * 256;
        const size_t maxOptinSharedBytes = prop.sharedMemPerBlockOptin > 0
            ? static_cast<size_t>(prop.sharedMemPerBlockOptin)
            : static_cast<size_t>(prop.sharedMemPerBlock);
        const size_t throughputStaticSharedBytes =
            2 * DSMEM_WARPS * sizeof(unsigned long long);
        const size_t maxDynamicSharedBytes = maxOptinSharedBytes > throughputStaticSharedBytes
            ? maxOptinSharedBytes - throughputStaticSharedBytes
            : 0;
        size_t dsmemSharedBytes = payloadSharedBytes;
        bool forcedOneBlockPerSm = false;

        if (forceOneBlockBytes >= payloadSharedBytes &&
            forceOneBlockBytes <= maxDynamicSharedBytes) {
            dsmemSharedBytes = forceOneBlockBytes;
            forcedOneBlockPerSm = true;
        }

        DsmemResults* d_dsmem_results = nullptr;
        int* d_dsmem_sink = nullptr;
        DsmemResults h_dsmem_results = {};
        CUDA_CHECK(cudaMalloc(&d_dsmem_results, sizeof(DsmemResults)));
        CUDA_CHECK(cudaMalloc(
            &d_dsmem_sink, 2 * DSMEM_BLOCK_SIZE * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_dsmem_results, 0, sizeof(DsmemResults)));
        CUDA_CHECK(cudaMemset(
            d_dsmem_sink, 0, 2 * DSMEM_BLOCK_SIZE * sizeof(int)));

        CUDA_CHECK(cudaFuncSetAttribute(
            dsmem_sync_bench,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(dsmemSharedBytes)));
        CUDA_CHECK(cudaFuncSetAttribute(
            dsmem_read_latency_bench,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(dsmemSharedBytes)));
        CUDA_CHECK(cudaFuncSetAttribute(
            dsmem_throughput_bench,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(dsmemSharedBytes)));
        CUDA_CHECK(cudaFuncSetAttribute(
            dsmem_visibility_bench,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(dsmemSharedBytes)));

        dsmem_sync_bench<<<2, DSMEM_BLOCK_SIZE, dsmemSharedBytes>>>(
            d_dsmem_results);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        dsmem_read_latency_bench<<<2, DSMEM_BLOCK_SIZE, dsmemSharedBytes>>>(
            d_dsmem_results, d_dsmem_sink);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        dsmem_throughput_bench<<<2, DSMEM_BLOCK_SIZE, dsmemSharedBytes>>>(
            d_dsmem_results, d_dsmem_sink);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        dsmem_visibility_bench<<<2, DSMEM_BLOCK_SIZE, dsmemSharedBytes>>>(
            d_dsmem_results, d_dsmem_sink);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(
            &h_dsmem_results,
            d_dsmem_results,
            sizeof(DsmemResults),
            cudaMemcpyDeviceToHost));

        printf("----- Inter-SM DSMEM (2 CTAs, %d threads/CTA) -----\n",
               DSMEM_BLOCK_SIZE);
        printf("Dynamic shared memory/CTA         : %zu bytes%s\n",
               dsmemSharedBytes,
               forcedOneBlockPerSm ? " (forces <=1 CTA/SM)" : "");

        const char* metricNames[DSMEM_METRIC_COUNT] = {
            "cluster sync", "read latency", "throughput", "visibility"
        };
        bool placementValid = true;
        for (int metric = 0; metric < DSMEM_METRIC_COUNT; ++metric) {
            unsigned int requesterSm = h_dsmem_results.sm_id[metric][0];
            unsigned int providerSm = h_dsmem_results.sm_id[metric][1];
            printf("%-34s : SM %u -> SM %u\n",
                   metricNames[metric], requesterSm, providerSm);
            placementValid &= requesterSm != providerSm;
        }
        if (!placementValid) {
            fprintf(stderr,
                    "WARNING: at least one DSMEM sample used two CTAs on the same SM; "
                    "do not treat that sample as inter-SM.\n");
        }

        print_metric(
            "Cluster sync latency",
            "cycles/sync",
            summarize_scaled(
                h_dsmem_results.sync_cycles,
                DSMEM_TRIALS,
                1.0 / DSMEM_SYNC_ITERS));
        print_metric(
            "Local SMEM read latency",
            "cycles/load",
            summarize_scaled(
                h_dsmem_results.local_read_latency_cycles,
                DSMEM_TRIALS,
                1.0 / DSMEM_LATENCY_ITERS));
        print_metric(
            "DSMEM read latency",
            "cycles/load",
            summarize_scaled(
                h_dsmem_results.remote_read_latency_cycles,
                DSMEM_TRIALS,
                1.0 / DSMEM_LATENCY_ITERS));

        const double throughputBytes =
            static_cast<double>(DSMEM_BLOCK_SIZE) *
            DSMEM_THROUGHPUT_ITERS * sizeof(int);
        print_metric(
            "Local SMEM read throughput",
            "B/cycle/CTA",
            summarize_throughput(
                h_dsmem_results.local_read_throughput_cycles,
                DSMEM_TRIALS,
                throughputBytes));
        print_metric(
            "DSMEM read throughput",
            "B/cycle/CTA",
            summarize_throughput(
                h_dsmem_results.remote_read_throughput_cycles,
                DSMEM_TRIALS,
                throughputBytes));
        print_metric(
            "Local SMEM store throughput",
            "B/cycle/CTA",
            summarize_throughput(
                h_dsmem_results.local_store_throughput_cycles,
                DSMEM_TRIALS,
                throughputBytes));
        print_metric(
            "DSMEM store issue throughput",
            "B/cycle/CTA",
            summarize_throughput(
                h_dsmem_results.remote_store_throughput_cycles,
                DSMEM_TRIALS,
                throughputBytes));

        MetricSummary visibilityRoundtrip = summarize_scaled(
            h_dsmem_results.visibility_roundtrip_cycles,
            DSMEM_TRIALS,
            1.0 / DSMEM_VISIBILITY_ITERS);
        print_metric(
            "Store visibility round trip",
            "cycles/roundtrip",
            visibilityRoundtrip);
        print_metric(
            "One-way visibility estimate",
            "cycles",
            {visibilityRoundtrip.minimum / 2.0,
             visibilityRoundtrip.median / 2.0,
             visibilityRoundtrip.maximum / 2.0});

        CUDA_CHECK(cudaFree(d_dsmem_results));
        CUDA_CHECK(cudaFree(d_dsmem_sink));
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
