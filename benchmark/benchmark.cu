#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// -------------------------------------------------------------------------
// CONSTANTS & MACROS
// -------------------------------------------------------------------------
/*
RTX 5090 Spces
L1 cache: 128KB per SM
L2 cache: 96MB (L2 is shared across all SMs)
Global memory: 32GB
*/
#define ITERATIONS 10000
#define MB (1024 * 1024)
#define L1_SIZE_BYTES (24 * 1024)      // 24KB, small enough to fit in L1
#define L2_SIZE_BYTES (24 * 1024 * 1024) // 24MB, fits in L2, spills L1
#define GLOBAL_SIZE_BYTES (256 * 1024 * 1024) // 256MB, spills L2

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (Line %d)\n", cudaGetErrorString(err), __LINE__); \
            return 1; \
        } \
    } while (0)

// P1: memory latency (pointer chasing, cannot be pipelined)
__global__ void latency_kernel(const int* __restrict__ data, int* __restrict__ sink, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0) return; // Single thread measurement to avoid contention

    int k = 0;
    long long start_time = clock64();

    // Critical loop: dependent loads
    #pragma unroll 1
    for (int i = 0; i < ITERATIONS; ++i) {
        k = data[k];
    }

    long long end_time = clock64();
    sink[0] = k; // Prevent compiler optimization
    
    float lat = (float)(end_time - start_time) / ITERATIONS;
    printf("  [Size: %d MB] Latency: %.2f cycles\n", N / (1024*1024/4), lat);

    return;
}

// Helper to initialize pointer chasing array
void init_stride_array(int* h_data, int size_elements, int stride_elements) {
    for (int i = 0; i < size_elements; ++i) {
        h_data[i] = (i + stride_elements) % size_elements;
    }
}

// -------------------------------------------------------------------------
// PART 2: MEMORY BANDWIDTH
// -------------------------------------------------------------------------
// Uses int4 (128-bit) loads to maximize memory pressure.
__global__ void bandwidth_kernel(const int4* __restrict__ src, int4* __restrict__ dst, long long N_vectors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (long long i = idx; i < N_vectors; i += stride) {
        dst[i] = src[i];
    }
}

// -------------------------------------------------------------------------
// PART 3: CLUSTER / DSMEM
// -------------------------------------------------------------------------
__global__ void __cluster_dims__(2, 1, 1) cluster_bench(int* dump_sink) {
    // Get current rank in the cluster
    cg::grid_group grid = cg::this_grid();
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int rank = cluster.block_rank();

    const int num_ints = 1024; // 4KB per thread block pass
    const int iters = 1000;

    // 3.1 Cluster Barrier Latency [cite: 36]
    // Measure barrier.cluster.arrive/wait cycle cost
    if (threadIdx.x == 0 && rank == 0) {
        long long start = clock64();
        cluster.sync(); // Calls barrier.cluster.arrive + wait
        long long end = clock64();
        printf("  [Cluster] Sync Latency (2 Blocks): %lld cycles\n", (end - start));
    } else {
        cluster.sync();
    }

    // 3.2 DSMEM Bandwidth (Push Pattern) [cite: 95, 120]
    // Block 0 writes to Block 1's shared memory.
    extern __shared__ int local_smem[];
    
    // Map address of Block 1's shared memory
    int* remote_smem = cluster.map_shared_rank(local_smem, 1 - rank);

    if (rank == 0) {
        // I am the PRODUCER (Block 0)
        // Write to CONSUMER (Block 1)
        long long start = clock64();
        
        
        #pragma unroll
        for(int i=0; i < iters; i++) {
             // Each thread writes one int to remote shared memory (push pattern)
             if(threadIdx.x < num_ints) {
                 remote_smem[threadIdx.x] = i; 
             }
        }
        cluster.sync(); // Ensure data is landed
        long long end = clock64();

        if (threadIdx.x == 0) {
             // Total bytes = 1000 iter * 1024 ints * 4 bytes
             double bytes = (double)iters * num_ints * sizeof(int);
             double cycles = (double)(end - start);
             // Approximate bytes/cycle
             printf("  [Cluster] DSMEM PUSH Bandwidth: %.2f Bytes/Cycle\n", bytes/cycles);
        }
    } else {
        // I am the CONSUMER (Block 1)
        // Just wait for data
        cluster.sync();
        if(threadIdx.x == 0) dump_sink[0] = local_smem[0]; // Side effect
    }

    // 3.3 Pull pattern
    // First, Rank 1 prepares data for Rank 0 to pull
    if (rank == 1 && threadIdx.x < num_ints) {
        local_smem[threadIdx.x] = threadIdx.x;
    }
    cluster.sync(); // Ensure data is ready in Rank 1's SMEM

    if (rank == 0) {
        int thread_sum = 0;
        long long start = clock64();
        
        for(int i=0; i < iters; i++) {
            if(threadIdx.x < num_ints) {
                // PULL: Explicitly reading from remote
                thread_sum += remote_smem[threadIdx.x];
            }
        }
        
        cluster.sync(); // Ensure all reads finished
        long long end = clock64();

        if (threadIdx.x == 0) {
            dump_sink[0] = thread_sum; // Prevent optimization
            double bytes = (double)iters * num_ints * sizeof(int);
            printf("  [Cluster] DSMEM PULL Bandwidth: %.2f Bytes/Cycle\n", bytes/(double)(end - start));
        }
    } else {
        cluster.sync(); // Participate in the timing barrier
    }

    // Measure the PULL latency
    if (threadIdx.x == 0) {
        local_smem[0] = 0; 
    }
    cluster.sync();

    if (rank == 0 && threadIdx.x == 0) {
        int k = 0;
        long long start = clock64();

        #pragma unroll 1
        for (int i = 0; i < ITERATIONS; ++i) {
            // PULL Latency: Each load must cross the NoC to Rank 1
            // and return before the next iteration can start.
            k = remote_smem[k]; 
        }

        long long end = clock64();
        dump_sink[0] = k;
        printf("  [Cluster] DSMEM PULL Latency: %.2f cycles\n", (float)(end - start) / ITERATIONS);
    }
    else {
        cluster.sync();
    }

    return;
}

// Dynamic Cluster Scaling Kernel
// No __cluster_dims__ macro here; we set it in host code.
__global__ void myClusterScaling(int* dump_sink, int num_iters) {
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int rank = cluster.block_rank();
    unsigned int cluster_size = cluster.num_blocks();

    // 1. SYNC Latency
    // Measure barrier cost for N blocks
    if (rank == 0 && threadIdx.x == 0) {
        long long start = clock64();
        cluster.sync();
        long long end = clock64();
        printf("[Cluster Size %d] SYNC Latency: %lld cycles\n", cluster_size, (end - start));
    } else {
        cluster.sync();
    }

    // Setup for Bandwidth Tests (Ring Pattern)
    // Rank i talks to Rank (i+1)%Size
    extern __shared__ int local_smem[];
    int neighbor_rank = (rank + 1) % cluster_size;
    int* remote_smem = cluster.map_shared_rank(local_smem, neighbor_rank);

    const int num_ints = 1024; // 4KB payload
    
    // 2. PUSH Bandwidth (Aggregate)
    // Every block pushes to its neighbor simultaneously
    cluster.sync();
    
    long long start_push = clock64();
    #pragma unroll
    for(int i=0; i < num_iters; i++) {
        if(threadIdx.x < num_ints) {
            remote_smem[threadIdx.x] = i; 
        }
    }
    cluster.sync(); // Wait for all pushes to land
    long long end_push = clock64();

    // Only Rank 0 prints the *Aggregate* bandwidth
    if (rank == 0 && threadIdx.x == 0) {
        double total_bytes = (double)num_iters * num_ints * sizeof(int);
        double cycles = (double)(end_push - start_push);
        printf("[Cluster Size %d] PUSH Bandwidth (Aggregate): %.2f Bytes/Cycle\n", cluster_size, total_bytes/cycles);
    }

    // 3. PULL Bandwidth (Aggregate)
    // Every block pulls from its neighbor
    // First, ensure data exists
    if (threadIdx.x < num_ints) local_smem[threadIdx.x] = rank;
    cluster.sync();

    long long start_pull = clock64();
    int sum = 0;
    #pragma unroll
    for(int i=0; i < num_iters; i++) {
        if(threadIdx.x < num_ints) {
            sum += remote_smem[threadIdx.x]; 
        }
    }
    cluster.sync();
    long long end_pull = clock64();

    if (rank == 0 && threadIdx.x == 0) {
        dump_sink[0] = sum;
        double total_bytes = (double)num_iters * num_ints * sizeof(int);
        double cycles = (double)(end_pull - start_pull);
        printf("[Cluster Size %d] PULL Bandwidth (Aggregate): %.2f Bytes/Cycle\n", cluster_size, total_bytes/cycles);
    }

    // 4. PULL Latency (Single Link)
    // Only Rank 0 reads from Rank 1 (or Neighbor). 
    // We don't scale this because latency is a point-to-point metric.
    if (threadIdx.x == 0) local_smem[0] = 0;
    cluster.sync();

    if (rank == 0 && threadIdx.x == 0) {
        int k = 0;
        long long start_lat = clock64();
        #pragma unroll 1
        for (int i = 0; i < 10000; ++i) {
             k = remote_smem[k]; 
        }
        long long end_lat = clock64();
        dump_sink[0] = k;
        printf("[Cluster Size %d] PULL Latency: %.2f cycles\n", cluster_size, (float)(end_lat - start_lat) / 10000.0f);
    } else {
        cluster.sync();
    }
}

// -------------------------------------------------------------------------
// MAIN
// -------------------------------------------------------------------------
int main() {
    int dev_id = 0;
    cudaSetDevice(dev_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev_id);
    printf("Benchmarking NVIDIA GPU: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("--------------------------------------------------\n");

    int clockRateKHz;
    cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, 0);
    float clockRateGHz = clockRateKHz / 1e6;

    // ================== LATENCY ==================
    printf("### 1. Memory Access Latency (Pointer Chasing)\n");
    int *d_data, *d_sink;
    int max_elements = GLOBAL_SIZE_BYTES / sizeof(int);
    CHECK_CUDA(cudaMalloc(&d_data, GLOBAL_SIZE_BYTES));
    CHECK_CUDA(cudaMalloc(&d_sink, sizeof(int)));
    int* h_data = new int[max_elements];

    // L1 Test (Small array, hit L1)
    int l1_elements = L1_SIZE_BYTES / sizeof(int);
    init_stride_array(h_data, l1_elements, 1);
    cudaMemcpy(d_data, h_data, L1_SIZE_BYTES, cudaMemcpyHostToDevice);
    latency_kernel<<<1, 1>>>(d_data, d_sink, L1_SIZE_BYTES);

    // L2 Test (Medium array, miss L1, hit L2)
    int l2_elements = L2_SIZE_BYTES / sizeof(int);
    init_stride_array(h_data, l2_elements, 16); // Stride to ensure cache lines are jumped
    cudaMemcpy(d_data, h_data, L2_SIZE_BYTES, cudaMemcpyHostToDevice);
    latency_kernel<<<1, 1>>>(d_data, d_sink, L2_SIZE_BYTES);

    // Global Test (Large array, miss L2)
    init_stride_array(h_data, max_elements, 32); 
    cudaMemcpy(d_data, h_data, GLOBAL_SIZE_BYTES, cudaMemcpyHostToDevice);
    latency_kernel<<<1, 1>>>(d_data, d_sink, GLOBAL_SIZE_BYTES);
    CHECK_CUDA(cudaDeviceSynchronize());

    // ================== BANDWIDTH ==================
    printf("\n### 2. Global Memory Bandwidth\n");
    int4 *d_src_v, *d_dst_v;
    size_t bw_size = 1024 * 1024 * 1024; // 1GB
    long long n_vectors = bw_size / sizeof(int4);
    CHECK_CUDA(cudaMalloc(&d_src_v, bw_size));
    CHECK_CUDA(cudaMalloc(&d_dst_v, bw_size));
    
    // Warmup
    bandwidth_kernel<<<1024, 256>>>(d_src_v, d_dst_v, n_vectors); 
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    bandwidth_kernel<<<80*114, 256>>>(d_src_v, d_dst_v, n_vectors); // Saturation grid
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    // 2x because Read + Write
    printf("  Global Bandwidth: %.2f GB/s\n", (bw_size * 2.0) / (ms * 1e6));

    // ================== CLUSTER ==================
    // Only runs if CC >= 9.0
    if (prop.major >= 9) {
        printf("\n### 3. Cluster / DSMEM Network (Inter-SM)\n");
        // Launch 2 blocks (dimension of cluster)
        // Shared mem size: 4KB
        int smem_size = 4096; 
        
        // Use cudaLaunchKernelEx to strictly launch a cluster
        cudaLaunchConfig_t config = {0};
        config.gridDim = 2; // 2 blocks total
        config.blockDim = 1024;
        config.dynamicSmemBytes = smem_size;
        
        // Setup Cluster Attribute
        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2; // Cluster size 2
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        config.attrs = attribute;
        config.numAttrs = 1;

        cudaLaunchKernelEx(&config, cluster_bench, d_sink);
        CHECK_CUDA(cudaDeviceSynchronize());
    } else {
        printf("\n[Skipped] Cluster benchmarks require Compute Capability 9.0+\n");
    }

    // ================== CLUSTER SCALING ==================
    if (prop.major >= 9) {
        printf("\n### 3. Cluster Scaling Benchmarks\n");
        
        int smem_size = 4096;
        int num_iters = 1000;
        
        // Define sizes to test. Note: Size 16 is max for H100/Blackwell.
        // Size 1 is usually invalid for "inter-SM" tests.
        std::vector<int> cluster_sizes = {2, 4, 8};

        for (int size : cluster_sizes) {
            // Check if hardware supports this cluster size
            if (size > prop.maxThreadsPerMultiProcessor / 1024 * prop.multiProcessorCount) { 
                // Simple sanity check, though cudaLaunchKernelEx handles validation better
                continue; 
            }

            cudaLaunchConfig_t config = {0};
            // Grid dimension must be a multiple of cluster size
            config.gridDim = size;  
            config.blockDim = 1024;
            config.dynamicSmemBytes = smem_size;

            cudaLaunchAttribute attribute[1];
            attribute[0].id = cudaLaunchAttributeClusterDimension;
            attribute[0].val.clusterDim.x = size;
            attribute[0].val.clusterDim.y = 1;
            attribute[0].val.clusterDim.z = 1;
            config.attrs = attribute;
            config.numAttrs = 1;

            // Launch
            cudaError_t err = cudaLaunchKernelEx(&config, myClusterScaling, d_sink, num_iters);
            
            if (err != cudaSuccess) {
                printf("[Skipped] Cluster Size %d failed to launch: %s\n", size, cudaGetErrorString(err));
            } else {
                CHECK_CUDA(cudaDeviceSynchronize());
            }
        }
    }

    // Cleanup
    cudaFree(d_data); cudaFree(d_sink); cudaFree(d_src_v); cudaFree(d_dst_v);
    delete[] h_data;
    return 0;
}
