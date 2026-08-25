#include <iostream>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void clusterTest() {
    cg::cluster_group cluster = cg::this_cluster();
    extern __shared__ int smem[];
    smem[threadIdx.x] = 0;
    cluster.sync();

    int nextBlock = (cluster.block_rank() + 1) % cluster.dim_blocks().x;
    int *dst_smem = cluster.map_shared_rank(smem, nextBlock);

    dst_smem[threadIdx.x] = cluster.block_rank() * 10;

    cluster.sync();

    int prevBlock = (cluster.block_rank() - 1 + cluster.dim_blocks().x) % cluster.dim_blocks().x;
    if (threadIdx.x == 0) {
        printf("Current Block %d, value: %d, it comes from Block %d.\n", cluster.block_rank(), smem[threadIdx.x], prevBlock);
    }

    return;
}

int main() {
    int threads_per_block = 2;
    int cluster_size = 4;

    size_t dynamic_smem_bytes = threads_per_block * sizeof(int);

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(cluster_size);
    config.blockDim = dim3(threads_per_block);
    config.dynamicSmemBytes = dynamic_smem_bytes;

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = cluster_size;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    config.attrs = attribute;
    config.numAttrs = 1;

    cudaError_t err = cudaLaunchKernelEx(&config, clusterTest);

    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Device execution failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    return 0;
}
