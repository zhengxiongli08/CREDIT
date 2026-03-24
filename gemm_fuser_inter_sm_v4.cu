#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <mma.h>
#include <cooperative_groups.h>

#include <cstdlib>
#include <iostream>

namespace cg = cooperative_groups;
using namespace nvcuda;

#ifndef FF_M_GLOBAL
#define FF_M_GLOBAL 128
#endif

#ifndef FF_K_GLOBAL
#define FF_K_GLOBAL 4096
#endif

#ifndef FF_N_GLOBAL
#define FF_N_GLOBAL 16384
#endif

// Workload from gemm_pytorch.py / gemm_tensorrt.py
constexpr int M_GLOBAL = FF_M_GLOBAL;
constexpr int K_GLOBAL = FF_K_GLOBAL;
constexpr int N_GLOBAL = FF_N_GLOBAL;

#ifndef FF_BM
#define FF_BM 32
#endif

#ifndef FF_BK_IN
#define FF_BK_IN 64
#endif

#ifndef FF_BK_OUT
#define FF_BK_OUT 128
#endif

#ifndef FF_CLUSTER_SIZE
#define FF_CLUSTER_SIZE 16
#endif

#ifndef FF_PAD
#define FF_PAD 8
#endif

#ifndef FF_WARPS_M
#define FF_WARPS_M 1
#endif

#ifndef FF_WARPS_N
#define FF_WARPS_N 8
#endif

#ifndef FF_PHASE2_PIPELINE_B2
#define FF_PHASE2_PIPELINE_B2 0
#endif

#ifndef FF_KERNEL_TAG
#define FF_KERNEL_TAG "fused_ffn_inter_sm_v4"
#endif

// Inter-SM fused tile design
constexpr int BM = FF_BM;
constexpr int BK_IN = FF_BK_IN;
constexpr int BK_OUT = FF_BK_OUT;
constexpr int CLUSTER_SIZE = FF_CLUSTER_SIZE;
constexpr int BN_STEP = CLUSTER_SIZE * BK_OUT;

constexpr int PAD = FF_PAD;
constexpr int WARPS_M = FF_WARPS_M;
constexpr int WARPS_N = FF_WARPS_N;
constexpr int WARP_COUNT = WARPS_M * WARPS_N;
constexpr int THREADS = WARP_COUNT * 32;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int ROW_TILES_PER_WARP = BM / (WARPS_M * WMMA_M);
constexpr int COL_TILES_PER_WARP = BK_OUT / (WARPS_N * WMMA_N);
constexpr int FRAGS_PER_WARP = ROW_TILES_PER_WARP * COL_TILES_PER_WARP;

static_assert((M_GLOBAL % BM) == 0, "M_GLOBAL must be divisible by BM");
static_assert((K_GLOBAL % BK_OUT) == 0, "K_GLOBAL must be divisible by BK_OUT");
static_assert((K_GLOBAL % BK_IN) == 0, "K_GLOBAL must be divisible by BK_IN");
static_assert((N_GLOBAL % (CLUSTER_SIZE * BK_OUT)) == 0, "N_GLOBAL must be divisible by CLUSTER_SIZE * BK_OUT");
static_assert((BM % (WARPS_M * WMMA_M)) == 0, "BM must be divisible by WARPS_M * WMMA_M");
static_assert((BK_OUT % (WARPS_N * WMMA_N)) == 0, "BK_OUT must be divisible by WARPS_N * WMMA_N");
static_assert(ROW_TILES_PER_WARP >= 1 && COL_TILES_PER_WARP >= 1, "Invalid warp tile decomposition");
static_assert(FRAGS_PER_WARP <= 4, "Increase fragment array size");

#define CUDA_CHECK(cmd)                                                                          \
  do {                                                                                           \
    cudaError_t e = (cmd);                                                                       \
    if (e != cudaSuccess) {                                                                      \
      std::cerr << "CUDA error: " << cudaGetErrorString(e) << " @ line " << __LINE__ << '\n'; \
      std::exit(1);                                                                              \
    }                                                                                            \
  } while (0)

template <int Bytes>
__device__ __forceinline__ void cp_async_ca(void* smem_ptr, const void* gmem_ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  __pipeline_memcpy_async(smem_ptr, gmem_ptr, Bytes);
#else
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" :: "r"(smem), "l"(gmem_ptr), "n"(Bytes));
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  __pipeline_commit();
#else
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  __pipeline_wait_prior(N);
#else
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  __pipeline_wait_prior(0);
#else
  asm volatile("cp.async.wait_all;\n" ::);
#endif
}

__global__ void fill_kernel(half* p, int n, float v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = __float2half(v + 0.0001f * (i % 31));
}

// One thread-block computes E tile [BM x BK_OUT].
// CLUSTER_SIZE blocks in a cluster cooperate along the N dimension via DSM.
__global__ __cluster_dims__(CLUSTER_SIZE, 1, 1) void fused_ffn_inter_sm_v4_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B1,
    const half* __restrict__ B2,
    half* __restrict__ E) {

  extern __shared__ __align__(16) half smem[];

  const int stride_c = BK_OUT + PAD;
  const int stride_a = BK_IN + PAD;
  const int stride_b = BK_OUT + PAD;

  const int size_c = BM * stride_c;
  const int size_a = BM * stride_a;
  const int size_b = BK_IN * stride_b;
  half* s_C_local = smem;
  half* s_C_stage = s_C_local + size_c;
  half* s_A = s_C_stage + size_c;
  half* s_B = s_A + 2 * size_a;
  half* s_B2 = s_B + 2 * size_b;
#if FF_PHASE2_PIPELINE_B2
  half* s_B2_next = s_B2 + (BK_OUT * stride_b);
#endif

  half* s_A_pipe[2] = {s_A, s_A + size_a};
  half* s_B_pipe[2] = {s_B, s_B + size_b};

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = cluster.block_rank();
  const int tid = threadIdx.x;

  if (cluster.dim_blocks().x != CLUSTER_SIZE) {
    return;
  }

  const int m_tile = blockIdx.y * BM;
  const int cluster_idx_x = blockIdx.x / CLUSTER_SIZE;
  const int k_cluster = cluster_idx_x * (BK_OUT * CLUSTER_SIZE);
  const int k_tile = k_cluster + rank * BK_OUT;

  const int warp_id = tid / 32;
  const int warp_m = warp_id / WARPS_N;
  const int warp_n = warp_id % WARPS_N;
  const int row_base = warp_m * ROW_TILES_PER_WARP * WMMA_M;
  const int col_base = warp_n * COL_TILES_PER_WARP * WMMA_N;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_E[FRAGS_PER_WARP];
#pragma unroll
  for (int i = 0; i < FRAGS_PER_WARP; ++i) wmma::fill_fragment(acc_E[i], __float2half(0.0f));

  for (int n_step = 0; n_step < N_GLOBAL; n_step += BN_STEP) {
    const int n_local = n_step + rank * BK_OUT;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_C[FRAGS_PER_WARP];
#pragma unroll
    for (int i = 0; i < FRAGS_PER_WARP; ++i) wmma::fill_fragment(acc_C[i], __float2half(0.0f));

    {
      const int k0 = 0;
      for (int i = tid; i < (BM * BK_IN) / 8; i += THREADS) {
        int r = i / (BK_IN / 8);
        int c = (i % (BK_IN / 8)) * 8;
        cp_async_ca<16>(&s_A_pipe[0][r * stride_a + c], &A[(m_tile + r) * K_GLOBAL + (k0 + c)]);
      }
      for (int i = tid; i < (BK_IN * BK_OUT) / 8; i += THREADS) {
        int r = i / (BK_OUT / 8);
        int c = (i % (BK_OUT / 8)) * 8;
        cp_async_ca<16>(&s_B_pipe[0][r * stride_b + c], &B1[(k0 + r) * N_GLOBAL + (n_local + c)]);
      }
      cp_async_commit();
    }

    for (int k_step = 0; k_step < K_GLOBAL; k_step += BK_IN) {
      int curr = (k_step / BK_IN) & 1;
      int next = 1 - curr;
      int next_k = k_step + BK_IN;

      if (next_k < K_GLOBAL) {
        for (int i = tid; i < (BM * BK_IN) / 8; i += THREADS) {
          int r = i / (BK_IN / 8);
          int c = (i % (BK_IN / 8)) * 8;
          cp_async_ca<16>(&s_A_pipe[next][r * stride_a + c], &A[(m_tile + r) * K_GLOBAL + (next_k + c)]);
        }
        for (int i = tid; i < (BK_IN * BK_OUT) / 8; i += THREADS) {
          int r = i / (BK_OUT / 8);
          int c = (i % (BK_OUT / 8)) * 8;
          cp_async_ca<16>(&s_B_pipe[next][r * stride_b + c], &B1[(next_k + r) * N_GLOBAL + (n_local + c)]);
        }
        cp_async_commit();
      }

      cp_async_wait_group<1>();
      __syncthreads();

      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;

#pragma unroll
      for (int ki = 0; ki < BK_IN; ki += WMMA_K) {
#pragma unroll
        for (int mi = 0; mi < ROW_TILES_PER_WARP; ++mi) {
#pragma unroll
          for (int nj = 0; nj < COL_TILES_PER_WARP; ++nj) {
            wmma::load_matrix_sync(a_frag, &s_A_pipe[curr][(row_base + mi * 16) * stride_a + ki], stride_a);
            wmma::load_matrix_sync(b_frag, &s_B_pipe[curr][ki * stride_b + (col_base + nj * 16)], stride_b);
            wmma::mma_sync(acc_C[mi * COL_TILES_PER_WARP + nj], a_frag, b_frag, acc_C[mi * COL_TILES_PER_WARP + nj]);
          }
        }
      }
      __syncthreads();
    }
    cp_async_wait_all();

#pragma unroll
    for (int f = 0; f < FRAGS_PER_WARP; ++f) {
#pragma unroll
      for (int t = 0; t < acc_C[f].num_elements; ++t) {
        acc_C[f].x[t] = __hmax(acc_C[f].x[t], __float2half(0.0f));
      }
      wmma::store_matrix_sync(
          &s_C_local[(row_base + (f / COL_TILES_PER_WARP) * 16) * stride_c +
                     (col_base + (f % COL_TILES_PER_WARP) * 16)],
          acc_C[f],
          stride_c,
          wmma::mem_row_major);
    }

    cluster.sync();

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> c_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b2_frag;

    const int vec_count = (BM * (BK_OUT + PAD)) / 8;

    for (int src_rank = 0; src_rank < CLUSTER_SIZE; ++src_rank) {
      const int n_src = n_step + src_rank * BK_OUT;

#if FF_PHASE2_PIPELINE_B2
      const bool has_next = (src_rank + 1) < CLUSTER_SIZE;
      const int n_next = n_step + (src_rank + 1) * BK_OUT;
      half* b2_curr = (src_rank & 1) ? s_B2_next : s_B2;
      half* b2_next = (src_rank & 1) ? s_B2 : s_B2_next;

      if (src_rank == 0) {
        for (int i = tid; i < (BK_OUT * BK_OUT) / 8; i += THREADS) {
          int r = i / (BK_OUT / 8);
          int c = (i % (BK_OUT / 8)) * 8;
          cp_async_ca<16>(&b2_curr[r * stride_b + c], &B2[(n_src + r) * K_GLOBAL + (k_tile + c)]);
        }
        cp_async_commit();
      }

      if (has_next) {
        for (int i = tid; i < (BK_OUT * BK_OUT) / 8; i += THREADS) {
          int r = i / (BK_OUT / 8);
          int c = (i % (BK_OUT / 8)) * 8;
          cp_async_ca<16>(&b2_next[r * stride_b + c], &B2[(n_next + r) * K_GLOBAL + (k_tile + c)]);
        }
        cp_async_commit();
      }
#else
      half* b2_curr = s_B2;
      for (int i = tid; i < (BK_OUT * BK_OUT) / 8; i += THREADS) {
        int r = i / (BK_OUT / 8);
        int c = (i % (BK_OUT / 8)) * 8;
        cp_async_ca<16>(&b2_curr[r * stride_b + c], &B2[(n_src + r) * K_GLOBAL + (k_tile + c)]);
      }
      cp_async_commit();
#endif

      half* src_c_ptr = (src_rank == rank) ? s_C_local : cluster.map_shared_rank(s_C_local, src_rank);
      int4* src_vec = reinterpret_cast<int4*>(src_c_ptr);
      int4* dst_vec = reinterpret_cast<int4*>(s_C_stage);
      for (int i = tid; i < vec_count; i += THREADS) {
        dst_vec[i] = src_vec[i];
      }

#if FF_PHASE2_PIPELINE_B2
      if ((src_rank + 1) < CLUSTER_SIZE) {
        cp_async_wait_group<1>();
      } else {
        cp_async_wait_group<0>();
      }
#else
      cp_async_wait_group<0>();
#endif
      __syncthreads();

#pragma unroll
      for (int kk = 0; kk < BK_OUT; kk += WMMA_K) {
#pragma unroll
        for (int mi = 0; mi < ROW_TILES_PER_WARP; ++mi) {
#pragma unroll
          for (int nj = 0; nj < COL_TILES_PER_WARP; ++nj) {
            wmma::load_matrix_sync(c_frag, &s_C_stage[(row_base + mi * 16) * stride_c + kk], stride_c);
            wmma::load_matrix_sync(b2_frag, &b2_curr[kk * stride_b + (col_base + nj * 16)], stride_b);
            wmma::mma_sync(acc_E[mi * COL_TILES_PER_WARP + nj], c_frag, b2_frag,
                           acc_E[mi * COL_TILES_PER_WARP + nj]);
          }
        }
      }
      __syncthreads();
    }

    cluster.sync();
  }

#pragma unroll
  for (int f = 0; f < FRAGS_PER_WARP; ++f) {
    wmma::store_matrix_sync(
        &E[(m_tile + row_base + (f / COL_TILES_PER_WARP) * 16) * K_GLOBAL +
           (k_tile + col_base + (f % COL_TILES_PER_WARP) * 16)],
        acc_E[f],
        K_GLOBAL,
        wmma::mem_row_major);
  }
}

int main(int argc, char** argv) {
  int warmup = 20;
  int iters = 100;
  if (argc >= 2) warmup = std::atoi(argv[1]);
  if (argc >= 3) iters = std::atoi(argv[2]);

  CUDA_CHECK(cudaSetDevice(0));

  int cluster_launch = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch, 0));
  if (!cluster_launch) {
    std::cerr << "Device does not support thread-block cluster launch (DSM).\n";
    return 1;
  }

  CUDA_CHECK(cudaFuncSetAttribute(
      fused_ffn_inter_sm_v4_kernel,
      cudaFuncAttributeNonPortableClusterSizeAllowed,
      1));

  int max_smem = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
  CUDA_CHECK(cudaFuncSetAttribute(
      fused_ffn_inter_sm_v4_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_smem));

  half *dA, *dB1, *dB2, *dE;
  CUDA_CHECK(cudaMalloc(&dA, sizeof(half) * M_GLOBAL * K_GLOBAL));
  CUDA_CHECK(cudaMalloc(&dB1, sizeof(half) * K_GLOBAL * N_GLOBAL));
  CUDA_CHECK(cudaMalloc(&dB2, sizeof(half) * N_GLOBAL * K_GLOBAL));
  CUDA_CHECK(cudaMalloc(&dE, sizeof(half) * M_GLOBAL * K_GLOBAL));

  fill_kernel<<<(M_GLOBAL * K_GLOBAL + 255) / 256, 256>>>(dA, M_GLOBAL * K_GLOBAL, 0.5f);
  fill_kernel<<<(K_GLOBAL * N_GLOBAL + 255) / 256, 256>>>(dB1, K_GLOBAL * N_GLOBAL, 0.25f);
  fill_kernel<<<(N_GLOBAL * K_GLOBAL + 255) / 256, 256>>>(dB2, N_GLOBAL * K_GLOBAL, 0.125f);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  dim3 block(THREADS, 1, 1);
  const int num_clusters_x = K_GLOBAL / (BK_OUT * CLUSTER_SIZE);
  dim3 grid(num_clusters_x * CLUSTER_SIZE, M_GLOBAL / BM, 1);

  int stride_c = BK_OUT + PAD;
  int stride_a = BK_IN + PAD;
  int stride_b = BK_OUT + PAD;
    size_t smem_bytes = sizeof(half) *
      (2 * BM * stride_c + 2 * (BM * stride_a) + 2 * (BK_IN * stride_b) +
       (BK_OUT * stride_b) * (FF_PHASE2_PIPELINE_B2 ? 2 : 1));

    std::cout << "Kernel=" << FF_KERNEL_TAG << std::endl;
  std::cout << "Shape(M,K,N)=" << M_GLOBAL << "," << K_GLOBAL << "," << N_GLOBAL << std::endl;
  std::cout << "GridClusters=" << num_clusters_x << "x" << grid.y
            << ", BlocksPerCluster=" << CLUSTER_SIZE
            << ", ThreadsPerBlock=" << THREADS << std::endl;
    std::cout << "Phase2B2Pipeline=" << FF_PHASE2_PIPELINE_B2 << std::endl;
  std::cout << "DynamicSmemKB=" << (smem_bytes / 1024.0f) << std::endl;

  cudaLaunchConfig_t launch_cfg = {};
  launch_cfg.gridDim = grid;
  launch_cfg.blockDim = block;
  launch_cfg.dynamicSmemBytes = static_cast<unsigned int>(smem_bytes);
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = CLUSTER_SIZE;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  launch_cfg.attrs = attrs;
  launch_cfg.numAttrs = 1;

  auto launch_once = [&]() {
    return cudaLaunchKernelEx(&launch_cfg, fused_ffn_inter_sm_v4_kernel, dA, dB1, dB2, dE);
  };

  for (int i = 0; i < warmup; ++i) {
    CUDA_CHECK(launch_once());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(launch_once());
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  float avg_ms = elapsed_ms / iters;

  double flops = 2.0 * M_GLOBAL * K_GLOBAL * N_GLOBAL * 2.0;
  double tflops = flops / (avg_ms * 1.0e-3) / 1.0e12;

  std::cout << "Average Latency: " << avg_ms << " ms" << std::endl;
  std::cout << "LATENCY_MS=" << avg_ms << std::endl;
  std::cout << "THROUGHPUT_TFLOPS=" << tflops << std::endl;

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB1));
  CUDA_CHECK(cudaFree(dB2));
  CUDA_CHECK(cudaFree(dE));
  return 0;
}
