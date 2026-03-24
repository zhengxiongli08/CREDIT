/*
 * gemm_fuser_inter_sm_v7.cu
 *
 * Inter-SM fused FFN — M-split / Weight-Sharing Cluster design
 *
 *   E = ReLU(A @ B1.T) @ B2.T
 *   A  : [M, K]   input activations  (row-major)
 *   B1 : [N, K]   FC1 weight         (row-major, B1[n, k])
 *   B2 : [N, K]   FC2 weight         (row-major, B2[n, k])
 *   E  : [M, K]   output             (row-major)
 *
 * ── Core idea (contrast with v4/v5 N-split) ─────────────────────────────────
 *
 *   v4/v5 N-split:
 *     - Each SM owns a different [BM × BK_OUT] output tile (different N columns).
 *     - After GEMM-1 (A@B1), the intermediate C tile [BM × BK_OUT] lives in
 *       each SM's local SMEM.
 *     - GEMM-2 (C@B2) requires each SM to read all other SMs' C tiles via DSMEM,
 *       so there is a cluster.sync() between phase-1 and phase-2 AND
 *       a cluster.sync() inside every src_rank iteration of the phase-2 loop.
 *     - That cluster.sync() latency (serialised CTA synchronisation) dominates
 *       for small M, causing the 6× slowdown vs TRT observed in NCU.
 *
 *   v7 M-split (this kernel):
 *     - The cluster splits along M instead of N.
 *     - Each SM owns a different [BM_LOCAL × K] row-stripe of A and E.
 *       (BM_LOCAL = BM_CLUSTER / CLUSTER_SIZE)
 *     - B1 and B2 weight tiles are SHARED across the cluster through DSMEM:
 *       rank 0 loads the current B1/B2 tile; all other ranks read it via
 *       cluster.map_shared_rank().
 *     - There is NO need to share intermediate activations between SMs at all —
 *       each SM independently accumulates its own rows of C and E entirely in
 *       registers and local SMEM.
 *     - The only synchronisation on the critical path is a single lightweight
 *       cluster.arrive()/cluster.wait() barrier PER WEIGHT TILE, which signals
 *       "rank 0 has finished loading this B tile into its SMEM".
 *
 * ── Tiling overview ──────────────────────────────────────────────────────────
 *
 *   Grid:
 *     x : N_GLOBAL / BN   (one cluster per N-column block)
 *     y : M_GLOBAL / BM_CLUSTER  (one cluster per M-row cluster group)
 *   Each grid point launches one cluster of CLUSTER_SIZE thread-blocks.
 *
 *   Per-cluster tile:
 *     BM_CLUSTER rows of A/E   →  each SM handles BM_LOCAL = BM_CLUSTER/CLUSTER_SIZE rows
 *     BN columns of B1/B2      →  the entire BN strip is held in rank-0's SMEM and
 *                                  read by all SMs via DSMEM
 *     K is iterated in BK_IN steps for GEMM-1, BK_OUT steps for GEMM-2.
 *
 *   SMEM layout per block:
 *     rank 0:  s_A[2][BM_LOCAL × BK_IN]   (double-buffered A tile)
 *              s_B [2][BN      × BK_IN]   (double-buffered B1 tile, SHARED via DSMEM)
 *              s_C_local[BM_LOCAL × BN]   (intermediate activation, local only)
 *              s_B2[2][BN × BK_OUT]       (double-buffered B2 tile, SHARED via DSMEM)
 *     rank 1..CLUSTER_SIZE-1:
 *              s_A[2][BM_LOCAL × BK_IN]   (each SM loads its own rows of A)
 *              s_B  → not allocated: reads rank-0's s_B via DSMEM
 *              s_C_local[BM_LOCAL × BN]
 *              s_B2 → reads rank-0's s_B2 via DSMEM
 *
 *   SMEM size:
 *     rank 0:  2*BM_LOCAL*stride_a + 2*BN*stride_b1 + BM_LOCAL*stride_c + 2*BN*stride_b2
 *     rank 1+: 2*BM_LOCAL*stride_a + BM_LOCAL*stride_c
 *     We allocate the SAME smem_bytes for all ranks (max of the two), with rank>0
 *     simply leaving the weight-tile region unused.
 *
 * ── Synchronisation ──────────────────────────────────────────────────────────
 *
 *   Only cluster.sync() is used, placed:
 *     1. After rank-0 finishes loading a B1/B2 tile (so other SMs can read via DSMEM)
 *     2. After all SMs finish consuming the tile (so rank-0 can overwrite it next iter)
 *   These are the minimum 2 barriers per tile iteration, replacing the O(N/BK_OUT)
 *   cluster.sync() calls per src_rank iteration in v4/v5.
 *
 * ── Hardware: RTX 5090 (GB202 SM 12.0) ──────────────────────────────────────
 *   - mma.sync.aligned.m16n8k16 (Ampere HMMA) — available on SM 12.0
 *   - cluster.map_shared_rank (DSMEM) — available since Hopper (SM 9.0+)
 *   - cp.async.ca + cp.async.commit/wait — available since Ampere
 *
 * Compile:
 *   nvcc -arch=sm_120 -O3 --use_fast_math -lcuda gemm_fuser_inter_sm_v7.cu -o v7
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <cooperative_groups.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace cg = cooperative_groups;

// ── Problem dimensions ────────────────────────────────────────────────────────
#ifndef FF_M_GLOBAL
#define FF_M_GLOBAL 128
#endif
#ifndef FF_K_GLOBAL
#define FF_K_GLOBAL 4096
#endif
#ifndef FF_N_GLOBAL
#define FF_N_GLOBAL 16384
#endif

constexpr int M_GLOBAL = FF_M_GLOBAL;
constexpr int K_GLOBAL = FF_K_GLOBAL;
constexpr int N_GLOBAL = FF_N_GLOBAL;

// ── Cluster / tile geometry ───────────────────────────────────────────────────
// CLUSTER_SIZE:  number of SMs in a cluster (each handles BM_LOCAL rows of M)
// BM_CLUSTER:    total M rows covered by one cluster  (= CLUSTER_SIZE * BM_LOCAL)
// BM_LOCAL:      M rows handled by ONE SM within the cluster
// BN:            N columns of B1/B2 tiles covered collectively by one cluster
//                (loaded once into rank-0's SMEM, read by all via DSMEM)
// BK_IN:         K-reduction tile width for GEMM-1 (A @ B1.T)
// BK_OUT:        K-reduction tile width for GEMM-2 (C @ B2.T)  [= BK_OUT cols of E]

#ifndef FF_CLUSTER_SIZE
#define FF_CLUSTER_SIZE 8
#endif
#ifndef FF_BM_LOCAL
#define FF_BM_LOCAL 16      // M rows per SM; BM_CLUSTER = 8*16 = 128
#endif
#ifndef FF_BN
#define FF_BN 128           // N columns per cluster tile; must divide N_GLOBAL
#endif
#ifndef FF_BK_IN
#define FF_BK_IN 64
#endif
#ifndef FF_BK_OUT
#define FF_BK_OUT 64        // K-output tile (K columns of E per cluster)
#endif
#ifndef FF_PAD
#define FF_PAD 8
#endif
// Consumer warp layout within BM_LOCAL × BK_OUT tile
// For mma.sync m16n8k16: MMA_M=16, MMA_N=8, MMA_K=16
// BM_LOCAL / MMA_M must be integer; BK_OUT / MMA_N must be integer.
// Default: WARPS_M=1, WARPS_N=4 → 4 warps × 32 = 128 threads/block
//   ROW_TILES  = BM_LOCAL/(WARPS_M*MMA_M)  = 16/(1*16) = 1
//   COL_TILES  = BK_OUT/(WARPS_N*MMA_N)    = 64/(4*8)  = 2
//   C_COL_TILES= BN/(WARPS_N*MMA_N)        = 128/(4*8) = 4
#ifndef FF_WARPS_M
#define FF_WARPS_M 1        // warps covering M within one SM
#endif
#ifndef FF_WARPS_N
#define FF_WARPS_N 4        // warps covering K-output (BK_OUT cols) within one SM
#endif

#ifndef FF_KERNEL_TAG
#define FF_KERNEL_TAG "fused_ffn_inter_sm_v7"
#endif

constexpr int CLUSTER_SIZE = FF_CLUSTER_SIZE;
constexpr int BM_LOCAL     = FF_BM_LOCAL;
constexpr int BM_CLUSTER   = CLUSTER_SIZE * BM_LOCAL;
constexpr int BN           = FF_BN;
constexpr int BK_IN        = FF_BK_IN;
constexpr int BK_OUT       = FF_BK_OUT;
constexpr int PAD          = FF_PAD;

// mma.sync.aligned.m16n8k16 tile sizes
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

constexpr int WARPS_M  = FF_WARPS_M;
constexpr int WARPS_N  = FF_WARPS_N;
constexpr int WARP_CNT = WARPS_M * WARPS_N;
constexpr int THREADS  = WARP_CNT * 32;

// Tiles per warp in the MMA output space
constexpr int ROW_TILES = BM_LOCAL / (WARPS_M * MMA_M);  // M rows per warp
constexpr int COL_TILES = BK_OUT   / (WARPS_N * MMA_N);  // N cols per warp (output K)

// Number of mma.sync accumulators per warp (phase-2 output E, and phase-1 C)
// Phase-1 C:  BM_LOCAL rows × BN cols  → tiles: (BM_LOCAL/MMA_M) × (BN/MMA_N) per warp
//   But we accumulate ALL N cols in the register file, so we need:
constexpr int C_COL_TILES = BN / (WARPS_N * MMA_N);
constexpr int FRAGS_C     = ROW_TILES * C_COL_TILES;   // phase-1 accumulators
constexpr int FRAGS_E     = ROW_TILES * COL_TILES;     // phase-2 accumulators

// Padded strides (in elements)
constexpr int STRIDE_A  = BK_IN  + PAD;   // row stride for s_A   [BM_LOCAL × BK_IN]
constexpr int STRIDE_B1 = BK_IN  + PAD;   // row stride for s_B1  [BN × BK_IN]  (B1 stored col-major in reduction dim)
constexpr int STRIDE_C  = BN     + PAD;   // row stride for s_C   [BM_LOCAL × BN]
constexpr int STRIDE_B2 = BK_OUT + PAD;   // row stride for s_B2  [BN × BK_OUT]

// smem sizes (in half elements)
constexpr int SZ_A  = BM_LOCAL * STRIDE_A;
constexpr int SZ_B1 = BN       * STRIDE_B1;
constexpr int SZ_C  = BM_LOCAL * STRIDE_C;
constexpr int SZ_B2 = BN       * STRIDE_B2;

// ── Static assertions ─────────────────────────────────────────────────────────
static_assert((M_GLOBAL  % BM_CLUSTER) == 0, "M_GLOBAL must be divisible by BM_CLUSTER");
static_assert((N_GLOBAL  % BN)         == 0, "N_GLOBAL must be divisible by BN");
static_assert((K_GLOBAL  % BK_IN)      == 0, "K_GLOBAL must be divisible by BK_IN");
static_assert((K_GLOBAL  % BK_OUT)     == 0, "K_GLOBAL must be divisible by BK_OUT");
static_assert((BM_LOCAL  % MMA_M)      == 0, "BM_LOCAL must be divisible by MMA_M");
static_assert((BN        % MMA_N)      == 0, "BN must be divisible by MMA_N");
static_assert((BK_IN     % MMA_K)      == 0, "BK_IN must be divisible by MMA_K");
static_assert((BK_OUT    % MMA_K)      == 0, "BK_OUT must be divisible by MMA_K");
static_assert((BK_OUT    % MMA_N)      == 0, "BK_OUT must be divisible by MMA_N");
static_assert((BM_LOCAL  % (WARPS_M * MMA_M)) == 0, "BM_LOCAL must be divisible by WARPS_M*MMA_M");
static_assert((BK_OUT    % (WARPS_N * MMA_N)) == 0, "BK_OUT must be divisible by WARPS_N*MMA_N");
static_assert((BN        % (WARPS_N * MMA_N)) == 0, "BN must be divisible by WARPS_N*MMA_N");
// With defaults: FRAGS_C = 1*4 = 4, FRAGS_E = 1*2 = 2
static_assert(FRAGS_C <= 64, "Too many phase-1 accumulators; reduce BN or increase WARPS_N");
static_assert(FRAGS_E <= 32, "Too many phase-2 accumulators; reduce BK_OUT or increase WARPS_N");

// ── CUDA helpers ──────────────────────────────────────────────────────────────
#define CUDA_CHECK(cmd) do {                                                   \
  cudaError_t _e = (cmd);                                                      \
  if (_e != cudaSuccess) {                                                     \
    std::cerr << "CUDA error: " << cudaGetErrorString(_e)                      \
              << " @ line " << __LINE__ << '\n';                               \
    std::exit(1);                                                               \
  }                                                                             \
} while (0)

// ── cp.async helpers ──────────────────────────────────────────────────────────
template <int Bytes>
__device__ __forceinline__ void cp_async_ca(void* smem_ptr, const void* gmem_ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    __pipeline_memcpy_async(smem_ptr, gmem_ptr, Bytes);
#else
    uint32_t smem = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n"
                 :: "r"(smem), "l"(gmem_ptr), "n"(Bytes));
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
__device__ __forceinline__ void cp_async_wait() {
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

// ── mma.sync.aligned.m16n8k16 helpers ────────────────────────────────────────
//
// PTX: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//   A operand: 4 × uint32 (row-major, 16×16 fp16 tile)
//   B operand: 2 × uint32 (col-major, 16×8  fp16 tile)
//   C/D:       4 × float
//
// A lane layout (m16n8k16, row-major):
//   lane l owns: a[0]=smem[r0*ld+kc], a[1]=smem[r0*ld+kc+8],
//                a[2]=smem[(r0+8)*ld+kc], a[3]=smem[(r0+8)*ld+kc+8]
//   where r0 = (l>>2)*2, kc = (l&3)*2
//
// B lane layout (m16n8k16, col-major, stored as row-major [N×K] in smem):
//   smem layout B1: [BN rows × BK_IN cols], stride = STRIDE_B1
//   lane l owns: b[0]=smem[nr*ld+kc], b[1]=smem[nr*ld+kc+8]
//   where nr = l>>2, kc = (l&3)*2
//
// D lane layout:
//   d[0]=D[r0,   c0], d[1]=D[r0,   c0+1],
//   d[2]=D[r0+8, c0], d[3]=D[r0+8, c0+1]
//   where r0=l>>2, c0=(l&3)*2

__device__ __forceinline__
void mma_m16n8k16(float (&d)[4],
                  const uint32_t (&a)[4],
                  const uint32_t (&b)[2],
                  const float    (&c)[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// Load A fragment (row-major, 16×16 fp16) from smem_tile = &smem[row_base*ld + k_base]
//
// PTX mma.sync.aligned.m16n8k16 row-major A-fragment register layout:
//   lane l owns two packed fp16 pairs:
//     a[0] = smem[r0      * ld + kc]     (rows 0..7,  k-cols 0,2,4,..,6 and 8,10,..,14 interleaved)
//     a[1] = smem[r0      * ld + kc + 8] (rows 0..7,  upper k half)
//     a[2] = smem[(r0+8)  * ld + kc]     (rows 8..15, lower k half)
//     a[3] = smem[(r0+8)  * ld + kc + 8] (rows 8..15, upper k half)
//   where r0 = lane >> 2   (NOT (lane>>2)*2),  kc = (lane & 3) * 2
//   r0 in [0..7], r0+8 in [8..15] — both within the 16-row MMA tile.
__device__ __forceinline__
void load_a_frag(uint32_t (&a)[4], const half* smem_tile, int ld)
{
    const int lane = threadIdx.x & 31;
    const int r0   = lane >> 2;          // row within first  8-row group [0..7]
    const int kc   = (lane & 3)  * 2;   // k-column [0,2,4,6]
    a[0] = *reinterpret_cast<const uint32_t*>(smem_tile +  r0      * ld + kc);
    a[1] = *reinterpret_cast<const uint32_t*>(smem_tile +  r0      * ld + kc + 8);
    a[2] = *reinterpret_cast<const uint32_t*>(smem_tile + (r0 + 8) * ld + kc);
    a[3] = *reinterpret_cast<const uint32_t*>(smem_tile + (r0 + 8) * ld + kc + 8);
}

// Load B fragment from row-major B[N×K] smem_tile = &smem[n_base*ld + k_base]
// B is row-major in smem, but used as col-major by the MMA instruction.
__device__ __forceinline__
void load_b_frag(uint32_t (&b)[2], const half* smem_tile, int ld)
{
    const int lane = threadIdx.x & 31;
    const int nr   = lane >> 2;
    const int kc   = (lane & 3) * 2;
    b[0] = *reinterpret_cast<const uint32_t*>(smem_tile + nr * ld + kc);
    b[1] = *reinterpret_cast<const uint32_t*>(smem_tile + nr * ld + kc + 8);
}

// Store C accumulator to smem with ReLU (for GEMM-2 A-operand preparation)
// smem_out = &smem[row_base*ld + col_base]
__device__ __forceinline__
void store_c_smem_relu(half* smem_out, int ld, const float (&d)[4])
{
    const int lane = threadIdx.x & 31;
    const int r0   = lane >> 2;
    const int c0   = (lane & 3) * 2;
    smem_out[ r0      * ld + c0]     = __float2half(d[0] > 0.f ? d[0] : 0.f);
    smem_out[ r0      * ld + c0 + 1] = __float2half(d[1] > 0.f ? d[1] : 0.f);
    smem_out[(r0 + 8) * ld + c0]     = __float2half(d[2] > 0.f ? d[2] : 0.f);
    smem_out[(r0 + 8) * ld + c0 + 1] = __float2half(d[3] > 0.f ? d[3] : 0.f);
}

// Atomic-add phase-2 accumulator to global memory E (fp32 atomicAdd)
// Each cluster contributes a partial sum; 128 clusters sum to produce final E.
// gmem_out = &E_fp32[grow * K_GLOBAL + gcol], row_stride = K_GLOBAL
__device__ __forceinline__
void atomic_add_e_gmem(float* gmem_out, int row_stride, const float (&d)[4])
{
    const int lane = threadIdx.x & 31;
    const int r0   = lane >> 2;
    const int c0   = (lane & 3) * 2;
    atomicAdd(&gmem_out[ r0      * row_stride + c0],     d[0]);
    atomicAdd(&gmem_out[ r0      * row_stride + c0 + 1], d[1]);
    atomicAdd(&gmem_out[(r0 + 8) * row_stride + c0],     d[2]);
    atomicAdd(&gmem_out[(r0 + 8) * row_stride + c0 + 1], d[3]);
}

// Convert fp32 E buffer to fp16 output E (separate pass, 1 thread per element)
__global__ void fp32_to_fp16_kernel(const float* __restrict__ src,
                                     half*        __restrict__ dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// ── Fill kernel ───────────────────────────────────────────────────────────────
__global__ void fill_kernel(half* p, int n, float v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = __float2half(v + 0.0001f * (i % 31));
}
__global__ void fill_fp32_kernel(float* p, int n, float v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

// ── Main fused kernel ─────────────────────────────────────────────────────────
//
// Algorithm: N-split grid, M-split within cluster, outer-product accumulation.
//
// Grid: (N_GLOBAL/BN * CLUSTER_SIZE, M_GLOBAL/BM_CLUSTER)
//   - Each cluster owns a fixed n_tile (BN columns of N) and BM_CLUSTER M-rows.
//   - Each rank within the cluster owns BM_LOCAL M-rows.
//
// Per-cluster computation:
//   GEMM-1: acc_C[BM_LOCAL × BN] += A[BM_LOCAL, :] @ B1[n_tile, :].T   (k_in loop)
//   ReLU applied to acc_C in-place (stored back to s_C in SMEM)
//   GEMM-2: for each k_out strip:
//     acc_E[BM_LOCAL × BK_OUT] += s_C[BM_LOCAL × BN] @ B2[n_tile, k_out:k_out+BK_OUT].T
//     atomicAdd partial acc_E to E_fp32[BM_LOCAL × BK_OUT]
//   Final pass: E_fp32 → E_fp16 (separate kernel)
//
// DSMEM usage: rank-0 loads B1/B2 tiles; all ranks read via cluster.map_shared_rank.
// A and s_C are private per-rank (no sharing needed).
//
// SMEM layout (same smem_bytes for all ranks — required for DSMEM pointer arithmetic):
//   s_B1[0]  : [BN × BK_IN]     rank-0 writes, all read via DSMEM
//   s_B1[1]  : [BN × BK_IN]     (double buffer — not used; kept for alignment)
//   s_B2     : [BN × BK_OUT]    rank-0 writes, all read via DSMEM
//   s_A[0]   : [BM_LOCAL × BK_IN] private per-rank
//   s_A[1]   : [BM_LOCAL × BK_IN] double buffer
//   s_C      : [BM_LOCAL × BN]  intermediate activation (post-GEMM1 + ReLU)

__global__ __cluster_dims__(CLUSTER_SIZE, 1, 1)
void fused_ffn_inter_sm_v7_kernel(
    const half* __restrict__ A,     // [M_GLOBAL × K_GLOBAL]
    const half* __restrict__ B1,    // [N_GLOBAL × K_GLOBAL]
    const half* __restrict__ B2,    // [N_GLOBAL × K_GLOBAL]
    float*      __restrict__ E_fp32) // [M_GLOBAL × K_GLOBAL]  (fp32 accumulation buffer)
{
    extern __shared__ __align__(16) half smem[];

    // ── SMEM pointers ─────────────────────────────────────────────────────────
    // Weight tiles at the FRONT of smem[] (same offset in all blocks → DSMEM works)
    half* s_B1 = smem;                              // [BN × BK_IN]
    half* s_B2 = smem + SZ_B1;                     // [BN × BK_OUT]
    // Private tiles after weight tiles
    half* s_A_pipe[2]  = { smem + SZ_B1 + SZ_B2,
                            smem + SZ_B1 + SZ_B2 + SZ_A };
    half* s_C          =   smem + SZ_B1 + SZ_B2 + 2 * SZ_A;

    // ── Thread/block/cluster indices ──────────────────────────────────────────
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = cluster.block_rank();  // 0 .. CLUSTER_SIZE-1
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    (void)(tid & 31);  // lane — used inside load_*_frag via threadIdx.x

    // M-rows owned by this block
    const int m_cluster_base = blockIdx.y * BM_CLUSTER;
    const int m_local_base   = m_cluster_base + rank * BM_LOCAL;

    // N-tile owned by this cluster
    const int n_tile = (blockIdx.x / CLUSTER_SIZE) * BN;

    // Warp layout within the MMA output [BM_LOCAL × (BN or BK_OUT)]
    const int warp_m = warp / WARPS_N;
    const int warp_n = warp % WARPS_N;

    // ── Phase-1 accumulators: acc_C[BM_LOCAL × BN] ───────────────────────────
    // Each warp covers ROW_TILES × C_COL_TILES MMA tiles
    float acc_C[FRAGS_C][4] = {};

    // ── GEMM-1 loop: acc_C += A[m_local, :] @ B1[n_tile, :].T ───────────────
    // Sync protocol per k_step:
    //   1. rank-0 issues cp_async for B1[n_tile, k_step]
    //   2. All ranks: cp_async for A[m_local, k_step] (independent)
    //   3. rank-0: cp_async_wait_all + __syncthreads
    //   4. cluster.sync()  ← B1 tile now visible in rank-0's SMEM via DSMEM
    //   5. All: MMA
    //   6. cluster.sync()  ← done consuming B1 tile

    // Pre-load first A tile
    for (int i = tid; i < BM_LOCAL * (BK_IN / 8); i += THREADS) {
        int r = i / (BK_IN / 8);
        int c = (i % (BK_IN / 8)) * 8;
        cp_async_ca<16>(&s_A_pipe[0][r * STRIDE_A + c],
                        &A[(m_local_base + r) * K_GLOBAL + c]);
    }
    cp_async_commit();

    for (int k_step = 0; k_step < K_GLOBAL; k_step += BK_IN) {
        const int curr   = (k_step / BK_IN) & 1;
        const int nxt    = 1 - curr;
        const int next_k = k_step + BK_IN;

        // rank-0: load B1 tile for current k_step into s_B1
        if (rank == 0) {
            for (int i = tid; i < BN * (BK_IN / 8); i += THREADS) {
                int r = i / (BK_IN / 8);
                int c = (i % (BK_IN / 8)) * 8;
                cp_async_ca<16>(&s_B1[r * STRIDE_B1 + c],
                                &B1[(n_tile + r) * K_GLOBAL + (k_step + c)]);
            }
            cp_async_commit();
        }

        // All ranks: prefetch next A tile
        if (next_k < K_GLOBAL) {
            for (int i = tid; i < BM_LOCAL * (BK_IN / 8); i += THREADS) {
                int r = i / (BK_IN / 8);
                int c = (i % (BK_IN / 8)) * 8;
                cp_async_ca<16>(&s_A_pipe[nxt][r * STRIDE_A + c],
                                &A[(m_local_base + r) * K_GLOBAL + (next_k + c)]);
            }
            cp_async_commit();
        }

        // Wait for current A tile to land
        if (next_k < K_GLOBAL) cp_async_wait<1>();
        else                    cp_async_wait_all();
        __syncthreads();

        // rank-0: ensure B1 is also done
        if (rank == 0) { cp_async_wait_all(); __syncthreads(); }

        // Expose rank-0's s_B1 to cluster
        cluster.sync();

        half* b1_src = (rank == 0) ? s_B1
                                   : cluster.map_shared_rank(s_B1, 0);

        // MMA: acc_C += A_tile @ B1_tile.T
        const int row_off = warp_m * ROW_TILES  * MMA_M;
        const int col_off = warp_n * C_COL_TILES * MMA_N;
        #pragma unroll
        for (int ki = 0; ki < BK_IN; ki += MMA_K) {
            #pragma unroll
            for (int mi = 0; mi < ROW_TILES; ++mi) {
                uint32_t a_frag[4];
                load_a_frag(a_frag,
                            s_A_pipe[curr] + (row_off + mi * MMA_M) * STRIDE_A + ki,
                            STRIDE_A);
                #pragma unroll
                for (int nj = 0; nj < C_COL_TILES; ++nj) {
                    uint32_t b_frag[2];
                    load_b_frag(b_frag,
                                b1_src + (col_off + nj * MMA_N) * STRIDE_B1 + ki,
                                STRIDE_B1);
                    mma_m16n8k16(acc_C[mi * C_COL_TILES + nj],
                                 a_frag, b_frag,
                                 acc_C[mi * C_COL_TILES + nj]);
                }
            }
        }

        // Signal done consuming B1 tile
        cluster.sync();
    }

    cp_async_wait_all();
    __syncthreads();

    // ── ReLU + store acc_C → s_C ──────────────────────────────────────────────
    {
        const int row_off = warp_m * ROW_TILES  * MMA_M;
        const int col_off = warp_n * C_COL_TILES * MMA_N;
        #pragma unroll
        for (int mi = 0; mi < ROW_TILES; ++mi) {
            #pragma unroll
            for (int nj = 0; nj < C_COL_TILES; ++nj) {
                store_c_smem_relu(
                    s_C + (row_off + mi * MMA_M) * STRIDE_C + (col_off + nj * MMA_N),
                    STRIDE_C,
                    acc_C[mi * C_COL_TILES + nj]);
            }
        }
    }
    __syncthreads();

    // ── GEMM-2 loop: for each k_out strip, acc_E += s_C @ B2[n_tile, k_out].T ──
    // acc_E is atomicAdd-ed into E_fp32 (fp32 accumulation across all n_tile clusters)
    // rank-0 loads B2[n_tile, k_out] → s_B2; all ranks read via DSMEM

    for (int k_out = 0; k_out < K_GLOBAL; k_out += BK_OUT) {
        float acc_E[FRAGS_E][4] = {};

        // rank-0: load B2 tile for this k_out
        if (rank == 0) {
            for (int i = tid; i < BN * (BK_OUT / 8); i += THREADS) {
                int r = i / (BK_OUT / 8);
                int c = (i % (BK_OUT / 8)) * 8;
                cp_async_ca<16>(&s_B2[r * STRIDE_B2 + c],
                                &B2[(n_tile + r) * K_GLOBAL + (k_out + c)]);
            }
            cp_async_commit();
            cp_async_wait_all();
        }
        __syncthreads();
        cluster.sync();  // ← B2 tile now visible via DSMEM

        half* b2_src = (rank == 0) ? s_B2
                                   : cluster.map_shared_rank(s_B2, 0);

        // MMA: acc_E += s_C @ B2_tile.T
        // s_C is A-operand [BM_LOCAL × BN], B2 is B-operand [BN × BK_OUT]
        const int row_off = warp_m * ROW_TILES * MMA_M;
        const int col_off = warp_n * COL_TILES  * MMA_N;
        #pragma unroll
        for (int ki = 0; ki < BN; ki += MMA_K) {
            #pragma unroll
            for (int mi = 0; mi < ROW_TILES; ++mi) {
                uint32_t a_frag[4];
                // s_C is [BM_LOCAL × BN] with stride STRIDE_C
                // A-operand row: row_off + mi*MMA_M, k-dim: ki
                load_a_frag(a_frag,
                            s_C + (row_off + mi * MMA_M) * STRIDE_C + ki,
                            STRIDE_C);
                #pragma unroll
                for (int nj = 0; nj < COL_TILES; ++nj) {
                    uint32_t b_frag[2];
                    // B2 tile: [BN × BK_OUT], A-operand is ki row, B-operand col is col_off+nj*MMA_N
                    load_b_frag(b_frag,
                                b2_src + ki * STRIDE_B2 + (col_off + nj * MMA_N),
                                STRIDE_B2);
                    mma_m16n8k16(acc_E[mi * COL_TILES + nj],
                                 a_frag, b_frag,
                                 acc_E[mi * COL_TILES + nj]);
                }
            }
        }

        // Signal done consuming B2 tile
        cluster.sync();

        // atomicAdd partial acc_E into E_fp32
        #pragma unroll
        for (int mi = 0; mi < ROW_TILES; ++mi) {
            #pragma unroll
            for (int nj = 0; nj < COL_TILES; ++nj) {
                int grow = m_local_base + row_off + mi * MMA_M;
                int gcol = k_out        + col_off + nj * MMA_N;
                atomic_add_e_gmem(
                    &E_fp32[grow * K_GLOBAL + gcol],
                    K_GLOBAL,
                    acc_E[mi * COL_TILES + nj]);
            }
        }
    }
}

// ── Shared memory size ────────────────────────────────────────────────────────
static size_t compute_smem_bytes() {
    // Weight tiles (at front — fixed offset for DSMEM):
    //   s_B1 : [BN × BK_IN]
    //   s_B2 : [BN × BK_OUT]
    // Private tiles:
    //   2 × s_A: [BM_LOCAL × BK_IN] double-buffered
    //   s_C:     [BM_LOCAL × BN]
    return sizeof(half) * (SZ_B1 + SZ_B2 + 2 * SZ_A + SZ_C);
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    int warmup = 20;
    int iters  = 100;
    if (argc >= 2) warmup = std::atoi(argv[1]);
    if (argc >= 3) iters  = std::atoi(argv[2]);

    CUDA_CHECK(cudaSetDevice(0));

    int cluster_ok = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&cluster_ok, cudaDevAttrClusterLaunch, 0));
    if (!cluster_ok) {
        std::cerr << "Device does not support thread-block cluster launch (DSMEM).\n";
        return 1;
    }

    int max_smem = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    CUDA_CHECK(cudaFuncSetAttribute(
        fused_ffn_inter_sm_v7_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem));
    CUDA_CHECK(cudaFuncSetAttribute(
        fused_ffn_inter_sm_v7_kernel,
        cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

    const size_t smem_bytes = compute_smem_bytes();
    if ((int)smem_bytes > max_smem) {
        std::cerr << "smem_bytes=" << smem_bytes << " > max_smem=" << max_smem << '\n';
        return 1;
    }

    half  *dA, *dB1, *dB2, *dE;
    float *dE_fp32;
    CUDA_CHECK(cudaMalloc(&dA,      sizeof(half)  * M_GLOBAL * K_GLOBAL));
    CUDA_CHECK(cudaMalloc(&dB1,     sizeof(half)  * N_GLOBAL * K_GLOBAL));
    CUDA_CHECK(cudaMalloc(&dB2,     sizeof(half)  * N_GLOBAL * K_GLOBAL));
    CUDA_CHECK(cudaMalloc(&dE,      sizeof(half)  * M_GLOBAL * K_GLOBAL));
    CUDA_CHECK(cudaMalloc(&dE_fp32, sizeof(float) * M_GLOBAL * K_GLOBAL));

    fill_kernel<<<(M_GLOBAL * K_GLOBAL + 255) / 256, 256>>>(dA,  M_GLOBAL * K_GLOBAL, 0.5f);
    fill_kernel<<<(N_GLOBAL * K_GLOBAL + 255) / 256, 256>>>(dB1, N_GLOBAL * K_GLOBAL, 0.25f);
    fill_kernel<<<(N_GLOBAL * K_GLOBAL + 255) / 256, 256>>>(dB2, N_GLOBAL * K_GLOBAL, 0.125f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Grid: (N_GLOBAL/BN * CLUSTER_SIZE, M_GLOBAL/BM_CLUSTER)
    const int grid_x = N_GLOBAL / BN;
    const int grid_y = M_GLOBAL / BM_CLUSTER;
    dim3 block(THREADS);
    dim3 grid(grid_x * CLUSTER_SIZE, grid_y, 1);

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim          = grid;
    cfg.blockDim         = block;
    cfg.dynamicSmemBytes = smem_bytes;
    cudaLaunchAttribute attr[1];
    attr[0].id               = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = CLUSTER_SIZE;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs    = attr;
    cfg.numAttrs = 1;

    const int E_elems  = M_GLOBAL * K_GLOBAL;
    const int fill_blk = (E_elems + 255) / 256;

    auto launch_once = [&]() -> cudaError_t {
        // Zero the fp32 accumulation buffer
        fill_fp32_kernel<<<fill_blk, 256>>>(dE_fp32, E_elems, 0.f);
        cudaError_t e = cudaLaunchKernelEx(&cfg, fused_ffn_inter_sm_v7_kernel,
                                            dA, dB1, dB2, dE_fp32);
        // Convert fp32 → fp16
        fp32_to_fp16_kernel<<<fill_blk, 256>>>(dE_fp32, dE, E_elems);
        return e;
    };

    std::cout << "Kernel=" << FF_KERNEL_TAG << '\n'
              << "Shape(M,K,N)=" << M_GLOBAL << ',' << K_GLOBAL << ',' << N_GLOBAL << '\n'
              << "BM_LOCAL=" << BM_LOCAL << " BM_CLUSTER=" << BM_CLUSTER
              << " BN=" << BN << " BK_IN=" << BK_IN << " BK_OUT=" << BK_OUT << '\n'
              << "CLUSTER_SIZE=" << CLUSTER_SIZE
              << " WARPS_M=" << WARPS_M << " WARPS_N=" << WARPS_N
              << " THREADS=" << THREADS << '\n'
              << "GridClusters=" << grid_x << " x " << grid_y
              << "  TotalBlocks=" << (grid_x * grid_y * CLUSTER_SIZE) << '\n'
              << "DynamicSmemKB=" << (smem_bytes / 1024.0f) << '\n';

    // Warmup
    for (int i = 0; i < warmup; ++i) CUDA_CHECK(launch_once());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) CUDA_CHECK(launch_once());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float elapsed_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, t0, t1));
    const float avg_ms = elapsed_ms / iters;

    const double flops  = 2.0 * M_GLOBAL * K_GLOBAL * N_GLOBAL * 2.0;
    const double tflops = flops / (avg_ms * 1e-3) / 1e12;

    std::cout << "Average Latency: " << avg_ms << " ms\n"
              << "LATENCY_MS=" << avg_ms << '\n'
              << "THROUGHPUT_TFLOPS=" << tflops << '\n';

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB1));
    CUDA_CHECK(cudaFree(dB2));
    CUDA_CHECK(cudaFree(dE));
    CUDA_CHECK(cudaFree(dE_fp32));
    return 0;
}
