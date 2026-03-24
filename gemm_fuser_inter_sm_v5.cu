/*
 * gemm_fuser_inter_sm_v5.cu
 *
 * Inter-SM fused FFN:  E = ReLU(A @ B1.T) @ B2.T
 *   A   : [M, K]      input activations (row-major)
 *   B1  : [N, K]      FC1 weight stored as [N, K] (B1[n,k]), so A @ B1.T loads B1 col-wise
 *   B2  : [N, K]      FC2 weight stored as [N, K] (B2[n,k])
 *   E   : [M, K]      output
 *
 * Improvements over v4:
 *   1. TMA (cp.async.bulk.tensor.2d) replaces per-thread cp.async.ca,
 *      reducing instruction count and address arithmetic overhead.
 *   2. mbarrier replaces cp_async_wait_group + __syncthreads for finer-grained
 *      producer-consumer synchronization.
 *   3. Warp specialization: warp 0 is a dedicated TMA producer;
 *      warps 1..C_WARPS are MMA consumers.  The producer issues the next
 *      tile's TMA while consumers compute on the current tile.
 *   4. mma.sync.aligned.m16n8k16.f32.f16.f16.f32 PTX replaces the WMMA C++
 *      API, giving the compiler explicit register-file visibility.
 *
 * Hardware: RTX 5090 (GB202, SM_12.0, consumer Blackwell).
 *   Confirmed working on sm_120:
 *     cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
 *     mbarrier.{init,arrive.expect_tx.release.cta,try_wait.parity}.shared::cta.b64
 *     mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
 *     cluster.map_shared_rank (DSMEM)
 *
 * Compile:
 *   nvcc -arch=sm_120 -O3 --use_fast_math -lcuda gemm_fuser_inter_sm_v5.cu -o v5
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace cg = cooperative_groups;

// ---- Problem dimensions ----
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

// ---- Tile geometry ----
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
// Consumer warp layout within the BM x BK_OUT tile
// WARPS_M * WARPS_N consumer warps + 1 producer warp
#ifndef FF_WARPS_M
#define FF_WARPS_M 1
#endif
#ifndef FF_WARPS_N
#define FF_WARPS_N 4
#endif

constexpr int BM         = FF_BM;
constexpr int BK_IN      = FF_BK_IN;
constexpr int BK_OUT     = FF_BK_OUT;
constexpr int CLUSTER_SIZE = FF_CLUSTER_SIZE;
constexpr int BN_STEP    = CLUSTER_SIZE * BK_OUT;
constexpr int PAD        = FF_PAD;

// mma.sync m16n8k16 ISA tile sizes
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

constexpr int WARPS_M = FF_WARPS_M;
constexpr int WARPS_N = FF_WARPS_N;
constexpr int C_WARPS = WARPS_M * WARPS_N;        // consumer warps
constexpr int THREADS = (C_WARPS + 1) * 32;       // +1 producer warp

// Tiles per consumer warp
constexpr int ROW_TILES = BM     / (WARPS_M * MMA_M); // 32/(1*16)=2
constexpr int COL_TILES = BK_OUT / (WARPS_N * MMA_N); // 128/(4*8)=4
constexpr int FRAGS     = ROW_TILES * COL_TILES;        // 8

// ---- Static checks ----
static_assert((M_GLOBAL % BM) == 0,               "M_GLOBAL % BM != 0");
static_assert((K_GLOBAL % BK_OUT) == 0,            "K_GLOBAL % BK_OUT != 0");
static_assert((K_GLOBAL % BK_IN) == 0,             "K_GLOBAL % BK_IN != 0");
static_assert((BK_IN  % MMA_K) == 0,              "BK_IN % MMA_K != 0");
static_assert((BK_OUT % MMA_K) == 0,              "BK_OUT % MMA_K != 0");
static_assert((BM    % (WARPS_M * MMA_M)) == 0,   "BM % (WARPS_M*MMA_M) != 0");
static_assert((BK_OUT % (WARPS_N * MMA_N)) == 0,  "BK_OUT % (WARPS_N*MMA_N) != 0");
static_assert((N_GLOBAL % BN_STEP) == 0,          "N_GLOBAL % BN_STEP != 0");

// ---- Error helpers ----
#define CUDA_CHECK(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess){                    \
  std::cerr<<"CUDA "<<cudaGetErrorString(_e)<<" @"<<__LINE__<<'\n'; std::exit(1);} } while(0)
#define CU_CHECK(x)   do { CUresult _r=(x); if(_r!=CUDA_SUCCESS){                      \
  const char*_s=nullptr; cuGetErrorString(_r,&_s);                                     \
  std::cerr<<"CU "<<(_s?_s:"?")<<" @"<<__LINE__<<'\n'; std::exit(1);} } while(0)

// ============================================================
// mbarrier helpers — correct PTX for sm_120
// ============================================================
__device__ __forceinline__ void mbar_init(uint64_t* mbar, uint32_t count) {
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(count)
        : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t* mbar) {
    uint64_t state;
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 %0, [%1];"
        : "=l"(state)
        : "r"((uint32_t)__cvta_generic_to_shared(mbar))
        : "memory");
    (void)state;
}

// Arrive AND pre-declare expected TMA transaction bytes.
// Must be called before tma_load_2d for the barrier to complete correctly.
__device__ __forceinline__ void mbar_arrive_tx(uint64_t* mbar, uint32_t tx_bytes) {
    uint64_t state;
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;"
        : "=l"(state)
        : "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(tx_bytes)
        : "memory");
    (void)state;
}

__device__ __forceinline__ void mbar_wait(uint64_t* mbar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "LAB_WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "@!P bra LAB_WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(phase)
        : "memory");
}

// ============================================================
// TMA 2D tile load (from rank-3 tensor map with degenerate batch dim=1)
// coord convention (innermost first): {col_offset, row_offset, 0}
// ============================================================
__device__ __forceinline__ void tma_load_2d(
    void*              smem_dst,
    const CUtensorMap* tmap,
    int32_t            col_off,
    int32_t            row_off,
    uint64_t*          mbar)
{
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem_dst)),
           "l"((uint64_t)tmap),
           "r"(col_off), "r"(row_off), "r"((int32_t)0),
           "r"((uint32_t)__cvta_generic_to_shared(mbar))
        : "memory");
}

// ============================================================
// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
// A: 4 x uint32, B: 2 x uint32, D/C: 4 x float
// ============================================================
__device__ __forceinline__ void mma_m16n8k16(
    float        (&d)[4],
    const uint32_t (&a)[4],
    const uint32_t (&b)[2],
    const float    (&c)[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d[0]),"=f"(d[1]),"=f"(d[2]),"=f"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),
          "r"(b[0]),"r"(b[1]),
          "f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]));
}

// ============================================================
// Fragment loaders
//
// A operand: row-major [16 rows x 16 k-cols], stored in smem as
//            smem[row * ld + k_col] (fp16 elements).
// PTX layout for m16n8k16 row A:
//   lane l: a[0] = smem[(l>>2)*2     * ld + (l&3)*2]  (packed 2 fp16)
//            a[1] = smem[(l>>2)*2     * ld + (l&3)*2+8]
//            a[2] = smem[(l>>2)*2 + 8 * ld + (l&3)*2]
//            a[3] = smem[(l>>2)*2 + 8 * ld + (l&3)*2+8]
// ============================================================
__device__ __forceinline__ void load_a_frag(
    uint32_t       (&a)[4],
    const half*    smem_tile,  // &smem[row_base * ld + k_base]
    int            ld)
{
    const int lane = threadIdx.x & 31;
    const int r0   = (lane >> 2) * 2;
    const int kc   = (lane & 3)  * 2;
    a[0] = *reinterpret_cast<const uint32_t*>(smem_tile +  r0      * ld + kc);
    a[1] = *reinterpret_cast<const uint32_t*>(smem_tile +  r0      * ld + kc + 8);
    a[2] = *reinterpret_cast<const uint32_t*>(smem_tile + (r0 + 8) * ld + kc);
    a[3] = *reinterpret_cast<const uint32_t*>(smem_tile + (r0 + 8) * ld + kc + 8);
}

// B operand: col-major [16 k-rows x 8 n-cols].
// Stored in smem as B[n, k] row-major (B1/B2 layout), so smem[n_row * ld + k_col].
// PTX layout for m16n8k16 col B:
//   lane l: b[0] = smem[(l>>2) * ld + (l&3)*2]     (n_row = l>>2, k_col pairs)
//            b[1] = smem[(l>>2) * ld + (l&3)*2 + 8]
// smem_tile = &smem[n_base * ld + k_base]
__device__ __forceinline__ void load_b_frag(
    uint32_t       (&b)[2],
    const half*    smem_tile,  // &smem[n_base * ld + k_base]
    int            ld)
{
    const int lane = threadIdx.x & 31;
    const int nr   = lane >> 2;      // n-row index within 8-row tile
    const int kc   = (lane & 3) * 2; // k-col index
    b[0] = *reinterpret_cast<const uint32_t*>(smem_tile + nr * ld + kc);
    b[1] = *reinterpret_cast<const uint32_t*>(smem_tile + nr * ld + kc + 8);
}

// ============================================================
// Accumulator store helpers
//
// D layout for m16n8k16 (per lane l):
//   d[0] = D[l>>2,         (l&3)*2    ]
//   d[1] = D[l>>2,         (l&3)*2 + 1]
//   d[2] = D[l>>2 + 8,     (l&3)*2    ]
//   d[3] = D[l>>2 + 8,     (l&3)*2 + 1]
// ============================================================
__device__ __forceinline__ void store_accum_smem_relu(
    half*          smem_out,   // &smem[row_base * ld + col_base]
    int            ld,
    const float    (&d)[4])
{
    const int lane = threadIdx.x & 31;
    const int r0   = lane >> 2;
    const int c0   = (lane & 3) * 2;
    auto relu16 = [](float v) { return __float2half(v > 0.f ? v : 0.f); };
    smem_out[ r0      * ld + c0]     = relu16(d[0]);
    smem_out[ r0      * ld + c0 + 1] = relu16(d[1]);
    smem_out[(r0 + 8) * ld + c0]     = relu16(d[2]);
    smem_out[(r0 + 8) * ld + c0 + 1] = relu16(d[3]);
}

__device__ __forceinline__ void store_accum_gmem(
    half*          gmem_out,   // &E[grow * K_GLOBAL + gcol]
    int            row_stride,
    const float    (&d)[4])
{
    const int lane = threadIdx.x & 31;
    const int r0   = lane >> 2;
    const int c0   = (lane & 3) * 2;
    gmem_out[ r0      * row_stride + c0]     = __float2half(d[0]);
    gmem_out[ r0      * row_stride + c0 + 1] = __float2half(d[1]);
    gmem_out[(r0 + 8) * row_stride + c0]     = __float2half(d[2]);
    gmem_out[(r0 + 8) * row_stride + c0 + 1] = __float2half(d[3]);
}

// ============================================================
// Utility
// ============================================================
__global__ void fill_kernel(half* p, int n, float v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = __float2half(v + 0.0001f * (i % 31));
}

// ============================================================
// TMA map creation (host)
// cuTensorMapEncodeTiled requires rank >= 3.  We use rank=3 with a
// degenerate outer dimension of 1 and zero padding for the extra dim.
//   dim layout (innermost first): {cols, rows, 1}
//   strides (rank-1=2 entries):   {cols*sizeof(half), rows*cols*sizeof(half)}
// ============================================================
static CUtensorMap create_tma_map(
    const void* ptr, uint64_t rows, uint64_t cols,
    uint32_t tile_rows, uint32_t tile_cols)
{
    CUtensorMap tm;
    // rank=3: dims[0]=cols (innermost), dims[1]=rows, dims[2]=1 (batch)
    uint64_t g_dims[3]    = {cols, rows, 1};
    // strides: [0] = byte stride between dim-0 slices (rows), [1] = stride between dim-1 slices
    uint64_t g_strides[2] = {cols * sizeof(half),                  // stride along rows
                              rows * cols * sizeof(half)};          // stride along batch (unused)
    uint32_t b_dims[3]    = {tile_cols, tile_rows, 1};
    uint32_t b_strides[3] = {1, 1, 1};
    CU_CHECK(cuTensorMapEncodeTiled(
        &tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3,
        const_cast<void*>(ptr),
        g_dims, g_strides, b_dims, b_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tm;
}

// ============================================================
// Shared memory size
// ============================================================
static size_t compute_smem_bytes() {
    constexpr int sa  = BK_IN  + PAD;
    constexpr int sb1 = BK_IN  + PAD;  // B1 [N×K]: stride over K dimension
    constexpr int sb2 = BK_OUT + PAD;  // B2 [N×K]: stride over K dimension
    constexpr int sc  = BK_OUT + PAD;
    return 128
        + 2 * sizeof(half) * BM    * sa   // s_A  double buffer
        + 2 * sizeof(half) * BK_OUT * sb1 // s_B1 double buffer  (BK_OUT rows of K=BK_IN)
        + sizeof(half) * BK_OUT * sb2     // s_B2
        + sizeof(half) * BM    * sc       // s_C_local
        + sizeof(half) * BM    * sc;      // s_C_stage
}

// ============================================================
// Kernel
// ============================================================
__global__ __cluster_dims__(CLUSTER_SIZE, 1, 1)
void fused_ffn_inter_sm_v5_kernel(
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B1,
    const __grid_constant__ CUtensorMap tma_B2,
    half* __restrict__ E)
{
    // stride = leading dimension (elements) including padding
    constexpr int stride_a  = BK_IN  + PAD;
    constexpr int stride_b1 = BK_IN  + PAD;  // B1 stored [N x K]: stride over K
    constexpr int stride_b2 = BK_OUT + PAD;  // B2 stored [N x K]: stride over K
    constexpr int stride_c  = BK_OUT + PAD;

    constexpr int size_a  = BM    * stride_a;
    constexpr int size_b1 = BK_OUT * stride_b1; // BK_OUT n-rows, BK_IN k-cols
    constexpr int size_b2 = BK_OUT * stride_b2;
    constexpr int size_c  = BM    * stride_c;

    extern __shared__ __align__(128) char smem_raw[];

    // mbarrier region
    uint64_t* mbar_A  = (uint64_t*)(smem_raw);      // [2]
    uint64_t* mbar_B1 = (uint64_t*)(smem_raw + 16); // [2]
    uint64_t* mbar_B2 = (uint64_t*)(smem_raw + 32); // [1]

    // Tile buffers (starting at 128B offset)
    char* p = smem_raw + 128;
    half* s_A0  = (half*)p; p += sizeof(half) * size_a;
    half* s_A1  = (half*)p; p += sizeof(half) * size_a;
    half* s_B10 = (half*)p; p += sizeof(half) * size_b1;
    half* s_B11 = (half*)p; p += sizeof(half) * size_b1;
    half* s_B2  = (half*)p; p += sizeof(half) * size_b2;
    half* s_C_local = (half*)p; p += sizeof(half) * size_c;
    half* s_C_stage = (half*)p;

    half* s_A_pipe[2]  = {s_A0, s_A1};
    half* s_B1_pipe[2] = {s_B10, s_B11};

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const bool is_prod = (warp_id == 0);
    // consumer warp index [0..C_WARPS-1] — valid only when !is_prod
    const int c_warp   = warp_id - 1;
    const int c_warp_m = c_warp / WARPS_N;
    const int c_warp_n = c_warp % WARPS_N;
    const int row_base = c_warp_m * ROW_TILES * MMA_M; // row within BM tile
    const int col_base = c_warp_n * COL_TILES * MMA_N; // col within BK_OUT tile

    // Cluster / problem indices
    cg::cluster_group cluster = cg::this_cluster();
    const int rank   = cluster.block_rank();
    const int m_tile = blockIdx.y * BM;
    const int k_tile = (blockIdx.x / CLUSTER_SIZE) * (BK_OUT * CLUSTER_SIZE) + rank * BK_OUT;

    // ---- init mbarriers ----
    if (tid == 0) {
        mbar_init(&mbar_A[0], 1);
        mbar_init(&mbar_A[1], 1);
        mbar_init(&mbar_B1[0], 1);
        mbar_init(&mbar_B1[1], 1);
        mbar_init(&mbar_B2[0], 1);
    }
    __syncthreads();

    // ---- Phase-2 accumulator (persistent across n_steps) ----
    float acc_E[FRAGS][4] = {};

    // ================================================================
    // Outer loop: N-dimension in cluster strides
    // ================================================================
    for (int n_step = 0; n_step < N_GLOBAL; n_step += BN_STEP) {
        const int n_local = n_step + rank * BK_OUT; // this block's N-tile start

        float acc_C[FRAGS][4] = {};

        // ---- Pre-issue first tile (k_step=0) ----
        if (is_prod && tid == 0) {
            mbar_arrive_tx(&mbar_A[0],  (uint32_t)(BM    * BK_IN  * sizeof(half)));
            mbar_arrive_tx(&mbar_B1[0], (uint32_t)(BK_OUT * BK_IN  * sizeof(half)));
            // A[m_tile:m_tile+BM, 0:BK_IN]  — TMA coord: col=0, row=m_tile
            tma_load_2d(s_A_pipe[0], &tma_A, 0, m_tile, &mbar_A[0]);
            // B1[n_local:n_local+BK_OUT, 0:BK_IN] — TMA coord: col=0, row=n_local
            tma_load_2d(s_B1_pipe[0], &tma_B1, 0, n_local, &mbar_B1[0]);
        }

        // ================================================================
        // Phase-1 k-loop
        // ================================================================
        for (int k_step = 0; k_step < K_GLOBAL; k_step += BK_IN) {
            const int curr    = (k_step / BK_IN) & 1;
            const int nxt     = 1 - curr;
            const int next_k  = k_step + BK_IN;
            // mbarrier parity semantics: after N arrives, barrier phase toggles.
            // A freshly init'd barrier starts at phase=0.  After the 1st completion,
            // phase = 1.  After 2nd completion, phase = 0.  etc.
            // try_wait.parity(p) returns true when barrier's phase currently equals p,
            // i.e., it has completed with parity p.
            // For a double-buffered pipeline, each buffer is re-init'd before reuse,
            // so it always starts at phase=0 and completes to phase=1.
            // Consumers always wait with phase=1 (waiting for the first completion).
            const int wait_phase = 1;

            // Producer: pre-fetch next tile
            if (is_prod && tid == 0 && next_k < K_GLOBAL) {
                mbar_init(&mbar_A[nxt], 1);
                mbar_init(&mbar_B1[nxt], 1);
                mbar_arrive_tx(&mbar_A[nxt],  (uint32_t)(BM    * BK_IN  * sizeof(half)));
                mbar_arrive_tx(&mbar_B1[nxt], (uint32_t)(BK_OUT * BK_IN  * sizeof(half)));
                tma_load_2d(s_A_pipe[nxt],  &tma_A,  next_k, m_tile,  &mbar_A[nxt]);
                tma_load_2d(s_B1_pipe[nxt], &tma_B1, next_k, n_local, &mbar_B1[nxt]);
            }

            // Consumers: wait, then compute
            if (!is_prod) {
                mbar_wait(&mbar_A[curr],  wait_phase);
                mbar_wait(&mbar_B1[curr], wait_phase);

                #pragma unroll
                for (int ki = 0; ki < BK_IN; ki += MMA_K) {
                    uint32_t a_frag[4], b_frag[2];
                    #pragma unroll
                    for (int mi = 0; mi < ROW_TILES; ++mi) {
                        load_a_frag(a_frag,
                            s_A_pipe[curr] + (row_base + mi * MMA_M) * stride_a + ki,
                            stride_a);
                        #pragma unroll
                        for (int nj = 0; nj < COL_TILES; ++nj) {
                            // B1 stored [N x K] → s_B1[n * stride_b1 + k]
                            load_b_frag(b_frag,
                                s_B1_pipe[curr] + (col_base + nj * MMA_N) * stride_b1 + ki,
                                stride_b1);
                            mma_m16n8k16(acc_C[mi * COL_TILES + nj],
                                         a_frag, b_frag,
                                         acc_C[mi * COL_TILES + nj]);
                        }
                    }
                }
            }
            __syncthreads();
        }

        // Consumer: ReLU → s_C_local
        if (!is_prod) {
            #pragma unroll
            for (int mi = 0; mi < ROW_TILES; ++mi) {
                #pragma unroll
                for (int nj = 0; nj < COL_TILES; ++nj) {
                    store_accum_smem_relu(
                        s_C_local + (row_base + mi * MMA_M) * stride_c + (col_base + nj * MMA_N),
                        stride_c,
                        acc_C[mi * COL_TILES + nj]);
                }
            }
        }
        __syncthreads();

        cluster.sync(); // all blocks have written their s_C_local

        // ================================================================
        // Phase-2: accumulate E += C_local @ B2 over all CLUSTER_SIZE blocks
        // ================================================================
        for (int src_rank = 0; src_rank < CLUSTER_SIZE; ++src_rank) {
            const int n_src = n_step + src_rank * BK_OUT;

            // Re-arm B2 mbarrier
            if (tid == 0) {
                mbar_init(&mbar_B2[0], 1);
            }
            __syncthreads();

            // Producer: TMA load B2[n_src:n_src+BK_OUT, k_tile:k_tile+BK_OUT]
            if (is_prod && tid == 0) {
                mbar_arrive_tx(&mbar_B2[0], (uint32_t)(BK_OUT * BK_OUT * sizeof(half)));
                tma_load_2d(s_B2, &tma_B2, k_tile, n_src, &mbar_B2[0]);
            }

            // All threads: DSMEM copy C_local from src_rank into s_C_stage
            {
                half* src_c = (src_rank == rank)
                    ? s_C_local
                    : cluster.map_shared_rank(s_C_local, src_rank);
                int4* src_v = reinterpret_cast<int4*>(src_c);
                int4* dst_v = reinterpret_cast<int4*>(s_C_stage);
                const int nv = (BM * stride_c) / 8;
                for (int vi = tid; vi < nv; vi += THREADS)
                    dst_v[vi] = src_v[vi];
            }

            // Wait: consumers need both s_C_stage and s_B2
            if (!is_prod) mbar_wait(&mbar_B2[0], 1);  // phase=1 = first completion
            __syncthreads(); // ensure both C_stage copy and B2 load done

            // Consumer: Phase-2 mma.sync
            if (!is_prod) {
                #pragma unroll
                for (int kk = 0; kk < BK_OUT; kk += MMA_K) {
                    uint32_t a_frag[4], b_frag[2];
                    #pragma unroll
                    for (int mi = 0; mi < ROW_TILES; ++mi) {
                        // C_stage is [BM × BK_OUT] row-major, stride = stride_c
                        load_a_frag(a_frag,
                            s_C_stage + (row_base + mi * MMA_M) * stride_c + kk,
                            stride_c);
                        #pragma unroll
                        for (int nj = 0; nj < COL_TILES; ++nj) {
                            // B2 stored [N × K] → s_B2[n * stride_b2 + k]
                            load_b_frag(b_frag,
                                s_B2 + (col_base + nj * MMA_N) * stride_b2 + kk,
                                stride_b2);
                            mma_m16n8k16(acc_E[mi * COL_TILES + nj],
                                         a_frag, b_frag,
                                         acc_E[mi * COL_TILES + nj]);
                        }
                    }
                }
            }
            __syncthreads();
        }

        cluster.sync();
    } // end n_step

    // ---- Write output (consumers only) ----
    if (!is_prod) {
        #pragma unroll
        for (int mi = 0; mi < ROW_TILES; ++mi) {
            #pragma unroll
            for (int nj = 0; nj < COL_TILES; ++nj) {
                int grow = m_tile + row_base + mi * MMA_M;
                int gcol = k_tile + col_base + nj * MMA_N;
                store_accum_gmem(
                    &E[grow * K_GLOBAL + gcol],
                    K_GLOBAL,
                    acc_E[mi * COL_TILES + nj]);
            }
        }
    }
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv) {
    int warmup = 20, iters = 100;
    if (argc >= 2) warmup = std::atoi(argv[1]);
    if (argc >= 3) iters  = std::atoi(argv[2]);

    CUDA_CHECK(cudaSetDevice(0));

    int cluster_ok = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&cluster_ok, cudaDevAttrClusterLaunch, 0));
    if (!cluster_ok) {
        std::cerr << "Device does not support cluster launch\n"; return 1;
    }

    int max_smem = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    CUDA_CHECK(cudaFuncSetAttribute(fused_ffn_inter_sm_v5_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem));
    CUDA_CHECK(cudaFuncSetAttribute(fused_ffn_inter_sm_v5_kernel,
        cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

    size_t smem_bytes = compute_smem_bytes();
    if ((int)smem_bytes > max_smem) {
        std::cerr << "smem " << smem_bytes << " > max " << max_smem << '\n'; return 1;
    }

    half *dA, *dB1, *dB2, *dE;
    CUDA_CHECK(cudaMalloc(&dA,  sizeof(half) * M_GLOBAL * K_GLOBAL));
    CUDA_CHECK(cudaMalloc(&dB1, sizeof(half) * N_GLOBAL * K_GLOBAL)); // B1[n,k]
    CUDA_CHECK(cudaMalloc(&dB2, sizeof(half) * N_GLOBAL * K_GLOBAL)); // B2[n,k]
    CUDA_CHECK(cudaMalloc(&dE,  sizeof(half) * M_GLOBAL * K_GLOBAL));

    fill_kernel<<<(M_GLOBAL*K_GLOBAL+255)/256,256>>>(dA,  M_GLOBAL*K_GLOBAL, 0.5f);
    fill_kernel<<<(N_GLOBAL*K_GLOBAL+255)/256,256>>>(dB1, N_GLOBAL*K_GLOBAL, 0.25f);
    fill_kernel<<<(N_GLOBAL*K_GLOBAL+255)/256,256>>>(dB2, N_GLOBAL*K_GLOBAL, 0.125f);
    CUDA_CHECK(cudaDeviceSynchronize());

    // TMA maps:
    //   A  [M x K]:  tile [BM x BK_IN]
    //   B1 [N x K]:  tile [BK_OUT x BK_IN]   (BK_OUT n-rows, BK_IN k-cols)
    //   B2 [N x K]:  tile [BK_OUT x BK_OUT]
    CUtensorMap tm_A  = create_tma_map(dA,  M_GLOBAL, K_GLOBAL, BM,     BK_IN);
    CUtensorMap tm_B1 = create_tma_map(dB1, N_GLOBAL, K_GLOBAL, BK_OUT, BK_IN);
    CUtensorMap tm_B2 = create_tma_map(dB2, N_GLOBAL, K_GLOBAL, BK_OUT, BK_OUT);

    const int num_cx = K_GLOBAL / (BK_OUT * CLUSTER_SIZE);
    dim3 block(THREADS), grid(num_cx * CLUSTER_SIZE, M_GLOBAL / BM, 1);

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem_bytes;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim = {(unsigned)CLUSTER_SIZE, 1, 1};
    cfg.attrs = attr; cfg.numAttrs = 1;

    auto launch = [&]() {
        return cudaLaunchKernelEx(&cfg, fused_ffn_inter_sm_v5_kernel,
                                  tm_A, tm_B1, tm_B2, dE);
    };

    std::cout << "Kernel=fused_ffn_inter_sm_v5\n"
              << "Shape(M,K,N)=" << M_GLOBAL << ',' << K_GLOBAL << ',' << N_GLOBAL << '\n'
              << "Instr=mma.sync.m16n8k16.row.col.f32.f16 + TMA + mbarrier + warp-spec\n"
              << "Threads/block=" << THREADS
              << " (1 prod + " << C_WARPS << " consumer warps)\n"
              << "TileA=[" << BM << 'x' << BK_IN << "]  "
              << "TileB1=[" << BK_OUT << 'x' << BK_IN << "]  "
              << "TileB2=[" << BK_OUT << 'x' << BK_OUT << "]\n"
              << "GridClusters=" << num_cx << 'x' << grid.y
              << "  ClusterSize=" << CLUSTER_SIZE << '\n'
              << "DynamicSmemKB=" << (smem_bytes / 1024.f) << '\n';

    for (int i = 0; i < warmup; ++i) CUDA_CHECK(launch());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) CUDA_CHECK(launch());
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    ms /= iters;

    double flops  = 2.0 * M_GLOBAL * K_GLOBAL * N_GLOBAL * 2.0;
    double tflops = flops / (ms * 1e-3) / 1e12;
    std::cout << "Average Latency: " << ms << " ms\n"
              << "LATENCY_MS=" << ms << '\n'
              << "THROUGHPUT_TFLOPS=" << tflops << '\n';

    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB1));
    CUDA_CHECK(cudaFree(dB2)); CUDA_CHECK(cudaFree(dE));
    return 0;
}
