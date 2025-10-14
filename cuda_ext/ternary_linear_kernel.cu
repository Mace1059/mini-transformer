// cuda/ternary_linear_kernel.cu
#include <cuda.h>
#include <cuda_fp16.h>
#include <ATen/ATen.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

// ----------------- Tunables (safe & solid) -----------------
#define BM 128        // rows per block of C/A
#define BN 128        // cols per block of C/B
#define BK 64         // K-slab per iteration

#define WARPS_PER_BLOCK 4
#define WARP_SIZE 32
#define THREADS (WARPS_PER_BLOCK * WARP_SIZE)

// Warp tile (covered per warp after all phases)
#define WM 64
#define WN 64

// Per-thread sub-tile (SAFE: no word crossing)
#define TM 4   // rows per thread
#define TN 4   // cols per thread

// Lane grouping: PR * PC == 32
// SAFE mapping: PC=8 (8 col-groups -> TN=4 => 32 cols/phase), PR=4 (4 row-groups -> TM=4 => 16 rows/phase)
#define PC 8
#define PR (32 / PC)         // 4

#define ROW_STRIDE (PR * TM) // 4 * 4 = 16
#define COL_STRIDE (PC * TN) // 8 * 4 = 32
#define ROW_PHASES (WM / ROW_STRIDE)  // 64/16 = 4
#define COL_PHASES (WN / COL_STRIDE)  // 64/32 = 2

// 128-bit copy quanta
#define A_COPY_BYTES 16              // 8 halves

// ----------------- helpers -----------------
__device__ __forceinline__ uint8_t* align_ptr(uint8_t* p, int align) {
    uintptr_t v = reinterpret_cast<uintptr_t>(p);
    v = (v + (align - 1)) & ~(uintptr_t)(align - 1);
    return reinterpret_cast<uint8_t*>(v);
}

#if __CUDA_ARCH__ >= 800
static __device__ __forceinline__
void cp_async_16B(void* smem_ptr, const void* gmem_ptr) {
    unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(gmem_ptr));
}
static __device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
static __device__ __forceinline__ void cp_async_wait()   { asm volatile("cp.async.wait_group 0;\n"); }
#else
static __device__ __forceinline__ void cp_async_16B(void* smem_ptr, const void* gmem_ptr) {
    *reinterpret_cast<uint4*>(smem_ptr) = *reinterpret_cast<const uint4*>(gmem_ptr); // requires 16B alignment
}
static __device__ __forceinline__ void cp_async_commit() {}
static __device__ __forceinline__ void cp_async_wait()   {}
#endif

// ----------------- Kernel -----------------
extern "C" __global__
void ternary_bitserial_gemm_tiled(
    const half* __restrict__ A,           // [M x K], row-major
    const uint32_t* __restrict__ B_nz,    // [K x ceil(N/32)]
    const uint32_t* __restrict__ B_sgn,   // [K x ceil(N/32)]
    const float* __restrict__ bias,       // [N] or nullptr
    half* __restrict__ C,                 // [M x N], row-major
    int M, int N, int K,
    float alpha,
    int relu
) {
    // ----------------- Block / Warp coords -----------------
    const int block_m0 = blockIdx.y * BM;
    const int block_n0 = blockIdx.x * BN;
    if (block_m0 >= M || block_n0 >= N) return;

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;     // 0..3
    const int lane    = tid % WARP_SIZE;

    // Warps arranged 2x2 per CTA
    const int warp_m0 = block_m0 + (warp_id % 2) * WM;
    const int warp_n0 = block_n0 + (warp_id / 2) * WN;

    // Thread coords inside one phase of the warp tile
    const int lane_row = (lane / PC) * TM;   // PR groups along rows
    const int lane_col = (lane % PC) * TN;   // PC groups along cols

    // ----------------- Shared memory (A only) -----------------
    extern __shared__ uint8_t smem[];
    uint8_t* p = smem;
    p = align_ptr(p, 128);
    half* As = reinterpret_cast<half*>(p);
    const size_t As_elems = size_t(BM) * BK;
    p += 2 * As_elems * sizeof(half); // double buffer
    half* As_buf[2] = { As, As + As_elems };

    // ----------------- Accumulator (per-thread TMxTN tile) -----------------
    float acc[TM][TN];

    // ----------------- Prefetch A-tile (zero all rows; safe tails) -----------------
    auto prefetch_A_tile = [&](int k0, int buf) {
        // Zero A slab (avoid stale reads for M<BM or tail K)
        for (int r = tid; r < BM; r += blockDim.x) {
            half* sA_row = As_buf[buf] + r * BK;
            #pragma unroll 1
            for (int c = 0; c < BK; ++c) sA_row[c] = __float2half(0.f);
        }
        __syncthreads();

        // Copy valid A rows
        const int valid_rows_A = min(BM, M - block_m0);
        const int valid_cols_A = min(BK, K - k0);         // halves
        const int row_bytes_A  = valid_cols_A * int(sizeof(half));
        for (int r = tid; r < valid_rows_A; r += blockDim.x) {
            const char* gA_row = reinterpret_cast<const char*>(A + (block_m0 + r) * K + k0);
            char*       sA_row = reinterpret_cast<char*>(As_buf[buf] + r * BK);
            const uintptr_t gA = reinterpret_cast<uintptr_t>(gA_row);
            const uintptr_t sA = reinterpret_cast<uintptr_t>(sA_row);
            const bool aligned16 = ((gA | sA) & 0xF) == 0;

            if (aligned16) {
                const int full16 = row_bytes_A / A_COPY_BYTES;
                const int tail   = row_bytes_A - full16 * A_COPY_BYTES;
                for (int cch = 0; cch < full16; ++cch) {
                    cp_async_16B(sA_row + cch * A_COPY_BYTES, gA_row + cch * A_COPY_BYTES);
                }
                for (int t = 0; t < tail; ++t) {
                    sA_row[full16 * A_COPY_BYTES + t] = gA_row[full16 * A_COPY_BYTES + t];
                }
            } else {
                // scalar halves
                const half* gA_h = reinterpret_cast<const half*>(gA_row);
                half*       sA_h = reinterpret_cast<half*>(sA_row);
                #pragma unroll 1
                for (int c = 0; c < valid_cols_A; ++c) sA_h[c] = gA_h[c];
            }
        }
#if __CUDA_ARCH__ >= 800
        cp_async_commit();
#endif
    };

    // Preload A stage 0
    int buf = 0;
    if (K > 0) {
        prefetch_A_tile(0, buf);
#if __CUDA_ARCH__ >= 800
        cp_async_wait();
#endif
        __syncthreads();
    }

    // Geometry for B (global access)
    const int words_per_row = (N + 31) >> 5;

    // ----------------- Phase loops (cover full 64x64 warp tile) -----------------
    for (int fr = 0; fr < ROW_PHASES; ++fr) {
        for (int fc = 0; fc < COL_PHASES; ++fc) {

            // Preload bias for this phase (TN columns per thread)
            float bias_reg[TN];
            #pragma unroll
            for (int j=0; j<TN; ++j) {
                int gcol = warp_n0 + fc*COL_STRIDE + lane_col + j;
                bias_reg[j] = (bias && gcol < N) ? bias[gcol] : 0.f;
            }

            // zero accumulators
            #pragma unroll
            for (int ii=0; ii<TM; ++ii)
                #pragma unroll
                for (int jj=0; jj<TN; ++jj)
                    acc[ii][jj] = 0.f;

            // Iterate K with double-buffered A tiles
            for (int k0 = 0; k0 < K; k0 += BK) {
                const int nxt = buf ^ 1;
                if (k0 + BK < K) prefetch_A_tile(k0 + BK, nxt);

                half* As_cur = As_buf[buf];

                // ---- Compute on current A tile; read B directly per-column ----
                #pragma unroll 4
                for (int kk = 0; kk < BK; ++kk) {
                    const int gk = k0 + kk;
                    if (gk >= K) break;

                    // Load A rows this thread needs (phase-adjusted)
                    float a_reg[TM];
                    #pragma unroll
                    for (int i=0; i<TM; ++i) {
                        int row_local = (warp_m0 - block_m0) + fr*ROW_STRIDE + lane_row + i; // 0..BM-1
                        a_reg[i] = (row_local >= 0 && row_local < BM) ? __half2float(As_cur[row_local*BK + kk]) : 0.f;
                    }

                    // Process TN columns: compute word and bit per column j
                    #pragma unroll
                    for (int j=0; j<TN; ++j) {
                        const int col = warp_n0 + fc*COL_STRIDE + lane_col + j;
                        if (col >= N) break;

                        const int w = col >> 5;           // word index for this column
                        const int b = col & 31;           // bit index in that word
                        // Bounds check on words_per_row
                        if ((unsigned)w >= (unsigned)words_per_row) continue;

                        const uint32_t nz_bits = B_nz[gk * words_per_row + w];
                        if (((nz_bits >> b) & 1u) == 0u) continue;

                        const uint32_t sg_bits = B_sgn[gk * words_per_row + w];
                        const float add = ((sg_bits >> b) & 1u) ? -1.f : 1.f;

                        #pragma unroll
                        for (int i=0; i<TM; ++i) {
                            acc[i][j] += a_reg[i] * add;
                        }
                    }
                }

                // Wait only at tile boundary
                if (k0 + BK < K) {
#if __CUDA_ARCH__ >= 800
                    cp_async_wait();
#endif
                    __syncthreads();
                }

                buf = nxt;
            } // k0

            // ----------------- Epilogue store for this (fr,fc) phase -----------------
            #pragma unroll
            for (int i=0; i<TM; ++i) {
                int row = warp_m0 + fr*ROW_STRIDE + lane_row + i;  // global row
                if (row >= M) continue;
                #pragma unroll
                for (int j=0; j<TN; ++j) {
                    int col = warp_n0 + fc*COL_STRIDE + lane_col + j;  // global col
                    if (col >= N) continue;
                    float v = acc[i][j] * alpha;                // scale
                    v += bias_reg[j];                           // bias from registers
                    if (relu) v = v > 0.f ? v : 0.f;            // activation
                    C[row * N + col] = __float2half_rn(v);
                }
            }

        } // fc
    } // fr
}

// ----------------- Host launcher -----------------
extern "C"
void launch_ternary_linear_cuda(const at::Half* A,
                                const uint32_t* B_nz,
                                const uint32_t* B_sgn,
                                at::Half* C,
                                int M, int N, int K,
                                float alpha,
                                const float* bias,
                                int relu) {
    dim3 block(THREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    const size_t smem = 2 * (size_t)BM * BK * sizeof(half); // A only (double buffer)

    ternary_bitserial_gemm_tiled<<<grid, block, smem>>>(
        reinterpret_cast<const half*>(A),
        B_nz, B_sgn,
        bias,
        reinterpret_cast<half*>(C),
        M, N, K,
        alpha, relu);
}

// ===================== DEBUG KERNELS =====================

extern "C" __global__
void ternary_decode_to_dense_kn_kernel(
    const uint32_t* __restrict__ B_nz,    // [K x ceil(N/32)]
    const uint32_t* __restrict__ B_sgn,   // [K x ceil(N/32)]
    int8_t* __restrict__ W_kn,            // [K x N], signed int8 {-1,0,1}
    int K, int N
) {
    int k = blockIdx.y * blockDim.y + threadIdx.y; // 0..K-1
    int n = blockIdx.x * blockDim.x + threadIdx.x; // 0..N-1
    if (k >= K || n >= N) return;
    int words_per_row = (N + 31) >> 5;
    int w = n >> 5;
    int b = n & 31;

    uint32_t nz = B_nz[k * words_per_row + w];
    if (((nz >> b) & 1u) == 0u) {
        W_kn[k * N + n] = 0;
        return;
    }
    uint32_t sg = B_sgn[k * words_per_row + w];
    int8_t v = ((sg >> b) & 1u) ? int8_t(-1) : int8_t(1);
    W_kn[k * N + n] = v;
}

extern "C" void launch_ternary_decode_to_dense_kn(
    const uint32_t* B_nz,
    const uint32_t* B_sgn,
    int8_t* W_kn,
    int K, int N
) {
    dim3 block(32, 8); // 256 thr
    dim3 grid((N + block.x - 1) / block.x,
              (K + block.y - 1) / block.y);
    ternary_decode_to_dense_kn_kernel<<<grid, block>>>(
        B_nz, B_sgn, W_kn, K, N
    );
}

// Slow but correct reference GEMM on GPU (FP32 accumulate, then cast to FP16)
extern "C" __global__
void ternary_naive_forward_kernel(
    const half* __restrict__ A,         // [M x K], row-major
    const uint32_t* __restrict__ B_nz,  // [K x ceil(N/32)]
    const uint32_t* __restrict__ B_sgn, // [K x ceil(N/32)]
    const float* __restrict__ bias,     // [N] or nullptr
    half* __restrict__ C,               // [M x N], row-major
    int M, int N, int K,
    float alpha,
    int relu
) {
    int m = blockIdx.y * blockDim.y + threadIdx.y; // row
    int n = blockIdx.x * blockDim.x + threadIdx.x; // col
    if (m >= M || n >= N) return;

    const int words_per_row = (N + 31) >> 5;
    const int w = n >> 5;
    const int b = n & 31;

    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        uint32_t nz = B_nz[k * words_per_row + w];
        if (((nz >> b) & 1u) == 0u) continue;
        uint32_t sg = B_sgn[k * words_per_row + w];
        float add = ((sg >> b) & 1u) ? -1.f : 1.f;
        float a = __half2float(A[m * K + k]);
        acc += a * add;
    }
    acc *= alpha;
    if (bias) acc += bias[n];
    if (relu) acc = acc > 0.f ? acc : 0.f;
    C[m * N + n] = __float2half_rn(acc);
}

extern "C" void launch_ternary_linear_naive(
    const at::Half* A,
    const uint32_t* B_nz,
    const uint32_t* B_sgn,
    at::Half* C,
    int M, int N, int K,
    float alpha,
    const float* bias,
    int relu
) {
    dim3 block(32, 8); // 256 thr
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);
    ternary_naive_forward_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(A),
        B_nz, B_sgn,
        bias,
        reinterpret_cast<half*>(C),
        M, N, K,
        alpha, relu
    );
}
