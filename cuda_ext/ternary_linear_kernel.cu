#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <ATen/ATen.h>

using namespace nvcuda;

// Block tiling params
#define BM 128
#define BN 128
#define BK 16

#define WARPS_PER_BLOCK 8
#define WARPS_M 4
#define WARPS_N 2

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WM_TILE (BM / WARPS_M)        // 32
#define WN_TILE (BN / WARPS_N)        // 64
#define WARP_TILES_M (WM_TILE / WMMA_M)  // 2
#define WARP_TILES_N (WN_TILE / WMMA_N)  // 4

extern "C" __global__
void ternary_wmma_gemm_fused(
    const half* __restrict__ A,           // [M x K], row-major
    const uint32_t* __restrict__ B_nz,    // [K x (N/32)]
    const uint32_t* __restrict__ B_sgn,   // [K x (N/32)]
    const float* __restrict__ bias,       // [N] or nullptr
    half* __restrict__ C,                 // [M x N], row-major (fp16)
    int M, int N, int K,
    float alpha,
    int relu
) {
    // Block origin in C
    int block_m0 = blockIdx.y * BM;
    int block_n0 = blockIdx.x * BN;
    if (block_m0 >= M || block_n0 >= N) return;

    // Lane/warp ids
    int lane_id = threadIdx.x & 31;           // 0..31
    int warp_id = threadIdx.x >> 5;           // 0..7
    int warp_m  = warp_id % WARPS_M;          // 0..3
    int warp_n  = warp_id / WARPS_M;          // 0..1

    int warp_m0 = block_m0 + warp_m * WM_TILE; // 32-step
    int warp_n0 = block_n0 + warp_n * WN_TILE; // 64-step

    // Shared memory layout: [A_smem (BM*BK half)] [B_smem (BK*BN half)] [C_scratch (WARPS_PER_BLOCK*WMMA_M*WMMA_N float)]
    extern __shared__ unsigned char smem_uc[];
    half*  A_smem = reinterpret_cast<half*>(smem_uc);
    half*  B_smem = A_smem + BM * BK;

    // Align C_scratch to 128B just to be safe for WMMA store
    unsigned char* afterB = reinterpret_cast<unsigned char*>(B_smem + BK * BN);
    size_t aligned = (reinterpret_cast<size_t>(afterB) + 127) & ~size_t(127);
    float* C_scratch = reinterpret_cast<float*>(aligned);

    // Accumulators for this warp's 2x4 tiles
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[WARP_TILES_M][WARP_TILES_N];

    #pragma unroll
    for (int tm = 0; tm < WARP_TILES_M; ++tm)
        #pragma unroll
        for (int tn = 0; tn < WARP_TILES_N; ++tn)
            wmma::fill_fragment(acc[tm][tn], 0.0f);

    // Iterate over K in BK=16 chunks
    for (int k0 = 0; k0 < K; k0 += BK) {
        // Cooperative load A tile [BM x BK] into shared
        // (All 8 warps redundantly do it; safe but can be optimized later.)
        for (int i = lane_id; i < BM * BK; i += 32) {
            int r = i / BK;   // 0..127
            int c = i % BK;   // 0..15
            int g_r = block_m0 + r;
            int g_c = k0 + c;
            half v = __float2half(0.f);
            if (g_r < M && g_c < K) v = A[g_r * K + g_c];
            A_smem[r * BK + c] = v;
        }

        // Decode B tile [BK x BN] into shared as half in row-major
        for (int i = lane_id; i < BK * BN; i += 32) {
            int kk = i / BN;           // 0..15
            int j  = i % BN;           // 0..127
            int g_k = k0 + kk;
            int g_n = block_n0 + j;
            half hv = __float2half(0.f);
            if (g_k < K && g_n < N) {
                int word_idx = g_n >> 5;
                int bit_lane = g_n & 31;
                int stride_words = (N + 31) >> 5;  // words per K-row
                uint32_t nzb  = B_nz[g_k * stride_words + word_idx];
                uint32_t sgnb = B_sgn[g_k * stride_words + word_idx];
                int nz  = (nzb  >> bit_lane) & 1;
                int sgn = (sgnb >> bit_lane) & 1;
                if (nz) hv = __int2half_rn(sgn ? -1 : 1);
            }
            B_smem[kk * BN + j] = hv;
        }

        __syncthreads();

        // MMA (A row-major, B row-major)
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            #pragma unroll
            for (int tm = 0; tm < WARP_TILES_M; ++tm) {
                #pragma unroll
                for (int tn = 0; tn < WARP_TILES_N; ++tn) {
                    int a_row = (warp_m * WM_TILE) + tm * WMMA_M;
                    int b_col = (warp_n * WN_TILE) + tn * WMMA_N;

                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                                   half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                                   half, wmma::col_major> b_frag;

                    const half* a_ptr = &A_smem[a_row * BK + kk];
                    const half* b_ptr = &B_smem[kk * BN + b_col];

                    wmma::load_matrix_sync(a_frag, a_ptr, BK);  // ld = BK
                    wmma::load_matrix_sync(b_frag, b_ptr, BN);  // ld = BN
                    wmma::mma_sync(acc[tm][tn], a_frag, b_frag, acc[tm][tn]);
                }
            }
        }

        __syncthreads();
    }

    // ---- Store (via per-warp shared scratch) + epilogue ----
    // Each warp has a 256-float scratch at:
    float* warp_scratch = C_scratch + warp_id * (WMMA_M * WMMA_N);

    #pragma unroll
    for (int tm = 0; tm < WARP_TILES_M; ++tm) {
        #pragma unroll
        for (int tn = 0; tn < WARP_TILES_N; ++tn) {
            int tile_m0 = warp_m0 + tm * WMMA_M;
            int tile_n0 = warp_n0 + tn * WMMA_N;

            // Store fragment into shared scratch (warp-cooperative)
            wmma::store_matrix_sync(warp_scratch, acc[tm][tn], WMMA_N, wmma::mem_row_major);
            __syncwarp(); // ensure scratch is visible within the warp

            // Each lane writes multiple elements from scratch to global
            for (int t = lane_id; t < WMMA_M * WMMA_N; t += 32) {
                int i = t / WMMA_N;
                int j = t % WMMA_N;
                int row = tile_m0 + i;
                int col = tile_n0 + j;
                if (row < M && col < N) {
                    float v = warp_scratch[t] * alpha;
                    if (bias) v += bias[col];
                    if (relu) v = v > 0.f ? v : 0.f;
                    C[row * N + col] = __float2half_rn(v);
                }
            }
        }
    }
}

// Host wrapper (signature matches your bindings.cpp)
extern "C"
void launch_ternary_linear_cuda(const at::Half* A,
                                const uint32_t* B_nz,
                                const uint32_t* B_sgn,
                                at::Half* C,
                                int M, int N, int K,
                                float alpha,
                                const float* bias,
                                int relu) {
    dim3 block(256);  // 8 warps × 32 threads
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    // dynamic shared memory:
    // A_smem (BM*BK half) + B_smem (BK*BN half) + C_scratch (WARPS_PER_BLOCK*WMMA_M*WMMA_N float) + a little for alignment
    size_t bytesA = BM * BK * sizeof(half);                  // 128*16*2 = 4096
    size_t bytesB = BK * BN * sizeof(half);                  // 16*128*2 = 4096
    size_t bytesScratch = WARPS_PER_BLOCK * WMMA_M * WMMA_N * sizeof(float); // 8*256*4 = 8192
    size_t smem_size = bytesA + bytesB + bytesScratch + 128; // +128 to allow alignment of scratch

    ternary_wmma_gemm_fused<<<grid, block, smem_size>>>(
        reinterpret_cast<const half*>(A),
        B_nz,
        B_sgn,
        bias,
        reinterpret_cast<half*>(C),
        M, N, K,
        alpha,
        relu);
}
