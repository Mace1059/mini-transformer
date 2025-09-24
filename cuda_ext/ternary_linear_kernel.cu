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

#define WM_TILE (BM / WARPS_M)            // 32
#define WN_TILE (BN / WARPS_N)            // 64
#define WARP_TILES_M (WM_TILE / WMMA_M)   // 2
#define WARP_TILES_N (WN_TILE / WMMA_N)   // 4

// Kernel
extern "C" __global__
void ternary_wmma_gemm_fused(
    const half* __restrict__ A,           // [M x K], row-major
    const uint32_t* __restrict__ B_nz,    // [K x ceil(N/32)], bitplane: nonzero?
    const uint32_t* __restrict__ B_sgn,   // [K x ceil(N/32)], bitplane: sign (1=-1)
    const float* __restrict__ bias,       // [N] or nullptr
    half* __restrict__ C,                 // [M x N], row-major
    int M, int N, int K,
    float alpha,
    int relu
) {
    // Block origin in C
    const int block_m0 = blockIdx.y * BM;
    const int block_n0 = blockIdx.x * BN;
    if (block_m0 >= M || block_n0 >= N) return;

    const int lane_id = threadIdx.x & 31;     // 0..31
    const int warp_id = threadIdx.x >> 5;     // 0..7
    const int warp_m  = warp_id % WARPS_M;    // 0..3
    const int warp_n  = warp_id / WARPS_M;    // 0..1

    const int warp_m0 = block_m0 + warp_m * WM_TILE; // 32-step
    const int warp_n0 = block_n0 + warp_n * WN_TILE; // 64-step

    // Shared memory: A_smem [BM x BK] (row-major), B_smem [BK x BN] (col-major)
    extern __shared__ half smem[];
    half* A_smem = smem;                 // ld = BK
    half* B_smem = A_smem + BM * BK;     // stored col-major: ld = BK

    // Accumulators (2x4 WMMA tiles per warp)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[WARP_TILES_M][WARP_TILES_N];

    #pragma unroll
    for (int tm = 0; tm < WARP_TILES_M; ++tm) {
        #pragma unroll
        for (int tn = 0; tn < WARP_TILES_N; ++tn) {
            wmma::fill_fragment(acc[tm][tn], 0.0f);
        }
    }

    const int words_per_row = (N + 31) >> 5;  // ceil(N/32)

    // Iterate over K in tiles of BK=16
    for (int k0 = 0; k0 < K; k0 += BK) {

        // ---- Cooperative load A tile [BM x BK] into shared (row-major) ----
        // All 256 threads share the work; no redundant loads.
        for (int t = threadIdx.x; t < BM * BK; t += blockDim.x) {
            int r = t / BK;           // 0..127
            int c = t % BK;           // 0..15
            int gr = block_m0 + r;
            int gc = k0 + c;
            half v = __float2half(0.0f);
            if (gr < M && gc < K) v = A[gr * K + gc];
            A_smem[r * BK + c] = v;
        }

        // ---- Decode B for this K-slice into shared (col-major) ----
        // For each kk in [0..BK), we have BN columns inside the block-N tile.
        // Expand 32-bit words -> 32 halfs at a time, and store at B_smem[col * BK + kk]
        for (int t = threadIdx.x; t < BK * ((BN + 31) >> 5); t += blockDim.x) {
            int kk   = t / ((BN + 31) >> 5);             // 0..15 within this k-slice
            int widx = t % ((BN + 31) >> 5);             // 0..ceil(BN/32)-1 within tile
            int gk   = k0 + kk;
            int gword = (block_n0 >> 5) + widx;          // global word index along N

            uint32_t nz = 0u, sg = 0u;
            if (gk < K && (gword < words_per_row)) {
                nz = B_nz[gk * words_per_row + gword];
                sg = B_sgn[gk * words_per_row + gword];
            }

            // Expand to 32 columns (may spill past N at right edge; guard later)
            int base_col = (gword << 5) - block_n0;      // local col 0-based within BN
            // For each bit -> write one half into B_smem at [col * BK + kk]
            // Keep B_smem col-major so WMMA_B col_major with ld=BK works.
            #pragma unroll
            for (int b = 0; b < 32; ++b) {
                int j = base_col + b;                   // 0..BN-1 (local col)
                if (j >= 0 && j < BN) {
                    int g_n = block_n0 + j;
                    half hv = __float2half(0.0f);
                    if (gk < K && g_n < N) {
                        int bit = 1 & (nz >> b);
                        if (bit) {
                            int s = 1 & (sg >> b);
                            hv = __int2half_rn(s ? -1 : 1);
                        }
                    }
                    B_smem[j * BK + kk] = hv;  // col-major write
                }
            }
        }

        __syncthreads();

        // ---- MMA on this A/B tile ----
        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            #pragma unroll
            for (int tm = 0; tm < WARP_TILES_M; ++tm) {
                #pragma unroll
                for (int tn = 0; tn < WARP_TILES_N; ++tn) {

                    const int a_row = (warp_m * WM_TILE) + tm * WMMA_M; // in [0..127]
                    const int b_col = (warp_n * WN_TILE) + tn * WMMA_N; // in [0..127]

                    // Fragments
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                                   half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                                   half, wmma::col_major> b_frag;

                    // A_smem is [BM x BK], ld = BK, row-major
                    const half* a_ptr = &A_smem[a_row * BK + kk];
                    // B_smem is [BK x BN] but stored col-major, ld = BK
                    const half* b_ptr = &B_smem[b_col * BK + kk];

                    // NOTE: For col_major matrix_b, leading dimension MUST be BK
                    wmma::load_matrix_sync(a_frag, a_ptr, BK);
                    wmma::load_matrix_sync(b_frag, b_ptr, BK);

                    wmma::mma_sync(acc[tm][tn], a_frag, b_frag, acc[tm][tn]);
                }
            }
        }

        __syncthreads();
    }

    // ---- Store (directly to global) with fused epilogue ----
    #pragma unroll
    for (int tm = 0; tm < WARP_TILES_M; ++tm) {
        #pragma unroll
        for (int tn = 0; tn < WARP_TILES_N; ++tn) {
            const int tile_m0 = warp_m0 + tm * WMMA_M;
            const int tile_n0 = warp_n0 + tn * WMMA_N;

            float out_frag[WMMA_M * WMMA_N];
            wmma::store_matrix_sync(out_frag, acc[tm][tn], WMMA_N, wmma::mem_row_major);

            #pragma unroll
            for (int i = 0; i < WMMA_M; ++i) {
                int row = tile_m0 + i;
                if (row >= M) break;
                #pragma unroll
                for (int j = 0; j < WMMA_N; ++j) {
                    int col = tile_n0 + j;
                    if (col >= N) break;

                    float v = out_frag[i * WMMA_N + j] * alpha;
                    if (bias) v += bias[col];
                    if (relu) v = v > 0.f ? v : 0.f;
                    C[row * N + col] = __float2half_rn(v);
                }
            }
        }
    }
}

// Host wrapper (unchanged signature)
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

    // Shared: A (BM*BK half) + B (BK*BN half)
    size_t smem_size = (BM * BK + BK * BN) * sizeof(half); // 4096 + 4096 = 8192 bytes

    ternary_wmma_gemm_fused<<<grid, block, smem_size>>>(
        reinterpret_cast<const half*>(A),
        B_nz, B_sgn, bias,
        reinterpret_cast<half*>(C),
        M, N, K,
        alpha, relu);
}
