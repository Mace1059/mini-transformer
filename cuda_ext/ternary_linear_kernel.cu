#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <ATen/ATen.h>   // pulls in at::Half definition

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

#define WM_TILE (BM / WARPS_M)
#define WN_TILE (BN / WARPS_N)
#define WARP_TILES_M (WM_TILE / WMMA_M)
#define WARP_TILES_N (WN_TILE / WMMA_N)

extern "C" __global__
void ternary_wmma_gemm_fused(
    const half* __restrict__ A,
    const uint32_t* __restrict__ B_nz,
    const uint32_t* __restrict__ B_sgn,
    const float* __restrict__ bias,
    half* __restrict__ C,
    int M, int N, int K,
    float alpha,
    int relu
) {
    int block_m0 = blockIdx.y * BM;
    int block_n0 = blockIdx.x * BN;
    if (block_m0 >= M || block_n0 >= N) return;

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int warp_m = warp_id % WARPS_M;
    int warp_n = warp_id / WARPS_M;

    int warp_m0 = block_m0 + warp_m * WM_TILE;
    int warp_n0 = block_n0 + warp_n * WN_TILE;

    extern __shared__ half smem[];
    half* A_smem = smem;
    half* B_smem = A_smem + BM * BK;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[WARP_TILES_M][WARP_TILES_N];

    #pragma unroll
    for (int tm = 0; tm < WARP_TILES_M; ++tm)
        for (int tn = 0; tn < WARP_TILES_N; ++tn)
            wmma::fill_fragment(acc[tm][tn], 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        // load A tile
        for (int i = lane_id; i < BM * BK; i += 32) {
            int r = i / BK;
            int c = i % BK;
            int g_r = block_m0 + r;
            int g_c = k0 + c;
            half v = __float2half(0.0f);
            if (g_r < M && g_c < K) v = A[g_r * K + g_c];
            A_smem[r * BK + c] = v;
        }

        // decode B tile
        for (int i = lane_id; i < BK * BN; i += 32) {
            int kk = i / BN;
            int j  = i % BN;
            int g_k = k0 + kk;
            int g_n = block_n0 + j;

            half hv = __float2half(0.0f);
            if (g_k < K && g_n < N) {
                int word_idx = g_n >> 5;
                int bit_lane = g_n & 31;
                int stride = (N + 31) >> 5;
                uint32_t nzb  = B_nz[g_k * stride + word_idx];
                uint32_t sgnb = B_sgn[g_k * stride + word_idx];
                int nz  = (nzb  >> bit_lane) & 1;
                int sgn = (sgnb >> bit_lane) & 1;
                if (nz) hv = __int2half_rn(sgn ? -1 : 1);
            }
            B_smem[kk * BN + j] = hv;
        }

        __syncthreads();

        // MMA
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            for (int tm = 0; tm < WARP_TILES_M; ++tm) {
                for (int tn = 0; tn < WARP_TILES_N; ++tn) {
                    int a_row = (warp_m * WM_TILE) + tm * WMMA_M;
                    int b_col = (warp_n * WN_TILE) + tn * WMMA_N;

                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                                   half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                                   half, wmma::col_major> b_frag;

                    const half* a_ptr = &A_smem[a_row * BK + kk];
                    const half* b_ptr = &B_smem[kk * BN + b_col];

                    wmma::load_matrix_sync(a_frag, a_ptr, BK);
                    wmma::load_matrix_sync(b_frag, b_ptr, BN);
                    wmma::mma_sync(acc[tm][tn], a_frag, b_frag, acc[tm][tn]);
                }
            }
        }

        __syncthreads();
    }

    // Store
    for (int tm = 0; tm < WARP_TILES_M; ++tm) {
        for (int tn = 0; tn < WARP_TILES_N; ++tn) {
            int tile_m0 = warp_m0 + tm * WMMA_M;
            int tile_n0 = warp_n0 + tn * WMMA_N;

            float out_frag[WMMA_M * WMMA_N];
            wmma::store_matrix_sync(out_frag, acc[tm][tn],
                                    WMMA_N, wmma::mem_row_major);

            for (int i = 0; i < WMMA_M; ++i) {
                int row = tile_m0 + i;
                if (row >= M) break;
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

// Host wrapper
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
    dim3 grid((N + 127) / 128, (M + 127) / 128);
    size_t smem_size = (128 * 16 + 16 * 128) * sizeof(half);

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
