// ternary_linear_kernel.cu
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <ATen/ATen.h>

using namespace nvcuda;

// ----------------- Block tiling -----------------
#define BM 128
#define BN 128
#define BK 32      // two 16-wide K steps

// Block decomposition (8 warps => 256 threads)
#define WARPS_PER_BLOCK 8
#define WARPS_M 4
#define WARPS_N 2

// WMMA (tensor core) tile
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WM_TILE (BM / WARPS_M)          // 32
#define WN_TILE (BN / WARPS_N)          // 64
#define WARP_TILES_M (WM_TILE / WMMA_M) // 2
#define WARP_TILES_N (WN_TILE / WMMA_N) // 4

// ----------------- Helpers -----------------
__device__ __forceinline__ half hsel(int nz, int sgn) {
    return nz ? (sgn ? __int2half_rn(-1) : __int2half_rn(1)) : __float2half(0.f);
}

__device__ __forceinline__ unsigned char* align128(unsigned char* p) {
    uintptr_t v = reinterpret_cast<uintptr_t>(p);
    v = (v + 127u) & ~uintptr_t(127u);
    return reinterpret_cast<unsigned char*>(v);
}

// ----------------- Kernel -----------------
extern "C" __global__
void ternary_wmma_gemm_fused(
    const half* __restrict__ A,           // [M x K], row-major
    const uint32_t* __restrict__ B_nz,    // [K x (N/32)]
    const uint32_t* __restrict__ B_sgn,   // [K x (N/32)]
    const float* __restrict__ bias,       // [N] or nullptr
    half* __restrict__ C,                 // [M x N], row-major
    int M, int N, int K,
    float alpha,
    int relu
) {
    // --- block origin ---
    const int block_m0 = blockIdx.y * BM;
    const int block_n0 = blockIdx.x * BN;
    if (block_m0 >= M || block_n0 >= N) return;

    // --- thread/warp ids ---
    const int tid     = threadIdx.x;
    const int lane_id = tid & 31;      // 0..31
    const int warp_id = tid >> 5;      // 0..7
    const int warp_m  = warp_id % WARPS_M;     // 0..3
    const int warp_n  = warp_id / WARPS_M;     // 0..1

    const int warp_m0 = block_m0 + warp_m * WM_TILE;  // 32-step
    const int warp_n0 = block_n0 + warp_n * WN_TILE;  // 64-step

    // words per K-row in bitplanes
    const int words_per_row   = (N + 31) >> 5;     // ceil(N/32)
    const int words_in_blockN = BN >> 5;           // BN/32 (128 -> 4)

    // --- shared memory layout (double buffered) ---
    extern __shared__ unsigned char smem_uc[];
    unsigned char* p = smem_uc;

    // Align for A tiles
    p = align128(p);
    half* As0 = reinterpret_cast<half*>(p); p += BM * BK * sizeof(half);
    p = align128(p);
    half* As1 = reinterpret_cast<half*>(p); p += BM * BK * sizeof(half);
    half* As[2] = { As0, As1 };

    // Bw_* word buffers (uint32)
    p = align128(p);
    uint32_t* Bw_nz0 = reinterpret_cast<uint32_t*>(p); p += BK * words_in_blockN * sizeof(uint32_t);
    p = align128(p);
    uint32_t* Bw_sg0 = reinterpret_cast<uint32_t*>(p); p += BK * words_in_blockN * sizeof(uint32_t);
    p = align128(p);
    uint32_t* Bw_nz1 = reinterpret_cast<uint32_t*>(p); p += BK * words_in_blockN * sizeof(uint32_t);
    p = align128(p);
    uint32_t* Bw_sg1 = reinterpret_cast<uint32_t*>(p); p += BK * words_in_blockN * sizeof(uint32_t);
    uint32_t* Bw_nz[2] = { Bw_nz0, Bw_nz1 };
    uint32_t* Bw_sg[2] = { Bw_sg0, Bw_sg1 };

    // Align for B decoded tiles (used by WMMA matrix_b loads)
    p = align128(p);
    half* Bs0 = reinterpret_cast<half*>(p); p += BK * BN * sizeof(half);
    p = align128(p);
    half* Bs1 = reinterpret_cast<half*>(p); p += BK * BN * sizeof(half);
    half* Bs[2] = { Bs0, Bs1 };

    // Per-warp store scratch (float 16x16 = 256 per warp)
    p = align128(p);
    float* warp_scratch = reinterpret_cast<float*>(p); // length: WARPS_PER_BLOCK * 256
    p += WARPS_PER_BLOCK * WMMA_M * WMMA_N * sizeof(float);
    float* my_scratch = warp_scratch + warp_id * (WMMA_M * WMMA_N);

    // --- accumulators (per warp) ---
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[WARP_TILES_M][WARP_TILES_N];
    #pragma unroll
    for (int tm = 0; tm < WARP_TILES_M; ++tm)
        #pragma unroll
        for (int tn = 0; tn < WARP_TILES_N; ++tn)
            wmma::fill_fragment(acc[tm][tn], 0.0f);

    // ----------------- Prefetch (synchronous & aligned) -----------------
    auto prefetch_tile = [&](int k0, int buf_idx) {
        // --- A tile: [BM x BK] ---
        for (int idx = tid; idx < BM * BK; idx += blockDim.x) {
            const int r = idx / BK;
            const int c = idx % BK;
            const int gr = block_m0 + r;
            const int gc = k0 + c;
            half v = __float2half(0.f);
            if (gr < M && gc < K) v = A[gr * K + gc];
            As[buf_idx][idx] = v;
        }

        // --- B bitplane words for this block-N: BK rows, words_in_blockN cols ---
        const int word_base = block_n0 >> 5; // starting word index for this block-N
        for (int kk = tid; kk < BK * words_in_blockN; kk += blockDim.x) {
            const int r  = kk / words_in_blockN;    // 0..BK-1
            const int w  = kk % words_in_blockN;    // 0..(BN/32-1)
            const int gk = k0 + r;
            const int gw = word_base + w;
            uint32_t nzv = 0u, sgv = 0u;
            if (gk < K && gw < words_per_row) {
                nzv = B_nz[gk * words_per_row + gw];
                sgv = B_sgn[gk * words_per_row + gw];
            }
            Bw_nz[buf_idx][kk] = nzv;
            Bw_sg[buf_idx][kk] = sgv;
        }
        __syncthreads();

        // --- Decode B words -> Bs[buf_idx] (row-major [BK x BN]) ---
        // clamp columns to N when at the right edge
        for (int r = tid; r < BK; r += blockDim.x) {
            for (int w = 0; w < words_in_blockN; ++w) {
                const uint32_t nzb  = Bw_nz[buf_idx][r * words_in_blockN + w];
                const uint32_t sgnb = Bw_sg[buf_idx][r * words_in_blockN + w];
                #pragma unroll
                for (int b = 0; b < 32; ++b) {
                    const int col = (w << 5) + b;           // 0..127 within block
                    const int g_n = block_n0 + col;         // global column
                    half hv = __float2half(0.f);
                    if (g_n < N) {
                        const int nz  = (nzb  >> b) & 1;
                        const int sgn = (sgnb >> b) & 1;
                        if (nz) hv = __int2half_rn(sgn ? -1 : 1);
                    }
                    Bs[buf_idx][r * BN + col] = hv;
                }
            }
        }
        __syncthreads();
    };

    int buf = 0;
    if (K > 0) prefetch_tile(0, buf);

    // ----------------- Main loop over K tiles -----------------
    for (int k0 = 0; k0 < K; k0 += BK) {
        const int next_k0 = k0 + BK;
        const int cur = buf;
        const int nxt = buf ^ 1;

        if (next_k0 < K) {
            prefetch_tile(next_k0, nxt);
        }

        // ---- MMA on current tile (BK=32 -> two 16-wide steps) ----
        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) { // 0,16
            #pragma unroll
            for (int tm = 0; tm < WARP_TILES_M; ++tm) {
                #pragma unroll
                for (int tn = 0; tn < WARP_TILES_N; ++tn) {
                    const int a_row = (warp_m * WM_TILE) + tm * WMMA_M;  // 0..127
                    const int b_col = (warp_n * WN_TILE) + tn * WMMA_N;  // 0..127

                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                                   half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                                   half, wmma::row_major> b_frag;

                    const half* a_ptr = &As[cur][a_row * BK + kk]; // ldA = BK
                    const half* b_ptr = &Bs[cur][kk * BN + b_col]; // ldB = BN

                    wmma::load_matrix_sync(a_frag, a_ptr, BK);
                    wmma::load_matrix_sync(b_frag, b_ptr, BN);
                    wmma::mma_sync(acc[tm][tn], a_frag, b_frag, acc[tm][tn]);
                }
            }
        }

        buf = nxt;
    }

    // --------- Store + epilogue (alpha, bias, ReLU) ----------
    #pragma unroll
    for (int tm = 0; tm < WARP_TILES_M; ++tm) {
        #pragma unroll
        for (int tn = 0; tn < WARP_TILES_N; ++tn) {
            const int tile_m0 = warp_m0 + tm * WMMA_M;
            const int tile_n0 = warp_n0 + tn * WMMA_N;

            // store to per-warp shared scratch (aligned)
            wmma::store_matrix_sync(my_scratch, acc[tm][tn], WMMA_N, wmma::mem_row_major);
            __syncwarp();

            for (int t = lane_id; t < WMMA_M * WMMA_N; t += 32) {
                const int i = t / WMMA_N;
                const int j = t % WMMA_N;
                const int row = tile_m0 + i;
                const int col = tile_n0 + j;
                if (row < M && col < N) {
                    float v = my_scratch[t] * alpha;
                    if (bias) v += bias[col];
                    if (relu) v = v > 0.f ? v : 0.f;
                    C[row * N + col] = __float2half_rn(v);
                }
            }
        }
    }
}

// ----------------- Host launcher (unchanged API) -----------------
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

    // Shared memory size (+alignment headroom between regions)
    size_t bytesA = 2 * (BM * BK * sizeof(half));
    size_t bytesBw = 2 * (BK * (BN >> 5) * sizeof(uint32_t)) * 2; // nz + sgn
    size_t bytesB = 2 * (BK * BN * sizeof(half));
    size_t bytesScratch = WARPS_PER_BLOCK * WMMA_M * WMMA_N * sizeof(float);
    size_t alignPad = 6 * 128; // alignment gaps before each region

    size_t smem = bytesA + bytesBw + bytesB + bytesScratch + alignPad;

    ternary_wmma_gemm_fused<<<grid, block, smem>>>(
        reinterpret_cast<const half*>(A),
        B_nz, B_sgn,
        bias,
        reinterpret_cast<half*>(C),
        M, N, K,
        alpha, relu);
}
