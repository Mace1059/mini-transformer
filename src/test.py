import torch
import torch.nn.functional as F
import ternary_linear_cuda


def make_ternary_bitplanes(W):
    """
    Convert a dense fp16/fp32 weight matrix W [K, N] into
    bitplanes B_nz, B_sgn (both [K, N/32], int32).
    """
    K, N = W.shape
    words_per_row = (N + 31) // 32
    B_nz = torch.zeros(K, words_per_row, dtype=torch.int32, device="cuda")
    B_sgn = torch.zeros_like(B_nz)
    for k in range(K):
        for n in range(N):
            val = W[k, n].item()
            nz = 1 if val != 0 else 0
            sgn = 1 if val < 0 else 0
            word_idx = n // 32
            bit_idx = n % 32
            if nz:
                B_nz[k, word_idx] |= (1 << bit_idx)
                if sgn:
                    B_sgn[k, word_idx] |= (1 << bit_idx)
    return B_nz, B_sgn


def benchmark(M=1024, K=1024, N=1024, runs=10):
    print("[DEBUG] Allocating tensors...")
    torch.manual_seed(0)

    # Random dense inputs
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)

    # Dense weight for cuBLAS baseline
    W_dense = torch.randint(-1, 2, (K, N), device="cuda", dtype=torch.float16)  # -1,0,1
    bias = torch.randn(N, device="cuda", dtype=torch.float32)

    print("[DEBUG] Converting weights to bitplanes...")
    B_nz, B_sgn = make_ternary_bitplanes(W_dense)
    print("[DEBUG] B_nz shape:", B_nz.shape, " B_sgn shape:", B_sgn.shape)

    print("[DEBUG] Running warmup...")
    for _ in range(3):
        _ = F.linear(A, W_dense.t().contiguous(), bias.half())
        _ = ternary_linear_cuda.forward(A, B_nz, B_sgn, bias, 1.0, False)
    torch.cuda.synchronize()
    print("[DEBUG] Warmup done")

    # cuBLAS dense GEMM benchmark
    print("[DEBUG] Benchmarking cuBLAS...")
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(runs):
        out_dense = F.linear(A, W_dense.t().contiguous(), bias.half())
    end_evt.record()
    torch.cuda.synchronize()
    dense_time = start_evt.elapsed_time(end_evt) / runs
    print(f"[RESULT] cuBLAS dense:   {dense_time:.3f} ms")

    # Custom ternary kernel benchmark
    print("[DEBUG] Benchmarking custom kernel...")
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(runs):
        out_tern = ternary_linear_cuda.forward(A, B_nz, B_sgn, bias, 1.0, False)
    end_evt.record()
    torch.cuda.synchronize()
    ternary_time = start_evt.elapsed_time(end_evt) / runs
    print(f"[RESULT] Ternary kernel: {ternary_time:.3f} ms")

    # Quick correctness check
    print("[DEBUG] Sample outputs")
    print("Dense out[0,:5]:  ", out_dense[0, :5])
    print("Ternary out[0,:5]:", out_tern[0, :5])


if __name__ == "__main__":
    # Start small to validate, then scale up
    benchmark(M=256, K=256, N=256, runs=10)
    benchmark(M=1024, K=1024, N=1024, runs=10)
