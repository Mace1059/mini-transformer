import torch
from cuda_ext import ternary_linear_cuda

M, K, N = 64, 128, 256  # small toy sizes

# Random A in fp16
A = torch.randn(M, K, device="cuda", dtype=torch.float16)

# Random ternary B (store as int32 bitplanes)
words_per_row = (N + 31) // 32
B_nz = torch.randint(0, 2, (K, words_per_row), dtype=torch.int32, device="cuda")
B_sgn = torch.randint(0, 2, (K, words_per_row), dtype=torch.int32, device="cuda")

# Optional bias
bias = torch.randn(N, device="cuda", dtype=torch.float32)

# Forward
C = ternary_linear_cuda.forward(A, B_nz, B_sgn, bias, alpha=1.0, relu=True)
print("C shape:", C.shape)
print("C dtype:", C.dtype)
print("C[0,:5]:", C[0, :5])
