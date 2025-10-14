# src/ternary_packed.py
import torch
import torch.nn as nn
import ternary_linear_cuda  # built by setup.py
import numpy as np

@torch.no_grad()
def pack_ternary_weights(W_int8: torch.Tensor):
    """
    Input:  W_int8 [K, N] with values in {-1,0,1}  (K=in_dim, N=out_dim)
    Output: B_nz/B_sg [K, ceil(N/32)] as int32 bitplanes
    """
    K, N = W_int8.shape
    words = (N + 31) // 32
    B_nz = torch.empty((K, words), dtype=torch.int32, device=W_int8.device)
    B_sg = torch.empty((K, words), dtype=torch.int32, device=W_int8.device)

    for k in range(K):
        for w in range(words):
            bits_nz, bits_sg = 0, 0
            base = w * 32
            limit = min(32, N - base)
            for b in range(limit):
                v = int(W_int8[k, base + b].item())
                if v == 0:
                    continue
                bits_nz |= (1 << b)
                if v < 0:
                    bits_sg |= (1 << b)

            # Mask to 32 bits and reinterpret to signed int
            nz32 = np.uint32(bits_nz).view(np.int32)
            sg32 = np.uint32(bits_sg).view(np.int32)
            B_nz[k, w] = nz32
            B_sg[k, w] = sg32

    return B_nz, B_sg

@torch.no_grad()
def unpack_to_dense(B_nz: torch.Tensor, B_sg: torch.Tensor, N: int) -> torch.Tensor:
    """
    Reconstruct dense ternary matrix from bitplanes.
    Inputs:
      B_*: [K, ceil(N/32)] int32
      N:   out_dim
    Returns:
      W: [N, K] == [out_dim, in_dim] in int8 with {-1,0,1}
    """
    K = B_nz.size(0)
    words = B_nz.size(1)
    WkN = torch.zeros(K, N, device=B_nz.device, dtype=torch.int8)
    for k in range(K):
        for w in range(words):
            nz = int(B_nz[k, w].item()) & 0xFFFFFFFF
            sg = int(B_sg[k, w].item()) & 0xFFFFFFFF
            base = w * 32
            bits = min(32, N - base)
            for b in range(bits):
                if (nz >> b) & 1:
                    WkN[k, base + b] = -1 if ((sg >> b) & 1) else 1
    return WkN.t().contiguous()  # [N, K] == [out_dim, in_dim]

class TernaryLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B_nz, B_sgn, bias, alpha: float = 1.0, relu: bool = False):
        """
        x: [M, K] fp16
        B_*: [K, ceil(N/32)] int32
        bias: [N] fp32 or None
        returns: [M, N] fp16
        """
        assert x.is_cuda and x.dtype == torch.float16
        assert B_nz.is_cuda and B_sgn.is_cuda and B_nz.dtype == torch.int32 and B_sgn.dtype == torch.int32
        out = ternary_linear_cuda.forward(x, B_nz, B_sgn, bias, float(alpha), bool(relu))
        return out

class TernaryPackedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, alpha=1.0, relu=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.relu = relu
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)
        # init buffers (correct shapes)
        self.register_buffer("B_nz", torch.zeros(in_features, (out_features + 31) // 32, dtype=torch.int32))
        self.register_buffer("B_sgn", torch.zeros_like(self.B_nz))

    @torch.no_grad()
    def set_weights(self, W_int8: torch.Tensor):
        """
        W_int8: [out_features, in_features], values in {-1,0,1}
        """
        assert W_int8.shape == (self.out_features, self.in_features)
        # Transpose so we have [K=in_dim, N=out_dim]
        W_t = W_int8.t().contiguous()
        B_nz, B_sg = pack_ternary_weights(W_t)
        # Replace buffers
        self.register_buffer("B_nz", B_nz)
        self.register_buffer("B_sgn", B_sg)

    def forward(self, x: torch.Tensor):
        return TernaryLinearFn.apply(x, self.B_nz, self.B_sgn, self.bias, self.alpha, self.relu)
