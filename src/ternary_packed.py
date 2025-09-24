# # ternary_packed.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Function

# # after you build the extension (setup.py), this import will work
# from cuda_ext import ternary_linear_cuda

# @torch.no_grad()
# def pack_bitplanes(W_tern: torch.Tensor):
#     """
#     W_tern: (out_features=N, in_features=K) in {-1,0,1} on CUDA
#     Returns:
#       NZ, SGN: (N, K//32) uint32 bitplanes
#         nz bit = 1 if weight != 0
#         sgn bit = 1 if weight == -1
#     """
#     N, K = W_tern.shape
#     assert K % 32 == 0, "in_features (K) must be a multiple of 32 for packing."
#     words = K // 32
#     NZ  = torch.zeros((N, words), dtype=torch.int32, device=W_tern.device)
#     SGN = torch.zeros((N, words), dtype=torch.int32, device=W_tern.device)

#     # vectorized per 32-chunk packing
#     for w in range(words):
#         block = W_tern[:, w*32:(w+1)*32]  # (N, 32)
#         # block ∈ {-1,0,1}
#         nz = (block != 0)
#         sgn = (block < 0)
#         # pack bits along dim=1
#         # create [1,2,4,...,2^31] mask on device
#         mask = (1 << torch.arange(32, device=W_tern.device, dtype=torch.int32)).unsqueeze(0)  # (1,32)
#         NZ[:, w]  = (nz.int()  * mask).sum(dim=1)
#         SGN[:, w] = (sgn.int() * mask).sum(dim=1)
#     return NZ, SGN


# class TernaryPackedLinearFn(Function):
#     @staticmethod
#     def forward(ctx, x, weight_fp, bias, thresh, alpha):
#         """
#         x: (M,K) float/half on CUDA
#         weight_fp: (N,K) float (master weights, trainable)
#         bias: (N) float or None
#         thresh: scalar float threshold for ternary
#         alpha: (N) float per-row scale (optional; can be None)
#         """
#         assert x.is_cuda and weight_fp.is_cuda, "x and weight_fp must be CUDA tensors"
#         M, K = x.shape
#         N = weight_fp.shape[0]
#         assert weight_fp.shape[1] == K

#         # ternarize (STE): {-1,0,1}
#         with torch.no_grad():
#             Wt = torch.zeros_like(weight_fp)
#             Wt[weight_fp >  thresh] =  1.0
#             Wt[weight_fp < -thresh] = -1.0

#         # pack bitplanes
#         NZ, SGN = pack_bitplanes(Wt)

#         # run CUDA forward (FP16 input recommended, FP32 accum output)
#         x_in = x
#         if x_in.dtype == torch.float32:
#             # kernel expects half for best perf; casting here is fine
#             x_in = x_in.to(torch.float16)

#         y = ternary_linear_cuda.forward(
#             x_in, NZ, SGN,
#             alpha if alpha is not None else torch.empty(0, device=x.device),
#             bias  if bias  is not None else torch.empty(0, device=x.device)
#         )  # (M,N) float32

#         # save for backward (STE)
#         ctx.save_for_backward(x, weight_fp, Wt, alpha)
#         ctx.has_bias = bias is not None
#         ctx.thresh = thresh
#         return y

#     @staticmethod
#     def backward(ctx, grad_y):
#         """
#         STE backward:
#           grad_x = grad_y @ W_tern
#           grad_w = grad_y^T @ x   (straight-through; no mask by sign)
#           grad_alpha = (grad_y * (x @ W_tern^T)).sum(dim=0)  (optional)
#           grad_bias = grad_y.sum(dim=0)
#         """
#         x, weight_fp, Wt, alpha = ctx.saved_tensors
#         # grad wrt input
#         grad_x = grad_w = grad_b = grad_alpha = grad_thresh = None

#         if ctx.needs_input_grad[0]:  # x
#             grad_x = grad_y @ Wt

#         if ctx.needs_input_grad[1]:  # weight_fp
#             grad_w = grad_y.t() @ x  # STE; simple, effective

#         if ctx.has_bias and ctx.needs_input_grad[2]:  # bias
#             grad_b = grad_y.sum(dim=0)
#         else:
#             grad_b = None

#         # no grads for thresh (hyperparameter)
#         grad_thresh = None

#         if alpha is not None and ctx.needs_input_grad[4]:
#             # alpha scales each output row; dL/dalpha = sum over batch of (grad_y * prealpha_out)
#             # prealpha_out ≈ x @ W_tern^T
#             pre = x @ Wt.t()
#             grad_alpha = (grad_y * pre).sum(dim=0)
#         else:
#             grad_alpha = None

#         return grad_x, grad_w, grad_b, grad_thresh, grad_alpha


# class TernaryPackedLinear(nn.Module):
#     """
#     Drop-in Linear:
#       - keeps float master weights (trainable)
#       - forward packs to 2-bit and calls CUDA kernel
#       - optional per-row scale alpha
#     """
#     def __init__(self, in_features, out_features, bias=True, ternary_threshold=0.0, use_alpha=True):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.ternary_threshold = float(ternary_threshold)
#         self.weight_fp = nn.Parameter(torch.empty(out_features, in_features))
#         self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
#         self.alpha = nn.Parameter(torch.ones(out_features)) if use_alpha else None
#         nn.init.xavier_uniform_(self.weight_fp)

#     def forward(self, x):
#         return TernaryPackedLinearFn.apply(
#             x, self.weight_fp, self.bias, self.ternary_threshold, self.alpha
#         )


import torch
import torch.nn as nn
from torch.autograd import Function

# Import compiled CUDA extension
import cuda_ext


class TernaryPackedLinearFn(Function):
    @staticmethod
    def forward(ctx, x, nz, sgn, bias, alpha):
        y = cuda_ext.ternary_linear_forward(x, nz, sgn, alpha, bias)
        ctx.save_for_backward(x, nz, sgn, alpha)
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, nz, sgn, alpha = ctx.saved_tensors
        grad_x = grad_b = grad_alpha = None

        if ctx.needs_input_grad[0]:
            grad_x = cuda_ext.ternary_linear_backward_x(grad_y, nz, sgn)

        if ctx.has_bias and ctx.needs_input_grad[3]:
            grad_b = grad_y.sum(dim=0)

        if alpha is not None and ctx.needs_input_grad[4]:
            grad_alpha = (grad_y * grad_y.mean(dim=0)).sum(dim=0)

        return grad_x, None, None, grad_b, grad_alpha


class TernaryPackedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, use_alpha=True):
        super().__init__()
        assert in_features % 32 == 0, "K must be multiple of 32"
        self.in_features = in_features
        self.out_features = out_features
        words = in_features // 32

        self.nz = nn.Parameter(torch.zeros(out_features, words, dtype=torch.int32))
        self.sgn = nn.Parameter(torch.zeros(out_features, words, dtype=torch.int32))
        self.alpha = nn.Parameter(torch.ones(out_features)) if use_alpha else None
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            w = torch.randint(-1, 2, (self.out_features, self.in_features))
            words = self.in_features // 32
            nz = torch.zeros((self.out_features, words), dtype=torch.int32)
            sgn = torch.zeros((self.out_features, words), dtype=torch.int32)
            for wi in range(words):
                block = w[:, wi*32:(wi+1)*32]
                mask = (1 << torch.arange(32)).to(torch.int32)
                nz[:, wi] = ((block != 0).int() * mask).sum(dim=1)
                sgn[:, wi] = ((block < 0).int() * mask).sum(dim=1)
            self.nz.data.copy_(nz)
            self.sgn.data.copy_(sgn)

    def forward(self, x):
        return TernaryPackedLinearFn.apply(x, self.nz, self.sgn, self.bias, self.alpha)
