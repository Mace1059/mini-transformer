# src/test.py
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.nn.functional as F
from src.ternary_packed import TernaryPackedLinear, unpack_to_dense

torch.manual_seed(0)
device = "cuda"

# Tiny debug case
B, in_dim, out_dim = 1, 4, 8
x = torch.randn(B, in_dim, device=device, dtype=torch.float16)
w_int = torch.randint(-1, 2, (out_dim, in_dim), device=device, dtype=torch.int8)

ref_out = F.linear(x, w_int.to(torch.float16))

layer = TernaryPackedLinear(in_dim, out_dim).to(device)
with torch.no_grad():
    layer.set_weights(w_int)

# Guardrail
B_nz, B_sg = layer.B_nz, layer.B_sgn
W_dense = unpack_to_dense(B_nz, B_sg, out_dim)         # [out_dim, in_dim]
assert (W_dense - w_int).abs().max().item() == 0

cpu_out = F.linear(x, W_dense.to(torch.float16))
cuda_out = layer(x)

print("w_int:")
print(w_int.cpu().numpy())
print("B_nz:", B_nz.cpu().numpy())
print("B_sgn:", B_sg.cpu().numpy())
print("\nref_out (dense):", ref_out.cpu().numpy())
print("cpu_out (unpack):", cpu_out.cpu().numpy())
print("cuda_out (kernel):", cuda_out.detach().cpu().numpy())
print("\nmax abs diff (ref vs cpu):", (ref_out - cpu_out).abs().max().item())
print("max abs diff (ref vs cuda):", (ref_out - cuda_out).abs().max().item())

# Medium correctness
B, in_dim, out_dim = 16, 512, 2048
x = torch.randn(B, in_dim, device=device, dtype=torch.float16)
w_int = torch.randint(-1, 2, (out_dim, in_dim), device=device, dtype=torch.int8)

ref_fp16 = F.linear(x, w_int.to(torch.float16))
ref_fp32acc = (x.float() @ w_int.float().t()).half()

layer = TernaryPackedLinear(in_dim, out_dim).to(device)
with torch.no_grad():
    layer.set_weights(w_int)

B_nz, B_sg = layer.B_nz, layer.B_sgn
W_dense = unpack_to_dense(B_nz, B_sg, out_dim)
assert (W_dense - w_int).abs().max().item() == 0

out = layer(x)
print("\nMEDIUM max abs diff vs FP16-accum ref:", (ref_fp16 - out).abs().max().item())
print("MEDIUM max abs diff vs FP32-accum ref:", (ref_fp32acc - out).abs().max().item())
