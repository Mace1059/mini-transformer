# benchmark.py
import torch
import torch.nn as nn
import time
from tqdm import tqdm

from src.ternary_packed import TernaryPackedLinear

# -------------------------------
# Benchmark configuration
# -------------------------------
B = 8             # batch size (smaller for laptop GPU)
d_model = 512     # input dim (smaller test size)
ffn_mult = 4
d_hidden = d_model * ffn_mult

# pad hidden dim to multiple of 32
d_hidden = ((d_hidden + 31) // 32) * 32

n_iter = 10       # fewer iterations for quick test
device = "cuda"

torch.manual_seed(0)

# -------------------------------
# Inputs
# -------------------------------
x = torch.randn(B, d_model, device=device, dtype=torch.float16)

# -------------------------------
# Baseline FP16 Linear (cuBLAS)
# -------------------------------
linear = nn.Linear(d_model, d_hidden, bias=False, device=device, dtype=torch.float16)

torch.cuda.synchronize()
print("\n[Baseline cuBLAS FP16]")
start = time.time()
for _ in tqdm(range(n_iter), desc="FP16 Linear"):
    out = linear(x)
    torch.cuda.synchronize()
end = time.time()

elapsed = end - start
print(f"Total time: {elapsed:.3f} s")
print(f"Avg per forward: {elapsed / n_iter * 1000:.3f} ms")
print(f"Throughput: {n_iter/elapsed:.2f} it/s")

# -------------------------------
# Ternary Linear
# -------------------------------
print("\n[Now testing Ternary]")
torch.cuda.synchronize()

w_int = torch.randint(-1, 2, (d_hidden, d_model), device=device, dtype=torch.int8)

layer = TernaryPackedLinear(d_model, d_hidden).to(device)
with torch.no_grad():
    layer.set_weights(w_int)

torch.cuda.synchronize()
print("\n[TernaryPackedLinear (bit-serial)]")
start = time.time()
for _ in tqdm(range(n_iter), desc="Ternary Linear"):
    out = layer(x)
    torch.cuda.synchronize()
end = time.time()

torch.cuda.synchronize()
elapsed = time.time() - start
print(f"Total time: {elapsed:.3f} s")
print(f"Avg per forward: {elapsed / n_iter * 1000:.3f} ms")
print(f"Throughput: {n_iter/elapsed:.2f} it/s")
