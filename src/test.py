import torch, time
from src.ternary_packed import TernaryPackedLinear
from src.utils import TernaryLinear

B, K, N = 256, 1024, 2048
x = torch.randn(B, K, device="cuda", dtype=torch.float16)

lin_fp = torch.nn.Linear(K, N).cuda().half()
lin_simple = TernaryLinear(K, N).cuda()
lin_packed = TernaryPackedLinear(K, N).cuda()

def bench(fn, x, iters=200, profile=False, name=""):
    torch.cuda.synchronize()
    t0 = time.time()

    if profile:
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            for _ in range(iters):
                _ = fn(x)
        torch.cuda.synchronize()
        print(f"\n---- CUDA kernel profile for {name} ----")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    else:
        for _ in range(iters):
            _ = fn(x)

    torch.cuda.synchronize()
    return (time.time() - t0) / iters

print("FP16 Linear:", bench(lin_fp, x))
print("Simple Ternary:", bench(lin_simple, x))

# Run packed with profiling ONCE to see kernel
print("Packed Ternary CUDA:", bench(lin_packed, x, profile=True, name="Packed Ternary CUDA"))
