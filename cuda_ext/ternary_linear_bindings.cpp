// ternary_linear_bindings.cpp
#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/ATen.h>   // at::Half

// Forward declare launcher from .cu
extern "C"
void launch_ternary_linear_cuda(const at::Half* A,
                                const uint32_t* B_nz,
                                const uint32_t* B_sgn,
                                at::Half* C,
                                int M, int N, int K,
                                float alpha,
                                const float* bias,
                                int relu);

torch::Tensor ternary_linear_forward(torch::Tensor A_half,       // [M,K], float16
                                     torch::Tensor B_nz_i32,     // [K, N/32], int32
                                     torch::Tensor B_sgn_i32,    // [K, N/32], int32
                                     c10::optional<torch::Tensor> bias_opt,
                                     double alpha,
                                     bool relu) {
    TORCH_CHECK(A_half.is_cuda(), "A must be CUDA");
    TORCH_CHECK(B_nz_i32.is_cuda() && B_sgn_i32.is_cuda(), "B bitplanes must be CUDA");
    TORCH_CHECK(A_half.scalar_type() == torch::kHalf, "A must be float16");
    TORCH_CHECK(B_nz_i32.scalar_type() == torch::kInt, "B_nz must be int32");
    TORCH_CHECK(B_sgn_i32.scalar_type() == torch::kInt, "B_sgn must be int32");

    auto M = A_half.size(0);
    auto K = A_half.size(1);
    auto words_per_row = B_nz_i32.size(1);
    TORCH_CHECK(B_nz_i32.size(0) == K && B_sgn_i32.size(0) == K, "B planes K mismatch");
    auto N = words_per_row * 32;

    torch::Tensor C_half = torch::empty({M, N}, A_half.options());

    const float* bias_ptr = nullptr;
    torch::Tensor bias_fp32;
    if (bias_opt.has_value() && bias_opt->defined()) {
        bias_fp32 = bias_opt.value();
        TORCH_CHECK(bias_fp32.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(bias_fp32.scalar_type() == torch::kFloat, "bias must be float32");
        TORCH_CHECK(bias_fp32.numel() == N, "bias length must match N");
        bias_ptr = bias_fp32.data_ptr<float>();
    }

    // Raw pointers
    const at::Half* A_ptr = A_half.data_ptr<at::Half>();

    // Get int32* from tensors, then reinterpret as uint32_t*
    const int32_t* Bnz_i32 = B_nz_i32.data_ptr<int32_t>();
    const int32_t* Bsgn_i32 = B_sgn_i32.data_ptr<int32_t>();
    const uint32_t* Bnz_ptr = reinterpret_cast<const uint32_t*>(Bnz_i32);
    const uint32_t* Bsgn_ptr = reinterpret_cast<const uint32_t*>(Bsgn_i32);

    at::Half* C_ptr = C_half.data_ptr<at::Half>();

    // Call kernel launcher
    launch_ternary_linear_cuda(A_ptr, Bnz_ptr, Bsgn_ptr, C_ptr,
                               M, N, K,
                               static_cast<float>(alpha),
                               bias_ptr,
                               relu ? 1 : 0);

    return C_half;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &ternary_linear_forward,
          "Ternary WMMA GEMM fused (A fp16, B ternary, C fp16) [CUDA]");
}
