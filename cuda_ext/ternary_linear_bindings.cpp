// cuda/ternary_linear_bindings.cpp
#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/ATen.h>

extern "C"
void launch_ternary_linear_cuda(const at::Half* A,
                                const uint32_t* B_nz,
                                const uint32_t* B_sgn,
                                at::Half* C,
                                int M, int N, int K,
                                float alpha,
                                const float* bias,
                                int relu);

extern "C"
void launch_ternary_linear_naive(const at::Half* A,
                                 const uint32_t* B_nz,
                                 const uint32_t* B_sgn,
                                 at::Half* C,
                                 int M, int N, int K,
                                 float alpha,
                                 const float* bias,
                                 int relu);

extern "C"
void launch_ternary_decode_to_dense_kn(const uint32_t* B_nz,
                                       const uint32_t* B_sgn,
                                       int8_t* W_kn,
                                       int K, int N);


torch::Tensor debug_forward_naive(torch::Tensor A_half,
                                  torch::Tensor B_nz_i32,
                                  torch::Tensor B_sgn_i32,
                                  c10::optional<torch::Tensor> bias_opt,
                                  double alpha,
                                  bool relu) {
    TORCH_CHECK(A_half.is_cuda(), "A must be CUDA");
    TORCH_CHECK(B_nz_i32.is_cuda() && B_sgn_i32.is_cuda(), "B planes must be CUDA");
    TORCH_CHECK(A_half.scalar_type() == torch::kHalf, "A must be float16");
    TORCH_CHECK(B_nz_i32.scalar_type() == torch::kInt, "B_nz must be int32");
    TORCH_CHECK(B_sgn_i32.scalar_type() == torch::kInt, "B_sgn must be int32");

    int M = (int)A_half.size(0);
    int K = (int)A_half.size(1);
    int words = (int)B_nz_i32.size(1);

    int N = bias_opt.has_value() && bias_opt->defined()
          ? (int)bias_opt->numel()
          : words * 32;

    torch::Tensor C_half = torch::empty({M, N}, A_half.options());

    const float* bias_ptr = nullptr;
    torch::Tensor bias_fp32;
    if (bias_opt.has_value() && bias_opt->defined()) {
        bias_fp32 = bias_opt.value();
        TORCH_CHECK(bias_fp32.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(bias_fp32.scalar_type() == torch::kFloat, "bias must be float32");
        TORCH_CHECK((int)bias_fp32.numel() == N, "bias length must match N");
        bias_ptr = bias_fp32.data_ptr<float>();
    }

    const at::Half* A_ptr = A_half.data_ptr<at::Half>();
    const uint32_t* Bnz_ptr = reinterpret_cast<const uint32_t*>(B_nz_i32.data_ptr<int32_t>());
    const uint32_t* Bsgn_ptr = reinterpret_cast<const uint32_t*>(B_sgn_i32.data_ptr<int32_t>());
    at::Half* C_ptr = C_half.data_ptr<at::Half>();

    launch_ternary_linear_naive(A_ptr, Bnz_ptr, Bsgn_ptr, C_ptr,
                                M, N, K,
                                static_cast<float>(alpha),
                                bias_ptr,
                                relu ? 1 : 0);
    return C_half;
}

torch::Tensor debug_unpack_cuda(torch::Tensor B_nz_i32,
                                torch::Tensor B_sgn_i32,
                                int64_t N) {
    TORCH_CHECK(B_nz_i32.is_cuda() && B_sgn_i32.is_cuda(), "B planes must be CUDA");
    TORCH_CHECK(B_nz_i32.scalar_type() == torch::kInt, "B_nz must be int32");
    TORCH_CHECK(B_sgn_i32.scalar_type() == torch::kInt, "B_sgn must be int32");
    int K = (int)B_nz_i32.size(0);
    auto W_kn = torch::empty({K, (int)N}, B_nz_i32.options().dtype(torch::kChar)); // int8
    const uint32_t* Bnz_ptr = reinterpret_cast<const uint32_t*>(B_nz_i32.data_ptr<int32_t>());
    const uint32_t* Bsgn_ptr = reinterpret_cast<const uint32_t*>(B_sgn_i32.data_ptr<int32_t>());
    int8_t* W_ptr = W_kn.data_ptr<int8_t>();
    launch_ternary_decode_to_dense_kn(Bnz_ptr, Bsgn_ptr, W_ptr, K, (int)N);
    return W_kn; // [K,N]
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &ternary_linear_forward,
          "Ternary FFN bit-serial (A fp16, B ternary packed, C fp16) [CUDA]");
    m.def("debug_forward_naive", &debug_forward_naive, "Debug: slow reference on GPU");
    m.def("debug_unpack_cuda",  &debug_unpack_cuda,  "Debug: unpack on GPU to dense [K,N] int8");
}
