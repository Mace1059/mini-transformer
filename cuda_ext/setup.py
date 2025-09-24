from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="ternary_linear_cuda",
    ext_modules=[
        CUDAExtension(
            name="ternary_linear_cuda",
            sources=[
                "ternary_linear_bindings.cpp",
                "ternary_linear_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["/O2"],  # for MSVC
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                    "--extended-lambda",
                    "--expt-relaxed-constexpr",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_89,code=sm_89",
                    "-gencode=arch=compute_90,code=sm_90",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
