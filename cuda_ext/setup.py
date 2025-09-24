from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="ternary_linear_cuda",
    ext_modules=[
        CUDAExtension(
            name="ternary_linear_cuda",
            sources=["ternary_linear_bindings.cpp", "ternary_linear_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "--maxrregcount=64",
                    "--extra-device-vectorization",
                    "-lineinfo"
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
