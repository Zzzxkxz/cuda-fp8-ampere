from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fp8wgt_ext",
    ext_modules=[
        CUDAExtension(
            name="fp8wgt_ext",
            sources=["fp8wgt_ext.cpp", "fp8wgt_ext_cuda.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
