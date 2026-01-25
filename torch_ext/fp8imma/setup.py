import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

import torch


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


FP8IMMA_BUILD_DIR = os.path.join(_repo_root(), "build")

CUDA_HOME = os.environ.get("CUDA_HOME", "/opt/cuda")
CUDA_INCLUDE_DIR = os.path.join(CUDA_HOME, "include")

TORCH_LIB_DIR = os.path.join(os.path.dirname(torch.__file__), "lib")

setup(
    name="fp8imma_ext",
    ext_modules=[
        CppExtension(
            name="fp8imma_ext",
            sources=[
                "fp8imma_ext.cpp",
            ],
            include_dirs=[CUDA_INCLUDE_DIR],
            library_dirs=[FP8IMMA_BUILD_DIR, TORCH_LIB_DIR],
            libraries=["fp8imma"],
            extra_compile_args={
                "cxx": ["-O3"],
            },
            extra_link_args=[
                f"-Wl,-rpath,{FP8IMMA_BUILD_DIR}",
                f"-Wl,-rpath,{TORCH_LIB_DIR}",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
