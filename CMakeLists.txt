cmake_minimum_required(VERSION 3.22)
project(rtx3090_fp8_exps LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

option(FP8IMMA_ENABLE_TORCH_TESTS "Enable building/running PyTorch extension smoke tests (skips at runtime if torch/cuda unavailable)" ON)

# RTX 3090 is SM86.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

add_executable(gpu_bench
  src/gpu_bench.cu
)

set(FP8IMMA_SOURCES
  src/fp8imma/fp8_lut.cu
  src/fp8imma/imma_fp8_v2.cu
  src/fp8imma/imma_fp8_v3.cu
  src/fp8imma/imma_fp8_v4.cu
  src/fp8imma/imma_fp8_v4_texscale.cu
  src/fp8imma/imma_fp8_cabi.cu
)

add_library(fp8imma_kernels STATIC
  ${FP8IMMA_SOURCES}
)

add_library(fp8imma_kernels_shared SHARED
  ${FP8IMMA_SOURCES}
)

target_include_directories(fp8imma_kernels PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src/fp8imma
)
target_include_directories(fp8imma_kernels_shared PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src/fp8imma
)

set_target_properties(fp8imma_kernels_shared PROPERTIES
  OUTPUT_NAME fp8imma
)

set_target_properties(fp8imma_kernels PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

set_target_properties(fp8imma_kernels_shared PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(gpu_bench PRIVATE fp8imma_kernels)

target_include_directories(gpu_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/tools/util/include
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(gpu_bench PRIVATE CUDA::cublas CUDA::cublasLt)

# NVTX (optional): enables NVTX ranges for Nsight Systems/Compute.
if(TARGET CUDAToolkit::nvToolsExt)
  target_link_libraries(gpu_bench PRIVATE CUDAToolkit::nvToolsExt)
elseif(TARGET CUDA::nvToolsExt)
  target_link_libraries(gpu_bench PRIVATE CUDA::nvToolsExt)
else()
  find_library(NVTOOLSEXT_LIBRARY nvToolsExt)
  if(NVTOOLSEXT_LIBRARY)
    target_link_libraries(gpu_bench PRIVATE ${NVTOOLSEXT_LIBRARY})
  endif()
endif()

target_compile_options(gpu_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

target_compile_options(fp8imma_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
target_compile_options(fp8imma_kernels_shared PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

# Optional: show register count / spill stats in build output.
option(GPU_BENCH_PTXAS_VERBOSE "Enable ptxas verbose output (-Xptxas=-v)" OFF)
if(GPU_BENCH_PTXAS_VERBOSE)
  target_compile_options(gpu_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
  )
endif()

# Optional: enable more aggressive optimization in Release.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ------------------------------ Tests (CTest) ------------------------------
include(CTest)
if(BUILD_TESTING)
  enable_testing()

  add_executable(fp8imma_smoke
    tests/fp8imma_smoke.cu
  )

  target_link_libraries(fp8imma_smoke PRIVATE
    fp8imma_kernels
    CUDA::cudart
  )

  target_include_directories(fp8imma_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

  target_compile_options(fp8imma_smoke PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  )

  add_test(NAME fp8imma.smoke COMMAND $<TARGET_FILE:fp8imma_smoke>)

  # Basic sanity: ensure the benchmark harness can execute key benches.
  add_test(NAME gpu_bench.list COMMAND $<TARGET_FILE:gpu_bench> --list)
  add_test(NAME gpu_bench.fp8imma_v2 COMMAND $<TARGET_FILE:gpu_bench> --bench imma_fp8_jit_v2 --M 64 --N 64 --K 64 --iters 1)
  add_test(NAME gpu_bench.fp8imma_v4 COMMAND $<TARGET_FILE:gpu_bench> --bench imma_fp8_jit_v4_act_f16 --M 64 --N 64 --K 64 --iters 1)

  if(FP8IMMA_ENABLE_TORCH_TESTS)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
      add_test(
        NAME torch.fp8imma_ext.smoke
        COMMAND ${Python3_EXECUTABLE} -u ${CMAKE_SOURCE_DIR}/tests/torch_fp8imma_ext_smoke.py
      )
    else()
      message(STATUS "Python3 interpreter not found; skipping torch.* tests")
    endif()
  endif()
endif()
