cmake_minimum_required(VERSION 3.22)
project(rtx3090_fp8_exps LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# RTX 3090 is SM86.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

add_executable(gpu_bench
  src/main.cu
)

add_library(fp8imma_kernels STATIC
  src/fp8imma/imma_fp8_kernels.cu
)

add_library(fp8imma_kernels_shared SHARED
  src/fp8imma/imma_fp8_kernels.cu
)

target_include_directories(fp8imma_kernels PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(fp8imma_kernels_shared PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(fp8imma_kernels_shared PROPERTIES
  OUTPUT_NAME fp8imma
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
