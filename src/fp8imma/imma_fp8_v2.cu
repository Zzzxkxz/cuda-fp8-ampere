#include "fp8imma/imma_fp8.h"

#include "fp8imma_internal.cuh"

#include <cuda_pipeline.h>
#include <mma.h>

namespace {

using fp8imma_internal::cvt_f16_to_s8_sat;
using fp8imma_internal::fp8_lut_decode_shared;

#include "impl/v2_kernels.inl"

} // namespace

namespace fp8imma {

cudaError_t launch_imma_fp8_jit_v2(
    int kChunk,
    const int8_t* A_row_i8,
    const uint8_t* B_col_fp8,
    const uint16_t* col_scales_u16,
    half* D_col_f16,
    int M,
    int N,
    int K,
    float global_scale,
    cudaStream_t stream) {

  init_fp8_e4m3_lut();

  dim3 block(512);
  dim3 grid((N + 63) / 64, (M + 63) / 64);

  if (kChunk == 32) {
    (void)cudaFuncSetAttribute(
        imma_gemm_fp8_int8_colscale_fp16_colmajor_kernel_v2<32>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)fp8imma_internal::smem_bytes_v2(32));
    imma_gemm_fp8_int8_colscale_fp16_colmajor_kernel_v2<32><<<grid, block, fp8imma_internal::smem_bytes_v2(32), stream>>>(
        A_row_i8, B_col_fp8, col_scales_u16, D_col_f16, M, N, K, global_scale);
    return cudaGetLastError();
  }
  if (kChunk == 64) {
    (void)cudaFuncSetAttribute(
        imma_gemm_fp8_int8_colscale_fp16_colmajor_kernel_v2<64>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)fp8imma_internal::smem_bytes_v2(64));
    imma_gemm_fp8_int8_colscale_fp16_colmajor_kernel_v2<64><<<grid, block, fp8imma_internal::smem_bytes_v2(64), stream>>>(
        A_row_i8, B_col_fp8, col_scales_u16, D_col_f16, M, N, K, global_scale);
    return cudaGetLastError();
  }
  return cudaErrorInvalidValue;
}

cudaError_t launch_imma_fp8_jit_v2_i8lut(
    int kChunk,
    const int8_t* A_row_i8,
    const uint8_t* B_col_fp8,
    const uint16_t* col_scales_u16,
    half* D_col_f16,
    int M,
    int N,
    int K,
    float global_scale,
    cudaStream_t stream) {

  init_fp8_e4m3_lut();

  dim3 block(512);
  dim3 grid((N + 63) / 64, (M + 63) / 64);

  if (kChunk == 32) {
    (void)cudaFuncSetAttribute(
        imma_gemm_fp8_int8_colscale_fp16_colmajor_kernel_v2_i8lut<32>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)fp8imma_internal::smem_bytes_v2_i8lut(32));
    imma_gemm_fp8_int8_colscale_fp16_colmajor_kernel_v2_i8lut<32><<<grid, block, fp8imma_internal::smem_bytes_v2_i8lut(32), stream>>>(
        A_row_i8, B_col_fp8, col_scales_u16, D_col_f16, M, N, K, global_scale);
    return cudaGetLastError();
  }
  if (kChunk == 64) {
    (void)cudaFuncSetAttribute(
        imma_gemm_fp8_int8_colscale_fp16_colmajor_kernel_v2_i8lut<64>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)fp8imma_internal::smem_bytes_v2_i8lut(64));
    imma_gemm_fp8_int8_colscale_fp16_colmajor_kernel_v2_i8lut<64><<<grid, block, fp8imma_internal::smem_bytes_v2_i8lut(64), stream>>>(
        A_row_i8, B_col_fp8, col_scales_u16, D_col_f16, M, N, K, global_scale);
    return cudaGetLastError();
  }
  return cudaErrorInvalidValue;
}

} // namespace fp8imma
