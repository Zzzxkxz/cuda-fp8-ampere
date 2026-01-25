#include "fp8imma/imma_fp8.h"

#include "fp8imma_internal.cuh"

#include <cuda_pipeline.h>
#include <mma.h>

namespace {

using fp8imma_internal::cvt_f16_to_s8_sat;
using fp8imma_internal::fp8_lut_decode_shared;
using fp8imma_internal::to_half;

#include "impl/v4_kernels.inl"

} // namespace

namespace fp8imma {

cudaError_t launch_imma_fp8_actquant_v4_f16(
    int kChunk,
    const half* A_row_f16,
    const uint8_t* B_col_fp8,
    const uint16_t* col_scales_u16,
    half* D_col_f16,
    int M,
    int N,
    int K,
    float global_scale,
    float a_inv_scale,
    cudaStream_t stream) {

  init_fp8_e4m3_lut();

  dim3 block(512);
  dim3 grid((N + 63) / 64, (M + 63) / 64);

  half a_inv_scale_h = __float2half(a_inv_scale);

  if (kChunk == 32) {
    (void)cudaFuncSetAttribute(
        imma_gemm_fp8_actquant_cpasync_int8_colscale_fp16_colmajor_kernel_v4<32, half>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)fp8imma_internal::smem_bytes_v4(32, sizeof(half)));
    imma_gemm_fp8_actquant_cpasync_int8_colscale_fp16_colmajor_kernel_v4<32, half>
        <<<grid, block, fp8imma_internal::smem_bytes_v4(32, sizeof(half)), stream>>>(
            A_row_f16, B_col_fp8, col_scales_u16, D_col_f16, M, N, K, global_scale, a_inv_scale_h);
    return cudaGetLastError();
  }
  if (kChunk == 64) {
    (void)cudaFuncSetAttribute(
        imma_gemm_fp8_actquant_cpasync_int8_colscale_fp16_colmajor_kernel_v4<64, half>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)fp8imma_internal::smem_bytes_v4(64, sizeof(half)));
    imma_gemm_fp8_actquant_cpasync_int8_colscale_fp16_colmajor_kernel_v4<64, half>
        <<<grid, block, fp8imma_internal::smem_bytes_v4(64, sizeof(half)), stream>>>(
            A_row_f16, B_col_fp8, col_scales_u16, D_col_f16, M, N, K, global_scale, a_inv_scale_h);
    return cudaGetLastError();
  }
  return cudaErrorInvalidValue;
}

cudaError_t launch_imma_fp8_actquant_v4_bf16(
    int kChunk,
    const __nv_bfloat16* A_row_bf16,
    const uint8_t* B_col_fp8,
    const uint16_t* col_scales_u16,
    half* D_col_f16,
    int M,
    int N,
    int K,
    float global_scale,
    float a_inv_scale,
    cudaStream_t stream) {

  init_fp8_e4m3_lut();

  dim3 block(512);
  dim3 grid((N + 63) / 64, (M + 63) / 64);

  half a_inv_scale_h = __float2half(a_inv_scale);

  if (kChunk == 32) {
    (void)cudaFuncSetAttribute(
        imma_gemm_fp8_actquant_cpasync_int8_colscale_fp16_colmajor_kernel_v4<32, __nv_bfloat16>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)fp8imma_internal::smem_bytes_v4(32, sizeof(__nv_bfloat16)));
    imma_gemm_fp8_actquant_cpasync_int8_colscale_fp16_colmajor_kernel_v4<32, __nv_bfloat16>
        <<<grid, block, fp8imma_internal::smem_bytes_v4(32, sizeof(__nv_bfloat16)), stream>>>(
            A_row_bf16, B_col_fp8, col_scales_u16, D_col_f16, M, N, K, global_scale, a_inv_scale_h);
    return cudaGetLastError();
  }
  if (kChunk == 64) {
    (void)cudaFuncSetAttribute(
        imma_gemm_fp8_actquant_cpasync_int8_colscale_fp16_colmajor_kernel_v4<64, __nv_bfloat16>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)fp8imma_internal::smem_bytes_v4(64, sizeof(__nv_bfloat16)));
    imma_gemm_fp8_actquant_cpasync_int8_colscale_fp16_colmajor_kernel_v4<64, __nv_bfloat16>
        <<<grid, block, fp8imma_internal::smem_bytes_v4(64, sizeof(__nv_bfloat16)), stream>>>(
            A_row_bf16, B_col_fp8, col_scales_u16, D_col_f16, M, N, K, global_scale, a_inv_scale_h);
    return cudaGetLastError();
  }
  return cudaErrorInvalidValue;
}

} // namespace fp8imma
