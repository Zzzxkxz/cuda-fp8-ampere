#pragma once

#include <cstddef>
#include <cstdint>

// Public, reusable API for the IMMA FP8-as-storage kernels.
//
// Notes:
// - This API is intentionally minimal and CUDA-centric.
// - It is designed to be callable from the bench harness, a PyTorch extension,
//   and a tinygrad wrapper.

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

namespace fp8imma {

// Initializes the internal FP8(E4M3)->FP16 LUT in constant memory.
// Safe to call multiple times.
void init_fp8_e4m3_lut();

// v2: INT8 activations (row-major) + FP8(E4M3) weights (col-major), JIT transcode FP8->INT8.
// Output: FP16, col-major.
// KChunk must be 32 or 64.
#ifdef __CUDACC__
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
    cudaStream_t stream);

// v2 experimental: precompute a per-column FP8->INT8 table in shared memory.
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
    cudaStream_t stream);

// v3: fused A quantization (register path) + FP8->INT8 JIT.
cudaError_t launch_imma_fp8_actquant_v3_f16(
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
    cudaStream_t stream);

cudaError_t launch_imma_fp8_actquant_v3_bf16(
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
    cudaStream_t stream);

// v4: FP16/BF16 activations (row-major) + FP8(E4M3) weights (col-major), fused act quant to INT8.
// Output: FP16, col-major.
// KChunk must be 32 or 64.
// a_inv_scale is the multiplier applied before sat-cvt to int8.

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
    cudaStream_t stream);

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
    cudaStream_t stream);

// v4_texscale: loads per-column scales via texture object bound to u16 scale bits.
cudaError_t launch_imma_fp8_actquant_v4_texscale_f16(
    int kChunk,
    const half* A_row_f16,
    const uint8_t* B_col_fp8,
    cudaTextureObject_t scale_tex_u16,
    half* D_col_f16,
    int M,
    int N,
    int K,
    float global_scale,
    float a_inv_scale,
    cudaStream_t stream);

cudaError_t launch_imma_fp8_actquant_v4_texscale_bf16(
    int kChunk,
    const __nv_bfloat16* A_row_bf16,
    const uint8_t* B_col_fp8,
    cudaTextureObject_t scale_tex_u16,
    half* D_col_f16,
    int M,
    int N,
    int K,
    float global_scale,
    float a_inv_scale,
    cudaStream_t stream);
#endif

} // namespace fp8imma

// ------------------------ C ABI (for ctypes/tinygrad) ------------------------
// We keep the ABI very simple:
// - pointers are passed as void*
// - the CUDA stream is passed as a void* (cudaStream_t)
// - return value is a CUDA error code (0 == success)

#ifdef __cplusplus
extern "C" {
#endif

void fp8imma_init_fp8_e4m3_lut();

int fp8imma_launch_imma_fp8_jit_v2(
    int kChunk,
    const void* A_row_i8,
    const void* B_col_fp8,
    const void* col_scales_u16,
    void* D_col_f16,
    int M,
    int N,
    int K,
    float global_scale,
    void* stream);

int fp8imma_launch_imma_fp8_jit_v2_i8lut(
    int kChunk,
    const void* A_row_i8,
    const void* B_col_fp8,
    const void* col_scales_u16,
    void* D_col_f16,
    int M,
    int N,
    int K,
    float global_scale,
    void* stream);

int fp8imma_launch_imma_fp8_actquant_v3_f16(
    int kChunk,
    const void* A_row_f16,
    const void* B_col_fp8,
    const void* col_scales_u16,
    void* D_col_f16,
    int M,
    int N,
    int K,
    float global_scale,
    float a_inv_scale,
    void* stream);

int fp8imma_launch_imma_fp8_actquant_v3_bf16(
    int kChunk,
    const void* A_row_bf16,
    const void* B_col_fp8,
    const void* col_scales_u16,
    void* D_col_f16,
    int M,
    int N,
    int K,
    float global_scale,
    float a_inv_scale,
    void* stream);

int fp8imma_launch_imma_fp8_actquant_v4_texscale_f16(
    int kChunk,
    const void* A_row_f16,
    const void* B_col_fp8,
    void* scale_tex_u16,
    void* D_col_f16,
    int M,
    int N,
    int K,
    float global_scale,
    float a_inv_scale,
    void* stream);

int fp8imma_launch_imma_fp8_actquant_v4_texscale_bf16(
    int kChunk,
    const void* A_row_bf16,
    const void* B_col_fp8,
    void* scale_tex_u16,
    void* D_col_f16,
    int M,
    int N,
    int K,
    float global_scale,
    float a_inv_scale,
    void* stream);

int fp8imma_launch_imma_fp8_actquant_v4_f16(
    int kChunk,
    const void* A_row_f16,
    const void* B_col_fp8,
    const void* col_scales_u16,
    void* D_col_f16,
    int M,
    int N,
    int K,
    float global_scale,
    float a_inv_scale,
    void* stream);

int fp8imma_launch_imma_fp8_actquant_v4_bf16(
    int kChunk,
    const void* A_row_bf16,
    const void* B_col_fp8,
    const void* col_scales_u16,
    void* D_col_f16,
    int M,
    int N,
    int K,
    float global_scale,
    float a_inv_scale,
    void* stream);

#ifdef __cplusplus
} // extern "C"
#endif
