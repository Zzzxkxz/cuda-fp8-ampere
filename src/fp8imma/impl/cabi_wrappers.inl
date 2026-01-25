extern "C" {

void fp8imma_init_fp8_e4m3_lut() {
  fp8imma::init_fp8_e4m3_lut();
}

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
    void* stream) {
  return (int)fp8imma::launch_imma_fp8_jit_v2(
      kChunk,
      (const int8_t*)A_row_i8,
      (const uint8_t*)B_col_fp8,
      (const uint16_t*)col_scales_u16,
      (half*)D_col_f16,
      M,
      N,
      K,
      global_scale,
      (cudaStream_t)stream);
}

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
    void* stream) {
  return (int)fp8imma::launch_imma_fp8_jit_v2_i8lut(
      kChunk,
      (const int8_t*)A_row_i8,
      (const uint8_t*)B_col_fp8,
      (const uint16_t*)col_scales_u16,
      (half*)D_col_f16,
      M,
      N,
      K,
      global_scale,
      (cudaStream_t)stream);
}

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
    void* stream) {
  return (int)fp8imma::launch_imma_fp8_actquant_v3_f16(
      kChunk,
      (const half*)A_row_f16,
      (const uint8_t*)B_col_fp8,
      (const uint16_t*)col_scales_u16,
      (half*)D_col_f16,
      M,
      N,
      K,
      global_scale,
      a_inv_scale,
      (cudaStream_t)stream);
}

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
    void* stream) {
  return (int)fp8imma::launch_imma_fp8_actquant_v3_bf16(
      kChunk,
      (const __nv_bfloat16*)A_row_bf16,
      (const uint8_t*)B_col_fp8,
      (const uint16_t*)col_scales_u16,
      (half*)D_col_f16,
      M,
      N,
      K,
      global_scale,
      a_inv_scale,
      (cudaStream_t)stream);
}

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
    void* stream) {
  return (int)fp8imma::launch_imma_fp8_actquant_v4_f16(
      kChunk,
      (const half*)A_row_f16,
      (const uint8_t*)B_col_fp8,
      (const uint16_t*)col_scales_u16,
      (half*)D_col_f16,
      M,
      N,
      K,
      global_scale,
      a_inv_scale,
      (cudaStream_t)stream);
}

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
    void* stream) {
  return (int)fp8imma::launch_imma_fp8_actquant_v4_bf16(
      kChunk,
      (const __nv_bfloat16*)A_row_bf16,
      (const uint8_t*)B_col_fp8,
      (const uint16_t*)col_scales_u16,
      (half*)D_col_f16,
      M,
      N,
      K,
      global_scale,
      a_inv_scale,
      (cudaStream_t)stream);
}

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
    void* stream) {
  return (int)fp8imma::launch_imma_fp8_actquant_v4_texscale_f16(
      kChunk,
      (const half*)A_row_f16,
      (const uint8_t*)B_col_fp8,
      (cudaTextureObject_t)scale_tex_u16,
      (half*)D_col_f16,
      M,
      N,
      K,
      global_scale,
      a_inv_scale,
      (cudaStream_t)stream);
}

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
    void* stream) {
  return (int)fp8imma::launch_imma_fp8_actquant_v4_texscale_bf16(
      kChunk,
      (const __nv_bfloat16*)A_row_bf16,
      (const uint8_t*)B_col_fp8,
      (cudaTextureObject_t)scale_tex_u16,
      (half*)D_col_f16,
      M,
      N,
      K,
      global_scale,
      a_inv_scale,
      (cudaStream_t)stream);
}

} // extern "C"
