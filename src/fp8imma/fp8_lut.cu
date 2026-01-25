#include "fp8imma/imma_fp8.h"

#include "fp8imma_internal.cuh"

// Single definition of the constant LUT used by all kernels.
// Must be in global scope to match the extern declaration in fp8imma_internal.cuh.
__constant__ uint16_t fp8imma_k_fp8_e4m3_to_f16_bits[256];

namespace fp8imma {

void init_fp8_e4m3_lut() {
  static bool inited = false;
  if (inited) return;

  uint16_t host_lut[256];
  for (int i = 0; i < 256; ++i) {
    host_lut[i] = fp8imma_internal::f32_to_f16_bits(fp8imma_internal::fp8_e4m3_to_f32((uint8_t)i));
  }

  // Best-effort: if this fails, downstream launch wrappers return CUDA error codes.
  (void)cudaMemcpyToSymbol(fp8imma_k_fp8_e4m3_to_f16_bits, host_lut, sizeof(host_lut));
  inited = true;
}

} // namespace fp8imma
