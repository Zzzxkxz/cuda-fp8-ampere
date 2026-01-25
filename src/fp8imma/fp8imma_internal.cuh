#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

// Internal helpers for the fp8imma kernel library.
// Not part of the public API; may change at any time.

#define FP8IMMA_CUDA_CHECK(call)                                      \
  do {                                                                \
    cudaError_t _e = (call);                                          \
    if (_e != cudaSuccess) {                                          \
      fprintf(stderr, "fp8imma CUDA error %s:%d: %s\n",              \
              __FILE__, __LINE__, cudaGetErrorString(_e));            \
      return _e;                                                      \
    }                                                                 \
  } while (0)

// Defined once in fp8_lut.cu
extern __constant__ uint16_t fp8imma_k_fp8_e4m3_to_f16_bits[256];

namespace fp8imma_internal {

__device__ __forceinline__ int8_t cvt_f16_to_s8_sat(half h) {
  int32_t r;
  unsigned short us = __half_as_ushort(h);
  asm("cvt.rni.sat.s8.f16 %0, %1;" : "=r"(r) : "h"(us));
  return (int8_t)r;
}

template <typename T>
__device__ __forceinline__ half to_half(T v);

template <>
__device__ __forceinline__ half to_half<half>(half v) {
  return v;
}

template <>
__device__ __forceinline__ half to_half<__nv_bfloat16>(__nv_bfloat16 v) {
  return __float2half_rn(__bfloat162float(v));
}

static inline uint16_t f32_to_f16_bits(float x) {
  __half h = __float2half(x);
  return reinterpret_cast<uint16_t&>(h);
}

static inline float fp8_e4m3_to_f32(uint8_t v) {
  const int sign = (v >> 7) & 1;
  const int exp = (v >> 3) & 0xF;
  const int mant = v & 0x7;
  const float s = sign ? -1.0f : 1.0f;
  const int bias = 7;
  if (exp == 0) {
    if (mant == 0) return s * 0.0f;
    return s * std::ldexp((float)mant, (1 - bias) - 3);
  }
  if (exp == 15) {
    if (mant == 0) return s * INFINITY;
    return NAN;
  }
  float frac = 1.0f + ((float)mant / 8.0f);
  return s * std::ldexp(frac, exp - bias);
}

__device__ __forceinline__ half fp8_lut_decode_shared(const uint16_t* __restrict__ lut_s, uint8_t v) {
  uint16_t bits = lut_s[v];
  union {
    uint16_t u;
    __half h;
  } u;
  u.u = bits;
  return u.h;
}

static inline size_t smem_bytes_v2(int kChunk) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  size_t k = (size_t)kChunk;
  size_t bytes_A = 2ull * BM * k;
  size_t bytes_B = 2ull * BN * k;
  size_t bytes_C = (size_t)BM * BN * 4ull;
  size_t bytes_scales = (size_t)BN * 4ull;
  size_t bytes_lut = 256ull * 2ull;
  size_t bytes_inv = (size_t)BN * 2ull;
  return bytes_A + bytes_B + bytes_C + bytes_scales + bytes_lut + bytes_inv;
}

static inline size_t smem_bytes_v2_i8lut(int kChunk) {
  constexpr int BN = 64;
  return smem_bytes_v2(kChunk) + (size_t)BN * 256ull;
}

static inline size_t smem_bytes_v4(int kChunk, size_t aTypeBytes) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  size_t k = (size_t)kChunk;
  size_t bytes_Af = 1ull * BM * k * aTypeBytes;
  size_t bytes_Ai8 = 2ull * BM * k;
  size_t bytes_Bi8 = 2ull * BN * k;
  size_t bytes_C = (size_t)BM * BN * 4ull;
  size_t bytes_scales = (size_t)BN * 4ull;
  size_t bytes_lut = 256ull * 2ull;
  size_t bytes_inv = (size_t)BN * 2ull;
  return bytes_Af + bytes_Ai8 + bytes_Bi8 + bytes_C + bytes_scales + bytes_lut + bytes_inv;
}

} // namespace fp8imma_internal
