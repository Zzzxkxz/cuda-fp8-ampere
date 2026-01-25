// Legacy monolithic translation unit (kept for archaeology).
// The fp8imma library is now built from modular sources in src/fp8imma/*.cu.

#include "fp8imma/imma_fp8.h"

#include <cmath>
#include <cstdint>
#include <cstdio>

#include <cuda_pipeline.h>
#include <mma.h>

namespace {

#define FP8IMMA_CUDA_CHECK(call)                                      \
  do {                                                                \
    cudaError_t _e = (call);                                          \
    if (_e != cudaSuccess) {                                          \
      fprintf(stderr, "fp8imma CUDA error %s:%d: %s\n",               \
              __FILE__, __LINE__, cudaGetErrorString(_e));            \
      return _e;                                                      \
    }                                                                 \
  } while (0)

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

// Private constant LUT for this library (avoid cross-TU device linking requirements).
__constant__ uint16_t fp8imma_k_fp8_e4m3_to_f16_bits[256];

static uint16_t f32_to_f16_bits(float x) {
  __half h = __float2half(x);
  return reinterpret_cast<uint16_t&>(h);
}

static float fp8_e4m3_to_f32(uint8_t v) {
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

// ------------------------ Kernels (ported from src/main.cu) ------------------------

// Variant: precompute a per-column FP8->INT8 table in shared memory.
template <int KChunk>
__global__ void imma_gemm_fp8_int8_colscale_fp16_colmajor_kernel_v2_i8lut(
    const int8_t* __restrict__ A_row,
    const uint8_t* __restrict__ B_col_fp8,
    const uint16_t* __restrict__ scale_u16,
    half* __restrict__ D_col,
    int M,
    int N,
    int K,
    float global_scale) {

  constexpr int WM = 16;
  constexpr int WN = 16;
  constexpr int WK = 16;
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int kWarps = (BM / WM) * (BN / WN);
  constexpr int kKChunk = KChunk;

  int tid = (int)threadIdx.x;
  int warp_id = tid >> 5;
  if (warp_id >= kWarps) return;

  int block_m0 = (int)blockIdx.y * BM;
  int block_n0 = (int)blockIdx.x * BN;
  if (block_m0 >= M || block_n0 >= N) return;

  int warp_m = warp_id >> 2;
  int warp_n = warp_id & 3;

  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a, WM, WN, WK, signed char, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WM, WN, WK, signed char, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WM, WN, WK, int> acc_frag;
  wmma::fill_fragment(acc_frag, 0);

  extern __shared__ uint8_t smem[];
  int8_t* A_sh = reinterpret_cast<int8_t*>(smem);
  int8_t* B_sh = A_sh + 2 * (BM * kKChunk);
  int32_t* C_sh = reinterpret_cast<int32_t*>(B_sh + 2 * (BN * kKChunk));
  float* scales_sh = reinterpret_cast<float*>(C_sh + (BM * BN));
  uint16_t* LutS = reinterpret_cast<uint16_t*>(scales_sh + BN);
  half* inv_scales_sh = reinterpret_cast<half*>(LutS + 256);
  int8_t* i8lut_sh = reinterpret_cast<int8_t*>(inv_scales_sh + BN); // [BN][256]

  if (tid < 256) {
    LutS[tid] = fp8imma_k_fp8_e4m3_to_f16_bits[tid];
  }

  if (tid < BN) {
    int col = block_n0 + tid;
    const __half* scale_h = reinterpret_cast<const __half*>(scale_u16);
    __half h = (col < N) ? scale_h[(size_t)col] : __float2half(0.0f);
    float s = __half2float(h);
    scales_sh[tid] = s;
    float inv = (fabsf(s) > 1e-8f) ? (1.0f / s) : 0.0f;
    inv_scales_sh[tid] = __float2half(inv);
  }
  __syncthreads();

  for (int idx = tid; idx < BN * 256; idx += (int)blockDim.x) {
    int col = idx >> 8;
    uint8_t v8 = (uint8_t)(idx & 0xFF);
    half h = fp8_lut_decode_shared(LutS, v8);
    half inv_s = inv_scales_sh[col];
    h = __hmul(h, inv_s);
    i8lut_sh[idx] = cvt_f16_to_s8_sat(h);
  }
  __syncthreads();

  int stage = 0;
  for (int k0 = 0; k0 < K; k0 += kKChunk) {
    int8_t* A_dst = A_sh + stage * (BM * kKChunk);
    int8_t* B_dst = B_sh + stage * (BN * kKChunk);

    constexpr int kVecElems = 16;
    int a_vecs = (BM * kKChunk) / kVecElems;
    for (int vi = tid; vi < a_vecs; vi += (int)blockDim.x) {
      int elem0 = vi * kVecElems;
      int r = elem0 / kKChunk;
      int c = elem0 - r * kKChunk;
      int gm = block_m0 + r;
      int gk = k0 + c;

      void* dst_ptr = (void*)(A_dst + (size_t)r * kKChunk + c);
      const void* src_ptr = (const void*)(A_row + (size_t)gm * K + gk);

      if (gm < M && (gk + (kVecElems - 1)) < K) {
        __pipeline_memcpy_async(dst_ptr, src_ptr, 16);
      } else {
        int4 v{};
        *reinterpret_cast<int4*>(dst_ptr) = v;
      }
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);

    int b_items = (BN * kKChunk) / 4;
    if (tid < b_items) {
      int vi = tid;
      int col = vi >> 3;
      int k_off = (vi & 7) * 4;
      int gn = block_n0 + col;
      int gk = k0 + k_off;

      uint32_t raw_fp8 = 0;
      if (gn < N && (gk + 3) < K) {
        raw_fp8 = *reinterpret_cast<const uint32_t*>(B_col_fp8 + (size_t)gn * K + gk);
      }

      uint32_t packed_int8 = 0;
      int base = (col << 8);
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        uint8_t v8 = (raw_fp8 >> (b * 8)) & 0xFF;
        int8_t v = i8lut_sh[base + (int)v8];
        packed_int8 |= ((uint32_t)(uint8_t)v) << (b * 8);
      }
      *reinterpret_cast<uint32_t*>(B_dst + (size_t)col * kKChunk + k_off) = packed_int8;
    }

    __syncthreads();

    const int8_t* A_tile0 = A_dst + (size_t)(warp_m * WM) * kKChunk;
    const int8_t* B_tile0 = B_dst + (size_t)(warp_n * WN) * kKChunk;
#pragma unroll
    for (int kk = 0; kk < kKChunk; kk += WK) {
      wmma::load_matrix_sync(a_frag, A_tile0 + kk, kKChunk);
      wmma::load_matrix_sync(b_frag, B_tile0 + kk, kKChunk);
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    __syncthreads();
    stage ^= 1;
  }

  int32_t* C_base = C_sh + (size_t)(warp_n * WN) * BM + (warp_m * WM);
  wmma::store_matrix_sync(C_base, acc_frag, BM, wmma::mem_col_major);
  __syncthreads();

  int total = BM * BN;
  for (int idx = tid; idx < total; idx += (int)blockDim.x) {
    int col = idx / BM;
    int row = idx - col * BM;
    int gm = block_m0 + row;
    int gn = block_n0 + col;
    if (gm < M && gn < N) {
      float s = scales_sh[col] * global_scale;
      float v = (float)C_sh[(size_t)col * BM + row] * s;
      D_col[(size_t)gn * M + gm] = __float2half_rn(v);
    }
  }
}

template <int KChunk>
__global__ void imma_gemm_fp8_int8_colscale_fp16_colmajor_kernel_v2(
    const int8_t* __restrict__ A_row,
    const uint8_t* __restrict__ B_col_fp8,
    const uint16_t* __restrict__ scale_u16,
    half* __restrict__ D_col,
    int M,
    int N,
    int K,
    float global_scale) {

  constexpr int WM = 16;
  constexpr int WN = 16;
  constexpr int WK = 16;
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int kWarps = (BM / WM) * (BN / WN);
  constexpr int kKChunk = KChunk;

  int tid = (int)threadIdx.x;
  int warp_id = tid >> 5;
  if (warp_id >= kWarps) return;

  int block_m0 = (int)blockIdx.y * BM;
  int block_n0 = (int)blockIdx.x * BN;
  if (block_m0 >= M || block_n0 >= N) return;

  int warp_m = warp_id >> 2;
  int warp_n = warp_id & 3;

  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a, WM, WN, WK, signed char, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WM, WN, WK, signed char, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WM, WN, WK, int> acc_frag;
  wmma::fill_fragment(acc_frag, 0);

  extern __shared__ uint8_t smem[];
  int8_t* A_sh = reinterpret_cast<int8_t*>(smem);
  int8_t* B_sh = A_sh + 2 * (BM * kKChunk);
  int32_t* C_sh = reinterpret_cast<int32_t*>(B_sh + 2 * (BN * kKChunk));
  float* scales_sh = reinterpret_cast<float*>(C_sh + (BM * BN));
  uint16_t* LutS = reinterpret_cast<uint16_t*>(scales_sh + BN);
  half* inv_scales_sh = reinterpret_cast<half*>(LutS + 256);

  if (tid < 256) {
    LutS[tid] = fp8imma_k_fp8_e4m3_to_f16_bits[tid];
  }

  if (tid < BN) {
    int col = block_n0 + tid;
    const __half* scale_h = reinterpret_cast<const __half*>(scale_u16);
    __half h = (col < N) ? scale_h[(size_t)col] : __float2half(0.0f);
    float s = __half2float(h);
    scales_sh[tid] = s;
    float inv = (fabsf(s) > 1e-8f) ? (1.0f / s) : 0.0f;
    inv_scales_sh[tid] = __float2half(inv);
  }
  __syncthreads();

  int stage = 0;
  for (int k0 = 0; k0 < K; k0 += kKChunk) {
    int8_t* A_dst = A_sh + stage * (BM * kKChunk);
    int8_t* B_dst = B_sh + stage * (BN * kKChunk);

    constexpr int kVecElems = 16;
    int a_vecs = (BM * kKChunk) / kVecElems;
    for (int vi = tid; vi < a_vecs; vi += (int)blockDim.x) {
      int elem0 = vi * kVecElems;
      int r = elem0 / kKChunk;
      int c = elem0 - r * kKChunk;
      int gm = block_m0 + r;
      int gk = k0 + c;

      void* dst_ptr = (void*)(A_dst + (size_t)r * kKChunk + c);
      const void* src_ptr = (const void*)(A_row + (size_t)gm * K + gk);

      if (gm < M && (gk + (kVecElems - 1)) < K) {
        __pipeline_memcpy_async(dst_ptr, src_ptr, 16);
      } else {
        int4 v{};
        *reinterpret_cast<int4*>(dst_ptr) = v;
      }
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);

    int b_items = (BN * kKChunk) / 4;
    int items_per_col = kKChunk / 4;
    for (int vi = tid; vi < b_items; vi += (int)blockDim.x) {
      int col = vi / items_per_col;
      int inner = vi - col * items_per_col;
      int k_off = inner * 4;

      int gn = block_n0 + col;
      int gk = k0 + k_off;

      uint32_t raw_fp8 = 0;
      if (gn < N && (gk + 3) < K) {
        raw_fp8 = *reinterpret_cast<const uint32_t*>(B_col_fp8 + (size_t)gn * K + gk);
      }

      half inv_s = inv_scales_sh[col];

      uint32_t packed_int8 = 0;
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        uint8_t val8 = (raw_fp8 >> (b * 8)) & 0xFF;
        half val_h = fp8_lut_decode_shared(LutS, val8);
        val_h = __hmul(val_h, inv_s);
        int8_t val_i8 = cvt_f16_to_s8_sat(val_h);
        packed_int8 |= ((uint32_t)(uint8_t)val_i8) << (b * 8);
      }

      *reinterpret_cast<uint32_t*>(B_dst + (size_t)col * kKChunk + k_off) = packed_int8;
    }
    __syncthreads();

    const int8_t* A_tile0 = A_dst + (size_t)(warp_m * WM) * kKChunk;
    const int8_t* B_tile0 = B_dst + (size_t)(warp_n * WN) * kKChunk;
#pragma unroll
    for (int kk = 0; kk < kKChunk; kk += WK) {
      wmma::load_matrix_sync(a_frag, A_tile0 + kk, kKChunk);
      wmma::load_matrix_sync(b_frag, B_tile0 + kk, kKChunk);
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    __syncthreads();

    stage ^= 1;
  }

  int32_t* C_base = C_sh + (size_t)(warp_n * WN) * BM + (warp_m * WM);
  wmma::store_matrix_sync(C_base, acc_frag, BM, wmma::mem_col_major);
  __syncthreads();

  int total = BM * BN;
  for (int idx = tid; idx < total; idx += (int)blockDim.x) {
    int col = idx / BM;
    int row = idx - col * BM;
    int gm = block_m0 + row;
    int gn = block_n0 + col;
    if (gm < M && gn < N) {
      float s = scales_sh[col] * global_scale;
      float v = (float)C_sh[(size_t)col * BM + row] * s;
      D_col[(size_t)gn * M + gm] = __float2half_rn(v);
    }
  }
}

// v3: fused A quantization (register path) + FP8->INT8 JIT.
template <int KChunk, typename AType>
__global__ void imma_gemm_fp8_actquant_int8_colscale_fp16_colmajor_kernel_v3(
    const AType* __restrict__ A_row_f,
    const uint8_t* __restrict__ B_col_fp8,
    const uint16_t* __restrict__ scale_u16,
    half* __restrict__ D_col,
    int M,
    int N,
    int K,
    float global_scale,
    half a_inv_scale_h) {

  constexpr int WM = 16;
  constexpr int WN = 16;
  constexpr int WK = 16;
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int kWarps = (BM / WM) * (BN / WN);
  constexpr int kKChunk = KChunk;

  int tid = (int)threadIdx.x;
  int warp_id = tid >> 5;
  if (warp_id >= kWarps) return;

  int block_m0 = (int)blockIdx.y * BM;
  int block_n0 = (int)blockIdx.x * BN;
  if (block_m0 >= M || block_n0 >= N) return;

  int warp_m = warp_id >> 2;
  int warp_n = warp_id & 3;

  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a, WM, WN, WK, signed char, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WM, WN, WK, signed char, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WM, WN, WK, int> acc_frag;
  wmma::fill_fragment(acc_frag, 0);

  extern __shared__ uint8_t smem[];
  int8_t* A_sh = reinterpret_cast<int8_t*>(smem);
  int8_t* B_sh = A_sh + 2 * (BM * kKChunk);
  int32_t* C_sh = reinterpret_cast<int32_t*>(B_sh + 2 * (BN * kKChunk));
  float* scales_sh = reinterpret_cast<float*>(C_sh + (BM * BN));
  uint16_t* LutS = reinterpret_cast<uint16_t*>(scales_sh + BN);
  half* inv_scales_sh = reinterpret_cast<half*>(LutS + 256);

  if (tid < 256) {
    LutS[tid] = fp8imma_k_fp8_e4m3_to_f16_bits[tid];
  }
  if (tid < BN) {
    int col = block_n0 + tid;
    const __half* scale_h = reinterpret_cast<const __half*>(scale_u16);
    __half h = (col < N) ? scale_h[(size_t)col] : __float2half(0.0f);
    float s = __half2float(h);
    scales_sh[tid] = s;
    float inv = (fabsf(s) > 1e-8f) ? (1.0f / s) : 0.0f;
    inv_scales_sh[tid] = __float2half(inv);
  }
  __syncthreads();

  int stage = 0;
  for (int k0 = 0; k0 < K; k0 += kKChunk) {
    int8_t* A_dst = A_sh + stage * (BM * kKChunk);
    int8_t* B_dst = B_sh + stage * (BN * kKChunk);

    constexpr int kVecElems = 16;
    int a_vecs = (BM * kKChunk) / kVecElems;
    for (int vi = tid; vi < a_vecs; vi += (int)blockDim.x) {
      int elem0 = vi * kVecElems;
      int r = elem0 / kKChunk;
      int c = elem0 - r * kKChunk;
      int gm = block_m0 + r;
      int gk = k0 + c;

      union {
        int4 v;
        int8_t s8[16];
      } pack;
#pragma unroll
      for (int j = 0; j < 16; ++j) pack.s8[j] = 0;

      if (gm < M && (gk + (kVecElems - 1)) < K) {
        const AType* src = A_row_f + (size_t)gm * K + gk;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          half h = to_half<AType>(src[j]);
          h = __hmul(h, a_inv_scale_h);
          pack.s8[j] = cvt_f16_to_s8_sat(h);
        }
      }
      *reinterpret_cast<int4*>(A_dst + (size_t)r * kKChunk + c) = pack.v;
    }

    int b_items = (BN * kKChunk) / 4;
    int items_per_col = kKChunk / 4;
    for (int vi = tid; vi < b_items; vi += (int)blockDim.x) {
      int col = vi / items_per_col;
      int inner = vi - col * items_per_col;
      int k_off = inner * 4;
      int gn = block_n0 + col;
      int gk = k0 + k_off;

      uint32_t raw_fp8 = 0;
      if (gn < N && (gk + 3) < K) {
        raw_fp8 = *reinterpret_cast<const uint32_t*>(B_col_fp8 + (size_t)gn * K + gk);
      }
      half inv_s = inv_scales_sh[col];

      uint32_t packed_int8 = 0;
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        uint8_t val8 = (raw_fp8 >> (b * 8)) & 0xFF;
        half val_h = fp8_lut_decode_shared(LutS, val8);
        val_h = __hmul(val_h, inv_s);
        int8_t val_i8 = cvt_f16_to_s8_sat(val_h);
        packed_int8 |= ((uint32_t)(uint8_t)val_i8) << (b * 8);
      }
      *reinterpret_cast<uint32_t*>(B_dst + (size_t)col * kKChunk + k_off) = packed_int8;
    }
    __syncthreads();

    const int8_t* A_tile0 = A_dst + (size_t)(warp_m * WM) * kKChunk;
    const int8_t* B_tile0 = B_dst + (size_t)(warp_n * WN) * kKChunk;
#pragma unroll
    for (int kk = 0; kk < kKChunk; kk += WK) {
      wmma::load_matrix_sync(a_frag, A_tile0 + kk, kKChunk);
      wmma::load_matrix_sync(b_frag, B_tile0 + kk, kKChunk);
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    __syncthreads();
    stage ^= 1;
  }

  int32_t* C_base = C_sh + (size_t)(warp_n * WN) * BM + (warp_m * WM);
  wmma::store_matrix_sync(C_base, acc_frag, BM, wmma::mem_col_major);
  __syncthreads();

  int total = BM * BN;
  for (int idx = tid; idx < total; idx += (int)blockDim.x) {
    int col = idx / BM;
    int row = idx - col * BM;
    int gm = block_m0 + row;
    int gn = block_n0 + col;
    if (gm < M && gn < N) {
      float s = scales_sh[col] * global_scale;
      float v = (float)C_sh[(size_t)col * BM + row] * s;
      D_col[(size_t)gn * M + gm] = __float2half_rn(v);
    }
  }
}

// v4_texscale: same as v4, but loads per-column scales via TEX (u16 bits).
template <int KChunk, typename AType>
__global__ void imma_gemm_fp8_actquant_cpasync_int8_colscale_fp16_colmajor_kernel_v4_texscale(
    const AType* __restrict__ A_row_f,
    const uint8_t* __restrict__ B_col_fp8,
    cudaTextureObject_t scale_tex_u16,
    half* __restrict__ D_col,
    int M,
    int N,
    int K,
    float global_scale,
    half a_inv_scale_h) {

  constexpr int WM = 16;
  constexpr int WN = 16;
  constexpr int WK = 16;
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int kWarps = (BM / WM) * (BN / WN);
  constexpr int kKChunk = KChunk;

  int tid = (int)threadIdx.x;
  int warp_id = tid >> 5;
  if (warp_id >= kWarps) return;

  int block_m0 = (int)blockIdx.y * BM;
  int block_n0 = (int)blockIdx.x * BN;
  if (block_m0 >= M || block_n0 >= N) return;

  int warp_m = warp_id >> 2;
  int warp_n = warp_id & 3;

  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a, WM, WN, WK, signed char, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WM, WN, WK, signed char, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WM, WN, WK, int> acc_frag;
  wmma::fill_fragment(acc_frag, 0);

  extern __shared__ uint8_t smem[];

  size_t off = 0;
  size_t bytes_A_f = 1ull * BM * (size_t)kKChunk * sizeof(AType);
  AType* A_f_sh = reinterpret_cast<AType*>(smem + off);
  off += bytes_A_f;
  int8_t* A_i8_sh = reinterpret_cast<int8_t*>(smem + off);
  off += 2ull * BM * (size_t)kKChunk;
  int8_t* B_sh = reinterpret_cast<int8_t*>(smem + off);
  off += 2ull * BN * (size_t)kKChunk;
  int32_t* C_sh = reinterpret_cast<int32_t*>(smem + off);
  off += (size_t)BM * BN * sizeof(int32_t);
  float* scales_sh = reinterpret_cast<float*>(smem + off);
  off += (size_t)BN * sizeof(float);
  uint16_t* LutS = reinterpret_cast<uint16_t*>(smem + off);
  off += 256ull * sizeof(uint16_t);
  half* inv_scales_sh = reinterpret_cast<half*>(smem + off);

  if (tid < 256) {
    LutS[tid] = fp8imma_k_fp8_e4m3_to_f16_bits[tid];
  }

  if (tid < BN) {
    int col = block_n0 + tid;
    unsigned short bits = (col < N) ? tex1Dfetch<unsigned short>(scale_tex_u16, col) : (unsigned short)0;
    union {
      unsigned short u;
      half h;
    } cvt;
    cvt.u = bits;
    float s = __half2float(cvt.h);
    scales_sh[tid] = s;
    float inv = (fabsf(s) > 1e-8f) ? (1.0f / s) : 0.0f;
    inv_scales_sh[tid] = __float2half(inv);
  }
  __syncthreads();

  int stage = 0;
  int k0 = 0;

  {
    uint8_t* A_f_stage_bytes = reinterpret_cast<uint8_t*>(A_f_sh);
    int a_copy_items = (int)((BM * (size_t)kKChunk * sizeof(AType)) / 16ull);
    if (tid < a_copy_items) {
      int vi = tid;
      int byte_off = vi * 16;
      int elem0 = byte_off / (int)sizeof(AType);
      int r = elem0 / kKChunk;
      int c = elem0 - r * kKChunk;
      int gm = block_m0 + r;
      int gk = c;

      void* dst_ptr = (void*)(A_f_stage_bytes + (size_t)byte_off);
      const void* src_ptr = (const void*)(A_row_f + (size_t)gm * K + gk);
      if (gm < M && (gk + 7) < K) {
        __pipeline_memcpy_async(dst_ptr, src_ptr, 16);
      } else {
        int4 z{};
        *reinterpret_cast<int4*>(dst_ptr) = z;
      }
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
  }

  {
    uint8_t* A_f_stage_bytes = reinterpret_cast<uint8_t*>(A_f_sh);
    int8_t* A_i8_stage = A_i8_sh;
    int8_t* B_dst = B_sh;

    int a_copy_items = (int)((BM * (size_t)kKChunk * sizeof(AType)) / 16ull);
    if (tid < a_copy_items) {
      int vi = tid;
      int byte_off = vi * 16;
      int elem0 = byte_off / (int)sizeof(AType);
      int r = elem0 / kKChunk;
      int c = elem0 - r * kKChunk;
      const AType* src = reinterpret_cast<const AType*>(A_f_stage_bytes) + (size_t)r * kKChunk + c;

      uint32_t p0 = 0;
      uint32_t p1 = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        half h = to_half<AType>(src[j]);
        h = __hmul(h, a_inv_scale_h);
        int8_t q = cvt_f16_to_s8_sat(h);
        if (j < 4) {
          p0 |= ((uint32_t)(uint8_t)q) << (j * 8);
        } else {
          p1 |= ((uint32_t)(uint8_t)q) << ((j - 4) * 8);
        }
      }
      *reinterpret_cast<uint32_t*>(A_i8_stage + (size_t)r * kKChunk + c) = p0;
      *reinterpret_cast<uint32_t*>(A_i8_stage + (size_t)r * kKChunk + c + 4) = p1;
    }

    int b_items = (BN * kKChunk) / 4;
    int items_per_col = kKChunk / 4;
    for (int vi = tid; vi < b_items; vi += (int)blockDim.x) {
      int col = vi / items_per_col;
      int inner = vi - col * items_per_col;
      int k_off = inner * 4;
      int gn = block_n0 + col;
      int gk = k_off;

      uint32_t raw_fp8 = 0;
      if (gn < N && (gk + 3) < K) {
        raw_fp8 = *reinterpret_cast<const uint32_t*>(B_col_fp8 + (size_t)gn * K + gk);
      }
      half inv_s = inv_scales_sh[col];

      uint32_t packed_int8 = 0;
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        uint8_t val8 = (raw_fp8 >> (b * 8)) & 0xFF;
        half val_h = fp8_lut_decode_shared(LutS, val8);
        val_h = __hmul(val_h, inv_s);
        int8_t val_i8 = cvt_f16_to_s8_sat(val_h);
        packed_int8 |= ((uint32_t)(uint8_t)val_i8) << (b * 8);
      }

      *reinterpret_cast<uint32_t*>(B_dst + (size_t)col * kKChunk + k_off) = packed_int8;
    }

    __syncthreads();
  }

  for (k0 = 0; k0 < K; k0 += kKChunk) {
    int next_k0 = k0 + kKChunk;
    int next_stage = stage ^ 1;

    if (next_k0 < K) {
      uint8_t* A_f_next_bytes = reinterpret_cast<uint8_t*>(A_f_sh);
      int a_copy_items = (int)((BM * (size_t)kKChunk * sizeof(AType)) / 16ull);
      if (tid < a_copy_items) {
        int vi = tid;
        int byte_off = vi * 16;
        int elem0 = byte_off / (int)sizeof(AType);
        int r = elem0 / kKChunk;
        int c = elem0 - r * kKChunk;
        int gm = block_m0 + r;
        int gk = next_k0 + c;

        void* dst_ptr = (void*)(A_f_next_bytes + (size_t)byte_off);
        const void* src_ptr = (const void*)(A_row_f + (size_t)gm * K + gk);
        if (gm < M && (gk + 7) < K) {
          __pipeline_memcpy_async(dst_ptr, src_ptr, 16);
        } else {
          int4 z{};
          *reinterpret_cast<int4*>(dst_ptr) = z;
        }
      }
      __pipeline_commit();
    }

    int8_t* A_i8_stage = A_i8_sh + (size_t)stage * BM * (size_t)kKChunk;
    int8_t* B_dst = B_sh + (size_t)stage * BN * (size_t)kKChunk;
    const int8_t* A_tile0 = A_i8_stage + (size_t)(warp_m * WM) * kKChunk;
    const int8_t* B_tile0 = B_dst + (size_t)(warp_n * WN) * kKChunk;
#pragma unroll
    for (int kk = 0; kk < kKChunk; kk += WK) {
      wmma::load_matrix_sync(a_frag, A_tile0 + kk, kKChunk);
      wmma::load_matrix_sync(b_frag, B_tile0 + kk, kKChunk);
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    if (next_k0 >= K) break;
    __pipeline_wait_prior(0);

    {
      uint8_t* A_f_stage_bytes = reinterpret_cast<uint8_t*>(A_f_sh);
      int8_t* A_i8_next = A_i8_sh + (size_t)next_stage * BM * (size_t)kKChunk;
      int8_t* B_next = B_sh + (size_t)next_stage * BN * (size_t)kKChunk;

      int a_copy_items = (int)((BM * (size_t)kKChunk * sizeof(AType)) / 16ull);
      if (tid < a_copy_items) {
        int vi = tid;
        int byte_off = vi * 16;
        int elem0 = byte_off / (int)sizeof(AType);
        int r = elem0 / kKChunk;
        int c = elem0 - r * kKChunk;
        const AType* src = reinterpret_cast<const AType*>(A_f_stage_bytes) + (size_t)r * kKChunk + c;

        uint32_t p0 = 0;
        uint32_t p1 = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          half h = to_half<AType>(src[j]);
          h = __hmul(h, a_inv_scale_h);
          int8_t q = cvt_f16_to_s8_sat(h);
          if (j < 4) {
            p0 |= ((uint32_t)(uint8_t)q) << (j * 8);
          } else {
            p1 |= ((uint32_t)(uint8_t)q) << ((j - 4) * 8);
          }
        }
        *reinterpret_cast<uint32_t*>(A_i8_next + (size_t)r * kKChunk + c) = p0;
        *reinterpret_cast<uint32_t*>(A_i8_next + (size_t)r * kKChunk + c + 4) = p1;
      }

      int b_items = (BN * kKChunk) / 4;
      int items_per_col = kKChunk / 4;
      for (int vi = tid; vi < b_items; vi += (int)blockDim.x) {
        int col = vi / items_per_col;
        int inner = vi - col * items_per_col;
        int k_off = inner * 4;
        int gn = block_n0 + col;
        int gk = next_k0 + k_off;

        uint32_t raw_fp8 = 0;
        if (gn < N && (gk + 3) < K) {
          raw_fp8 = *reinterpret_cast<const uint32_t*>(B_col_fp8 + (size_t)gn * K + gk);
        }
        half inv_s = inv_scales_sh[col];

        uint32_t packed_int8 = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          uint8_t val8 = (raw_fp8 >> (b * 8)) & 0xFF;
          half val_h = fp8_lut_decode_shared(LutS, val8);
          val_h = __hmul(val_h, inv_s);
          int8_t val_i8 = cvt_f16_to_s8_sat(val_h);
          packed_int8 |= ((uint32_t)(uint8_t)val_i8) << (b * 8);
        }

        *reinterpret_cast<uint32_t*>(B_next + (size_t)col * kKChunk + k_off) = packed_int8;
      }
    }

    __syncthreads();
    stage ^= 1;
  }

  int32_t* C_base = C_sh + (size_t)(warp_n * WN) * BM + (warp_m * WM);
  wmma::store_matrix_sync(C_base, acc_frag, BM, wmma::mem_col_major);
  __syncthreads();

  int total = BM * BN;
  for (int idx = tid; idx < total; idx += (int)blockDim.x) {
    int col = idx / BM;
    int row = idx - col * BM;
    int gm = block_m0 + row;
    int gn = block_n0 + col;
    if (gm < M && gn < N) {
      float s = scales_sh[col] * global_scale;
      float v = (float)C_sh[(size_t)col * BM + row] * s;
      D_col[(size_t)gn * M + gm] = __float2half_rn(v);
    }
  }
}

template <int KChunk, typename AType>
__global__ void imma_gemm_fp8_actquant_cpasync_int8_colscale_fp16_colmajor_kernel_v4(
    const AType* __restrict__ A_row_f,
    const uint8_t* __restrict__ B_col_fp8,
    const uint16_t* __restrict__ scale_u16,
    half* __restrict__ D_col,
    int M,
    int N,
    int K,
    float global_scale,
    half a_inv_scale_h) {

  constexpr int WM = 16;
  constexpr int WN = 16;
  constexpr int WK = 16;
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int kWarps = (BM / WM) * (BN / WN);
  constexpr int kKChunk = KChunk;

  int tid = (int)threadIdx.x;
  int warp_id = tid >> 5;
  if (warp_id >= kWarps) return;

  int block_m0 = (int)blockIdx.y * BM;
  int block_n0 = (int)blockIdx.x * BN;
  if (block_m0 >= M || block_n0 >= N) return;

  int warp_m = warp_id >> 2;
  int warp_n = warp_id & 3;

  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a, WM, WN, WK, signed char, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WM, WN, WK, signed char, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WM, WN, WK, int> acc_frag;
  wmma::fill_fragment(acc_frag, 0);

  extern __shared__ uint8_t smem[];

  size_t off = 0;
  size_t bytes_A_f = 1ull * BM * (size_t)kKChunk * sizeof(AType);
  AType* A_f_sh = reinterpret_cast<AType*>(smem + off);
  off += bytes_A_f;
  int8_t* A_i8_sh = reinterpret_cast<int8_t*>(smem + off);
  off += 2ull * BM * (size_t)kKChunk * sizeof(int8_t);
  int8_t* B_sh = reinterpret_cast<int8_t*>(smem + off);
  off += 2ull * BN * (size_t)kKChunk * sizeof(int8_t);
  int32_t* C_sh = reinterpret_cast<int32_t*>(smem + off);
  off += (size_t)BM * BN * sizeof(int32_t);
  float* scales_sh = reinterpret_cast<float*>(smem + off);
  off += (size_t)BN * sizeof(float);
  uint16_t* LutS = reinterpret_cast<uint16_t*>(smem + off);
  off += 256ull * sizeof(uint16_t);
  half* inv_scales_sh = reinterpret_cast<half*>(smem + off);

  if (tid < 256) {
    LutS[tid] = fp8imma_k_fp8_e4m3_to_f16_bits[tid];
  }

  if (tid < BN) {
    int col = block_n0 + tid;
    const __half* scale_h = reinterpret_cast<const __half*>(scale_u16);
    __half h = (col < N) ? scale_h[(size_t)col] : __float2half(0.0f);
    float s = __half2float(h);
    scales_sh[tid] = s;
    float inv = (fabsf(s) > 1e-8f) ? (1.0f / s) : 0.0f;
    inv_scales_sh[tid] = __float2half(inv);
  }
  __syncthreads();

  int stage = 0;
  int k0 = 0;

  // Prefetch first A stage and wait.
  {
    uint8_t* A_f_stage_bytes = reinterpret_cast<uint8_t*>(A_f_sh);
    int a_copy_items = (int)((BM * (size_t)kKChunk * sizeof(AType)) / 16ull);
    if (tid < a_copy_items) {
      int vi = tid;
      int byte_off = vi * 16;
      int elem0 = byte_off / (int)sizeof(AType);
      int r = elem0 / kKChunk;
      int c = elem0 - r * kKChunk;
      int gm = block_m0 + r;
      int gk = c;

      void* dst_ptr = (void*)(A_f_stage_bytes + (size_t)byte_off);
      const void* src_ptr = (const void*)(A_row_f + (size_t)gm * K + gk);
      if (gm < M && (gk + 7) < K) {
        __pipeline_memcpy_async(dst_ptr, src_ptr, 16);
      } else {
        int4 z{};
        *reinterpret_cast<int4*>(dst_ptr) = z;
      }
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
  }

  // Prepare first stage: quantize A + transcode B.
  {
    uint8_t* A_f_stage_bytes = reinterpret_cast<uint8_t*>(A_f_sh);
    int8_t* A_i8_stage = A_i8_sh;
    int8_t* B_dst = B_sh;

    int a_copy_items = (int)((BM * (size_t)kKChunk * sizeof(AType)) / 16ull);
    if (tid < a_copy_items) {
      int vi = tid;
      int byte_off = vi * 16;
      int elem0 = byte_off / (int)sizeof(AType);
      int r = elem0 / kKChunk;
      int c = elem0 - r * kKChunk;
      const AType* src = reinterpret_cast<const AType*>(A_f_stage_bytes) + (size_t)r * kKChunk + c;

      uint32_t p0 = 0;
      uint32_t p1 = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        half h = to_half<AType>(src[j]);
        h = __hmul(h, a_inv_scale_h);
        int8_t q = cvt_f16_to_s8_sat(h);
        if (j < 4) {
          p0 |= ((uint32_t)(uint8_t)q) << (j * 8);
        } else {
          p1 |= ((uint32_t)(uint8_t)q) << ((j - 4) * 8);
        }
      }
      *reinterpret_cast<uint32_t*>(A_i8_stage + (size_t)r * kKChunk + c) = p0;
      *reinterpret_cast<uint32_t*>(A_i8_stage + (size_t)r * kKChunk + c + 4) = p1;
    }

    int b_items = (BN * kKChunk) / 4;
    int items_per_col = kKChunk / 4;
    for (int vi = tid; vi < b_items; vi += (int)blockDim.x) {
      int col = vi / items_per_col;
      int inner = vi - col * items_per_col;
      int k_off = inner * 4;
      int gn = block_n0 + col;
      int gk = k_off;

      uint32_t raw_fp8 = 0;
      if (gn < N && (gk + 3) < K) {
        raw_fp8 = *reinterpret_cast<const uint32_t*>(B_col_fp8 + (size_t)gn * K + gk);
      }
      half inv_s = inv_scales_sh[col];

      uint32_t packed_int8 = 0;
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        uint8_t val8 = (raw_fp8 >> (b * 8)) & 0xFF;
        half val_h = fp8_lut_decode_shared(LutS, val8);
        val_h = __hmul(val_h, inv_s);
        int8_t val_i8 = cvt_f16_to_s8_sat(val_h);
        packed_int8 |= ((uint32_t)(uint8_t)val_i8) << (b * 8);
      }

      *reinterpret_cast<uint32_t*>(B_dst + (size_t)col * kKChunk + k_off) = packed_int8;
    }

    __syncthreads();
  }

  for (k0 = 0; k0 < K; k0 += kKChunk) {
    int next_k0 = k0 + kKChunk;
    int next_stage = stage ^ 1;

    if (next_k0 < K) {
      uint8_t* A_f_next_bytes = reinterpret_cast<uint8_t*>(A_f_sh);
      int a_copy_items = (int)((BM * (size_t)kKChunk * sizeof(AType)) / 16ull);
      if (tid < a_copy_items) {
        int vi = tid;
        int byte_off = vi * 16;
        int elem0 = byte_off / (int)sizeof(AType);
        int r = elem0 / kKChunk;
        int c = elem0 - r * kKChunk;
        int gm = block_m0 + r;
        int gk = next_k0 + c;

        void* dst_ptr = (void*)(A_f_next_bytes + (size_t)byte_off);
        const void* src_ptr = (const void*)(A_row_f + (size_t)gm * K + gk);
        if (gm < M && (gk + 7) < K) {
          __pipeline_memcpy_async(dst_ptr, src_ptr, 16);
        } else {
          int4 z{};
          *reinterpret_cast<int4*>(dst_ptr) = z;
        }
      }
      __pipeline_commit();
    }

    int8_t* A_i8_stage = A_i8_sh + (size_t)stage * BM * (size_t)kKChunk;
    int8_t* B_dst = B_sh + (size_t)stage * BN * (size_t)kKChunk;
    const int8_t* A_tile0 = A_i8_stage + (size_t)(warp_m * WM) * kKChunk;
    const int8_t* B_tile0 = B_dst + (size_t)(warp_n * WN) * kKChunk;
#pragma unroll
    for (int kk = 0; kk < kKChunk; kk += WK) {
      wmma::load_matrix_sync(a_frag, A_tile0 + kk, kKChunk);
      wmma::load_matrix_sync(b_frag, B_tile0 + kk, kKChunk);
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    if (next_k0 >= K) break;

    __pipeline_wait_prior(0);

    {
      uint8_t* A_f_stage_bytes = reinterpret_cast<uint8_t*>(A_f_sh);
      int8_t* A_i8_next = A_i8_sh + (size_t)next_stage * BM * (size_t)kKChunk;
      int8_t* B_next = B_sh + (size_t)next_stage * BN * (size_t)kKChunk;

      int a_copy_items = (int)((BM * (size_t)kKChunk * sizeof(AType)) / 16ull);
      if (tid < a_copy_items) {
        int vi = tid;
        int byte_off = vi * 16;
        int elem0 = byte_off / (int)sizeof(AType);
        int r = elem0 / kKChunk;
        int c = elem0 - r * kKChunk;
        const AType* src = reinterpret_cast<const AType*>(A_f_stage_bytes) + (size_t)r * kKChunk + c;

        uint32_t p0 = 0;
        uint32_t p1 = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          half h = to_half<AType>(src[j]);
          h = __hmul(h, a_inv_scale_h);
          int8_t q = cvt_f16_to_s8_sat(h);
          if (j < 4) {
            p0 |= ((uint32_t)(uint8_t)q) << (j * 8);
          } else {
            p1 |= ((uint32_t)(uint8_t)q) << ((j - 4) * 8);
          }
        }
        *reinterpret_cast<uint32_t*>(A_i8_next + (size_t)r * kKChunk + c) = p0;
        *reinterpret_cast<uint32_t*>(A_i8_next + (size_t)r * kKChunk + c + 4) = p1;
      }

      int b_items = (BN * kKChunk) / 4;
      int items_per_col = kKChunk / 4;
      for (int vi = tid; vi < b_items; vi += (int)blockDim.x) {
        int col = vi / items_per_col;
        int inner = vi - col * items_per_col;
        int k_off = inner * 4;
        int gn = block_n0 + col;
        int gk = next_k0 + k_off;

        uint32_t raw_fp8 = 0;
        if (gn < N && (gk + 3) < K) {
          raw_fp8 = *reinterpret_cast<const uint32_t*>(B_col_fp8 + (size_t)gn * K + gk);
        }
        half inv_s = inv_scales_sh[col];

        uint32_t packed_int8 = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          uint8_t val8 = (raw_fp8 >> (b * 8)) & 0xFF;
          half val_h = fp8_lut_decode_shared(LutS, val8);
          val_h = __hmul(val_h, inv_s);
          int8_t val_i8 = cvt_f16_to_s8_sat(val_h);
          packed_int8 |= ((uint32_t)(uint8_t)val_i8) << (b * 8);
        }

        *reinterpret_cast<uint32_t*>(B_next + (size_t)col * kKChunk + k_off) = packed_int8;
      }
    }

    __syncthreads();
    stage ^= 1;
  }

  int32_t* C_base = C_sh + (size_t)(warp_n * WN) * BM + (warp_m * WM);
  wmma::store_matrix_sync(C_base, acc_frag, BM, wmma::mem_col_major);
  __syncthreads();

  int total = BM * BN;
  for (int idx = tid; idx < total; idx += (int)blockDim.x) {
    int col = idx / BM;
    int row = idx - col * BM;
    int gm = block_m0 + row;
    int gn = block_n0 + col;
    if (gm < M && gn < N) {
      float s = scales_sh[col] * global_scale;
      float v = (float)C_sh[(size_t)col * BM + row] * s;
      D_col[(size_t)gn * M + gm] = __float2half_rn(v);
    }
  }
}

static size_t smem_bytes_v2(int kChunk) {
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

static size_t smem_bytes_v2_i8lut(int kChunk) {
  // Adds per-column [256] int8 LUT (BN*256 bytes).
  constexpr int BN = 64;
  return smem_bytes_v2(kChunk) + (size_t)BN * 256ull;
}

static size_t smem_bytes_v4(int kChunk, size_t aTypeBytes) {
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

} // namespace

namespace fp8imma {

void init_fp8_e4m3_lut() {
  static bool inited = false;
  if (inited) return;
  uint16_t host_lut[256];
  for (int i = 0; i < 256; ++i) {
    host_lut[i] = f32_to_f16_bits(fp8_e4m3_to_f32((uint8_t)i));
  }
  // Best-effort: if this fails, downstream launch wrappers return CUDA error codes.
  cudaMemcpyToSymbol(fp8imma_k_fp8_e4m3_to_f16_bits, host_lut, sizeof(host_lut));
  inited = true;
}

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
        (int)smem_bytes_v2(32));
    imma_gemm_fp8_int8_colscale_fp16_colmajor_kernel_v2<32><<<grid, block, smem_bytes_v2(32), stream>>>(
        A_row_i8, B_col_fp8, col_scales_u16, D_col_f16, M, N, K, global_scale);
    return cudaGetLastError();
  }
  if (kChunk == 64) {
    (void)cudaFuncSetAttribute(
        imma_gemm_fp8_int8_colscale_fp16_colmajor_kernel_v2<64>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_bytes_v2(64));
    imma_gemm_fp8_int8_colscale_fp16_colmajor_kernel_v2<64><<<grid, block, smem_bytes_v2(64), stream>>>(
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
        (int)smem_bytes_v2_i8lut(32));
    imma_gemm_fp8_int8_colscale_fp16_colmajor_kernel_v2_i8lut<32><<<grid, block, smem_bytes_v2_i8lut(32), stream>>>(
        A_row_i8, B_col_fp8, col_scales_u16, D_col_f16, M, N, K, global_scale);
    return cudaGetLastError();
  }
  if (kChunk == 64) {
    (void)cudaFuncSetAttribute(
        imma_gemm_fp8_int8_colscale_fp16_colmajor_kernel_v2_i8lut<64>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_bytes_v2_i8lut(64));
    imma_gemm_fp8_int8_colscale_fp16_colmajor_kernel_v2_i8lut<64><<<grid, block, smem_bytes_v2_i8lut(64), stream>>>(
        A_row_i8, B_col_fp8, col_scales_u16, D_col_f16, M, N, K, global_scale);
    return cudaGetLastError();
  }
  return cudaErrorInvalidValue;
}

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
    cudaStream_t stream) {

  init_fp8_e4m3_lut();

  dim3 block(512);
  dim3 grid((N + 63) / 64, (M + 63) / 64);
  half a_inv_scale_h = __float2half(a_inv_scale);

  if (kChunk == 32) {
    (void)cudaFuncSetAttribute(
        imma_gemm_fp8_actquant_int8_colscale_fp16_colmajor_kernel_v3<32, half>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_bytes_v2(32));
    imma_gemm_fp8_actquant_int8_colscale_fp16_colmajor_kernel_v3<32, half><<<grid, block, smem_bytes_v2(32), stream>>>(
        A_row_f16, B_col_fp8, col_scales_u16, D_col_f16, M, N, K, global_scale, a_inv_scale_h);
    return cudaGetLastError();
  }
  if (kChunk == 64) {
    (void)cudaFuncSetAttribute(
        imma_gemm_fp8_actquant_int8_colscale_fp16_colmajor_kernel_v3<64, half>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_bytes_v2(64));
    imma_gemm_fp8_actquant_int8_colscale_fp16_colmajor_kernel_v3<64, half><<<grid, block, smem_bytes_v2(64), stream>>>(
        A_row_f16, B_col_fp8, col_scales_u16, D_col_f16, M, N, K, global_scale, a_inv_scale_h);
    return cudaGetLastError();
  }
  return cudaErrorInvalidValue;
}

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
    cudaStream_t stream) {

  init_fp8_e4m3_lut();

  dim3 block(512);
  dim3 grid((N + 63) / 64, (M + 63) / 64);
  half a_inv_scale_h = __float2half(a_inv_scale);

  if (kChunk == 32) {
    (void)cudaFuncSetAttribute(
        imma_gemm_fp8_actquant_int8_colscale_fp16_colmajor_kernel_v3<32, __nv_bfloat16>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_bytes_v2(32));
    imma_gemm_fp8_actquant_int8_colscale_fp16_colmajor_kernel_v3<32, __nv_bfloat16><<<grid, block, smem_bytes_v2(32), stream>>>(
        A_row_bf16, B_col_fp8, col_scales_u16, D_col_f16, M, N, K, global_scale, a_inv_scale_h);
    return cudaGetLastError();
  }
  if (kChunk == 64) {
    (void)cudaFuncSetAttribute(
        imma_gemm_fp8_actquant_int8_colscale_fp16_colmajor_kernel_v3<64, __nv_bfloat16>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_bytes_v2(64));
    imma_gemm_fp8_actquant_int8_colscale_fp16_colmajor_kernel_v3<64, __nv_bfloat16><<<grid, block, smem_bytes_v2(64), stream>>>(
        A_row_bf16, B_col_fp8, col_scales_u16, D_col_f16, M, N, K, global_scale, a_inv_scale_h);
    return cudaGetLastError();
  }
  return cudaErrorInvalidValue;
}

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
        (int)smem_bytes_v4(32, sizeof(half)));
    imma_gemm_fp8_actquant_cpasync_int8_colscale_fp16_colmajor_kernel_v4<32, half>
        <<<grid, block, smem_bytes_v4(32, sizeof(half)), stream>>>(
            A_row_f16, B_col_fp8, col_scales_u16, D_col_f16, M, N, K, global_scale, a_inv_scale_h);
    return cudaGetLastError();
  }
  if (kChunk == 64) {
    (void)cudaFuncSetAttribute(
        imma_gemm_fp8_actquant_cpasync_int8_colscale_fp16_colmajor_kernel_v4<64, half>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_bytes_v4(64, sizeof(half)));
    imma_gemm_fp8_actquant_cpasync_int8_colscale_fp16_colmajor_kernel_v4<64, half>
        <<<grid, block, smem_bytes_v4(64, sizeof(half)), stream>>>(
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
        (int)smem_bytes_v4(32, sizeof(__nv_bfloat16)));
    imma_gemm_fp8_actquant_cpasync_int8_colscale_fp16_colmajor_kernel_v4<32, __nv_bfloat16>
        <<<grid, block, smem_bytes_v4(32, sizeof(__nv_bfloat16)), stream>>>(
            A_row_bf16, B_col_fp8, col_scales_u16, D_col_f16, M, N, K, global_scale, a_inv_scale_h);
    return cudaGetLastError();
  }
  if (kChunk == 64) {
    (void)cudaFuncSetAttribute(
        imma_gemm_fp8_actquant_cpasync_int8_colscale_fp16_colmajor_kernel_v4<64, __nv_bfloat16>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_bytes_v4(64, sizeof(__nv_bfloat16)));
    imma_gemm_fp8_actquant_cpasync_int8_colscale_fp16_colmajor_kernel_v4<64, __nv_bfloat16>
        <<<grid, block, smem_bytes_v4(64, sizeof(__nv_bfloat16)), stream>>>(
            A_row_bf16, B_col_fp8, col_scales_u16, D_col_f16, M, N, K, global_scale, a_inv_scale_h);
    return cudaGetLastError();
  }
  return cudaErrorInvalidValue;
}

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
    cudaStream_t stream) {

  init_fp8_e4m3_lut();
  dim3 block(512);
  dim3 grid((N + 63) / 64, (M + 63) / 64);
  half a_inv_scale_h = __float2half(a_inv_scale);

  if (kChunk == 32) {
    (void)cudaFuncSetAttribute(
        imma_gemm_fp8_actquant_cpasync_int8_colscale_fp16_colmajor_kernel_v4_texscale<32, half>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_bytes_v4(32, sizeof(half)));
    imma_gemm_fp8_actquant_cpasync_int8_colscale_fp16_colmajor_kernel_v4_texscale<32, half>
        <<<grid, block, smem_bytes_v4(32, sizeof(half)), stream>>>(
            A_row_f16, B_col_fp8, scale_tex_u16, D_col_f16, M, N, K, global_scale, a_inv_scale_h);
    return cudaGetLastError();
  }
  if (kChunk == 64) {
    (void)cudaFuncSetAttribute(
        imma_gemm_fp8_actquant_cpasync_int8_colscale_fp16_colmajor_kernel_v4_texscale<64, half>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_bytes_v4(64, sizeof(half)));
    imma_gemm_fp8_actquant_cpasync_int8_colscale_fp16_colmajor_kernel_v4_texscale<64, half>
        <<<grid, block, smem_bytes_v4(64, sizeof(half)), stream>>>(
            A_row_f16, B_col_fp8, scale_tex_u16, D_col_f16, M, N, K, global_scale, a_inv_scale_h);
    return cudaGetLastError();
  }
  return cudaErrorInvalidValue;
}

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
    cudaStream_t stream) {

  init_fp8_e4m3_lut();
  dim3 block(512);
  dim3 grid((N + 63) / 64, (M + 63) / 64);
  half a_inv_scale_h = __float2half(a_inv_scale);

  if (kChunk == 32) {
    (void)cudaFuncSetAttribute(
        imma_gemm_fp8_actquant_cpasync_int8_colscale_fp16_colmajor_kernel_v4_texscale<32, __nv_bfloat16>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_bytes_v4(32, sizeof(__nv_bfloat16)));
    imma_gemm_fp8_actquant_cpasync_int8_colscale_fp16_colmajor_kernel_v4_texscale<32, __nv_bfloat16>
        <<<grid, block, smem_bytes_v4(32, sizeof(__nv_bfloat16)), stream>>>(
            A_row_bf16, B_col_fp8, scale_tex_u16, D_col_f16, M, N, K, global_scale, a_inv_scale_h);
    return cudaGetLastError();
  }
  if (kChunk == 64) {
    (void)cudaFuncSetAttribute(
        imma_gemm_fp8_actquant_cpasync_int8_colscale_fp16_colmajor_kernel_v4_texscale<64, __nv_bfloat16>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_bytes_v4(64, sizeof(__nv_bfloat16)));
    imma_gemm_fp8_actquant_cpasync_int8_colscale_fp16_colmajor_kernel_v4_texscale<64, __nv_bfloat16>
        <<<grid, block, smem_bytes_v4(64, sizeof(__nv_bfloat16)), stream>>>(
            A_row_bf16, B_col_fp8, scale_tex_u16, D_col_f16, M, N, K, global_scale, a_inv_scale_h);
    return cudaGetLastError();
  }
  return cudaErrorInvalidValue;
}

} // namespace fp8imma

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

} // extern "C"
