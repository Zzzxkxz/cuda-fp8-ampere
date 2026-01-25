#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cstdint>
#include <mutex>

#include <mma.h>

namespace wmma = nvcuda::wmma;

#define CUDA_CHECK(x) TORCH_CHECK((x) == cudaSuccess, "CUDA error: ", cudaGetErrorString((x)))

// ---------------- FP8 E4M3 decode ----------------
// We use a LUT mapping byte -> FP16 bits (uint16) stored in constant memory.
__constant__ uint16_t k_fp8_e4m3_to_f16_bits[256];

static __host__ __device__ inline float fp8_e4m3_to_f32_host(uint8_t v) {
  // Canonical-ish E4M3 decode:
  // sign:1, exp:4 (bias 7), mant:3.
  // This is not guaranteed to match NVIDIA's e4m3fn exactly, but is consistent and usable
  // for experimentation and weight storage.
  uint32_t sign = (v >> 7) & 1u;
  uint32_t exp = (v >> 3) & 0xFu;
  uint32_t mant = v & 0x7u;

  if (exp == 0) {
    if (mant == 0) return sign ? -0.0f : 0.0f;
    // subnormal: (-1)^sign * 2^(1-bias) * (mant / 2^3)
    float m = (float)mant / 8.0f;
    float val = ldexpf(m, 1 - 7);
    return sign ? -val : val;
  }
  if (exp == 0xF) {
    // inf/nan
    if (mant == 0) return sign ? -INFINITY : INFINITY;
    return NAN;
  }

  // normal: (-1)^sign * 2^(exp-bias) * (1 + mant/2^3)
  float m = 1.0f + ((float)mant / 8.0f);
  float val = ldexpf(m, (int)exp - 7);
  return sign ? -val : val;
}

static inline uint16_t f32_to_f16_bits_host(float f) {
  // Use CUDA half conversion on host via __half_raw-like bitcast is annoying; do a simple
  // IEEE fp16 conversion. This is fine for LUT generation.
  // Reference conversion: round-to-nearest-even.
  union { float f; uint32_t u; } v;
  v.f = f;
  uint32_t x = v.u;
  uint32_t sign = (x >> 31) & 1u;
  int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127;
  uint32_t mant = x & 0x7FFFFFu;

  uint16_t out_sign = (uint16_t)(sign << 15);

  if (((x >> 23) & 0xFFu) == 0xFFu) {
    // inf/nan
    uint16_t out_exp = 0x1Fu << 10;
    uint16_t out_m = (mant ? 0x200u : 0u);
    return (uint16_t)(out_sign | out_exp | out_m);
  }

  // Handle zero/denorm in fp32
  if (((x >> 23) & 0xFFu) == 0) {
    return out_sign;
  }

  // fp16 exponent = exp + 15
  int32_t e16 = exp + 15;
  if (e16 >= 31) {
    // overflow -> inf
    return (uint16_t)(out_sign | (0x1Fu << 10));
  }
  if (e16 <= 0) {
    // underflow -> subnormal or zero
    // shift mantissa (with implicit 1)
    uint32_t m = mant | (1u << 23);
    int32_t shift = (1 - e16);
    if (shift > 24) return out_sign;
    // m is 24-bit (including implicit 1 at bit 23); we need 10-bit mantissa
    // We want: (m >> (13 + shift)) with rounding
    uint32_t rshift = 13 + (uint32_t)shift;
    uint32_t m_shifted = m >> rshift;
    uint32_t rem = m & ((1u << rshift) - 1u);
    uint32_t halfway = 1u << (rshift - 1u);
    if (rem > halfway || (rem == halfway && (m_shifted & 1u))) m_shifted++;
    return (uint16_t)(out_sign | (uint16_t)m_shifted);
  }

  // Normal fp16
  // mant16 = mant >> 13 with rounding
  uint32_t mant16 = mant >> 13;
  uint32_t rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (mant16 & 1u))) {
    mant16++;
    if (mant16 == 0x400u) {
      // mantissa overflow
      mant16 = 0;
      e16++;
      if (e16 >= 31) return (uint16_t)(out_sign | (0x1Fu << 10));
    }
  }

  return (uint16_t)(out_sign | ((uint16_t)e16 << 10) | (uint16_t)mant16);
}

static void ensure_lut_initialized() {
  static std::once_flag once;
  std::call_once(once, [] {
    uint16_t host_lut[256];
    for (int i = 0; i < 256; ++i) {
      float f = fp8_e4m3_to_f32_host((uint8_t)i);
      host_lut[i] = f32_to_f16_bits_host(f);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(k_fp8_e4m3_to_f16_bits, host_lut, sizeof(host_lut)));
  });
}

__device__ __forceinline__ __half fp8_lut_decode_const(uint8_t v) {
  union { uint16_t u; __half h; } cvt;
  cvt.u = k_fp8_e4m3_to_f16_bits[v];
  return cvt.h;
}

// ---------------- Fused decode + WMMA GEMM ----------------
// A: [M,K] FP16 row-major
// B: [N,K] uint8 (represents KxN col-major with ld=K)
// C: [M,N] FP32 row-major
// Each block computes 32x32 tile using 4 warps; BK=16.
__global__ void wmma_fp16a_fp8e4m3b_gemm_kernel_tiled_vec4(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B8_colmajor,
    float* __restrict__ C,
    int M, int N, int K,
    float scale_b) {
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 16;

  int block_col = (int)blockIdx.x;
  int block_row = (int)blockIdx.y;

  int warp_id = (int)threadIdx.x >> 5;
  int warp_row = (warp_id >> 1) & 1;
  int warp_col = warp_id & 1;

  __shared__ __half As[BM * BK];
  __shared__ __half Bs[BK * BN];
  __shared__ uint16_t LutS[256];

  // Copy LUT once per block.
  for (int i = (int)threadIdx.x; i < 256; i += (int)blockDim.x) {
    LutS[i] = k_fp8_e4m3_to_f16_bits[i];
  }

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  int row0 = block_row * BM;
  int col0 = block_col * BN;

  for (int k0 = 0; k0 < K; k0 += BK) {
    // Load A tile to shared.
    for (int idx = (int)threadIdx.x; idx < BM * BK; idx += (int)blockDim.x) {
      int r = idx / BK;
      int c = idx - r * BK;
      int ar = row0 + r;
      int ac = k0 + c;
      As[idx] = (ar < M && ac < K) ? A[(size_t)ar * K + ac] : __float2half(0.0f);
    }

    // Decode B tile into shared (col-major), vector4 decode.
    {
      int t = (int)threadIdx.x;
      int items = BN * (BK / 4); // 32 * 4 = 128
      if (t < items) {
        int c = t / (BK / 4);
        int pack = t - c * (BK / 4);
        int r0 = pack * 4;
        const uint8_t* p = B8_colmajor + (size_t)(col0 + c) * K + (k0 + r0);
        uint32_t w = *reinterpret_cast<const uint32_t*>(p);

        uint8_t v0 = (uint8_t)(w & 0xFFu);
        uint8_t v1 = (uint8_t)((w >> 8) & 0xFFu);
        uint8_t v2 = (uint8_t)((w >> 16) & 0xFFu);
        uint8_t v3 = (uint8_t)((w >> 24) & 0xFFu);

        union { uint16_t u; __half h; } cvt;

        cvt.u = LutS[v0]; Bs[c * BK + (r0 + 0)] = __float2half(__half2float(cvt.h) * scale_b);
        cvt.u = LutS[v1]; Bs[c * BK + (r0 + 1)] = __float2half(__half2float(cvt.h) * scale_b);
        cvt.u = LutS[v2]; Bs[c * BK + (r0 + 2)] = __float2half(__half2float(cvt.h) * scale_b);
        cvt.u = LutS[v3]; Bs[c * BK + (r0 + 3)] = __float2half(__half2float(cvt.h) * scale_b);
      }
    }

    __syncthreads();

    const __half* A_tile = As + (warp_row * 16) * BK;
    const __half* B_tile = Bs + (warp_col * 16) * BK;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    wmma::load_matrix_sync(a_frag, A_tile, BK);
    wmma::load_matrix_sync(b_frag, B_tile, BK);
    wmma::mma_sync(acc, a_frag, b_frag, acc);

    __syncthreads();
  }

  int c_row = row0 + warp_row * 16;
  int c_col = col0 + warp_col * 16;
  if (c_row < M && c_col < N) {
    wmma::store_matrix_sync(C + (size_t)c_row * N + c_col, acc, N, wmma::mem_row_major);
  }
}

torch::Tensor fp8_linear_forward(torch::Tensor A_fp16, torch::Tensor B_fp8_u8, double scale_b) {
  TORCH_CHECK(A_fp16.is_cuda(), "A must be CUDA");
  TORCH_CHECK(B_fp8_u8.is_cuda(), "B must be CUDA");
  TORCH_CHECK(A_fp16.scalar_type() == at::kHalf, "A must be FP16");
  TORCH_CHECK(B_fp8_u8.scalar_type() == at::kByte, "B must be uint8");
  TORCH_CHECK(A_fp16.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B_fp8_u8.is_contiguous(), "B must be contiguous");
  TORCH_CHECK(A_fp16.dim() == 2, "A must be 2D");
  TORCH_CHECK(B_fp8_u8.dim() == 2, "B must be 2D");

  int64_t M = A_fp16.size(0);
  int64_t K = A_fp16.size(1);
  int64_t N = B_fp8_u8.size(0);
  TORCH_CHECK(B_fp8_u8.size(1) == K, "B must have shape [N,K] (col-major KxN with ld=K)");

  TORCH_CHECK((M % 16) == 0 && (N % 16) == 0 && (K % 16) == 0,
              "M,N,K must be multiples of 16 for WMMA");

  ensure_lut_initialized();

  auto C = torch::empty({M, N}, torch::TensorOptions().device(A_fp16.device()).dtype(torch::kFloat32));

  dim3 grid((unsigned)(N / 32), (unsigned)(M / 32), 1);
  dim3 block(128, 1, 1);

  cudaStream_t stream = at::cuda::getDefaultCUDAStream();
  wmma_fp16a_fp8e4m3b_gemm_kernel_tiled_vec4<<<grid, block, 0, stream>>>(
      (const __half*)A_fp16.data_ptr<at::Half>(),
      (const uint8_t*)B_fp8_u8.data_ptr<uint8_t>(),
      (float*)C.data_ptr<float>(),
      (int)M, (int)N, (int)K,
      (float)scale_b);
  CUDA_CHECK(cudaGetLastError());

  return C;
}
