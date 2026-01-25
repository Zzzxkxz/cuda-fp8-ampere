#include <fp8imma/imma_fp8.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void die(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e));
    std::exit(1);
  }
}

static int8_t clamp_i8(int v) {
  if (v < -128) return -128;
  if (v > 127) return 127;
  return (int8_t)v;
}

static void fill_i8(std::vector<int8_t>& v) {
  for (size_t i = 0; i < v.size(); ++i) v[i] = (int8_t)((int)(i * 1315423911u + 12345u) % 17 - 8);
}

static void fill_u8(std::vector<uint8_t>& v) {
  for (size_t i = 0; i < v.size(); ++i) v[i] = (uint8_t)((i * 2654435761u) & 0xFFu);
}

static void fill_f16(std::vector<half>& v) {
  for (size_t i = 0; i < v.size(); ++i) {
    float x = std::sin((float)i * 0.01f) * 0.75f;
    v[i] = __float2half_rn(x);
  }
}

static void fill_bf16(std::vector<__nv_bfloat16>& v) {
  for (size_t i = 0; i < v.size(); ++i) {
    float x = std::cos((float)i * 0.01f) * 0.75f;
    v[i] = __float2bfloat16_rn(x);
  }
}

static void fill_scale_u16(std::vector<uint16_t>& v) {
  for (size_t i = 0; i < v.size(); ++i) {
    // Store half bits as uint16_t (as expected by kernels)
    float s = 0.5f + 0.01f * (float)(i % 31); // avoid tiny scales
    half h = __float2half_rn(s);
    v[i] = (uint16_t)__half_as_ushort(h);
  }
}

static void check_finite_output(const std::vector<half>& out, const char* tag) {
  for (size_t i = 0; i < out.size(); ++i) {
    float x = __half2float(out[i]);
    if (!std::isfinite(x)) {
      fprintf(stderr, "Non-finite output in %s at idx=%zu: %f\n", tag, i, x);
      std::exit(2);
    }
  }
}

static cudaTextureObject_t make_u16_texture(const uint16_t* d_u16, size_t count) {
  cudaResourceDesc res{};
  res.resType = cudaResourceTypeLinear;
  res.res.linear.devPtr = (void*)d_u16;
  res.res.linear.sizeInBytes = count * sizeof(uint16_t);
  res.res.linear.desc = cudaCreateChannelDesc<unsigned short>();

  cudaTextureDesc tex{};
  tex.addressMode[0] = cudaAddressModeClamp;
  tex.filterMode = cudaFilterModePoint;
  tex.readMode = cudaReadModeElementType;
  tex.normalizedCoords = 0;

  cudaTextureObject_t obj = 0;
  die(cudaCreateTextureObject(&obj, &res, &tex, nullptr), "cudaCreateTextureObject");
  return obj;
}

int main() {
  int device_count = 0;
  cudaError_t e = cudaGetDeviceCount(&device_count);
  if (e != cudaSuccess || device_count <= 0) {
    fprintf(stderr, "SKIP: no CUDA device found.\n");
    return 0;
  }

  die(cudaSetDevice(0), "cudaSetDevice");
  cudaStream_t stream = nullptr; // default stream

  constexpr int M = 64;
  constexpr int N = 64;
  constexpr int K = 64;
  constexpr float global_scale = 1.0f;
  constexpr float a_inv_scale = 1.0f;

  // Host buffers
  std::vector<int8_t> h_A_i8((size_t)M * K);
  std::vector<uint8_t> h_B_fp8((size_t)N * K);
  std::vector<uint16_t> h_scales_u16((size_t)N);
  std::vector<half> h_A_f16((size_t)M * K);
  std::vector<__nv_bfloat16> h_A_bf16((size_t)M * K);

  fill_i8(h_A_i8);
  fill_u8(h_B_fp8);
  fill_scale_u16(h_scales_u16);
  fill_f16(h_A_f16);
  fill_bf16(h_A_bf16);

  // Device buffers
  int8_t* d_A_i8 = nullptr;
  uint8_t* d_B_fp8 = nullptr;
  uint16_t* d_scales_u16 = nullptr;
  half* d_A_f16 = nullptr;
  __nv_bfloat16* d_A_bf16 = nullptr;
  half* d_D = nullptr;

  die(cudaMalloc(&d_A_i8, h_A_i8.size()), "cudaMalloc d_A_i8");
  die(cudaMalloc(&d_B_fp8, h_B_fp8.size()), "cudaMalloc d_B_fp8");
  die(cudaMalloc(&d_scales_u16, h_scales_u16.size() * sizeof(uint16_t)), "cudaMalloc d_scales_u16");
  die(cudaMalloc(&d_A_f16, h_A_f16.size() * sizeof(half)), "cudaMalloc d_A_f16");
  die(cudaMalloc(&d_A_bf16, h_A_bf16.size() * sizeof(__nv_bfloat16)), "cudaMalloc d_A_bf16");
  die(cudaMalloc(&d_D, (size_t)N * M * sizeof(half)), "cudaMalloc d_D");

  die(cudaMemcpy(d_A_i8, h_A_i8.data(), h_A_i8.size(), cudaMemcpyHostToDevice), "cudaMemcpy A_i8");
  die(cudaMemcpy(d_B_fp8, h_B_fp8.data(), h_B_fp8.size(), cudaMemcpyHostToDevice), "cudaMemcpy B_fp8");
  die(cudaMemcpy(d_scales_u16, h_scales_u16.data(), h_scales_u16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "cudaMemcpy scales_u16");
  die(cudaMemcpy(d_A_f16, h_A_f16.data(), h_A_f16.size() * sizeof(half), cudaMemcpyHostToDevice), "cudaMemcpy A_f16");
  die(cudaMemcpy(d_A_bf16, h_A_bf16.data(), h_A_bf16.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "cudaMemcpy A_bf16");

  // Texture object for texscale variants
  cudaTextureObject_t tex = make_u16_texture(d_scales_u16, (size_t)N);

  auto run_and_check = [&](const char* tag, auto&& fn) {
    die(cudaMemset(d_D, 0, (size_t)N * M * sizeof(half)), "cudaMemset D");
    cudaError_t err = fn();
    if (err != cudaSuccess) {
      fprintf(stderr, "%s returned %s\n", tag, cudaGetErrorString(err));
      return 1;
    }
    die(cudaGetLastError(), "cudaGetLastError");
    die(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::vector<half> h_D((size_t)N * M);
    die(cudaMemcpy(h_D.data(), d_D, h_D.size() * sizeof(half), cudaMemcpyDeviceToHost), "cudaMemcpy D->H");
    check_finite_output(h_D, tag);
    return 0;
  };

  // ---- v2 / v2_i8lut ----
  for (int kChunk : {32, 64}) {
    int rc = run_and_check("v2", [&]() {
      return fp8imma::launch_imma_fp8_jit_v2(kChunk, d_A_i8, d_B_fp8, d_scales_u16, d_D, M, N, K, global_scale, stream);
    });
    if (rc) return rc;

    rc = run_and_check("v2_i8lut", [&]() {
      return fp8imma::launch_imma_fp8_jit_v2_i8lut(kChunk, d_A_i8, d_B_fp8, d_scales_u16, d_D, M, N, K, global_scale, stream);
    });
    if (rc) return rc;

    // Also hit C ABI wrappers (same paths)
    rc = run_and_check("v2_cabi", [&]() {
      int ierr = fp8imma_launch_imma_fp8_jit_v2(kChunk, d_A_i8, d_B_fp8, d_scales_u16, d_D, M, N, K, global_scale, stream);
      return (ierr == 0) ? cudaSuccess : cudaErrorUnknown;
    });
    if (rc) return rc;

    rc = run_and_check("v2_i8lut_cabi", [&]() {
      int ierr = fp8imma_launch_imma_fp8_jit_v2_i8lut(kChunk, d_A_i8, d_B_fp8, d_scales_u16, d_D, M, N, K, global_scale, stream);
      return (ierr == 0) ? cudaSuccess : cudaErrorUnknown;
    });
    if (rc) return rc;
  }

  // ---- v3 ----
  for (int kChunk : {32, 64}) {
    int rc = run_and_check("v3_f16", [&]() {
      return fp8imma::launch_imma_fp8_actquant_v3_f16(kChunk, d_A_f16, d_B_fp8, d_scales_u16, d_D, M, N, K, global_scale, a_inv_scale, stream);
    });
    if (rc) return rc;

    rc = run_and_check("v3_bf16", [&]() {
      return fp8imma::launch_imma_fp8_actquant_v3_bf16(kChunk, d_A_bf16, d_B_fp8, d_scales_u16, d_D, M, N, K, global_scale, a_inv_scale, stream);
    });
    if (rc) return rc;

    rc = run_and_check("v3_f16_cabi", [&]() {
      int ierr = fp8imma_launch_imma_fp8_actquant_v3_f16(kChunk, d_A_f16, d_B_fp8, d_scales_u16, d_D, M, N, K, global_scale, a_inv_scale, stream);
      return (ierr == 0) ? cudaSuccess : cudaErrorUnknown;
    });
    if (rc) return rc;

    rc = run_and_check("v3_bf16_cabi", [&]() {
      int ierr = fp8imma_launch_imma_fp8_actquant_v3_bf16(kChunk, d_A_bf16, d_B_fp8, d_scales_u16, d_D, M, N, K, global_scale, a_inv_scale, stream);
      return (ierr == 0) ? cudaSuccess : cudaErrorUnknown;
    });
    if (rc) return rc;
  }

  // ---- v4 ----
  for (int kChunk : {32, 64}) {
    int rc = run_and_check("v4_f16", [&]() {
      return fp8imma::launch_imma_fp8_actquant_v4_f16(kChunk, d_A_f16, d_B_fp8, d_scales_u16, d_D, M, N, K, global_scale, a_inv_scale, stream);
    });
    if (rc) return rc;

    rc = run_and_check("v4_bf16", [&]() {
      return fp8imma::launch_imma_fp8_actquant_v4_bf16(kChunk, d_A_bf16, d_B_fp8, d_scales_u16, d_D, M, N, K, global_scale, a_inv_scale, stream);
    });
    if (rc) return rc;

    // texscale
    rc = run_and_check("v4_texscale_f16", [&]() {
      return fp8imma::launch_imma_fp8_actquant_v4_texscale_f16(kChunk, d_A_f16, d_B_fp8, tex, d_D, M, N, K, global_scale, a_inv_scale, stream);
    });
    if (rc) return rc;

    rc = run_and_check("v4_texscale_bf16", [&]() {
      return fp8imma::launch_imma_fp8_actquant_v4_texscale_bf16(kChunk, d_A_bf16, d_B_fp8, tex, d_D, M, N, K, global_scale, a_inv_scale, stream);
    });
    if (rc) return rc;

    // texscale C ABI wrappers
    rc = run_and_check("v4_texscale_f16_cabi", [&]() {
      int ierr = fp8imma_launch_imma_fp8_actquant_v4_texscale_f16(kChunk, d_A_f16, d_B_fp8, (void*)tex, d_D, M, N, K, global_scale, a_inv_scale, stream);
      return (ierr == 0) ? cudaSuccess : cudaErrorUnknown;
    });
    if (rc) return rc;

    rc = run_and_check("v4_texscale_bf16_cabi", [&]() {
      int ierr = fp8imma_launch_imma_fp8_actquant_v4_texscale_bf16(kChunk, d_A_bf16, d_B_fp8, (void*)tex, d_D, M, N, K, global_scale, a_inv_scale, stream);
      return (ierr == 0) ? cudaSuccess : cudaErrorUnknown;
    });
    if (rc) return rc;

    // v4 C ABI wrappers
    rc = run_and_check("v4_f16_cabi", [&]() {
      int ierr = fp8imma_launch_imma_fp8_actquant_v4_f16(kChunk, d_A_f16, d_B_fp8, d_scales_u16, d_D, M, N, K, global_scale, a_inv_scale, stream);
      return (ierr == 0) ? cudaSuccess : cudaErrorUnknown;
    });
    if (rc) return rc;

    rc = run_and_check("v4_bf16_cabi", [&]() {
      int ierr = fp8imma_launch_imma_fp8_actquant_v4_bf16(kChunk, d_A_bf16, d_B_fp8, d_scales_u16, d_D, M, N, K, global_scale, a_inv_scale, stream);
      return (ierr == 0) ? cudaSuccess : cudaErrorUnknown;
    });
    if (rc) return rc;
  }

  die(cudaDestroyTextureObject(tex), "cudaDestroyTextureObject");

  cudaFree(d_A_i8);
  cudaFree(d_B_fp8);
  cudaFree(d_scales_u16);
  cudaFree(d_A_f16);
  cudaFree(d_A_bf16);
  cudaFree(d_D);

  fprintf(stdout, "fp8imma_smoke: OK\n");
  return 0;
}
