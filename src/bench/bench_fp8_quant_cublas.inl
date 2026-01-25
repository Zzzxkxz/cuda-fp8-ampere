// ------------------------ BENCH: FP8 quantization + cuBLAS ------------------------
__global__ void quantize_fp16_to_fp8e4m3_kernel(const half* __restrict__ in, uint8_t* __restrict__ out8, int n, float inv_scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float x = __half2float(in[idx]) * inv_scale;
  // Device-side encode using a small approximation: reuse host encoder logic by mirroring key steps.
  // For perf experiments, LUT decode cost matters more than perfect encode.
  // We'll clamp to representable range and do a rough normal encode.
  int sign = x < 0.0f;
  float ax = fabsf(x);
  if (!isfinite(ax)) {
    out8[idx] = (uint8_t)((sign << 7) | (14u << 3) | 7u);
    return;
  }
  if (ax == 0.0f) {
    out8[idx] = (uint8_t)(sign << 7);
    return;
  }
  ax = fminf(ax, 240.0f);
  int e = (int)floorf(log2f(ax));
  int exp = e + 7;
  if (exp <= 0) {
    float scaled = ldexpf(ax, (7 - 1) + 3);
    int mant = (int)nearbyintf(scaled);
    mant = max(0, min(7, mant));
    out8[idx] = (uint8_t)((sign << 7) | mant);
    return;
  }
  exp = max(1, min(14, exp));
  float frac = ax / ldexpf(1.0f, e);
  int mant = (int)nearbyintf((frac - 1.0f) * 8.0f);
  if (mant >= 8) {
    mant = 0;
    exp = min(14, exp + 1);
  }
  mant = max(0, min(7, mant));
  out8[idx] = (uint8_t)((sign << 7) | ((uint32_t)exp << 3) | (uint32_t)mant);
}

__global__ void dequantize_fp8e4m3_to_fp16_kernel(const uint8_t* __restrict__ in8, half* __restrict__ out16, int n, float scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  half h = fp8_lut_decode_const(in8[idx]);
  float x = __half2float(h) * scale;
  out16[idx] = __float2half(x);
}

__global__ void dequantize_fp8e4m3_to_fp16_kernel_vec4(
    const uint8_t* __restrict__ in8,
    half* __restrict__ out16,
    int n,
    float scale) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int base = tid * 4;
  if (base >= n) return;

  // n is expected to be a multiple of 4 in our benches (K*N for 1024 multiples).
  // Handle tail safely anyway.
  if (base + 3 < n) {
    uint32_t packed = *reinterpret_cast<const uint32_t*>(in8 + base);
    uint8_t b0 = (uint8_t)(packed & 0xFFu);
    uint8_t b1 = (uint8_t)((packed >> 8) & 0xFFu);
    uint8_t b2 = (uint8_t)((packed >> 16) & 0xFFu);
    uint8_t b3 = (uint8_t)((packed >> 24) & 0xFFu);
    float s = scale;
    out16[base + 0] = __float2half(__half2float(fp8_lut_decode_const(b0)) * s);
    out16[base + 1] = __float2half(__half2float(fp8_lut_decode_const(b1)) * s);
    out16[base + 2] = __float2half(__half2float(fp8_lut_decode_const(b2)) * s);
    out16[base + 3] = __float2half(__half2float(fp8_lut_decode_const(b3)) * s);
  } else {
    for (int i = 0; i < 4; ++i) {
      int idx = base + i;
      if (idx < n) {
        out16[idx] = __float2half(__half2float(fp8_lut_decode_const(in8[idx])) * scale);
      }
    }
  }
}

__global__ void dequantize_fp8e4m3_to_fp16_blockscale_f16_kernel(
    const uint8_t* __restrict__ in8_col,
    half* __restrict__ out16_col,
    int K,
    int N,
    const half* __restrict__ scales16,
    int block_k) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int n = K * N;
  if (idx >= n) return;
  int col = idx / K;
  int row = idx - col * K;
  int kb = row / block_k;
  int scale_idx = col * (K / block_k) + kb;
  half h = fp8_lut_decode_const(in8_col[idx]);
  float s = __half2float(scales16[scale_idx]);
  out16_col[idx] = __float2half(__half2float(h) * s);
}

__global__ void dequantize_fp8e4m3_to_fp16_blockscale_f16_vec4_kernel(
    const uint8_t* __restrict__ in8_col,
    half* __restrict__ out16_col,
    int K,
    int N,
    const half* __restrict__ scales16,
    int block_k) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int base = tid * 4;
  int n = K * N;
  if (base >= n) return;

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    int idx = base + i;
    if (idx >= n) break;
    int col = idx / K;
    int row = idx - col * K;
    int kb = row / block_k;
    int scale_idx = col * (K / block_k) + kb;
    half h = fp8_lut_decode_const(in8_col[idx]);
    float s = __half2float(scales16[scale_idx]);
    out16_col[idx] = __float2half(__half2float(h) * s);
  }
}

__global__ void dequantize_fp8e4m3_to_fp16_blockscale_fp8_kernel(
    const uint8_t* __restrict__ in8_col,
    half* __restrict__ out16_col,
    int K,
    int N,
    const uint8_t* __restrict__ scales8,
    int block_k) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int n = K * N;
  if (idx >= n) return;
  int col = idx / K;
  int row = idx - col * K;
  int kb = row / block_k;
  int scale_idx = col * (K / block_k) + kb;
  half h = fp8_lut_decode_const(in8_col[idx]);
  half hs = fp8_lut_decode_const(scales8[scale_idx]);
  out16_col[idx] = __float2half(__half2float(h) * __half2float(hs));
}

__global__ void dequantize_int8_to_fp16_vec4_kernel(
    const int8_t* __restrict__ in8,
    half* __restrict__ out16,
    int n,
    float scale) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int base = tid * 4;
  if (base >= n) return;

  if (base + 3 < n) {
    uint32_t packed = *reinterpret_cast<const uint32_t*>(in8 + base);
    int8_t b0 = (int8_t)(packed & 0xFFu);
    int8_t b1 = (int8_t)((packed >> 8) & 0xFFu);
    int8_t b2 = (int8_t)((packed >> 16) & 0xFFu);
    int8_t b3 = (int8_t)((packed >> 24) & 0xFFu);
    float s = scale;
    out16[base + 0] = __float2half((float)b0 * s);
    out16[base + 1] = __float2half((float)b1 * s);
    out16[base + 2] = __float2half((float)b2 * s);
    out16[base + 3] = __float2half((float)b3 * s);
  } else {
    for (int i = 0; i < 4; ++i) {
      int idx = base + i;
      if (idx < n) out16[idx] = __float2half((float)in8[idx] * scale);
    }
  }
}

__global__ void dequantize_int8_to_fp16_blockscale_f16_vec4_kernel(
    const int8_t* __restrict__ in8_col,
    half* __restrict__ out16_col,
    int K,
    int N,
    const half* __restrict__ scales16,
    int block_k) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int base = tid * 4;
  int n = K * N;
  if (base >= n) return;

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    int idx = base + i;
    if (idx >= n) break;
    int col = idx / K;
    int row = idx - col * K;
    int kb = row / block_k;
    int scale_idx = col * (K / block_k) + kb;
    float s = __half2float(scales16[scale_idx]);
    out16_col[idx] = __float2half((float)in8_col[idx] * s);
  }
}

__global__ void dequantize_fp8e4m3_to_fp16_tex_kernel(
    const uint8_t* __restrict__ in8,
    half* __restrict__ out16,
    int n,
    float scale,
    cudaTextureObject_t lut_tex) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  half h = fp8_lut_decode_tex(lut_tex, in8[idx]);
  float x = __half2float(h) * scale;
  out16[idx] = __float2half(x);
}

template <bool UseTex>
__global__ void dequantize_fp8e4m3_to_fp16_long_kernel(
    const uint8_t* __restrict__ in8,
    half* __restrict__ out16,
    int n,
    float scale,
    cudaTextureObject_t lut_tex,
    int extra_iters) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  uint8_t v = in8[idx];
  half h0 = UseTex ? fp8_lut_decode_tex(lut_tex, v) : fp8_lut_decode_const(v);
  float acc = __half2float(h0);
  #pragma unroll 1
  for (int i = 0; i < extra_iters; ++i) {
    half h = UseTex ? fp8_lut_decode_tex(lut_tex, v) : fp8_lut_decode_const(v);
    acc = fmaf(acc, 1.000061f, __half2float(h));
  }
  out16[idx] = __float2half(acc * scale);
}

static void make_fp8_lut_device_and_tex(uint16_t** d_lut_out, cudaTextureObject_t* tex_out) {
  uint16_t host_lut[256];
  for (int i = 0; i < 256; ++i) {
    float f = fp8_e4m3_to_f32((uint8_t)i);
    host_lut[i] = f32_to_f16_bits(f);
  }
  CUDA_CHECK(cudaMemcpyToSymbol(k_fp8_e4m3_to_f16_bits, host_lut, sizeof(host_lut)));

  uint16_t* d_lut = nullptr;
  CUDA_CHECK(cudaMalloc(&d_lut, 256 * sizeof(uint16_t)));
  CUDA_CHECK(cudaMemcpy(d_lut, host_lut, 256 * sizeof(uint16_t), cudaMemcpyHostToDevice));
  cudaTextureObject_t tex = make_lut_texture_object_u16(d_lut);
  *d_lut_out = d_lut;
  *tex_out = tex;
}

static void run_bench_fp8quant() {
  upload_fp8_lut();

  constexpr int M = 1024;
  constexpr int N = 1024;
  constexpr int K = 1024;
  constexpr int repeats = 50;

  size_t bytes_Ah = (size_t)M * K * sizeof(half);
  size_t bytes_Bh = (size_t)K * N * sizeof(half);
  size_t bytes_A8 = (size_t)M * K;
  size_t bytes_B8 = (size_t)K * N;
  size_t bytes_C = (size_t)M * N * sizeof(float);

  // Host FP16 inputs.
  std::vector<half> h_A((size_t)M * K);
  std::vector<half> h_B_col((size_t)K * N);

  std::mt19937 rng(123);
  std::normal_distribution<float> nd(0.0f, 0.5f);

  float max_abs = 0.0f;
  for (int i = 0; i < M * K; ++i) {
    float v = nd(rng);
    max_abs = std::max(max_abs, std::fabs(v));
    h_A[(size_t)i] = __float2half(v);
  }
  for (int c = 0; c < N; ++c) {
    for (int r = 0; r < K; ++r) {
      float v = nd(rng);
      max_abs = std::max(max_abs, std::fabs(v));
      h_B_col[(size_t)c * K + r] = __float2half(v);
    }
  }

  // Simple per-tensor scale: map max_abs to max finite (240).
  float scale = (max_abs > 0.0f) ? (max_abs / 240.0f) : 1.0f;
  float inv_scale = 1.0f / scale;

  half* d_Ah = nullptr;
  half* d_Bh_col = nullptr;
  uint8_t* d_A8 = nullptr;
  uint8_t* d_B8 = nullptr;
  half* d_Ah_deq = nullptr;
  half* d_Bh_deq = nullptr;
  float* d_C = nullptr;

  CUDA_CHECK(cudaMalloc(&d_Ah, bytes_Ah));
  CUDA_CHECK(cudaMalloc(&d_Bh_col, bytes_Bh));
  CUDA_CHECK(cudaMalloc(&d_A8, bytes_A8));
  CUDA_CHECK(cudaMalloc(&d_B8, bytes_B8));
  CUDA_CHECK(cudaMalloc(&d_Ah_deq, bytes_Ah));
  CUDA_CHECK(cudaMalloc(&d_Bh_deq, bytes_Bh));
  CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

  CUDA_CHECK(cudaMemcpy(d_Ah, h_A.data(), bytes_Ah, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Bh_col, h_B_col.data(), bytes_Bh, cudaMemcpyHostToDevice));

  // cuBLAS setup
  cublasHandle_t handle{};
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUBLAS_CHECK(cublasSetStream(handle, stream));

  float alpha = 1.0f;
  float beta = 0.0f;

  // One-time quantization (measure separately)
  {
    GpuTimer t;
    t.begin(stream);
    quantize_fp16_to_fp8e4m3_kernel<<<((M * K) + 255) / 256, 256, 0, stream>>>(d_Ah, d_A8, M * K, inv_scale);
    quantize_fp16_to_fp8e4m3_kernel<<<((K * N) + 255) / 256, 256, 0, stream>>>(d_Bh_col, d_B8, K * N, inv_scale);
    CUDA_CHECK(cudaGetLastError());
    float ms = t.end_ms(stream);
    printf("[fp8quant] scale=%.6g max_abs=%.6g quantize_once: %.3f ms\n", scale, max_abs, ms);
  }

  // Baseline: FP16 cuBLAS
  {
    GpuTimer t;
    // Warmup
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        d_Ah, CUDA_R_16F, K,
        d_Bh_col, CUDA_R_16F, K,
        &beta,
        d_C, CUDA_R_32F, M,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CUDA_CHECK(cudaGetLastError());

    t.begin(stream);
    for (int r = 0; r < repeats; ++r) {
      CUBLAS_CHECK(cublasGemmEx(
          handle,
          CUBLAS_OP_T, CUBLAS_OP_N,
          M, N, K,
          &alpha,
          d_Ah, CUDA_R_16F, K,
          d_Bh_col, CUDA_R_16F, K,
          &beta,
          d_C, CUDA_R_32F, M,
          CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
            CUDA_CHECK(cudaGetLastError());
            float ms = t.end_ms(stream);
    double flops = 2.0 * (double)M * (double)N * (double)K * repeats;
    double tflops = (flops / (ms / 1e3)) / 1e12;
    printf("[fp8quant] fp16_cublas: %.3f ms total (%d reps)  (%.2f TFLOP/s)\n", ms, repeats, tflops);
  }

  // FP8 storage: dequant -> FP16, then cuBLAS.
  {
    GpuTimer t;
    // Warmup
    dequantize_fp8e4m3_to_fp16_kernel<<<((M * K) + 255) / 256, 256, 0, stream>>>(d_A8, d_Ah_deq, M * K, scale);
    dequantize_fp8e4m3_to_fp16_kernel<<<((K * N) + 255) / 256, 256, 0, stream>>>(d_B8, d_Bh_deq, K * N, scale);
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        d_Ah_deq, CUDA_R_16F, K,
        d_Bh_deq, CUDA_R_16F, K,
        &beta,
        d_C, CUDA_R_32F, M,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CUDA_CHECK(cudaGetLastError());

    t.begin(stream);
    for (int r = 0; r < repeats; ++r) {
      dequantize_fp8e4m3_to_fp16_kernel<<<((M * K) + 255) / 256, 256, 0, stream>>>(d_A8, d_Ah_deq, M * K, scale);
      dequantize_fp8e4m3_to_fp16_kernel<<<((K * N) + 255) / 256, 256, 0, stream>>>(d_B8, d_Bh_deq, K * N, scale);
      CUBLAS_CHECK(cublasGemmEx(
          handle,
          CUBLAS_OP_T, CUBLAS_OP_N,
          M, N, K,
          &alpha,
          d_Ah_deq, CUDA_R_16F, K,
          d_Bh_deq, CUDA_R_16F, K,
          &beta,
          d_C, CUDA_R_32F, M,
          CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CUDA_CHECK(cudaGetLastError());
    float ms = t.end_ms(stream);
    double flops = 2.0 * (double)M * (double)N * (double)K * repeats;
    double tflops = (flops / (ms / 1e3)) / 1e12;
    printf("[fp8quant] fp8_storage(dequant+fp16_cublas): %.3f ms total (%d reps)  (%.2f TFLOP/s)\n", ms, repeats, tflops);
  }

  cublasDestroy(handle);
  CUDA_CHECK(cudaStreamDestroy(stream));
  cudaFree(d_Ah);
  cudaFree(d_Bh_col);
  cudaFree(d_A8);
  cudaFree(d_B8);
  cudaFree(d_Ah_deq);
  cudaFree(d_Bh_deq);
  cudaFree(d_C);
}

static void run_bench_fp8sweep() {
  // Sweep K to move between bandwidth-ish and compute-ish regimes.
  constexpr int M = 1024;
  constexpr int N = 1024;
  const int Ks[] = {64, 128, 256, 512, 1024};

  uint16_t* d_lut = nullptr;
  cudaTextureObject_t lut_tex = 0;
  make_fp8_lut_device_and_tex(&d_lut, &lut_tex);

  cublasHandle_t handle{};
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUBLAS_CHECK(cublasSetStream(handle, stream));

  float alpha = 1.0f;
  float beta = 0.0f;

  printf("[fp8sweep] M=%d N=%d (varying K)\n", M, N);
  printf("[fp8sweep] columns: K repeats fp16_cublas(TF) fp8reuse_const(TF) fp8reuse_tex(TF) long8_const(TF) long8_tex(TF)\n");

  // Keep total work roughly bounded.
  const double base_flops = 2.0 * (double)M * (double)N * 1024.0 * 200.0;

  for (int kk = 0; kk < (int)(sizeof(Ks) / sizeof(Ks[0])); ++kk) {
    int K = Ks[kk];
    if ((K % 8) != 0) continue;

    int repeats = (int)llround(base_flops / (2.0 * (double)M * (double)N * (double)K));
    repeats = std::max(20, std::min(400, repeats));

    size_t bytes_Ah = (size_t)M * K * sizeof(half);
    size_t bytes_Bh = (size_t)K * N * sizeof(half);
    size_t bytes_A8 = (size_t)M * K;
    size_t bytes_B8 = (size_t)K * N;
    size_t bytes_C = (size_t)M * N * sizeof(float);

    std::vector<half> h_A((size_t)M * K);
    std::vector<half> h_B_col((size_t)K * N);

    std::mt19937 rng(123 + K);
    std::normal_distribution<float> nd(0.0f, 0.5f);
    float max_abs = 0.0f;
    for (int i = 0; i < M * K; ++i) {
      float v = nd(rng);
      max_abs = std::max(max_abs, std::fabs(v));
      h_A[(size_t)i] = __float2half(v);
    }
    for (int c = 0; c < N; ++c) {
      for (int r = 0; r < K; ++r) {
        float v = nd(rng);
        max_abs = std::max(max_abs, std::fabs(v));
        h_B_col[(size_t)c * K + r] = __float2half(v);
      }
    }
    float scale = (max_abs > 0.0f) ? (max_abs / 240.0f) : 1.0f;
    float inv_scale = 1.0f / scale;

    half* d_Ah = nullptr;
    half* d_Bh_col = nullptr;
    uint8_t* d_A8 = nullptr;
    uint8_t* d_B8 = nullptr;
    half* d_Ah_deq = nullptr;
    half* d_Bh_deq = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_Ah, bytes_Ah));
    CUDA_CHECK(cudaMalloc(&d_Bh_col, bytes_Bh));
    CUDA_CHECK(cudaMalloc(&d_A8, bytes_A8));
    CUDA_CHECK(cudaMalloc(&d_B8, bytes_B8));
    CUDA_CHECK(cudaMalloc(&d_Ah_deq, bytes_Ah));
    CUDA_CHECK(cudaMalloc(&d_Bh_deq, bytes_Bh));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    CUDA_CHECK(cudaMemcpy(d_Ah, h_A.data(), bytes_Ah, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bh_col, h_B_col.data(), bytes_Bh, cudaMemcpyHostToDevice));

    // Quantize once.
    quantize_fp16_to_fp8e4m3_kernel<<<((M * K) + 255) / 256, 256, 0, stream>>>(d_Ah, d_A8, M * K, inv_scale);
    quantize_fp16_to_fp8e4m3_kernel<<<((K * N) + 255) / 256, 256, 0, stream>>>(d_Bh_col, d_B8, K * N, inv_scale);
    CUDA_CHECK(cudaGetLastError());

    // Weights reuse: dequantize B once (const).
    dequantize_fp8e4m3_to_fp16_kernel<<<((K * N) + 255) / 256, 256, 0, stream>>>(d_B8, d_Bh_deq, K * N, scale);
    CUDA_CHECK(cudaGetLastError());

    auto tflops_for = [&](float ms_total) {
      double flops = 2.0 * (double)M * (double)N * (double)K * repeats;
      return (flops / (ms_total / 1e3)) / 1e12;
    };

    // FP16 baseline.
    {
      CUBLAS_CHECK(cublasGemmEx(
          handle,
          CUBLAS_OP_T, CUBLAS_OP_N,
          M, N, K,
          &alpha,
          d_Ah, CUDA_R_16F, K,
          d_Bh_col, CUDA_R_16F, K,
          &beta,
          d_C, CUDA_R_32F, M,
          CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      GpuTimer t;
      t.begin(stream);
      for (int r = 0; r < repeats; ++r) {
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            d_Ah, CUDA_R_16F, K,
            d_Bh_col, CUDA_R_16F, K,
            &beta,
            d_C, CUDA_R_32F, M,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      }
      float ms = t.end_ms(stream);
      double tf = tflops_for(ms);

      // fp8 reuse (dequant A only) const
      float tf_const = 0.0;
      {
        dequantize_fp8e4m3_to_fp16_kernel<<<((M * K) + 255) / 256, 256, 0, stream>>>(d_A8, d_Ah_deq, M * K, scale);
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            d_Ah_deq, CUDA_R_16F, K,
            d_Bh_deq, CUDA_R_16F, K,
            &beta,
            d_C, CUDA_R_32F, M,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        GpuTimer tt;
        tt.begin(stream);
        for (int r = 0; r < repeats; ++r) {
          dequantize_fp8e4m3_to_fp16_kernel<<<((M * K) + 255) / 256, 256, 0, stream>>>(d_A8, d_Ah_deq, M * K, scale);
          CUBLAS_CHECK(cublasGemmEx(
              handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              M, N, K,
              &alpha,
              d_Ah_deq, CUDA_R_16F, K,
              d_Bh_deq, CUDA_R_16F, K,
              &beta,
              d_C, CUDA_R_32F, M,
              CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        float ms2 = tt.end_ms(stream);
        tf_const = tflops_for(ms2);
      }

      // fp8 reuse (dequant A only) tex
      float tf_tex = 0.0;
      {
        dequantize_fp8e4m3_to_fp16_tex_kernel<<<((M * K) + 255) / 256, 256, 0, stream>>>(d_A8, d_Ah_deq, M * K, scale, lut_tex);
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            d_Ah_deq, CUDA_R_16F, K,
            d_Bh_deq, CUDA_R_16F, K,
            &beta,
            d_C, CUDA_R_32F, M,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        GpuTimer tt;
        tt.begin(stream);
        for (int r = 0; r < repeats; ++r) {
          dequantize_fp8e4m3_to_fp16_tex_kernel<<<((M * K) + 255) / 256, 256, 0, stream>>>(d_A8, d_Ah_deq, M * K, scale, lut_tex);
          CUBLAS_CHECK(cublasGemmEx(
              handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              M, N, K,
              &alpha,
              d_Ah_deq, CUDA_R_16F, K,
              d_Bh_deq, CUDA_R_16F, K,
              &beta,
              d_C, CUDA_R_32F, M,
              CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        float ms2 = tt.end_ms(stream);
        tf_tex = tflops_for(ms2);
      }

      // long decode (extra 8) const
      float tf_long_const = 0.0;
      {
        dequantize_fp8e4m3_to_fp16_long_kernel<false><<<((M * K) + 255) / 256, 256, 0, stream>>>(d_A8, d_Ah_deq, M * K, scale, lut_tex, 8);
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            d_Ah_deq, CUDA_R_16F, K,
            d_Bh_deq, CUDA_R_16F, K,
            &beta,
            d_C, CUDA_R_32F, M,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        GpuTimer tt;
        tt.begin(stream);
        for (int r = 0; r < repeats; ++r) {
          dequantize_fp8e4m3_to_fp16_long_kernel<false><<<((M * K) + 255) / 256, 256, 0, stream>>>(d_A8, d_Ah_deq, M * K, scale, lut_tex, 8);
          CUBLAS_CHECK(cublasGemmEx(
              handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              M, N, K,
              &alpha,
              d_Ah_deq, CUDA_R_16F, K,
              d_Bh_deq, CUDA_R_16F, K,
              &beta,
              d_C, CUDA_R_32F, M,
              CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        float ms2 = tt.end_ms(stream);
        tf_long_const = tflops_for(ms2);
      }

      // long decode (extra 8) tex
      float tf_long_tex = 0.0;
      {
        dequantize_fp8e4m3_to_fp16_long_kernel<true><<<((M * K) + 255) / 256, 256, 0, stream>>>(d_A8, d_Ah_deq, M * K, scale, lut_tex, 8);
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            d_Ah_deq, CUDA_R_16F, K,
            d_Bh_deq, CUDA_R_16F, K,
            &beta,
            d_C, CUDA_R_32F, M,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        GpuTimer tt;
        tt.begin(stream);
        for (int r = 0; r < repeats; ++r) {
          dequantize_fp8e4m3_to_fp16_long_kernel<true><<<((M * K) + 255) / 256, 256, 0, stream>>>(d_A8, d_Ah_deq, M * K, scale, lut_tex, 8);
          CUBLAS_CHECK(cublasGemmEx(
              handle,
              CUBLAS_OP_T, CUBLAS_OP_N,
              M, N, K,
              &alpha,
              d_Ah_deq, CUDA_R_16F, K,
              d_Bh_deq, CUDA_R_16F, K,
              &beta,
              d_C, CUDA_R_32F, M,
              CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        float ms2 = tt.end_ms(stream);
        tf_long_tex = tflops_for(ms2);
      }

      printf("[fp8sweep] K=%4d reps=%3d  fp16=%.2f  reuse_const=%.2f  reuse_tex=%.2f  long8_const=%.2f  long8_tex=%.2f\n",
             K, repeats, tf, tf_const, tf_tex, tf_long_const, tf_long_tex);
    }

    cudaFree(d_Ah);
    cudaFree(d_Bh_col);
    cudaFree(d_A8);
    cudaFree(d_B8);
    cudaFree(d_Ah_deq);
    cudaFree(d_Bh_deq);
    cudaFree(d_C);
  }

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaDestroyTextureObject(lut_tex));
  cudaFree(d_lut);
}

static void run_bench_fp8reuse() {
  upload_fp8_lut();

  constexpr int M = 1024;
  constexpr int N = 1024;
  constexpr int K = 1024;
  constexpr int repeats = 200;

  size_t bytes_Ah = (size_t)M * K * sizeof(half);
  size_t bytes_Bh = (size_t)K * N * sizeof(half);
  size_t bytes_A8 = (size_t)M * K;
  size_t bytes_B8 = (size_t)K * N;
  size_t bytes_C = (size_t)M * N * sizeof(float);

  std::vector<half> h_A((size_t)M * K);
  std::vector<half> h_B_col((size_t)K * N);

  std::mt19937 rng(123);
  std::normal_distribution<float> nd(0.0f, 0.5f);

  float max_abs = 0.0f;
  for (int i = 0; i < M * K; ++i) {
    float v = nd(rng);
    max_abs = std::max(max_abs, std::fabs(v));
    h_A[(size_t)i] = __float2half(v);
  }
  for (int c = 0; c < N; ++c) {
    for (int r = 0; r < K; ++r) {
      float v = nd(rng);
      max_abs = std::max(max_abs, std::fabs(v));
      h_B_col[(size_t)c * K + r] = __float2half(v);
    }
  }

  float scale = (max_abs > 0.0f) ? (max_abs / 240.0f) : 1.0f;
  float inv_scale = 1.0f / scale;

  half* d_Ah = nullptr;
  half* d_Bh_col = nullptr;
  uint8_t* d_A8 = nullptr;
  uint8_t* d_B8 = nullptr;
  half* d_Ah_deq = nullptr;
  half* d_Bh_deq = nullptr;
  float* d_C = nullptr;

  CUDA_CHECK(cudaMalloc(&d_Ah, bytes_Ah));
  CUDA_CHECK(cudaMalloc(&d_Bh_col, bytes_Bh));
  CUDA_CHECK(cudaMalloc(&d_A8, bytes_A8));
  CUDA_CHECK(cudaMalloc(&d_B8, bytes_B8));
  CUDA_CHECK(cudaMalloc(&d_Ah_deq, bytes_Ah));
  CUDA_CHECK(cudaMalloc(&d_Bh_deq, bytes_Bh));
  CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

  CUDA_CHECK(cudaMemcpy(d_Ah, h_A.data(), bytes_Ah, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Bh_col, h_B_col.data(), bytes_Bh, cudaMemcpyHostToDevice));

  cublasHandle_t handle{};
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUBLAS_CHECK(cublasSetStream(handle, stream));

  float alpha = 1.0f;
  float beta = 0.0f;

  // One-time quantize A and B to FP8.
  {
    GpuTimer t;
    t.begin(stream);
    quantize_fp16_to_fp8e4m3_kernel<<<((M * K) + 255) / 256, 256, 0, stream>>>(d_Ah, d_A8, M * K, inv_scale);
    quantize_fp16_to_fp8e4m3_kernel<<<((K * N) + 255) / 256, 256, 0, stream>>>(d_Bh_col, d_B8, K * N, inv_scale);
    CUDA_CHECK(cudaGetLastError());
    float ms = t.end_ms(stream);
    printf("[fp8reuse] scale=%.6g max_abs=%.6g quantize_once(A+B): %.3f ms\n", scale, max_abs, ms);
  }

  auto report = [&](const char* name, float ms_total) {
    double flops = 2.0 * (double)M * (double)N * (double)K * repeats;
    double tflops = (flops / (ms_total / 1e3)) / 1e12;
    printf("[fp8reuse] %s: %.3f ms total (%d reps)  (%.2f TFLOP/s)\n", name, ms_total, repeats, tflops);
  };

  // Prepare cuBLASLt once so we can use its heuristic-picked algo across scenarios.
  LtGemmF16 lt;
  bool lt_ok = init_lt_gemm_f16_colmajor_atr_bn(lt, M, N, K, 64ull * 1024 * 1024, stream);

  // Baseline: FP16 cuBLASLt (public heuristic-picked algo) using same layout trick as cuBLAS.
  if (lt_ok && lt.ready) {
    lt_gemm_f16_run(lt, &alpha, d_Ah, d_Bh_col, &beta, d_C, stream);
    GpuTimer t;
    t.begin(stream);
    for (int r = 0; r < repeats; ++r) {
      lt_gemm_f16_run(lt, &alpha, d_Ah, d_Bh_col, &beta, d_C, stream);
    }
    float ms = t.end_ms(stream);
    report("fp16_cublasLt(heuristic)", ms);
  } else {
    printf("[fp8reuse] fp16_cublasLt(heuristic): skipped (no algo)\n");
  }

  // Baseline: FP16 cuBLAS.
  {
    // Warmup
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        d_Ah, CUDA_R_16F, K,
        d_Bh_col, CUDA_R_16F, K,
        &beta,
        d_C, CUDA_R_32F, M,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    GpuTimer t;
    t.begin(stream);
    for (int r = 0; r < repeats; ++r) {
      CUBLAS_CHECK(cublasGemmEx(
          handle,
          CUBLAS_OP_T, CUBLAS_OP_N,
          M, N, K,
          &alpha,
          d_Ah, CUDA_R_16F, K,
          d_Bh_col, CUDA_R_16F, K,
          &beta,
          d_C, CUDA_R_32F, M,
          CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    float ms = t.end_ms(stream);
    report("fp16_cublas", ms);
  }

  // Scenario 1: dequantize both A and B every iteration (worst case).
  {
    // Warmup
    dequantize_fp8e4m3_to_fp16_kernel<<<((M * K) + 255) / 256, 256, 0, stream>>>(d_A8, d_Ah_deq, M * K, scale);
    dequantize_fp8e4m3_to_fp16_kernel<<<((K * N) + 255) / 256, 256, 0, stream>>>(d_B8, d_Bh_deq, K * N, scale);
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        d_Ah_deq, CUDA_R_16F, K,
        d_Bh_deq, CUDA_R_16F, K,
        &beta,
        d_C, CUDA_R_32F, M,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    GpuTimer t;
    t.begin(stream);
    for (int r = 0; r < repeats; ++r) {
      dequantize_fp8e4m3_to_fp16_kernel<<<((M * K) + 255) / 256, 256, 0, stream>>>(d_A8, d_Ah_deq, M * K, scale);
      dequantize_fp8e4m3_to_fp16_kernel<<<((K * N) + 255) / 256, 256, 0, stream>>>(d_B8, d_Bh_deq, K * N, scale);
      CUBLAS_CHECK(cublasGemmEx(
          handle,
          CUBLAS_OP_T, CUBLAS_OP_N,
          M, N, K,
          &alpha,
          d_Ah_deq, CUDA_R_16F, K,
          d_Bh_deq, CUDA_R_16F, K,
          &beta,
          d_C, CUDA_R_32F, M,
          CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    float ms = t.end_ms(stream);
    report("fp8_storage(dequant A+B each iter) + cublas", ms);
  }

  // Scenario 1b: dequantize both A and B every iteration, then cuBLASLt heuristic-picked GEMM.
  if (lt_ok && lt.ready) {
    dequantize_fp8e4m3_to_fp16_kernel<<<((M * K) + 255) / 256, 256, 0, stream>>>(d_A8, d_Ah_deq, M * K, scale);
    dequantize_fp8e4m3_to_fp16_kernel<<<((K * N) + 255) / 256, 256, 0, stream>>>(d_B8, d_Bh_deq, K * N, scale);
    lt_gemm_f16_run(lt, &alpha, d_Ah_deq, d_Bh_deq, &beta, d_C, stream);
    CUDA_CHECK(cudaGetLastError());

    GpuTimer t;
    t.begin(stream);
    for (int r = 0; r < repeats; ++r) {
      dequantize_fp8e4m3_to_fp16_kernel<<<((M * K) + 255) / 256, 256, 0, stream>>>(d_A8, d_Ah_deq, M * K, scale);
      dequantize_fp8e4m3_to_fp16_kernel<<<((K * N) + 255) / 256, 256, 0, stream>>>(d_B8, d_Bh_deq, K * N, scale);
      lt_gemm_f16_run(lt, &alpha, d_Ah_deq, d_Bh_deq, &beta, d_C, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    float ms = t.end_ms(stream);
    report("fp8_storage(dequant A+B each iter) + cublasLt", ms);
  }

  // Scenario 2: weights reused: dequantize B once, only dequantize A each iteration.
  {
    // One-time B dequant
    dequantize_fp8e4m3_to_fp16_kernel<<<((K * N) + 255) / 256, 256, 0, stream>>>(d_B8, d_Bh_deq, K * N, scale);
    CUDA_CHECK(cudaGetLastError());

    // Warmup
    dequantize_fp8e4m3_to_fp16_kernel<<<((M * K) + 255) / 256, 256, 0, stream>>>(d_A8, d_Ah_deq, M * K, scale);
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        d_Ah_deq, CUDA_R_16F, K,
        d_Bh_deq, CUDA_R_16F, K,
        &beta,
        d_C, CUDA_R_32F, M,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    GpuTimer t;
    t.begin(stream);
    for (int r = 0; r < repeats; ++r) {
      dequantize_fp8e4m3_to_fp16_kernel<<<((M * K) + 255) / 256, 256, 0, stream>>>(d_A8, d_Ah_deq, M * K, scale);
      CUBLAS_CHECK(cublasGemmEx(
          handle,
          CUBLAS_OP_T, CUBLAS_OP_N,
          M, N, K,
          &alpha,
          d_Ah_deq, CUDA_R_16F, K,
          d_Bh_deq, CUDA_R_16F, K,
          &beta,
          d_C, CUDA_R_32F, M,
          CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    float ms = t.end_ms(stream);
    report("fp8_storage(weights reuse: dequant A only) + cublas", ms);
  }

  // Scenario 2b: weights reused + cuBLASLt heuristic-picked GEMM.
  if (lt_ok && lt.ready) {
    dequantize_fp8e4m3_to_fp16_kernel<<<((K * N) + 255) / 256, 256, 0, stream>>>(d_B8, d_Bh_deq, K * N, scale);
    dequantize_fp8e4m3_to_fp16_kernel<<<((M * K) + 255) / 256, 256, 0, stream>>>(d_A8, d_Ah_deq, M * K, scale);
    lt_gemm_f16_run(lt, &alpha, d_Ah_deq, d_Bh_deq, &beta, d_C, stream);
    CUDA_CHECK(cudaGetLastError());

    GpuTimer t;
    t.begin(stream);
    for (int r = 0; r < repeats; ++r) {
      dequantize_fp8e4m3_to_fp16_kernel<<<((M * K) + 255) / 256, 256, 0, stream>>>(d_A8, d_Ah_deq, M * K, scale);
      lt_gemm_f16_run(lt, &alpha, d_Ah_deq, d_Bh_deq, &beta, d_C, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    float ms = t.end_ms(stream);
    report("fp8_storage(weights reuse: dequant A only) + cublasLt", ms);
  }

  destroy_lt_gemm(lt);

  cublasDestroy(handle);
  CUDA_CHECK(cudaStreamDestroy(stream));
  cudaFree(d_Ah);
  cudaFree(d_Bh_col);
  cudaFree(d_A8);
  cudaFree(d_B8);
  cudaFree(d_Ah_deq);
  cudaFree(d_Bh_deq);
  cudaFree(d_C);
}

