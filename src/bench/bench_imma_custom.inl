// ------------------------ BENCH: custom IMMA fused (int8xint8 + per-col scale -> fp16) ------------------------
static void run_bench_imma_int8bfp_fused(bool use_l2pin) {
  constexpr int M = 1024;
  constexpr int N = 1024;
  constexpr int K = 1024;
  constexpr int repeats = 200;
  constexpr float global_scale = 1.0f;

  static_assert((M % 32) == 0 && (N % 32) == 0, "M and N must be multiples of 32");
  static_assert((K % 16) == 0, "K must be multiple of 16");

  size_t bytes_A = (size_t)M * K * sizeof(int8_t);
  size_t bytes_B = (size_t)K * N * sizeof(int8_t);
  size_t bytes_D = (size_t)M * N * sizeof(half);
  size_t bytes_scale = (size_t)N * sizeof(uint16_t);

  std::vector<int8_t> h_A((size_t)M * K);
  std::vector<int8_t> h_B((size_t)K * N);
  std::vector<uint16_t> h_scale_u16((size_t)N);

  std::mt19937 rng(123);
  std::uniform_int_distribution<int> dist(-8, 7);
  std::uniform_real_distribution<float> sd(0.25f, 4.0f);
  for (int i = 0; i < M * K; ++i) h_A[(size_t)i] = (int8_t)dist(rng);
  for (int i = 0; i < K * N; ++i) h_B[(size_t)i] = (int8_t)dist(rng);
  for (int n = 0; n < N; ++n) h_scale_u16[(size_t)n] = f32_to_f16_bits(sd(rng));

  int8_t* d_A = nullptr;
  int8_t* d_B_col = nullptr;
  half* d_D = nullptr;
  uint16_t* d_scale_u16 = nullptr;

  CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
  CUDA_CHECK(cudaMalloc(&d_B_col, bytes_B));
  CUDA_CHECK(cudaMalloc(&d_D, bytes_D));
  CUDA_CHECK(cudaMalloc(&d_scale_u16, bytes_scale));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scale_u16, h_scale_u16.data(), bytes_scale, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D, 0, bytes_D));

  // Upload B as col-major on device.
  {
    std::vector<int8_t> h_B_col((size_t)K * N);
    for (int r = 0; r < K; ++r) {
      for (int c = 0; c < N; ++c) {
        h_B_col[(size_t)c * K + r] = h_B[(size_t)r * N + c];
      }
    }
    CUDA_CHECK(cudaMemcpy(d_B_col, h_B_col.data(), bytes_B, cudaMemcpyHostToDevice));
  }

  cudaStream_t stream = 0;
  if (use_l2pin) {
    try_enable_persisting_l2(bytes_B + bytes_scale);
    try_set_stream_access_policy_persisting(stream, d_B_col, bytes_B, 1.0f);
    try_set_stream_access_policy_persisting(stream, d_scale_u16, bytes_scale, 1.0f);
  }

  // Default config: 2x2 warp tiles per block (4 warps).
  constexpr int BM = 2;
  constexpr int BN = 2;
  dim3 block(32 * BM * BN, 1, 1);
  dim3 grid((N + (BN * 16 - 1)) / (BN * 16), (M + (BM * 16 - 1)) / (BM * 16), 1);
  // Shared: int32 tiles (warps * 256) + float scales (warps * 16)
  size_t smem_bytes = (BM * BN * 16 * 16) * sizeof(int32_t) + (BM * BN * 16) * sizeof(float);

  auto run_once = [&] {
    imma_gemm_int8_colscale_fp16_colmajor_kernel<BM, BN><<<grid, block, smem_bytes, stream>>>(
        d_A, d_B_col, d_scale_u16, d_D, M, N, K, global_scale);
  };

  // Warmup
  run_once();
  CUDA_CHECK(cudaDeviceSynchronize());

  GpuTimer t;
  t.begin(stream);
  for (int r = 0; r < repeats; ++r) run_once();
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms_avg = t.end_ms(stream) / (float)repeats;

  double ops = 2.0 * (double)M * (double)N * (double)K;
  double tops = (ops / (ms_avg / 1e3)) / 1e12;
  printf("[imma_int8bfp_fused] int8xint8 + per-col scale -> fp16 (single kernel): %.3f ms/iter  (%.2f TOPS)\n", ms_avg, tops);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B_col));
  CUDA_CHECK(cudaFree(d_D));
  CUDA_CHECK(cudaFree(d_scale_u16));
  if (use_l2pin) {
    (void)cudaCtxResetPersistingL2Cache();
  }
}

// ------------------------ BENCH: custom IMMA fused v2 (64x64 tile + shared staging) ------------------------
static void run_bench_imma_int8bfp_fused_v2(bool use_l2pin) {
  constexpr int M = 1024;
  constexpr int N = 1024;
  constexpr int K = 1024;
  constexpr int repeats = 200;
  constexpr float global_scale = 1.0f;

  static_assert((K % 16) == 0, "K must be multiple of 16");

  size_t bytes_A = (size_t)M * K * sizeof(int8_t);
  size_t bytes_B = (size_t)K * N * sizeof(int8_t);
  size_t bytes_D = (size_t)M * N * sizeof(half);
  size_t bytes_scale = (size_t)N * sizeof(uint16_t);

  std::vector<int8_t> h_A((size_t)M * K);
  std::vector<int8_t> h_B((size_t)K * N);
  std::vector<uint16_t> h_scale_u16((size_t)N);

  std::mt19937 rng(123);
  std::uniform_int_distribution<int> dist(-8, 7);
  std::uniform_real_distribution<float> sd(0.25f, 4.0f);
  for (int i = 0; i < M * K; ++i) h_A[(size_t)i] = (int8_t)dist(rng);
  for (int i = 0; i < K * N; ++i) h_B[(size_t)i] = (int8_t)dist(rng);
  for (int n = 0; n < N; ++n) h_scale_u16[(size_t)n] = f32_to_f16_bits(sd(rng));

  int8_t* d_A = nullptr;
  int8_t* d_B_col = nullptr;
  half* d_D = nullptr;
  uint16_t* d_scale_u16 = nullptr;

  CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
  CUDA_CHECK(cudaMalloc(&d_B_col, bytes_B));
  CUDA_CHECK(cudaMalloc(&d_D, bytes_D));
  CUDA_CHECK(cudaMalloc(&d_scale_u16, bytes_scale));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scale_u16, h_scale_u16.data(), bytes_scale, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D, 0, bytes_D));

  // Upload B as col-major on device.
  {
    std::vector<int8_t> h_B_col((size_t)K * N);
    for (int r = 0; r < K; ++r) {
      for (int c = 0; c < N; ++c) {
        h_B_col[(size_t)c * K + r] = h_B[(size_t)r * N + c];
      }
    }
    CUDA_CHECK(cudaMemcpy(d_B_col, h_B_col.data(), bytes_B, cudaMemcpyHostToDevice));
  }

  cudaStream_t stream = 0;
  if (use_l2pin) {
    try_enable_persisting_l2(bytes_B + bytes_scale);
    try_set_stream_access_policy_persisting(stream, d_B_col, bytes_B, 1.0f);
    try_set_stream_access_policy_persisting(stream, d_scale_u16, bytes_scale, 1.0f);
  }

  dim3 block(512, 1, 1);
  dim3 grid((N + 63) / 64, (M + 63) / 64, 1);
  constexpr int kKChunk = 32;
  // Shared bytes: A_sh(2*64*kKChunk) + B_sh(2*64*kKChunk) + C_sh(64*64*i32) + scales(64*f32)
  size_t smem_bytes = (2ull * 64 * (size_t)kKChunk) + (2ull * 64 * (size_t)kKChunk) + (64ull * 64 * sizeof(int32_t)) + (64ull * sizeof(float));

  // Explicitly opt-in to larger dynamic shared memory (Ampere default is 48KB).
  CUDA_CHECK(cudaFuncSetAttribute(
      imma_gemm_int8_colscale_fp16_colmajor_kernel_v2<kKChunk>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      (int)smem_bytes));

  auto run_once = [&] {
    imma_gemm_int8_colscale_fp16_colmajor_kernel_v2<kKChunk><<<grid, block, smem_bytes, stream>>>(
        d_A, d_B_col, d_scale_u16, d_D, M, N, K, global_scale);
    CUDA_CHECK(cudaGetLastError());
  };

  run_once();
  CUDA_CHECK(cudaDeviceSynchronize());
  GpuTimer t;
  t.begin(stream);
  for (int r = 0; r < repeats; ++r) run_once();
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms = t.end_ms(stream) / (float)repeats;

  double ops = 2.0 * (double)M * (double)N * (double)K;
  double tops = (ops / (ms / 1e3)) / 1e12;
  printf("[imma_int8bfp_fused_v2] fused int8xint8->int32 + colscale->fp16: %.3f ms/iter  (%.2f TOPS)\n", ms, tops);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B_col));
  CUDA_CHECK(cudaFree(d_D));
  CUDA_CHECK(cudaFree(d_scale_u16));
  if (use_l2pin) {
    (void)cudaCtxResetPersistingL2Cache();
  }
}

static void run_bench_imma_fp8_jit_v2(bool use_l2pin) {
  fp8imma::init_fp8_e4m3_lut();
  constexpr int M = 4096;
  constexpr int N = 4096;
  constexpr int K = 4096;
  constexpr int repeats = 200;
  constexpr float global_scale = 1.0f;

  static_assert((K % 16) == 0, "K must be multiple of 16");

  // A is INT8 (activations)
  size_t bytes_A = (size_t)M * K * sizeof(int8_t);
  // B is FP8 (weights) - one byte per element
  size_t bytes_B_fp8 = (size_t)K * N * sizeof(uint8_t);
  size_t bytes_D = (size_t)M * N * sizeof(half);
  size_t bytes_scale = (size_t)N * sizeof(uint16_t);

  std::vector<int8_t> h_A((size_t)M * K);
  std::vector<uint8_t> h_B_fp8((size_t)K * N);
  std::vector<uint16_t> h_scale_u16((size_t)N);

  std::mt19937 rng(123);
  std::uniform_int_distribution<int> dist(-127, 127); 
  std::uniform_real_distribution<float> sd(0.25f, 4.0f);
  
  for (size_t i = 0; i < h_A.size(); ++i) h_A[i] = (int8_t)dist(rng);
  for (size_t i = 0; i < h_B_fp8.size(); ++i) h_B_fp8[i] = (uint8_t)(rng() & 0xFF);
  for (int n = 0; n < N; ++n) h_scale_u16[(size_t)n] = f32_to_f16_bits(sd(rng));

  int8_t* d_A = nullptr;
  uint8_t* d_B_col_fp8 = nullptr;
  half* d_D = nullptr;
  uint16_t* d_scale_u16 = nullptr;

  CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
  CUDA_CHECK(cudaMalloc(&d_B_col_fp8, bytes_B_fp8));
  CUDA_CHECK(cudaMalloc(&d_D, bytes_D));
  CUDA_CHECK(cudaMalloc(&d_scale_u16, bytes_scale));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scale_u16, h_scale_u16.data(), bytes_scale, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D, 0, bytes_D));

  // Upload B as col-major on device (software transpose before upload)
  {
    std::vector<uint8_t> h_B_col((size_t)K * N);
    for (int r = 0; r < K; ++r) {
      for (int c = 0; c < N; ++c) {
        h_B_col[(size_t)c * K + r] = h_B_fp8[(size_t)r * N + c];
      }
    }
    CUDA_CHECK(cudaMemcpy(d_B_col_fp8, h_B_col.data(), bytes_B_fp8, cudaMemcpyHostToDevice));
  }

  cudaStream_t stream = 0;
  if (use_l2pin) {
    try_enable_persisting_l2(bytes_B_fp8 + bytes_scale);
    try_set_stream_access_policy_persisting(stream, d_B_col_fp8, bytes_B_fp8, 1.0f);
    try_set_stream_access_policy_persisting(stream, d_scale_u16, bytes_scale, 1.0f);
  }

  dim3 block(512, 1, 1);
  dim3 grid((N + 63) / 64, (M + 63) / 64, 1);
  constexpr int kKChunk = 32;

  auto run_once = [&] {
    CUDA_CHECK(fp8imma::launch_imma_fp8_jit_v2(
      kKChunk,
      d_A,
      d_B_col_fp8,
      d_scale_u16,
      d_D,
      M,
      N,
      K,
      global_scale,
      stream));
  };

  run_once();
  CUDA_CHECK(cudaDeviceSynchronize());
  GpuTimer t;
  t.begin(stream);
  for (int r = 0; r < repeats; ++r) run_once();
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms = t.end_ms(stream) / (float)repeats;

  double ops = 2.0 * (double)M * (double)N * (double)K;
  double tops = (ops / (ms / 1e3)) / 1e12;
  printf("[imma_fp8_jit_v2] JIT fp8->int8 + imma: %.3f ms/iter  (%.2f TOPS)\n", ms, tops);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B_col_fp8));
  CUDA_CHECK(cudaFree(d_D));
  CUDA_CHECK(cudaFree(d_scale_u16));
  if (use_l2pin) {
    (void)cudaCtxResetPersistingL2Cache();
  }
}

static void run_bench_imma_fp8_jit_v2_i8lut(bool use_l2pin) {
  fp8imma::init_fp8_e4m3_lut();
  constexpr int M = 4096;
  constexpr int N = 4096;
  constexpr int K = 4096;
  constexpr int repeats = 200;
  constexpr float global_scale = 1.0f;

  static_assert((K % 16) == 0, "K must be multiple of 16");

  size_t bytes_A = (size_t)M * K * sizeof(int8_t);
  size_t bytes_B_fp8 = (size_t)K * N * sizeof(uint8_t);
  size_t bytes_D = (size_t)M * N * sizeof(half);
  size_t bytes_scale = (size_t)N * sizeof(uint16_t);

  std::vector<int8_t> h_A((size_t)M * K);
  std::vector<uint8_t> h_B_fp8((size_t)K * N);
  std::vector<uint16_t> h_scale_u16((size_t)N);

  std::mt19937 rng(123);
  std::uniform_int_distribution<int> dist(-127, 127);
  std::uniform_real_distribution<float> sd(0.25f, 4.0f);
  for (size_t i = 0; i < h_A.size(); ++i) h_A[i] = (int8_t)dist(rng);
  for (size_t i = 0; i < h_B_fp8.size(); ++i) h_B_fp8[i] = (uint8_t)(rng() & 0xFF);
  for (int n = 0; n < N; ++n) h_scale_u16[(size_t)n] = f32_to_f16_bits(sd(rng));

  int8_t* d_A = nullptr;
  uint8_t* d_B_col_fp8 = nullptr;
  half* d_D = nullptr;
  uint16_t* d_scale_u16 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
  CUDA_CHECK(cudaMalloc(&d_B_col_fp8, bytes_B_fp8));
  CUDA_CHECK(cudaMalloc(&d_D, bytes_D));
  CUDA_CHECK(cudaMalloc(&d_scale_u16, bytes_scale));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scale_u16, h_scale_u16.data(), bytes_scale, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D, 0, bytes_D));

  {
    std::vector<uint8_t> h_B_col((size_t)K * N);
    for (int r = 0; r < K; ++r) {
      for (int c = 0; c < N; ++c) {
        h_B_col[(size_t)c * K + r] = h_B_fp8[(size_t)r * N + c];
      }
    }
    CUDA_CHECK(cudaMemcpy(d_B_col_fp8, h_B_col.data(), bytes_B_fp8, cudaMemcpyHostToDevice));
  }

  cudaStream_t stream = 0;
  if (use_l2pin) {
    try_enable_persisting_l2(bytes_B_fp8 + bytes_scale);
    try_set_stream_access_policy_persisting(stream, d_B_col_fp8, bytes_B_fp8, 1.0f);
    try_set_stream_access_policy_persisting(stream, d_scale_u16, bytes_scale, 1.0f);
  }

  dim3 block(512, 1, 1);
  dim3 grid((N + 63) / 64, (M + 63) / 64, 1);
  constexpr int kKChunk = 32;

  auto run_once = [&] {
    CUDA_CHECK(fp8imma::launch_imma_fp8_jit_v2_i8lut(
      kKChunk,
      d_A,
      d_B_col_fp8,
      d_scale_u16,
      d_D,
      M,
      N,
      K,
      global_scale,
      stream));
  };

  run_once();
  CUDA_CHECK(cudaDeviceSynchronize());
  GpuTimer t;
  t.begin(stream);
  for (int r = 0; r < repeats; ++r) run_once();
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms = t.end_ms(stream) / (float)repeats;

  double ops = 2.0 * (double)M * (double)N * (double)K;
  double tops = (ops / (ms / 1e3)) / 1e12;
  printf("[imma_fp8_jit_v2_i8lut] JIT fp8->int8 via per-column LUT + imma: %.3f ms/iter  (%.2f TOPS)\n", ms, tops);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B_col_fp8));
  CUDA_CHECK(cudaFree(d_D));
  CUDA_CHECK(cudaFree(d_scale_u16));
  if (use_l2pin) {
    (void)cudaCtxResetPersistingL2Cache();
  }
}

static void run_bench_imma_fp8_jit_v3_act_f16(bool use_l2pin) {
  fp8imma::init_fp8_e4m3_lut();
  constexpr int M = 4096;
  constexpr int N = 4096;
  constexpr int K = 4096;
  constexpr int repeats = 200;
  constexpr float global_scale = 1.0f;

  static_assert((K % 16) == 0, "K must be multiple of 16");

  size_t bytes_A = (size_t)M * K * sizeof(half);
  size_t bytes_B_fp8 = (size_t)K * N * sizeof(uint8_t);
  size_t bytes_D = (size_t)M * N * sizeof(half);
  size_t bytes_scale = (size_t)N * sizeof(uint16_t);

  std::vector<half> h_A((size_t)M * K);
  std::vector<uint8_t> h_B_fp8((size_t)K * N);
  std::vector<uint16_t> h_scale_u16((size_t)N);

  std::mt19937 rng(123);
  std::uniform_int_distribution<int> dist(-127, 127);
  std::uniform_real_distribution<float> sd(0.25f, 4.0f);

  for (size_t i = 0; i < h_A.size(); ++i) {
    int v = dist(rng);
    h_A[i] = __float2half_rn((float)v);
  }
  for (size_t i = 0; i < h_B_fp8.size(); ++i) h_B_fp8[i] = (uint8_t)(rng() & 0xFF);
  for (int n = 0; n < N; ++n) h_scale_u16[(size_t)n] = f32_to_f16_bits(sd(rng));

  half* d_A = nullptr;
  uint8_t* d_B_col_fp8 = nullptr;
  half* d_D = nullptr;
  uint16_t* d_scale_u16 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
  CUDA_CHECK(cudaMalloc(&d_B_col_fp8, bytes_B_fp8));
  CUDA_CHECK(cudaMalloc(&d_D, bytes_D));
  CUDA_CHECK(cudaMalloc(&d_scale_u16, bytes_scale));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scale_u16, h_scale_u16.data(), bytes_scale, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D, 0, bytes_D));

  {
    std::vector<uint8_t> h_B_col((size_t)K * N);
    for (int r = 0; r < K; ++r) {
      for (int c = 0; c < N; ++c) {
        h_B_col[(size_t)c * K + r] = h_B_fp8[(size_t)r * N + c];
      }
    }
    CUDA_CHECK(cudaMemcpy(d_B_col_fp8, h_B_col.data(), bytes_B_fp8, cudaMemcpyHostToDevice));
  }

  cudaStream_t stream = 0;
  if (use_l2pin) {
    try_enable_persisting_l2(bytes_B_fp8 + bytes_scale);
    try_set_stream_access_policy_persisting(stream, d_B_col_fp8, bytes_B_fp8, 1.0f);
    try_set_stream_access_policy_persisting(stream, d_scale_u16, bytes_scale, 1.0f);
  }

  dim3 block(512, 1, 1);
  dim3 grid((N + 63) / 64, (M + 63) / 64, 1);
  constexpr int kKChunk = 32;
  auto run_once = [&] {
    CUDA_CHECK(fp8imma::launch_imma_fp8_actquant_v3_f16(
      kKChunk,
      d_A,
      d_B_col_fp8,
      d_scale_u16,
      d_D,
      M,
      N,
      K,
      global_scale,
      1.0f,
      stream));
  };

  run_once();
  CUDA_CHECK(cudaDeviceSynchronize());
  GpuTimer t;
  t.begin(stream);
  for (int r = 0; r < repeats; ++r) run_once();
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms = t.end_ms(stream) / (float)repeats;

  double ops = 2.0 * (double)M * (double)N * (double)K;
  double tops = (ops / (ms / 1e3)) / 1e12;
  printf("[imma_fp8_jit_v3_act_f16] FP16 A -> INT8 (fused) + FP8->INT8 JIT + imma: %.3f ms/iter  (%.2f TOPS)\n", ms, tops);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B_col_fp8));
  CUDA_CHECK(cudaFree(d_D));
  CUDA_CHECK(cudaFree(d_scale_u16));
  if (use_l2pin) {
    (void)cudaCtxResetPersistingL2Cache();
  }
}

static void run_bench_imma_fp8_jit_v3_act_bf16(bool use_l2pin) {
  fp8imma::init_fp8_e4m3_lut();
  constexpr int M = 4096;
  constexpr int N = 4096;
  constexpr int K = 4096;
  constexpr int repeats = 200;
  constexpr float global_scale = 1.0f;

  static_assert((K % 16) == 0, "K must be multiple of 16");

  size_t bytes_A = (size_t)M * K * sizeof(__nv_bfloat16);
  size_t bytes_B_fp8 = (size_t)K * N * sizeof(uint8_t);
  size_t bytes_D = (size_t)M * N * sizeof(half);
  size_t bytes_scale = (size_t)N * sizeof(uint16_t);

  std::vector<__nv_bfloat16> h_A((size_t)M * K);
  std::vector<uint8_t> h_B_fp8((size_t)K * N);
  std::vector<uint16_t> h_scale_u16((size_t)N);

  std::mt19937 rng(123);
  std::uniform_int_distribution<int> dist(-127, 127);
  std::uniform_real_distribution<float> sd(0.25f, 4.0f);

  for (size_t i = 0; i < h_A.size(); ++i) {
    int v = dist(rng);
    h_A[i] = __float2bfloat16_rn((float)v);
  }
  for (size_t i = 0; i < h_B_fp8.size(); ++i) h_B_fp8[i] = (uint8_t)(rng() & 0xFF);
  for (int n = 0; n < N; ++n) h_scale_u16[(size_t)n] = f32_to_f16_bits(sd(rng));

  __nv_bfloat16* d_A = nullptr;
  uint8_t* d_B_col_fp8 = nullptr;
  half* d_D = nullptr;
  uint16_t* d_scale_u16 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
  CUDA_CHECK(cudaMalloc(&d_B_col_fp8, bytes_B_fp8));
  CUDA_CHECK(cudaMalloc(&d_D, bytes_D));
  CUDA_CHECK(cudaMalloc(&d_scale_u16, bytes_scale));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scale_u16, h_scale_u16.data(), bytes_scale, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D, 0, bytes_D));

  {
    std::vector<uint8_t> h_B_col((size_t)K * N);
    for (int r = 0; r < K; ++r) {
      for (int c = 0; c < N; ++c) {
        h_B_col[(size_t)c * K + r] = h_B_fp8[(size_t)r * N + c];
      }
    }
    CUDA_CHECK(cudaMemcpy(d_B_col_fp8, h_B_col.data(), bytes_B_fp8, cudaMemcpyHostToDevice));
  }

  cudaStream_t stream = 0;
  if (use_l2pin) {
    try_enable_persisting_l2(bytes_B_fp8 + bytes_scale);
    try_set_stream_access_policy_persisting(stream, d_B_col_fp8, bytes_B_fp8, 1.0f);
    try_set_stream_access_policy_persisting(stream, d_scale_u16, bytes_scale, 1.0f);
  }

  dim3 block(512, 1, 1);
  dim3 grid((N + 63) / 64, (M + 63) / 64, 1);
  constexpr int kKChunk = 32;
  auto run_once = [&] {
    CUDA_CHECK(fp8imma::launch_imma_fp8_actquant_v3_bf16(
      kKChunk,
      d_A,
      d_B_col_fp8,
      d_scale_u16,
      d_D,
      M,
      N,
      K,
      global_scale,
      1.0f,
      stream));
  };

  run_once();
  CUDA_CHECK(cudaDeviceSynchronize());
  GpuTimer t;
  t.begin(stream);
  for (int r = 0; r < repeats; ++r) run_once();
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms = t.end_ms(stream) / (float)repeats;

  double ops = 2.0 * (double)M * (double)N * (double)K;
  double tops = (ops / (ms / 1e3)) / 1e12;
  printf("[imma_fp8_jit_v3_act_bf16] BF16 A -> INT8 (fused) + FP8->INT8 JIT + imma: %.3f ms/iter  (%.2f TOPS)\n", ms, tops);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B_col_fp8));
  CUDA_CHECK(cudaFree(d_D));
  CUDA_CHECK(cudaFree(d_scale_u16));
  if (use_l2pin) {
    (void)cudaCtxResetPersistingL2Cache();
  }
}

template <typename AType>
static inline AType host_from_f32(float v);

template <>
inline half host_from_f32<half>(float v) {
  return __float2half_rn(v);
}

template <>
inline __nv_bfloat16 host_from_f32<__nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}

template <int KChunk, typename AType>
static void run_bench_imma_fp8_jit_v4_act_impl(bool use_l2pin, const char* tag) {
  fp8imma::init_fp8_e4m3_lut();
  constexpr int M = 4096;
  constexpr int N = 4096;
  constexpr int K = 4096;
  constexpr int repeats = 200;
  constexpr float global_scale = 1.0f;
  static_assert((K % 16) == 0, "K must be multiple of 16");
  static_assert((KChunk % 16) == 0, "KChunk must be multiple of 16");

  size_t bytes_A = (size_t)M * K * sizeof(AType);
  size_t bytes_B_fp8 = (size_t)K * N * sizeof(uint8_t);
  size_t bytes_D = (size_t)M * N * sizeof(half);
  size_t bytes_scale = (size_t)N * sizeof(uint16_t);

  std::vector<AType> h_A((size_t)M * K);
  std::vector<uint8_t> h_B_fp8((size_t)K * N);
  std::vector<uint16_t> h_scale_u16((size_t)N);

  std::mt19937 rng(123);
  std::uniform_int_distribution<int> dist(-127, 127);
  std::uniform_real_distribution<float> sd(0.25f, 4.0f);
  for (size_t i = 0; i < h_A.size(); ++i) {
    int v = dist(rng);
    h_A[i] = host_from_f32<AType>((float)v);
  }
  for (size_t i = 0; i < h_B_fp8.size(); ++i) h_B_fp8[i] = (uint8_t)(rng() & 0xFF);
  for (int n = 0; n < N; ++n) h_scale_u16[(size_t)n] = f32_to_f16_bits(sd(rng));

  AType* d_A = nullptr;
  uint8_t* d_B_col_fp8 = nullptr;
  half* d_D = nullptr;
  uint16_t* d_scale_u16 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
  CUDA_CHECK(cudaMalloc(&d_B_col_fp8, bytes_B_fp8));
  CUDA_CHECK(cudaMalloc(&d_D, bytes_D));
  CUDA_CHECK(cudaMalloc(&d_scale_u16, bytes_scale));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scale_u16, h_scale_u16.data(), bytes_scale, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D, 0, bytes_D));

  {
    std::vector<uint8_t> h_B_col((size_t)K * N);
    for (int r = 0; r < K; ++r) {
      for (int c = 0; c < N; ++c) {
        h_B_col[(size_t)c * K + r] = h_B_fp8[(size_t)r * N + c];
      }
    }
    CUDA_CHECK(cudaMemcpy(d_B_col_fp8, h_B_col.data(), bytes_B_fp8, cudaMemcpyHostToDevice));
  }

  cudaStream_t stream = 0;
  if (use_l2pin) {
    try_enable_persisting_l2(bytes_B_fp8 + bytes_scale);
    try_set_stream_access_policy_persisting(stream, d_B_col_fp8, bytes_B_fp8, 1.0f);
    try_set_stream_access_policy_persisting(stream, d_scale_u16, bytes_scale, 1.0f);
  }

  dim3 block(512, 1, 1);
  dim3 grid((N + 63) / 64, (M + 63) / 64, 1);
  auto run_once = [&] {
    if constexpr (std::is_same<AType, half>::value) {
      CUDA_CHECK(fp8imma::launch_imma_fp8_actquant_v4_f16(
          KChunk,
          (const half*)d_A,
          d_B_col_fp8,
          d_scale_u16,
          d_D,
          M,
          N,
          K,
          global_scale,
          1.0f,
          stream));
    } else {
      CUDA_CHECK(fp8imma::launch_imma_fp8_actquant_v4_bf16(
          KChunk,
          (const __nv_bfloat16*)d_A,
          d_B_col_fp8,
          d_scale_u16,
          d_D,
          M,
          N,
          K,
          global_scale,
          1.0f,
          stream));
    }
  };

  run_once();
  CUDA_CHECK(cudaDeviceSynchronize());
  GpuTimer t;
  t.begin(stream);
  for (int r = 0; r < repeats; ++r) run_once();
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms = t.end_ms(stream) / (float)repeats;

  double ops = 2.0 * (double)M * (double)N * (double)K;
  double tops = (ops / (ms / 1e3)) / 1e12;
  printf("[%s] %.3f ms/iter  (%.2f TOPS)\n", tag, ms, tops);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B_col_fp8));
  CUDA_CHECK(cudaFree(d_D));
  CUDA_CHECK(cudaFree(d_scale_u16));
  if (use_l2pin) {
    (void)cudaCtxResetPersistingL2Cache();
  }
}

template <int KChunk, typename AType>
static void run_bench_imma_fp8_jit_v4_act_texscale_impl(bool use_l2pin, const char* tag) {
  fp8imma::init_fp8_e4m3_lut();
  constexpr int M = 4096;
  constexpr int N = 4096;
  constexpr int K = 4096;
  constexpr int repeats = 200;
  constexpr float global_scale = 1.0f;
  static_assert((K % 16) == 0, "K must be multiple of 16");
  static_assert((KChunk % 16) == 0, "KChunk must be multiple of 16");

  size_t bytes_A = (size_t)M * K * sizeof(AType);
  size_t bytes_B_fp8 = (size_t)K * N * sizeof(uint8_t);
  size_t bytes_D = (size_t)M * N * sizeof(half);
  size_t bytes_scale = (size_t)N * sizeof(uint16_t);

  std::vector<AType> h_A((size_t)M * K);
  std::vector<uint8_t> h_B_fp8((size_t)K * N);
  std::vector<uint16_t> h_scale_u16((size_t)N);

  std::mt19937 rng(123);
  std::uniform_int_distribution<int> dist(-127, 127);
  std::uniform_real_distribution<float> sd(0.25f, 4.0f);
  for (size_t i = 0; i < h_A.size(); ++i) {
    int v = dist(rng);
    h_A[i] = host_from_f32<AType>((float)v);
  }
  for (size_t i = 0; i < h_B_fp8.size(); ++i) h_B_fp8[i] = (uint8_t)(rng() & 0xFF);
  for (int n = 0; n < N; ++n) h_scale_u16[(size_t)n] = f32_to_f16_bits(sd(rng));

  AType* d_A = nullptr;
  uint8_t* d_B_col_fp8 = nullptr;
  half* d_D = nullptr;
  uint16_t* d_scale_u16 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
  CUDA_CHECK(cudaMalloc(&d_B_col_fp8, bytes_B_fp8));
  CUDA_CHECK(cudaMalloc(&d_D, bytes_D));
  CUDA_CHECK(cudaMalloc(&d_scale_u16, bytes_scale));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scale_u16, h_scale_u16.data(), bytes_scale, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D, 0, bytes_D));

  {
    std::vector<uint8_t> h_B_col((size_t)K * N);
    for (int r = 0; r < K; ++r) {
      for (int c = 0; c < N; ++c) {
        h_B_col[(size_t)c * K + r] = h_B_fp8[(size_t)r * N + c];
      }
    }
    CUDA_CHECK(cudaMemcpy(d_B_col_fp8, h_B_col.data(), bytes_B_fp8, cudaMemcpyHostToDevice));
  }

  cudaStream_t stream = 0;
  if (use_l2pin) {
    try_enable_persisting_l2(bytes_B_fp8 + bytes_scale);
    try_set_stream_access_policy_persisting(stream, d_B_col_fp8, bytes_B_fp8, 1.0f);
    try_set_stream_access_policy_persisting(stream, d_scale_u16, bytes_scale, 1.0f);
  }

  cudaTextureObject_t scale_tex = make_u16_tex_object(d_scale_u16, (size_t)N);

  dim3 block(512, 1, 1);
  dim3 grid((N + 63) / 64, (M + 63) / 64, 1);
  auto run_once = [&] {
    if constexpr (std::is_same<AType, half>::value) {
      CUDA_CHECK(fp8imma::launch_imma_fp8_actquant_v4_texscale_f16(
          KChunk,
          (const half*)d_A,
          d_B_col_fp8,
          scale_tex,
          d_D,
          M,
          N,
          K,
          global_scale,
          1.0f,
          stream));
    } else {
      CUDA_CHECK(fp8imma::launch_imma_fp8_actquant_v4_texscale_bf16(
          KChunk,
          (const __nv_bfloat16*)d_A,
          d_B_col_fp8,
          scale_tex,
          d_D,
          M,
          N,
          K,
          global_scale,
          1.0f,
          stream));
    }
  };

  run_once();
  CUDA_CHECK(cudaDeviceSynchronize());
  GpuTimer t;
  t.begin(stream);
  for (int r = 0; r < repeats; ++r) run_once();
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms = t.end_ms(stream) / (float)repeats;

  double ops = 2.0 * (double)M * (double)N * (double)K;
  double tops = (ops / (ms / 1e3)) / 1e12;
  printf("[%s] %.3f ms/iter  (%.2f TOPS)\n", tag, ms, tops);

  CUDA_CHECK(cudaDestroyTextureObject(scale_tex));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B_col_fp8));
  CUDA_CHECK(cudaFree(d_D));
  CUDA_CHECK(cudaFree(d_scale_u16));
  if (use_l2pin) {
    (void)cudaCtxResetPersistingL2Cache();
  }
}

static void run_bench_imma_fp8_jit_v4_act_f16(bool use_l2pin) {
  run_bench_imma_fp8_jit_v4_act_impl<32, half>(
      use_l2pin,
      "imma_fp8_jit_v4_act_f16 FP16 A cp.async->shmem + shmem->INT8 quant + FP8->INT8 JIT + imma:");
}

static void run_bench_imma_fp8_jit_v4_act_bf16(bool use_l2pin) {
  run_bench_imma_fp8_jit_v4_act_impl<32, __nv_bfloat16>(
      use_l2pin,
      "imma_fp8_jit_v4_act_bf16 BF16 A cp.async->shmem + shmem->INT8 quant + FP8->INT8 JIT + imma:");
}

static void run_bench_imma_fp8_jit_v4_act_f16_texscale(bool use_l2pin) {
  run_bench_imma_fp8_jit_v4_act_texscale_impl<32, half>(
      use_l2pin,
      "imma_fp8_jit_v4_act_f16_texscale v4 + TEX scales:");
}

static void run_bench_imma_fp8_jit_v4_act_bf16_texscale(bool use_l2pin) {
  run_bench_imma_fp8_jit_v4_act_texscale_impl<32, __nv_bfloat16>(
      use_l2pin,
      "imma_fp8_jit_v4_act_bf16_texscale v4 + TEX scales:");
}

static void run_bench_imma_fp8_jit_v4_act_f16_k64(bool use_l2pin) {
  run_bench_imma_fp8_jit_v4_act_impl<64, half>(
      use_l2pin,
      "imma_fp8_jit_v4_act_f16_k64 FP16 A (KChunk=64) cp.async->shmem + shmem->INT8 quant + FP8->INT8 JIT + imma:");
}

static void run_bench_imma_fp8_jit_v4_act_bf16_k64(bool use_l2pin) {
  run_bench_imma_fp8_jit_v4_act_impl<64, __nv_bfloat16>(
      use_l2pin,
      "imma_fp8_jit_v4_act_bf16_k64 BF16 A (KChunk=64) cp.async->shmem + shmem->INT8 quant + FP8->INT8 JIT + imma:");
}

static void run_bench_imma_fp8_jit_v4_act_f16_texscale_k64(bool use_l2pin) {
  run_bench_imma_fp8_jit_v4_act_texscale_impl<64, half>(
      use_l2pin,
      "imma_fp8_jit_v4_act_f16_texscale_k64 (KChunk=64) v4 + TEX scales:");
}

static void run_bench_imma_fp8_jit_v4_act_bf16_texscale_k64(bool use_l2pin) {
  run_bench_imma_fp8_jit_v4_act_texscale_impl<64, __nv_bfloat16>(
      use_l2pin,
      "imma_fp8_jit_v4_act_bf16_texscale_k64 (KChunk=64) v4 + TEX scales:");
}

static void run_bench_imma_int8bfp_fused_v2_texscale(bool use_l2pin) {
  constexpr int M = 1024;
  constexpr int N = 1024;
  constexpr int K = 1024;
  constexpr int repeats = 200;
  constexpr float global_scale = 1.0f;

  size_t bytes_A = (size_t)M * K * sizeof(int8_t);
  size_t bytes_B = (size_t)K * N * sizeof(int8_t);
  size_t bytes_D = (size_t)M * N * sizeof(half);
  size_t bytes_scale = (size_t)N * sizeof(uint16_t);

  std::vector<int8_t> h_A((size_t)M * K);
  std::vector<int8_t> h_B((size_t)K * N);
  std::vector<uint16_t> h_scale_u16((size_t)N);

  std::mt19937 rng(123);
  std::uniform_int_distribution<int> dist(-8, 7);
  std::uniform_real_distribution<float> sd(0.25f, 4.0f);
  for (int i = 0; i < M * K; ++i) h_A[(size_t)i] = (int8_t)dist(rng);
  for (int i = 0; i < K * N; ++i) h_B[(size_t)i] = (int8_t)dist(rng);
  for (int n = 0; n < N; ++n) h_scale_u16[(size_t)n] = f32_to_f16_bits(sd(rng));

  int8_t* d_A = nullptr;
  int8_t* d_B_col = nullptr;
  half* d_D = nullptr;
  uint16_t* d_scale_u16 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
  CUDA_CHECK(cudaMalloc(&d_B_col, bytes_B));
  CUDA_CHECK(cudaMalloc(&d_D, bytes_D));
  CUDA_CHECK(cudaMalloc(&d_scale_u16, bytes_scale));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scale_u16, h_scale_u16.data(), bytes_scale, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D, 0, bytes_D));

  {
    std::vector<int8_t> h_B_col((size_t)K * N);
    for (int r = 0; r < K; ++r) {
      for (int c = 0; c < N; ++c) {
        h_B_col[(size_t)c * K + r] = h_B[(size_t)r * N + c];
      }
    }
    CUDA_CHECK(cudaMemcpy(d_B_col, h_B_col.data(), bytes_B, cudaMemcpyHostToDevice));
  }

  cudaStream_t stream = 0;
  if (use_l2pin) {
    try_enable_persisting_l2(bytes_B + bytes_scale);
    try_set_stream_access_policy_persisting(stream, d_B_col, bytes_B, 1.0f);
    try_set_stream_access_policy_persisting(stream, d_scale_u16, bytes_scale, 1.0f);
  }

  cudaTextureObject_t scale_tex = make_u16_tex_object(d_scale_u16, (size_t)N);

  dim3 block(512, 1, 1);
  dim3 grid((N + 63) / 64, (M + 63) / 64, 1);
  constexpr int kKChunk = 32;
  size_t smem_bytes = (2ull * 64 * (size_t)kKChunk) + (2ull * 64 * (size_t)kKChunk) + (64ull * 64 * sizeof(int32_t)) + (64ull * sizeof(float));
  CUDA_CHECK(cudaFuncSetAttribute(
      imma_gemm_int8_colscale_fp16_colmajor_kernel_v2_texscale<kKChunk>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      (int)smem_bytes));

  auto run_once = [&] {
    imma_gemm_int8_colscale_fp16_colmajor_kernel_v2_texscale<kKChunk><<<grid, block, smem_bytes, stream>>>(
        d_A, d_B_col, scale_tex, d_D, M, N, K, global_scale);
    CUDA_CHECK(cudaGetLastError());
  };

  run_once();
  CUDA_CHECK(cudaDeviceSynchronize());
  GpuTimer t;
  t.begin(stream);
  for (int r = 0; r < repeats; ++r) run_once();
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms = t.end_ms(stream) / (float)repeats;

  double ops = 2.0 * (double)M * (double)N * (double)K;
  double tops = (ops / (ms / 1e3)) / 1e12;
  printf("[imma_int8bfp_fused_v2_texscale] fused + TEX scales: %.3f ms/iter  (%.2f TOPS)\n", ms, tops);

  CUDA_CHECK(cudaDestroyTextureObject(scale_tex));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B_col));
  CUDA_CHECK(cudaFree(d_D));
  CUDA_CHECK(cudaFree(d_scale_u16));
  if (use_l2pin) {
    (void)cudaCtxResetPersistingL2Cache();
  }
}

static void run_bench_imma_int8bfp_fused_v2_autotune(bool use_l2pin) {
  constexpr int M = 1024;
  constexpr int N = 1024;
  constexpr int K = 1024;
  constexpr int repeats = 200;
  constexpr float global_scale = 1.0f;

  size_t bytes_A = (size_t)M * K * sizeof(int8_t);
  size_t bytes_B = (size_t)K * N * sizeof(int8_t);
  size_t bytes_D = (size_t)M * N * sizeof(half);
  size_t bytes_scale = (size_t)N * sizeof(uint16_t);

  std::vector<int8_t> h_A((size_t)M * K);
  std::vector<int8_t> h_B((size_t)K * N);
  std::vector<uint16_t> h_scale_u16((size_t)N);

  std::mt19937 rng(123);
  std::uniform_int_distribution<int> dist(-8, 7);
  std::uniform_real_distribution<float> sd(0.25f, 4.0f);
  for (int i = 0; i < M * K; ++i) h_A[(size_t)i] = (int8_t)dist(rng);
  for (int i = 0; i < K * N; ++i) h_B[(size_t)i] = (int8_t)dist(rng);
  for (int n = 0; n < N; ++n) h_scale_u16[(size_t)n] = f32_to_f16_bits(sd(rng));

  int8_t* d_A = nullptr;
  int8_t* d_B_col = nullptr;
  half* d_D = nullptr;
  uint16_t* d_scale_u16 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
  CUDA_CHECK(cudaMalloc(&d_B_col, bytes_B));
  CUDA_CHECK(cudaMalloc(&d_D, bytes_D));
  CUDA_CHECK(cudaMalloc(&d_scale_u16, bytes_scale));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scale_u16, h_scale_u16.data(), bytes_scale, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D, 0, bytes_D));
  {
    std::vector<int8_t> h_B_col((size_t)K * N);
    for (int r = 0; r < K; ++r) {
      for (int c = 0; c < N; ++c) {
        h_B_col[(size_t)c * K + r] = h_B[(size_t)r * N + c];
      }
    }
    CUDA_CHECK(cudaMemcpy(d_B_col, h_B_col.data(), bytes_B, cudaMemcpyHostToDevice));
  }

  cudaStream_t stream = 0;
  if (use_l2pin) {
    try_enable_persisting_l2(bytes_B + bytes_scale);
    try_set_stream_access_policy_persisting(stream, d_B_col, bytes_B, 1.0f);
    try_set_stream_access_policy_persisting(stream, d_scale_u16, bytes_scale, 1.0f);
  }

  struct Cand { int kchunk; float ms; };
  std::vector<Cand> cands;
  cands.reserve(2);

  dim3 block(512, 1, 1);
  dim3 grid((N + 63) / 64, (M + 63) / 64, 1);

  // KChunk=32
  {
    constexpr int kKChunk = 32;
    size_t smem_bytes = (2ull * 64 * (size_t)kKChunk) + (2ull * 64 * (size_t)kKChunk) + (64ull * 64 * sizeof(int32_t)) + (64ull * sizeof(float));
    CUDA_CHECK(cudaFuncSetAttribute(
        imma_gemm_int8_colscale_fp16_colmajor_kernel_v2<kKChunk>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_bytes));

    constexpr int warm = 5;
    for (int w = 0; w < warm; ++w) {
      imma_gemm_int8_colscale_fp16_colmajor_kernel_v2<kKChunk><<<grid, block, smem_bytes, stream>>>(
          d_A, d_B_col, d_scale_u16, d_D, M, N, K, global_scale);
      CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer t;
    t.begin(stream);
    for (int r = 0; r < repeats; ++r) {
      imma_gemm_int8_colscale_fp16_colmajor_kernel_v2<kKChunk><<<grid, block, smem_bytes, stream>>>(
          d_A, d_B_col, d_scale_u16, d_D, M, N, K, global_scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = t.end_ms(stream) / (float)repeats;
    printf("[imma_int8bfp_fused_v2_autotune] KChunk=32: %.3f ms/iter\n", ms);
    cands.push_back({kKChunk, ms});
  }
  // KChunk=64
  {
    constexpr int kKChunk = 64;
    size_t smem_bytes = (2ull * 64 * (size_t)kKChunk) + (2ull * 64 * (size_t)kKChunk) + (64ull * 64 * sizeof(int32_t)) + (64ull * sizeof(float));
    CUDA_CHECK(cudaFuncSetAttribute(
        imma_gemm_int8_colscale_fp16_colmajor_kernel_v2<kKChunk>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_bytes));

    constexpr int warm = 5;
    for (int w = 0; w < warm; ++w) {
      imma_gemm_int8_colscale_fp16_colmajor_kernel_v2<kKChunk><<<grid, block, smem_bytes, stream>>>(
          d_A, d_B_col, d_scale_u16, d_D, M, N, K, global_scale);
      CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer t;
    t.begin(stream);
    for (int r = 0; r < repeats; ++r) {
      imma_gemm_int8_colscale_fp16_colmajor_kernel_v2<kKChunk><<<grid, block, smem_bytes, stream>>>(
          d_A, d_B_col, d_scale_u16, d_D, M, N, K, global_scale);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = t.end_ms(stream) / (float)repeats;
    printf("[imma_int8bfp_fused_v2_autotune] KChunk=64: %.3f ms/iter\n", ms);
    cands.push_back({kKChunk, ms});
  }

  std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b){ return a.ms < b.ms; });
  double ops = 2.0 * (double)M * (double)N * (double)K;
  double tops = (ops / (cands[0].ms / 1e3)) / 1e12;
  printf("[imma_int8bfp_fused_v2_autotune] best: KChunk=%d  %.3f ms/iter  (%.2f TOPS)\n", cands[0].kchunk, cands[0].ms, tops);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B_col));
  CUDA_CHECK(cudaFree(d_D));
  CUDA_CHECK(cudaFree(d_scale_u16));
  if (use_l2pin) {
    (void)cudaCtxResetPersistingL2Cache();
  }
}

// ------------------------ BENCH: custom IMMA fused with autotuned tile config ------------------------
static void run_bench_imma_int8bfp_fused_autotune(bool use_l2pin) {
  constexpr int M = 1024;
  constexpr int N = 1024;
  constexpr int K = 1024;
  constexpr int repeats = 200;
  constexpr float global_scale = 1.0f;

  static_assert((K % 16) == 0, "K must be multiple of 16");

  size_t bytes_A = (size_t)M * K * sizeof(int8_t);
  size_t bytes_B = (size_t)K * N * sizeof(int8_t);
  size_t bytes_D = (size_t)M * N * sizeof(half);
  size_t bytes_scale = (size_t)N * sizeof(uint16_t);

  std::vector<int8_t> h_A((size_t)M * K);
  std::vector<int8_t> h_B((size_t)K * N);
  std::vector<uint16_t> h_scale_u16((size_t)N);

  std::mt19937 rng(123);
  std::uniform_int_distribution<int> dist(-8, 7);
  std::uniform_real_distribution<float> sd(0.25f, 4.0f);
  for (int i = 0; i < M * K; ++i) h_A[(size_t)i] = (int8_t)dist(rng);
  for (int i = 0; i < K * N; ++i) h_B[(size_t)i] = (int8_t)dist(rng);
  for (int n = 0; n < N; ++n) h_scale_u16[(size_t)n] = f32_to_f16_bits(sd(rng));

  int8_t* d_A = nullptr;
  int8_t* d_B_col = nullptr;
  half* d_D = nullptr;
  uint16_t* d_scale_u16 = nullptr;

  CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
  CUDA_CHECK(cudaMalloc(&d_B_col, bytes_B));
  CUDA_CHECK(cudaMalloc(&d_D, bytes_D));
  CUDA_CHECK(cudaMalloc(&d_scale_u16, bytes_scale));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scale_u16, h_scale_u16.data(), bytes_scale, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D, 0, bytes_D));

  // Upload B as col-major on device.
  {
    std::vector<int8_t> h_B_col((size_t)K * N);
    for (int r = 0; r < K; ++r) {
      for (int c = 0; c < N; ++c) {
        h_B_col[(size_t)c * K + r] = h_B[(size_t)r * N + c];
      }
    }
    CUDA_CHECK(cudaMemcpy(d_B_col, h_B_col.data(), bytes_B, cudaMemcpyHostToDevice));
  }

  cudaStream_t stream = 0;
  if (use_l2pin) {
    try_enable_persisting_l2(bytes_B + bytes_scale);
    try_set_stream_access_policy_persisting(stream, d_B_col, bytes_B, 1.0f);
    try_set_stream_access_policy_persisting(stream, d_scale_u16, bytes_scale, 1.0f);
  }

  struct Candidate {
    int bm;
    int bn;
    const char* name;
  };
  // Keep warps <= 8 (<=256 threads). All are multiples of 16.
  static const Candidate cands[] = {
      {1, 2, "1x2"},
      {2, 1, "2x1"},
      {2, 2, "2x2"},
      {1, 4, "1x4"},
      {4, 1, "4x1"},
      {2, 4, "2x4"},
      {4, 2, "4x2"},
  };

  auto smem_for = [&](int bm, int bn) -> size_t {
    int warps = bm * bn;
    return (size_t)warps * 16 * 16 * sizeof(int32_t) + (size_t)warps * 16 * sizeof(float);
  };

  auto grid_for = [&](int bm, int bn) -> dim3 {
    return dim3((N + (bn * 16 - 1)) / (bn * 16), (M + (bm * 16 - 1)) / (bm * 16), 1);
  };

  auto block_for = [&](int bm, int bn) -> dim3 {
    return dim3(32 * bm * bn, 1, 1);
  };

  auto time_candidate = [&](int bm, int bn) -> float {
    dim3 grid = grid_for(bm, bn);
    dim3 block = block_for(bm, bn);
    size_t smem = smem_for(bm, bn);

    auto launch = [&] {
      if (bm == 1 && bn == 2) {
        imma_gemm_int8_colscale_fp16_colmajor_kernel<1, 2><<<grid, block, smem, stream>>>(d_A, d_B_col, d_scale_u16, d_D, M, N, K, global_scale);
      } else if (bm == 2 && bn == 1) {
        imma_gemm_int8_colscale_fp16_colmajor_kernel<2, 1><<<grid, block, smem, stream>>>(d_A, d_B_col, d_scale_u16, d_D, M, N, K, global_scale);
      } else if (bm == 2 && bn == 2) {
        imma_gemm_int8_colscale_fp16_colmajor_kernel<2, 2><<<grid, block, smem, stream>>>(d_A, d_B_col, d_scale_u16, d_D, M, N, K, global_scale);
      } else if (bm == 1 && bn == 4) {
        imma_gemm_int8_colscale_fp16_colmajor_kernel<1, 4><<<grid, block, smem, stream>>>(d_A, d_B_col, d_scale_u16, d_D, M, N, K, global_scale);
      } else if (bm == 4 && bn == 1) {
        imma_gemm_int8_colscale_fp16_colmajor_kernel<4, 1><<<grid, block, smem, stream>>>(d_A, d_B_col, d_scale_u16, d_D, M, N, K, global_scale);
      } else if (bm == 2 && bn == 4) {
        imma_gemm_int8_colscale_fp16_colmajor_kernel<2, 4><<<grid, block, smem, stream>>>(d_A, d_B_col, d_scale_u16, d_D, M, N, K, global_scale);
      } else if (bm == 4 && bn == 2) {
        imma_gemm_int8_colscale_fp16_colmajor_kernel<4, 2><<<grid, block, smem, stream>>>(d_A, d_B_col, d_scale_u16, d_D, M, N, K, global_scale);
      }
    };

    // Warmup
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer t;
    t.begin(stream);
    for (int r = 0; r < repeats; ++r) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    return t.end_ms(stream) / (float)repeats;
  };

  float best_ms = 1e9f;
  const char* best_name = "";
  int best_bm = 0;
  int best_bn = 0;
  for (auto &c : cands) {
    if (c.bm * c.bn > 8) continue;
    float ms = time_candidate(c.bm, c.bn);
    if (ms < best_ms) {
      best_ms = ms;
      best_name = c.name;
      best_bm = c.bm;
      best_bn = c.bn;
    }
  }

  double ops = 2.0 * (double)M * (double)N * (double)K;
  double tops = (ops / (best_ms / 1e3)) / 1e12;
  printf("[imma_int8bfp_fused_autotune] best=%s (BM=%d BN=%d): %.3f ms/iter  (%.2f TOPS)\n", best_name, best_bm, best_bn, best_ms, tops);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B_col));
  CUDA_CHECK(cudaFree(d_D));
  CUDA_CHECK(cudaFree(d_scale_u16));
  if (use_l2pin) {
    (void)cudaCtxResetPersistingL2Cache();
  }
}

// ------------------------ BENCH: FP8 stored -> Offline Convert -> IMMA ------------------------
static void run_bench_fp8wgt_imma_offline() {
  printf("[fp8wgt_imma_offline] Simulating offline FP8->INT8+Scale conversion followed by IMMA GEMM\n");
  constexpr int M = 4096;
  constexpr int N = 4096;
  constexpr int K = 4096;
  constexpr int repeats = 200;
  constexpr float global_scale = 1.0f;

  size_t bytes_B_fp8 = (size_t)K * N * sizeof(uint8_t);
  
  // Host Data Generation (FP8)
  std::vector<uint8_t> h_B_fp8((size_t)K * N);
  std::vector<int8_t> h_A((size_t)M * K);
  std::mt19937 rng(123);
  std::uniform_int_distribution<int> bd(0, 255);
  std::uniform_int_distribution<int> ad(-127, 127);
  for (int i = 0; i < K * N; ++i) h_B_fp8[i] = (uint8_t)bd(rng);
  for (int i = 0; i < M * K; ++i) h_A[i] = (int8_t)ad(rng);

  // Offline Conversion (FP8 -> INT8 + Scale)
  std::vector<int8_t> h_B_int8_col((size_t)K * N);
  std::vector<uint16_t> h_scale_u16((size_t)N);
  
  for (int c = 0; c < N; ++c) {
     float max_val = 0.0f;
     std::vector<float> col_vals(K);
     for (int r = 0; r < K; ++r) {
         uint8_t val8 = h_B_fp8[r * N + c];
         float val = fp8_e4m3_to_f32(val8);
         col_vals[r] = val;
         if (fabsf(val) > max_val) max_val = fabsf(val);
     }
     
     // Scale to map max_val to 127
     float scale = max_val / 127.0f;
     if (scale < 1e-8f) scale = 1.0f;
     float inv_scale = 1.0f / scale;
     
     h_scale_u16[c] = f32_to_f16_bits(scale);

     for (int r = 0; r < K; ++r) {
         float v = col_vals[r] * inv_scale;
         int i = (int)roundf(v);
         if (i < -127) i = -127;
         if (i > 127) i = 127;
         h_B_int8_col[c * K + r] = (int8_t)i;
     }
  }

  // Device Alloc
  size_t bytes_A = (size_t)M * K * sizeof(int8_t);
  size_t bytes_B = (size_t)K * N * sizeof(int8_t);
  size_t bytes_D = (size_t)M * N * sizeof(half);
  size_t bytes_scale = (size_t)N * sizeof(uint16_t);

  int8_t* d_A = nullptr;
  int8_t* d_B_col = nullptr;
  half* d_D = nullptr;
  uint16_t* d_scale_u16 = nullptr;

  CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
  CUDA_CHECK(cudaMalloc(&d_B_col, bytes_B));
  CUDA_CHECK(cudaMalloc(&d_D, bytes_D));
  CUDA_CHECK(cudaMalloc(&d_scale_u16, bytes_scale));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B_col, h_B_int8_col.data(), bytes_B, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scale_u16, h_scale_u16.data(), bytes_scale, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_D, 0, bytes_D));

  cudaStream_t stream = 0;
  dim3 block(512, 1, 1);
  dim3 grid((N + 63) / 64, (M + 63) / 64, 1);
  constexpr int kKChunk = 32;
  size_t smem_bytes = (2ull * 64 * (size_t)kKChunk) + (2ull * 64 * (size_t)kKChunk) + (64ull * 64 * sizeof(int32_t)) + (64ull * sizeof(float));

  CUDA_CHECK(cudaFuncSetAttribute(
      imma_gemm_int8_colscale_fp16_colmajor_kernel_v2<kKChunk>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      (int)smem_bytes));

  auto run_once = [&] {
    imma_gemm_int8_colscale_fp16_colmajor_kernel_v2<kKChunk><<<grid, block, smem_bytes, stream>>>(
        d_A, d_B_col, d_scale_u16, d_D, M, N, K, global_scale);
    CUDA_CHECK(cudaGetLastError());
  };

  run_once();
  CUDA_CHECK(cudaDeviceSynchronize());
  GpuTimer t;
  t.begin(stream);
  for (int r = 0; r < repeats; ++r) run_once();
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms = t.end_ms(stream) / (float)repeats;

  double ops = 2.0 * (double)M * (double)N * (double)K;
  double tops = (ops / (ms / 1e3)) / 1e12;
  printf("[fp8wgt_imma_offline] Offline-Int8-Converted GEMM: %.3f ms/iter  (%.2f TOPS)\n", ms, tops);
  
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B_col));
  CUDA_CHECK(cudaFree(d_D));
  CUDA_CHECK(cudaFree(d_scale_u16));
}

