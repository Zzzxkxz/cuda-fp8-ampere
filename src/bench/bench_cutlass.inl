// ------------------------ BENCH: CUTLASS FP16 tensor-op GEMM ------------------------
static void run_bench_cutlass_f16() {
  // Purpose: provide a CUTLASS-style GEMM baseline without WMMA.
  constexpr int M = 1024;
  constexpr int N = 1024;
  constexpr int K = 1024;
  constexpr int repeats = 200;

  size_t bytes_A = (size_t)M * K * sizeof(cutlass::half_t);
  size_t bytes_B = (size_t)K * N * sizeof(cutlass::half_t);
  size_t bytes_C = (size_t)M * N * sizeof(float);

  std::vector<cutlass::half_t> h_A((size_t)M * K);
  std::vector<cutlass::half_t> h_Bcol((size_t)K * N);

  std::mt19937 rng(123);
  std::normal_distribution<float> nd(0.0f, 0.5f);
  for (int i = 0; i < M * K; ++i) {
    h_A[(size_t)i] = cutlass::half_t(nd(rng));
  }
  for (int c = 0; c < N; ++c) {
    for (int r = 0; r < K; ++r) {
      h_Bcol[(size_t)c * K + r] = cutlass::half_t(nd(rng));
    }
  }

  cutlass::half_t* d_A = nullptr;
  cutlass::half_t* d_Bcol = nullptr;
  float* d_C = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
  CUDA_CHECK(cudaMalloc(&d_Bcol, bytes_B));
  CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Bcol, h_Bcol.data(), bytes_B, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));

    // Use an explicit, smaller tile to avoid excessive dynamic shared memory.
    using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor,
      cutlass::half_t, cutlass::layout::ColumnMajor,
      float, cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
        // This CUTLASS snapshot doesn't provide DefaultGemmConfiguration for Sm86.
        // Sm80 kernels are valid on Ampere sm_86 devices.
        cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<32, 32, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>>;

  Gemm gemm;
  float alpha = 1.0f;
  float beta = 0.0f;

  cutlass::TensorRef<cutlass::half_t const, cutlass::layout::RowMajor> refA(
      d_A, cutlass::layout::RowMajor(K));
  cutlass::TensorRef<cutlass::half_t const, cutlass::layout::ColumnMajor> refB(
      d_Bcol, cutlass::layout::ColumnMajor(K));
  cutlass::TensorRef<float const, cutlass::layout::RowMajor> refC(
      d_C, cutlass::layout::RowMajor(N));
  cutlass::TensorRef<float, cutlass::layout::RowMajor> refD(
      d_C, cutlass::layout::RowMajor(N));

  typename Gemm::EpilogueOutputOp::Params epilogue(alpha, beta);
  typename Gemm::Arguments args(
      {M, N, K},
      refA, refB, refC, refD,
      epilogue);

  cutlass::Status can = Gemm::can_implement(args);
  if (can != cutlass::Status::kSuccess) {
    fprintf(stderr, "[cutlass_f16] can_implement failed: %s (%d)\n", cutlass::cutlassGetStatusString(can), (int)can);
    std::exit(1);
  }

  size_t workspace_bytes = Gemm::get_workspace_size(args);
  void* workspace = nullptr;
  if (workspace_bytes) CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));

  // Warmup
  cutlass::Status st = gemm(args, workspace, 0);
  if (st != cutlass::Status::kSuccess) {
    cudaError_t last = cudaGetLastError();
    fprintf(stderr,
            "[cutlass_f16] run failed: %s (%d); cudaGetLastError=%s\n",
            cutlass::cutlassGetStatusString(st), (int)st, cudaGetErrorString(last));
    std::exit(1);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  GpuTimer t;
  t.begin();
  for (int r = 0; r < repeats; ++r) {
    st = gemm(args, workspace, 0);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms_avg = t.end_ms() / (float)repeats;

  double flops = 2.0 * (double)M * (double)N * (double)K;
  double tflops = (flops / (ms_avg / 1e3)) / 1e12;
  printf("[cutlass_f16] cutlass_fp16_tensorop: %.3f ms/iter  (%.2f TFLOP/s)\n", ms_avg, tflops);

  if (workspace) cudaFree(workspace);
  cudaFree(d_A);
  cudaFree(d_Bcol);
  cudaFree(d_C);
}

// ------------------------ BENCH: FP16 A + FP8(E4M3) B via CUTLASS (decode->GEMM) ------------------------
static void run_bench_cutlass_fp8wgt(bool use_l2pin) {
  upload_fp8_lut();

  constexpr int M = 1024;
  constexpr int N = 1024;
  constexpr int K = 1024;
  constexpr int repeats = 200;
  constexpr float scale_b = 1.0f;
  constexpr int block_k = 32;
  static_assert((K % block_k) == 0, "K must be divisible by block_k");

  size_t bytes_Ah = (size_t)M * K * sizeof(cutlass::half_t);
  size_t bytes_B8 = (size_t)K * N;
  size_t bytes_Bh = (size_t)K * N * sizeof(cutlass::half_t);
  size_t bytes_C = (size_t)M * N * sizeof(float);

  // Host data.
  std::vector<cutlass::half_t> h_Ah((size_t)M * K);
  std::vector<uint8_t> h_B8((size_t)K * N);
  std::vector<cutlass::half_t> h_Bh_col((size_t)K * N);

  std::mt19937 rng(123);
  std::normal_distribution<float> nd(0.0f, 0.5f);
  std::uniform_int_distribution<int> bd(0, 255);
  for (int i = 0; i < M * K; ++i) h_Ah[(size_t)i] = cutlass::half_t(nd(rng));
  for (int r = 0; r < K; ++r) {
    for (int c = 0; c < N; ++c) {
      uint8_t v = (uint8_t)bd(rng);
      h_B8[(size_t)r * N + c] = v;
      float f = fp8_e4m3_to_f32(v) * scale_b;
      h_Bh_col[(size_t)c * K + r] = cutlass::half_t(f);
    }
  }

  cutlass::half_t* d_Ah = nullptr;
  uint8_t* d_B8_col = nullptr;
  cutlass::half_t* d_Bh_col = nullptr;
  cutlass::half_t* d_Bh_upcast = nullptr;
  float* d_C = nullptr;
  half* d_scales16 = nullptr;
  uint8_t* d_scales8 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_Ah, bytes_Ah));
  CUDA_CHECK(cudaMalloc(&d_B8_col, bytes_B8));
  CUDA_CHECK(cudaMalloc(&d_Bh_col, bytes_Bh));
  CUDA_CHECK(cudaMalloc(&d_Bh_upcast, bytes_Bh));
  CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
  int n_scales = (K / block_k) * N;
  CUDA_CHECK(cudaMalloc(&d_scales16, (size_t)n_scales * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_scales8, (size_t)n_scales * sizeof(uint8_t)));

  CUDA_CHECK(cudaMemcpy(d_Ah, h_Ah.data(), bytes_Ah, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Bh_col, h_Bh_col.data(), bytes_Bh, cudaMemcpyHostToDevice));

  // Upload B8 as col-major.
  {
    std::vector<uint8_t> h_B8_col((size_t)K * N);
    for (int r = 0; r < K; ++r) {
      for (int c = 0; c < N; ++c) {
        h_B8_col[(size_t)c * K + r] = h_B8[(size_t)r * N + c];
      }
    }
    CUDA_CHECK(cudaMemcpy(d_B8_col, h_B8_col.data(), bytes_B8, cudaMemcpyHostToDevice));
  }

  // Prepare blockwise scales (one scale per (k_block, n)).
  {
    std::vector<half> h_scales16((size_t)n_scales);
    std::uniform_real_distribution<float> sd(0.25f, 4.0f);
    for (int i = 0; i < n_scales; ++i) h_scales16[(size_t)i] = __float2half(sd(rng));
    CUDA_CHECK(cudaMemcpy(d_scales16, h_scales16.data(), (size_t)n_scales * sizeof(half), cudaMemcpyHostToDevice));
    // Quantize the scales to FP8 on device (approx encoder), inv_scale=1.
    quantize_fp16_to_fp8e4m3_kernel<<<(n_scales + 255) / 256, 256>>>(d_scales16, d_scales8, n_scales, 1.0f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  cudaStream_t stream = 0;
  if (use_l2pin) {
    // B8 and per-block scales are relatively small (~1MiB + ~64KiB). Try to keep them in persisting L2.
    try_enable_persisting_l2(bytes_B8 + (size_t)n_scales * (sizeof(half) + sizeof(uint8_t)));
    try_set_stream_access_policy_persisting(stream, d_B8_col, bytes_B8, 1.0f);
    try_set_stream_access_policy_persisting(stream, d_scales16, (size_t)n_scales * sizeof(half), 1.0f);
    try_set_stream_access_policy_persisting(stream, d_scales8, (size_t)n_scales * sizeof(uint8_t), 1.0f);
  }

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor,
      cutlass::half_t, cutlass::layout::ColumnMajor,
      float, cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<32, 32, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>>;

  Gemm gemm;
  float alpha = 1.0f;
  float beta = 0.0f;

  auto make_args = [&](cutlass::half_t const* A, cutlass::half_t const* Bcol) {
    cutlass::TensorRef<cutlass::half_t const, cutlass::layout::RowMajor> refA(
        A, cutlass::layout::RowMajor(K));
    cutlass::TensorRef<cutlass::half_t const, cutlass::layout::ColumnMajor> refB(
        Bcol, cutlass::layout::ColumnMajor(K));
    cutlass::TensorRef<float const, cutlass::layout::RowMajor> refC(
        d_C, cutlass::layout::RowMajor(N));
    cutlass::TensorRef<float, cutlass::layout::RowMajor> refD(
        d_C, cutlass::layout::RowMajor(N));
    typename Gemm::EpilogueOutputOp::Params epilogue(alpha, beta);
    return typename Gemm::Arguments({M, N, K}, refA, refB, refC, refD, epilogue);
  };

  auto time_avg_ms = [&](auto&& launch) {
    GpuTimer t;
    t.begin();
    for (int r = 0; r < repeats; ++r) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return t.end_ms() / (float)repeats;
  };

  auto report = [&](const char* name, float ms_avg) {
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = (flops / (ms_avg / 1e3)) / 1e12;
    printf("[cutlass_fp8wgt] %s: %.3f ms/iter  (%.2f TFLOP/s)\n", name, ms_avg, tflops);
  };

  // Workspace sized for this problem.
  auto args_fp16 = make_args(d_Ah, d_Bh_col);
  cutlass::Status can = Gemm::can_implement(args_fp16);
  if (can != cutlass::Status::kSuccess) {
    fprintf(stderr, "[cutlass_fp8wgt] can_implement failed: %s (%d)\n", cutlass::cutlassGetStatusString(can), (int)can);
    std::exit(1);
  }
  size_t workspace_bytes = Gemm::get_workspace_size(args_fp16);
  void* workspace = nullptr;
  if (workspace_bytes) CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));

  // Warmup: pure FP16.
  {
    cutlass::Status st = gemm(args_fp16, workspace, 0);
    if (st != cutlass::Status::kSuccess) {
      cudaError_t last = cudaGetLastError();
      fprintf(stderr,
              "[cutlass_fp8wgt] warmup failed: %s (%d); cudaGetLastError=%s\n",
              cutlass::cutlassGetStatusString(st), (int)st, cudaGetErrorString(last));
      std::exit(1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  report("cutlass_fp16_tensorop", time_avg_ms([&] {
    auto args = make_args(d_Ah, d_Bh_col);
    (void)gemm(args, workspace, stream);
  }));

  // Naive upcast-per-iter + CUTLASS.
  report("fp8->fp16 upcast (global) + cutlass_tensorop", time_avg_ms([&] {
    dequantize_fp8e4m3_to_fp16_kernel_vec4<<<((K * N / 4) + 255) / 256, 256>>>(d_B8_col, (half*)d_Bh_upcast, K * N, scale_b);
    auto args = make_args(d_Ah, d_Bh_upcast);
    (void)gemm(args, workspace, stream);
  }));

  // Blockwise scale (FP16 scales) + CUTLASS.
  report("fp8->fp16 blockscale(fp16) + cutlass_tensorop", time_avg_ms([&] {
    dequantize_fp8e4m3_to_fp16_blockscale_f16_vec4_kernel<<<((K * N / 4) + 255) / 256, 256>>>(
        d_B8_col, (half*)d_Bh_upcast, K, N, d_scales16, block_k);
    auto args = make_args(d_Ah, d_Bh_upcast);
    (void)gemm(args, workspace, stream);
  }));

  // Blockwise scale (FP8 scales) + CUTLASS.
  report("fp8->fp16 blockscale(fp8) + cutlass_tensorop", time_avg_ms([&] {
    dequantize_fp8e4m3_to_fp16_blockscale_fp8_kernel<<<((K * N) + 255) / 256, 256>>>(
        d_B8_col, (half*)d_Bh_upcast, K, N, d_scales8, block_k);
    auto args = make_args(d_Ah, d_Bh_upcast);
    (void)gemm(args, workspace, stream);
  }));

  // Practical inference baseline: decode once (weights reuse) + CUTLASS.
  {
    dequantize_fp8e4m3_to_fp16_kernel_vec4<<<((K * N / 4) + 255) / 256, 256>>>(d_B8_col, (half*)d_Bh_upcast, K * N, scale_b);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    report("fp8->fp16 upcast_once (weights reuse) + cutlass_tensorop", time_avg_ms([&] {
      auto args = make_args(d_Ah, d_Bh_upcast);
      (void)gemm(args, workspace, stream);
    }));
  }

  if (workspace) cudaFree(workspace);
  cudaFree(d_Ah);
  cudaFree(d_B8_col);
  cudaFree(d_Bh_col);
  cudaFree(d_Bh_upcast);
  cudaFree(d_C);
  cudaFree(d_scales16);
  cudaFree(d_scales8);

  if (use_l2pin) {
    (void)cudaCtxResetPersistingL2Cache();
  }
}

// ------------------------ BENCH: FP16 A + INT8 B (weight-only) via CUTLASS (dequant->GEMM) ------------------------
static void run_bench_cutlass_int8wgt(bool use_l2pin) {
  constexpr int M = 1024;
  constexpr int N = 1024;
  constexpr int K = 1024;
  constexpr int repeats = 200;
  constexpr float scale_b = 1.0f;
  constexpr int block_k = 32;
  static_assert((K % block_k) == 0, "K must be divisible by block_k");

  size_t bytes_Ah = (size_t)M * K * sizeof(cutlass::half_t);
  size_t bytes_B8 = (size_t)K * N * sizeof(int8_t);
  size_t bytes_Bh = (size_t)K * N * sizeof(cutlass::half_t);
  size_t bytes_C = (size_t)M * N * sizeof(float);

  std::vector<cutlass::half_t> h_Ah((size_t)M * K);
  std::vector<int8_t> h_B8_row((size_t)K * N);

  std::mt19937 rng(123);
  std::normal_distribution<float> nd(0.0f, 0.5f);
  std::uniform_int_distribution<int> id(-8, 7);
  for (int i = 0; i < M * K; ++i) h_Ah[(size_t)i] = cutlass::half_t(nd(rng));
  for (int i = 0; i < K * N; ++i) h_B8_row[(size_t)i] = (int8_t)id(rng);

  cutlass::half_t* d_Ah = nullptr;
  int8_t* d_B8_col = nullptr;
  cutlass::half_t* d_Bh_upcast = nullptr;
  float* d_C = nullptr;
  half* d_scales16 = nullptr;

  CUDA_CHECK(cudaMalloc(&d_Ah, bytes_Ah));
  CUDA_CHECK(cudaMalloc(&d_B8_col, bytes_B8));
  CUDA_CHECK(cudaMalloc(&d_Bh_upcast, bytes_Bh));
  CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

  CUDA_CHECK(cudaMemcpy(d_Ah, h_Ah.data(), bytes_Ah, cudaMemcpyHostToDevice));

  // Upload B8 as col-major.
  {
    std::vector<int8_t> h_B8_col((size_t)K * N);
    for (int r = 0; r < K; ++r) {
      for (int c = 0; c < N; ++c) {
        h_B8_col[(size_t)c * K + r] = h_B8_row[(size_t)r * N + c];
      }
    }
    CUDA_CHECK(cudaMemcpy(d_B8_col, h_B8_col.data(), bytes_B8, cudaMemcpyHostToDevice));
  }

  int n_scales = (K / block_k) * N;
  CUDA_CHECK(cudaMalloc(&d_scales16, (size_t)n_scales * sizeof(half)));
  {
    std::uniform_real_distribution<float> sd(0.25f, 4.0f);
    std::vector<half> h_scales16((size_t)n_scales);
    for (int i = 0; i < n_scales; ++i) h_scales16[(size_t)i] = __float2half(sd(rng));
    CUDA_CHECK(cudaMemcpy(d_scales16, h_scales16.data(), (size_t)n_scales * sizeof(half), cudaMemcpyHostToDevice));
  }

  cudaStream_t stream = 0;
  if (use_l2pin) {
    try_enable_persisting_l2(bytes_B8 + (size_t)n_scales * sizeof(half));
    try_set_stream_access_policy_persisting(stream, d_B8_col, bytes_B8, 1.0f);
    try_set_stream_access_policy_persisting(stream, d_scales16, (size_t)n_scales * sizeof(half), 1.0f);
  }

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor,
      cutlass::half_t, cutlass::layout::ColumnMajor,
      float, cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<32, 32, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>>;

  Gemm gemm;
  float alpha = 1.0f;
  float beta = 0.0f;

  auto make_args = [&](cutlass::half_t const* A, cutlass::half_t const* Bcol) {
    cutlass::TensorRef<cutlass::half_t const, cutlass::layout::RowMajor> refA(
        A, cutlass::layout::RowMajor(K));
    cutlass::TensorRef<cutlass::half_t const, cutlass::layout::ColumnMajor> refB(
        Bcol, cutlass::layout::ColumnMajor(K));
    cutlass::TensorRef<float const, cutlass::layout::RowMajor> refC(
        d_C, cutlass::layout::RowMajor(N));
    cutlass::TensorRef<float, cutlass::layout::RowMajor> refD(
        d_C, cutlass::layout::RowMajor(N));
    typename Gemm::EpilogueOutputOp::Params epilogue(alpha, beta);
    return typename Gemm::Arguments({M, N, K}, refA, refB, refC, refD, epilogue);
  };

  auto time_avg_ms = [&](auto&& launch) {
    GpuTimer t;
    t.begin();
    for (int r = 0; r < repeats; ++r) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return t.end_ms() / (float)repeats;
  };

  auto report = [&](const char* name, float ms_avg) {
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = (flops / (ms_avg / 1e3)) / 1e12;
    printf("[cutlass_int8wgt] %s: %.3f ms/iter  (%.2f TFLOP/s)\n", name, ms_avg, tflops);
  };

  // Workspace.
  // Warmup uses dequant + GEMM (since B is int8).
  {
    dequantize_int8_to_fp16_vec4_kernel<<<((K * N / 4) + 255) / 256, 256>>>(d_B8_col, (half*)d_Bh_upcast, K * N, scale_b);
    auto args = make_args(d_Ah, d_Bh_upcast);
    cutlass::Status can = Gemm::can_implement(args);
    if (can != cutlass::Status::kSuccess) {
      fprintf(stderr, "[cutlass_int8wgt] can_implement failed: %s (%d)\n", cutlass::cutlassGetStatusString(can), (int)can);
      std::exit(1);
    }
    size_t workspace_bytes = Gemm::get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_bytes) CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));
    cutlass::Status st = gemm(args, workspace, stream);
    if (st != cutlass::Status::kSuccess) {
      cudaError_t last = cudaGetLastError();
      fprintf(stderr,
              "[cutlass_int8wgt] warmup failed: %s (%d); cudaGetLastError=%s\n",
              cutlass::cutlassGetStatusString(st), (int)st, cudaGetErrorString(last));
      std::exit(1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    report("int8->fp16 upcast (global) + cutlass_tensorop", time_avg_ms([&] {
      dequantize_int8_to_fp16_vec4_kernel<<<((K * N / 4) + 255) / 256, 256>>>(d_B8_col, (half*)d_Bh_upcast, K * N, scale_b);
      auto a = make_args(d_Ah, d_Bh_upcast);
      (void)gemm(a, workspace, stream);
    }));

    report("int8->fp16 blockscale(fp16) + cutlass_tensorop", time_avg_ms([&] {
      dequantize_int8_to_fp16_blockscale_f16_vec4_kernel<<<((K * N / 4) + 255) / 256, 256>>>(
          d_B8_col, (half*)d_Bh_upcast, K, N, d_scales16, block_k);
      auto a = make_args(d_Ah, d_Bh_upcast);
      (void)gemm(a, workspace, stream);
    }));

    // Decode once baseline.
    dequantize_int8_to_fp16_vec4_kernel<<<((K * N / 4) + 255) / 256, 256>>>(d_B8_col, (half*)d_Bh_upcast, K * N, scale_b);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    report("int8->fp16 upcast_once (weights reuse) + cutlass_tensorop", time_avg_ms([&] {
      auto a = make_args(d_Ah, d_Bh_upcast);
      (void)gemm(a, workspace, stream);
    }));

    if (workspace) cudaFree(workspace);
  }

  if (use_l2pin) {
    (void)cudaCtxResetPersistingL2Cache();
  }
  cudaFree(d_Ah);
  cudaFree(d_B8_col);
  cudaFree(d_Bh_upcast);
  cudaFree(d_C);
  cudaFree(d_scales16);
}

// ------------------------ FP8 (E4M3) LUT ------------------------

static uint16_t f32_to_f16_bits(float x) {
  __half h = __float2half(x);
  return reinterpret_cast<uint16_t&>(h);
}

static float fp8_e4m3_to_f32(uint8_t v) {
  // Assumption: IEEE-ish E4M3 with sign(1), exp(4, bias=7), mant(3).
  // - exp==0: subnorm/zero
  // - exp==15: inf/nan
  const int sign = (v >> 7) & 1;
  const int exp = (v >> 3) & 0xF;
  const int mant = v & 0x7;
  const float s = sign ? -1.0f : 1.0f;
  const int bias = 7;
  if (exp == 0) {
    if (mant == 0) return s * 0.0f;
    // subnormal: 2^(1-bias) * (mant / 2^3)
    return s * std::ldexp((float)mant, (1 - bias) - 3);
  }
  if (exp == 15) {
    if (mant == 0) return s * INFINITY;
    return NAN;
  }
  // normal: 2^(exp-bias) * (1 + mant/2^3)
  float frac = 1.0f + ((float)mant / 8.0f);
  return s * std::ldexp(frac, exp - bias);
}

[[maybe_unused]] static uint8_t fp8_e4m3_from_f32_sat(float x) {
  // Encode float -> FP8 E4M3 with saturation to max finite (no inf output).
  // This is intended for quantization experiments, not full IEEE compliance.
  if (std::isnan(x)) return 0x7F; // NaN-ish
  if (x == 0.0f) return (std::signbit(x) ? 0x80 : 0x00);

  int sign = std::signbit(x) ? 1 : 0;
  float ax = std::fabs(x);

  // Max finite for E4M3 (exp=14, mant=7): (1 + 7/8) * 2^(14-7) = 1.875 * 128 = 240
  constexpr float max_finite = 240.0f;
  if (ax >= max_finite) {
    return (uint8_t)((sign << 7) | (14u << 3) | 7u);
  }

  // Compute exponent.
  int e = (int)std::floor(std::log2(ax));
  int bias = 7;
  int exp = e + bias;

  // Handle subnormals: exp <= 0
  if (exp <= 0) {
    // subnormal value = 2^(1-bias) * (mant/8). Solve mant ~= ax / 2^(1-bias) * 8.
    float scaled = std::ldexp(ax, (bias - 1) + 3); // ax * 2^(bias-1) * 8
    int mant = (int)std::nearbyint(scaled);
    if (mant < 0) mant = 0;
    if (mant > 7) mant = 7;
    return (uint8_t)((sign << 7) | (0u << 3) | (uint32_t)mant);
  }

  // Clamp exp to normal range (1..14)
  if (exp > 14) {
    return (uint8_t)((sign << 7) | (14u << 3) | 7u);
  }

  // Normal mantissa rounding.
  float frac = ax / std::ldexp(1.0f, e); // ax / 2^e in [1,2)
  float mant_f = (frac - 1.0f) * 8.0f;
  int mant = (int)std::nearbyint(mant_f);
  if (mant == 8) {
    // rounding overflow -> increment exponent
    mant = 0;
    exp += 1;
    if (exp > 14) {
      exp = 14;
      mant = 7;
    }
  }
  if (mant < 0) mant = 0;
  if (mant > 7) mant = 7;

  return (uint8_t)((sign << 7) | ((uint32_t)exp << 3) | (uint32_t)mant);
}

static void upload_fp8_lut() {
  uint16_t host_lut[256];
  for (int i = 0; i < 256; ++i) {
    float f = fp8_e4m3_to_f32((uint8_t)i);
    host_lut[i] = f32_to_f16_bits(f);
  }
  CUDA_CHECK(cudaMemcpyToSymbol(k_fp8_e4m3_to_f16_bits, host_lut, sizeof(host_lut)));
}

static cudaTextureObject_t make_lut_texture_object_u16(const uint16_t* d_ptr) {
  cudaResourceDesc res{};
  res.resType = cudaResourceTypeLinear;
  res.res.linear.devPtr = const_cast<uint16_t*>(d_ptr);
  res.res.linear.desc = cudaCreateChannelDesc<unsigned short>();
  res.res.linear.sizeInBytes = 256 * sizeof(uint16_t);

  cudaTextureDesc tex{};
  tex.addressMode[0] = cudaAddressModeClamp;
  tex.filterMode = cudaFilterModePoint;
  tex.readMode = cudaReadModeElementType;
  tex.normalizedCoords = 0;

  cudaTextureObject_t texObj = 0;
  CUDA_CHECK(cudaCreateTextureObject(&texObj, &res, &tex, nullptr));
  return texObj;
}

// Forward decl (defined later near the RNS helpers).
static cudaTextureObject_t make_u32_tex_object(const uint32_t* d_ptr, size_t count);
static cudaTextureObject_t make_u16_tex_object(const uint16_t* d_ptr, size_t count);

static __device__ __forceinline__ uint32_t lop3_xor3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t out;
  // 3-input XOR (odd parity) LUT.
  asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(out) : "r"(a), "r"(b), "r"(c));
  return out;
}

static __device__ __forceinline__ uint32_t lop3_maj3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t out;
  // 3-input majority (full-adder carry) LUT.
  asm volatile("lop3.b32 %0, %1, %2, %3, 0xE8;" : "=r"(out) : "r"(a), "r"(b), "r"(c));
  return out;
}

