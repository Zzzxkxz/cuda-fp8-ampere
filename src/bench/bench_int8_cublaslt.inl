// ------------------------ BENCH: INT8 tensor-core GEMM baseline (cuBLASLt) ------------------------
static void run_bench_int8gemm() {
  // Simple baseline: INT8 A/B, INT32 accumulation/output.
  constexpr int M = 1024;
  constexpr int N = 1024;
  constexpr int K = 1024;
  constexpr int repeats = 50;

  size_t bytes_A = (size_t)M * K * sizeof(int8_t);
  size_t bytes_B = (size_t)K * N * sizeof(int8_t);
  size_t bytes_C = (size_t)M * N * sizeof(int32_t);

  std::vector<int8_t> h_A((size_t)M * K);
  std::vector<int8_t> h_B((size_t)K * N);

  std::mt19937 rng(123);
  std::uniform_int_distribution<int> dist(-8, 7);
  for (int i = 0; i < M * K; ++i) h_A[(size_t)i] = (int8_t)dist(rng);
  for (int i = 0; i < K * N; ++i) h_B[(size_t)i] = (int8_t)dist(rng);

  int8_t* d_A = nullptr;
  int8_t* d_B_col = nullptr;
  int32_t* d_C = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
  CUDA_CHECK(cudaMalloc(&d_B_col, bytes_B));
  CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

  // Upload A row-major.
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
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

  cublasLtHandle_t lt{};
  CUBLAS_CHECK(cublasLtCreate(&lt));

  cublasLtMatmulDesc_t op{};
  CUBLAS_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

  cublasLtMatrixLayout_t aLayout{};
  cublasLtMatrixLayout_t bLayout{};
  cublasLtMatrixLayout_t cLayout{};
  cublasLtMatrixLayout_t dLayout{};
  // Same layout trick: A is row-major MxK, reinterpret as col-major KxM (ld=K) and op(A)=T.
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&aLayout, CUDA_R_8I, K, M, K));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&bLayout, CUDA_R_8I, K, N, K));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&cLayout, CUDA_R_32I, M, N, M));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&dLayout, CUDA_R_32I, M, N, M));

  cublasLtMatmulPreference_t pref{};
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  size_t workspaceBytes = 1 << 22; // 4 MiB
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceBytes, sizeof(workspaceBytes)));
  void* workspace = nullptr;
  CUDA_CHECK(cudaMalloc(&workspace, workspaceBytes));

  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      lt, op, aLayout, bLayout, cLayout, dLayout, pref, 1, &heuristic, &returned));
  if (returned <= 0) {
    fprintf(stderr, "[int8gemm] cuBLASLt: no heuristic algorithm found for M=%d N=%d K=%d\n", M, N, K);
    std::exit(1);
  }

  int32_t alpha = 1;
  int32_t beta = 0;

  auto run_once = [&] {
    CUBLAS_CHECK(cublasLtMatmul(
        lt, op,
        &alpha,
        d_A, aLayout,
        d_B_col, bLayout,
        &beta,
        d_C, cLayout,
        d_C, dLayout,
        &heuristic.algo,
        workspace, workspaceBytes,
        0));
  };

  // Warmup
  run_once();
  CUDA_CHECK(cudaDeviceSynchronize());

  GpuTimer t;
  t.begin();
  for (int r = 0; r < repeats; ++r) run_once();
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms_avg = t.end_ms() / (float)repeats;

  double ops = 2.0 * (double)M * (double)N * (double)K;
  double tops = (ops / (ms_avg / 1e3)) / 1e12;
  printf("[int8gemm] cublasLt_int8xint8->int32: %.3f ms/iter  (%.2f TOPS)\n", ms_avg, tops);

  CUDA_CHECK(cudaFree(workspace));
  CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(aLayout));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(bLayout));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(cLayout));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(dLayout));
  CUBLAS_CHECK(cublasLtMatmulDescDestroy(op));
  CUBLAS_CHECK(cublasLtDestroy(lt));

  cudaFree(d_A);
  cudaFree(d_B_col);
  cudaFree(d_C);
}

// ------------------------ BENCH: INT8 mantissa (BFP-style) + post-scale ------------------------
__global__ void postscale_int32_to_fp16_colmajor_vec4_tex_kernel(
    const int32_t* __restrict__ in_i32,
    half* __restrict__ out_h,
    int M,
    int N,
    float global_scale,
    cudaTextureObject_t scale_tex_u16) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int base = tid * 4;
  int n = M * N;
  if (base >= n) return;

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    int idx = base + i;
    if (idx >= n) break;
    int col = idx / M;
    unsigned short bits = tex1Dfetch<unsigned short>(scale_tex_u16, col);
    union { unsigned short u; half h; } cvt;
    cvt.u = bits;
    float s = __half2float(cvt.h) * global_scale;
    out_h[idx] = __float2half((float)in_i32[idx] * s);
  }
}

static void run_bench_int8bfp(bool use_l2pin) {
  // Concept: represent values as int8 mantissa + per-column scale (block-float-ish).
  // Compute uses INT8 tensor cores (IMMA) and then applies scales as a separate tiled pass.
  constexpr int M = 1024;
  constexpr int N = 1024;
  constexpr int K = 1024;
  constexpr int repeats = 50;
  constexpr float global_scale = 1.0f;

  size_t bytes_A = (size_t)M * K * sizeof(int8_t);
  size_t bytes_B = (size_t)K * N * sizeof(int8_t);
  size_t bytes_C = (size_t)M * N * sizeof(int32_t);
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
  for (int n = 0; n < N; ++n) {
    h_scale_u16[(size_t)n] = f32_to_f16_bits(sd(rng));
  }

  int8_t* d_A = nullptr;
  int8_t* d_B_col = nullptr;
  int32_t* d_C = nullptr;
  half* d_D = nullptr;
  uint16_t* d_scale_u16 = nullptr;

  CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
  CUDA_CHECK(cudaMalloc(&d_B_col, bytes_B));
  CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
  CUDA_CHECK(cudaMalloc(&d_D, bytes_D));
  CUDA_CHECK(cudaMalloc(&d_scale_u16, bytes_scale));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scale_u16, h_scale_u16.data(), bytes_scale, cudaMemcpyHostToDevice));

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

  cudaTextureObject_t scale_tex = make_u16_tex_object(d_scale_u16, (size_t)N);

  cublasLtHandle_t lt{};
  CUBLAS_CHECK(cublasLtCreate(&lt));

  cublasLtMatmulDesc_t op{};
  CUBLAS_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

  cublasLtMatrixLayout_t aLayout{};
  cublasLtMatrixLayout_t bLayout{};
  cublasLtMatrixLayout_t cLayout{};
  cublasLtMatrixLayout_t dLayout{};
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&aLayout, CUDA_R_8I, K, M, K));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&bLayout, CUDA_R_8I, K, N, K));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&cLayout, CUDA_R_32I, M, N, M));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&dLayout, CUDA_R_32I, M, N, M));

  cublasLtMatmulPreference_t pref{};
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  size_t workspaceBytes = 1 << 22;
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceBytes, sizeof(workspaceBytes)));
  void* workspace = nullptr;
  CUDA_CHECK(cudaMalloc(&workspace, workspaceBytes));

  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      lt, op, aLayout, bLayout, cLayout, dLayout, pref, 1, &heuristic, &returned));
  if (returned <= 0) {
    fprintf(stderr, "[int8bfp] cuBLASLt: no heuristic algorithm found for M=%d N=%d K=%d\n", M, N, K);
    std::exit(1);
  }

  int32_t alpha = 1;
  int32_t beta = 0;

  auto run_gemm_once = [&] {
    CUBLAS_CHECK(cublasLtMatmul(
        lt, op,
        &alpha,
        d_A, aLayout,
        d_B_col, bLayout,
        &beta,
        d_C, cLayout,
        d_C, dLayout,
        &heuristic.algo,
        workspace, workspaceBytes,
        stream));
  };

  auto run_post_once = [&] {
    postscale_int32_to_fp16_colmajor_vec4_tex_kernel<<<((M * N / 4) + 255) / 256, 256, 0, stream>>>(
        d_C, d_D, M, N, global_scale, scale_tex);
  };

  // Warmup
  run_gemm_once();
  run_post_once();
  CUDA_CHECK(cudaDeviceSynchronize());

  GpuTimer t;
  t.begin();
  for (int r = 0; r < repeats; ++r) run_gemm_once();
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms_gemm = t.end_ms() / (float)repeats;

  t.begin();
  for (int r = 0; r < repeats; ++r) run_post_once();
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms_post = t.end_ms() / (float)repeats;

  double ops = 2.0 * (double)M * (double)N * (double)K;
  double tops = (ops / (ms_gemm / 1e3)) / 1e12;
  printf("[int8bfp] int8xint8->int32 GEMM: %.3f ms/iter  (%.2f TOPS)\n", ms_gemm, tops);
  printf("[int8bfp] postscale int32->fp16 (per-col scale via TEX): %.3f ms/iter\n", ms_post);
  printf("[int8bfp] end2end (gemm+post): %.3f ms/iter\n", ms_gemm + ms_post);

  CUDA_CHECK(cudaDestroyTextureObject(scale_tex));
  CUDA_CHECK(cudaFree(workspace));
  CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(aLayout));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(bLayout));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(cLayout));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(dLayout));
  CUBLAS_CHECK(cublasLtMatmulDescDestroy(op));
  CUBLAS_CHECK(cublasLtDestroy(lt));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B_col));
  CUDA_CHECK(cudaFree(d_C));
  CUDA_CHECK(cudaFree(d_D));
  CUDA_CHECK(cudaFree(d_scale_u16));

  if (use_l2pin) {
    (void)cudaCtxResetPersistingL2Cache();
  }
}

// ------------------------ BENCH: probe cuBLASLt heuristics for int8xint8->int32 GEMM ------------------------
static void run_bench_int8bfp_probe() {
  // Emit cuBLASLt's internal heuristics trace to a log file for inspection.
  // Level 4 == Heuristics Trace.
  // Mask: 2 (Performance Trace) | 8 (Heuristics Trace).
  LtLoggerScope log("cublaslt_heuristics.log", 4, (2 | 8));
  if (log.enabled) {
    (void)cublasLtLoggerSetCallback(lt_logger_callback);
    printf("[int8bfp_probe] cuBLASLt logger enabled -> cublaslt_heuristics.log\n");
  }

  constexpr int M = 1024;
  constexpr int N = 1024;
  constexpr int K = 1024;

  size_t bytes_A = (size_t)M * K * sizeof(int8_t);
  size_t bytes_B = (size_t)K * N * sizeof(int8_t);
  size_t bytes_C = (size_t)M * N * sizeof(int32_t);

  std::vector<int8_t> h_A((size_t)M * K);
  std::vector<int8_t> h_B((size_t)K * N);

  std::mt19937 rng(123);
  std::uniform_int_distribution<int> dist(-8, 7);
  for (int i = 0; i < M * K; ++i) h_A[(size_t)i] = (int8_t)dist(rng);
  for (int i = 0; i < K * N; ++i) h_B[(size_t)i] = (int8_t)dist(rng);

  int8_t* d_A = nullptr;
  int8_t* d_B_col = nullptr;
  int32_t* d_C = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
  CUDA_CHECK(cudaMalloc(&d_B_col, bytes_B));
  CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));

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
  CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));

  cudaStream_t stream = 0;

  cublasLtHandle_t lt{};
  CUBLAS_CHECK(cublasLtCreate(&lt));

  cublasLtMatmulDesc_t op{};
  CUBLAS_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

  cublasLtMatrixLayout_t aLayout{};
  cublasLtMatrixLayout_t bLayout{};
  cublasLtMatrixLayout_t cLayout{};
  cublasLtMatrixLayout_t dLayout{};
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&aLayout, CUDA_R_8I, K, M, K));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&bLayout, CUDA_R_8I, K, N, K));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&cLayout, CUDA_R_32I, M, N, M));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&dLayout, CUDA_R_32I, M, N, M));

  cublasLtMatmulPreference_t pref{};
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  // Use a larger workspace for probing; some fast algorithms need it.
  size_t workspaceBytes = 1ull << 26;
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceBytes, sizeof(workspaceBytes)));
  void* workspace = nullptr;
  CUDA_CHECK(cudaMalloc(&workspace, workspaceBytes));

  constexpr int kMax = 32;
  std::vector<cublasLtMatmulHeuristicResult_t> heur(kMax);
  int returned = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      lt, op, aLayout, bLayout, cLayout, dLayout, pref, kMax, heur.data(), &returned));

  printf("[int8bfp_probe] returned %d heuristic candidates\n", returned);
  if (returned <= 0) {
    printf("[int8bfp_probe] no algorithms returned\n");
    std::exit(1);
  }

  int32_t alpha = 1;
  int32_t beta = 0;

  struct ResultRow { int idx; float ms; };
  std::vector<ResultRow> times;
  times.reserve((size_t)returned);

  // Disable logging during timing loops to avoid skew.
  if (log.enabled) {
    (void)cublasLtLoggerSetMask(0);
  }

  // Dump a compact set of config fields + measure each candidate.
  int to_dump = returned;
  if (to_dump > 16) to_dump = 16;

  for (int i = 0; i < returned; ++i) {
    auto &h = heur[i];

    int32_t algo_id = -1;
    uint32_t tile_id = 0;
    int32_t splitk = 0;
    uint32_t red = 0;
    uint32_t swz = 0;
    uint32_t custom = 0;
    uint32_t stages = 0;
    uint16_t inner = 0;
    uint16_t cluster = 0;

    (void)lt_algo_get_i32(&h.algo, CUBLASLT_ALGO_CONFIG_ID, algo_id);
    (void)lt_algo_get_u32(&h.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, tile_id);
    (void)lt_algo_get_i32(&h.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, splitk);
    (void)lt_algo_get_u32(&h.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, red);
    (void)lt_algo_get_u32(&h.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, swz);
    (void)lt_algo_get_u32(&h.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, custom);
    (void)lt_algo_get_u32(&h.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, stages);
    (void)lt_algo_get_u16(&h.algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, inner);
    (void)lt_algo_get_u16(&h.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, cluster);

    if (i < to_dump) {
          printf("[int8bfp_probe] #%02d state=%d waves=%.3f ws=%zu id=%d tile=%u stages=%u splitk=%d\n",
            i, (int)h.state, (double)h.wavesCount, (size_t)h.workspaceSize,
            (int)algo_id, (unsigned)tile_id, (unsigned)stages, (int)splitk);
    }

    if (h.state != CUBLAS_STATUS_SUCCESS) continue;

    auto run_once = [&] {
      CUBLAS_CHECK(cublasLtMatmul(
          lt, op,
          &alpha,
          d_A, aLayout,
          d_B_col, bLayout,
          &beta,
          d_C, cLayout,
          d_C, dLayout,
          &h.algo,
          workspace, workspaceBytes,
          stream));
    };

    // Quick timing to rank algorithms.
    constexpr int reps = 50;
    run_once();
    CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer t;
    t.begin(stream);
    for (int r = 0; r < reps; ++r) run_once();
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = t.end_ms(stream) / (float)reps;
    times.push_back({i, ms});
  }

  // Sort by time.
  std::sort(times.begin(), times.end(), [](const ResultRow& a, const ResultRow& b){ return a.ms < b.ms; });

  int topk = (int)times.size();
  if (topk > 8) topk = 8;
  printf("[int8bfp_probe] top %d by measured time:\n", topk);
  for (int j = 0; j < topk; ++j) {
    int i = times[j].idx;
    float ms = times[j].ms;
    int32_t algo_id = -1;
    uint32_t tile_id = 0;
    int32_t splitk = 0;
    (void)lt_algo_get_i32(&heur[i].algo, CUBLASLT_ALGO_CONFIG_ID, algo_id);
    (void)lt_algo_get_u32(&heur[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, tile_id);
    (void)lt_algo_get_i32(&heur[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, splitk);
    printf("  #%02d: %.3f ms  id=%d tile=%u splitk=%d\n", i, ms, (int)algo_id, (unsigned)tile_id, (int)splitk);
  }

  // --- Deeper probe: enumerate all algo IDs, check support, and time the best ones.
  printf("[int8bfp_probe] --- AlgoGetIds + AlgoCheck enumeration ---\n");
  constexpr int kMaxIds = 1024;
  std::vector<int> algoIds((size_t)kMaxIds);
  int idCount = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoGetIds(
      lt, CUBLAS_COMPUTE_32I, CUDA_R_32I, CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUDA_R_32I,
      kMaxIds, algoIds.data(), &idCount));
  printf("[int8bfp_probe] AlgoGetIds returned %d algo IDs\n", idCount);

  struct CheckedCand {
    int algoId;
    cublasLtMatmulHeuristicResult_t check;
    float ms;
    bool timed;
  };
  std::vector<CheckedCand> checked;
  checked.reserve((size_t)idCount);

  auto cap_get_u32_array = [](const cublasLtMatmulAlgo_t* algo, cublasLtMatmulAlgoCapAttributes_t attr) {
    size_t bytes = 0;
    cublasStatus_t s0 = cublasLtMatmulAlgoCapGetAttribute(algo, attr, nullptr, 0, &bytes);
    if (s0 != CUBLAS_STATUS_SUCCESS || bytes == 0) return std::vector<uint32_t>{};
    std::vector<uint32_t> v(bytes / sizeof(uint32_t));
    size_t written = 0;
    cublasStatus_t s1 = cublasLtMatmulAlgoCapGetAttribute(algo, attr, v.data(), bytes, &written);
    if (s1 != CUBLAS_STATUS_SUCCESS || written == 0) return std::vector<uint32_t>{};
    v.resize(written / sizeof(uint32_t));
    return v;
  };

  int supported = 0;
  int supported_ws = 0;
  for (int ii = 0; ii < idCount; ++ii) {
    int algoId = algoIds[(size_t)ii];
    cublasLtMatmulAlgo_t base{};
    cublasStatus_t sInit = cublasLtMatmulAlgoInit(
        lt, CUBLAS_COMPUTE_32I, CUDA_R_32I, CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUDA_R_32I,
        algoId, &base);
    if (sInit != CUBLAS_STATUS_SUCCESS) continue;

    // Many algos need TILE_ID / STAGES_ID explicitly configured; otherwise AlgoCheck will reject.
    auto tiles = cap_get_u32_array(&base, CUBLASLT_ALGO_CAP_TILE_IDS);
    auto stages = cap_get_u32_array(&base, CUBLASLT_ALGO_CAP_STAGES_IDS);
    if (tiles.empty()) continue;
    if (stages.empty()) stages.push_back(0u);

    for (uint32_t tile : tiles) {
      for (uint32_t stage : stages) {
        cublasLtMatmulAlgo_t algo = base;
        (void)cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile));
        (void)cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stage, sizeof(stage));

        cublasLtMatmulHeuristicResult_t chk{};
        cublasStatus_t sChk = cublasLtMatmulAlgoCheck(lt, op, aLayout, bLayout, cLayout, dLayout, &algo, &chk);
        if (sChk != CUBLAS_STATUS_SUCCESS) continue;
        if (chk.state != CUBLAS_STATUS_SUCCESS) continue;
        ++supported;
        if (chk.workspaceSize <= workspaceBytes) ++supported_ws;
        checked.push_back({algoId, chk, 0.0f, false});
      }
    }
  }
  printf("[int8bfp_probe] AlgoCheck supported=%d (workspace<=%zu: %d)\n",
         supported, (size_t)workspaceBytes, supported_ws);

  // Time a subset (sorted by wavesCount proxy) to keep runtime reasonable.
  std::sort(checked.begin(), checked.end(), [](const CheckedCand& a, const CheckedCand& b) {
    return a.check.wavesCount > b.check.wavesCount;
  });

  int to_time = (int)checked.size();
  if (to_time > 64) to_time = 64;
  printf("[int8bfp_probe] timing top %d by wavesCount...\n", to_time);

  constexpr int reps2 = 30;
  for (int i = 0; i < to_time; ++i) {
    auto& c = checked[(size_t)i];
    if (c.check.workspaceSize > workspaceBytes) continue;

    // Grab a single Matmul trace message for this algoId (best-effort).
    if (log.enabled && g_lt_matmul_trace_by_algo.find(c.algoId) == g_lt_matmul_trace_by_algo.end()) {
      g_lt_trace_target_algo = c.algoId;
      g_lt_trace_need_one = true;
      // Enable just enough logging to emit the Matmul trace line.
      (void)cublasLtLoggerSetMask(2);
      CUBLAS_CHECK(cublasLtMatmul(
          lt, op,
          &alpha,
          d_A, aLayout,
          d_B_col, bLayout,
          &beta,
          d_C, cLayout,
          d_C, dLayout,
          &c.check.algo,
          workspace, workspaceBytes,
          stream));
      CUDA_CHECK(cudaDeviceSynchronize());
      (void)cublasLtLoggerSetMask(0);
      g_lt_trace_need_one = false;
    }

    auto run_once2 = [&] {
      CUBLAS_CHECK(cublasLtMatmul(
          lt, op,
          &alpha,
          d_A, aLayout,
          d_B_col, bLayout,
          &beta,
          d_C, cLayout,
          d_C, dLayout,
          &c.check.algo,
          workspace, workspaceBytes,
          stream));
    };

    run_once2();
    CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer t2;
    t2.begin(stream);
    for (int r = 0; r < reps2; ++r) run_once2();
    CUDA_CHECK(cudaDeviceSynchronize());
    c.ms = t2.end_ms(stream) / (float)reps2;
    c.timed = true;
  }

  // Rank measured.
  std::vector<CheckedCand*> timed;
  timed.reserve(checked.size());
  for (auto& c : checked) if (c.timed) timed.push_back(&c);
  std::sort(timed.begin(), timed.end(), [](const CheckedCand* a, const CheckedCand* b) {
    return a->ms < b->ms;
  });

  int top2 = (int)timed.size();
  if (top2 > 12) top2 = 12;
  printf("[int8bfp_probe] top %d (AlgoCheck-enumerated) by measured time:\n", top2);
  for (int j = 0; j < top2; ++j) {
    const auto& c = *timed[(size_t)j];

        int32_t cfg_id = -1;
        uint32_t tile_id = 0;
        int32_t splitk = 0;
        uint32_t stages = 0;
    (void)lt_algo_get_i32(&c.check.algo, CUBLASLT_ALGO_CONFIG_ID, cfg_id);
    (void)lt_algo_get_u32(&c.check.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, tile_id);
    (void)lt_algo_get_i32(&c.check.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, splitk);
    (void)lt_algo_get_u32(&c.check.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, stages);

    std::string tile_str;
    std::string stages_str;
    auto it = g_lt_matmul_trace_by_algo.find(c.algoId);
    if (it != g_lt_matmul_trace_by_algo.end()) {
      tile_str = extract_token_after(it->second.c_str(), "tile=");
      stages_str = extract_token_after(it->second.c_str(), "stages=");
    }

    if (!tile_str.empty() || !stages_str.empty()) {
          printf("  %.3f ms  algoId=%d tile=%u stages=%u\n",
            (double)c.ms, (int)c.algoId, (unsigned)tile_id, (unsigned)stages);
          printf("           %s %s\n",
            tile_str.empty() ? "" : tile_str.c_str(),
            stages_str.empty() ? "" : stages_str.c_str());
    } else {
      printf("  %.3f ms  algoId=%d cfgId=%d tile=%u stages=%u splitk=%d ws=%zu waves=%.3f\n",
             (double)c.ms, (int)c.algoId, (int)cfg_id, (unsigned)tile_id, (unsigned)stages, (int)splitk,
             (size_t)c.check.workspaceSize, (double)c.check.wavesCount);
    }
  }

  CUDA_CHECK(cudaFree(workspace));
  CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(aLayout));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(bLayout));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(cLayout));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(dLayout));
  CUBLAS_CHECK(cublasLtMatmulDescDestroy(op));
  CUBLAS_CHECK(cublasLtDestroy(lt));

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B_col));
  CUDA_CHECK(cudaFree(d_C));
}

