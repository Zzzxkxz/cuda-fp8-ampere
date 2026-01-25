// ------------------------ BENCH: TEX vs LDG ------------------------
__global__ void bench_global_load_kernel(const float* __restrict__ in, float* __restrict__ out, int n, int alu_iters) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  float x = in[idx];

  // Synthetic ALU to create latency hiding opportunities.
  #pragma unroll 1
  for (int i = 0; i < alu_iters; ++i) {
    x = fmaf(x, 1.0000001f, 0.0000003f);
    x = fmaf(x, 0.9999997f, 0.0000002f);
  }

  out[idx] = x;
}

__global__ void bench_tex_load_kernel(cudaTextureObject_t tex, float* __restrict__ out, int n, int alu_iters) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  // tex1D uses float coordinates in [0, n) when unnormalized.
  float x = tex1D<float>(tex, (float)idx + 0.5f);

  #pragma unroll 1
  for (int i = 0; i < alu_iters; ++i) {
    x = fmaf(x, 1.0000001f, 0.0000003f);
    x = fmaf(x, 0.9999997f, 0.0000002f);
  }

  out[idx] = x;
}

__global__ void bench_tex2d_load_kernel(cudaTextureObject_t tex, float* __restrict__ out, int n, int alu_iters, int width_log2) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  int x_i = idx & ((1 << width_log2) - 1);
  int y_i = idx >> width_log2;

  float x = tex2D<float>(tex, (float)x_i + 0.5f, (float)y_i + 0.5f);

  #pragma unroll 1
  for (int i = 0; i < alu_iters; ++i) {
    x = fmaf(x, 1.0000001f, 0.0000003f);
    x = fmaf(x, 0.9999997f, 0.0000002f);
  }

  out[idx] = x;
}

static cudaTextureObject_t make_1d_texture_object(const float* d_ptr, size_t count) {
  cudaResourceDesc res{};
  res.resType = cudaResourceTypeLinear;
  res.res.linear.devPtr = const_cast<float*>(d_ptr);
  res.res.linear.desc = cudaCreateChannelDesc<float>();
  res.res.linear.sizeInBytes = count * sizeof(float);

  cudaTextureDesc tex{};
  tex.addressMode[0] = cudaAddressModeClamp;
  tex.filterMode = cudaFilterModePoint;
  tex.readMode = cudaReadModeElementType;
  tex.normalizedCoords = 0;

  cudaTextureObject_t texObj = 0;
  CUDA_CHECK(cudaCreateTextureObject(&texObj, &res, &tex, nullptr));
  return texObj;
}

static cudaTextureObject_t make_2d_texture_object_linear(cudaArray_t arr) {
  cudaResourceDesc res{};
  res.resType = cudaResourceTypeArray;
  res.res.array.array = arr;

  cudaTextureDesc tex{};
  tex.addressMode[0] = cudaAddressModeClamp;
  tex.addressMode[1] = cudaAddressModeClamp;
  tex.filterMode = cudaFilterModeLinear;
  tex.readMode = cudaReadModeElementType;
  tex.normalizedCoords = 0;

  cudaTextureObject_t texObj = 0;
  CUDA_CHECK(cudaCreateTextureObject(&texObj, &res, &tex, nullptr));
  return texObj;
}

static void run_bench_tex() {
  // Note: cudaArray-backed 1D textures have a max width; keep this comfortably below limits.
  const int n = 1 << 24;  // ~16 million floats (~64 MiB)
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;

  float* d_in = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_in, 0x3f, n * sizeof(float)));

  cudaTextureObject_t tex_point = make_1d_texture_object(d_in, (size_t)n);

  // For true linear filtering, use a 2D cudaArray and tex2D().
  // Choose a power-of-two width so idx -> (x,y) uses shift+mask.
  constexpr int width_log2 = 14;          // 16384
  constexpr int width = 1 << width_log2;
  constexpr int height = n / width;       // with n=2^24, height=1024
  static_assert((width * height) == n, "n must be exactly width*height");

  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  cudaArray_t arr2d = nullptr;
  CUDA_CHECK(cudaMallocArray(&arr2d, &desc, (size_t)width, (size_t)height));
  CUDA_CHECK(cudaMemcpy2DToArray(arr2d, 0, 0, d_in,
                                 (size_t)width * sizeof(float),
                                 (size_t)width * sizeof(float),
                                 (size_t)height,
                                 cudaMemcpyDeviceToDevice));
  cudaTextureObject_t tex_linear = make_2d_texture_object_linear(arr2d);

  std::vector<int> alu_sweep = {0, 16, 64, 256};

  printf("[tex] n=%d (%.1f MiB) blocks=%d threads=%d\n", n, (n * sizeof(float)) / (1024.0 * 1024.0), blocks, threads);

  // Warmup
  bench_global_load_kernel<<<blocks, threads>>>(d_in, d_out, n, 16);
  bench_tex_load_kernel<<<blocks, threads>>>(tex_point, d_out, n, 16);
  bench_tex2d_load_kernel<<<blocks, threads>>>(tex_linear, d_out, n, 16, width_log2);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int alu_iters : alu_sweep) {
    GpuTimer t;

    t.begin();
    bench_global_load_kernel<<<blocks, threads>>>(d_in, d_out, n, alu_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms_g = t.end_ms();

    t.begin();
    bench_tex_load_kernel<<<blocks, threads>>>(tex_point, d_out, n, alu_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms_t_point = t.end_ms();

    t.begin();
    bench_tex2d_load_kernel<<<blocks, threads>>>(tex_linear, d_out, n, alu_iters, width_log2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms_t_linear = t.end_ms();

    double bytes = (double)n * sizeof(float);
    double gbps_g = (bytes / 1e9) / (ms_g / 1e3);
    double gbps_t_point = (bytes / 1e9) / (ms_t_point / 1e3);
    double gbps_t_linear = (bytes / 1e9) / (ms_t_linear / 1e3);

    printf("[tex] alu_iters=%d | global: %.3f ms (%.1f GB/s) | tex(point): %.3f ms (%.1f GB/s) | tex(linear): %.3f ms (%.1f GB/s)\n",
           alu_iters, ms_g, gbps_g, ms_t_point, gbps_t_point, ms_t_linear, gbps_t_linear);
  }

  CUDA_CHECK(cudaDestroyTextureObject(tex_point));
  CUDA_CHECK(cudaDestroyTextureObject(tex_linear));
  CUDA_CHECK(cudaFreeArray(arr2d));
  cudaFree(d_in);
  cudaFree(d_out);
}

// ------------------------ BENCH: TEX prefetch pipeline (report §3.3) ------------------------
template <int Prefetch, bool UseTex>
__global__ void bench_texpipe_kernel(
    const float* __restrict__ in,
    cudaTextureObject_t tex,
    float* __restrict__ out,
    int n,
    int iters,
    int alu_iters) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = blockDim.x * gridDim.x;
  if (tid >= total_threads) return;

  // Use a power-of-two n so we can wrap with a mask.
  int mask = n - 1;
  uint32_t pos = ((uint32_t)tid * 2654435761u) & (uint32_t)mask;
  // Per-thread LCG step; must be odd to be full-period mod 2^k.
  uint32_t step = ((uint32_t)tid * 2246822519u) | 1u;

  float pref[Prefetch];

  #pragma unroll
  for (int i = 0; i < Prefetch; ++i) {
    float v;
    if constexpr (UseTex) {
      v = tex1D<float>(tex, (float)pos + 0.5f);
    } else {
      v = in[pos];
    }
    pref[i] = v;
    pos = (pos + step) & (uint32_t)mask;
  }

  float acc = 0.0f;
  #pragma unroll 1
  for (int it = 0; it < iters; ++it) {
    int slot = it & (Prefetch - 1);

    // Issue fetch for the "future" value.
    float next;
    if constexpr (UseTex) {
      next = tex1D<float>(tex, (float)pos + 0.5f);
    } else {
      next = in[pos];
    }
    pos = (pos + step) & (uint32_t)mask;

    // Independent ALU work (latency hiding budget).
    #pragma unroll 1
    for (int k = 0; k < alu_iters; ++k) {
      acc = fmaf(acc, 1.0000001f, 0.0000003f);
      acc = fmaf(acc, 0.9999997f, 0.0000002f);
    }

    // Consume the older prefetched value.
    acc = fmaf(pref[slot], 0.000001f, acc);
    pref[slot] = next;
  }

  // One output per thread to keep compiler honest.
  out[tid] = acc;
}

static void run_bench_texpipe() {
  // Keep n as power-of-two to use masking in-kernel.
  const int n = 1 << 24;
  const int threads = 256;
  const int blocks = 1024;
  const int total_threads = blocks * threads;
  const int iters = 4096;

  float* d_in = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, (size_t)n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, (size_t)total_threads * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_in, 0x3f, (size_t)n * sizeof(float)));

  cudaTextureObject_t tex = make_1d_texture_object(d_in, (size_t)n);

  std::vector<int> alu_sweep = {0, 16, 64, 256, 1024};

  auto run_case = [&](const char* name, bool use_tex, int prefetch, int alu_iters) {
    // Warmup
    if (prefetch == 1) {
      if (use_tex) {
        bench_texpipe_kernel<1, true><<<blocks, threads>>>(d_in, tex, d_out, n, 256, alu_iters);
      } else {
        bench_texpipe_kernel<1, false><<<blocks, threads>>>(d_in, tex, d_out, n, 256, alu_iters);
      }
    } else if (prefetch == 2) {
      if (use_tex) {
        bench_texpipe_kernel<2, true><<<blocks, threads>>>(d_in, tex, d_out, n, 256, alu_iters);
      } else {
        bench_texpipe_kernel<2, false><<<blocks, threads>>>(d_in, tex, d_out, n, 256, alu_iters);
      }
    } else if (prefetch == 4) {
      if (use_tex) {
        bench_texpipe_kernel<4, true><<<blocks, threads>>>(d_in, tex, d_out, n, 256, alu_iters);
      } else {
        bench_texpipe_kernel<4, false><<<blocks, threads>>>(d_in, tex, d_out, n, 256, alu_iters);
      }
    } else if (prefetch == 8) {
      if (use_tex) {
        bench_texpipe_kernel<8, true><<<blocks, threads>>>(d_in, tex, d_out, n, 256, alu_iters);
      } else {
        bench_texpipe_kernel<8, false><<<blocks, threads>>>(d_in, tex, d_out, n, 256, alu_iters);
      }
    } else if (prefetch == 16) {
      if (use_tex) {
        bench_texpipe_kernel<16, true><<<blocks, threads>>>(d_in, tex, d_out, n, 256, alu_iters);
      } else {
        bench_texpipe_kernel<16, false><<<blocks, threads>>>(d_in, tex, d_out, n, 256, alu_iters);
      }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer t;
    t.begin();
    if (prefetch == 1) {
      if (use_tex) {
        bench_texpipe_kernel<1, true><<<blocks, threads>>>(d_in, tex, d_out, n, iters, alu_iters);
      } else {
        bench_texpipe_kernel<1, false><<<blocks, threads>>>(d_in, tex, d_out, n, iters, alu_iters);
      }
    } else if (prefetch == 2) {
      if (use_tex) {
        bench_texpipe_kernel<2, true><<<blocks, threads>>>(d_in, tex, d_out, n, iters, alu_iters);
      } else {
        bench_texpipe_kernel<2, false><<<blocks, threads>>>(d_in, tex, d_out, n, iters, alu_iters);
      }
    } else if (prefetch == 4) {
      if (use_tex) {
        bench_texpipe_kernel<4, true><<<blocks, threads>>>(d_in, tex, d_out, n, iters, alu_iters);
      } else {
        bench_texpipe_kernel<4, false><<<blocks, threads>>>(d_in, tex, d_out, n, iters, alu_iters);
      }
    } else if (prefetch == 8) {
      if (use_tex) {
        bench_texpipe_kernel<8, true><<<blocks, threads>>>(d_in, tex, d_out, n, iters, alu_iters);
      } else {
        bench_texpipe_kernel<8, false><<<blocks, threads>>>(d_in, tex, d_out, n, iters, alu_iters);
      }
    } else if (prefetch == 16) {
      if (use_tex) {
        bench_texpipe_kernel<16, true><<<blocks, threads>>>(d_in, tex, d_out, n, iters, alu_iters);
      } else {
        bench_texpipe_kernel<16, false><<<blocks, threads>>>(d_in, tex, d_out, n, iters, alu_iters);
      }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = t.end_ms();

    double total_reads = (double)total_threads * (double)iters; // one 4B fetch per iter
    double gbps = (total_reads * 4.0 / 1e9) / (ms / 1e3);
    printf("[texpipe] %-6s prefetch=%2d alu_iters=%4d | %.3f ms | %.1f GB/s\n", name, prefetch, alu_iters, ms, gbps);
  };

  printf("[texpipe] blocks=%d threads=%d total_threads=%d iters=%d n=%d\n", blocks, threads, total_threads, iters, n);
  printf("[texpipe] Goal: see how prefetch depth hides TEX/global latency (report §3.3)\n");

  const int prefetches[] = {1, 2, 4, 8, 16};
  for (int alu_iters : alu_sweep) {
    for (int p : prefetches) {
      run_case("global", false, p, alu_iters);
      run_case("tex", true, p, alu_iters);
    }
  }

  CUDA_CHECK(cudaDestroyTextureObject(tex));
  cudaFree(d_in);
  cudaFree(d_out);
}

// ------------------------ BENCH: TRANSPOSE ------------------------
__device__ __forceinline__ uint32_t shfl_xor(uint32_t v, int lane_mask) {
  return __shfl_xor_sync(0xFFFFFFFFu, v, lane_mask);
}

__device__ __forceinline__ uint32_t bit_swap_mask(uint32_t x, uint32_t y, uint32_t mask, int shift) {
  // Swap masked bits between x and y with a shift. Returns new x; caller computes y similarly.
  uint32_t t = ((x >> shift) ^ y) & mask;
  y ^= t;
  x ^= (t << shift);
  return x;
}

__global__ void bench_transpose_kernel(uint32_t* out, int iters) {
  // Each lane starts with a 32-bit pattern.
  uint32_t v = 0xA5A5A5A5u ^ (uint32_t)threadIdx.x;

  // We treat the warp registers as a 32x32 bit matrix: lane = row, bit = col.
  // Butterfly exchange stages: 16,8,4,2,1. After each shuffle, swap bit groups.
  #pragma unroll 1
  for (int it = 0; it < iters; ++it) {
    uint32_t x = v;

    // Stage 16
    {
      uint32_t y = shfl_xor(x, 16);
      uint32_t mask = 0x0000FFFFu;
      uint32_t t = ((x & mask) << 16) | (y & mask);
      uint32_t u = ((y & ~mask) >> 16) | (x & ~mask);
      x = (threadIdx.x & 16) ? u : t;
    }

    // Stage 8
    {
      uint32_t y = shfl_xor(x, 8);
      uint32_t mask = 0x00FF00FFu;
      uint32_t t = ((x & mask) << 8) | (y & mask);
      uint32_t u = ((y & ~mask) >> 8) | (x & ~mask);
      x = (threadIdx.x & 8) ? u : t;
    }

    // Stage 4
    {
      uint32_t y = shfl_xor(x, 4);
      uint32_t mask = 0x0F0F0F0Fu;
      uint32_t t = ((x & mask) << 4) | (y & mask);
      uint32_t u = ((y & ~mask) >> 4) | (x & ~mask);
      x = (threadIdx.x & 4) ? u : t;
    }

    // Stage 2
    {
      uint32_t y = shfl_xor(x, 2);
      uint32_t mask = 0x33333333u;
      uint32_t t = ((x & mask) << 2) | (y & mask);
      uint32_t u = ((y & ~mask) >> 2) | (x & ~mask);
      x = (threadIdx.x & 2) ? u : t;
    }

    // Stage 1
    {
      uint32_t y = shfl_xor(x, 1);
      uint32_t mask = 0x55555555u;
      uint32_t t = ((x & mask) << 1) | (y & mask);
      uint32_t u = ((y & ~mask) >> 1) | (x & ~mask);
      x = (threadIdx.x & 1) ? u : t;
    }

    v = x ^ (uint32_t)it;
  }

  // One warp per block is enough for this microbench.
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  out[idx] = v;
}

static void run_bench_transpose() {
  constexpr int blocks = 256;
  constexpr int threads = 32;  // one warp
  constexpr int iters = 1 << 15;

  uint32_t* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, blocks * threads * sizeof(uint32_t)));

  // Warmup
  bench_transpose_kernel<<<blocks, threads>>>(d_out, 256);
  CUDA_CHECK(cudaDeviceSynchronize());

  GpuTimer t;
  t.begin();
  bench_transpose_kernel<<<blocks, threads>>>(d_out, iters);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms = t.end_ms();

  double total_warp_iters = (double)blocks * (double)iters;
  double ns_per_transpose_per_warp = (ms * 1e6) / total_warp_iters;

  printf("[transpose] blocks=%d warps=%d iters=%d\n", blocks, blocks, iters);
  printf("[transpose] %.3f ms total, %.3f ns / transpose / warp\n", ms, ns_per_transpose_per_warp);

  cudaFree(d_out);
}

// ------------------------ BENCH: RNS lane-split ------------------------
// Pseudo-Mersenne modulus p = 2^31 - c, with small c.
// This is a perf-oriented microbench; primality of p is not required.
__constant__ uint32_t k_rns_c_list[32];

__host__ __device__ __forceinline__ uint32_t rns_p_from_c(uint32_t c) {
  return 0x80000000u - c;
}

__device__ __forceinline__ uint32_t rns_reduce_p31c(uint64_t x, uint32_t c) {
  constexpr uint32_t mask = 0x7FFFFFFFu;
  uint32_t p = rns_p_from_c(c);

  // First fold: x = (lo31) + c*(hi)
  uint64_t t = (x & mask) + (uint64_t)c * (x >> 31);
  // Second fold to bring back near 31-bit.
  t = (t & mask) + (uint64_t)c * (t >> 31);
  // One more is cheap and stabilizes for larger c.
  t = (t & mask) + (uint64_t)c * (t >> 31);

  uint32_t r = (uint32_t)t;
  // r can still be in [0, p + small], do up to 2 conditional subtracts.
  if (r >= p) r -= p;
  if (r >= p) r -= p;
  return r;
}

__device__ __forceinline__ uint32_t rns_add_p31c(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t p = rns_p_from_c(c);
  uint64_t s = (uint64_t)a + (uint64_t)b;
  if (s >= p) s -= p;
  return (uint32_t)s;
}

__device__ __forceinline__ uint32_t rns_mul_p31c(uint32_t a, uint32_t b, uint32_t c) {
  return rns_reduce_p31c((uint64_t)a * (uint64_t)b, c);
}

static cudaTextureObject_t make_u32_tex_object(const uint32_t* d_ptr, size_t count) {
  cudaResourceDesc res{};
  res.resType = cudaResourceTypeLinear;
  res.res.linear.devPtr = const_cast<uint32_t*>(d_ptr);
  res.res.linear.desc = cudaCreateChannelDesc<unsigned int>();
  res.res.linear.sizeInBytes = count * sizeof(uint32_t);

  cudaTextureDesc tex{};
  tex.addressMode[0] = cudaAddressModeClamp;
  tex.filterMode = cudaFilterModePoint;
  tex.readMode = cudaReadModeElementType;
  tex.normalizedCoords = 0;

  cudaTextureObject_t texObj = 0;
  CUDA_CHECK(cudaCreateTextureObject(&texObj, &res, &tex, nullptr));
  return texObj;
}

static cudaTextureObject_t make_u16_tex_object(const uint16_t* d_ptr, size_t count) {
  cudaResourceDesc res{};
  res.resType = cudaResourceTypeLinear;
  res.res.linear.devPtr = const_cast<uint16_t*>(d_ptr);
  res.res.linear.desc = cudaCreateChannelDesc<unsigned short>();
  res.res.linear.sizeInBytes = count * sizeof(uint16_t);

  cudaTextureDesc tex{};
  tex.addressMode[0] = cudaAddressModeClamp;
  tex.filterMode = cudaFilterModePoint;
  tex.readMode = cudaReadModeElementType;
  tex.normalizedCoords = 0;

  cudaTextureObject_t texObj = 0;
  CUDA_CHECK(cudaCreateTextureObject(&texObj, &res, &tex, nullptr));
  return texObj;
}

template <int LoadMode>
__global__ void bench_rns_lanesplit_kernel(
    const uint32_t* __restrict__ A,
    const uint32_t* __restrict__ B,
    uint32_t* __restrict__ Out,
    const uint32_t* __restrict__ twiddles,
    cudaTextureObject_t tw_tex,
    int num_items,
    int k_residues,
    int tw_mask,
    int iters) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;
  if (warp >= num_items) return;
  if (lane >= k_residues) return;

  uint32_t c = k_rns_c_list[lane];
  uint32_t p = rns_p_from_c(c);

  uint32_t x = A[warp * k_residues + lane] % p;
  uint32_t y = B[warp * k_residues + lane] % p;

  // Broadcast a per-item seed across lanes (register-cache concept).
  uint32_t seed = __shfl_sync(0xFFFFFFFFu, (uint32_t)warp * 2654435761u, 0);

  #pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    int idx = (int)((seed + (uint32_t)i * 1013904223u) & (uint32_t)tw_mask);
    uint32_t tw;
    if constexpr (LoadMode == 0) {
      tw = twiddles[idx];
    } else if constexpr (LoadMode == 1) {
#if __CUDA_ARCH__ >= 350
      tw = __ldg(&twiddles[idx]);
#else
      tw = twiddles[idx];
#endif
    } else {
      tw = (uint32_t)tex1Dfetch<unsigned int>(tw_tex, idx);
    }

    // Keep tw in-range. This isn't a correctness benchmark; it's about throughput of the data path.
    tw &= 0x7FFFFFFFu;
    if (tw >= p) tw -= p;

    x = rns_mul_p31c(x, tw, c);
    y = rns_add_p31c(y, tw, c);
    x = rns_add_p31c(x, y, c);
  }

  Out[warp * k_residues + lane] = x ^ (y + lane);
}

static void run_bench_rns() {
  constexpr int k_residues = 8;
  constexpr int num_items = 1 << 15;     // warps
  constexpr int threads = 256;
  constexpr int blocks = (num_items * 32 + threads - 1) / threads;
  constexpr int iters = 4096;
  constexpr int launches = 40;

  constexpr int tw_len = 1 << 20;
  constexpr int tw_mask = tw_len - 1;
  static_assert((tw_len & (tw_len - 1)) == 0, "tw_len must be power of two");

  // Choose small c values (p = 2^31 - c).
  uint32_t c_list[32]{};
  const uint32_t chosen[k_residues] = {1u, 19u, 61u, 89u, 107u, 127u, 521u, 607u};
  for (int i = 0; i < k_residues; ++i) c_list[i] = chosen[i];
  CUDA_CHECK(cudaMemcpyToSymbol(k_rns_c_list, c_list, sizeof(c_list)));

  std::vector<uint32_t> h_A((size_t)num_items * k_residues);
  std::vector<uint32_t> h_B((size_t)num_items * k_residues);
  std::vector<uint32_t> h_tw((size_t)tw_len);

  std::mt19937 rng(123);
  std::uniform_int_distribution<uint32_t> dist(0u, 0x7FFFFFFFu);
  for (int i = 0; i < num_items; ++i) {
    for (int r = 0; r < k_residues; ++r) {
      uint32_t p = rns_p_from_c(chosen[r]);
      h_A[(size_t)i * k_residues + r] = dist(rng) % p;
      h_B[(size_t)i * k_residues + r] = dist(rng) % p;
    }
  }
  for (int i = 0; i < tw_len; ++i) h_tw[(size_t)i] = dist(rng);

  uint32_t* d_A = nullptr;
  uint32_t* d_B = nullptr;
  uint32_t* d_O = nullptr;
  uint32_t* d_tw = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, (size_t)num_items * k_residues * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_B, (size_t)num_items * k_residues * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_O, (size_t)num_items * k_residues * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_tw, (size_t)tw_len * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), (size_t)num_items * k_residues * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), (size_t)num_items * k_residues * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_tw, h_tw.data(), (size_t)tw_len * sizeof(uint32_t), cudaMemcpyHostToDevice));

  cudaTextureObject_t tw_tex = make_u32_tex_object(d_tw, (size_t)tw_len);

  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreate(&stream));

  auto run_mode = [&](const char* name, int mode) {
    // Warmup
    if (mode == 0) {
      bench_rns_lanesplit_kernel<0><<<blocks, threads, 0, stream>>>(d_A, d_B, d_O, d_tw, tw_tex, num_items, k_residues, tw_mask, 256);
    } else if (mode == 1) {
      bench_rns_lanesplit_kernel<1><<<blocks, threads, 0, stream>>>(d_A, d_B, d_O, d_tw, tw_tex, num_items, k_residues, tw_mask, 256);
    } else {
      bench_rns_lanesplit_kernel<2><<<blocks, threads, 0, stream>>>(d_A, d_B, d_O, d_tw, tw_tex, num_items, k_residues, tw_mask, 256);
    }
    CUDA_CHECK(cudaGetLastError());

    GpuTimer t;
    t.begin(stream);
    for (int i = 0; i < launches; ++i) {
      if (mode == 0) {
        bench_rns_lanesplit_kernel<0><<<blocks, threads, 0, stream>>>(d_A, d_B, d_O, d_tw, tw_tex, num_items, k_residues, tw_mask, iters);
      } else if (mode == 1) {
        bench_rns_lanesplit_kernel<1><<<blocks, threads, 0, stream>>>(d_A, d_B, d_O, d_tw, tw_tex, num_items, k_residues, tw_mask, iters);
      } else {
        bench_rns_lanesplit_kernel<2><<<blocks, threads, 0, stream>>>(d_A, d_B, d_O, d_tw, tw_tex, num_items, k_residues, tw_mask, iters);
      }
    }
    CUDA_CHECK(cudaGetLastError());
    float ms = t.end_ms(stream);

        double total_lane_iters = (double)num_items * (double)k_residues * (double)iters * (double)launches;
        double lane_iters_per_s = total_lane_iters / (ms / 1e3);
        printf("[rns] %-12s blocks=%d threads=%d items=%d k=%d iters=%d launches=%d | %.3f ms total | %.2f G lane-iters/s\n",
          name, blocks, threads, num_items, k_residues, iters, launches, ms, lane_iters_per_s / 1e9);
  };

  printf("[rns] p_i = 2^31 - c_i, k=%d residues per item\n", k_residues);
  run_mode("global", 0);
  run_mode("ldg", 1);
  run_mode("tex", 2);

  CUDA_CHECK(cudaDestroyTextureObject(tw_tex));
  CUDA_CHECK(cudaStreamDestroy(stream));
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_O);
  cudaFree(d_tw);
}

// ------------------------ BENCH: VBS (bit-sliced add with LOP3) ------------------------
template <int Bits>
__global__ void bench_vbs_add_lop3_kernel(uint32_t* out, int iters) {
  // Each uint32_t is a bit-slice holding 32 independent 1-bit lanes.
  uint32_t a[Bits];
  uint32_t b[Bits];

  uint32_t seed = 0x9E3779B9u ^ (uint32_t)(blockIdx.x * 0x7F4A7C15u) ^ (uint32_t)threadIdx.x;
  #pragma unroll
  for (int i = 0; i < Bits; ++i) {
    // Simple xorshift-ish mixing for variety.
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    a[i] = seed ^ (0xA5A5A5A5u + (uint32_t)i * 0x3C6EF372u);
    b[i] = (seed * 0x85EBCA6Bu) ^ (0xC3C3C3C3u - (uint32_t)i * 0x27D4EB2Du);
  }

  uint32_t mix = 0;
  #pragma unroll 1
  for (int it = 0; it < iters; ++it) {
    uint32_t carry = (uint32_t)it;
    #pragma unroll
    for (int bit = 0; bit < Bits; ++bit) {
      uint32_t sum = lop3_xor3(a[bit], b[bit], carry);
      uint32_t c2 = lop3_maj3(a[bit], b[bit], carry);
      a[bit] = sum;
      carry = c2;
      mix ^= sum + (c2 ^ (uint32_t)bit);
    }
  }

  out[blockIdx.x * blockDim.x + threadIdx.x] = mix ^ a[0] ^ b[0];
}

template <int Bits>
__global__ void bench_vbs_add_bool_kernel(uint32_t* out, int iters) {
  uint32_t a[Bits];
  uint32_t b[Bits];

  uint32_t seed = 0x9E3779B9u ^ (uint32_t)(blockIdx.x * 0x7F4A7C15u) ^ (uint32_t)threadIdx.x;
  #pragma unroll
  for (int i = 0; i < Bits; ++i) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    a[i] = seed ^ (0xA5A5A5A5u + (uint32_t)i * 0x3C6EF372u);
    b[i] = (seed * 0x85EBCA6Bu) ^ (0xC3C3C3C3u - (uint32_t)i * 0x27D4EB2Du);
  }

  uint32_t mix = 0;
  #pragma unroll 1
  for (int it = 0; it < iters; ++it) {
    uint32_t carry = (uint32_t)it;
    #pragma unroll
    for (int bit = 0; bit < Bits; ++bit) {
      uint32_t sum = a[bit] ^ b[bit] ^ carry;
      uint32_t c2 = (a[bit] & b[bit]) | (a[bit] & carry) | (b[bit] & carry);
      a[bit] = sum;
      carry = c2;
      mix ^= sum + (c2 ^ (uint32_t)bit);
    }
  }

  out[blockIdx.x * blockDim.x + threadIdx.x] = mix ^ a[0] ^ b[0];
}

static void run_bench_vbs() {
  // Goal: quantify the cost of a bit-sliced ripple-carry adder over Bits bit-planes.
  // Each bit plane computes 32 independent additions (one per bit in the uint32_t mask).
  constexpr int blocks = 256;
  constexpr int threads = 256;
  constexpr int iters = 2048;

  uint32_t* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, blocks * threads * sizeof(uint32_t)));

  auto run_case = [&](int bits) {
    GpuTimer t;
    float ms_lop3 = 0.0f;
    float ms_bool = 0.0f;

    auto report = [&](const char* name, float ms) {
      double total_adds = (double)blocks * threads * (double)iters * 32.0; // 32 packed adds per thread-iter
      // Each add processes `bits` bit-planes.
      double adds_per_s = total_adds / (ms / 1e3);
      printf("[vbs] bits=%d %-8s: %.3f ms | %.2f Gadds/s (bit-sliced, 32-way)\n", bits, name, ms, adds_per_s / 1e9);
    };

    // Warmup + timed.
    if (bits == 32) {
      bench_vbs_add_lop3_kernel<32><<<blocks, threads>>>(d_out, 128);
      bench_vbs_add_bool_kernel<32><<<blocks, threads>>>(d_out, 128);
      CUDA_CHECK(cudaDeviceSynchronize());

      t.begin();
      bench_vbs_add_lop3_kernel<32><<<blocks, threads>>>(d_out, iters);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      ms_lop3 = t.end_ms();

      t.begin();
      bench_vbs_add_bool_kernel<32><<<blocks, threads>>>(d_out, iters);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      ms_bool = t.end_ms();
    } else if (bits == 64) {
      bench_vbs_add_lop3_kernel<64><<<blocks, threads>>>(d_out, 64);
      bench_vbs_add_bool_kernel<64><<<blocks, threads>>>(d_out, 64);
      CUDA_CHECK(cudaDeviceSynchronize());

      t.begin();
      bench_vbs_add_lop3_kernel<64><<<blocks, threads>>>(d_out, iters);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      ms_lop3 = t.end_ms();

      t.begin();
      bench_vbs_add_bool_kernel<64><<<blocks, threads>>>(d_out, iters);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      ms_bool = t.end_ms();
    } else if (bits == 128) {
      bench_vbs_add_lop3_kernel<128><<<blocks, threads>>>(d_out, 32);
      bench_vbs_add_bool_kernel<128><<<blocks, threads>>>(d_out, 32);
      CUDA_CHECK(cudaDeviceSynchronize());

      t.begin();
      bench_vbs_add_lop3_kernel<128><<<blocks, threads>>>(d_out, iters);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      ms_lop3 = t.end_ms();

      t.begin();
      bench_vbs_add_bool_kernel<128><<<blocks, threads>>>(d_out, iters);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      ms_bool = t.end_ms();
    } else if (bits == 256) {
      bench_vbs_add_lop3_kernel<256><<<blocks, threads>>>(d_out, 16);
      bench_vbs_add_bool_kernel<256><<<blocks, threads>>>(d_out, 16);
      CUDA_CHECK(cudaDeviceSynchronize());

      t.begin();
      bench_vbs_add_lop3_kernel<256><<<blocks, threads>>>(d_out, iters);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      ms_lop3 = t.end_ms();

      t.begin();
      bench_vbs_add_bool_kernel<256><<<blocks, threads>>>(d_out, iters);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      ms_bool = t.end_ms();
    }

    report("lop3", ms_lop3);
    report("bool", ms_bool);
  };

  printf("[vbs] blocks=%d threads=%d iters=%d (each iter does a full ripple add)\n", blocks, threads, iters);
  run_case(32);
  run_case(64);
  run_case(128);
  run_case(256);

  cudaFree(d_out);
}

// ------------------------ BENCH: FP32/INT32 pipe contention (Deep Dive §2.1) ------------------------
__device__ __forceinline__ float ptx_fma_rn_f32(float a, float b, float c) {
  float out;
  asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(out) : "f"(a), "f"(b), "f"(c));
  return out;
}

__device__ __forceinline__ int ptx_mad_lo_s32(int a, int b, int c) {
  int out;
  asm volatile("mad.lo.s32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(c));
  return out;
}

template <int ImadGroups>
__global__ void bench_pipe_mix_kernel(float* __restrict__ out, int iters) {
  // One output per warp to keep stores minimal.
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int warps_per_block = blockDim.x >> 5;
  int out_idx = blockIdx.x * warps_per_block + warp;

  float a0 = 1.001f, a1 = 1.002f, a2 = 1.003f, a3 = 1.004f;
  float b0 = 1.0001f, b1 = 0.9999f, b2 = 1.0002f, b3 = 0.9998f;
  float c0 = 0.0003f, c1 = 0.0004f, c2 = 0.0005f, c3 = 0.0006f;

  int x0 = (int)(lane + 1), x1 = (int)(lane + 3), x2 = (int)(lane + 5), x3 = (int)(lane + 7);
  int y0 = 1103515245, y1 = 12345, y2 = 214013, y3 = 2531011;
  int z0 = 1013904223, z1 = 1664525, z2 = 69069, z3 = 362437;

  auto group = [&](auto group_idx) {
    constexpr int G = decltype(group_idx)::value;
    if constexpr (G < ImadGroups) {
      x0 = ptx_mad_lo_s32(x0, y0, z0);
      x1 = ptx_mad_lo_s32(x1, y1, z1);
      x2 = ptx_mad_lo_s32(x2, y2, z2);
      x3 = ptx_mad_lo_s32(x3, y3, z3);
    } else {
      a0 = ptx_fma_rn_f32(a0, b0, c0);
      a1 = ptx_fma_rn_f32(a1, b1, c1);
      a2 = ptx_fma_rn_f32(a2, b2, c2);
      a3 = ptx_fma_rn_f32(a3, b3, c3);
    }
  };

  #pragma unroll 1
  for (int it = 0; it < iters; ++it) {
    // 16 groups * 4 ops/group = 64 instructions/iter, with a compile-time mix.
    group(std::integral_constant<int, 0>{});
    group(std::integral_constant<int, 1>{});
    group(std::integral_constant<int, 2>{});
    group(std::integral_constant<int, 3>{});
    group(std::integral_constant<int, 4>{});
    group(std::integral_constant<int, 5>{});
    group(std::integral_constant<int, 6>{});
    group(std::integral_constant<int, 7>{});
    group(std::integral_constant<int, 8>{});
    group(std::integral_constant<int, 9>{});
    group(std::integral_constant<int, 10>{});
    group(std::integral_constant<int, 11>{});
    group(std::integral_constant<int, 12>{});
    group(std::integral_constant<int, 13>{});
    group(std::integral_constant<int, 14>{});
    group(std::integral_constant<int, 15>{});
  }

  if (lane == 0) {
    // Fold int state in so the compiler can't discard either side.
    float f = (float)((x0 ^ x1 ^ x2 ^ x3) & 0xFF);
    out[out_idx] = (a0 + a1 + a2 + a3) + f;
  }
}

static void run_bench_pipe_mix() {
  const int threads = 256;
  const int blocks = 4096;
  const int iters = 4096;
  const int launches = 40;
  const int warps_per_block = threads / 32;
  const int out_elems = blocks * warps_per_block;

  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, (size_t)out_elems * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_out, 0, (size_t)out_elems * sizeof(float)));

  auto time_case = [&](const char* name, auto kernel) {
    // Warmup
    kernel<<<blocks, threads>>>(d_out, 256);
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer t;
    t.begin();
    for (int i = 0; i < launches; ++i) {
      kernel<<<blocks, threads>>>(d_out, iters);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = t.end_ms();

    // Per-iter instruction count is fixed: 64 ops.
    // We report an "effective Gops/s" so ratios are easy to compare.
    double total_ops = (double)blocks * threads * (double)iters * 64.0 * (double)launches;
    double gops = (total_ops / 1e9) / (ms / 1e3);
    printf("[pipe_mix] %s | blocks=%d threads=%d iters=%d launches=%d | %.3f ms | %.1f Gops/s (64 ops/iter)\n",
           name, blocks, threads, iters, launches, ms, gops);
  };

  printf("[pipe_mix] Validates FP32/INT32 contention on the flexible datapath (Deep Dive §2.1).\n");
  // ImadGroups is number of 4-IMAD groups out of 16 groups total.
  // So {0,4,8,12,16} => {0,16,32,48,64} IMAD ops per iter.
  time_case("ffma64 imad0", bench_pipe_mix_kernel<0>);
  time_case("ffma48 imad16", bench_pipe_mix_kernel<4>);
  time_case("ffma32 imad32", bench_pipe_mix_kernel<8>);
  time_case("ffma16 imad48", bench_pipe_mix_kernel<12>);
  time_case("ffma0  imad64", bench_pipe_mix_kernel<16>);

  cudaFree(d_out);
}

// ------------------------ BENCH: LSU+TEX dual read (Deep Dive §2.2) ------------------------
template <bool UseTex>
__global__ void bench_l1tex_dualread_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    cudaTextureObject_t B_tex,
    float* __restrict__ out,
    int n,
    int alu_iters) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  float a = A[idx];
  float b;
  if constexpr (UseTex) {
    b = tex1D<float>(B_tex, (float)idx + 0.5f);
  } else {
    b = B[idx];
  }

  float acc = a + b;
  #pragma unroll 1
  for (int k = 0; k < alu_iters; ++k) {
    acc = fmaf(acc, 1.0000001f, 0.0000003f);
    acc = fmaf(acc, 0.9999997f, 0.0000002f);
  }
  out[idx] = acc;
}

static void run_bench_l1tex_dualread() {
  printf("[l1tex_dualread] Validates LDG+TEX vs LDG+LDG in the same loop (Deep Dive §2.2).\n");

  const int threads = 256;
  std::vector<int> alu_sweep = {0, 64, 256};
  std::vector<int> n_sweep = {
      1 << 18,  // 256K floats  (~1 MiB)  : L2-friendly
      1 << 24,  // 16M floats   (~64 MiB) : exceeds L2
  };

  for (int n : n_sweep) {
    int blocks = (n + threads - 1) / threads;

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_A, 0x3f, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_B, 0x3f, (size_t)n * sizeof(float)));

    cudaTextureObject_t texB = make_1d_texture_object(d_B, (size_t)n);

    // Warmup
    bench_l1tex_dualread_kernel<false><<<blocks, threads>>>(d_A, d_B, texB, d_out, n, 64);
    bench_l1tex_dualread_kernel<true><<<blocks, threads>>>(d_A, d_B, texB, d_out, n, 64);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("[l1tex_dualread] n=%d (%.1f MiB) blocks=%d threads=%d\n",
           n, (n * sizeof(float)) / (1024.0 * 1024.0), blocks, threads);

    for (int alu_iters : alu_sweep) {
      GpuTimer t;

      t.begin();
      bench_l1tex_dualread_kernel<false><<<blocks, threads>>>(d_A, d_B, texB, d_out, n, alu_iters);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      float ms_lsu = t.end_ms();

      t.begin();
      bench_l1tex_dualread_kernel<true><<<blocks, threads>>>(d_A, d_B, texB, d_out, n, alu_iters);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      float ms_tex = t.end_ms();

      double bytes = (double)n * sizeof(float) * 2.0;  // read A and B once
      double gbps_lsu = (bytes / 1e9) / (ms_lsu / 1e3);
      double gbps_tex = (bytes / 1e9) / (ms_tex / 1e3);
      printf("[l1tex_dualread] alu_iters=%d | ldg+ldg: %.3f ms (%.1f GB/s) | ldg+tex: %.3f ms (%.1f GB/s)\n",
             alu_iters, ms_lsu, gbps_lsu, ms_tex, gbps_tex);
    }

    CUDA_CHECK(cudaDestroyTextureObject(texB));
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_out);
  }
}

// ------------------------ CLI ------------------------
