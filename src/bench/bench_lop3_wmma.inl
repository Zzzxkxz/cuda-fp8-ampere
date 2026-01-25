// ------------------------ BENCH: LOP3 ------------------------
__global__ void bench_lop3_kernel(uint32_t* out, int iters) {
  uint32_t a = 0x13579BDFu ^ (uint32_t)threadIdx.x;
  uint32_t b = 0x2468ACE0u ^ (uint32_t)(blockIdx.x * 1315423911u);
  uint32_t c = 0x0F0F0F0Fu ^ (uint32_t)(threadIdx.x * 2654435761u);

  uint32_t sum = 0;
  uint32_t carry = 0;

  #pragma unroll 4
  for (int i = 0; i < iters; ++i) {
    sum = lop3_xor3(a, b, c);
    carry = lop3_maj3(a, b, c);
    a ^= sum + 0x9E3779B9u;
    b ^= carry + 0x7F4A7C15u;
    c ^= (sum ^ carry) + 0xD1B54A35u;
  }

  out[blockIdx.x * blockDim.x + threadIdx.x] = sum ^ carry ^ a ^ b ^ c;
}

__global__ void bench_bool_kernel(uint32_t* out, int iters) {
  uint32_t a = 0x13579BDFu ^ (uint32_t)threadIdx.x;
  uint32_t b = 0x2468ACE0u ^ (uint32_t)(blockIdx.x * 1315423911u);
  uint32_t c = 0x0F0F0F0Fu ^ (uint32_t)(threadIdx.x * 2654435761u);

  uint32_t sum = 0;
  uint32_t carry = 0;

  #pragma unroll 4
  for (int i = 0; i < iters; ++i) {
    sum = a ^ b ^ c;
    carry = (a & b) | (a & c) | (b & c);
    a ^= sum + 0x9E3779B9u;
    b ^= carry + 0x7F4A7C15u;
    c ^= (sum ^ carry) + 0xD1B54A35u;
  }

  out[blockIdx.x * blockDim.x + threadIdx.x] = sum ^ carry ^ a ^ b ^ c;
}

static void run_bench_lop3() {
  constexpr int blocks = 256;
  constexpr int threads = 256;
  constexpr int iters = 1 << 15;

  uint32_t* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, blocks * threads * sizeof(uint32_t)));

  // Warmup
  bench_lop3_kernel<<<blocks, threads>>>(d_out, 1024);
  CUDA_CHECK(cudaDeviceSynchronize());

  GpuTimer t;

  t.begin();
  bench_lop3_kernel<<<blocks, threads>>>(d_out, iters);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms_lop3 = t.end_ms();

  t.begin();
  bench_bool_kernel<<<blocks, threads>>>(d_out, iters);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms_bool = t.end_ms();

  // Each iter does 2 logical results (sum + carry). We'll report ns per iter per thread.
  double total_thread_iters = (double)blocks * threads * (double)iters;
  double ns_per_iter_lop3 = (ms_lop3 * 1e6) / total_thread_iters;
  double ns_per_iter_bool = (ms_bool * 1e6) / total_thread_iters;

  printf("[lop3] blocks=%d threads=%d iters=%d\n", blocks, threads, iters);
  printf("[lop3] lop3: %.3f ms total, %.3f ns / iter / thread\n", ms_lop3, ns_per_iter_lop3);
  printf("[lop3] bool: %.3f ms total, %.3f ns / iter / thread\n", ms_bool, ns_per_iter_bool);

  cudaFree(d_out);
}

// ------------------------ BENCH: FP8 E4M3 decode + WMMA ------------------------
namespace wmma = nvcuda::wmma;

__device__ __forceinline__ half fp8_lut_decode_const(uint8_t v) {
  union {
    uint16_t u;
    half h;
  } cvt;
  cvt.u = k_fp8_e4m3_to_f16_bits[v];
  return cvt.h;
}

__device__ __forceinline__ half fp8_lut_decode_tex(cudaTextureObject_t lut_tex, uint8_t v) {
  union {
    uint16_t u;
    half h;
  } cvt;
  cvt.u = (uint16_t)tex1Dfetch<unsigned short>(lut_tex, (int)v);
  return cvt.h;
}




template <bool UseTex>
__global__ void wmma_fp16a_fp8e4m3b_gemm_kernel(
    const half* __restrict__ A,
    const uint8_t* __restrict__ B8_colmajor,
    float* __restrict__ C,
    int M, int N, int K,
    cudaTextureObject_t lut_tex,
    float scale_b) {
  // One warp computes one 16x16 tile.
  int tile_col = blockIdx.x;
  int tile_row = blockIdx.y;

  int lane = threadIdx.x & 31;

  __shared__ half Bs[16 * 16];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  for (int k0 = 0; k0 < K; k0 += 16) {
    // Decode B tile into shared memory (col-major for wmma).
    for (int i = lane; i < 256; i += 32) {
      int r = i >> 4;  // /16
      int c = i & 15;  // %16

      int b_row = k0 + r;
      int b_col = tile_col * 16 + c;
      uint8_t bv = B8_colmajor[b_col * K + b_row];
      half hb = UseTex ? fp8_lut_decode_tex(lut_tex, bv) : fp8_lut_decode_const(bv);
      float fb = __half2float(hb) * scale_b;
      // Store B tile in col-major layout for wmma.
      Bs[c * 16 + r] = __float2half(fb);
    }
    __syncwarp();

    const half* A_tile = A + (tile_row * 16) * K + k0;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;

    wmma::load_matrix_sync(a_frag, A_tile, K);
    wmma::load_matrix_sync(b_frag, Bs, 16);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
    __syncwarp();
  }

  // Store to C (row-major)
  int c_row = tile_row * 16;
  int c_col = tile_col * 16;
  wmma::store_matrix_sync(C + c_row * N + c_col, acc, N, wmma::mem_row_major);
}

// Block-tiled variant: one block computes a 32x32 output tile using 4 warps.
// This amortizes B decode across multiple output tiles within the block.
// DecodeMode:
// 0 = const LUT scalar (k_fp8_e4m3_to_f16_bits)
// 1 = texture LUT scalar (lut_tex)
// 2 = shared LUT scalar (copy const->shared once per block)
// 3 = shared LUT vector4 (uint32 loads, decode 4 rows at once)
template <int DecodeMode>
__global__ void wmma_fp16a_fp8e4m3b_gemm_kernel_tiled(
    const half* __restrict__ A,
    const uint8_t* __restrict__ B8_colmajor,
  cudaTextureObject_t b8_u32_tex,
    float* __restrict__ C,
    int M, int N, int K,
    cudaTextureObject_t lut_tex,
    float scale_b) {
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 16;

  int block_col = blockIdx.x;  // BN tile
  int block_row = blockIdx.y;  // BM tile

  int warp_id = threadIdx.x >> 5;

  // 4 warps per block: (warp_row, warp_col) in {0,1}x{0,1}
  int warp_row = (warp_id >> 1) & 1;
  int warp_col = warp_id & 1;

  __shared__ half As[BM * BK];     // row-major, ld=BK
  __shared__ half Bs[BK * BN];     // col-major, ld=BK
  __shared__ uint16_t LutS[256];

  if constexpr (DecodeMode == 2 || DecodeMode == 3) {
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
      LutS[i] = k_fp8_e4m3_to_f16_bits[i];
    }
    __syncthreads();
  }

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  int row0 = block_row * BM;
  int col0 = block_col * BN;

  for (int k0 = 0; k0 < K; k0 += BK) {
    // Load A tile into shared.
    for (int idx = threadIdx.x; idx < BM * BK; idx += blockDim.x) {
      int r = idx / BK;
      int c = idx - r * BK;
      As[idx] = A[(row0 + r) * K + (k0 + c)];
    }

    // Decode B tile into shared (col-major).
    if constexpr (DecodeMode == 5) {
      // TEX weight loads + 1-stage prefetch pipeline (distance = 1 K-tile).
      int t = threadIdx.x;
      int items = BN * (BK / 4);
      if (t < items) {
        int c = t / (BK / 4);
        int pack = t - c * (BK / 4);
        int r0 = pack * 4;

        int base_byte = (col0 + c) * K + (0 + r0);
        uint32_t w_cur = tex1Dfetch<unsigned int>(b8_u32_tex, base_byte >> 2);

        for (int kk = 0; kk < K; kk += BK) {
          // Prefetch next tile's 4 bytes early.
          uint32_t w_next = 0;
          int kk_next = kk + BK;
          if (kk_next < K) {
            int base_next = (col0 + c) * K + (kk_next + r0);
            w_next = tex1Dfetch<unsigned int>(b8_u32_tex, base_next >> 2);
          }

          // Decode current prefetched bytes into Bs.
          uint8_t v0 = (uint8_t)(w_cur & 0xFFu);
          uint8_t v1 = (uint8_t)((w_cur >> 8) & 0xFFu);
          uint8_t v2 = (uint8_t)((w_cur >> 16) & 0xFFu);
          uint8_t v3 = (uint8_t)((w_cur >> 24) & 0xFFu);

          half h0 = fp8_lut_decode_shared(LutS, v0);
          half h1 = fp8_lut_decode_shared(LutS, v1);
          half h2 = fp8_lut_decode_shared(LutS, v2);
          half h3 = fp8_lut_decode_shared(LutS, v3);

          Bs[c * BK + (r0 + 0)] = __float2half(__half2float(h0) * scale_b);
          Bs[c * BK + (r0 + 1)] = __float2half(__half2float(h1) * scale_b);
          Bs[c * BK + (r0 + 2)] = __float2half(__half2float(h2) * scale_b);
          Bs[c * BK + (r0 + 3)] = __float2half(__half2float(h3) * scale_b);

          __syncthreads();

          const half* A_tile = As + (warp_row * 16) * BK;
          const half* B_tile = Bs + (warp_col * 16) * BK;
          wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
          wmma::load_matrix_sync(a_frag, A_tile, BK);
          wmma::load_matrix_sync(b_frag, B_tile, BK);
          wmma::mma_sync(acc, a_frag, b_frag, acc);

          __syncthreads();

          // Rotate pipeline.
          w_cur = w_next;

          // Advance A tile for next iteration by having all threads load it.
          if (kk_next < K) {
            for (int idx = threadIdx.x; idx < BM * BK; idx += blockDim.x) {
              int r = idx / BK;
              int cA = idx - r * BK;
              As[idx] = A[(row0 + r) * K + (kk_next + cA)];
            }
          }
        }
      } else {
        // Threads not participating still need to help load A for the first tile.
        // (Already done above) and must participate in the barriers below, which are inside the loop.
        for (int kk = 0; kk < K; kk += BK) {
          __syncthreads();
          const half* A_tile = As + (warp_row * 16) * BK;
          const half* B_tile = Bs + (warp_col * 16) * BK;
          wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
          wmma::load_matrix_sync(a_frag, A_tile, BK);
          wmma::load_matrix_sync(b_frag, B_tile, BK);
          wmma::mma_sync(acc, a_frag, b_frag, acc);
          __syncthreads();
          int kk_next = kk + BK;
          if (kk_next < K) {
            for (int idx = threadIdx.x; idx < BM * BK; idx += blockDim.x) {
              int r = idx / BK;
              int cA = idx - r * BK;
              As[idx] = A[(row0 + r) * K + (kk_next + cA)];
            }
          }
        }
      }
      // Entire computation done in the custom loop above.
      break;
    } else if constexpr (DecodeMode == 4) {
      // TEX weight loads (no prefetch pipeline), decode 4 rows at once.
      int t = threadIdx.x;
      int items = BN * (BK / 4);
      if (t < items) {
        int c = t / (BK / 4);
        int pack = t - c * (BK / 4);
        int r0 = pack * 4;
        int base_byte = (col0 + c) * K + (k0 + r0);
        uint32_t w = tex1Dfetch<unsigned int>(b8_u32_tex, base_byte >> 2);
        uint8_t v0 = (uint8_t)(w & 0xFFu);
        uint8_t v1 = (uint8_t)((w >> 8) & 0xFFu);
        uint8_t v2 = (uint8_t)((w >> 16) & 0xFFu);
        uint8_t v3 = (uint8_t)((w >> 24) & 0xFFu);
        half h0 = fp8_lut_decode_shared(LutS, v0);
        half h1 = fp8_lut_decode_shared(LutS, v1);
        half h2 = fp8_lut_decode_shared(LutS, v2);
        half h3 = fp8_lut_decode_shared(LutS, v3);
        Bs[c * BK + (r0 + 0)] = __float2half(__half2float(h0) * scale_b);
        Bs[c * BK + (r0 + 1)] = __float2half(__half2float(h1) * scale_b);
        Bs[c * BK + (r0 + 2)] = __float2half(__half2float(h2) * scale_b);
        Bs[c * BK + (r0 + 3)] = __float2half(__half2float(h3) * scale_b);
      }
    } else if constexpr (DecodeMode == 3) {
      // Assign exactly BN * (BK/4) work-items: each loads 4 consecutive rows for one column.
      // For BK=16, that's BN*4 = 128 items (perfect match for blockDim=128 in fp8wgt).
      int t = threadIdx.x;
      int items = BN * (BK / 4);
      if (t < items) {
        int c = t / (BK / 4);
        int pack = t - c * (BK / 4);
        int r0 = pack * 4;
        const uint8_t* p = B8_colmajor + (col0 + c) * K + (k0 + r0);
        uint32_t w = *reinterpret_cast<const uint32_t*>(p);
        uint8_t v0 = (uint8_t)(w & 0xFFu);
        uint8_t v1 = (uint8_t)((w >> 8) & 0xFFu);
        uint8_t v2 = (uint8_t)((w >> 16) & 0xFFu);
        uint8_t v3 = (uint8_t)((w >> 24) & 0xFFu);

        half h0 = fp8_lut_decode_shared(LutS, v0);
        half h1 = fp8_lut_decode_shared(LutS, v1);
        half h2 = fp8_lut_decode_shared(LutS, v2);
        half h3 = fp8_lut_decode_shared(LutS, v3);

        Bs[c * BK + (r0 + 0)] = __float2half(__half2float(h0) * scale_b);
        Bs[c * BK + (r0 + 1)] = __float2half(__half2float(h1) * scale_b);
        Bs[c * BK + (r0 + 2)] = __float2half(__half2float(h2) * scale_b);
        Bs[c * BK + (r0 + 3)] = __float2half(__half2float(h3) * scale_b);
      }
    } else {
      for (int idx = threadIdx.x; idx < BK * BN; idx += blockDim.x) {
        int r = idx / BN;
        int c = idx - r * BN;
        uint8_t bv = B8_colmajor[(col0 + c) * K + (k0 + r)];
        half hb;
        if constexpr (DecodeMode == 0) {
          hb = fp8_lut_decode_const(bv);
        } else if constexpr (DecodeMode == 1) {
          hb = fp8_lut_decode_tex(lut_tex, bv);
        } else {
          hb = fp8_lut_decode_shared(LutS, bv);
        }
        float fb = __half2float(hb) * scale_b;
        Bs[c * BK + r] = __float2half(fb);
      }
    }

    __syncthreads();

    const half* A_tile = As + (warp_row * 16) * BK;
    const half* B_tile = Bs + (warp_col * 16) * BK;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::load_matrix_sync(a_frag, A_tile, BK);
    wmma::load_matrix_sync(b_frag, B_tile, BK);
    wmma::mma_sync(acc, a_frag, b_frag, acc);

    __syncthreads();
  }

  int c_row = row0 + warp_row * 16;
  int c_col = col0 + warp_col * 16;
  wmma::store_matrix_sync(C + c_row * N + c_col, acc, N, wmma::mem_row_major);
}

// ------------------------ FP16 A + FP8(E4M3) B: larger tiling variants (64x64) ------------------------
// Motivation:
// - increase tile size to amortize decode work
// - explore dataflow that reduces shared-memory pressure / synchronization
//
// Variant A: stage A into shared (like the 32x32 kernel) + decode B into shared.
// Block computes 64x64 output using 16 warps (512 threads). BK=16.
template <int DecodeMode>
__global__ void wmma_fp16a_fp8e4m3b_gemm_kernel_tiled_64x64(
    const half* __restrict__ A,
    const uint8_t* __restrict__ B8_colmajor,
    cudaTextureObject_t b8_u32_tex,
    float* __restrict__ C,
    int M, int N, int K,
    cudaTextureObject_t lut_tex,
    float scale_b) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 16;

  int block_col = (int)blockIdx.x;
  int block_row = (int)blockIdx.y;

  int warp_id = (int)threadIdx.x >> 5;
  if (warp_id >= 16) return;
  int warp_row = warp_id >> 2;  // 0..3
  int warp_col = warp_id & 3;   // 0..3

  __shared__ half As[BM * BK];
  __shared__ half Bs[BK * BN];
  __shared__ uint16_t LutS[256];

  if constexpr (DecodeMode == 2 || DecodeMode == 3 || DecodeMode == 4) {
    for (int i = (int)threadIdx.x; i < 256; i += (int)blockDim.x) {
      LutS[i] = k_fp8_e4m3_to_f16_bits[i];
    }
  }

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  int row0 = block_row * BM;
  int col0 = block_col * BN;

  for (int k0 = 0; k0 < K; k0 += BK) {
    // Load A tile into shared.
    for (int idx = (int)threadIdx.x; idx < BM * BK; idx += (int)blockDim.x) {
      int r = idx / BK;
      int c = idx - r * BK;
      int ar = row0 + r;
      int ac = k0 + c;
      As[idx] = (ar < M && ac < K) ? A[(size_t)ar * K + (size_t)ac] : __float2half(0.0f);
    }

    // Decode B tile into shared (col-major).
    if constexpr (DecodeMode == 4) {
      // TEX weight loads (u32), decode 4 rows at once.
      int t = (int)threadIdx.x;
      int items = BN * (BK / 4);  // 64 * 4 = 256
      if (t < items) {
        int c = t / (BK / 4);
        int pack = t - c * (BK / 4);
        int r0 = pack * 4;
        int base_byte = (col0 + c) * K + (k0 + r0);
        uint32_t w = tex1Dfetch<unsigned int>(b8_u32_tex, base_byte >> 2);
        uint8_t v0 = (uint8_t)(w & 0xFFu);
        uint8_t v1 = (uint8_t)((w >> 8) & 0xFFu);
        uint8_t v2 = (uint8_t)((w >> 16) & 0xFFu);
        uint8_t v3 = (uint8_t)((w >> 24) & 0xFFu);
        half h0 = fp8_lut_decode_shared(LutS, v0);
        half h1 = fp8_lut_decode_shared(LutS, v1);
        half h2 = fp8_lut_decode_shared(LutS, v2);
        half h3 = fp8_lut_decode_shared(LutS, v3);
        Bs[c * BK + (r0 + 0)] = __float2half(__half2float(h0) * scale_b);
        Bs[c * BK + (r0 + 1)] = __float2half(__half2float(h1) * scale_b);
        Bs[c * BK + (r0 + 2)] = __float2half(__half2float(h2) * scale_b);
        Bs[c * BK + (r0 + 3)] = __float2half(__half2float(h3) * scale_b);
      }
    } else if constexpr (DecodeMode == 3) {
      // Global u32 loads, decode 4 rows at once.
      int t = (int)threadIdx.x;
      int items = BN * (BK / 4);
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
        half h0 = fp8_lut_decode_shared(LutS, v0);
        half h1 = fp8_lut_decode_shared(LutS, v1);
        half h2 = fp8_lut_decode_shared(LutS, v2);
        half h3 = fp8_lut_decode_shared(LutS, v3);
        Bs[c * BK + (r0 + 0)] = __float2half(__half2float(h0) * scale_b);
        Bs[c * BK + (r0 + 1)] = __float2half(__half2float(h1) * scale_b);
        Bs[c * BK + (r0 + 2)] = __float2half(__half2float(h2) * scale_b);
        Bs[c * BK + (r0 + 3)] = __float2half(__half2float(h3) * scale_b);
      }
    } else {
      // Scalar decode path (const/tex/shared LUT)
      for (int idx = (int)threadIdx.x; idx < BK * BN; idx += (int)blockDim.x) {
        int r = idx / BN;
        int c = idx - r * BN;
        uint8_t bv = B8_colmajor[(size_t)(col0 + c) * K + (k0 + r)];
        half hb;
        if constexpr (DecodeMode == 0) {
          hb = fp8_lut_decode_const(bv);
        } else if constexpr (DecodeMode == 1) {
          hb = fp8_lut_decode_tex(lut_tex, bv);
        } else {
          hb = fp8_lut_decode_shared(LutS, bv);
        }
        Bs[c * BK + r] = __float2half(__half2float(hb) * scale_b);
      }
    }

    __syncthreads();

    const half* A_tile = As + (warp_row * 16) * BK;
    const half* B_tile = Bs + (warp_col * 16) * BK;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::load_matrix_sync(a_frag, A_tile, BK);
    wmma::load_matrix_sync(b_frag, B_tile, BK);
    wmma::mma_sync(acc, a_frag, b_frag, acc);

    __syncthreads();
  }

  int c_row = row0 + warp_row * 16;
  int c_col = col0 + warp_col * 16;
  if (c_row < M && c_col < N) {
    wmma::store_matrix_sync(C + (size_t)c_row * N + (size_t)c_col, acc, N, wmma::mem_row_major);
  }
}

// Variant B: do NOT stage A into shared; each warp loads A directly from global.
// This removes the A shared array and may reduce barriers/shared pressure.
template <int DecodeMode>
__global__ void wmma_fp16a_fp8e4m3b_gemm_kernel_tiled_64x64_noAs(
    const half* __restrict__ A,
    const uint8_t* __restrict__ B8_colmajor,
    cudaTextureObject_t b8_u32_tex,
    float* __restrict__ C,
    int M, int N, int K,
    cudaTextureObject_t lut_tex,
    float scale_b) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 16;

  int block_col = (int)blockIdx.x;
  int block_row = (int)blockIdx.y;
  int warp_id = (int)threadIdx.x >> 5;
  if (warp_id >= 16) return;
  int warp_row = warp_id >> 2;
  int warp_col = warp_id & 3;

  __shared__ half Bs[BK * BN];
  __shared__ uint16_t LutS[256];
  if constexpr (DecodeMode == 2 || DecodeMode == 3 || DecodeMode == 4) {
    for (int i = (int)threadIdx.x; i < 256; i += (int)blockDim.x) {
      LutS[i] = k_fp8_e4m3_to_f16_bits[i];
    }
  }

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  int row0 = block_row * BM;
  int col0 = block_col * BN;
  int c_row = row0 + warp_row * 16;
  int c_col = col0 + warp_col * 16;

  // Pipeline state for DecodeMode == 5 (Dist=1)
  uint32_t w_next = 0;
  if constexpr (DecodeMode == 5) {
    int t = (int)threadIdx.x;
    int items = BN * (BK / 4);
    if (t < items) {
      int c = t / (BK / 4);
      int pack = t - c * (BK / 4);
      int r0 = pack * 4;
      // Prefetch for k0=0
      int base_byte = (col0 + c) * K + (0 + r0);
      w_next = tex1Dfetch<unsigned int>(b8_u32_tex, base_byte >> 2);
    }
  }

  for (int k0 = 0; k0 < K; k0 += BK) {
    // Decode B tile into shared.
    if constexpr (DecodeMode == 5) {
      // Pipelined path (Dist=1)
      int t = (int)threadIdx.x;
      int items = BN * (BK / 4);
      if (t < items) {
        int c = t / (BK / 4);
        int pack = t - c * (BK / 4);
        int r0 = pack * 4;
        
        uint32_t w = w_next;
        // Prefetch for next tile
        if (k0 + BK < K) {
          int base_byte = (col0 + c) * K + (k0 + BK + r0);
          w_next = tex1Dfetch<unsigned int>(b8_u32_tex, base_byte >> 2);
        }

        // Decode w
        uint8_t v0 = (uint8_t)(w & 0xFFu);
        uint8_t v1 = (uint8_t)((w >> 8) & 0xFFu);
        uint8_t v2 = (uint8_t)((w >> 16) & 0xFFu);
        uint8_t v3 = (uint8_t)((w >> 24) & 0xFFu);
        half h0 = fp8_lut_decode_shared(LutS, v0);
        half h1 = fp8_lut_decode_shared(LutS, v1);
        half h2 = fp8_lut_decode_shared(LutS, v2);
        half h3 = fp8_lut_decode_shared(LutS, v3);
        Bs[c * BK + (r0 + 0)] = __float2half(__half2float(h0) * scale_b);
        Bs[c * BK + (r0 + 1)] = __float2half(__half2float(h1) * scale_b);
        Bs[c * BK + (r0 + 2)] = __float2half(__half2float(h2) * scale_b);
        Bs[c * BK + (r0 + 3)] = __float2half(__half2float(h3) * scale_b);
      }
    } else if constexpr (DecodeMode == 4) {
      int t = (int)threadIdx.x;
      int items = BN * (BK / 4);
      if (t < items) {
        int c = t / (BK / 4);
        int pack = t - c * (BK / 4);
        int r0 = pack * 4;
        int base_byte = (col0 + c) * K + (k0 + r0);
        uint32_t w = tex1Dfetch<unsigned int>(b8_u32_tex, base_byte >> 2);
        uint8_t v0 = (uint8_t)(w & 0xFFu);
        uint8_t v1 = (uint8_t)((w >> 8) & 0xFFu);
        uint8_t v2 = (uint8_t)((w >> 16) & 0xFFu);
        uint8_t v3 = (uint8_t)((w >> 24) & 0xFFu);
        half h0 = fp8_lut_decode_shared(LutS, v0);
        half h1 = fp8_lut_decode_shared(LutS, v1);
        half h2 = fp8_lut_decode_shared(LutS, v2);
        half h3 = fp8_lut_decode_shared(LutS, v3);
        Bs[c * BK + (r0 + 0)] = __float2half(__half2float(h0) * scale_b);
        Bs[c * BK + (r0 + 1)] = __float2half(__half2float(h1) * scale_b);
        Bs[c * BK + (r0 + 2)] = __float2half(__half2float(h2) * scale_b);
        Bs[c * BK + (r0 + 3)] = __float2half(__half2float(h3) * scale_b);
      }
    } else if constexpr (DecodeMode == 3) {
      int t = (int)threadIdx.x;
      int items = BN * (BK / 4);
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
        half h0 = fp8_lut_decode_shared(LutS, v0);
        half h1 = fp8_lut_decode_shared(LutS, v1);
        half h2 = fp8_lut_decode_shared(LutS, v2);
        half h3 = fp8_lut_decode_shared(LutS, v3);
        Bs[c * BK + (r0 + 0)] = __float2half(__half2float(h0) * scale_b);
        Bs[c * BK + (r0 + 1)] = __float2half(__half2float(h1) * scale_b);
        Bs[c * BK + (r0 + 2)] = __float2half(__half2float(h2) * scale_b);
        Bs[c * BK + (r0 + 3)] = __float2half(__half2float(h3) * scale_b);
      }
    } else {
      for (int idx = (int)threadIdx.x; idx < BK * BN; idx += (int)blockDim.x) {
        int r = idx / BN;
        int c = idx - r * BN;
        uint8_t bv = B8_colmajor[(size_t)(col0 + c) * K + (k0 + r)];
        half hb;
        if constexpr (DecodeMode == 0) hb = fp8_lut_decode_const(bv);
        else if constexpr (DecodeMode == 1) hb = fp8_lut_decode_tex(lut_tex, bv);
        else hb = fp8_lut_decode_shared(LutS, bv);
        Bs[c * BK + r] = __float2half(__half2float(hb) * scale_b);
      }
    }

    __syncthreads();

    // A pointer for this warp's 16x16 tile, loaded directly from global.
    const half* A_tile = A + (size_t)c_row * K + (size_t)k0;
    const half* B_tile = Bs + (warp_col * 16) * BK;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::load_matrix_sync(a_frag, A_tile, K);
    wmma::load_matrix_sync(b_frag, B_tile, BK);
    wmma::mma_sync(acc, a_frag, b_frag, acc);

    __syncthreads();
  }

  if (c_row < M && c_col < N) {
    wmma::store_matrix_sync(C + (size_t)c_row * N + (size_t)c_col, acc, N, wmma::mem_row_major);
  }
}

// Variant C: 64x64 noAs with cp.async double-buffering for the FP8 B tile.
// This overlaps global->shared B8 copies for k0+BK with WMMA compute for k0.
// Decode still happens in a separate phase (not overlapped), but the B8 load is.
__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.commit_group;" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait0() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  asm volatile("cp.async.wait_group 0;" ::);
#endif
}

__device__ __forceinline__ void cp_async_cg_16B(void* smem_ptr, const void* gmem_ptr, bool pred) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  unsigned smem_u32 = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.b32 p, %2, 0;\n"
      "  @p cp.async.cg.shared.global [%0], [%1], 16;\n"
      "}\n"
      :: "r"(smem_u32), "l"(gmem_ptr), "r"((int)pred));
#else
  if (pred) {
    // Fallback: regular 16B load+store.
    *reinterpret_cast<uint4*>(smem_ptr) = *reinterpret_cast<const uint4*>(gmem_ptr);
  }
#endif
}

template <int DecodeMode>
__global__ void wmma_fp16a_fp8e4m3b_gemm_kernel_tiled_64x64_noAs_cpasync(
    const half* __restrict__ A,
    const uint8_t* __restrict__ B8_colmajor,
    float* __restrict__ C,
    int M, int N, int K,
    cudaTextureObject_t lut_tex,
    float scale_b) {
  (void)lut_tex;
  // This kernel is currently wired for shared-LUT decode (no texture LUT).
  static_assert(DecodeMode == 2, "cpasync kernel currently expects shared-LUT decode mode (2)");

  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 16;
  constexpr int BYTES_B_TILE = BK * BN;  // 1024 bytes

  int block_col = (int)blockIdx.x;
  int block_row = (int)blockIdx.y;
  int warp_id = (int)threadIdx.x >> 5;
  if (warp_id >= 16) return;
  int warp_row = warp_id >> 2;
  int warp_col = warp_id & 3;

  // Double-buffered B8 staging (uint4 == 16B) + decoded half tile (as 16B segments).
  __shared__ uint4 B8s[2][BYTES_B_TILE / 16];
  __shared__ __align__(16) uint4 Bs4[2][(BK * BN) / 8];
  __shared__ uint16_t LutS[256];

  for (int i = (int)threadIdx.x; i < 256; i += (int)blockDim.x) {
    LutS[i] = k_fp8_e4m3_to_f16_bits[i];
  }
  __syncthreads();

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  int row0 = block_row * BM;
  int col0 = block_col * BN;
  int c_row = row0 + warp_row * 16;
  int c_col = col0 + warp_col * 16;

  auto stage_b8_tile = [&](int stage, int k0) {
    // One cp.async per column: copy 16 bytes (BK) for that column.
    int t = (int)threadIdx.x;
    if (t < BN) {
      int c = t;
      bool pred = (col0 + c) < N && (k0 + (BK - 1)) < K;
      const void* gptr = (const void*)(B8_colmajor + (size_t)(col0 + c) * K + k0);
      void* sptr = (void*)(&B8s[stage][c]);
      cp_async_cg_16B(sptr, gptr, pred);
    }
    cp_async_commit();
  };

  auto decode_b8_to_half = [&](int stage) {
    // Decode from shared B8s[stage] into shared Bs4[stage] (col-major half).
    // Only the loader threads participate (t < BN). This allows us to drop the
    // pre-decode __syncthreads() and keep only a single barrier after decode.
    int c = (int)threadIdx.x;
    if (c < BN) {
      const uint8_t* p = reinterpret_cast<const uint8_t*>(&B8s[stage][c]);
      half scale_h = __float2half(scale_b);

      auto pack2 = [](half a, half b) -> uint32_t {
        __half2 h2p = __halves2half2(a, b);
        return *reinterpret_cast<uint32_t*>(&h2p);
      };

      // rows [0..7]
      {
        half h0 = __hmul(fp8_lut_decode_shared(LutS, p[0]), scale_h);
        half h1 = __hmul(fp8_lut_decode_shared(LutS, p[1]), scale_h);
        half h2 = __hmul(fp8_lut_decode_shared(LutS, p[2]), scale_h);
        half h3 = __hmul(fp8_lut_decode_shared(LutS, p[3]), scale_h);
        half h4 = __hmul(fp8_lut_decode_shared(LutS, p[4]), scale_h);
        half h5 = __hmul(fp8_lut_decode_shared(LutS, p[5]), scale_h);
        half h6 = __hmul(fp8_lut_decode_shared(LutS, p[6]), scale_h);
        half h7 = __hmul(fp8_lut_decode_shared(LutS, p[7]), scale_h);
        uint4 out;
        out.x = pack2(h0, h1);
        out.y = pack2(h2, h3);
        out.z = pack2(h4, h5);
        out.w = pack2(h6, h7);
        Bs4[stage][2 * c + 0] = out;
      }

      // rows [8..15]
      {
        half h0 = __hmul(fp8_lut_decode_shared(LutS, p[8]), scale_h);
        half h1 = __hmul(fp8_lut_decode_shared(LutS, p[9]), scale_h);
        half h2 = __hmul(fp8_lut_decode_shared(LutS, p[10]), scale_h);
        half h3 = __hmul(fp8_lut_decode_shared(LutS, p[11]), scale_h);
        half h4 = __hmul(fp8_lut_decode_shared(LutS, p[12]), scale_h);
        half h5 = __hmul(fp8_lut_decode_shared(LutS, p[13]), scale_h);
        half h6 = __hmul(fp8_lut_decode_shared(LutS, p[14]), scale_h);
        half h7 = __hmul(fp8_lut_decode_shared(LutS, p[15]), scale_h);
        uint4 out;
        out.x = pack2(h0, h1);
        out.y = pack2(h2, h3);
        out.z = pack2(h4, h5);
        out.w = pack2(h6, h7);
        Bs4[stage][2 * c + 1] = out;
      }
    }
  };

  // Prime stage 0.
  int stage = 0;
  int k0 = 0;
  stage_b8_tile(stage, k0);
  cp_async_wait0();
  decode_b8_to_half(stage);
  __syncthreads();

  for (k0 = 0; k0 < K; k0 += BK) {
    int next_k0 = k0 + BK;
    int next_stage = stage ^ 1;
    if (next_k0 < K) {
      stage_b8_tile(next_stage, next_k0);
    }

    // WMMA compute for current stage.
    const half* A_tile = A + (size_t)c_row * K + (size_t)k0;
    const half* B_tile = reinterpret_cast<const half*>(Bs4[stage]) + (warp_col * 16) * BK;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::load_matrix_sync(a_frag, A_tile, K);
    wmma::load_matrix_sync(b_frag, B_tile, BK);
    wmma::mma_sync(acc, a_frag, b_frag, acc);

    if (next_k0 < K) {
      // Finish the async copy, decode next stage, then sync once before the next iteration.
      cp_async_wait0();
      decode_b8_to_half(next_stage);
      __syncthreads();
      stage = next_stage;
    }
  }

  if (c_row < M && c_col < N) {
    wmma::store_matrix_sync(C + (size_t)c_row * N + (size_t)c_col, acc, N, wmma::mem_row_major);
  }
}

__global__ void wmma_f16_gemm_kernel_tiled(
    const half* __restrict__ A,
    const half* __restrict__ B_colmajor,
    float* __restrict__ C,
    int M, int N, int K) {
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 16;

  int block_col = blockIdx.x;
  int block_row = blockIdx.y;

  int warp_id = threadIdx.x >> 5;
  int warp_row = (warp_id >> 1) & 1;
  int warp_col = warp_id & 1;

  __shared__ half As[BM * BK];
  __shared__ half Bs[BK * BN];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  int row0 = block_row * BM;
  int col0 = block_col * BN;

  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int idx = threadIdx.x; idx < BM * BK; idx += blockDim.x) {
      int r = idx / BK;
      int c = idx - r * BK;
      As[idx] = A[(row0 + r) * K + (k0 + c)];
    }
    for (int idx = threadIdx.x; idx < BK * BN; idx += blockDim.x) {
      int r = idx / BN;
      int c = idx - r * BN;
      // B_colmajor is KxN with ld=K, so element (r,c) in the BKxBN tile is:
      Bs[c * BK + r] = B_colmajor[(col0 + c) * K + (k0 + r)];
    }

    __syncthreads();

    const half* A_tile = As + (warp_row * 16) * BK;
    const half* B_tile = Bs + (warp_col * 16) * BK;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::load_matrix_sync(a_frag, A_tile, BK);
    wmma::load_matrix_sync(b_frag, B_tile, BK);
    wmma::mma_sync(acc, a_frag, b_frag, acc);

    __syncthreads();
  }

  int c_row = row0 + warp_row * 16;
  int c_col = col0 + warp_col * 16;
  wmma::store_matrix_sync(C + c_row * N + c_col, acc, N, wmma::mem_row_major);
}

template <bool UseTex>
__global__ void wmma_fp8e4m3_gemm_kernel(
    const uint8_t* __restrict__ A8,
    const uint8_t* __restrict__ B8_colmajor,
    float* __restrict__ C,
    int M, int N, int K,
    cudaTextureObject_t lut_tex) {
  // One warp computes one 16x16 tile.
  int tile_col = blockIdx.x;
  int tile_row = blockIdx.y;

  int lane = threadIdx.x & 31;

  __shared__ half As[16 * 16];
  __shared__ half Bs[16 * 16];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  for (int k0 = 0; k0 < K; k0 += 16) {
    // Decode A and B tiles into shared memory.
    for (int i = lane; i < 256; i += 32) {
      int r = i >> 4;      // /16
      int c = i & 15;      // %16

      int a_row = tile_row * 16 + r;
      int a_col = k0 + c;
      uint8_t av = A8[a_row * K + a_col];
      As[i] = UseTex ? fp8_lut_decode_tex(lut_tex, av) : fp8_lut_decode_const(av);

      int b_row = k0 + r;
      int b_col = tile_col * 16 + c;
      uint8_t bv = B8_colmajor[b_col * K + b_row];
      // Store B tile in col-major layout for wmma.
      Bs[c * 16 + r] = UseTex ? fp8_lut_decode_tex(lut_tex, bv) : fp8_lut_decode_const(bv);
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;

    wmma::load_matrix_sync(a_frag, As, 16);
    wmma::load_matrix_sync(b_frag, Bs, 16);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
    __syncwarp();
  }

  // Store to C (row-major)
  int c_row = tile_row * 16;
  int c_col = tile_col * 16;
  // Each warp stores the full tile.
  wmma::store_matrix_sync(C + c_row * N + c_col, acc, N, wmma::mem_row_major);
}

__global__ void wmma_f16_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B_colmajor,
    float* __restrict__ C,
    int M, int N, int K) {
  int tile_col = blockIdx.x;
  int tile_row = blockIdx.y;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  for (int k0 = 0; k0 < K; k0 += 16) {
    const half* A_tile = A + (tile_row * 16) * K + k0;
    const half* B_tile = B_colmajor + (tile_col * 16) * K + k0;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::load_matrix_sync(a_frag, A_tile, K);
    wmma::load_matrix_sync(b_frag, B_tile, K);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
  }

  int c_row = tile_row * 16;
  int c_col = tile_col * 16;
  wmma::store_matrix_sync(C + c_row * N + c_col, acc, N, wmma::mem_row_major);
}

__global__ void decode_fp8e4m3_to_f16_kernel(const uint8_t* __restrict__ in8, half* __restrict__ out16, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out16[idx] = fp8_lut_decode_const(in8[idx]);
}

// Forward decl (defined later in the FP8 quantization section).
__global__ void dequantize_fp8e4m3_to_fp16_kernel(
    const uint8_t* __restrict__ in8,
    half* __restrict__ out16,
    int n,
    float scale);

static void run_bench_fp8e4m3() {
  upload_fp8_lut();

  // Keep it simple: square sizes divisible by 16.
  constexpr int M = 1024;
  constexpr int N = 1024;
  constexpr int K = 1024;

  size_t bytes_A8 = (size_t)M * K;
  size_t bytes_B8 = (size_t)K * N;
  size_t bytes_Ah = (size_t)M * K * sizeof(half);
  size_t bytes_Bh = (size_t)K * N * sizeof(half);
  size_t bytes_C = (size_t)M * N * sizeof(float);

  std::vector<uint8_t> h_A8(bytes_A8);
  std::vector<uint8_t> h_B8(bytes_B8);

  // Random FP8 bytes; for perf purposes we mostly care about decode+compute throughput.
  std::mt19937 rng(123);
  std::uniform_int_distribution<int> dist(0, 255);
  for (size_t i = 0; i < bytes_A8; ++i) h_A8[i] = (uint8_t)dist(rng);
  for (size_t i = 0; i < bytes_B8; ++i) h_B8[i] = (uint8_t)dist(rng);

  // Build FP16 equivalents on host using the same LUT semantics.
  std::vector<half> h_Ah((size_t)M * K);
  std::vector<half> h_Bh((size_t)K * N);
  for (int i = 0; i < M * K; ++i) {
    float f = fp8_e4m3_to_f32(h_A8[(size_t)i]);
    h_Ah[(size_t)i] = __float2half(f);
  }
  // B is stored col-major on device for wmma.
  for (int r = 0; r < K; ++r) {
    for (int c = 0; c < N; ++c) {
      uint8_t v = h_B8[(size_t)r * N + c];
      float f = fp8_e4m3_to_f32(v);
      h_Bh[(size_t)c * K + r] = __float2half(f);
    }
  }

  uint8_t* d_A8 = nullptr;
  uint8_t* d_B8_col = nullptr;
  half* d_Ah = nullptr;
  half* d_Bh_col = nullptr;
  half* d_Ah_upcast = nullptr;
  half* d_Bh_upcast = nullptr;
  float* d_C = nullptr;
  uint16_t* d_lut = nullptr;

  CUDA_CHECK(cudaMalloc(&d_A8, bytes_A8));
  CUDA_CHECK(cudaMalloc(&d_B8_col, bytes_B8));
  CUDA_CHECK(cudaMalloc(&d_Ah, bytes_Ah));
  CUDA_CHECK(cudaMalloc(&d_Bh_col, bytes_Bh));
  CUDA_CHECK(cudaMalloc(&d_Ah_upcast, bytes_Ah));
  CUDA_CHECK(cudaMalloc(&d_Bh_upcast, bytes_Bh));
  CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
  CUDA_CHECK(cudaMalloc(&d_lut, 256 * sizeof(uint16_t)));

  // Upload A8 row-major
  CUDA_CHECK(cudaMemcpy(d_A8, h_A8.data(), bytes_A8, cudaMemcpyHostToDevice));
  // Upload B8 as col-major on device
  {
    std::vector<uint8_t> h_B8_col((size_t)K * N);
    for (int r = 0; r < K; ++r) {
      for (int c = 0; c < N; ++c) {
        h_B8_col[(size_t)c * K + r] = h_B8[(size_t)r * N + c];
      }
    }
    CUDA_CHECK(cudaMemcpy(d_B8_col, h_B8_col.data(), bytes_B8, cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMemcpy(d_Ah, h_Ah.data(), bytes_Ah, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Bh_col, h_Bh.data(), bytes_Bh, cudaMemcpyHostToDevice));

  // Upload LUT to global for texture path.
  {
    uint16_t host_lut[256];
    for (int i = 0; i < 256; ++i) host_lut[i] = f32_to_f16_bits(fp8_e4m3_to_f32((uint8_t)i));
    CUDA_CHECK(cudaMemcpy(d_lut, host_lut, 256 * sizeof(uint16_t), cudaMemcpyHostToDevice));
  }
  cudaTextureObject_t lut_tex = make_lut_texture_object_u16(d_lut);

  dim3 grid(N / 16, M / 16);
  dim3 block(32, 1, 1);

  // Warmup
  wmma_f16_gemm_kernel<<<grid, block>>>(d_Ah, d_Bh_col, d_C, M, N, K);
  wmma_fp8e4m3_gemm_kernel<false><<<grid, block>>>(d_A8, d_B8_col, d_C, M, N, K, 0);
  wmma_fp8e4m3_gemm_kernel<true><<<grid, block>>>(d_A8, d_B8_col, d_C, M, N, K, lut_tex);
  decode_fp8e4m3_to_f16_kernel<<<(M * K + 255) / 256, 256>>>(d_A8, d_Ah_upcast, M * K);
  decode_fp8e4m3_to_f16_kernel<<<(K * N + 255) / 256, 256>>>(d_B8_col, d_Bh_upcast, K * N);
  wmma_f16_gemm_kernel<<<grid, block>>>(d_Ah_upcast, d_Bh_upcast, d_C, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());

  auto report = [&](const char* name, float ms) {
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = (flops / (ms / 1e3)) / 1e12;
    printf("[fp8e4m3] %s: %.3f ms  (%.2f TFLOP/s)\n", name, ms, tflops);
  };

  GpuTimer t;

  t.begin();
  wmma_f16_gemm_kernel<<<grid, block>>>(d_Ah, d_Bh_col, d_C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  report("fp16_wmma", t.end_ms());

  // Measure decode-only cost for the naive upcast path.
  float ms_decode_only = 0.0f;
  {
    t.begin();
    decode_fp8e4m3_to_f16_kernel<<<(M * K + 255) / 256, 256>>>(d_A8, d_Ah_upcast, M * K);
    decode_fp8e4m3_to_f16_kernel<<<(K * N + 255) / 256, 256>>>(d_B8_col, d_Bh_upcast, K * N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    ms_decode_only = t.end_ms();
    printf("[fp8e4m3] decode_only (global fp16 outputs): %.3f ms\n", ms_decode_only);
  }

  // Library baseline: cuBLAS FP16 GEMM using tensor cores.
  {
    cublasHandle_t handle{};
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    CUBLAS_CHECK(cublasSetStream(handle, 0));

    float alpha = 1.0f;
    float beta = 0.0f;

    // cuBLAS is column-major. We reuse d_Ah (row-major MxK) by interpreting it as
    // a column-major KxM matrix and using op(A)=T.
    // B is already stored column-major (KxN) in d_Bh_col.

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
    CUDA_CHECK(cudaDeviceSynchronize());

    t.begin();
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
    CUDA_CHECK(cudaDeviceSynchronize());
    report("cublas_fp16_tensorop", t.end_ms());

    // Naive (but strong) baseline: upcast to global FP16, then let cuBLAS do the GEMM.
    t.begin();
    decode_fp8e4m3_to_f16_kernel<<<(M * K + 255) / 256, 256>>>(d_A8, d_Ah_upcast, M * K);
    decode_fp8e4m3_to_f16_kernel<<<(K * N + 255) / 256, 256>>>(d_B8_col, d_Bh_upcast, K * N);
    CUBLAS_CHECK(cublasGemmEx(
      handle,
      CUBLAS_OP_T, CUBLAS_OP_N,
      M, N, K,
      &alpha,
      d_Ah_upcast, CUDA_R_16F, K,
      d_Bh_upcast, CUDA_R_16F, K,
      &beta,
      d_C, CUDA_R_32F, M,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CUDA_CHECK(cudaDeviceSynchronize());
    report("fp8->fp16 upcast (global) + cublas_tensorop", t.end_ms());

    cublasDestroy(handle);
  }

  // Naive baseline: split decode (to global FP16) + FP16 WMMA.
  t.begin();
  decode_fp8e4m3_to_f16_kernel<<<(M * K + 255) / 256, 256>>>(d_A8, d_Ah_upcast, M * K);
  decode_fp8e4m3_to_f16_kernel<<<(K * N + 255) / 256, 256>>>(d_B8_col, d_Bh_upcast, K * N);
  wmma_f16_gemm_kernel<<<grid, block>>>(d_Ah_upcast, d_Bh_upcast, d_C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  report("fp8->fp16 upcast (global) + fp16_wmma", t.end_ms());

  t.begin();
  wmma_fp8e4m3_gemm_kernel<false><<<grid, block>>>(d_A8, d_B8_col, d_C, M, N, K, 0);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  report("fp8->fp16 (const LUT) + wmma", t.end_ms());

  t.begin();
  wmma_fp8e4m3_gemm_kernel<true><<<grid, block>>>(d_A8, d_B8_col, d_C, M, N, K, lut_tex);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  report("fp8->fp16 (tex LUT) + wmma", t.end_ms());

  CUDA_CHECK(cudaDestroyTextureObject(lut_tex));
  cudaFree(d_A8);
  cudaFree(d_B8_col);
  cudaFree(d_Ah);
  cudaFree(d_Bh_col);
  cudaFree(d_Ah_upcast);
  cudaFree(d_Bh_upcast);
  cudaFree(d_C);
  cudaFree(d_lut);
}

// ------------------------ BENCH: FP16 A + FP8(E4M3) B fused decode + WMMA ------------------------
static void run_bench_fp8wgt() {
  upload_fp8_lut();

  // Inference-like: activations in FP16, weights stored as FP8 (E4M3).
  // Keep sizes divisible by 16 for WMMA.
  constexpr int M = 1024;
  constexpr int N = 1024;
  constexpr int K = 1024;
  constexpr int repeats = 50;
  constexpr float scale_b = 1.0f;

  size_t bytes_Ah = (size_t)M * K * sizeof(half);
  size_t bytes_B8 = (size_t)K * N;
  size_t bytes_B8_u32 = (size_t)K * N / 4 * sizeof(uint32_t);
  size_t bytes_Bh = (size_t)K * N * sizeof(half);
  size_t bytes_C = (size_t)M * N * sizeof(float);

  // Host data.
  std::vector<half> h_Ah((size_t)M * K);
  std::vector<uint8_t> h_B8((size_t)K * N);
  std::vector<half> h_Bh_col((size_t)K * N);

  std::mt19937 rng(123);
  std::normal_distribution<float> nd(0.0f, 0.5f);
  std::uniform_int_distribution<int> bd(0, 255);

  for (int i = 0; i < M * K; ++i) h_Ah[(size_t)i] = __float2half(nd(rng));
  for (int r = 0; r < K; ++r) {
    for (int c = 0; c < N; ++c) {
      uint8_t v = (uint8_t)bd(rng);
      h_B8[(size_t)r * N + c] = v;
      float f = fp8_e4m3_to_f32(v) * scale_b;
      h_Bh_col[(size_t)c * K + r] = __float2half(f);
    }
  }

  half* d_Ah = nullptr;
  uint8_t* d_B8_col = nullptr;
  uint32_t* d_B8_col_u32 = nullptr;
  half* d_Bh_col = nullptr;
  half* d_Bh_upcast = nullptr;
  float* d_C = nullptr;
  uint16_t* d_lut = nullptr;

  CUDA_CHECK(cudaMalloc(&d_Ah, bytes_Ah));
  CUDA_CHECK(cudaMalloc(&d_B8_col, bytes_B8));
  CUDA_CHECK(cudaMalloc(&d_B8_col_u32, bytes_B8_u32));
  CUDA_CHECK(cudaMalloc(&d_Bh_col, bytes_Bh));
  CUDA_CHECK(cudaMalloc(&d_Bh_upcast, bytes_Bh));
  CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
  CUDA_CHECK(cudaMalloc(&d_lut, 256 * sizeof(uint16_t)));

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

    // Also upload a packed u32 view for TEX-weight loads (4 bytes per uint32).
    std::vector<uint32_t> h_B8_col_u32((size_t)K * N / 4);
    for (size_t i = 0; i < (size_t)K * N / 4; ++i) {
      uint32_t w = 0;
      w |= (uint32_t)h_B8_col[i * 4 + 0] << 0;
      w |= (uint32_t)h_B8_col[i * 4 + 1] << 8;
      w |= (uint32_t)h_B8_col[i * 4 + 2] << 16;
      w |= (uint32_t)h_B8_col[i * 4 + 3] << 24;
      h_B8_col_u32[i] = w;
    }
    CUDA_CHECK(cudaMemcpy(d_B8_col_u32, h_B8_col_u32.data(), bytes_B8_u32, cudaMemcpyHostToDevice));
  }

  // Upload LUT to global for texture path.
  {
    uint16_t host_lut[256];
    for (int i = 0; i < 256; ++i) host_lut[i] = f32_to_f16_bits(fp8_e4m3_to_f32((uint8_t)i));
    CUDA_CHECK(cudaMemcpy(d_lut, host_lut, 256 * sizeof(uint16_t), cudaMemcpyHostToDevice));
  }
  cudaTextureObject_t lut_tex = make_lut_texture_object_u16(d_lut);
  cudaTextureObject_t b8_u32_tex = make_u32_tex_object(d_B8_col_u32, (size_t)K * N / 4);

  dim3 grid(N / 32, M / 32);
  dim3 block(128, 1, 1);

  // Warmup
  wmma_f16_gemm_kernel_tiled<<<grid, block>>>(d_Ah, d_Bh_col, d_C, M, N, K);
  wmma_fp16a_fp8e4m3b_gemm_kernel_tiled<0><<<grid, block>>>(d_Ah, d_B8_col, 0, d_C, M, N, K, 0, scale_b);
  wmma_fp16a_fp8e4m3b_gemm_kernel_tiled<1><<<grid, block>>>(d_Ah, d_B8_col, 0, d_C, M, N, K, lut_tex, scale_b);
  wmma_fp16a_fp8e4m3b_gemm_kernel_tiled<2><<<grid, block>>>(d_Ah, d_B8_col, 0, d_C, M, N, K, 0, scale_b);
  wmma_fp16a_fp8e4m3b_gemm_kernel_tiled<3><<<grid, block>>>(d_Ah, d_B8_col, 0, d_C, M, N, K, 0, scale_b);
  wmma_fp16a_fp8e4m3b_gemm_kernel_tiled<4><<<grid, block>>>(d_Ah, d_B8_col, b8_u32_tex, d_C, M, N, K, 0, scale_b);
  dequantize_fp8e4m3_to_fp16_kernel<<<(K * N + 255) / 256, 256>>>(d_B8_col, d_Bh_upcast, K * N, scale_b);
  wmma_f16_gemm_kernel_tiled<<<grid, block>>>(d_Ah, d_Bh_upcast, d_C, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());

  auto report = [&](const char* name, float ms_avg) {
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = (flops / (ms_avg / 1e3)) / 1e12;
    printf("[fp8wgt] %s: %.3f ms/iter  (%.2f TFLOP/s)\n", name, ms_avg, tflops);
  };

  auto time_avg_ms = [&](auto&& launch) {
    GpuTimer t;
    t.begin();
    for (int r = 0; r < repeats; ++r) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return t.end_ms() / (float)repeats;
  };

  report("fp16_wmma", time_avg_ms([&] {
    wmma_f16_gemm_kernel_tiled<<<grid, block>>>(d_Ah, d_Bh_col, d_C, M, N, K);
  }));

  report("fp8->fp16 upcast (global) + fp16_wmma", time_avg_ms([&] {
    dequantize_fp8e4m3_to_fp16_kernel<<<(K * N + 255) / 256, 256>>>(d_B8_col, d_Bh_upcast, K * N, scale_b);
    wmma_f16_gemm_kernel_tiled<<<grid, block>>>(d_Ah, d_Bh_upcast, d_C, M, N, K);
  }));

  // Practical inference baseline: weights are reused, so decode once and keep FP16 weights resident.
  // This is the simplest way to beat naive upcasting on SM86 (no native FP8 MMA).
  {
    dequantize_fp8e4m3_to_fp16_kernel<<<(K * N + 255) / 256, 256>>>(d_B8_col, d_Bh_upcast, K * N, scale_b);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    report("fp8->fp16 upcast_once (weights reuse) + fp16_wmma", time_avg_ms([&] {
      wmma_f16_gemm_kernel_tiled<<<grid, block>>>(d_Ah, d_Bh_upcast, d_C, M, N, K);
    }));
  }

  report("fp8->fp16 (const LUT) fused + wmma", time_avg_ms([&] {
    wmma_fp16a_fp8e4m3b_gemm_kernel_tiled<0><<<grid, block>>>(d_Ah, d_B8_col, 0, d_C, M, N, K, 0, scale_b);
  }));

  report("fp8->fp16 (tex LUT) fused + wmma", time_avg_ms([&] {
    wmma_fp16a_fp8e4m3b_gemm_kernel_tiled<1><<<grid, block>>>(d_Ah, d_B8_col, 0, d_C, M, N, K, lut_tex, scale_b);
  }));

  report("fp8->fp16 (shared LUT) fused + wmma", time_avg_ms([&] {
    wmma_fp16a_fp8e4m3b_gemm_kernel_tiled<2><<<grid, block>>>(d_Ah, d_B8_col, 0, d_C, M, N, K, 0, scale_b);
  }));

  report("fp8->fp16 (shared LUT vector4) fused + wmma", time_avg_ms([&] {
    wmma_fp16a_fp8e4m3b_gemm_kernel_tiled<3><<<grid, block>>>(d_Ah, d_B8_col, 0, d_C, M, N, K, 0, scale_b);
  }));

  report("fp8->fp16 (TEX weights + shared LUT vector4) fused + wmma", time_avg_ms([&] {
    wmma_fp16a_fp8e4m3b_gemm_kernel_tiled<4><<<grid, block>>>(d_Ah, d_B8_col, b8_u32_tex, d_C, M, N, K, 0, scale_b);
  }));

  report("fp8->fp16 (TEX weights pipelined(dist=1) + shared LUT vector4) fused + wmma", time_avg_ms([&] {
    wmma_fp16a_fp8e4m3b_gemm_kernel_tiled<5><<<grid, block>>>(d_Ah, d_B8_col, b8_u32_tex, d_C, M, N, K, 0, scale_b);
  }));

  // cuBLAS tensor-core baseline (strong reference point).
  {
    cublasHandle_t handle{};
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    CUBLAS_CHECK(cublasSetStream(handle, 0));

    float alpha = 1.0f;
    float beta = 0.0f;

    auto cublas_run = [&](const half* B_colmajor) {
      // cuBLAS is column-major. Reuse A row-major MxK by interpreting it as column-major KxM and using op(A)=T.
      CUBLAS_CHECK(cublasGemmEx(
          handle,
          CUBLAS_OP_T, CUBLAS_OP_N,
          M, N, K,
          &alpha,
          d_Ah, CUDA_R_16F, K,
          B_colmajor, CUDA_R_16F, K,
          &beta,
          d_C, CUDA_R_32F, M,
          CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };

    // Warmup
    cublas_run(d_Bh_col);
    CUDA_CHECK(cudaDeviceSynchronize());

    // FP16 cuBLAS baseline
    report("cublas_fp16_tensorop", time_avg_ms([&] { cublas_run(d_Bh_col); }));

    // Naive upcast-per-iter + cuBLAS
    report("fp8->fp16 upcast (global) + cublas_tensorop", time_avg_ms([&] {
      dequantize_fp8e4m3_to_fp16_kernel<<<(K * N + 255) / 256, 256>>>(d_B8_col, d_Bh_upcast, K * N, scale_b);
      cublas_run(d_Bh_upcast);
    }));

    // Practical decode-once weights reuse + cuBLAS
    {
      dequantize_fp8e4m3_to_fp16_kernel<<<(K * N + 255) / 256, 256>>>(d_B8_col, d_Bh_upcast, K * N, scale_b);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      report("fp8->fp16 upcast_once (weights reuse) + cublas_tensorop", time_avg_ms([&] {
        cublas_run(d_Bh_upcast);
      }));
    }

    cublasDestroy(handle);
  }

  CUDA_CHECK(cudaDestroyTextureObject(lut_tex));
  CUDA_CHECK(cudaDestroyTextureObject(b8_u32_tex));
  cudaFree(d_Ah);
  cudaFree(d_B8_col);
  cudaFree(d_B8_col_u32);
  cudaFree(d_Bh_col);
  cudaFree(d_Bh_upcast);
  cudaFree(d_C);
  cudaFree(d_lut);
}

// ------------------------ BENCH: FP16 A + FP8(E4M3) B fused-only (no FP16 weights allocated) ------------------------
static void run_bench_fp8wgt_fused_only() {
  upload_fp8_lut();

  // Intended goal: store weights in FP8, expand only transiently inside the GEMM.
  // This benchmark avoids allocating a persistent FP16 weights buffer (VRAM saver).
  constexpr int M = 1024;
  constexpr int N = 1024;
  constexpr int K = 1024;
  int repeats = g_profile ? 1 : 50;
  constexpr float scale_b = 1.0f;

  size_t bytes_Ah = (size_t)M * K * sizeof(half);
  size_t bytes_B8 = (size_t)K * N;
  size_t bytes_B8_u32 = (size_t)K * N / 4 * sizeof(uint32_t);
  size_t bytes_C = (size_t)M * N * sizeof(float);

  printf("[fp8wgt_fused_only] weight storage: FP8=%zu bytes (%.2f MiB) vs FP16=%zu bytes (%.2f MiB) for KxN\n",
         bytes_B8, bytes_B8 / (1024.0 * 1024.0),
         (size_t)K * (size_t)N * sizeof(half),
         ((size_t)K * (size_t)N * sizeof(half)) / (1024.0 * 1024.0));

  // Host data.
  std::vector<half> h_Ah((size_t)M * K);
  std::vector<uint8_t> h_B8((size_t)K * N);

  std::mt19937 rng(123);
  std::normal_distribution<float> nd(0.0f, 0.5f);
  std::uniform_int_distribution<int> bd(0, 255);
  for (int i = 0; i < M * K; ++i) h_Ah[(size_t)i] = __float2half(nd(rng));
  for (int i = 0; i < K * N; ++i) h_B8[(size_t)i] = (uint8_t)bd(rng);

  half* d_Ah = nullptr;
  uint8_t* d_B8_col = nullptr;
  uint32_t* d_B8_col_u32 = nullptr;
  float* d_C = nullptr;
  uint16_t* d_lut = nullptr;
  CUDA_CHECK(cudaMalloc(&d_Ah, bytes_Ah));
  CUDA_CHECK(cudaMalloc(&d_B8_col, bytes_B8));
  CUDA_CHECK(cudaMalloc(&d_B8_col_u32, bytes_B8_u32));
  CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
  CUDA_CHECK(cudaMalloc(&d_lut, 256 * sizeof(uint16_t)));

  CUDA_CHECK(cudaMemcpy(d_Ah, h_Ah.data(), bytes_Ah, cudaMemcpyHostToDevice));

  // Upload B8 as col-major (and a packed u32 view for TEX-weight loads).
  {
    std::vector<uint8_t> h_B8_col((size_t)K * N);
    for (int r = 0; r < K; ++r) {
      for (int c = 0; c < N; ++c) {
        h_B8_col[(size_t)c * K + r] = h_B8[(size_t)r * N + c];
      }
    }
    CUDA_CHECK(cudaMemcpy(d_B8_col, h_B8_col.data(), bytes_B8, cudaMemcpyHostToDevice));

    std::vector<uint32_t> h_B8_col_u32((size_t)K * N / 4);
    for (size_t i = 0; i < (size_t)K * N / 4; ++i) {
      uint32_t w = 0;
      w |= (uint32_t)h_B8_col[i * 4 + 0] << 0;
      w |= (uint32_t)h_B8_col[i * 4 + 1] << 8;
      w |= (uint32_t)h_B8_col[i * 4 + 2] << 16;
      w |= (uint32_t)h_B8_col[i * 4 + 3] << 24;
      h_B8_col_u32[i] = w;
    }
    CUDA_CHECK(cudaMemcpy(d_B8_col_u32, h_B8_col_u32.data(), bytes_B8_u32, cudaMemcpyHostToDevice));
  }

  // Upload LUT to global for texture path.
  {
    uint16_t host_lut[256];
    for (int i = 0; i < 256; ++i) host_lut[i] = f32_to_f16_bits(fp8_e4m3_to_f32((uint8_t)i));
    CUDA_CHECK(cudaMemcpy(d_lut, host_lut, 256 * sizeof(uint16_t), cudaMemcpyHostToDevice));
  }
  cudaTextureObject_t lut_tex = make_lut_texture_object_u16(d_lut);
  cudaTextureObject_t b8_u32_tex = make_u32_tex_object(d_B8_col_u32, (size_t)K * N / 4);

  dim3 grid(N / 32, M / 32);
  dim3 block(128, 1, 1);

  // Warmup
  wmma_fp16a_fp8e4m3b_gemm_kernel_tiled<3><<<grid, block>>>(d_Ah, d_B8_col, 0, d_C, M, N, K, 0, scale_b);
  wmma_fp16a_fp8e4m3b_gemm_kernel_tiled<4><<<grid, block>>>(d_Ah, d_B8_col, b8_u32_tex, d_C, M, N, K, 0, scale_b);
  wmma_fp16a_fp8e4m3b_gemm_kernel_tiled<5><<<grid, block>>>(d_Ah, d_B8_col, b8_u32_tex, d_C, M, N, K, 0, scale_b);
  {
    dim3 grid64(N / 64, M / 64);
    dim3 block64(512, 1, 1);
    wmma_fp16a_fp8e4m3b_gemm_kernel_tiled_64x64<4><<<grid64, block64>>>(
        d_Ah, d_B8_col, b8_u32_tex, d_C, M, N, K, 0, scale_b);
    wmma_fp16a_fp8e4m3b_gemm_kernel_tiled_64x64_noAs<4><<<grid64, block64>>>(
        d_Ah, d_B8_col, b8_u32_tex, d_C, M, N, K, 0, scale_b);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  auto report = [&](const char* name, float ms_avg) {
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = (flops / (ms_avg / 1e3)) / 1e12;
    printf("[fp8wgt_fused_only] %s: %.3f ms/iter  (%.2f TFLOP/s)\n", name, ms_avg, tflops);
  };

  auto time_avg_ms = [&](auto&& launch) {
    GpuTimer t;
    t.begin();
    for (int r = 0; r < repeats; ++r) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return t.end_ms() / (float)repeats;
  };

  auto should_run = [&](const char* name) -> bool {
    if (!g_profile_only || !*g_profile_only) return true;
    return std::strstr(name, g_profile_only) != nullptr;
  };

  auto profile_or_time = [&](const char* name, auto&& launch) {
    if (!should_run(name)) return;
    NvtxRange r(name);
    if (g_profile) {
      // Keep the capture tight: one launch per kernel.
      CUDA_CHECK(cudaProfilerStart());
      launch();
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaProfilerStop());
      // Still print an approximate time (single launch) for sanity.
      float ms = time_avg_ms(launch);
      report(name, ms);
    } else {
      report(name, time_avg_ms(launch));
    }
  };

  // Choose the best-looking fused variants from fp8wgt without requiring FP16 weights.
  profile_or_time("fp8->fp16 (shared LUT vector4) fused + wmma", [&] {
    wmma_fp16a_fp8e4m3b_gemm_kernel_tiled<3><<<grid, block>>>(d_Ah, d_B8_col, 0, d_C, M, N, K, 0, scale_b);
  });

  profile_or_time("fp8->fp16 (TEX weights + shared LUT vector4) fused + wmma", [&] {
    wmma_fp16a_fp8e4m3b_gemm_kernel_tiled<4><<<grid, block>>>(d_Ah, d_B8_col, b8_u32_tex, d_C, M, N, K, 0, scale_b);
  });

  profile_or_time("fp8->fp16 (TEX weights pipelined(dist=1) + shared LUT vector4) fused + wmma", [&] {
    wmma_fp16a_fp8e4m3b_gemm_kernel_tiled<5><<<grid, block>>>(d_Ah, d_B8_col, b8_u32_tex, d_C, M, N, K, 0, scale_b);
  });

  // Larger tiling experiments (64x64) to explore different data paths / memory pressure.
  {
    dim3 grid64(N / 64, M / 64);
    dim3 block64(512, 1, 1);

    profile_or_time("fp8->fp16 (64x64, sharedA, TEX weights u32) fused + wmma", [&] {
      wmma_fp16a_fp8e4m3b_gemm_kernel_tiled_64x64<4><<<grid64, block64>>>(
          d_Ah, d_B8_col, b8_u32_tex, d_C, M, N, K, 0, scale_b);
    });

    profile_or_time("fp8->fp16 (64x64, noAs, TEX weights u32) fused + wmma", [&] {
      wmma_fp16a_fp8e4m3b_gemm_kernel_tiled_64x64_noAs<4><<<grid64, block64>>>(
          d_Ah, d_B8_col, b8_u32_tex, d_C, M, N, K, 0, scale_b);
    });

    profile_or_time("fp8->fp16 (64x64, noAs, TEX weights pipelined(dist=1)) fused + wmma", [&] {
      wmma_fp16a_fp8e4m3b_gemm_kernel_tiled_64x64_noAs<5><<<grid64, block64>>>(
          d_Ah, d_B8_col, b8_u32_tex, d_C, M, N, K, 0, scale_b);
    });

    profile_or_time("fp8->fp16 (64x64, noAs, cp.async B8 + shared LUT) fused + wmma", [&] {
      wmma_fp16a_fp8e4m3b_gemm_kernel_tiled_64x64_noAs_cpasync<2><<<grid64, block64>>>(
          d_Ah, d_B8_col, d_C, M, N, K, 0, scale_b);
    });
  }

  CUDA_CHECK(cudaDestroyTextureObject(lut_tex));
  CUDA_CHECK(cudaDestroyTextureObject(b8_u32_tex));
  cudaFree(d_Ah);
  cudaFree(d_B8_col);
  cudaFree(d_B8_col_u32);
  cudaFree(d_C);
  cudaFree(d_lut);
}

