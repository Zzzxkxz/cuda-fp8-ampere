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

