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

