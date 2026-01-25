#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <cstdint>

extern "C" {
void fp8imma_init_fp8_e4m3_lut();

int fp8imma_launch_imma_fp8_actquant_v4_f16(
        int kChunk,
        const void* A_row_f16,
        const void* B_col_fp8,
        const void* col_scales_u16,
        void* D_col_f16,
        int M,
        int N,
        int K,
        float global_scale,
        float a_inv_scale,
        void* stream);

int fp8imma_launch_imma_fp8_actquant_v4_bf16(
        int kChunk,
        const void* A_row_bf16,
        const void* B_col_fp8,
        const void* col_scales_u16,
        void* D_col_f16,
        int M,
        int N,
        int K,
        float global_scale,
        float a_inv_scale,
        void* stream);
}

static void check_launch(int err, const char* what) {
    TORCH_CHECK(err == 0, what, " failed, cudaError_t=", err);
}

torch::Tensor imma_fp8_v4_act(
        torch::Tensor A,
        torch::Tensor B_col_fp8,
        torch::Tensor col_scales_f16,
        double global_scale,
        double a_inv_scale,
        int64_t kChunk) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B_col_fp8.is_cuda(), "B_col_fp8 must be a CUDA tensor");
    TORCH_CHECK(col_scales_f16.is_cuda(), "col_scales_f16 must be a CUDA tensor");

    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B_col_fp8.is_contiguous(), "B_col_fp8 must be contiguous");
    TORCH_CHECK(col_scales_f16.is_contiguous(), "col_scales_f16 must be contiguous");

    TORCH_CHECK(A.dim() == 2, "A must be 2D [M,K]");
    TORCH_CHECK(B_col_fp8.dim() == 2, "B_col_fp8 must be 2D [N,K]");
    TORCH_CHECK(col_scales_f16.dim() == 1, "col_scales_f16 must be 1D [N]");

    TORCH_CHECK(B_col_fp8.scalar_type() == at::kByte, "B_col_fp8 must be uint8");
    TORCH_CHECK(
            col_scales_f16.scalar_type() == at::kHalf || col_scales_f16.scalar_type() == at::kShort,
            "col_scales_f16 must be fp16 (scale bits) or uint16");

    TORCH_CHECK(kChunk == 32 || kChunk == 64, "kChunk must be 32 or 64");

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B_col_fp8.size(0);
    TORCH_CHECK(B_col_fp8.size(1) == K, "B_col_fp8 must have shape [N,K]");
    TORCH_CHECK(col_scales_f16.size(0) == N, "col_scales_f16 must have shape [N]");

    TORCH_CHECK((M % 16) == 0 && (N % 16) == 0 && (K % 16) == 0, "M,N,K must be multiples of 16");

    fp8imma_init_fp8_e4m3_lut();

    auto out = torch::empty(
            {N, M},
            torch::TensorOptions().device(A.device()).dtype(torch::kFloat16));

    const void* scales_u16 = nullptr;
    if (col_scales_f16.scalar_type() == at::kHalf) {
        scales_u16 = (const void*)reinterpret_cast<const uint16_t*>(col_scales_f16.data_ptr<at::Half>());
    } else {
        scales_u16 = (const void*)reinterpret_cast<const uint16_t*>(col_scales_f16.data_ptr<int16_t>());
    }

    // Respect PyTorch stream semantics.
    void* stream = (void*)at::cuda::getCurrentCUDAStream();

    if (A.scalar_type() == at::kHalf) {
        check_launch(
                fp8imma_launch_imma_fp8_actquant_v4_f16(
                        (int)kChunk,
                        (const void*)A.data_ptr<at::Half>(),
                        (const void*)B_col_fp8.data_ptr<uint8_t>(),
                        scales_u16,
                        (void*)out.data_ptr<at::Half>(),
                        (int)M,
                        (int)N,
                        (int)K,
                        (float)global_scale,
                        (float)a_inv_scale,
                        stream),
                "fp8imma_launch_imma_fp8_actquant_v4_f16");
    } else if (A.scalar_type() == at::kBFloat16) {
        check_launch(
                fp8imma_launch_imma_fp8_actquant_v4_bf16(
                        (int)kChunk,
                        (const void*)A.data_ptr<at::BFloat16>(),
                        (const void*)B_col_fp8.data_ptr<uint8_t>(),
                        scales_u16,
                        (void*)out.data_ptr<at::Half>(),
                        (int)M,
                        (int)N,
                        (int)K,
                        (float)global_scale,
                        (float)a_inv_scale,
                        stream),
                "fp8imma_launch_imma_fp8_actquant_v4_bf16");
    } else {
        TORCH_CHECK(false, "A must be fp16 or bf16");
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "imma_fp8_v4_act",
      &imma_fp8_v4_act,
      "IMMA FP8-as-storage GEMM (v4 actquant)\n"
      "A: [M,K] fp16/bf16 contiguous (row-major)\n"
      "B_col_fp8: [N,K] uint8 contiguous (represents col-major KxN)\n"
      "col_scales_f16: [N] fp16 contiguous (per-column scales)\n"
      "Returns D_col: [N,M] fp16 contiguous (represents col-major MxN)");
}
