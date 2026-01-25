#include <torch/extension.h>

// CUDA forward
torch::Tensor fp8_linear_forward(torch::Tensor A_fp16, torch::Tensor B_fp8_u8, double scale_b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_linear_forward", &fp8_linear_forward,
        "FP8(E4M3) weight-only linear forward (FP16 compute, fused decode)");
}
