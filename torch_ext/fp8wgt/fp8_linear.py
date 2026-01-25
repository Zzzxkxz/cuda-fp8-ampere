import torch
import fp8wgt_ext


class FP8Linear(torch.nn.Module):
    """Inference-only linear: FP16 activations, FP8(E4M3) weights stored as uint8.

    Weight tensor layout for this extension:
    - B_fp8_u8 is [N, K] contiguous uint8
    - interpreted as col-major KxN with ld=K (same indexing as repo benchmarks)

    Output:
    - FP32 [M, N]

    This is meant as a minimal plug-in building block; no backward.
    """

    def __init__(self, B_fp8_u8: torch.Tensor, scale_b: float = 1.0):
        super().__init__()
        assert B_fp8_u8.is_cuda and B_fp8_u8.dtype == torch.uint8 and B_fp8_u8.dim() == 2
        self.register_buffer("B_fp8_u8", B_fp8_u8.contiguous())
        self.scale_b = float(scale_b)

    def forward(self, A_fp16: torch.Tensor) -> torch.Tensor:
        assert A_fp16.is_cuda and A_fp16.dtype == torch.float16 and A_fp16.dim() == 2
        return fp8wgt_ext.fp8_linear_forward(A_fp16.contiguous(), self.B_fp8_u8, self.scale_b)
