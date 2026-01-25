# fp8wgt_ext (PyTorch CUDA extension)

This is a minimal “plug-and-play” wrapper around the repo’s **fused decode + WMMA** idea:

- store weights in FP8(E4M3) as `uint8` on GPU
- decode transiently inside the matmul kernel
- compute on FP16 tensor cores (WMMA)
- no persistent FP16 weight buffer

## Build

From this directory:

```bash
python -m pip install -U pip
python -m pip install -U torch
python setup.py install
```

## Use

```python
import torch
from fp8_linear import FP8Linear

# B_fp8_u8 must be [N,K] uint8 contiguous on CUDA
M, K, N = 1024, 1024, 1024
A = torch.randn(M, K, device="cuda", dtype=torch.float16)
B_fp8 = torch.randint(0, 256, (N, K), device="cuda", dtype=torch.uint8)

layer = FP8Linear(B_fp8, scale_b=1.0)
C = layer(A)  # FP32 [M,N]
print(C.shape, C.dtype)
```

## Notes / limitations

- Forward-only (inference). No backward.
- Requires M/N/K multiples of 16.
- FP8 format decode is a reasonable E4M3 interpretation but may not match NVIDIA’s exact FP8 variants; treat as experimental.
