# fp8imma_ext (PyTorch)

Minimal PyTorch CUDA extension that exposes the `fp8imma` IMMA kernels.

This extension is **C++-only** and links against the prebuilt shared library:
- `build/libfp8imma.so`

Currently exposed:
- `imma_fp8_v4_act(A, B_col_fp8, col_scales_f16, global_scale, a_inv_scale, kChunk)`

Conventions:
- `A`: `[M,K]` fp16/bf16, contiguous
- `B_col_fp8`: `[N,K]` uint8, contiguous (this is **row-major** `[N,K]`, which corresponds to **col-major** weights of shape `[K,N]` with leading dimension `K`)
- `col_scales_f16`: `[N]` fp16, contiguous (scale values stored as fp16)
- return value: `[N,M]` fp16, contiguous (row-major `[N,M]` == col-major `[M,N]`)

Build/install (dev):

```bash
cd ../..
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

cd torch_ext/fp8imma
python -m pip install -v --no-build-isolation -e .
```

Smoke test:

```python
import torch
import fp8imma_ext

M=N=K=256
A = torch.randn(M, K, device='cuda', dtype=torch.float16)
B_col = torch.randint(0, 256, (N, K), device='cuda', dtype=torch.uint8)
scales = torch.ones(N, device='cuda', dtype=torch.float16)
D_col = fp8imma_ext.imma_fp8_v4_act(A, B_col, scales, 1.0, 1.0, 32)
print(D_col.shape, D_col.dtype, D_col.is_contiguous())
```
