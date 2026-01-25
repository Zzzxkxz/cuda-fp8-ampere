# rtx3090_fp8_exps

IMMA-based **FP8-as-storage** GEMM experiments for Ampere (sm_86 / RTX 3090 Ti).

Goal: keep weights stored as **1-byte FP8(E4M3) bits** in VRAM, decode + per-column scale on the fly, and use **INT8 tensor cores (WMMA/IMMA)** to get high throughput on hardware without native FP8 MMA.

This repo contains:
- a reusable CUDA kernel library (C++ API + C ABI): `include/fp8imma/imma_fp8.h`
- a benchmark harness: `src/gpu_bench.cu` → `build/gpu_bench`
- a minimal PyTorch extension: `torch_ext/fp8imma`

Large exploratory markdown notes were moved into `reports/` and are git-ignored.

## Architecture (high level)

- Weights are stored as `uint8` FP8(E4M3) **bit patterns**.
- A 256-entry LUT decodes FP8→FP16 values.
- Per-output-channel scales are applied (stored as FP16 bits / `uint16`).
- Values are quantized to int8 and consumed by WMMA IMMA (`signed char` fragments).

Kernel variants exposed by the library include:
- v2: int8 activations + FP8 weights (JIT FP8→int8)
- v3: fused activation quantization (register path)
- v4: pipelined activations + fused quantization (shared staging)

### Kernel dataflow (FP8-as-storage)

```text
	A (fp16/bf16)                B (uint8 fp8-e4m3 bits)         col_scales (u16 bits)
   [M,K] row-major               [N,K] (represents KxN col-major)      [N]
		  |                               |                               |
		  |                               | (LUT in __constant__)        |
		  |                               v                               |
		  |                        fp8 -> fp16 decode                     |
		  |                               |                               |
		  |                               +-----------(per-column)--------+
		  |                                           scale
		  |                               |
		  |                               v
		  |                        fp16 -> int8 (sat)
		  |                               |
		  +--------------- int8 A --------+
						  (act quant)
										  |
										  v
								WMMA/IMMA (int8) accumulate (int32)
										  |
										  v
								 D (fp16) written as [N,M]
								 (represents MxN col-major)
```

Code organization:

```text
include/fp8imma/imma_fp8.h      Public C++ API + C ABI entry points
src/fp8imma/*.cu               Modular kernel implementations + wrappers
src/fp8imma/impl/*.inl         Per-variant kernel bodies
src/gpu_bench.cu + src/bench/* Benchmark harness
torch_ext/fp8imma              Minimal PyTorch extension (links libfp8imma.so)
```

## Results (measured locally)

These are **microbenchmarks** on RTX 3090 Ti, CUDA visible, using the included scripts.

Apples-to-apples FP8-storage baseline comparison (M=N=K=4096, fp16 activations, FP8 bytes weights):
- Fused `fp8imma_ext.imma_fp8_v4_act`: ~2.95 ms/iter (~46.6 TOPS), peak alloc ~120 MiB
- Naive Torch (decode FP8→fp16 each iter + `A @ B.T`): ~2.43 ms/iter (~56.6 TOPS), peak alloc ~248 MiB

Notes:
- “Torch matmul only (weights already fp16)” is faster, but it assumes you keep fp16 weights resident, which loses FP8 VRAM savings.
- Peak alloc above is per-call peak allocated bytes (not full model VRAM footprint).

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Tests

From the build directory:

```bash
ctest --output-on-failure
```

PyTorch extension smoke-test (builds + imports + runs one tiny CUDA call):

```bash
ctest -R torch --output-on-failure
```

If you don’t have PyTorch installed (or no CUDA device is available), the torch test prints `SKIP:` and exits successfully.

To disable adding torch tests at configure-time:

```bash
cmake .. -DFP8IMMA_ENABLE_TORCH_TESTS=OFF
```

## Run

```bash
./build/gpu_bench --list
./build/gpu_bench --bench tex
./build/gpu_bench --bench lop3
./build/gpu_bench --bench transpose

# FP8 / INT8 / WMMA-related
./build/gpu_bench --bench fp8reuse
./build/gpu_bench --bench int8bfp_probe

# IMMA FP8-as-storage benches
./build/gpu_bench --bench imma_fp8_jit_v2
./build/gpu_bench --bench imma_fp8_jit_v4_act_f16
```

## Profiling (optional)

- Nsight Compute: `ncu --set full ./build/gpu_bench --bench tex`
- Nsight Systems: `nsys profile -t cuda,nvtx ./build/gpu_bench --bench tex`

### Register/spill stats (ptxas)

To print register count and spill stats in the build output:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGPU_BENCH_PTXAS_VERBOSE=ON
cmake --build build -j
```

## PyTorch extension

The extension exposes a single entry point today:
- `fp8imma_ext.imma_fp8_v4_act(A, B_col_fp8, col_scales_f16, global_scale, a_inv_scale, kChunk)`

Build/install (dev):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

. .venv_torch_cuda312/bin/activate
python -m pip install -v --no-build-isolation -e torch_ext/fp8imma
```

Smoke test:

```python
import torch
import fp8imma_ext

M=N=K=128
A = torch.randn((M, K), device='cuda', dtype=torch.float16)
B = torch.randint(0, 256, (N, K), device='cuda', dtype=torch.uint8)
scales = torch.ones((N,), device='cuda', dtype=torch.float16)

out_nm = fp8imma_ext.imma_fp8_v4_act(A, B, scales, 1.0, 1.0, 32)
print(out_nm.shape, out_nm.dtype)
```

## Benchmark: Torch vs fused

For an apples-to-apples “FP8-as-storage” comparison:

```bash
. .venv_torch_cuda312/bin/activate
python scripts/bench_torch_vs_fp8imma.py --M 4096 --N 4096 --K 4096 --kChunk 32 --report_mem
```
