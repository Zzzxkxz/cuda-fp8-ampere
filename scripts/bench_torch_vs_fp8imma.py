#!/usr/bin/env python3

import argparse
import math
import time
from typing import Tuple

import torch


def fp8_e4m3_to_f32(v: int) -> float:
    # Matches src/fp8imma/imma_fp8_kernels.cu::fp8_e4m3_to_f32
    sign = (v >> 7) & 1
    exp = (v >> 3) & 0xF
    mant = v & 0x7
    s = -1.0 if sign else 1.0
    bias = 7

    if exp == 0:
        if mant == 0:
            return s * 0.0
        return s * math.ldexp(float(mant), (1 - bias) - 3)

    if exp == 15:
        if mant == 0:
            return s * float("inf")
        return float("nan")

    frac = 1.0 + (float(mant) / 8.0)
    return s * math.ldexp(frac, exp - bias)


def make_fp8_e4m3_lut(device: torch.device) -> torch.Tensor:
    # Build on CPU in float32 for exactness, then cast to fp16 and move.
    vals = [fp8_e4m3_to_f32(i) for i in range(256)]
    lut = torch.tensor(vals, dtype=torch.float32).to(torch.float16)
    return lut.to(device)


def cuda_time_ms(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


def peak_mem_bytes(fn) -> int:
    # Measures peak allocated bytes during a single call.
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def tops_from_ms(ms: float, m: int, n: int, k: int) -> float:
    # 2*M*N*K ops
    ops = 2.0 * float(m) * float(n) * float(k)
    return (ops / (ms * 1e-3)) / 1e12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--kChunk", type=int, default=32, choices=[32, 64])
    ap.add_argument("--global_scale", type=float, default=1.0)
    ap.add_argument("--a_inv_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--report_mem",
        action="store_true",
        help="Report peak CUDA allocated bytes for each path (single call)",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    assert torch.cuda.is_available(), "CUDA must be available"

    device = torch.device("cuda")

    # Make inputs. We use fp16 activations, and FP8 bytes for weights.
    A = torch.randn((args.M, args.K), device=device, dtype=torch.float16)
    B_u8 = torch.randint(0, 256, (args.N, args.K), device=device, dtype=torch.uint8)
    scales = torch.ones((args.N,), device=device, dtype=torch.float16)

    # LUT for naive decode path.
    lut = make_fp8_e4m3_lut(device)

    # Import extension lazily, so torch-only baselines can still run.
    import fp8imma_ext

    def fused_once():
        # returns [N,M]
        _ = fp8imma_ext.imma_fp8_v4_act(
            A, B_u8, scales, args.global_scale, args.a_inv_scale, args.kChunk
        )

    def naive_decode_each_iter():
        # Decode weights every iteration and matmul in fp16.
        # This is the closest "default torch" approach for FP8-as-storage.
        B_fp16 = torch.take(lut, B_u8.to(torch.int64).reshape(-1)).reshape(args.N, args.K)
        B_fp16 = B_fp16 * scales[:, None]
        _ = A @ B_fp16.t()

    # A "decode once" steady-state baseline (unfair to FP8-as-storage, but useful context).
    B_fp16_cached = torch.take(lut, B_u8.to(torch.int64).reshape(-1)).reshape(args.N, args.K)
    B_fp16_cached = (B_fp16_cached * scales[:, None]).contiguous()

    def naive_decode_once_matmul_only():
        _ = A @ B_fp16_cached.t()

    # Timings
    ms_fused = cuda_time_ms(fused_once, args.iters, args.warmup)
    ms_naive_full = cuda_time_ms(naive_decode_each_iter, args.iters, args.warmup)
    ms_naive_matmul = cuda_time_ms(naive_decode_once_matmul_only, args.iters, args.warmup)

    mem_fused = mem_naive_full = mem_naive_matmul = None
    if args.report_mem:
        # Use a single call to estimate peak allocation.
        mem_fused = peak_mem_bytes(fused_once)
        mem_naive_full = peak_mem_bytes(naive_decode_each_iter)
        mem_naive_matmul = peak_mem_bytes(naive_decode_once_matmul_only)

    print("Sizes: M,N,K =", args.M, args.N, args.K)
    print("kChunk:", args.kChunk)
    print()

    print("FUSED (fp8imma_ext imma_fp8_v4_act):")
    print(f"  {ms_fused:.4f} ms/iter  ({tops_from_ms(ms_fused, args.M, args.N, args.K):.2f} TOPS)")
    if mem_fused is not None:
        print(f"  peak alloc: {mem_fused / (1024**2):.1f} MiB")

    print("NAIVE Torch (decode FP8->FP16 each iter + fp16 matmul):")
    print(f"  {ms_naive_full:.4f} ms/iter  ({tops_from_ms(ms_naive_full, args.M, args.N, args.K):.2f} TOPS)")
    if mem_naive_full is not None:
        print(f"  peak alloc: {mem_naive_full / (1024**2):.1f} MiB")

    print("Torch matmul only (weights already decoded/cached as fp16):")
    print(f"  {ms_naive_matmul:.4f} ms/iter  ({tops_from_ms(ms_naive_matmul, args.M, args.N, args.K):.2f} TOPS)")
    if mem_naive_matmul is not None:
        print(f"  peak alloc: {mem_naive_matmul / (1024**2):.1f} MiB")


if __name__ == "__main__":
    main()
