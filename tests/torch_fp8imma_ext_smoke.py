#!/usr/bin/env python3
import os
import sys
import subprocess


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, ".."))


def _smoke() -> None:
    try:
        import torch
    except Exception as e:
        print(f"SKIP: torch import failed: {e}")
        return

    if not torch.cuda.is_available():
        print("SKIP: torch.cuda.is_available() is false")
        return

    ext_dir = os.path.join(_repo_root(), "torch_ext", "fp8imma")

    # Build extension in-place (fast incremental rebuild).
    subprocess.check_call([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=ext_dir)

    # Import the built extension from its local folder.
    sys.path.insert(0, ext_dir)
    import fp8imma_ext  # type: ignore

    device = torch.device("cuda")

    # Smallest valid multiple-of-16 shapes.
    M = N = K = 16
    A = torch.randn((M, K), device=device, dtype=torch.float16).contiguous()
    B = torch.randint(0, 256, (N, K), device=device, dtype=torch.uint8).contiguous()
    col_scales = torch.ones((N,), device=device, dtype=torch.float16).contiguous()

    out = fp8imma_ext.imma_fp8_v4_act(A, B, col_scales, 1.0, 1.0, 32)

    assert tuple(out.shape) == (N, M), f"unexpected out shape: {tuple(out.shape)}"
    assert out.is_cuda, "out must be CUDA"
    assert out.dtype == torch.float16, f"unexpected out dtype: {out.dtype}"

    if not torch.isfinite(out).all().item():
        raise AssertionError("out contains non-finite values")

    # Synchronize to surface async CUDA failures.
    torch.cuda.synchronize()

    print("OK: fp8imma_ext build+import+call")


if __name__ == "__main__":
    _smoke()
