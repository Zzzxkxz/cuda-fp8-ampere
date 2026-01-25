// Entry point for the benchmark binary.
//
// This stays as a single CUDA translation unit (for NVCC simplicity), but the
// implementation is split into readable modules under src/bench/.

#include "bench/preamble.inl"

#include "bench/bench_cutlass.inl"
#include "bench/bench_lop3_wmma.inl"
#include "bench/bench_int8_cublaslt.inl"
#include "bench/bench_imma_custom.inl"
#include "bench/bench_fp8_quant_cublas.inl"
#include "bench/bench_misc.inl"

#include "bench/cli.inl"
