static void print_list() {
  printf("Available benchmarks:\n");
  printf("  tex        - texture fetch vs global load under ALU load\n");
  printf("  texpipe    - software-pipelined TEX/global prefetch (report §3.3)\n");
  printf("  pipe_mix   - FP32/INT32 contention on the flexible datapath (Deep Dive §2.1)\n");
  printf("  l1tex_dualread - LDG+LDG vs LDG+TEX concurrent reads (Deep Dive §2.2)\n");
  printf("  lop3       - LOP3 full-adder primitives (XOR3 + MAJ3)\n");
  printf("  rns        - lane-split RNS modmul/modadd with global vs ldg vs tex table loads\n");
  printf("  vbs        - vertical bit-slicing add: LOP3 ripple-carry over bit-planes\n");
  printf("  transpose  - warp-local 32x32 bit transpose cost\n");
  printf("  fp8e4m3    - decode FP8(E4M3)->FP16 then WMMA (tensor cores)\n");
  printf("  fp8wgt     - FP16 activations + FP8(E4M3) weights: fused decode inside WMMA vs global upcast\n");
  printf("  fp8wgt_fused_only - FP16 activations + FP8(E4M3) weights: fused WMMA, no FP16 weights allocation (includes 64x64 tiling experiments)\n");
  printf("  fp8quant   - FP16->FP8(E4M3) quantize, FP8->FP16 dequant, then cuBLAS GEMM\n");
  printf("  fp8reuse   - FP8 quant + cuBLAS; compares dequant-both vs weights-reuse dequant\n");
  printf("  fp8sweep   - sweep K to study TEX-vs-const dequant and longer decode paths\n");
  printf("  int8gemm   - INT8 tensor-core GEMM baseline via cuBLASLt heuristics\n");
  printf("  cutlass_f16 - FP16 tensor-op GEMM baseline via CUTLASS\n");
  printf("  cutlass_fp8wgt - FP16 A + FP8(E4M3) weights pipeline via CUTLASS (decode then GEMM)\n");
  printf("  cutlass_fp8wgt_l2pin - same as cutlass_fp8wgt, but tries persisting-L2 pinning for B/scales\n");
  printf("  cutlass_int8wgt - FP16 A + INT8 weights pipeline via CUTLASS (dequant then GEMM)\n");
  printf("  cutlass_int8wgt_l2pin - same as cutlass_int8wgt, but tries persisting-L2 pinning for B/scales\n");
  printf("  int8bfp    - INT8 TC GEMM + postscale (per-column scale via TEX)\n");
  printf("  int8bfp_l2pin - int8bfp with persisting-L2 hinting for B/scales\n");
  printf("  int8bfp_probe - dump+time cuBLASLt heuristic candidates for int8xint8->int32 GEMM\n");
  printf("  imma_int8bfp_fused - custom IMMA (WMMA) int8 GEMM with fused per-column scaling -> fp16\n");
  printf("  imma_int8bfp_fused_l2pin - imma_int8bfp_fused with persisting-L2 hinting for B/scales\n");
  printf("  imma_int8bfp_fused_v2 - IMMA fused v2 (64x64 tile + shared staging)\n");
  printf("  imma_int8bfp_fused_v2_l2pin - imma_int8bfp_fused_v2 with persisting-L2 hinting for B/scales\n");
  printf("  imma_int8bfp_fused_v2_texscale - IMMA fused v2, but loads per-column scales via TEX\n");
  printf("  imma_int8bfp_fused_v2_texscale_l2pin - v2_texscale with persisting-L2 hinting for B/scales\n");
  printf("  imma_int8bfp_fused_v2_autotune - autotunes IMMA fused v2 (KChunk)\n");
  printf("  imma_int8bfp_fused_v2_autotune_l2pin - v2 autotune with persisting-L2 hinting for B/scales\n");
  printf("  imma_int8bfp_fused_autotune - autotunes IMMA fused tile config (BMxBN)\n");
  printf("  imma_int8bfp_fused_autotune_l2pin - autotune bench with persisting-L2 hinting for B/scales\n");
  printf("  fp8wgt_imma_offline - FP8 weights -> Offline INT8 conversion -> IMMA GEMM (Phase B1)\n");
  printf("  imma_fp8_jit_v2        - IMMA fused v2 using FP8->INT8 JIT transcoding (Phase B2)\n");
  printf("  imma_fp8_jit_v2_l2pin  - imma_fp8_jit_v2 with persisting-L2 hinting for B/scales\n");
  printf("  imma_fp8_jit_v2_i8lut       - IMMA fused v2 using FP8->INT8 via per-column shared LUT (experimental)\n");
  printf("  imma_fp8_jit_v2_i8lut_l2pin - imma_fp8_jit_v2_i8lut with persisting-L2 hinting for B/scales\n");
  printf("  imma_fp8_jit_v3_act_f16      - FP16 A + FP8 weights: fused A FP16->INT8 quant + FP8->INT8 JIT + IMMA\n");
  printf("  imma_fp8_jit_v3_act_f16_l2pin - same as imma_fp8_jit_v3_act_f16, with persisting-L2 hinting for B/scales\n");
  printf("  imma_fp8_jit_v3_act_bf16      - BF16 A + FP8 weights: fused A BF16->INT8 quant + FP8->INT8 JIT + IMMA\n");
  printf("  imma_fp8_jit_v3_act_bf16_l2pin - same as imma_fp8_jit_v3_act_bf16, with persisting-L2 hinting for B/scales\n");
  printf("  imma_fp8_jit_v4_act_f16      - FP16 A + FP8 weights: cp.async A to shared + shared quant to INT8 + FP8->INT8 JIT + IMMA\n");
  printf("  imma_fp8_jit_v4_act_f16_l2pin - same as imma_fp8_jit_v4_act_f16, with persisting-L2 hinting for B/scales\n");
  printf("  imma_fp8_jit_v4_act_bf16      - BF16 A + FP8 weights: cp.async A to shared + shared quant to INT8 + FP8->INT8 JIT + IMMA\n");
  printf("  imma_fp8_jit_v4_act_bf16_l2pin - same as imma_fp8_jit_v4_act_bf16, with persisting-L2 hinting for B/scales\n");
  printf("  imma_fp8_jit_v4_act_f16_texscale      - v4_act_f16 but loads per-column scales via TEX (u16)\n");
  printf("  imma_fp8_jit_v4_act_f16_texscale_l2pin - v4_act_f16_texscale with persisting-L2 hinting for B/scales\n");
  printf("  imma_fp8_jit_v4_act_bf16_texscale      - v4_act_bf16 but loads per-column scales via TEX (u16)\n");
  printf("  imma_fp8_jit_v4_act_bf16_texscale_l2pin - v4_act_bf16_texscale with persisting-L2 hinting for B/scales\n");
  printf("  imma_fp8_jit_v4_act_f16_k64      - v4_act_f16 but KChunk=64\n");
  printf("  imma_fp8_jit_v4_act_f16_k64_l2pin - v4_act_f16_k64 with persisting-L2 hinting for B/scales\n");
  printf("  imma_fp8_jit_v4_act_bf16_k64      - v4_act_bf16 but KChunk=64\n");
  printf("  imma_fp8_jit_v4_act_bf16_k64_l2pin - v4_act_bf16_k64 with persisting-L2 hinting for B/scales\n");
  printf("  imma_fp8_jit_v4_act_f16_texscale_k64      - v4_act_f16_texscale but KChunk=64\n");
  printf("  imma_fp8_jit_v4_act_f16_texscale_k64_l2pin - v4_act_f16_texscale_k64 with persisting-L2 hinting for B/scales\n");
  printf("  imma_fp8_jit_v4_act_bf16_texscale_k64      - v4_act_bf16_texscale but KChunk=64\n");
  printf("  imma_fp8_jit_v4_act_bf16_texscale_k64_l2pin - v4_act_bf16_texscale_k64 with persisting-L2 hinting for B/scales\n");
}

static const char* get_arg(int argc, char** argv, const char* key) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (strcmp(argv[i], key) == 0) return argv[i + 1];
  }
  return nullptr;
}

static bool has_flag(int argc, char** argv, const char* key) {
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], key) == 0) return true;
  }
  return false;
}

int main(int argc, char** argv) {
  if (has_flag(argc, argv, "--list")) {
    print_list();
    return 0;
  }

  g_profile = has_flag(argc, argv, "--profile");
  g_profile_only = get_arg(argc, argv, "--profile_only");

  const char* bench = get_arg(argc, argv, "--bench");
  if (!bench) {
    printf("Usage: %s --list | --bench <name> [--profile] [--profile_only <substr>]\n", argv[0]);
    return 1;
  }

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

  if (strcmp(bench, "tex") == 0) {
    run_bench_tex();
    return 0;
  }
  if (strcmp(bench, "texpipe") == 0) {
    run_bench_texpipe();
    return 0;
  }
  if (strcmp(bench, "pipe_mix") == 0) {
    run_bench_pipe_mix();
    return 0;
  }
  if (strcmp(bench, "l1tex_dualread") == 0) {
    run_bench_l1tex_dualread();
    return 0;
  }
  if (strcmp(bench, "lop3") == 0) {
    run_bench_lop3();
    return 0;
  }
  if (strcmp(bench, "rns") == 0) {
    run_bench_rns();
    return 0;
  }
  if (strcmp(bench, "vbs") == 0) {
    run_bench_vbs();
    return 0;
  }
  if (strcmp(bench, "transpose") == 0) {
    run_bench_transpose();
    return 0;
  }
  if (strcmp(bench, "fp8e4m3") == 0) {
    run_bench_fp8e4m3();
    return 0;
  }
  if (strcmp(bench, "fp8wgt") == 0) {
    run_bench_fp8wgt();
    return 0;
  }
  if (strcmp(bench, "fp8wgt_fused_only") == 0) {
    run_bench_fp8wgt_fused_only();
    return 0;
  }
  if (strcmp(bench, "fp8quant") == 0) {
    run_bench_fp8quant();
    return 0;
  }
  if (strcmp(bench, "fp8reuse") == 0) {
    run_bench_fp8reuse();
    return 0;
  }
  if (strcmp(bench, "fp8sweep") == 0) {
    run_bench_fp8sweep();
    return 0;
  }
  if (strcmp(bench, "int8gemm") == 0) {
    run_bench_int8gemm();
    return 0;
  }
  if (strcmp(bench, "cutlass_f16") == 0) {
    run_bench_cutlass_f16();
    return 0;
  }
  if (strcmp(bench, "cutlass_fp8wgt") == 0) {
    run_bench_cutlass_fp8wgt(false);
    return 0;
  }
  if (strcmp(bench, "cutlass_fp8wgt_l2pin") == 0) {
    run_bench_cutlass_fp8wgt(true);
    return 0;
  }
  if (strcmp(bench, "cutlass_int8wgt") == 0) {
    run_bench_cutlass_int8wgt(false);
    return 0;
  }
  if (strcmp(bench, "cutlass_int8wgt_l2pin") == 0) {
    run_bench_cutlass_int8wgt(true);
    return 0;
  }
  if (strcmp(bench, "int8bfp") == 0) {
    run_bench_int8bfp(false);
    return 0;
  }
  if (strcmp(bench, "int8bfp_l2pin") == 0) {
    run_bench_int8bfp(true);
    return 0;
  }
  if (strcmp(bench, "int8bfp_probe") == 0) {
    run_bench_int8bfp_probe();
    return 0;
  }

  if (strcmp(bench, "imma_int8bfp_fused") == 0) {
    run_bench_imma_int8bfp_fused(false);
    return 0;
  }
  if (strcmp(bench, "imma_int8bfp_fused_l2pin") == 0) {
    run_bench_imma_int8bfp_fused(true);
    return 0;
  }

  if (strcmp(bench, "imma_int8bfp_fused_v2") == 0) {
    run_bench_imma_int8bfp_fused_v2(false);
    return 0;
  }
  if (strcmp(bench, "imma_int8bfp_fused_v2_l2pin") == 0) {
    run_bench_imma_int8bfp_fused_v2(true);
    return 0;
  }

  if (strcmp(bench, "imma_int8bfp_fused_v2_texscale") == 0) {
    run_bench_imma_int8bfp_fused_v2_texscale(false);
    return 0;
  }
  if (strcmp(bench, "imma_int8bfp_fused_v2_texscale_l2pin") == 0) {
    run_bench_imma_int8bfp_fused_v2_texscale(true);
    return 0;
  }

  if (strcmp(bench, "imma_int8bfp_fused_v2_autotune") == 0) {
    run_bench_imma_int8bfp_fused_v2_autotune(false);
    return 0;
  }
  if (strcmp(bench, "imma_int8bfp_fused_v2_autotune_l2pin") == 0) {
    run_bench_imma_int8bfp_fused_v2_autotune(true);
    return 0;
  }

  if (strcmp(bench, "imma_int8bfp_fused_autotune") == 0) {
    run_bench_imma_int8bfp_fused_autotune(false);
    return 0;
  }
  if (strcmp(bench, "imma_int8bfp_fused_autotune_l2pin") == 0) {
    run_bench_imma_int8bfp_fused_autotune(true);
    return 0;
  }

  if (strcmp(bench, "imma_fp8_jit_v2") == 0) {
    run_bench_imma_fp8_jit_v2(false);
    return 0;
  }
  if (strcmp(bench, "imma_fp8_jit_v2_l2pin") == 0) {
    run_bench_imma_fp8_jit_v2(true);
    return 0;
  }

  if (strcmp(bench, "imma_fp8_jit_v2_i8lut") == 0) {
    run_bench_imma_fp8_jit_v2_i8lut(false);
    return 0;
  }
  if (strcmp(bench, "imma_fp8_jit_v2_i8lut_l2pin") == 0) {
    run_bench_imma_fp8_jit_v2_i8lut(true);
    return 0;
  }

  if (strcmp(bench, "imma_fp8_jit_v3_act_f16") == 0) {
    run_bench_imma_fp8_jit_v3_act_f16(false);
    return 0;
  }
  if (strcmp(bench, "imma_fp8_jit_v3_act_f16_l2pin") == 0) {
    run_bench_imma_fp8_jit_v3_act_f16(true);
    return 0;
  }

  if (strcmp(bench, "imma_fp8_jit_v3_act_bf16") == 0) {
    run_bench_imma_fp8_jit_v3_act_bf16(false);
    return 0;
  }
  if (strcmp(bench, "imma_fp8_jit_v3_act_bf16_l2pin") == 0) {
    run_bench_imma_fp8_jit_v3_act_bf16(true);
    return 0;
  }

  if (strcmp(bench, "imma_fp8_jit_v4_act_f16") == 0) {
    run_bench_imma_fp8_jit_v4_act_f16(false);
    return 0;
  }
  if (strcmp(bench, "imma_fp8_jit_v4_act_f16_l2pin") == 0) {
    run_bench_imma_fp8_jit_v4_act_f16(true);
    return 0;
  }

  if (strcmp(bench, "imma_fp8_jit_v4_act_bf16") == 0) {
    run_bench_imma_fp8_jit_v4_act_bf16(false);
    return 0;
  }
  if (strcmp(bench, "imma_fp8_jit_v4_act_bf16_l2pin") == 0) {
    run_bench_imma_fp8_jit_v4_act_bf16(true);
    return 0;
  }

  if (strcmp(bench, "imma_fp8_jit_v4_act_f16_texscale") == 0) {
    run_bench_imma_fp8_jit_v4_act_f16_texscale(false);
    return 0;
  }
  if (strcmp(bench, "imma_fp8_jit_v4_act_f16_texscale_l2pin") == 0) {
    run_bench_imma_fp8_jit_v4_act_f16_texscale(true);
    return 0;
  }

  if (strcmp(bench, "imma_fp8_jit_v4_act_bf16_texscale") == 0) {
    run_bench_imma_fp8_jit_v4_act_bf16_texscale(false);
    return 0;
  }
  if (strcmp(bench, "imma_fp8_jit_v4_act_bf16_texscale_l2pin") == 0) {
    run_bench_imma_fp8_jit_v4_act_bf16_texscale(true);
    return 0;
  }

  if (strcmp(bench, "imma_fp8_jit_v4_act_f16_k64") == 0) {
    run_bench_imma_fp8_jit_v4_act_f16_k64(false);
    return 0;
  }
  if (strcmp(bench, "imma_fp8_jit_v4_act_f16_k64_l2pin") == 0) {
    run_bench_imma_fp8_jit_v4_act_f16_k64(true);
    return 0;
  }

  if (strcmp(bench, "imma_fp8_jit_v4_act_bf16_k64") == 0) {
    run_bench_imma_fp8_jit_v4_act_bf16_k64(false);
    return 0;
  }
  if (strcmp(bench, "imma_fp8_jit_v4_act_bf16_k64_l2pin") == 0) {
    run_bench_imma_fp8_jit_v4_act_bf16_k64(true);
    return 0;
  }

  if (strcmp(bench, "imma_fp8_jit_v4_act_f16_texscale_k64") == 0) {
    run_bench_imma_fp8_jit_v4_act_f16_texscale_k64(false);
    return 0;
  }
  if (strcmp(bench, "imma_fp8_jit_v4_act_f16_texscale_k64_l2pin") == 0) {
    run_bench_imma_fp8_jit_v4_act_f16_texscale_k64(true);
    return 0;
  }

  if (strcmp(bench, "imma_fp8_jit_v4_act_bf16_texscale_k64") == 0) {
    run_bench_imma_fp8_jit_v4_act_bf16_texscale_k64(false);
    return 0;
  }
  if (strcmp(bench, "imma_fp8_jit_v4_act_bf16_texscale_k64_l2pin") == 0) {
    run_bench_imma_fp8_jit_v4_act_bf16_texscale_k64(true);
    return 0;
  }

  if (strcmp(bench, "fp8wgt_imma_offline") == 0) {
    run_bench_fp8wgt_imma_offline();
    return 0;
  }

  fprintf(stderr, "Unknown benchmark: %s\n", bench);
  print_list();
  return 1;
}
