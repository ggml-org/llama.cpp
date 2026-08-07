Yes, I agree. I moved the weight repack and CUTLASS GEMM path into the CUDA backend so it is no longer tied to MoE. I tested both dense and MoE Qwen models on the same RTX PRO 6000.

Qwen3.6-27B-NVFP4 dense:

| Prompt | Existing CUDA | CUTLASS | Speedup |
|---:|---:|---:|---:|
| pp512 | 4,613 tok/s | 5,932 tok/s | 1.29x |
| pp2048 | 4,624 tok/s | 6,014 tok/s | 1.30x |
| pp8192 | 5,092 tok/s | 6,780 tok/s | 1.33x |

Qwen3.6-35B-A3B-NVFP4 MoE:

| Prompt | Existing CUDA, ub2048 | CUTLASS, ub2048 | CUTLASS, ub8192 |
|---:|---:|---:|---:|
| pp512 | 10,150 tok/s | 12,429 tok/s | 13,038 tok/s |
| pp2048 | 11,249 tok/s | 12,680 tok/s | 14,042 tok/s |
| pp8192 | 12,633 tok/s | 16,728 tok/s | 13,380 tok/s |

| Prompt | Existing CUDA | Best CUTLASS | Speedup |
|---:|---:|---:|---:|
| pp512 | 10,150 tok/s | 13,038 tok/s | 1.29x |
| pp2048 | 11,249 tok/s | 14,042 tok/s | 1.25x |
| pp8192 | 12,633 tok/s | 16,728 tok/s | 1.32x |

The weight conversion and GEMM code are shared. Only the expert scheduling and fused epilogues remain MoE-specific.
