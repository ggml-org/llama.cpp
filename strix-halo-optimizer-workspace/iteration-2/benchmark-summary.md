# Strix Halo Optimizer — Iteration 2 Benchmark Summary

## Overall Scores

| Configuration | Score | Pass Rate |
|--------------|-------|-----------|
| **With skill** | **12/12** | **100%** |
| Without skill | 9/12 | 75% |

## Per-Eval Breakdown

### Eval 4: Vulkan vs HIP Backend Choice
| Assertion | With Skill | Without Skill |
|-----------|-----------|---------------|
| Vulkan build command | PASS | PASS |
| HIP recommended over Vulkan | PASS | PASS |
| Vulkan known issues (#18741) | PASS | **FAIL** |
| Vulkan as valid fallback | PASS | PASS |
| **Score** | **4/4** | **3/4** |

### Eval 5: CPU-Only ZenDNN Setup
| Assertion | With Skill | Without Skill |
|-----------|-----------|---------------|
| ZenDNN build flags | PASS | PASS |
| ZENDNNL_MATMUL_ALGO=1 | PASS | PASS |
| BF16 recommendation | PASS | PASS |
| Thread count (16 physical) | PASS | PASS |
| **Score** | **4/4** | **4/4** |

### Eval 6: llama-server Production Deployment
| Assertion | With Skill | Without Skill |
|-----------|-----------|---------------|
| Parallel sequences (-np) | PASS | PASS |
| KV cache quantization | PASS | PASS |
| Essential flags (ngl 99, no-mmap, hipBLASLt) | PASS | **FAIL** |
| Production stability (systemd/monitoring) | PASS | PASS |
| **Score** | **4/4** | **2/4** |

## Analysis

### Key Differentiators (Where Skill Adds Value)

1. **Strix Halo-specific issues**: The skill consistently surfaces hardware-specific knowledge like issue #18741 (Vulkan model loading failures) that the base model doesn't know about.

2. **HIP/ROCm critical flags**: The skill's biggest advantage is on server deployment — it correctly provides `ROCBLAS_USE_HIPBLASLT=1`, `--no-mmap`, and `-ngl 99` as essential Strix Halo flags. Without the skill, the baseline defaults to CPU-only inference and misses the hipBLASLt optimization entirely.

3. **Unified memory awareness**: The skill ensures `--no-mmap` is recommended for HIP workloads (avoiding page-locking overhead), a subtlety the base model misses.

### Non-Discriminating Evals

- **Eval 5 (ZenDNN)**: Both with and without skill scored 4/4. This is likely because ZenDNN knowledge is relatively well-known and the assertions are broad enough that general knowledge suffices. Consider adding more specific assertions (e.g., ZenDNN only accelerates FP32/BF16, not quantized models).

### Combined with Iteration 1

| Iteration | With Skill | Without Skill |
|-----------|-----------|---------------|
| Iteration 1 (3 evals) | 12/12 (100%) | 4/12 (33%) |
| Iteration 2 (3 evals) | 12/12 (100%) | 9/12 (75%) |
| **Total (6 evals)** | **24/24 (100%)** | **13/24 (54%)** |

The skill maintains perfect scores across all 6 evals while the baseline scores 54% overall.
