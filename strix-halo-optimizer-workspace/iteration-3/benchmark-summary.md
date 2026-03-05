# Strix Halo Optimizer — Iteration 3 Benchmark Summary

## Overall Scores

| Configuration | Score | Pass Rate |
|--------------|-------|-----------|
| **With skill** | **24/24** | **100%** |
| Without skill | 14/24 | 58% |

## Per-Eval Breakdown

### Eval 1: Full Setup Guide (6 assertions)
| Assertion | With Skill | Without Skill |
|-----------|-----------|---------------|
| TTM memory fix | PASS | **FAIL** — Uses GGML_CUDA_ENABLE_UNIFIED_MEMORY instead |
| GPU_TARGETS=gfx1150 | PASS | PASS |
| ROCBLAS_USE_HIPBLASLT=1 | PASS | **FAIL** — Not mentioned |
| --no-mmap flag | PASS | **FAIL** — Not mentioned |
| -ngl 99 full offload | PASS | PASS |
| Quantization recommendation | PASS | PASS |
| **Score** | **6/6** | **3/6** |

### Eval 2: Slow Prompt Processing (3 assertions)
| Assertion | With Skill | Without Skill |
|-----------|-----------|---------------|
| Identifies rocBLAS kernel issue | PASS | **FAIL** — Misdiagnoses as CPU-only, recommends Vulkan |
| Recommends ROCBLAS_USE_HIPBLASLT=1 | PASS | **FAIL** — Not mentioned |
| Mentions ~2.5x improvement | PASS | **FAIL** — Claims 1500-3000+ t/s (wrong diagnosis) |
| **Score** | **3/3** | **0/3** |

### Eval 3: OOM Memory Issue (3 assertions)
| Assertion | With Skill | Without Skill |
|-----------|-----------|---------------|
| Explains unified memory / 4GB VRAM | PASS | PASS |
| TTM pages_limit fix | PASS | **FAIL** — Recommends GGML_CUDA_ENABLE_UNIFIED_MEMORY + BIOS UMA |
| Reboot/initramfs required | PASS | **FAIL** — No TTM context |
| **Score** | **3/3** | **1/3** |

### Eval 4: Vulkan vs HIP Backend Choice (4 assertions)
| Assertion | With Skill | Without Skill |
|-----------|-----------|---------------|
| Vulkan build command | PASS | PASS |
| HIP recommended over Vulkan | PASS | PASS |
| Vulkan known issues (#18741) | PASS | **FAIL** — Not mentioned |
| Vulkan as valid fallback | PASS | PASS |
| **Score** | **4/4** | **3/4** |

### Eval 5: CPU-Only ZenDNN Setup (4 assertions)
| Assertion | With Skill | Without Skill |
|-----------|-----------|---------------|
| ZenDNN build flags | PASS | PASS |
| ZENDNNL_MATMUL_ALGO=1 | PASS | PASS |
| BF16 recommendation | PASS | PASS |
| Thread count (16 physical) | PASS | PASS |
| **Score** | **4/4** | **4/4** |

### Eval 6: llama-server Production Deploy (4 assertions)
| Assertion | With Skill | Without Skill |
|-----------|-----------|---------------|
| Parallel sequences (-np) | PASS | PASS |
| KV cache quantization | PASS | PASS |
| Essential flags (ngl 99, no-mmap, hipBLASLt) | PASS | **FAIL** — Uses Vulkan, missing hipBLASLt and --no-mmap |
| Production stability (systemd/monitoring) | PASS | PASS |
| **Score** | **4/4** | **3/4** |

## Timing

| Eval | With Skill (tokens) | With Skill (sec) | Without Skill (tokens) | Without Skill (sec) |
|------|---------------------|------------------|------------------------|---------------------|
| 1: Full Setup | 29,148 | 81.2s | 51,555 | 161.2s |
| 2: Slow PP | 25,226 | 77.1s | 37,938 | 154.4s |
| 3: OOM Memory | 21,403 | 67.4s | 23,557 | 133.3s |
| 4: Vulkan vs HIP | 35,309 | 149.8s | 63,853 | 199.5s |
| 5: CPU ZenDNN | 41,200 | 155.9s | 56,518 | 158.3s |
| 6: Server Deploy | 37,721 | 204.5s | 42,128 | 242.5s |
| **Average** | **31,668** | **122.6s** | **45,925** | **174.9s** |

## Cross-Iteration Comparison

| Iteration | With Skill | Without Skill |
|-----------|-----------|---------------|
| Iteration 1 (3 evals) | 12/12 (100%) | 4/12 (33%) |
| Iteration 2 (3 evals) | 12/12 (100%) | 9/12 (75%) |
| Iteration 3 (6 evals) | 24/24 (100%) | 14/24 (58%) |

## Analysis

### Key Differentiators
1. **TTM pages_limit**: The #1 gap. Without the skill, agents consistently recommend GGML_CUDA_ENABLE_UNIFIED_MEMORY or BIOS UMA changes instead — neither is the actual fix.
2. **ROCBLAS_USE_HIPBLASLT=1**: Without the skill, agents completely miss this critical optimization and often misdiagnose slow PP as a GPU offload issue.
3. **--no-mmap**: Consistently missed without the skill, especially harmful for large model deployment.
4. **Vulkan issue #18741**: Hardware-specific knowledge the base model cannot discover from code alone.

### Non-Discriminating Eval
- **Eval 5 (ZenDNN)**: 4/4 both with and without skill, same as iteration 2. The ZenDNN backend is well-documented in the codebase and the assertions are broad enough.

### Harmful Advice Without Skill
- Eval 1 recommends HSA_OVERRIDE_GFX_VERSION which hurts performance
- Eval 2 misdiagnoses the problem entirely and suggests Vulkan instead of fixing rocBLAS
- Eval 3 recommends BIOS UMA settings which is the wrong fix
- Eval 6 uses Vulkan backend for production instead of the better-optimized HIP

### Token/Time Efficiency
- With skill uses **31% fewer tokens** on average (31.7k vs 45.9k) while achieving 100% pass rate
- With skill is **30% faster** (122.6s vs 174.9s) — the skill provides focused knowledge so agents don't need extensive codebase exploration
