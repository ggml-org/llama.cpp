# Benchmark Summary — Iteration 1

## Overall Scores

| Eval | With Skill | Without Skill | Improvement |
|------|-----------|---------------|-------------|
| 1: Full Setup Guide | 6/6 (100%) | 3/6 (50%) | +50pp |
| 2: Slow PP Diagnosis | 3/3 (100%) | 0/3 (0%) | +100pp |
| 3: OOM Memory Issue | 3/3 (100%) | 1/3 (33%) | +67pp |
| **Total** | **12/12 (100%)** | **4/12 (33%)** | **+67pp** |

## Timing

| Eval | With Skill (tokens) | With Skill (sec) | Without Skill (tokens) | Without Skill (sec) |
|------|---------------------|------------------|------------------------|---------------------|
| 1: Full Setup | 29,689 | 90.2s | 15,013 | 73.5s |
| 2: Slow PP | 27,881 | 57.2s | 13,878 | 61.3s |
| 3: OOM Memory | 28,008 | 71.5s | 13,640 | 56.3s |
| **Average** | **28,526** | **73.0s** | **14,177** | **63.7s** |

## Per-Assertion Results

### Eval 1: Full Setup Guide (6 assertions)

| Assertion | With Skill | Without Skill |
|-----------|-----------|---------------|
| TTM pages_limit config | PASS | **FAIL** — Not mentioned at all |
| GPU_TARGETS=gfx1150 | PASS | PASS (used gfx1151 variant) |
| ROCBLAS_USE_HIPBLASLT=1 | PASS | **FAIL** — Not mentioned |
| --no-mmap for 70B | PASS | **FAIL** — Not mentioned |
| -ngl 99 full offload | PASS | PASS |
| Quantization recommendation | PASS | PASS |

### Eval 2: Slow PP Diagnosis (3 assertions)

| Assertion | With Skill | Without Skill |
|-----------|-----------|---------------|
| Identifies rocBLAS kernel issue | PASS | **FAIL** — Misdiagnoses as bandwidth/ubatch/flash-attn |
| Recommends ROCBLAS_USE_HIPBLASLT=1 | PASS | **FAIL** — Recommends counterproductive HSA_OVERRIDE_GFX_VERSION |
| Mentions ~2.5x improvement | PASS | **FAIL** — Claims 1500-4000+ t/s (wildly inaccurate) |

### Eval 3: OOM Memory Issue (3 assertions)

| Assertion | With Skill | Without Skill |
|-----------|-----------|---------------|
| Explains unified memory / 4GB VRAM | PASS | PASS |
| TTM pages_limit fix | PASS | **FAIL** — Recommends BIOS UMA setting instead |
| Reboot/initramfs required | PASS | **FAIL** — No TTM context for reboot |

## Key Findings

1. **The skill provides massive value for Strix Halo-specific knowledge.** Without the skill, the model completely misses the three most critical Strix Halo optimizations:
   - TTM pages_limit (the #1 OOM fix)
   - ROCBLAS_USE_HIPBLASLT=1 (the #1 prompt processing fix)
   - --no-mmap for large models

2. **Without-skill responses contain harmful advice.** Eval 2 without skill recommends `HSA_OVERRIDE_GFX_VERSION=11.5.1` which actually *hurts* performance by disabling native RDNA 3.5 optimizations. Eval 3 without skill recommends BIOS UMA changes which is the wrong fix entirely.

3. **Token usage is ~2x higher with skill** (28.5k vs 14.2k avg) due to reading reference files, but execution time is comparable (~73s vs ~64s avg). The quality improvement vastly outweighs the cost.

4. **Score: 100% with skill vs 33% without skill.** The skill turns every eval from partially/fully wrong to fully correct.

## Conclusion

The skill is highly effective. No changes needed for iteration 1 — all assertions pass at 100%. The skill correctly encodes the three critical pieces of Strix Halo knowledge that the base model lacks: TTM memory configuration, hipBLASLt environment variable, and --no-mmap for large models.
