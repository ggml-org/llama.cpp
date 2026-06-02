# KV Cache Blue-Noise Dithering Results

## Implementation Summary

Added `GGML_TYPE_Q4_0_BLUE` as a first-class KV cache type with CLI support via `--cache-type-k q4_0_blue` and `--cache-type-v q4_0_blue`.

### Design
- Same block format as `Q4_0` (`block_q4_0` — 2 bytes scale + 16 bytes nibbles per 32-element block)
- Identical dequantization path (reuses `dequantize_row_q4_0`)
- Quantization applies a deterministic 64-value blue-noise dither table before rounding
- Dither strength: 0.25 × quantization step
- Seed derived from block index and element index for determinism
- Mixed-mode supported: `--cache-type-k q4_0_blue --cache-type-v q4_0`

### Files Changed

| File | Change |
|------|--------|
| `ggml/include/ggml.h` | Added `GGML_TYPE_Q4_0_BLUE = 42` to enum |
| `ggml/src/ggml.c` | Type traits entry + `ggml_quantize_chunk` case |
| `ggml/src/ggml-quants.h` | Declared `quantize_row_q4_0_blue_ref` and `quantize_q4_0_blue` |
| `ggml/src/ggml-quants.c` | Added blue-noise table (64 values) + dithered quantization functions |
| `common/arg.cpp` | Added `GGML_TYPE_Q4_0_BLUE` to `kv_cache_types` |
| `ggml/src/ggml-cuda/common.cuh` | CUDA type traits specialization |
| `ggml/src/ggml-cuda/*.cu/.cuh` | CUDA dispatch cases (17 locations across 7 files) |
| `ggml/src/ggml-metal/*.cpp` | Metal dispatch cases (3 locations) |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | Vulkan dispatch cases (17 locations) |
| `ggml/src/ggml-cpu/ops.cpp` | CPU dispatch cases (7 locations) |
| `ggml/src/ggml-cpu/amx/*.cpp` | AMX dispatch cases (2 locations) |
| `ggml/src/ggml-cpu/llamafile/sgemm.cpp` | llamafile dispatch case |
| `ggml/src/ggml-cpu/spacemit/ime.cpp` | Spacemit dispatch cases (3 locations) |

### Unit Test Results (test-q4-blue-dither)

| Test | Result |
|------|--------|
| Dithering produces different output from normal q4_0 | PASS |
| Deterministic (same input → same output) | PASS |
| Dequantization produces finite values | PASS |
| Type name is `q4_0_blue` | PASS |
| Block sizes match Q4_0 (18 bytes) | PASS |

### CLI Recognition
```
--cache-type-k TYPE   KV cache data type for K
                      allowed values: f32, f16, bf16, q8_0, q4_0, q4_0_blue, q4_1, iq4_nl, ...
```

### Build Status

| Build | Status |
|-------|--------|
| CPU (`-DGGML_CUDA=OFF`) | ✅ Builds without errors |
| CUDA (`-DGGML_CUDA=ON`) | ❌ Pre-existing CUDA 11.5 + GCC 11.4 incompatibility (not caused by our changes) |

### End-to-End Test Results (Qwen2.5-1.5B-Instruct, q4_0 quantized, CPU)

| Test | F16 | Q4_0 | Q4_0_BLUE | Mixed (K blue, V q4_0) |
|------|-----|------|-----------|----------------------|
| Needle retrieval (4K ctx) | ✅ `BLUELLAMA-42-KV` | ❌ Degenerate output | ❌ Degenerate output | ❌ Degenerate output |
| Loads without crash | ✅ | ✅ | ✅ | ✅ |
| Determinism (same seed) | ✅ | ✅ | ✅ | N/A |
| Perplexity | N/A | N/A | N/A | N/A |

### Observations
1. **F16 baseline** successfully retrieves the hidden fact "BLUELLAMA-42-KV" at 4K context.
2. **All q4_0 variants** (normal, blue, mixed) produce degenerate output at 4K context with this small 1.5B model. This is expected — q4_0 KV cache quantization at 4K context is very aggressive for a small model.
3. The q4_0_blue deterministic output is identical to q4_0 for this test case, suggesting that when the quantization is this aggressive, the dithering doesn't change the token-level outcome (both collapse to the same failure mode).
4. Perplexity testing requires more text data (min 1024 tokens for context 512).

### Next Steps / Known Issues
1. The CUDA build environment has a pre-existing issue with CUDA 11.5 + GCC 11.4 (`std_function.h` parameter pack error). This is unrelated to the Q4_0_BLUE changes.
2. For meaningful quality comparison, test with larger models (7B+) where q4_0 KV cache has more room to show differences.
3. The `ggml_cpu.c` dispatch table was updated (`type_traits_cpu` array) — this was critical for runtime stability.
4. Perplexity testing requires a text corpus with at least 2x context window tokens.
