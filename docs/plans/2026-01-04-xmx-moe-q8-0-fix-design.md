# XMX MoE Q8_0 Weight Loading Fix - Design Document

**Date:** 2026-01-04
**Status:** Approved for implementation

## Problem Statement

The XMX MoE GEMM kernel for Q8_0 quantized weights produces garbage output because it incorrectly loads weights directly from global memory, treating Q8_0 block-packed data as raw int8 arrays.

### Root Cause

Q8_0 block layout: `[2 bytes fp16 scale][32 int8 values]` = 34 bytes per block

The current code (moe-xmx.hpp lines 311-313):
```cpp
auto w_offset = w_ptr + col * in_dim + k;
joint_matrix_load(sg, mat_b, w_mptr, static_cast<size_t>(in_dim));
```

This loads garbage because:
1. It treats packed blocks as contiguous int8
2. It reads scale bytes as data values
3. The stride `in_dim` is wrong for block-packed format

## Solution Architecture

Mirror the MXFP4 kernel pattern: **unpack weights to SLM before XMX loads**.

### Changes Required

1. **Add SLM weight buffer** (2048 bytes for TILES_N × XMX_N × XMX_K)
2. **Add cooperative Q8_0 unpacking** to extract int8 values from blocks
3. **Load mat_b from SLM** instead of global memory
4. **Remove broken direct global load**

### Q8_0 Unpacking Logic

```cpp
// Q8_0 block: [2 bytes scale][32 bytes int8 data] = 34 bytes
constexpr int Q8_0_BLOCK_SIZE = 34;
constexpr int Q8_0_SCALE_SIZE = 2;

// For each output column and K-block:
int64_t block_offset = (global_col * num_k_blocks + k_block) * Q8_0_BLOCK_SIZE;
int8_t value = w_ptr[block_offset + Q8_0_SCALE_SIZE + k_elem];  // Skip scale bytes
slm_weights[out_col_local * XMX_K + k_elem] = value;
```

## MXFP4 Kernel Status

The MXFP4 kernel is **structurally complete**:
- Token loading to SLM ✅
- Weight unpacking via kvalues_mxfp4 LUT ✅
- E8M0 scale extraction ✅
- XMX mat_a/mat_b loads from SLM ✅
- Scale application and float accumulation ✅

Cannot test until MXFP4 model fits in VRAM (requires <12GB or dual-GPU).

## Testing Strategy

### Primary Test: Numerical Correctness
```bash
GGML_SYCL_XMX_MOE=1 GGML_SYCL_DISABLE_GRAPH=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -ngl 99 --flash-attn on --no-conversation \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```
**Expected:** `1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15`

### Baseline Comparison
Run same test without `GGML_SYCL_XMX_MOE=1` - outputs must match exactly.

### Performance Benchmark
```bash
GGML_SYCL_XMX_MOE=1 GGML_SYCL_DISABLE_GRAPH=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-bench -m gpt-oss-20b-Q8_0.gguf -p 512 -n 128 -ngl 99 -fa 1
```

## Risk Assessment

- **Low risk**: Following proven MXFP4 pattern
- **Testable immediately**: GPT-OSS 20B Q8_0 available
- **Reversible**: XMX path gated behind GGML_SYCL_XMX_MOE=1

## Success Criteria

1. Count prompt produces correct sequential numbers
2. XMX output matches baseline output exactly
3. No performance regression vs baseline (ideally improvement)
4. All skeleton comments removed from production code
