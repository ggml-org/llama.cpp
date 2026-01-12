# XMX MoE Implementation Completion Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the XMX-accelerated MoE implementation to produce correct numerical output end-to-end.

**Architecture:** Fix F32↔F16 type conversions in sort/scatter phases, implement Q8_0 scale extraction, and use device-aware tile configuration. The XMX GEMM kernels in moe-xmx.hpp are already complete.

**Tech Stack:** Intel SYCL, XMX joint_matrix API, Intel Arc B580 GPU (M=8, N=16, K=32)

---

## Prerequisites

Before starting:
```bash
source /opt/intel/oneapi/setvars.sh --force
cd /Apps/llama.cpp/.worktrees/sycl-coalescing
```

---

### Task 1: Add F32→F16 Token Conversion Kernel

**Files:**
- Modify: `ggml/src/ggml-sycl/moe-sort.hpp`

**Step 1: Add the conversion kernel**

Add before the `moe_count_tokens_per_expert` function:

```cpp
// Convert F32 tokens to F16 for XMX processing
inline void moe_convert_f32_to_f16(
    const float* tokens_f32,     // [n_tokens, hidden_dim] F32 input
    sycl::half* tokens_f16,      // [n_tokens, hidden_dim] F16 output
    int64_t n_tokens,
    int64_t hidden_dim,
    sycl::queue& queue)
{
    constexpr int SG_SIZE = 16;
    int64_t total_elements = n_tokens * hidden_dim;

    queue.parallel_for(
        sycl::nd_range<1>(
            ((total_elements + SG_SIZE - 1) / SG_SIZE) * SG_SIZE,
            SG_SIZE),
        [=](sycl::nd_item<1> item) {
            int64_t idx = item.get_global_id(0);
            if (idx < total_elements) {
                tokens_f16[idx] = sycl::half(tokens_f32[idx]);
            }
        }).wait();
}
```

**Step 2: Verify build compiles**

Run: `cmake --build build -j 16 2>&1 | tail -5`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/moe-sort.hpp
git commit -m "feat(sycl): add F32→F16 conversion kernel for XMX MoE"
```

---

### Task 2: Modify Scatter to Convert F16→F32

**Files:**
- Modify: `ggml/src/ggml-sycl/moe-sort.hpp`

**Step 1: Add F16→F32 scatter function**

Add after the existing `moe_scatter_results` function:

```cpp
// Scatter results back with F16→F32 conversion
// Use when sorted output is F16 but final output must be F32
inline void moe_scatter_results_f16_to_f32(
    const sycl::half* sorted_output,  // [total_pairs, output_dim] F16
    float* final_output,               // [n_tokens * n_ids, output_dim] F32
    const MoETokenMapping* token_map,
    int64_t total_pairs,
    int64_t output_dim,
    sycl::queue& queue)
{
    queue.parallel_for(
        sycl::range<1>(total_pairs),
        [=](sycl::id<1> idx) {
            int32_t original_pos = token_map[idx].original_idx;

            for (int64_t d = 0; d < output_dim; d++) {
                final_output[original_pos * output_dim + d] =
                    static_cast<float>(sorted_output[idx * output_dim + d]);
            }
        }).wait();
}
```

**Step 2: Verify build compiles**

Run: `cmake --build build -j 16 2>&1 | tail -5`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/moe-sort.hpp
git commit -m "feat(sycl): add F16→F32 scatter conversion for XMX MoE output"
```

---

### Task 3: Add Q8_0 Scale Extraction Helper

**Files:**
- Modify: `ggml/src/ggml-sycl/moe-xmx.hpp`

**Step 1: Add Q8_0 scale extraction function**

Add after the `preprocess_tokens_q8` function (around line 109):

```cpp
// Extract fp16 scales from Q8_0 weight blocks
// Q8_0 block layout: [32 int8 values][2 bytes fp16 scale] = 34 bytes per block
// Output: scales[out_dim * (in_dim/32)] in row-major order
inline void extract_q8_0_scales(
    const void* weights_qs,   // [out_dim, in_dim] Q8_0 packed
    sycl::half* scales,       // [out_dim, in_dim/32] output scales
    int64_t out_dim,
    int64_t in_dim,
    sycl::queue& queue)
{
    constexpr int QK = 32;    // Q8_0 block size
    constexpr int Q8_0_BLOCK_SIZE = 34;  // 32 int8 + 2 bytes fp16 scale

    int64_t num_blocks_per_row = in_dim / QK;
    int64_t total_blocks = out_dim * num_blocks_per_row;

    const uint8_t* w_ptr = static_cast<const uint8_t*>(weights_qs);

    queue.parallel_for(
        sycl::range<1>(total_blocks),
        [=](sycl::id<1> idx) {
            // Q8_0 block layout: first 2 bytes are fp16 scale (little-endian)
            // then 32 bytes of int8 values
            int64_t block_offset = idx * Q8_0_BLOCK_SIZE;

            // Load fp16 scale (stored at start of block in GGML Q8_0 format)
            uint16_t scale_bits = w_ptr[block_offset] |
                                 (static_cast<uint16_t>(w_ptr[block_offset + 1]) << 8);

            // Reinterpret as fp16
            sycl::half scale;
            std::memcpy(&scale, &scale_bits, sizeof(sycl::half));

            scales[idx] = scale;
        }).wait();
}
```

**Step 2: Verify build compiles**

Run: `cmake --build build -j 16 2>&1 | tail -5`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/moe-xmx.hpp
git commit -m "feat(sycl): add Q8_0 scale extraction for XMX MoE GEMM"
```

---

### Task 4: Update Dispatcher to Use New Kernels

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp`

**Step 1: Allocate F32-typed tokens_f16 buffer and add conversion**

Replace lines 11502-11509 (the F32 to F16 conversion section):

```cpp
    // Phase 1a: Convert F32 tokens to F16 for XMX processing
    const float* tokens_f32 = static_cast<const float*>(src1->data);
    sycl::half* tokens_f16_input = sycl::malloc_device<sycl::half>(
        n_tokens * in_dim, *stream);

    if (!tokens_f16_input) {
        GGML_SYCL_DEBUG("[XMX MoE] Failed to allocate tokens_f16_input\n");
        sycl::free(tokens_sorted, *stream);
        sycl::free(token_map, *stream);
        sycl::free(expert_counts, *stream);
        sycl::free(expert_offsets, *stream);
        sycl::free(sorted_output, *stream);
        return false;
    }

    moe_convert_f32_to_f16(
        tokens_f32, tokens_f16_input, n_tokens, in_dim, *stream);

    // Phase 1b: Sort tokens by expert
    moe_sort_tokens_by_expert(
        tokens_f16_input, tokens_sorted, expert_ids, expert_write_pos,
        token_map, n_tokens, n_ids, in_dim, n_experts, *stream);
```

**Step 2: Allocate and populate Q8_0 expert scales buffer**

Before the expert loop (around line 11534), add scale buffer allocation:

```cpp
    // Allocate buffer for Q8_0 expert scales (reused across experts)
    // Each expert has out_dim * (in_dim/QK) scales
    int64_t num_k_blocks = in_dim / QK;
    sycl::half* expert_scale_buf = nullptr;
    if (src0->type == GGML_TYPE_Q8_0) {
        expert_scale_buf = sycl::malloc_device<sycl::half>(
            out_dim * num_k_blocks, *stream);
        if (!expert_scale_buf) {
            GGML_SYCL_DEBUG("[XMX MoE] Failed to allocate expert_scale_buf\n");
            // ... cleanup ...
            return false;
        }
    }
```

**Step 3: Update Q8_0 expert dispatch to extract and use scales**

Replace the Q8_0 kernel call section (lines 11543-11559):

```cpp
        if (src0->type == GGML_TYPE_Q8_0) {
            // Extract Q8_0 scales for this expert's weights
            moe_xmx::extract_q8_0_scales(
                expert_weights, expert_scale_buf,
                out_dim, in_dim, *stream);

            // Pre-quantize fp16 tokens to int8 with per-block scales
            const sycl::half* expert_tokens = tokens_sorted + h_offsets[e] * in_dim;
            moe_xmx::preprocess_tokens_q8(
                expert_tokens, q_tokens, token_scales,
                h_counts[e], in_dim, *stream);

            moe_xmx::launch_xmx_moe_gemm_q8_0<4, 4>(
                expert_weights, expert_scale_buf,
                q_tokens, token_scales,
                sorted_output + h_offsets[e] * out_dim,
                h_counts[e], out_dim, in_dim, xmx_cfg, *stream);
        }
```

**Step 4: Update scatter to use F16→F32 conversion**

Replace lines 11580-11587:

```cpp
    // Phase 3: Scatter results back to original positions with F16→F32 conversion
    float* final_output = static_cast<float*>(dst->data);
    moe_scatter_results_f16_to_f32(
        sorted_output, final_output, token_map,
        total_pairs, out_dim, *stream);
```

**Step 5: Update cleanup to free new buffers**

Add to the cleanup section (around line 11589):

```cpp
    // Free temporary buffers
    sycl::free(tokens_f16_input, *stream);
    if (expert_scale_buf) sycl::free(expert_scale_buf, *stream);
    sycl::free(tokens_sorted, *stream);
    // ... rest of existing cleanup ...
```

**Step 6: Verify build compiles**

Run: `cmake --build build -j 16 2>&1 | tail -5`
Expected: Build succeeds

**Step 7: Commit**

```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "fix(sycl): use proper F32↔F16 conversion and Q8_0 scales in XMX MoE"
```

---

### Task 5: Verify Q8_0 MoE Model Correctness

**Files:**
- Test only (no modifications)

**Step 1: Run numerical correctness test**

```bash
source /opt/intel/oneapi/setvars.sh --force
GGML_SYCL_XMX_MOE=1 GGML_SYCL_DISABLE_GRAPH=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -ngl 99 --flash-attn on --no-conversation \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

Expected: "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15"

**Step 2: Compare with baseline (XMX disabled)**

```bash
GGML_SYCL_DISABLE_GRAPH=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -ngl 99 --flash-attn on --no-conversation \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

Expected: Output should match XMX path output

---

### Task 6: Run Performance Benchmark

**Files:**
- Test only (no modifications)

**Step 1: Benchmark XMX path**

```bash
GGML_SYCL_XMX_MOE=1 GGML_SYCL_DISABLE_GRAPH=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -p 512 -n 128 -ngl 99 -fa 1
```

Record: pp512 t/s, tg128 t/s

**Step 2: Benchmark baseline (fused MoE path)**

```bash
GGML_SYCL_DISABLE_GRAPH=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -p 512 -n 128 -ngl 99 -fa 1
```

Record: pp512 t/s, tg128 t/s for comparison

---

### Task 7: Final Integration Commit

**Files:**
- All modified files

**Step 1: Stage and create final commit**

```bash
git add -A
git commit -m "feat(sycl): complete XMX MoE implementation with proper type conversion

Completes the XMX-accelerated MoE dispatch path to produce correct output:

- Add F32→F16 conversion before token sorting (moe_convert_f32_to_f16)
- Add F16→F32 conversion during result scatter (moe_scatter_results_f16_to_f32)
- Implement Q8_0 scale extraction (extract_q8_0_scales)
- Use extracted scales in XMX GEMM kernel

The XMX MoE path is enabled via GGML_SYCL_XMX_MOE=1 environment variable
and requires GGML_SYCL_DISABLE_GRAPH=1 due to host synchronization needs.

Tested with GPT-OSS 20B Q8_0 model producing correct numerical output."
```

---

## Summary of Changes

| File | Change |
|------|--------|
| moe-sort.hpp | Add `moe_convert_f32_to_f16`, `moe_scatter_results_f16_to_f32` |
| moe-xmx.hpp | Add `extract_q8_0_scales` |
| ggml-sycl.cpp | Use new conversion/scale functions, fix buffer allocation |

## Risk Assessment

- **Medium risk**: New kernels are straightforward conversions
- **Testing required**: Numerical output must match baseline
- **Performance unknown**: May be slower than fused path for some batch sizes

## Future Work (Not in This Plan)

1. Dynamic tile parameter dispatch (currently hardcoded `<4, 4>`)
2. MXFP4 model testing (once MXFP4 models are available)
3. Graph recording compatibility (remove host sync requirement)
