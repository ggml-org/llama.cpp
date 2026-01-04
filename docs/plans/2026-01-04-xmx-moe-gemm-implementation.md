# XMX MoE GEMM Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement working XMX-accelerated GEMM kernels for MoE dispatch, replacing skeleton implementations that produce zeros.

**Architecture:** Two-phase approach: (1) Pre-quantize fp16 tokens to int8 with scales, (2) XMX GEMM per expert with SLM staging and per-K-block scale accumulation. Reference implementation in `mmq_xmx.cpp`.

**Tech Stack:** Intel SYCL, XMX joint_matrix API, Intel Arc B580 GPU (M=8, N=16, K=32)

---

### Task 1: Add Token Pre-Quantization Kernel

**Files:**
- Modify: `ggml/src/ggml-sycl/moe-xmx.hpp:17-46`

**Step 1: Add preprocess_tokens_q8 function declaration**

After the `MoEXMXConfig` struct (~line 46), add:

```cpp
// Pre-quantize fp16 tokens to int8 with per-block scales
// Output: q_tokens[batch * in_dim] int8, scales[batch * (in_dim/32)] fp16
void preprocess_tokens_q8(
    const sycl::half* tokens,   // [batch, in_dim] fp16 input
    int8_t* q_tokens,           // [batch, in_dim] int8 output
    sycl::half* scales,         // [batch, in_dim/32] per-block scales
    int64_t batch,
    int64_t in_dim,
    sycl::queue& queue);
```

**Step 2: Implement the kernel**

Add implementation after the declaration:

```cpp
inline void preprocess_tokens_q8(
    const sycl::half* tokens,
    int8_t* q_tokens,
    sycl::half* scales,
    int64_t batch,
    int64_t in_dim,
    sycl::queue& queue)
{
    constexpr int QK = 32;      // Quantization block size (matches XMX_K)
    constexpr int SG_SIZE = 16;

    int64_t num_blocks = batch * (in_dim / QK);

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_blocks * SG_SIZE, SG_SIZE),
            [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                auto sg = item.get_sub_group();
                int64_t block_id = item.get_group(0);
                int64_t row = block_id / (in_dim / QK);
                int64_t k_block = block_id % (in_dim / QK);

                int lane = sg.get_local_linear_id();

                // Each lane loads 2 values (32 total per sub-group)
                int64_t base = row * in_dim + k_block * QK;
                float v0 = static_cast<float>(tokens[base + lane * 2]);
                float v1 = static_cast<float>(tokens[base + lane * 2 + 1]);

                // Find max absolute value via sub-group reduction
                float local_max = sycl::fmax(sycl::fabs(v0), sycl::fabs(v1));
                float amax = sycl::reduce_over_group(sg, local_max, sycl::maximum<float>());

                // Compute scale and inverse scale
                float scale = amax / 127.0f;
                float inv_scale = (amax > 0.0f) ? 127.0f / amax : 0.0f;

                // Quantize values
                int8_t q0 = static_cast<int8_t>(sycl::round(v0 * inv_scale));
                int8_t q1 = static_cast<int8_t>(sycl::round(v1 * inv_scale));

                // Store quantized values
                q_tokens[base + lane * 2] = q0;
                q_tokens[base + lane * 2 + 1] = q1;

                // Store scale (one per block, lane 0 only)
                if (lane == 0) {
                    scales[row * (in_dim / QK) + k_block] = sycl::half(scale);
                }
            });
    });
}
```

**Step 3: Verify build compiles**

Run: `source /opt/intel/oneapi/setvars.sh --force && cmake --build build -j 16 2>&1 | tail -10`
Expected: Build succeeds with no errors

**Step 4: Commit**

```bash
git add ggml/src/ggml-sycl/moe-xmx.hpp
git commit -m "feat(sycl): add token pre-quantization kernel for XMX MoE"
```

---

### Task 2: Update Q8_0 Kernel Signature for Pre-Quantized Tokens

**Files:**
- Modify: `ggml/src/ggml-sycl/moe-xmx.hpp:48-68`

**Step 1: Update function signature**

Change `launch_xmx_moe_gemm_q8_0` signature from:

```cpp
template<int TILES_M = 4, int TILES_N = 4>
void launch_xmx_moe_gemm_q8_0(
    const void* weights_qs,       // [out_dim, in_dim] int8 quantized
    const sycl::half* weights_d,  // [out_dim, in_dim/32] Q8_0 scales
    const sycl::half* tokens,     // [batch, in_dim] fp16 activations
    sycl::half* output,           // [batch, out_dim]
```

To:

```cpp
template<int TILES_M = 4, int TILES_N = 4>
void launch_xmx_moe_gemm_q8_0(
    const void* weights_qs,           // [out_dim, in_dim] int8 quantized
    const sycl::half* weights_d,      // [out_dim, in_dim/32] Q8_0 scales
    const int8_t* q_tokens,           // [batch, in_dim] pre-quantized int8
    const sycl::half* token_scales,   // [batch, in_dim/32] token scales
    sycl::half* output,               // [batch, out_dim]
```

**Step 2: Remove the (void)tokens suppression at the end**

Delete lines 192-196 that suppress unused parameter warnings.

**Step 3: Verify build compiles**

Run: `source /opt/intel/oneapi/setvars.sh --force && cmake --build build -j 16 2>&1 | tail -10`
Expected: Build may fail (caller not updated yet), that's OK for this step

**Step 4: Commit**

```bash
git add ggml/src/ggml-sycl/moe-xmx.hpp
git commit -m "refactor(sycl): update Q8_0 MoE kernel to accept pre-quantized tokens"
```

---

### Task 3: Implement SLM Loading and XMX GEMM for Q8_0

**Files:**
- Modify: `ggml/src/ggml-sycl/moe-xmx.hpp:85-191`

**Step 1: Add SLM allocation for tokens and scales**

In the `queue.submit` lambda, add after `slm_weights`:

```cpp
        // SLM for token tiles (TILES_M * XMX_M * XMX_K int8 = 1024 bytes)
        sycl::local_accessor<int8_t, 1> slm_tokens(
            sycl::range<1>(TILES_M * XMX_M * XMX_K), cgh);

        // SLM for scales (token + weight scales for current K-block)
        sycl::local_accessor<float, 1> slm_token_scales(
            sycl::range<1>(TILES_M), cgh);
        sycl::local_accessor<float, 1> slm_weight_scales(
            sycl::range<1>(TILES_N), cgh);
```

**Step 2: Add cooperative token loading inside K-loop**

Replace the TODO comment and `joint_matrix_fill(sg, mat_a, 0)` with:

```cpp
                    // === Cooperative token loading to SLM ===
                    // Each sub-group loads part of the token tile
                    int items_per_sg = (TILES_M * XMX_M * XMX_K) / (cfg.wg_size / SG_SIZE);
                    int sg_offset = sg_id * items_per_sg;
                    for (int i = 0; i < items_per_sg; i += SG_SIZE) {
                        int idx = sg_offset + i + item.get_local_id(1);
                        if (idx < TILES_M * XMX_M * XMX_K) {
                            int tile_row = idx / XMX_K;
                            int tile_k = idx % XMX_K;
                            int global_row = wg_row + tile_row;
                            int global_k = k + tile_k;
                            if (global_row < batch && global_k < in_dim) {
                                slm_tokens[idx] = q_tokens[global_row * in_dim + global_k];
                            } else {
                                slm_tokens[idx] = 0;
                            }
                        }
                    }

                    // Load token scales for this K-block
                    if (sg_id < TILES_M && item.get_local_id(1) == 0) {
                        int global_row = wg_row + sg_id * XMX_M;
                        int64_t k_block_idx = k / XMX_K;
                        if (global_row < batch) {
                            slm_token_scales[sg_id] = static_cast<float>(
                                token_scales[global_row * (in_dim / XMX_K) + k_block_idx]);
                        } else {
                            slm_token_scales[sg_id] = 0.0f;
                        }
                    }

                    item.barrier(sycl::access::fence_space::local_space);
```

**Step 3: Load mat_a from SLM**

Replace `joint_matrix_fill(sg, mat_a, 0)` with proper SLM load:

```cpp
                    // Load mat_a from SLM (requires address_space_cast)
                    auto slm_tokens_ptr = sycl::address_space_cast<
                        sycl::access::address_space::local_space,
                        sycl::access::decorated::no>(
                            slm_tokens.get_pointer() + tm * XMX_M * XMX_K);
                    joint_matrix_load(sg, mat_a, slm_tokens_ptr, XMX_K);
```

**Step 4: Verify build compiles**

Run: `source /opt/intel/oneapi/setvars.sh --force && cmake --build build -j 16 2>&1 | tail -10`
Expected: Build succeeds

**Step 5: Commit**

```bash
git add ggml/src/ggml-sycl/moe-xmx.hpp
git commit -m "feat(sycl): implement SLM token loading for XMX MoE Q8_0"
```

---

### Task 4: Implement Scale Application and Output Store for Q8_0

**Files:**
- Modify: `ggml/src/ggml-sycl/moe-xmx.hpp:158-191`

**Step 1: Add float accumulators before K-loop**

Before the K-loop, add per-tile float accumulators:

```cpp
                // Float accumulators for precision across K-blocks
                float float_acc[TILES_M][TILES_N][XMX_M * XMX_N] = {{{0.0f}}};
```

**Step 2: Replace the output store section**

Replace the placeholder output store (~lines 166-188) with:

```cpp
                    // === Apply scales and accumulate ===
                    // For each tile, extract int32 result, apply scales, add to float accumulator
                    for (int tm = 0; tm < TILES_M; tm++) {
                        for (int tn = 0; tn < TILES_N; tn++) {
                            int row = wg_row + tm * XMX_M;
                            int col = wg_col + tn * XMX_N;

                            if (row < batch && col < out_dim) {
                                // Store accumulator to SLM to extract values
                                auto acc_ptr = sycl::address_space_cast<
                                    sycl::access::address_space::local_space,
                                    sycl::access::decorated::no>(
                                        slm_weights.get_pointer());  // Reuse weight SLM
                                joint_matrix_store(sg, acc[tm][tn], acc_ptr, XMX_N,
                                    layout::row_major);

                                item.barrier(sycl::access::fence_space::local_space);

                                // Apply scales and accumulate in float
                                float t_scale = slm_token_scales[tm];
                                float w_scale = slm_weight_scales[tn];
                                float combined_scale = t_scale * w_scale;

                                int lane = sg.get_local_linear_id();
                                for (int i = lane; i < XMX_M * XMX_N; i += SG_SIZE) {
                                    int32_t raw = reinterpret_cast<int32_t*>(
                                        slm_weights.get_pointer())[i];
                                    float_acc[tm][tn][i] += raw * combined_scale;
                                }

                                // Reset accumulator for next K-block
                                joint_matrix_fill(sg, acc[tm][tn], 0);

                                item.barrier(sycl::access::fence_space::local_space);
                            }
                        }
                    }
                } // End K-loop

                // === Final output store ===
                for (int tm = 0; tm < TILES_M; tm++) {
                    for (int tn = 0; tn < TILES_N; tn++) {
                        int row = wg_row + tm * XMX_M;
                        int col = wg_col + tn * XMX_N;

                        if (row < batch && col < out_dim) {
                            int lane = sg.get_local_linear_id();
                            for (int i = lane; i < XMX_M * XMX_N; i += SG_SIZE) {
                                int tile_row = i / XMX_N;
                                int tile_col = i % XMX_N;
                                if (row + tile_row < batch && col + tile_col < out_dim) {
                                    output[(row + tile_row) * out_dim + col + tile_col] =
                                        sycl::half(float_acc[tm][tn][i]);
                                }
                            }
                        }
                    }
                }
```

**Step 3: Verify build compiles**

Run: `source /opt/intel/oneapi/setvars.sh --force && cmake --build build -j 16 2>&1 | tail -10`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add ggml/src/ggml-sycl/moe-xmx.hpp
git commit -m "feat(sycl): implement scale application for XMX MoE Q8_0"
```

---

### Task 5: Update Dispatcher to Call Pre-Quantization

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp:11429-11548`

**Step 1: Find the XMX sorted MoE dispatch section**

Search for `launch_xmx_moe_gemm_q8_0` in ggml-sycl.cpp.

**Step 2: Add temporary buffer allocation for quantized tokens**

Before the expert loop, add:

```cpp
    // Allocate temporary buffers for quantized tokens
    int64_t in_dim = src0->ne[0];
    int64_t num_k_blocks = in_dim / 32;

    int8_t* q_tokens = sycl::malloc_device<int8_t>(
        total_tokens * in_dim, *main_stream);
    sycl::half* token_scales = sycl::malloc_device<sycl::half>(
        total_tokens * num_k_blocks, *main_stream);

    // Pre-quantize sorted tokens (one pass for all experts)
    moe_xmx::preprocess_tokens_q8(
        sorted_tokens_ptr, q_tokens, token_scales,
        total_tokens, in_dim, *main_stream);
```

**Step 3: Update the kernel call to pass quantized tokens**

Change the `launch_xmx_moe_gemm_q8_0` call to use `q_tokens` and `token_scales` instead of `sorted_tokens_ptr`.

**Step 4: Add buffer cleanup after expert loop**

After the expert loop:

```cpp
    sycl::free(q_tokens, *main_stream);
    sycl::free(token_scales, *main_stream);
```

**Step 5: Verify build compiles**

Run: `source /opt/intel/oneapi/setvars.sh --force && cmake --build build -j 16 2>&1 | tail -10`
Expected: Build succeeds

**Step 6: Commit**

```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "feat(sycl): integrate token pre-quantization in XMX MoE dispatch"
```

---

### Task 6: Test Q8_0 XMX MoE Path

**Files:**
- Test: existing benchmarks

**Step 1: Run correctness test with XMX enabled**

```bash
source /opt/intel/oneapi/setvars.sh --force
GGML_SYCL_XMX_MOE=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -ngl 99 --flash-attn on \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

Expected: "1, 2, 3, 4, 5, 6, 7, 8, 9, 10..." (correct counting)

**Step 2: Run performance benchmark**

```bash
GGML_SYCL_XMX_MOE=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -p 512 -n 128 -ngl 99 -fa 1
```

Expected: Performance similar to or better than fused ESIMD path

**Step 3: Compare with XMX disabled**

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -p 512 -n 128 -ngl 99 -fa 1
```

Document performance difference.

**Step 4: Commit test results**

```bash
git commit --allow-empty -m "test(sycl): verify XMX MoE Q8_0 correctness and performance

Correctness: Counting test produces expected output
Performance: [fill in actual numbers]"
```

---

### Task 7: Implement MXFP4 Weight Unpacking

**Files:**
- Modify: `ggml/src/ggml-sycl/moe-xmx.hpp:203-350`

**Step 1: Add kvalues_mxfp4 constant**

At the top of the namespace (after `using namespace`), add:

```cpp
// MXFP4 E2M1 lookup table (values fit in int8 range)
constexpr int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12,
    0, -1, -2, -3, -4, -6, -8, -12
};
```

**Step 2: Update MXFP4 kernel signature**

Change to accept pre-quantized tokens:

```cpp
template<int TILES_M = 4, int TILES_N = 4>
void launch_xmx_moe_gemm_mxfp4(
    const void* weights_qs,           // [out_dim, in_dim/32 * 17] MXFP4 packed
    const int8_t* q_tokens,           // [batch, in_dim] pre-quantized int8
    const sycl::half* token_scales,   // [batch, in_dim/32] token scales
    sycl::half* output,               // [batch, out_dim]
```

**Step 3: Add MXFP4 unpacking to SLM**

In the K-loop, add weight unpacking:

```cpp
                    // === Cooperative MXFP4 weight unpacking to SLM ===
                    // MXFP4 block: 16 bytes packed + 1 byte E8M0 = 17 bytes per 32 values
                    constexpr int MXFP4_BLOCK = 17;
                    const uint8_t* w_base = static_cast<const uint8_t*>(weights_qs);

                    // Each sub-group unpacks part of weight tile
                    for (int tn = 0; tn < TILES_N; tn++) {
                        int col = wg_col + tn * XMX_N;
                        if (col < out_dim && sg_id == tn % (cfg.wg_size / SG_SIZE)) {
                            // Block offset in MXFP4 layout
                            int64_t k_block = k / XMX_K;
                            int64_t blocks_per_row = in_dim / XMX_K;
                            const uint8_t* block_ptr = w_base +
                                (col * blocks_per_row + k_block) * MXFP4_BLOCK;

                            // Extract E8M0 exponent
                            uint8_t e8m0 = block_ptr[16];
                            float exp_scale = sycl::pow(2.0f, static_cast<float>(e8m0) - 127.0f);

                            // Store exponent scale for later
                            if (item.get_local_id(1) == 0) {
                                slm_weight_scales[tn] = exp_scale * 0.5f;  // Include 0.5 for LUT
                            }

                            // Unpack 4-bit values to int8 using LUT
                            int lane = sg.get_local_linear_id();
                            for (int i = lane; i < 16; i += SG_SIZE) {
                                uint8_t packed = block_ptr[i];
                                int8_t low = kvalues_mxfp4[packed & 0x0F];
                                int8_t high = kvalues_mxfp4[packed >> 4];
                                int slm_offset = tn * XMX_N * XMX_K + i * 2;
                                slm_weights[slm_offset] = low;
                                slm_weights[slm_offset + 1] = high;
                            }
                        }
                    }

                    item.barrier(sycl::access::fence_space::local_space);
```

**Step 4: Verify build compiles**

Run: `source /opt/intel/oneapi/setvars.sh --force && cmake --build build -j 16 2>&1 | tail -10`
Expected: Build succeeds

**Step 5: Commit**

```bash
git add ggml/src/ggml-sycl/moe-xmx.hpp
git commit -m "feat(sycl): implement MXFP4 weight unpacking for XMX MoE"
```

---

### Task 8: Complete MXFP4 GEMM Implementation

**Files:**
- Modify: `ggml/src/ggml-sycl/moe-xmx.hpp:280-350`

**Step 1: Add mat_b loading from unpacked SLM**

After weight unpacking, add:

```cpp
                    // Load mat_b from unpacked weights in SLM
                    auto slm_weights_ptr = sycl::address_space_cast<
                        sycl::access::address_space::local_space,
                        sycl::access::decorated::no>(
                            slm_weights.get_pointer() + tn * XMX_N * XMX_K);
                    joint_matrix_load(sg, mat_b, slm_weights_ptr, XMX_N);
```

**Step 2: Copy scale accumulation pattern from Q8_0**

Apply the same scale accumulation and output store pattern used in Q8_0 kernel.

**Step 3: Verify build compiles**

Run: `source /opt/intel/oneapi/setvars.sh --force && cmake --build build -j 16 2>&1 | tail -10`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add ggml/src/ggml-sycl/moe-xmx.hpp
git commit -m "feat(sycl): complete MXFP4 XMX MoE GEMM implementation"
```

---

### Task 9: Update Dispatcher for MXFP4

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp:11529-11548`

**Step 1: Update the MXFP4 kernel call**

In the type dispatch section, update the MXFP4 call to pass quantized tokens:

```cpp
} else if (src0->type == GGML_TYPE_MXFP4) {
    moe_xmx::launch_xmx_moe_gemm_mxfp4<4, 4>(
        expert_weights, q_tokens, token_scales,
        output_ptr, batch_size, out_dim, in_dim,
        xmx_cfg, *main_stream);
}
```

**Step 2: Verify build compiles**

Run: `source /opt/intel/oneapi/setvars.sh --force && cmake --build build -j 16 2>&1 | tail -10`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "feat(sycl): update MXFP4 dispatch to use pre-quantized tokens"
```

---

### Task 10: Final Integration Test

**Files:**
- Test: existing benchmarks

**Step 1: Test dense model (no MoE)**

```bash
source /opt/intel/oneapi/setvars.sh --force
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p 512 -n 128 -ngl 99 -fa 1
```

Expected: ~42 t/s tg128 (no regression)

**Step 2: Test MoE model with XMX**

```bash
GGML_SYCL_XMX_MOE=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -ngl 99 --flash-attn on \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

Expected: "1, 2, 3, 4, 5, 6, 7, 8, 9, 10..."

**Step 3: Final commit with test results**

```bash
git add -A
git commit -m "feat(sycl): complete XMX MoE GEMM implementation

Implements working XMX-accelerated GEMM kernels for MoE dispatch:
- Token pre-quantization (fp16 -> int8 with per-block scales)
- Q8_0 weight handling with SLM staging
- MXFP4 weight unpacking via kvalues LUT
- Per-K-block scale accumulation

Correctness verified with counting test.
Performance: [fill in numbers]"
```

---

## Summary of Changes

| File | Changes |
|------|---------|
| `moe-xmx.hpp` | Add preprocess_tokens_q8, update kernel signatures, implement SLM loading, add scale accumulation, add MXFP4 unpacking |
| `ggml-sycl.cpp` | Allocate quantized token buffers, call pre-quantization, update kernel calls |

## Risk Assessment

- **Medium complexity**: Following proven patterns from mmq_xmx.cpp
- **SLM constraints**: Verified 5-6KB usage fits well within 64KB limit
- **Scale precision**: Float accumulation across K-blocks prevents overflow
- **Fallback**: XMX path only activates with GGML_SYCL_XMX_MOE=1
