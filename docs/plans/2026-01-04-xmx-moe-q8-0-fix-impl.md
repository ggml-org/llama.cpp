# XMX MoE Output Layout Fix Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Fix XMX MoE GEMM kernels to produce correct numerical output instead of garbage.

**Tech Stack:** Intel SYCL, XMX joint_matrix API, Intel Arc B580 GPU

---

## Root Cause Analysis (CORRECTED)

**Original plan was WRONG about output layout.**

Investigation proved MoE and MMQ have DIFFERENT output contexts:

| Context | Output Target | Required Layout | Consumer |
|---------|---------------|-----------------|----------|
| **MMQ** | ggml tensor `dst` | Column-major | ggml tensor system |
| **MoE XMX** | intermediate `sorted_output` | **Row-major** | `moe_scatter_results_f16_to_f32` |

**Evidence:** Scatter function reads `sorted_output[idx * output_dim + d]` which is row-major.

**Status:**
- ✅ Q8_0 output: Already fixed to row-major (lines 416-439)
- ❌ MXFP4 output: Still uses column-major (lines 750-772) - NEEDS FIX

---

## Task 1: Fix MXFP4 Output to Row-Major

**File:** `ggml/src/ggml-sycl/moe-xmx.hpp`
**Lines:** 750-772

**Step 1: Change comment at line 750-751**

FROM:
```cpp
                // === Final output store (column-major to match reference MMQ) ===
                // Reference: dst[col * nrows_dst + row] where nrows_dst = batch
```

TO:
```cpp
                // === Final output store (row-major for MoE scatter) ===
                // MoE scatter expects: sorted_output[pair_idx * output_dim + dim]
                // This differs from MMQ which outputs column-major to ggml tensors
```

**Step 2: Change store formula at line 762-764**

FROM:
```cpp
                                if (row + tile_row < batch && col + tile_col < out_dim) {
                                    // Column-major: output[col * batch + row]
                                    output[(col + tile_col) * batch + row + tile_row] =
                                        sycl::half(float_acc[tm][tn][i]);
                                }
```

TO:
```cpp
                                if (row + tile_row < batch && col + tile_col < out_dim) {
                                    // Row-major: output[row * out_dim + col]
                                    output[(row + tile_row) * out_dim + col + tile_col] =
                                        sycl::half(float_acc[tm][tn][i]);
                                }
```

---

## Task 2: Rebuild

**Command:**
```bash
source /opt/intel/oneapi/setvars.sh --force && cmake --build build -j 16
```

**Expected:** Successful compilation with no errors.

---

## Task 3: Test Correctness

**Step 1: Run XMX-enabled test**
```bash
GGML_SYCL_XMX_MOE=1 GGML_SYCL_DISABLE_GRAPH=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -ngl 99 --flash-attn on --no-conversation \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

**Expected output:** `1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20`

**Step 2: Compare with baseline (XMX disabled)**
```bash
GGML_SYCL_DISABLE_GRAPH=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -ngl 99 --flash-attn on --no-conversation \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

**Verification:** Outputs must match exactly.

---

## Task 4: Run Performance Benchmark

**Command:**
```bash
GGML_SYCL_XMX_MOE=1 GGML_SYCL_DISABLE_GRAPH=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -p 512 -n 128 -ngl 99 -fa 1
```

**Record:** t/s for prompt processing and token generation.

---

## Success Criteria

1. Count prompt produces correct sequential numbers
2. XMX output matches baseline output exactly
3. No performance regression vs baseline
