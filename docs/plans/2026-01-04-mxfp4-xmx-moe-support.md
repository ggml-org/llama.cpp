# MXFP4 XMX MoE Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the XMX sorted MoE dispatch path to support MXFP4-quantized expert weights in addition to Q8_0.

**Architecture:** The XMX sorted MoE path uses Intel XMX matrix extensions with token sorting for batched GEMM. Currently only Q8_0 is supported. We add a parallel MXFP4 kernel template using the same three-phase approach (Sort → GEMM → Scatter) and type dispatch in the main dispatcher.

**Tech Stack:** Intel SYCL, XMX joint_matrix API (8x16x32 tiles), MXFP4 E2M1 format with E8M0 exponent

---

## Background: MXFP4 Format

MXFP4 (Microscaling 4-bit Floating Point) uses:
- 32 4-bit E2M1 values packed in 16 bytes
- 1 byte E8M0 shared exponent
- Total: 17 bytes per block (vs Q8_0's 34 bytes)
- Dequantization: `d * kvalues_mxfp4[nibble] * 0.5f` where d = 2^(e8m0_exp - 127)

---

### Task 1: Add MXFP4 XMX GEMM Kernel Skeleton

**Files:**
- Modify: `ggml/src/ggml-sycl/moe-xmx.hpp`

**Step 1: Add MXFP4 kernel template**

Add after `launch_xmx_moe_gemm_q8_0`:

```cpp
// MXFP4 XMX GEMM for a single expert's token batch
// SKELETON STATUS: This kernel compiles but produces INCORRECT output.
// Full implementation requires:
// 1. Proper MXFP4 unpacking (4-bit -> 8-bit for XMX int8 operands)
// 2. E8M0 exponent application per block
// 3. LUT-based dequantization using kvalues_mxfp4
template<int TILES_M = 4, int TILES_N = 4>
void launch_xmx_moe_gemm_mxfp4(
    const void* weights_qs,       // [out_dim, in_dim] MXFP4 quantized (17 bytes per 32 elements)
    const sycl::half* tokens,     // [batch, in_dim] fp16 activations
    sycl::half* output,           // [batch, out_dim]
    int64_t batch,
    int64_t out_dim,
    int64_t in_dim,
    const MoEXMXConfig& cfg,
    sycl::queue& queue)
{
    constexpr int XMX_M = 8;
    constexpr int XMX_N = 16;
    constexpr int XMX_K = 32;
    constexpr int SG_SIZE = 16;
    constexpr int MXFP4_BLOCK_SIZE = 17;  // 16 bytes packed + 1 byte E8M0 exponent

    // ... skeleton implementation that writes zeros ...
    // See full implementation in moe-xmx.hpp
}
```

**Step 2: Verify build compiles**

Run: `source /opt/intel/oneapi/setvars.sh --force && cmake --build build -j 16 2>&1 | tail -10`
Expected: Build succeeds with no errors

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/moe-xmx.hpp
git commit -m "feat(sycl): add MXFP4 XMX GEMM kernel skeleton for MoE"
```

---

### Task 2: Add Type Dispatch in XMX MoE Dispatcher

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp:11429-11548`

**Step 1: Update type check to accept MXFP4**

Change line ~11430 from:
```cpp
if (src0->type != GGML_TYPE_Q8_0) {
```

To:
```cpp
if (src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_MXFP4) {
```

**Step 2: Add type dispatch in GEMM loop**

Around line ~11536, add type switch:
```cpp
if (src0->type == GGML_TYPE_Q8_0) {
    moe_xmx::launch_xmx_moe_gemm_q8_0<4, 4>(...);
} else if (src0->type == GGML_TYPE_MXFP4) {
    moe_xmx::launch_xmx_moe_gemm_mxfp4<4, 4>(...);
}
```

**Step 3: Update comment to reflect MXFP4 support**

Change comment at ~line 11599:
```cpp
// Try XMX sorted MoE path (experimental, Q8_0/MXFP4, requires GGML_SYCL_XMX_MOE=1)
```

**Step 4: Verify build compiles**

Run: `source /opt/intel/oneapi/setvars.sh --force && cmake --build build -j 16 2>&1 | tail -10`
Expected: Build succeeds

**Step 5: Commit**

```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "feat(sycl): add MXFP4 type dispatch in XMX MoE path"
```

---

### Task 3: Verify No Regression on Q8_0 Models

**Files:**
- Test only (no modifications)

**Step 1: Run Q8_0 MoE model benchmark (XMX path disabled)**

```bash
source /opt/intel/oneapi/setvars.sh --force
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -p 512 -n 128 -ngl 99 -fa 1
```

Expected: Performance matches baseline (~280 t/s pp, ~31 t/s tg)

**Step 2: Run Q8_0 correctness test**

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -ngl 99 --flash-attn on -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

Expected: "1, 2, 3, 4, 5, 6, 7, 8, 9, 10..."

**Step 3: Run dense model canary test**

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p 512 -n 128 -ngl 99 -fa 1
```

Expected: No regression from baseline (~42 t/s tg128)

---

### Task 4: Final Commit

**Files:**
- All modified files

**Step 1: Stage and commit all changes**

```bash
git add ggml/src/ggml-sycl/moe-xmx.hpp ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "feat(sycl): add MXFP4 support to XMX sorted MoE path

Extends the experimental XMX-accelerated MoE dispatch to handle
MXFP4-quantized expert weights alongside Q8_0:

- Add launch_xmx_moe_gemm_mxfp4 kernel template in moe-xmx.hpp
- Update type check to accept GGML_TYPE_MXFP4
- Add type dispatch to call appropriate kernel

Both kernels are skeleton status (produce zeros) - full XMX
implementation is future work. The infrastructure enables
testing the sort/scatter phases with MXFP4 models."
```

---

## Summary of Changes

1. **moe-xmx.hpp**: Add `launch_xmx_moe_gemm_mxfp4` kernel template
2. **ggml-sycl.cpp:11430**: Update type check to accept MXFP4
3. **ggml-sycl.cpp:11529-11548**: Add type dispatch for MXFP4 kernel
4. **ggml-sycl.cpp:11599**: Update comment to reflect MXFP4 support

## Risk Assessment

- **Low risk**: MXFP4 path only activates with `GGML_SYCL_XMX_MOE=1` env var
- **Skeleton status**: Kernels compile but produce incorrect output (by design)
- **No regression**: Q8_0 path unchanged, verified with benchmarks
