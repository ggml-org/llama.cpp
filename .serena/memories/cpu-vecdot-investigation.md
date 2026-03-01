# CPU vec_dot Implementation Investigation (SYCL cpu-dispatch)

## Overview
Comprehensive investigation of how CPU vec_dot is called in the SYCL cpu-dispatch path, including the `nrc` (number of rows computed) parameter and available CPU instruction sets.

## 1. SYCL CPU Dispatch: cpu_mul_mat Function (ggml-sycl/cpu-dispatch.cpp)

### Location: `/Apps/llama.cpp/ggml/src/ggml-sycl/cpu-dispatch.cpp:1118-1373`

**Function signature:**
```cpp
static bool cpu_mul_mat(ggml_backend_sycl_context & ctx, ggml_tensor * dst)
```

### Key Implementation Details:

#### vec_dot Usage (Lines 1207-1318):
- **M threshold for vec_dot**: `M <= 4` (line 1213)
  - Activations are quantized on-the-fly (quantize_row_q4_0 -> Q8_0, etc)
  - Direct quantized vec_dot replaces dequantization + BLAS overhead
  - ~5x less memory bandwidth vs dequant path

#### vec_dot Call Signature (Lines 1289-1291, 1301-1303, 1313-1315):
```cpp
cpu_traits->vec_dot(
    static_cast<int>(K),           // n: number of elements
    &dot_result,                   // s: output scalar
    sizeof(float),                 // bs: output stride (always sizeof(float) for single-row)
    weight_row,                    // x: quantized weights
    0,                             // bx: stride for x (always 0 - sequential)
    src1_q_data + m * q_row_size, // y: quantized activations
    0,                             // by: stride for y (always 0 - sequential)
    1                              // nrc: ALWAYS 1 in this path
);
```

#### Row Processing:
- **Outer loops** (lines 1238-1239): batch dimensions (ne13, ne12)
- **Inner loops** (lines 1285-1318): N (weight rows) x M (activation rows)
  - When M <= 4: quantized path
  - Otherwise: GEMM path (dnnl_sgemm at line 1335)

**Parallelization** (lines 1272-1306):
- **TBB parallel_for** when:
  - N > 1 AND n_threads_hint > 1 
  - total_work (N*M) >= GGML_SYCL_CPU_VECDOT_MIN_PARALLEL_WORK() (line 1272)
  - Grain size: max(grain_from_target, GGML_SYCL_CPU_VECDOT_MIN_ROWS_PER_TASK()) (line 1276)
- **Serial path** for small N or single thread (lines 1297-1305)

#### Key Insight:
**NRC is ALWAYS 1 in cpu_dispatch.cpp** - the function never calls vec_dot with nrc > 1. Multi-row processing happens at the tensor level (iterating M rows), not within vec_dot.

---

## 2. CPU Backend vec_dot: type_traits_cpu Structure

### Location: `/Apps/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c:204-307`

### Type Traits Array Definition (Q4_0 example):
```cpp
[GGML_TYPE_Q4_0] = {
    .from_float               = quantize_row_q4_0,
    .vec_dot                  = ggml_vec_dot_q4_0_q8_0,
    .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
    .nrows                    = 2,   // <-- MULTI-ROW SUPPORT ON ARM
#else
    .nrows                    = 1,   // <-- SINGLE-ROW ON x86/GENERIC
#endif
},
```

### Key Point:
- `.nrows` field indicates **max simultaneous rows** a vec_dot can process
- **ARM MATMUL INT8 support**: nrows=2 for Q4_0, Q4_1, Q8_0, Q6_K
- **x86/generic**: nrows=1 only
- **Intel SYCL uses x86 path**: nrows=1 always

---

## 3. Generic vec_dot_q4_0_q8_0 Implementation

### Location: `/Apps/llama.cpp/ggml/src/ggml-cpu/quants.c:115-149`

```cpp
void ggml_vec_dot_q4_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs,
                                    const void * GGML_RESTRICT vx, size_t bx,
                                    const void * GGML_RESTRICT vy, size_t by,
                                    int nrc) {
    const int qk = QK8_0;  // 32
    const int nb = n / qk; // number of blocks
    
    assert(n % qk == 0);
    assert(nrc == 1);      // <-- GENERIC ASSERTS nrc == 1
    UNUSED(nrc);
    UNUSED(bx);            // always 0
    UNUSED(by);            // always 0
    UNUSED(bs);            // stride not used in single-row
    
    const block_q4_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;
    
    int ib = 0;
    float sumf = 0;
    
    for (; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;
        
        // Inner loop: process block (32 elements = QK8_0)
        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F) - 8;  // lower nibble
            const int v1 = (x[ib].qs[j] >>   4) - 8;  // upper nibble
            
            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }
        
        int sumi = sumi0 + sumi1;
        // Scale by block scale: d * scale
        sumf += sumi * GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d);
    }
    
    *s = sumf;
}
```

### Memory Access Pattern:
- **Sequential**: both x and y accessed sequentially (no stride)
- **Block-wise**: processes 32 elements per block (QK8_0)
- **Within block**: 16 x (2-nibble unpacking + multiply-accumulate)
- **L1-friendly**: ~2KB weight block + ~32B activation block per dot

### Data Access:
- `x[ib].qs[j]` - Q4_0 quantized weights (2x 4-bit values)
- `y[ib].qs[j]` - Q8_0 quantized activations (1x 8-bit value)
- Both are **affine integer operations**, no floating point until final scale multiply

---

## 4. x86 Specialized vec_dot_q4_0_q8_0 Implementation

### Location: `/Apps/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c:543-715`

### SIMD Codegen Paths (in order of specificity):
1. **AVX2** (lines 560-599): Most common on modern Intel
   - Two blocks at a time (`ib + 1`)
   - `_mm256` operations (256-bit vectors)
   - Uses `mul_sum_i8_pairs_float()` for int8x8 -> float accumulation
   - Prefetch hints for next blocks (L1/L2 temporal hints)

2. **AVX** (lines 600-627): 256-bit float but 128-bit int handling
   - Loads int8 data with `_mm_loadu_si128`
   - Manual nibble unpacking and subtraction
   - Conversion to float via `_mm_cvtepi32_ps`

3. **SSSE3** (lines 628-695): 128-bit operations
   - Four 32-bit float accumulators (`acc_0`, `acc_1`, `acc_2`, `acc_3`)
   - Processes two blocks per iteration
   - `mul_sum_i8_pairs()` returns 128-bit int results

4. **Fallback Generic** (lines 698-714): scalar C code
   - Same as generic implementation above
   - Used when AVX/SSSE3 unavailable (rare)

### Key SIMD Characteristics:
- **Prefetch**: L0 temporal hints for next blocks (predictive)
- **Loop unrolling**: 2-block chunks in most paths
- **No VNNI**: Intel AVX-VNNI (vpdpbusd) NOT used for int8x8 dot
- **No AVX-512**: No specialized AVX-512 path (falls through to AVX2)

---

## 5. ARM MATMUL INT8 Multi-Row Path

### Location: `/Apps/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c:140-228`

**CRITICAL**: ARM can process `nrc=2` rows simultaneously!

```cpp
void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs,
                            const void * GGML_RESTRICT vx, size_t bx,
                            const void * GGML_RESTRICT vy, size_t by,
                            int nrc) {
    assert((nrc == 2) || (nrc == 1));  // <-- ALLOWS nrc=2
    
#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (nrc == 2) {
        // Extract two separate weight/activation streams
        const block_q4_0 * GGML_RESTRICT vx0 = vx;
        const block_q4_0 * GGML_RESTRICT vx1 = (const block_q4_0 *)((const uint8_t*)vx + bx);
        const block_q8_0 * GGML_RESTRICT vy0 = vy;
        const block_q8_0 * GGML_RESTRICT vy1 = (const block_q8_0 *)((const uint8_t*)vy + by);
        
        float32x4_t sumv0 = vdupq_n_f32(0.0f);  // result 1
        
        for (int i = 0; i < nb; i++) {
            // ... unpack Q4_0 nibbles to int8x16 ...
            
            // ARM MATMUL instruction: processes two 8-element dot products
            sumv0 = vmlaq_f32(sumv0,
                              (vcvtq_f32_s32(vmmlaq_s32(
                                   vmmlaq_s32(...x0_l, r0),
                                   ...x0_h, r0_h),
                                   ...x1_l, r1),
                                   ...x1_h, r1_h))
                              ), scale);
        }
        
        // Extract both results from sumv0
        vst1_f32(s,      vget_low_f32(sumv0));
        vst1_f32(s + bs, vget_high_f32(sumv0));  // <-- uses bs stride
        return;
    }
#endif
    
    // nrc == 1 path (same as generic)
    ...
}
```

### Multi-Row Specifics:
- **bx, by parameters**: byte strides to second weight/activation row
- **bs parameter**: output stride (typically 16 = 4 floats)
- **vmmlaq_s32**: ARM Integer Matrix Multiply accumulate (4x4 int8 blocks)
- **Interleave/zip operations**: Prepare data for mmla (lines 205-215)

### ARM Limitations (Intel SYCL unaffected):
- Only Q4_0, Q4_1, Q8_0, Q6_K support nrc=2
- Q5_0, Q5_1, Q2_K, Q3_K, Q5_K do NOT support nrc=2

---

## 6. CPU Backend MUL_MAT with vec_dot_num_rows

### Location: `/Apps/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c:1226-1410`

### Multi-Row Integration in ggml_compute_forward_mul_mat (CPU backend only):

```cpp
enum ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
int64_t vec_dot_num_rows = type_traits_cpu[src0->type].nrows;  // 1 or 2

// Line 1407-1409: Check if nrc=2 is safe
if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) 
    || ((ir1_end - ir1_start) % 2 != 0)) {
    num_rows_per_vec_dot = 1;  // Fall back to nrc=1 if dimensions don't align
}

// Line 1215: Call vec_dot with dynamic nrc
vec_dot(ne00,                           // n
        &tmp[ir0 - iir0],               // s
        (num_rows_per_vec_dot > 1 ? 16 : 0),  // bs: stride or 0
        src0_row + ir0 * nb01,           // x
        (num_rows_per_vec_dot > 1 ? nb01 : 0), // bx: stride or 0
        src1_col,                        // y
        (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), // by: stride or 0
        num_rows_per_vec_dot);           // nrc: 1 or 2
```

### Key Points:
- **ggml/src/ggml-cpu backend ONLY** supports nrc > 1
- **cpu-dispatch (SYCL) NEVER** uses nrc > 1 (always 1)
- **Dimension alignment checks** ensure even-ness before attempting nrc=2
- **Stride parameters**: only used when nrc=2

---

## 7. CPU Feature Detection & Build Configuration

### Compiler Feature Defines:
- **`__AVX2__`**: Auto-defined by compiler when `-march=native` or higher
- **`__AVX__`**: SSE2 + floating-point operations (older baseline)
- **`__SSSE3__`**: Supplemental SSE3 (shuffle intrinsics)
- **`__AVX_VNNI__`**: AVX Vector Neural Network Instructions (not used in Q4_0)
- **`__AVX512F__`**: AVX-512 Foundation (not actively used)
- **`__ARM_FEATURE_MATMUL_INT8`**: ARM v8.2 integer matrix multiply

### CMakeLists.txt Configuration:
- **GGML_NATIVE** (default ON): Uses `-march=native` (lines 309)
- **GGML_SSE42**: Fallback to `-msse4.2` if not native
- **GGML_CPU_GENERIC**: Flag-based generic implementation (line 545)
- **x86 selection**: autodetects to CPU arch (arch/x86/quants.c)

### On Intel SYCL System (Arc B580):
- **Likely flags**: `-march=native` → AVX2/SSE4.2/AES-NI
- **Actual path**: AVX2 vec_dot (lines 560-599) for Q4_0
- **No VNNI**: Even if available, Q4_0 doesn't use it

---

## 8. CPU Dispatch vs CPU Backend

### SYCL cpu-dispatch.cpp (Host compute path):
- ✓ Quantized dot product for M <= 4
- ✓ Parallel TBB for large N
- ✓ **Always nrc=1**
- ✓ Host-pinned buffers (zero-copy GPU access)
- ✗ No nrc>1 support

### ggml-cpu backend (Regular CPU inference):
- ✓ dnnl_sgemm GEMM for all sizes
- ✓ Quantized dot with nrc=1 or nrc=2
- ✓ ARM-only multi-row (nrc=2)
- ✗ Not used in SYCL context

---

## 9. Summary: nrc Parameter Meaning

| Parameter | Q4_0 Generic | Q4_0 x86 AVX2 | Q4_0 ARM MATMUL |
|-----------|---|---|---|
| **nrc=1** | Single (N,K)→float | 2 blocks at once, 1 output | 1 output |
| **nrc=2** | ASSERTS FAIL | Not reached | 2 outputs, stride bx/by |
| **bs stride** | Ignored | Ignored | **Used**: space between outputs |
| **bx stride** | Always 0 | Always 0 | **Used**: offset to 2nd weight |
| **by stride** | Always 0 | Always 0 | **Used**: offset to 2nd activation |

### cpu-dispatch.cpp Reality:
- **Always passes nrc=1** (lines 1290, 1302, 1314)
- bs/bx/by always 0
- Per-element vec_dot call (scalar output)
- Outer loops handle multiple rows

---

## 10. Memory Layout & Access Patterns

### Q4_0 Weight Block:
```
struct block_q4_0 {
    ggml_fp16_t d;      // 2 bytes: scale
    uint8_t qs[16];     // 16 bytes: 32 x 4-bit values
};
// 18 bytes per block = 18B * (K/32) per weight row
```

### Q8_0 Activation Block:
```
struct block_q8_0 {
    ggml_fp16_t d;      // 2 bytes: scale
    int8_t qs[32];      // 32 bytes: 32 x 8-bit values
};
// 34 bytes per block = 34B * (K/32) per activation row
```

### Contiguous Access:
- Weights: sequential block_q4_0 array (AOS layout after cache)
- Activations: quantized on-the-fly from F32 rows (contiguous F32 src)
- **Cache-friendly**: One 2KB weight block + 4KB activations per dot

---

## Files & Line Numbers Summary

| File | Lines | Content |
|------|-------|---------|
| `/Apps/llama.cpp/ggml/src/ggml-sycl/cpu-dispatch.cpp` | 1118-1373 | cpu_mul_mat, vec_dot calls |
| `/Apps/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` | 204-307 | type_traits_cpu array |
| `/Apps/llama.cpp/ggml/src/ggml-cpu/quants.c` | 115-149 | Generic Q4_0 vec_dot |
| `/Apps/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c` | 543-715 | x86 Q4_0 SIMD paths |
| `/Apps/llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c` | 140-228 | ARM MATMUL nrc=2 support |
| `/Apps/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` | 1226-1410 | MUL_MAT multi-row logic |
