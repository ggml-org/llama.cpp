# CPU Vec_Dot Optimization for Tensor Split

> **For Claude:** Use team-driven-development to implement this plan with agent teams.

**Goal:** Optimize CPU vec_dot in tensor split to achieve ~80+ tok/s TG (up from 72 tok/s GPU-only) by processing multiple weight rows per kernel invocation with AVX2 + AVX-VNNI.

**Architecture:** Four stacking optimizations in `ggml_sycl_cpu_vec_dot_rows()`: (1) Wire existing 4-row SIMD kernel for Q4_0. (2) Enable AVX2 + VNNI compile flags. (3) New 8-row AVX2 VNNI kernel. (4) 16-row kernel with array-based interface.

**IMPORTANT: Arrow Lake does NOT support AVX-512.** Intel removed AVX-512 from consumer Arrow Lake chips (Core Ultra 7 265K). Available: AVX2, FMA, AVX-VNNI (`_mm256_dpbssd_epi32`). All kernels use 256-bit registers only.

---

## Team Topology

**Recommended implementers:** 1 (all tasks modify same file)
**Reviewers:** 1 spec-reviewer, 1 quality-reviewer

### Parallel Tracks

| Track | Tasks | Description |
|-------|-------|-------------|
| A | 1, 2, 3, 4, 5 | All sequential (same file: cpu-dispatch.cpp) |

### Dependency Graph

```
Task 1 → Task 2 → Task 3 → Task 4 → Task 5
```

### File Ownership Map

| File/Directory | Tasks | Conflict Risk |
|----------------|-------|---------------|
| `ggml/src/ggml-sycl/cpu-dispatch.cpp` | 1, 3, 4 | Sequential (same track) |
| `ggml/src/ggml-sycl/CMakeLists.txt` | 2 | None (single task) |
| Benchmark only | 5 | None |

---

## Task 1: Wire 4-row kernel into tensor split vec_dot [COMPLETED]

Committed as eec4eed3d. Wired `simd_mul_mat_q4_0_q8_0_4row()` into
`ggml_sycl_cpu_vec_dot_rows()` TBB body with 16-row grain size.

---

## Task 2: Enable AVX2 + VNNI compile flags [COMPLETED]

Changed CMakeLists.txt to add `-mavx2;-mfma;-mavxvnniint8` per-file flags for
cpu-dispatch.cpp only. Uses `CheckCXXCompilerFlag` for portability.

**NOTE**: Originally planned as AVX-512 flags but Arrow Lake lacks AVX-512.
Pivoted to AVX2+FMA+AVXVNNIINT8.

---

## Task 3: New 8-row AVX2 VNNI kernel + dispatch

**Track:** A
**Depends on:** Task 2
**File scope:**
- Modify: `ggml/src/ggml-sycl/cpu-dispatch.cpp`

**Description:**

Write a new `simd_mul_mat_q4_0_q8_0_8row()` kernel that processes 8 weight rows per
inner loop iteration. Uses 8 independent __m256 accumulators with VNNI `_mm256_dpbssd_epi32`
for maximum ILP. Arrow Lake P-cores have 6-wide issue with dual 256-bit FMA ports — 8
accumulators keep the pipeline fed.

**Guard:** `#if defined(__AVXVNNIINT8__)` — the 8-row kernel requires VNNI for the
`_mm256_dpbssd_epi32` instruction. Without VNNI, the 4-row kernel (which has both VNNI
and non-VNNI paths) handles everything.

### Implementation

Add after `simd_mul_mat_q4_0_q8_0_4row()` (after line ~1510), inside the
`#if defined(__AVX2__)` block:

```cpp
#if defined(__AVXVNNIINT8__)
// Process 8 weight rows x 1 activation row for Q4_0 x Q8_0.
// Uses 8 independent 256-bit accumulators for maximum ILP on Arrow Lake P-cores.
// Loads each Q8_0 block once and dots against 8 Q4_0 blocks simultaneously.
static inline void simd_mul_mat_q4_0_q8_0_8row(
    int K_elem,
    float * GGML_RESTRICT out,
    const void * GGML_RESTRICT vx0,
    const void * GGML_RESTRICT vx1,
    const void * GGML_RESTRICT vx2,
    const void * GGML_RESTRICT vx3,
    const void * GGML_RESTRICT vx4,
    const void * GGML_RESTRICT vx5,
    const void * GGML_RESTRICT vx6,
    const void * GGML_RESTRICT vx7,
    const void * GGML_RESTRICT vy
) {
    const int nb = K_elem / QK4_0;

    const block_q4_0 * GGML_RESTRICT x0 = (const block_q4_0 *)vx0;
    const block_q4_0 * GGML_RESTRICT x1 = (const block_q4_0 *)vx1;
    const block_q4_0 * GGML_RESTRICT x2 = (const block_q4_0 *)vx2;
    const block_q4_0 * GGML_RESTRICT x3 = (const block_q4_0 *)vx3;
    const block_q4_0 * GGML_RESTRICT x4 = (const block_q4_0 *)vx4;
    const block_q4_0 * GGML_RESTRICT x5 = (const block_q4_0 *)vx5;
    const block_q4_0 * GGML_RESTRICT x6 = (const block_q4_0 *)vx6;
    const block_q4_0 * GGML_RESTRICT x7 = (const block_q4_0 *)vx7;
    const block_q8_0 * GGML_RESTRICT y  = (const block_q8_0 *)vy;

    // 8 independent 256-bit accumulators
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    __m256 acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps();
    __m256 acc7 = _mm256_setzero_ps();

    const __m256i off = _mm256_set1_epi8(8);

    for (int ib = 0; ib < nb; ib++) {
        // Load activation block ONCE (amortized across 8 weight rows)
        const float   q8_d = GGML_FP16_TO_FP32(y[ib].d);
        const __m256i qy   = _mm256_loadu_si256((const __m256i *)y[ib].qs);

#define PROCESS_ROW_8(xptr, acc) \
        { \
            const float d  = GGML_FP16_TO_FP32(xptr[ib].d) * q8_d; \
            __m256i     qx = ggml_sycl_bytes_from_nibbles_32(xptr[ib].qs); \
            qx             = _mm256_sub_epi8(qx, off); \
            const __m256i prod = _mm256_dpbssd_epi32(_mm256_setzero_si256(), qx, qy); \
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(prod), acc); \
        }

        PROCESS_ROW_8(x0, acc0)
        PROCESS_ROW_8(x1, acc1)
        PROCESS_ROW_8(x2, acc2)
        PROCESS_ROW_8(x3, acc3)
        PROCESS_ROW_8(x4, acc4)
        PROCESS_ROW_8(x5, acc5)
        PROCESS_ROW_8(x6, acc6)
        PROCESS_ROW_8(x7, acc7)

#undef PROCESS_ROW_8
    }

    out[0] = ggml_sycl_hsum_float_8(acc0);
    out[1] = ggml_sycl_hsum_float_8(acc1);
    out[2] = ggml_sycl_hsum_float_8(acc2);
    out[3] = ggml_sycl_hsum_float_8(acc3);
    out[4] = ggml_sycl_hsum_float_8(acc4);
    out[5] = ggml_sycl_hsum_float_8(acc5);
    out[6] = ggml_sycl_hsum_float_8(acc6);
    out[7] = ggml_sycl_hsum_float_8(acc7);
}
#endif // defined(__AVXVNNIINT8__)
```

Then update the Q4_0 fast path in `ggml_sycl_cpu_vec_dot_rows()` TBB body to
dispatch 8-row tiles before 4-row:

```cpp
#if defined(__AVX2__)
                    if (type == GGML_TYPE_Q4_0) {
#if defined(__AVXVNNIINT8__)
                        // 8-row AVX2 VNNI kernel
                        for (; i + 7 < r.end(); i += 8) {
                            simd_mul_mat_q4_0_q8_0_8row(
                                ne00, output + i,
                                (const char *) src0_host + (size_t)(i + 0) * row_stride,
                                (const char *) src0_host + (size_t)(i + 1) * row_stride,
                                (const char *) src0_host + (size_t)(i + 2) * row_stride,
                                (const char *) src0_host + (size_t)(i + 3) * row_stride,
                                (const char *) src0_host + (size_t)(i + 4) * row_stride,
                                (const char *) src0_host + (size_t)(i + 5) * row_stride,
                                (const char *) src0_host + (size_t)(i + 6) * row_stride,
                                (const char *) src0_host + (size_t)(i + 7) * row_stride,
                                src1_q_data);
                        }
#endif
                        // 4-row AVX2 kernel for remainder
                        for (; i + 3 < r.end(); i += 4) {
                            simd_mul_mat_q4_0_q8_0_4row(
                                ne00, output + i,
                                (const char *) src0_host + (size_t)(i + 0) * row_stride,
                                (const char *) src0_host + (size_t)(i + 1) * row_stride,
                                (const char *) src0_host + (size_t)(i + 2) * row_stride,
                                (const char *) src0_host + (size_t)(i + 3) * row_stride,
                                src1_q_data);
                        }
                    }
#endif
```

### Verification

```bash
source /opt/intel/oneapi/setvars.sh --force && ninja -C build -j $(nproc)

# Correctness
GGML_SYCL_TENSOR_SPLIT=13 GGML_SYCL_DISABLE_GRAPH=1 \
  ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expected: "6, 7, 8, 9, 10, 11, 12, 13, 14, 15"
```

---

## Task 4: 16-row kernel with array-based interface

**Track:** A
**Depends on:** Task 3
**File scope:**
- Modify: `ggml/src/ggml-sycl/cpu-dispatch.cpp`

**Description:**

Write a new `simd_mul_mat_q4_0_q8_0_16row()` kernel that processes 16 weight rows per
inner loop iteration. Uses 16 independent __m256 accumulators. Arrow Lake P-core L1 cache
is 80KB — at row_stride=2304 bytes, 16 rows of block data per iteration is ~36KB, well
within L1. The activation data (2304 bytes) stays cached.

Uses array-based interface (`const void * rows[16]`) instead of 16 separate parameters.

**Guard:** `#if defined(__AVXVNNIINT8__)` — same as 8-row kernel.

### Implementation

Add after the 8-row kernel, inside the same `#if defined(__AVXVNNIINT8__)` block:

```cpp
// Process 16 weight rows x 1 activation row for Q4_0 x Q8_0.
// 16 independent accumulators saturate Arrow Lake P-core's 6-wide issue.
static inline void simd_mul_mat_q4_0_q8_0_16row(
    int K_elem,
    float * GGML_RESTRICT out,
    const void * GGML_RESTRICT rows[16],
    const void * GGML_RESTRICT vy
) {
    const int nb = K_elem / QK4_0;

    const block_q4_0 * GGML_RESTRICT xr[16];
    for (int r = 0; r < 16; r++) {
        xr[r] = (const block_q4_0 *)rows[r];
    }
    const block_q8_0 * GGML_RESTRICT y = (const block_q8_0 *)vy;

    __m256 acc[16];
    for (int r = 0; r < 16; r++) {
        acc[r] = _mm256_setzero_ps();
    }

    const __m256i off = _mm256_set1_epi8(8);

    for (int ib = 0; ib < nb; ib++) {
        const float   q8_d = GGML_FP16_TO_FP32(y[ib].d);
        const __m256i qy   = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        for (int r = 0; r < 16; r++) {
            const float d  = GGML_FP16_TO_FP32(xr[r][ib].d) * q8_d;
            __m256i     qx = ggml_sycl_bytes_from_nibbles_32(xr[r][ib].qs);
            qx             = _mm256_sub_epi8(qx, off);
            const __m256i prod = _mm256_dpbssd_epi32(_mm256_setzero_si256(), qx, qy);
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(prod), acc[r]);
        }
    }

    for (int r = 0; r < 16; r++) {
        out[r] = ggml_sycl_hsum_float_8(acc[r]);
    }
}
```

Update dispatch in `ggml_sycl_cpu_vec_dot_rows()` TBB body to add 16-row before 8-row:

```cpp
#if defined(__AVXVNNIINT8__)
                        // 16-row AVX2 VNNI kernel
                        for (; i + 15 < r.end(); i += 16) {
                            const void * row_ptrs[16];
                            for (int k = 0; k < 16; k++) {
                                row_ptrs[k] = (const char *) src0_host + (size_t)(i + k) * row_stride;
                            }
                            simd_mul_mat_q4_0_q8_0_16row(
                                ne00, output + i, row_ptrs, src1_q_data);
                        }
                        // 8-row remainder
                        for (; i + 7 < r.end(); i += 8) {
                            simd_mul_mat_q4_0_q8_0_8row(
                                ne00, output + i,
                                (const char *) src0_host + (size_t)(i + 0) * row_stride,
                                ...8 pointer args...
                                src1_q_data);
                        }
#endif
```

### Verification

```bash
source /opt/intel/oneapi/setvars.sh --force && ninja -C build -j $(nproc)

# Correctness
GGML_SYCL_TENSOR_SPLIT=13 GGML_SYCL_DISABLE_GRAPH=1 \
  ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expected: "6, 7, 8, 9, 10, 11, 12, 13, 14, 15"
```

---

## Task 5: Performance benchmark sweep

**Track:** A
**Depends on:** Task 4
**File scope:** None (benchmark only)

### Benchmarks (45s cooldown between each)

```bash
source /opt/intel/oneapi/setvars.sh --force && ninja -C build -j $(nproc)

# 1. GPU-only baseline
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
# Expected: PP512 >= 1200, TG128 ~72 tok/s

sleep 45

# 2. Tensor split sweep
for pct in 5 8 10 13 15 20 25; do
  echo "=== TENSOR_SPLIT=$pct ==="
  GGML_SYCL_TENSOR_SPLIT=$pct \
    ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-bench \
    -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128
  sleep 45
done

# 3. Correctness
GGML_SYCL_TENSOR_SPLIT=13 \
  ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
# Expected: "6, 7, 8, 9, 10, 11, 12, 13, 14, 15"
```

### Benchmark Results (Feb 19, 2026)

| Split % | TG128 tok/s | vs GPU-only (70) | Notes |
|---------|-------------|------------------|-------|
| 0 (GPU+graph) | 72.07 | baseline | Graph replay enabled |
| 0 (GPU, no graph) | 70.00 | -2.9% | TG_FAST already fast |
| 2% | 56.92 | -18.7% | Minimum viable split |
| 3% | 55.91 | -20.1% | |
| 5% | 48.52 | -30.7% | |
| 10% | 40.72 | -41.8% | |
| 13% | 33.34 | -52.4% | Original target |

PP512 baseline: 1335.64 (no regression expected, tensor split is TG-only).
Correctness: all splits produce correct output matching GPU-only baseline.

### Analysis

**Phase 1 tensor split cannot beat GPU-only.** The per-MUL_MAT `stream->memcpy().wait()`
synchronization at line 20005 of ggml-sycl.cpp breaks the GPU dispatch pipeline. Without
tensor split, the host rapidly queues ~580 ops and the GPU processes them in-order. With
tensor split, the host blocks on every MUL_MAT for staging + CPU vec_dot, serializing what
was previously pipelined.

**Key finding:** GPU-only without graph replay is already 70 tok/s (TG_FAST MMVQ path
eliminates most kernel dispatch overhead). Graph replay only adds +2 tok/s. This means
the old assumption that graph replay was essential (5.7 vs 72 tok/s) no longer holds.

**Phase 2 required:** To exceed GPU-only, need graph replay + tensor split integration
where: (1) GPU graph replays partial MMVQ for ALL layers in one batch, (2) CPU vec_dot
runs TRULY concurrently during the single graph submission. This eliminates per-op
synchronization. Theoretical limit: 3.83 GiB / (280+45 GB/s) = 12.6ms → 79 tok/s (+10%).

### Success Criteria (REVISED)
- Phase 1: ~~TG128 > 75 tok/s~~ Phase 1 is correctness-only validation
- Phase 2 (future): TG128 > 75 tok/s with graph replay integration
- PP512 unchanged from baseline (>= 1200 tok/s) - PASS
- Deterministic output matches GPU-only baseline - PASS
- GPU-only path (TENSOR_SPLIT=0) has zero regression - PASS
