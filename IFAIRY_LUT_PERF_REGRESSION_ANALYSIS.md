# iFairy ARM LUT Performance Regression Analysis

## Summary

This document analyzes the performance regression observed in the iFairy LUT implementation between commit `0ec52a5a` (2025-12-17, ~16.99 tok/s) and `34d8df05` (~5-7 tok/s). The analysis identifies specific code changes that likely contributed to the ~60-70% performance drop and provides recommendations for recovery.

## Performance Timeline (from IFAIRY_ARM_3W_LUT_STATUS.md)

| Commit | Date | Layout | tok/s | Notes |
|--------|------|--------|-------|-------|
| `0ec52a5a` | 2025-12-17T13:17 | legacy | **15.39** | Baseline (high performance) |
| `0ec52a5a` | 2025-12-17T13:17 | compact | **16.99** | Peak performance |
| `0aeaa6c9` | 2025-12-17T14:20 | legacy | 2.71 | **First major regression** |
| `0aeaa6c9` | 2025-12-17T14:20 | compact | 4.75 | |
| `a785693e+dirty` | 2025-12-17T17:50 | legacy | 8.21 | Partial recovery |
| `34d8df05` | 2025-12-18T04:04 | legacy | 5.01 | Current state (noisy) |
| `34d8df05` | 2025-12-18T04:04 | compact | 4.98 | Current state (noisy) |

**Key Observation**: Performance dropped from ~17 tok/s to ~3-5 tok/s after commit `0aeaa6c9` ("减少重复 preprocess 与同步开销").

---

## Root Cause Analysis

### 1. `ggml_ifairy_lut_preprocess_ex` Changes (HIGH IMPACT)

**File**: `ggml/src/ggml-ifairy-lut.cpp`

#### Original Code (0ec52a5a):
```c
memset(tbl0, 0, k_ifairy_lut_pos_bytes);
memset(tbl1, 0, k_ifairy_lut_pos_bytes);
memset(tbl2, 0, k_ifairy_lut_pos_bytes);

// position 0
tbl0[0 * 4 + 0] = (int8_t) -xr0;
tbl0[0 * 4 + 1] = (int8_t) -xi0;
tbl0[1 * 4 + 0] = (int8_t)  xr0;
tbl0[1 * 4 + 1] = (int8_t)  xi0;
// ... direct byte assignments
```

#### Current Code (`34d8df05`):
```c
const uint8_t xr0_p = (uint8_t) xr0;
const uint8_t xi0_p = (uint8_t) xi0;
// ... 12 cast operations
const uint8_t xr0_n = (uint8_t) (int8_t) -xr0;
// ... more casts

const uint64_t tbl0_lo = ggml_ifairy_pack_u8_8(xr0_n, xi0_n, 0, 0, xr0_p, xi0_p, 0, 0);
const uint64_t tbl0_hi = ggml_ifairy_pack_u8_8(0, 0, xr0_n, xi0_n, 0, 0, xr0_p, xi0_p);
// ... 6 function calls

#if defined(__ARM_NEON) && defined(__aarch64__)
const uint8x16_t v0 = vcombine_u8(vcreate_u8(tbl0_lo), vcreate_u8(tbl0_hi));
vst1q_u8((uint8_t *) tbl0, v0);
// ... NEON vector operations
#endif
```

**Problems**:
1. **Excessive intermediate variables**: 12 `uint8_t` temporaries + 6 `uint64_t` pack results = 18 intermediate values vs. direct stores
2. **Function call overhead**: `ggml_ifairy_pack_u8_8()` is called 6 times per group (even if inlined, the bit manipulation is more complex)
3. **NEON vector creation overhead**: `vcreate_u8()` + `vcombine_u8()` add latency before the store
4. **Original was already optimal**: Direct byte stores with `memset` initialization is cache-friendly and compiler-optimizable

**Estimated Impact**: HIGH (30-40% of regression)

**Validation (worktree)**: Switching `compact` preprocess back to a simple “memset + direct byte stores” implementation immediately recovered to ~`18-19 tok/s` on Apple M4 (see latest entries in `IFAIRY_ARM_3W_LUT_STATUS.md`). This supports the hypothesis that the pack/NEON-setup-heavy rewrite was a primary regression source.

---

### 2. Loop Unroll Strategy Changes (MEDIUM IMPACT)

**File**: `ggml/src/ggml-ifairy-lut.cpp` (`ggml_ifairy_lut_qgemm_ex_legacy`)

#### Original Code (0ec52a5a):
```c
for (; gi + 1 < groups_per_block; gi += 2) {
    // 2-way unroll with isum0, isum1
    __builtin_prefetch(grp0 + 2 * k_ifairy_lut_group_bytes, 0, 1);
    // ...
}
```

#### Current Code (`34d8df05`):
```c
for (; gi + 3 < groups_per_block; gi += 4) {
    // 4-way unroll
    if (prefetch) {
        __builtin_prefetch(grp0 + 4 * k_ifairy_lut_group_bytes, 0, 1);
    }
    // ...
}
```

**Problems**:
1. **Increased register pressure**: 4-way unroll requires more registers for pat0-pat3, c00-c32, grp0-grp3, t00-t32, p00-p32, s160-s163
2. **Conditional prefetch overhead**: `if (prefetch)` branch adds pipeline stall risk in hot loop
3. **Instruction cache pressure**: 4-way unroll generates ~2x more code per iteration
4. **Prefetch distance mismatch**: Prefetching 4 groups ahead instead of 2 may not match cache line timing

**Estimated Impact**: MEDIUM (15-25% of regression)

---

### 3. N==1 Fast-Path Code Quality (MEDIUM IMPACT)

**File**: `ggml/src/ggml-ifairy-lut.cpp`

#### Original Code (0ec52a5a):
```c
#if 0  // <-- DISABLED!
// Fast-path for decode: N == 1 avoids the col loop
if (n == 1) {
    // ...
}
#endif
```

#### Current Code (`34d8df05`):
```c
// Fast-path for decode: N == 1 avoids the col loop
if (n == 1 && !strict) {  // <-- ENABLED
    // ... 142 lines of new fast-path code
}
```

**Problems**:
1. **Fast-path is now enabled but slower**: The original code disabled the fast-path (`#if 0`), meaning the generic loop was used
2. **New fast-path has redundant code**: Duplicates much of the generic loop structure with minor modifications
3. **Different accumulator strategy**: The new fast-path initializes `float32x4_t accv` differently than the generic path
4. **Additional index arithmetic**: `idx_blk[gi + 0..3]` vs `idx_g[0..3]` with pointer advancement

**Estimated Impact**: MEDIUM (10-20% of regression for decode scenarios)

---

### 4. Activation Quantization Parallelization (MEDIUM IMPACT)

**File**: `ggml/src/ggml-cpu/ggml-cpu.c`

#### Original Code (0ec52a5a):
```c
// quantize activations once (thread 0) if needed
if (src1->type == GGML_TYPE_F32) {
    if (ith == 0) {
        const float * act_f32 = (const float *) src1->data;
        for (int64_t c = 0; c < N; ++c) {
            quantize_row_ifairy_q16(act_f32 + c * (nb11 / sizeof(float)), act_q + c * blocks_per_col, K);
        }
    }
    ggml_barrier(params->threadpool);
}
```

#### Current Code (`34d8df05`):
```c
if (N >= nth) {
    // Shard by columns.
    for (int64_t c = ith; c < N; c += nth) {
        quantize_row_ifairy_q16(act_f32 + c * act_f32_col_stride, act_q + c * blocks_per_col, K);
    }
} else {
    // Decode-like: N is small (often 1). Shard each column by K-block ranges.
    const int64_t ib0 = (blocks_per_col * ith) / nth;
    const int64_t ib1 = (blocks_per_col * (ith + 1)) / nth;
    if (ib1 > ib0) {
        const int64_t k_part = (ib1 - ib0) * QK_K;
        for (int64_t c = 0; c < N; ++c) {
            quantize_row_ifairy_q16(x, y, k_part);
        }
    }
}
ggml_barrier(params->threadpool);
```

**Problems**:
1. **K-block sharding overhead**: For decode (N=1), each thread processes a K-block range, introducing:
   - Division operations per thread: `(blocks_per_col * ith) / nth`
   - More complex memory access patterns
   - Potential cache line conflicts at K-block boundaries
2. **Multiple quantize calls**: Instead of one call with full K, multiple threads call with partial K
3. **Barrier still required**: No reduction in synchronization overhead

**Estimated Impact**: MEDIUM (10-15% of regression)

---

### 5. Overflow Check Assertions (LOW-MEDIUM IMPACT)

**Files**: `ggml/src/ggml-ifairy-lut.cpp`, `ggml/src/ggml-cpu/ggml-cpu.c`

#### Added Code (`34d8df05`):
```c
// In ggml_ifairy_lut_get_wsize():
GGML_ASSERT(a == 0 || b <= SIZE_MAX / a);  // per multiplication
// ... 20+ new GGML_ASSERT calls

// In ggml-cpu.c:
GGML_ASSERT(M >= 0);
GGML_ASSERT(K >= 0);
GGML_ASSERT(N >= 0);
GGML_ASSERT(blocks_per_col >= 0);
GGML_ASSERT(blocks_tile >= 0);
GGML_ASSERT(groups_tile >= 0);
// ... many more
```

**Problems**:
1. **Hot path assertions**: `ggml_ifairy_lut_get_wsize()` is called per mul_mat operation
2. **Division in checks**: `b <= SIZE_MAX / a` includes division operations
3. **Release builds**: `GGML_ASSERT` may still evaluate its condition depending on build configuration

**Estimated Impact**: LOW (5-10% of regression)

---

### 6. Environment Variable Parsing (LOW IMPACT)

**Files**: `ggml/src/ggml-ifairy-lut.h`, `ggml/src/ggml-ifairy-lut.cpp`, `ggml/src/ggml-cpu/ggml-cpu.c`

#### Added Code (`34d8df05`):
```c
// New helper functions called repeatedly:
static inline bool ggml_ifairy_env_enabled(const char * name);
static inline int ggml_ifairy_env_get_int_nonzero(const char * name, int def);

// Called in hot paths:
const bool prefetch = ggml_ifairy_lut_prefetch_enabled();  // getenv() call
const bool dbg = ggml_ifairy_env_enabled("GGML_IFAIRY_LUT_DEBUG");  // getenv() call
```

**Problems**:
1. **Repeated getenv() calls**: Each call involves system call overhead
2. **Not cached**: Environment variables are re-read on every operation
3. **Multiple variables**: PREFETCH, DEBUG, STRICT, LAYOUT, BK_BLOCKS, BM, FULLACC all checked

**Estimated Impact**: LOW (3-5% of regression)

---

## Recommendations

### Priority 1: Revert Preprocess Changes (HIGH IMPACT)

Revert `ggml_ifairy_lut_preprocess_ex` to use direct byte assignment:

```c
// Instead of pack_u8_8 + NEON store, use:
memset(tbl0, 0, k_ifairy_lut_pos_bytes);
memset(tbl1, 0, k_ifairy_lut_pos_bytes);
memset(tbl2, 0, k_ifairy_lut_pos_bytes);

tbl0[0 * 4 + 0] = (int8_t) -xr0;
tbl0[0 * 4 + 1] = (int8_t) -xi0;
tbl0[1 * 4 + 0] = (int8_t)  xr0;
tbl0[1 * 4 + 1] = (int8_t)  xi0;
// ... direct assignments
```

**Expected Recovery**: 30-40% of lost performance

### Priority 2: Revert Loop Unroll to 2-Way

Change the inner loop back to 2-way unroll:

```c
// Change from:
for (; gi + 3 < groups_per_block; gi += 4) { ... }

// Back to:
for (; gi + 1 < groups_per_block; gi += 2) { ... }
```

And remove the conditional prefetch:

```c
// Change from:
if (prefetch) {
    __builtin_prefetch(grp0 + 4 * k_ifairy_lut_group_bytes, 0, 1);
}

// Back to:
__builtin_prefetch(grp0 + 2 * k_ifairy_lut_group_bytes, 0, 1);
```

**Expected Recovery**: 15-25% of lost performance

### Priority 3: Disable N==1 Fast-Path

Wrap the N==1 fast-path with `#if 0` again until it's properly optimized:

```c
#if 0  // TODO: optimize before re-enabling
// Fast-path for decode: N == 1
if (n == 1 && !strict) {
    // ...
}
#endif
```

**Expected Recovery**: 10-20% of lost performance (decode scenarios)

### Priority 4: Simplify Activation Quantization

Revert to thread 0 only quantization for now:

```c
if (src1->type == GGML_TYPE_F32) {
    if (ith == 0) {
        const float * act_f32 = (const float *) src1->data;
        for (int64_t c = 0; c < N; ++c) {
            quantize_row_ifairy_q16(act_f32 + c * (nb11 / sizeof(float)),
                                    act_q + c * blocks_per_col, K);
        }
    }
    ggml_barrier(params->threadpool);
}
```

**Expected Recovery**: 10-15% of lost performance

### Priority 5: Cache Environment Variables (if still measurable)

Recent work already centralizes env parsing helpers; if profiling still shows measurable getenv/parse overhead on your target machine, add static caching for the truly hot toggles (e.g. `GGML_IFAIRY_LUT_PREFETCH`, layout) and measure again.

Example (pattern only):

```c
static bool g_prefetch_checked = false;
static bool g_prefetch_enabled = true;

static inline bool ggml_ifairy_lut_prefetch_enabled(void) {
    if (!g_prefetch_checked) {
        const char * env = getenv("GGML_IFAIRY_LUT_PREFETCH");
        g_prefetch_enabled = !(env && strcmp(env, "0") == 0);
        g_prefetch_checked = true;
    }
    return g_prefetch_enabled;
}
```

**Expected Recovery**: 0-5% (likely minor; validate with profile)

### Priority 6: Move Overflow Checks to Debug Only

Wrap overflow checks in debug conditionals:

```c
#ifdef GGML_DEBUG
GGML_ASSERT(a == 0 || b <= SIZE_MAX / a);
#endif
```

Or move them to a separate validation function called only once per session.

**Expected Recovery**: 0-10% (depends on whether checks are evaluated in Release)

---

## Verification Plan

After implementing fixes (each step in isolation; do not batch changes):

1. **Rebuild**: `cmake --build build-rel --config Release -j $(nproc)`

2. **Sanity Check**:
```bash
./build-rel/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf \
    --gpu-layers 0 -t 4 -b 1 --seed 1 -p "I believe life is" -n 16 -no-cnv
```

3. **Benchmark**:
```bash
GGML_IFAIRY_LUT=1 GGML_IFAIRY_LUT_BK_BLOCKS=0 GGML_IFAIRY_LUT_BM=0 \
GGML_IFAIRY_LUT_FULLACC=0 GGML_IFAIRY_LUT_LAYOUT=legacy \
./build-rel/bin/llama-cli -m models/Fairy-plus-minus-i-700M/ifairy.gguf \
    --gpu-layers 0 -t 4 -b 1 -c 2048 --seed 1 -p "I believe life is" -n 256 -no-cnv
```

4. **Target**: Recover to **15+ tok/s** (legacy) or **17+ tok/s** (compact)

For the canonical commands and the tok/s table format, follow `IFAIRY_ARM_3W_LUT_STATUS.md`.

---

## Conclusion

The performance regression is primarily caused by:

1. **Preprocess function complexity** (30-40%): Over-engineering with pack functions and NEON stores
2. **Loop unroll over-optimization** (15-25%): 4-way unroll with increased register pressure
3. **N==1 fast-path quality** (10-20%): Newly enabled but unoptimized code path
4. **Quantization parallelization overhead** (10-15%): Complex K-block sharding
5. **Safety checks overhead** (8-15%): Overflow assertions + env parsing

Total estimated regression sources account for ~70-115% (overlapping effects).

**Recommended Action**: Apply fixes in priority order, measuring performance after each change. Target is to recover to `0ec52a5a` baseline (~17 tok/s compact, ~15 tok/s legacy).

---

## Why This Happened (and How to Avoid It)

### Why performance dropped

- **Hot path got more complex, not simpler**: replacing straightforward byte stores with “packing + vector construction + stores” increased instruction count, register pressure, and dependency chains in a function that runs for every `(col, group)` during decode.
- **Optimization direction mismatch**: optimizing for “fewer stores” can backfire when the platform/compiler already turns `memset + a few stores` into efficient store-pairs, while manual packing introduces extra ops.
- **Regression masked by noise**: tok/s is noisy on desktop systems (thermals/background load); without a strict A/B workflow, it is easy to misattribute changes.

### How to prevent similar regressions

- **Treat preprocess/qgemm as perf-critical APIs**: any change in these should be isolated (one knob at a time), benchmarked, and reverted quickly if it does not win.
- **Prefer simple codegen in hot loops**: fewer temporaries, fewer helpers, avoid “clever” packing unless assembly inspection shows a win.
- **Always keep a stable benchmark contract**: fixed command/seed/ctx/threads, record tok/s in `IFAIRY_ARM_3W_LUT_STATUS.md`, and rerun at least twice if the delta is within noise.
- **Keep a “perf-safe mode”**: when introducing optional fast-paths (e.g. `N==1`), gate them behind an env/compile flag until proven stable.

### Practical reproducibility rules (what we actually do)

- **3-run rule**: for any performance claim, run the exact baseline command 3 times back-to-back for both `legacy` and `compact`, then record `min/max/mean` (not just the best run).
- **Thermal/noise awareness**: if the 3 runs show a large monotonic drop (typical on laptops/Apple Silicon), cool down and rerun; otherwise “A/B” conclusions are unreliable.
- **A/B without code churn**: prefer env-gated switches for risky fast-paths (e.g. `GGML_IFAIRY_LUT_N1_FASTPATH=0/1`), so we can do ABABAB runs and revert immediately if it does not win.
