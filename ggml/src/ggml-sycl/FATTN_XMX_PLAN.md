# SYCL Flash Attention XMX Acceleration Plan

## Overview

This document outlines the plan to accelerate the SYCL Flash Attention kernel using Intel XMX (Xe Matrix eXtensions) hardware matrix units available on Intel Arc and Data Center GPUs.

---

## Current Status Summary

### Phase 1: Basic XMX Implementation ✅ COMPLETE
- Created `fattn-xmx-f16.hpp` with XMX-accelerated Q·K^T
- 16 threads (single sub-group) - **7% slower than scalar**

### Phase 2: Thread Count Optimization ✅ COMPLETE
- Increased to 128 threads (8 sub-groups)
- All threads participate in memory loads
- **Now matches scalar performance**

### Phase 3: XMX Correctness Debugging ✅ FIXED (2024-11-27)
- **Root cause found**: `joint_matrix_load` requires stride divisible by 8!
- Old code used `XMX_PAD = 1` giving `KT_STRIDE = 65` (not divisible by 8) → wrong results
- Fix: Changed `XMX_PAD = 8` giving `KT_STRIDE = 72` (divisible by 8) → correct results
- Created test programs `test_joint_matrix_local.cpp` and `test_joint_matrix_alignment.cpp` to isolate bug

### Current Performance (2024-11-27)

#### gpt-oss-20b (D=64, GQA=8, sinks) - Short Prompt (16 tokens)
| Kernel | Prompt Eval | Generation | Status |
|--------|-------------|------------|--------|
| MMA F16 (scalar) | ~29.7 t/s | ~13.4 t/s | ✅ Working |
| XMX F16 (no double buf) | ~30.2 t/s | ~13.5 t/s | ✅ Working |
| XMX F16 (double buf) | ~30.5 t/s | ~13.6 t/s | ✅ Working |

#### gpt-oss-20b (D=64, GQA=8, sinks) - Medium Prompt (170 tokens)
| Kernel | Prompt Eval | Generation | Status |
|--------|-------------|------------|--------|
| MMA F16 (scalar) | ~49.6 t/s | ~13.3 t/s | ✅ Working |
| XMX F16 (double buf) | ~49.1 t/s | ~13.2 t/s | ✅ Working |

**Analysis**: XMX shows small gains on short prompts but is roughly equivalent on medium prompts.
The workload appears to be compute-bound rather than memory-bound at these context lengths.

### Key Finding - joint_matrix Stride Alignment
`joint_matrix_load` has an undocumented requirement: **stride must be divisible by 8**.
- Stride 16, 24, 32: ✅ Work
- Stride 17, 18, 19, 20, etc.: ❌ Produce incorrect results
This applies to both local and global memory.

---

## Resolved Issue: joint_matrix Stride Alignment ✅

### Root Cause (FOUND 2024-11-27)
`joint_matrix_load` requires the stride parameter to be **divisible by 8**.
This is NOT documented in Intel's SYCL joint_matrix documentation.

### Debugging Process
1. Created `test_joint_matrix_local.cpp` - tested joint_matrix with local memory
2. Found Test 6 (with PAD=1) failed while Tests 1-5 passed
3. Created `test_joint_matrix_alignment.cpp` - systematic stride testing
4. Results showed: stride 16, 24, 32 pass; stride 17-23, 25-31 fail

### The Fix
```cpp
// OLD (broken): XMX_PAD = 1 → KT_STRIDE = 64 + 1 = 65 (NOT divisible by 8)
constexpr int XMX_PAD = 1;

// NEW (working): XMX_PAD = 8 → KT_STRIDE = 64 + 8 = 72 (divisible by 8)
constexpr int XMX_PAD = 8;
```

### Test Programs Created
- `test_joint_matrix_local.cpp` - Tests joint_matrix with local_accessor
- `test_joint_matrix_stride.cpp` - Isolates stride alignment issue
- `test_joint_matrix_alignment.cpp` - Systematic stride testing (16-32)

---

## Optimization Roadmap (Once XMX Works)

### Priority 1: Memory Bandwidth Optimizations
These target the actual bottleneck.

#### 1.1 Double Buffering for K/V Loads
- **Status**: ✅ Implemented (2024-11-27)
- **Difficulty**: Medium
- **Actual Gain**: ~1-2% on short prompts, marginal on long contexts
- **Description**: Load next K batch into alternate buffer while computing with current
- **Note**: Added second KT buffer (2x KT_SIZE shared memory). The gain is smaller than
  expected because the kernel is compute-bound rather than memory-bound on this model.
  Benefits may be more visible on longer contexts (>1K tokens) or larger head dimensions.

#### 1.2 Coalesced Memory Access Patterns
- **Status**: 🔲 Not Started
- **Difficulty**: Medium
- **Expected Gain**: 5-15%
- **Description**: Ensure memory accesses are coalesced across threads

#### 1.3 Reduce Shared Memory Bank Conflicts
- **Status**: ✅ Implemented (XMX_PAD = 8, must be divisible by 8 for joint_matrix!)
- **Difficulty**: Medium
- **Expected Gain**: 5-10%
- **Description**: Pad shared memory arrays to avoid bank conflicts

#### 1.4 Prefetch to L1/L2 Cache
- **Status**: 🔲 Not Started (SYCL prefetch not portable)
- **Difficulty**: Low
- **Expected Gain**: 5-10%

### Priority 2: Compute Optimizations

#### 2.1 XMX for Softmax·V Accumulation
- **Status**: ✅ IMPLEMENTED (2024-11-27)
- **Difficulty**: High
- **Actual Gain**: ~4% on long prompts (146 tokens: 49.8 → 51.95 t/s)
- **Note**: Implemented XMX-based S@V computation. Key changes:
  - Added tile_S buffer for softmax weights in half precision
  - Added SV_acc buffer for XMX output
  - Used shared memory for batch_max to synchronize across threads
  - XMX computes S[ncols_padded, BATCH_KV] @ V[BATCH_KV, D] in tiles
  - Results accumulated into per-thread VKQ registers
  - Gain is modest because kernel remains compute-bound

#### 2.2 Fused Mask + Softcap Application
- **Status**: ✅ IMPLEMENTED (2024-11-28)
- **Difficulty**: Low
- **Actual Gain**: ~0% (model doesn't use softcap, mask application was not a bottleneck)
- **Note**: Vectorized mask+softcap using float4 loads/stores and half4 for mask values

#### 2.3 Vectorized Softmax Max/Sum Reduction
- **Status**: ✅ IMPLEMENTED (2024-11-28)
- **Difficulty**: Medium
- **Actual Gain**: ~1-3% (short: 42.70→43.09 t/s, long: 51.59→51.82 t/s)
- **Note**: Used float4 for max reduction, half8→float4 for sum reduction

### Priority 3: Algorithmic Optimizations

#### 3.1 Increase BATCH_KV for Long Contexts
- **Status**: 🔲 Not Started
- **Difficulty**: Low
- **Expected Gain**: Variable

#### 3.2 Adaptive ncols Based on Query Count
- **Status**: ✅ Already implemented in dispatch logic
- **Difficulty**: Low
- **Expected Gain**: 5-10% on prefill

#### 3.3 Skip Masked Positions Early
- **Status**: ✅ IMPLEMENTED (2024-11-27)
- **Difficulty**: High
- **Expected Gain**: Variable (depends on prompt length and context)
- **Note**: Successfully implemented causal skip during prefill. Key insights:
  - For causal attention, query at position P can attend to KV positions 0..P
  - During prefill (ne01 > ncols), `ic0 + ncols - 1` gives the last query's position
  - We limit `kv_loop_end = min(ne11, last_q_pos + BATCH_KV)` during prefill
  - During generation, the optimization is disabled since `ic0` doesn't map to
    absolute sequence position (KV cache is already populated)
  - Prefetch also respects the causal boundary to avoid loading unused data

### Priority 4: Architecture-Specific Tuning

#### 4.1 Tune for Intel Arc A770/A750
- **Status**: 🔲 Ready to start (XMX now working!)
- **Difficulty**: Medium
- **Expected Gain**: 5-15%

#### 4.2 Tune for Intel Data Center GPU Max
- **Status**: 🔲 Not Started
- **Difficulty**: Medium
- **Expected Gain**: Unknown

#### 4.3 Runtime Kernel Selection
- **Status**: 🔲 Not Started
- **Difficulty**: Low
- **Expected Gain**: Ensures best kernel for each GPU

---

## Implementation Order

### Sprint 0: Fix XMX Bug ✅ COMPLETE (2024-11-27)
1. [x] Identify bug: XMX produces wrong Q@K^T results
2. [x] Verify Q/K loading is correct
3. [x] Add scalar verification showing expected vs actual
4. [x] Fix ncols padding for XMX_TM alignment
5. [x] **Test joint_matrix with local memory isolation** → Created test_joint_matrix_local.cpp
6. [x] **Isolate stride alignment issue** → Found stride must be divisible by 8
7. [x] **Fix: Change XMX_PAD from 1 to 8** → KT_STRIDE now 72 instead of 65
8. [x] **Verify fix: XMX kernel produces correct output**

### Sprint 1: Quick Wins (XMX Now Working!)
1. [ ] **2.2 Fused mask** - Low effort, small gain
2. [ ] **2.3 Vectorized reductions** - Medium effort

### Sprint 2: Memory Optimizations ✅ COMPLETE (2024-11-27)
3. [x] **1.1 Double buffering** - Implemented, ~1-2% gain (compute-bound workload)
4. [ ] **1.2 Coalesced access** - Deferred (current access patterns adequate)

### Sprint 3: Compute Optimizations (XMX Working) ✅ COMPLETE (2024-11-27)
5. [x] **2.1 XMX for V accumulation** - ✅ Implemented, ~4% gain on long prompts
6. [x] **3.3 Skip masked positions** - ✅ Implemented for prefill

### Sprint 4: Tuning (XMX Working) ✅ COMPLETE (2024-11-28)
7. [x] **3.1 Tune BATCH_KV** - ✅ COMPLETE: BATCH_KV=32 is optimal (32 > 64 > 128)
8. [x] **4.1 Arc tuning** - ✅ COMPLETE: NTHREADS=128 is optimal (128 ≥ 256 > 64)
9. [x] **4.3 Runtime selection** - ✅ COMPLETE: XMX kernel auto-selected on Intel GPUs with matrix support

### Sprint 4.5: Continuous Batching Support ✅ COMPLETE (2024-11-28)
10. [x] **Sequence ID tensor allocation** - ✅ COMPLETE: Fixed graph traversal timing issue
    - **Problem**: seq_ids tensors not getting buffers allocated in unified KV mode
    - **Root cause**: `ggml_flash_attn_ext_set_seq_ids()` was called AFTER `ggml_build_forward_expand()`
    - **Solution**: Move seq_ids setting INTO `build_attn_mha()` before graph expansion
    - **Secondary fix**: Changed from I8→I32 to F32→I32 cast (CPU backend limitation)
    - **Status**: Server starts with `[SEQ_IDS] SUCCESS: Buffers on host, optimization enabled!`

#### Files Modified for Continuous Batching
- `src/llama-graph.h:707-718` - Added seq_ids parameters to `build_attn_mha()`
- `src/llama-graph.cpp:1350-1401` - Implementation with seq_ids before graph expand
- `src/llama-graph.cpp:1597-1607` - F32 tensor type for host, I32 cast for device
- `src/llama-kv-cache.cpp:1337-1420` - Write F32 sequence IDs from ubatch

#### Technical Details
The optimization enables cross-sequence masking in unified KV mode:
1. **Host tensors (F32)**: Written by CPU in `set_input_seq_ids()` with per-token sequence IDs
2. **Device tensors (I32)**: F32→I32 cast result passed to flash attention kernel
3. **Kernel behavior**: Can skip attention computation between tokens of different sequences
4. **Graceful fallback**: If buffers not available, kernel falls back to mask-based detection

### Performance Results (2024-11-28 - Intel Arc with tuned parameters)

#### gpt-oss-20b (D=64, GQA=8, sinks)
| Context Length | Prompt Eval (t/s) |
|----------------|-------------------|
| 47 tokens      | 42.76             |
| 120 tokens     | 51.59             |
| 211 tokens     | 54.95             |

#### Mistral-7B Q4_0 (D=128, GQA=4) - Flash Attention Benchmark
| Test           | Throughput (t/s)  |
|----------------|-------------------|
| pp512          | 341.98 ± 0.44     |
| tg128          | 37.32 ± 0.72      |

#### gpt-oss-20b Q8_0 (D=64, GQA=8, sinks) - Post-seq_ids Fix Verification
| Test           | Throughput (t/s)  |
|----------------|-------------------|
| pp128          | 53.01 ± 0.37      |
| tg64           | 13.22 ± 0.03      |

*All tests on Intel Arc Pro B50 Graphics with unified KV mode and seq_ids optimization enabled.*

#### Parameter Tuning Results

**BATCH_KV tuning (47 tokens):**
| BATCH_KV | Prompt Eval (t/s) | Generation (t/s) |
|----------|-------------------|------------------|
| 32       | 43.00             | 13.41            |
| 64       | 42.53             | 13.26            |
| 128      | 41.11             | 12.68            |

**NTHREADS tuning (120 tokens):**
| NTHREADS | Prompt Eval (t/s) | Generation (t/s) |
|----------|-------------------|------------------|
| 64       | 51.51             | 13.25            |
| 128      | 51.71             | 13.45            |
| 256      | 51.74             | 13.37            |

**Optimal configuration:**
- XMX_BATCH_KV = 32 (smaller batches = better cache utilization)
- XMX_NTHREADS = 128 (best balance of parallelism and overhead)

---

## Debugging Information

### Debug Output from XMX Kernel (2024-11-27)
```
[XMX-Q] head=0 tile_Q[0..3]: -0.0100 0.0036 -0.0626 0.1033
[XMX-KT] kv_start=0 tile_KT[d=0..3][k=0]: -0.5010 -4.4141 0.6665 -3.8574
[XMX-QK] BEFORE mask QK_acc[q=0][k=0..3]: -0.0037 -0.0444 0.7391 0.0555 (scalar dot=-1.0840)
```

### Debug Output from MMA F16 Kernel (Working)
```
[MMA-Q] head=0 tile_Q[0..3]: -0.0100 0.0036 -0.0626 0.1033
[MMA-K] kv_start=0 tile_K[k=0][d=0..3]: -0.5010 -4.4141 0.6665 -3.8574
[MMA-QK] AFTER mask KQ_shared[q=0][k=0..3]: -1.0840 -inf -inf -inf
```

### Key Observations
- Q values match between XMX and MMA: ✅
- K values match between XMX and MMA: ✅
- Scalar dot product in XMX matches MMA: ✅ (-1.0840)
- XMX joint_matrix result: ❌ (-0.0037, completely wrong)

---

## Benchmarking Protocol

For each optimization, measure:

1. **Correctness**: Output matches scalar kernel (within 1e-3)
2. **Prompt eval throughput**: tokens/second with 30+ token prompt
3. **Generation throughput**: tokens/second generating 20 tokens
4. **Memory usage**: Shared memory per work-group

Test models:
- gpt-oss-20b (D=64, GQA=8, sinks) - Complex case
- Mistral-7B (D=128, GQA=4) - Large head dim
- GPT-2 (D=64, GQA=1) - Simple case

Test contexts:
- Short: 32 tokens
- Medium: 256 tokens
- Long: 2048 tokens

---

## Technical Notes

### Intel XMX Specifications (Arc)
- Tile sizes: 8×16×16 (M×N×K)
- Sub-group size: 16
- Data types: FP16 inputs, FP32 accumulator
- Throughput: ~128 TOPS (theoretical)

### Memory Hierarchy (Arc A770)
- L1 cache: 192 KB per Xe-core
- L2 cache: 16 MB shared
- Memory bandwidth: 512 GB/s (GDDR6)
- Shared local memory: 64 KB per work-group

### Current Kernel Configuration (Tuned 2024-11-29 - Sprint 6)
```cpp
XMX_TM = 8           // Queries per XMX tile
XMX_TN = 16          // KV positions per XMX tile
XMX_TK = 16          // Reduction dimension
XMX_SG = 16          // Sub-group size
XMX_NTHREADS = 256   // Total threads (benchmarked: 256 > 128 > 512)
XMX_N_SG = 16        // Number of sub-groups (256/16)
XMX_BATCH_KV = 16    // KV positions per main loop (benchmarked: 16 > 32 > 64 > 128)
XMX_PAD = 0          // No padding needed (16 is divisible by 8)
```

### Sprint 6 Optimization Summary (2024-11-29)

**FA OFF vs FA ON Gap Reduction:**
| Phase | pp128 Gap | pp512 Gap | pp2048 Gap |
|-------|-----------|-----------|------------|
| Before Sprint 6 | -6% | -17% | -14% |
| After 6.1 (BATCH_KV=16) | -1% | -6% | -2% |
| After 6.2 (native::exp) | -1% | -6% | -2% |
| After 6.3 (NTHREADS=256) | -1% | -6% | -2% |
| After 6.4 (vec Q/V loads) | -0.9% | -5.0% | -0.6% |

**Current Benchmark (After Sprint 6.1-6.4):**
| Prompt | FA OFF (t/s) | FA ON (t/s) | Gap |
|--------|--------------|-------------|-----|
| pp128  | 236.63       | 234.42      | -0.9% |
| pp256  | 334.02       | 325.09      | -2.7% |
| pp512  | 405.17       | 384.79      | -5.0% |
| pp1024 | 397.90       | 383.86      | -3.5% |
| pp2048 | 385.65       | 383.23      | -0.6% |

**Key Achievement**: Reduced FA overhead from 10-17% to <1-5%

### Sprint 6 COMPLETE ✅ (2024-11-29)

All 6 phases completed. Final summary:

| Phase | Optimization | Result |
|-------|--------------|--------|
| 6.1 | BATCH_KV=16 | +13% FA throughput |
| 6.2 | native::exp, PAD=0 | +0.3% |
| 6.3 | NTHREADS=256 | +1% |
| 6.4 | Vectorized Q/V loads | +1.3% |
| 6.5 | oneDNN hybrid research | Not viable (documented) |
| 6.6 | Tile tuning | Hardware-fixed, config finalized |

**Total Improvement**: FA overhead reduced from **10-17%** to **0.1-6%**

**When to use Flash Attention:**
- ✅ Long contexts (2K+ tokens): Gap is <1%, memory savings worth it
- ✅ Memory-constrained: FA uses O(N) vs O(N²) memory
- ⚠️ Generation: 6% slower for D=128, 0.5% for D=64
- ⚠️ Short prompts (<256): 2-3% slower, may not be worth it

### Shared Memory Layout (with padding)
```cpp
ncols_padded = max(ncols, XMX_TM)  // Pad to XMX tile size
tile_Q:  [ncols_padded][D] half    // Query tile (padded for XMX)
tile_KT: [D][XMX_BATCH_KV + PAD] half // K transposed with padding
tile_V:  [XMX_BATCH_KV][D] half    // Value tile
QK_acc:  [ncols_padded][XMX_BATCH_KV] float // QK scores
```

### Files
- `fattn-xmx-f16.hpp` - XMX kernel implementation (currently broken)
- `fattn-mma-f16.hpp` - Working scalar MMA kernel (fallback)
- `fattn.cpp` - Kernel dispatch (currently uses MMA F16)
- `test_joint_matrix.cpp` - Standalone XMX test (passes with global memory)

---

## Sprint 5: Unified KV Mode SEGFAULT Debugging (IN PROGRESS - 2024-11-28)

### Problem Statement
Server crashes with SEGFAULT during warmup when using unified KV mode (`-kvu --parallel 4`) with flash attention.

**Symptom**:
```
common_init_from_params: warming up the model with an empty run
[SEQ_IDS] SUCCESS: Buffers on host, optimization enabled!
Segmentation fault (core dumped)
```

### Root Cause Analysis

The issue is that **input tensors have their `->data` pointing to HOST memory**, but the SYCL kernel tries to access this pointer from the GPU, causing a segfault.

**Data flow:**
1. `llama-graph.cpp:1600-1615` creates seq_ids tensors as I32, marks with `ggml_set_input()`
2. `llama-kv-cache.cpp:1594-1643` writes seq_ids to tensors via `tensor->data` (HOST buffer)
3. Scheduler copies data from host to device internally
4. `fattn.cpp:475-499` accesses `q_seq_ids->data` and `kv_seq_ids->data` directly
5. **CRASH**: Kernel tries to read host pointer from GPU

### What We've Already Tried

#### Attempt 1: Disable seq_ids debug output ❌
- Changed `#if 1` to `#if 0` on debug fprintf in fattn.cpp
- Result: Still crashed

#### Attempt 2: Disable seq_offsets optimization only ❌
- Commented out `params.n_seqs`, `params.seq_q_offsets`, `params.seq_kv_offsets`
- Kept `params.q_seq_ids` and `params.kv_seq_ids` from tensor->data
- Result: Still crashed (kernel still accessing host memory)

#### Attempt 3: Set seq_ids to nullptr (workaround) ❌
- User rejected: "instead of doing a work around... let's fix the root problem"

#### Attempt 4: Create I32 tensors directly instead of F32+ggml_cast ❌
- Changed llama-graph.cpp to create I32 tensors directly
- Changed llama-kv-cache.cpp to write I32 directly
- Result: Still crashed - `tensor->data` still points to host buffer

#### Attempt 5: Add H2D copy in fattn.cpp (CURRENT)
- Added code to detect host buffers and copy to device memory
- Use `sycl::malloc_device()` and `stream->memcpy()`
- Use thread-local device buffers to avoid per-call allocation
- **Status**: H2D copy debug message NOT appearing in output

### Current Debug Investigation

**Debug messages added:**
1. `[SEQ_IDS] SUCCESS` in llama-kv-cache.cpp:1596 - ✅ Appears
2. `[FATTN DEBUG] q_seq_ids=...` in fattn.cpp:483 - ❓ Not appearing
3. `[SEQ_IDS] H2D copy:...` in fattn.cpp:521 - ❓ Not appearing

**Hypothesis**: The seq_ids tensors might be:
- NULL when passed to fattn.cpp
- Have NULL buffer
- Have different type than expected

### Next Steps to Try

1. **Add debug before the if-check** to see tensor state
2. **Check if buffer is NULL** (default to host copy if so)
3. **Verify tensor type is GGML_TYPE_I32**
4. **Check if tensors are passed to flash attention operation at all**
5. **Temporarily disable seq_ids completely** to verify server works without it
6. **Add debug in kernel dispatch** to see if we reach the XMX kernel

### Files Modified for This Fix

| File | Changes |
|------|---------|
| `fattn.cpp:471-557` | H2D copy logic for seq_ids tensors |
| `llama-graph.cpp:1600-1615` | I32 tensor creation (from F32+cast) |
| `llama-kv-cache.cpp:1600-1643` | Write I32 directly (from F32) |
| `llama-graph.h:308-314` | Updated comments |

---

## Sprint 6: FA Performance Optimization (IN PROGRESS - 2024-11-29)

### Problem Statement

Flash Attention is **slower** than the non-FA path despite using XMX hardware acceleration.

**Current Benchmark Results (Mistral-7B Q4_0, Intel Arc Pro B50):**

| Prompt Length | FA OFF (t/s) | FA ON (t/s) | FA Speedup |
|--------------|--------------|-------------|------------|
| pp128        | 237.05       | 222.60      | 0.94x (6% slower) |
| pp256        | 334.23       | 300.45      | 0.90x (10% slower) |
| pp512        | 405.71       | 336.44      | 0.83x (17% slower) |
| pp1024       | 398.08       | 334.71      | 0.84x (16% slower) |
| pp2048       | 386.16       | 333.31      | 0.86x (14% slower) |

### Root Cause Analysis

The non-FA path uses **oneDNN** (Intel's highly optimized Deep Neural Network Library), while FA uses manual XMX operations:

| Component | Non-FA Path | FA Path |
|-----------|-------------|---------|
| Q×K matmul | oneDNN GEMM (optimized) | Manual XMX joint_matrix |
| Softmax | Dedicated optimized kernel | Inline computation |
| S×V matmul | oneDNN GEMM (optimized) | Manual XMX joint_matrix |
| Memory | Separate buffers, optimized | Shared memory tiles |

**Key insight**: oneDNN has years of optimization for Intel GPUs. Our manual XMX implementation can't match it without significant work.

### Optimization Strategy

#### Phase 6.1: Tune XMX_BATCH_KV ✅ COMPLETE (2024-11-29)
- **Status**: ✅ Complete
- **Finding**: Smaller BATCH_KV = better performance (opposite of initial hypothesis!)
- **Results** (Mistral-7B Q4_0, Arc B50):

| BATCH_KV | pp128 | pp256 | pp512 | pp1024 | pp2048 |
|----------|-------|-------|-------|--------|--------|
| 128      | 199   | 267   | 274   | 268    | 264    |
| 64       | 223   | 300   | 336   | 335    | 333    |
| 32       | 231   | 316   | 363   | 361    | 359    |
| **16**   | **234** | **323** | **379** | **378** | **377** |

- **Optimal**: BATCH_KV=16 (minimum allowed by XMX_TN=16 constraint)
- **Improvement**: +13% at pp512-pp2048 compared to BATCH_KV=64
- **Analysis**: Smaller tiles fit better in L1 cache, reducing memory latency
- **Gap vs non-FA**: Reduced from 10-17% to just 1-6%!

#### Phase 6.2: Optimize Softmax Computation ✅ COMPLETE (2024-11-29)
- **Status**: ✅ Complete
- **Changes**:
  1. Replaced `sycl::exp()` with `sycl::native::exp()` (4 locations)
  2. Removed XMX_PAD (16 is already divisible by 8)
- **Results**:

| Prompt | Before | After | Improvement |
|--------|--------|-------|-------------|
| pp128  | 233.54 | 233.00 | -0.2% (noise) |
| pp256  | 322.88 | 322.02 | -0.3% (noise) |
| pp512  | 379.17 | 380.23 | +0.3% |
| pp1024 | 377.86 | 378.97 | +0.3% |
| pp2048 | 376.86 | 378.13 | +0.3% |

- **Actual Gain**: ~0.3% (marginal, but reduces shared memory usage)
- **Note**: native::exp provides ~1% less precision but faster execution

#### Phase 6.3: Reduce Synchronization Overhead ✅ COMPLETE (2024-11-29)
- **Status**: ✅ Complete
- **Barrier Analysis Results**: All 10 group_barrier() calls are necessary:
  1. After Q load (line ~240) - Required: All threads read Q after
  2. After KT load (line ~340) - Required: XMX reads KT after
  3. After QK XMX compute (line ~450) - Required: All threads read QK_acc
  4. After KQ scale/mask (line ~550) - Required: Before reduction
  5. After softmax max/sum (line ~660) - Required: All threads need max/sum
  6. After V load (line ~700) - Required: XMX reads V after
  7. After SV XMX compute (line ~765) - Required: All threads read result
  8. Before double-buffer swap (line ~800) - Required: Sync before next iter
  9. After final reduction (line ~850) - Required: Before output write
  10. Before output write (line ~900) - Required: All threads done
- **Alternative Optimization**: Tuned XMX_NTHREADS instead

**NTHREADS Tuning Results** (pp128-pp2048 average):
| NTHREADS | Avg Throughput | vs 128 |
|----------|----------------|--------|
| 128      | 339 t/s        | baseline |
| 256      | 343 t/s        | +1.2% |
| 512      | 331 t/s        | -2.4% |

- **Optimal**: NTHREADS=256 gives ~1% improvement
- **Analysis**: 256 threads (16 sub-groups) better utilizes Arc B50 EU resources
  while 512 causes register pressure and scheduling overhead

#### Phase 6.4: Improve Memory Access Patterns ✅ COMPLETE (2024-11-29)
- **Status**: ✅ Complete
- **Changes**:
  1. Vectorized Q loading with half4 (4 elements per memory transaction)
  2. Vectorized V loading with half4 (4 elements per memory transaction)
  3. K loading kept as-is (transpose during load prevents easy vectorization)
- **Results**:

| Prompt | Before | After | FA Improvement | New Gap vs OFF |
|--------|--------|-------|----------------|----------------|
| pp128  | 233.68 | 234.42 | +0.3% | -0.9% |
| pp256  | 323.00 | 325.09 | +0.6% | -2.7% |
| pp512  | 380.03 | 384.79 | +1.3% | -5.0% |
| pp1024 | 379.40 | 383.86 | +1.2% | -3.5% |
| pp2048 | 378.80 | 383.23 | +1.2% | -0.6% |

- **Actual Gain**: 0.3-1.3% improvement in FA throughput
- **Note**: Larger gains at longer contexts where memory bandwidth matters more

#### Phase 6.5: Hybrid oneDNN + XMX Research ✅ COMPLETE (2024-11-29)
- **Status**: ✅ Complete (Not viable - documented findings)
- **Conclusion**: Hybrid approach is NOT viable for Flash Attention

**Why it won't work:**

1. **Architecture mismatch**: oneDNN dispatches host-level kernels, while FA
   requires fused in-kernel operations to keep data in shared memory

2. **Memory trade-off**: Using oneDNN would require:
   - Q×K result materialized to global memory (O(N²) storage)
   - Separate softmax kernel launch
   - S×V result materialized to global memory
   - This defeats FA's core memory advantage (O(N) vs O(N²))

3. **Prefetch limitations**: Intel's GPU Optimization Guide notes that
   `sycl::global_ptr::prefetch` has limited effectiveness on Arc GPUs
   due to lack of cache control

**Why oneDNN is faster for non-FA:**
- oneDNN GEMM is a single large operation with optimized tile scheduling
- FA does many small XMX operations with online softmax overhead
- The remaining 1-5% gap is acceptable given FA's memory advantage

**Recommendation**: Accept current performance. FA's value is at very long
contexts (4K+ tokens) where O(N) memory matters more than per-op speed.
At pp2048, FA is already within 0.6% of non-FA performance.

#### Phase 6.6: Tile Size Tuning for Arc B50 ✅ COMPLETE (2024-11-29)
- **Status**: ✅ Complete
- **Finding**: XMX tile sizes (TM=8, TN=16, TK=16) are hardware-fixed by Intel's
  joint_matrix specification. Tunable parameters already optimized in 6.1/6.3.

**Final Optimized Configuration:**
```cpp
XMX_TM = 8           // Hardware-fixed
XMX_TN = 16          // Hardware-fixed
XMX_TK = 16          // Hardware-fixed
XMX_NTHREADS = 256   // Tuned in Phase 6.3
XMX_BATCH_KV = 16    // Tuned in Phase 6.1
XMX_PAD = 0          // Optimized in Phase 6.2
```

**Comprehensive Benchmark Results (Final Configuration):**

| Model | Test | FA OFF (t/s) | FA ON (t/s) | Gap |
|-------|------|--------------|-------------|-----|
| Mistral 7B (D=128) | pp128 | 236.6 | 236.4 | -0.1% |
| Mistral 7B (D=128) | pp256 | 334.0 | 326.9 | -2.1% |
| Mistral 7B (D=128) | pp512 | 402.7 | 388.4 | -3.6% |
| Mistral 7B (D=128) | pp1024 | 397.9 | 387.1 | -2.7% |
| Mistral 7B (D=128) | pp2048 | 385.7 | 384.2 | -0.4% |
| Mistral 7B (D=128) | tg128 | 43.5 | 40.9 | -6.0% |
| GPT-OSS 20B (D=64) | pp128 | 54.1 | 53.0 | -2.1% |
| GPT-OSS 20B (D=64) | pp512 | 58.3 | 56.8 | -2.5% |
| GPT-OSS 20B (D=64) | tg64 | 13.7 | 13.6 | -0.5% |

**Key Observations:**
1. **Prompt processing**: FA gap is 0.1-3.6%, best at short/very long contexts
2. **Generation (tg)**: FA gap is 0.5-6%, worse for D=128 than D=64
3. **D=64 vs D=128**: Smaller head dimension has smaller FA overhead
4. **XMX underutilization**: Generation (ncols=1) only uses 1/8 of Q tile rows

### Implementation Order

1. **Phase 6.1** - Quick test of larger BATCH_KV (30 min)
2. **Phase 6.6** - Systematic tile tuning (2-4 hours with script)
3. **Phase 6.2** - Softmax optimization (2-3 hours)
4. **Phase 6.3** - Barrier reduction (1-2 hours)
5. **Phase 6.4** - Memory patterns (2-3 hours)
6. **Phase 6.5** - oneDNN hybrid (research, potentially days)

### Success Criteria

- **Minimum**: FA ON matches FA OFF performance (1.0x)
- **Goal**: FA ON is 10-20% faster than FA OFF at long contexts (>1K tokens)
- **Stretch**: FA ON is 30%+ faster at very long contexts (>4K tokens)

### Benchmarking Commands

```bash
# Quick comparison
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p 128,256,512,1024,2048 -n 0 -ngl 99 -fa 0,1

# With generation
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p 512 -n 128 -ngl 99 -fa 0,1

# Large model test
ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -p 512 -n 128 -ngl 99 -fa 0,1
```

### Notes

- The non-FA path scales poorly at very long contexts due to O(N²) memory
- FA should eventually win at long enough contexts even if slower per-op
- Focus optimizations on the common case (256-2048 token prompts)
- Consider adding runtime selection: use non-FA for short, FA for long

---

## References

- [Intel SYCL joint_matrix documentation](https://intel.github.io/llvm-docs/dpcpp/matrix.html)
- [Intel XMX architecture whitepaper](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691)
- [Intel Arc GPU architecture](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/desktop/a-series/overview.html)
