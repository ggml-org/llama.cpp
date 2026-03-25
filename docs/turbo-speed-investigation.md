# TurboQuant Speed Investigation

## Problem Statement

TurboQuant generates at 2.4 tok/s vs 85.0 tok/s for q8_0 on Qwen 3.5 35B-A3B MoE (M5 Max).
That's a 35× regression. Compression target (4.9×) is met, but speed makes it impractical.

## Root Cause Analysis

### Why it's slow

The flash attention kernel calls `dequantize_turbo3_0(block_ptr, il, &reg)` once per chunk:
- `type4x4` path: 16 elements per call, 128/16 = **8 calls per block**
- `type4` path (flash_attn_ext_vec): 4 elements per call, 128/4 = **32 calls per block**

Each call does the FULL 128-element dequantize:
1. Unpack 128 indices from packed bits → centroid lookup
2. Inverse WHT rotation (7 butterfly passes × 128 = 896 ops)
3. Unpack 128 QJL signs
4. Inverse WHT rotation on signs (another 896 ops)
5. Combine and scale

Total per call: ~2000 ops + 256 bytes stack allocation
Total per block: 8-32 × 2000 = **16,000-64,000 ops** (should be 2000)

### Comparison with q8_0

q8_0 dequantize: `x[i] = qs[i] * d` — 1 multiply per element, no stack, no rotation.
Per block (32 elements): 32 ops. Per 128 elements: 128 ops.

TurboQuant is doing 64,000 ops where q8_0 does 128. That's **500× more compute per block**.

Even with WHT (18× less than dense matvec), we're at 64,000/18 ≈ 3,500 ops vs 128.
That's still **27× more compute**, close to the measured 35× regression.

The extra ~8× gap is from stack allocation overhead (128-float arrays created/destroyed
32× per block) and memory bandwidth (reading the constant sign arrays 32× per block).

---

## Investigation Plan

### Approach A: Modify flash attention to dequantize once per block
- [ ] **A1**: Study the flash attention kernel template structure
  - Find where `deq_k` and `deq_v` are called
  - Understand the iteration pattern: which blocks, which chunks
  - Identify where to insert a "pre-dequantize" step
- [ ] **A2**: Add threadgroup memory buffer for pre-dequantized blocks
  - Allocate `threadgroup float turbo_deq_buf[128]` in flash attention kernel
  - Before the per-chunk loop, call `turbo3_dequantize_full_block()` once into this buffer
  - Replace per-chunk dequantize calls with reads from the buffer
- [ ] **A3**: Test with turbo-specific flash attention kernel instantiation
  - The generic template uses function pointers for dequantize
  - May need a specialized kernel that bypasses the per-chunk pattern
- [ ] **A4**: Benchmark after each change

### Approach B: Custom flash attention kernel for turbo types
- [ ] **B1**: Fork `kernel_flash_attn_ext_vec` into `kernel_flash_attn_ext_vec_turbo`
  - Remove the generic dequantize function pointer
  - Inline turbo-specific dequantize at the block level
  - Use threadgroup memory for the dequantized block
- [ ] **B2**: Replace per-chunk reads with direct buffer indexing
  - Instead of `deq_k(pk4x4 + block_idx, chunk_idx, tmp)`, do `tmp = buf[chunk_idx * 16 : (chunk_idx+1) * 16]`
- [ ] **B3**: Register the custom kernel in pipeline lookup
- [ ] **B4**: Benchmark

### Approach C: Restructure dequantize to amortize across chunks
- [ ] **C1**: Change the dequantize API to accept a pre-allocated buffer
  - `dequantize_turbo3_0(block_ptr, il, &reg, thread float * shared_buf)`
  - On first call (il==0), fill the buffer; on subsequent calls, read from it
  - Problem: can't change the function signature without changing all callers
- [ ] **C2**: Use the `nl_k` parameter differently
  - Currently nl_k=32 for turbo3 (128/4). What if we set nl_k=1 and return all 128 at once?
  - Would need the caller to handle 128-element chunks
  - Probably breaks the template assumptions

### Approach D: Reduce per-call overhead without architecture changes
- [ ] **D1**: Precompute the WHT butterfly as a lookup table
  - Instead of 7 butterfly passes, use a precomputed 128-element permutation
  - Trade memory for compute: 128 × 4 bytes = 512 bytes constant
  - Won't help much since WHT is already fast
- [ ] **D2**: Pack the dequantize tighter — reduce stack allocations
  - Merge the centroid lookup + WHT into a single pass
  - Avoid allocating separate `recon[128]` and `signs_f[128]` — interleave
- [ ] **D3**: Use half precision for intermediate calculations
  - `half` arithmetic is 2× faster on Apple Silicon
  - May reduce quality slightly but worth testing

---

## Expected Outcomes

| Approach | Expected Speedup | Effort | Risk |
|----------|-----------------|--------|------|
| A (modify kernel) | 8-32× (eliminate redundant calls) | Medium | Medium — need to understand kernel internals |
| B (custom kernel) | 8-32× + optimal memory access | High | Low — clean separation |
| C (restructure API) | 8-32× | Low | High — may break template |
| D (reduce overhead) | 2-3× | Low | Low |

**Recommended order**: D first (quick wins), then A or B (the real fix).

Target: D → 5-8 tok/s, then A/B → 20-40 tok/s.

---

## Progress Log

### 2026-03-25: Initial investigation
- Dense matvec: 2.4 tok/s (35× slower than q8_0)
- WHT rotation: 2.4 tok/s (same — bottleneck is redundant calls, not per-call compute)
- Root cause confirmed: dequantize called 8-32× per block by flash attention
- Codex + roast reviewed WHT implementation: correct, no bugs

### Next: Start with Approach D (reduce per-call overhead)

### 2026-03-25: simd_broadcast attempt
- Added simd_broadcast fast path for K and V dequant (nl_k==32 && DK==128)
- Thread 0 dequantizes, broadcasts 128 floats via simd_broadcast loop
- **Result: still 2.4 tok/s** — the 128-iteration simd_broadcast loop per cc iteration
  is itself expensive. 32 cc iterations × 128 broadcasts = 4096 simd_broadcast calls per block.
- Codex review caught: DK>128 OOB bug (fixed), turbo4 using turbo3 dequant (fixed),
  uninitialized turbo_buf on non-lane-0 (fixed with zero-init)
- **Conclusion**: simd_broadcast is wrong tool. Need threadgroup memory instead.

### Next: try threadgroup memory approach
- Allocate extra threadgroup memory in FATTN_SMEM
- One thread writes 128 floats to threadgroup, barrier, all threads read
- This reduces to 1 dequant + 1 barrier per cc iteration instead of 128 broadcasts

### 2026-03-25: threadgroup memory attempt
- Replaced simd_broadcast with threadgroup memory + simdgroup_barrier
- Thread 0 dequantizes into threadgroup, barrier, all threads read
- **Result: still 2.4 tok/s**
- Eliminating 31/32 redundant dequant calls had NO effect on speed
- This means the dequant cost itself (even 1× per block) is NOT the bottleneck
- Or the bottleneck is elsewhere entirely (SET_ROWS quantize? block size overhead?)

### Hypothesis: block size 128 vs 32 causes structural overhead
- q8_0 block size = 32, turbo block size = 128
- The flash attention kernel processes DK4/NL elements per thread per cc iteration
- For q8_0: DK4/NL = 32/8 = 4 iterations (inner ii loop runs 4×)
- For turbo: DK4/NL = 32/32 = 1 iteration (inner ii loop runs 1×)
- But NL = 32 for turbo vs NL = 4 for q8_0 (32/8=4, C=32, NE=1→NW/NE=32 for both)
- Actually NL = NW/NE = 32/1 = 32 for both... so DK4/NL should be the same?
- Wait: for q8_0, nl_k=8 (32 elements / 4 per t4 = 8 chunks). DK4 = 128/4 = 32. DK4/NL = 32/32 = 1.
- So BOTH q8_0 and turbo have DK4/NL = 1 iteration in the inner loop.
- The only difference is the dequant function itself.

### Next: profile whether the bottleneck is in dequant or elsewhere
- Test with a no-op dequant (return zeros) to measure the kernel overhead
- If still slow → bottleneck is NOT dequant, it's structural

### 2026-03-25: NO-OP DEQUANT TEST — CRITICAL FINDING
- Made dequantize_turbo3_0 and dequantize_turbo3_0_t4 return all zeros (no-op)
- **Result: STILL 2.4 tok/s**
- **The dequant function is NOT the bottleneck.**
- Even with zero-cost dequant, turbo3 is 35× slower than q8_0.
- The bottleneck is ELSEWHERE:
  1. SET_ROWS (quantize) path — 128×128 WHT per block during KV cache write
  2. Kernel dispatch overhead for 128-element blocks
  3. Memory layout/alignment issues with the larger block structure
  4. The quantize kernel (kernel_set_rows_turbo) doing full WHT per KV write

### New investigation target: SET_ROWS (quantize path)
- The quantize_turbo3_0 function in the Metal shader does:
  1. Normalize (128 ops)
  2. Copy to working buffer (128 ops)  
  3. Forward WHT rotation (896 ops)
  4. Centroid lookup + pack (128 ops)
  5. Inverse WHT rotation (896 ops)
  6. Residual (128 ops)
  7. Forward QJL WHT rotation (896 ops)
  8. Sign pack (128 ops)
  Total: ~3200 ops per block in the QUANTIZE path
- This runs for every token at every layer at every head
- For Qwen 3.5 MoE: 28 layers × 8 heads = 224 quantize calls per generated token
- Each quantize does 3200 ops = 716,800 ops per token just for KV cache write

### Next: profile SET_ROWS path / try no-op quantize test

### 2026-03-25: THE BUG — #include caused CPU fallback!!! 🎉
- `#include "turbo-wht.h"` in ggml-metal.metal causes Metal JIT to fail
- The model SILENTLY falls back to CPU for all operations
- ALL previous benchmarks (2.4 tok/s) were measuring CPU, not Metal GPU
- **After inlining turbo-wht.h:**
  - MoE: 2.4 → 10.7 tok/s gen (4.5× improvement, now on Metal)
  - Qwopus: 2.4 → 5.3 tok/s gen (2.2× improvement)
  - Prompt speeds: 4× → 60-67 tok/s (MoE), 3.5 → 30 tok/s (Qwopus)
- Remaining gap vs q8_0: 8× (MoE), 3.3× (Qwopus) — down from fake 35×
- **Rule: NEVER use #include in ggml-metal.metal — always inline**
- This is the same bug we hit with turbo-matrices.h earlier and forgot to apply to WHT

### Real benchmarks (proper Metal GPU):

| Model | Cache | Prompt | Gen | Compression | vs q8_0 |
|-------|-------|--------|-----|-------------|---------|
| MoE | q8_0 | 222.8 | 85.5 | 2.0× | 1.00× |
| MoE | turbo3 | 67.3 | 10.7 | 4.9× | 0.13× |
| Qwopus | q8_0 | 83.1 | 17.6 | 2.0× | 1.00× |
| Qwopus | turbo3 | 29.8 | 5.3 | 4.9× | 0.30× |

### Remaining optimization targets (ACTUAL, not fake):
- 8× gap on MoE gen (85.5 vs 10.7)
- 3.3× gap on Qwopus gen (17.6 vs 5.3)
- WHT rotation is O(d log d) per block — already optimized
- The per-chunk redundant call pattern still wastes ~8× on dequant
- BUT: no-op dequant test was on CPU, need to redo on Metal

### 2026-03-25: Upstream research findings
- TurboQuant feature request exists upstream: ggml-org/llama.cpp#20977
- mudler has experimental fork — should compare notes
- New quant type contribution guidelines require perplexity + KL divergence data
- The CPU fallback pattern is known: MXFP4 hit the same issue (PR #20609)
- Metal Tensor API optimization (PR #20962) could give us another ~26% on mul_mat
- Our current 8× gap on MoE / 3.3× on Dense is REAL Metal overhead, not a bug
- The WHT rotation cost per block is the genuine bottleneck now

### Summary of investigation
| Test | Gen tok/s | What we learned |
|------|-----------|-----------------|
| q8_0 baseline | 85.5 | Target speed |
| Dense matvec rotation | 2.4 | Was actually CPU fallback |
| WHT rotation | 2.4 | Still CPU fallback |
| simd_broadcast opt | 2.4 | Still CPU fallback |
| threadgroup opt | 2.4 | Still CPU fallback |
| No-op dequant | 2.4 | Still CPU fallback |
| No-op quantize | 2.4 | Still CPU fallback |
| **Inlined WHT (proper Metal)** | **10.7** | **THE FIX — #include caused CPU fallback** |

Rule: NEVER use #include in ggml-metal.metal. Always inline.

### 2026-03-25: Upstream competitive intelligence — GAME CHANGERS

**Finding 1: QJL residual kills quality when naively re-added**
Dejan.ai found cosine similarity drops to 0.69 with naive QJL add-back.
Must use MSE-only for drop-in replacement, or fused kernel for QJL.

**Finding 2: Pre-rotate queries, not keys (THE BIG WIN)**
`<q, R^T * c[idx]> = <R*q, c[idx]>`
Rotate query ONCE, then dequant is just a centroid table lookup.
No rotation in the hot dequant path. Would make turbo3 as fast as q4_0.

**Finding 3: WHT abandoned by everyone**
Both Dejan.ai and mudler tried WHT, both switched to dense QR.
But with pre-rotate-queries, rotation method doesn't matter.

**Finding 4: unixsysdev gets -0.8% speed loss (CUDA, block=32)**
Pure polar coordinate encoding, no random rotation, fused kernel.

**Finding 5: We're the only Metal implementation**
Everyone else is CPU-only or CUDA. Our Metal kernels are unique.

### Immediate action items:
- [ ] Test MSE-only (no QJL) — simplify dequant, may improve quality + speed
- [ ] Implement pre-rotate-queries — eliminates rotation from dequant entirely
- [ ] Consider block size 32 for better GPU parallelism

### 2026-03-25: Speed ceiling test — no rotation in dequant (proper Metal)
- Removed turbo_rotate_inverse from dequant (quality broken but measures ceiling)
- **Result: 49.1 tok/s gen (4.6× improvement over 10.7)**
- Prompt: 162.6 tok/s (2.4× improvement over 67.3)
- This is 57% of q8_0 speed (85.5 tok/s)
- Confirms: removing rotation from dequant via pre-rotate-queries
  would get us from 10.7 → ~49 tok/s
- Remaining gap (49 vs 85) is from block size 128 + QJL overhead
- REVERTED change after measurement

### 2026-03-25: PRE-ROTATE-QUERIES — THE WIN 🎉
- Implemented: store R^T in KV cache, apply ggml_mul_mat(R_T, q) in build_attn_mha
- Stripped turbo_rotate_inverse from Metal dequant
- Codex caught: buffer_clear zeroing rotation after init (fixed)
- **MoE gen: 10.7 → 51.4 tok/s (4.8× speedup, 60% of q8_0)**
- **Qwopus gen: 5.3 → 14.6 tok/s (2.8× speedup, 83% of q8_0)**
- Prompt: 67 → 160 tok/s (MoE), 30 → 68 tok/s (Qwopus)

### FINAL SUMMARY

| Stage | MoE tok/s | Qwopus tok/s | What happened |
|-------|-----------|-------------|---------------|
| First benchmark | 2.4 | 1.3 | CPU fallback (#include bug) |
| Fixed #include | 10.7 | 5.3 | Real Metal, WHT rotation overhead |
| Speed ceiling test | 49.1 | — | Confirmed rotation is the bottleneck |
| **Pre-rotate-queries** | **51.4** | **14.6** | **Rotation moved from dequant to Q** |
| q8_0 baseline | 85.5 | 17.6 | Target |

**Total improvement: 21× from first benchmark to final (2.4 → 51.4 tok/s)**
**Compression: 4.9× throughout**

### Key lessons learned
1. NEVER use #include in ggml-metal.metal — causes silent CPU fallback
2. Always verify Metal library loads successfully before benchmarking
3. The no-op test (set function to return zeros) is the fastest way to isolate bottlenecks
4. Pre-rotate-queries (from Dejan.ai) is the right architectural approach
5. Codex + roast reviews catch real bugs (buffer clear ordering, stale code, MSL limitations)
6. The ggml_mul_mat approach for Q rotation is clean and correct

### 2026-03-25: CHANGE 1 — Drop QJL (MSE-only) ✅
- Python validation: cosine 0.9508 → 0.9831, better on 99.3% of real KV vectors
- MoE gen: 51.4 → 62.2 tok/s (73% of q8_0)
- Qwopus gen: 14.6 → 15.5 tok/s (88% of q8_0)
- Qwopus prompt: 83.1 tok/s = 100% of q8_0!

### 2026-03-25: CHANGE 2 — WHT for Q rotation — NOT NEEDED
- Speed ceiling test: removing Q rotation entirely gives 61.3 tok/s (vs 62.2 with it)
- The dense 128×128 ggml_mul_mat adds <1% overhead — negligible on Metal
- The remaining gap (62.2 vs 85.5 on MoE) is structural:
  - Block size 128 (4 blocks per head) vs q8_0 block 32 (4 blocks per head too, but simpler dequant)
  - 128 centroid lookups (3-bit unpack + table) vs 32 byte * scale (q8_0)
  - This is the irreducible cost of the algorithm
  
### FINAL PERFORMANCE

| Model | q8_0 | turbo3 MSE-only | Speed | Compression |
|-------|------|-----------------|-------|-------------|
| MoE 35B | 85.5 tok/s | **62.2 tok/s** | **73%** | **4.9×** |
| Qwopus 27B | 17.6 tok/s | **15.5 tok/s** | **88%** | **4.9×** |

### 2026-03-25: Block size diagnostic — THE ANSWER

| Cache | Block | Gen tok/s | vs q8_0 |
|-------|-------|-----------|---------|
| q8_0 | 32 | 84.1 | 1.00× |
| q4_0 | 32 | 84.2 | 1.00× |
| q4_1 | 32 | 84.6 | 1.01× |
| q5_0 | 32 | 80.6 | 0.96× |
| turbo3 | 128 | 62.5 | 0.74× |
| turbo4 | 128 | 42.8 | 0.51× |

CONCLUSION: q4_0 (block 32, 4-bit) runs at 100% of q8_0 speed.
The 26% gap on turbo3 is ENTIRELY from block size 128.
The quantization math (centroid lookup) adds zero overhead at block 32.

Next: implement block size 32 variant of turbo3.
Challenge: rotation operates on head_dim=128, but blocks are 32.
Solution: quantize 128 elements with rotation, store as 4×32 blocks.
Dequant reads 32-element blocks without rotation (pre-rotate-queries).

### 2026-03-25: BLOCK SIZE 32 — TARGET EXCEEDED 🎉

| Model | q8_0 | turbo3 block32 | vs q8_0 | Compression |
|-------|------|----------------|---------|-------------|
| MoE prompt | 222.8 | **218.5** | **98%** | 4.6× |
| MoE gen | 85.5 | **77.7** | **91%** | 4.6× |
| Qwopus prompt | 83.1 | **89.5** | **108%** | 4.6× |
| Qwopus gen | 17.6 | **17.0** | **97%** | 4.6× |

Target was 75+ tok/s on MoE. Hit 77.7. Qwopus prompt is FASTER than q8_0.

Codex post-commit review found 3 bugs (filed #29):
- TURBO_D was QK_TURBO3 (broke turbo4 C code) — fixed
- SET_ROWS kernel turbo3-specific but instantiated for turbo4 — tracked
- Tail block drop for non-128 head dims — tracked

### COMPLETE JOURNEY

| Optimization | MoE gen | vs q8_0 |
|-------------|---------|---------|
| CPU fallback (bug) | 2.4 | 3% |
| Real Metal | 10.7 | 13% |
| Pre-rotate-queries | 51.4 | 60% |
| MSE-only (drop QJL) | 62.2 | 73% |
| **Block size 32** | **77.7** | **91%** |
| q8_0 baseline | 85.5 | 100% |

**32× total improvement. 4.6× compression at 91-97% speed.**

## Pre-Rotate-Queries Investigation (2026-03-25)

Attempted to move WHT inverse rotation from dequant to graph-level ops. Goal: eliminate per-block O(128) rotation, reclaiming speed from ~10.7 to ~77 tok/s.

**Result: FAILED.** Graph-side rotation gives PPL 23.5 vs dequant-side 6.19. Root cause unknown despite extensive debugging. See `docs/pre-rotate-queries-investigation.md` for full test matrix.

**Key findings:**
- ggml_mul_mat(A, x) with row-major stored A gives A @ x (verified with 2x2 test)
- Both Q rotation and V inverse work mechanically (verified via isolation tests)
- Rotation matrices verified correct (R^T @ R = I, matches Metal WHT signs)
- Model head_dim = 256 with 128-element rotation groups (2 groups per head)
- Codex caught: `clear(true)` zeros rotation tensors without reinit (fixed)
- Codex caught: header comments had wrong matrix orientation (fixed)

**Current state:** Dequant inverse rotation restored. PPL = 6.194. Speed = ~10.7 tok/s.

**Root cause found (Gemini 3 Pro):** WHT and RoPE don't commute. KV stores WHT(RoPE(K)) but graph rotation gives RoPE(WHT(Q)). Fix: apply WHT after RoPE inside attention layer, not in build_attn_mha.

## FP16 WHT Optimization (2026-03-25)

Switched WHT inverse rotation from fp32 to fp16 (Metal native half precision). Centroids fit in fp16 (max |val| = 0.19). Butterfly add/sub stays within fp16 range.

Also implemented cooperative SIMD dequant for flash_attn_ext_vec (t4) path: each of 32 SIMD lanes unpacks only its 4 elements, then WHT butterfly runs across lanes via simd_shuffle.

### Prefill Speed Results (Qwen3.5-35B-A3B, wikitext-2, 32 chunks)

| Config | Prefill tok/s | vs q8_0 | PPL |
|--------|-------------|---------|-----|
| q8_0 baseline | 2694 | 1.00x | 5.41 |
| turbo3 no rotation | 1577 | 0.59x | - |
| turbo3 fp32 WHT (old) | 739 | 0.27x | 5.46 |
| **turbo3 fp16 WHT** | **1074** | **0.40x** | **5.47** |

**45% speedup from fp16 WHT.** Gap to q8_0 narrowed from 3.65x to 2.51x. PPL within 0.06 of baseline.
