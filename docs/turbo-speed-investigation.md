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
