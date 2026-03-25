# Pre-Rotate Queries Optimization Plan

## The Trick

`⟨q, R^T · centroids[idx]⟩ = ⟨R·q, centroids[idx]⟩`

Instead of inverse-rotating every K vector during dequant (128 WHT ops × every KV position),
rotate Q once before attention (128 WHT ops × 1 query).

## Changes Required

### 1. Store rotation parameters in model/context
- [ ] Add WHT sign arrays to llama_context or llama_kv_cache
- [ ] Accessible during graph building via hparams or cparams

### 2. Insert Q rotation in attention graph (llama-graph.cpp:1799)
- [ ] After `q = ggml_permute(ctx0, q, 0, 2, 1, 3);`
- [ ] Check: `if (k->type == GGML_TYPE_TURBO3_0 || k->type == GGML_TYPE_TURBO4_0)`
- [ ] Apply: `q = ggml_mul_mat(ctx0, rotation_matrix_tensor, q);`
- [ ] The rotation matrix is 128×128 per head_dim — shared across all heads/layers
- [ ] Q shape at this point: (head_dim, n_tokens, n_head, n_stream)
- [ ] Rotation operates on the head_dim dimension (dim 0)

### 3. Simplify Metal dequant to skip rotation
- [ ] `dequantize_turbo3_0_t4`: just centroid lookup + norm scale (no WHT)
- [ ] `dequantize_turbo4_0_t4`: same
- [ ] `turbo3_dequantize_full_block`: remove turbo_rotate_inverse calls
- [ ] Keep QJL correction? Dejan.ai says NO for drop-in replacement

### 4. Simplify Metal quantize similarly
- [ ] `quantize_turbo3_0`: still needs forward WHT (writing to KV cache)
- [ ] BUT: if we pre-rotate Q, the stored K values are in ROTATED space
- [ ] So quantize writes rotated centroids WITHOUT inverse rotation
- [ ] And dequant returns rotated centroids directly
- [ ] The Q·K dot product works because Q is also rotated

### 5. Handle QJL decision
- [ ] Option A: Drop QJL entirely (MSE-only, all bits to Lloyd-Max)
  - Simpler dequant (just centroid lookup)
  - Dejan.ai found this gives BETTER quality than naive QJL add-back
  - Saves 1 bit per element (or gives 1 extra bit to MSE)
- [ ] Option B: Keep QJL but handle in fused kernel
  - More complex, but theoretically better IP preservation
  - Requires keeping the two-part representation separate

## Expected Performance Impact

### Current (with WHT rotation in dequant):
- Dequant: unpack + centroid + inverse WHT + QJL WHT + scale = ~2000 ops/block
- MoE gen: 10.7 tok/s (8× slower than q8_0)

### After pre-rotate-queries (no rotation in dequant):
- Dequant: unpack + centroid + scale = ~256 ops/block
- Expected: close to q4_0 speed (84.5 tok/s on MoE)
- The only overhead vs q4_0 is the 128-element block vs 32-element block

### After dropping QJL (MSE-only):
- Dequant: unpack + centroid + scale = ~200 ops/block
- Block struct shrinks: no signs array needed
- Even simpler, even faster

## Test Plan

- [ ] A/B test 1: Current turbo3 vs MSE-only turbo3 (drop QJL) — quality + speed
- [ ] A/B test 2: Current turbo3 vs pre-rotated-Q turbo3 — speed
- [ ] A/B test 3: Pre-rotated-Q + MSE-only — the full optimization
- [ ] Quality: cosine sim, attention output comparison
- [ ] Speed: tok/s on MoE + Qwopus

## Implementation Order

1. **Quick win**: Remove WHT from dequant, test speed (broken quality but measures ceiling)
2. **MSE-only test**: Remove QJL from both quant and dequant, test quality
3. **Pre-rotate-Q**: Insert rotation in graph, full correct implementation

## Codex Review Findings

### Finding 1: R must be fixed and shared — VERIFIED ✅
Our WHT sign arrays are `constant float` in the Metal shader — same for ALL blocks,
layers, and heads. The rotation is indeed global and fixed. The optimization is valid.

### Finding 2: Use WHT for Q rotation, not dense matmul — AGREED
Insert a custom ggml op for WHT rotation, not ggml_mul_mat with a dense matrix.
Or implement as a Metal compute kernel that does in-place WHT on the Q tensor.

### Finding 3: Normalization must be consistent — VERIFIED ✅
Our WHT is normalized by 1/sqrt(128). The same normalization is applied in both
quantize (forward WHT) and dequant (inverse WHT). Pre-rotating Q with the same
normalized WHT preserves the dot product scale.

### Finding 4: Gate on KV cache config type, not transient tensor type — GOOD POINT
The `k->type` at graph build time may be F16 (intermediate). Need to check the
actual KV cache type from `cparams.cache_type_k` or similar.

### Finding 5: QJL removal is a new format — AGREED
Will test as a separate path, not modify existing turbo3 behavior.

### Finding 6: Prompt/decode consistency — MUST VERIFY
The quantize path (SET_ROWS) and dequant path (flash_attn) must agree on whether
rotation is stored or applied at read time.

### Finding 7: Backend parity — ACKNOWLEDGED
CPU backend still uses dense rotation in C code. Metal uses WHT. These produce
different results. Need to align or document the divergence.

## Implementation Progress

### Challenge: inserting rotation tensor in ggml graph
- ggml tensors don't have allocated data at graph build time
- Can't memcpy into tensor during graph construction
- Need to create the rotation matrix as a persistent tensor at init time
- Store in KV cache object or context, access via mctx in graph builder

### Simplest path forward
1. Add a `ggml_tensor * turbo_rotation` member to `llama_kv_cache`
2. Allocate and fill during KV cache construction when type_k is turbo
3. In `build_attn_mha`, access via `mctx` and apply `ggml_mul_mat` to Q
4. Strip rotation from Metal dequant (already tested — gives 49 tok/s)

### What we know for certain (from speed ceiling test)
- Removing rotation from dequant: 10.7 → 49.1 tok/s (confirmed on Metal)
- The math is sound (R is fixed, orthogonal, normalized — verified by codex)
- The insertion point is llama-graph.cpp line ~1799
- k->type at that point IS the KV cache storage type (verified)

### Files to modify
1. `src/llama-kv-cache.h` — add turbo_rotation tensor member
2. `src/llama-kv-cache.cpp` — allocate + fill rotation during construction
3. `src/llama-kv-cache-context.h` — expose rotation to graph context
4. `src/llama-graph.cpp` — apply rotation to Q before flash_attn
5. `ggml/src/ggml-metal/ggml-metal.metal` — strip rotation from dequant
6. `src/turbo-rotation-data.h` — pre-computed 128x128 rotation matrix
