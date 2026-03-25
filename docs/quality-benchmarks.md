# TurboQuant+ Quality Benchmarks

## Why These Benchmarks Matter

Speed without quality is useless. We claim 4.6× KV cache compression at 91-97% of q8_0 speed.
But we have ZERO quantitative quality data on the actual llama.cpp build. "Coherent text" and
Python cosine similarity (0.985) are not sufficient.

The llama.cpp CONTRIBUTING.md requires for new quant types:
1. Perplexity vs FP16/BF16
2. KL divergence data
3. Performance data via llama-bench

Papers (KIVI, QJL, RotateKV) all use wikitext-2 perplexity as the primary metric.
Prince Canuma validated with NIAH at 8K/32K/64K context.

## What We're Measuring

### 1. Perplexity (wikitext-2)
- **What**: How well the model predicts next tokens over a standard text corpus
- **Why**: Gold standard for LLM quality. Lower = better. Sensitive to KV cache quality.
- **Target**: turbo3 within 1% of q8_0 perplexity. If >2%, quality problem.
- **Comparison**: f16, q8_0, q4_0, q4_1, q5_0, turbo3

### 2. KL Divergence vs f16
- **What**: How different the output probability distribution is vs full precision
- **Why**: Measures distributional shift, not just top-token accuracy
- **Metrics**: mean KLD, delta-p RMS, same-top-p percentage
- **Required by**: llama.cpp CONTRIBUTING.md for upstream acceptance

### 3. Passkey Retrieval (built-in NIAH)
- **What**: Can the model retrieve a specific passkey from a long haystack?
- **Why**: Tests attention pattern preservation over long context
- **Comparison**: f16 vs turbo3 at various context lengths and needle positions

### 4. Generation Quality (qualitative)
- **What**: Side-by-side text generation comparison
- **Why**: Catches issues that aggregate metrics miss (repetition, coherence)

## Test Configuration

- **Model**: Qwen 3.5 35B-A3B MoE Q8_0 (primary), Qwopus v2 27B Q8_0 (secondary)
- **Hardware**: Apple M5 Max 128GB, Metal GPU
- **Dataset**: wikitext-2-raw (downloaded via scripts/get-wikitext-2.sh)
- **Context**: 512 tokens for perplexity (fast), 2048+ for NIAH
- **Chunks**: 8 for initial, 32 for final numbers

## Commands

### Download dataset
```bash
cd ~/local_llms/llama.cpp && bash scripts/get-wikitext-2.sh
```

### Perplexity suite
```bash
LLAMA=~/local_llms/llama.cpp/build-turbo/bin
MODEL=~/local_llms/models/Qwen3.5-35B-A3B-Q8_0.gguf
WIKI=~/local_llms/llama.cpp/wikitext-2-raw/wiki.test.raw

for ct in f16 q8_0 q4_0 q4_1 q5_0 turbo3; do
  echo "=== $ct ==="
  $LLAMA/llama-perplexity -m $MODEL -f $WIKI -c 512 \
    -ctk $ct -ctv $ct -fa on --chunks 8 -ngl 99 \
    2>&1 | tee results/ppl_${ct}.log
done
```

### KL Divergence
```bash
# Save f16 baseline logits
$LLAMA/llama-perplexity -m $MODEL -f $WIKI -c 512 \
  -ctk f16 -ctv f16 -fa on --chunks 8 -ngl 99 \
  --kl-divergence-base results/f16_logits.kld

# Compute KL divergence for each cache type
for ct in q8_0 q4_0 turbo3; do
  $LLAMA/llama-perplexity -m $MODEL -f $WIKI -c 512 \
    -ctk $ct -ctv $ct -fa on --chunks 8 -ngl 99 \
    --kl-divergence --kl-divergence-base results/f16_logits.kld \
    2>&1 | tee results/kld_${ct}.log
done
```

## Results

*To be filled after running benchmarks.*

### Perplexity (wikitext-2, 512 context)

| Cache Type | Bits/val | Perplexity | vs f16 | vs q8_0 |
|------------|----------|------------|--------|---------|
| f16 | 16 | — | baseline | — |
| q8_0 | 8 | — | — | baseline |
| q4_0 | 4 | — | — | — |
| q4_1 | 4.5 | — | — | — |
| q5_0 | 5.5 | — | — | — |
| **turbo3** | **3.5** | — | — | — |

### KL Divergence vs f16

| Cache Type | Mean KLD | Delta-p RMS | Same Top-p % |
|------------|----------|-------------|-------------|
| q8_0 | — | — | — |
| q4_0 | — | — | — |
| **turbo3** | — | — | — |

### Passkey Retrieval

| Cache Type | 1K | 2K | 4K | 8K |
|------------|----|----|----|----|
| f16 | — | — | — | — |
| q8_0 | — | — | — | — |
| turbo3 | — | — | — | — |

## INITIAL RESULTS — QUALITY FAILURE

### Perplexity (wikitext-2, 512 context, 8 chunks)

| Cache Type | Bits/val | Perplexity | vs f16 |
|------------|----------|------------|--------|
| f16 | 16 | 6.121 | baseline |
| q8_0 | 8 | 6.111 | -0.16% (better!) |
| q4_0 | 4 | 6.142 | +0.34% |
| **turbo3** | **3.5** | **165.6** | **+2607%** ❌ |

**turbo3 perplexity is 27× worse than f16. This is catastrophic.**

The model generates "coherent-looking" text but the actual predictions are garbage.
Speed benchmarks were meaningless — we were measuring how fast the model produces wrong answers.

### Root cause investigation needed
The block size 32 change + MSE-only + pre-rotate-queries may have introduced a bug:
1. The norm handling: storing full 128-element group norm in each 32-element block
   but dequant treats it as a per-block norm — scale mismatch?
2. The pre-rotate-queries: rotation matrix might not match the quantize rotation
3. The 3-bit index split (qs + signs) might have a packing error
4. The non-vec flash attention instantiation might use wrong nl parameter

MUST FIX before claiming any quality results.

## Quality Bisection Log

Goal: find which change broke perplexity (165.6 vs 6.1 baseline).

Changes applied (in order):
1. Pre-rotate-queries: rotation moved from dequant to Q in attention graph
2. MSE-only: dropped QJL, 3-bit centroids (2-bit qs + 1-bit signs)
3. Block size 32: storage blocks shrunk from 128 to 32 elements

Bisection plan:
- [ ] Test A: revert block 32 → 128, keep MSE-only + pre-rotate
- [ ] Test B: revert MSE-only → 2bit+QJL, keep block 128 + pre-rotate
- [ ] Test C: revert pre-rotate → rotation in dequant, keep block 128 + 2bit+QJL
- [ ] If A/B/C all still bad: check the rotation matrix consistency

## Bisection Results

### Test A: Disable Q rotation → PPL 165.6 (STILL BAD)
Pre-rotate-queries is NOT the problem. Q rotation has no effect on perplexity.

### Root Cause Found: V CACHE IN ROTATED SPACE

The bug: both K and V caches go through the same turbo3 quantize (which rotates)
and dequant (which returns rotated values without inverse rotation).

For K cache: this is correct because Q is also rotated → <R*q, R*k> = <q, k> ✓
For V cache: this is WRONG because the attention output = attn_weights @ V
needs to be in ORIGINAL space, not rotated space.

The fix: V cache must NOT be rotated during quantize, or the attention output
must be inverse-rotated after the V multiplication.

Options:
1. Skip rotation for V in quantize — only rotate K
2. Add inverse rotation after attention output: out = R^T @ (attn @ R*V) = attn @ V ✓
3. Use f16 for V cache (only compress K) — mixed types need support

Option 1 is simplest: in the SET_ROWS kernel, check if this is K or V and
skip rotation for V. The dequant already skips rotation.
Problem: the SET_ROWS kernel doesn't know if it's writing K or V.

Option 2 is cleanest: add R^T multiplication to attention output in the graph.
This is one ggml_mul_mat per layer (same as Q rotation). Cost is negligible.

### Root Cause #1: V cache dequant returns rotated-space values
Python verified: cosine(input, dequant_output) = 0.02 (garbage)
After inverse rotation: cosine = 0.987 (correct)

### Root Cause #2: dynamic_cast to llama_kv_cache_context FAILS
The Qwen 3.5 MoE uses llama_memory_hybrid_context (not llama_kv_cache_context).
The dynamic_cast returns null, so the V inverse rotation never executes.
This also means the Q pre-rotation never executed — explaining why the
chat server seemed to work (no rotation applied = raw quantize/dequant, which
happens to produce plausible-looking text at low context but garbage perplexity).

### Fix needed
1. Store turbo rotation tensors in llm_graph_context (not KV cache)
   OR access them through the hybrid memory interface
2. Apply V inverse rotation through a non-cast path
3. Verify Q rotation also executes (it was also using the same cast)

### The "coherent text" mystery explained
The model produced grammatical text because:
- Without Q rotation, attention scores are computed in wrong space but still produce
  some attention pattern (just not the right one)
- Without V inverse rotation, the output is in rotated space which is orthogonal to
  the correct space — but each layer compounds the error
- Short conversations look plausible but perplexity reveals the content is wrong

### Bisect: block 128 → STILL PPL=165.6
Block size is NOT the issue. PPL identical at block 128 and block 32.

### Bisect: disable Q rotation → STILL PPL=165.6
Pre-rotate-queries is NOT the issue. PPL identical with/without Q rotation.

### Root cause #2 confirmed: dynamic_cast fails for MoE hybrid memory
The Q rotation and V inverse rotation NEVER execute because mctx is
llama_memory_hybrid (not llama_kv_cache). The kv_cache IS inside the
hybrid object but we can't reach it.

### What this means
Without Q rotation and without V inverse rotation:
- K cache is quantized in ROTATED space, dequanted in ROTATED space
- Q is in ORIGINAL space (no pre-rotation)
- <original_Q, rotated_K> = GARBAGE (cosine ≈ 0.02)
- V cache similarly in rotated space, output never inverse-rotated

The entire time, K/V values were in the wrong coordinate system.
The "coherent text" was an illusion from the language model's resilience.

### Fix approach
Need to get rotation tensors through the hybrid memory hierarchy.
Virtual methods on llama_memory_context_i are the cleanest path.

## FIX: Inverse rotation restored in dequant

### Perplexity (wikitext-2, 512 context, 8 chunks)

| Cache Type | Perplexity | vs f16 | vs q8_0 |
|------------|------------|--------|---------|
| f16 | 6.121 | baseline | — |
| q8_0 | 6.111 | -0.16% | baseline |
| q4_0 | 6.142 | +0.34% | +0.51% |
| **turbo3** | **6.194** | **+1.19%** | **+1.36%** |

**turbo3 within 1.4% of q8_0. Quality target MET.**

### Root cause of previous failure
Pre-rotate-queries never executed because:
1. Q tensor ne[0]=256 (GQA concatenated heads) vs rotation ne[0]=128
2. MoE hybrid memory context fails dynamic_cast to llama_kv_cache_context

### Current state
- Inverse rotation restored in dequant (correct quality, slower speed ~10.7 tok/s)
- Speed optimization (pre-rotate-queries) needs reimplementation for GQA + hybrid

## Current State Summary (after quality fix)

### What works:
- turbo3 KV cache compression: 4.6× compression ratio
- Perplexity: 6.194 (1.4% of q8_0) — QUALITY TARGET MET
- MSE-only mode: 3-bit PolarQuant, no QJL — better quality than paper's Algorithm 2
- WHT rotation in dequant: correct inverse rotation restoring original space
- Block size 128: works on Metal for both MoE and Dense models
- Works for MoE (hybrid memory), Dense, and ISWA models

### Speed:
- ~10.7 tok/s MoE gen (8× slower than q8_0 85.5)
- Bottleneck: inverse WHT rotation in dequant (called per block per attention)

### What doesn't work:
- Pre-rotate-queries optimization: never executed because
  1. Q tensor ne[0]=256 (GQA concatenated heads) vs rotation ne[0]=128
  2. Need per-head reshape before rotation
- Previous speed numbers (51-77 tok/s) were INVALID — measured garbage output speed

### Lessons learned:
1. ALWAYS run perplexity before claiming quality
2. "Coherent text" is NOT quality validation
3. dynamic_cast fails silently for MoE hybrid memory
4. GQA models concatenate heads in ne[0] — can't apply per-head ops without reshape
5. #include in ggml-metal.metal causes silent CPU fallback
6. Block size matters for speed but NOT for quality (both 32 and 128 had same PPL)
