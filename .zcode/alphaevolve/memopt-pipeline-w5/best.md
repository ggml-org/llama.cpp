# memopt-pipeline-w5 best.md (champion) - FULL

## Champion: g1 (explicit v_trans for tessera_paged_attn)

- **Branch**: `evolve-review/memopt-pipeline-w5` on the main repo, off bbfc3493d.
- **Champion tip**: 30b692672 (gene worktree), 6efe8bdc5 (review branch cherry-pick).
- **Patch**: `integration/patches/g1.patch` (88 lines, 4 files).
- **Lineage**: gen2 (gen0 baseline 3.8pct, gen1 diagnostic instrumentation that located the bug, gen2 the one-line semantic fix promoted).
- **Candidates evaluated**: 2.
- **Stacked with**: nothing else (single gene). Stacks cleanly on bbfc3493d (the wave-4 tip).

## The actual bug (root cause)

`ggml_tessera_paged_attn` inferred whether V is stored transposed from the
tensor shapes:

```c
const bool v_trans = v->ne[0] == k->ne[2];   // ggml.c, before fix
```

i.e. "if V's first dim equals K's n_kv dim, V must be transposed". This is
ambiguous: for the gemma-4-12B **SWA cache** the V head dimension is 256 and the
padded KV arena size n_kv is also 256, so `v->ne[0] == k->ne[2]` is true even
though the cache stores V **non-transposed** (v_trans was forced false by the
wave-4 fix at cache construction). The op then selected the transposed-V access
pattern and read V rows at the wrong offsets, producing multilingual garbage
from decode step 1 onward.

Why decode step 0 still matched: the first generated token survived the
corrupted attention only because it happened to be the argmax of both the
correct and corrupted distributions (token 562 = " A" is a high-prior token).
From step 1 onward every output was wrong.

Why only the hybrid (gemma) case broke:
- qwen2.5-0.5B (non-hybrid, head_dim=128 != n_kv=256): the inference picked
  v_trans=false, which is correct, so paged worked.
- gemma BASE layers (head_dim=512 != n_kv=256): also picked v_trans=false
  correctly.
- gemma **SWA layers** (head_dim=256 == n_kv=256): inference flipped to
  v_trans=true, reading V wrong. Every SWA layer produced noise, and since
  most gemma layers are SWA (5 of every 6), the whole decode collapsed.

This is precisely why wave 4 (which verified K/V data, strides, head dims, GQA
ratios all correct, and which fixed the unrelated v_trans cache-layout bug for
non-hybrid models) could not resolve gemma: the data was never wrong. The bug
was in the op's *self-inflicted* layout detection, not in the cache.

## The fix (4 files, +17/-6)

1. **ggml/include/ggml.h**: add `bool v_trans` parameter to
   `ggml_tessera_paged_attn`.
2. **ggml/src/ggml.c**: drop the shape-inferred `v_trans`; use the caller-
   supplied value. Replace the old ambiguous assert with explicit
   per-layout asserts (`v_trans ? v->ne[0]==k->ne[2] : v->ne[0]==k->ne[0]`).
3. **src/llama-graph.cpp**: both `build_attn` paged call sites (the plain
   kv_cache path and the kv_cache_iswa path) pass `mctx_cur->is_v_trans()`.
   The caller already knows the authoritative cache layout.
4. **tests/test-backend-ops.cpp**: the tessera_paged_attn case passes its
   existing `v_trans` member to the op.

The wave-4 gating (`!mctx_cur->is_v_trans()` in the paged dispatch condition)
already ensures paged is only selected when V is non-transposed, so the op now
always receives v_trans=false via this path. The v_trans=true branch in the
kernel is retained for completeness and is still exercised by the backend-ops
unit test.

## Scores (review branch; 80-token greedy decode, temp=0, argmax)

- **gemma-4-12B f16 paged vs flash (seed=0)**: 100pct (80/80). Was 3.8pct.
- **gemma-4-12B f16 paged vs flash (seed=7, Mars prompt)**: 100pct (60/60).
- **qwen2.5-0.5B f16 paged vs flash**: 100pct (80/80). No regression.
- **ctest -R test-server-prompt-cache**: PASS.
- **test-backend-ops TESSERA_PAGED_ATTN (CPU + Metal)**: PASS (both v_trans
  true/false cases).

All gates from the spec are met: gemma >= 95pct (got 100pct), ctest passes,
qwen non-hybrid stays 100pct.

## Culprit layer/cache

- **Which cache**: the **SWA** half of the hybrid-iswa cache (head_dim 256).
  The base half (head_dim 512) was unaffected because 512 != 256.
- **Which layers**: every SWA attention layer of gemma-4-12B (layers where
  `is_swa_impl[il]` is true, i.e. 5 of every 6 layers). The 8 layers with
  `n_head_kv=1` (layers 5,11,17,23,29,35,41,47) were not the cause - that is a
  genuine architectural feature of gemma-4-12B and the kernel handles their
  GQA correctly; they were a red herring that the diagnostic ruled out.
- **Root cause**: shape-based V-layout inference in the op construction, which
  is ambiguous when head_dim == n_kv. Fixed by making the layout explicit.

## Bugs found

- **Primary**: `ggml_tessera_paged_attn` v_trans shape inference ambiguity
  (fixed, this gene).
- **Diagnostic notes** (not bugs, recorded for the next agent):
  - The wave-4 `!mctx_cur->is_v_trans()` gating means the paged op only ever
    runs with v_trans=false in production. The v_trans=true branch of the
    kernel is currently only reachable via the test-backend-ops unit test.
    If a future change relaxes that gating, the v_trans=true Metal/CPU
    kernels need an independent correctness audit on a real transposed cache.
  - The gemma-4-12B Q5_K_M telemetry build has `n_head_kv=1` on layers
    5,11,17,23,29,35,41,47 by design (verified against the GGUF
    `attention.head_count_kv` array, length 48). Not a bug.

## Inspect

```
git log --oneline bbfc3493d..evolve-review/memopt-pipeline-w5
git diff bbfc3493d...evolve-review/memopt-pipeline-w5
```
