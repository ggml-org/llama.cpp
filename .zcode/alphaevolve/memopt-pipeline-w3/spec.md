# memopt-pipeline-w3 spec (single-instance all-phases, single gene S4)

## Goal
Extend S1's q8_0 quantized-KV paged-attn to a 2-bit KIVI-style asymmetric
scheme (per-channel K, per-token V) to further reduce KV cache RSS, with the
dequant fused into the TESSERA_PAGED_ATTN kernel (CPU + Metal).

## Target / regions
- ggml/src/ggml-cpu/ops.cpp: ggml_compute_forward_tessera_paged_attn load_k/load_v
- ggml/src/ggml-metal/ggml-metal.metal: kernel_tessera_paged_attn_q8_0 mirror
- src/llama-graph.cpp: build_attn type gate (use_tessera_paged)
- src/llama-kv-cache.cpp: type acceptance, attn_rot gating
- possibly a new ggml 2-bit type or reuse q2_K

## Metric (MAXIMIZE)
- Primary: NEGATIVE peak RSS (B). Floor = 8,300,724,224 B (~8.30 GB) at pp512/n32.
- Correctness gate (HARD):
  - ctest -R test-server-prompt-cache must pass.
  - logit-diff vs f16 baseline: 64+ token greedy decode of a substantial prompt;
    report max_abs_delta. Tiny deltas that flip a few tokens are expected at 2-bit.
    Total output collapse (max_abs_delta huge, >5, or garbage text) = FAIL.
- Secondary: decode tok/s (must not regress catastrophically).

## Evaluator
- Build: cmake --build build-g1 --target llama-server llama-bench -- -j8
- RSS: gtime -l on llama-bench pp512 -n32, 3-run median, take "maximum resident
  set size" field. SERIALIZE across candidates.
- Correctness: see scripts under the run dir; compare q8_0 vs 2-bit greedy logits.

## Baseline sha
45fec50982fd3080450996b74a8c58ed6389ae1e (S1 q8_0 paged-attn + s2 lazy clear OFF + WIP)

## Budget
6 generations OR 50 min wall OR 15 candidates, whichever first.
stagnation_limit = 4.

## Constraints
- 16 GB RAM M1. 12B Q5_K_M (~8 GB weights). pp512 only (OOMs higher).
- SERIALIZE llama-bench across candidates.
- 3-run median for RSS, wide tie band.
- ASCII only. Commits on evolve/memopt-pipeline-w3/* only.
- Never weaken a test. Never master/main. Never push, never gh.

## Honest expectation
2-bit KV at 12B/512 has real RSS headroom but quality risk is higher than q8_0.
The Hadamard rotation tessera already applies to quantized K/V (attn_rot_k/v,
the QServe trick) is what makes 2-bit viable. If S4 can't hold quality, ship
"partial" and document - a clean q8_0 (S1) may be the practical sweet spot.
