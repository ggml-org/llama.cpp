# Wave 3 pre-flight facts (resolved by the orchestrator)

## CRITICAL: baseline correction from wave 2
Wave 2's s2 ("LLAMA_KV_LAZY_CLEAR") reported -1.94% RSS but DOES NOT reproduce
on the cleaner w3 baseline. Re-measured with S1+s2 both compiled in:
  - lazy clear OFF: 8,300,724,224 B  (~8.30 GB)  <- this is the real floor
  - lazy clear ON:  8,891,383,808 B  (~8.89 GB)  <- a REGRESSION (+0.59 GB)
The wave-2 "win" was within measurement noise / order-dependent. TREAT s2 AS
SUSPECT. The w3 baseline ships with s2's code present but the env var DEFAULTS
OFF. Do not assume s2 is a win; if you re-test it and it regresses, prune it.

## Build
- Baseline sha: 45fec50982fd3080450996b74a8c58ed6389ae1e
- Contains: S1 (q8_0 paged-attn) + s2 (lazy clear, off by default) + untracked
  tessera WIP (server-admission/metrics/prefill-policy + ggml-ane). Builds clean.
- Build: cmake --build build --target llama-server llama-bench -- -j8

## Model (16GB machine constraint - unchanged)
- /Volumes/Julian T7/models/gemma-4-12B-it-qat-unified-mtp-Q5_K_M-telemetry.gguf
- 12B Q5_K_M, 8.04 GiB. pp512 -n 32 only. DO NOT exceed pp512 (OOMs at 16GB).
- SERIALIZE eval across genes (one llama-bench at a time).

## True baseline (re-measured on w3 baseline tree)
- pp512: 69.00 t/s, peak RSS: 8,300,724,224 B (~8.30 GB) with lazy clear OFF.
- This is the floor to beat. Anything above 8.30 GB is a regression.

## Candidates in scope for wave 3
The plan called for S4, S5, S6, S8. Reality:
- S4 (KIVI 2-bit KV) extends S1 -> S1 is in baseline. OK to run.
- S5 (InfiniGen prefetch) needs a host-tier KV cache. S2 did not ship a real
  win, so S5's prerequisite is NOT met. SKIP S5.
- S6 (MoE disk offload) is independent but the workload model (gemma-4-12B) is
  DENSE, not MoE. S6 has no payoff on a dense model. SKIP S6 unless you switch
  to the Qwen3.6-35B-A3B MoE model (which won't fit in 16GB anyway). SKIP.
- S8 (speculative expert prefetch) extends S6. SKIP.

So wave 3 is effectively S4 only on this hardware/model. That's fine - S4 is
the highest-payoff candidate (2-bit KV ~2.6x KV reduction) and extends S1's
mechanism. Run it as a single focused gene.

## S4 target
KIVI-style asymmetric 2-bit KV: per-channel K, per-token V. Builds on S1's
q8_0 reader - add a q2_K or similar 2-bit type with on-the-fly dequant in the
paged-attn path.
Region: ggml/src/ggml-cpu/ops.cpp (paged attn), ggml/src/ggml-metal/ggml-metal.metal,
        src/llama-kv-cache.cpp (type acceptance), possibly a new ggml type.
Risk: kernel complexity; quality at 2 bits needs the asymmetric scheme to be
      worth it. Correctness gate MUST include a perplexity or logit-diff check
      strong enough to catch 2-bit quality collapse.

## Mechanics
- Single gene (S4), one worktree off 45fec509.
- Budget: 6 gens OR 50 min OR 15 candidates. stagnation_limit=4.
- ASCII only. Commits on evolve/memopt-pipeline-w3/* only. Never master/main.
- Never weaken a test. 3-run median, wide tie band.

## Output
- review branch evolve-review/memopt-pipeline-w3 off 45fec509.
- If S4 wins, it should stack with S1 (already in baseline). Re-verify s2
  (LLAMA_KV_LAZY_CLEAR) against the S4 tree - if it still regresses, leave it
  off by default and document.
- Final message: S4 verdict, final peak RSS vs 8.30 GB, correctness, bugs.

Honest expectation: 2-bit KV at 12B/512-context may show a real RSS win (KV is
~larger fraction now), but quality risk is higher than q8_0. A clean q8_0
baseline (S1) that we already have may be the practical sweet spot. If S4
can't hold quality, ship "partial" and document - that's a useful result.
