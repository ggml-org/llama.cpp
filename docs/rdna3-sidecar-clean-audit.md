# `feat/qwen38-sidecar-clean` integration audit

This note records the selective integration into
`feat/unified-rdna2-rdna3`. The two branches are not a linear continuation:
the sidecar-clean branch contains a newer upstream merge and a different
Qwen4Exp experiment, so it must not be merged wholesale.

## Active RDNA3 path

The verified Qwen3.8 MTP launcher remains the active path. Its dry run selects
both identity-matched gfx1100 devices and emits:

```text
--device ROCm0,ROCm1 --split-mode tensor --tensor-split 1,1
--spec-type draft-mtp,ngram-map-k4v --device-draft ROCm0
```

The current MTP sidecar is already ahead of sidecar-clean's
`502224204` MTP top-k commit: `0a723d2e5` uses capability-gated rocPRIM
runtime selection with a fallback and was previously validated on gfx1100 and
gfx1030. Cherry-picking `502224204` would replace that implementation with an
older, gfx1030-only form and was intentionally rejected.

The unified build script already enables MTP/DFlash sidecars by default,
passes the exact selected HIP architecture, and keeps `GGML_HIP_RCCL=ON`.
Therefore the older sidecar-clean build-script commit `565396668` is not
needed.

## Integrated

- `996796b91` — adapted the validated behavior of sidecar-clean
  `2bad4731f`: permit target TP output-head sharding only for a validated
  sidecar-only DFlash/DSpark candidate, use the target vocabulary mask token
  when no host draft exists, account for unconstrained auxiliary layers, and
  fail closed if a promised sidecar preflight later fails.
- `9122100d5` — adapted sidecar-clean `0b54c1f69`: DFlash committed KV
  storage grows geometrically from 16K rows to the required position up to a
  131072 hard ceiling, invalidates the sequence graph after relocation,
  releases grown storage on reset, and returns an empty draft at the ceiling.
  This is compile-qualified only because no DFlash artifact bundle is present
  on the host.
- `60f003e36` — documents the unified sidecar behavior and safety contract.

## Intentionally not integrated

- `f05d6d8ab`, `921a73c38`, and `3456f8628`: DFlash gfx1030-specific kernel
  optimizations; useful for the unavailable physical gfx1030 target, but not
  RDNA3 changes.
- `56bcb1cf7`, `a4d7cb256`, `b3252ffbd`, `c44f55ecc`, `60b1d023b`,
  `661d16ae6`, and `9a8d34fb5`: Qwen4Exp/Flash-Next-specific provider,
  device-view, or TP4 reduction work. The current Qwen35 MTP device-view path
  intentionally remains host fallback until architecture-specific correctness
  is demonstrated.
- Revert commits in sidecar-clean are not imported; no dormant Qwen4Exp
  provider is enabled in the unified branch.

## Verification

From the remote repository at `/home/edwin/llama.cpp-rdna2`, both unified
`llama-server`, MTP, and DFlash sidecar targets built successfully for gfx1100
and gfx1030. The focused host suite passed 4/4:

- `test-spec-sidecar-artifact`
- `test-speculative-sidecar-cap`
- `test-rdna3-auto-policy`
- `test-speculative-backend-policy`

The final launcher dry run selected native gfx1100 `ROCm0,ROCm1`, tensor split
`1,1`, RCCL-linked HIP, and the fixed-width MTP sidecar. No DFlash runtime
long-context claim is made without a matching prepared artifact bundle.

## RocPRIM A/B result

For a fair native-RDNA3 comparison, the old `502224204` source was compiled
with its rocPRIM path explicitly enabled on gfx1100; its original
`SPEC_SIDECAR_GFX1030` guard would otherwise use the fallback on gfx1100. Both
variants also compiled successfully for gfx1030.

- A 12-run HIP top-k microbenchmark (`N=40960`, `K=32`, alternating order,
  5000/10000 iterations) measured `444.470 us/call` for `0a723d2e5` and
  `444.685 us/call` for `502224204`: a `0.048%` advantage for current, well
  inside run-to-run noise. Selection outputs matched.
- The actual MTP sidecar ABI was exercised directly on gfx1100: two no-debug
  500-call runs per variant, three stochastic proposals per call. The means
  were `3.529171 ms/call` current and `3.528476 ms/call` old, a `0.020%`
  advantage for old. Proposal IDs and checksums were identical.

Thus neither rocPRIM implementation has a meaningful performance advantage in
this test. Keep `0a723d2e5`: it is effectively tied, activates rocPRIM on both
architectures when available, and safely falls back when the API or scratch
allocation is unavailable. A full target-model server A/B was not completed
because startup encountered the known SVM resident-memory failure before the
sidecar was initialized; no end-to-end speed claim is based on that attempt.
Evidence is under `rocprim-variant-topk-microbench-20260831T141203Z` and
`rocprim-variant-mtp-direct-20260831T141621Z`.
