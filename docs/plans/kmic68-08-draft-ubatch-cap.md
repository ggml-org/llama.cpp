# P08 - Draft-context ubatch cap

**Kind:** port
**Depends on:** none

## Purpose

At long context the draft context's compute buffer is dominated by the KQ mask,
whose size is `n_kv * n_ubatch * sizeof(f16)`. The target needs a wide ubatch
because that is what buys prefill throughput, but the draft is a single layer that
simply loops over more chunks, so its ubatch can be capped independently.

Kmic-68 quote their worst case as 1024 MiB at `262144 x 2048`. Our depths are
smaller, but the mechanism is the same, and it is the same buffer whose 11 MiB
growth PR #90 T24 spent a day accounting for. This plan reduces it by design
rather than measuring it after the fact.

## Source

- Kmic-68 `common/speculative.cpp`, the `n_ubatch` clamp in
  `common_base_params_to_speculative`.

## In this fork

- `common/common.h:497`, `n_ubatch` on the common params; the speculative params
  struct has no per-draft equivalent today.
- `src/llama-context.cpp:1035-1053`, `sched_reserve`, which now reserves the
  draft catch-up shape after `15f318275`.

## Requirements

- **R08.1** (optional feature) WHERE a speculative draft ubatch is configured, the <params conversion> shall cap the draft context's `n_ubatch` at that value independently of the target's.
- **R08.2** (ubiquitous) The <draft context> shall have `n_batch` at least equal to its `n_ubatch`.
- **R08.3** (event-driven) WHEN the draft ubatch is capped, the <draft context> shall decode the same draft tokens to the same result, chunked over more `llama_decode` calls rather than fewer, larger ones.
- **R08.4** (unwanted) IF the configured draft ubatch exceeds the target's, THEN the <params conversion> shall leave the target's value unchanged.
- **R08.4a** (ubiquitous) The <effective draft ubatch> shall be at least `n_rs_seq + 2`, because `llama-context.cpp:390-402` raises `n_batch` and `n_ubatch` to that floor with only a warning, so a cap below it is silently undone at context creation.
- **R08.4b** (unwanted) IF the fit path reports a draft compute buffer, THEN it shall use the same normalised value as the runtime context, because `common.cpp:1308` sets the fit context's `n_rs_seq` to zero and would otherwise report a buffer smaller than the runtime reserves.
- **R08.5** (event-driven) WHEN the draft ubatch cap is active, the <fit path> shall report the reduced draft compute buffer.
- **R08.6** (unwanted) IF the cap would force a draft microbatch below 32, THEN the <params conversion> shall clamp at 32, since that is the BLAS floor the CPU backend needs.

## Acceptance

Same draft tokens and same acceptance rate with and without the cap, from the P04
log. A770, named driver, P01's `-n 512`.

The run must go through an executable that actually constructs `ctx_dft`.
`tools/llama-bench` has no draft-model or speculative path, so a depth sweep with
it creates only the target context: it cannot exercise the cap, compare draft
tokens, or report a draft compute buffer, and would return identical numbers
whether or not P08 works. Use the server speculative harness, and record the
draft context's compute buffer from the fit breakdown rather than from `llama-bench`.
