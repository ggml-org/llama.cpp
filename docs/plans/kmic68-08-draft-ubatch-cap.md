# P08 - Draft-context ubatch cap

**Kind:** common-layer feature
**Depends on:** none
**Precedes:** P12

## Context

A draft context does not need the target context's prefill-oriented microbatch
width. At long context, inheriting the target `n_ubatch` can make the draft
compute buffer, especially its attention-mask dimension, unnecessarily large.
P08 adds one independent draft cap while retaining the target settings and the
existing draft decode/chunking machinery.

The user-facing field is `common_params_speculative_draft::n_ubatch`.
`--spec-draft-ubatch-size`, `-ubd`, and `--ubatch-size-draft` map to it;
`LLAMA_ARG_SPEC_DRAFT_UBATCH_SIZE` is the environment alias. Unset or `0`
means inherit the live target context. Positive values are strict maxima capped
at the live target width and must already satisfy the repository/draft-architecture
minimums; invalid or too-small values fail. The default remains zero.

Source provenance: `Kmic-68/llama.cpp` branch `p100-optimizations`, the
draft `n_ubatch` field, argument aliases, and clamp in
`common_base_params_to_speculative`.

## EARS requirements

- **R08.1 (Optional feature):** WHERE independent draft microbatch sizing is included, `common_params_speculative_draft` shall expose `n_ubatch` with `0` as the inherit value.
- **R08.2 (Ubiquitous):** The post-target speculative resolver shall keep draft `n_batch` at least as large as both live target `n_batch` and draft `n_ubatch`.
- **R08.3 (Event-driven):** WHEN a valid positive draft ubatch cap is configured, `common_resolve_speculative_batch_params` shall set draft `n_ubatch` to `min(llama_n_ubatch(ctx_tgt),configured_cap)`.
- **R08.4 (Ubiquitous):** The post-target speculative resolver shall leave the live target context and caller's target parameters unchanged.
- **R08.5 (Event-driven):** WHEN the draft ubatch cap is active, the draft fit diagnostics shall report provisional fit dimensions, authoritative live-target dimensions, recurrent architecture hard minimum, effective draft `n_ubatch`, reconciliation outcome, and reserved draft compute-buffer size.
- **R08.6 (Unwanted behaviour):** IF an explicit draft cap is below `max(32,recurrent_hard_min)`, THEN the post-target speculative resolver shall reject the configuration instead of raising past the cap.
- **R08.7 (State-driven):** WHILE the draft ubatch value is unset or zero, `common_resolve_speculative_batch_params` shall inherit `llama_n_ubatch(ctx_tgt)`.
- **R08.8 (Unwanted behaviour):** IF a draft ubatch value is negative or not an integer, THEN the common argument parser shall fail with the option name and rejected value.
- **R08.9 (Event-driven):** WHEN draft work contains more rows than effective draft `n_ubatch`, including an MTP chain wider than an explicit cap, the speculative implementations shall route those rows through their existing chunked or sequential decode paths.
- **R08.10 (Ubiquitous):** The post-target speculative resolver shall read batch dimensions from `llama_n_batch(ctx_tgt)` and `llama_n_ubatch(ctx_tgt)` after target normalization.
- **R08.11 (Unwanted behaviour):** IF the live target width is below `max(32,recurrent_hard_min)`, THEN the post-target speculative resolver shall reject the configuration.
- **R08.12 (Event-driven):** WHEN authoritative effective draft dimensions exceed the provisional dimensions supplied to the joint fit, initialization shall rerun the fit and joint-placement validation or reject the configuration before creating the draft context.

## Approach

1. Add `int32_t n_ubatch=0` plus the three CLI aliases/environment name to
   `common_params_speculative_draft`; document zero/inherit and minimum 32.
2. Before `ctx_tgt` exists, derive provisional target dimensions from the
   pre-target `cparams` passed to `common_fit_params`. Apply the requested cap
   to produce provisional `draft_n_ubatch` and
   `draft_n_batch=max(provisional_target_n_batch,draft_n_ubatch)` in
   `params_dft`/`cparams_dft`, so the extra-model fit measures the capped
   draft shapes. Label these values as estimates and retain them for comparison.
3. Keep the pre-context joint target+draft device-memory fit at
   `common/common.cpp:1297-1317`, using those provisional capped draft
   dimensions. Resolve `draft_min_ubatch=max(32,recurrent_hard_min)`; MTP
   `chain_required_rows` is deliberately not part of this hard minimum.
4. Create/load the target context. Before draft-context creation, call
   `common_resolve_speculative_batch_params(params,ctx_tgt,draft_kind)` and read
   `target_batch=llama_n_batch(ctx_tgt)` plus
   `target_ubatch=llama_n_ubatch(ctx_tgt)`. For zero, inherit `target_ubatch`.
   For an explicit cap require `cap>=draft_min_ubatch`, set
   `draft_ubatch=min(target_ubatch,cap)`, reject if the result is below the hard
   minimum, and set `draft_n_batch=max(target_batch,draft_ubatch)`. Never mutate
   the live target context or silently raise a cap.
5. Compare the authoritative effective draft dimensions with the provisional
   values supplied to the fit. Any authoritative `draft_n_batch` or
   `draft_n_ubatch` above fitted capacity requires a new fit and joint-placement
   validation or clean rejection before draft-context creation; continuing with
   an undersized fit is forbidden. Revalidate any other difference as well. If
   placement must change, use the supported fit/reload sequence; if that cannot
   complete, fail before creating the draft context. The
   `common_base_params_to_speculative` helper remains responsible only for
   model/device/common copying, and `spec_check_draft_ctx` validates the final
   draft context against live target values.
6. Reuse current wider-work paths: EAGLE3 encoder chunking; DFlash/DSpark
   injection chunking; MTP chain-fit with sequential fallback; and existing
   `llama_process` splitting. An explicit cap below MTP chain width is valid and
   selects sequential fallback; add no cap-specific scheduler.
7. Re-run authoritative resolution and fit reconciliation after target reload or
   sleep wake before creating a new draft context. Diagnostics print requested cap,
   provisional fit dimensions, live target dimensions, recurrent hard minimum,
   effective draft values, whether reconciliation reran, and buffer MiB.
8. Add parser/resolver tests for provisional target normalization differing from
   the live context, target `n_ubatch>n_batch` normalization, recurrent hard
   minima, zero, cap 32/64/equal/above target, explicit cap 1/31 rejection, valid
   cap 32 below MTP chain width, and invalid target/draft dimensions.
9. Add `tests/test-spec-draft-ubatch.cpp` with generated minimal fixtures for
   EAGLE3, DFlash, DSpark, and MTP. Each fixture forces rows above cap 32/64,
   verifies the named existing chunk/sequential path, and compares token IDs and
   accepted counts with inherit mode. The MTP fixture must use a chain wider than
   cap 32 and prove that sequential fallback succeeds rather than startup failing.
10. Add planned `scripts/perf/fixtures/draft-ubatch.json` naming real A770
   target/draft GGUFs and exact `--spec-type` arguments for each family. The
   depth/performance campaign uses the measured production MTP fixture; the
   other families remain mandatory correctness/capacity fixtures.
11. Update speculative/CLI/server option documentation.

## Critical files & anchors

- `common/common.h:328-360` - `common_params_speculative_draft`.
- `common/common.h:496-497` - target `n_batch` and `n_ubatch`; minimum 32 comments.
- `common/arg.cpp:4489-4495` - nearby speculative-draft option pattern.
- `common/speculative.cpp:153-168` - draft/target context compatibility checks.
- `common/speculative.cpp:3260-3322` - `common_base_params_to_speculative`.
- `common/common.cpp:1297-1317,1425` - provisional joint-fit dimensions before live target creation and authoritative reconciliation afterward.
- `common/common.cpp:1770-1786` - common-to-context batch conversion.
- `common/speculative.cpp:781-805` - EAGLE3 encoder chunking.
- `common/speculative.cpp:1335-1405` - DFlash/DSpark injection chunking.
- `common/speculative.cpp:1636-1639,2125-2128` - MTP capacity and chain fallback.
- `src/llama-context.cpp:1035-1053,1079-1084` - draft reservation and compute-buffer log.
- `tests/test-arg-parser.cpp`, `tests/test-qwen4exp-mtp.cpp`, and planned `tests/test-spec-draft-ubatch.cpp` - conversion and four-family behavior.
- `scripts/perf/fixtures/draft-ubatch.json` - planned concrete model/argument fixture manifest.
- `scripts/perf/verify-draft-ubatch.py` - planned token, acceptance, reservation, and depth comparison.

## Verification

Run focused conversion and all-family fixtures:

```bash
timeout 240 ctest --test-dir build-sycl -R 'test-arg-parser|test-qwen4exp-mtp|test-spec-draft-ubatch' --output-on-failure
```

On A770, use the fixture manifest and the same build/driver/boot/seed. The MTP
production fixture uses the P01 depth set and `n_predict=512`; plain
`llama-bench` is not feature evidence because it creates no draft context.

```bash
timeout 2400 python3 scripts/perf/verify-draft-ubatch.py --server ./build-sycl/bin/llama-server --fixtures scripts/perf/fixtures/draft-ubatch.json --production-fixture qwen4exp-mtp --prompts scripts/perf/prompts.jsonl --depths 0,4096,16384 --n-predict 512 --caps 0,32,64 --seed 123 --paired-launches 4 --fit-print --report /tmp/p08-draft-ubatch.json
```

Expected evidence:

- diagnostics show provisional fit dimensions, authoritative live target
  dimensions, unchanged target state, resolved recurrent hard minimum, inherited
  cap-0 width, and cap-32/cap-64 widths;
- tests prove provisional parameter-derived dimensions are replaced by live target
  dimensions after target creation, authoritative values above fitted capacity
  force re-fit or rejection, and coverage includes normalization changes,
  recurrent hard minima, wider MTP chains with cap 32, and impossible targets;
- cap-64 reduces the production draft buffer at long depth with raw MiB retained;
- EAGLE3, DFlash, DSpark, and MTP fixtures each force rows beyond the cap, prove
  their named existing path, and match inherit-mode tokens/accepted counts;
- the MTP ABBA report shows no statistically detected regression;
- explicit caps 1/31 and any cap below a recurrent architecture hard minimum fail
  rather than being silently raised; a cap of 32 below an MTP chain width remains
  valid and uses sequential fallback, above-target caps resolve to target width,
  and zero inherits;
- the post-run two-driver fault gate passes and the service restarts.

## Assumptions & contingencies

- Draft `n_batch` remains at least live target `n_batch`; P08 reduces only
  draft microbatch width and derived shapes.
- Minimum 32 and recurrent architecture hard minima are validation constraints,
  not values an explicit cap may be silently raised past. MTP chain width is not
  a hard minimum because the existing sequential fallback handles a wider chain.
- Every draft family must pass its concrete cap-crossing fixture.
- P12 consumes the post-target effective draft `n_ubatch`.
