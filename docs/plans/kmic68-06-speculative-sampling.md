# P06 - Distribution-recording speculative acceptance

**Kind:** common-layer feature
**Depends on:** P04
**Supplies:** recorded distributions and per-token fallback for P07

## Context

The current acceptor samples the target and accepts a drafted token only when
the token IDs match. That preserves the target distribution but wastes the
draft probability `q(x)`. For a stateless stochastic drafter, exact rejection
sampling accepts with `min(1, p(x)/q(x))`; after rejection it samples from the
normalized residual `max(p-q, 0)`. The output distribution remains the target
distribution `p` while acceptance can increase.

P06 is opt-in and limited to the non-chained MTP host path. The runtime contract
is exact: `LLAMA_SPEC_SAMPLE_TEMP` unset or `0` disables the feature; a
finite value greater than zero enables draft sampling. `LLAMA_SPEC_DRAFT_TOPP`
is optional, defaults to `1.0`, and must be finite in `(0, 1]`. The draft
uses existing `LLAMA_DRAFT_TOP_K=10`. Backend draft sampling must be disabled
because a backend-selected token has no matching host-side `q`.

Source provenance: `Kmic-68/llama.cpp` branch `p100-optimizations`,
`common_sampler_sample_and_accept_n_dist` plus the draft-side `dists`,
`sample_temp`, and `sample_top_p` data path.

## EARS requirements

- **R06.1 (Optional feature):** WHERE distribution-recording speculative sampling is included, `common_speculative_draft_params` shall carry one token-ID-keyed normalized draft distribution `q` aligned with each emitted draft token.
- **R06.2 (State-driven):** WHILE `LLAMA_SPEC_SAMPLE_TEMP` is finite and greater than zero and the MTP sampler chain is stateless, the non-chained MTP drafter shall produce each draft token together with the exact distribution `q` from which it was sampled.
- **R06.3 (State-driven):** WHILE `0 < LLAMA_SPEC_DRAFT_TOPP < 1`, the non-chained MTP drafter shall truncate `q` to the smallest descending-probability prefix whose cumulative mass reaches the configured value.
- **R06.4 (Event-driven):** WHEN a draft token with valid `q` is verified against target distribution `p`, `common_sampler_sample_and_accept_n_dist` shall accept it with probability `min(1, p(x)/q(x))`.
- **R06.5 (Event-driven):** WHEN distribution-recording acceptance rejects a draft token, `common_sampler_sample_and_accept_n_dist` shall select the correction token from normalized `max(p-q, 0)`.
- **R06.6 (Ubiquitous):** The distribution-recording acceptor shall preserve the target output distribution `p`.
- **R06.7 (State-driven):** WHILE `LLAMA_SPEC_SAMPLE_TEMP` is unset or zero, speculative acceptance shall use the existing exact-match path.
- **R06.8 (State-driven):** WHILE grammar, penalties, DRY, mirostat, a reasoning budget, or any unsupported stateful sampler is active, speculative acceptance shall use the existing exact-match path.
- **R06.9 (State-driven):** WHILE a draft or target token is backend-selected or a chained in-graph argmax token is in use, speculative acceptance shall use the existing exact-match path.
- **R06.10 (Unwanted behaviour):** IF any block distribution is missing, misaligned, non-normalized, non-finite, or assigns zero mass to its emitted token, THEN the speculative acceptor shall use the existing exact-match path for the entire untouched block.
- **R06.11 (Unwanted behaviour):** IF either distribution-sampling environment value is outside its accepted finite range, THEN the speculative initializer shall fail with the variable name and rejected value.
- **R06.12 (Ubiquitous):** The `common_sampler` clone, copy, and reset operations shall maintain deterministic distribution-acceptance RNG state.
- **R06.13 (Event-driven):** WHEN all `G` draft tokens are accepted, the distribution acceptor shall append one terminal target token sampled from `p_G`.
- **R06.14 (State-driven):** WHILE P04 cycle logging and P06 are enabled, the cycle artifact shall serialize the proposal identity, drafted token IDs, and every sparse token/probability pair in `q`.
- **R06.15 (State-driven):** WHILE a proposal is consumed, invalidated, or replaced by checkpoint replay, speculative acceptance shall use the existing exact-match path.
- **R06.16 (Event-driven):** WHEN distribution-recording acceptance begins, the acceptor shall prevalidate every proposal distribution and target row before mutating the live sampler or RNG.

## Approach

1. Parse both environment variables once during speculative initialization.
   Unset top-p becomes 1.0; malformed, non-finite, negative temperature, or top-p
   outside `(0,1]` is a startup error.
2. Extend `common_speculative_draft_params` with a monotonic single-use
   `proposal_id` and `dists` parallel to `result`. Each distribution is a
   sparse token-ID-keyed normalized vector. Final MTP clear/clamp/selection in
   `common_speculative_draft` updates result and distributions in lockstep and
   stamps the immutable P04 proposal snapshot.
3. On the non-chained MTP CPU path, construct the draft sampler from top-k 10,
   configured temperature/top-p, and distribution selection. Call
   `common_sampler_sample`, copy `common_sampler_get_candidates` immediately
   afterward and before accept, and record the emitted token with that exact
   normalized `q` under the same proposal ID.
4. Permit only the supported stateless chain. Grammar, penalties, DRY, mirostat,
   reasoning budget, unknown sampler, backend-selected token, or chain mode
   makes the whole proposal ineligible. Log one fallback reason per proposal.
5. Before touching the live sampler or acceptance RNG, preflight the entire
   block: proposal ID matches the slot and is unconsumed; replay is false;
   `result.size()==dists.size()`; every sparse distribution is finite,
   normalized within the declared tolerance, token-ID unique, and gives positive
   mass to its emitted token; and `llama_get_sampled_token_ith` is null for
   every target index. Any failure calls the old exact-match overload on the
   original untouched state.
6. Clone `common_sampler` and its domain-separated acceptance RNG after
   preflight. Build all required target distributions on the working clone. If a
   later validation fails, discard the clone and run exact-match on the pristine
   original. Only a fully successful distribution decision copies final sampler
   and RNG state back atomically.
7. For each draft position, read normalized target `p` from the working clone,
   treat absent sparse-`q` IDs as zero, draw one acceptance uniform, and commit
   an accepted token on the clone. On rejection, build the residual over the
   union of supports, clamp negative round-off, normalize, sample/commit one
   correction, and stop. If clipped residual mass is zero, sample from `p`.
8. Preserve the existing `G+1` return contract. If all `G` drafts survive,
   sample, accept, and append the terminal token at `idxs[G]` from `p_G`.
   Server and speculative-simple consumers must receive `G+1`, not mistake full
   acceptance for replay. This includes a final accepted draft token that is
   end-of-generation (EOG): the acceptor still returns the bonus, but consumers
   stop at the first EOG under the active stopping policy and discard the rest
   of the returned vector. They must not emit or decode the post-EOG bonus;
   terminate/release the sequence through the existing stop path. An explicit
   ignore-EOG policy retains its existing behavior.
9. Mark a proposal consumed after one acceptance attempt. When replay replaces
   `slot.spec_draft` with an accepted prefix plus correction, clear/invalidate
   its distributions even if vector length happens to match. Replayed rounds
   always use exact-match until a fresh proposal ID and distributions are made.
10. Extend the P04 selected cycle row with versioned `q_records`: proposal ID,
   draft index/token, sparse token/probability pairs, probability sum, and schema
   version. The verifier rejects missing, duplicate, corrupt, or cross-proposal
   records. Baseline rows explicitly state `q_records=null`.
11. Add seeded corpus and server tests for full acceptance/terminal bonus,
   full acceptance with EOG at the final and an earlier draft position,
   rejection, invalid final-row distribution, target backend token at any index,
   late validation failure with pristine fallback, consumed proposal reuse, same-
   length replay replacement, server, and speculative-simple.
12. Feed enabled/disabled A770 runs into the P04 schema and report acceptance per
   drafted token at temperature 0.8; distribution parity is established by the
   confidence-bounded corpus, not token identity.

## Critical files & anchors

- `common/speculative.h:72-91` - `common_speculative_draft_params` result contract.
- `common/speculative.cpp:2313-2415` - non-chained MTP host sampling loop.
- `common/speculative.cpp:3639-3690` - shared draft wrapper and `n_max` result clamping.
- `common/sampling.cpp:594-675` - target CPU sampling and candidate distribution.
- `common/sampling.cpp:678-714` - existing exact-match acceptor.
- `common/sampling.cpp:509-539` - sampler clone/copy; extend with acceptance RNG state.
- `common/sampling.h:85-89` - current acceptor declarations.
- `tools/server/server-context.cpp:3414-3416,4302-4377` - proposal handoff, replay replacement, acceptor selection, and final stats.
- `examples/speculative-simple/speculative-simple.cpp:193-200,269-270` - non-server `G+1` consumer.
- `tools/server/server-context.cpp:4388-4407` and `examples/speculative-simple/speculative-simple.cpp:317-325` - consumer stop handling must discard the post-EOG suffix.
- `tests/test-sampling.cpp` - seeded confidence-bounded distribution corpus.
- `scripts/perf/verify-spec-distribution.py` - planned P04/`q_records` A770 comparison.

## Verification

Add a fixed-seed corpus covering vocabularies 2 through 64, sparse/dense `p/q`,
`p=q`, disjoint support, near-zero residual mass, full `G+1` acceptance, and
multi-token prefixes. Draw `N=100000` outputs per enabled and disabled case.
For each case, fail immediately if any token with target probability zero is
emitted. Deterministically pool the lowest-probability bins into one `other`
bin until every Pearson expected count is at least 5. Run Pearson multinomial
goodness-of-fit tests for enabled-versus-`p` and disabled-versus-`p`, plus a
Pearson two-sample homogeneity test for enabled versus disabled. Apply
Holm-Bonferroni across every case/test at family-wise `alpha=0.01`. Report TV
and maximum error as diagnostics only, never fixed pass thresholds.

```bash
timeout 180 ctest --test-dir build-sycl -R '^test-sampling$' --output-on-failure
```

After P04 is complete, stop the service, verify A770 sole tenancy, name the
kernel driver/build/model, and compare the current exact-match path with P06 at
temperature 0.8:

The planned verifier launches fresh processes with explicit environments. Map
`--draft-temperature 0.8` to `LLAMA_SPEC_SAMPLE_TEMP=0.8` and
`--draft-top-p 1.0` to `LLAMA_SPEC_DRAFT_TOPP=1.0` only in the enabled arm.
Unset both variables in the baseline, regardless of the parent environment.
Set `LLAMA_DRAFT_TOP_K=10` in both arms, pass `--target-temperature` to the
request's target sampling parameters, and keep backend draft sampling disabled.
Enable P04 logging separately for both arms. Record the resolved settings and
require an eligible enabled proposal with non-null `q_records` plus baseline
exact-match selection with `q_records=null`; CLI arguments alone are not proof
that distribution recording ran.

```bash
timeout 1800 python3 scripts/perf/verify-spec-distribution.py --server ./build-sycl/bin/llama-server --model target-qwen4exp.gguf --draft-model qwen4exp-mtp.gguf --prompts scripts/perf/prompts.jsonl --repeats 2 --seed 123 --target-temperature 0.8 --draft-temperature 0.8 --draft-top-p 1.0 --spec-args='--spec-type draft-mtp --no-spec-draft-backend-sampling' --baseline-log /tmp/p06-baseline.jsonl --enabled-log /tmp/p06-enabled.jsonl --q-artifact /tmp/p06-q-records.jsonl --report /tmp/p06-report.json
```

Expected evidence:

- every enabled/disabled corpus case passes the predeclared Pearson tests after
  Holm-Bonferroni at family-wise `alpha=0.01`, with no zero-probability output;
- every selected P04 proposal has one valid cycle-keyed `q_records` entry per
  drafted token with matching proposal/token identity, finite normalized sparse
  support, and positive sampled-token mass;
- full acceptance returns exactly `G+1` tokens to both server and
  speculative-simple consumers;
- EOG fixtures retain the acceptor's `G+1` contract while both consumers emit
  nothing and schedule no decode after the first effective EOG, including the
  bonus; the existing ignore-EOG policy is tested separately;
- invalid or late target/distribution checks leave original sampler/RNG state
  untouched and restart the old whole-block exact-match path;
- consumed and replay-replaced proposals cannot reuse stale distributions,
  including same-length correction replacement;
- baseline and enabled P04 logs report accepted/drafted totals for the same six
  requests and the report records their acceptance ratio;
- subprocess-environment fixtures prove enabled variable mapping and baseline
  unsetting even under conflicting inherited values; runtime evidence confirms
  enabled `q_records` and baseline exact-match rather than two disabled arms;
- every disabled/fallback condition reaches exact-match, invalid environment
  values name the variable, and the post-run GPU fault gate passes.

Run the exact README regression floor before and after the feature.

## Assumptions & contingencies

- P06 stores sparse distributions bounded by draft top-k 10; it does not retain
  full-vocabulary `q` arrays.
- The feature is disabled by default. No target sampler behavior changes until a
  positive `LLAMA_SPEC_SAMPLE_TEMP` is supplied and every whole-block check
  passes.
- Proposal distributions are single-use evidence bound to immutable proposal
  identity; size equality alone is never provenance.
- Empirical parity is decided by the predeclared family-wise Pearson/Holm test;
  TV and maximum error are diagnostics, not acceptance thresholds.
- P07 may consume only unconsumed P06 proposals with complete `p/q` records;
  every P06 fallback remains a P07 fallback.
