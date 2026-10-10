# P05 - Guarded CPU sampler top-k prefilter

**Kind:** common-layer feature
**Depends on:** none

## Context

The current CPU sampler materializes one `llama_token_data` entry for every
vocabulary token before a leading top-k sampler immediately discards all but
`k`. Kmic-68 avoids that work by selecting at most 128 logits directly from
the raw array. This plan ports only that bounded prefilter and retains the
existing full-vocabulary path for every ineligible call.

The optimization applies to CPU sampling. It covers draft-simple and the CPU
fallback or explicit `--no-spec-draft-backend-sampling` paths for EAGLE3,
DFlash/DSpark, and non-chained MTP. Backend-selected tokens and chained in-graph
argmax do not consume this path. The environment contract is exact:
`LLAMA_SAMPLER_PREFILTER` unset or `1` permits guarded use, `0` disables
it, and any other value warns once and disables it.

Source provenance: `Kmic-68/llama.cpp` branch `p100-optimizations`,
`common/sampling.cpp` additions `top_k_from_logits`, `prefilter_k`, and
`can_prefilter`.

## EARS requirements

- **R05.1 (Event-driven):** WHEN `common_sampler_init` finds that the first effective sampler is TOP_K, `0 < top_k <= 128`, merged user/model logit bias is empty, mirostat is disabled, and prefilter configuration permits the feature, the `common_sampler` shall initialize `prefilter_k` to `top_k`.
- **R05.2 (Event-driven):** WHEN two candidate logits are equal, the CPU top-k prefilter and full-vocabulary top-k path shall order the lower token ID first.
- **R05.3 (Unwanted behaviour):** IF TOP_K is not the first effective sampler, `top_k` is outside 1 through 128, merged logit bias is nonempty, mirostat is enabled, or `grammar_first` is true, THEN the `common_sampler_sample` function shall use the existing full-vocabulary path.
- **R05.4 (Unwanted behaviour):** IF `common_reasoning_budget_get_state` returns `REASONING_BUDGET_FORCING`, THEN the `common_sampler_sample` function shall use the existing full-vocabulary path.
- **R05.5 (State-driven):** WHILE `LLAMA_SAMPLER_PREFILTER` is `0` or invalid, `common_sampler_sample` shall use the existing full-vocabulary path.
- **R05.6 (Event-driven):** WHEN grammar rejection triggers a second sampling pass, `common_sampler_sample` shall rematerialize the full vocabulary before applying grammar.
- **R05.7 (Ubiquitous):** The `common_sampler_clone` and `common_sampler_copy` operations shall preserve `prefilter_k`.
- **R05.8 (State-driven):** WHILE `llama_get_sampled_token_ith` returns a backend-selected token, `common_sampler_sample` shall materialize the full candidate array and return that selected token without using the prefilter.
- **R05.9 (State-driven):** WHILE no backend token exists and every static/current-call guard passes, `common_sampler::set_logits` shall populate `cur_p` with only the highest `prefilter_k` raw logits before the sampler chain runs.
- **R05.10 (Unwanted behaviour):** IF any raw vocabulary logit is non-finite, THEN the `common_sampler::set_logits` method shall abandon the prefilter and materialize the existing full-vocabulary candidate array.
- **R05.11 (State-driven):** WHILE `LLAMA_SAMPLER_PREFILTER` is `1`, the prefilter shall remain subject to every static and per-call eligibility guard.
- **R05.12 (Event-driven):** WHEN CPU draft sampling accumulates host time, the server metrics endpoint shall expose its monotonic microsecond total.

## Approach

1. Add `prefilter_k` to `common_sampler`, initialized to zero. Compute only
   static eligibility in `common_sampler_init` after merging user bias with
   model suppress-token bias.
2. Reuse the exact constructor no-op predicates: PENALTIES is ineffective when
   `penalty_last_n==0` or all three rates are neutral; DRY is ineffective when
   multiplier is zero, base is below 1, or history is zero; TOP_N_SIGMA is
   ineffective when its value is at most zero. Factor/reuse these predicates so
   initialization cannot drift from the constructors. Any other effective entry
   before TOP_K disables `prefilter_k`.
3. Parse `LLAMA_SAMPLER_PREFILTER` once. Unset or `1` permits guarded use,
   `0` disables it, and another value warns once and disables. `1` is not a
   force-through switch and cannot override bias, grammar, reasoning, mirostat,
   range, candidate-source, or non-finite guards.
4. Add `top_k_from_logits` next to `common_sampler::set_logits`: one bounded
   heap of at most 128 entries, one raw-logit pass, no vocabulary-sized vector,
   comparison by logit descending then token ID ascending. Detect NaN and both
   infinities before heap comparison; on any non-finite value discard the heap
   and execute the existing full materialization for that call.
5. After context synchronization, query `llama_get_sampled_token_ith(ctx,idx)`
   before deciding `allow_prefilter`. If non-null, call the existing full
   `set_logits(...,false)`, locate/mark that backend token in
   `cur_p.selected`, and return it. This preserves current candidate-array
   observability for `common_sampler_get_candidates` while recording zero
   prefilter calls. Only when the backend token is null compute
   `allow_prefilter` from static eligibility, raw-logit/first-pass state,
   `grammar_first==false`, and reasoning state not FORCING. Grammar resampling
   and sampled-probability/logit arrays always keep full/current handling.
6. Reuse one comparator in `src/llama-sampler.cpp` for both partial-sort
   helpers so the full path shares deterministic equal-logit ordering.
7. Copy `prefilter_k` in clone/copy and test every static/dynamic guard,
   constructor no-op boundary, clone/copy, equal logits, `k=1`, `k=128`,
   invalid `k`, and each non-finite value.
8. Expose a read-only `common_sampler::t_total_us` snapshot through
   `common/sampling.h`, aggregate the internal CPU draft samplers through
   `common_speculative`, and publish monotonic draft-only
   `spec_draft_sampler_us_total` server telemetry beside existing
   `spec_decode_num_drafts_total`. Add no timing branch inside the prefilter
   scan; reuse the draft sampler's existing timer and exclude target samplers.
9. The verifier reads matching before/after metric deltas. This makes draft-side
   host microseconds per speculative cycle executable without P04.

## Critical files & anchors

- `common/sampling.cpp:130-153` - `common_sampler::set_logits` materialization.
- `common/sampling.cpp:187-427` - `common_sampler_init`, merged biases, sampler-chain construction, and backend-sampling policy.
- `common/sampling.cpp:509-539` - clone and copy.
- `common/sampling.cpp:594-675` - first sample, backend early return, grammar pre-application, and full resample.
- `src/llama-sampler.cpp:135-213` - both partial-sort comparators.
- `common/common.h:231-255` - sampler defaults used to identify no-op entries.
- `src/llama-ext.h:124` - `LLAMA_DRAFT_TOP_K=10`.
- `common/arg.cpp:4489-4495` - CPU fallback control `--no-spec-draft-backend-sampling`.
- `tests/test-sampling.cpp` - seeded sampler, no-op-boundary, non-finite, and equal-logit cases.
- `tools/server/server-task.cpp:1560-1565` - existing speculative metrics and new sampler-time metric.
- `scripts/perf/bench_spec.py` - parser coverage for structured sampler telemetry.
- `scripts/perf/verify-sampler-prefilter.py` - planned oracle and host-time comparison.

## Verification

Build and run the focused sampler tests:

```bash
timeout 180 ctest --test-dir build-sycl -R '^test-sampling$' --output-on-failure
```

On the A770, stop the service, verify sole tenancy, name the driver/build/model,
and run the same six requests and seed through both environment states with
backend draft sampling disabled:

```bash
timeout 1800 python3 scripts/perf/verify-sampler-prefilter.py --server ./build-sycl/bin/llama-server --model target-qwen4exp.gguf --draft-model qwen4exp-mtp.gguf --prompts scripts/perf/prompts.jsonl --repeats 2 --repetitions 3 --seed 123 --spec-args='--spec-type draft-mtp --spec-draft-n-max 7 --no-spec-draft-backend-sampling' --k 10 --report /tmp/p05-prefilter.json
```

The verifier launches the same binary with `LLAMA_SAMPLER_PREFILTER=0` and
`=1`, preserves raw results, and computes host microseconds per speculative
cycle from matching `/metrics` deltas. It does not consume P04.

Expected evidence:

- forced-off and forced-on runs produce identical token IDs for every seeded
  oracle request in all three repetitions;
- equal-logit vocabularies select ascending token IDs for `k=1..128` in both
  full and prefilter paths;
- exact constructor boundary cases identify no-op PENALTIES, DRY, and
  TOP_N_SIGMA entries consistently;
- `grammar_first`, runtime `REASONING_BUDGET_FORCING`, merged bias,
  mirostat, invalid range/configuration, grammar resampling, sampled candidate
  arrays, NaN, positive infinity, and negative infinity all take the full path;
- `LLAMA_SAMPLER_PREFILTER=1` never overrides an ineligible guard;
- clone/copy retain the same `prefilter_k` and output;
- each launch reports monotonic `spec_draft_sampler_us_total` and
  `spec_decode_num_drafts_total`; their deltas produce median draft CPU
  sampling microseconds per cycle for both states;
- a backend-token fixture reports a full candidate array, the correct
  `cur_p.selected`, zero prefilter calls, and the unchanged returned token;
  chained paths also show no control-flow change, and the A770 run passes the
  post-run fault gate.

Run the exact README regression floor before and after the feature.

## Assumptions & contingencies

- The full path's new token-ID tie break is an intentional determinism fix
  required for forced-on/off parity. Record any changed equal-logit golden value.
- No SIMD-specific implementation is required initially. If the bounded portable
  scan does not lower median host time, P05 fails acceptance; do not ship the
  unset-means-enabled contract without revising and remeasuring the plan.
- P05 does not accelerate the default backend-sampling path. Its production
  benefit depends on a workload that reaches CPU sampling.
- Non-finite input always takes the explicit full-vocabulary fallback; heap
  ordering never receives a NaN or infinity.
- P05 timing telemetry is feature-local and monotonic; it introduces no P04
  dependency.
