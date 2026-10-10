# P10 - Cumulative-probability draft-width clamp

**Kind:** common-layer feature
**Depends on:** P04; P01 re-baseline evidence
**Independent of:** P09

## Context

Every host-side drafter already has a per-token `p_min` confidence stop. P10
adds a second, cumulative stop: multiply successive top-1 probabilities and stop
before the first token that takes the product below a threshold. This can avoid
low-value target verification rows at deep context without replacing existing
stops or adaptive width.

The eligible scope is host-loop EAGLE3, DFlash/DFlash2, and non-chained MTP.
Chained MTP and draft-simple have no supported host stop point. DSpark is
explicitly excluded: its confidence-head value is not a calibrated token
probability. DFlash/DFlash2 decode the noise block before host filtering, so P10
can reduce target verification rows there but cannot avoid that draft decode.

`LLAMA_SPEC_P_CUM` is the only runtime control. Unset disables the rule. A
finite non-negative value is the explicit threshold. Any finite negative value
selects the A770-measured depth ramp. NaN, infinity, and malformed values fail
initialization. The ramp uses the committed target position at cycle start and
is unavailable until the P04/P01 calibration report has been recorded.

Source provenance: `Kmic-68/llama.cpp` branch `p100-optimizations`, its
`p_cum`, `p_cum_min`, and cumulative stop beside the existing `p_min`
stop. P100 threshold values are not defaults for the A770.

## EARS requirements

- **R10.1 (State-driven):** WHILE `LLAMA_SPEC_P_CUM` is a finite non-negative value, each eligible host-loop drafter shall use that value as its cumulative top-1 probability threshold.
- **R10.2 (Ubiquitous):** The cumulative-probability rule shall operate in addition to every existing per-token confidence, capacity, and model-specific stop.
- **R10.3 (State-driven):** WHILE `LLAMA_SPEC_P_CUM` is unset, eligible drafters shall preserve their current stop behavior and batch shapes.
- **R10.4 (State-driven):** WHILE `LLAMA_SPEC_P_CUM` is finite and negative and non-chained MTP is active, the MTP drafter shall obtain its threshold from the recorded MTP A770 depth ramp at the cycle's committed target position.
- **R10.5 (Event-driven):** WHEN a non-first candidate makes the running top-1 probability product strictly less than the active threshold, the eligible drafter shall discard that candidate and end the draft round.
- **R10.6 (Unwanted behaviour):** IF a first candidate has passed every pre-existing stop, THEN the cumulative-probability rule shall preserve that candidate even when its product is below threshold.
- **R10.7 (State-driven):** WHILE chained MTP or draft-simple is active, cumulative-probability drafting shall preserve the existing path without a cumulative stop.
- **R10.8 (Event-driven):** WHEN an adaptive controller chooses a draft-width cap, the cumulative-probability rule shall only shorten the resulting draft.
- **R10.9 (Unwanted behaviour):** IF `LLAMA_SPEC_P_CUM` is empty, malformed, partially parsed, overflowed, or non-finite, THEN the speculative initializer shall fail with the rejected value.
- **R10.10 (State-driven):** WHILE DFlash or DFlash2 is active, the cumulative-probability rule shall apply only to the host-filtered returned token list.
- **R10.11 (State-driven):** WHILE a negative setting is used with EAGLE3, DFlash, or DFlash2, the cumulative-probability rule shall remain disabled for that drafter.
- **R10.12 (Ubiquitous):** The cumulative-probability extractor shall use the declared probability domain for each eligible drafter.
- **R10.13 (State-driven):** WHILE DSpark is active, cumulative-probability drafting shall preserve the existing path without a cumulative stop.

## Approach

1. Complete P01 and collect P04 synchronized rows for non-chained MTP only.
   `scripts/perf/calibrate-spec-pcum.py` replays those MTP top-1 sequences over
   a predeclared threshold grid and shortlists real A770 A/B candidates.
2. Store ordered `(position,threshold)` knots at 0, 4096, and 16384 keyed by
   `drafter=draft-mtp`, model identity, route, build, and driver. Clamp outside
   endpoints and interpolate between adjacent knots. A negative setting on any
   other drafter logs one ineligibility reason and applies no cumulative stop;
   it never reuses MTP calibration.
3. Parse the variable with complete string consumption and finite range checks.
   Unset disables. Finite non-negative values are explicit constants; any finite
   negative value selects the keyed MTP ramp. Signed `-0` compares as zero and
   is constant mode. Reject empty strings, whitespace/trailing text, overflow,
   underflow-to-nonfinite, NaN, and infinities.
4. At each eligible round start, set `p_cum=1`. For each candidate, first apply
   all existing model-specific confidence/`p_min` stops. If it survives, obtain
   the declared token probability, compute `p_next=p_cum*p_top1`, preserve the
   first surviving candidate, and for later candidates stop before pushing when
   `p_next<threshold`. Otherwise push and update the product.
5. Define probability domains explicitly:
   - EAGLE3 and non-chained MTP use normalized `cur_p->data[0].p` from the host
     sampler that emitted the candidate.
   - DFlash2 uses the selected predecessor's selector-score softmax probability;
     compute that softmax whenever cumulative mode is enabled, even if `p_min`
     is disabled.
   - DFlash drafted tokens use normalized host-sampler `cur_p->data[0].p`.
   - DSpark remains excluded because its confidence-head value is not token
     top-1 probability and no calibrated transform is provided.
6. Track `stop_reason`. When cumulative stopping leaves one or more tokens,
   preserve the first token and bypass generic `n_min` clearing caused solely
   by that cumulative shortening. Existing `p_min`/confidence failure before
   the first token and non-cumulative `n_min` behavior remain unchanged.
7. EAGLE3/MTP break before the next draft decode. DFlash/DFlash2 filter their
   already-decoded block, so report target-row savings separately from unchanged
   draft decode work. DSpark remains unchanged.
8. Start from each existing/adaptive maximum. Never raise it or mutate adaptive
   controller state; final length is the minimum of existing stops and the
   cumulative clamp. Leave chained MTP, draft-simple, and P09 independent.
9. Extend cycle evidence with drafter kind, probability domain, stop reason,
   crossing probability, threshold, and product before/after. MTP negative-mode
   calibration consumes P04; other drafter tests use deterministic local traces.
10. Choose each MTP ramp knot only from a paired A770 run. Require highest median
   throughput with accepted-tokens-per-drafted-token not below baseline. If no
   candidate qualifies, emit no ramp and fail P10 rather than copy P100 values.

## Critical files & anchors

- `common/speculative.cpp:61-64` - existing one-time environment-gate pattern.
- `common/speculative.cpp:886-1009` - EAGLE3 host loop and existing `p_min` stop.
- `common/speculative.cpp:1415-1561` - DFlash/DFlash2 filtering and excluded DSpark boundary.
- `common/speculative.cpp:2208-2228` - excluded chained MTP output.
- `common/speculative.cpp:2313-2415` - non-chained MTP host loop and existing `p_min` stop.
- `common/speculative.cpp:2117-2121,2443-2450` - existing/adaptive cap and controller update.
- `docs/plans/kmic68-01-bench-token-count.md` - comparable `-n 512` depth evidence.
- `docs/plans/kmic68-04-cycle-log.md` - per-cycle probability and phase schema.
- `scripts/perf/calibrate-spec-pcum.py` - planned MTP-only offline threshold replay.
- `scripts/perf/verify-spec-pcum.py` - planned MTP A770 calibration and paired depth A/B.
- `tests/test-speculative-pcum.cpp` - planned EAGLE3, DFlash2, DFlash, MTP, excluded-DSpark, and parser fixtures.
- `docs/research/kmic68-a770-pcum-ramp.md` - planned MTP-keyed ramp provenance and decision record.

## Verification

Add deterministic tests for unset, zero, above-one, equality, strict crossing,
first-token preservation after prior stops, cumulative-only `n_min` bypass,
adaptive caps, complete numeric parsing, and every included/excluded drafter.
DFlash2 fixtures vary selector scores with `p_min` on/off; DSpark fixtures
prove both confidence-head and sampler-probability changes leave P10 disabled.

```bash
timeout 240 ctest --test-dir build-sycl -R 'test-qwen4exp-mtp|test-speculative-pcum' --output-on-failure
```

After P01/P04, run MTP calibration at the P01 token count and depths:

```bash
timeout 3600 python3 scripts/perf/verify-spec-pcum.py --server ./build-sycl/bin/llama-server --model target-qwen4exp.gguf --draft-model qwen4exp-mtp.gguf --prompts scripts/perf/prompts.jsonl --depths 0,4096,16384 --n-predict 512 --repetitions 3 --threshold-grid 0:1:0.01 --seed 123 --spec-args='--spec-type draft-mtp --spec-draft-n-max 7' --cycle-log /tmp/p10-cycles.jsonl --report docs/research/kmic68-a770-pcum-ramp.md
```

Expected evidence:

- unset mode reproduces current output and complete numeric-parser boundaries;
- each included path uses its declared probability domain and preserves the
  first candidate that passed existing stops;
- cumulative-only shortening cannot be erased by generic `n_min`, while an
  earlier existing confidence/`p_min` stop remains authoritative;
- deterministic temperature-zero runs emit identical target tokens;
- each MTP depth reports drafted/accepted counts, target rows, draft work,
  accepted-per-drafted, and paired throughput;
- DFlash2 tests vary selector softmax independently; DSpark tests prove exclusion;
- DFlash/DFlash2 report unchanged block decode separately from target-row savings;
- negative mode works only for keyed MTP and is disabled on EAGLE3/DFlash;
- DSpark, chained MTP, draft-simple, and P09 remain unchanged;
- the post-run two-driver fault gate passes and the service is restarted.

## Assumptions & contingencies

- Ramp depth is committed target `pos0`, not requested context or KV size.
- Explicit thresholds above 1 are legal and reduce a draft to the first token
  that already passed pre-existing stops; the measured MTP ramp remains `[0,1]`.
- DFlash/DFlash2 can save target work but not an already-issued draft block.
- DSpark is ineligible until a token-probability transform is separately
  calibrated and validated.
- The negative ramp is MTP-only until another eligible drafter/model/route
  receives its own probability dataset and real A770 A/B.
- If no MTP ramp improves the objective, leave P10 disabled and record it.
