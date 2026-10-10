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

`LLAMA_SPEC_P_CUM` selects the runtime mode. Unset disables the rule. A
finite non-negative value is the explicit threshold. Any finite negative value
selects the A770-measured depth ramp. NaN, infinity, and malformed values fail
initialization. The ramp uses the committed target position at cycle start and
requires the machine-readable artifact at `LLAMA_SPEC_P_CUM_RAMP=<path>`.
The P04/P01 calibration report alone does not enable negative mode. Unset and
constant modes, and excluded drafters, do not read the artifact.

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
- **R10.14 (Unwanted behaviour):** IF negative MTP mode lacks a valid calibration artifact matching the active target model, draft model, backend devices, route, build, and driver, THEN the speculative initializer shall fail before drafting.

## Approach

1. Complete P01 and collect P04 synchronized rows for non-chained MTP only.
   `scripts/perf/calibrate-spec-pcum.py` replays those MTP top-1 sequences over
   a predeclared threshold grid and shortlists real A770 A/B candidates.
2. Have `verify-spec-pcum.py --phase calibrate --ramp-output <path>` write a
   version-1 JSON artifact alongside the human-readable report only after the
   paired measurements qualify. Store `schema_version`, `drafter=draft-mtp`,
   target/draft model SHA-256 identities, backend device identities and their
   target/draft assignments, route, build identity, driver identity,
   and exactly three ordered `(position,threshold)` knots at 0, 4096, and 16384.
   Build identity includes hashes of the server and loaded backend libraries;
   driver identity includes kernel driver and compute-runtime versions. GPU
   identity includes vendor/device ID, architecture, VRAM capacity, and device
   UUID from the active backend. Match the complete assigned device set, not
   just the default GPU name; missing identity or a different GPU/topology fails
   initialization. Calibration records the devices that actually executed it.
   The common speculative initializer reads `LLAMA_SPEC_P_CUM_RAMP` once, with
   a 64 KiB input limit, only for negative non-chained MTP mode. Require version 1,
   all fields, exact identity matches, and finite thresholds in `[0,1]`; missing,
   unreadable, oversized, malformed, or mismatched artifacts fail initialization.
   Keep the validated knots immutable for the run and record the artifact hash
   in cycle evidence. Clamp outside endpoints and interpolate between adjacent
   knots. A negative setting on another drafter logs one ineligibility reason
   and applies no cumulative stop; it never loads MTP calibration.
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
- `common/speculative.cpp` - planned bounded ramp-artifact loader and runtime identity validation.
- `tests/test-speculative-pcum.cpp` - planned EAGLE3, DFlash2, DFlash, MTP, excluded-DSpark, and parser fixtures.
- `docs/research/kmic68-a770-pcum-ramp.md` - planned MTP-keyed ramp provenance and decision record.

## Verification

Add deterministic tests for unset, zero, above-one, equality, strict crossing,
first-token preservation after prior stops, cumulative-only `n_min` bypass,
adaptive caps, complete numeric parsing, and every included/excluded drafter.
Add artifact-loader fixtures for the three knots, interpolation and endpoint
clamping, missing path/file, malformed/oversized data, unsupported schema,
missing/duplicate/out-of-order positions, non-finite/out-of-range thresholds,
and each identity mismatch, including another GPU with the same driver and a
changed target/draft device assignment. Unset, constant, and excluded modes must
perform no artifact I/O. Replacing the artifact after initialization must not change the
in-memory knots or recorded hash.
DFlash2 fixtures vary selector scores with `p_min` on/off; DSpark fixtures
prove both confidence-head and sampler-probability changes leave P10 disabled.

```bash
timeout 240 ctest --test-dir build-sycl -R 'test-qwen4exp-mtp|test-speculative-pcum' --output-on-failure
```

After P01/P04, run MTP calibration at the P01 token count and depths:

```bash
timeout 3600 python3 scripts/perf/verify-spec-pcum.py --phase calibrate --ramp-output /tmp/p10-ramp.json --server ./build-sycl/bin/llama-server --model target-qwen4exp.gguf --draft-model qwen4exp-mtp.gguf --prompts scripts/perf/prompts.jsonl --depths 0,4096,16384 --n-predict 512 --repetitions 3 --threshold-grid 0:1:0.01 --seed 123 --spec-args='--spec-type draft-mtp --spec-draft-n-max 7' --cycle-log /tmp/p10-calibration-cycles.jsonl --report docs/research/kmic68-a770-pcum-ramp.md
timeout 3600 python3 scripts/perf/verify-spec-pcum.py --phase validate --ramp-input /tmp/p10-ramp.json --modes=unset,-1 --server ./build-sycl/bin/llama-server --model target-qwen4exp.gguf --draft-model qwen4exp-mtp.gguf --prompts scripts/perf/prompts.jsonl --depths 0,4096,16384 --n-predict 512 --repetitions 3 --seed 123 --spec-args='--spec-type draft-mtp --spec-draft-n-max 7' --cycle-log /tmp/p10-validation-cycles.jsonl --report docs/research/kmic68-a770-pcum-ramp.md
```

Calibration launches use only unset/explicit constant modes and clear inherited
ramp settings. Validation uses the same binaries without rebuilding, sets
`LLAMA_SPEC_P_CUM_RAMP` from `--ramp-input` and `LLAMA_SPEC_P_CUM=-1` for the
negative arm, and clears both for the unset arm. Append the validation results
and consumed artifact hash to the report; do not refit knots during validation.
No deployment or performance claim is supported until this second phase passes.

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
- negative mode consumes the recorded artifact only for matching MTP, fails on
  missing/invalid/mismatched artifacts, and is disabled on EAGLE3/DFlash;
- calibration and validation record identical binary/model/device/route/driver
  identities, and every negative-mode cycle carries the validated artifact hash;
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
- A changed binary, backend library, model, backend device/assignment, route, or
  driver invalidates the artifact and requires calibration plus validation again;
  there is no built-in
  ramp or silent fallback for negative MTP mode.
