# P10 - Cumulative-probability draft width

**Kind:** port
**Depends on:** P01 for the baseline, P04 for the dataset that sets the default

## Purpose

Narrow the draft once the running product of the drafted tokens' top-1
probabilities falls below a threshold, so a deep context does not pay to verify
rows that cannot be accepted. Kmic-68 stop when the running product drops under a
threshold that ramps with depth, and report that the benefit only appears deep in
context, where each extra verify row costs a full attention pass over the KV
cache.

## Source

- Kmic-68 `common/speculative.cpp`: `p_cum`, `p_cum_min`, and the stop condition
  `pc_next < p_cum_min(pos0)` alongside the existing `p_min` test.

## In this fork

- `common/speculative.cpp:2362-2367`, the non-chained per-token stop
  `cur_p->data[0].p < params.p_min`, which `continue`s before `result.push_back`.
- `common/speculative.cpp:2221`, the chained per-token stop `p < params.p_min`,
  over the packed `[id, prob]` rows that the in-graph decode emits at `:2210-2216`.

Both paths already carry a per-token confidence stop, so this is an additional
stop condition rather than a new mechanism. Note that the chained path fuses the
*decode* into one graph but still runs a host-side *selection* loop at
`:2213-2229`, so chain mode is in scope. In both paths the crossing token is
discarded: the stop `break`s before `result.push_back(id)`. The cumulative check
belongs beside the `p_min` test, not at `:2403-2406`, which is the separate
`draft_add` failure path.

## Requirements

- **R10.1** (optional feature) WHERE `LLAMA_SPEC_P_CUM` is set, the <drafter> shall multiply each successive drafted token's top-1 probability into a running product initialised to one, and shall stop drafting when that product falls below the threshold.
- **R10.2** (ubiquitous) The <cumulative rule> shall apply in addition to, and never in place of, the existing `p_min` stop at `:2221` and `:2363`.
- **R10.3** (optional feature) WHERE the threshold is not set, the <drafter> shall apply no cumulative-probability stop, regardless of padded verify state.
- **R10.4** (event-driven) WHEN the threshold is depth-dependent, the <default> shall be derived from a paired A770 measurement over our production depth range, not copied from Kmic-68's P100 values.
- **R10.5** (event-driven) WHEN `LLAMA_SPEC_P_CUM` is set to a negative value, the <drafter> shall use the measured default ramp.
- **R10.6** (event-driven) WHEN the running product falls below the threshold, the <drafter> shall discard that token and end the round, matching the existing `p_min` behaviour which breaks before pushing.
- **R10.7** (event-driven) WHEN the rule stops drafting at depth zero, the <drafter> shall fall back to the ordinary `p_min` stop for that round.
- **R10.8** (event-driven) WHEN the adaptive controller selects the draft width, the <cumulative rule> shall clamp the selected width rather than replace the controller's choice.
- **R10.9** (event-driven) WHEN the drafter is the chained path, the <cumulative rule> shall read the probability from the packed row rather than from a host-side sampler.
- **R10.10** (event-driven) WHEN the threshold is active, the <acceptor> shall not treat a token discarded by it as drafted, so it cannot appear in the verification batch.

## Acceptance

Paired depth sweep on the A770 reporting accepted-tokens-per-drafted-token and tg,
against P01's re-baselined numbers, with the P04 log as the dataset. Any default
threshold chosen here is recorded with the measurement that produced it.
