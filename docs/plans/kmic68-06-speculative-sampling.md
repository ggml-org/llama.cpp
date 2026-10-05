# P06 - Distribution-recording acceptance

**Kind:** port
**Depends on:** P04 for validation

## Purpose

Our acceptance rule (`common/sampling.cpp:678`) accepts a draft token only if it
equals the target's sample. That is exact for greedy decoding but discards the
draft's own distribution `q`: off-greedy, acceptance probability is `p(x)`
rather than `min(1, p(x)/q(x))`.

The correct rule accepts with `min(1, p(x)/q(x))` and, on rejection, draws the
correction from the residual `max(p - q, 0)`. The output distribution is unchanged;
the acceptance rate is not.

Requires the drafter to *record* the distribution it sampled from. Kmic-68 adds
this with an opt-in temperature and top-p on the draft side; we default it on when
the chain is stateless and temperature is above zero.

## Source

- Kmic-68 `common/sampling.cpp`: `common_sampler_sample_and_accept_n_dist`, and in
  `common/speculative.cpp` the `dists` field, `sample_temp`, `sample_top_p`.

## In this fork

- `common/sampling.cpp:678` `common_sampler_sample_and_accept_n`, the rule replaced.
- `common/speculative.h:72-91` `common_speculative_draft_params`, which has no
  `dists` field today.
- `common/speculative.cpp:2348-2375`, the non-chained MTP host-side sample and push.

## Requirements

- **R06.1** (ubiquitous) The <draft params> shall carry, per drafted token, the distribution `q` that token was drawn from.
- **R06.1a** (ubiquitous) The <recorded distributions> shall be moved, truncated, replaced and replayed in lockstep with the token vector at every site that mutates a draft, including checkpoint save and load, `spec_draft` replacement by an accepted prefix plus correction, and any accepted-token replay, so a token can never be verified against another token's `q`.
- **R06.1b** (event-driven) WHEN a correction token is emitted, the <acceptor> shall carry no `q` for it, since it was drawn from the residual rather than from `q`.
- **R06.2** (optional feature) WHERE `LLAMA_SPEC_SAMPLE_TEMP` is greater than zero, the <MTP drafter> shall draw each draft token from the distribution that R06.3 leaves in force, rather than taking the argmax, and shall record that distribution.
- **R06.3** (optional feature) WHERE `LLAMA_SPEC_DRAFT_TOPP` is less than one, the <MTP drafter> shall first truncate the temperature-scaled distribution to its smallest prefix reaching that mass, matching the target sampler rule, and shall draw from the truncated result. Truncating only the recorded copy after the draw would leave a token sampled from the discarded tail with q(x) = 0, and R06.4 ratio would divide by zero.
- **R06.4** (event-driven) WHEN a drafted token is verified and `q` was recorded, the <acceptor> shall accept it with probability `min(1, p(x)/q(x))`.
- **R06.5** (event-driven) WHEN a drafted token is rejected, the <acceptor> shall draw the correction token from the residual `max(p - q, 0)` renormalised, and shall emit that correction in place of the rejected token.
- **R06.6** (ubiquitous) The <acceptance rule> shall output the target distribution `p` exactly, for the same recorded target and draft distributions. Parity with the greedy rule is not the invariant: the greedy rule already reproduces `p`, so testing against it would not detect a sampler that preserves neither.
- **R06.7** (unwanted) IF a grammar, penalty, DRY, mirostat or reasoning budget is active, THEN the <acceptor> shall use the existing exact-match acceptance path, which continues to sample from the target's configured distribution; it shall not switch to argmax.
- **R06.8** (unwanted) IF a backend sampler already picked the token, THEN the <acceptor> shall use the existing exact-match acceptance path, because `cur_p` is not the distribution those tokens came from.
- **R06.9** (optional feature) WHERE `LLAMA_SPEC_SAMPLE_TEMP` is unset or zero, the <drafter> shall take the argmax and record no distribution, so the <acceptor> shall use the existing exact-match acceptance path and current behaviour stays reachable without a rebuild.
- **R06.10** (event-driven) WHEN the drafter is the chained path, the <acceptor> shall continue to use the existing exact-match acceptance path, because its tokens are produced by an in-graph argmax with no recorded distribution.

## Acceptance

Over a seeded corpus of at least 100000 draws per token position, spanning a
case where `p` and `q` agree closely and one where they diverge substantially, the
empirical frequency of each emitted token shall match the analytic target `p` under
a two-sample chi-square goodness-of-fit at p > 0.001, with total variation
distance below 0.01. The same test shall be run against the existing exact-match
path as a control. Acceptance rate before and after at temperature 0.8 on the real
trunk, A770, named driver, sole tenancy.
