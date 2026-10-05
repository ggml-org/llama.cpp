# P07 - Block verification

**Kind:** port
**Depends on:** P06

## Purpose

Sun et al., *Block Verification Accelerates Speculative Decoding* (arXiv:2403.10444,
ICLR 2025), Algorithm 2. Under per-token rejection sampling a rejection at
position *i* ends the block. Block verification does not: it tracks `p_i`, the
prefix acceptance probability, and a later position can still set `tau` after an
earlier one failed. Their Theorem 2 states it is never worse in expected
accepted tokens.

Two review passes disagreed on the recurrence for `p_i`. Algorithm 2 line 4
settles it: `p_i = min{ p_{i-1} * M_b(X_i) / M_s(X_i), 1 }`, the clipped product,
so `p_i` is non-increasing. Kmic-68 ships that form and this plan now requires
it. An earlier revision of this file had it backwards.

The instrumentation matters as much as the rule. `LLAMA_SPEC_BLOCK_VERIFY=2`
reports the expected accepted tokens per drafted token under both rules,
`sum_i p_i` against the per-token `sum_i prod_{j<=i} min(1, p_j/q_j)`, which
measures the gain with no timing run at all.

## Source

- Kmic-68 `common/sampling.cpp`: `common_spec_block_verify`, and the block branch
  inside `common_sampler_sample_and_accept_n_dist`.

## In this fork

- `common/sampling.cpp:678`, the per-token rule that P06 keeps for stateful chains
  and that this plan supersedes for stateless ones.

## Requirements

- **R07.1** (event-driven) WHEN `LLAMA_SPEC_BLOCK_VERIFY` is not zero and the chain is stateless, the <acceptor> shall evaluate every position in the block rather than stopping at the first rejection.
- **R07.2** (event-driven) WHEN position *i* is evaluated, the <acceptor> shall set `p_i = min(p_{i-1} * p(x_i)/q(x_i), 1)`, clamping the product, exactly as Sun et al. Algorithm 2 line 4 specifies. `p_i` is therefore non-increasing.
- **R07.2a** (event-driven) WHEN `p_i` is clamped, the <acceptor> shall not instead carry an unclipped cumulative ratio and clamp it at definition time; that is a different recurrence, it can rise after falling, and it is not the algorithm.
- **R07.2b** (event-driven) WHEN position *i* is evaluated, the <acceptor> shall take the accept test from the algorithm's `h_i^block` (Equation 5) and the residual from `p_res^block` (Equation 4), which is `p_tau * M_b - M_s`, rather than reusing the per-token forms.
- **R07.3** (event-driven) WHEN a position fails its `h_i^block` test, the <acceptor> shall continue the loop rather than break, setting `tau = i` whenever a later test succeeds, which is Algorithm 2 lines 6-10.
- **R07.4** (event-driven) WHEN `LLAMA_SPEC_BLOCK_VERIFY=2`, the <acceptor> shall every 256 blocks log the expected accepted tokens per drafted token under both rules, `sum_i p_i` against the per-token `sum_i prod_{j<=i} min(1, p_j/q_j)`. The two differ by design: the block rule clips at every step, so it is expected to trail the per-token product while still being never worse in accepted tokens.
- **R07.5** (event-driven) WHEN a position is rejected, the <acceptor> shall draw the correction from the residual scaled by the surviving prefix's `keep` value.
- **R07.6** (unwanted) IF a grammar, penalty, DRY, mirostat or reasoning budget is active, THEN the <acceptor> shall use the per-token rule from P06.
- **R07.7** (unwanted) IF `LLAMA_SPEC_BLOCK_VERIFY=0`, THEN the <acceptor> shall use the per-token rule from P06.
- **R07.8** (ubiquitous) The <block rule> shall produce the same output distribution as the per-token rule.

## Acceptance

The R07.4 output shows a non-negative gain over a six-request run, with the
block figure trailing the per-token figure as the recurrence implies. Output
distribution parity against P06 over the seeded corpus, as P06 R06.6 requires.
