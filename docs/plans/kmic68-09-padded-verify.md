# P09 - Padded fixed-width verify

**Kind:** decoupling
**Depends on:** none

## Purpose

Kmic-68 measured roughly 26 ms to rebuild the target graph on every change of
draft length, and mitigated it with a padded fixed-width verify: pad the target
batch to a constant width so the graph is built once and reused across draft
lengths.

Our chain grows `mtp_chain_rows` by doubling for the same reason
(`src/llama-context.cpp:2378-2382`).

This plan is deliberately *not* coupled to P10. In Kmic-68 the cumulative
probability threshold returns zero unless padded verify is enabled, which welds
two independent risks to one flag; separating them lets each be adopted,
measured and reverted on its own.

## Source

- Kmic-68 `params.pad_verify` and its use in `p_cum_min`; the padding itself lives
  in their server path.

## In this fork

- `common/speculative.cpp:2104-2230`, the chained path that consumes
  `mtp_chain_rows`.
- `src/llama-context.cpp:2378-2382`, the doubling growth of `mtp_chain_rows`.

## Requirements

- **R09.1** (optional feature) WHERE padded verify is enabled, the <target batch> shall be padded to the configured fixed width on every speculative step regardless of the current draft length.
- **R09.2** (ubiquitous) The <padded rows> shall be masked so they cannot contribute to the output or to the KV cache.
- **R09.3** (event-driven) WHEN padded verify is enabled, the <target graph> shall not be rebuilt for a draft length that fits inside the fixed width.
- **R09.4** (unwanted) IF padded verify is enabled, THEN the <acceptor> shall emit no token for a padded row.
- **R09.5** (event-driven) WHEN padded verify is disabled, the <target batch> shall be exactly the draft length, preserving current behaviour.
- **R09.6** (unwanted) IF the fixed width exceeds the batch or microbatch capacity, THEN the <padded verify> shall fall back to the unpadded path and log the reason.
- **R09.7** (ubiquitous) The <fixed width> shall default to a power of two at or above `n_max + 1`, because a verify batch carries the previously sampled token plus the draft rows and `common_sampler_sample_and_accept_n` asserts `idxs.size() == draft.size() + 1`. A width of `n_max` is one row short whenever `n_max` is itself a power of two.
- **R09.8** (optional feature) WHERE padded verify is enabled, the <target graph> shall record a build-or-reuse counter, incrementing only on a rebuild, so R09.3 can be measured.
- **R09.9** (unwanted) IF padded rows reach a speculative implementation, THEN the <implementation> shall ignore them explicitly, because the server passes `batch.view` unchanged to `common_speculative_process` at `tools/server/server-context.cpp:4130-4133` and each implementation mirrors that batch into the draft context. Masking them from target outputs and target KV is not enough: their tokens and positions would otherwise enter draft KV or recurrent state and change later drafts.

## Acceptance

Identical tokens with and without padding over the oracle corpus. Graph rebuild
count per 100 steps before and after, from the counter R09.8 adds. P04's profile
does not supply this: it instruments the drafter's catch-up and draft-step
phases, not the target graph this plan changes.
