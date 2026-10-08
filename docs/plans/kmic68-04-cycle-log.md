# P04 - Speculative cycle log and phase profile

**Kind:** instrumentation
**Depends on:** none
**Feeds:** P06, P07, P10

## Purpose

Every measurement-driven plan in this set needs the same dataset: per draft
cycle, how long drafting took, how long verification and sampling took, how many
tokens were accepted, and how confident the drafter was at each step. Building it
once, correctly, is cheaper than reconstructing it three times.

This is pure instrumentation. It changes no output and no performance when unset.

## Source

- Kmic-68 `common/speculative.cpp` +204/-5: `LLAMA_SPEC_PROFILE`, `LLAMA_SPEC_LOG`,
  `prof_catchup_us`, `prof_step_us`, `spec_log_rec`.

## In this fork

- `common/speculative.cpp:1576`, `common_speculative_impl_draft_mtp`, the only
  drafter these plans instrument.
- `common/speculative.cpp:2348` host-side sample and `:2375` result push, the two
  points the log brackets.

## Requirements

- **R04.1** (optional feature) WHERE `LLAMA_SPEC_LOG=<path>` is set, the <MTP drafter> shall append one line per draft cycle and sequence carrying a request or generation identifier, position, draft time, verify-and-sample time, accepted count, and the top-1 probability of every drafted token including the one that stopped drafting.
- **R04.2** (optional feature) WHERE `LLAMA_SPEC_PROFILE` is set, the <MTP drafter> shall report synchronised catch-up and draft-step wall times averaged per draft call, every 64 calls.
- **R04.3** (unwanted) IF `LLAMA_SPEC_PROFILE` is set, THEN the <drafter> shall synchronise the draft context before reading each phase timestamp.
- **R04.4** (ubiquitous) The <log format> shall remain parseable when the optional fields are absent, so a consumer written before this plan still reads it.
- **R04.4a** (event-driven) WHEN a server slot is reused by a new request, the <request identifier> shall differ from the previous request on that slot, so two requests beginning at the same position produce distinguishable records.
- **R04.4b** (event-driven) WHEN a checkpoint rollback replays accepted tokens, the <cycle record> shall use the server's replay-adjusted accepted count, because `post_decode` sets `spec_is_replay` and returns before `common_speculative_accept`, the server passes `accepted.size() - 1` on replay, and a record finalised only in the MTP `accept` callback would otherwise disagree with `slot.stats.n_draft_accepted`.
- **R04.5** (event-driven) WHEN neither `LLAMA_SPEC_LOG` nor `LLAMA_SPEC_PROFILE` is set, the <drafter> shall not synchronise and shall not branch inside the decode loop.

## Acceptance

A six-request run producing a log whose accepted counts reconcile with the
server's own speculative statistics, and a profile line whose catch-up plus
draft-step total is consistent with the cycle wall time to within the synchronise
overhead.
