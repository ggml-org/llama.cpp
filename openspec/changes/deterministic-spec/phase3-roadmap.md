# Phase 3 Roadmap (Forward-Looking Design Notes)

> **Status**: pre-planning teaser, not committed scope. Nothing here is in the
> current PR. These notes document known limitations of the Phase 2 design and
> the extension directions already considered for addressing them, so reviewers
> can see the constraints are understood and the API contract was shaped to
> accommodate them. Feasibility notes reflect analysis against the Phase 2
> plugin contract (`external/include/deterministic_draft_plugin.h`) as of
> 2026-07-19.

## What Phase 2 ships (and its known limits)

Phase 2 delivers a single deterministic grammar filter in the speculative
pipeline: drafts from the MTP head are validated against an XGrammar/GBNF
grammar, truncated at the first invalid token, and optionally committed without
target verification (`--det-draft-accept-all`).

Known limits, by design:

1. **It is a blunt instrument.** The filter can only truncate. A draft that is
   90% useful loses its tail at the first invalid token even when an obvious
   repair (a missing `;` or `}`) exists.
2. **Single domain per buffer.** One grammar per slot. Content that
   intentionally leaves the grammar's domain (prose, markdown, a second
   language, an unexpected but valid construct) is truncated the same as a
   genuine error. This is the documented accept-all limitation: safe only for
   single-language code-only generation.
3. **The niche is narrow on purpose.** Deterministic rejection of anything
   outside the grammar is the feature - but it confines the approach to
   fixed-language codegen workloads.

Both extensions below address these limits without weakening the deterministic
guarantee for workloads that want it: the default configuration remains a
single strict filter, exactly as shipped in Phase 2.

## Direction 1: Chain of responsibility (filters that transform, not just truncate)

Extend the single filter into an ordered chain of plugins. Each chain element
may reject (truncate, as today) or **transform**: at the position where the
draft token is invalid, return a grammar-valid alternative instead of cutting
the draft.

Why it is cheap: at the rejection position the filter already holds (a) the
grammar bitmask - the exact set of valid next tokens, and (b) the draft head's
logits - a free ranking over that set. A meaningful repair is argmax over
bitmask intersect draft logits. No target-model distribution and no resampling
is involved. XGrammar's existing `get_jump_forward()` is already a degenerate
form of this (deterministic insertion); this direction generalizes it to
repair. Note this revives the Phase 1 PoC's fix-injection idea at the
draft-batch level, avoiding the per-token sampler overhead that motivated
Phase 1's replacement.

Feasibility notes:

- Chain orchestration is host-side; default chain length 1 is today's behavior.
  Backward compatible.
- Hard constraint: token-stream state consistency. Truncation only ever
  removes draft tokens, so target KV, draft KV, and grammar state always
  describe the same stream. A transform changes the stream, so it is coherent
  only in accept-all mode, where the emitted stream is redefined as the chain
  output and ingested into the target as a plain batch. The draft KV for
  changed positions goes stale and needs rollback/re-decode, and
  `common_speculative_accept()` must learn to accept a stream that differs in
  content (not just length) from the draft. This is the real implementation
  cost.
- API impact: additive. New capability bit (e.g. `CAPABILITY_TRANSFORM`) plus
  one new function returning a rewritten token array. `filter_draft()` and all
  Phase 2 plugins are unaffected. The capability-based contract shipped in
  Phase 2 was designed for exactly this kind of extension.

## Direction 2: Threshold-gated pass-through on domain switch

Today, a draft token outside the grammar is always treated as an error. But a
strong, confident out-of-grammar prediction usually signals an *intentional*
domain switch (code to prose, embedded second language, markdown), not a
mistake. The complementary case is a stream that stays grammatically valid but
drifts semantically (comment spam): also a possible context switch, invisible
to rejection alone. The proposal: detect drift inside the filter, and on a
confirmed drift, stop filtering and let the target arbitrate - passthrough,
not rejection and not silent acceptance.

The stripped-down design (the feasible one):

1. **Trigger.** A drift event fires inside the filter. Candidates, in
   implementation order:
   - Rejected-token confidence: the filter rejects a token, but the target's
     distribution at that position puts high probability on it (above a
     configurable threshold, e.g. 0.9, or relative like min-p). In accept-all
     mode these logits come free: the committed prefix is already ingested
     through the target as a batch, so the next-token distribution at exactly
     the rejection point falls out of that pass at zero extra cost.
   - Repetition heuristic: catches the observed degenerate-spam failure mode
     cheaply, without new plumbing.
   - Valid-stream drift: draft-head probability collapse while still
     in-grammar. The direct signal for the spam case, but needs
     draft-sampling probabilities plumbed into the filter - v2.
2. **Confirmation.** A drift event sets a high-water mark; only if a second
   event fires within N tokens does the filter latch off. One-off blips
   (genuinely broken drafts) keep today's truncate behavior.
3. **Latch.** Once latched, the filter becomes a passthrough for the rest of
   the request: drafts flow to standard target verification, which the
   machinery already has (non-accept-all mode). The deterministic guarantee is
   never weakened while the filter is active - it is either strict or off.

Why the transition is clean:

- At the latch point, filter state and target KV are in sync by construction
  (in accept-all the target ingested exactly what the filter committed).
- The grammar matcher is abandoned, not resynced. Passthrough mode never
  consults grammar state, so the hard problem - resyncing a parser to
  arbitrary mid-stream text - is sidestepped entirely.
- Re-engagement is free at natural reset points: slot reset, next request.
  Within-request re-arm is explicitly out of scope for v1. A v2 may re-arm at
  structural landmarks (blank line + dedent, markdown fences) by resetting the
  matcher and constraining only forward - reset-and-resume, not resync.

Implementation notes:

- Host-side: one `deferred` flag per slot in the det filter state, checked at
  the existing draft/accept branches; plus the drift counter and window. No
  new speculative machinery - the fallback path is the standard verification
  branch that already exists.
- The flag rides in the slot checkpoint blob (the state serialization SPI
  shipped in Phase 2), so a checkpoint/restore preserves deferral instead of
  silently re-arming.
- API impact: none for the latch itself. Trigger 1 needs no new SPI. Trigger 3
  may add an optional drift-score getter later, behind a capability bit as
  usual.
- Estimated size: ~50 lines host-side plus tests in the existing harness.

The two directions compose: the threshold gate is a natural chain-terminator
element in Direction 1 ("if the grammar filter rejected but the target
strongly wants this token, hand off to pass-through instead of truncating").

## Cross-cutting caveat

Both directions amplify token-stream state consistency (draft KV vs target KV
vs grammar state) - the same surface as the unresolved baseline-MTP failure
seen on the N100, documented in
`deterministic-draft-model-poc/docs/observations.md` (root cause not isolated;
fork integration, benchmark tool, or upstream all still candidates). That
investigation (tasks 16.3.x in tasks.md) should conclude before building
transform semantics on top of the speculative path.

## Relationship to this PR

Phase 2 ships the single strict filter and the capability-based C API. Both
directions land later as new capability bits and optional functions; plugins
written against the Phase 2 contract continue to work unchanged, and the host
degrades gracefully when capabilities are absent - the extension story the
contract was designed for.

Both directions were kept out of this PR deliberately, not for feasibility -
Direction 2 in particular is small and well-scoped - but because the PR already
carries the pipeline integration, the plugin contract, the reference plugin,
and the checkpoint serialization. If the PR is accepted, Direction 2 is the
natural first follow-up: it reuses the fallback branch and the serialization
path shipped here, and answers the most predictable reviewer question ("what
happens when generation legitimately leaves the grammar?").
