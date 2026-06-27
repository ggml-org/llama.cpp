# Deterministic Draft Filter

A shared-library plugin that validates speculative draft tokens against a
deterministic rule set -- typically a grammar compiled by XGrammar -- before
they reach target-model verification. Structurally invalid drafts are pruned
before wasting verification slots.

The contribution is the plugin contract and the integration point -- not the
specific validator. The reference implementation uses XGrammar for
grammar-constrained decoding with jump-forward support as a concrete
demonstration of one domain. Any domain with a deterministic correctness
criterion -- legal citation formats, schema validation, regulatory constraints,
structured query languages, private proprietary rules -- can implement the same
contract and plug in without touching llama.cpp core.

## Background: MTP and why accept rate matters

In standard autoregressive decoding, the main output head produces one token
per forward pass -- 1:1. MTP-enabled models add auxiliary prediction heads that
run on top of the same shared backbone hidden state, each drafting one
additional token. A model with 3 MTP heads produces 4 candidate tokens per
forward pass: 1 from the main head plus 3 draft tokens from the auxiliary heads
(`--spec-draft-n-max 3`).

The auxiliary heads are cheap relative to the shared backbone computation, which
is already done. The efficiency gain depends entirely on accept rate -- how many
draft tokens the target accepts before rejecting. Rejected draft tokens waste
the cost of the auxiliary heads and fall back to single-token decoding.

On constrained hardware, this failure mode is severe. On the N100 in the
original benchmark configuration (n_max 100), baseline MTP accept rate was
under 1% -- the auxiliary heads generated draft tokens that were rejected over
99% of the time. Tuning the draft budget recovers some acceptance, but the
underlying issue remains: there is no structural guarantee between draft and
target. The filter addresses this at the source -- validating draft tokens
against a language grammar before they reach target verification, so only
structurally valid tokens consume verification slots.

The filter is compatible with multiple drafter types: `--spec-type draft-mtp`,
`draft-eagle3`, `draft-dspark`, and `draft-dflash`.

## What this is not

The filter validates structural correctness only, not semantic correctness.
Code that parses as valid C may still be semantically wrong -- a shadowed
variable, an incorrect algorithm, an off-by-one. The grammar-level constraint
cannot check meaning. Semantic correctness remains the caller's responsibility.

## Integration points

Changes to llama.cpp core are confined to:

- `include/deterministic_draft_plugin.h` -- the C SPI contract that plugin
  authors implement
- `include/llama_deterministic_draft.h` -- the consumer-facing API used by
  llama.cpp core
- `src/llama-deterministic-draft-serviceloader.cpp` -- plugin loader
  (dlopen/dlsym), feeds draft tokens to the plugin via the C API
- `common/speculative.{h,cpp}` -- pipeline integration, including the filter
  hook inside `common_speculative_draft()`, the conditional bypass of
  `common_sampler_sample_and_accept_n`, and the stats reporting in
  `common_speculative_print_stats()`

The plugin loader is always compiled into libllama, so the `--det-draft-*`
flags work in any build.

The plugin has no access to core sampling routines. It receives tokens via the
C API contract, returns a validation result, and core decides what to do with
that result based on the flags. Domain logic stays entirely outside core.

A single plugin instance is shared across all inference slots. Each call
carries an `int32_t slot_id`; plugins maintain per-slot state maps
internally. The host serializes all calls on one instance, so plugins do
not need per-instance thread safety.

## Plugin contract

The contract is capability-based. Plugins report their features through
`deterministic_draft_get_capabilities()` and the host degrades gracefully
for missing capabilities.

**CAPABILITY_BITMASK** -- the plugin fills a bitmask of valid token IDs
before each sampling step. `deterministic_draft_filter_draft()` filters a
batch of draft tokens against the bitmask, commits valid tokens to the
grammar state, and stops at the first invalid token (commit-on-accept).
This is the primary validation mechanism used by the reference implementation.

**CAPABILITY_JUMP_FORWARD** -- the plugin returns strings that are uniquely
determined by the current grammar state, allowing deterministic sequences to
be skipped without model sampling. Present in the contract and implemented
in the reference implementation. Not yet wired into `common/speculative.cpp` --
integration into the draft stream is non-trivial and deferred to follow-up
work.

**Rollback** -- `deterministic_draft_rollback()` undoes the last N commit
calls for a given slot, keeping grammar state consistent with what was
actually emitted when the target model accepts fewer tokens than the plugin
already committed. Required in default mode (without intercept-all).

## Plugin architecture

```
PLUGIN AUTHOR          DISTRIBUTION           END USER
-------------          ------------           --------
plugin.cpp
#include plugin.h
implements contract
      |
      v
deterministic-        ships .so          downloads .so
draft.so         ------------------>           |
                                               v
                                     llama.cpp
                                     --det-draft-model ./plugin.so
                                           |
                                     ServiceLoader (in libllama)
                                     dlopen / dlsym
                                     deterministic_draft_filter_draft
                                     deterministic_draft_commit
                                     deterministic_draft_rollback
                                     deterministic_draft_reset
                                     deterministic_draft_destroy
                                           |
                                     common/speculative.cpp
                                     feeds draft tokens to plugin
                                     applies filter result
```

Three parties, three distinct concerns:

**Plugin author** -- writes a domain-specific validator, includes
`deterministic_draft_plugin.h`, implements `deterministic_draft_create`,
`deterministic_draft_filter_draft`, `deterministic_draft_commit`,
`deterministic_draft_rollback`, `deterministic_draft_reset`,
`deterministic_draft_destroy`, and compiles to a `.so`. The only party that
needs the headers. The reference implementation (XGrammar grammar-constrained
decoding) is one example; a regulated organisation's private validator is
another.

**Distribution** -- the plugin author ships the `.so`. Open source on GitHub,
a private artifact in a corporate repo, or anything in between. llama.cpp
carries no opinion on how plugins are distributed.

**End user** -- downloads or receives the `.so`, passes it via
`--det-draft-model ./plugin.so`, and runs llama.cpp. They never see a header.
The loader (compiled into libllama) handles `dlopen`/`dlsym` at runtime,
resolves the function pointer table, and calls into the plugin through
`common/speculative.cpp`.

## Flags

| Flag | Description |
|---|---|
| `--det-draft-model <path>` | Path to the plugin (.so/.dylib/.dll). Requires a compatible `--spec-type` (see below). Does not auto-enable any `--spec-type`. |
| `--det-draft-n-max <N>` | Max tokens to validate per draft step. `-1` = all (default), `0` = disabled, `>0` = caps filter output. When positive, the value is also applied to `--spec-draft-n-max`. |
| `--det-draft-intercept-all` | Skip target verification of draft tokens; the filter is the sole verifier. Default: false. Requires `--det-draft-model`. MTP only (see below). |

The examples in `README.md` also use standard llama.cpp / speculative-decoding flags:

**Model and server**

| Flag | Meaning |
|---|---|
| `-m <path>` | Target model (GGUF). The distribution you actually sample from. |
| `-md <path>` | Separate draft/speculator model (eagle3/dspark only). For dspark, `-m` is the target and `-md` is the draft. |
| `-c <N>` | Context length (KV cache size). |
| `-ngl <N>` | Layers offloaded to GPU (`99` = all). |
| `-fa on` | Flash attention. |
| `--port <N>` | HTTP listen port. |
| `--reasoning off` | Disable reasoning mode. |

**Sampling** (`--temp 0 --top-k 1 --top-p 1.0` together = greedy, deterministic)

| Flag | Meaning |
|---|---|
| `--temp 0` | Temperature 0. |
| `--top-k 1` | Keep only the single highest-probability token. |
| `--top-p 1.0` | Nucleus at 1.0 (no truncation). |

**Speculative decoding**

| Flag | Meaning |
|---|---|
| `--spec-type <t>` | Drafter: `draft-mtp`, `draft-eagle3`, `draft-dspark`, `draft-dflash`. |
| `--spec-draft-n-max <N>` | Max tokens per draft step. |

The plugin's own flags are `--det-draft-model`, `--det-draft-n-max`, and `--det-draft-intercept-all` (table above).

**Parse errors:**

- `--det-draft-intercept-all` without `--det-draft-model`:
  `--det-draft-intercept-all requires --deterministic-draft-model (plugin not loaded)`
- `--det-draft-model` without a compatible `--spec-type`:
  `--det-draft-model requires a compatible drafter type (--spec-type draft-mtp, draft-eagle3, draft-dspark, or draft-dflash)`

**Intercept-all downgrade:** if `--det-draft-intercept-all` is requested with a
non-MTP drafter, the flag is cleared at parse time with a warning and the
run proceeds in default mode with the bonus token also filtered:

```
W --det-draft-intercept-all requires the standard autoregressive MTP flow. Falling back to default mode with the bonus token also filtered (target still verifies, so performance may degrade)
```

| Environment Variable | Description |
|---|---|
| `DETERMINISTIC_DRAFT_GRAMMAR_DIR` | Override directory for bundled grammar files. Defaults to `<plugin_dir>/grammars/`. |

## Modes

### Default mode

The plugin filters each draft batch: `filter_draft()` commits each accepted
token to the constraint state and stops at the first invalid token. The
target model then verifies the surviving draft tokens using standard
rejection sampling. On disagreement -- the target rejects a token the filter
accepted -- the plugin state is rolled back to the accepted prefix so the
constraint remains consistent with what was actually emitted. The final
token of each step (the correction or bonus) is also constrained through
the same plugin, so the grammar state never desyncs from the emitted output.

Available for all four drafter types: MTP, Eagle3, DSpark, and DFlash.

### Intercept-all mode

No target verification of draft tokens occurs. The filter is the sole
verifier: every token the filter accepts becomes output. The target forward
still runs every round -- what intercept-all removes is the verification
comparison (the rejection-sampling check), not the forward pass.

Intercept-all is MTP only. If a non-MTP drafter is selected, the flag is
cleared at parse time (see downgrade warning above) and the run continues
in default mode with the bonus token also filtered.

**Why MTP only:** MTP auxiliary heads run on top of the same shared backbone
hidden state, so removing rejection sampling yields a real speedup --
approximately 2x on the tested RTX 3060 setup, because grammar-vetted
drafts are accepted at near-100% and each round emits far more tokens at
the same per-round cost. DSpark, Eagle3, and DFlash encoders consume target
layer inputs that only the target forward produces each round; intercept-all
gives them no speedup, so it is disabled for them.

The filtering is applied to the same set of tokens in intercept-all and in the
non-MTP fallback - both the draft head and the bonus token pass through the
filter. What differs is whether the target model still gets a veto over the
draft head:

| mode | draft head | bonus | target verifies draft head? |
|---|---|---|---|
| default | filtered | unfiltered | **yes** |
| intercept-all (MTP) | filtered | filtered | **no** (filter is sole verifier) |
| intercept-all fallback (non-MTP) | filtered | filtered | **yes** (default + filtered bonus) |

So "the bonus is filtered" is identical in the last two rows, but draft-head
acceptance is not: intercept-all emits every filter-surviving token (nothing
is rejected), while the fallback still lets the target reject filter-survivors.
That is why intercept-all is only meaningful as a speedup when the target veto
is safe to drop - i.e. MTP.

## How the filter integrates

Drafts pass through `llama_deterministic_draft_filter_draft()` inside
`common_speculative_draft()` before verification. The function commits each
accepted token to the plugin constraint state as a side effect and stops at
the first filter-invalid token. If the filter rejects any token, the result
is truncated to the valid prefix.

The final token -- the bonus in intercept-all mode, or the correction/bonus in
default mode -- is constrained through the same plugin in
`common_speculative_sample_and_accept()`.

In default mode, after the target verification loop, the plugin state is
rolled back to the prefix the target actually accepted, using
`deterministic_draft_rollback()`. This keeps the plugin constraint
consistent with the tokens that were actually emitted.

## Relationship to MTP

MTP is reused as-is. The filter composes with the existing MTP framework rather
than replacing or forking it. `--det-draft-model` does not auto-enable
`--spec-type draft-mtp`; the user must pass a compatible `--spec-type`
explicitly. If the model lacks MTP auxiliary heads (`n_layer_nextn == 0`),
llama.cpp fails to start with the standard MTP error.

## Tests

`tests/test-deterministic-draft.cpp` registers 34 tests covering plugin
loading, capability negotiation, error paths, per-slot state isolation,
filter/commit/rollback semantics, bitmask constraint, bootstrap language
detection, and the intercept-all downgrade behavior
(`test_intercept_all_non_mtp_fallback`).
