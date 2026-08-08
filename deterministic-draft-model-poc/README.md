# Deterministic Draft Filter - Plugin Framework for MTP Speculative Decoding

A plugin framework that intercepts MTP draft tokens before target verification,
filtering structurally invalid tokens using domain-specific validators loaded at
runtime via `dlopen`/`dlsym`.

This is a proof of concept. The contribution is the plugin contract and the
integration point - not the specific validator. The reference implementation
uses XGrammar for grammar-constrained decoding with jump-forward support
as a concrete demonstration of one domain. Any domain with a deterministic
correctness criterion - legal citation formats, schema validation, regulatory
constraints, structured query languages, private proprietary rules - can
implement the same contract and plug in without touching llama.cpp core.

## Background: MTP and why accept rate matters

In standard autoregressive decoding, the main output head produces one token
per forward pass - 1:1. MTP-enabled models add auxiliary prediction heads that
run on top of the same shared backbone hidden state, each drafting one
additional token. A model with 3 MTP heads produces 4 candidate tokens per
forward pass: 1 from the main head plus 3 draft tokens from the auxiliary heads
(`--spec-draft-n-max 3`).

The auxiliary heads are cheap relative to the shared backbone computation, which
is already done. The efficiency gain depends entirely on accept rate - how many
draft tokens the target accepts before rejecting. Rejected draft tokens waste
the cost of the auxiliary heads and fall back to single-token decoding.

On constrained hardware, this failure mode is severe. On the N100 in the
original benchmark configuration (n_max 100), baseline MTP accept rate was
under 1% - the auxiliary heads generated draft tokens that were rejected over
99% of the time. Tuning the draft budget recovers some acceptance, but the
underlying issue remains: there is no structural guarantee between draft and
target. The filter addresses this at the source - validating draft tokens
against a language grammar before they reach target verification, so only
structurally valid tokens consume verification slots.

## Core changes to llama.cpp

Changes to core are confined to:

- Shared data structures and headers
- Three CLI flags (see below)
- `src/llama-deterministic-draft-serviceloader.cpp` - plugin loader
  (dlopen/dlsym), feeds draft tokens from the MTP auxiliary heads to the plugin
  via the C API contract
- `include/llama_deterministic_draft.h` - consumer API header
- `include/deterministic_draft_plugin.h` - plugin contract header
- `common/speculative.{h,cpp}` - pipeline integration, including a conditional
  bypass of `common_sampler_sample_and_accept_n` when both `--det-draft-model`
  and `--det-draft-accept-all` are set

The bypass follows the same gating pattern as MTP's own conditional behaviour.
Without `--det-draft-model` the code path is identical to unmodified llama.cpp.
With `--det-draft-model` but without `--det-draft-accept-all`, the plugin
validates draft tokens before they reach target verification; the target then
verifies accepted drafts via standard rejection sampling as normal, and the
final token of each step (the rejection correction or the bonus) is
grammar-constrained before emission, so the plugin's grammar state never
desyncs from the emitted output. Only when both flags are set does core bypass
target verification entirely.

The plugin has no access to core sampling routines. It receives tokens via the
C API contract, returns a validation result, and core decides what to do with
that result based on the flags. Domain logic stays entirely outside core.

The plugin loader is always compiled into libllama, so the `--det-draft-*`
flags work in any build. `-DDETERMINISTIC_SPEC_ENABLED=ON` additionally builds
the standalone SDK artifacts (`external/`) and the benchmark tool.

## Plugin contract

The contract is capability-based. Plugins declare what they support via
`deterministic_draft_get_capabilities()` and the host degrades gracefully for
missing capabilities.

**CAPABILITY_BITMASK** - the plugin fills a bitmask of valid token IDs before
each sampling step. `deterministic_draft_filter_draft()` filters a batch of
draft tokens against the bitmask, commits valid tokens to the grammar state, and
stops at the first invalid token. This is the primary validation mechanism used
by the reference implementation.

**CAPABILITY_JUMP_FORWARD** - the plugin returns strings that are uniquely
determined by the current grammar state, allowing deterministic sequences to be
skipped without model sampling. Present in the plugin contract and implemented
in the reference implementation. Not yet wired into `common/speculative.cpp` -
integration into the MTP draft stream is non-trivial and deferred to follow-up
work.

**Rollback** - `deterministic_draft_rollback()` undoes the last N commit calls
for a given slot, keeping grammar state consistent with what was actually
emitted when the target model accepts fewer tokens than the plugin already
committed. Required for correct behaviour in default mode (without accept-all).

**Multi-slot isolation** - each inference slot is identified by a `slot_id`
parameter. Plugins maintain per-slot state internally. The server's concurrent
inference slots are correctly isolated.

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

**Plugin author** - writes a domain-specific validator, includes
`deterministic_draft_plugin.h`, implements `deterministic_draft_create`,
`deterministic_draft_filter_draft`, `deterministic_draft_commit`,
`deterministic_draft_rollback`, `deterministic_draft_reset`,
`deterministic_draft_destroy`, and compiles to a
`.so`. The only party that needs the headers. The reference implementation
(XGrammar grammar-constrained decoding) is one example; a regulated organisation's
private validator is another.

**Distribution** - the plugin author ships the `.so`. Open source on GitHub,
a private artifact in a corporate repo, or anything in between. llama.cpp
carries no opinion on how plugins are distributed.

**End user** - downloads or receives the `.so`, passes it via
`--det-draft-model ./plugin.so`, and runs llama.cpp. They never see a header.
The loader (compiled into libllama) handles `dlopen`/`dlsym` at runtime,
resolves the function pointer table, and calls into the plugin through
`common/speculative.cpp`.

The plugin has no dependency on llama.cpp internals. It receives draft tokens,
returns a validation result, and core decides what to do with that result based
on the active flags.

## Relationship to MTP

MTP is reused as-is. The filter composes with the existing MTP framework rather
than replacing or forking it. Enabling `--det-draft-model` auto-enables
`--spec-type draft-mtp`. If the model lacks MTP auxiliary heads
(`n_layer_nextn == 0`), llama.cpp fails to start with the standard MTP error.

## Demonstrated results

Single-run, indicative numbers - see [Benchmark Overview](docs/benchmark-overview.md)
for full tables, hardware details, and caveats, and
[Comprehensive Benchmark](docs/benchmark-comprehensive.md) for methodology.

| Hardware / model | Mode | Throughput | Speedup |
|---|---|---|---|
| N100 (CPU), 4B mxfp4 | accept-all, n_max 64 | 0.35 -> 5.97 t/s | 17.0x |
| N100 (CPU), 4B mxfp4 | default, n_max 16 | 2.02 -> 2.68 t/s | 1.32x |
| N100 (CPU), 2B Q6_K | default, n_max 16 | 4.38 -> 5.54 t/s | 1.26x |
| RTX 3060, 2B Q8_0 | accept-all | see overview | 8.3-29.1x |
| RTX 3060, 2B Q8_0 | default | see overview | 1.1-2.8x |

What these show:

- **Accept rate**: on the N100, the accepted-token share rises from 3.8% to
  100% in accept-all mode (4B, n_max 64), and from 17.6% / 33.6% to
  29.1% / 45.4% in default mode (4B / 2B, n_max 16) - the filter removes
  structurally invalid drafts before they waste verification slots.
- **The validity contract holds in both modes**: every emitted token passes
  the grammar. Post-fix runs record zero grammar/emission desync events.
- **Accept-all is an explicit opt-in tradeoff**: it measures grammar-only
  verification (no target-model verification) and changes the output
  distribution - appropriate where the validator is authoritative, not a
  like-for-like speculative-decoding comparison.
- **On weak CPUs, results depend on n_max tuning**: with the original
  misconfigured n_max 100 the filter is slower than baseline; the tuned
  configurations above are the correct comparison (see the N100 Tuning
  Cautionary Note in the overview).

## Usage guidance

| Flag | Description |
|---|---|
| `--det-draft-model FNAME` | Path to the plugin (.so/.dylib/.dll). Auto-enables `--spec-type draft-mtp`. Requires an MTP-enabled model. |
| `--det-draft-n-max N` | Max draft tokens to validate per step (-1=no cap, 0=disabled). When >0, also sets `--spec-draft-n-max` to N - controls the draft budget at both the filter and the MTP auxiliary heads with a single flag. |
| `--det-draft-accept-all` | Bypass target verification entirely. The filter is the sole verifier. Default: false. |

| Environment Variable | Description |
|---|---|
| `DETERMINISTIC_DRAFT_GRAMMAR_DIR` | Override directory for bundled grammar files. Defaults to `<plugin_dir>/grammars/`. |

Language selection: the reference plugin auto-detects the language from
generated content (bootstrap detection - it tries candidate grammars and
narrows as tokens commit). The benchmark tool additionally accepts
`--det-draft-language <name>` to pin a language explicitly; pinning is
recommended for strict guarantees, because while unpinned a token valid in
*any* surviving candidate language passes (see
[observations](docs/observations.md)). The server path has no language
override.

`--det-draft-accept-all` is appropriate only where the domain validator can be
trusted as the authority on token validity - single-language code-only
generation, not mixed content, markdown, or chat output.

The filter validates structural correctness only, not semantic correctness. Code
that parses as valid C may still be semantically wrong - a shadowed variable,
an incorrect algorithm, an off-by-one. XGrammar operates at the grammar
level; name resolution, type checking, and logic errors are outside its scope
and remain the caller's responsibility.

## Supported languages (reference implementation)

C, Java, Python, JavaScript

## Documentation

- [Quick Start](docs/quick-start.md) - build and run instructions
- [Benchmark Overview](docs/benchmark-overview.md) - results and usage guidance
- [Comprehensive Benchmark](docs/benchmark-comprehensive.md) - full methodology and data
- [Observations and Limitations](docs/observations.md) - known boundaries, usage requirements, resolved findings
