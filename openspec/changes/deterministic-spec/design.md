## Context

Current code generation with LLMs uses GBNF grammar masking for syntax enforcement and speculative decoding (`--spec-draft-model`) for token acceleration. Neither validates structural correctness via AST nor provides deterministic error diagnostics. llama.cpp has no tree-sitter integration; it uses a custom PEG parser (`common/peg-parser.h`) and Jinja engine (`common/jinja/`). The PoC was initially implemented as a `common_sampler` extension but is being rearchitected into the `common_speculative` pipeline as a proper speculative type composing with MTP.

## Goals / Non-Goals

**Goals:**
- Guarantee structural validity of generated code via XGrammar bitmask-based grammar-constrained decoding
- Filter MTP draft tokens through deterministic structural validation; truncate at first error and let main model resume naturally (truncate + reverify)
- Integrate as `COMMON_SPECULATIVE_TYPE_DRAFT_DETERMINISTIC` speculative type composing with MTP
- Support per-token validation and draft-scale batches (Multi-Token eXchange / MTX: 100s to 1000s of tokens)

**Non-Goals:**
- Style rule enforcement (camelCase, indentation) - follow-on phase after structural validity proven
- Multi-language projects (JSX, Vue SFCs) - single-language gatekeeper for PoC
- Grammar masking replacement (GBNF remains opt-in; grammar-constrained decoder is additive)

## Decisions

| Decision | Rationale | Alternatives |
|---|---|---|
| **Clean SDK separation** | Main tree has zero domain-specific code; plugin loaded at runtime via dlopen; external artifacts distributed as header + .so | Embed XGrammar in core (bloat, coupling), single repo (no separation) |
| **`external/` directory** | Gated by `DETERMINISTIC_SPEC_ENABLED`; produces distributable header + spec loader .so with no llama dependency | Install to system paths (less portable), header-only (insufficient for real SDK) |
| **`deterministic-draft-model-poc/`** | Standalone consumer project; zero main tree dependency; simulates private repo (law firm, gov agency) | Build in main tree (violates separation), no reference implementation (harder to adopt) |
| **Fail-fast draft validation** | One error per draft stops validation; preserves valid prefix without discarding | Full-draft re-parse (slower), per-token only (higher overhead) |
| **Truncate + reverify** | Mirrors how MTP rejection already works; simplest; no fix coherence risk | Fix injection (fix may be rejected by main model), diagnostic injection (consumes context) |
| **New `COMMON_SPECULATIVE_TYPE_DRAFT_DETERMINISTIC`** | Proper speculative pipeline stage; composes with MTP; clean separation | Sampler extension (bolt-on, decoupled from MTP), filter inside MTP impl (couples logic) |
| **Auto-imply draft-mtp** | `--deterministic-draft-model` requires MTP heads; auto-adding `draft-mtp` avoids user error | Require explicit `--spec-type draft-mtp` (error-prone) |
| **Plugin state via accept() only** (revised 2026-07-18) | Superseded: `filter_draft` has commit-on-accept semantics and advances grammar state during draft filtering; `deterministic_draft_rollback()` restores consistency when the target accepts fewer tokens than committed. Reset + prompt commit happen in `common_speculative_begin()` | Advance during draft() without rollback (breaks on partial accept) |
| **Commit full prompt as code context** | Plugin needs bracket context from prompt code | Empty buffer (misses context), last code block only (requires heuristic parsing) |

## Risks / Trade-offs

- [Tokenization mismatch] Subword tokens may split UTF-8 boundaries; XGrammar rejects incomplete fragments -> **Mitigation**: BPE models preserve UTF-8; verify per target model |
- [XGrammar as new dependency] Adds C++ library + per-language grammars + build complexity -> **Mitigation**: Optional linkage (`FetchContent`); default to no grammar if unavailable |
- [Deep nesting latency] Templates/C++ nested structures increase grammar compilation cost -> **Mitigation**: Benchmark worst-case; lazy grammar loading for inactive rules |
- [Non-code content in chat templates] Chat-formatted prompts include system/user tags that may not match any grammar -> **Mitigation**: Recommend raw completion mode for PoC; plugin should handle gracefully |
- [MTP model availability] At the time of writing, Qwen3.5/3.6 and Step3 were the MTP-head architectures; llama.cpp has since added nextn/MTP tensors for more architectures (e.g. deepseek32, exaone, bailingmoe2, gemma4). Qwen3.5/3.6 also use M-RoPE which requires correct position tracking in benchmark tooling -> **Mitigation**: The bench tool position accounting bug has been fixed; Qwen3.5/3.6 are compatible with the deterministic draft filter. |

## Open Questions

- Draft size policy: fixed N vs. AST-boundary (function/class completion) - **Resolved** (flag-based)
- MISSING node tolerance at trailing edge: accept incomplete code mid-generation - **Resolved** (grammar-constrained decoder triggers on error only)
- Rule config format (YAML/TOML/DSL) for follow-on style phase - deferred; no config for phase 1 structural validation
- Serialization of grammar state for server contexts - **Resolved** (shipped: `deterministic_draft_state_get_size/get_data/set_data` in the SPI, packed into the slot checkpoint blob in `common_speculative_get_state/set_state` with a magic + u32 version header)
- Polyglot prompt parsing: how to extract code from chat-formatted prompts (deferred)
- Non-code extension: pluggable validators for non-coding tasks (future work)

## Decisions Deferred to Phase 2

- ~~Server state serialization~~: shipped (see above) - matcher state is checkpointed via the serialization SPI; a server restart still loses it (checkpoints are per-slot, in-memory)
- Grammar fallback: XGrammar only; no alternative grammar library for PoC
- UTF-8 boundary guard: BPE models preserve UTF-8; Unigram mid-character splits not in scope for PoC

## Additional Decisions

| Decision | Rationale | Alternatives |
|---|---|---|
| **Draft size flag (`--det-draft-n-max`)** | `--det-draft-n-max` sets BOTH the filter cap and MTP draft count when > 0 (auto-derives --spec-draft-n-max). Default -1 = no filter cap (MTP default of 3). | Separate flags (more complex), AST-boundary cutoff (requires heuristic) |

## Accept-All Correctness/Perf Decisions (2026-07-18)

Recorded after tracing why accept-all produced invalid C and was slower than baseline on qwen35 (Qwopus3.5-9B-Coder-MTP). These are the decisions behind the fixes; measured evidence is in benchmark-overview.md "Phase 2".

| Decision | Rationale | Alternatives |
|---|---|---|
| **Ingest prompt into draft context via `common_speculative_process`** | The bench decoded the prompt on the target but never processed it through the speculative impl, so the MTP draft head generated from an empty context (zero `pending_h`, empty draft KV) and produced garbage drafts from the first token. Spec init moved before prompt eval (so nextn embeddings are enabled in time) + prompt batch processed. This was THE correctness fix. | Re-prefill the draft context separately (position-collision risk), share KV (unsupported on qwen35) |
| **O(1) `AcceptToken` probes for the bonus token** | `FillNextTokenBitmask` cost ~1.1-1.8s per call over the ~150k vocab and dominated decode time (~2900ms/run). The bonus is now selected by probing the sampler's probability-sorted shortlist with `filter_draft` (commits on accept); sample phase dropped to ~0.5ms/run. Bitmask kept only as fallback. | Keep bitmask (too slow), per-token `AcceptToken` only for drafts (already done; bonus was the gap) |
| **Grammar-parseable prompt (newline after `#include`)** | A no-newline prompt leaves the c.gbnf matcher stuck in its `preprocessor` rule (`[^\n]* "\n"`), so it accepts any non-newline token instead of validating C. Usage requirement, not a code bug. | Make preprocessor rule newline-optional (would swallow everything), skip grammar priming (wrong validation scope) |
| **SPI vs utility split; loader renamed ServiceLoader** | Core keeps only the SPI (contract headers + basic datastructures), the `--det-draft-*` arg flags, the ServiceLoader (dlopen), and the minimal hooks to integrate/parse/execute the plugin. Plugin-internal utilities stay in the plugin. `src/llama-deterministic-draft.cpp` renamed to `...-serviceloader.cpp` to name the role. Dead `common_speculative_target_step` removed. | Full extraction out of core (breaks integration), leave generic name (role unclear) |
| **Small `--det-draft-n-max` on weak draft heads** | With a weak MTP head, large drafts (32/100) drift into grammar-valid-but-degenerate output (comment spam) that fails gcc. At n_max 16 the target re-anchors every ~16 tokens, staying coherent - valid AND fastest. Grammar validates syntax only. | Larger drafts + stronger head (unavailable), more frequent target re-anchoring (same as small n_max) |
| **qwen35 accept-all speedup is per-decode, not skipped decode** | Upstream `ctx_other` (KV sharing) is only honored for GEMMA4_ASSISTANT/EAGLE3, so qwen35's draft context keeps a separate KV/recurrent cache and the target must still decode accepted tokens to populate its own cache. The speedup comes from accepting far more tokens per target-decode, not from skipping the target decode. | Extend `ctx_other` to qwen35 (core change, out of scope for the plugin) |
