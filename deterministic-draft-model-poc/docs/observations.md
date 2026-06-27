# Deterministic Draft Filter: Observations and Limitations

This is a proof-of-concept, not a production-ready plugin. Its sole purpose is
to demonstrate how to implement a deterministic-draft plugin for a
specific domain (grammar-constrained code generation) against the
`deterministic_draft_plugin.h` contract - the bundled XGrammar/GBNF
grammars, the language-detection heuristics, and the fixes below are all
in service of that demonstration, not a claim that this is ready to run
unmodified in production. A real deployment would need its own grammars
(audited for its actual domain), its own decision on language
detection/selection, and resolution of the observations tracked below.

This document tracks active gaps and unresolved findings separately from
[quick-start.md](quick-start.md) so they aren't lost or silently assumed
fixed. Update this file as observations are resolved or new ones are added - do
not let it go stale.

## Upstream / Out-of-Scope Investigations

These are observations made during development that are NOT caused by this
PR's changes. They are tracked here for completeness but do not block
upstream submission.

### N100 follow-up investigation (2026-07-19): baseline MTP fails on the N100 test configuration

During the post-fix N100 re-verification, baseline MTP speculative decoding
failed on the N100 test configuration: 0.0% acceptance over 20100 drafts with
gibberish output, while plain autoregressive decode on the same host is correct
at 13.2 t/s. The failure requires the MTP draft/verify path. Root cause is NOT
isolated - candidates are this fork's spec-decode integration, the benchmark
tool, or upstream CPU behavior. A `ctx_dft pos_max < N-1` process()-hook warning
on CPU prefill was also observed and is part of the same investigation.

- Impact on published numbers: the 2026-07-19 N100 row (baseline 0.05 t/s vs
  treatment 4.31 t/s, 86.1x) is recorded as "filter + accept-all completed
  while baseline MTP did not", not a healthy like-for-like comparison. At
  n_max=16 the baseline is healthy (37.9% accept, 3/3 valid, 5.28 t/s).
- Owed: bisection across n_max, KV cache types, threads, BLAS on/off, and an
  llama-server repro outside the benchmark tool (tasks 16.3.1-16.3.3 in
  `openspec/changes/deterministic-spec/tasks.md`). Do not attribute to upstream
  without this bisection.

### qwen35 does not share the draft/target KV cache

The "accept-all means the target only computes the bonus token" ideal
requires the MTP draft head to write its K/V into the target's cache.
Upstream llama.cpp only honors `ctx_other` (KV sharing) for
GEMMA4_ASSISTANT and EAGLE3 arches; for qwen35 (and other MTP models)
the draft context keeps a separate KV/recurrent cache. So the target
must still run a full forward pass over the accepted draft tokens to
populate its own cache. The accept-all speedup on qwen35 comes from
accepting far more tokens per target-decode (fewer verification cycles),
not from skipping the target decode. Making the KV shared for qwen35
would be a core change and is out of scope for the plugin.

## Known Limitations

These are inherent limitations of the CFG-based grammar-constrained
approach, not bugs introduced by this PR. They are documented here so
users understand the boundaries of what the filter can and cannot do.

### c.gbnf's identifier-as-type permissiveness is inherent, not a bug to fix

`c.gbnf`'s `type_keyword` rule includes a bare `identifier` fallback
(any word can be treated as a type name), which lets non-C content that
merely looks declaration-shaped parse as valid C (e.g. a JavaScript
snippet using `const`/`let`/`function`). This was confirmed to be the
*industry-standard* approach, not a defect specific to this grammar -
`tree-sitter-c`'s real, actively-maintained grammar does the exact same
thing (`_type_identifier: $ => alias($.identifier, $.type_identifier)`),
because representing C as a context-free grammar at all requires it (the
alternative - real typedef-name resolution - needs a symbol table, which
neither GBNF nor tree-sitter's GLR parser without semantic help can do
without this same permissiveness).

Mitigated, not fixed: bootstrap language detection deliberately tries "c"
last in its candidate order (see `list_bundled_languages()` in
`src/plugin.cpp`), so more discriminating grammars get first chance to
correctly reject non-matching content. This reduces false positives for
detection but does not change the underlying grammar's permissiveness
when "c" is used directly (e.g. via bootstrap detection resolving to "c", or an explicit `set_language` call from programmatic/C-API use).

### Bundled grammars have not been exhaustively audited

Two confirmed gaps in `c.gbnf` were found and fixed this session (missing
integer/floating literal suffixes like `1.0f`/`100u`/`100L`; missing
parenthesized function-pointer declarators like `int (*fp)(int, int);`),
found via targeted spot-checks against the ISO C11 grammar and
`tree-sitter-c`, not a line-by-line audit. Other gaps in `c.gbnf` likely
remain unfound. `java.gbnf`, `python.gbnf`, and `javascript.gbnf` have not
been audited against their respective language specs at all this session.

### Bootstrap detection fallthrough has no bound on tie duration

While 2+ candidates remain tied (see "Bootstrap language detection" in
quick-start.md once documented there), each fallthrough attempt reloads a
candidate grammar (cheap - cache hit) and replays the full committed
token history via `AcceptToken` (linear in history length). This is fine
for the short ties the design expects, but there is no hard limit or
safeguard if a pathological input stays genuinely ambiguous across
multiple candidates for an unusually long time - replay cost would grow
with history length on every subsequent fallthrough in that slot.

### Draft-size vs coherence tradeoff on weak draft heads (accept-all)

The grammar validates syntax only, not coherence. In accept-all mode the
target model never rejects draft tokens, so a weak MTP draft head can
drift into grammar-valid-but-degenerate output (e.g. comment spam, or
`123.000f;` float-literal spam) that passes the grammar but fails
`gcc -fsyntax-only` (incomplete program, unclosed brace). Observed on
Qwopus3.5-9B-Coder-MTP (qwen35): at n_max 32/100 the whole generation is
a handful of long drafts, so the head drifts before the target can
re-anchor it. At n_max 16 the target re-anchors every ~16 tokens via the
bonus, which keeps the head coherent - output is gcc-valid AND it is the
fastest configuration (see benchmark-overview.md Phase 2). Keep n_max
small on weak draft heads; a stronger head (or more frequent
re-anchoring) would tolerate larger drafts.

### Real-language validators check static semantics a CFG cannot express

The bundled grammars are context-free (syntax only). Real validators
enforce more: `gcc -fsyntax-only` reports redefinition; `node --check`
reports `SyntaxError: Identifier 'left' has already been declared`
(`const` redeclaration); `javac` reports duplicate methods, undeclared
variables, and type errors (`new Scanner(System.out)`). These are
static-semantic early errors - they need a symbol table / type checker,
which a CFG cannot represent. So a grammar can accept output that the
real parser rejects. Observed on Qwopus3.5-9B (Phase 2): C and Python
converge (grammar-valid ~= real-parser-valid), but JavaScript and Java
diverge - treatment output is CFG-valid yet fails `node`/`javac` because
the model degenerates into redeclaration/semantic-error spam. This is
inherent to CFG-based validation, not a grammar bug - do NOT try to fix
it by adding scope/symbol tracking to the grammar (that is the whole
reason CFGs are used here). Mitigation, same as the draft-size finding:
keep n_max small so the target re-anchors and the model stays coherent
(completes a clean program) instead of drifting into redeclaration spam.

### Benchmark validity scores depend on the system gcc version

The bench validates generated C with `gcc -fsyntax-only`. GCC 14+
(observed with 15.2) turns implicit function declarations into hard
errors, so grammar-valid output that calls `printf`/`scanf` without
`#include <stdio.h>` fails validation there, while older gcc only warns
and scores the identical output valid. This shifts baseline and
treatment validity counts equally - it is a toolchain property, not a
filter regression. Seed the includes via the prompt (the documented
benchmark commands do this) and record `gcc --version` alongside any
published numbers.

### Output distribution / perplexity impact of this session's changes

**Normal speculative decoding mode (target verification enabled, the default):**
Verified draft tokens are unaffected: the target model verifies every accepted
draft token via standard rejection sampling, so those tokens follow the target
model's true distribution regardless of draft/grammar-filter changes. The one
exception is the final token of each step (the rejection correction or the
bonus): it is grammar-constrained before emission (see "Unconstrained final
token" in Resolved below), so it follows the grammar-constrained target
distribution rather than the unconstrained target - the same class of effect
as llama.cpp's own `--grammar` sampling, and deliberate: it is what makes the
every-emitted-token-passes-the-grammar contract hold. With that scope, this
session's changes (per-slot grammar isolation, bootstrap language
auto-detection, the fail-open-to-fail-closed termination fix, c.gbnf grammar
corrections, per-token allocation removal, benchmark loop fixes) affect only
draft acceptance rate, the final-token constraint, and generation speed.

**Accept-all mode (`--det-draft-accept-all`):**
This is a pre-existing feature, predating this session, with a known and
explicit tradeoff: draft tokens are accepted without target-model rejection
sampling, so the grammar filter becomes the sole verifier and the output
distribution follows the grammar-constrained draft rather than the
unconstrained target model. This session's changes to the filter (bug fixes
and grammar coverage improvements) make the filter more correct in this mode,
never less correct, and do not alter the fundamental accept-all tradeoff,
which remains an explicit user opt-in and is out of scope for this work.

### Bench validation inherits gcc's warning policy

`validate_output()` in the bench uses `gcc -fsyntax-only`, and constructs
like extra tokens after an `#include` directive (e.g.
`<string.h>mbString.h>`) are only *warnings* - exit code 0, so such runs
are judged VALID. "Grammar-filtered valid" numbers therefore mean "no gcc
errors", not "strictly clean C"; a grammar-validity ground truth requires
re-feeding the emitted text through the pinned plugin itself.

## Usage Requirements

These are requirements for correct operation of the plugin, not bugs.

### Prompt must be grammar-parseable (newline after #include)

The grammar matcher is primed with the prompt, so the prompt must be
valid input for the grammar. A C prompt without a newline after the
last `#include` line (`#include <stdio.h> int main(void) {`) leaves the
c.gbnf matcher stuck mid-`include_directive` waiting for the line to
end, so every continuation token is rejected and drafts are truncated to
zero until a newline finally arrives. Use a newline-terminated prompt
(`#include <stdio.h>\nint main(void) {`). This is a usage requirement,
not a code bug - see the note in quick-start.md.

## Resolved (Phase 2, 2026-07-18)

- **Prompt never ingested into the MTP draft context**: the bench decoded
  the prompt on the target but never called `common_speculative_process`
  on it, so the draft head generated from an empty context (zero
  `pending_h`, empty draft KV) and produced garbage drafts from the first
  token (symptom: drafts like `123.0000:...` immediately after a clean
  prompt). Fixed by initializing speculative before the prompt eval (so
  nextn embeddings are enabled in time) and processing the prompt batch
  through the speculative impl. This was THE correctness fix - after it,
  drafts are coherent C (`int i = 0; if (...) ...`).
- **Bonus verification used O(vocab) `FillNextTokenBitmask`**: the bonus
  token was constrained via `FillNextTokenBitmask`, which cost ~1.1-1.8s
  per call and dominated decode time (sample/accept phase ~2900ms per
  run). Replaced with O(1) `AcceptToken` probes over the sampler's
  probability-sorted shortlist (highest-probability grammar-valid token
  wins), keeping the bitmask only as a fallback. Sample phase dropped to
  ~0.5ms. This was the main accept-all speedup blocker.
- **Empty-draft fallthrough bypassed the grammar**: in accept-all mode, a
  fully-rejected draft fell through to unconstrained sampling (the token
  bypassed the "grammar is the sole verifier" guarantee). Now the single
  emitted token is grammar-constrained via the same fast path.
- **Bonus double-commit in accept-all**: the bonus token was committed
  both during selection and again in `common_speculative_accept`. It is
  now committed exactly once (during selection); `common_speculative_accept`
  skips the accept-all bonus commit.
- **Loader renamed to ServiceLoader**: `src/llama-deterministic-draft.cpp`
  renamed to `src/llama-deterministic-draft-serviceloader.cpp` to name its
  role (dlopen/dlsym of the SPI). Build references updated.
- **Removed dead `common_speculative_target_step`**: it was declared and
  defined but never called, and its shared-KV shortcut is inapplicable to
  qwen35.
- **python.gbnf assignment-target allowed the call trailer**: `target ::
  identifier trailer*` and `trailer` includes `"(" arglist? ")"`, so
  `self.dfs(key) = []` parsed as a valid assignment (a Python syntax
  error - a call is not a valid target). Grammar-filtered scored 0/5 valid
  because of it. Fixed by restricting the target trailer to attribute
  (`.id`) and subscript (`[expr]`) only; treatment then scored 5/5 valid
  (caught 2 baseline-invalid runs).

## Resolved this session (for reference - do not re-investigate)

- **`filter_draft` fail-open bug on bitmask-fill exception**: if
  `FillNextTokenBitmask` threw (e.g. matcher already terminated), the
  `catch` block only logged and left `token_valid` defaulted to `true`
  (fail-open), then called `AcceptToken` on an already-terminated matcher
  anyway, producing a XGrammar warning - once per remaining token in the
  batch, with no `break`. Fixed: fail closed on that exception, and break
  immediately once `IsTerminated()` becomes true, in both the bitmask-fill
  path and the normal accept path.
- **Missing termination signal in the plugin contract**: added
  `deterministic_draft_is_terminated(state, slot_id)` to
  `deterministic_draft_plugin.h` (optional, gracefully degrades to
  `false` for plugins that don't implement it, matching the existing
  pattern for `rollback`/`set_language`/etc.), wired through the loader
  (`src/llama-deterministic-draft.cpp`), exposed via
  `common_speculative_is_terminated()` in `common/speculative.{h,cpp}`,
  and used in `bench-deterministic-draft.cpp`'s generation loop to stop
  as soon as the grammar completes, the same way `has_eos` already does.
- **Per-slot grammar was plugin-wide, not per-slot**: `set_language()`/
  `set_grammar()` ignored the `slot_id` the header already documented as
  supporting "multi-language / polyglot support" - calling it for one
  slot silently reassigned the grammar for every slot. Fixed by moving
  `compiled_grammar`/`current_language` into `SlotState`.
- **Empty-draft crash in accept-all mode**: `common_speculative_sample_and_accept`
  returned an empty vector when the very first draft was empty and the
  bonus-token bitmask also failed, tripping a caller-side
  `GGML_ASSERT(ids.size() > 0)`. Fixed by falling through to normal
  sampling when the draft is empty.
- **Permissive `preprocessor` rule in c.gbnf**: the old rule
  (`"#" ws [a-zA-Z_][a-zA-Z0-9_]* [^\n]* "\n"`) accepted anything after `#`
  word-characters, so accept-all mode emitted garbage like
  `#include <ctype backing in to include <stdlib.h>` and it passed the
  filter (found on the N100 0.8B/2B runs, 2026-07-23). Fixed by spelling
  out each directive: `include` requires `<` path `>` or `"` path `"` with
  a filename-shaped body, `define`/`undef`/`ifdef`/`ifndef` require an
  identifier, `error`/`warning`/`pragma`/`line`/`if`/`elif` keep loose
  single-line bodies, and directives can no longer span lines (horizontal
  whitespace only inside them). The   `type_keyword ::= ... | identifier`
  permissiveness above remains the known way grammar-valid but
  semantically-wrong declarations (e.g. `fxd = 0x ...`) get through.
- **Unconstrained final token in standard (target-verified) mode**: the
  last token of every step (the rejection correction or the bonus) was
  sampled from the target distribution with no grammar check and only
  committed to the plugin post-hoc in `common_speculative_accept()`. When
  the plugin rejected it, the token had already been emitted, and the
  grammar state desynced from the actual output by one token - every
  later filter decision then ran against a stale matcher, so garbage
  passed the filter from then on (found on the N100 0.8B standard-mode
  runs, 2026-07-25: `commit_tokens ... REJECT` in the debug log right
  before invalid output). Fixed by constraining the final token before
  emission in `common_speculative_sample_and_accept()`, reusing the same
  helper as accept-all mode (`common_speculative_sample_det_token()`:
  shortlist `filter_draft` probe, full-vocab bitmask fallback, skip when
  terminated), and rolling the grammar back to the accepted draft prefix
  *before* validating the correction instead of post-hoc in
  `common_speculative_accept()`. This affects every standard-mode caller
  (server, bench, speculative-simple), not just the bench.
- **Bootstrap detection validates against a union of languages**: while
  no language is pinned, bootstrap keeps every bundled grammar whose
  candidate still accepts the committed history, and a token valid in
  *any* surviving candidate passes. An all-`#include` C prompt keeps the
  python candidate alive (comment lines), and python happily accepts
  C-shaped garbage like `<string.h>le` or `le = 3000000000;`. Mitigated
  in the bench with `--det-draft-language <name>` to pin the language;
  the server path remains unpinned by design.
- **Per-token heap allocation + exception-path leak in `filter_draft`**:
  `filter_draft()` in `plugin.cpp` allocated a fresh ~16KB
  `std::vector<int32_t>` bitmask buffer and a separate `new int64_t[1]`
  shape allocation on *every* draft token before calling
  `FillNextTokenBitmask`, instead of reusing a pre-allocated buffer. This
  was both allocation overhead and a leak on the exception path (the
  `int64_t[1]` was never freed). Fixed by reusing the slot-scoped
  `slot.bitmask_scratch` buffer and a stack-local shape variable,
  eliminating both the per-token allocation cost and the leak. This was a
  primary contributor to the near-termination stall (see "MTP draft head
  overhead" below).
- **Benchmark tool dead-code progress guard + wasted post-termination work**:
  in `bench-deterministic-draft.cpp`, the progress guard checked
  `draft.empty()` *after* `draft.clear()` had already run, so it never
  fired; fixed by capturing the draft-empty-this-iteration state before
  clearing. Also added an early exit when grammar termination is detected
  (via `common_speculative_is_terminated`), skipping a wasted
  target-decode/verify cycle that previously ran even after termination was
  already known. This removed the indefinite stall in `--compare` runs.
- **Removed `DETERMINISTIC_DRAFT_LANGUAGE` env var and `--lang` CLI flag**:
  both were removed entirely, leaving grammar selection to the plugin's
  bootstrap language auto-detection. A bench-only `--det-draft-language
  <name>` flag was later added to pin the language explicitly (see the
  bootstrap-union entry above); the server path has no override.

- **c.gbnf `type_specifier` unbounded repetition**: `c.gbnf`'s
  `type_specifier` rule allowed the type-keyword sequence to repeat without
  bound, so a construct like `long long long long int` would parse as a
  valid (if degenerate) type specifier. In accept-all mode this produced
  degenerate output - the grammar accepted an effectively infinite run of
  repeated type keywords, which the structural filter could not reject because
  it was grammatically well-formed. Fixed by bounding the repeated-keyword
  sequence to at most 3 repetitions (matching the realistic maximum of
  `long long long` for `long long int`/`long long double` plus one spare),
  so the degenerate "long long long long..." behavior no longer occurs.
  Note: fixing this bug did NOT make accept-all output fully valid. The
  remaining invalid output is a separate, pre-existing cause (the
  identifier-as-type permissiveness described in Known Limitations above).

- **MTP draft head overhead near grammar termination**: the indefinite
  stall near grammar termination was caused by per-token allocation and
  benchmark-loop bugs (fixed above). The residual overhead was caused by
  the O(vocab) `FillNextTokenBitmask` call (replaced with O(1)
  `AcceptToken` in Phase 2, see "Resolved (Phase 2, 2026-07-18)" above).
  The only remaining sub-question - whether there is inherent MTP
  draft-head overhead beyond these fixes - is upstream/out-of-scope for
  this plugin (would need instrumenting `common_speculative_draft()`
  directly).
