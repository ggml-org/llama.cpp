---
name: code-review
description: Review llama.cpp changes against project conventions and common reviewer pitfalls before a PR. Use when the user wants to review a diff, branch, or PR.
---

# Review llama.cpp changes

This skill reviews changes against llama.cpp's conventions and the pitfalls that reviewers flag most often, so the contributor can fix them before a maintainer has to. It has two modes:

- **Self-review (default):** review the contributor's own local changes (uncommitted work, or a branch vs `master`) as a pre-PR pass. Ask which if it's ambiguous; default to `git diff master...HEAD` plus any uncommitted changes.
- **Read-only review of a PR/file:** if the user points at a PR number or specific files (including code they didn't write), review those and report findings.

In both modes the output is **private review notes for the user to read and act on** - it is never something to post. This is a hard rule from `AGENTS.md`: an agent must NEVER write, or help write, a PR comment, a review comment, or a reply to a reviewer, by any means including `gh`. Do not offer to. If the user asks you to post the notes, refuse and point them at that rule. Present findings in the conversation only.

Before starting, read `AGENTS.md` and `CONTRIBUTING.md` if not already in context - the "Coding guidelines", "Naming guidelines", and AI usage sections are the baseline this review enforces. For a diff that adds a new model architecture, also read `docs/development/HOWTO-add-model.md` and consider the dedicated `add-new-model` skill.

## Step 0 - Scope the diff and pick the checklists

Identify what actually changed and which area checklists below apply. Run `git diff --stat` (or `gh pr view <n> --json files` for PR mode) and bucket the touched paths:

- `conversion/`, `gguf-py/`, `src/models/`, `src/llama-arch.*` -> **New model / architecture**
- `ggml/` (any backend, op, or `ggml.h`) -> **ggml / backend**
- `include/llama.h` and other public headers -> **Public API**
- `tools/server/` -> **Server**
- `tools/ui/` (or the legacy `tools/server/webui/`) -> **WebUI**
- `tests/`, `tools/server/tests/`, or any other added/changed test -> **Tests**
- anything else, plus all of the above -> **General** (always runs)

Always run the **Scope and quick-reject gate**, the **Security review**, and the **General** checklist. Run each area checklist whose paths were touched. Additionally, if the diff introduces a new component, subsystem, or piece of infrastructure (a new file/class/module, a new abstraction, or hand-rolled machinery), run the **Approach and design** review. Tell the user which checklists you're running and why.

## Scope and quick-reject gate (always)

These are the patterns that get PRs closed without a full review. Check them first - a finding here is more important than any code nit, because it can mean the change shouldn't be a PR in its current form at all.

- Is there a prior issue/discussion for this? Features are supposed to start as an issue, not a PR (`CONTRIBUTING.md`). If this is a nontrivial feature with no linked issue, flag it and suggest opening one first.
- Is it a duplicate of existing/in-flight work? Suggest `gh search prs` / `gh search issues` for the feature. Many closed PRs were duplicates of something already queued.
- Is it self-contained and single-purpose? Multiple unrelated changes/optimizations bundled together get sent back to be split. Flag unrelated changes and suggest separate PRs.
- Does it touch multiple ggml backends at once? Initial support should be CPU-only, other backends as follow-ups (`CONTRIBUTING.md`). Flag CUDA/Metal/Vulkan/etc. changes bundled into a feature's first PR.
- Does it add a new `ggml_type` / quantization type? That carries a disproportionate maintenance burden and needs the full justification package (GGUF sample upload, perplexity vs FP16/BF16 and similar sizes, KL-divergence data, CPU perf numbers). Absent that, it will be rejected regardless of code quality.
- Is it invasive - new subsystem, core-API reshaping, changes to shared graph/sampler code that other models don't need? Flag it and suggest a discussion with maintainers before investing further.
- Does it add a model-specific CLI argument to any binary (`common/arg.cpp`, tools, examples) or conversion script (`convert_*.py`)? Not allowed - model-specific behavior must come from GGUF metadata or be detected automatically, not from a per-model flag.
- Is it niche/vendor-specific in a way that adds a maintenance burden nobody will own long-term? Flag the maintenance-ownership question.
- Is the change semantically correct, or a plausible-looking "fix" that misunderstands the code? Sanity-check the actual behavior, not just that it compiles.
- AI-disclosure: if AI meaningfully contributed, is the PR template's disclosure section filled in? Remind the user. Never suggest writing the PR description or commit message for them.

## Security review (mandatory)

Mandatory on every review; any finding here is **blocking**. Rule of thumb: GGUF metadata, tensor shapes, tokenizer/grammar input, and all server/RPC fields are attacker-controlled - bound them before use.

- **Sizes/counts from tensor dims:** validate before allocating. Products like `ne[i]*nb[i]`/nbytes can overflow on crafted dims into an undersized alloc then heap overflow. Overflow checks must run BEFORE the arithmetic they guard - padding/alignment macros wrap to 0 near `SIZE_MAX`, so a guard after the pad passes.
- **GGUF strings/arrays:** cap declared lengths and element counts before using them to size a loop or buffer; validate element type and length before casting an array to a pointer or reading fixed indices (`[i+1]`, `[0..2]`).
- **Element-type confusion:** casting `gguf_get_arr_data()` or `tensor->data` to `float *`/`int32_t *` needs an element-type check first (`gguf_get_kv_type() == GGUF_TYPE_ARRAY` then `gguf_get_arr_type()`; `type == GGML_TYPE_F32` for tensors). A `UINT8` array or `I8` tensor passes every length check, then gets read 4 bytes per element - a nearby length check is not a type check.
- **Loaders:** `GGML_ASSERT` on a file-derived value aborts the process; throw instead where the caller already catches (vocab, model loader, clip).
- **File-supplied counts indexing fixed arrays:** bound any count (e.g. layer/block count into a `LLAMA_MAX_*` array) before indexing; watch checks that only fire when an optional key is present.
- **Declared vs actual array length:** check the declared length of a GGUF array against the count actually read, not just against a buffer size.
- **Bounds comparisons:** flag narrowing casts (`size_t`->`int32_t`) and signed/unsigned mixing that can bypass a length check and copy past a buffer.
- **Parsed/derived indices:** range-check `stoi`/`atoi` results and catch parse throws; never use a default or derived token id (EOS/BOS/...) as an index without a bounds check.
- **Reused/reserved buffers:** recheck bounds after a buffer is shrunk or reused; watch `reserve()` then index-by-assumed-size, and header fields read before their length is checked.
- **Server JSON ints:** clamp client-supplied integers (token/discard counts, offsets) to non-negative and an upper bound before they reach index/pointer arithmetic.
- **RPC-deserialized fields:** treat every field (type/buffer/data/ne/nb/op_params) as hostile - validate before use. Null/zero buffers skipping validation, attacker data pointers, out-of-range type indices, and negative strides sign-extending past a corner-only assert all give arbitrary read/write.
- **Recursion on untrusted input:** a function that recurses once per level (or per char) of request JSON, model output, a grammar, a template or a regex pattern overflows the stack - the HTTP threads have 512 KB on macOS, so a few hundred levels are enough. Every such function needs its own depth bound, checked in the function itself, not a pre-walk at a caller (the next caller skips it): libcommon uses `COMMON_RECURSIVE_LIMIT` (`common/common.h`), jinja `JINJA_RECURSIVE_LIMIT` (`common/jinja/utils.h`), the grammar `GRAMMAR_RECURSIVE_LIMIT` (`src/llama-grammar.h`). Count the levels of the input, not the frames, and throw `std::invalid_argument` for request input so the server answers 400. Look for recursion the code does not spell out: mutual recursion, recursive `std::function` lambdas, visitor/virtual dispatch, `std::regex` (libstdc++ recurses per char of the input), and per-char helpers that recurse where a loop would do.
- **Lifetime/UAF:** flag stored raw pointers to caller/temporary storage, cached pointers to buffers a later free releases, async ops whose source may drop before completion, and structures not invalidated on free/realloc. Null-check conditionally-built or "not required" tensors before dereferencing.

## Approach and design (when a new component/infra is introduced)

Run this whenever the diff adds a new component, subsystem, or piece of infrastructure. Reviews too often stop at "does it work" - a diff can be correct and still be the wrong approach, and a messy design costs more long-term than a bug. Evaluate the *approach*, not just the behavior; raising a cleaner one is a high-value finding, not a nit. If you see a better design, describe it concretely rather than just calling the current one bad.

- **Simpler approach upstream:** the biggest win is often a different data model or design that removes whole subsystems, not tweaks to the code as written. Complexity must be justified by the problem, not by the first thing that worked.
- **Reuse over reinvention:** grep for an existing helper, library, object, or mechanism before adding a new one. Reimplementing what the codebase already has reintroduces solved bugs and adds maintenance surface.
- **Clear ownership/lifetime:** prefer RAII and obvious ownership over manual liveness flags, hand-tracked pointers, and "is it still alive?" checks - manual lifetime tracking is a recurring source of subtle bugs.
- **Right-sized machinery:** flag redundant, overkill, or heavier-than-needed primitives and abstractions; use the minimum the design actually needs.
- **Right structure and fit:** a new type should earn its place (split it if it serves two roles); follow existing patterns, idioms, and naming, and avoid constructs the project shuns.
- **Root cause vs symptom:** fixes layered on fixes signal a design to correct, not guard around.

## New model / architecture

See the `add-new-model` skill and `docs/development/HOWTO-add-model.md` for the full workflow; this is the review-time subset that reviewers most often catch:

- Don't branch on `model.arch` when the real dependency is a config/capability value - gate on the hparam/capability, not the architecture enum.
- If the model is a close variant of an existing arch, is the delta justified? Prefer reusing or subclassing the existing arch/model class over duplicating it. A near-duplicate class or `src/models/<name>.cpp` will be asked to merge with its sibling.
- New tensor names go through `tensor_mapping.py`, not ad-hoc name matching.
- For QKV, split the *activation* with `ggml_view`, not the *weight* tensor; rely on ggml broadcasting instead of manually duplicating tensors.
- New graph inputs are declared at the top of the graph-build function, not inline where first used.
- Don't end a text model graph with `ggml_build_forward_expand(gf, cur)` on `t_logits`/`t_embd` - it's redundant, since `llama_model::build_graph()` expands them via `build_logits()` and `build_pooling()` (and skips the LM head for pooled embeddings).
- Hparams that the model can't run correctly without must be mandatory (hard-error if missing), not read with a silent default fallback. Only genuinely-optional-across-configs values get a fallback accessor.
- New/optional weight tensors (scales, etc.) must route through `build_lora_mm` and the existing helpers, matching convention - don't leave raw matmuls copied from another arch.
- Don't hack RoPE with a custom sin/cos implementation. If `ggml_rope_ext` genuinely can't express it, that's an issue for discussion, not a PR.
- Test the quantized-KV path (`-ctk`/`-ctv q8_0`), not just default f16 - new speculative/attention features silently break there.
- Preserve existing explanatory comments about model-specific quirks when copying code; note the provenance ("copied from X, with Y added").
- Remove dead code/branches left over from adapting a reference implementation.

## ggml / backend

- `supports_op` (and any dispatch/gating condition) must be scoped exactly to the cases being changed - a condition meant for a few quant types must not silently disable or enable everything else.
- No hardcoded warp/lane size - use `ggml_cuda_get_physical_warp_size()` (32 on CUDA, 64 on HIP/ROCm) and the portable helpers.
- Strip leftover debug/profiling/logging code before review.
- New or changed op? Update `docs/ops.md` and the relevant `docs/ops/*.csv` for the touched backend.
- New op or operator change needs corresponding `test-backend-ops` cases, and (per `CONTRIBUTING.md`) consistency across at least two backends.
- New kernels are expected to come with concrete perf data (throughput across realistic tensor shapes), not just correctness.
- Don't have a backend mutate the cgraph as a shortcut - that's an unresolved architectural question, not something to slip in.
- Expect this to need two maintainer approvals; that's normal for `ggml/` changes, not a sign something is wrong.
- For CUDA: Avoid excessively templating kernels, only add this where it shows visible performance gain.

## Public API (`include/llama.h`)

Public API changes carry a higher bar than internal ones (`CONTRIBUTING.md`). Review for:

- Justification: why doesn't an existing mechanism (e.g. `cb_eval`, existing batch/sampler knobs) suffice? If it does, the change likely shouldn't add public surface. This is the single most common reason these PRs are rejected.
- Experimental or stop-gap surface belongs in a side header (`llama-ext.h`), not in `llama.h`.
- Keep it minimal and general: prefer one general call over several narrow convenience wrappers; make new calls forward-compatible (e.g. mixed-modality batches) rather than assuming today's shape.
- The C API is the first-class, stable, ABI-defining surface - don't propose a parallel C++ API as a replacement. `llama-cpp.h` stays a thin convenience layer.
- Types and naming: sized integer types (`int32_t`, `size_t` for sizes/offsets); `snake_case`; `<class>_<method>` = `<class>_<action>_<noun>`; enum values upper-case and prefixed with the enum name; `_t` suffix for opaque types. Avoid gratuitous signature/ABI changes to existing exported functions.
- Every new API needs a working example/tool exercising it in the same PR - reviewers find real bugs by requiring it to be wired into `server`, `embedding`, `perplexity`, etc.

## Server (`tools/server/`)

- Is the feature within server's defined scope? Check `tools/server/README-dev.md` - out-of-scope features get declined.
- Security: don't trust client-supplied headers (e.g. `X-Forwarded-For`) or add footguns; things like IP allowlisting belong at a reverse proxy unless there's a trusted-proxy design.
- Wire new behavior into the existing request/response and checkpoint paths correctly; watch for resource leaks across requests.

## WebUI (`tools/ui/`)

These apply only to changes under `tools/ui/` (formerly `tools/server/webui/`) - the SvelteKit + Svelte 5 frontend. They are frontend conventions, not repo-wide ones, so do not apply them to C++ or other tool code.

The dominant review theme in this area is placement: most accepted nits are about putting a value, function, or piece of logic in the layer where it belongs.

### Constants and enums - decision rules

Trigger: a literal that is reused, domain-significant, or crosses a boundary (wire/persisted format, selector, key, shared UI text). One-off component-local values stay inline (see the boundary at the end of this section) - flagging those is noise. Before adding, grep `$lib/constants` / `$lib/enums` - reuse beats a new near-duplicate. What becomes what:

- **String enum** (`$lib/enums`, `enum X { KEY = 'value' }`): a closed set of categorical values the code branches on, compares, or stores as a type discriminator - roles, statuses, kinds, types, actions, sort fields. Anything written as `x.type === 'some-string'` or a closed string union in a `.d.ts` is an enum. Keys are SCREAMING_SNAKE with a domain prefix (`ModelDownloadStopRequest.CANCEL`); values are lowercase by default and are the persisted/wire representation, so they may intentionally differ from the key (`TOOL_USE = 'tools'`, `MTP = 'mtp'`) or mirror an external API exactly - say so in the enum's JSDoc when they do (see `HfModelSort`). No numeric-valued enums.
- **Constant object keyed by enum** (constants): a lookup/mapping - icon per file type, label per capability, HTTP code to message (`FILE_TYPE_ICONS`, `HTTP_CODE_TO_STRING`, `Record<Enum, string>`). The enum enumerates, the object maps.
- **Declarative config array/object** (constants): a set of related items consumed by more than one renderer (e.g. dropdown + sheet) - menu items whose `action`/`id`/`enabledWhen` fields are enum values, not strings (`ATTACHMENT_FILE_ITEMS`). If two surfaces render the same menu, both must render from this one config.
- **Grouped `as const` object** (constants): related literals that belong to one feature - error-message taxonomy (`ERROR_MESSAGES` with `HTTP`/`NETWORK` groups), UI copy shared across surfaces (`ATTACHMENT_MENU_TEXT`), a feature's prompt + fallback + patterns (`TITLE_GENERATION`). Prefer one grouped object over scattering same-feature constants.
- **Scalar constants**:
  - strings: CSS class strings, selectors, `data-*` attribute names, header names, storage keys, API routes/paths, file extensions.
  - numbers: any threshold, timeout, size, limit, or count - suffixed with unit or context (`_MS`, `_PX`, `_BYTES`, `_CHARS`, `_MAX_LENGTH`).
  - regexes: any non-trivial literal regex, `_REGEX` suffix, grouped in the feature's constants file.
  - characters/separators: named by meaning in their domain, not by character - `PATH_SEPARATOR`, `ORG_SEPARATOR`, `SSE_RECORD_SEPARATOR`, `ISO_DATE_TIME_SEPARATOR`, `NEWLINE`. No universal `SLASH` constant.
- **Key-name constants**: strings used as keys into another object or persisted config get an object constant mirroring the property names (`SETTINGS_KEYS.CUSTOM_CSS = 'customCss'`), so a rename breaks the build instead of silently missing.

Placement and composition:

- Shared across features -> `$lib/constants/<topic>.constants.ts` / `$lib/enums/<topic>.enums.ts`; feature-dedicated values may live in the feature folder, but only what is truly dedicated to it.
- Compose, don't duplicate: constants may derive from other constants/enums (`SANDBOX_TOOL_NAME = BuiltInTool.BROWSER_RUN_JAVASCRIPT`, storage keys built from the `STORAGE_APP_NAME` prefix).
- Tests and Storybook stories use the same constants/enums for structural assertions - they don't re-declare the literals they assert against. Exception: persisted/wire-contract assertions keep independent literals - there the literal value itself is the contract, and reusing the production constant would make the test tautological (change the constant, the expectation changes with it, the regression still passes).

What stays inline: one-off user-facing copy inside a single component, console/log messages, and import specifiers. When inline copy gains a second consumer, it graduates into a grouped constant object.

Adjacent data-shape rule: empty string is not a valid "unset" initial state - use `undefined`.

### Placement and layering

- Components render; they don't own business logic. Simple presentation-local derivation stays in the component as `$derived` from props and store values - what leaves is shared business logic, multi-store orchestration, and naming/parsing logic that a store method, `$lib/utils` function, or `$lib/hooks` hook should own. A component should call `store.doSomething()` rather than reimplement it.
- Keep the layers deliberate: stores (`*.svelte.ts`, stateful), services (stateless clients), utils (pure functions), hooks (`*.svelte.ts`, reusable reactivity), `$lib/types` (types), `$lib/constants` (constants). If a new file blurs these, flag it and name the layer it belongs in.
- A stateful "service" is misnamed - it's a store or belongs in the scope that uses it. Conversely, a one-off `.svelte.ts` module (which is effectively a store) for logic only one component uses is overkill; keep it in that component or a context.
- Before introducing a new abstraction (store/service/hook/context), ask whether it will ever have a second consumer; if not, keep the logic in the existing scope. KISS first, iterate later.
- Barrel files (`index.ts`) match the existing barrel style - file-level JSDoc plus per-export group comments for stores, plain re-export lists for constants/enums - and add JSDoc only where an export isn't self-explanatory; constants and types are not re-exported from store barrels.

### DRY and dead code

- Reuse before writing: grep for an existing helper, component, or store method before adding a new one (truncation, copy/download, file preview, sanitization, and formatting helpers all already exist).
- Sibling variants (e.g. Dropdown vs Sheet versions of the same picker) must render from one shared config / single source of truth, not duplicate markup and logic.
- Flag dead code, commented-out markup, and leftovers from renamed or removed features. Intentional-but-temporary leftovers get `@deprecated` JSDoc so they can be found and removed later.

### Components and Svelte 5

- Break bloated components into focused sub-components (plus dedicated `*Skeleton.svelte` loading states); use `{#snippet}` for repeated in-component markup.
- Use the existing `$lib/components/ui` primitives instead of hand-rolling dialogs, tooltips, inputs, or dropdowns.
- Rune syntax only: `$props()` deconstruction, `$derived` (reach for `$derived.by` only when a simple expression won't do), `$state`. Avoid `$effect` for things `$state`/`$derived` can express - a wall of `$effect` is a design smell.
- Handle window/document events via `<svelte:window>` / `<svelte:document>`, not manual add/remove-listener plumbing.
- Svelte 5 supports arrays in the `class` attribute - don't reach for `cn()` for simple conditional classes.
- Use `SvelteMap`/`SvelteSet` for reactive collections.

### Comments

- Same bar as the General checklist, enforced harder here: no comments that restate the code, no verbose JSDoc on obvious functions, no commented-out code. Examples and edge-case enumerations live in unit tests, not prose; non-obvious why/rationale comments stay, because a test cannot carry that context at the code location.

### Types, naming, and style

- `===` over `==`, `includes()` over index checks, no redundant null-coalescing after an `Array.isArray` check.
- One spelling per concept across the codebase (e.g. pick `cwd` vs `currentWorkingDirectory` and apply it everywhere); explicit, self-describing names over clever ones.

### Performance (streaming UI)

- Anything running per streamed chunk must be cheap: throttle before computing, not after; apply block-level regex passes globally and scan lines backwards with early exit; don't re-parse the whole message per update.
- Smooth height/expand transitions consistent with the rest of the UI - animate auto-height via `interpolate-size: allow-keywords` with a `grid-template-rows: 0fr/1fr` fallback for browsers without it (see `CollapsibleRegion.svelte`).

### Security (frontend side)

- Sanitize anything rendered from model or remote data: the rehype/`{@html}` pipeline, SVG content, and URLs (use the `$lib/utils` URL sanitizer rather than ad-hoc checks or a third-party lib for trivial cases).
- Proxy headers and agent/tool filesystem access are security boundaries - scope enforcement (e.g. browse-root sandboxing) must be validated end-to-end, not just in the UI; loop in server reviewers for the C++ side.

### PR hygiene specific to `tools/ui`

- Build artifacts are not committed - the UI is built by CI (`.github/workflows/ui-build.yml`) and downloaded/bundled at release time. `npm run build` must still succeed locally; a red UI build job is blocking.
- `npm run lint && npm run check` (and `npm run format`) must pass; a failing Storybook test is blocking.
- PRs with user-facing interactive or visual changes need a short video demo in the description showing the change working (screenshots are acceptable for pure-visual tweaks). Test-only, refactor, docs, and config PRs don't need a demo.
- Scope discipline: changes that belong to another stacked PR don't ride along in this one - call them out and move them there. Flag unrelated edits to deprecated/legacy files and ask whether they're intentional.
- New nontrivial utils and parsing logic get Vitest unit tests; reusable UI components get Storybook stories/tests when practical.

## Multimodal (`tools/mtmd/`)

- Tensor names must be prefixed by `v.`, `a.`, `mm.` or `a.mm.` (legacy naming doesn't follow this convention - this is expected, but new code should follow it).
- Do not use explicit sin/cos for RoPE; use `ggml_rope_ext` instead, see `HOWTO-add-model.md`. If it can't express the needed behavior, that's a design discussion, not a PR.
- New GGML ops must not be introduced in the same PR, you must push it as a separate PR.
- In most cases, `build_vit` should be enough to build the transformer graph for vision models. Do not add a loop to build the transformer graph manually, unless you have a very good reason to do so. If you do, please explain why in the PR description.
- If you need a dedicated preprocessor, there is a high chance that it can be a derived class from one of the existing preprocessors. Check carefully before adding a new preprocessor class.
- If the model need a new public API in `mtmd.h`, open a discussion first.
- For audio generation models, see `tools/mtmd/README-dev.md`

## Tests

- Follow the existing testing patterns. Do not add a new testing system. Before adding a new file under `tests/*`, think carefully about whether the tests can go into an existing file first.
- In most cases, new test cases belong in an existing test file - check for one covering the same component before adding anything new.
- Only add tests that bring meaningful results. Too-trivial tests just bloat the suite and CI.
- No time-sensitive tests (timing thresholds, sleeps, races against wall-clock); they are flaky on CI.
- Think twice about tests that significantly increase CI run time (expensive computation, large inputs, or long sleep/wait delays) or download large amounts of data from the internet (big models, datasets). Flag them and ask whether a smaller model/input or an existing fixture would do.
- Server: small tests of internal components (helpers, `server_tokens`, queues, format converters, ...) belong in `tests/test-server-component.cpp`. The end-to-end pytest suite in `tools/server/tests` is costly in CI time; only add to it when the test requires end-to-end behavior with a real model.

## General (always)

Enforce the `AGENTS.md` / `CONTRIBUTING.md` coding and naming guidelines on every changed line - this is a distinct pass from checking that the code works, and matters just as much for review speed:

- ASCII only in code and comments - no emdash, unicode arrows, `x`, `...` used as unicode; use `-`, `->`, `x`, `...` ASCII equivalents.
- Comments are concise and explain non-obvious *why*, not *what*. Flag verbose comments, comments that restate the code, comments that reference the current task/PR, and comments hard-wrapped to a fixed column width.
- Do not force-wrap prose/comments to a fixed character count or split a sentence across lines.
- `snake_case` names; `kebab-case` (lowercase-with-dashes) file names for C/C++, `.h` headers; Python files lowercase-with-underscores. Naming optimizes for longest common prefix (`number_small`, not `small_number`).
- 4-space indentation, brackets on the same line, `void * ptr`, `int & a`, no trailing whitespace; match the surrounding style.
- Before pushing the PR, run the code style check and editorconfig check locally (see `.github/workflows/code-style.yml` and `.github/workflows/editorconfig.yml`). If the change touches Python code, also run the Python type check (`.github/workflows/python-type-check.yml`).
- Reuse existing infrastructure over introducing new components; no new third-party dependencies, extra headers, or files unless clearly justified.
- Keep it simple: a simpler change doing 90% is often preferable to a complex one doing 100%. Flag unnecessary templates/fancy STL; basic `for` loops are fine here.
- Every added line should be something the contributor can explain and defend to a reviewer without AI help - flag anything that looks copied-in without understanding.
- `Co-authored-by:` must be reserved for human co-authors; AI contributions (claude, cursor, codex, etc.) must use `Assisted-by:`; if this point is violated, it's a blocking finding.
- Any mentions of Minja must be treated as blocking; see `AGENTS.md` for why.

## Reporting

Group findings by severity so the user knows what actually blocks a merge:

1. **Blocking** - quick-reject/scope issues and correctness bugs; these can sink the PR regardless of everything else.
2. **Will slow the review** - convention/naming/comment violations, missing tests/docs/perf data, missing API justification or example.
3. **Nits** - minor style, optional cleanups.

For each finding, point to the file and line and say concretely what to change and why. Do not rewrite the whole diff unprompted; let the contributor make the fixes so they own and understand them. And do not draft any PR text, commit message, or reviewer reply - that is the contributor's to write.
