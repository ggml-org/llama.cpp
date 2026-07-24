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
- Is it niche/vendor-specific in a way that adds a maintenance burden nobody will own long-term? Flag the maintenance-ownership question.
- Is the change semantically correct, or a plausible-looking "fix" that misunderstands the code? Sanity-check the actual behavior, not just that it compiles.
- AI-disclosure: if AI meaningfully contributed, is the PR template's disclosure section filled in? Remind the user. Never suggest writing the PR description or commit message for them.

## Security review (mandatory)

Mandatory on every review; any finding here is **blocking**. Rule of thumb: GGUF metadata, tensor shapes, tokenizer/grammar input, and all server/RPC fields are attacker-controlled - bound them before use.

- **Sizes/counts from tensor dims:** validate before allocating. Products like `ne[i]*nb[i]`/nbytes can overflow on crafted dims into an undersized alloc then heap overflow. Overflow checks must run BEFORE the arithmetic they guard - padding/alignment macros wrap to 0 near `SIZE_MAX`, so a guard after the pad passes.
- **GGUF strings/arrays:** cap declared lengths and element counts before using them to size a loop or buffer; validate element type and length before casting an array to a pointer or reading fixed indices (`[i+1]`, `[0..2]`).
- **File-supplied counts indexing fixed arrays:** bound any count (e.g. layer/block count into a `LLAMA_MAX_*` array) before indexing; watch checks that only fire when an optional key is present.
- **Bounds comparisons:** flag narrowing casts (`size_t`->`int32_t`) and signed/unsigned mixing that can bypass a length check and copy past a buffer.
- **Parsed/derived indices:** range-check `stoi`/`atoi` results and catch parse throws; never use a default or derived token id (EOS/BOS/...) as an index without a bounds check.
- **Reused/reserved buffers:** recheck bounds after a buffer is shrunk or reused; watch `reserve()` then index-by-assumed-size, and header fields read before their length is checked.
- **Server JSON ints:** clamp client-supplied integers (token/discard counts, offsets) to non-negative and an upper bound before they reach index/pointer arithmetic.
- **RPC-deserialized fields:** treat every field (type/buffer/data/ne/nb/op_params) as hostile - validate before use. Null/zero buffers skipping validation, attacker data pointers, out-of-range type indices, and negative strides sign-extending past a corner-only assert all give arbitrary read/write.
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

## General (always)

Enforce the `AGENTS.md` / `CONTRIBUTING.md` coding and naming guidelines on every changed line - this is a distinct pass from checking that the code works, and matters just as much for review speed:

- ASCII only in code and comments - no emdash, unicode arrows, `x`, `...` used as unicode; use `-`, `->`, `x`, `...` ASCII equivalents.
- Comments are concise and explain non-obvious *why*, not *what*. Flag verbose comments, comments that restate the code, comments that reference the current task/PR, and comments hard-wrapped to a fixed column width.
- Do not force-wrap prose/comments to a fixed character count or split a sentence across lines.
- `snake_case` names; `kebab-case` (lowercase-with-dashes) file names for C/C++, `.h` headers; Python files lowercase-with-underscores. Naming optimizes for longest common prefix (`number_small`, not `small_number`).
- 4-space indentation, brackets on the same line, `void * ptr`, `int & a`, no trailing whitespace; match the surrounding style.
- Reuse existing infrastructure over introducing new components; no new third-party dependencies, extra headers, or files unless clearly justified.
- Keep it simple: a simpler change doing 90% is often preferable to a complex one doing 100%. Flag unnecessary templates/fancy STL; basic `for` loops are fine here.
- Every added line should be something the contributor can explain and defend to a reviewer without AI help - flag anything that looks copied-in without understanding.

## Reporting

Group findings by severity so the user knows what actually blocks a merge:

1. **Blocking** - quick-reject/scope issues and correctness bugs; these can sink the PR regardless of everything else.
2. **Will slow the review** - convention/naming/comment violations, missing tests/docs/perf data, missing API justification or example.
3. **Nits** - minor style, optional cleanups.

For each finding, point to the file and line and say concretely what to change and why. Do not rewrite the whole diff unprompted; let the contributor make the fixes so they own and understand them. And do not draft any PR text, commit message, or reviewer reply - that is the contributor's to write.
