# PR #27 gh-resolve triage report

PR: https://github.com/Raudbjorn/ggml-llama.cpp/pull/27
Head: b069fea45 (`fix(review): portable timeout + accurate force-param comment (PR #27)`)
Merge commit: 9ea6be2e3 (`Merge origin/master into test/sycl-correctness-ci-gate`)
PR head before merge: 79c616b3c
Base: master @ 028084d00
Outcome: 4 unresolved threads at start, 0 at end. `gh-resolve list 27 --json` -> unresolved `threads[]` length 0; `--resolved` -> 4.

This PR also carried two real merge conflicts against master (`scripts/turbo-quality-gate.sh` and `tests/test-sycl-turbo-correctness.cpp`); both were resolved as part of this work (see "Merge conflict resolution" below).

## Real issues (fixed in code)

| Thread ID | Source | File | Issue | Fix |
|---|---|---|---|---|
| PRRT_kwDOTGTyNs6TsfIB | gemini-code-assist | scripts/turbo-quality-gate.sh:32 | The correctness stage calls GNU coreutils `timeout` directly; that binary is not installed by default on macOS, so on such hosts the gate aborts with "command not found" instead of running the stage. The PR's stated goal is portability, so this defeats it. | Added a `run_timeout` helper that resolves `timeout`/`gtimeout` once at startup into `TIMEOUT_CMD` and, when neither exists, runs the command directly (so the stage still executes; the device-lost hang cap is best-effort). Routed the correctness stage through it. |
| PRRT_kwDOTGTyNs6Tshat | copilot-pull-request-reviewer | scripts/turbo-quality-gate.sh:37 | Same `timeout` portability concern across the gate; the thread also flagged a unicode em dash in a FAIL message, which violates the repo's ASCII-only rule (AGENTS.md). | The `run_timeout` helper (above) was applied to ALL six timeout call sites, not just the flagged one: the two 180s correctness runs plus the four 600s perplexity/context-scaling runs, so stages 1-2 are portable too. The em dash lived only in the PR-head inline `[0/3]` block, which the merge replaced with master's function-based `stage_correctness` (no em dash); `grep -nP '[\x{2014}\x{2192}\x{00d7}\x{2026}]' scripts/turbo-quality-gate.sh` now returns nothing. |
| PRRT_kwDOTGTyNs6TshZ_ | copilot-pull-request-reviewer | tests/test-sycl-turbo-correctness.cpp:276 | The `force=true` doc comment said "(f16/q8_0)", but this harness only passes `force=true` for `GGML_TYPE_Q8_0` (the f16 baseline uses the separate `probe_fa_f16`), so the comment is misleading about which path is forced. | Corrected both the doc-block comment and the inline comment at the call site to say q8_0 only and to note that the f16 baseline goes through `probe_fa_f16`. Comment-only change; no logic touched. |

## Addressed by the merge (resolved, no new edit this turn)

| Thread ID | Source | File | Reviewer's point | Disposition |
|---|---|---|---|---|
| PRRT_kwDOTGTyNs6TshaZ | copilot-pull-request-reviewer | tests/test-sycl-turbo-correctness.cpp:347 | The skip string read "kernel not yet implemented", but `ggml_sycl_flash_attn_ext_supported()` vetoes turbo KV because the VEC turbo FA kernel is known-broken/hang-prone on the A770, not "unimplemented"; the message would mislead CI logs. | Valid observation, but already corrected by master: after the merge the skip string reads "SYCL reports turbo FA unsupported for this D/n_q/head combo" (no "not yet implemented"). Resolved without a new edit. This is NOT a false alarm -- the reviewer was right; master's superset simply reworded the string first. |

## False alarms (verified and dismissed)

None.

Each of the four threads was evaluated on the post-merge tree. Three were real and fixed in code (table above); the fourth (@347) was a real observation already corrected by master. No thread turned out to be a false alarm: in every case the reviewer's underlying point held, and the only reason one needed no new edit is that master had independently fixed the same string. The borderline candidate was @347 (one could call "not yet implemented" a defensible shorthand for a vetoed kernel), but the in-source comment at that site documents the veto as a known-hang avoidance, so the original string genuinely mislabelled it; the reviewer's correction stands.

## Merge conflict resolution

The branch was 2 commits ahead / 183 commits behind master, and the two named files conflicted.

- `tests/test-sycl-turbo-correctness.cpp` (12 conflict regions): in every region the master side was a strict superset of the PR head -- GQA `nh_q`/`nh_kv` parameters, the XMX `probe_fa_f16_nomask` probe, the `[8]`/`[8a]` InnerQ skeleton, `norm_ratio` bands in `meets_pass`/`meets_warn`, and turbo3/turbo4 flash attention promoted from XFAIL to GATE. The PR head was an older snapshot (no GQA/XMX/[8], turbo FA still XFAIL, d=128 only). Taking the PR-head side would have regressed master, so the file was resolved to master's side.
- `scripts/turbo-quality-gate.sh` (1 region): master refactored the three stages into `stage_correctness`/`stage_ppl`/`stage_scaling` functions (a strict superset of the PR's inline `[0/3]`/`[1/3]`/`[2/3]` blocks, including a second `LLAMA_TEST_TURBO_FA=1` correctness pass and a `TIMEOUT_COUNT`/exit-124 policy). Resolved to master's side; the PR's only unmerged intent (timeout portability) existed on neither side and is delivered by the `run_timeout` fix above.

Proof no PR-unique content was lost: immediately after taking master's side for both files, `git diff origin/master -- tests/test-sycl-turbo-correctness.cpp scripts/turbo-quality-gate.sh` was empty -- the PR's edits to these files had already been squash-merged upstream, so the PR head was a stale snapshot and the merge reproduced master exactly. The pre-merge working-tree stash contained only the `unsetenv()` return-propagation if-form, which master already carries at line 68 (commit 64f5307b7); it was therefore redundant and was dropped. After the review-fix commit the test file's net diff vs master is comment-only (the @276 correction), confirming no logic diverges from master.

## Verification

- `bash -n scripts/turbo-quality-gate.sh` -> rc=0 (syntax OK).
- `grep -nE '^\s*timeout |if timeout ' scripts/turbo-quality-gate.sh` (excluding the helper definition and the `TIMEOUT_CMD`/`TIMEOUT_COUNT` variables) -> empty; all six timeout call sites route through `run_timeout`.
- `git diff origin/master -- tests/test-sycl-turbo-correctness.cpp` -> three changed lines, all `//` comments (the @276 fix); no logic change.
- `git diff --check origin/master` -> clean (no whitespace errors).
- Merge-time proof: `git diff origin/master -- tests/test-sycl-turbo-correctness.cpp scripts/turbo-quality-gate.sh` empty at the merge commit (see "Merge conflict resolution").
- `gh-resolve list 27 -r Raudbjorn/ggml-llama.cpp --json` -> unresolved `threads[]` length 0.
- `gh-resolve list 27 -r Raudbjorn/ggml-llama.cpp --resolved --json` -> resolved `threads[]` length 4 (PRRT_kwDOTGTyNs6TsfIB, PRRT_kwDOTGTyNs6TshZ_, PRRT_kwDOTGTyNs6TshaZ, PRRT_kwDOTGTyNs6Tshat).
- `git push origin test/sycl-correctness-ci-gate` -> 79c616b3c..b069fea45.
