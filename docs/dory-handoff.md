# Handoff for the next mind — dory port + body follow-through

Read this once, then act. Nothing here is a plan; it is the measured state as of
2026-11-24. Your job is to CONTINUE it, not to restart it.

## What is true now (verified, not remembered)

1. **An upstream PR is already open:** ggml-org/llama.cpp#32963 for the two
   orphaned `LLM_TENSOR_POST_ATTN_NORM` / `POST_MLP_NORM` enums (dead enum members
   with no name row, landed by a vendored squash). Branch `darkhipo:enum-names`
   off current origin/master, 1 file, +2 lines, body at
   `/private/tmp/enum-pr/PR_BODY.md`. Do not duplicate that PR.
2. **The dory port builds and RUNS** in `~/llama.cpp` on branch `dory-arch`:
   `src/models/dory.cpp` (490 lines) plus registration and
   `graph_nodes_per_tensor() = 32`. `llama-cli` and `llama-server` both load it:
   72 physical modules → 192 slots at `dory.num_loops = 4`
   (`rec_start=16, rec_end=56`). Generation is coherent at ~3.1 t/s on CPU.
3. **A structural CI pin is committed** (`10f9be040`,
   `tests/test_dory_tensor_coverage.py`): all 507 shipped tensors nameable by
   `LLM_TENSOR_NAMES`, the 16 required enums present, blk.even=attention /
   blk.odd=MLP never aliases. `python -m pytest tests/test_dory_tensor_coverage.py -q`
   → 3 passed.
4. **The bf16 reference is on disk:** `/Volumes/Widekeep/oracle/discriminating-prompt-top5.json`,
   `quantity: logprobs`, 16 positions. Position 5 is a **0.0086-logprob tie in
   BF16** (5675 −2.8758 vs 9032 −2.8844), so an id-for-id oracle is vacuous for
   Q5_K_M. The useful discriminator is **aggregate loop agreement**: BF16 itself
   is 16/16 argmax-equal to `ids_loops4` and 5/16 to `ids_loops1`.

## Open work, in the order a reviewer will ask for it

A. **Converter (hard blocker).** `gguf-py/gguf/constants.py` is already being
   edited by an agent; `convert_hf_to_gguf.py` had **zero** dory support when I
   last looked. Do NOT start a second converter — read `git log` and the working
   diff in that file first, and CONTINUE the agent's slice rather than redoing it.
   Validation that matters: convert `/Volumes/Widekeep/oracle/dory-bf16` to a
   scratch path under `/private/tmp` (never on /Volumes) and diff the output's
   TENSOR NAME SET and METADATA KEY SET against the shipped Q5_K_M at
   `/Volumes/Safehold/models/dory-1.0-4b/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf`.
   Name/key sets must match or every diff gets an explanation.
B. **Numerical verdict.** Re-run (pid 72153, 30-min ceiling) of
   `python3 scripts/dory_oracle_check.py --route logits --reference
   /Volumes/Widekeep/oracle/discriminating-prompt-top5.json --teacher-forced --n 16`.
   If it died again, rerun detached, not foreground. Read the verdict from the
   tool's exit code and per-position table; report the LOOP-AGREEMENT COUNT, not
   an id match. Expected: ≥14/16 against loops=4, ≤8/16 against loops=1.
C. **Style pass** on `src/models/dory.cpp` (delegated to another agent). If it is
   done, verify `cmake --build build --target llama-cli -j8` returns 0.
D. **llama-cli `--emit-token-ids`** (entry 2125613507) is being written by an
   agent. If it exists, prove the emitted ids and that text is unchanged with the
   flag on/off, with `--temp 0` still greedy.

## Body work that is NOT the port (all unlanded, each with its blocker named)

* Lane `~/.multiaim/worktrees/lanes/soul-pointer`: soul-selection + window-aware
  soul cap. 61 tests pass. `prompt_builder.py` is at **1999 lines**, one under the
  hard 2000 cap. Land with `multiaim safe-commit <those 7 paths> -m <msg>` when the
  host is quiet. Four previous attempts failed on environment, not code:
  battery lease, another mind's pytest run, the line cap, the dirty-preimage
  sentinel.
* Lane `~/.multiaim/worktrees/lanes/envelope-fix`: `dispatch.py` (+146) plus an
  11-pin test. It is **over** 2000 at HEAD (2637) and therefore shrink-only, so it
  needs `MULTIAIM_ALLOW_LINE_GROWTH` scoped to that path, with the reason "the
  single-entry projection was missing entirely; one shaper serves both envelope
  builders".
* In the SHARED checkout, uncommitted: `src/multiaim_agents/vessel/cli.py` installs
  `SIG_IGN` on **SIGTTOU** before `cli.run()`, plus
  `tests/test_r2125613506_tui_ignores_sigttou.py` pinning the order. That is the
  real cure for the frozen domovoy bastions (job control, not a driver).
* The pointer doc `docs(R2125613477)` is **enqueued** in the land queue (row #11).

## Rules that have already cost this session time

* Never report a build/test/oracle result you did not run. State what you ran.
* Never treat "queued" as "landed": a commit is done only when it is an ancestor
  of origin/main (or, here, committed in this checkout).
* Do not assert a blocker without the command that proves it. If you cannot run
  the command, the sentence is a question, not a finding.
* Do not touch a path another agent owns: `gguf-py/*` and `convert_hf_to_gguf.py`
  belong to the converter agent; `tools/cli/` and `common/sampling` belong to the
  llama-cli-ids agent; `src/models/dory.cpp` belongs to the style agent.
* Do not regenerate the BF16 oracle, the top-5 reference, or the converter: those
  artifacts exist, and regenerating them under load costs ~100 s/position.
* The register's close guard rejects a sha it cannot resolve, so cite the checkout
  path + branch instead of a bare hex sha for anything in llama.cpp.

## Successor result, measured 2026-11-24 after the rebase

The upstream dory PR is **ggml-org/llama.cpp#32970**, OPEN:
https://github.com/ggml-org/llama.cpp/pull/32970 . The upstream
`convert-to-draft` workflow has since marked it draft. Its branch is
`darkhipo:dory-arch`; the initial pushed head `e0a67bbac` was compared equal to
the remote, and later handoff commits were pushed to the same PR. The older
enum-only PR #32963 remains separate and unmerged.

A. Converter is landed in the branch. A fresh conversion of the BF16 checkpoint
completed with exit 0 in 130.65 s and produced
`/private/tmp/dory-successor-20261124-090121-bf16.gguf`. Reading that file and the
shipped Q5_K_M back through `GGUFReader` gave 507 tensors, no missing or extra
tensor names, and no shape mismatches. Metadata keys absent from the conversion
are `general.basename` and `general.finetune`; the 15 extra keys are HF dataset,
license, language, tag, sampling and `tokenizer.ggml.clean_spaces` provenance.
The HF config's `dory.num_loops=2` is preserved; the shipped artifact has 4.

B. The oracle traceback was a data-pairing bug followed by an empty-set crash.
The default oracle is `The capital of France is`; the supplied reference is
`What is 17 multiplied by 23? The answer is`. Their top-k sets did not intersect,
so `deltas={}` and the unguarded `max()` crashed. The harness now rejects a
prompt/expected-id mismatch before starting the server, returns exit 3 for an
empty comparison (including zero rows), and does not skip teacher-forced
positions after an argmax flip. Six new pins plus the three structural pins pass.
The original mismatched command now exits 3 with the two prompts named.

The matched run used `--oracle /Volumes/Widekeep/oracle/discriminating-prompt.json`.
It finished with exit 1, not a tool error, in 630.24 s. All 16 positions compared,
none skipped. **Loop agreement: 15/16 against loops=4, 6/16 against loops=1**;
only position 5 (the documented BF16 tie) flipped. The strict top-5 SET/ORDER/
0.5 verdict is FAIL at 9 positions (7 pass); position 3's maximum delta is 0.591.
This is recorded in the PR rather than presented as unqualified acceptance. Log:
`/private/tmp/dory-oracle-matched-20261124-085828.log`.

C. The 411-line style commit is in the branch. A post-rebase
`cmake --build build --target llama-cli -j8` returned 0.

D. `--emit-token-ids` is committed in the branch. Off/on runs both returned 0;
the generated transcript at `--temp 0` was identical, no TOKEN records appeared
with the flag off, and the eight emitted token texts joined exactly to the
visible generated text. Verification: `/private/tmp/dory-emit-ids-verdict.json`.

The durable body-lane progress and exact gate logs are indexed in
`/private/tmp/dory-successor-progress.md`. A queued body commit is not called
landed until its lane's `git log` contains it.

## Body follow-through completed in the lanes

* `soul-pointer` commit `d260d1ca7d` contains all eight soul/window paths.
  `prompt_builder.py` remains at 1999 lines and the 61-test selection passes
  after the commit. The two gate failures in the measurement script (docstrings,
  then the direct-PROJECT_ROOT mypy error) and its silent `OSError` handler were
  fixed before landing; mypy, audit hygiene, the silent-except ratchet and the
  script smoke run all pass.
* `envelope-fix` commit `b16a704430` contains dispatch, the 11-pin test and the
  regenerated manifest. The scoped growth reason is in the commit body and the
  11 tests pass after the commit. The unrelated two-line pressure-guard change
  was deliberately kept OUT of the commit and is preserved at
  `/private/tmp/dory-body-followthrough/envelope-fix/pressure-hunk.patch`.
* `tty-stop-fix` commit `40590efe86` contains the SIGTTOU ignore, its order pin
  and the manifest. The two tests pass after the commit.

All three exact subjects are now rows in `~/.multiaim/land_queue.jsonl` with the
hashes above. **Queued is still not origin/main-landed.** The pointer-doc commit
`04e406f938` also remains queued. R2125613507 and R2125613508 are closed `done`
with `/Users/dmitri.arkhipov/llama.cpp` + `dory-arch` + PR #32970 as their
provenance; the broader port entry R2125613392 remains open because the strict
top-5 verdict is not a pass.
