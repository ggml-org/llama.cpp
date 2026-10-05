# Speculative checkpoint ON_DEVICE A/B on Arc A770, 2026-10-04

## Conclusion

`LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` at the six speculative checkpoint sites in
`tools/server/server-context.cpp` is a small win where the sites execute and a
no-op where they do not:

- Where both contexts take full-sequence checkpoints (hybrid target and hybrid
  draft, `draft-simple`): +3.5% decode, 95% CI +/-3.6% over 4 paired launches,
  so the launch-paired interval still touches zero. On runs where both arms
  produced the same token stream with no rejected draft the gap is +4.0% to
  +4.2% with non-overlapping ranges.
- Under `--spec-type draft-mtp` on Qwen4Exp (the PR #90 configuration) none of
  the six sites execute: the target rolls back through recurrent-state
  snapshots (`n_rs_seq = 6` in the live log) and the MTP draft context supports
  partial removal. 0 checkpoints created and 0 restored over 95 verified draft
  rounds. The flag cannot change that configuration.
- Under `ngram-mod` on the Qwen4Exp trunk the target sites execute, but only 5
  draft rounds occurred in 4 requests and launch noise on prompts with no draft
  at all was 2% to 5%. No campaign was run there; it could not resolve an effect
  of this size.

The flag is applied at the same six sites as upstream PR
ggml-org/llama.cpp#28104. Upstream pairs it with dropping `DRAFT_MTP` from
`need_n_rs_seq()`, which this fork does not do (it keeps the snapshot rollback
from #28123, commit `0eadefebd`). Since review the fork also differs in that
the flag is conditional: a checkpoint stays on the host when the device copy
would eat into the `--fit-target` margin (see "Device-margin guard"). Both
measured arms predate the guard; the on-device arm passed the flag at every
site.

## The change and which sites execute

`LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` became
`LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` at six
calls on `slot.spec_ckpt` (line numbers below are those after the guard was
added). The prompt-cache checkpoints (`:453-477`, `:2681-2682`,
`:3749-3750`) stay host-resident. `slot.spec_ckpt` is used nowhere else, so a
device handle never reaches a host reader.

| site | call | executes when |
|---|---|---|
| `:3403` | `update_dft` before drafting | draft context is full-removal only |
| `:3442` | `load_dft` after drafting | draft context is full-removal only |
| `:3461` | `update_tgt` before verify | target is full-removal only, or bounded rollback with a draft longer than `n_rs_seq` |
| `:3473` | `update_dft` before verify | draft context is bounded rollback: unreachable, the draft context is always built with `n_rs_seq = 0` (`common/common.cpp:1308`, `common/speculative.cpp:3348`) |
| `:4330` | `load_tgt` on partial acceptance | as `:3461`, and a draft token was rejected |
| `:4333` | `load_dft` on partial acceptance | as `:4330`, with a draft context |

The removal type comes from `common_context_can_seq_rm`. `need_n_rs_seq()`
(`common/common.h:414`) returns `draft.n_max` for the MTP, EAGLE3, DFlash and
DSpark types and 0 otherwise, so a hybrid target is full-removal only under
`draft-simple` and the ngram types.

## Method

Commit ids quoted in this note name the tree a result was measured on. They are commits of
the PR 90 branch, which was merged as one squashed commit, so they are not in the history
of `master`; `git fetch origin pull/90/head` retrieves them.

- Tree: `99db3ec3f` plus the six-line change. Two snapshots of one SYCL build
  directory; only `libllama-server-impl.so` differs (sha256 `5c19b2e2898d94f9...`
  host, `d0165b6cf835c7ce...` on-device), every other library is byte-identical.
- Build: icpx 2026.1.1, JIT (no device arch), `GGML_SYCL_F16=ON`,
  `GGML_SYCL_DNN=OFF`, `GGML_NATIVE=ON`. intel-compute-runtime-git
  22.43.24558.r13063, IGC 2.41.10, kernel 7.3.0-rc5 xe.
- Harness: `scripts/perf/bench_spec.py` with the new `MODE=ab`: one server
  launch per arm in ABBA order, first request of every launch discarded, median
  of `REPEATS` per prompt per launch, paired 95% t intervals over launches. It
  refuses to start while another process holds `/dev/dri/renderD128` (exit 70)
  and reports new `xe` fault lines from `dmesg`. Both arms run with
  `LLAMA_TRACE=1`, which logs one line per verified draft and marks restores.
  Review tightened the gates after these runs: i915 as well as xe fault lines,
  a kernel log that could not be compared, a `fuser` error, a response that
  fails the target-argmax verifier and a speculative arm without draft
  statistics now each make the run fail. Applied to the summaries kept here,
  the last two exclude no launch. An odd `LAUNCHES` is flagged as an unbalanced
  order, the hash recorded per arm is that of the configured binary, and each
  response records how many of its rows carried argmax evidence (see
  "Correctness evidence").
- Requests: `prompts.jsonl`, `n_predict` 256, temperature 0, `cache_prompt`
  false, `--parallel 1`, q8_0 KV, flash attention on.
- No persistent SYCL cache and no SYCL graph (`GGML_SYCL_ENABLE_GRAPH` unset).

## Results

### Hybrid target and hybrid draft, all weights on the GPU

Qwen3.5-9B Q4_K_M (arch `qwen35`) as target and as its own `draft-simple` draft,
`--spec-draft-n-max 8`, ctx 8192, 4 launches per arm, 2 repeats. Both contexts
report "does not support partial sequence removal", so every draft round takes
a 50.25 MiB target checkpoint and a 50.25 MiB draft checkpoint. The
debug-verbosity probe prints the pair as `size = 100.5 MiB, draft = 50.25 MiB`,
where the first figure is the sum; an earlier revision of this report read it
as the target alone.

| prompt | host t/s | on-device t/s | delta | 95% CI half-width |
|---|---|---|---|---|
| code_edit | 16.13 | 16.63 | +3.10% | 5.92% |
| multi_turn | 16.05 | 16.60 | +3.42% | 3.66% |
| free_prose | 15.77 | 16.42 | +4.07% | 2.78% |
| all | 15.98 | 16.55 | +3.53% | 3.62% |

Checkpoint activity over the campaign: 831 verified draft rounds and 18
restores on the host arm, 840 and 26 on the on-device arm. New `xe` fault lines
in `dmesg`: 0.

The interval is wide because temperature-0 output is not reproducible run to
run in either arm (see below): a run that happens to reject a draft is 4% to 12%
slower. Restricting to runs that produced the most common token stream for the
prompt (a post-hoc cut, chosen after seeing the data):

| prompt | host runs | host t/s (min-max) | on-device runs | on-device t/s (min-max) | delta |
|---|---|---|---|---|---|
| code_edit | 5 | 16.226 (16.209-16.242) | 5 | 16.882 (16.870-16.890) | +4.04% |
| multi_turn | 7 | 16.226 (16.199-16.260) | 5 | 16.900 (16.891-16.907) | +4.15% |
| free_prose | 7 | 15.816 (15.428-16.233) | 7 | 16.464 (16.151-16.913) | +4.10% |

Derived, not measured directly: 0.61 s saved per 256-token run over about 29
draft rounds, about 21 ms per round for roughly 150 MiB of state that the host
arm moves across PCIe (two 50.25 MiB reads and one 50.25 MiB write).

### Qwen4Exp trunk with the MTP head (`draft-mtp`)

Qwen3.8-Flash-Next IQ1_M trunk plus `mtp-Qwen3.8-Flash-Next-Q8_0.gguf`,
`--spec-draft-n-max 6`, `--fit on --fit-target 1024`, ctx 16384, host binary at
debug verbosity. Loads in 74 s (9.2 GiB of weights on the GPU, the rest mapped
on the host). Two 192-token requests: 288 tokens drafted, 143 and 142 accepted.
The log shows `n_rs_seq = 6` on the target, "supports bounded partial sequence
removal", 95 accept lines, 0 "created speculative checkpoint" and 0 "restoring
speculative checkpoint". Throughput from that run (10.3 and 13.0 t/s) was taken
at debug verbosity and is not a benchmark.

### Qwen4Exp trunk with `ngram-mod`

One launch per arm, 1 repeat, `--fit-target 3072`.

| prompt | host t/s | on-device t/s | drafted / accepted (host) | drafted / accepted (on-device) |
|---|---|---|---|---|
| code_edit | 13.14 | 31.35 | 64 / 34 | 227 / 227 |
| multi_turn | 12.86 | 12.63 | none | none |
| free_prose | 13.29 | 12.58 | none | none |

5 verified draft rounds per arm including the warmup, 2 restores on the host
arm and 0 on the on-device arm. The code_edit gap is the two arms taking
different token streams (one accepted a long ngram draft, the other rejected
it), not checkpoint cost. The two prompts with no draft differ by 1.8% and 5.3%
between arms that are functionally identical there, which is the launch noise
floor for this partly host-resident model. Expected checkpoint saving here is
far below that (estimate: one state plane is about 113 MiB, one or two rounds
per request).

With `--fit-target 1024` the host (baseline) binary aborted during the warmup
request after about 170 generated tokens: `Failed to allocate physical memory`
in `ggml_sycl_pool_vmm::alloc` (`ggml-sycl.cpp:2089`) under `mul_mat`. That is
VRAM exhaustion in the unmodified code path and is independent of the flag.

## Correctness evidence

- `test-save-load-state -m Qwen3.5-9B.Q4_K_M.gguf -ngl 999` on the A770 exits 1.
  Test 7 "seq copy (device, scatter)", which restores through the on-device
  path and compares the state blob byte for byte, passes, as do tests 2, 6, 8
  and 9. Tests 3 (file load), 4 (host seq copy) and 5 (device seq copy) fail
  with the same value, `NMSE at step 0 is 6.785894e-05 (threshold 1.0e-05)`.
  The host and device paths agreeing to seven digits says the failure is the
  replay-versus-batch comparison on a real quantized model, not the device
  path. This test covers full sequence state, not `PARTIAL_ONLY`.
- Every run in both arms passed the harness verifier. That says less than its
  name: the server sends a top list only for a token it samples on its normal
  path, and the tokens of a verified draft round are emitted without one
  (`// TODO: set result.probs` in the accept path of
  `tools/server/server-context.cpp`). For those rows the verifier can only
  check the token id and a finite log-probability. The harness has counted the
  rows with a top list since a later review pass: two single-launch runs of
  this configuration had 1 such row in 256 per response. The verifier therefore
  did not establish that the generated tokens are the target's argmax here;
  that rests on the server's own draft verification.
- Token streams are not a usable oracle. The same binary, prompt and seed gave
  different streams on consecutive requests in both arms, including pairs where
  every drafted token was accepted (host arm, code_edit: `7011...` and
  `b683...`, both 226 of 226). On Qwen4Exp the streams differed between launches
  on prompts with no draft at all. The cause was not investigated.

## Cost and risk

- The on-device arm keeps one extra copy of the checkpointed state per context
  and sequence in VRAM (50.2 MiB per context in the 9B run, target and draft
  alike; an estimated 113 MiB for the Qwen4Exp trunk). It is allocated on the
  first checkpoint, so `--fit` does not budget for it, and it is never freed.
  The guard below keeps it from eating into the fit margin. The baseline abort above shows
  that a 1024 MiB fit margin is already too small for batched ngram verifies on
  this trunk.
- The flag invalidates earlier on-device states for the same sequence. The
  server keeps exactly one speculative checkpoint per slot and context, so this
  holds today; a second on-device user of the same sequence would break it.

## Device-margin guard

Added in review, after the measurements above. The six sites no longer pass the
flag unconditionally. Each slot picks the flags for its target and for its
draft context before every checkpoint update, and the load that follows reuses
them (`spec_ckpt_place()` and `spec_ckpt_flags()` in
`tools/server/server-context.cpp`, decision in
`common_speculative_checkpoint_flags()` in `common/speculative.cpp`):

- the device copy would hold `size(PARTIAL_ONLY) - size(PARTIAL_ONLY | ON_DEVICE)`
  bytes, the tensor data the host form carries;
- the context still holds the copy of its latest device checkpoint, and the state
  writer frees that before it allocates one of another size, so only the growth
  over that copy needs new memory. A checkpoint that is no larger keeps the
  device without a memory query;
- a larger one stays on the device only if every device of the context's model
  reports at least the growth in free memory plus the `--fit-target` margin the
  user gave (default 1024 MiB, taken before the server adds an mmproj to it).
  The margins follow the device order of the target model, so a device is
  looked up there by identity; one the target does not use keeps the largest
  margin;
- otherwise that checkpoint goes to the host, as before this change. A GPU that
  reports no memory figures counts as full.

The first version of the guard decided once, at the first checkpoint. That is
enough for a state of constant size, such as the recurrent state of the 9B run
below. It is not enough where the partial state includes attention cells and
grows with the sequence (a sliding-window cache under a hybrid memory, the
DeepSeek-V4 raw window): the writer allocates a larger copy on every change of
size. A later review pointed this out and the decision moved to every update.

A decision is logged when it is first made and when it changes. One-launch runs
of the 9B self-draft configuration on the A770, `guarded` being the build with
the guard. The first two rows are the first version
(`summary_ab-review-guard-room.json`, `summary_ab-review-guard-tight.json`), the
last two the per-update version (`summary_ab-review-guard2-room.json`,
`summary_ab-review-guard2-tight.json`):

| run | margin | log line of the guarded arm (draft and target context) | restores | exit |
|---|---|---|---|---|
| room | default 1024 MiB | `seq 0 checkpoint stays on the device (50.2 MiB)`, twice | 2 | 0 |
| tight | `--fit off --fit-target 15000` | `seq 0 checkpoint stays on the host: SYCL0 has 5665.7 MiB free, the device copy needs 50.2 MiB on top of the 15000.0 MiB margin`, twice (5665.4 MiB the second time) | 3 | 0 |
| room, per update | default 1024 MiB | `seq 0 checkpoint stays on the device (50.2 MiB)`, twice | 1 | 0 |
| tight, per update | `--fit off --fit-target 15000` | the host line twice (5665.8 and 5665.5 MiB free) | 2 | 0 |

All four runs completed all requests with 0 new i915/xe fault lines, and the
per-update runs logged each decision once: the state did not change size. The
guard checks, it does not reserve: a compute buffer that grows between two
checkpoints can still take the space, and a backend that over-reports free
memory defeats it. It also only decides where the copy goes; a failed device
allocation inside the state writer still aborts.

The growth path has no real-model run. `test_checkpoint_placement` in
`tests/test-qwen4exp-mtp.cpp` covers it on the fixture, whose head context saves
its whole cache: on the A770 a checkpoint that outgrew its device copy goes to
the host under a margin no device can keep, a checkpoint of unchanged size
keeps the device under the same margin, and a grown one with room goes back to
the device. With the first version's keep-the-first-decision rule put back, the
test fails at the first of these.

## Related findings from the same session

- `SYCL_CACHE_PERSISTENT=1` crashed the unmodified server once inside libsycl
  (`PersistentDeviceCodeCache::getItemFromDisc`, SIGSEGV) during warmup. All
  runs above leave it unset, as the production unit does.
- Upstream #28123 (snapshot rollback for qwen4exp) is in this fork as
  `0eadefebd`. Every tensor and key name added by the closed #28243 and the
  merged #29761 exists here, and the draft-path load fix from both does not
  apply (the fork passes draft params whose `model.path` is the draft file; the
  live log shows the head file being loaded).
- At the time of this run two behaviours of those PRs were absent here.
  Shared-embedding heads (`--mtp-shared-embd` in #28243, which borrow
  `token_embd`/`output` from the target) aborted on the `GGML_ASSERT` in
  `graph_mtp`; that was ported afterwards on the same branch, see "Borrowed
  tables and chain sampler follow-up" in `qwen4exp-mtp-correctness-2026-10-04.md`.
  Still absent and untested: the MTP block attends densely over a plain KV
  cache, as #28243 did, while the merged #29761 runs it through the indexer
  cache and block-sparse attention; the two differ once the context exceeds
  `indexer_top_k + 3` = 2051 cells.

## Not claimed

- No measurement of the flag under `draft-mtp` throughput: the sites do not
  execute there, which was shown by log counts on one launch, not by a timed
  A/B.
- No campaign on the Qwen4Exp trunk. The `ngram-mod` numbers are one launch per
  arm and say nothing about the flag.
- The 9B self-draft configuration is a stress shape for checkpoint cost, not a
  configuration anyone would deploy; the +4% does not transfer to other state
  sizes or acceptance rates.
- The stream-matched table is post-hoc. The pre-planned statistic is the
  launch-paired one, whose interval includes zero.
- Site `:3473` was not executed and cannot be in this fork. Sites `:4330` and
  `:4333` executed only on the 18 to 26 rejections per arm.
- VRAM figures for the extra device copy are derived from checkpoint sizes, not
  read from the device.
- The guard was exercised on one model and one device, one launch per case. Its
  two outcomes were forced with `--fit-target`, not reached by a model that
  fills the card through `--fit`. Multi-device placement, a draft on another
  device than its target, and more than one slot were not run.
- The four guard runs are single launches: their throughput columns are not a
  measurement of the guard. The per-update decision adds two state-size passes
  to every checkpoint update; their cost was not measured.
- The margin lookup by device was tested with one GPU: a model on that GPU
  against a host-only reference model. No run had two GPUs or a draft on
  another device than its target.
- No model whose partial state grows (hybrid memory over a sliding-window cache,
  DeepSeek-V4) was run with the guard. That the writer frees the old copy before
  it allocates the new one was read from source, not observed on the device.
- `PARTIAL_ONLY | ON_DEVICE` restore exactness was not tested in isolation; the
  byte-exact evidence is for the full-state on-device path.
- JIT build only, eager submission only. SYCL graph replay
  (`GGML_SYCL_ENABLE_GRAPH=1`, the production setting) was not exercised with
  on-device checkpoints.
- Run-to-run nondeterminism at temperature 0 was observed, not explained.

## Reproduce

```bash
# oneAPI env block from CLAUDE.md, then:
export ONEAPI_DEVICE_SELECTOR=level_zero:0
MODE=ab SETVARS= OUT_TAG=selfdraft-q35-9b CTX=8192 REPEATS=2 LAUNCHES=4 \
MODEL=/mnt/ssd1/models/Qwen3.5-9B.Q4_K_M.gguf \
DRAFT_MODEL=/mnt/ssd1/models/Qwen3.5-9B.Q4_K_M.gguf \
SPEC_ARGS="--spec-type draft-simple --spec-draft-n-max 8 --spec-draft-ngl 999" \
SERVER_BIN_A=<host build>/llama-server NAME_A=host \
SERVER_BIN_B=<on-device build>/llama-server NAME_B=ondevice \
python3 scripts/perf/bench_spec.py
```

Raw output: `scripts/perf/results/summary_ab-selfdraft-q35-9b.json` (campaign),
`summary_ab-probe-selfdraft.json` (debug-verbosity probe), `summary_ab-determinism-selfdraft.json`
and `summary_ab-probe-qwen4exp-ngrammod.json`. The per-launch `*.log` files beside
them are git-ignored.
