# DeepSeek-V4 ROCm benchmark harness

This directory contains the controlled PP and target-only raw-decode
baseline/profiling harness for
`docs/deepseek-v4-flash-rocm-performance.md`.

## Safety

`run-pp.sh` and `run-tg.sh` acquire `$HOME/llama-jobs/gpu.lock`, then check
`rocm-smi --showpids` twice: once before manifest capture and again immediately
before launch. Discovery errors fail closed. Any active ROCm process causes a
refusal. The benchmark runs in its own process group; cleanup and both watchdogs
signal the complete group with bounded TERM-to-KILL handling. The script never
signals a process that existed before its own launch.

Do not use `DSV4_ALLOW_BUSY_GPUS=1` for controlled measurements.

Inspect the exact command without querying ROCm or starting a GPU process:

```bash
scripts/dsv4-rocm/run-pp.sh --dry-run
scripts/dsv4-rocm/profile-pp.sh --dry-run
scripts/dsv4-rocm/run-tg.sh --dry-run
DSV4_TG_MODE=residency scripts/dsv4-rocm/run-tg.sh --dry-run
DSV4_TG_DEPTHS=16384 scripts/dsv4-rocm/profile-tg.sh --dry-run
```

## Target-only raw decode baseline

`run-tg.sh` uses llama-bench's `--n-depth` path: prefix/depth setup happens
outside `samples_ns`, then `test_gen` performs exactly the fixed number of
single-token target evaluations. There is no sampler, EOS stop, draft model,
or speculative flag. Default performance mode runs the required depth sweep
with tg32 and six raw repetitions; it predeclares the first target-depth sample
as graph-cold and reports the remaining five. `summarize-tg.py` recomputes t/s
and ms/token from raw nanoseconds, requires exact depth/config/repetition
coverage, and reports MAD/median stability. To keep the mandatory tg32 samples
unperturbed, the one-second `rocm-smi` loop runs only during each depth's setup;
any query tail can overlap only repetition 1, which is predeclared/discarded.
Accepted repetitions 2+ have no in-band telemetry query. Pre/post snapshots,
setup telemetry, the power policy, and clock behavior remain preserved.

Scheduler logging is deliberately separate so verbose debug output cannot
perturb accepted TG. Residency mode runs one target evaluation per depth with
`GGML_SCHED_DEBUG=2 --verbose`; `parse-sched-debug.py` records DSV4 LID/TOP_K
backend assignments plus CPU and ROCm/meta split counts and scheduled split-input
copies. It counts only real operation lines (not `CONT`/`SET_ROWS` consumers)
and requires exactly 21 TOP_K plus 21 LIGHTNING_INDEXER nodes per measured graph
from 2K upward (none at depth 0). Attestation also requires one ordered
benchmark/generation marker per depth, the expected nonzero depth marker, split
`#0 = CPU/0 inputs`, and split `#1 = Meta(ROCm0,...,ROCm3)/22-or-25 inputs`,
with no extra splits or parser warnings. A CPU/unknown DSV4 assignment or count,
marker, split, or backend-correlation mismatch exits nonzero as valid pre-fix or
incomplete evidence, not as a deployment pass. The scheduler exposes the four
devices as one composite Meta backend, so its split/copy counts are aggregate
Meta counts, not independent per-GPU counts.

Dry-run and non-GPU fixture validation:

```bash
cd /home/edwin/llama.cpp-rdna2
scripts/dsv4-rocm/test-tg-tools.py
scripts/dsv4-rocm/test-tg-profile.py
scripts/dsv4-rocm/test-bf16-screen.py
scripts/dsv4-rocm/run-tg.sh --dry-run
DSV4_TG_MODE=residency scripts/dsv4-rocm/run-tg.sh --dry-run
DSV4_TG_DEPTHS=16384 scripts/dsv4-rocm/profile-tg.sh --dry-run
```

Before performance mode may reuse llama-bench's saved full context state, run
the model-dependent equivalence gate:

```bash
cmake --build build --target test-state-restore-equivalence -j 12
scripts/dsv4-rocm/run-state-restore-equivalence.sh --dry-run
DSV4_STATE_API=context DSV4_LABEL=context-state-equivalence \
  scripts/dsv4-rocm/run-state-restore-equivalence.sh
```

The gate computes a deterministic fresh prefix at 2K, 3K, and 16K by default,
saves either sequence-only or full context state, and runs four greedy target
steps while retaining every full-vocabulary logit. It then clears memory and
recomputes the same prefix/continuation as a fresh-repeat control before a
second clear, state restore, and exact-input replay. It requires all three
argmax paths to match and both the original-vs-fresh-repeat and
fresh-repeat-vs-restored full-logit comparisons to satisfy
`abs_diff <= 1e-5 + 1e-5*max(abs(a),abs(b))`. Exact prefix/input/argmax
token IDs, state/logit hashes, byte counts, manifests, DSOs, and telemetry are
preserved under `$HOME/llama-jobs/dsv4-rocm-state-equivalence/`.

This is deliberately a same-context/same-benchmark-instance gate. The fixed
`run-tg.sh` command has one generation/batch configuration and unique depths,
so llama-bench restores the saved state only for repetitions 2-6 inside the
same context; the next instance has a different depth and performs fresh setup.
The greedy path is a semantic state-equivalence test, not a literal reproduction
of llama-bench's random timing inputs.

On DSV4, sequence-only `llama_state_seq_get/set_data` restoration is rejected:
fresh re-prefill is bit-identical, but restored full-vocabulary logits diverge.
Full `llama_state_get/set_data` context restoration is bit-identical at all
three gate depths. Consequently `run-tg.sh` requires
`LLAMA_BENCH_DEPTH_STATE_API=context`; overriding it back to sequence fails
closed.

After a GPU window is confirmed and this gate passes:

```bash
# Performance: tg32, six raw / five accepted samples at every depth.
# Use full hashing for a final evidence run; it reads every GGUF shard before load.
DSV4_HASH_MODE=full DSV4_LABEL=raw-tg-baseline scripts/dsv4-rocm/run-tg.sh

# Separate scheduler-residency audit; timings are not baseline TG.
DSV4_HASH_MODE=full DSV4_TG_MODE=residency DSV4_LABEL=raw-tg-residency \
  scripts/dsv4-rocm/run-tg.sh
```

Default depths are actual starting KV depths
`0,2048,3072,4096,8192,16384,32768,65536`. Depth 0 means the context is empty
before `test_gen` evaluates its first BOS/random token; it is recorded as 0,
not relabeled as a user-visible zero-token prompt. llama-bench saves the
first target-only depth state and restores it for later repetitions. That
fresh-vs-restored equivalence remains a separate M5.0 acceptance gate.

Performance mode sets scheduler debug to zero. Residency mode sets it to two.
Both modes explicitly default to the accepted J16/HC/LID controls (16/1/4),
F16 K/V, flash on, tensor split `1/1/1/1` (llama-bench's device-list spelling
of `1,1,1,1`), batch 512, and ubatch 256. Set
`DSV4_REQUIRE_ACCEPTED_STACK=0` only for a separately declared A/B arm.
Artifacts are written under `$HOME/llama-jobs/dsv4-rocm-tg/`.

Each setup phase (model/context/depth) and each generation sample has a
separate watchdog deadline (`DSV4_TG_SETUP_TIMEOUT` and
`DSV4_TG_SAMPLE_TIMEOUT`). Phase transitions and every measured-sample start
are preserved in `phase-events.tsv` and `measurement-start.ns`; setup time is
never reported as TG. A timeout preserves completed depth records and marks the
sweep incomplete.

Do not use scheduler-debug TG as a production throughput number and do not
combine performance and residency samples.

## Unprofiled RCCL raw-TG candidate screen

After two independent exact-role profiles at both 16K and 64K selected
communication, routine profiling stops. `screen-rccl-tg.sh` runs reversible
RCCL algorithm/protocol candidates through the real target-only TG path:

```bash
# Run this first. It uses one model load and 5 accepted tg32 samples/depth.
scripts/dsv4-rocm/screen-rccl-tg.sh tree-ll

# Run the matched control only if tree-ll is plausibly >=3% above the accepted
# historical 64K median. ring-ll is the one fallback candidate.
scripts/dsv4-rocm/screen-rccl-tg.sh auto
scripts/dsv4-rocm/screen-rccl-tg.sh ring-ll
```

The wrapper forces 16K/32K/64K, tg32, six raw repetitions, one predeclared
discard, full model hashes, accepted MMQ/HC/LID settings, exact tensor split,
F16 K/V, no profiler, and no speculative path. It rejects every inherited
`NCCL_*`/`RCCL_*` variable plus `GGML_CUDA_DISABLE_GRAPHS` before setting the
complete candidate environment. The comparator requires
`GGML_HIP_GRAPHS:BOOL=ON` in the captured build cache and confirms the runtime
graph-disable variable remained absent; the similarly named environment value
is recorded but does not substitute for this build/runtime proof.
`auto` leaves algorithm/protocol unset; `tree-ll` and `ring-ll` force exactly
those algorithms with the LL protocol. All use
`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ENV` to retain RCCL's runtime environment
acknowledgement without enabling per-collective tuning diagnostics. RCCL may
emit one small setup-time function/protocol/algorithm matrix under this mask;
the comparator permits it but rejects per-collective AllReduce/channel-tuning
markers. The complete raw stream is preserved verbatim in `bench.stdout.log`, while non-JSON lines are
also separated into `bench.stdout-nonjson.log`; only parsed JSON records enter
`result.jsonl` and the result-completion timestamps. `stdout-classification.json`
fails closed on malformed JSON-like data, unterminated output, capture errors, or
excessive diagnostics. The settings are recorded
along with set-versus-unset controls in `command.sh`, `executed-command.sh`,
`effective-settings.sh`, and `contract.json`.

Channel forcing is excluded. The installed RCCL 2.30.4 binary explicitly says
`NCCL_MIN_NCHANNELS` is ignored below eight GPUs, so a 1/2-channel sweep would
not be a valid four-V620 candidate. Compare a contemporaneous control and
candidate with:

```bash
scripts/dsv4-rocm/compare-rccl-tg.py CONTROL_DIR CANDIDATE_DIR \
  --json CANDIDATE_DIR/rccl-screen-comparison.json
```

The fail-closed gate requires complete stable identity-matched runs (including
normalized run-local source provenance), compiled graph support with no runtime
disable, runtime acknowledgement of every forced algorithm/protocol, no in-band TUNING output,
>=3% median TG gain at 64K, and no >2% median regression at 16K or 32K. Passing
five accepted samples selects only a full 31-repetition validation; it never
accepts an optimization. `test-rccl-screen.py` covers exact wrapper isolation,
inherited NCCL rejection, runtime acknowledgement, tuning-noise and legacy
capture rejection, positive/no-go gates, and identity mismatch. The non-GPU
`test-tg-tools.py` fixture exercises the real `run-tg.sh` FIFO path with mixed
RCCL/JSON output and proves malformed/high-volume capture failures propagate.

Current M5.4 outcome (2026-08-04): the first `ENV,TUNING` Tree+LL run is invalid
instrumentation and must never be compared. The repaired Tree+LL screen is
unstable and falls to 14.847 t/s at 64K; the Ring+LL fallback is unstable at
32K and reaches only 18.318 t/s at 64K (-0.460% versus the accepted historical
median). Neither candidate reaches the preliminary gate, so `auto` was not run
and neither candidate advances. The canonical performance document records the
exact artifacts. Next is the predeclared guarded RDNA2/four-way BF16 hidden
reduction, not another runtime candidate or routine profile.

## Short guarded BF16 hidden-AllReduce gate

`GGML_HIP_RDNA2_BF16_HIDDEN_ALLREDUCE=1` is an experimental, fail-closed
shape-scoped option. It requires a HIP build with RCCL/NCCL, explicit
`GGML_CUDA_ALLREDUCE=nccl`, exactly four distinct physical RDNA2 devices, and
contiguous F32 rank tensors of exact shape `[4096,1,1,1]`. The force-FP32 flag
is ORed across all ranks and always wins. Qualifying calls reuse the existing
F32-to-BF16, BF16 RCCL sum, and BF16-to-F32 implementation; all misses retain
the existing size heuristic. Unset or exact `0` disables the candidate and any
other value aborts. The optional
`GGML_HIP_RDNA2_BF16_HIDDEN_ALLREDUCE_AUDIT=/path` writes one JSONL summary per
communication context and is correctness-only; performance runs must leave it
unset.

The user-directed iteration gate is deliberately short:

```bash
cmake --build build --target test-cuda-allreduce-precision \
  test-dsv4-bf16-allreduce-equivalence llama-bench -j 12
build/bin/test-cuda-allreduce-precision
scripts/dsv4-rocm/run-bf16-allreduce-equivalence.sh --dry-run
scripts/dsv4-rocm/run-bf16-allreduce-equivalence.sh

# Only after the correctness comparison reports PASS:
DSV4_BF16_EQ_RESULT=/path/to/passing-correctness-artifact \
  scripts/dsv4-rocm/screen-bf16-tg.sh
```

The deterministic correctness A/B uses only 2K context and four fixed target
inputs. It captures raw full-vocabulary F32 logits, requires matching argmax
tokens, finite values, every element within `0.05 + 0.01*scale`, RMSE at most
0.02, exactly 344 eligible hidden reductions, zero candidate BF16 calls in the
control, 344 in the candidate, and the model-observed exact dynamic force-FP32 count of zero (rank-wise precedence is proved by the host selector test). It does not use
state restore, sampling, a profiler, or speculative decoding.

The matched performance triage uses only 0/2K/8K, tg8, six raw repetitions,
one discarded, and five retained. A regression beyond 2% at any depth or less
than 4% gain at 8K is a NO-GO. A pass is only
`PROMISING_SHORT_SCREEN`; it never accepts the optimization. The script does
not launch 16K, 32K, or 64K. Any later confirmation is a separate explicit
user decision.

## Target-only raw-decode profile

`profile-tg.sh` is the disk-safe M5.3 profile wrapper. It requires exactly one
starting depth and fails closed unless the evidence contract is tg32, at least
six raw repetitions, exactly the first discarded/at least five profiled, full
GGUF hashing,
full-context state, batch/ubatch 512/256, F16 K/V, tensor split `1/1/1/1`,
12 host threads, mmap loading, and the guarded J16/HC1/LID4 stack. A ROCm-only optional llama-bench hook calls
`roctxProfilerResume(0)` immediately around `test_gen` for repetitions 2+ and
pauses after each synchronized generation. The benchmark process writes its own
authoritative `CLOCK_MONOTONIC` `resume_return`/`pause_call` pairs to
`rocprof-selected-regions.tsv`; the summarizer requires every trace event to fit
wholly within those exact boundaries. `rocprofv3 --selected-regions` therefore
records no model load, depth setup, context-state restore, or discarded first
repetition. The compact profile writes CSV only, but requires kernel,
memory-copy, RCCL, and HIP-runtime domain files so launch/synchronization calls
remain visible. Never substitute an unscoped whole-process trace at 32K/64K.

Run independent decision-context profiles as separate processes/artifacts.
The wrapper defaults to six raw repetitions; if the 3% profiled-wall stability
gate misses, increase `DSV4_TG_REPS` only and retain exactly one discard. The
scope-validated family attribution is still preserved for diagnosis, but the
runner exits 4 and the perturbed throughput cannot establish TG or decide CSA:

```bash
DSV4_TG_DEPTHS=16384 DSV4_TG_REPS=11 DSV4_LABEL=raw-tg-profile-16k-a \
  scripts/dsv4-rocm/profile-tg.sh
DSV4_TG_DEPTHS=65536 DSV4_TG_REPS=11 DSV4_LABEL=raw-tg-profile-64k-a \
  scripts/dsv4-rocm/profile-tg.sh
```

The wrapper requires full GGUF hashing and writes under
`$HOME/llama-jobs/dsv4-rocm-tg-profile/`. `run-tg.sh` invokes
`summarize-tg-profile.py` after the normal stability gate and preserves
`profile-summary.{txt,json}` plus `profile-families.tsv`. The summarizer verifies
that every trace event is wholly inside an authoritative accepted generation interval, then
reports accepted wall time, target-token count, dispatches, summed device time,
per-agent and per-repetition totals, HIP/RCCL/copy calls, exclusive families,
and top kernel/grid/workgroup groups. Summed device time spans four
devices/queues and is the branch-share denominator; it is not wall time. For
the fully hashed V4-Flash IQ2_M model, routed/shared MMVQ attribution requires
exact type/fusion/workgroup/grid signatures plus exactly one
`deepseek4.block_count=43` manifest record. Normal and anomalous tensor-type
multiplicities must match `target tokens * layers * four GPU agents` globally,
per agent, and per repetition; a partial signature match fails closed. At 16K
and 64K, exact F16 concat type/dimension/grid signatures similarly split CSA/HCA
K materialization and final-mask concatenation. Their 21/20-layer counts must
match globally, per agent, and per repetition; unknown near-shapes fail closed.
Other depths remain explicitly unclassified rather than borrowing these grids.
`non_moe_quantized_matmul` combines attention/indexer/final-output projections.
Unclassified `other` time is not silently assigned. RCCL/HIP API durations are
reported separately, may overlap, and are never added to device-kernel time.

`analyze-tg-communication.py` performs read-only forensics across preserved
selected-region artifacts. It requires the exact rocprof RCCL schema and exact
43-block/tg32/four-GPU cadence: 86 AllReduce groups per token, 344 rank calls per
token, and 2,752 groups / 11,008 rank calls plus device kernels per repetition.
It reports long device-kernel intervals and same-agent timestamp overlap without
calling them stalls or proving a critical path. The supported RCCL trace has no
count/datatype/buffer/communicator/rank/message-byte arguments, and its API and
device-kernel correlation-ID sets are disjoint; message sizes and direct
API-to-kernel attribution are therefore unavailable, not inferred from launch
geometry. Example (output only after every run validates):

```bash
python3 scripts/dsv4-rocm/analyze-tg-communication.py \
  "$PROFILE_16K_A" "$PROFILE_16K_B" "$PROFILE_64K_A" \
  --json "$PROFILE_64K_A/communication-forensics.json" \
  > "$PROFILE_64K_A/communication-forensics.txt"
```

Run `scripts/dsv4-rocm/test-tg-communication.py` as the non-GPU fixture gate.

## Five-minute quick PP baseline

After the GPU owner has stopped inference and confirmed a benchmark window:

```bash
cd /home/edwin/llama.cpp-rdna2
DSV4_LABEL=base-quick \
DSV4_PROMPTS=512,2048 DSV4_UBATCHES=256 \
DSV4_REPS=3 DSV4_TIMEOUT=300 \
scripts/dsv4-rocm/run-pp.sh
```

Defaults match the current target:

- IQ2_M first shard under `/home/edwin/models/DeepSeek-V4-Flash-0731-GGUF/`;
- all layers on four GPUs;
- tensor split `1/1/1/1`;
- F16 K/V cache;
- flash attention on;
- batch 512 and ubatch 256;
- NCCL all-reduce, P2P, HIP graphs, gfx1030 override, and no scratch reclaim.

The five-minute absolute cap starts when llama-bench emits its first measured
prompt-run marker, after the initial model load, context creation, and first
warmup. TERM is sent two seconds before the deadline and KILL is sent at the
absolute deadline if any process-group member remains. A separate 20-minute
startup cap prevents an infinite load/warmup hang.

JSONL preserves fully completed cases if a run is truncated. A truncated or
shape-incomplete run is explicitly marked `complete=false` and exits 3 after
writing its summary. It is evidence, not a matched A/B result. Compare only
identical complete shape/repetition sets. Run 8K separately so a slower 8K case
cannot hide completion of the short grid:

```bash
DSV4_LABEL=base-8k DSV4_PROMPTS=8192 DSV4_REPS=2 \
DSV4_TIMEOUT=300 scripts/dsv4-rocm/run-pp.sh
```

The script runs PP only (`--n-gen 0`). Speculative draft settings are excluded
because they do not belong in first-pass PP attribution.

## Frozen baseline binaries

Selecting `/home/edwin/baseline-bin/bin/llama-bench` also selects sibling DSOs
through the harness-managed `LD_LIBRARY_PATH`, preventing a candidate rebuild
from silently changing the baseline libraries:

```bash
DSV4_BENCH=/home/edwin/baseline-bin/bin/llama-bench \
DSV4_LABEL=frozen-base scripts/dsv4-rocm/run-pp.sh
```

The manifest stores the executable SHA-256, `ldd` resolution, SHA-256 for all
resolved local llama/ggml DSOs, selected CMake cache when available, git diff,
and untracked-file hashes. Verify resolved DSO paths before accepting an A/B.

## RDNA2 routed-MMQ screening

`GGML_HIP_RDNA2_MMQ_J` is an experimental, opt-in tile-width override for
routed MMQ on RDNA2 only. The default dispatch is unchanged when it is unset.
Valid values are supported multiples of eight from 8 through 128; an
unsupported type/configuration fails rather than silently changing the test.
For example:

```bash
GGML_HIP_RDNA2_MMQ_J=16 \
DSV4_LABEL=mmq-j16-quick DSV4_PROMPTS=512,2048 DSV4_REPS=3 \
scripts/dsv4-rocm/run-pp.sh
```

On the target IQ2_M model, J=16 is the current exploratory winner. It is not
yet a general RDNA2 default: the focused fixture shows that strongly skewed
expert routing changes the optimum. Keep the variable and its value in the
manifest, use complete paired runs, and do not carry this setting to unrelated
models without screening their routing shape.

`test-mmid-rdna2` defaults to a fast prototype-weight fixture for performance
screens. Its `--fixture unique` mode independently quantizes every
expert/output row for correctness. The target-shape check uses N=512,
batch=256, 256 experts, top-6, and K=256 (one quant block) so coupled
expert/row addressing is distinguishable without minutes of setup. Dump a J64
reference and compare J16 for both `iq2_xxs`/`iq3_xxs` and `uniform`/`hot`:

```bash
GGML_HIP_RDNA2_MMQ_J=64 build/bin/test-mmid-rdna2 \
  --type iq2_xxs --fixture unique --k 256 --n 512 --batch 256 \
  --experts 256 --top-k 6 --routing uniform --dump-output /tmp/mmid.bin
GGML_HIP_RDNA2_MMQ_J=16 build/bin/test-mmid-rdna2 \
  --type iq2_xxs --fixture unique --k 256 --n 512 --batch 256 \
  --experts 256 --top-k 6 --routing uniform --compare-output /tmp/mmid.bin
```

## RDNA2 DSV4 hidden-channel mixer screening

`GGML_HIP_RDNA2_HC_MIXES=1` opts the exact contiguous F32
M=24,N=256,K=16384 DSV4 hidden-channel mixer shape into a 12x16x256 LDS-tiled
kernel on wave32 RDNA2. Unset/`0`, other devices, other types/layouts/shapes,
and all non-HIP backends keep the existing dispatcher and rocBLAS/generic
fallbacks. Values other than `0` or `1` fail closed.

The setting is still a screening control, not a general F32 GEMM default. Run
the focused benchmark through the normal GGML backend graph before whole-model
A/B:

```bash
unset GGML_HIP_RDNA2_HC_MIXES
build/bin/test-hc-mixes-rdna2 --iterations 100 --dump-output /tmp/hc-mixes.bin
GGML_HIP_RDNA2_HC_MIXES=1 \
build/bin/test-hc-mixes-rdna2 --iterations 100 \
  --compare-output /tmp/hc-mixes.bin
```

The benchmark also checks a double-accumulation CPU reference. Hold
`GGML_HIP_RDNA2_MMQ_J=16` constant in every arm of whole-model tests so only
the hidden-channel path changes.

## RDNA2 DSV4 lightning-indexer subwave path

`GGML_HIP_RDNA2_LID_SUBWAVE=4` enables the bitwise-preserving four-lane
subwave reduction for the guarded DSV4 F16 shape: 128 embedding values, 64
heads, batch 256, one stream, and KV 1–4096 on wave32 RDNA2. Unset/`0` and all
other devices, shapes, types, and backends keep the generic vector kernel.
Other values fail closed. The manual fixture requires explicit path identity;
run reference and candidate in separate processes:

```bash
GGML_HIP_RDNA2_LID_SUBWAVE=0 build/bin/test-lightning-indexer-rdna2 \
  --kv 4096 --expect-path reference --dump-output /tmp/lid-kv4096.bin
GGML_HIP_RDNA2_LID_SUBWAVE=4 build/bin/test-lightning-indexer-rdna2 \
  --kv 4096 --expect-path subwave4 --compare-output /tmp/lid-kv4096.bin
```

The fixture's candidate expectation is intentionally bound to the validated
Radeon Pro V620 target. Acceptance artifacts must additionally preserve
reference/candidate kernel-name traces; byte equality alone cannot prove the
candidate dispatched.

## Natural-text proxy validation

`scripts/dsv4-rocm/corpus/technical-proxy.txt` is a fixed 2,527-token
engineering proxy, not a user-supplied production corpus. Its SHA-256 is
`396c178b3f77e7a920473fedaa54d79d3c98df5a27baebfa9b7de62a793a71df`.
Set `DSV4_OUTPUT_DIR` when running `tests/test-dsv4-validation.sh` to preserve
layer/tensor server logs and response JSON. Run the same command with the MMQ
override unset and set to 16, using different output directories, then compare:

```bash
scripts/dsv4-rocm/compare-validation.py \
  "$artifact_root/base" "$artifact_root/candidate" \
  --json "$artifact_root/comparison.json"
```

The comparison fails if required response fields are missing/malformed, if
timings are internally inconsistent, or if content, generated token IDs,
prompts, or token counts differ. Its timing row is only one natural-text
observation; use bracketed `run-pp.sh` repetitions for performance claims.

For an acceptance-quality base/J16 run, use the safety-guarded attested wrapper
from a clean checkout:

```bash
cd /home/edwin/llama.cpp-rdna2
DSV4_HASH_MODE=full scripts/dsv4-rocm/run-corpus-validation.sh
```

The default arms remain no overrides versus J16. To isolate the second
optimization, hold J16 in both arms and enable only the candidate HC mixer:

```bash
DSV4_BASE_MMQ_J=16 DSV4_CANDIDATE_MMQ_J=16 \
DSV4_BASE_HC_MIXES=0 DSV4_CANDIDATE_HC_MIXES=1 \
scripts/dsv4-rocm/run-corpus-validation.sh
```

To isolate the lightning-indexer optimization, hold J16 and HC fixed:

```bash
DSV4_BASE_MMQ_J=16 DSV4_CANDIDATE_MMQ_J=16 \
DSV4_BASE_HC_MIXES=1 DSV4_CANDIDATE_HC_MIXES=1 \
DSV4_BASE_LID_SUBWAVE=0 DSV4_CANDIDATE_LID_SUBWAVE=4 \
scripts/dsv4-rocm/run-corpus-validation.sh
```

Empty per-arm controls mean unset; HC controls accept `0`/`1` and LID controls
accept `0`/`4`. MMQ controls must be supported multiples of eight. The wrapper
removes inherited MMQ/HC/LID variables before applying each arm, and records
the resolved controls in both self-contained command files and
`effective-settings.sh`.

It holds the shared GPU lock, refuses active ROCm processes, rechecks before
each variant, pins server batch/ubatch to 512/256 by default, and reads the
exact UTF-8 prompt bytes through `DSV4_PROMPT_FILE`,
and requires full hashes of all model shards. The artifact includes clean
source identity, executable and resolved llama/ggml DSO hashes, exact exported
settings and base/candidate command files, ROCm/hardware details, all response
JSON/logs, comparison output, and response hashes. Full model hashing occurs
before inference and can take time, but it does not consume the measured PP
budget.

## Artifacts

Runs are written to collision-resistant directories under
`$HOME/llama-jobs/dsv4-rocm-pp/`:

- `manifest.txt`: host, source, binary/all-resolved-DSO hashes, model shard metadata or full hashes, ROCm, topology, clock/performance level, power cap/profile, and environment;
- `source.patch`, `source-status.txt`, `untracked-files.sha256`: dirty-source identity;
- `effective-settings.sh`, `command.sh`, `executed-command.sh`: exact effective controls and argv;
- `result.jsonl`: raw llama-bench records, including every `samples_ts`/`samples_ns` value;
- `summary.tsv` and `summary.json`: completion state, expected/missing shapes, median/range, latency, and raw samples;
- `bench.log`: stderr loader/progress/errors;
- `bench.stdout.log`: complete raw stdout stream preserved verbatim;
- `bench.stdout-nonjson.log`, `stdout-classification.json`: separated diagnostics and fail-closed capture counts/status;
- `measurement-start.ns`, `result-completed-at.ns`: trace-alignment timestamps for parsed benchmark records only;
- `clock-domain.txt`: run-time realtime-to-monotonic clock mapping and boot ID;
- `measured-region-summary.{txt,json}`: optional filtered rocprof attribution;
- `rocm-smi.log`: one-second utilization/memory/power/clock samples (raw-TG restricts in-band sampling to setup so accepted tg32 repetitions remain unperturbed);
- `status.txt`: nanosecond timestamps, truncation, and process exit code.

Quick three-sample summaries report median and range. p05/p95 are deliberately
`NA` until at least 20 samples are available. Use `DSV4_HASH_MODE=full` only when
a one-time full read of every GGUF shard is acceptable.

## Kernel trace

Start with one shape because rocprof tracing is expensive:

```bash
DSV4_LABEL=trace-base-8k \
DSV4_PROMPTS=8192 DSV4_UBATCHES=256 DSV4_REPS=1 \
scripts/dsv4-rocm/profile-pp.sh
```

For long-context attribution where full HIP API and JSON output would consume
several GiB, use compact CSV-only tracing. It retains the kernel dispatch,
memory-copy, and RCCL inputs required by `summarize-trace.py`:

```bash
DSV4_PROFILE=kernel DSV4_PROMPTS=16384 \
DSV4_LABEL=kernel-trace-base-16k scripts/dsv4-rocm/profile-pp.sh
```

The profile wrapper disables llama-bench's warmup. Full `trace` mode records
whole-process HIP runtime, kernel, memory, and RCCL CSV/JSON under `rocprof/`;
compact `kernel` mode omits HIP API events and JSON. Model-load events are still
present in either mode. Do **not** use whole-process summary percentages for
optimization decisions. Filter the trace to the interval beginning at
`measurement-start.ns` and ending at the corresponding line in
`result-completed-at.ns`; traced throughput is not comparable to ordinary A/B
throughput.

Each new run stores start/end calibrations in `clock-domain.txt`, which map the
harness's realtime markers to rocprof's monotonic timestamps and bound clock
drift across the run. Generate measured-region text and JSON summaries with:

```bash
run_dir=/home/edwin/llama-jobs/dsv4-rocm-pp/<trace-run>
scripts/dsv4-rocm/summarize-trace.py "$run_dir" --top 30 \
  --json "$run_dir/measured-region-summary.json" \
  | tee "$run_dir/measured-region-summary.txt"
```

Kernel durations are clipped to the measured interval and summed across all
devices/queues, so their wall-equivalent percentage can exceed 100%. The
summarizer requires a kernel trace/nonempty measured interval and rejects more
than 1 ms of start-to-end clock-offset drift by default. A legacy trace without
`clock-domain.txt` requires its run-time realtime-minus-monotonic offset via
`--clock-offset-ns`; do not reconstruct that offset after a reboot. A single
legacy calibration has unknown boundary uncertainty and is labeled as such.

For multi-GPU imbalance, preserve a per-agent view after generating the main
summary:

```bash
scripts/dsv4-rocm/analyze-trace-agents.py "$run_dir" \
  --json "$run_dir/measured-region-agents.json" \
  | tee "$run_dir/measured-region-agents.txt"
```

This maps rocprof KFD agents to PCI BDFs through the run's `agent_info.csv` and
reports measured-region all-kernel sums plus conservative kernel-name-match
families and copy endpoints. Name matches overlap and are not operation-level
proof. Durations are clipped and then summed over queues; they expose imbalance
but do not by themselves prove PCIe causality or
link bandwidth.

## Production MTP validation

After the main-model PP controls pass, exercise the accepted stack with the
production context/KV/batch settings and the Q4 MTP draft model:

```bash
timeout --signal=TERM --kill-after=30s 2400s env \
GGML_HIP_RDNA2_MMQ_J=16 GGML_HIP_RDNA2_HC_MIXES=1 \
DSV4_OUTER_TIMEOUT=2400 \
scripts/dsv4-rocm/run-production-mtp-validation.sh
```

The runner holds the shared GPU lock, rejects active ROCm processes again
immediately before each server launch, uses a 262144-token unified F16 KV
context with batch/ubatch 512/256, and compares a fresh main-only server with a
fresh MTP server (`draft-mtp`, Q4_0, n-max 3, ROCm0+ROCm1). It fully hashes the
main shards, draft model, server/DSOs, corpus, and its embedded request client.
The fixed request processes the engineering proxy before generating 128 tokens.
Content and token IDs must match exactly; the MTP arm must report nonzero draft
attempts and drafted tokens. PP and TG timings plus draft acceptance are one
matched main-then-MTP observation, not a stable speed claim.

Default output is a new `$HOME/llama-jobs/dsv4-production-mtp-*` directory. The
outer timeout includes model hashing/loading; the request itself is capped at
five minutes, so model preparation remains outside the measured PP/TG fields.
Do not use production port 8080.

Bounded diagnostics may set `DSV4_DRAFT_N_MAX=1`, `DSV4_N_PREDICT=64`, and
`DSV4_HASH_MODE=quick` after linking back to one full-hash run. Set
`GGML_CUDA_DISABLE_GRAPHS=1` to disable runtime HIP/CUDA graph capture; absence
means enabled. `GGML_HIP_GRAPHS` is a build-time CMake option in this checkout,
not a runtime off switch. Every diagnostic arm records the resolved n-max,
hash mode, and graph-disable presence.

## Summarize existing output

```bash
scripts/dsv4-rocm/summarize.py result.jsonl \
  --expected-prompts 512,2048 --expected-ubatches 256 --expected-reps 3 \
  --json summary.json --tsv summary.tsv
```

`--allow-trailing-partial` is accepted only together with `--truncated`. It may
drop one malformed final JSONL fragment after forced termination; malformed
middle records remain errors and the raw file is never rewritten.

## Validation sequence

1. Run a complete 512/2K quick baseline at ubatch 256.
2. Run a separate capped 8K baseline.
3. Run one 8K trace and filter it to the recorded measured interval.
4. Select a patch only from matched complete shapes and measured-region trace data.
5. Validate exploratory wins on one fixed, recorded production-representative token corpus because llama-bench's synthetic tokens can change MoE routing.
6. Use paired/interleaved base-candidate-base ordering and review clocks/temperature.
7. Add ubatch 128/512 and 32K+ only for final validation.
8. Preserve all raw directories; never cite only the fastest sample.