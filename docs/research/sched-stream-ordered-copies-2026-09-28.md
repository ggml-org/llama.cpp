# Stream-ordered split input copies (Andrei-Dr 0018/0020 port) - 2026-09-28

Development log for porting Andrei-Dr local-ai patches 0018 and 0020 (skip the host sync
before a stream-ordered split input copy). It records the choices, the obstacles and the
evidence.

Current policy: synchronous copies are the default. Only exact
`GGML_SCHED_COPY_SYNC=0` enables the experiment; the earlier default-on campaign
below records historical behavior. The failed output comparison remains a gate
for any future default-on promotion.

Evidence labels: **measured** means tool output on this host; **source** means read from code;
**author** means Andrei-Dr's numbers on their hardware.

## What the patches do

Source: github.com/Andrei-Dr/local-ai, `research/patches/mainline-series/0018-*.patch` and
`0020-*.patch`.

In `ggml_backend_sched_compute_splits` (`ggml/src/ggml-backend.cpp`), a split input from
another backend used to be handled like this when there are no pipeline-parallel events
(single-GPU):

1. the host synchronizes the split backend fully, so the device drains;
2. SYCL has no `cpy_tensor_async`, so the generic branch synchronizes again and runs a blocking
   `ggml_backend_tensor_copy`. For a SYCL destination that is
   `ggml_backend_sycl_buffer_set_tensor`, which runs `queues_wait_and_throw()` on the device
   and then waits on the memcpy.

In decode with host-resident experts (`--n-cpu-moe`, `--fit`), every MoE layer returns its CPU
expert output to the GPU through this path. 0018 notices that a copy enqueued with
`set_tensor_async` on the split backend's own stream is already ordered after that stream's
earlier reads of the destination, so steps 1 and 2 become one async enqueue. 0020 makes this
the default and adds `GGML_SCHED_COPY_SYNC=1` to restore the wait. Author: host overhead per
MoE layer went from 89.5 to 62.7 us (nsys, GTX 1650 SUPER, CUDA). Output was IDENTICAL x4.
Decode t/s did not change beyond noise.

## Choices

1. **0018 + 0020 only, without 0010.** 0018's second hunk is written against 0010's version of
   the generic branch, which master does not have. Its `stream_ordered` branch already contains
   the async upload that 0010 introduces, so for single-GPU (no events) nothing else is needed.
   0010's remaining change affects only the pipeline-parallel event path, which this fork does
   not run. That path is excluded explicitly: `stream_ordered` requires `n_copies == 1`
   as well as no event for the split backend. A null event alone is insufficient: parallel
   schedulers can lack events, for example with `GGML_SYCL_NO_PEER_COPY`. Review caught both
   the initial event-path change and this eventless parallel case.
2. **Only backends that declare stream order, currently SYCL only** (the user's choice).
   - A new proc-address hook, `ggml_backend_async_is_stream_ordered`, with its typedef in
     `ggml-backend.h`. The scheduler caches its answer per backend in `ggml_backend_sched_new`.
   - It also requires the scheduler buffer type for that backend to be the backend's default:
     tensor copies live there, and SYCL's `set_tensor_async` asserts on any other buffer type.
   - Andrei's patches apply to every backend with `set_tensor_async`. Here Vulkan keeps the
     synchronize. Vulkan records async writes into its compute context, or into a separate
     transfer queue on AMD dGPUs (`ggml-vulkan.cpp`, `ggml_backend_vk_set_tensor_2d_async`).
     Whether such a write is barriered against an earlier dispatch still reading the buffer was
     not audited. Its `synchronize` also runs `ggml_vk_graph_cleanup`.
3. **Evidence that SYCL is stream-ordered** (source, grep of `ggml/src/ggml-sycl/`):
   - every op submits to `ctx.stream()`;
   - the buffer set/get/copy calls, the host allocator (`common.cpp`, `malloc_host`) and the
     MoE cache (`moe-cache.cpp`, `sctx->stream()`) all use the device's one dpct in-order default
     queue;
   - the only other queue is PR 72's private stream, which belongs to the prefetch backend and
     is never a scheduler split backend;
   - the in-order property itself was confirmed on the A770 in
     `sycl-prefetch-second-queue-2026-09-28.md`.
   - **This holds for one device only.** With several SYCL devices, tensor-split `MUL_MAT`
     submits to other devices' queues (`ctx.stream(i, is)`) and joins them back onto the main
     queue with barriers only at the end of the op. Stream order would then depend on every such
     path always joining before it returns, which has not been audited, so the hook returns
     `ggml_sycl_info().device_count == 1`. Review caught this on the first push.
4. **A host-source lifetime guard, which is new relative to the patches.** The device reads the
   host source when the async memcpy executes, not at submit time. 0010's comment argues the
   source cannot be overwritten early, because any later host split first synchronizes the
   device backend. That argument does not cover two cases:
   - a host split whose inputs are all user inputs, which synchronizes only itself;
   - the next graph. `ggml_gallocr_free_node` (`ggml-alloc.c`) protects only OUTPUT tensors, so
     an intermediate may reuse a graph input's memory. The next ubatch's `set_inputs` then
     writes there while the previous graph's last upload may still be queued.

   The guard works like this:
   - `pending_host_reads[b]` is set when a stream-ordered upload reads a non-WEIGHTS host
     source. An event is recorded immediately after the upload, before later kernels.
     Weights are immutable, so they never set it. Mutable sources retain the blocking
     copy path when the backend cannot provide an upload-completion event.
   - Before a host split (`ggml_backend_buft_is_host(bufts[b])`, which covers CPU and BLAS)
     starts, including its own input copies, each pending upload event is waited.
   - After every `ggml_backend_sched_compute_splits`, whatever its status (in
     `ggml_backend_sched_graph_compute_async`), the same happens, and `ggml_backend_sched_synchronize`
     clears the flags.
   - Upload-event waits leave later device kernels in flight. A CPU split that depends on
     their output still performs its existing backend synchronization. The first revision
     drained the whole backend in the guard, causing a redundant full wait on that path
     and draining final device work before async graph return. That claim of merely moving
     an existing wait was incorrect. Event waits add work; their isolated cost is unmeasured. End-to-end timings are recorded below.
5. **MoE-weight and prefetch inputs.** When the stream-ordered condition holds for these, the
   step-1 wait is skipped as well. This is safe because the SYCL stream is in-order: the routed
   copy into the destination runs after every earlier read of it. The ids readback is not the
   argument, because it does not always synchronize the split backend. The weight sources are
   immutable. Prefetch writes a staging slot, not the destination's own memory.
   CPU-from-pointer buffers are excluded from the new stream-ordered path, including ordinary
   mapped MUL_MAT weights. Their existing copy path preserves the SYCL/PVC malloc staging
   workaround; the scheduler must not pass a mapped address directly to the async setter.
6. **Counters.** The number of stream-ordered copies, flushes at a host split and flushes at graph
   end are printed as one `GGML_LOG_DEBUG` line when the scheduler is freed, visible with `-v`.
   These count upload-event waits separately from full backend synchronization; they do not
   establish a performance improvement.

## Obstacles

- **0018 is not self-contained.** Its context lines are 0010's code, so it had to be applied by
  hand against the pre-0010 generic branch.
- **The author's lifetime argument has gaps**, as described in choice 4. This was found while
  reviewing the patch, not from a failure.
- **The GPU was reserved for another session's Ornith benchmark** during development. The
  GPU runtime checks were pending at that stage; the 2026-10-06 campaign is recorded below.

## Verification

2026-10-05 review follow-up, after rebasing onto master `a17b8400d`:

- The initial CPU-only `test-sched-stream-ordered` at `c834a7490` used real scheduler splits with a CPU-backed
  destination advertising stream order but returning null events. It checks the selected
  copy path and output values across repeated graph computations in both scheduler modes.
- Before the single-copy fix, the serial control passed with three async uploads; parallel mode
  incorrectly made nine async uploads and failed. With `n_copies == 1` required, serial
  still made three async uploads and parallel mode made zero, retaining synchronization.
- That initial test checked scheduler selection, not delayed device execution or GPU host-source lifetime.
  GPU correctness, model output comparisons, and timing runs were pending at that revision.

The subsequent lifetime follow-up replaces that immediate-copy mock with a deterministic
deferred queue. The CPU-only build and CTest pass seven cases: successful async return,
an independent host split with only user inputs, a dependent host split, failed device
compute after upload submission, serial and parallel null-event fallback, and mapped-buffer
WEIGHTS in ordinary MUL_MAT. Successful and failed async returns release the upload source;
successful returns leave unrelated compute queued. The host-input-only case overwrites the
source before graph return and checks that the uploaded values remain correct. The dependent
host split requires exactly one full producer synchronization before its computation.

Four temporary negative controls each failed the test: removing the host-split flush,
removing the graph-return flush, removing mapped-buffer exclusion, and replacing event waits
with full destination synchronization. The production source was restored and CTest passed
again. Run it with `ctest --test-dir BUILD -R '^test-sched-stream-ordered$' --output-on-failure`.
These synthetic checks do not verify SYCL event execution, the PVC driver workaround itself,
or runtime performance.

## 2026-10-06 author-requested CPU/GPU campaign

**The tests were executed, but the byte-identical output criterion failed. This
campaign does not justify enabling the path by default.** The forced-sync baseline also
changes output between repetitions; this does not isolate a regression to the
new copy path and does not prove that path equivalent.

Tested source: `9b231926c38beff3b5d887facd5b410a1a28725e`, clean before testing.
This follow-up changes evidence only. Machine-readable commands, prompts, full
generated outputs, hashes, counters, CPU/GPU gate output, and benchmark records:
[`sched-stream-ordered-copies-2026-10-06.json`](sched-stream-ordered-copies-2026-10-06.json).
Raw local logs: `/home/svnbjrn/pr84-validation-2026-10-06`.

### Configuration and isolation

- Fresh Release JIT build with icpx/icx 2026.1.1, SYCL F16 enabled, DNN disabled,
  compiler launchers and ccache disabled. Built `llama-completion`, `llama-bench`,
  `llama-tokenize`, `test-sycl-turbo-correctness`, and `test-sched-stream-ordered`
  with four build jobs; configure and build returned 0.
- Ryzen 9 7900X3D, one visible Arc A770 on the `xe` driver;
  kernel `7.3.0-rc5-273-linux73-tkg-bore-rc5-xe`, IGC `2.41.10`, compute-runtime
  package `22.43.24558.r13063.gbe85a8d685-1`, Level Zero loader
  `1.34.0.r2.gae3db48-1`. Full version strings are in the JSON.
- Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL, 13,806,312,608 bytes, SHA256
  `69cd7578d77dffc0b17e34bad9ef998d08ae0e20ccceef21bad4e7eb3d8c553b`.
- `ONEAPI_DEVICE_SELECTOR=level_zero:0`, `SYCL_CACHE_PERSISTENT=1`,
  `GGML_SYCL_ENABLE_GRAPH=0`; inherited `GGML_*`, `LLAMA_ARG_*`, `LLAMA_TEST_*`,
  `SYCL_*`, and `UR_L0_*` overrides were removed before setting these values.
  The backend applied its default xe copy-engine workaround.
- The llama service was inactive initially, was explicitly stopped before
  timing, and remained inactive afterwards. GPU ownership was checked before
  each process and every five seconds while it ran: 124 probes across 29
  successful GPU processes, no foreign owner observed. This is polling, not an
  exclusive device lock; the desktop and CPU were not exclusively reserved.
  Each model process had a 1,800-second timeout, the GPU gate 600 seconds.

The original kernel filter missed `Timedout job` and reverse-order
`Fence expiration time out i915-...`. After review, the retained kernel journal
was re-read for 00:38:50-00:49:30 UTC, enclosing all 29 processes. Its 27
unfiltered lines produced zero matches using the independent driver and failure
patterns from `scripts/perf/bench_spec.py`. Both reported timeout spellings match
that predicate. The JSON preserves the original executed filter and adds this
rescan's command, patterns, count, and raw-log hash; unrelated firewall entries
are not copied into the repository.

### Correctness results

| Check | Observed result |
| --- | --- |
| Fresh CPU scheduler CTest | Pass; all seven deferred-queue cases |
| A770 default `test-sycl-turbo-correctness` | rc 0, 0 GATE-FAIL, 0 XPASS |
| 65-token prompt, 256 generated tokens, three runs per mode | Fail: six distinct output byte hashes |
| Exactly 2,500 prompt tokens, `-b 16 -ub 16`, 256 generated tokens, three runs per mode | Fail: six distinct output byte hashes |
| Forced sync, one CPU thread, two short-prompt runs | Two distinct outputs |
| Forced sync, SYCL fusion and FFN fusion disabled, two short-prompt runs | Two distinct outputs |
| Forced sync, `--moe-cache off`, two short-prompt runs | Two distinct outputs |
| Kernel GPU reset/hang/timeout/fault/error/GuC filter | Zero matching lines before and after |

The default GPU gate reported three turbo2 accuracy warnings. Turbo FA, d=256 FA,
and InnerQ opt-in sections remained disabled; its summary's `0 SKIP` counter does
not count those disabled sections. CPU testing here means the scheduler's
CPU-backed regression, not a CPU-only run of the 30B model.

For both prompts, each mode itself produced three distinct outputs. These are
actual generated-text differences, not timestamps or logging in stdout. All
12 processes returned 0; the comparison stage returned 1. Neither reducing
CPU threads nor disabling fusion or the MoE cache made the sync baseline
repeatable. Earlier temperature-zero nondeterminism is recorded in
[the checkpoint campaign](speculative/sycl-a770-spec-checkpoint-on-device-ab-2026-10-04.md),
but that history is not a root-cause explanation for this failure.

Every enabled short-prompt run recorded 5,377 stream-ordered copies and 4,864
host-split upload-event waits; every enabled long-prompt run recorded 8,673 and
7,847 respectively. Graph-end waits were zero for these model graphs. Forced-sync
runs emitted no stream-ordered counter line. The real model therefore exercises
the branch; synthetic CPU tests remain the evidence for graph-return lifetime
handling and failure paths.

Common completion arguments (prompts are embedded in the JSON):

```bash
llama-completion -m MODEL -f PROMPT -ngl 99 --n-cpu-moe 20 -c 4096 -n 256 \
  -b 512 -ub 512 -t 12 -tb 12 -fa on -ctk f16 -ctv f16 --fit off \
  --temp 0 --seed 1 --ignore-eos -no-cnv --no-display-prompt \
  --simple-io --color off --log-verbosity 5
# Long prompt: replace -b 512 -ub 512 with -b 16 -ub 16.
# At tested revision 9b231926c: baseline=1, enabled arm unset.
# With the opt-in follow-up: baseline unset or 1, enabled arm exactly 0.
```

### Alternating decode timing

Five pairs, enabled then forced-sync, one process per repetition. Each process
used `llama-bench -p 0 -n 128 -ncmoe 20 -ngl 99 -b 512 -ub 512 -t 12
-fa 1 -ctk f16 -ctv f16 -r 1 -o json -v` and the model above. Verbose logging
exposes the branch counters in both arms. The default automatic MoE cache was
active in these benchmark runs; these are not cache-disabled measurements.
The benchmark uses a fixed token generator with the same seed for repetition
zero in every process, rather than sampling from its varying logits.

| Pair | Enabled tok/s | Forced-sync tok/s |
| --- | ---: | ---: |
| 1 | 19.218901 | 19.321719 |
| 2 | 19.466073 | 19.452151 |
| 3 | 19.212976 | 19.501421 |
| 4 | 19.410218 | 19.439509 |
| 5 | 19.539309 | 19.414470 |
| Mean +/- sample SD | 19.3695 +/- 0.1475 | 19.4259 +/- 0.0663 |

The ratio of means is **-0.29%**. No throughput improvement is demonstrated.
Every enabled benchmark recorded 2,709 stream-ordered copies and 2,451 host-split
upload-event waits; forced-sync runs recorded none. These timings measure the
whole selected configuration, not upload-event overhead in isolation.

## 2026-10-06 opt-in follow-up

The review allowed an opt-in fallback while real-model equivalence is unproven.
`GGML_SCHED_COPY_SYNC` now defaults to synchronization: only the exact string `0`
selects the experimental path. Unset, empty, and every other string retain the
existing copy behavior. Extra mutable-upload events are not allocated while the
experiment is disabled; normal pipeline events are unchanged.

The CPU regression now runs with explicit `0`, unset, and explicit `1`. Its
synchronous-selection case uses an event-capable destination and asserts zero
async uploads and zero allocated upload events, so lack of backend support cannot
hide an accidental default-on path. Additional subprocess checks cover empty,
invalid, `00`, whitespace, and negative values; forcing `0` while expecting sync
is a negative control. The seven opt-in lifetime scenarios are retained.

Validation: incremental SYCL build passed; all three CTest registrations passed
(seven opt-in lifetime cases, unset default, explicit `1`). The extra environment
checks passed, and explicit `0` correctly failed the sync expectation. On the
A770, the default GPU gate again returned 0 with zero GATE-FAIL/XPASS. One real
model run with the variable unset recorded no stream-ordered copies; one with
explicit `0` recorded 5,377 copies and 4,864 host-split event waits. Both exited 0.
These are selection checks, not an output-equivalence claim. The broader kernel
journal predicate found no new GPU faults; the service stayed inactive. The
JSON's `opt_in_followup` records the tested source hashes, commands, and results.

The complete runtime contract is documented in
[SYCL.md](../backend/SYCL.md#scheduler-input-copy-synchronization), with usage and
validation reminders in `AGENTS.md`, `CLAUDE.md`, and the scheduler's code comment.
This policy change does not fix or waive the earlier nondeterminism, and does not
claim a speedup.

## Not claimed

- Vulkan and OpenVINO behaviour is unchanged by construction: they do not export the hook. That
  is verified from source only.
- Pipeline-parallel schedulers (`n_copies > 1`) cannot take the stream-ordered branch.
  The eventless case was exercised with the CPU-backed test above; GPU event-allocation failure was not.
  Multi-device SYCL keeps the old syncs because the hook returns false there; it was not run.
- Byte-identical model output and merge readiness are not established. The baseline
  nondeterminism remains unexplained; no fix for that nondeterminism is claimed.
- SYCL graph replay, AOT, multi-GPU, PVC hardware, explicit `--moe-cache on`, and
  injected GPU event failures were not tested. Automatic MoE caching was active
  in the timing runs; no separate cache-correctness claim follows from them.
- No CPU-only 30B inference, isolated event-overhead profile, or installed-binary
  update was performed. No production dependency or state is added by this
  opt-in follow-up.
