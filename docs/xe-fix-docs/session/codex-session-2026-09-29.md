# Codex xe investigation and research session

Archived 2026-09-30 at the user's request. Investigation and research took place
on 2026-09-29. Times below are UTC (also local Atlantic/Reykjavik time).

## Read this first

This is the record of the source-audit, copy-tracer, driver/firmware and
interactivity-research session. It complements the pre-existing documentation
from other investigations; it does not claim ownership of their experiments,
production changes or later kernel work.

- [Full final research report](../FINAL_REPORT.md): the report requested by the
  user, with portable links and ASCII-normalized text.
- [Final user-facing response](final-response-2026-09-29.md): the delivered
  explanation, with its report link pointed at this archive.
- [Correction guide](#corrections-and-evidence-precedence): essential when
  reading older summaries in this directory.
- [Source manifest](source-manifest-2026-09-30.json): original paths, archived
  paths, current and original byte counts and SHA-256 for 173 committed files.
  Nineteen unbundled artifacts are listed separately under `omitted`; this is
  a partial archive, and historical tracer wrappers are not directly runnable.

This narrative reconstructs the session from available conversation context,
saved artifacts and tool results. It is not represented as a verbatim transcript
of every earlier message or tool call. Files copied now reflect the available
source versions at archival time; timestamps and narratives inside them establish
their individual provenance. Later edits by other investigators can be present.

## User objective and constraints

The user initially requested fetch/pull of remote master and an explanation of
xe behavior on Ornith using the saved benchmark campaign. They supplied both
performance comparisons and failure descriptions, directed us to the September
27 research directory, and objected to recommending an immediate i915 rollback
without trying to fix xe.

The user then requested coordination with another investigation already using
the GPU, later indicated it might be free, and asked for a recheck. Finally they
requested research into the llama.cpp/driver interaction, xe/i915 differences,
newer firmware and configuration for interactive inference on both drivers.

The agreed approach was evidence before attribution: preserve production work,
avoid overlapping GPU tests, collect concrete copy metadata and retain the
useful xe configuration while identifying a narrower failure. Research alone
did not authorize a firmware flash or require a reboot; none was performed.

Repository master was fetched and fast-forwarded 33 commits to `4e7400c3a` from
a clean checkout earlier in this session. The source audit and benchmark binary
were pinned to that revision. This session made no backend-code patch or PR.
At archival time other sessions have source changes and reproducers in the
working tree; they have been left untouched.

## Initial evidence, before corrections

The supplied Ornith benchmark compared i915 September 28 round 2 with five xe
configurations. Reported i915 values were approximately 125.3 pp512, 18.13 tg64,
107.1 pp512 at 8k and 16.81 tg64 at 8k. Xe prefill was roughly 55-62% higher;
decode was 20-29% lower under the tested placement.

The initial cases included silent hangs and a BCS reset followed by an
`OUT_OF_DEVICE_MEMORY` error. Copy-engine-off completed and improved the
bench decode number modestly. Production subsequently also reset BCS. The
initial inference was that host-expert streaming exposed a copy-path problem.

The user already qualified several claims: enqueue stacks alone do not prove
an unsignaled dependency; command-stream failure does not identify its author;
OOM after reset is not proof of physical VRAM exhaustion; placement capacity
does not imply equal cache hits. Those qualifications remain valid.

Two later discoveries materially changed the account: the trace copied expert
weights rather than activations, and the benchmark used `--moe-cache auto`
despite production explicitly setting `off`. The established slowdown belongs
to the benchmark placement. It is neither erased nor proven for production.

## Source audit and coordination chronology

### Before GPU access: inspect the actual paths

The backend scheduler was checked instead of assuming the optional MoE cache
implementation caused the transfers. At `4e7400c3a`, direct scheduler async-set
sites included `copy_experts` and the separate prefetch path. Prefetch was not
enabled in these tests. `copy_experts` uploads selected host-resident weights
for GPU `MUL_MAT_ID`. The SYCL backend's async tensor-copy interface entry was
NULL, so generic CPU/GPU tensor-copy fallback has different synchronization.

This establishes the code paths available in the revision. It does not prove
that every placement exercises the expert-upload path. Our earlier extension
of this finding to production `off` was unsupported and was corrected.

Queue construction, graph gating, pinned-memory settings and installed runtime
symbols were inspected. The actual graph knob is `GGML_SYCL_ENABLE_GRAPH`.
The stale `GGML_SYCL_DISABLE_GRAPHS` variable in older guidance is not parsed.

### Coredump attribution

The file named `xe-devcoredump-132039.txt.gz` contains an earlier event:

| Item | Saved server dump | Separate 13:20 event |
| --- | --- | --- |
| Time | 13:15:16, snapshot `1790687716.869280196` | 13:20:39 |
| PID | 264942 | 318738 |
| Sequence number | 11684 | 7811 |
| Saved IPEHR | `0x72080025` | Not established by that dump |

The 13:15 identity was subsequently matched to the kernel journal, saved in
[kernel-131516-confirmed.txt](../artifacts/kernel-131516-confirmed.txt).
This resolved the mismatch; it was not a reason to discard the dump or merge
the two crashes. The pinned-off dump had `IPEHR=0xfffff000`, a distinct value.

The server dump's `0x72080025` equals the Xe-HPG COMPUTE_WALKER initial header:

```text
0x25 | (2 << 18) | (2 << 24) | (2 << 27) | (3 << 29) = 0x72080025
```

The source excerpt is archived in
[compute-walker-source-excerpt.txt](../artifacts/compute-walker-source-excerpt.txt).
This strengthens the command-stream/mapping lead but does not prove wrong-engine
submission: reused storage, overwritten buffers, stale mappings/visibility and
misleading capture remain possible. VM state capture failed with -19 and there
was insufficient batch content to settle the cause.

Both initially inspected dumps showed queue Timeout=0 and preemption timeout
640000 us. Attributing their reset solely to the ordinary 5000 ms BCS job
timeout was unsupported. LR mode does not mean that no reset can ever occur.

### 16:43-16:54: tracer preparation, no GPU overlap

A bounded LD_PRELOAD tracer was implemented in
[trace-copies.c](../artifacts/trace-copies.c). It intercepts exported graph-submit
and async-set calls, logs metadata for the first 64 graph-internal uploads and
up to four outside uploads, and emits totals on normal exit. It records tensor
name, type, shape, offset and size, not tensor payloads. Atomic counters and
thread-local graph context keep the diagnostic state bounded.

The [CPU forwarding check](../artifacts/check-trace.c) compiled with warnings
as errors and passed a real four-float upload/readback. PLT inspection showed
that interception was applicable to the installed library. This was initially
only a CPU/forwarding check, not proof of GPU-path coverage.

Compile/check commands used:

```sh
cc -shared -fPIC -Wall -Wextra -Werror \
  -I /mnt/mrgr/ggml-llama.cpp/ggml/include trace-copies.c -ldl -o trace-copies.so
cc -Wall -Wextra -Werror \
  -I /mnt/mrgr/ggml-llama.cpp/ggml/include check-trace.c \
  -lggml-cpu -lggml-base -lggml-sycl -o check-trace
LD_PRELOAD=./trace-copies.so ./check-trace
```

An executable [wrapper](../artifacts/bin/llama-bench) applies LD_PRELOAD only
to llama-bench, not the shell/timeout process, and passed `bash -n`. Existing
runner supervision provided service stop/restart and the 420-second timeout.
Saved scripts contain original absolute paths and are historical recipes, not
portable commands to execute without inspecting them.

At 16:54 the other session's `N-xe-inorder0` was still active, so this session
did not launch GPU work. The other investigator had completed additional
copy-off rounds and six server requests and had installed the production
copy-off drop-in. That work was preserved.

### 17:35 onward: two instrumented GPU runs

After the GPU handoff, the default-copy traced run produced approximately
197.77 pp512 and 13.60 tg64, then stalled before the 8k prefill result. Two
stacks placed the host in `urQueueFinish -> queue_impl::wait`. BCS counters
advanced while CCS counters did not; the host consumed approximately one core.
This differs from the earlier CCS-busy/BCS-idle observation and must not be
flattened into one universal engine state.

Trace records identified multi-megabyte Q4_K gate/up expert-weight uploads.
A second run with `UR_L0_DISABLE_EVENTS_CACHING=1` stalled before its first
result row in `enqueueMemCopyHelper -> urEnqueueUSMMemcpy -> async set`.
GDB identified the pending tensor precisely:

```text
name   = SYCL0#blk.1.ffn_down_exps.weight#0
type   = GGML_TYPE_Q6_K
shape  = {512, 2048, 256, 1}
offset = 202137600
size   = 860672 bytes
```

Separately rebuilt debug symbols had byte-identical `.text` to the loaded
tracer before relocation into GDB. Original-address GDB commands are archived
in [inspect-live.gdb](../artifacts/inspect-live.gdb); their PID/address values
are not reusable across processes.

Both runs were manually SIGTERM-terminated after repeated stalled snapshots,
exit 143, before their watchdog deadline. Neither is a completed benchmark.
The contemporaneous capture reported no new reset observed by this session;
later files/coredumps added to the shared evidence directory must be attributed
using their own metadata. Do not infer the absence of all later kernel events
from that observation. The service was restored and its health endpoint returned
`{"status":"ok"}`; the existing copy-off environment was verified.

The [corrected trace account](../evidence/copy-path-trace.md) and raw
[default trace](../evidence/raw/N-xe-copytrace/) and
[event-cache-off trace](../evidence/raw/N-xe-copytrace-nocache/) retain stacks,
maps, metadata, counter samples and partial output. Instrumentation changes
timing, so these numbers are diagnostic, not clean performance comparisons.

### 17:49-18:05: placement correction from the other investigation

The campaign audit established that historical bench runs inherited `auto`,
including i915 references. Production's explicit `off` selects different
placement. Recorded xe copy-off bench decode was 14.52/13.70 t/s for auto at
short/8k depth, versus 47.55/42.07 for off. Real-text off placement was 32.78
t/s copy-off versus 32.38 copy-on. Those are not an i915/xe comparison.

Our trace write-up was corrected to state its actual auto placement. The claim
that random tokens are necessarily a worst case for this workload's locality
was not retained as an established result.

## Driver, firmware and interactivity research

The research skill was used, with three delegated, read-only research lanes:

| Lane | Archived note | Responsibility |
| --- | --- | --- |
| Driver mechanisms | [driver-mechanisms.md](../artifacts/driver-mechanisms.md) | VM_BIND, eviction/rebinding, dependencies, LR mode, BCS/CCS and support status |
| Firmware | [firmware-research.md](../artifacts/firmware-research.md) | Loaded/disk/initramfs/upstream comparison, DG2 compatibility, board inventory |
| Interactive tuning | [interactive-tuning.md](../artifacts/interactive-tuning.md) | Current configuration, latency controls, placement and fair comparisons |

The mechanism lane's opening wording about `--moe-cache off` predates the final
placement correction. It is retained as historical research, not the final
statement on production transfers. The final report supersedes it.

Primary-source work established:

- Xe separates bind and execution ordering and uses TTM eviction/rebind machinery;
  switching KMD changes NEO's implementation as well as the kernel path.
- Both drivers default to GuC submission on DG2; that is not the differentiator.
- NEO 26.35 creates Xe LR VMs; ordinary job timeout is not a universal watchdog.
- DG2 remains force-probe under xe in kernel 7.3-rc1.
- DG2's NEO direct-submission table enables CCS but has no BCS enable entry;
  a Battlemage report cannot be transplanted as an A770 diagnosis.
- Immediate binding is not the same as kernel pinning. NEO residency terminology
  must not be confused with permanent physical residency.
- Source flags found in a binary are not proof that a knob is enabled, honored
  by the active adapter, or relevant to the active engine.

Runtime source snapshots are in `artifacts/`. `neo-26.35-*` refers to the
installed NEO release tag. `ur-current-*` was fetched from Intel LLVM's current
SYCL branch, not proven identical to the installed UR. `ur-reference-*` is an
older weekly-2025-01-17 reference, not the installed source. The generated
command excerpt was checked against local NEO commit
`d70df548ba9ddc71b8cf0cd76983944827a7087f`. Exact provenance limits matter.

Firmware inspection found loaded GuC 70.53.0 matching current upstream,
installed file and the active kernel's initramfs byte-for-byte; loaded DMC 2.08
matched current firmware. Xe's DG2 HuC N/A corresponds to explicit upstream
non-support. Read-only IGSC inventory found board GSC `DG02_1.3266` and
subsystem `172f:3937`; no authoritative newer compatible board image was
established. No firmware was installed or flashed.

## Host snapshot used for the report

| Item | Observed September 29 |
| --- | --- |
| Kernel | `7.3.0-rc1-273-tkg-bore` |
| NEO | `26.35.39758.10-1.1` |
| IGC | `1:2.41.5-1.1` |
| Level Zero loader | `1.32.0-1.1` |
| Linux firmware | `20260929.33b68e2c-1` |
| Installed llama.cpp | `b12305.4e7400c3a-1` |
| GPU | A770, `0000:03:00.0`, `8086:56a0`, xe |
| Frequency request bounds | 700-2400 MHz; RPe 700; idle actual 0 |
| Power cap | 230 W |
| Engine timeslice | 1000 us |
| CCS/RCS preemption timeout | 7500000 us |
| Other engine preemption timeout | 640000 us |
| CCS/BCS ordinary job timeout | 10000 ms |
| CCS mode | 1 |
| Runtime power management | auto |

Production service at that snapshot:
`llama-gpu@Ornith-1.5-35B-Q4_K_M.service`.
Actual drop-in path recorded by the coordinated investigation:
`/etc/systemd/system/llama-gpu@Ornith-1.5-35B-Q4_K_M.service.d/xe-copy-engine.conf`.
Older summaries naming `llama-sycl.service.d` are not the observed unit path.

Whitelisted environment inspected: `ONEAPI_DEVICE_SELECTOR=level_zero:0`,
`UR_L0_USE_COPY_ENGINE=0`, `GGML_SYCL_ENABLE_GRAPH=1`,
`GGML_SYCL_GRAPH_EVICTION_TIMEOUT=300`. Selected arguments: context 131072,
parallel 1, fit on/target 1024, threads and batch threads 12, flash attention on,
q8_0 K/V, ngram-mod speculation and moe-cache off. This is a historical snapshot,
not a claim of current state after later kernel or application work.

## Corrections and evidence precedence

Use the final report and explicit corrections here before interpreting older
summaries. Keep raw captures unchanged and identify each capture independently.

| Older claim | Correct interpretation |
| --- | --- |
| Root cause is a proven cross-engine deadlock | Ordering/lifetime/mapping failure is the leading family; exact defect and owner remain unknown |
| Copy-off eliminated the failure | Four benchmark rounds and six server requests passed; finite, heterogeneous evidence |
| No inherent xe decode regression | Regression measured for auto placement; production off comparison is missing |
| i915 production achieves approximately 32 t/s | No matched measurement established; do not invent a baseline |
| The tracer proves production off streams those experts | Traced bench used auto; tensor identity is proven only for those runs |
| i915 success rules out application bugs | Timing-exposed application lifetime/ordering bugs remain possible |
| Event caching off rules out event bugs | One altered configuration stalled; event defects are not excluded |
| The two server crashes share one coredump identity | Saved filename names 13:20, header/journal establish 13:15; separate events |
| IPEHR proves the wrong engine received a compute batch | Header match is a lead; batch/mapping/capture alternatives remain |
| No first result row means model-load stall/no kernels | Trace contains kernel activity; output absence alone does not locate the phase |
| Timeout inflation is harmless | It may delay recovery; it does not speed scheduling or repair ordering |
| LR means resets cannot happen | Ordinary job-timeout limits differ; preemption/reset paths still exist |
| Copy-off disables every possible BCS operation | It changes runtime copy selection; it does not disable all kernel migration |
| All failures and passes are exchangeable statistical trials | Configurations, contexts, load and instrumentation differ |

The old `SESSION_DOCUMENTATION.md`, `session/session-log.md` and adversarial
artifacts contain some of these stronger claims. They remain historical records,
not proof of those conclusions. Later rc5 work is separately documented in
[kernel follow-up](../analysis/kernel-7.3-rc5-xe-build-2026-09-30.md); it was not
performed or validated by this source-audit/research session.

## What was changed, verified and left open

This session produced the tracer/check, diagnostic captures, coordination notes,
source snapshots, three research notes, corrected copy-path account and final
research report. Another session installed the copy-off service drop-in. No
rollback, flash, driver change, backend fix, upstream issue submission or PR was
performed by this session.

Verification actually performed: CPU tracer forwarding/readback; tracer compile
with warnings as errors; wrapper shell syntax; live GPU interception; GDB symbol
text identity; restored-service health; firmware bytes/version checks; local
source inspection; report ASCII and link checks. The unfinished GPU tests do
not establish numerical output correctness or a successful fix.

The research recommendation is to retain the useful copy-off workaround, obtain
the missing matched i915 production-placement result, compare kernels holding
userspace/firmware constant, and reduce the copy failure to an event/mapping
reproducer. Do not repeat failed configuration levers without a new hypothesis.
Interactivity measurements should cover TTFT after idle and warm, p95 token
gaps, longest pause and failure rate, with placement, context, speculation and
contending load recorded.

## Archive map and reproducibility limits

- `../FINAL_REPORT.md`: exact full research report with primary citations.
- `final-response-2026-09-29.md`: delivered response, report link localized.
- `../artifacts/`: tracer source/binaries, GDB commands, coordination, source
  excerpts and research-lane notes. Preserve third-party notices in source files.
- `../evidence/`: available September 29 campaign, including scripts, raw
  outputs, coredumps, trace fragments and saved service/configuration evidence.
- `../evidence/baseline-2026-09-27/`: complete available baseline campaign,
  including the September 28 comparison rounds referenced in that campaign.
- `source-manifest-2026-09-30.json`: source-to-archive mapping and SHA-256 checks.

Binary artifacts are historical outputs; they are not promised portable to a
different runtime. Scripts can stop services or start GPU work if executed;
archiving did not execute them. Absolute paths and source-relative links in
verbatim artifacts are intentionally preserved. The archive index supplies
local navigation. Model weights, entire external source repositories, OS images
and complete chat-platform transcripts are not bundled. Their absence is not
hidden by calling this a fully self-contained reproducer.
