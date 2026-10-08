# SYCL second queue for MoE expert prefetch (PR #72) - 2026-09-28

Development log for making `--prefetch-experts-slots` (port of ggml-org/llama.cpp#28414, fork
PR #72) actually overlap host-to-device expert uploads with compute on the SYCL backend. It
records the measurements, the design choices and the obstacles hit along the way.

Evidence labels: **measured** means tool output on this host; **observed once** means a single
probe run on one driver/runtime version and not a campaign; **source** means read from code.

## Starting point

- #28414 overlaps the upload of host-resident MoE expert weights (`--n-cpu-moe`, `--fit`
  auto-offload) with compute. It creates a *second backend instance* on the same device and
  assumes that instance has its own stream. That holds on CUDA, where each context creates its
  own streams.
- It does not hold in this fork (source). `ggml_backend_sycl_context::stream()`
  (`ggml-sycl/common.hpp`) resolves every stream slot to `dpct::get_device(device).default_queue()`,
  a per-device singleton. Vulkan's `ggml_vk_get_device()` likewise caches one `vk_device` with
  one compute queue, and on Intel it does not use the transfer queue for async copies unless
  `GGML_VK_ASYNC_USE_TRANSFER_QUEUE` is set. So a second instance serializes on the same queue.
- `ggml_backend_sycl_event_wait` did a host `sycl::event::wait()`, which blocks the scheduler
  thread instead of making one queue wait for another.
- Commit `4f73cf1f9` (pushed by a parallel session while this work started) had turned the
  feature off on SYCL entirely, by matching on the backend name. That is the gate this work
  replaces.

## Probe: can a second queue overlap copy with compute on the A770?

Standalone program, `docs/research/sycl/sycl-prefetch-second-queue-probe.cpp` (not built by CMake).
Two in-order
queues share one `sycl::context`: a busy kernel runs on q1 and a 512 MiB H2D `memcpy` runs on q2.
Overlap is judged from device profiling intervals (`command_start..command_end`) intersecting,
not from wall clock, so other GPU holders cannot fake it. A single-queue control must show zero
overlap. The production llama-server was stopped for the run and restarted afterwards. `dmesg`
was clean before and after.

Host: Arc A770, i915, compute-runtime 26.35.39758, oneAPI 2026.1 (UR 0.12). Observed once:

| configuration | copy/kernel overlap | notes |
|---|---|---|
| default adapter (= `SYCL_UR_USE_LEVEL_ZERO_V2=0`) | full (23.5 of 23.5 ms) | wall 53 ms vs 77 ms single-queue; `intel_gpu_top`: Blitter and Compute both ~100% busy in the same samples |
| `SYCL_UR_USE_LEVEL_ZERO_V2=1` | none | the cross-queue barrier also blocked the host (next submit took 23.8 ms) |
| `UR_L0_USE_COPY_ENGINE=0` | none | copies fall back to the compute engine |

- q2 built from `q1.get_context()` shares the context, so USM device pointers are valid on both
  (measured). A queue built from the device alone also lands in the default context.
- `q1.ext_oneapi_submit_barrier({q2_marker})` is a device-side wait: it returns in ~0.01 ms,
  the copy is still running at that point, the dependent kernel starts ~0.03 ms after the copy
  ends, and the data is correct (0 mismatches over 512 MiB). Measured.
- H2D bandwidth 22.7-22.9 GB/s in every memory kind below.

Host-side cost of *submitting* the async copy per 512 MiB (measured, default adapter):

| source memory | submit (host blocked) |
|---|---|
| `sycl::malloc_host` (pinned) | 0.01-0.8 ms (0.8 on first touch) |
| `malloc` + `prepare_for_device_copy` | 0.01-0.06 ms |
| pageable `malloc` | 5-14 ms |
| file-backed `mmap` (MAP_SHARED, prefaulted) | 108-134 ms |
| file-backed `mmap` + `prepare_for_device_copy` | 107-116 ms (the import returned in 0.7 ms and changed nothing) |

Host `memcpy` from the same mmap'd file into anonymous memory: 34-35 ms per 512 MiB
(15.5 GB/s, single thread, measured).

## Source cross-check (deep-research pass, same day)

A multi-agent source review (Unified Runtime Level Zero adapter at intel/llvm `17c6a87c`, the
`sycl_ext_oneapi_enqueue_barrier` spec, compute-runtime docs, i915 patches) agreed with the probe
and added these points. They are read from source, not measured, unless noted.

- The v1 adapter lowers `ext_oneapi_submit_barrier(events)` on an in-order queue *without*
  profiling to a wait+signal (`UR_L0_IN_ORDER_BARRIER_BY_SIGNAL`, default on). *With* profiling
  it uses a full `zeCommandListAppendBarrier`. Both are non-blocking. The probe queues had
  profiling on, so they exercised the second lowering. The llama.cpp runs below used production
  queues (no profiling), and there the speedup plus byte-identical output is indirect evidence
  that the first lowering is device-side and ordered too.
- An *empty* wait list is a SYCL no-op, but UR's `urEnqueueEventsWait` with zero events
  host-synchronizes. `ggml_backend_sycl_event_wait` always passes exactly one event.
- Do not express the wait as `cgh.depends_on(e); cgh.ext_oneapi_barrier({})`. DPC++ releases
  before intel/llvm PR #23231 (merged 2026-09-22) drop the `depends_on` events there. The
  queue-level `ext_oneapi_submit_barrier({ev})` used here is not affected.
- In v1, an H2D USM memcpy on an in-order queue goes to the copy engine. If either pointer is
  `malloc_shared`, DG2 forces it to compute, and device-to-device copies go to compute unless
  `UR_L0_USE_COPY_ENGINE_FOR_D2D_COPY=1`. The slots are device USM and the sources are host
  memory, so the upload stays on the copy engine; a staging hop through device memory would not.
- Events from another context are undefined in a barrier wait list, which is why the
  device-side wait is limited to same-device events (dpct puts all queues on the platform
  default context).
- On i915, DG2 exposes a single CCS, so two compute queues still run their kernels one after
  another. The only overlap available is copy (BCS) beside compute (CCS). The xe driver was
  not checked.
- Unanswered by that pass: the prior art in ggml-org #29398 and #21067, and the Intel GPU
  optimization guide's per-architecture notes. No claim about them survived verification.

## Design choices

1. **Private queue only for the prefetch instance, not for every SYCL context.** Changing
   `stream()` globally would move SYCL-Graph record/replay, the oneDNN engine, the memory pools,
   the FA scratch buffers and the MoE cache off the queue they all assume. The prefetch backend
   only does `set_tensor_async`, `event_record`, `event_wait` and `synchronize`. None of those
   touch that state. `ggml_backend_sycl_context::use_private_queue()` points every stream slot
   of that one context at an in-order queue built on the default queue's context and device.
   The queue is deliberately not registered with dpct's `_queues` list, so
   `queues_wait_and_throw()` calls elsewhere (buffer clear, synchronous `set_tensor`) do not
   wait on in-flight prefetch uploads.
2. **A generic hook instead of a SYCL name check in the scheduler.** The new proc address is
   `"ggml_backend_init_private_stream"` (typedef `ggml_backend_init_private_stream_t` in
   `ggml-backend.h`). The scheduler requires it; a backend that does not export it gets a
   one-line warning and prefetch stays off. This keeps `ggml-backend.cpp` free of
   backend-specific knowledge. It also closes the Vulkan hole: on this fork Vulkan ran the
   feature with no overlap, which is pure overhead (a full-tensor upload replacing the
   routed-experts-only copy). The cost is that Vulkan loses a code path that was verified
   byte-identical in `4f73cf1f9` but gave no speedup there.
3. **The SYCL hook declines configurations measured not to overlap.** These are the Level Zero
   v2 adapter (detected from the platform name, `"... over Level-Zero V2"`, rather than from
   the env var, so a future runtime that makes v2 the default is caught too), copies routed to
   the compute engine (`UR_L0_USE_COPY_ENGINE=0` or `UR_L0_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE=0`,
   plus their `SYCL_PI_` names; matched as the exact string "0" because these variables also
   accept `lower:upper` engine ranges), and non-Level Zero backends (unmeasured, so not
   assumed). The in-order-queue variable comes from the source cross-check and was not probed.
4. **`event_wait` becomes a device-side barrier for same-device events.** It now calls
   `stream->ext_oneapi_submit_barrier({ev})`, the SYCL equivalent of `cudaStreamWaitEvent`.
   Cross-device events keep the host wait, so the multi-GPU pipeline-parallel behavior is
   unchanged. In practice only the prefetch path waits on a same-device SYCL event.
5. **Pinned bounce only for mmap-backed sources.** llama.cpp's `--n-cpu-moe` override forces
   `ggml_backend_cpu_buffer_type()`, so with default mmap the experts live in a `CPU_Mapped`
   buffer (file pages). For those, the scheduler memcpys into a per-slot pinned buffer from
   `ggml_backend_dev_host_buffer_type()` and uploads from there: about 35 ms instead of about
   120 ms of host time per 512 MiB. Anonymous memory (`--no-mmap`) is left to the driver, which
   was faster than a memcpy (5-14 ms). Registering host memory (`prepare_for_device_copy`) was
   rejected for this change. It helps only anonymous memory, it pins the whole CPU weight buffer
   (on this host swap was already full), and it needs a registration/refcount lifetime across
   schedulers that share one model.
6. **Slots come from the scheduler's buffer type for that backend** (`sched->bufts[backend_id]`),
   not the backend default, because a slot stands in for the tensor copy the scheduler
   allocated. If that buffer type is not the private-stream backend's default (SYCL
   `set_tensor_async` asserts on it), prefetch turns off with a warning instead of asserting.
7. **One teardown path.** `ggml_backend_sched_prefetch_free()` waits for each used slot's
   last consumer (its `prefetch_free` event) and for the private stream, then frees every
   array entry up to the compile-time maximum rather than the current slot count, and resets
   the cursor. The setter (disable or count
   change), the disable path and `ggml_backend_sched_free()` all use it. The OOM fallback frees
   the dropped slots immediately.

## Verification of the implementation

Build `~/build-pr-28414` (SYCL, JIT, Release). Model Qwen3-Coder-30B-A3B-Instruct UD-Q3_K_XL,
`--n-cpu-moe 20 -ngl 99 -c 4096 -fa on`, a 2500-token prompt, greedy decoding
(`--temp 0 --top-k 1 --seed 42`), 32 generated tokens. The experts were confirmed host-resident:
the log shows `CPU_Mapped model buffer size = 5426.94 MiB`. The production server was stopped
for the runs and restarted afterwards. `dmesg` was unchanged (1261 lines before and after).

- `test-sycl-turbo-correctness`: 0 GATE-FAIL, 0 XPASS, 0 xfail, 0 SKIP (measured, final binary).
- Prefetch fires: 300 stagings per run (debug log). With the default mmap load all 300 went
  through the pinned bounce; with `--no-mmap` 0 did (direct path).
- Output byte-identical to `--prefetch-experts-slots 0` for slots 2, 3 and 4, for slots 3 with
  `--no-mmap`, and for slots 3 under `SYCL_UR_USE_LEVEL_ZERO_V2=1`. In that last case the hook
  declined with `the Level Zero v2 adapter serializes queues` and prefetch stayed off.
- Prefill, 3 alternating reps with a warm page cache (observed, no confidence interval):
  slots 0 gave 133.2 / 132.3 / 132.2 t/s, slots 3 gave 154.7 / 154.2 / 151.9 t/s, about +16%.
  Single runs: `--no-mmap` with slots 3 gave 157.5 t/s, and the v2 adapter with prefetch off
  gave 138.5 t/s. The very first slots-0 run after each server stop took about 27 s (92 t/s)
  and is excluded; it repeated in both scripts, which fits cold page-cache faults on the 5.4 GiB
  of mapped experts.
- `intel_gpu_top` at 20 ms samples: compute busy rose from 67% to 76% of active samples and
  blitter from 26% to 41%. That is consistent with overlap but too coarse to prove it; the
  probe above is the direct evidence.

## Obstacles

- **Concurrent session.** The branch head moved from `21b2ea228` to `4f73cf1f9` while this work
  was being planned (a parallel session pushed round-2 fixes and the SYCL off-switch). Work was
  rebased onto the remote head before editing. Five review threads had also been resolved by
  that session.
- **The earlier premise was wrong.** The hold comment on PR #72 said SYCL has no second queue.
  The hardware (i915 sysfs lists `bcs0`, `ccs0`, `rcs0`) and the API both have one; it was only
  unwired. The earlier survey (`fork-cannibalization-survey-2026-09-27.md`, F.3 and C 0024)
  had flagged "SYCL twin needs a second queue" as unverified. This probe answers it.
- **The L0 v2 adapter kills overlap.** Both adapters ship in oneAPI 2026.1 and v1 is the default
  on DG2 today. If a runtime update flips the default, the hook declines and the feature is off
  until someone re-probes.
- **mmap is the real bottleneck, not the queue.** The driver stages copies from file-backed
  pages on the calling thread at about 4.5 GB/s. `prepare_for_device_copy` on the mapping is
  accepted but does nothing. The old path paid this same cost too. The pinned bounce cuts it
  but does not remove it: the memcpy is still on the scheduler thread.
- **Segfault at exit, found by the first verification run.** The unified teardown originally
  synchronized every `sched->backends[b]` before freeing the slots. `llama_context` declares
  `sched` before `backends`, so the backends are destroyed first. `ggml_backend_sched_free()`
  was therefore synchronizing freed backends, and every prefetching run exited with SIGSEGV
  (139) after printing correct output. No core dump was captured. The source order confirmed
  the cause. The fix waits on each used slot's `prefetch_free` event instead: the event is
  owned by the scheduler and records the slot's last consumer, so no backend pointer is needed.
  After the fix every run exits 0.
- **clangd noise.** The editor diagnostics for SYCL files and for the new typedef are from an
  include path without SYCL and with an installed `ggml-backend.h`. The real build is the
  arbiter.

## Still open

- The batched-decode heuristic (codex P1 on PR #72) is unchanged. A ubatch routing
  `>= 2*n_expert` ids takes the prefetch path whether it is prefill or batched decode. At that
  ratio about 86% of experts are routed anyway (1 - e^-2), so the full-tensor copy moves at most
  about 16% more bytes than the routed-only copy, and it overlaps. The wording in the CLI help,
  `llama.h` and `ggml-backend.h` no longer claims decode is unaffected.
- Graph mode: `GGML_SYCL_ENABLE_GRAPH=1` (off by default) keys replay on source data pointers
  (`ggml_sycl_graph_update_required`), so a slot swap forces a re-record rather than a stale
  replay. That is correct, but it gives no replay benefit on prefetched splits.
- `--moe-cache` together with prefetch remains untested (production runs `--moe-cache off`).
- Not exercised at runtime: the setter's reconfiguration paths (disable after init, slot-count
  change), the OOM fallback that drops slots, the buffer-type mismatch decline, multi-GPU, and
  the Vulkan decline. They are verified from source only.
- The pinned bounce memcpy still runs on the scheduler thread (about 7 ms per 100 MB tensor at
  15.5 GB/s). Moving it to a worker thread would hide it too; not attempted.
- Vulkan could export the hook later if its transfer queue proves independent. That is
  unmeasured.

## How to re-probe

Build the probe with `icpx -fsycl -O2 sycl-prefetch-second-queue-probe.cpp -o probe` (oneAPI env
as in `CLAUDE.md`) and run
`ONEAPI_DEVICE_SELECTOR=level_zero:0 ./probe 512 host_usm,pageable,imported,mmap <scratch file>`
with the GPU otherwise idle, optionally under `sudo intel_gpu_top -J`. The verdict lines are
`overlap X ms` for the two-queue rows against `0.00` for the control row, plus the `xq-wait` rows.
