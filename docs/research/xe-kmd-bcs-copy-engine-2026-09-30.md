# Arc A770 on the xe KMD: why blitter copies hang, and what to do - 2026-09-30

## Decision

- **Root cause established (NEO, not kernel, not llama.cpp):** intel-compute-runtime
  26.35.39758 submits the blitter's work through the KMD with per-flush residency, and its
  unused-allocation eviction unbinds the copy command buffer while that job is still
  pending. The blitter then executes zeros (scratch page) up to the next mapped allocation,
  halts on an invalid instruction, the host stalls, and the next LR-mode suspend turns the
  halt into `Engine reset: engine_class=bcs` after the GuC's 640 ms preempt timeout.
- **Fork default (this PR):** ggml-sycl sets `UR_L0_USE_COPY_ENGINE=0` at startup when an
  Intel GPU is bound to `xe` and the user has not set the variable. Copies run on the
  compute queue, which uses direct submission and is unaffected. Real-text decode on the
  production placement was within 1 % of the blitter path (32.78 vs 32.38 t/s, 09-29).
- **Kept the blitter:** `NEOReadDebugKeys=1 DirectSubmissionOverrideBlitterSupport=1`
  passed one full run (X5) but decodes at 8.7 t/s instead of 14.5 on the random-token
  bench. Not a production candidate; useful as a second confirmation of the mechanism.
- **Upstream:** report drafted below for intel/compute-runtime. Not filed from this session.
- Kernel 7.3-rc5 changed nothing: the failure reproduced on the first attempt (X1).

## Host and stack

`vinbonesjr`, Ryzen 9 7900X3D, 64 GB, Arc A770 16 GB at `0000:03:00.0`, headless. Kernel
`7.3.0-rc5-273-linux73-tkg-bore-rc5-xe` (no local xe patch; rc1 for the 09-29 data),
intel-compute-runtime 26.35.39758.10, level-zero-loader 1.32, oneAPI 2026.1.4 (UR Level Zero
adapter v1, two immediate command lists, `inOrder: 0`), GuC 70.53.0, no HuC.
llama.cpp fork b12305 `4e7400c3a` (installed package binaries). Model
Ornith-1.5-35B-A3B Q4_K_M with `--fit`, so roughly half of the routed experts stay
host-resident and `ggml_backend_sched` uploads the used experts every token
(`ggml/src/ggml-backend.cpp`, `copy_experts`).

Earlier evidence (09-29, four coredumps, fix ladder, UR trace, tracer runs) is in
`docs/xe-fix-docs/` and `~/research-llama.cpp/sycl-oneapi-benchmarks-2026-09-29/`. New runs
from this session are under `/mnt/nvme1/oneapi-ab/N-xe-x*/` and use `scratchpad/xn.sh`:
unit stopped, `perf record -a -k CLOCK_MONOTONIC -e xe:...` (scheduling
enable/disable/done, userptr invalidate, rebind worker, evict, reset, timedout, CAT error),
NEO `NEOReadDebugKeys=1 LogAllocationStdout=1 LogAllocationType=1 LogAllocationMemoryPool=1
PrintBOBindingResult=1 PrintBOCreateDestroyResult=1` with a wall-clock stamp on every line,
kernel journal, devcoredump capture, `timeout 600`.

## Runs

| run | env (copy engine ON unless noted) | placement | outcome |
|---|---|---|---|
| X1 `N-xe-x1-trace` | default | auto | **stall at t+11 s, bcs reset at t+73 s**, abort; coredump captured |
| X5 `N-xe-x5-bcsds1` | `DirectSubmissionOverrideBlitterSupport=1` | auto | clean: pp512 182.8, tg64 8.67, pp@8k 150.3, tg@8k 8.42 (instrumented) |
| X4 `N-xe-x4-pinned-ceon` | `--moe-cache off --load-mode none` | off, pinned | clean (instrumented): 362.8 / 48.7 / 300.8 / 41.9 |
| X4b `N-xe-x4b-pinned-ceoff` | copy engine OFF, `--moe-cache off --load-mode none`, clean | off, pinned | clean: **366.0 +- 7.6 / 49.4 +- 0.4 / 304.2 +- 2.0 / 42.3 +- 0.1** |
| X4c `N-xe-x4c-pinned-ceon` | `--moe-cache off --load-mode none`, clean | off, pinned | clean: 363.8 +- 6.7 / 47.9 +- 1.3 / 302.4 +- 2.5 / 41.6 +- 0.2 |
| X4d `N-xe-x4d-mmap-ceoff` | copy engine OFF, `--moe-cache off`, clean (control, same boot) | off, mmap | clean: 325.1 +- 18.4 / 49.2 +- 0.6 / 274.4 +- 3.1 / 41.9 +- 0.3 |

Instrumented t/s are not benchmark numbers (NEO logs several MB/s of text).

## What X1 shows, in order

Times are `CLOCK_MONOTONIC` seconds from `perf`; NEO log lines carry wall clock, anchored
in `clock-anchor.txt` (epoch = mono + 1790730047.6946).

1. 434.398: NEO's blitter exec queue (`guc_id=6`, class 3, flags `0x20` = low latency) is
   enabled. From here the NEO log shows roughly 100 userptr imports per second: each expert
   upload is `EXTERNAL_HOST_PTR` -> `DRM_XE_VM_BIND_OP_MAP_USERPTR` on the LR-mode VM (VM
   flags `0xb` = 64K | LR_MODE | SCRATCH_PAGE), unbound after use. 2810 imports in 75 s;
   `gt0/stats tlb_inval_count` 2933 -> 30959.
2. 434.438: `xe_vma_userptr_invalidate` from llama-bench's own thread (the loader unmapping
   a mmap fragment it had just uploaded from). 11 us later `scheduling_disable` for both
   the compute queue (`guc_id=2`) and the blitter (`guc_id=6`), `guc_state=0x29`
   (REGISTERED|PENDING_DISABLE|SUSPENDED), both done within 60 us, rebind worker,
   `scheduling_enable` 1.3 ms later. A suspend cycle is normal and harmless here.
3. 443.2 (epoch 1790730490.87): **the NEO log stops.** The last lines are one
   residency cycle of the blitter CSR: eleven unbinds, then binds of the tag buffer
   (BO-4), a 32 KB buffer (BO-41), the 1 MB command buffer (BO-39), the timestamp buffer
   (BO-59) and the 64 KB local-memory `COMMAND_BUFFER` BO-121 at .868131, then
   **`unbind BO-121` at .868257, 126 us later**, then one more userptr import and a
   17 MB `INTERNAL_HOST_MEMORY` allocation. Nothing else for 61 s: the host spins waiting
   for the blitter. BO-121 was allocated at 488.428 and had been bound/unbound 503 times in
   2.4 s (269 cycles under 1 ms, minimum 0.126 ms); the older command buffer BO-66 cycled
   at a 12 ms median.
4. 504.575: `scheduling_disable` for both queues again (no userptr event this time; the
   candidates are a BO validate/evict through `xe_bo_trigger_rebind`, whose
   `xe_vma_evict` tracepoint fires only after the wait, or a REMAP bind). The compute queue
   acknowledges in 28 us. The blitter never does.
5. 505.2155: `xe_exec_queue_reset` for `guc_id=6`, exactly 0.640 s after the disable:
   the GuC preempt timeout (`Preempt timeout: 640000 us` in the dump). Kernel prints
   `Engine reset ... state=0x29` and, from the reset cleanup, `Timedout job: seqno=4052`.
   `xe_sched_job_timedout` carries `batch_addr=0xd556a8e46480`.
6. The coredump (`xe-devcoredump-x1.txt.gz`): `batch_addr[0]=0xd556a8e46480`, which is
   inside BO-121 (`0xd556a8e40000-0xd556a8e4ffff`); `RING_ESR=0x1` (instruction error);
   `ACTHD=0xd556a9db0004`, `RING_BBADDR=0xd556a9db0005`, `IPEHR=0xfffff000`. In this
   process `0xd556a9db0000` is `CONSTANT_SURFACE` handle 67 (kernel constant data, 64 KB
   local memory, allocated at load and never unbound). It is the first mapped allocation
   above the unmapped batch: with scratch pages the unmapped 16 MB read as zeros
   (`MI_NOOP`), the parser walked up to the constant surface and halted on its second
   dword. `RING_DMA_FADD = ACTHD + 0x200` is the prefetch.
7. After the reset every `VM_BIND` returns `EPERM` (banned VM); NEO maps that to
   `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`, which is what `set_tensor_async` reported all
   along. It was never memory.

The four 09-29 coredumps carry the same registers (`ESR 1`, second-level batch, ACTHD at
the first mapped allocation above a NEO batch region: three at `0xd556a9db0004` with
`0xfffff000`, one at `0xd556a9be0018` on a `COMPUTE_WALKER` header, i.e. a compute command
buffer occupied that slot in that process). Four of five 09-29 resets were logged with
`state=0x29` for the same reason as X1: the reset is the suspend that found a dead blitter.

## The NEO code path

`shared/source/os_interface/linux/drm_command_stream.inl`, `flushInternal`: with
`vmBindAvailable` (xe) the command buffer is pushed into `allocationsForResidency` and
`mergeWithResidencyContainer` binds whatever is not bound (`makeResidentWithinOsContext`,
`evictable = true`, immediate bind); then `exec` submits through `DRM_IOCTL_XE_EXEC`.
`drm_memory_operations_handler.cpp`, `evictUnusedAllocationsImpl`: for every allocation of
the device, unbind it unless some engine has it `isUsedByOsContext(ctx)` with
`getTaskCount(ctx) > *tagAddress` (or it is always-resident or locked). It is invoked on a
bind failure (`Drm::bindBufferObject`, retried after eviction), on an exec failure
(`BufferObject::exec`), and in the direct-submission-light retry path. On i915 a premature
unbind is harmless because execbuf holds every object of the job until it completes; on xe
`VM_BIND` is decoupled from `EXEC`, so an unbind removes the mapping under a queued job.
The compute queue is not exposed: DG2's direct submission (`hw_info_dg2.cpp`, CCS only)
keeps its ring and command buffers always-resident. Which engine's tag/task count
comparison lets BO-121 slip through was not established from logs alone; the sub-millisecond
bind/unbind churn of the blitter's residency set (thousands per minute) and the userptr
bind/unbind of every copy (`EXTERNAL_HOST_PTR` 90 261 times in X5's four minutes) provide
the retries.

## X5: direct submission for the blitter

`NEOReadDebugKeys=1 DirectSubmissionOverrideBlitterSupport=1` made NEO allocate blitter ring
buffers and semaphore buffers (`RING_BUFFER`/`SEMAPHORE_BUFFER` in local memory, bound once),
i.e. the ULLS path for the copy engine. The full bench passed (four rows, 8k depth), three
LR-mode suspend cycles occurred and were survived, no kernel message. Decode fell to
8.67 t/s (random-token bench, auto placement, instrumented) against 14.5 with copies on the
compute queue. One pass is consistent with the mechanism (no per-flush residency, no
premature unbind), not a proof of safety.

## X4: host-resident experts in pinned memory (`--load-mode none`)

With mmap off the loader's CPU buffer list resolves the fit's CPU-placed tensors to
`SYCL_Host` (`sycl::malloc_host`), because the only thing forcing them back to `CPU_Mapped`
is the mmap guard in `src/llama-model-loader.cpp` (`use_mmap && buft == host buft`). The
copy source is then a USM allocation NEO already knows: no userptr import, no per-copy
`VM_BIND`/unbind, no MMU-notifier exposure.

Instrumented run, copy engine ON, production placement (`--moe-cache off`):

| test | X4 pinned, copy engine on (instrumented) | 09-29 mmap, copy engine off (clean) |
|---|--:|--:|
| pp512 | 362.75 +- 6.99 | 325.0 +- 19.4 |
| tg64 | 48.73 +- 0.15 | 47.55 +- 0.42 |
| pp512 @8k | 300.84 +- 1.69 | 274.0 +- 2.6 |
| tg64 @8k | 41.89 +- 0.12 | 42.07 +- 0.18 |

Clean, no kernel message. The expert-sized userptr imports (4-23 MB in X1) are gone; the
11 601 imports left are 523 KB and 32 KB activation shuttles for the CPU-side layers, about
two per token. The clean repeats (X4b/X4c/X4d, same boot, quiet host) put the pinned gain at
**+12.6 % pp512 and +10.9 % pp512@8k, decode within noise** against the mmap control with
the copy engine off; the copy engine on top of pinned changes nothing measurable (X4c) and
passed twice. Values are t/s, columns pp512 / tg64 / pp512@8k / tg64@8k, random tokens. Caveats: every CPU-placed weight moves to
pinned memory (an owned copy instead of shared page cache), load is a file read, and a
`malloc_host` failure silently falls back to a plain CPU buffer (check the `model buffer
size` lines of `llama-completion`, `llama-bench` prints none). System-memory BOs on an
LR VM can still be swapped by the xe shrinker under memory pressure, which also suspends
the queues; that is harmless while the blitter is not in use.

## Verification of the fork change (build `~/build-xe-kmd`, JIT, branch `xe-kmd-copy-engine`)

- `test-sycl-xe-defaults`: OK (fake sysfs tree, env decision).
- `test-sycl-turbo-correctness` default sweep with `UR_L0_USE_COPY_ENGINE` unset on xe:
  `0 GATE-FAIL, 0 XPASS, 0 xfail, 0 SKIP`; stderr carries
  `Intel GPU 0000:03:00.0 is bound to the xe kernel driver: defaulting UR_L0_USE_COPY_ENGINE=0`.
- `llama-completion` real text (bash man page, 524-token prompt, 128 tokens, temp 0,
  production flags `--fit on --fit-target 1024 --fit-ctx 32768 -ctk q8_0 -ctv q8_0 -fa on
  --moe-cache off -t 12`, `/mnt/nvme1/oneapi-ab/N-xe-rt-*`):

| run | load mode | copy engine | pp t/s | tg t/s |
|---|---|---|--:|--:|
| a | mmap (default) | hook -> off | 71.5 (cold JIT, first launch of the build) | 33.62 |
| b | `--load-mode none` | hook -> off | 232.8 | 34.34 |
| b2 | `--load-mode none`, `--verbose` | hook -> off | 233.9 | 34.28 |
| c | `--load-mode none` | `UR_L0_USE_COPY_ENGINE=1` (override honoured) | 232.4 | 33.99 |

  b2 shows the placement: `SYCL0 model buffer size = 13522.05 MiB`, `SYCL_Host model
  buffer size = 6653.66 MiB`, no `CPU_Mapped` buffer. No kernel message in any run.
  Generated text is coherent in all runs but diverges after roughly 40 tokens between
  runs, including b vs c and the 09-29 pair (`out-off.txt` vs `out-off-copyeng-on.txt`);
  temperature-0 output on this stack is not run-to-run reproducible, which predates this
  change (see the standing decision on batch invariance in `CLAUDE.md`).

## Not claimed

- The exact task-count/tag comparison that misjudges the in-flight command buffer is
  inferred from NEO source and the 126 us bind-to-unbind gap, not observed with NEO
  symbols. A NEO developer can confirm with `ResidencyDebugEnable`.
- What issued the second suspend (504.575) is unknown; it only matters for timing.
- X5 is one pass on one host; the earlier `UR_L0_USE_DRIVER_COUNTER_BASED_EVENTS=0` run
  also passed once and was not a fix.
- No i915 production-placement baseline exists yet; the 09-29 decode regression numbers
  are for the `auto` placement only.
- The 7.3-rc5 kernel was booted for this session; rc1 vs rc5 was not compared for speed.
- Temperature-0 outputs differ run to run (also between the 09-29 pair), so the real-text
  runs verify stability and throughput, not bit-identical output.
- `--load-mode none` gains are from one boot, quiet host, random-token bench plus one
  real-text prompt; the production unit still runs mmap until the user changes it.

## Upstream report draft (intel/compute-runtime)

Title: xe KMD, DG2: blitter command buffer unbound while its job is pending; engine halts,
later `Engine reset: engine_class=bcs` after preempt timeout.

Environment: Arc A770 (`8086:56a0`), Linux 7.3-rc1 and 7.3-rc5 xe (`xe.force_probe=56a0`),
intel-compute-runtime 26.35.39758.10, level-zero-loader 1.32.0, oneAPI 2026.1.4 SYCL/UR
(Level Zero adapter v1, copy engine enabled by default).

Symptom: a SYCL in-order queue that interleaves ~1 MB host-to-device `memcpy` from pageable
host memory with kernels stalls after seconds to minutes. The host spins in
`urEnqueueUSMMemcpy`/`urQueueFinish`. Later the kernel logs `Engine reset:
engine_class=bcs ... state=0x29` and `Timedout job`, all binds return `EPERM`, and the
application gets `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`.

Evidence: device coredump with `RING_ESR=0x1`, `batch_addr` inside a 64 KB
`COMMAND_BUFFER` allocation that `PrintBOBindingResult` shows unbound 126 us after it was
bound for the submission; `ACTHD` at the first mapped allocation above that batch
(`CONSTANT_SURFACE`), `IPEHR=0xfffff000`; xe tracepoints showing the reset is the 640 ms
preempt timeout of a suspend request made a minute after the stall. Not reproducible with
`UR_L0_USE_COPY_ENGINE=0` (compute queue, direct submission) or with
`DirectSubmissionOverrideBlitterSupport=1`.

Reproducer: llama.cpp SYCL build, an MoE model with experts left on the host (`--fit`),
`llama-bench -p 512 -n 64 -d 0,8192` on the copy-engine default; fails in 12 of 15 runs.
Artifacts available: 5 coredumps, `perf script` of xe events, NEO allocation/bind logs.
