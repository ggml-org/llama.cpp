# Arc A770 on the xe KMD: driver switch, dense and MoE benchmarks, failures (2026-09-29)

Host `vinbonesjr`: Ryzen 9 7900X3D, 64 GB, Arc A770 16 GB at `0000:03:00.0`, headless
(both monitors on the Raphael iGPU). Kernel `7.3.0-rc1-273-tkg-bore`,
intel-compute-runtime 26.35.39758, level-zero-loader 1.32, oneAPI DLE 2026.1.4,
`llama.cpp-sycl-f16-git b12305` (`4e7400c3a`, oneDNN off, SYCL graphs on). i915
reference numbers are the 2026-09-28 variant-N rounds from
`../sycl-oneapi-benchmarks-2026-09-27/`. Operational write-up of the switch itself:
`~/._claude/arc-a770-xe-driver-switch-2026-09-29.md`.

**Bottom line**

- **Dense model, all on device:** xe prefill +6 % (pp512) and +17 % (pp2048), decode
  equal. Clean, reproducible.
- **Ornith-1.5-35B-A3B Q4_K_M with `--fit`, `llama-bench` default `--moe-cache auto`
  (about half the routed experts host-resident and streamed):** xe prefill +55-62 %,
  **decode -20 to -29 %** (+14 to +22 ms per token), consistent over five runs. This
  placement is not the production one; see the `--moe-cache` section.
- **Stability on xe: 4 failures in 7 long-context exercises**, i915 clean in every
  round. Two silent hangs (host spinning in the L0 adapter's memcpy enqueue), two
  blitter `Engine reset ... Timedout job` events with device coredumps, one of them the
  **production `llama-server` in default config on its second real request**.
- `UR_L0_USE_COPY_ENGINE=0` was the only lever that finished and moved decode
  (+5-7 %). One pass; not evidence it removes the hang.
- **Fix ladder (16:37-17:32):** `UR_L0_USE_COPY_ENGINE=0` (all copies on the compute
  queue) went **0 for 10** (4 bench rounds + 6 real 7k-14k-token production requests)
  where the default path went 11 failures in 14. Five submission-model variants did not
  rescue the default path; a UR trace shows the copy-engine memcpy chain simply stops
  completing. **Production stays on xe with a unit drop-in setting that variable.**
  i915 revert stays documented as the fallback.
- **Benchmark caveat found late (17:49):** every Ornith `llama-bench` row, including
  the i915 reference, used `--moe-cache auto` (bench default); production pins `off`.
  On the production placement xe decodes real text at ~32.5 t/s; `llama-bench` says
  47.6 on random tokens. The -20 to -29 % decode figure is for `auto` only; the
  production placement has no i915 number yet.
- Side findings: xe on DG2 has **no HuC**, so VAAPI encode is CQP-only (VBR/CBR fail);
  `LIBVA_DRIVER_NAME=radeonsi` in `/etc/environment` had been breaking Intel VAAPI on
  both drivers (removed); xe's job timeout is capped at 10 s by the tkg kernel config;
  journald silently recorded no system entries on the first xe boot until restarted;
  multi-CCS (`ZEX_NUMBER_OF_CCS`) cannot be configured under xe with this NEO.

## Driver configuration under test

- ZBM cmdline (`zroot/ROOT`): `i915.force_probe=!56a0 xe.force_probe=56a0`;
  `/etc/mkinitcpio.conf` `MODULES=(xe amdgpu zfs)`; `/etc/modprobe.d/intel-xe.conf`
  comments only (no blacklists). Snapshot in `raw/config/`.
- xe at bind: `xe-a770-tune.service` (udev `80-xe-a770-tune.rules`, drop-in on
  `llama-gpu@.service`) sets `ccs`/`rcs` `preempt_timeout_us=7500000` (i915 parity; xe
  default 640 000) and `ccs` `job_timeout_ms=10000`. The intended 60 s is refused:
  `CONFIG_DRM_XE_JOB_TIMEOUT_MAX=10000` caps the sysfs `_max` store. Note (18:55): the
  job timeout never arms for NEO's queues (LR-mode VM); the blitter "Timedout job" lines
  are GuC engine resets reported through the TDR error path, not timer expiries.
- Live under xe: GuC `i915/dg2_guc_70.bin` 70.53.0 RUNNING, HuC N/A, GSC N/A;
  `wedged_mode=1`, `guc_log_level=1`, `probe_display=Y`; GT freq floor 750 MHz (RPe;
  i915 floors at 300) - observation only; the research pass found no support for frequency management explaining the performance deltas; hwmon `power2_max` 230 W, pkg 50  degC, VRAM 62  degC at idle.
- Engines: sysfs classes `bcs ccs rcs vcs vecs`; debugfs `hw_engines` `rcs0 bcs0
  ccs0 ccs1 ccs2 ccs3`; `tile0/gt0/ccs_mode = 1`.

## Method

Same `raw/bench.sh` as the 09-27 set (Ornith unit stopped, offline `llama-bench`,
restarted afterwards). `raw/bench-xe.sh` adds a `GRAPH` override and a
`BENCH_TIMEOUT` watchdog for the lever runs. Runners: `raw/run-xe.sh` (two rounds,
quiet-host wait), `raw/run-xe-levers.sh`. Logs: `raw/run-xe.log`,
`raw/run-xe-levers.log`.

```
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_ENABLE_GRAPH=1 \
llama-bench -m /mnt/ssd2/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf \
  -fitt 1024 -fitc 32768 -ctk q8_0 -ctv q8_0 -fa 1 \
  -p 512 -n 64 -d 0,8192 -r 5 -t 12 -o md
```

Host quiet for every run (load 0.7-2.0 at start), swap 0/15 GiB, 30 GB RAM free,
nothing else on the GPU. Values are mean +/- sd over 5 repetitions in t/s. `@8k` = KV
depth 8192.

## Results

### 1. Dense 8B, i915 vs xe (`raw/dense-8b/`)

| test | i915 | xe (clean) | delta |
|---|--:|--:|--:|
| pp512 | 1408.08 +/- 2.41 | 1495.56 +/- 0.49 | +6.2 % |
| pp2048 | 848.93 +/- 0.57 | 990.40 +/- 1.28 | +16.7 % |
| tg128 | 60.15 +/- 0.07 | 60.58 +/- 0.01 | +0.7 % |

A contended xe run (Ornith unit resident, idle) matched the clean one within noise.

### 2. Ornith, i915 vs xe, five xe runs (`raw/N-xe-*`)

| Run | env | pp512 | tg64 | pp512 @8k | tg64 @8k | outcome |
|---|---|---|---|---|---|---|
| N r2 (i915, 09-28) | - | 125.30 +/- 5.67 | 18.13 +/- 0.15 | 107.13 +/- 1.93 | 16.81 +/- 0.70 | clean |
| N-xe-r1 | - | 197.85 +/- 13.41 | 13.85 +/- 0.14 | 167.36 +/- 4.65 | 12.50 +/- 0.13 | clean, 3 min |
| N-xe-r2 | - | 202.86 +/- 9.81 | 13.38 +/- 0.17 | 170.24 +/- 3.03 | **hang** | stuck in tg64@8k 22 min, killed |
| N-xe-copyeng0 | `UR_L0_USE_COPY_ENGINE=0` | 202.38 +/- 9.98 | 14.50 +/- 0.07 | 167.68 +/- 2.70 | 13.41 +/- 0.20 | clean |
| N-xe-pinned0 | `GGML_SYCL_ENABLE_HOST_PINNED_MEM=0` | 200.42 +/- 9.03 | 12.87 +/- 0.08 | 168.44 +/- 2.07 | **abort** | bcs engine reset, `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`, exit 134 |
| N-xe-graph0 | `GGML_SYCL_ENABLE_GRAPH=0` | 194.33 +/- 9.80 | 13.67 +/- 0.19 | **hang** | - | stuck in pp512@8k, watchdog 7 min, exit 124 |

Per token: i915 55 ms, xe 69-78 ms at depth 0; 60 ms vs 75-80 ms at 8k.

### 3. What the decode path is (corrected 17:45)

`--moe-cache` defaults to **off** in this build (`common/arg.cpp`) and the production
unit pins it off. The fit log line "MoE cache fit selected main-device dense placement
... (up to 51.3 % coverage)" is the planner's name for the layout it chose; the loader
then warns `tensor overrides to CPU are used with mmap enabled`, i.e. roughly half of
the routed expert tensors are host-resident. A `llama-completion` run with
`GGML_CUDA_MOE_CACHE_STATS=1` printed neither a cache configuration nor hit/fill
counters (`raw/N-xe-cachestats/`), so the fork's streaming cache in
`ggml-sycl/moe-cache.cpp` is not active.

What *is* active, with the cache off, is the scheduler's own expert streaming for
host-resident MoE weights consumed by a GPU `MUL_MAT_ID`
(`ggml/src/ggml-backend.cpp`, `copy_experts` at line 2430): per ubatch it reads the
routing ids back, marks the used experts, groups consecutive ids and issues one
`ggml_backend_tensor_set_async` per group of expert slices, padded by up to 512 bytes.
The 17:35 follow-up (`copy-path-trace.md`) caught one such copy live under GDB:
`SYCL0#blk.1.ffn_down_exps.weight#0`, `GGML_TYPE_Q6_K`, `ne={512,2048,256,1}`,
offset 202 137 600 = 235 x 860 160 (expert 235), size 860 672 = one expert + 512
padding. The earlier claim in this README that the per-token transfers were only
activation shuttles to CPU-computed experts was **wrong**; they are expert-weight
uploads, hundreds per token, 0.4-2 MB each. That is the decode cost that xe's copy
path makes 20-29 % more expensive, and the reason the copy-engine lever moves decode
at all. Placement is byte-identical on both drivers; bytes moved per token are
therefore identical too, given identical routing (llama-bench feeds seeded
pseudo-random tokens on both drivers), so the delta is per-copy cost, not volume.
`--prefetch-experts-slots` was off in every run.

`llama-bench`'s random tokens are the worst case for expert locality on either
driver. The relative comparison holds; the absolute 13-14 t/s does not represent
production. Real text on xe, `llama-completion`, 38-token prompt, 63 tokens: 24.7 t/s
(40.4 ms/token), two runs within 0.1 t/s.

### 4. Failures

**N-xe-r2, silent hang** (`raw/N-xe-r2/hang-diagnostics.txt`). Three rows done by
12:09, fourth test never finished, killed 12:31. Main thread 99.9 % CPU in user space,
`eu-stack` twice identical:

```
__sched_yield
(libur_adapter_level_zero x5)
enqueueMemCopyHelper(ur_command_t, ur_queue_handle_t_*, void*, unsigned char, unsigned long, void const*, ...)
ur::level_zero::urEnqueueUSMMemcpy
sycl::_V1::detail::MemoryManager::copy_usm
sycl::_V1::queue::memcpy
(libggml-sycl)
ggml_backend_sched_graph_compute_async
llama_context::graph_compute -> process_ubatch -> decode
```

12 worker threads in `futex_wait`. `fdinfo` over 5 s: `drm-cycles-ccs` +96 088 348 of
`drm-total-cycles` +96 095 483 (ccs context scheduled ~100 %), `drm-cycles-bcs` +0.
No xe kernel messages during the run; SIGTERM ended the process in 2 s with no engine
reset; GPU idle afterwards. Established: submission stuck inside a memcpy enqueue.
Consistent with, not proof of, an unresolved device-side dependency (an in-ring
semaphore wait counts as context runtime). NEO's compute queues sit in LR-mode VMs, so
xe's job timeout never fires for them.

**N-xe-pinned0, blitter reset** (`raw/N-xe-pinned0/`). 12:38:41:

```
xe 0000:03:00.0: [drm] Tile0: GT0: Engine reset: engine_class=bcs, logical_mask: 0x1, guc_id=6, state=0x3
xe 0000:03:00.0: [drm] Tile0: GT0: Timedout job: seqno=106926, lrc_seqno=106926, guc_id=6, flags=0x20 in llama-bench [260404]
```

then `level_zero backend failed with error: 39 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)`
in `ggml_backend_sycl_set_tensor_async` (`ggml-sycl.cpp:6230`), `ggml_abort`, exit
134. Device coredump `xe-devcoredump-123841.txt.gz` (532 KB, GuC log included; the
kernel deleted the original 77 s later): bcs0 context runtime 0 ms, `RING_ESR=0x1`,
`IPEHR=0xfffff000`, `ACTHD`/`RING_BBADDR` inside the batch buffer. A command-stream
failure: the copy engine never progressed. The dump does not say who produced the bad
state (command generation, buffer lifetime, mapping/residency, visibility), does not
tie this to the silent hangs, and the `OUT_OF_DEVICE_MEMORY` code alone does not
establish exhausted VRAM.

**N-xe-graph0, silent hang** in pp512@8k after two rows, watchdog at 7 min, no kernel
messages.

**Production server, default config, twice** (`raw/N-xe-server/`). The unit reset
the blitter at **13:15:16** (`Timedout job: seqno=11684 ... in llama-server [264942]`,
`state=0x29`, unit exited `status=6/ABRT` at 13:15:18) on traffic nobody in this
session sent, and again at 13:20:39 on request 2 below. The device coredump in
`raw/N-xe-server/xe-devcoredump-132039.txt.gz` is the **13:15** one (xe keeps a single
pending dump until it is read or expires; the file names the process
`llama-server [264942]`). Through the running unit (tools, `ngram-mod` speculation,
128k ctx): request 1, 542-token prompt, 256 tokens: pp 163.5 t/s, tg 38.0 t/s
(speculation on a repetitive prompt; not a comparison number). Request 2, same body,
13:20:39:

```
xe 0000:03:00.0: [drm] Tile0: GT0: Engine reset: engine_class=bcs, logical_mask: 0x1, guc_id=6, state=0x3
xe 0000:03:00.0: [drm] Tile0: GT0: Timedout job: seqno=7811, lrc_seqno=7811, guc_id=6, flags=0x20 in llama-server [318738]
```

then the same `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` at `ggml-sycl.cpp:6230` from
`server_context_impl::update_slots()`, SIGABRT, `Failed with result 'core-dump'`,
systemd restart. Journal and kernel excerpts alongside. No lever was set.

**17:35 follow-up, other session** (`copy-path-trace.md`, `raw/N-xe-copytrace*`,
`raw/copy-tracer/`). An `LD_PRELOAD` tracer on `ggml_backend_tensor_set_async`, default
copy engine, `bench-xe.sh` flags. Run 1 completed pp512 (197.8) and tg64 (13.60), then
stalled before pp512@8k with the host in `urQueueFinish -> ur_queue_handle_t_::synchronize
-> sched_yield`; `fdinfo` over 3 s: **bcs +57 603 176 of total +57 603 652, ccs +0**, the
mirror image of round 2 (ccs 100 %, bcs 0). Run 2 with `UR_L0_DISABLE_EVENTS_CACHING=1`
stalled before the first row, host in `enqueueMemCopyHelper`, again bcs scheduled and
ccs idle (+38 405 183 / +38 405 660 over 2 s). Both terminated by SIGTERM (exit 143).
Verified here from their samples. Their note says no xe reset appeared during these
runs; the kernel log disagrees: `17:39:37 Engine reset: engine_class=bcs ... Timedout job:
seqno=88884 ... in llama-bench [428746]`, `state=0x29`, fifth bcs reset this boot, fourth
device coredump saved as `raw/N-xe-copytrace/xe-devcoredump-173937.txt.gz`.

Reading of the two stall shapes together: whichever engine's context holds the pending
in-ring wait shows as "scheduled" in `fdinfo`; the counters record waiting, not
progress. In round 2 the compute context waited (on a copy event), in the traced runs
the copy context waited (on a compute event or on its own predecessor) while the host
sat in a queue finish or the next enqueue. Both shapes vanish when the copy list does
not exist. Event caching is not the lever: turning it off changes allocation and
lifetime behaviour and still stalls.

**Cause ranking**, best supported first: (1) runtime copy/event ordering or submission
defect in UR/NEO on xe (stuck enqueue; copy-engine lever moves decode and was the only
lever to finish); (2) mapping/residency or command-buffer lifetime/visibility defect
(the two bcs dumps); (3) an application buffer-ordering defect exposed by different
timing, which success on i915 does not exclude. The exact defect is not established.


## Fix ladder (16:37 GMT onward)

The first verdict stopped at three single-pass levers. This rung works the one that
finished.

### Blitter job timeout (kept, but irrelevant - corrected 18:55)

`xe-a770-tune` also sets `bcs` `job_timeout_ms=10000`. At the time this was written the
blitter resets were read as 5 s job timeouts. Wrong: NEO puts every queue on an LR-mode
VM, and xe gives LR queues `MAX_SCHEDULE_TIMEOUT` (no watchdog; `xe_guc_submit.c`: "LR
jobs can only get here if queue has been killed or hit an error"). Every reset we logged
was GuC's own `Engine reset` notification (CS error, `RING_ESR=0x1`), with the TDR run
afterwards as the error path that prints "Timedout job". The knob changes nothing for
this workload; it stays because it is harmless. See `deep-research-xe-i915-2026-09-29.md`.

### `UR_L0_USE_COPY_ENGINE=0` soak, Ornith bench (`raw/N-xe-copyeng0*`)

Same command as above, `BENCH_TIMEOUT=420` watchdog, host quiet (load 0.5-5).

| round | pp512 | tg64 | pp512 @8k | tg64 @8k | outcome |
|---|--:|--:|--:|--:|---|
| r1 (12:32) | 202.38 +/- 9.98 | 14.50 +/- 0.07 | 167.68 +/- 2.70 | 13.41 +/- 0.20 | clean |
| r2 (16:38) | 201.83 +/- 10.74 | 14.52 +/- 0.11 | 167.58 +/- 2.59 | 13.70 +/- 0.11 | clean |
| r3 (16:41) | 200.99 +/- 10.77 | 14.54 +/- 0.08 | 167.78 +/- 3.35 | 13.33 +/- 0.12 | clean |
| r4 (16:45) | 201.84 +/- 10.45 | 14.57 +/- 0.18 | 168.85 +/- 2.39 | 12.30 +/- 0.22 | clean |

4/4 clean with copies on the compute queue, against 3 failures in 5 with the copy
engine on. No xe kernel messages during any of them. Decode 14.5 t/s at depth 0 is
stable to +/-0.05 across rounds; i915 was 18.1.

### Production unit on xe with the lever (`raw/server-soak/`)

Drop-in `llama-gpu@Ornith-1.5-35B-Q4_K_M.service.d/xe-copy-engine.conf` sets
`Environment=UR_L0_USE_COPY_ENGINE=0` (verified in the server's `/proc/<pid>/environ`).
Six real-text requests through the running unit (man-page prose, `cache_prompt=false`,
128 tokens each, temperature 0; the unit runs `ngram-mod` speculation so tg is not a
bench-comparable number):

| req | prompt tokens | pp t/s | tg t/s | http |
|---|--:|--:|--:|---|
| r1 | 6948 | 213.5 | 29.31 | 200 |
| r2 | 6948 | 250.5 | 22.45 | 200 |
| r3 | 10185 | 240.5 | 23.26 | 200 |
| r4 | 10185 | 245.0 | 31.59 | 200 |
| r5 | 14100 | 227.1 | 31.30 | 200 |
| r6 | 6948 | 246.5 | 34.14 | 200 |

6/6, no kernel messages, no restarts. Earlier today the same unit with the default
copy-engine setting reset the blitter twice in six minutes (13:15, 13:20).

### Localization rung (`raw/run-xe-levers2.sh`, copy engine back on default)

Host load was 12-16 during these runs (`mergerfs` + an `rg` sweep of the pool, not
part of the test); t/s values are contaminated, pass/fail is not.

| run | env | outcome |
|---|---|---|
| N-xe-inorder0 | `UR_L0_USE_DRIVER_INORDER_LISTS=0` | **bcs reset** before the first row (`seqno=4070`, `state=0x29`), abort; third coredump in `raw/N-xe-inorder0/` |
| N-xe-cbevents0 | `UR_L0_USE_DRIVER_COUNTER_BASED_EVENTS=0` | clean pass (165.3 / 12.82 / 161.0 / 12.11, load 17) |
| N-xe-immcl0 | `UR_L0_USE_IMMEDIATE_COMMANDLISTS=0` | no reset, but **decode 2.17 t/s** (batched command lists pay per-submit latency every token); watchdog cut tg64@8k. Not a fix. |
| N-xe-nodirectsub | `NEOReadDebugKeys=1 EnableDirectSubmission=0` | two rows (176.4 / 10.96), then watchdog in pp512@8k with no kernel messages: silent stall or extreme slowdown, not separable. Not a fix. |
| N-xe-norelaxed | `NEOReadDebugKeys=1 DirectSubmissionRelaxedOrdering=0` | zero rows in 7 min, no kernel messages: silent stall. Not a fix. |

Localization result: nothing in the submission model rescues the default copy path.
Turning driver in-order lists off fails faster; counter-based events off passes once
(no proof); batched command lists and NEO direct-submission variants either collapse
decode or stall silently. The one variable that flips the outcome is where copies
execute.

### UR Level Zero trace of the default path (`raw/N-xe-trace/`)

`UR_L0_DEBUG=1`, default env, same bench. The adapter created two immediate command
lists, `ordinal 0` (compute) and `ordinal 1` (copy: "main blitter/copy engine is
available, link blitter/copy engines are not"), both `inOrder: 0`, so ordering is done
by UR event chaining. The log (716 001 lines, 62.5 MB, `bench.err.gz`) stops growing
13 s after start, still inside model load, 0 rows produced, process alive until the
10-minute watchdog, no kernel messages. The last 3 000 lines
(`bench.err.tail3000.txt`) are a chain of `zeCommandListAppendMemoryCopy`, each with
`NumEventsInWaitList 1: <previous copy's event>`, host-visible profiling events from
the adapter's cache; the chain simply stops completing. That is the silent-hang
shape: a copy whose wait never resolves never starts, so xe's job timeout never fires.
The reset shape (bcs `RING_ESR=1`, context runtime 0 ms) is the other outcome, a copy
that starts and errors. Both live on the copy-engine list; with
`UR_L0_USE_COPY_ENGINE=0` there is no such list.

Whole-log call counts before the stall: 157 543 `zeCommandListAppendLaunchKernel`,
21 217 `zeCommandListAppendMemoryCopy`, 11 `zeCommandListAppendMemoryFill`.

Final tally on xe today, long-context exercises: **copy engine on: 11 failures in 14**
(bench r2 hang, pinned0 reset, graph0 hang, server 13:15 reset, server 13:20 reset,
inorder0 reset, immcl0 decode collapse, nodirectsub stall, norelaxed stall, trace
stall, copytrace stall + 17:39 reset, copytrace-nocache stall; passes: r1, cbevents0,
server request 1; counting the decode collapse as a failure). **Copy engine off: 0
failures in 10** (4 bench rounds, 6 server requests). Five bcs resets this boot, five
`Timedout job` lines, all `guc_id=6`, four coredumps saved.
Every kernel-visible failure is the same event: `Engine reset: engine_class=bcs ...
Timedout job ... guc_id=6`, bcs0 context runtime 0 ms.

### Upstream report material

Component: Unified Runtime Level Zero adapter (oneAPI 2026.1.4, `libur_adapter_level_zero.so`)
on intel-compute-runtime 26.35.39758 with the xe KMD (kernel 7.3.0-rc1), DG2 / Arc A770.
Symptom: USM memcpy chain on the main copy engine (immediate list ordinal 1,
`inOrder: 0`, event-chained) stops completing; either silent (no job ever starts) or
`bcs` engine reset with `RING_ESR=0x1`, `IPEHR=0xfffff000`. Not seen on i915 with the
same user stack. Workaround: `UR_L0_USE_COPY_ENGINE=0`. Artifacts: four xe device
coredumps (`raw/N-xe-pinned0`, `raw/N-xe-server` (13:15 event), `raw/N-xe-inorder0`,
`raw/N-xe-copytrace` (17:39 event)), the UR trace (`raw/N-xe-trace`), kernel excerpts,
host backtraces in three shapes (memcpy enqueue: `raw/N-xe-r2/hang-diagnostics.txt`,
`raw/N-xe-copytrace-nocache/stack.txt`; queue finish: `raw/N-xe-copytrace/stack.txt`),
engine-counter samples showing either engine parked (`stall-samples.json`), and the
live pending copy (`raw/N-xe-copytrace-nocache/live-copy.txt`: one Q6_K expert slice,
860 672 bytes, from `ggml_backend_sched`'s `copy_experts`). Reproducer: `raw/bench.sh` with
`llama.cpp-sycl-f16-git b12305` and an MoE model that leaves expert tensors on the CPU
(`--fit`), failure in 6-9 of 12 runs within 3-10 minutes.

### Revised verdict

xe + `UR_L0_USE_COPY_ENGINE=0` survived 4 bench rounds and 6 long-context production
requests; the default path failed 9 of 12 exercises. Intermittent failures and ten
clean passes are not a proof, but the odds of ten clean passes at the default failure
rate are well under 1 %. With the lever, xe gives prefill +55-62 % and decode -20 %
on the random-token bench versus i915 (real-text decode through the unit 22-34 t/s
with speculation). **Production stays on xe with the drop-in.** The i915 revert
commands below remain the fallback if the unit ever logs a `bcs` reset again; the
check is `journalctl -k -g "Timedout job"`.


## `--moe-cache` placement: the benchmarks were not measuring the production config (17:49-18:05)

`llama-bench` in this fork defaults to `--moe-cache auto`; the production unit pins
`--moe-cache off`. Every Ornith `llama-bench` row above, **including the 09-27/28 i915
reference rounds** (their `bench.err` carries the same `MoE cache fit selected
main-device dense placement` lines), ran the `auto` placement: all 42 layers on the GPU,
about half of the routed expert tensors host-resident, and the scheduler streaming the
used experts every token. That is the placement with the -20 to -29 % decode on xe and
the copy-engine failures. Nobody runs it in production.

Same bench, xe, copy engine off, three placements (`raw/N-xe-copyeng0-moesoft`,
`raw/N-xe-copyeng0-moeoff`; `auto` = r2-r4 above):

| `--moe-cache` | placement (fit log) | pp512 | tg64 | pp512 @8k | tg64 @8k |
|---|---|--:|--:|--:|--:|
| auto | dense, ~50 % experts host-resident, all streamed | 201.8 | 14.52 | 167.6 | 13.70 |
| soft | 18/41 layers' experts GPU-resident, 9.6 GB evicted, ~0.5 GB cache | 272.9 +/- 7.5 | 22.72 +/- 0.06 | 226.3 +/- 3.6 | 20.08 +/- 0.19 |
| off | standard `--fit` (no fit-log lines) | 325.0 +/- 19.4 | **47.55 +/- 0.42** | 274.0 +/- 2.6 | **42.07 +/- 0.18** |

Then real text (`llama-completion`, 562-token man-page prompt, 128 tokens, temperature
0, same fit/KV/FA flags; `raw/N-xe-realtext-moecache/`):

| `--moe-cache` | copy engine | pp t/s | tg t/s | ms/token |
|---|---|--:|--:|--:|
| off | off (`UR_L0_USE_COPY_ENGINE=0`) | 204.8 | **32.78** | 30.5 |
| off | on (default) | 199.0 | 32.38 | 30.9 |
| soft | off | 187.4 | 18.46 | 54.2 |

Readings:

- **Random tokens reward streaming placements.** `off` scores 47.6 t/s on the bench and
  32.8 on text; `soft` 22.7 on the bench and 18.5 on text; the ordering between `soft`
  and `off` even flips against `auto` depending on input. `llama-bench` decode numbers
  for this model are not production numbers on any driver.
- **Production placement on xe: ~32.5 t/s real-text decode at short context**, coherent
  output. The copy-engine lever is neutral there (+1 %), so the production drop-in costs
  nothing; it still matters because the two production crashes (13:15, 13:20) happened
  on exactly this placement with the copy engine on.
- **The xe-vs-i915 decode regression is established only for the `auto` placement.**
  For the production placement there is no i915 number in either set. The ledger's
  16.9 t/s (i915, through the server with tools context, b12275) is a different
  measurement and not comparable. Settling the driver question for production needs one
  reboot on i915 and two runs: this bench with `--moe-cache off`, and the real-text
  `llama-completion` above.
- `soft` is not a production candidate on this evidence: slower than `off` on text.

## Fallback: revert to i915

Only if the copy-engine-off configuration regresses. Revert (permanent):

```bash
sudo zfs set org.zfsbootmenu:commandline="$(zfs get -H -o value org.zfsbootmenu:commandline zroot/ROOT | sed 's/i915.force_probe=!56a0 xe.force_probe=56a0/i915.force_probe=56a0 xe.force_probe=!56a0/')" zroot/ROOT
sudo sed -i 's/^MODULES=(xe amdgpu zfs)$/MODULES=(i915 amdgpu zfs)/' /etc/mkinitcpio.conf && sudo mkinitcpio -P
```

One-boot test: edit the two `force_probe` params at the ZBM prompt. `xe-a770-tune`
and the `UR_L0_USE_COPY_ENGINE=0` drop-in are harmless under i915.

## Side findings

- **Encode** (`raw/encode/`): xe.ko carries no DG2 HuC (`strings`: only
  `i915/dg2_guc_70.bin`, `dg2_dmc_ver2_08.bin`; debugfs `huc_info: N/A`; under i915 HuC
  7.10.16 RUNNING). iHD 26.2.4 still advertises 18 encode entrypoints incl. AV1; real
  ffmpeg test: H.264/HEVC/AV1 **CQP OK**, H.264/AV1 **VBR and CBR fail** (`-5 EIO`).
  BRC lives on the HuC. `/etc/environment` had `LIBVA_DRIVER_NAME=radeonsi` globally,
  which made `vainfo` on the Intel node fail under both drivers; removed 11:48 (backup
  `/etc/environment.bak-20260929-114756`), libva auto-selects correctly without it.
- **journald**: on the 11:00 xe boot journald wrote only UID-1000 entries
  (`SplitMode=uid`); kernel, PID 1 and root entries were dropped until
  `systemctl restart systemd-journald` at 11:17 (`journalctl --verify` PASS, disk
  fine). `dmesg` had lost the first 27 s of boot to 624 `amdgpu ... Unsupported screen
  format RA24` lines. No kernel record of the first xe probe exists; firmware state
  came from debugfs.
- **Multi-CCS**: 4 CCS in hardware, 1 exposed. Intel's `MULTI_CCS_MODES.md` configures
  it through `/sys/class/drm/cardX/gt/gtY/ccs_mode` (i915 layout); xe's file is
  `.../device/tile0/gt0/ccs_mode`, and `libze_intel_gpu.so.1` (26.35) composes only the
  i915 path (`/sys/class/drm/` + `/gt/gt` + `/ccs_mode`), so `ZEX_NUMBER_OF_CCS` has
  nothing to write under xe. A root write returns `EBUSY` while any DRM client holds the
  device (kwin, Xorg, Xwayland, logind, llama-server all do). `fdinfo` on the server
  showed `drm-cycles-bcs 14.0 M` vs `ccs 8.0 M` after load: UR already routes copies to
  the blitter under xe.
- **Live driver swap** is not possible on this workstation: kwin and friends hold
  `card0` even though the A770 has no display; every driver change is a reboot.
- **PCIe**: endpoint reports `2.5GT/s x1` (DG2 internal switch virtual link); real link
  `16GT/s x16` at `00:01.1 <-> 01:00.0`, ReBAR 16 GB.

## Not claimed

- No UR/L0 debug trace was captured before killing the hung runs; the device-side-wait
  reading is inferred from engine counters, the backtrace and the bcs coredumps, not
  from adapter internals.
- First-rung levers were one pass each; the copy-engine-off lever has 10 passes, the
  rest of the localization rung one each. Host load was 12-20 during the localization
  rung, so its t/s values are contaminated (pass/fail is not).
- The UR trace stalled during model load, not in the decode test that failed earlier;
  same copy-chain shape, different moment. `UR_L0_DEBUG` changes timing.
- The 17:35 follow-up runs used an `LD_PRELOAD` tracer and manual termination; they are
  diagnostics, not benchmarks, and the tracer itself changes timing. Their note's
  "no new xe reset" is contradicted by the kernel log (17:39:37).
- Whether both stall shapes have one cause is inferred from the single variable that
  removes both, not shown from adapter internals.
- i915 reference for Ornith is the 09-28 set, one day earlier, same package; not re-run
  on 09-29. The dense-8B i915 run was on 09-29, before the reboot.
- The dense-8B i915 stderr was lost with `/tmp` at the reboot; values are from the
  session transcript.
- Real-text server numbers are single requests with speculation on; only their
  existence and the crash matter, not the t/s.
- Encode test: synthetic source, 90 frames, VAAPI only; QSV untested.
- journald root cause not found; restart was a workaround.
- Multi-CCS conclusion rests on strings in the NEO library, not on source.

## Files

- `raw/dense-8b/`: 8B A/B table and the xe stderr.
- `raw/N-xe-r1`, `raw/N-xe-r2` (+ `hang-diagnostics.txt`), `raw/N-xe-copyeng0`,
  `raw/N-xe-pinned0` (+ `kernel-log-123841.txt`, `xe-devcoredump-123841.txt.gz`),
  `raw/N-xe-graph0`: `bench.md` / `bench.err` per run.
- `raw/N-xe-cachestats/`: `llama-completion` real-text runs with and without
  `GGML_CUDA_MOE_CACHE_STATS=1`.
- `raw/N-xe-server/`: request body, request-1 response with timings, server journal
  and kernel excerpts for the 13:20 crash, second device coredump.
- `raw/encode/`: ffmpeg stderr per encoder/rc mode and `results.md`.
- `raw/config/`: `xe-a770-tune`, its unit, udev rule, `intel-xe.conf`, the
  `llama-gpu@` drop-in, cmdline and `MODULES=` as booted.
- `deep-research-xe-i915-2026-09-29.md` (verified findings, refutations, cross-check,
  config table) + `raw/deep-research-wf_9c488c33-result.json` (machine result, 111 agents).
- `copy-path-trace.md`, `raw/N-xe-copytrace/` (+ fourth coredump, kernel log),
  `raw/N-xe-copytrace-nocache/`, `raw/copy-tracer/` (other session, 17:35).
- `raw/N-xe-copyeng0-moesoft`, `raw/N-xe-copyeng0-moeoff`, `raw/N-xe-realtext-moecache/`,
  `raw/run-xe-moecache.sh`, `raw/bench-xe2.sh` (accepts `BENCH_EXTRA`).
- `raw/N-xe-copyeng0-r2..r4`, `raw/server-soak/` (six requests, responses with timings,
  `soak.log`), `raw/N-xe-inorder0` (+ third coredump), `raw/N-xe-cbevents0`,
  `raw/N-xe-immcl0`, `raw/N-xe-nodirectsub`, `raw/N-xe-norelaxed`, `raw/N-xe-trace/`
  (head/tail extracts + full `bench.err.gz`).
- `raw/bench.sh` (09-27 original), `raw/bench-xe.sh`, `raw/run-xe.sh`,
  `raw/run-xe-levers.sh`, `raw/run-xe-soak.sh`, `raw/run-xe-levers2.sh`,
  `raw/run-xe-trace.sh` and their logs.
- Outputs also remain under `/mnt/nvme1/oneapi-ab/N-xe-*`.
