# Copy-path trace, 2026-09-29 17:35 follow-up

Installed binary: 4e7400c3a, xe, existing bench-xe.sh flags, bounded to 420 s.
Production service stopped for each run and restarted by the runner trap.
KWin remained a DRM client. These instrumented runs are diagnostic, not clean
performance comparisons. No source-code or driver-configuration patch was applied.

## Observed

- Default copy-engine configuration with a bounded LD_PRELOAD tracer completed
  pp512 and tg64, then stalled before the pp512@8192 result. Two stacks show
  the host in urQueueFinish -> queue_impl::wait. In a 3 s sample, BCS cycles
  increased by 57,603,176 while CCS cycles stayed unchanged; the host consumed
  one CPU core. No new xe reset appeared during these runs.
- The first 64 traced graph-internal async uploads include Q4_K
  blk.0.ffn_gate_exps.weight and blk.0.ffn_up_exps.weight, in multi-megabyte
  ranges. These are weights, not activation tensors.
- Retrying with UR_L0_DISABLE_EVENTS_CACHING=1, otherwise the same tracer and
  flags, stalled before the first result row. Two stacks put the host in
  enqueueMemCopyHelper -> urEnqueueUSMMemcpy -> ggml_backend_tensor_set_async.
  A 2 s counter sample again showed BCS scheduled, CCS unchanged.
- Live GDB inspection of the wrapper's arguments identified the pending copy:
  name=SYCL0#blk.1.ffn_down_exps.weight#0, type=GGML_TYPE_Q6_K,
  shape={512,2048,256,1}, offset=202137600, size=860672 bytes.
  Debug symbols were compiled separately; their .text matched the loaded
  tracer byte-for-byte before symbols were relocated into the debugger.
- Both runs were manually SIGTERM-terminated after repeated stalled snapshots
  (exit 143), before the 420 s watchdog. Neither is a completed benchmark.
- Ornith was restored; /health returned {"status":"ok"}.

## What this resolves

Later placement audit: these bench invocations inherited --moe-cache auto,
not the production unit's explicit off. They therefore do not establish
expert streaming in the production placement. At this revision,
ggml-backend.cpp's copy_experts helper (line 2436) calls async-set for
host-resident weights consumed by GPU MUL_MAT_ID. The live pending-copy
metadata confirms this path in the traced auto-placement runs. The previous
claim that these traced transfers were only CPU-expert activation shuttles
was incorrect. See the campaign README's 17:49-18:05 placement audit.

Disabling UR event caching did not provide a workaround in this trial.
That does not exclude event lifetime/ordering bugs: the switch also changes
allocation and destruction behavior. The exact failing event and its producer
remain unidentified. Engine counters show scheduling, not useful progress.
The practical copy-engine-off workaround installed by the other session was
preserved; no new production knob was enabled.

## Artifacts and reproduction

- raw/N-xe-copytrace/: stderr metadata, partial table, two stacks, maps, counters.
- raw/N-xe-copytrace-nocache/: same plus live-copy.txt and captured test setting.
- raw/copy-tracer/: tracer source, CPU forwarding check, archived GDB commands.
  The GDB file contains that run's PID/address and is evidence, not a reusable
  invocation without updating those fields.
- /mnt/nvme1/oneapi-ab/xe-investigation-20260929/bin/llama-bench wraps only the
  binary with LD_PRELOAD; preloading the shell itself is unsupported.

CPU tracer check: compile trace-copies.c as a shared library with -fPIC and
-ldl against ggml/include; compile check-trace.c against libggml-cpu,
libggml-base and libggml-sycl. LD_PRELOAD=./trace-copies.so ./check-trace
passed and preserved four uploaded/read-back floats. GPU graph interception
was subsequently observed in the two runs above. No claim of a runtime fix,
output correctness for the unfinished GPU runs, or a new failure probability.
