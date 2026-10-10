# Codex source-audit lane, 2026-09-29

GPU testing belongs to the other investigation running run-xe-soak.sh and
run-xe-levers2.sh. This session will not stop/start services or launch GPU work
until explicitly handed a slot. No driver configuration changes.

Findings for the runtime investigator:

1. The saved N-xe-server/xe-devcoredump-132039.txt.gz is NOT the 13:20:39
   journal event: its header says PID 264942, seqno 11684, snapshot
   1790687716.869280196 (13:15:16 UTC). Journal says PID 318738, seqno 7811.
   Its IPEHR is 0x72080025, versus 0xfffff000 in the pinned-off dump.
   Preserve both; do not describe these as matching register signatures.
2. At master 4e7400c3a the scheduler has only two direct async-set call sites:
   prefetch_stage (off here), and copy_experts in compute_splits, line 2436.
   The latter uploads host-resident WEIGHTS consumed by a GPU MUL_MAT_ID.
   Ordinary CPU activation return uses tensor_copy's synchronous fallback
   because SYCL's iface.cpy_tensor_async is NULL. Thus CPU-resident does not
   imply CPU-executed; --moe-cache off disables the optional cache, not the
   scheduler's routed expert upload optimization. Capture tensor names/sizes
   before accepting either interpretation of the failure stack.
3. Both saved dumps have queue Timeout: 0, Preempt timeout: 640000 us.
   Their reset cannot be explained solely by the sysfs bcs job timeout=5000.

Preparing a bounded LD_PRELOAD async-copy tracer here for your next offline
run. It does not change queue ordering or copy parameters. Instrumented
throughput is diagnostic, not a benchmark. No remote messages or PRs sent.

## Tracer ready (16:43 UTC)

`trace-copies.so` intercepts the exported ggml async-set and graph-submit APIs.
It records the first 64 async uploads made inside graph submission (tensor
name, quant type, shape, offset, bytes), plus totals at normal exit. It also
prints up to 4 uploads outside graph submission. No payloads are logged.

Set LD_PRELOAD on the llama-bench executable itself, NOT on bench-xe.sh (the
shell has no ggml symbols). For the existing runner, put it on the final env
invocation before timeout/llama-bench only with a direct binary wrapper, or
invoke the binary directly under your existing service/timeout supervision:

    LD_PRELOAD=/mnt/nvme1/oneapi-ab/xe-investigation-20260929/trace-copies.so /usr/bin/llama-bench ...

CPU-only smoke verification passed: real libggml CPU tensor upload/readback,
all 4 float values preserved, wrapper printed tracer_cpu_check, bytes=16,
total=1. Compile commands:

    cc -shared -fPIC -Wall -Wextra -Werror -I /mnt/mrgr/ggml-llama.cpp/ggml/include trace-copies.c -ldl -o trace-copies.so
    cc -Wall -Wextra -Werror -I /mnt/mrgr/ggml-llama.cpp/ggml/include check-trace.c -lggml-cpu -lggml-base -lggml-sycl -o check-trace
    LD_PRELOAD=./trace-copies.so ./check-trace

This verifies forwarding and metadata access, not that GPU graph-internal
calls are intercepted at runtime. The installed libggml-base has PLT calls
for async-set (objdump), so LD_PRELOAD is applicable; graph trace still needs
an actual offline run. Instrumented t/s is not clean benchmark evidence.

## Concrete coredump lead (16:46 UTC)

The EARLIER server dump's IPEHR=0x72080025 is exactly Xe-HPG COMPUTE_WALKER
DW0, while Capture_source=GuC labels the engine bcs0, full-capture. Source:
intel/compute-runtime d70df548ba9ddc71b8cf0cd76983944827a7087f,
shared/source/generated/xe_hpg_core/hw_cmds_generated_xe_hpg_core.inl,
COMPUTE_WALKER struct/init (lines 5160-5318 in this checkout).

    0x25 | (2 << 18) | (2 << 24) | (2 << 27) | (3 << 29) == 0x72080025

Fields: length=37, CFE subopcode=2, compute opcode=2, pipeline=2, type=3.
This is a valid compute header, not random garbage. A compute command seen by
a blitter points toward wrong command stream, recycled/overwritten batch
storage, stale mapping/visibility, or misleading/stale capture. The dump
cannot distinguish these (VM state failed with -19; no batch contents).
Do not call it proven wrong-engine submission. It DOES justify prioritizing
immediate-list/direct-submission isolation over another timeout adjustment.
Installed libze_intel_gpu contains the strings EnableDirectSubmission,
DirectSubmissionRelaxedOrdering, DirectSubmissionRelaxedOrderingForBcs,
NEOReadDebugKeys; actual knob application still needs trace confirmation.

## Existing runner integration

A two-line executable wrapper is ready at `bin/llama-bench`. To use your
existing timeout and service trap unchanged (once the GPU lane is available):

    /mnt/nvme1/oneapi-ab/bench-xe.sh N-xe-copytrace '' /mnt/nvme1/oneapi-ab/xe-investigation-20260929/bin

This keeps preload out of the shell/timeout processes. The wrapper passed
bash -n; the underlying tracer passed the CPU forwarding check. GPU run is
not performed by this source-audit session.

The new NEO command-buffer pool switch is not the first lever to add: in the
installed-version source tag 26.35.39758.10, its default follows
is2MBLocalMemAlignmentEnabled(), whose generic implementation is false and
which DG2 does not override in its product-helper specialization. Avoid a
no-op pool-disable run unless tracing shows that pool was enabled.

## 16:54 ownership check

User offered GPU tentatively; checked before running. N-xe-inorder0 started
16:53:32 under the other session's run-xe-levers2.sh, so no GPU work launched
here. Copy-engine-off rounds r2/r3/r4 all exited 0, and server-soak/soak.log
records six HTTP 200 completions. The other session installed
/etc/systemd/system/llama-gpu@Ornith-1.5-35B-Q4_K_M.service.d/xe-copy-engine.conf
with UR_L0_USE_COPY_ENGINE=0. Preserve that when restoring service.

Related upstream reports (leads, NOT attribution of this host's failure):
- https://github.com/intel/compute-runtime/issues/948: other Intel hardware,
  direct-submission ring/semaphore residency suspected under xe; not proven.
- https://github.com/intel/llvm/issues/18424: L0 v1 deadlocks including A770,
  in-order-list selection changes behavior. Different application/version.

13:15 dump attribution now confirmed against this boot's kernel journal:
13:15:16, PID 264942, seqno 11684, bcs state=0x29. Saved matching kernel lines
in kernel-131516-confirmed.txt. This was an additional actual reset, not just
a timestamp interpretation. The saved dump remains distinct from 13:20.
