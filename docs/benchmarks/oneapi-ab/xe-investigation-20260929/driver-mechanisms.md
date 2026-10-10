# Xe/i915 mechanisms relevant to Ornith, 2026-09-29

Source research only. No GPU tests, service changes, package installs, or configuration changes performed in this lane.

## What is actually failing

The [local copy trace](/home/svnbjrn/research-llama.cpp/sycl-oneapi-benchmarks-2026-09-29/copy-path-trace.md) establishes scheduler expert-weight uploads despite `--moe-cache off`. The pending Q6_K copy was 860672 bytes. The latest hangs showed BCS scheduled and CCS unchanged, first in queue finish and then inside memcpy enqueue. Counters establish scheduling, not useful progress or the particular instruction being executed. These observations replace the earlier activation-only interpretation and must not be conflated with the earlier CCS-busy observation.

## Stack and important differences

The relevant stack is ggml scheduler -> SYCL queue -> Unified Runtime Level Zero adapter -> Level Zero loader -> Intel compute-runtime (NEO) -> i915 or xe DRM ioctls -> firmware/GPU. Changing KMD also changes NEO's kernel-facing implementation. It does not merely replace a scheduler underneath an identical submission path.

| Area | Source-grounded difference and implication |
|---|---|
| Submission and dependencies | i915 execbuf traditionally carries a buffer-object list and binds at submission. Xe separates VM_BIND from EXEC and requires userspace to provide inter-exec and bind/exec dependencies. Xe's compute VM revalidation uses preempt fences/rebind work. More explicit dependency bookkeeping makes missing ordering or residency a plausible interface defect, not proof of one. [Xe command-submission documentation](https://docs.kernel.org/gpu/xe/xe_cs.html) |
| Residency | Xe uses TTM placement/eviction. User BOs can be evicted; moving them requires mapping repair before use. Kernel-owned objects and userspace-created runtime command buffers are different categories: a command buffer created by NEO is not automatically a pinned kernel BO. Therefore a NEO residency flag must not be read as a kernel pin guarantee. [Xe memory management](https://docs.kernel.org/gpu/xe/xe_mm.html) |
| LR queues | NEO 26.35's `getFlagsForVmCreate()` starts with `DRM_XE_VM_CREATE_FLAG_LR_MODE`. The kernel API explicitly permits jobs without an execution-time upper bound in LR VMs. Changing a sysfs ordinary job timeout cannot be assumed to bound these jobs. This does not disable preemption failure detection or every possible engine reset. [NEO tagged source](https://github.com/intel/compute-runtime/blob/26.35.39758.10/shared/source/os_interface/linux/xe/ioctl_helper_xe.cpp#L1337), [kernel 7.3-rc1 API](https://github.com/torvalds/linux/blob/v7.3-rc1/include/uapi/drm/xe_drm.h#L966) |
| GuC | GuC is not unique to xe: current i915 defaults enable GuC submission and HuC authentication on DG2 (it falls through the exclusions for older platforms). Thus 'xe uses GuC, i915 does not' is the wrong explanation. The drivers still have different queue, memory, reset, and firmware integration code. [i915 7.3-rc1 defaults](https://github.com/torvalds/linux/blob/v7.3-rc1/drivers/gpu/drm/i915/gt/uc/intel_uc.c#L26) |
| Support status | DG2 still has `require_force_probe = true` in kernel 7.3-rc1; the driver reports it as not officially supported unless forced. This is a specific support boundary for A770, not a claim about all Xe GPUs. [Current DG2 descriptor](https://github.com/torvalds/linux/blob/v7.3-rc1/drivers/gpu/drm/xe/xe_pci.c#L351) |

NEO's Xe `getFlagsForVmBind()` converts immediate/make-resident requests to `DRM_XE_VM_BIND_FLAG_IMMEDIATE`; it does not turn `bindLock` into a kernel pin. Its user-pointer mapping uses `MAP_USERPTR`. CPU caching defaults to WC, selecting WB for coherent system-memory-only allocations. These are concrete places to audit bind lifetimes and CPU/GPU visibility; they do not establish that the installed stack violates coherence. [NEO tagged helper](https://github.com/intel/compute-runtime/blob/26.35.39758.10/shared/source/os_interface/linux/xe/ioctl_helper_xe.cpp#L715)

For comparison, i915 discrete-GPU API rules also select WC for allocations with device-memory placement and WB/coherence for system-memory-only allocations. Do not assert a universal 'xe uncached, i915 cached' difference. [i915 API caching rules](https://docs.kernel.org/gpu/driver-uapi.html#c.drm_i915_gem_set_domain)

## Why copy-engine-off is a meaningful clue

BCS copies and CCS computation require their producer/consumer ordering and completion visibility to agree. Removing BCS changes engine selection, overlap, events, command-stream generation and potentially memory handling simultaneously. A successful run therefore implicates that combined path; it does not isolate PCIe bandwidth, a single fence bug, or a bad blitter.

The distinct 13:15:16 dump's BCS IPEHR=0x72080025 matches the Xe-HPG COMPUTE_WALKER header, as preserved in [COORDINATION.md](/mnt/nvme1/oneapi-ab/xe-investigation-20260929/COORDINATION.md). Wrong stream selection, reused/overwritten batch storage, stale mapping/visibility, or a misleading capture remain alternatives. It is more specific than generic instability but is not a proven wrong-engine submission. The 13:20 crash and pinned-off dump must keep their separate identities.

## External leads, with hardware boundaries

[NEO issue 948](https://github.com/intel/compute-runtime/issues/948) reports BMG faults associated with direct-submission ring/semaphore addresses and discusses residency. It is relevant as a failure mechanism to investigate, not a match or upstream confirmation for this A770.

Crucially, the [26.35 DG2 capability table](https://github.com/intel/compute-runtime/blob/26.35.39758.10/shared/source/xe_hpg_core/hw_info_dg2.cpp#L33) enables direct submission for CCS/CCS1-3 but contains no BCS enable entry. Do not assume the BMG BCS direct-submission configuration applies here. Live queue selection or runtime overrides must be established first. The [Linux direct-submission residency helper](https://github.com/intel/compute-runtime/blob/26.35.39758.10/shared/source/direct_submission/linux/drm_direct_submission.inl#L205) waits on a paging fence; understanding its callers and Xe rebind handling is necessary before calling that a missing residency check.

[NEO issue 973](https://github.com/intel/compute-runtime/issues/973) documents another A770 residency defect involving reused private kernel surfaces. It is a different workload and failure signature, useful evidence that runtime residency bugs are real, not attribution here. [Intel LLVM issue 18424](https://github.com/intel/llvm/issues/18424) reports Level Zero v1 deadlock on A770 with a different application and versions; again a lead, not a diagnosis.

## Discriminating next experiments, not performed here

1. Obtain a Level Zero/NEO trace with command-list engine class, queue identities, event wait/signal addresses, and VM bindings for the pending upload. Capture the missing producer or bad batch mapping instead of inferring it from engine utilization.
2. A/B immediate versus regular command lists, retaining BCS, with identical installed libraries. If reproducible failure follows immediate lists, focus on list/event/batch reuse. Confirm that the adapter actually honors the switch; do not stack speculative knobs.
3. Isolate direct submission only after confirming which engines use it. A passing CCS-direct-submission-off run cannot alone prove a BCS direct-submission bug.
4. Test a matched stable kernel against this 7.3-rc1 with the same userspace and firmware. That isolates KMD changes more cleanly than changing NEO, UR, kernel and firmware together.
5. For performance, measure total upload bytes/count, enqueue/wait time, VM_BIND time, and actual PCIe/GT clocks on matched real-text runs. Seeded random tokens do not guarantee identical expert routing across floating-point execution differences. Long-context pressure and transfer synchronization can explain sensitivity without proving eviction or bandwidth loss.

For the current service retain the empirically useful copy-engine-off setting while investigating. Timeout inflation and event-cache disabling are not established fixes; the latter already failed in the local trace. No newly suggested lever is validated by this research.
