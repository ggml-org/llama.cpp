# Deep research: xe vs i915 for llama.cpp SYCL on the Arc A770 (DG2) - 2026-09-29

Method: `/deep-research` workflow (`wf_9c488c33-b94`): 6 search angles, 111 agents, 1568
tool uses, 10.9 M tokens, 37 min. Each claim faced three adversarial verifiers; a claim
survived only without two refutes. Full machine result: `raw/deep-research-wf_9c488c33-result.json`.
Below: what survived, what was refuted, how it squares with what this set measured, and
what is actionable. Anything marked **[local]** is our own measurement, not the research.

## Verified findings

1. **Submission model is the real difference, not GuC vs execlists.** On DG2 both drivers
   drive the *same* GuC firmware. i915 fronts it with its own `i915_sched_engine`, a full
   buffer-object residency list per exec, kernel-side implicit sync via dma-resv, and one
   global watchdog (`i915.request_timeout_ms`, 20 s). xe fronts it with `drm_sched` in a 1:1
   scheduler-per-queue mapping, and compute-runtime (NEO) **creates every xe VM in LR
   (long-running) mode** with one `USER_FENCE` out-sync per exec: the exec ioctl carries no
   BO list and adds **zero kernel-side dependencies between execs**. Ordering a blitter copy
   against a compute kernel is therefore entirely the user-mode runtime's job (L0 events,
   GPU-side semaphores). Eviction uses preempt fences plus a rebind worker. On DG2 (no
   recoverable page faults) an A770 VM is necessarily LR + preempt-fence mode. *(high;
   kernel xe_cs / vm-bind docs, xe_drm.h, xe_exec.c, NEO ioctl_helper_xe.cpp)*
2. **LR queues have no job watchdog.** `xe_guc_submit.c` sets the scheduler timeout to
   `MAX_SCHEDULE_TIMEOUT` for queues on an LR VM; the TDR comment reads "LR jobs can only
   get here if queue has been killed or hit an error". *(high; verified here in v7.3-rc1
   source, see cross-check below)*
3. **Timeout knobs.** `CONFIG_DRM_XE_JOB_TIMEOUT_MAX=10000` is an absolute ceiling on the
   sysfs `job_timeout_ms` (and on `job_timeout_max`), `-EINVAL` above it regardless of
   privilege; only a kernel rebuild raises it. `CONFIG_DRM_XE_PREEMPT_TIMEOUT=640000 us` is
   the boot default, runtime-changeable per engine class within [1 us, 10 s], passed to
   GuC as the PREEMPTION_TIMEOUT KLV; a context that does not reach an arbitration point
   before expiry is reset. Exec queues copy both values **at creation**, so sysfs writes
   affect only later queues. *(high; Kconfig.profile help text upstream 5e34374d6531,
   xe_hw_engine_class_sysfs.c; matches [local] `xe-a770-tune` behaviour)*
4. **Firmware is identical across drivers.** DG2 GuC is one blob, `i915/dg2_guc_70.bin`,
   loaded by both i915 and xe; shipped version 70.53.0 (compat 1.26.0, linux-firmware tag
   20251111, still current in linux-firmware main and Arch on 2026-09-29; both drivers
   *want* 70.53.0). GuC version cannot explain any i915-vs-xe difference on this host.
   *(high; drm-firmware MR !43, linux-firmware WHENCE, xe_uc_fw.c, intel_uc_fw.c, [local]
   debugfs)*
5. **A newer DG2 GuC exists upstream-adjacent.** GuC **70.74.0** (UAPI 1.38.4) merged into
   the `intel-staging` branch of gitlab.com/kernel-firmware/drm-firmware on 2026-09-02
   (MR !84, "Update GUC to v70.74.0 for MTL, DG2, BMG, LNL, PTL, NVL-S"). Not in
   linux-firmware, not in Arch, not in `linux-firmware-git` (tracks linux-firmware main),
   not in Intel's GitHub firmware repo. Only source:
   `https://gitlab.com/kernel-firmware/drm-firmware/-/raw/intel-staging/i915/dg2_guc_70.bin`.
   Intel's test patches ("drm/xe/guc: Test GuC v70.74.0 for MTL, DG2") passed xe KUnit;
   i915 full CI had failures on unrelated shards. *(high)*
6. **Installing it is a drop-in.** Both drivers match only the major version; a newer
   minor loads silently, an older one logs "is recommended, but only ... was found" and
   loads anyway; hard floor 70.29.2. Steps on Arch: put an uncompressed
   `/usr/lib/firmware/i915/dg2_guc_70.bin` next to the shipped `.zst` (the kernel prefers
   the uncompressed file), `mkinitcpio -P` (xe is early-loaded here), reboot; or point
   `xe.guc_firmware_path=` at another path. Revert = reinstall `linux-firmware`. *(high;
   xe_uc_fw.c version checks; the Arch packaging steps are standard practice, not a
   verified claim)*
7. **HuC on DG2 under xe: unsupported, and stalled.** `xe_uc_fw.c` has no DG2 HuC entry
   and asserts `platform != XE_DG2` in the CPD parser ("We don't support DG2 HuC right
   now"). The only effort is an out-of-tree "[PATCH v4] drm/xe: add DG2 HuC GSC support and
   MEI integration" (non-Intel author, 2026-08-02): CI refused it (author not allowlisted),
   zero reviews, the one maintainer reply asked the author to stop resending. So no VAAPI
   CBR/VBR low-power encode on xe for the A770 for the foreseeable future. *(high; matches
   [local] encode test: CQP OK, VBR/CBR fail)*
8. **No upstream report matches our blitter signature.** Searched intel/compute-runtime,
   intel/llvm, drm/xe GitLab, kernel lists: nothing with bcs `Engine reset` + `Timedout
   job` + `RING_ESR=0x1` + `IPEHR=0xfffff000` on DG2. Nearest analog:
   intel/compute-runtime **#999** (2026-09-17, open): Battlemage Arc Pro B65 under xe, a bcs
   context is consumed by GuC but never dispatched, times out "not started" after ~108 s.
   Same failure *shape* (copy queue stuck), different *signature* (idle engine, no CS
   error), different silicon. Two candidate xe fixes were ruled out for DG2 (USM-reserved
   BCS timestamp fix: has_usm platforms only, already in 7.3-rc1; "stop re-submitting
   signalled jobs": suspend/resume race, already in 7.3-rc1). *(medium)*
9. **The performance question is open.** Every claim attributing the prefill gain or
   decode loss to GT frequency management (GuC SLPC policy, RPe-750 vs RPn-300 floors,
   sysfs pinning as a supported ramp-latency fix) was refuted 0-3. So was "LR-mode VM_BIND
   forces synchronous bind round-trips". Nothing on ULLS/direct submission, immediate
   command lists, counter-based events, RC6, ASPM or multi-CCS survived. Candidate
   contributors that *are* verified structure: user-fence completion instead of
   dma-fences; no kernel-side implicit sync; preempt-fence eviction; drm_sched 1:1 front
   end; and the copy-engine path (our own observation). None measured. *(low)*

## Refuted (do not cite)

- xe's per-engine-class job timeout default "5000 ms hardcoded, sysfs only" (0-3).
- Linux 6.12.36's "stop re-submitting signalled jobs" as the fix for a bcs Timedout job
  triple (0-3): it is a suspend/resume race.
- "GT frequency under xe is a pure GuC SLPC policy between min and max" and "writing
  max_freq <= min_freq pins the clock as the supported ramp-latency mechanism" (0-3 each).
- LR-mode VM_BIND cannot be pipelined, each bind is a synchronous round-trip (0-3).
- The June 2026 check_timeout() change (never-scheduled job -> GT reset) as relevant to
  our copy-queue timeouts (1-2).

## Cross-check against this set's measurements

- **Finding 2 corrects this README's earlier wording.** NEO's queues sit on an LR VM, so
  `job_timeout_ms` never armed for them. Every kernel line we logged reads `Engine reset:
  engine_class=bcs ...` **before** `Timedout job ...`: GuC detected the failure (the
  coredumps show `RING_ESR=0x1`, `IPEHR=0xfffff000`, context runtime 0 ms - a command
  stream error, not a slow job), the queue was marked reset, and the TDR ran immediately
  as the error path that prints "Timedout job" and writes the devcoredump. Raising `bcs`
  `job_timeout_ms` to 10 s (done 16:37) therefore changed nothing; it stays only because
  it is harmless. The silent hangs are the other face of the same fact: a queue that is
  merely waiting on a never-signalled event is not an error, so nothing ever fires.
- **Finding 1 is the mechanism behind the workaround.** With no kernel-side ordering
  between the compute and copy immediate lists, correctness of every ccs<->bcs edge rests on
  UR's event chaining. `UR_L0_USE_COPY_ENGINE=0` removes the second list and with it every
  cross-engine edge; 0 failures in 10 afterwards versus 11 in 14 before. Which layer drops
  the edge (UR adapter, NEO, GuC) is still not identified; the research found no bug that
  names it. Our four coredumps plus the UR trace are, as far as the research could tell,
  the first DG2 record of it.
- **Finding 4 kills one hypothesis we had written down**: firmware differences. Same blob,
  same wanted version, both drivers.
- **Finding 9 removes another**: the RPe-750 floor under xe was listed here as a possible
  part of the prefill gain. The research found no support for frequency management as an
  explanation and refuted the "pin via sysfs" recipe as a supported mechanism. Treat the
  floor difference as an observation only.
- **Finding 7 matches the encode test exactly** (CQP works, BRC modes fail).
- **Not covered by the research at all**: the user-mode knob space (UR_L0_*, NEO debug
  keys, GGML_SYCL_*) and Intel's llama.cpp SYCL guidance produced no verified claim. Our
  own localization rung (`README.md`, "Localization rung") is the only data on those.

## Actionable configuration (both drivers)

| Layer | xe (current) | i915 (fallback) |
|---|---|---|
| Driver selection | ZBM cmdline `i915.force_probe=!56a0 xe.force_probe=56a0`, `MODULES=(xe amdgpu zfs)` | swap the `!`, `MODULES=(i915 amdgpu zfs)`, `mkinitcpio -P` |
| Preemption | `preempt_timeout_us` per engine class via sysfs; `xe-a770-tune` sets ccs/rcs to 7 500 000 (i915 parity). Applies to queues created afterwards -> restart the unit after changes. | Kconfig `DRM_I915_PREEMPT_TIMEOUT_COMPUTE=7500` already; `i915.request_timeout_ms` (20 000) is the only watchdog |
| Job timeout | Irrelevant for NEO's LR queues. Ceiling 10 s by Kconfig for anything else. | `request_timeout_ms`, global |
| Copies | **`UR_L0_USE_COPY_ENGINE=0`** in the unit (drop-in `xe-copy-engine.conf`) - the one lever with evidence | Not needed on the evidence (0 failures in the 09-27/28 rounds); harmless if left set |
| Placement | `--moe-cache off` (production; 32.5 t/s real text). Not `soft`, not `auto`. | Same; the i915 reference rows were `auto` and need re-running with `off` |
| Firmware | 70.53.0 (stock). Optional experiment: 70.74.0 from drm-firmware intel-staging, drop-in, unsupported by any distro | Same blob, same option |
| Encode | CQP only (no HuC) | Full BRC (HuC 7.10.16 via mei_gsc) |
| Not worth touching on evidence | `probe_display`, `wedged_mode`, `guc_log_level`, GT freq pinning, ASPM, `ccs_mode` (EBUSY while kwin holds card0; NEO composes the i915 path only) | `enable_guc` (auto already enables GuC submission + HuC on DG2), `enable_dc`, `enable_fbc/psr` (headless) |

## Open questions after the research

1. Which layer drops the ccs<->bcs event edge under xe on DG2 (UR adapter, NEO, GuC)? A
   minimal Level Zero reproducer (two immediate lists, event-chained memcpys against a
   kernel) would decide it and is what an upstream report needs.
2. Does GuC 70.74.0 change the default copy path? One boot with the intel-staging blob and
   the failing bench (`bench-xe.sh`, copy engine on) answers it; 11-of-14 failure rate
   makes a clean pass informative.
3. Why is prefill +55-62 % on xe for the streamed-expert placement and +6-17 % dense? No
   verified explanation exists. A per-copy latency microbenchmark (L0 `zeCommandListAppendMemoryCopy`
   of 1 MB slices, both drivers) would separate copy cost from kernel-launch cost.
4. The production placement (`--moe-cache off`) has no i915 number. One reboot.

## Sources (as returned by the workflow; verified by its adversarial pass)

- https://www.kernel.org/doc/html/next/gpu/xe/xe_cs.html
- https://dri.freedesktop.org/docs/drm/gpu/drm-vm-bind-async.html
- https://github.com/torvalds/linux/blob/master/include/uapi/drm/xe_drm.h
- https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/xe/xe_exec.c
- https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/xe/xe_guc_submit.c
- https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/xe/Kconfig.profile
- https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/xe/xe_hw_engine_class_sysfs.c
- https://raw.githubusercontent.com/torvalds/linux/master/drivers/gpu/drm/xe/xe_uc_fw.c
- https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/i915/gt/uc/intel_uc_fw.c
- https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/i915/gt/uc/intel_guc_submission.c
- https://docs.kernel.org/gpu/rfc/i915_scheduler.html
- https://github.com/intel/compute-runtime/blob/master/shared/source/os_interface/linux/xe/ioctl_helper_xe.cpp
- https://github.com/intel/compute-runtime/blob/master/shared/source/os_interface/linux/ioctl_helper_i915.cpp
- https://github.com/intel/compute-runtime/issues/999
- https://gitlab.com/kernel-firmware/drm-firmware/-/merge_requests/43
- https://gitlab.com/kernel-firmware/drm-firmware/-/merge_requests/84
- https://gitlab.com/kernel-firmware/linux-firmware/-/blob/main/WHENCE
- https://gitlab.freedesktop.org/drm/xe/kernel/-/merge_requests/361
- https://github.com/intel/media-driver/blob/master/README.md
- https://www.phoronix.com/news/Intel-Alchemist-HuC-Xe-Patches
- https://cdn.kernel.org/pub/linux/kernel/v6.x/ChangeLog-6.12.36
- Mailing-list threads read via the ratatoskr.run mirror (lore/patchwork were bot-walled):
  intel-xe 2026/06/17094663, 2026/06/17109909, 2026/08/17442060, 2026/08/17355490;
  intel-gfx 2026/08/17441870.

## Not claimed

- The workflow's own caveats: question (2) unanswered; UMD knobs unresearched; #999 is a
  single unconfirmed Battlemage report; 70.74.0 has no DG2 field testing outside Intel CI;
  two findings passed 2-1; firmware/patch status is a 2026-09-29 snapshot.
- The LR/TDR correction above is from reading v7.3-rc1 `xe_exec_queue.c` and
  `xe_guc_submit.c` (fetched from GitHub, tag v7.3-rc1); the tkg tree on this host is
  headers-only. Whether NEO ever creates a non-LR VM for its copy queues was not checked
  beyond the research's reading of `ioctl_helper_xe.cpp` (LR unconditional).
