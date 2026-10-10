# Linux 7.3-rc5 build for the Arc A770 Xe investigation

**Follow-up, 2026-09-30 21:57 UTC:** The host now runs this GCC rc5 kernel.
The [runtime correction and AOCC experiment](gaema-runtime-and-aocc-2026-09-30.md)
records the updated Intel packages and passing bounded host-IPC tests.
The first AOCC profiles failed correctness validation. The later build using
x86-64 instructions with Zen 4 tuning passed full build/module checks and QEMU,
and is installed as a separate entry for a manual trial boot. Existing kernels
and the rc1 boot default are preserved. No native AOCC result or speedup is claimed.
The earlier BCS failure is still not established as fixed. The build-session
record below describes the state before that first host boot.

**Build-session status, 2026-09-30 00:27 UTC:** Built, packaged, tested in QEMU and installed as a
separate kernel. The host still runs `7.3.0-rc1-273-tkg-bore`; rc1 remains the
ZFSBootMenu default. No rc5 host boot or A770 workload test has been performed
by this build session.

This build provides a newer Xe implementation, additional diagnostics and
preserved rollback for investigating the BCS copy-engine failures. It contains
no identified fix for those failures. Keep `UR_L0_USE_COPY_ENGINE=0` in
production until hardware testing supports changing that decision.

The authoritative implementation record is the
kernel build audit (local-only: `/home/svnbjrn/dev/krnl/linux-tkg-7.3-rc5-xe/customization.cfg.audit.md`),
with the Xe source review (local-only: `/home/svnbjrn/dev/krnl/linux-tkg-7.3-rc5-xe/xe-research.md`).
This document connects that build to the [captured workload evidence](../evidence/README.md)
and the [corrected interactivity analysis](../../research/xe-i915-llama-interactivity-firmware-2026-09-29.md).

## Exact build identity

| Component | Recorded value |
| --- | --- |
| Mainline source | `v7.3-rc5`, released 2026-09-27; latest RC checked on 2026-09-29 |
| Kernel source commit | `72d3fcf802c45d00b300f25b848a93c3a2bd7c7e` |
| linux-tkg framework | `ff4ee038c94e8987c0c3f2d60a394cda08dd2bd5` |
| Local configuration/evidence commit | `3e2b5c99`, branch `build/v7.3-rc5-xe` |
| Packages | `linux73-tkg-bore-rc5-xe` and `linux73-tkg-bore-rc5-xe-headers`, both `7.3.rc5-273` |
| Kernel release | `7.3.0-rc5-273-linux73-tkg-bore-rc5-xe` |
| Build profile | GCC 16.2.1, znver4, O3, BORE, dynamic preemption, 1000 Hz, 24 CPUs, NUMA, BTF and Rust; eight build jobs |
| A770 | DG2/Xe-HPG, PCI `0000:03:00.0`, device `8086:56a0`, 16 GiB ReBAR observed |
| Preserved firmware | GuC 70.53.0 and DMC 2.08; no firmware replacement |
| Preserved GPU userspace | compute-runtime `26.35.39758.10-1.1`, Level Zero loader `1.32.0-1.1`; no runtime/compiler upgrade for the workload |
| ZFS source | `e8a0a6cd4ed324446361e26bac9be7d6a4cd4e2e` |
| New ZFS DKMS identity | `zfs/2.4.99.r0.ge8a0a6cd4.73rc5xe`, restricted to the exact rc5 release |

This is an upstream RC with the selected linux-tkg patches and local build
configuration. It is not an unmodified upstream kernel. Xe itself received
no additional local driver patch. The only new local code workaround is the
sched_ext BTF repair described below.

The configuration worktree is
`/home/svnbjrn/dev/krnl/linux-tkg-7.3-rc5-xe`; packages and DKMS archives are
under its `artifacts/` directory. Native source, objects and test staging are
under `/mnt/nvme1/build/linux-tkg-7.3-rc5-xe`. The shared bare source cache is
still under `/mnt/nvme1/build/linux-tkg-7.3-rc4/linux-tkg/linux-kernel.git`;
retain it while the source worktrees depend on it.

## What can help with the Xe issues

Configuration changes below are compared with the previously audited rc4
build. The running host's source baseline is rc1. Exact values are recorded in
the compiled config (local-only: `/home/svnbjrn/dev/krnl/linux-tkg-7.3-rc5-xe/evidence/config-rc5`)
and rc4-to-rc5 config diff (local-only: `/home/svnbjrn/dev/krnl/linux-tkg-7.3-rc5-xe/evidence/config-rc4-to-rc5.diff`).

| Change or retained feature | Relevance to the investigation | Limit of the claim |
| --- | --- | --- |
| Advance the kernel source to rc5 | Allows a controlled comparison with rc1 while keeping userspace and firmware fixed | No reviewed commit was established as the fix for this host's BCS failure |
| New `CONFIG_DYNAMIC_DEBUG=y` and `CONFIG_DYNAMIC_DEBUG_CORE=y` | Enables selected `pr_debug`/`dev_dbg` messages, including TTM fault diagnostics, without permanently enabling broad logging | Most `xe_dbg` messages still use `drm.debug`; this does not provide arbitrary per-file Xe logging |
| New `CONFIG_FW_LOADER_DEBUG=y` | Logs firmware filename and SHA-256 when the corresponding dynamic-debug callsite is enabled; helps pin the actual payload requested by the driver | Hashes firmware-loader data, not GPU memory; hashing occurs even when its debug message is disabled |
| New compiled probe defaults: Xe `56a0`, i915 `!56a0` | Matches the host's existing A770 driver selection; both modules remain available for controlled comparisons | The existing boot parameters already select xe, so this is configuration consistency rather than a performance feature |
| Preserved PMU, hwmon, debugfs, tracing and device coredumps | Supports engine activity/frequency observations and captures around queue resets, timeout reporting and VM/TLB activity | Counters alone do not prove useful progress, identify a missing event signal or establish the failing layer |
| Preserved finite `DRM_XE_JOB_TIMEOUT_MAX=10000` | Avoids extending recovery merely to hide a stall | NEO's LR-mode compute path means this setting cannot be assumed to bound every hang |
| Separate package and boot image, with older kernels intact | Makes kernel comparisons and rollback practical | Actual host boot, ZBM kexec and rollback boot behavior still need supervised validation |

Display, DP tunneling, `DRM_XE_GPUSVM`, `DRM_XE_PAGEMAP`, `PERF_EVENTS`,
`HWMON`, `DEBUG_FS`, `DEV_COREDUMP` and tracing were already enabled in the
rc4 configuration and are retained. Their presence is not a new A770 speedup.
The generic SVM options do not override DG2's lack of faulting-USM capability.

Broad Xe debugging, fault injection and the BROKEN-gated
`CONFIG_DRM_USE_DYNAMIC_DEBUG` conversion remain disabled. Existing engine
timeslices, power policy, firmware, CCS partitioning and observation
permissions were not changed. In particular, the host tuning script's
attempt to request a 60-second timeout cap was not made effective by raising
the kernel cap.

## Upstream fixes: useful context, not a BCS cure

The source review checked relevant rc1-to-rc5 Xe changes and the inspected
post-rc5 fixes branch. Examples show why commit descriptions must be matched
to this GPU and failure before claiming an improvement:

| Included commit | What it fixes | A770/BCS interpretation |
| --- | --- | --- |
| [`d0c095287819`](https://github.com/torvalds/linux/commit/d0c09528781938362655d9c336364e777119bd6b) | MMIO GEM destroy reference lifetime and fault-handler access | Relevant to driver correctness; no demonstrated connection to the observed copy/event chain |
| [`c7a925c84704`](https://github.com/torvalds/linux/commit/c7a925c84704ec431598f9411dd89c0e16ae34ae) | Completes invalidations during wedged-device cleanup to avoid warnings | Improves cleanup after a wedge; does not establish prevention of the initial failure |
| [`f5fcf7e638b9`](https://github.com/torvalds/linux/commit/f5fcf7e638b904397ec0f66d3ea6766ef0cfe25b) | LSC untyped L1 cache flush behavior; mentions Llama.cpp | The added behavior is gated to graphics version 20+. DG2 is 12.55, so this is not an A770 Llama.cpp fix |

The inspected post-rc5 Xe fixes concerned SR-IOV VF BAR handling and were not
applicable to this consumer DG2 device. No cherry-pick was added for them.
See the source review (local-only: `/home/svnbjrn/dev/krnl/linux-tkg-7.3-rc5-xe/xe-research.md`)
for the exact inspected branch tip and the remaining hardware gates.

The candidate DG2 HuC patch was excluded after finding unresolved workqueue
lifetime, MEI transport and forcewake issues. HuC media authentication is
distinct from GuC scheduling and is not an identified fix for SYCL copy
failures. The initramfs contains HuC firmware for i915, but this does not
enable HuC authentication in xe. SR-IOV, PXP and EU stall sampling were not
unlocked for A770 by configuration changes.

## Independent fixes that make the build usable

**sched_ext registration:** GCC 16.2.1/O3 emitted an unsuitable DWARF
parameter location for `scx_bpf_error_bstr`. Pahole omitted that function's
BTF entry, leaving a zero registration ID and disabling sched_ext at boot.
Compiling only this error-reporting function at O2 restores the entry.
The linked BTF and QEMU initialization checks pass; the rc1 control still
shows the original error. This repairs CPU scheduler infrastructure. It is
not a repair to Xe's GPU scheduler, and no BPF scheduler was benchmarked.

**ZFS compatibility and boot image:** The pinned Linux 7.3 compatibility
source supplies the final kernel-thread filesystem-access fix and the other
required API adaptations. Its four patches were matched to upstream fixes;
the unrelated crypto, marshaling and Haiku branches were not combined into
this build. ZFS META is unchanged and experimental-kernel support is explicit.

The candidate-only initramfs contains xe, i915, amdgpu, NVMe, ZFS and SPL,
plus `tr`, which the existing diagnostic ZFS hook requires. This removes
the missing-`tr` diagnostic without editing the global hook or regenerating
older boot images. Neither fix establishes GPU reliability.

## Verification completed

| Check | Observed result |
| --- | --- |
| Kernel and header compilation/packaging | Passed; final linked kernel has no unresolved `scx_bpf_error_bstr` warning |
| Resolved and packaged configuration | Passed the hardware, scheduler and BTF config gate |
| DKMS against packaged headers | ZFS, scap 9.1.0 and v4l2loopback 0.15.4 built successfully in an isolated tree |
| QEMU rc1 control and rc5 candidate | Both imported/mounted the disposable NVMe-backed ZFS pool, read and wrote data, scrubbed with zero errors and exported it |
| QEMU sched_ext | rc5 passed `SCX_INIT_PASS`; rc1 retained `Failed to register kfunc sets (-22)` |
| Installed kernel and DKMS binaries | Byte-for-byte match to the tested staging binaries |
| Package integrity | Both new packages reported zero altered files |
| Production initramfs | Required modules/firmware match installed files; host ID and pool cache match; test-only hook absent |
| Rollback integrity | Six older kernel/initramfs files and 19,423 files in their module trees match their pre-install hashes |
| Host state | Running rc1, existing xe binding and ZBM default unchanged; root pool ONLINE with zero read/write/checksum errors |

The kernel release in the rc5 guest was
`7.3.0-rc5-273-linux73-tkg-bore-rc5-xe`. Its ZFS module reported version
`2.4.99-1`, srcversion `9171D09E6CCD01E4098CD66`, and the matching vermagic.
Installed ZFS userspace remains 2.4.4; this pairing is experimental.

Evidence: QEMU results (local-only: `/home/svnbjrn/dev/krnl/linux-tkg-7.3-rc5-xe/evidence/qemu-results.txt`),
installed-image checks (local-only: `/home/svnbjrn/dev/krnl/linux-tkg-7.3-rc5-xe/evidence/installed-image-checks.txt`),
rollback checks (local-only: `/home/svnbjrn/dev/krnl/linux-tkg-7.3-rc5-xe/evidence/integrity-after.txt`),
archive hashes (local-only: `/home/svnbjrn/dev/krnl/linux-tkg-7.3-rc5-xe/evidence/artifacts.sha256`)
and boot-file hashes (local-only: `/home/svnbjrn/dev/krnl/linux-tkg-7.3-rc5-xe/evidence/installed-boot.sha256`).
These absolute links refer to the local kernel worktree, not files copied
into this llama.cpp repository. The build audit records compiler and package
hook warnings as well; a successful build is not a warning-free-build claim.

## First host boot and useful diagnostics

Select `vmlinuz-linux73-tkg-bore-rc5-xe` once in ZFSBootMenu. Leave the existing
default in place. After boot, confirm the exact release and device identity:

```sh
uname -r
lspci -nnk -s 03:00.0
sudo zpool status -v zroot
cat /sys/kernel/sched_ext/state
ls /sys/bus/pci/devices/0000:03:00.0/drm/
ls /sys/bus/event_source/devices/xe_0000_03_00.0/events
sudo ls /sys/kernel/tracing/events/xe
sudo journalctl -k -b
```

The sched_ext state is expected to be `disabled` while using BORE, with no
registration error. Derive the A770 card number from its PCI `drm/` directory
before reading `/sys/kernel/debug/dri/N/gt0/uc/guc_info`; do not assume card0
survives a reboot. Inspect actual engine limits before creating workload
queues. If the journal is unexpectedly empty, inspect `sudo dmesg` and
journald health rather than treating an empty log as absence of failures.

Use `gputop` or native Xe PMU events for observation. The installed
`intel_gpu_top` rejects xe. Queue-reset and timeout tracepoints include
`xe_exec_queue_reset` and `xe_sched_job_timedout`; enumerate the actual event
directory for other VM/TLB events. Preserve a coredump after an actual
failure, without deliberately resetting the production GPU to test capture.

For firmware identification, add this argument for one diagnostic boot:

```text
dyndbg="file drivers/base/firmware_loader/main.c +p"
```

For a TTM-fault investigation, record the current debug state first. If this
file's callsites were initially disabled, enable them for the diagnostic run
and return them to that state afterward:

```sh
echo 'file drivers/gpu/drm/ttm/ttm_bo_vm.c +p' | sudo tee /sys/kernel/debug/dynamic_debug/control
# Capture the bounded diagnostic run, then restore the original disabled state:
echo 'file drivers/gpu/drm/ttm/ttm_bo_vm.c -p' | sudo tee /sys/kernel/debug/dynamic_debug/control
```

These are post-boot instructions, not actions performed by this documentation
task. Logging changes timing; do not use an instrumented run as the clean
performance result. General dynamic debug does not replace `drm.debug` for
the Xe messages that still use DRM's category interface.

## Comparison plan for the copy-engine failure

1. Start rc5 with the existing production copy-off workaround and explicit
   `--moe-cache off`. Keep the same binary, model, prompts, context, KV types,
   thread counts, graph policy and speculation policy. Record actual loaded
   SYCL/UR/NEO libraries and build IDs, not just package names. The graph flag
   parsed by this fork is `GGML_SYCL_ENABLE_GRAPH`; setting it does not prove
   graph replay occurred.
2. Compare rc1 and rc5 under otherwise matched conditions. Measure output
   correctness, resets, time to first token, inter-token stalls and throughput.
   Establish sole GPU tenancy for timing. Keep diagnostic and clean timing
   runs separate; disable speculation for driver attribution or explicitly
   report its acceptance behavior in representative production tests.
3. Obtain the missing i915 production-placement baseline with the same
   settings. For a one-boot i915 comparison, replace the paired probe options
   with `i915.force_probe=56a0 xe.force_probe=!56a0` and verify the actual binding.
   Do not merely add i915's positive probe while leaving xe forced on.
4. Test the default copy path only in a controlled reproduction with recovery
   available. Preserve source/destination lifetimes, allocation sizes and the
   observed cross-engine event dependencies. Confirm actual engine selection
   in the trace. A subsequent direct Level Zero version can separate UR from
   lower layers, but it must select its engines explicitly: the UR environment
   variable does not configure direct Level Zero API calls.

The decision to remove the production workaround requires hardware evidence
from this workload. More kernel knobs, synchronous waits on every copy, HuC
patches or longer timeouts are not justified by the current evidence.

## Evidence interpretation and rollback

The [earlier adversarial report](adversarial-research-artifact.md) contains
stronger conclusions than the source evidence supports:

- Four clean benchmark rounds plus six server requests support retaining the
  workaround, not claiming the failure probability is zero.
- Both original i915 and xe benchmark sets used `--moe-cache auto`. Their
  measured difference remains relevant to that streaming workload. The
  32.78 versus 32.38 t/s comparison is between two xe configurations; it
  does not establish xe/i915 parity with production placement.
- Event deadlock, mapping/visibility and buffer lifetime remain hypotheses.
  Successful i915 execution and standard API calls do not exclude an
  application ordering/lifetime defect. Model-load traces and decode failures
  also must not be presented as one proven causal sequence.
- `UR_L0_USE_COPY_ENGINE` is documented in the
  [official UR reference](https://oneapi-src.github.io/unified-runtime/core/LEVEL_ZERO.html).
  That is not a permanent compatibility guarantee. Monitor and pin the actual
  UR adapter as well as NEO; [UR development lives in intel/llvm](https://github.com/oneapi-src/unified-runtime#contents-of-the-project).

For rollback, select `vmlinuz-linux73-tkg-bore` (rc1, the unchanged default),
`vmlinuz-linux73-tkg-bore-rc4`, or `vmlinuz-linux71-tkg-bore` in ZFSBootMenu.
All corresponding packages and modules were preserved. Driver selection is
separate from kernel selection: retain or deliberately replace the paired
probe parameters for the intended comparison. No pool features were upgraded.
The original promotion gate of three clean boots over at least 24 hours
remains pending.

## Not claimed

No A770 speedup, copy-engine cure, exact root cause, HuC authentication,
suspend/resume result or rc5 physical-host boot is established. QEMU tested
emulated storage and scheduler initialization, not the real GPU, real storage
controller, ZBM kexec, encrypted pools or sustained production I/O. No BPF CPU
scheduler was loaded or benchmarked. Existing 2.4.4 ZFS userspace with the
experimental 2.4.99 module remains a compatibility consideration. No production
service, runtime, firmware, boot default or GPU setting was changed while
writing this document.
