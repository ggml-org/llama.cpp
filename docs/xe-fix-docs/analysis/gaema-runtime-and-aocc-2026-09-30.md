# Gaema runtime correction and AOCC kernel experiment

Status, 2026-09-30: the host IPC correction is retained in the current runtime
and its regression passes on the A770. The host runs the existing GCC-built
`7.3.0-rc5-273-linux73-tkg-bore-rc5-xe`. The first AOCC kernel trial failed
correctness validation. A later guide-driven control found a viable profile
for the six previously affected compilation units: O2, x86-64 instructions
and Zen 4 tuning. That profile now passes the complete kernel and external-module
builds, package checks and disposable QEMU tests. The separate kernel, headers,
matching external modules and production boot image are installed. It is ready
for a manual trial boot; rc1 remains the ZFSBootMenu default and all recorded
rollback files are unchanged. No host reboot has occurred. Performance relative
to GCC remains unmeasured.

## Runtime correction

The correction was first built and installed as `intel-compute-runtime-git
22.43.24558.r13059.g8ae033266e-3`, retaining upstream commit
`8ae033266e170aba980e1f383d99c4c33d2edc8e` and the user's Gaema merge.
The old-looking package version comes from the existing git-version recipe;
use the source commit and local patch hashes to identify this build.

During the kernel trial, a separate update installed
`22.43.24558.r13063.gbe85a8d685-1` from commit
`be85a8d6855732cc8ab74714b8bc815007acd578`. This session did not perform that
upgrade. It retains the identical `030` and `040` patches and adds
`050-neo-retry-userptr-bind-readonly-on-eperm.patch`. Final checks against that
currently installed package pass both host IPC modes, Intel-only OpenCL and
Level Zero enumeration, and DG2 compilation. Package integrity reports 38
files with zero alterations. The package version and runtime library hash
were unchanged across those checks. The new `050` patch was not reviewed or
validated against the BCS workload by this session.

040-gaema-host-ipc-cpu-access.patch (local-only: `/home/svnbjrn/projects/intel-compute-runtime-git/040-gaema-host-ipc-cpu-access.patch`)
corrects `MemoryManager::importFdHandle()`: host IPC allocations are CPU-mapped,
so they must not assert the fork's no-CPU-access contract. Only device imports
retain that marking. The existing `030` patch still keeps the private flag
disabled unless explicitly requested. It must remain unset for ordinary use
on this kernel. An assertion inspecting the real DRM buffer object was added
to the existing host IPC unit test; the package still skips the unit suite.

The kernel was not missing a general A770 feature. Gaema's
[matching kernel patch](https://github.com/gaema/linux/commit/086ac7f76bc015d9e672b928718c1150e6d82e2d)
bypasses a coherency check for imported buffers. It does not enforce absence
of CPU mappings. DG2's ordinary write-back PAT is already classified
`XE_COH_1WAY`, and same-device imports reuse the original GEM object.
The fork's B70 motivation does not establish an A770 need. No bit-8 patch,
P2P topology bypass or ACS override was added. See the
source review (local-only: `/home/svnbjrn/dev/krnl/linux-tkg-7.3-rc5-xe/gaema-xe-review-2026-09-30.md`).

## Actual Intel stack used

| Package | Version |
| --- | --- |
| intel-compute-runtime-git | `22.43.24558.r13063.gbe85a8d685-1` |
| intel-gmmlib-git | `22.10.2.r0.g733c91a-1` |
| intel-graphics-compiler | `1:2.41.10-1` |
| level-zero-headers-git | `1.34.0.r2.gae3db48-1` |
| level-zero-loader-git | `1.34.0.r2.gae3db48-1` |
| intel-deep-learning-essentials | `2026.1.4-2` |
| intel-llvm-git | `23.0.0_r615667.591a75f17bac-1` |

The session's `r13059...-3` package `.BUILDINFO` records the updated GMMlib, IGC and Level Zero
dependencies. Loader diagnostics resolved `/usr/lib/libigdgmm.so.12`,
`/usr/lib/libigc.so.2`, `/usr/lib/libigdfcl.so.2` and the installed Intel
OpenCL runtime. These updates change the userspace baseline from the original
kernel-build report. They do not themselves add a kernel ioctl flag.

## Reproduced failure and verification

The regression test derives from upstream's
`zello_host_ipc_copy_dma_buf.cpp`. It transfers a host allocation between two
processes, writes it through the CPU mapping, reads it back with a compute
queue and checks the data. Adaptations are recorded: one child, 128 KiB
(the original odd size fails its four-byte fill-pattern validation), five-second
queue waits, checked child exit, resource cleanup, and a 20-second process-group
deadline. It exercises the legacy fd-transfer IPC path, not every opaque IPC
or external-memory API.

| Runtime and test setting | Result |
| --- | --- |
| Installed `-2`, private flag unset | PASS |
| Installed `-2`, private flag set only in the test process | FAIL at the importing process's GPU submission |
| Candidate `-3`, flag unset / set | Both PASS |
| Installed `-3`, flag unset / set | Both PASS |
| Final installed `r13063.gbe85a8d685-1`, flag unset / set | Both PASS |
| Installed package integrity | 38 files, zero altered files |
| Existing GCC kernel/initramfs hashes | Unchanged |
| Intel-only OpenCL enumeration | A770 found with both `-2` and `-3` |
| Level Zero-only SYCL enumeration | A770 found with `-3` |
| Installed ocloc with current IGC, `-device dg2` | OpenCL C smoke kernel compiled successfully |

The final runtime was also rechecked with Intel-only `clinfo`, Level Zero-only
`sycl-ls`, and `ocloc -device dg2`; all pass. Versions, library/patch hashes,
test-source hashes and results are in the
final runtime record (local-only: `/home/svnbjrn/dev/krnl/linux-tkg-7.3-rc5-xe/evidence/gaema-20260930/final-runtime-verification-inputs.json`).
Earlier `r13059...-3` IPC logs were preserved separately before rerunning.

This proves the bounded host-import regression is corrected. It does not make
bit 8 safe for device imports on this kernel, validate all fd ownership paths,
or establish a fix for the earlier BCS/event-chain failure. Keep
`UR_L0_USE_COPY_ENGINE=0` in production.

The runnable check, adapted source, exact source diff, before/after logs,
package versions, build/install logs and archive hashes are retained in
the evidence directory (local-only: `/home/svnbjrn/dev/krnl/linux-tkg-7.3-rc5-xe/evidence/gaema-20260930`).
Run against the installed runtime:

```sh
python /home/svnbjrn/dev/krnl/linux-tkg-7.3-rc5-xe/evidence/gaema-20260930/check-host-ipc.py
```

The earlier `r13059...-2` and `-3` releases remain under `/mnt/mrgr/build/pkgdest/`.
The package recipe and patches are not committed. Existing services were not
restarted; processes already holding old libraries need a normal restart to
adopt the new package.

## Separate mixed-OpenCL failure

Unfiltered `clinfo` and `sycl-ls` currently fail with an LLVM duplicate analysis
registration when the system Rusticl LLVM 23 and ROCm LLVM 23git are loaded
together. An A/B check with extracted `-2` and installed `-3` reproduces the
same failure; both Intel-only OpenCL checks pass. This is separate from the
host-import patch. No system ICD was removed or disabled.
The final `r13063...-1` unfiltered `clinfo` check still exits 1 with the same
diagnostic; its Intel-only check passes.

For an Intel Level Zero enumeration check:

```sh
ONEAPI_DEVICE_SELECTOR=level_zero:gpu sycl-ls
```

This selector is not a repair to mixed-ICD loading. The general OpenCL loader
conflict remains unresolved.

## First rejected AOCC profiles (historical)

AOCC 5.2 / AMD Clang 17.0.6 meets this rc5 source's LLVM minimum. Small Zen 4
C/assembly, objtool, BTF and linker probes passed. Rust availability passes
with `LIBCLANG_PATH=/opt/aocc/lib`. Those are compatibility checks, not
performance measurements. No conclusion about AOCC versus GCC speed is
established. See the toolchain review (local-only: `/home/svnbjrn/dev/krnl/AMD-Compiler/kernel-toolchain-review-2026-09-30.md`).

At the user's request, the first separate kernel build was attempted with real AOCC
tools, integrated assembly, no LTO, `znver4`, Rust and BTF retained, and a
distinct package identity. The trial was stopped after confirmed compiler
defects; no accepted kernel or headers archives were produced. AOCL and
ZenDNN are userspace libraries and were not linked into the trial. The
existing GCC bootable build and rollback entries are preserved.

| Attempted candidate property | Setting |
| --- | --- |
| Kernel and headers package base | `linux73-tkg-bore-rc5-xe-aocc` |
| Kernel release | `7.3.0-rc5-273-linux73-tkg-bore-rc5-xe-aocc` |
| Upstream source | `v7.3-rc5`, `72d3fcf802c45d00b300f25b848a93c3a2bd7c7e` |
| C compiler, assembler, linker | AOCC AMD Clang 17.0.6, integrated assembler, AOCC LLD |
| CPU and scheduler | Zen 4, O2, BORE, PREEMPT_DYNAMIC, HZ1000, NR_CPUS24 |
| Retained facilities | Xe/i915 support, Rust, BTF, dynamic debug and firmware logging |
| LTO | Disabled |
| Deliberate tool exception | GNU `objcopy` preserves `CONFIG_X86_X32_ABI=y` |
| Controls in the failed O2 build | `-mllvm -enable-shrink-wrap=false -mllvm -disable-branch-fold` |
| Stack/control-flow validation | `CONFIG_OBJTOOL_WERROR=y`; objtool warnings stop the build |
| Additional private Xe ABI | None |

Compiler-selected Kconfig differences are recorded in the
candidate audit (local-only: `/home/svnbjrn/dev/krnl/linux-tkg-7.3-rc5-xe-aocc/customization.cfg.audit.md`).
Clang 17 lacks some attributes available in GCC 16, so the effective
configurations are not byte-identical. The AOCC trial also used O2 after the
O3 diagnostics described below; the GCC baseline uses O3. This experiment
compares the resulting toolchain builds, including that optimization-profile
difference; it cannot isolate the compiler alone or assume identical safety
checks.
The existing GCC-only sched_ext BTF workaround remains in the source but does
not apply to AOCC. Final BTF and packaged-header acceptance were not reached
in this first trial; the accepted guide-driven build is recorded below.

The O3 trial exposed a FORTIFY diagnostic in the GUD USB-display driver.
A historical bounds-check patch (local-only: `/home/svnbjrn/dev/krnl/linux-tkg-7.3-rc5-xe-aocc/evidence/aocc/historical-patches/0002-gud-explicit-tv-mode-buffer-bound.mypatch`)
rejects a returned length larger than its 256-byte TV-mode buffer before
deriving the scan count. USB core already promises the requested-length
bound; no reachable USB overflow was demonstrated. The original translation
unit fails AOCC O3 and passes AOCC O2; the guarded version passes AOCC O3
with FORTIFY retained. Clang 23 still diagnoses the guarded control using
this Clang-17-generated configuration, so this is not a universal LLVM fix.
The runnable compile comparison (local-only: `/home/svnbjrn/dev/krnl/linux-tkg-7.3-rc5-xe-aocc/evidence/aocc/reproduce-gud-fortify.py`)
and logs preserve that distinction. The original GUD translation unit also
passes at O2 with both attempted AOCC controls. The bounds-check patch was
therefore removed from the active source and patch set; the candidate retains
the original GUD code. No GUD hardware test was performed.

Global linking then exposed a separate, substantive AOCC miscompile in
memory-cgroup static-key paths. The original object branches to register
pops without first executing their pushes. Disabling shrink-wrapping alone
removes that stack corruption but leaves ordinary jumps bypassing the bodies
of four functions. Independent inspection of machine code and `__jump_table`
relocations confirmed functional defects; these were not harmless unwind
warnings. O2 and removing the framework's `-enable-pipeliner` did not fix them.

Both compiler controls in the table initially repaired the full
`memcontrol.c` reproduction. Together they pass the kernel's actual objtool
checks with `--werror`; independent review confirmed balanced saves/restores
and correct body reachability for both static-key states in all four affected
functions. A small synthetic static-key test did not reproduce the original
bug. A Clang 23 control using Clang-17-generated configuration crashed, so it
does not establish an upstream LLVM defect. No affected kernel was installed
or booted.

At the user's suggestion, the exact reproduction was also run after
`source /opt/aocc/setenv_AOCC.sh` in a clean Bash subshell. That script sets
executable, library and include paths; it adds no optimization flags and
does not set `LIBCLANG_PATH`. All 101 resolved compiler-library paths were
identical after canonicalization. The original command still reports both
stack-frame errors under the vendor environment, while the command with
both corrective flags passes objtool cleanly. Both object pairs are also
byte-for-byte identical with and without the vendor script. The
vendor-environment control (local-only: `/home/svnbjrn/dev/krnl/linux-tkg-7.3-rc5-xe-aocc/evidence/aocc/vendor-environment-control.json`)
records the exact commands and relevant environment without dumping unrelated
session variables.

The attempted workaround was applied through the native
`scripts/Makefile.clang`. Objtool warnings were promoted to errors with
`CONFIG_OBJTOOL_WERROR=y`. Those settings did not establish whole-kernel
correctness and are not an approved workaround for reuse.
The O3 rebuild with both safeguards then exposed unreachable duplicate WARN
blocks in `kvm_apic_ack_interrupt()` and `mrp_attr_event()`. Independent review
found the live checks and functional paths intact, unlike the earlier cgroup
miscompile. Tail-duplication controls did not clean up both objects. O2 with
both safeguards passes all three isolated translation units with strict
objtool checks, so the full trial was rebuilt at O2. No warning
allowance or reachability suppression was introduced.
The full O2 rebuild then reported duplicate WARN blocks in Intel KVM and
exposed a separate live-path defect in `ept_save_pdptrs()`. Its EVMCS static
branch skips the only assignment of the `vcpu` pointer to RBX and subsequently
stores through RBX. Independent assembly and jump-table review confirmed this
in the actual build object; configured Hyper-V support can enable that path.
The normal path is intact. This is not a claim that the A770 or native AMD KVM
path has failed, but it prevents accepting the compiler profile as correct.

A narrower tail-merging control still fails the cgroup checks. It makes the
VMX object pass objtool while retaining the invalid RBX path, demonstrating
why clean stack/unwind diagnostics alone are insufficient. The full build was
stopped. No kernel from that failed profile was installed or booted. See the
candidate audit (local-only: `/home/svnbjrn/dev/krnl/linux-tkg-7.3-rc5-xe-aocc/customization.cfg.audit.md`)
for commands, diagnostics and assembly evidence.

The compiler pass trace (local-only: `/home/svnbjrn/dev/krnl/linux-tkg-7.3-rc5-xe-aocc/evidence/aocc/vmx-ept-register-pass-excerpts.txt`)
locates the bad register copy in greedy register allocation, before AOCC's
later register-allocation fixup pass. The copy is inserted after
`INLINEASM_BR`, while its taken target already requires that register.
Changing only the later fixup pass would not address where the defect begins.

As a diagnostic control, O2 with `-mllvm -regalloc=basic` and ordinary shrink
wrapping/branch folding passes all six affected translation units. Independent
assembly review confirms that the EPT pointer assignment now precedes both
static-key paths, and the four earlier cgroup paths have reachable bodies and
balanced stack handling. The
control results (local-only: `/home/svnbjrn/dev/krnl/linux-tkg-7.3-rc5-xe-aocc/evidence/aocc/basic-regalloc-o2-results.json`)
are retained. LLVM describes basic allocation as a bug-triage and performance
baseline option, explicitly not a production register allocator. It was not
promoted into the native recipe or used to justify installation.
[LLVM 17 register-allocation documentation](https://releases.llvm.org/17.0.1/docs/CodeGenerator.html#built-in-register-allocators).

This rejects the tested AOCC kernel profiles on correctness evidence. It says
nothing about their speed, does not establish an AOCC-only versus upstream
LLVM regression, and does not imply that all AOCC userspace builds are broken.

The kernel does not authenticate DG2 HuC through xe or change the BCS event
path merely by changing compilers. Its retained Xe facilities are described
in the [GCC rc5 build report](kernel-7.3-rc5-xe-build-2026-09-30.md).
Candidate-specific DKMS sources/settings were prepared for ZFS, scap and
v4l2loopback, but compilation was deliberately not started at this stage after
the kernel failed validation. The private workspace was preserved and global
DKMS sources/settings were unchanged at this boundary. Later results are in the
external-module report (local-only: `/home/svnbjrn/dev/krnl/aocc-dkms-validation-2026-09-30.md`).

The disposable VM check now also activates the cgroup2 memory controller,
places a child process in a group with a 64 MiB limit, writes 8 MiB into tmpfs,
and checks the resulting memory charge and zero OOM kills. It then reduces the
limit to 4 MiB, enables group OOM killing, and requires an oversized writer to
exit with status 137 and increment the group OOM-kill counter. A fresh control
using the existing GCC rc5 kernel passed both checks, sched_ext initialization,
and the ZFS import, read/write, zero-error scrub and export checks. The
[control log](/mnt/ssd2/build/linux-tkg-7.3-rc5-xe-aocc-qemu/evidence/gcc-control-oom-qemu.log)
records the expected bounded OOM event. Only a disposable disk-image copy was
attached; the original fixture hash is unchanged. This validates the test
harness and GCC control, not an AOCC kernel or every memory-cgroup path.

The first-trial closeout checks matched all 10 existing boot files, 111,008 files in existing
module trees, and 29 recorded configuration/firmware files against their
pre-trial hashes. DKMS registrations, ZFSBootMenu properties, EFI entries and
the running command line are unchanged. `zroot` remains ONLINE with zero
reported errors. No AOCC package, preset or native DKMS override was installed
at that boundary.
The [rollback evidence](/mnt/ssd2/build/linux-tkg-7.3-rc5-xe-aocc-qemu/evidence)
retains manifests and final comparison results.

The first trial therefore could not supply a GCC/AOCC performance comparison.
Use the same model, llama.cpp binary, Intel libraries, firmware, CPU placement
and power settings, with `UR_L0_USE_COPY_ENGINE=0` and identical explicit MoE
placement. Repeated warm runs are needed before claiming a difference in
prefill, decode or desktop responsiveness.

## Follow-up from the supplied AOCC guides

The user's three guides prompted two new controls. Adding guide 1's explicit
`-fno-PIC` to the recorded failing commands produced byte-identical memcontrol
and VMX objects: the existing `-fno-PIE` already selected static relocation.
This did not repair either failure.

Its generic CPU starting point did produce correct code in the reviewed
paths. At O2 with `-march=x86-64 -mtune=znver4`, all six affected units pass
strict objtool. Independent assembly review confirms correct static-key
paths in the four memcontrol functions, preservation of the EPT destination
pointer across both VMX paths, and retention of the original live WARN
branches without orphan duplicates. Normal greedy allocation, shrink
wrapping and branch folding are enabled; the earlier special pass-disabling
flags are absent.

The inverse control, `-march=znver4 -mtune=generic`, restores the memcontrol
failures and still miscompiles EPT even though that VMX object passes
objtool. The evidence supports retaining Zen 4 tuning while using x86-64
instruction selection for the new AOCC build. It does not establish a
performance advantage or identify the individual ISA feature involved.
The guide assessment (local-only: `/home/svnbjrn/dev/krnl/AMD-Compiler/kernel-toolchain-review-2026-09-30.md`)
links the exact commands, object hashes, results and corrections to the
other guides' build advice. The persistent recipe now applies x86-64
instruction selection and Zen 4 tuning to C/preprocessing. Rust keeps its
existing Zen 4 target through its separate LLVM 22 backend. Patch 0003's
compiler-pass overrides are archived and inactive; no global PIC override
was added. The actual native memcontrol compile passed before the full
rebuild started at 19:50 UTC. The full build, package validation, external-module
builds and disposable QEMU tests have now passed.

A comparison with the installed GCC kernel would initially compare whole
builds: that kernel uses O3 and the znver4 instruction profile. Attributing
any difference specifically to AOCC would require matching those settings
in a separate compiler comparison. No performance result is claimed here.

The first complete compile/link produced six section-lifetime warnings in
Xen and ACPI HEST. An independent source/ELF review identified three local
specialization clones whose only callers run before init memory is freed.
Their addresses do not escape, and the ordinary Xen walker retains its
runtime indirect callbacks. Modpost hides the clones' numeric suffixes in
the warnings. The [saved review](/mnt/nvme1/build/linux-tkg-7.3-rc5-xe-aocc/probes/guide-init-sections/init-section-review.md)
records every relevant incoming reference and the init lifetime chains.
No after-init path was identified for these exact references; the warnings
remain visible and no compiler or warning-suppression flag was added.
Xen PV and populated HEST runtime scenarios remain untested.

The command audit also caught a linked Rust C-helper object left over from
the earlier Zen 4 ISA attempt. That object and its command record were
preserved, then rebuilt with the intended flags; the kernel and all 6,445
modules were relinked. The subsequent audit passes 25,485 target Clang
command records. Three LZ4 units deliberately use O3 through their unchanged
kernel Makefile; those exact source-mandated exceptions are recorded rather
than treating the whole kernel as uniformly O2.

### External-module annotations

ZFS normally removes objtool's fatal-warning option. This candidate uses its
supported `--enable-objtool-werror` setting. Strict checking exposed eight
SPL errors: `spl_panic()` never returns, but an old compatibility workaround
withheld that annotation from ordinary kernel builds. Restoring the truthful
declaration fixes SPL. Its 308 instruction bytes remain identical before and
after the declaration change; only callers learn its existing terminal
behavior.

ZFS calls `spl_panic` across the module boundary, so objtool also requires
`NORETURN(spl_panic)` in its known external-callee list. This is the mechanism
documented in the pinned kernel's objtool manual. The candidate tool adds
exactly that entry, and the headers package now contains the rebuilt tool.
The kernel image, kernel package and in-tree modules are byte-identical.
The prior headers archive remains under the build's `artifacts/history/`.

The stricter analysis then exposed Lua's shared `l_noret` macro being disabled
for kernel builds. The new ZFS source restores it for Linux and annotates
three terminal public declarations: `lua_error`, `luaL_error` and
`luaL_argerror`. Their `int` signatures are preserved. Protected calls and
normal recovery APIs remain unchanged. No caller-by-caller edits or Lua
function names were added to the kernel tool's list.

The [SPL lifetime/assembly review](/mnt/nvme1/build/linux-tkg-7.3-rc5-xe-aocc/probes/spl-noreturn-review/spl-noreturn-review.md)
and [Lua control-flow review](/mnt/nvme1/build/linux-tkg-7.3-rc5-xe-aocc/probes/spl-noreturn-review/lua-noreturn-review.md)
verify the function contracts. With the corrected source, strict SPL passes
with either tool; strict ZFS produces 315 external-callee errors with the old
tool and zero with the revised packaged tool. All original checks and
`--werror` remain enabled. The module validation record (local-only: `/home/svnbjrn/dev/krnl/aocc-dkms-validation-2026-09-30.md`)
records the final clean module rebuild. ZFS passed 295 C command checks,
scap 28 (including compatibility probes), and v4l2loopback three. Compiler
and objtool diagnostics are zero. The existing experimental-Linux configure
notice and scap's missing `MODULE_DESCRIPTION` modpost notice remain visible.

Five syntactic returns in the three terminal Lua API implementations were
removed or changed to equivalent terminal calls. This resolves the compiler's
invalid-noreturn warnings without changing the complete instruction bytes or
relocation-aware disassembly of `lapi.o` or `lauxlib.o`. The exact-kernel ZFS
source and binary archives preserve these changes for installation and rebuilds.

The pinned ZFS Kbuild intentionally uses O3 for 24 bundled-zstd compilation
units. Those exact source-mandated exceptions are recorded and retain the
x86-64 ISA and Zen 4 tuning. Other audited module commands use O2. This led
to an additional VM check of actual compressed data and decompression.

The VM harness now additionally runs bounded, read-only ZFS channel programs:
an explicit Lua error, an argument-type error exercising the public error
chain, and then a successful normal return. Both the updated GCC control and
the final AOCC kernel pass those recovery checks and zero-error scrub/export.

### Accepted build and VM validation

| Property | Final accepted value |
| --- | --- |
| Release | `7.3.0-rc5-273-linux73-tkg-bore-rc5-xe-aocc` |
| Packages | `linux73-tkg-bore-rc5-xe-aocc` and `linux73-tkg-bore-rc5-xe-aocc-headers`, `7.3.rc5-273` |
| Source | `v7.3-rc5`, `72d3fcf802c45d00b300f25b848a93c3a2bd7c7e` |
| C tools and flags | AOCC 5.2 / AMD Clang 17.0.6, matching LLD and integrated assembler; `-O2 -march=x86-64 -mtune=znver4`, with the enumerated source-required O3 exceptions |
| Rust | Retained, Rust 1.98.1 / LLVM 22.1.8; `-Ctarget-cpu=znver4 -Ztune-cpu=znver4` |
| Retained configuration | BORE, PREEMPT_DYNAMIC, HZ1000, 24 CPUs, BTF, IA32/x32, Xe/i915, dynamic debug and firmware logging |
| Compiler passes / LTO | Normal production allocator, shrink wrapping and branch folding; LTO disabled |
| ZFS source identity | `2.4.99.r0.ge8a0a6cd4.73rc5xeaocc`, pinned compatibility source plus the reviewed SPL/Lua annotations |
| Other external modules | scap 9.1.0 and v4l2loopback 0.15.4 |
| Kernel archive SHA-256 | `2a17af9955eba48f8c7ab1b4b19ce8824b2c1297ac7d6436fbb941fabbc4a546` |
| Headers archive SHA-256 | `4861585c2c329c648c420fc7b73bf3e56f8bfcc51c5b3b08c7a15cefa232882f` |

The [final AOCC VM log](/mnt/ssd2/build/linux-tkg-7.3-rc5-xe-aocc-qemu/evidence/aocc-final-qemu.log)
records the exact release/compiler and passes sched_ext initialization,
memory-cgroup charging, bounded group OOM, ZFS import/mount/write/readback,
Lua error recovery, zstd compression/readback, zero-error scrub and export.
The [matching GCC control](/mnt/ssd2/build/linux-tkg-7.3-rc5-xe-aocc-qemu/evidence/gcc-control-final-qemu.log)
passes the same tests. The expected OOM trace is part of the test; no unexpected
BUG, Oops, panic, invalid opcode or general-protection fault was found.

The zstd check writes about 1.9 MB of nonzero patterned data to a dataset with
metadata-only caching, syncs it, drops the guest caches and reads it back.
Both builds reproduce SHA-256
`00d469b4a001a9a4e0eac0b1bb4a0ea923162b05465af2608fdf6b9de3e77108`,
with 253,952 bytes used versus 1,909,760 logical bytes. Only the disposable
guest pool copy received the zstd feature flag. No host pool feature was changed.
The VM has no GPU passthrough, so these checks do not exercise the A770.

### Native installation and rollback

Completed at 21:57 UTC on 2026-09-30. Both new packages pass `pacman -Qkk`
with zero altered files. Native ZFS/SPL, scap and v4l2loopback binaries match
the validated archives by SHA-256, vermagic and source version. Existing DKMS
registrations are preserved, with exactly three installed entries added for
the new release. The separate ZFS source embeds the compiler and strict-check
settings; scap/v4l2loopback use candidate-only overrides. Their existing source
files are unchanged and reproduce the validated sources.

The production image contains 32 modules and 743 firmware files matching their
installed payloads, including Xe, i915, amdgpu, NVMe, ZFS and SPL. Its host ID,
pool cache, normal ZFS hook and `tr` match the host. The disposable VM's
poweroff/test hook is absent. Only the new image was generated, with mkinitcpio
post hooks disabled. Temporary pacman hook overrides prevented broad TKG
cleanup, DKMS autoinstall and regeneration of older boot images.

| Installed boot file | SHA-256 |
| --- | --- |
| `/boot/vmlinuz-linux73-tkg-bore-rc5-xe-aocc` | `8112423661dec4a85f0d2ead06a831389b4e786a0336061d1be76e077e8cd5e7` |
| `/boot/initramfs-linux73-tkg-bore-rc5-xe-aocc.img` | `910ffa6e4f6c4da3d1a71c8b8bd542b99480c47bbea8a4de8223ee296b51fb59` |

The [final installed-state check](/mnt/ssd2/build/linux-tkg-7.3-rc5-xe-aocc-qemu/evidence/aocc-final-install-verification.json)
matches all 10 prior boot files, 111,008 prior module-tree files, 29 recorded
configuration/firmware files and the original VM disk fixture against their
saved hashes. ZFSBootMenu properties, EFI entries, the running command line and
Intel package versions are unchanged; `zroot` is healthy. Package installation
retained the existing ldconfig notices and experimental-ZFS warnings. No host
pool feature, firmware payload, boot default or running kernel was changed.

At a convenient reboot, select `vmlinuz-linux73-tkg-bore-rc5-xe-aocc` in
ZFSBootMenu. Confirm `uname -r` reports
`7.3.0-rc5-273-linux73-tkg-bore-rc5-xe-aocc` before running the comparison below.
If it fails to boot or behaves poorly, select the preserved GCC rc5 entry
`vmlinuz-linux73-tkg-bore-rc5-xe`, or the existing rc1 default. This is a
controlled experimental boot, not a claim of proven native hardware reliability.
No automatic promotion or reboot was performed.

## Benchmarking without the private binding flag

The missing private bit does not make ordinary A770 workloads fail. The
installed `030` runtime patch returns zero for that flag unless
`NEO_XE_VM_BIND_NO_CPU_ACCESS` is explicitly enabled. Normal compute and
supported imports therefore continue to use the upstream binding ABI.
The bounded native IPC test passed with it unset. Enabling it can still
make device imports issue an unsupported bit; the `040` fix excludes host
IPC allocations only. Neither this kernel nor the GCC baseline can test
Gaema's private P2P binding path with that bit enabled.

For the comparison after a trial boot, use the same installed
`llama-bench` on each kernel. The command below can also capture the existing
GCC baseline. Its binary and model exist locally; the options were checked
against the installed binary's help. This adapts the recorded Ornith command
by making copy-engine routing and production MoE placement explicit. It has
not been run as an AOCC performance test. Run without a competing kernel build
or inference workload, and with otherwise similar host load:

```sh
bench_dir="$PWD/xe-kernel-$(uname -r)-$(date +%Y%m%d-%H%M%S)"
mkdir -- "$bench_dir"
uname -a > "$bench_dir/kernel.txt"
pacman -Q intel-compute-runtime-git intel-gmmlib-git \
  intel-graphics-compiler level-zero-loader-git > "$bench_dir/packages.txt"
sha256sum /usr/bin/llama-bench > "$bench_dir/binary.sha256"
timeout -k 10s 900s env -u NEO_XE_VM_BIND_NO_CPU_ACCESS \
  UR_L0_USE_COPY_ENGINE=0 ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  GGML_SYCL_ENABLE_GRAPH=1 \
  /usr/bin/llama-bench \
  -m /mnt/ssd2/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf \
  -fitt 1024 -fitc 32768 --moe-cache off -ctk q8_0 -ctv q8_0 -fa on \
  -p 512 -n 64 -d 0,8192 -r 5 -t 12 -o json \
  > "$bench_dir/results.json" 2> "$bench_dir/stderr.log"
bench_status=$?
printf '%s\n' "$bench_status" > "$bench_dir/exit-status.txt"
```

Compare matching prefill/decode/depth rows only when both commands exit zero
and neither run reports GPU resets or invalid output. Preserve the same GPU
placement reported in stderr; identical command lines alone do not prove
identical automatic fitting. Repeat across boots before attributing small
differences to the compiler. Changing the copy-engine setting or enabling a
private ABI belongs in a separate experiment, since either would change more
than the kernel compiler.

## Not claimed

No native AOCC boot, A770 workload test on that kernel or performance speedup
was established. The accepted AOCC kernel passes the recorded VM checks;
that does not prove arbitrary compiler output correct or every kernel path safe.
Actual Xen PV and populated ACPI HEST firmware paths remain untested, and the
ZFS/Linux combination still uses explicit experimental-kernel support.
No full OpenCL conformance suite, mutable-dispatch suite,
long-context BCS workload or cross-device P2P test was run. The runtime unit
assertion was added but its unit suite was not compiled. Native IPC tests used
the real A770 with bounded small allocations; they are not long-duration
reliability or performance tests. The mixed-OpenCL LLVM conflict remains.
