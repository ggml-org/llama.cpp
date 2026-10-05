# XMX gather GEMMs vs DG2 AOT builds - 2026-09-28

Why `GGML_SYCL_XMX_GATHER` exists, what it does, and the evidence behind it. The user-facing
description is in `docs/backend/SYCL.md`, "XMX gather GEMMs and DG2 AOT builds". This note
records the investigation and its 2026-10-05 review correction.

Evidence labels: **measured** means tool output on this host; **source** means read from code;
**reported** means a packager or reviewer run, attributed below and not repeated here.
Historical measurements retain their original date and toolchain; they are not fresh
validation of the revised build policy.

## The problem

- Fork PR #67 (`e985ccb9e`, port of upstream #29245) added `ggml/src/ggml-sycl/fused-gemm.cpp`.
  It holds grouped `MUL_MAT_ID` and plain `MUL_MAT` dequant-in-GEMM kernels for nine IQ formats.
  They use a sub-group-16 8x16x16 fp16/fp16/fp32 `joint_matrix` shape (source).
- The A770 (`acm-g10`) reports no such matrix combination, so
  `ggml_sycl_fused_dequant_gemm_f16_device_ok()` returns false there and the kernels never run
  (measured earlier on this host, when PR #67's verification claim was corrected).
- JIT builds compile a kernel for the device only on first launch, so the kernels do no harm.
  AOT builds (`GGML_SYCL_DEVICE_ARCH=acm-g10`) compile every kernel at link time, and IGC crashes
  on them. The Arch packager hit this on the `llama.cpp-sycl-f16-git` build of `4e7400c3a`
  (reported: IGC 2.41.5, all 18 IQ instantiations, both the ALHP and the generic IGC build).
  The packager worked around it with a local patch that stubs the two entry points under
  `GGML_SYCL_NO_XMX_GATHER`, plus `-DCMAKE_CXX_FLAGS=-DGGML_SYCL_NO_XMX_GATHER` (packaging repo
  commit `28e9839`, not pushed).

## Reproduction (measured, this host)

IGC `intel-graphics-compiler 1:2.41.5-1.1`, compute-runtime 26.35.39758, oneAPI 2026.1. The
`fused-gemm.cpp` compile command was taken from `compile_commands.json` of an
`-DGGML_SYCL_DEVICE_ARCH=acm-g10` configure. The file was compiled on its own, then linked into
a shared object with `-fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen
"-device acm-g10" -shared`; that device-link step runs ocloc on the file's kernels.

| variant | compile | AOT device link |
|---|---|---|
| without `-DGGML_SYCL_NO_XMX_GATHER` | rc 0 | **rc 1**: `[acm-g10] IGC: Internal Compiler Error: Floating point exception`, `gen compiler command failed with exit code 245` |
| with `-DGGML_SYCL_NO_XMX_GATHER` | rc 0 | rc 0 |

## Original implementation and evidence (2026-09-28)

The first revision used an ON/OFF option and a DG2-name deny-list. It compiled out
all gather kernels for the entire target list if any entry matched `acm`, `dg2`,
`xe-hpg`, or IP `12.55`-`12.57`. That implementation was incomplete: unsupported
non-DG2 targets and alternate spellings escaped it, explicit ON was overridden,
and mixed-target packages lost kernels even on supported devices.

Exactly seven configure-only cases were recorded. "Kernels included" below means
the `fused-gemm.cpp` compile command lacked `GGML_SYCL_NO_XMX_GATHER`; it does not
mean that an AOT device link or a kernel execution passed.

| `GGML_SYCL_DEVICE_ARCH` | original configure result |
|---|---|
| (empty, JIT) | kernels included |
| `acm-g10` | compiled out, message printed |
| `bmg-g21` | kernels included |
| `acm-g10,bmg-g21` | compiled out, message printed |
| `12.55.8` | compiled out, message printed |
| `ACM-G11` | compiled out, message printed |
| `xe2-hpg` | kernels included |

The original commit message and PR body claimed ten combinations. This artifact
supports only these seven; the additional `dg2` and two explicit-OFF cases were
not recorded and are not treated as measured evidence. A default JIT build of
`ggml-sycl`, `llama-server`, and `test-sycl-turbo-correctness` was reported to
complete with rc 0. Its private directory name did not establish a reproducible
source identity, so that build is not validation of the current revision.

## Review evidence (reported, 2026-10-02)

The reviewer linked unstubbed `fused-gemm.cpp` from PR #76 head `40c9b932e593`
using the single-file method above. No crash or miscompile was reported in the
stub implementation; the problem was target selection.

| AOT targets | reported unstubbed device-link result |
|---|---|
| `mtl-h`, `arl-h` | rc 1; 18 IGC floating-point-exception ICEs |
| `tgl`, `adl-p` | rc 10; cooperative-matrix extension disabled, 18 diagnostics |
| `ats-m150`, `ats-m75` | rc 1; 18 ICEs |
| `xe_hpg`, `0x56A0`, `dg1:acm-g10`, `xe` | rc 1 or rc 10; individual codes not supplied |
| `acm-g11`, `acm-g12` | rc 1; 18 ICEs |
| `bmg-g21`, `xe2-hpg`, `pvc`, `lnl-m` | rc 0 |

The stubbed variant linked with rc 0 for `acm-g10`, `mtl-h`, and `tgl`.
The reviewer also queried `ocloc ids`: `ats-m150` is IP 12.55.8 (like `acm-g10`),
and `ats-m75` is IP 12.56.5 (like `acm-g11`). Leaving those aliases out was a
policy error, not evidence that they were safe. The earlier "acm-g11/acm-g12 not
compiled" statement applies only to the original investigation; the reviewer
subsequently linked them and observed failures.

A reported live A770 matrix-capability query lacked the required SG16 8x16x16
fp16 shape. This corrects the older
[performance note](sycl/ornith-a770-perf-research-2026-09-27.md): its passing
operator probes and dense 0.99x comparison exercised library fallback, not the
fused XMX gather implementation. They establish neither fused-kernel correctness
nor fused-kernel speed on the A770.

## Revised policy (2026-10-05)

`GGML_SYCL_XMX_GATHER` is AUTO/ON/OFF. AUTO uses a conservative AOT allow-list;
ON is an explicit opt-in and is not silently changed to OFF. OFF compiles out
the gather kernels. The canonical policy, separate-build inspection paths, and
provenance fields are documented in
[the SYCL guide](../backend/SYCL.md#xmx-gather-gemms-and-dg2-aot-builds).

This changes which device images are emitted. A successful AOT link alone does
not establish correct execution, and the runtime matrix-shape gate remains
necessary. No SG8 GEMM implementation is introduced.

## Follow-up validation (measured, 2026-10-05)

The review worktree was rebased onto fork master before these checks. These
results apply to the revised policy and kernel guards, not the original deny-list.
Toolchain: icpx 2026.1.1 (20260724), IGC package `1:2.41.10-1`,
compute-runtime-git `22.43.24558.r13063.gbe85a8d685-1`, and
level-zero-loader-git `1.34.0.r2.gae3db48-1`. The GPU check ran on the A770 with
**xe**, without timing measurements.

Thirteen actual CMake configurations checked both `build-metadata.json` fields
and the `fused-gemm.cpp` command's compile-out macro. Each row passed:

| requested | device arch | effective gather targets |
|---|---|---|
| AUTO | empty (JIT) | JIT |
| AUTO | `acm-g10` | OFF |
| AUTO | `mtl-h` | OFF |
| AUTO | `tgl` | OFF |
| AUTO | `ats-m150` | OFF |
| AUTO | `xe_hpg` | OFF |
| AUTO | `0x56A0` | OFF |
| AUTO | `dg1:acm-g10` | OFF |
| AUTO | `bmg-g21` | `bmg-g21` |
| AUTO | `acm-g10,bmg-g21` | `bmg-g21` |
| ON | `acm-g10` | `acm-g10` |
| OFF | empty (JIT) | OFF |
| OFF | `bmg-g21` | OFF |

The explicit ON row checks selection only; it does not claim that unsupported
`acm-g10` gather kernels now link. The persistent CMake policy test also checks
normalization, unknown targets, mixed lists, and invalid modes.

The enabled JIT `fused-gemm.cpp` object compiled. The compile-out test compiled,
linked, and called the three public fallback entry points, all returning false.
The policy and fallback CTests passed 2/2:

```bash
cmake --build BUILD --target test-sycl-xmx-gather-stubs
ctest --test-dir BUILD -R '^test-sycl-xmx-gather(-stubs)?$' --output-on-failure
```

Fresh partial AOT device links of the enabled gather object succeeded for all
four admitted target names: `bmg-g21`, `xe2-hpg`, `pvc`, and `lnl-m` (each rc 0).
These used `icpx -fsycl -fsycl-targets=spir64_gen -fsycl-device-code-split=per_kernel
-Xsycl-target-backend=spir64_gen "-device TARGET" -fsycl-link OBJECT -o OUTPUT`.

The mixed `acm-g10,bmg-g21` configure generated a gather image restricted to
`bmg-g21`. Linking its extracted host object and prelinked device object with
the full backend target list succeeded. Passing the device object directly to
the host linker preserves that restriction instead of recompiling its kernels
for every backend target. A standalone registration probe found nine named
gather kernels. It created no device queue and ran no GPU compute; registration
is evidence of retained images, not execution on Battlemage.

Stubbed partial AOT links also succeeded freshly for `acm-g10`, `mtl-h`, and
`tgl` (each rc 0). These are single-file device links, not full backend AOT builds.
The durable mixed-build CTest suite passed 3/3 for both shared and ELF static
configurations. The static test links through an archive member referenced by
the public capability function and verifies that all nine named kernels remain
registered. Additional configures passed for separator-only JIT input,
normalized mixed target names, and migration of a BOOL ON cache to STRING ON;
an attempted nested policy override was rejected with the outer-option diagnostic.
In addition,
outer and nested separate-build configures for AUTO `acm-g10` both reported OFF;
the outer metadata recorded OFF and the nested `fused-gemm.cpp` compile command
contained `GGML_SYCL_NO_XMX_GATHER`.

A full JIT build of `ggml-sycl` and `test-sycl-turbo-correctness` passed. The
A770 default correctness sweep ran with graph replay disabled under a 180-second
timeout and ended with zero `GATE-FAIL` and zero `XPASS`. This verifies the default
fallback behavior on that device, not execution of its unsupported gather kernels.

## Not claimed

- The reviewer target table is reported evidence, not independently repeated
  measurements from this documentation update. Compiler/IGC changes can change
  link results; aliases and targets outside the table were not tested there.
- No model, acceptance, throughput, or fused-kernel correctness result follows
  from configure-only or device-link checks. The default A770 correctness run
  exercises fallback, not the gather implementation.
- Historical JIT builds are not validation of the revised CMake policy.
- No full backend AOT link, Windows build, or supported-target GPU execution
  was performed in this follow-up. Windows static mixed AUTO builds are
  explicitly rejected; that is a documented build limitation, not verification
  of a Windows-specific image-link implementation.
- A full original-branch AOT build was not recorded. The packager reported a full
  `acm-g10` AOT build of `4e7400c3a` with its equivalent stub patch, a passing
  correctness gate, and coherent Ornith output; those results belong to that
  artifact, not every later revision.
