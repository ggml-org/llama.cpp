## Introduction

This wiki is intended to track the support for various hardware types within the BLIS framework source distribution.

We apologize if this wiki falls out of date. For the latest support, we recommend peeking inside of the relevant sub-configuration (specifically, in the `bli_cntx_init_<configname>.c` file) and looking at which kernels are registered. You may also contact the [blis-devel](http://groups.google.com/group/blis-devel) mailing list.


## Level-3 microkernels

The following table lists architectures for which there exist optimized level-3 microkernels, which microkernels are optimized, the name of the author or maintainer, and the current status of the microkernels.

A few remarks / reminders:
  * Optimizing only the [gemm microkernel](KernelsHowTo.md#gemm-microkernel) will result in optimal performance for all [level-3 operations](BLISTypedAPI#level-3-operations) except `trsm` (which will typically achieve 60 - 80% of attainable peak performance).
  * The [trsm](BLISTypedAPI#trsm) operation needs the [gemmtrsm microkernel(s)](KernelsHowTo.md#gemmtrsm-microkernels), in addition to the aforementioned [gemm microkernel](KernelsHowTo.md#gemm-microkernel), in order reach optimal performance.
  * Induced complex (1m) implementations are employed in all situations where the real domain [gemm microkernel](KernelsHowTo.md#gemm-microkernel) of the corresponding precision is available, but the "native" complex domain gemm microkernel is unavailable. Note that the table below lists native kernels, so if a microarchitecture lists only `sd`, support for both `c` and `z` datatypes will be provided via the 1m method. (Note: most people cannot tell the difference between native and 1m-based performance.) Please see our [ACM TOMS article on the 1m method](https://github.com/flame/blis#citations) for more info on this topic.
  * Some microarchitectures use the same sub-configuration. *This is not a typo.* For example, Haswell and Broadwell systems as well as "desktop" (non-server) versions of Skylake, Kaby Lake, and Coffee Lake all use the `haswell` sub-configuration and the kernels registered therein. Microkernels can be recycled in this manner because the key detail that determines level-3 performance outcomes is actually the vector ISA, not the microarchitecture. In the previous example, all of the microarchitectures listed support AVX2 (but not AVX-512), and therefore they can reuse the same microkernels.
  * Remember that you (usually) don't have to choose your sub-configuration manually! Instead, you can always request configure-time hardware detection via `./configure auto`. This will defer to internal logic (based on CPUID for x86_64 systems) that will attempt to choose the appropriate sub-configuration automatically.
  * There is a difficulty in automatically choosing the ideal sub-configuration for use on Skylake-X systems, which may have one or two FMA units. The `skx` sub-configuration is only beneficial when used on hardware with two FMA units. Otherwise the hardware is treated as a "desktop" Skylake system, which uses the `haswell` sub-configuration. Furthermore, the number of units can't be queried directly; instead, we rely on a manually-maintained list of CPU models (via logic in `frame/base/bli_cpuid.c`), which may be incorrect for new processors, particularly Gold models. In that case, you can either fix the code (and please raise an issue!) or manually target the `skx` at configure-time (i.e., `./configure [options] skx`). If your performance seems low, you can set `export BLIS_ARCH_DEBUG=1`, which will cause BLIS to output some basic debugging info to `stderr` that will reveal whether your system was detected as having one or two VPUs (FMA units).

| Vendor/Microarchitecture             | BLIS sub-configuration | `gemm` | `gemmtrsm` |
|:-------------------------------------|:-----------------------|:-------|:-----------|
| AMD Bulldozer (AVX/FMA4)             | `bulldozer`            | `sdcz` |            |
| AMD Piledriver (AVX/FMA3)            | `piledriver`           | `sdcz` |            |
| AMD Steamroller (AVX/FMA3)           | `steamroller`          | `sdcz` |            |
| AMD Excavator (AVX/FMA3)             | `excavator`            | `sdcz` |            |
| AMD Zen (AVX/FMA3)                   | `zen`                  | `sdcz` |  `sd`      |
| Intel Core2 (SSE3)                   | `penryn`               | `sd`   |   `d`      |
| Intel Sandy/Ivy Bridge (AVX/FMA3)    | `sandybridge`          | `sdcz` |            |
| Intel Haswell, Broadwell (AVX/FMA3)  | `haswell`              | `sdcz` |  `sd`      |
| Intel Sky/Kaby/CoffeeLake (AVX/FMA3) | `haswell`              | `sdcz` |  `sd`      |
| Intel Knights Landing (AVX-512/FMA3) | `knl`                  | `sd`   |            |
| Intel SkylakeX (AVX-512/2×FMA3)      | `skx`                  | `sd`   |            |
| Intel SkylakeX (AVX-512/1×FMA3)      | `haswell`              | `sdcz` |  `sd`      |
| ARMv7 Cortex-A9 (NEON)               | `cortex-a9`            | `sd`   |            |
| ARMv7 Cortex-A15 (NEON)              | `cortex-a15`           | `sd`   |            |
| ARMv8 Cortex-A53 (NEON)              | `cortex-a53`           | `sd`   |            |
| ARMv8 Cortex-A57 (NEON)              | `cortex-a57`           | `sd`   |            |
| ARMv8.1 ThunderX2 (NEON)             | `thunderx2`            | `sd`   |            |
| ARMv8.1 A64FX (SVE)                  | `a64fx`                |  `d`   |            |
| IBM Blue Gene/Q (QPX int)            | `bgq`                  |  `d`   |            |
| IBM Power7 (QPX int)                 | `power7`               |  `d`   |            |
| IBM Power9                           | `power9`               | `sdcz` |            |
| IBM Power10                          | `power10`              | `sdcz` |            |
| template (C99)                       | `template`             | `sdcz` | `sdcz`     |

## Level-1f kernels

Not yet written. Please see the relevant sub-configuration (`bli_cntx_init_<configname>.c`) to determine which kernels are implemented/registered.

## Level-1v kernels

Not yet written. Please see the relevant sub-configuration (`bli_cntx_init_<configname>.c`) to determine which kernels are implemented/registered.
