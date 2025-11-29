# Release Notes

*Note: For some releases, credit for individuals' contributions are shown in parentheses.*

## Contents

* [Changes in 3.0](ReleaseNotes.md#changes-in-30)
* [Changes in 2.0](ReleaseNotes.md#changes-in-20)
* [Changes in 1.0](ReleaseNotes.md#changes-in-10)
* [Changes in 0.9.0](ReleaseNotes.md#changes-in-090)
* [Changes in 0.8.1](ReleaseNotes.md#changes-in-081)
* [Changes in 0.8.0](ReleaseNotes.md#changes-in-080)
* [Changes in 0.7.0](ReleaseNotes.md#changes-in-070)
* [Changes in 0.6.1](ReleaseNotes.md#changes-in-061)
* [Changes in 0.6.0](ReleaseNotes.md#changes-in-060)
* [Changes in 0.5.2](ReleaseNotes.md#changes-in-052)
* [Changes in 0.5.1](ReleaseNotes.md#changes-in-051)
* [Changes in 0.5.0](ReleaseNotes.md#changes-in-050)
* [Changes in 0.4.1](ReleaseNotes.md#changes-in-041)
* [Changes in 0.4.0](ReleaseNotes.md#changes-in-040)
* [Changes in 0.3.2](ReleaseNotes.md#changes-in-032)
* [Changes in 0.3.1](ReleaseNotes.md#changes-in-031)
* [Changes in 0.3.0](ReleaseNotes.md#changes-in-030)
* [Changes in 0.2.2](ReleaseNotes.md#changes-in-022)
* [Changes in 0.2.1](ReleaseNotes.md#changes-in-021)
* [Changes in 0.2.0](ReleaseNotes.md#changes-in-020)
* [Changes in 0.1.8](ReleaseNotes.md#changes-in-018)
* [Changes in 0.1.7](ReleaseNotes.md#changes-in-017)
* [Changes in 0.1.6](ReleaseNotes.md#changes-in-016)
* [Changes in 0.1.5](ReleaseNotes.md#changes-in-015)
* [Changes in 0.1.4](ReleaseNotes.md#changes-in-014)
* [Changes in 0.1.3](ReleaseNotes.md#changes-in-013)
* [Changes in 0.1.2](ReleaseNotes.md#changes-in-012)
* [Changes in 0.1.1](ReleaseNotes.md#changes-in-011)
* [Changes in 0.1.0](ReleaseNotes.md#changes-in-010)
* [Changes in 0.0.9](ReleaseNotes.md#changes-in-009)
* [Changes in 0.0.8](ReleaseNotes.md#changes-in-008)
* [Changes in 0.0.7](ReleaseNotes.md#changes-in-007)
* [Changes in 0.0.6](ReleaseNotes.md#changes-in-006)
* [Changes in 0.0.5](ReleaseNotes.md#changes-in-005)
* [Changes in 0.0.4](ReleaseNotes.md#changes-in-004)
* [Changes in 0.0.3](ReleaseNotes.md#changes-in-003)
* [Changes in 0.0.2](ReleaseNotes.md#changes-in-002)
* [Changes in 0.0.1](ReleaseNotes.md#changes-in-001)

## Changes in 3.0:
In development

Improvements present in 3.0:

Framework:
- Fixed an issue which could cause a segfault on x86-64 with `-m32` (and potentially, on other 64-bit setups) stemming from how enum constants are passed to variadic functions. (Igor Zhuravlov)

## Changes in 2.0:
January 15, 2025

Improvements present in 2.0:

Known Issues:
- There is a performance regression in the `ztrmm` and `ztrsm` operations. On the Ampere Altra, performance is impacted by up to 30%; it is currently unknown if and how much this bug affects other architectures but the effect should be much smaller in most cases.

Framework:
- BLIS now supports "plugins", which provide additional functionality through user-defined kernels, blocksizes, and kernel preferences. Users can use an installed copy of BLIS (even a binary-only distribution) to create a plugin outside of the BLIS source tree. User-written reference kernels can then be registered into BLIS, and are compiled by the BLIS build system for all configured architecture. This also means that user-provided kernels participate in run-time kernel selection based on the actual hardware used! Additionally, users can provide and register optimized kernels for specific architectures which are automatically selected as appropriate. See `docs/PluginHowTo.md` for more information.
- A new API has been added which allows users to modify the default "control tree". This data structure defines the specific algorithmic steps used to implement a level-3 BLAS operation such as `gemm` or `syrk`. Users can start with a predefined control tree for one of the level-3 BLAS operations (except `trsm` currently) and then modify it to produce a custom operation. Users can change kernels for packing and computation, associated blocksizes, and provide additional information (such as external parameters or additional data) which is passed directly to the kernels. See `docs/PluginHowTo.md` for more information and a working example.
- All level-3 BLAS operations (except `trsm`) now support full mixed-precision mixed-domain computation. The A, B, and C matrices, as well as the alpha and beta scalars, may be provided in any of the supported data types (single/double precision and real/complex domain, currently), and an additionally-provided computational precision controls how the computation is actually performed internally. The computational precision can be set on the `obj_t` structure representing the C matrix.
- Added a `func2_t` struct for dealing with 2-type kernels (see below). A `func2_t` can be safely cast to `func_t` to refer to only kernels with equal type parameters. (Devin Matthews)
- The `bli_*_front` functions have been removed.
- Extensive other back-end changes and improvements.

Compatibility:
- Added a ScaLAPACK compatibility mode which disables some conflicting BLAS definitions. (Field Van Zee)
- Fixed issues with improperly escaped strings in python scripts for compatibility with python 3.12+. (@AngryLoki)
- Added a user-defined macro `BLIS_ENABLE_STD_COMPLEX` which uses `std::complex` typedefs in `blis.h` for C++ code.  (Devin Matthews)
- Fixed a bug in the definition of some scalar level-0 macros affecting compatibility of `bli_creal` and `bli_zreal`, for example. (Devin Matthews)
- Fixed improperly-quoted strings in Python scripts which affected compatibility with Python 3.12+. (@AngryLoki)
- The static initializer macros (`BLIS_*_INITIALIZER`) have been fixed for compatibility with C++. (Devin Matthews)
- Install "helper" `blis.h` and `cblas.h` headers directly to `INCDIR` (in addition to the full files in `INCDIR/blis`). (Field Van Zee, Jed Brown, Mo Zhou)

Kernels:
- Fixed an out-of-bounds read bug in the `haswell` `gemmsup` kernels. (John Mather)
- Fixed a bug in the complex-domain `gemm` kernels for `piledriver`. (@rmast)
- Kernel, blocksizes, and preference lookup functions now use `siz_t` rather than specific enums. (Devin Matthews)
- Fixed some issues with run-time kernel detection and add more ARM part numbers/manufacturer codes. (John Mather)
- Kernels can now be added which have two datatype parameters. Kernel IDs are assigned such that 1-type and 2-type kernels cannot be interchanged accidentally. (Devin Matthews)
- The packing microkernels and computational microkernels (`gemm` and `gemmtrsm`) now receive offsets into the global matrix. The latter are passed via the `auxinfo_t` struct. (Devin Matthews)
- The separate "MRxk" and "NRxk" packing kernels have been merged into one generic packing kernel. Packing kernels are now expected to pack any size micropanel, but may optimize for specific shapes. (Devin Matthews)
- Added explicit packing kernels for diagonal portions of matrices, and for certain mixed-domain/1m cases. (Devin Matthews)
- Improved support for duplication during packing ("broadcast-B") across all packing kernels.

Build system:
- The `cblas.h` file is now "flattened" immediately after `blis.h` is (if enabled), rather than later in the build process. (Jeff Diamond, Field Van Zee)
- Added script to help with preparing release candidate branches. (Field Van Zee)
- The configure script has been overhauled. In particular, using spaces in `CC`/`CXX` is now supported. (Devin Matthews)
- Improved support for C++ source files in BLIS or in plugins. (Devin Matthews)

Testing:
- test/3 drivers now allow using the "default" induced method, rather than forcing native or 1m operation. (Field Van Zee, Leick Robinson)
- Fix some segfaults in the test/3 drivers. (Field Van Zee, Leick Robinson)
- The testsuite now tests *all* possible type combinations when requested. (Devin Matthews)
- Improved detection of problems in `make check-blis` and related targets. (Devin Matthews)

Documentation:
- Added documentation for the new plugin system and for creating custom operations by modifying the BLIS control tree. (Devin Matthews)
- Updated documentation for downloading BLIS in `README.md` and instructions for maintainers in `RELEASING`. (Field Van Zee)

## Changes in 1.0
May 6, 2024

Improvements present in 1.0:

Framework:
- Initialize/finalize BLIS via a new `bli_pthread_switch_t` API. (Field Van Zee, Devin Matthews)
- Revamped `bli_init()` to use TLS where feasible. (Field Van Zee, Edward Smyth, Minh Quan Ho)
- Implemented support for fat multithreading.
- Implemented tile-level load balancing (tlb), or tile-level partitioning, in jr/ir loops for `gemm`, `gemmt`, and `trmm` macrokernels. (Field Van Zee, Devin Matthews, Leick Robinson, Minh Quan Ho)
- Added padding to `thrcomm_t` fields to avoid false sharing of cache lines. (Leick Robinson)
- Rewrote/fixed broken tree barrier implementation. (Leick Robinson)
- Refactored some `rntm_t` management code. (Field Van Zee, Devin Matthews)
- Initialize `rntm_t` nt/ways fields with 1 (not -1). (Field Van Zee, Jeff Diamond, Leick Robinson, Devin Matthews)
- Defined `invscalv`, `invscalm`, `invscald` operations.
- Added consistent `NaN`/`Inf` handling in `sumsqv`. (Devin Matthews)
- Implemented support for HPX as a threading backend option. (Christopher Taylor, Srinivas Yadav)
- Relocated the pba, sba pool (from the `rntm_t`), and `mem_t` (from the `cntl_t`) to the `thrinfo_t` object.
- Modified which communicator is associated with a given node of the `thrinfo_t` tree. (Devin Matthews)
- Refactored level-3 thread decorator into two parts: a thread launcher and a function to pass operands. (Devin Matthews)
- Refactored structure awareness in `bli_packm_blk_var1.c`. (Devin Matthews)
- Reimplemented `bli_l3_determine_kc()`. (Devin Matthews)
- Implemented `cntx_t` pointer caching in gks. (Field Van Zee, Harihara Sudhan S)
- Added `const` keyword to pointers in kernel APIs. (Field Van Zee, Nisanth M P)
- Migrated all kernel APIs to use `void*` pointers.
- Defined new global scalar constants: `BLIS_ONE_I`, `BLIS_MINUS_ONE_I`, `BLIS_NAN`. (Devin Matthews)
- Disabled modification of KC in the `gemmsup` kernels. (Devin Matthews)
- Defined `lt`, `lte`, `gt`, `gte` operations and other miscellaneous updates.
- Consolidated `INSERT_` macro sets via variadic macros. (Devin Matthews)
- De-templatized macrokernels for `gemmt`, `trmm`, and `trsm` to match that of `gemm`. (Devin Matthews)
- De-templatized `bli_l3_sup_var1n2m.c` and unified `_sup_packm_a/b()`. (Devin Matthews)
- Fixed 1m enablement for `herk`/`her2k`/`syrk`/`syr2k`. (Devin Matthews)
- Fixed `trmm[3]`/`trsm` performance bug introduced in `cf7d616`. (Field Van Zee, Leick Robinson)
- Fixed a 1m optimization bug in right-sided `hemm`/`symm`. (Field Van Zee, Nisanth M P)
- Fixed a bug in sup threshold registration. (Devin Matthews, Field Van Zee)
- Fixed brokenness in the small block allocator (sba) when the sba is disabled. (Field Van Zee, John Mather)
- Fixed type bug in `bli_cntx_set_ukr_prefs()`. (Field Van Zee, Leick Robinson, Devin Matthews, Jeff Diamond)
- Fixed incorrect `sizeof(type)` in edge case macros. (@moon-chilled)
- Fixed bugs and added sanity check in `bli_pool.c`. (Devin Matthews)
- Fixed a typo in the macro definition for `VEXTRACTF64X2` in `bli_x86_asm_macros.h`. (Harsh Dave)
- Fixed a typo in `bli_type_defs.h` where `BLIS_BLAS_INT_TYPE_SIZE` was misspelled. (Devin Matthews)
- Typecast `printf()` args in `bli_thread_range_tlb.c` to avoid compiler warnings. (Lee Killough)
- Minor tweaks to `bli_l3_check.c`.
- Partial addition of `const` to all interfaces above the (micro)kernels. (Devin Matthews)
- Fixed a harmless misspelling of `xpbys` in gemm macrokernel.
- Various internal API renaming/reorganization.
- Various other fixes.

Compatibility:
- Implemented `[cz]symv_()`, `[cz]syr_()`, `[cz]rot_()`. (Field Van Zee, James Foster)
- Fixed compilation errors when `BLIS_DISABLE_BLAS_DEFS` is defined. (Field Van Zee, Edward Smyth, Devin Matthews)
- Include `bli_config.h` before `bli_system.h` in `cblas.h` so that `BLIS_ENABLE_SYSTEM` is defined in time for proper OS detection. (Edward Smyth)

Kernels:
- Updated ARMv8a kernels to fix two prefetching issues and re-enable general stride IO. (Jeff Diamond)
- Restored general storage case to `armsve` kernels. (RuQing Xu)
- Added arm64 `dgemmsup` with extended MR and NR. (RuQing Xu)
- Reorganized the way `packm` kernels are stored within the `cntx_t` so that BLIS only stores two `packm` kernels per datatype: one for MRxk upanels and one for kxNR upanels. (Devin Matthews)
- Fixed bugs in `scal2v` reference kernel when alpha == 1.
- Fixed out-of-bounds read in `haswell` `gemmsup` kernels. (Daniël de Kok, Bhaskar Nallani, Madeesh Kannan)
- Fixed k = 0 edge case in `power10` microkernels. (Nisanth M P)
- Disabled `power10` kernels other than `sgemm`, `dgemm`. (Nisanth M P)
- Fixed `bli_gemm_small()` prototype mismatch. (Jeff Diamond)

Extras:
- Use the conventional level-3 sup thread decorator within the `gemmlike` sandbox.
- Fixed type-mismatch errors in `power10` sandbox. (Nisanth M P)
- Fixed `gemmlike` sandbox bug that stems from reuse of `bli_thrinfo_sup_grow()`.

Build system:
- Added two arm64 subconfigs: `altra` and `altramax`. (Jeff Diamond, Leick Robinson)
- Added support for RISC-V configuration targets. (Angelika Schwarz, Lee Killough)
- Auto-detect the RISC-V ABI of the compiler and use `-mabi=` during RISC-V builds. (Lee Killough)
- Added `sifive_x280` subconfig and kernel set. (Aaron Hutchinson, Lee Killough, Devin Matthews, and Angelika Schwarz)
- Added AddressSanitizer (--enable-asan) option to `configure`. (Devin Matthews)
- Added option to disable thread-local storage via `--disable-tls`. (Field Van Zee, Nick Knight)
- Exclude `-lrt` on Android with Bionic libraries. (Lee Killough)
- Omit `-fPIC` option when shared library build is disabled. (Field Van Zee, Nick Knight)
- Move `-fPIC` option insertion to subconfigs' `make_defs.mk` files. (Field Van Zee, Nick Knight)
- Install one-line helper headers to `INCDIR` prefix so that user can `#include "blis.h"` instead of `#include <blis/blis.h>` and/or `"cblas.h"` instead of `<blis/cblas.h>` if CBLAS is enabled). (Field Van Zee, Jed Brown, Devin Matthews, Mo Zhou)
- Enhanced detection of Fortran compiler when checking the version string for the purposes of determining a default return convention for complex domain values. (Bart Oldeman)
- Added detection of the NVIDIA nvhpc compiler (`nvc`) in `configure`. (Ajay Panyala)
- Updated `zen3` subconfig to support NVHPC compilers. (Abhishek Bagusetty)
- Use kernel CFLAGS for `kernels` subdirs in addons. (AMD, Mithun Mohan)
- Created `power` umbrella configuration family (which currently includes `power9` and `power10` subconfigs). (Nisanth M P)
- Defined `BLIS_VERSION_STRING` in `blis.h` instead of via command line argument during compilation. (Field Van Zee, Mohsen Aznaveh, Tim Davis)
- Rewrote `regen-symbols.sh` as `gen-libblis-symbols.sh`. (Field Van Zee)
- Support `clang` targetting MinGW. (Isuru Fernando)
- Added autodetection (via `/proc/cpuinfo`) for POWER7, POWER9 and POWER10 microarchitectures. (Alexander Grund)
- Added `#line` directives to flattened `blis.h` to facilitate easier debugging. (Devin Matthews)
- Added `--nosup` and `--sup` shorthand options to `configure`.
- Use here-document syntax for `configure --help` output. (Lee Killough)
- Updated `configure` to pass all `shellcheck` checks. (Lee Killough)
- Tweaks to `.dir-locals.el` to enhance emacs formatting of C files. (Lee Killough)
- Removed buggy cruft from `power10` subconfig. (Field Van Zee, Nicholai Tukanov)
- Added missing `#include <io.h>` for Windows. (@h-vetinari)
- Fixed hardware auto-detection for `firestorm` (Apple M1) subconfig. (Devin Matthews)
- Fixed bug in detection of Fortran compiler vendor. (Devin Matthews)
- Fixed version check for `znver3`, which needs gcc >= 10.3. (Jed Brown)
- Fixed typo in `configure --help` text. (Lee Killough)
- Fixed warning about regular expressions with stray backslashes as the result of recent changes to `grep`.
- Added `output.testsuite` to `.gitignore`.
- Minor changes to .gitignore and LICENSE files. (Jeff Diamond)
- Minor decluttering of top-level directory.
- Very minor tweaks to common.mk.

Testing:
- Rewrote `test/3` drivers to take parameters via command line arguments. (Field Van Zee, Jeff Diamond, Leick Robinson)
- Added `arm64` entry to `.travis.yml` so that Travis CI will compile/test ARM builds. (Field Van Zee, RuQing Xu)
- Test the `gemmlike` sandbox via AppVeyor. (Jeff Diamond)
- Added `-q` quiet mode option to testsuite.
- Fixed non-deterministic segfault in standalone `test/3` drivers. (Field Van Zee, Leick Robinson)
- Fixed a crash that occurs when either `cblat1` or `zblat1` are linked with a build of BLIS that was compiled with `--complex-return=intel`. (Bart Oldeman)
- Other minor fixes/tweaks.

Documentation:
- Added Discord documentation (`docs/Discord.md`) and logo to `README.md`.
- Added the `mm_algorithm` files (for bp and pb) to `docs/diagrams`.
- Added mention of Wilkinson Prize to `README.md`.
- Minor fixes and improvements to `docs/Multithreading.md`.
- Fix typos in docs + example code comments. (Igor Zhuravlov)
- Fixed broken "tagged releases" link in `README.md`.
- Added SMU citation to `README.md` intro.

## Changes in 0.9.0
April 1, 2022

Improvements present in 0.9.0:

Framework:
- Added various fields to `obj_t` that relate to storing function pointers to custom `packm` kernels, microkernels, etc as well as accessor functions to set and query those fields. (Devin Matthews)
- Enabled user-customized `packm` microkernels and variants via the aforementioned new `obj_t` fields. (Devin Matthews)
- Moved edge-case handling out of the macrokernel and into the `gemm` and `gemmtrsm` microkernels. This also required updating of APIs and definitions of all existing microkernels in `kernels` directory. Edge-case handling functionality is now facilitated via new preprocessor macros found in `bli_edge_case_macro_defs.h`. (Devin Matthews)
- Avoid `gemmsup` thread barriers when not packing A or B. This boosts performance for many small multithreaded problems. (Field Van Zee, AMD)
- Allow the 1m method to operate normally when single and double real-domain microkernels mix row and column I/O preference. (Field Van Zee, Devin Matthews, RuQing Xu)
- Removed support for execution of complex-domain level-3 operations via the 3m and 4m methods.
- Refactored `herk`, `her2k`, `syrk`, `syr2k` in terms of `gemmt`. (Devin Matthews)
- Defined `setijv` and `getijv` to set/get vector elements.
- Defined `eqsc`, `eqv`, and `eqm` operations to test equality between two scalars, vectors, or matrices.
- Added new bounds checking to `setijm` and `getijm` to prevent use of negative indices.
- Renamed `membrk` files/variables/functions to `pba`.
- Store error-checking level as a thread-local variable. (Devin Matthews)
- Add `err_t*` "return" parameter to `bli_malloc_*()` and friends.
- Switched internal mutexes of the `sba` and `pba` to static initialization.
- Changed return value method of `bli_pack_get_pack_a()`, `bli_pack_get_pack_b()`.
- Fixed a bug that allows `bli_init()` to be called more than once (without segfaulting). (@lschork2, Minh Quan Ho, Devin Matthews)
- Removed a sanity check in `bli_pool_finalize()` that prevented BLIS from being re-initialized. (AMD)
- Fixed insufficient `pool_t`-growing logic in `bli_pool.c`, and always allocate at least one element in `.block_ptrs` array. (Minh Quan Ho)
- Cleanups related to the error message array in `bli_error.c`. (Minh Quan Ho)
- Moved language-related definitions from `bli_macro_defs.h` to a new header, `bli_lang_defs.h`.
- Renamed `BLIS_SIMD_NUM_REGISTERS` to `BLIS_SIMD_MAX_NUM_REGISTERS` and `BLIS_SIMD_SIZE` to `BLIS_SIMD_MAX_SIZE` for improved clarity. (Devin Matthews)
- Many minor bugfixes.
- Many cleanups, including removal of old and commented-out code.

Compatibility:
- Expanded BLAS layer to include support for `?axpby_()` and `?gemm_batch_()`. (Meghana Vankadari, AMD)
- Added `gemm3m` APIs to BLAS and CBLAS layers. (Bhaskar Nallani, AMD)
- Handle `?gemm_()` invocations where m or n is unit by calling `?gemv_()`. (Dipal M Zambare, AMD)
- Removed option to finalize BLIS after every BLAS call.
- Updated default definitions of `bli_slamch()` and `bli_dlamch()` to use constants from standard C library rather than values computed at runtime. (Devin Matthews)

Kernels:
- Added 512-bit SVE-based `a64fx` subconfiguration that uses empirically-tuned blocksizes (Stepan Nassyr, RuQing Xu)
- Added a vector-length agnostic `armsve` subconfig that computes blocksizes via an analytical model. (Stepan Nassyr)
- Added vector-length agnostic d/s/sh `gemm` kernels for Arm SVE. (Stepan Nassyr)
- Added `gemmsup` kernels to the `armv8a` kernel set for use in new Apple Firestorm subconfiguration. (RuQing Xu)
- Added 512-bit SVE `dpackm` kernels (16xk and 10xk) with in-register transpose. (RuQing Xu)
- Extended 256-bit SVE `dpackm` kernels by Linaro Ltd. to 512-bit for size 12xk. (RuQing Xu)
- Reorganized register usage in `bli_gemm_armv8a_asm_d6x8.c` to accommodate clang. (RuQing Xu)
- Added `saxpyf`/`daxpyf`/`caxpyf` kernels to `zen` kernel set. (Dipal M Zambare, AMD)
- Added `vzeroupper` instruction to `haswell` microkernels. (Devin Matthews)
- Added explicit `beta == 0` handling in s/d `armsve` and `armv7a` `gemm` microkernels. (Devin Matthews)
- Added a unique tag to branch labels to accommodate clang. (Devin Matthews, Jeff Hammond)
- Fixed a copy-paste bug in the loading of `kappa_i` in the two assembly `cpackm` kernels in `haswell` kernel set. (Devin Matthews)
- Fixed a bug in Mx1 `gemmsup` `haswell` kernels whereby the `vhaddpd` instruction is used with uninitialized registers. (Devin Matthews)
- Fixed a bug in the `power10` microkernel I/O. (Nicholai Tukanov)
- Many other Arm kernel updates and fixes. (RuQing Xu)

Extras:
- Added support for addons, which are similar to sandboxes but do not require the user to implement any particular operation.
- Added a new `gemmlike` sandbox to allow rapid prototyping of `gemm`-like operations.
- Various updates and improvements to the `power10` sandbox, including a new testsuite. (Nicholai Tukanov)

Build system:
- Added explicit support for AMD's Zen3 microarchitecture. (Dipal M Zambare, AMD, Field Van Zee)
- Added runtime microarchitecture detection for Arm. (Dave Love, RuQing Xu, Devin Matthews)
- Added a new `configure` option `--[en|dis]able-amd-frame-tweaks` that allows BLIS to compile certain framework files (each with the `_amd` suffix) that have been customized by AMD for improved performance (provided that the targeted configuration is eligible). By default, the more portable counterparts to these files are compiled. (Field Van Zee, AMD)
- Added an explicit compiler predicate (`is_win`) for Windows in `configure`. (Devin Matthews)
- Use `-march=haswell` instead of `-march=skylake-avx512` on Windows. (Devin Matthews, @h-vetinari)
- Fixed `configure` breakage on MacOSX by accepting either `clang` or `LLVM` in vendor string. (Devin Matthews)
- Blacklist clang10/gcc9 and older for `armsve` subconfig.
- Added a `configure` option to control whether or not to use `@rpath`. (Devin Matthews)
- Added armclang detection to `configure`. (Devin Matthews)
- Use `@path`-based install name on MacOSX and use relocatable `RPATH` entries for testsuite binaries. (Devin Matthews)
- For environment variables `CC`, `CXX`, `FC`, `PYTHON`, `AR`, and `RANLIB`, `configure` will now print an error message and abort if a user specifies a specific tool and that tool is not found. (Field Van Zee, Devin Matthews)
- Added symlink to `blis.pc.in` for out-of-tree builds. (Andrew Wildman)
- Register optimized real-domain `copyv`, `setv`, and `swapv` kernels in `zen` subconfig. (Dipal M Zambare, AMD)
- Added Apple Firestorm (A14/M1) subconfiguration, `firestorm`. (RuQing Xu)
- Added `armsve` subconfig to `arm64` configuration family. (RuQing Xu)
- Allow using clang with the `thunderx2` subconfiguration. (Devin Matthews)
- Fixed a subtle substitution bug in `configure`. (Chengguo Sun)
- Updated top-level Makefile to reflect a dependency on the "flat" `blis.h` file for the BLIS and BLAS testsuite objects. (Devin Matthews)
- Mark `xerbla_()` as a "weak" symbol on MacOSX. (Devin Matthews)
- Fixed a long-standing bug in `common.mk` whereby the header path to `cblas.h` was omitted from the compiler flags when compiling CBLAS files within BLIS.
- Added a custom-made recursive `sed` script to `build` directory.
- Minor cleanups and fixes to `configure`, `common.mk`, and others.

Testing:
- Fixed a race condition in the testsuite when the SALT option (simulate application-level threading) is enabled. (Devin Matthews)
- Test 1m method execution during `make check`. (Devin Matthews)
- Test `make install` in Travis CI. (Devin Matthews)
- Test C++ in Travis CI to make sure `blis.h` is C++-compatible. (Devin Matthews)
- Disabled SDE testing of pre-Zen microarchitectures via Travis CI.
- Added Travis CI support for testing Arm SVE. (RuQing Xu)
- Updated SDE usage so that it is downloaded from a separate repository (ci-utils) in our GitHub organization. (Field Van Zee, Devin Matthews)
- Updated octave scripts in `test/3` to be robust against missing datasets as well as to fixed a few minor issues.
- Added `test_axpbyv.c` and `test_gemm_batch.c` test driver files to `test` directory. (Meghana Vankadari, AMD)
- Support all four datatypes in `her`, `her2`, `herk`, and `her2k` drivers in `test` directory. (Madan mohan Manokar, AMD)

Documentation:
- Added documentation for: `setijv`, `getijv`, `eqsc`, `eqv`, `eqm`.
- Added `docs/Addons.md`.
- Added dedicated "Performance" and "Example Code" sections to `README.md`.
- Updated `README.md`.
- Updated `docs/Sandboxes.md`.
- Updated `docs/Multithreading.md`. (Devin Matthews)
- Updated `docs/KernelHowTo.md`.
- Updated `docs/Performance.md` to report Fujitsu A64fx (512-bit SVE) results. (RuQing Xu)
- Updated `docs/Performance.md` to report Graviton2 Neoverse N1 results. (Nicholai Tukanov)
- Updated `docs/FAQ.md` with new questions.
- Fixed typos in `docs/FAQ.md`. (Gaëtan Cassiers)
- Various other minor fixes.

## Changes in 0.8.1
March 22, 2021

Improvements present in 0.8.1:

Framework:
- Implemented an automatic reduction in the number of threads when the user requests parallelism via a single number (ie: the automatic way) and (a) that number of threads is prime, and (b) that number exceeds a minimum threshold defined by the macro `BLIS_NT_MAX_PRIME`, which defaults to 11. If prime numbers are really desired, this feature may be suppressed by defining the macro `BLIS_ENABLE_AUTO_PRIME_NUM_THREADS` in the appropriate configuration family's `bli_family_*.h`. (Jeff Diamond)
- Changed default value of `BLIS_THREAD_RATIO_M` from 2 to 1, which leads to slightly different automatic thread factorizations.
- Enable the 1m method only if the real domain microkernel is not a reference kernel. BLIS now forgoes use of 1m if both the real and complex domain kernels are reference implementations.
- Relocated the general stride handling for `gemmsup`. This fixed an issue whereby `gemm` would fail to trigger to conventional code path for cases that use general stride even after `gemmsup` rejected the problem. (RuQing Xu)
- Disabled AMD's small matrix handling entry points for `syrk` and `trsm` due to lack of testing on our side.
- Fixed an incorrect function signature (and prototype) of `bli_?gemmt()`. (RuQing Xu)
- Redefined `BLIS_NUM_ARCHS` to be part of the `arch_t` enum, which means it will be updated automatically when defining future subconfigs.
- Minor code consolidation in all level-3 `_front()` functions.
- Reorganized Windows cpp branch of `bli_pthreads.c`.
- Implemented `bli_pthread_self()` and `_equals()`, but left them commented out (via cpp guards) due to issues with getting the Windows versions working. Thankfully, these functions aren't yet needed by BLIS.

Kernels:
- Added low-precision POWER10 `gemm` kernels via a `power10` sandbox. This sandbox also provides an API for implementations that use these kernels. See the `sandbox/power10/POWER10.md` document for more info. (Nicholai Tukanov)
- Added assembly `packm` kernels for the `haswell` kernel set and registered to `haswell`, `zen`, and `zen2` subconfigs accordingly. The `s`, `c`, and `z` kernels were modeled on the `d` kernel, which was contributed by AMD.
- Reduced KC in the `skx` subconfig from 384 to 256. (Tze Meng Low)
- Fixed bugs in two `haswell` dgemmsup kernels, which involved extraneous assembly instructions left over from when the kernels were first written. (Kiran Varaganti, Bhaskar Nallani)
- Minor updates to all of the `gemmtrsm` kernels to allow division by diagonal elements rather that scaling by pre-inverted elements. This change was applied to `haswell` and `penryn` kernel sets as well as reference kernels, 1m kernels, and the pre-broadcast B (bb) format kernels used by the `power9` subconfig. (Bhaskar Nallani)
- Fixed incorrect return type on `bli_diag_offset_with_trans()`. (Devin Matthews)

Build system:
- Output a pkgconfig file so that CMake users that use BLIS can find and incorporate BLIS build products. (Ajay Panyala)
- Fixed an issue in the the configure script's kernel-to-config map that caused `skx` kernel flags to be used when compiling kernels from the `zen` kernel set. This issue wasn't really fixed, but rather tweaked in such a way that it happens to now work. A more proper fix would require a serious rethinking of the configuration system. (Devin Matthews)
- Fixed the shared library build rule in top-level Makefile. The previous rule was incorrectly only linking prerequisites that were newer than the target (`$?`) rather than correctly linking all prerequisites (`$^`). (Devin Matthews)
- Fixed `cc_vendor` for crosstool-ng toolchains. (Isuru Fernando)
- Allow disabling of `trsm` diagonal pre-inversion at compile time via `--disable-trsm-preinversion`.

Testing:
- Fixed obscure testsuite bug for the `gemmt` test module that relates to its dependency on `gemv`.
- Allow the `amaxv` testsuite module to run with a dimension of 0. (Meghana Vankadari)

Documentation:
- Documented auto-reduction for prime numbers of threads in `docs/Multithreading.md`.
- Fixed a missing `trans_t` argument in the API documentation for `her2k`/`syr2k` in `docs/BLISTypedAPI.md`. (RuQing Xu)
- Removed an extra call to `free()` in the level-1v typed API example code. (Ilknur Mustafazade)

## Changes in 0.8.0
November 19, 2020

Improvements present in 0.8.0:

Framework:
- Implemented support for the level-3 operation `gemmt`, which performs a `gemm` on only the lower or only the upper triangle of a square matrix C. For now, only the conventional/large code path (and not the sup code path) is provided. This support also includes `gemmt` APIs in the BLAS and CBLAS compatibility layers. (AMD)
- Added a C++ template header, `blis.hh`, containing a BLAS-inspired wrapper to a set of polymorphic CBLAS-like function wrappers defined in another header, `cblas.hh`. These headers are installed only when running the `install` target with `INSTALL_HH` set to `yes`. (AMD)
- Disallow `randv`, `randm`, `randnv`, and `randnm` from producing vectors and matrices with 1-norms of zero.
- Changed the behavior of user-initialized `rntm_t` objects so that packing of A and B is disabled by default. (Kiran Varaganti)
- Transitioned to using `bool` keyword instead of the previous integer-based `bool_t` typedef. (RuQing Xu)
- Updated all inline function definitions to use the cpp macro `BLIS_INLINE` instead of the `static` keyword. (Giorgos Margaritis, Devin Matthews)
- Relocated `#include "cpuid.h"` directive from `bli_cpuid.h` to `bli_cpuid.c` so that applications can `#include` both `blis.h` and `cpuid.h`. (Bhaskar Nallani, Devin Matthews)
- Defined `xerbla_array_()` to complement the netlib routine `xerbla_array()`. (Isuru Fernando)
- Replaced the previously broken `ref99` sandbox with a simpler, functioning alternative. (Francisco Igual)
- Fixed a harmless bug whereby `herk` was calling `trmm`-related code for determining the blocksize of KC in the 4th loop.

Kernels:
- Implemented a full set of `sgemmsup` assembly millikernels and microkernels for `haswell` kernel set.
- Implemented POWER10 `sgemm` and `dgemm` microkernels. (Nicholai Tukanov)
- Added two kernels (`dgemm` and `dpackm`) that employ ARM SVE vector extensions. (Guodong Xu)
- Implemented explicit beta = 0 handling in the `sgemm` microkernel in `bli_gemm_armv7a_int_d4x4.c`. This omission was causing testsuite failures in the new `gemmt` testsuite module for `cortexa15` builds given that the `gemmt` correctness check relies on `gemm` with beta = 0.
- Updated `void*` function arguments in reference `packm` kernels to use the native pointer type, and fixed a related dormant type bug in `bli_kernels_knl.h`.
- Fixed missing `restrict` qualifier in `sgemm` microkernel prototype for `knl` kernel set header.
- Added some missing n = 6 edge cases to `dgemmsup` kernels.
- Fixed an erroneously disabled edge case optimization in `gemmsup` variant code.
- Various bugfixes and cleanups to `dgemmsup` kernels.

Build system:
- Implemented runtime subconfiguration selection override via `BLIS_ARCH_TYPE`. (decandia50)
- Output the python found during `configure` into the `PYTHON` variable set in `build/config.mk`. (AMD)
- Added configure support for Intel oneAPI via the `CC` environment variable. (Ajay Panyala, Devin Matthews)
- Use `-O2` for all framework code, potentially avoiding intermitten issues with `f2c`'ed packed and banded code. (Devin Matthews)
- Tweaked `zen2` subconfiguration's cache blocksizes and registered full suite of `sgemm` and `dgemm` millikernels.
- Use the `-fomit-frame-pointer` compiler optimization option in the `haswell` and `skx` subconfigurations. (Jeff Diamond, Devin Matthews)
- Tweaked Makefiles in `test`, `test/3`, and `test/sup` so that running any of the usual targets without having first built BLIS results in a helpful error message.
- Add support for `--complex-return=[gnu|intel]` to `configure`, which allows the user to toggle between the GNU and Intel return value conventions for functions such as `cdotc`, `cdotu`, `zdotc`, and `zdotu`.
- Updates to `cortexa9`, `cortexa53` compilation flags. (Dave Love)

Testing:
- Added a `gemmt` module to the testsuite and a standalone test driver to the `test` directory, both of which exercise the new `gemmt` functionality. (AMD)
- Support creating matrices with small or large leading dimensions in `test/sup` test drivers.
- Support executing `test/sup` drivers with unpacked or packed matrices.
- Added optional `numactl` usage to `test/3/runme.sh`.
- Updated and/or consolidated octave scripts in `test/3` and `test/sup`.
- Increased `dotxaxpyf` testsuite thresholds to avoid false `MARGINAL` results during normal execution. (nagsingh)

Documentation:
- Added Epyc 7742 Zen2 ("Rome") performance results (single- and multithreaded) to `Performance.md` and `PerformanceSmall.md`. (Jeff Diamond)
- Documented `gemmt` APIs in `BLISObjectAPI.md` and `BLISTypedAPI.md`. (AMD)
- Documented commonly-used object mutator functions in `BLISObjectAPI.md`. (Jeff Diamond)
- Relocated the operation indices of `BLISObjectAPI.md` and `BLISTypedAPI.md` to appear immediately after their respective tables of contents. (Jeff Diamond)
- Added missing perl prerequisite to `BuildSystem.md`. (pkubaj, Dilyn Corner)
- Fixed missing `conjy` parameter in `BLISTypedAPI.md` documentation for `her2` and `syr2`. (Robert van de Geijn)
- Fixed incorrect link to `shiftd` in `BLISTypedAPI.md`. (Jeff Diamond)
- Mention example code at the top of `BLISObjectAPI.md` and `BLISTypedAPI.md`.
- Minor updates to `README.md`, `FAQ.md`, `Multithreading.md`, and `Sandboxes.md` documents.

## Changes in 0.7.0
April 7, 2020

Improvements present in 0.7.0:

Framework:
- Implemented support for multithreading within the sup (skinny/small/unpacked) framework, which previously was single-threaded only. Note that this feature works harmoniously with the selective packing introduced into the sup framework in 0.6.1. (AMD)
- Renamed `bli_thread_obarrier()` and `bli_thread_obroadcast()` functions to drop the 'o', which was left over from when `thrcomm_t` objects tracked both "inner" and "outer" communicators.
- Fixed an obscure `int`-to-`packbuf_t` type conversion error that only affects certain C++ compilers (including g++) when compiling application code that includes the BLIS header file `blis.h`. (Ajay Panyala)
- Added a missing early `return` statement in `bli_thread_partition_2x2()`, which provides a slight optimization. (Kiran Varaganti)

Kernels:
- Fixed the semantics of the `bli_amaxv()` kernels ('s' and 'd') within the `zen` kernel set. Previously, the kernels (incorrectly) returned the index of the last element whose absolute value was largest (in the event there were multiple of equal value); now, it (correclty) returns the index of the first of such elements. The kernels also now return the index of the first NaN, if one is encountered. (Mat Cross, Devin Matthews)

Build system:
- Warn the user at configure-time when hardware auto-detection returns the `generic` subconfiguration since this is probably not what they were expecting. (Devin Matthews)
- Removed unnecessary sorting (and duplicate removal) on `LDFLAGS` in `common.mk`. (Isuru Fernando)
- Specify the full path to the location of the dynamic library on OSX so that other dynamic libraries that depend on BLIS know where to find the library. (Satish Balay, Jed Brown)

Testing:
- Updated and reorganized test drivers in `test/sup` so that they work for either single-threaded or multithreaded purposes. (AMD)
- Updated/optimized octave scripts in `test/sup` for use with octave 5.2.0.
- Minor updates/tweaks to `test/1m4m`.

Documentation:
- Updated existing single-threaded sup performance graphs with new data and added multithreaded sup graphs to `docs/PerformanceSmall.md`.
- Added mention of Gentoo support under the external packages section of the `README.md`.
- Tweaks to `docs/Multithreading.md` that clarify that setting any `BLIS_*_NT` variable to 1 will be considered manual specification for the purposes of determining whether to auto-factorize via `BLIS_NUM_THREADS`. (AMD)

## Changes in 0.6.1
January 14, 2020

Improvements present in 0.6.1:

Framework:
- Added support for pre-broadcast when packing B. This causes elements of B to be repeated (broadcast) in the packed copy of B so that subsequent vector loads will result in the element already being pre-broadcast into the vector register.
- Added support for selective packing to `gemmsup` (controlled via environment variables and/or the `rntm_t` object). (AMD)
- Fixed a bug in `sdsdot_sub()` that redundantly added the "alpha" scalar and a separate bug in the order of typecasting intermediate products in `sdsdot_()`. (Simon Lukas Märtens, Devin Matthews)
- Fixed an obscure bug in `bli_acquire_mpart_mdim()`/`bli_acquire_mpart_ndim()`. (Minh Quan Ho)
- Fixed a subtle and complicated bug that only manifested via the BLAS test drivers in the `generic` subconfiguration, and possibly any other subconfiguration that did not register complex-domain `gemm` ukernels, or registered ONLY real-domain ukernels as row-preferential. (Dave Love)
- Always use `sumsqv` to compute `normfv` instead of the "dot product trick" that was previously employed for performance reasons. (Roman Yurchak, Devin Matthews, and Isuru Fernando)
- Fixed bug in `thrinfo_t` debugging/printing code.

Kernels:
- Implemented and registered an optimized `dgemm` microkernel for the `power9` kernel set. (Nicholai Tukanov)
- Pacify a `restrict` warning in the `gemmtrsm4m1` reference ukernel. (Dave Love, Devin Matthews)

Build system:
- Fixed parsing in `vpu_count()` on some SkylakeX workstations. (Dave Love)
- Reimplemented `bli_cpuid_query()` for ARM to use `stdio`-based functions instead of `popen()`. (Dave Love)
- Use `-march=znver1` for clang on `zen2` subconfig.
- Updated `-march` flags for `sandybridge`, `haswell` subconfigurations to use newer syntax (e.g. `haswell` instead of `core-avx2` and `sandybridge` instead of `corei7-avx`.
- Correctly use `-qopenmp-simd` for reference kernels when compiling with icc. (Victor Eikjhout)
- Added `-march` support for select gcc version ranges where flag syntax changes or new flags are added. The ranges we identify are: versions older than 4.9.0; versions older than 6.1.0 (but newer than 4.9.0); versions older than 9.1.0 (but newer than 6.1.0).
- Use `-funsafe-math-optimizations` and `-ffp-contract=fast` for all reference kernels when using gcc or clang.
- Updated MC cache blocksizes used by `haswell` subconfig.
- Updated NC cache blocksizes used by `zen` subconfig.
- Fixed a typo in the context registration of the `cortexa53` subconfiguration in `bli_gks.c`. (Francisco Igual)
- Output a more informative error when the user manually targets a subconfiguration that configure places in the configuration blacklist. (Tze Meng Low)
- Set execute bits of shared library at install-time. (Adam J. Stewart)
- Added missing thread-related symbols for export to shared libraries. (Kyungmin Lee)
- Removed (finally) the `attic/windows` directory since we offer Windows DLL support via AppVeyor's build artifacts, and thus that directory was only likely confusing people.

Testing:
- Fixed latent testsuite microkernel module bug for `power9` subconfig. (Jeff Hammond)
- Added `test/1m4m` driver directory for test drivers related to the 1m paper.
- Added libxsmm support to `test/sup drivers`. (Robert van de Geijn)
- Updated `.travis.yml` and `do_sde.sh` to automatically accept SDE license and download SDE directly from Intel. (Devin Matthews, Jeff Hammond)
- Updated standalone test drivers to iterate backwards through the specified problem space. This often helps avoid the situation whereby the CPU doesn't immediately throttle up to its maximum clock frequency, which can produce strange discontinuities (sharply rising "cliffs") in performance graphs.
- Pacify an unused variable warning in `blastest/f2c/lread.c`. (Jeff Hammond)
- Various other minor fixes/tweaks to test drivers.

Documentation:
- Added libxsmm results to `docs/PerformanceSmall.md`.
- Added BLASFEO results to `docs/PerformanceSmall.md`.
- Added the page size and location of the performance drivers to `docs/Performance.md` and `docs/PerformanceSmall.md`. (Dave Love)
- Added notes to `docs/Multithreading.md` regarding the nuances of setting multithreading parameters the manual way vs. the automatic way. (Jérémie du Boisberranger)
- Added a section on reproduction to `docs/Performance.md` and `docs/PerformanceSmall.md`. (Dave Love)
- Documented Eigen `-march=native` hack in `docs/Performance.md` and `docs/PerformanceSmall.md`. (Sameer Agarwal)
- Inserted multithreading links and disclaimers to `BuildSystem.md`. (Jeff Diamond)
- Fixed typo in description for `bli_?axpy2v()` in `docs/BLISTypedAPI.md`. (Shmuel Levine)
- Added "How to Download BLIS" section to `README.md`. (Jeff Diamond)
- Various other minor documentation fixes.

## Changes in 0.6.0
June 3, 2019

Improvements present in 0.6.0:

Framework:
- Implemented small/skinny/unpacked (sup) framework for accelerated level-3 performance when at least one matrix dimension is small (or very small). For now, only `dgemm` is optimized, and this new implementation currently only targets Intel Haswell through Coffee Lake, and AMD Zen-based Ryzen/Epyc. (The existing kernels should extend without significant modification to Zen2-based Ryzen/Epyc once they are available.) Also, multithreaded parallelism is not yet implemented, though application-level threading should be fine. (AMD)
- Changed function pointer usages of `void*` to new, typedef'ed type `void_fp`.
- Allow compile-time disabling of BLAS prototypes in BLIS, in case the application already has access to prototypes.
- In `bli_system.h`, define `_POSIX_C_SOURCE` to `200809L` if the macro is not already defined. This ensures that things such as pthreads are properly defined by an application that has `#include "blis.h"` but omits the definition of `_POSIX_C_SOURCE` from the command-line compiler options. (Christos Psarras)

Kernels:
- None.

Build system:
- Updated the way configure and the top-level Makefile handle installation prefixes (`prefix`, `exec_prefix`, `libdir`, `includedir`, `sharedir`) to better conform with GNU conventions.
- Improved clang version detection. (Isuru Fernando)
- Use pthreads on MinGW and Cygwin. (Isuru Fernando)

Testing:
- Added Eigen support to test drivers in `test/3`.
- Fix inadvertently hidden `xerbla_()` in blastest drivers when building only shared libraries. (Isuru Fernando, M. Zhou)

Documentation:
- Added `docs/PerformanceSmall.md` to showcase new BLIS small/skinny `dgemm` performance on Kaby Lake and Epyc.
- Added Eigen results (3.3.90) to performance graphs showcased in `docs/Performance.md`.
- Added BLIS thread factorization info to `docs/Performance.md`.

## Changes in 0.5.2
March 19, 2019

Improvements present in 0.5.2:

Framework:
- Added support for IC loop parallelism to the `trsm` operation.
- Implemented a pool-based small block allocator and a corresponding `configure` option (enabled by default), which minimizes the number of calls to `malloc()` and `free()` for the purposes of allocating small blocks (on the order of 100 bytes). These small blocks are used by internal data structures, and the repeated allocation and freeing of these structures could, perhaps, cause memory fragmentation issues in certain application circumstances. This was never reproduced and observed, however, and remains entirely theoretical. Still, the sba should be no slower, and perhaps a little faster, than repeatedly calling `malloc()` and `free()` for these internal data structures. Also, the sba was designed to be thread-safe. (AMD)
- Refined and extended the output enabled by `--enable-mem-tracing`, which allows a developer to follow memory allocation and release performed by BLIS.
- Initialize error messages at compile-time rather than at runtime. (Minh Quan Ho)
- Fixed a potential situation whereby the multithreading parameters in a `rntm_t` object that is passed into an expert interface is ignored.
- Prevent a redefinition of `ftnlen` in the `f2c_types.h` in blastest. (Jeff Diamond)

Kernels:
- Adjusted the cache blocksizes in the `zen` sub-configuration for `float`, `scomplex`, and `dcomplex` datatypes. The previous values, taken directly from the `haswell` subconfig, were merely meant to be reasonable placeholders until more suitable values were determined, as had already taken place for the `double` datatype. (AMD)
- Rewrote reference kernels in terms of simplified indexing annotated by the `#pragma omp simd` directive, which a compiler can use to vectorize certain constant-bounded loops. The `#pragma` is disabled via a preprocessor macro layer if the compiler is found by `configure` to not support `-fopenmp-simd`. (Devin Matthews, Jeff Hammond)

Build system:
- Added symbol-export annotation macros to all of the function prototypes and global variable declarations for public symbols, and created a new `configure` option, `--export-shared=[public|all]`, that controls which symbols--only those that are meant to be public, or all symbols--are exported to the shared library. (Isuru Fernando)
- Standardized to using `-O3` in various subconfigs, and also `-funsafe-math-optimizations` for reference kernels. (Dave Love, Jeff Hammond)
- Disabled TBM, XOP, LWP instructions in all AMD subconfigs. (Devin Matthews)
- Fixed issues that prevented using BLIS on GNU Hurd. (M. Zhou)
- Relaxed python3 requirements to allow python 3.4 or later. Previously, python 3.5 or later was required if python3 was being used. (Dave Love)
- Added `thunderx2` sub-configuration. (Devangi Parikh)
- Added `power9` sub-configuration. For now, this subconfig only uses reference kernels. (Nicholai Tukanov)
- Fixed an issue with `configure` failing on OSes--including certain flavors of BSD--that contain a slash '/' character in the output of `uname -s`. (Isuru Fernando, M. Zhou)

Testing:
- Renamed `test/3m4m` directory to `test/3`.
- Lots of updates and improvements to Makefiles, shell scripts, and matlab scripts in `test/3`.

Documentation:
- Added a new `docs/Performance.md` document that showcases single-threaded, single-socket, and dual-socket performance results of `single`, `double`, `scomplex`, and `dcomplex` level-3 operations in BLIS, OpenBLAS, and MKL/ARMPL for Haswell, SkylakeX, ThunderX2, and Epyc hardware architectures. (Note: Other implementations such as Eigen and ATLAS may be added to these graphs in the future.)
- Updated `README.md` to include new language on external packages. (Dave Love)
- Updated `docs/Multithreading.md` to be more explicit about the fact that multithreading is disabled by default at configure-time, and the fact that BLIS will run executed single-threaded at runtime by default if no multithreaded specification is given. (M. Zhou)

## Changes in 0.5.1
December 18, 2018

Improvements present in 0.5.1:

Framework:
- Added mixed-precision support to the 1m method implementation.
- Track internal scalar datatypes in the `obj_t` info bitfield. This allows slightly better handling of scalars during mixed-datatype `gemm` computation.
- Fixed a bug that allowed execution of 1m with mixed-precision `gemm`, despite such usage not yet being officially supported. (Devangi Parikh)
- Added missing internal calls to `bli_init_once()` in `bli_thread_set_num_threads()` and `bli_thread_set_ways()`. (Ali Emre Gülcü)

Kernels:
- Redefined `packm` kernels to handle edge cases and zero-filling, and updated their APIs accordingly. This was needed in order to fully support the use of non-default/non-reference packm kernels. (Devin Matthews)

Build system:
- Disallow explicit requests to use 64-bit integers in the BLAS API while simultaneously using 32-bit integers in the BLIS API. (Jeff Hammond, Devin Matthews)
- Fixed an msys2/Windows build failure. (Isuru Fernando, Costas Yamin)
- Fixed a MinGW build failure. (Isuru Fernando)
- Disabled `arm32`, `arm64` configuration families since we don't yet have logic to choose the correct context at runtime.

Testing:
- Make sure the testsuite fails for `NaN`, `Inf` in input operands. (Devin Matthews)
- Added `hemm` driver to `test/3m4m`.
- Minor updates to `test/mixeddt` drivers, matlab scripts.
- Added additional matlab plotting scripts to `test/3m4m`.

Documentation:
- Updated `docs/Multithreading.md` to include discussion of setting affinity via OpenMP.
- Updated `docs/Testsuite.md` to include discussion of mixed-datatype settings.
- Updated `docs/MixedDatatypes.md` to include a brief section on running the testsuite to exercise mixed-datatype functionality, and other minor updates.
- Fixed broken links in `docs/KernelsHowTo.md`. (Richard Goldschmidt)
- Spelling fixes in FAQ. (Rhys Ulerich)
- Updated 3-clause license comment blocks to refer generically to copyright holders rather than just the original copyright holder, UT-Austin.

## Changes in 0.5.0
October 25, 2018

Improvements present in 0.5.0:

Framework:
- Implemented support for matrix operands of mixed datatypes (domains and precisions) within the `gemm` operation.
- Added configure-time option to use slab or round-robin partitioning within JR and IR loops of most level-3 operations' macrokernels.
- Allow parallelism in the JC loop for `trsm_l`, which previously was unnecessarily disabled. (Field Van Zee, Devangi Parikh)
- Added Fortran-77/90-compatible APIs for some thread-related functions. (Kay Dewhurst)
- Defined a new level-1d operation `shiftd`, which adds a scalar value to every element along an arbitrary diagonal of a matrix.
- Patched an issue (#267) that may arise when linking against OpenMP-configured BLIS from which parallelism is requested at runtime and a level-3 operation (e.g. `gemm`) is called from within an OpenMP parallel region of an application where OpenMP nested parallelism is disabled. (Devin Matthews)

Kernels:
- Imported SkylakeX `dgemm` microkernel from `skx-redux` branch, which contains optimizations (mostly better prefetching on C) over the previous implementation. (Devin Matthews)
- Renamed/relocated level-3 `zen` microkernels to the `haswell` kernel set. Please see a recent message to blis-devel for more information on this rename [1].
- BG/Q kernel fixes. (Ye Luo)

Build system:
- Added support for building Windows DLLs via AppVeyor [2], complete with a built-in implementation of pthreads for Windows, as well as an implementation of the `pthread_barrier_*()` APIs for use on OS X. (Isuru Fernando, Devin Matthews, Mathieu Poumeyrol, Matthew Honnibal)
- Defined a `cortexa53` sub-configuration, which is similar to `cortexa57` except that it uses slightly different compiler flags. (Mathieu Poumeyrol)
- Added python version checking to `configure` script.
- Added a script to automate the regeneration of the symbols list file (now located in `build/libblis-symbols.def`).
- Various tweaks in preparation for BLIS's inclusion within Debian. (M. Zhou)
- Various fixes and cleanups.

Testing:
- Added tests for `cortexa15` and `cortexa57` in Travis CI. (Mathieu Poumeyrol)
- Added tests for mixed-datatype `gemm` and the simulation of application-level threading (salt) in Travis CI.
- Add statistics-collecting `irun.py` script.
- Include various threading parameters in the initial comment block of testsuite output.
- Various fixes and cleanups.

Documentation:
- Added `MixedDatatypes.md` documentation for mixed-datatype `gemm`.
- Added example code demonstrating use of mixed-datatype `gemm` (object API only).
- Added description of `shiftd` to `BLISTypedAPI.md` and `BLISObjectAPI.md`.
- Added "Known issues" sections to `Multithreading.md` and `Sandboxes.md`.
- Updated `FAQ.md`.
- Various other documentation updates.

[1] https://groups.google.com/forum/?fromgroups#!topic/blis-devel/pytWRjIzxVY
[2] https://ci.appveyor.com/project/shpc/blis/

## Changes in 0.4.1
August 30, 2018

Improvements present in 0.4.1:

Framework:
- Improved thread safety by homogenizing all critical sections to unconditionally use pthread mutexes. (AMD)
- Fixed `bli_finalize()`, which had become uncallable due to sharing `pthread_once_t` objects between the initialization and finalization steps. This manifested as a rather large memory leak (many megabytes) if/when the application manually finalized BLIS in the middle of its execution. (Devangi Parikh, Field Van Zee)
- Fixed a minor memory leak in the global kernel structure. (Devangi Parikh, Field Van Zee)
- Replaced extensive use of function "chooser" macros in object API functions with use of a new set of functions using the suffix `_qfp()` ("query function pointer"). These functions can be used to query function pointers for most families of typed functions.
- Fixed an obscure integer size bug due to improper use of integer literal constants with `va_arg()`. This oddly manifested as LP64 systems using the general stride output case of microkernels even when the output matrix storage matched that of the microkernel output preference. (Devangi Parikh, Field Van Zee)

Kernels:
- Fixed compilation of `armv7a` kernels. (Mathieu Poumeyrol)

Build system:
- Generate makefile fragments within the `obj` directory rather than in `config`, `kernels`, `ref_kernels`, and `frame`. This allows a user to perform an out-of-tree build even if the BLIS source distribution is read-only. (Devin Matthews)
- Allow a dependent sub-project such as example code or the testsuite to compile and link against an installation of BLIS rather than implicitly searching for a local (uninstalled) copy. (Victor Eijkhout, Field Van Zee)
- Fixed a link error that manifested after building only a shared library (e.g. `--disable-static`) and then trying to build a dependent sub-project such as example code or the testsuite. (Sajid Ali)
- Changed `test` make target of top-level `Makefile` to behave more like `check` by printing a color-coded characterization of the test results.
- Fixed the `-p` option to `configure`, which had likely been broken since May 7, 2018. The `--prefix` option was unaffected. (Dave Love)
- Running `configure` no longer requires a C++ compiler given that a C++ compiler was only ever envisioned for *optional* use in the sandbox. (Devangi Parikh, Field Van Zee)

Testing:
- Added the ability to "simulate" multiple application-level threads in the testsuite by executing the individual experiments with multiple threads. This should make it easier to test for thread-safety in the future. (AMD)
- Removed borderline useless wall clock time from test drivers' output.

Documentation:
- Updated typed and object API documents to include language on `rntm_t` parameters in the expert interfaces.
- Updates to `README.md`, including language on sandboxes.
- Added table of make targets to `BuildSystem.md`.
- Added missing language to `ConfigurationHowTo.md` on updating the architecture string array in `bli_arch.c`. (Devangi Parikh, Field Van Zee)

## Changes in 0.4.0
July 27, 2018

Framework:
- Added support for "sandboxes" for employing alternative `gemm` implementations. A ready-to-use reference C99 sandbox provides developers with a starting point for experimentation.
- Separated expert, non-expert typed APIs (levels 1v, 1d, 1f, 1m, 2, and 3, and utility functions).
- Defined new `rntm_t` structure and API to provide a uniform way of storing user-level threading information (equivalent of `BLIS_NUM_THREADS` and `BLIS_*_NT` environment variables), and also conveying that information to expert APIs. (Matthew Honnibal, Nathaniel Smith)
- Renamed various `obj_t` accessor macros, converted to static functions, and inserted explicit typecasting to facilitate #including blis.h from a C++ application. (Jacob Gorm Hansen)
- Cache and reuse `arch_t` architecture query result at runtime. (Devin Matthews)
- Implemented object-based functions `bli_projm()`/`_projv()`, which project objects from one domain to another (within the same precision), and `bli_castm()`/`_castv()`, which typecast objects from one datatype to another.
- Implemented object-based functions `bli_setrm()`/`_setrv()`, `bli_setim()`/`_setiv()`, which allow the caller to broadcast a scalar to all real elements or all imaginary elements within an object.
- Enforce consistent datatypes in most object APIs.
- For native execution, initialize a context's virtual microkernel slots to the function pointers of native microkernels. This simplifies query routines and paves the way for more generalized use of virtual microkernels beyond those for induced methods.
- Various bugfixes. (Devangi Parikh)

Kernels:
- Re-expressed x86_64 microkernels in terms of assembly language macros, which support lower- and upper-case, AT&T and Intel syntax. (Devin Matthews)
- Various bugfixes. (Robin Christ, Francisco Igual, Devangi Parikh, qnerd)

Build system:
- Added support for `--libdir`, `--includedir` configure options. (Nico Schlömer)
- Adopted Linux-like shared library versioning and enabled building shared libraries by default.
- Improved shared library handling on OS X. (Alex Arslan)
- Added configure support for preset `CFLAGS`, `LDFLAGS`. (Dave Love)
- Improvements to version file handling.
- Implemented configure option hack for circumventing small/limited values of `ARG_MAX`.
- Reorganized `cc`, `cc_vendor` detection responsibilities from `Makefile` to `configure`. (Alex Arslan)
- Cross-compilation fixes.
- Preliminary Windows ABI suport using `clang`, appveyor. (Isuru Fernando)
- Better support for typical development environment on OpenBSD, FreeBSD. (Alex Arslan)
- Bumped shared library `soname` version number to 1.0.0.
- Various build system fixes and cleanups. (Mathieu Poumeyrol, Nico Schlömer, Tony Skjellum)

Testing:
- Rewrote Travis CI testing config file and supporting logic to use Intel's SDE emulator. This allows multiple x86_64 microarchitectures to be tested regardless of what hardware Travis happens to be using at the time. (Devin Matthews)
- Added `docs/studies` hardware-specific test driver directory to track individual performance studies. (Devangi Parikh)
- Streamlined `testsuite/input.operations` file format.

Documentation:
- Relocated all wiki documents to a `docs` directory and adjusted all links, and `README.md`, accordingly.
- Added a `CONTRIBUTING.md` file to top-level directory.
- Added `docs/CodingConventions.md`.
- Added `docs/Sandboxes.md`.
- Added `docs/BLISObjectAPI.md`.
- Renamed and updated `docs/BLISTypedAPI.md`.
- Updated `docs/KernelsHowTo.md`.
- Updated `docs/BuildSystem.md`. (Stefanos Mavros)
- Updated `docs/Multithreading.md`.
- Updated indentation in `docs/ConfigurationHowTo.md` for easier reading.
- Added example code for the BLIS typed API in `examples/tapi`.
- Expanded existing example code for the object API in `examples/oapi`.
- Added links to RHEL/Fedora and Debian packages to `README.md`.
- Various cleanups. (Tony Skjellum, Dave Love, Nico Schlömer)

## Changes in 0.3.2
April 28, 2018

- Added `setijm`, `getijm` operations for updating and querying individual matrix elements via the object API.
- Added `examples/oapi` directory containing a code-based tutorial on using the object-based API in BLIS.
- Track separate reference kernel `CFLAGS` for each sub-configuration.
- Added support for blacklisting sub-configurations based on the assembler/binutils.
- Added 64-bit support to BLAS test drivers.
- Various bugfixes.

## Changes in 0.3.1
April 4, 2018

- Enable use of new zen kernels in haswell sub-configuration.
- Added row-storage optimizations to zen `dotxf` kernels (now also used by haswell).
- Integrated an `f2c`ed version of the BLAS test drivers from netlib LAPACK into BLIS build system (e.g. `make testblas`, `make checkblas`). See the [Testsuite](Testsuite.md) document for more info. Also scheduled these BLAS drivers to execute regularly via Travis CI.
- Added a new `make check` target that executes a fast version of the BLIS testsuite as well as the BLAS test drivers (primarily targeting package maintainers).
- Allow individual operation overriding in the BLIS testsuite. (This makes it easy to quickly test one or two operations of interest.)
- Added build system support for libmemkind. If present, `hbw_malloc()` is used as the default value for `BLIS_MALLOC_POOL` instead of `malloc()`. It can be disabled via `--disable-memkind`.
- Tweaks and fixes to BLAS compatibility layer, courtesy of the new BLAS test drivers.
- Output the active sub-configuration in testsuite output header.
- Allow arbitrary nesting of "umbrella" configuration families in `config_registry`, allowing us to define x86_64 in terms of amd64 and intel64.
- Added skx and knl to intel64 (and by proxy, x86_64) configuration families.
- Implemented basic support for ARM hardware detection (via `/proc/cpuinfo`).
- Various bugfixes.

## Changes in 0.3.0
February 23, 2018

This version contains significant improvements from 0.2.2. Major changes include:
- Real and complex domain (s,d,c,z) assembly-based gemm microkernels for AMD's Zen microarchitecture. (AMD, Field Van Zee)
- Real domain (s,d) assembly-based `gemmtrsm_l` and `gemmtrsm_u` microkernels for Zen. (AMD, Field Van Zee)
- Real domain (s,d) intrinsics-based `amaxv`, `axpyv`, `dotv`, `dotxv`, `scalv`, `axpyf`, and `dotxf` kernels for Zen. (AMD, Field Van Zee)
- Generalized the configuration system to allow multi-configuration builds targeting configuration "families". A single sub-configuration is chosen at runtime via some heuristic, such as querying CPUID (e.g. runtime hardware detection). This change was extensive and required a reorganization of the build system, configuration semantics, reference kernels, a new naming scheme for native kernels, and a rewrite of the global kernel structure (gks). Please see the rewritten [Configuration Guide](ConfigurationHowTo.md) for details.
- Implemented runtime hardware detection for x86_64 hardware.
- Reimplemented configure-time hardware detection in terms of new runtime hardware detection code, which queries for CPU features rather than individual models.
- Implemented library self-initialization by rewriting `bli_init()` in terms of `pthread_once()` and inserting invocations to `bli_init()` in key places throughout BLIS. The expectation is that through normal use of any BLIS API (BLAS, typed BLIS, or object-based BLIS), the user no longer needs to explicitly initialize the library, and that `bli_finalize()` should never be called by the user unless he is absolutely sure he no longer needs BLIS functionality. Related to this: global scalar constants (`BLIS_ONE`, `BLIS_ZERO`, etc.) are now statically initialized and thus ready to use immediately. Collectively, these changes provide improved thread safety at the application level.
- Compile with and install a single monolithic (flattened) `blis.h` header to (1) speed up compilation and (2) reduce the number of build product files.
- Added a sub-API for setting multithreading environment variables at runtime. For a few examples, please see the [Multithreading](Multithreading.md) guide.
- Reimplemented OpenMP/pthread barriers in terms of GNU atomic built-ins.
- Other small changes and fixes.

## Changes in 0.2.2
May 2, 2017

- Implemented the 1m method for inducing complex matrix multiplication. (Please see ACM TOMS publication ["Implementing high-performance complex matrix multiplication via the 1m method"](https://github.com/flame/blis#citations) for more details.)
- Switched to simpler `trsm_r` implementation.
- Relaxed constraints that `MC % NR = 0` and `NC % MR = 0`, as this was only needed for the more sophisticated `trsm_r` implementation.
- Automatic loop thread assignment. (Devin Matthews)
- Updates to `.travis.yml` configuration file. (Devin Matthews)
- Updates to non-default haswell microkernels.
- Match storage format of the temporary micro-tiles in macrokernels to that of the microkernel storage preference for edge cases.
- Added support for Intel's Knight's Landing. (Devin Matthews)
- Added more flexible options to specify multithreading via the configure script. (Devin Matthews)
- OS X compatibility fixes. (Devin Matthews)
- Other small changes and fixes.

Also, thanks to Elmar Peise, Krzysztof Drewniak, and Francisco Igual for their contributions in reporting/fixing certain bugs that were addressed in this version.

## Changes in 0.2.1
October 5, 2016

- Implemented distributed `thrinfo_t` structure management. (Ricardo Magana)
- Redesigned BLIS's level-3 algorithmic control tree structure. (suggested by Tyler Smith)
- Consolidated `gemm`, `herk`, and `trmm` blocked variants into one set of three bidirectional variants.
- Integrated a new "memory broker" (`membrk_t`) abstraction in place of the previous memory allocator, which allows one set of pools per broker (or, in other words, per memory space). (Ricardo Magana)
- Reorganized multithreading APIs, including more consistent namespace prefixes: `bli_thrinfo_*()`, `bli_thrcomm_*()`, etc.
- Added `randnm`, `randnv` operations, which produce random powers of two in a narrow range, and integrated a corresponding option into the testsuite. (suggested by AMD)
- Reclassified `amaxv` as a level-1v operation and kernel.
- Added complex `gemm` microkernels for haswell, which have register allocations consistent with the existing 6x16 `sgemm` and 6x8 `dgemm` microkernels.
- Adjusted existing microkernels to work properly when BLIS is configured to use 32-bit integers. (Devin Matthews)
- Relaxed alignment constraints in sandybridge and haswell microkernels. (Devin Matthews)
- Define CBLAS API with `f77_int` instead of `int`, which means the BLAS compatibility integer size is inherited by the CBLAS compatibility layer. (Devin Matthews)
- Added an alignment switch to the testsuite to globally enable/disable starting address and leading dimension alignment. (suggested by Devin Matthews)
- Various enhancements to configure script. (Devin Matthews)
- Avoid compiling BLAS/CBLAS compatibility layer when it is disabled via configure. (suggested by Devin Matthews)
- Disabled compilation of object-based blocked partitioning code for level-2 operations, as it was already functionally disabled.
- Fixes and tweaks to POSIX thread support. (Tyler Smith, Jeff Hammond)
- Other small changes and fixes.

## Changes in 0.2.0
April 11, 2016

Most of BLIS 0.2.0's changes are contained within a single commit, 537a1f4 (aka "the big commit"). An executive summary of the most consequential of these changes follows:

- BLIS has been retrofitted with a new data structure, known as a "context," affecting virtually every internal API for every computational operation, as well as many supporting, non-computational functions that must access information within the context.
- In addition to appearing within these internal APIs, the context--specifically, a pointer to a `cntx_t`--is now present within all user-level datatype-aware APIs, e.g. `bli_zgemm()`, appearing as the last argument.
- User-level object APIs, e.g. `bli_gemm()`, were unaffected and continue to be "context-free." However, these APIs were duplicated so that corresponding "context-aware" APIs now also exist, differentiated with an `_ex` suffix (for "expert").
- Contexts are initialized very soon after a computational function is called (if one was not passed in by the caller) and are passed all the way down the function stack, even into the kernels, and thus allow the code at any level to query information about the runtime instantiation of the current operation being executed, such as kernel addresses, microkernel storage preferences, and cache/register blocksizes.
- Contexts are thread-friendly. For example, consider the situation where a developer wishes two or more threads to execute simultaneously with somewhat different runtime parameters. Contexts also inherently promote thread-safety, such as in the event that the original source of the information stored in the context changes at run-time (see next two bullets).
- BLIS now consolidates virtually all kernel/hardware information in a new "global kernel structure" (gks) API. This new API will allow the caller to initialize a context in a thread-safe manner according to the currently active kernel configuration. For now, the currently active configuration cannot be changed once the library is built. However, in the future, this API will be expanded to allow run-time management of kernels and related parameters.
- The most obvious application of this new infrastructure is the run-time detection of hardware (and the implied selection of appropriate kernels). With contexts, kernels may even be "hot swapped" within the gks, and once execution begins on a level-3 operation, the memory allocator will be reinitialized on-the-fly, if necessary, to accommodate the new kernels' blocksizes. If a different application thread is executing with another (previously loaded) kernel, it will finish in a deterministic fashion because its kernel info was loaded into its context before computation began, and also because the blocks it checked out from the memory pools will be unaffected by the newer threads' reinitialization of the allocator.

This version contains other changes that were committed prior to 537a1f4:

- Inline assembly FMA4 microkernels for AMD bulldozer. (Etienne Sauvage)
- A more feature-rich configure script and build system. Certain long-style options are now accepted, including convenient command-line switches for things like enabling debugging symbols. Important definitions were also consolidated into a new makefile fragment, `common.mk`, which can be included by the BLIS build system as well as quasi-independent build systems, such as the BLIS test suite. (Devin Matthews)
- Updated and improved armv8 microkernels. (Francisco Igual)
- Define `bli_clock()` in terms of `clock_gettime()` intead of `gettimeofday()`, which has been languishing on my to-do list for years, literally. (Devin Matthews)
- Minor but extensive modifications to parts of the BLAS compatibility layer to avoid potential namespace conflicts with external user code when `blis.h` is included. (Devin Matthews)
- Fixed a missing BLIS integer type definition (`BLIS_BLAS2BLIS_INT_TYPE_SIZE`) when CBLAS was enabled. Thanks to Tony Kelman reporting this bug.
- Merged `packm_blk_var2()` into `packm_blk_var1()`. The former's functionality is used by induced methods for complex level-3 operations. (Field Van Zee)
- Subtle changes to treatment of row and column strides in `bli_obj.c` that pertain to somewhat unusual use cases, in an effort to support certain situations that arise in the context of tensor computations. (Devin Matthews)
- Fixed an unimplemented `beta == 0` case in the penryn (formerly "dunnington") `sgemm` microkernel. (Field Van Zee)
- Enhancements to the internal memory allocator in anticipation of the context retrofit. (Field Van Zee)
- Implemented so-called "quadratic" matrix partitioning for thread-level parallelism, whereby threads compute thread index ranges to produce partitions of roughly equal area (and thus computation), subject to the (register) blocksize multiple, even when given a structured rectangular subpartition with an arbitrary diagonal offset. Thanks to Devangi Parikh for reporting bugs related to this feature. (Field Van Zee)
- Enabled use of Travis CI for automatic testing of github commits and pull requests. (Xianyi Zhang)
- New `README.md`, written in github markdown. (Field Van Zee)
- Many other minor bug fixes.

Special thanks go to Lee Killough for suggesting the use of a "context" data structure in discussions that transpired years ago, during the early planning stages of BLIS, and also for suggesting such a perfectly appropriate name.

## Changes in 0.1.8
July 29, 2015

This release contains only two commits, but they are non-trivial: we now have configuration support for AMD Excavator (Carrizo) and microkernels for Intel Haswell/Broadwell.

## Changes in 0.1.7
June 19, 2015

- Replaced the static memory allocator used to manage internal packing buffers with one that dynamically allocates memory, on-demand, and then recycles the allocated blocks in a software cache, or "pool". This significantly simplifies the memory-related configuration parameter set, and it completely eliminates the need to specify a maximum number of threads.
- Implemented default values for all macro constants previously found in `bli_config.h`. The default values are now set in `frame/include/bli_config_macro_defs.h`. Any value #defined in `bli_config.h` will override these defaults.
- Initial support for configure-time detection of hardware. By specifying the `auto` configuration at configure-time, the configure script chooses a configuration for you. If an optimized configuration does not exist, the reference implementation serves as a fallback.
- Completely reorganized implementations for complex induced methods and added support for new algorithms.
- Added optimized microkernels for AMD Piledriver family of hardware.
- Several bugfixes to multithreaded execution.
- Various other minor tweaks, code reorganizations, and bugfixes.

## Changes in 0.1.6
October 23, 2014

- New complex domain AVX microkernels are now available and used by default by the sandybridge configuration.
- Added new high-level 4m and 3m implementations presently known as "4mh" and "3mh".
- Cleaned up 4m/3m front-end layering and added routines to enable, disable, and query which implementation will be called for a given level-3 operation. The test suite now prints this information in its pre-test summary. 4m (not 4mh) is still the default when complex microkernels are not present.
- Consolidated control tree code and usage so that all level-3 multiplication operations use the same gemm_t structure, leaving only `trsm` to have a custom tree structure and associated code.
- Re-implemented micropanel alignment, which was removed in commit c2b2ab6 earlier this year.
- Relaxed the long-standing constraint that `KC` be a multiple of `MR and `NR` by allowing the developer to specify target values and then adjusting them up to the next multiple of `MR` or `NR`, as needed by the affected operations (`hemm`, `symm`, `trmm`, trsm`).
- Added a new "row preference" flag that the developer can use to signal to the framework that a microkernel prefers to output micro-tiles of C that are row-stored (rather than column-stored). Column storage preference is still the default.
- Changed semantics of blocksize extensions to instead be "maximum" blocksizes (and thus emphasizing the "extended" values rather than the difference).
- Various other minor tweaks, code reorganizations, and bugfixes.

Thanks go to those whose contributions, feedback, and bug reports led to these improvements--in particular, Tony Kelman, Kevin Locke, Devin Matthews, Tyler Smith, and perhaps others whose feedback I've lost track of.

## Changes in 0.1.5
August 4, 2014

- Added a CBLAS compatibility layer, which can be enabled at configure-time via `BLIS_ENABLE_CBLAS` in `bli_config.h`. Enabling the CBLAS layer implicitly forces the BLAS compatibility layer to also be enabled. Once enabled, the application may access CBLAS prototypes via `blis.h` or `cblas.h`.
- Fixed a packing bug for cases when `MR` or `NR` (or both) are 1.
- Redefined bit field macros in `bli_type_defs.h` with bitshift operator to ease future rearranging, expanding, or adding of info bits.

## Changes in 0.1.4
July 27, 2014

- Added shared library support to build system.
- Preliminary parallelization of `trsm` (Tyler Smith).
- Added generic `_void()` microkernel wrappers so that users (or developers) can call the microkernel without knowing the implementation/developer-specific function names, which are specified at configure-time.
- Added `bli_info_*()` API for querying general information about BLIS, including blocksizes.
- Reimplemented initialization/finalization for thread safety.
- Fixed a possible `Inf`/`NaN` issue in several level-3 operations when beta is zero.
- Minor fixes to BLAS compatibility layer.
- Added initial support for Emscripten (Marat Dukhan).

## Changes in 0.1.3
June 23, 2014

This is a relatively minor release. The changes can be summarized as:
- Added experimental support for PNaCL (Marat Dukhan).
- Fixed aligned memory allocation on Windows (Tony Kelman).
- Fixed missing version string in build products when downloading tarballs/zip files (Field Van Zee). Thanks to Victor Eijkhout for pointing out this bug.

## Changes in 0.1.2
June 2, 2014

Tyler has been hard at work developing and refining extensions to BLIS that provide multithreading support (currently via OpenMP, though POSIX threads may be supported in the future). These extensions enable multithreading within all level-3 operations except for `trsm`. We are pleased to announce that these code changes are now part of BLIS.

## Changes in 0.1.1
February 25, 2014

I. I am excited to announce that BLIS now provides high-performance complex domain support to ALL level-3 operations when ONLY the same-precision real domain equivalent gemm microkernel is present and optimized. In other words, BLIS's productivity lever just got twice as strong: optimize the `dgemm` microkernel, and you will get double-precision complex versions of all level-3 operations, for free. Same for `sgemm` microkernel and single-precision complex.

II. We also now offer complex domain support based on the 3m method, but this support is ONLY accessible via separate interfaces. This separation is a safety feature, since the 3m method's numerical properties are inherently less robust. Furthermore, we think the 3m method, as implemented, is somewhat performance-limited on systems with L1 caches that have less than 8-way associativity.

We plan on writing a paper on (I) and (II), so if you are curious how exactly we accomplish this, please be patient and wait for the paper. :)

III. The second, user-oriented change facilitates a much more developer-friendly configuration system. This "change" actually represents a family of smaller changes. What follows is a list of those changes taken from the git log:
- We now have standard names for reference kernels (levels-1v, -1f and 3) in the form of macro constants. Examples:
      `BLIS_SAXPYV_KERNEL_REF`
      `BLIS_DDOTXF_KERNEL_REF`
      `BLIS_ZGEMM_UKERNEL_REF`
- Developers no longer have to name all datatype instances of a kernel with a common base name; [sdcz] datatype flavors of each kernel or microkernel (level-1v, -1f, or 3) may now be named independently. This means you can now, if you wish, encode the datatype-specific register blocksizes in the name of the microkernel functions.
- Any datatype instances of any kernel (1v, 1f, or 3) that is left undefined in `bli_kernel.h` will default to the corresponding reference implementation. For example, if `BLIS_DGEMM_UKERNEL` is left undefined, it will be defined to be `BLIS_DGEMM_UKERNEL_REF`.
- Developers no longer need to name level-1v/-1f kernels with multiple datatype chars to match the number of types the kernel WOULD take in a mixed type environment, as in `bli_dddaxpyv_opt()`. Now, one char is sufficient, as in `bli_daxpyv_opt()`.
- There is no longer a need to define an obj_t wrapper to go along with your level-1v/-1f kernels. The framework now provides a `_kernel()` function, as in `bli_axpyv_kernel()`, which serves as the `obj_t` wrapper for whatever kernels are specified (or defaulted to) via `bli_kernel.h`.
- Developers no longer need to prototype their kernels, and thus no longer need to include any prototyping headers from within `bli_kernel.h`. The framework now generates kernel prototypes, with the proper type signature, based on the kernel names defined (or defaulted to) via `bli_kernel.h`.
- If the complex datatype x (of [cz]) implementation of the gemm microkernel is left undefined by `bli_kernel.h`, but its same-precision real domain equivalent IS defined, BLIS will enable the automatic complex domain feature described above in (1a) for the datatype x implementations of all level-3 operations, using only the corresponding real domain gemm microkernel. If the complex gemm microkernel for x IS defined, then all complex level-3 operations will be defined in terms of that microkernel.

The net effect of (III) is that your `bli_kernel.h` files can be MUCH simpler and less cluttered. (Extreme example: the reference configuration's `bli_kernel.h` is now completely empty!) I have updated all configurations and kernels that are currently part of BLIS by stripping out unnecessary/outdated definitions and migrating existing definitions to their new names. (If you ever need to reference the complete list of options and macros, please refer to the `bli_kernel.h` inside the template configuration.) Please set aside some time to test and, if necessary, tweak the configurations which you originally developed and submitted. I may have broken some of them. If so, please accept my apologies and contact me for assistance. I will work with you to get them functional again.

The changes mentioned in (I), (II), and (III), along with all other changes since 0.1.0, are included BLIS 0.1.1 (fde5f1fd).

I know these changes may be a little disruptive to some, but I think that most developers will find the new complex functionality very useful, and the new configuration system much easier to use.

## Changes in 0.1.0
November 9, 2013

- Added `sgemm` microkernel for dunnington.
- Added `dgemm` microkernels and configurations for sandybridge, bgq, mic, power7, piledriver, loonson3a, which were used to gather performance data in our second ACM TOMS paper. Many thanks to Francisco Igual, Tyler Smith, Mike Kistler, and Xianyi Zhang for developing, testing, and contributing these kernels.
- Migrated to signed integer for `dim_t`, `inc_t` (to facilitate calling BLIS from Fortran).
- Added "template" configuration and kernel set for developers to use as a starting point when developing new kernels from scratch.
- Improvements to test suite, including section overrides and standalone level-1f/level-3 kernel modules.
- Improvements to Windows build system (though it may still not yet be functional out-of-the-box). Thanks to Martin Schatz for his help here.
- Removed support for element "duplication" in level-3 macrokernels.
- Several bug fixes to BLAS compatibility layer. Thanks to Vladimir Sukharev for his numerous bug reports wrt the LAPACK test suite.
- Various other minor bugfixes.

## Changes in 0.0.9
July 18, 2013

- A few algorithmic optimizations and bug fixes to `trmm` and `trsm`.
- Parameter checking in the compatibility layer that mimics netlib BLAS.
- Default use of `stdint.h` types (`int64_t`, `uint64_t` by default).
- Optional (and very much untested) C99 built-in complex type/arithmetic support.

Note that `bli_config.h` has changed since 0.0.8. Added configuration macros are:
```c
  #define BLIS_ENABLE_C99_COMPLEX
  #define BLIS_ENABLE_BLAS2BLIS_INT64
  #define PASTEF770(name) // ...
```
The first macro enables C99 built-in complex types. The second causes a Fortran integer to be defined as an int64_t (rather than `int32_t`). The third is a macro to name-mangle a full routine name for Fortran (ie: add an underscore) and should be obtained from `config/reference/bli_config.h`.

## Changes in 0.0.8
June 12, 2013

This version includes several kernel optimizations and bug fixes.

While neither `bli_config.h` nor `bli_kernel.h` has changed formats since 0.0.7, `make_defs.mk` **has** changed, so please update your copy of this file when you git-pull. Specifically, we now define a new `CFLAGS_KERNELS` variable that allows one to use different compiler flags when compiling kernels. It works like this: At compile time, make will use `CFLAGS_KERNELS` to compile any source code that resides in any directory that begins with the name `kernels`. My recommendation is to simply apply this naming convention to the symbolic link to your kernels directory that resides in your configuration directory. Thanks to Tyler for suggesting this change.

## Changes in 0.0.7
April 30, 2013

This version incorporates many small fixes and feature enhancements made during our SC13 collaboration.

## Changes in 0.0.6
April 13, 2013

Several changes regarding memory alignment were made since 0.0.5, including modifications to `bli_config.h`. Also, this update fixes a few bugs.

## Changes in 0.0.5
March 24, 2013

The most obvious change in this version is the migration to the `bli` function (and source code filename) prefix, from the old `bl2` prefix, as well as a rename of the main BLIS header (`blis2.h` -> `blis.h`). The test suite seems to indicate that the change was successful.

A few other much more minor changes were made, one pertaining to a renamed constant in the `_config.h` file.

## Changes in 0.0.4
March 15, 2013

The changes included in 0.0.4 mostly relate to the contiguous (static) memory allocator. The previous implementation was intended as a temporary solution that would work for benchmarking purposes, until enough other priorities had been tended to that I could go back and do it right.

I began with the assumption that the benefit of packing matrices into contiguous memory is non-negligible and worth the effort. Furthermore, we assume that:
- the only portable way to acquire contiguous memory is to reserve a region of static memory and manage it ourselves;
- the cache blocksizes used for one level-3 operation will be the same as those used for another level-3 operation, since all of them boil down to some form of matrix-matrix multiplication;
- only three types of contiguous memory will ever be needed (for level-3 operations): a block of matrix A, a panel of matrix B, or a panel of matrix C--and the last case is not commonly used;
- when a block or panel is to be acquired from the allocator, the caller knows which of the three types of memory is needed.

Given these assumptions, I was able to come up with an implementation that is simple, easy to understand, and thread-safe (provided you add OpenMP directives to protect the critical sections, which are clearly marked with comments). It can also both allocate and release in O(1) time. And of course, page-alignment is taken care of behind the scenes. So while it is not a generalized solution by any means, I think it will work very well for our purposes.

Also, note that based on the level of the overall matrix multiplication algorithm at which you parallelize, the minimum number of blocks/panels of each type of contiguous memory will vary. For example, if you want all of your threads to work on different iterations of a single rank-k update (via block-panel multiply), the threads share the packed panel of B, but each one needs memory to hold its own packed block of A. Thus, the memory allocator needs to be initialized so that it contains enough memory for at least one panel of B and at least t blocks of A, where t is the number of threads. All of this can be adjusted at configure-time in `bl2_config.h`.

## Changes in 0.0.3
February 22, 2013

The biggest change in this version is that the BLAS-to-BLIS compatibility layer is now available. Virtually every BLAS interface is included, even those corresponding to functionality that BLIS does not implement (such as banded and packed level-2 operations). If the application code attempts to call one of these unimplemented routines, the code aborts with a generic not-yet-implemented error message.

The compatibility layer is enabled via a configuration option in `bl2_config.h`. For now, it is enabled by default (provided you have an up-to-date copy of `bl2_config.h`).

## Changes in 0.0.2
February 11, 2013

Most notably, this version contains the new test suite I've been working on for the last month.

What is the test suite? It is a highly configurable test driver that allows one to test an arbitrary set of BLIS operations, with an arbitrary set of parameter combinations, and matrix/vector storage formats, as well as whichever datatypes you are interested in. (For now, only homogeneous datatyping is supported, which is what most people want.) You can also specify an arbitrary problem size range with arbitrary increments, and arbitrary ratios between dimensions (or anchor a dimension to a single value), and you can output directly to files which store the output in matlab syntax, which makes it easy to generate performance graphs.

BLIS developers: note that 0.0.2 makes small changes to the configuration files. This new version also contains many bug fixes. (Most of these fixes address bugs which were found using the test suite.)

## Changes in 0.0.1
December 10, 2012

- Added auto-detection of string version (via `git`).
- Wrote basic INSTALL, CHANGELOG, AUTHORS, and CREDITS files.
- Updates to standalone `test` directory `Makefile`.
- Added initial build system
- Various code reorganizations.

