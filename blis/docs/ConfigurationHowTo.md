## Contents

* **[Contents](ConfigurationHowTo.md#contents)**
* **[Introduction](ConfigurationHowTo.md#introduction)**
* **[Sub-configurations](ConfigurationHowTo.md#sub-configurations)**
  * [`bli_cntx_init_*.c`](ConfigurationHowTo.md#bli_cntx_init_c)
  * [`bli_family_*.h`](ConfigurationHowTo.md#bli_family_h)
  * [`make_defs.mk`](ConfigurationHowTo.md#make_defsmk)
* **[Configuration families](ConfigurationHowTo.md#configuration-families)**
* **[Configuration registry](ConfigurationHowTo.md#configuration-registry)**
  * [Walkthrough](ConfigurationHowTo.md#walkthrough)
  * [Printing the configuration registry lists](ConfigurationHowTo.md#printing-the-configuration-registry-lists)
* **[Adding a new kernel set](ConfigurationHowTo.md#adding-a-new-kernel-set)**
* **[Adding a new configuration family](ConfigurationHowTo.md#adding-a-new-configuration-family)**
* **[Adding a new sub-configuration](ConfigurationHowTo.md#adding-a-new-sub-configuration)**
* **[Further development topics](ConfigurationHowTo.md#further-development-topics)**
  * [Querying the current configuration](ConfigurationHowTo.md#querying-the-current-configuration)
  * [Header dependencies](ConfigurationHowTo.md#header-dependencies)
  * [Still have questions?](ConfigurationHowTo.md#still-have-questions)

## Introduction

This document describes how to manage, edit, and create BLIS framework configurations. **The target audience is primarily BLIS developers** who wish to add support for new types of hardware, and developers who write (or tinker with) BLIS kernels.

The BLIS [Build System](BuildSystem.md) guide introduces the concept of a BLIS [configuration](BuildSystem.md#Step_1:_Choose_a_framework_configuration). There are actually two types of configurations: sub-configuration and configuration families.

A _sub-configuration_ encapsulates all of the information needed to build BLIS for a particular microarchitecture. For example, the `haswell` configuration allows a user or developer to build a BLIS library that targets hardware based on Intel Haswell (or Broadwell or Skylake/Kabylake desktop) microprocessors. Such a sub-configuration typically includes optimized kernels as well as the corresponding cache and register blocksizes that allow those kernels to work well on the target hardware.

A _configuration family_ simply specifies a collection of other registered sub-configurations. For example, the `intel64` configuration allows a user or developer to build a BLIS library that includes several Intel x86_64 configurations, and hence supports multiple microarchitectures simultaneously. The appropriate configuration information (e.g. kernels and blocksizes) will be selected via some hardware detection heuristic (e.g. the `CPUID` instruction) at runtime. (**Note:** Prior to 290dd4a, configuration families could only be defined in terms of sub-configurations. Starting with 290dd4a, configuration families may be defined in terms of other families.)

Both of these configuration types are organized as directories of files and then "registered" into a configuration registry file named `config_registry`, which resides in the top-level directory.



## Sub-configurations

A sub-configuration is represented by a sub-directory of the `config` directory in the top-level of the BLIS distribution:
```
$ ls config
amd64      cortexa15  excavator  intel64  old         power7       template
bgq        cortexa57  generic    knc      penryn      sandybridge  zen
bulldozer  cortexa9   haswell    knl      piledriver  steamroller
```
Let's inspect the `haswell` configuration as an example:
```
$ ls config/haswell
bli_cntx_init_haswell.c  bli_family_haswell.h  make_defs.mk
```
A sub-configuration (`haswell`, in this case) usually contains just three files:
  * `bli_cntx_init_haswell.c`. This file contains the initialization function for a context targeting the hardware in question, in this case, Intel Haswell. A context, or `cntx_t` object, in BLIS encapsulates all of the hardware-specific information--including kernel function pointers and cache and register blocksizes--necessary to support all of the main computational operations in BLIS. The initialization function inside this file should be named the same as the filename (excluding `.c` suffix), which should begin with prefix `bli_cntx_init_` and end with the (lowercase) name of the sub-configuration. The context initialization function (in this case, `bli_cntx_init_haswell()`) is used internally by BLIS when setting up the global kernel structure--a mechanism for managing and supporting multiple microarchitectures simultaneously, so that the choice of which context to use can be deferred until the computation is ready to execute.
  * `bli_family_haswell.h`. This header file is `#included` when the configuration in question, in this case `haswell`, was the target to `./configure`. This is where you would specify certain global parameters and settings. For example, if you wanted to specify custom implementations of `malloc()` and `free()`, this is where you would specify them. The file is oftentimes empty. (In the case of configuration families, the definitions in this file apply to the _entire_ build, and not any specific sub-configuration, but for consistency we support them for all configuration targets, whether they be singleton sub-configurations or configuration families.)
  * `make_defs.mk`. This makefile fragment defines the compiler and compiler flags to use during compilation. Specifically, the values defined in this file are used whenever compiling source code specific to the sub-configuration (i.e., reference kernels and optimized kernels). If the sub-configuration is the target of `configure`, then these flags are also used to compile general framework code.

Providing these three components constitutes a complete sub-configuration. A more detailed description of each file will follow.



### bli_cntx_init_*.c

As mentioned above, the kernels used by a sub-configuration are specified in the `bli_cntx_init_` function. This function is flexible in that the context is typically initialized with a set of "reference" kernels. Then, the kernel developer overwrites the fields in the context that correspond to kernel operations that have optimized counterparts that should be used instead.

Let's use the following hypothetical function definition to guide our walkthrough.
```c
#include "blis.h"

void bli_cntx_init_fooarch( cntx_t* cntx )
{
    blksz_t blkszs[ BLIS_NUM_BLKSZS ];

    // Set default kernel blocksizes and functions.
    bli_cntx_init_fooarch_ref( cntx );

    // -------------------------------------------------------------------------

    // Update the context with optimized native gemm microkernels and
    // their storage preferences.
    bli_cntx_set_l3_nat_ukrs
    (
      5,
      BLIS_GEMM_UKR,       BLIS_DOUBLE, bli_dgemm_bararch_asm,       FALSE,
      BLIS_GEMMTRSM_L_UKR, BLIS_DOUBLE, bli_dgemmtrsm_l_bararch_asm, FALSE,
      BLIS_GEMMTRSM_U_UKR, BLIS_DOUBLE, bli_dgemmtrsm_u_bararch_asm, FALSE,
      BLIS_TRSM_L_UKR,     BLIS_DOUBLE, bli_dtrsm_l_bararch_asm,     FALSE,
      BLIS_TRSM_U_UKR,     BLIS_DOUBLE, bli_dtrsm_u_bararch_asm,     FALSE,
      cntx
    );

    // Update the context with optimized packm kernels.
    bli_cntx_set_packm_kers
    (
      2,
      BLIS_PACKM_4XK_KER, BLIS_DOUBLE, bli_dpackm_bararch_asm_4xk,
      BLIS_PACKM_8XK_KER, BLIS_DOUBLE, bli_dpackm_bararch_asm_8xk,
      cntx
    );

    // Update the context with optimized level-1f kernels.
    bli_cntx_set_l1f_kers
    (
      5,
      BLIS_AXPY2V_KER,    BLIS_DOUBLE, bli_daxpy2v_fooarch_asm,
      BLIS_DOTAXPYV_KER,  BLIS_DOUBLE, bli_ddotaxpyv_fooarch_asm,
      BLIS_AXPYF_KER,     BLIS_DOUBLE, bli_daxpyf_fooarch_asm,
      BLIS_DOTXF_KER,     BLIS_DOUBLE, bli_ddotxf_fooarch_asm,
      BLIS_DOTXAXPYF_KER, BLIS_DOUBLE, bli_ddotxaxpyf_fooarch_asm,
      cntx
    );

    // Update the context with optimized level-1v kernels.
    bli_cntx_set_l1v_kers
    (
      2,
      BLIS_AXPYV_KER, BLIS_DOUBLE, bli_daxpyv_fooarch_asm,
      BLIS_DOTV_KER,  BLIS_DOUBLE, bli_ddotv_fooarch_asm,
      cntx
    );

    // Initialize level-3 blocksize objects with architecture-specific values.
    //                                           s      d      c      z
    bli_blksz_init_easy( &blkszs[ BLIS_MR ],     8,     8,     8,     4 );
    bli_blksz_init_easy( &blkszs[ BLIS_NR ],     8,     4,     4,     4 );
    bli_blksz_init_easy( &blkszs[ BLIS_MC ],   128,   128,   128,   128 );
    bli_blksz_init_easy( &blkszs[ BLIS_KC ],   256,   256,   256,   256 );
    bli_blksz_init_easy( &blkszs[ BLIS_NC ],  4096,  4096,  4096,  4096 );

    // Update the context with the current architecture's register and cache
    // blocksizes (and multiples) for native execution.
    bli_cntx_set_blkszs
    (
      5,
      BLIS_NC, &blkszs[ BLIS_NC ], BLIS_NR,
      BLIS_KC, &blkszs[ BLIS_KC ], BLIS_KR,
      BLIS_MC, &blkszs[ BLIS_MC ], BLIS_MR,
      BLIS_NR, &blkszs[ BLIS_NR ], BLIS_NR,
      BLIS_MR, &blkszs[ BLIS_MR ], BLIS_MR,
      cntx
    );
}
```
_**Function name/signature.**_ This function always takes one argument, a pointer to a `cntx_t` object. As with the name of the file, it should be named with the prefix `bli_cntx_init_` followed by the lowercase name of the configuration--in this case, `fooarch`.

_**Blocksize object array.**_ The `blkszs` array declaration is needed later in the function and should generally be consistent (and unchanged) across all configurations.

_**Reference initialization.**_ The first function call, `bli_cntx_init_fooarch_ref()`, initializes the context `cntx` with function pointers to reference implementations of all of the kernels supported by BLIS (as well as cache and register blocksizes, and other fields). This function is automatically generated by BLIS for every sub-configuration enabled at configure-time. The function prototype is generated by a preprocessor macro in `frame/include/bli_arch_config.h`.

_**Level-3 microkernels.**_ The second function call is to a variable argument function, `bli_cntx_set_l3_nat_ukrs()`, which updates `cntx` with five optimized double-precision complex level-3 microkernels. The first argument encodes the number of individual kernels being registered into the context. Every subsequent line, except for the last line, is associated with the registration of a single kernel, and each of these lines is independent of one another and can occur in any order, provided that the kernel parameters of each line occur in the same order--kernel ID, followed by datatype, followed by function name, followed by storage preference boolean (i.e., whether the microkernel prefers row storage). The last argument of the function call is the address of the context being updated, `cntx`. Notice that we are registering microkernels written for another type of hardware, `bararch`, because in our hypothetical universe `bararch` is very similar to `fooarch` and so we recycle the code between the two configurations. After the function returns, the context contains pointers to optimized double-precision level-3 real microkernels. Note that the context will still contain reference microkernels for single-precision real and complex, and double-precision complex computation, as those kernels were not updated.

_Note:_ Currently, BLIS only allows the kernel developer to signal a preference (row or column) for `gemm` microkernels. The preference of the `gemmtrsm` and `trsm` microkernels can (and must) be set, but are ignored by the framework during execution.

_**Level-1m (packm) kernels.**_ The third function call is to another variable argument function, `bli_cntx_set_packm_kers()`. This function works very similar to `bli_cntx_set_l3_nat_ukrs()`, except that it expects a different set of kernel IDs (because now we are registering level-1m kernels) and it does not take a storage preference boolean.  After this function returns, `cntx` contains function pointers to optimized double-precision real `packm` kernels. These kernels, like the level-3 kernels previously, are also borrowed from the `bararch` kernel set. Unregistered `packm` kernels will continue to point to reference code.

_**Level-1f kernels.**_ The third function call is to yet another variable argument function, `bli_cntx_set_l1f_kers()`. This function has the same signature as `bli_cntx_set_packm_kers()`, except that it expects a different set of kernel IDs (because now we are registering level-1f kernels). After this function returns, `cntx` contains function pointers to optimized double-precision real level-1f kernels. These kernels are written for `fooarch` specifically. The unregistered level-1f kernels will continue to point to reference code.

_**Level-1v kernels.**_ The fourth function call is to `bli_cntx_set_l1v_kers()`, which operates similarly to the `bli_cntx_set_l1f_kers()`, except here we are registering level-1v kernels. After the function returns, most kernels will continue to point to reference code, except double-precision real instances of `axpyv` and `dotv`.

For a complete list of kernel IDs, please see the definitions of `l3ukr_t`, `l1mkr_t`, `l1fkr_t`, `l1vkr_t` in [frame/include/bli_type_defs.h](https://github.com/flame/blis/blob/master/frame/include/bli_type_defs.h).

_**Setting blocksizes.**_ The next block of code initializes the `blkszs` array with register and cache blocksize values for each datatype. The values here are used by the level-3 operations that employ the level-3 microkernels we registered previously. We use `bli_blksz_init_easy()` when initializing only the primary value. If the auxiliary value needs to be set to a different value that the primary, `bli_blksz_init()` should be used instead, as in:
```c
    //                                           s      d      c      z
    bli_blksz_init_easy( &blkszs[ BLIS_MR ],     0,     8,     0,     0 );
    bli_blksz_init_easy( &blkszs[ BLIS_NR ],     0,     4,     0,     0 );
    bli_blksz_init     ( &blkszs[ BLIS_MC ],     0,   128,     0,     0,
                                                 0,   160,     0,     0 );
    bli_blksz_init     ( &blkszs[ BLIS_KC ],     0,   256,     0,     0,
                                                 0,   288,     0,     0 );
    bli_blksz_init_easy( &blkszs[ BLIS_NC ],     0,  4096,     0,     0 );
```
Here, we use `bli_blksz_init()` to set different auxiliary (maximum) cache blocksizes for _MC_ and _KC_. The same function could be used to set auxiliary (packing) register blocksizes for _MR_ and _NR_, which correspond to the _PACKMR_ and _PACKNR_ parameters. Other blocksizes, particularly those corresponding to level-1f operations, may be set. For a complete list of blocksize IDs, please see the definitions of `bszid_t` in [frame/include/bli_type_defs.h](https://github.com/flame/blis/blob/master/frame/include/bli_type_defs.h). For more information on interpretations of the auxiliary blocksize value, see the digressions below.

Note that we set level-3 blocksizes even for datatypes that retain reference code kernels; however, by passing in `0` for those blocksizes, we indicate to `bli_blksz_init()` and `bli_blksz_init_easy()` that the current value should be left untouched. In the example above, this leaves the blocksizes associated with the reference kernels (set by `bli_cntx_init_fooarch_ref()`) intact for the single real, single complex, and double complex datatypes.

_Digression:_ Auxiliary blocksize values for register blocksizes are interpreted as the "packing" register blocksizes. _PACKMR_ and _PACKNR_ serve as "leading dimensions" of the packed micropanels that are passed into the microkernel. Oftentimes, _PACKMR = MR_ and _PACKNR = NR_, and thus the developer does not typically need to set these values manually. (See the [implementation notes for gemm](KernelsHowTo.md#Implementation_Notes_for_gemm) in the BLIS Kernel guide for more details on these topics.)

_Digression:_ Auxiliary blocksize values for cache blocksizes are interpreted as the maximum cache blocksizes. The maximum cache blocksizes are a convenient and portable way of smoothing performance of the level-3 operations when computing with a matrix operand that is just slightly larger than a multiple of the preferred cache blocksize in that dimension. In these "edge cases," iterations run with highly sub-optimal blocking. We can address this problem by merging the "edge case" iteration with the second-to-last iteration, such that the cache blocksizes are slightly larger--rather than significantly smaller--than optimal. The maximum cache blocksizes allow the developer to specify the _maximum_ size of this merged iteration; if the edge case causes the merged iteration to exceed this maximum, then the edge case is _not_ merged and instead it is computed upon in separate (final) iteration.

_**Committing blocksizes.**_ Finally, we commit the values in `blkszs` to the context by calling the variable argument function `bli_cntx_set_blkszs()`. This function call generally should be considered boilerplate and thus should not changed unless you are altering the matrix multiplication _algorithm_ as specified in the control tree. If this is your goal, please get in contact with BLIS developers via the [blis-devel](http://groups.google.com/group/blis-devel) mailing list for guidance, if you have not done so already.

_**Availability of kernels.**_ Note that any kernel made available to the `fooarch` configuration within `config_registry` may be referenced inside `bli_cntx_init_fooarch()`. In this example, we referenced `fooarch` kernels as well as kernels native to another configuration, `bararch`. Thus, the `config_registry` would contain a line such as:
```
fooarch: fooarch/fooarch/bararch
```
Interpreting the line left-to-right: the `fooarch` configuration family contains only itself, `fooarch`, but must be able to refer to kernels from its own kernel set (`fooarch`) as well as kernels belonging to the `bararch` kernel set. The configuration registry is described more completely [in a later section](ConfigurationHowTo.md#configuration-registry).



### bli_family_*.h

This file is conditionally `#included` only for the configuration family targeted at configure-time. For example, if you run `./configure haswell`, `bli_family_haswell.h` will be `#included`, and if you run `./configure intel64`, `bli_family_intel64.h` will be `#included`. The header file is `#included` by [frame/include/bli_arch_config.h](https://github.com/flame/blis/blob/master/frame/include/bli_arch_config.h).

This header file is oftentimes empty. This is because the parameters specified here usually work fine with their default values, which are defined in [frame/include/bli_kernel_macro_defs.h](https://github.com/flame/blis/blob/master/frame/include/bli_kernel_macro_defs.h). However, there may be some configurations for which a kernel developer will wish to adjust some of these parameters. Furthermore, when creating a configuration family, the parameters set in the corresponding `bli_family_*.h` file must work for **all** sub-configurations in the family.

A description of the parameters that may be set in `bli_family_*.h` follows.

_**Memory allocation functions.**_ BLIS allows the developer to customize the functions called for memory allocation for three different categories of memory: user, pool, and internal. The functions for user allocation are called any time the creation of a BLIS matrix or vector `obj_t` requires that a matrix buffer be allocated, such as via `bli_obj_create()`. The functions for pool allocation are called only when allocating blocks to the memory pools used to manage packed matrix buffers. The function for internal allocation are called by BLIS when allocating internal data structures, such as control trees. By default, the three pairs of parameters are defined via preprocessor macros to call the implementation of `malloc()` and `free()` provided by `stdlib.h`:
```c
#define BLIS_MALLOC_USER  malloc
#define BLIS_FREE_USER    free

#define BLIS_MALLOC_POOL  malloc
#define BLIS_FREE_POOL    free

#define BLIS_MALLOC_INTL  malloc
#define BLIS_FREE_INTL    free
```
Any substitute for `malloc()` and `free()` defined by customizing these parameters must use the same function prototypes as the original functions. Namely:
```c
void* malloc( size_t size );
void  free( void* p );
```
Furthermore, if a header file needs to be included, such as `my_malloc.h`, it should be `#included` within the `bli_family_*.h` file (before `#defining` any of the `BLIS_MALLOC_` and `BLIS_FREE_` macros).

_**SIMD register file.**_ BLIS allows you to specify the _maximum_ number of SIMD registers available for use by your kernels, as well as the _maximum_ size (in bytes) of those registers. These values default to:
```c
#define BLIS_SIMD_MAX_NUM_REGISTERS  32
#define BLIS_SIMD_MAX_SIZE           64
```
These macros are used in computing the maximum amount of temporary storage (typically allocated statically, on the function stack) that will be needed to hold a single micro-tile of any datatype (and for any induced method or mixed-precision computation):
```c
#define BLIS_STACK_BUF_MAX_SIZE  ( BLIS_SIMD_MAX_NUM_REGISTERS * BLIS_SIMD_MAX_SIZE * 4 )
```
These temporary buffers are used when handling edge cases (m % _MR_ != 0 || n % _NR_ != 0) within the level-3 macrokernels, and also in the virtual microkernels of various implementations of induced methods for complex matrix multiplication. It is **very important** that these values be set correctly; otherwise, you may experience undefined behavior as stack data is overwritten at run-time. A kernel developer may set `BLIS_SIMD_MAX_NUM_REGISTERS` and `BLIS_SIMD_MAX_SIZE`, which will indirectly affect `BLIS_STACK_BUF_MAX_SIZE`, or he may set `BLIS_STACK_BUF_MAX_SIZE` directly. Notice that the default values are already set to work with modern x86_64 systems.

_**Memory alignment.**_ BLIS implements memory alignment internally, rather than relying on a function such as `posix_memalign()`, and thus it can provide aligned memory even with functions that adhere to the `malloc()` and `free()` API in the standard C library.
```c
#define BLIS_SIMD_ALIGN_SIZE             BLIS_SIMD_MAX_SIZE
#define BLIS_PAGE_SIZE                   4096

#define BLIS_STACK_BUF_ALIGN_SIZE        BLIS_SIMD_ALIGN_SIZE
#define BLIS_HEAP_ADDR_ALIGN_SIZE        BLIS_SIMD_ALIGN_SIZE
#define BLIS_HEAP_STRIDE_ALIGN_SIZE      BLIS_SIMD_ALIGN_SIZE
#define BLIS_POOL_ADDR_ALIGN_SIZE_A      BLIS_PAGE_SIZE
#define BLIS_POOL_ADDR_ALIGN_SIZE_B      BLIS_PAGE_SIZE
#define BLIS_POOL_ADDR_ALIGN_SIZE_C      BLIS_PAGE_SIZE
#define BLIS_POOL_ADDR_ALIGN_SIZE_GEN    BLIS_PAGE_SIZE
```
The value `BLIS_STACK_BUF_ALIGN_SIZE` defines the alignment of stack memory used as temporary internal buffers, such as for output matrices to the microkernel when computing edge cases. (See [implementation notes](KernelsHowTo#implementation-notes-for-gemm) for the `gemm` microkernel for details.) This value defaults to `BLIS_SIMD_ALIGN_SIZE`, which defaults to `BLIS_SIMD_MAX_SIZE`.

The value `BLIS_HEAP_ADDR_ALIGN_SIZE` defines the alignment used when allocating memory via the `malloc()` function defined by `BLIS_MALLOC_USER`. Setting this value to `BLIS_SIMD_ALIGN_SIZE` may speed up certain level-1v and -1f kernels.

The value `BLIS_HEAP_STRIDE_ALIGN_SIZE` defines the alignment used for so-called "leading dimensions" (i.e. column strides for column-stored matrices, and row strides for row-stored matrices) when creating BLIS matrices via the object-based API (e.g. `bli_obj_create()`). While setting `BLIS_HEAP_ADDR_ALIGN_SIZE` guarantees alignment for the first column (or row), creating a matrix with certain dimension values (_m_ and _n_) may cause subsequent columns (or rows) to be misaligned. Setting this value to `BLIS_SIMD_ALIGN_SIZE` is usually desirable. Additional alignment may or may not be beneficial.

The value `BLIS_POOL_ADDR_ALIGN_SIZE_*` define the alignments used when allocating blocks to the memory pools used to manage internal packing buffers for matrices A, B, C, and for general use. Any block of memory returned by the memory allocator is guaranteed to be aligned to this value. Aligning these blocks to the virtual memory page size (usually 4096 bytes) is standard practice.



### make_defs.mk

The `make_defs.mk` file primarily contains compiler and compiler flag definitions used by `make` when building a BLIS library.

The format of the file is mostly self-explanatory. However, we will expound on the contents here, using the `make_defs.mk` file for the `haswell` configuration as an example:
```make
# Declare the name of the current configuration and add it to the
# running list of configurations included by common.mk.
THIS_CONFIG    := haswell

ifeq ($(CC),)
CC             := gcc
CC_VENDOR      := gcc
endif

CPPROCFLAGS    := -D_POSIX_C_SOURCE=200112L
CMISCFLAGS     := -std=c99 -m64
CPICFLAGS      := -fPIC
CWARNFLAGS     := -Wall -Wno-unused-function -Wfatal-errors

ifneq ($(DEBUG_TYPE),off)
CDBGFLAGS      := -g
endif

ifeq ($(DEBUG_TYPE),noopt)
COPTFLAGS      := -O0
else
COPTFLAGS      := -O3
endif

CKOPTFLAGS     := $(COPTFLAGS)

ifeq ($(CC_VENDOR),gcc)
CVECFLAGS      := -mavx2 -mfma -mfpmath=sse -march=core-avx2
else
ifeq ($(CC_VENDOR),icc)
CVECFLAGS      := -xCORE-AVX2
else
ifeq ($(CC_VENDOR),clang)
CVECFLAGS      := -mavx2 -mfma -mfpmath=sse -march=core-avx2
else
$(error gcc, icc, or clang is required for this configuration.)
endif
endif
endif

# Store all of the variables here to new variables containing the
# configuration name.
$(eval $(call store-make-defs,$(THIS_CONFIG)))
```
_**Configuration name.**_ The first statement reaffirms the name of the configuration. The `THIS_CONFIG` variable is used later to attach the configuration name as a suffix to the remaining variables so that they can co-exist with variables read from other `make_defs.mk` files during multi-configuration builds. Note that if the configuration name defined here does not match the name of the directory in which `make_defs.mk` is stored, `make` will output an error when executing the top-level `Makefile`.

_**Compiler definitions.**_ Next, we set the values of `CC` and `CC_VENDOR`. The former is the name (or path) to the actual compiler executable to use during compilation. The latter is the compiler family. Currently, BLIS generally supports three compiler families: `gcc`, `clang`, and `icc`. `CC_VENDOR` is used when conditionally setting various variables based on the type of flags available--flags that might not vary across different versions or installations of the same compiler (e.g. `gcc-4.9` vs `gcc-5.0`, or `gcc` vs `/usr/local/bin/gcc`), but may vary across compiler families (e.g. `gcc` vs. `icc`). If the compiler you wish to use is in your `PATH` environment variable, `CC` and `CC_VENDOR` will usually contain the same value.

_**Basic compiler flags.**_ The variables `CPPROCFLAGS` and `CWARNFLAGS` should be assigned to C preprocessor flags and compiler warning flags, respectively, while `CPICFLAGS` should be assigned flags to enable position independent code (shared library) flags. Finally, `CMISCFLAGS` may be assigned any miscellaneous flags that do not neatly fit into any other category, such as language flags and 32-/64-bit flags. These four categories of flags are usually recognized across compiler families.

_**Debugging flags.**_ The `CDBGFLAGS` variable should be assigned to contain flags that insert debugging symbols into the object code emitted by the compiler. Typically, this amounts to no more than the `-g` flag, but some compilers or situations may call for different (or additional) flags. This variable is conditionally set only if `$(DEBUG_TYPE)`, which is set the by `configure` script, is not equal to `noopt`.

_**Optimization flags.**_ The `COPTFLAGS` variable should be assigned any flags relating to general compiler optimization. Usually this takes the form of `-O2` or `-O3`, but more specific optimization flags may be included as well, such as `-fomit-frame-pointer`. Note that, as with `CDBGFLAGS`, `COPTFLAGS` is conditionally assigned based on the value of `$(DEBUG_TYPE)`. A separate `CKOPTFLAGS` variable tracks optimizations flags used when compiling kernels. For most configurations, `CKOPTFLAGS` is assigned as a copy of `COPTFLAGS`, but if the kernel developer needs different optimization flags to be applied when compiling kernel source code, `CKOPTFLAGS` should be set accordingly.

_**Vectorization flags.**_ The second-to-last block sets the `CVECFLAGS`, which should be assigned any flags that must be given to the compiler in order to enable use of a vector instruction set needed or assumed by the kernel source code. Also, if you wish to enable automatic use of certain instruction sets (e.g. `-mfpmath=sse` for many Intel architectures), this is where you should set those flags. These flags often differ among compiler families, especially between `icc` and `gcc`/`clang`.

_**Variable storage/renaming.**_ Finally, the last statement commits the variables defined in the file to "storage". That is, they are copied to variable names that contain `THIS_CONFIG` as a suffix. This allows the variables for one configuration to co-exist with variables of another configuration.


## Configuration families

A configuration family is represented similarly to that of a sub-configuration: a sub-directory of the `config` directory. Additionally, there are two types of families: singleton families and umbrella families.

A _singleton_ family simply refers to a sub-configuration. The `configure` script only targets configuration families. But since every sub-configuration is also a valid configuration family, every sub-configuration is a valid configuration target.

An _umbrella_ family is the more interesting type of configuration family. These families are defined as collections of architecturally related sub-configurations. (**Important:** an umbrella family should always be named something different than any of its constituent sub-configurations.) BLIS provides a mechanism to define umbrella families so that users and developers can build a single instance of BLIS that supports multiple configurations, where some heuristic is used at runtime to choose among the configurations. For example, you may wish to deploy a BLIS library on a storage device that is shared among several computers, each of which is based on a different x86_64 microarchitecture.

Throughout the remainder of this document, we will sometimes refer to "umbrella families" as simply "families". Similarly, we will refer to "singleton families" and "sub-configurations" interchangeably. To the extent that any ambiguity may remain, context should clarify which type of family is germane to the discussion.

Let's inspect the `amd64` configuration family as an example:
```
$ ls config/amd64
bli_family_amd64.h  make_defs.mk
```
A configuration family contains a subset of the files contained within a sub-configuration: A `bli_family_*.h` header file and a `make_defs.mk` makefile fragment:
  * `bli_family_amd64.h`. This header file is `#included` only when the configuration family in question, in this case `amd64`, was the target to `./configure`. The file serves a similar purpose as with sub-configurations--a place to define various parameters, such as those relating to memory allocation and alignment. However, in the context of configuration families, the uniqueness of this file makes a bit more sense. Importantly, the definitions in this file will be affect **all** sub-configurations within the family. Thus, it is useful to think of these as "global" parameters. For example, if custom implementations of `malloc()` and `free()` are specified in the `bli_family_amd64.h` file, these implementations will be used for every sub-configuration member of the `amd64` family. (The configuration registry, described in [the next section](ConfigurationHowTo.md#configuration-registry), specifies each configuration family's membership.) As with sub-configurations, this file may be empty, in which case reasonable defaults are selected by the framework.
  * `make_defs.mk`. This makefile fragment defines the compiler and compiler flags in a manner identical to that of sub-configurations. However, these configuration flags are used when compiling source code that is not specific to any one particular sub-configuration. (The build system compiles a set of reference kernels and optimized kernels for each sub-configuration, during which it uses flags read from the individual sub-configurations' `make_defs.mk` files. By contrast, the general framework code is compiled once--using the flags read from the family's `make_defs.mk` file--and executed by all sub-configurations.)

For a more detailed walkthrough of these files' expected/allowed contents, please see the descriptions provided in the section on [sub-configurations](ConfigurationHowTo.md#sub-configurations):
 * [bli_family_*.h](ConfigurationHowTo.md#bli_family_h)
 * [make_defs.h](ConfigurationHowTo.md#make_defsmk)

With these two files defined and present, the configuration family is properly constituted and ready to be registered within the configuration registry.



## Configuration registry

The configuration registry is the official place for declaring a sub-configuration or configuration family. Unless a configuration (singleton or family) is declared within the registry, `configure` will not accept it as a valid configuration target at configure-time.

Before describing the syntax and semantics of the registry, we'll first briefly describe three types of information we wish to encode into the registry:

_**Configuration list.**_ First and foremost, the registry needs to enumerate the registered sub-configurations. That is, it needs to list the sub-configurations (or, singleton families) that are available to be targeted by `configure`. The registry also needs to specify configuration family membership--that is, the (umbrella) families to which those sub-configurations belong.

_**Kernel list.**_ Next, the registry needs to specify the list of kernel sets that will be needed by each sub-configuration, and by proxy, each configuration family. It's easy to think of different configurations as corresponding to different microarchitectures, and that generally holds true. However, sometimes we use the same configuration for multiple microarchitectures (e.g. `haswell` is used for Intel Haswell, Broadwell, and non-server Skylake variants). It might also be tempting to think of each microarchitecture as having its own set of kernels. However, in practice, we find that some microarchitectures' kernels are identical to those of a previous microarchitectural revision, or to those of another vendor's microarchitecture. Thus, sometimes a sub-configuration will wish to use a kernel set that is "native" to a different configuration. In these cases, there is not a one-to-one mapping of sub-configuration names to kernel set names, and therefore the configuration registry must separately specify the kernel sets needed by any sub-configuration (and by proxy, any configuration family).

_**Kernel-to-configuration map.**_ Lastly, and most subtly, for each kernel set in the kernel list, the registry needs to specify the sub-configuration(s) that depend on that particular kernel set. Notice that the kernel list can be obtained by mapping sub-configurations to kernel sets they require. By contrast, the kernel-to-configuration map tracks the reverse dependency and helps us answer: for any given kernel set, which sub-configurations caused the kernel set to be pulled into the build? This mapping is needed when determining which sub-configuration's compiler flags (as defined in its `make_defs.mk` file) to use when compiling that kernel set. The most obvious solution to this problem would have been to associate compiler flags with the individual kernel sets. However, given the desire to share kernel sets among sub-configurations, we needed the flexibility of applying different compiler flags to any given kernel set based on the sub-configuration that would be utilizing that kernel set. In the case that multiple sub-configurations pull in the same kernel set, a set of heuristics is used to choose between the sub-configurations so that a single set of compiler flags can be chosen for use when compiling that kernel set.



### Walkthrough

The configuration registry exists as a human-readable file, `config_registry`, located at the top-level of the BLIS distribution. What follows is an example of a `config_registry` file that is based on actual contents in a BLIS commit recent as of this writing. Note that lines containing only whitespace are ignored. Furthermore, any characters that appear after (and including) a `#` are treated as comments and also ignored.
```
#
# config_registry
#

# Processor families.
x86_64:      intel64 amd64
intel64:     haswell sandybridge penryn generic
amd64:       zen excavator steamroller piledriver bulldozer generic
arm64:       cortexa57 generic
arm32:       cortexa15 cortexa9 generic

# Intel architectures.
haswell:     haswell
sandybridge: sandybridge
penryn:      penryn
knl:         knl

# AMD architectures.
zen:         zen/haswell/sandybridge
excavator:   excavator/piledriver
steamroller: steamroller/piledriver
piledriver:  piledriver
bulldozer:   bulldozer

# ARM architectures.
cortexa57:   cortexa57/armv8a
cortexa15:   cortexa15/armv7a
cortexa9:    cortexa9/armv7a


# Generic architectures.
generic:     generic
```
Generally speaking, the registry can be thought of as defining a very simple grammar. (However, as you'll soon see, there are nuances that are un-grammar-like.) The registry can contain two kinds of lines. The first type defines a singleton configuration family. For example, the line
```
haswell:     haswell
```
defines a configuration family `haswell` (the left side of the `:`) as containing only itself: the sub-configuration by the same name, `haswell` (the right side of the `:`). When singleton families are defined in this way, it implicitly pulls in the kernel set by the same name as the sub-configuration (in this case, `haswell`). More specifically, the `haswell` sub-configuration depends on the kernels residing in the `kernels/haswell` sub-directory.

The second type of line defines an umbrella configuration family. For example, the line
```
intel64:     haswell sandybridge penryn generic
```
defines the configuration family `intel64` as containing the `haswell`, `sandybridge`, `penryn`, and `generic` sub-configurations as members (technically speaking, it is more accurate to think of the family as containing singleton families rather than their corresponding sub-configurations). Thus, if the user runs `./configure intel64`, the library will be built to support all sub-configurations defined within the `intel64` family.

**Note:** `generic` is a somewhat special sub-configuration that uses only reference kernels and reference blocksizes. It is included in every umbrella family so that when those families are instantiated into BLIS libraries and linked to an application, the application will be able to run even if none of the other sub-configurations (`haswell`, `sandybridge`, `penryn`) are chosen at runtime by the hardware detection heuristic.

Some sub-configurations, for various reasons, do not rely on their own set of kernels and instead use the kernel set that is native to another sub-configuration. For example, the `excavator` and `steamroller` configurations each correspond to hardware that is very similar to the hardware targeted by the `piledriver` configuration. In fact, the former two configurations rely exclusively on kernels written for the latter configuration. (Presently, there are no `excavator` or `steamroller` kernel sets in BLIS.) We denote this kernel dependency with a `/` character:
```
excavator:   excavator/piledriver
steamroller: steamroller/piledriver
```
Here, the first line (reading from left-to-right) defines the `excavator` singleton family as containing only itself, the `excavator` sub-configuration, and also specifies that this sub-configuration must have access to the `piledriver` kernel set. The second line defines the `steamroller` singleton family in a similar manner.

**Note:** Specifying non-native kernel sets via the `/` character is only allowed when defining singleton configuration families. They may NOT appear in the definitions of umbrella families! When an umbrella family includes a singleton family that is defined to require non-native kernels, this will be accounted for during the parsing of the `config_registry` file.

Sometimes, a sub-configuration may need access to more than one kernel set. If additional kernel sets are needed, they should be listed with additional `/` characters:
```
zen:         zen/haswell/sandybridge
```
The line above defines the `zen` singleton family as containing only itself, the `zen` sub-configuration, and also specifies that this sub-configuration must have access to the `haswell` kernel set as well as the `sandybridge` kernel set. What if there exists a `zen` kernel set as well, which the `zen` sub-configuration must access in addition to those of `haswell` and `sanydbridge`? In this case, it would need to be annotated explicitly as:
```
zen:         zen/zen/haswell/sandybridge
```
This line (which is hypothetical and does not appear in the `config_registry` example above) defines the `zen` singleton family in terms of only the `zen` sub-configuration, and provides that sub-configuration access to `zen`, `haswell`, and `sandybridge` kernel sets. (Also: the kernel sets may appear in any order.)

Notice that while kernel sets usually correspond to a sub-configuration, they do not always. For example, while the `armv7a` and `armv8a` kernel sets are referenced in the example `config_registry` file, there do not exist any registered sub-configurations by those names. However, the kernel directories exist and the kernel sets appear in the definitions of a few `cortex` singleton families.

One last thing to point out: take a look at the `x86_64` configuration family:
```
x86_64:      intel64 amd64
```
Unlike most of the registered families, which are defined in terms of sub-configurations, `x86_64` is defined in terms of *other* families--specifically, `intel64` and `amd64`:
```
intel64:     haswell sandybridge penryn generic
amd64:       zen excavator steamroller piledriver bulldozer generic
```
This multi-level style of specifying sub-configurations became available starting in 290dd4a. The behavior of `configure` in this situation is as you would expect; that is, including `intel64` and `amd64` in the definition of `x86_64` is equivalent to:
```
x86_64:      haswell sandybridge penryn zen excavator steamroller piledriver bulldozer generic
```
Any duplicates that may result are removed automatically.


### Printing the configuration registry lists

The configuration list, kernel list, and kernel-to-configuration map are constructed internally by `configure`, but these structures can be inspected by running `configure` with the `-c` (which is the short form of `--show-config-lists`) option. This can be useful as a sanity check to make sure `configure` is properly parsing and interpreting the `config_registry` file.

The first thing printed is the configuration list:
```
$ ./configure -c amd64
configure: reading configuration registry...done.
...
configure: configuration list:
configure:   amd64: zen excavator steamroller piledriver bulldozer generic
configure:   arm32: cortexa15 cortexa9 generic
configure:   arm64: cortexa57 generic
configure:   bulldozer: bulldozer
configure:   cortexa15: cortexa15
configure:   cortexa57: cortexa57
configure:   cortexa9: cortexa9
configure:   excavator: excavator
configure:   generic: generic
configure:   haswell: haswell
configure:   intel64: haswell sandybridge penryn generic
configure:   knl: knl
configure:   penryn: penryn
configure:   piledriver: piledriver
configure:   sandybridge: sandybridge
configure:   skx: skx
configure:   steamroller: steamroller
configure:   x86_64: haswell sandybridge penryn zen excavator steamroller piledriver bulldozer generic
```
This simply lists the sub-configurations associated with each defined configuration family (singleton or umbrella). Note that they are sorted alphabetically.

Next, the kernel list (actually, all kernel lists) is printed:
```
configure: kernel list:
configure:   amd64: zen piledriver bulldozer generic
configure:   arm32: armv7a generic
configure:   arm64: armv8a generic
configure:   bulldozer: bulldozer
configure:   cortexa15: armv7a
configure:   cortexa57: armv8a
configure:   cortexa9: armv7a
configure:   excavator: piledriver
configure:   generic: generic
configure:   haswell: haswell zen
configure:   intel64: haswell zen sandybridge penryn generic
configure:   knl: knl
configure:   penryn: penryn
configure:   piledriver: piledriver
configure:   sandybridge: sandybridge
configure:   skx: skx
configure:   steamroller: piledriver
configure:   x86_64: haswell sandybridge penryn zen piledriver bulldozer generic
configure:   zen: zen
```
This shows the kernel sets that are pulled in by each configuration family. For singleton families, this is specified in a straightforward manner via the `/` character described [in the previous section](ConfigurationHowTo.md#Walkthrough). For umbrella families, this is determined indirectly by looking up the definitions of the singleton families that are members of the umbrella family.

Next, the full kernel-to-configuration map is printed:
```
configure: kernel-to-config map for 'amd64':
configure:   bulldozer: bulldozer
configure:   generic: generic
configure:   piledriver: excavator steamroller piledriver
configure:   zen: zen
```
For each of the kernel sets required of the selected configuration family above, the kernel-to-configuration map shows the sub-configurations that required that kernel set. Notice that sometimes a single kernel set may be pulled in by more than one sub-configuration, as with the `piledriver` kernel set.

Lastly, we print a version of the kernel-to-configuration map in which we've used a set of heuristics to select a single sub-configuration for each kernel set in the map:
```
configure: kernel-to-config map for 'amd64' (chosen pairs):
configure:   bulldozer:bulldozer
configure:   generic:generic
configure:   piledriver:piledriver
configure:   zen:zen
```
This variant of the kernel-to-config map is formatted as a series of "sub-configuration:kernel-set" pairs. These pairs are used during the processing of the top-level `Makefile` to determine which sub-configuration's compiler flags should be used when compiling the source code within each kernel set.


## Adding a new kernel set

Adding support for a new set of kernels in BLIS is easy and can be done via the following steps.



1. _**Create and populate the kernel set directory.**_ First, we must create a directory in `kernels` that corresponds to the new kernel set. Suppose we wanted to add kernels for Intel's Knight's Landing microarchitecture. In BLIS, this corresponds to the `knl` configuration, and so we should name the directory `knl`. This is because we want the `knl` kernel set to be pulled by default into builds that include the `knl` sub-configuration.
   ```
   $ mkdir kernels/knl
   $ ls kernels
   armv7a  bgq        generic  knc  old     piledriver  sandybridge
   armv8a  bulldozer  haswell  knl  penryn  power7
   ```
   Next, we must write the `knl` kernels and locate them inside `kernels/knl`. (For more information on writing BLIS kernels, please see the [Kernels Guide](KernelsHowTo.md).) We recommend separating level-1v, level-1f, and level-3 kernels into separate `1`, `1f`, and `3` sub-directories, respectively. The kernel files and functions therein do not need to follow any particular naming convention, though we strongly recommend using the conventions already used by other kernel sets. Take a look at other kernel files, such as those for `haswell`, [for examples](https://github.com/flame/blis/tree/master/kernels). Finally, for the `knl` kernel set, you should insert a file named `bli_kernels_knl.h` into `kernels/knl` that prototypes all of your new kernel set's kernel functions. You are welcome to write your own prototypes, but to make the prototyping of kernels easier we recommend using the prototype-generating macros for level-1v, level-1f, level-1m, and level-3 functions defined in [frame/1/bli_l1v_ker_prot.h](https://github.com/flame/blis/blob/master/frame/1/bli_l1v_ker_prot.h), [frame/1f/bli_l1f_ker_prot.h](https://github.com/flame/blis/blob/master/frame/1f/bli_l1f_ker_prot.h), [frame/1m/bli_l1m_ker_prot.h](https://github.com/flame/blis/blob/master/frame/1m/bli_l1m_ker_prot.h), and [frame/3/bli_l3_ukr_prot.h](https://github.com/flame/blis/blob/master/frame/3/bli_l3_ukr_prot.h), respectively. The following example utilizes how a select subset of these macros can be used to generate kernel function prototypes.
   ```c
   GEMM_UKR_PROT( double, d, gemm_knl_asm_24x8 )

   PACKM_KER_PROT( double, d, packm_knl_asm_24xk )
   PACKM_KER_PROT( double, d, packm_knl_asm_8xk )

   AXPYF_KER_PROT( dcomplex, z, axpyf_knl_asm )
   DOTXF_KER_PROT( dcomplex, z, dotxf_knl_asm )

   AXPYV_KER_PROT( float, s, axpyv_knl_asm )
   DOTXV_KER_PROT( float, s, dotxv_knl_asm )
   ```
   The first line generates a function prototype for a double-precision real `gemm` microkernel named `bli_dgemm_knl_asm_24x8()`. Notice how the macro takes three arguments: the C language datatype, the single character corresponding to the datatype, and the base name of the function, which includes the operation (`gemm`), the kernel set name (`knl`), and a substring specifying its implementation (`asm_24x8`).

   The second and third lines generate prototypes for double-precision real `packm` kernels to go along with the `gemm` microkernel above. The fourth and fifth lines generate prototypes for double-precision complex instances of the level-1f kernels `axpyf` and `dotxf`. The last two lines generate prototypes for single-precision real instances of the level-1v kernels `axpyv` and `dotxv`.



2. _**Add support within the framework source code.**_ We also need to make a minor update to the framework to support the new kernels--specifically, to pull in the kernels' function prototypes.

   **`frame/include/bli_arch_config.h`**. When adding support for the `knl` kernel set to the framework, we must modify this file to `#include` the `bli_kernels_knl.h` header file:
   ```c
   #ifdef BLIS_KERNELS_KNL
   #include "bli_kernels_knl.h"
   #endif
   ```
   The `BLIS_KERNELS_KNL` macro, which guards the `#include` directive, is automatically defined by the build system when the `knl` kernel set is required by _any_ sub-configuration.


## Adding a new configuration family

Adding support for a new umbrella configuration family in BLIS is fairly straightforward and can be done via the following steps. The hypothetical examples used in these steps assume you are trying to create a new configuration family `intelavx` that supports only Intel microarchitectures that support the Intel AVX instruction set.



1. _**Create and populate the family directory.**_ First, we must create a directory in `config` that corresponds to the new family. Since we are adding a new family named `intelavx`, we would name our directory `intelavx`.
   ```
   $ mkdir config/intelavx
   $ ls config
   amd64      cortexa15  excavator  intel64   knl     piledriver   steamroller
   bgq        cortexa57  generic    intelavx  old     power7       template
   bulldozer  cortexa9   haswell    knc       penryn  sandybridge  zen
   ```
   We also need to create `bli_family_intelavx.h` and `make_defs.mk` files inside our new sub-directory. Since they will be very similar to those of the `intel64` family's files, we can copy those files over and then modify them accordingly:
   ```
   $ cp config/intel64/bli_family_intel64.h config/intelavx/bli_family_intelavx.h
   $ cp config/intel64/make_defs.mk config/intelavx/
   ```
   First, we update the configuration name inside of `make_defs.mk`:
   ```
   THIS_CONFIG    := intelavx
   ```
   and while we're editing the file, we can make any other changes to compiler flags we wish (if any). Similarly, the `bli_family_intelavx.h` header file should be updated, though in our case it does not need any changes; the original file is empty and thus the copied file can remain empty as well. Note that other configuration families may have different needs. Remember that all of the parameters set in this file, either explicitly or implicitly (via their defaults), must work for **all** sub-configurations in the family. When creating or modifying a family, it's worth reviewing the parameters' defaults, which are set in [frame/include/bli_kernel_macro_defs.h](https://github.com/flame/blis/blob/master/frame/include/bli_kernel_macro_defs.h) and convincing yourself that each parameter default (or overriding definition in `bli_family_*.h`) will work for each sub-configuration.



2. _**Add support within the framework source code.**_ Next, we need to update the BLIS framework source code so that the new configuration family is recognized and supported. Configuration families require updates to two files.

   * **`frame/include/bli_arch_config.h`**. This file must be updated to `#include` the `bli_family_intelavx.h` header file. Notice that the preprocessor directive should be guarded as follows:
      ```c
      #ifdef BLIS_FAMILY_INTELAVX
      #include "bli_family_intelavx.h"
      #endif
      ```
      The `BLIS_FAMILY_INTELAVX` will automatically be defined by the build system whenever the family was targeted by `configure` is `intelavx`. (In general, if the user runs `./configure foobar`, the C preprocessor macro `BLIS_FAMILY_FOOBAR` will be defined.)

   * **`frame/base/bli_arch.c`**. This file must be updated so that `bli_arch_query_id()` returns the correct `arch_t` microarchitecture ID value to the caller. This function is called when the framework is trying to choose which sub-configuration to use at runtime. For x86_64 architectures, this is supported via the `CPUID` instruction, as implemented via `bli_cpuid_query_id()`. Thus, you can simply mimic what is done for the `intel64` family by inserting lines such as:
      ```c
      #ifdef BLIS_FAMILY_INTELAVX
          id = bli_cpuid_query_id();
      #endif
      ```
      This results in `bli_cpuid_query_id()` being called, which will return the `arch_t` ID value corresponding to the hardware detected by `CPUID`. (If your configuration family does not consist of x86_64 architectures, then you'll need some other heuristic to determine how to choose the correct sub-configuration at runtime. When in doubt, please [open an issue](https://github.com/flame/blis/issues) to begin a dialogue with developers.)



3. _**Update the configuration registry.**_ The last step is to update the `config_registry` file so that it defines the new family. Since we want the family to include only Intel sub-configurations that support AVX, we would add the following line:
   ```
   intelavx: haswell sandybridge
   ```
   Notice that we left out the Core2-based `penryn` sub-configuration since it targets hardware that only supports SSE vector instructions.


## Adding a new sub-configuration

Adding support for a new-subconfiguration to BLIS is similar to adding support for a family, though there are a few additional steps. Throughout this section, we will use the `knl` (Knight's Landing) configuration as an example to illustrate the typical changes necessary to various files in BLIS.



1. _**Create and populate the family directory.**_ First, we must create a directory in `config` that corresponds to the new sub-configuration.
   ```
   $ mkdir config/knl
   $ ls config
   amd64      cortexa15  excavator  intel64  old         power7       template
   bgq        cortexa57  generic    knc      penryn      sandybridge  zen
   bulldozer  cortexa9   haswell    knl      piledriver  steamroller
   ```
   We also need to create `bli_cntx_init_knl.c`, `bli_family_intelavx.h`, and `make_defs.mk` files inside our new sub-directory. Since they will be very similar to those of the `haswell` sub-configuration's files, we can copy those files over and then modify them accordingly:
   ```
   $ cp config/haswell/bli_cntx_init_haswell.c config/knl/bli_cntx_init_knl.c
   $ cp config/haswell/bli_family_haswell.h config/knl/bli_family_knl.h
   $ cp config/haswell/make_defs.mk config/knl/
   ```
   First, we update the configuration name inside of `make_defs.mk`:
   ```
   THIS_CONFIG    := knl
   ```
   and while we're editing the file, we can make any other changes to compiler flags we wish (if any). Similarly, the `bli_family_knl.h` header file should be updated as needed. Since the number of vector registers and the vector register size on `knl` differ from the defaults, we must explicitly set them. (The role of these parameters was explained in a [previous section](ConfigurationHowTo.md#bli_family_h).) Furthermore, provided that a macro `BLIS_NO_HBWMALLOC` is not set, we use a different implementation of `malloc()` and `free()` and `#include` that implementation's header file.
   ```c
   #define BLIS_SIMD_MAX_NUM_REGISTERS  32
   #define BLIS_SIMD_MAX_SIZE           64

   #ifdef BLIS_NO_HBWMALLOC
     #include <stdlib.h>
     #define BLIS_MALLOC_POOL  malloc
     #define BLIS_FREE_POOL    free
   #else
     #include <hbwmalloc.h>
     #define BLIS_MALLOC_POOL  hbw_malloc
     #define BLIS_FREE_POOL    hbw_free
   #endif
   ```
   Finally, we update `bli_cntx_init_knl.c` to initialize the context with the appropriate kernel function pointers and blocksize values. The functions used to perform this initialization are explained in [an earlier section](ConfigurationHowTo.md#bli_cntx_init_c).



2. _**Add support within the framework source code.**_ Next, we need to update the BLIS framework source code so that the new sub-configuration is recognized and supported. Sub-configurations require updates to four files--six if hardware detection logic is added.

   * **`frame/include/bli_type_defs.h`**. First, we need to define an ID to associate with the microarchitecture for which we are adding support. All microarchitecture type IDs are defined in [bli_type_defs.h](https://github.com/flame/blis/blob/master/frame/include/bli_type_defs.h) as an enumerated type that we `typedef` to `arch_t`. To support `knl`, we add a new enumerated type value `BLIS_ARCH_KNL`:
      ```c
      typedef enum
      {
          BLIS_ARCH_KNL,
          BLIS_ARCH_KNC,
          BLIS_ARCH_HASWELL,
          BLIS_ARCH_SANDYBRIDGE,
          BLIS_ARCH_PENRYN,

          BLIS_ARCH_ZEN,
          BLIS_ARCH_EXCAVATOR,
          BLIS_ARCH_STEAMROLLER,
          BLIS_ARCH_PILEDRIVER,
          BLIS_ARCH_BULLDOZER,

          BLIS_ARCH_CORTEXA57,
          BLIS_ARCH_CORTEXA15,
          BLIS_ARCH_CORTEXA9,

          BLIS_ARCH_POWER7,
          BLIS_ARCH_BGQ,

          BLIS_ARCH_GENERIC,

          BLIS_NUM_ARCHS

      } arch_t;
      ```
      Notice that the total number of `arch_t` values, `BLIS_NUM_ARCHS`, is updated automatically.



   * **`frame/include/bli_gentconf_macro_defs.h`**. We must also update the macro which automatically generates code which
   should be executed for each enabled sub-configuration. This macro update requires changes in two places: first we must conditionally define a
   macro for our new sub-configuration, and then we can invoke (call) that macro from the generic `INSERT_GENTCONF` macro. For `knl`, the
   first, sub-configuration-specific macro takes the form,
      ```c
      // -- KNL microarchitecture --
      #ifdef BLIS_CONFIG_KNL
      #define INSERT_GENTCONF_KNL GENTCONF( KNL, knl )
      #else
      #define INSERT_GENTCONF_KNL
      #endif
      ```
      Note the upper-case `KNL` tag which is used in various pre-defined macros such as `BLIS_CONFIG_KNL`, and the lower-case
      tag `knl` which is used in generating function names such as `bli_cntx_init_knl_ref`. The second modification to make is
      to add a call to this macro from `INSERT_GENTCONF`,
      ```c
      #define INSERT_GENTCONF \
      ...
      INSERT_GENTCONF_KNL \
      ...
      ```
      This will automatically handle most code fragments which depend on a specific sub-configuration, such as creating
      reference contexts in the global kernel structure.

   * **`frame/include/bli_arch_config.h`**. This file must be modified by adding an `#include` to the `bli_family_knl.h`
   header file, just as we would if we were adding support for an umbrella family:
      ```c
      #ifdef BLIS_FAMILY_KNL
      #include "bli_family_knl.h"
      #endif
      ```
      As before with umbrella families, the `BLIS_FAMILY_KNL` macro is automatically defined by the build system for whatever family was targeted by `configure`. (That is, if the user runs `./configure foobar`, the C preprocessor macro `BLIS_FAMILY_FOOBAR` will be defined.)



   * **`frame/base/bli_arch.c`**. This file must be updated so that `bli_arch_query_id()` returns the correct `arch_t` architecture ID value to the caller. `bli_arch_query_id()` is called when the framework is trying to choose which sub-configuration to use at runtime. When adding support for a sub-configuration as a singleton family, this amounts to adding a block of code such as:
      ```c
      #ifdef BLIS_FAMILY_KNL
          id = BLIS_ARCH_KNL;
      #endif
      ```
      The `BLIS_FAMILY_KNL` macro is automatically `#defined` by the build system if the `knl` sub-configuration was targeted directly (as a singleton family) at configure-time. Other ID values are returned only if their respective family macros are defined. (Recall that only one family is ever enabled at time.) If, however, the `knl` sub-configuration was enabled indirectly via an umbrella family, `bli_arch_query_id()` will return the `arch_t` ID value via the lines similar to the following:
      ```c
      #ifdef BLIS_FAMILY_INTEL64
          id = bli_cpuid_query_id();
      #endif
      #ifdef BLIS_FAMILY_AMD64
          id = bli_cpuid_query_id();
      #endif
      ```
      Supporting runtime detection of `knl` microarchitectures requires adding `knl` support to `bli_cpuid_query_id()`, which is addressed in the next step (`bli_cpuid.c`).
      Before we finish editing the `bli_arch.c` file, we need to add a string label to the static array `config_name`:
      ```c
      static char* config_name[ BLIS_NUM_ARCHS ] =
      {
          "knl",
          "knc",
          "haswell",
          "sandybridge",
          "penryn",

          "zen",
          "excavator",
          "steamroller",
          "piledriver",
          "bulldozer",

          "cortexa57",
          "cortexa15",
          "cortexa9",

          "power7",
          "bgq",

          "generic"
      };
      ```
      This array is used by `bli_arch_string()` when mapping `arch_t` values to the strings associated with that architecture ID. Because the `arch_t` value is used as the index of each string, **the relative order of the strings in this array is important**. Be sure to insert the new string (in our case, `"knl"`) at the **same relative location** as the `arch_t` value inserted in `bli_type_defs.h`. This will ensure that each `arch_t` value will map to its corresponding string in the `config_name` array.



   * **`frame/base/bli_cpuid.c`**. To support the aforementioned runtime microarchitecture detection, the function `bli_cpuid_query_id()`, defined in [bli_cpuid.c](https://github.com/flame/blis/blob/master/frame/base/bli_cpuid.c), will need to be updated. Specifically, we need to insert logic that will detect the presence of the new hardware based on the results of the `CPUID` instruction (assuming the new microarchitecture belongs to the x86_64 architecture family). For example, when support for `knl` was added, this entailed adding the following code block to `bli_cpuid_query_id()`:
      ```c
      #ifdef BLIS_CONFIG_KNL
          if ( bli_cpuid_is_knl( family, model, features ) )
              return BLIS_ARCH_KNL;
      #endif
      ```
      Additionally, we had to define the function `bli_cpuid_is_knl()`, which checks for various processor features known to be present on `knl` systems and returns a boolean `TRUE` if all relevant feature checks are satisfied by the hardware. Note that the order in which we check for the sub-configurations is important. We must check for microarchitectural matches from most recent to most dated. This prevents an older sub-configuration from being selected on newer hardware when a newer sub-configuration would have also matched.



   * **`frame/base/bli_cpuid.h`**. After defining the function `bli_cpuid_is_knl()`, we must also update [bli_cpuid.h](https://github.com/flame/blis/blob/master/frame/base/bli_cpuid.h) to contain a prototype for the function.



3. _**Update the configuration registry.**_ Lastly, we update the `config_registry` file so that it defines the new sub-configuration. For example, if we want to define a sub-configuration called `knl` that used only `knl` kernels, we would add the following line:
   ```
   knl: knl
   ```
   If, when defining `bli_cntx_init_knl()`, we referenced kernels from a non-native kernel set--say, those of `haswell`--in addition to `knl`-specific kernels, we would need to explicitly pull in both `knl` and `haswell` kernel sets:
   ```
   knl: knl/knl/haswell
   ```


## Further Development Topics

### Querying the current configuration

If you are ever unsure which configuration is "active", or the configuration parameters that were specified (or implied by default) at configure-time, simply run:

```
$ make showconfig
configuration family:  intel64
sub-configurations:    haswell sandybridge penryn
requisite kernels:     haswell sandybridge penryn
kernel-to-config map:  haswell:haswell penryn:penryn sandybridge:sandybridge
-----------------------
BLIS version string:   0.2.2-73
install prefix:        /home/field/blis
debugging status:      off
multithreading status: no
enable BLAS API?       yes
enable CBLAS API?      no
build static library?  yes
build shared library?  no
```

This will tell you the current configuration name, the [configuration registry lists](ConfigurationHowTo.md#printing-the-configuration-registry-lists), as well as other information stored by `configure` in the `config.mk` file.



### Header dependencies

Due to the way the BLIS framework handles header files, **any** change to **any** header file will result in the entire library being rebuilt. This policy is in place mostly out of an abundance of caution. If two or more files use definitions in a header that is modified, and one or more of those files somehow does not get recompiled to reflect the updated definitions, you could end up sinking hours of time trying to track down a bug that didn't ever need to be an issue to begin with. Thus, to prevent developers (including the framework developer(s)) from shooting themselves in the foot with this problem, the BLIS build system recompiles **all** object files if any header file is touched. We apologize for the inconvenience this may cause.



### Still have questions?

If you have further questions about BLIS configurations, please do not hesitate to contact the BLIS developer community. To do so, simply join and post to the [blis-devel](http://groups.google.com/group/blis-devel) mailing list.
***
