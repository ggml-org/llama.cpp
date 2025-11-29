## Contents

* **[Introduction](Sandboxes.md#introduction)**
* **[Enabling a sandbox](Sandboxes.md#enabling-a-sandbox)**
* **[Sandbox rules](Sandboxes.md#sandbox-rules)**
* **[Caveats](Sandboxes.md#caveats)**
* **[Known issues](Sandboxes.md#known-issues)**
* **[Conclusion](Sandboxes.md#conclusion)**


## Introduction

This file briefly describes the requirements for building a custom BLIS
*sandbox*.

Simply put, a sandbox in BLIS provides an alternative implementation to the
`gemm` operation.

To get a little more specific, a sandbox provides an alternative implementation
to the function `bli_gemm_ex()`, which is the
[expert interface](BLISObjectAPI.md##basic-vs-expert-interfaces) for calling the
[object-based API](BLISObjectAPI.md#gemm) for the `gemm` operation.

Why sandboxes? Sometimes you want to experiment with tweaks or changes to
the `gemm` operation, but you want to do so in a simple environment rather than
the highly macroized and refactored (and somewhat obfuscated) code of the
core framework. By building a BLIS sandbox, you can experiment (within limits)
and still benefit from BLIS's existing build system, testsuite, and toolbox of
utility functions.

## Enabling a sandbox

To enable a sandbox at configure-time, you simply specify it as an option to
`configure`. Either of the following usages are accepted:
```
$ ./configure --enable-sandbox=gemmlike auto
$ ./configure -s gemmlike auto
```
Here, we tell `configure` that we want to use the `gemmlike` sandbox, which
corresponds to a sub-directory of `sandbox` named `gemmlike`. (Reminder: the
`auto` argument is the configuration target and thus unrelated to
sandboxes.)

NOTE: Using your own sandbox implementation means that BLIS will call your
sandbox for *all* problem sizes and shapes, for *all* datatypes supported
by BLIS. If you intend to only implement a subset of this functionality
within your sandbox, you should be sure to redirect execution back into
the core framework for the parts that you don't wish to reimplement yourself.

As `configure` runs, you should get output that includes lines
similar to:
```
configure: configuring for alternate gemm implementation:
configure:   sandbox/gemmlike
```
And when you build BLIS, the last files to be compiled will be the source
code in the specified sandbox:
```
Compiling obj/haswell/sandbox/gemmlike/bls_gemm.o ('haswell' CFLAGS for sandboxes)
Compiling obj/haswell/sandbox/gemmlike/bls_gemm_bp_var1.o ('haswell' CFLAGS for sandboxes)
...
```
That's it! After the BLIS library is built, it will contain your chosen
sandbox's implementation of `bli_gemm_ex()` instead of the default BLIS
implementation.

## Sandbox rules

Like any civilized sandbox, there are rules for playing here. Please follow
these guidelines for the best sandbox developer experience.

1. Don't bother worrying about makefiles. We've already taken care of the
boring/annoying/headache-inducing build system stuff for you. :) By configuring
BLIS with a sandbox enabled, `make` will scan your sandbox directory and compile
all of its source code using similar compilation rules as were used for the rest
of the framework. In addition, the compilation command line will automatically
contain one `-I<includepath>` option for every subdirectory in your sandbox,
so it doesn't matter where in your sandbox you place your header files. They
will be found!

2. Your sandbox must be written in C99 or C++11. If you write your sandbox in
C++11, you must use one of the BLIS-approved file extensions for your source
files (`.cc`, `.cpp`, `.cxx`) and your header files (`.hh`, `.hpp`, `.hxx`).
Note that `blis.h` already contains all of its definitions inside of an
`extern "C"` block, so you should be able to `#include "blis.h"` from your
C++11 source code without any issues.

3. All of your code to replace BLIS's default implementation of `bli_gemm_ex()`
should reside in the named sandbox directory, or some directory therein.
(Obviously.) For example, the "gemmlike" sandbox is located in
`sandbox/gemmlike`. All of the code associated with this sandbox will be
contained within `sandbox/gemmlike`. Note that you absolutely *may* include
additional code and interfaces within the sandbox, if you wish -- code and
interfaces that are not directly or indirectly needed for satisfying the
the "contract" set forth by the sandbox (i.e., including a local definition
of`bli_gemm_ex()`).

4. The *only* header file that is required of your sandbox is `bli_sandbox.h`.
It must be named `bli_sandbox.h` because `blis.h` will `#include` this file
when the sandbox is enabled at configure-time. That said, you will probably
want to keep the file empty. Why require a file that is supposed to be empty?
Well, it doesn't *have* to be empty. Anything placed in this file will be
folded into the flattened (monolithic) `blis.h` at compile-time. Therefore,
you should only place things (e.g. prototypes or type definitions) in
`bli_sandbox.h` if those things would be needed at compile-time by:
(a) the BLIS framework itself, or
(b) an *application* that calls your sandbox-enabled BLIS library.
Usually, neither of these situations will require any of your local definitions
since those local definitions are only needed to define your sandbox
implementation of `bli_gemm_ex()`, and this function is already prototyped by
BLIS. *But if you are adding additional APIs and/or operations to the sandbox
that are unrelated to `bli_gemm_ex()`, then you'll want to `#include` those
function prototypes from within `bli_sandbox.h`*

5. Your definition of `bli_gemm_ex()` should be the **only function you define**
in your sandbox that begins with `bli_`. If you define other functions that
begin with `bli_`, you risk a namespace collision with existing framework
functions. To guarantee safety, please prefix your locally-defined sandbox
functions with another prefix. Here, in the `gemmlike` sandbox, we use the prefix
`bls_`. (The `s` is for sandbox.) Also, please avoid the prefix `bla_` since that
prefix is also used in BLIS for BLAS compatibility functions.

If you follow these rules, you will be much more likely to have a pleasant
experience integrating your BLIS sandbox into the larger framework.

## Caveats

Notice that the BLIS sandbox is not all-powerful. You are more-or-less stuck
working with the existing BLIS infrastructure.

For example, with a BLIS sandbox you **can** do the following kinds of things:
- use a different `gemm` algorithmic partitioning path than the default
  Goto-like algorithm;
- experiment with different implementations of `packm` (not just `packm`
  kernels, which can already be customized within each sub-configuration);
- try inlining your functions manually;
- pivot away from using `obj_t` objects at higher algorithmic level (such as
  immediately after calling `bli_gemm_ex()`) to try to avoid some overhead;
- create experimental implementations of new BLAS-like operations (provided
  that you also provide an implementation of `bli_gemm_ex()`).

You **cannot**, however, use a sandbox to do the following kinds of things:
- define new datatypes (half-precision, quad-precision, short integer, etc.)
  and expect the rest of BLIS to "know" how to handle them;
- use a sandbox to replace the default implementation of a different level-3
  operation, such as Hermitian rank-k update;
- change the existing BLIS APIs (typed or object);
- remove support for one or more BLIS datatypes (to cut down on library size,
  for example).

Another important limitation is the fact that the build system currently uses
"framework `CFLAGS`" when compiling the sandbox source files. These are the same
`CFLAGS` used when compiling general framework source code,
```
# Example framework CFLAGS used by 'haswell' sub-configuration
-O3 -Wall -Wno-unused-function -Wfatal-errors -fPIC -std=c99
-D_POSIX_C_SOURCE=200112L -I./include/haswell -I./frame/3/
-I./frame/1m/ -I./frame/1f/ -I./frame/1/ -I./frame/include
-DBLIS_VERSION_STRING=\"0.3.2-51\"
```
which are likely more general-purpose than the `CFLAGS` used for, say,
optimized kernels or even reference kernels.
```
# Example optimized kernel CFLAGS used by 'haswell' sub-configuration
-O3 -mavx2 -mfma -mfpmath=sse -march=core-avx2 -Wall -Wno-unused-function
-Wfatal-errors -fPIC -std=c99 -D_POSIX_C_SOURCE=200112L -I./include/haswell
-I./frame/3/ -I./frame/1m/ -I./frame/1f/ -I./frame/1/ -I./frame/include
-DBLIS_VERSION_STRING=\"0.3.2-51\"
```
(To see precisely which flags are being employed for any given file, enable
verbosity at compile-time via `make V=1`.) Compiling sandboxes with these more
versatile `CFLAGS` compiler options means that we only need to compile one
instance of each sandbox source file, even when targeting multiple
configurations (for example, via `./configure x86_64`). However, it also means
that sandboxes are not ideal for microkernels, as they sometimes need additional
compiler flags not included in the set used for framework `CFLAGS` in order to
yield the highest performance. If you have a new microkernel you would like to
use within a sandbox, you can always develop it within a sandbox. However,
once it is stable and ready for use by others, it's best to formally register
the kernel(s) along with a new configuration, which will allow you to specify
kernel-specific compiler flags to be used when compiling your microkernel.
Please see the
[Configuration Guide](ConfigurationHowTo)
for more details, and when in doubt, please don't be shy about seeking
guidance from BLIS developers by opening a
[new issue](https://github.com/flame/blis/issues) or sending a message to the
[blis-devel](http://groups.google.com/d/forum/blis-devel) mailing list.

Notwithstanding these limitations, hopefully you still find BLIS sandboxes
useful!

## Known issues

* **Mixed datatype support.** Unless you *really* know what you are doing, you
should probably disable mixed datatype support when using a sandbox. (Mixed
datatype support can be disabled by configuring with `--disable-mixed-dt`.) The
BLIS testsuite is smart enough to verify that you've configured BLIS with mixed
datatype support before allowing you to test with mixed domains/precisions
enabled in `input.general`. However, if those options *are* enabled and BLIS was
built with mixed datatype support, then BLIS assumes that the implementation of
`gemm` will support mixing of datatypes. BLIS *must* assume this, because
there's no way for it to confirm at runtime that an implementation was written
to support mixing datatypes. Note that even the `gemmlike` sandbox included with
BLIS does not support mixed-datatype computation.

## Conclusion

If you encounter any problems, or are really bummed-out that `gemm` is the
only operation for which you can provide a sandbox implementation, please open
a new [issue on GitHub](https://github.com/flame/blis/issues).

If you are unsure about how something works, you can still open an issue. Or, you
can send a message to
[blis-devel](https://groups.google.com/d/forum/blis-devel) mailing list.

Happy sandboxing!

