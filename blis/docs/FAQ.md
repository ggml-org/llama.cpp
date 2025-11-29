## Introduction

Here we attempt to provide some frequently-asked questions about the BLIS framework
project, as well as those we think a new user or developer might ask. If you do not see the answer to your question here, please join and post your question to one of the [BLIS mailing lists](https://github.com/flame/blis#discussion).

## Contents

  * [Why did you create BLIS?](FAQ.md#why-did-you-create-blis)
  * [Why should I use BLIS instead of GotoBLAS / OpenBLAS / ATLAS / MKL / ESSL / ACML / Accelerate?](FAQ.md#why-should-i-use-blis-instead-of-gotoblas--openblas--atlas--mkl--essl--acml--accelerate)
  * [How is BLIS related to FLAME / libflame?](FAQ.md#how-is-blis-related-to-flame--libflame)
  * [What is the difference between BLIS and the AMD fork of BLIS found in AOCL?](FAQ.md#what-is-the-difference-between-blis-and-the-amd-fork-of-blis-found-in-aocl)
  * [Who do I contact if I have a question about the AMD version of BLIS?](FAQ.md#who-do-i-contact-if-i-have-a-question-about-the-amd-version-of-blis)
  * [Does BLIS automatically detect my hardware?](FAQ.md#does-blis-automatically-detect-my-hardware)
  * [I understand that BLIS is mostly a tool for developers?](FAQ.md#i-understand-that-blis-is-mostly-a-tool-for-developers)
  * [How do I link against BLIS?](FAQ.md#how-do-i-link-against-blis)
  * [Must I use git? Can I download a tarball?](FAQ.md#must-i-use-git-can-i-download-a-tarball)
  * [What is a microkernel?](FAQ.md#what-is-a-microkernel)
  * [What is a macrokernel?](FAQ.md#what-is-a-macrokernel)
  * [What is a context?](FAQ.md#what-is-a-context)
  * [I am used to thinking in terms of column-major/row-major storage and leading dimensions. What is a "row stride" / "column stride"?](FAQ.md#im-used-to-thinking-in-terms-of-column-majorrow-major-storage-and-leading-dimensions-what-is-a-row-stride--column-stride)
  * [I'm somewhat new to this matrix stuff. Can you remind me, what is the difference between a matrix row and a matrix column?](FAQ.md#im-somewhat-new-to-this-matrix-stuff-can-you-remind-me-what-is-the-difference-between-a-matrix-row-and-a-matrix-column)
  * [Why does BLIS have vector (level-1v) and matrix (level-1m) variations of most level-1 operations?](FAQ.md#why-does-blis-have-vector-level-1v-and-matrix-level-1m-variations-of-most-level-1-operations)
  * [What does it mean when a matrix with general stride is column-tilted or row-tilted?](FAQ.md#what-does-it-mean-when-a-matrix-with-general-stride-is-column-tilted-or-row-tilted)
  * [I am not really interested in all of these newfangled features in BLIS. Can I just use BLIS as a BLAS library?](FAQ.md#im-not-really-interested-in-all-of-these-newfangled-features-in-blis-can-i-just-use-blis-as-a-blas-library)
  * [What about CBLAS?](FAQ.md#what-about-cblas)
  * [Can I call the native BLIS API from Fortran-77/90/95/2000/C++/Python?](FAQ.md#can-i-call-the-native-blis-api-from-fortran-7790952000cpython)
  * [Do I need to call initialization/finalization functions before being able to use BLIS from my application?](FAQ.md#do-i-need-to-call-initializationfinalization-functions-before-being-able-to-use-blis-from-my-application)
  * [Does BLIS support multithreading?](FAQ.md#does-blis-support-multithreading)
  * [Does BLIS support NUMA environments?](FAQ.md#does-blis-support-numa-environments)
  * [Does BLIS work with GPUs?](FAQ.md#does-blis-work-with-gpus)
  * [Does BLIS work on (some architecture)?](FAQ.md#does-blis-work-on-some-architecture)
  * [What about distributed-memory parallelism?](FAQ.md#what-about-distributed-memory-parallelism)
  * [Can I build BLIS on Mac OS X?](FAQ.md#can-i-build-blis-on-mac-os-x)
  * [Can I build BLIS on Windows?](FAQ.md#can-i-build-blis-on-windows)
  * [Can I build BLIS as a shared library?](FAQ.md#can-i-build-blis-as-a-shared-library)
  * [Can I use the mixed domain / mixed precision support in BLIS?](FAQ.md#can-i-use-the-mixed-domain--mixed-precision-support-in-blis)
  * [Who is involved in the project?](FAQ.md#who-is-involved-in-the-project)
  * [Who funded the development of BLIS?](FAQ.md#who-funded-the-development-of-blis)
  * [I found a bug. How do I report it?](FAQ.md#i-found-a-bug-how-do-i-report-it)
  * [How do I request a new feature?](FAQ.md#how-do-i-request-a-new-feature)
  * [I'm a developer and I'd like to study the way matrix multiplication is implemented in BLIS. Where should I start?](FAQ.md#im-a-developer-and-id-like-to-study-the-way-matrix-multiplication-is-implemented-in-blis-where-should-i-start)
  * [Where did you get the photo for the BLIS logo / mascot?](FAQ.md#where-did-you-get-the-photo-for-the-blis-logo--mascot)

### Why did you create BLIS?

Initially, BLIS was conceived as simply "BLAS with a more flexible interface". The original BLIS was written as a wrapper layer around BLAS that allowed generalized matrix storage (i.e., separate row and column strides). We also took the opportunity to implement some complex domain features that were missing from the BLAS (mostly related to conjugating input operands). This "proto-BLIS" was deployed in [libflame](http://shpc.ices.utexas.edu/libFLAME.html) to facilitate cleaner implementations of some LAPACK-level operations.

Over time, we wanted more than just a more flexible interface; we wanted an entire framework from which we could build operations in the BLAS as well as those not present within the BLAS. After this new BLIS framework was created, it turned out that the interface improvements were much less interesting (albeit still of consequence) than some of the framework's other features, and the fact that it allowed developers to rapidly instantiate new BLAS libraries by optimizing only a small amount of code.

### Why should I use BLIS instead of GotoBLAS / OpenBLAS / ATLAS / MKL / ESSL / ACML / Accelerate?

BLIS has numerous advantages to existing BLAS implementations. Many of these advantages are summarized on the [BLIS
homepage](https://github.com/flame/blis#key-features). But here are a few reasons one might choose BLIS over some other implementation of BLAS:
  * BLIS facilitates high performance while remaining very portable. BLIS isolates performance-sensitive code to a microkernel which contains only one loop and which, when optimized, accelerates virtually all level-3 operations. Thus, BLIS serves as a powerful tool for quickly instantiating BLAS on new or experimental hardware architectures, as well as a flexible "laboratory" in which to conduct research and experiments.
  * BLIS provides robust multithreading support, allowing symmetric multicore/many-core parallelism via either OpenMP or POSIX threads. It also computes proper load balance for structured matrix subpartitions, regardless of the location of the diagonal, or whether the subpartition is lower- or upper-stored.
  * BLIS supports a superset of BLAS functionality, providing operations omitted from the BLAS as well as some complex domain support that is missing in BLAS operations. BLIS is especially useful to researchers who need to develop and prototype new BLAS-like operations that do not exist in the BLAS.
  * BLIS is backwards compatible with BLAS. BLIS contains a BLAS compatibility layer that allows an application to treat BLIS as if it were a traditional BLAS library.
  * BLIS supports generalized matrix storage, which can be used to express column-major, row-major, and general stride storage.
  * BLIS supports mixed-datatype computation for general matrix multiplication `gemm`, and does so while holding the impact on performance to a relative minimum.
  * BLIS is free software, available under a [new/modified/3-clause BSD license](http://opensource.org/licenses/BSD-3-Clause).

### How is BLIS related to FLAME / `libflame`?

As explained [above](FAQ.md#why-did-you-create-blis?), BLIS was initially a layer within `libflame` that allowed more convenient interfacing to the BLAS. So in some ways, BLIS is a spin-off project. Prior to developing BLIS, [its primary author](http://www.cs.utexas.edu/users/field/) worked as the primary maintainer of `libflame`. If you look closely, you can also see that the design of BLIS was influenced by some of the more useful and innovative aspects of `libflame`, such as internal object abstractions and control trees.

Note that various members of the [SHPC research group](http://shpc.ices.utexas.edu/people.html) and its [collaborators](http://shpc.ices.utexas.edu/collaborators.html) routinely provide insight, feedback, and also contribute code (especially kernels) to the BLIS project.

### What is the difference between BLIS and the AMD fork of BLIS found in AOCL?

BLIS, also known as "vanilla BLIS" or "upstream BLIS," is maintained by its [original developer](https://github.com/fgvanzee) (with the [support of others](http://shpc.ices.utexas.edu/collaborators.html)) in the [Science of High-Performance Computing](http://shpc.ices.utexas.edu/) (SHPC) group within the [The Oden Institute for Computational Engineering and Sciences](http://www.oden.utexas.edu/) at [The University of Texas at Austin](http://www.utexas.edu/). In 2015, [AMD](https://www.amd.com/) reorganized many of their software library efforts around existing open source projects. BLIS was chosen as the basis for their [CPU BLAS library](https://developer.amd.com/amd-aocl/blas-library/), and an AMD-maintained [fork of BLIS](https://github.com/amd/blis) was established.

AMD BLIS sometimes contains certain optimizations specific to AMD hardware. Many of these optimizations are (eventually) merged back into upstream BLIS. However, for various reasons, some changes may remain unique to AMD BLIS for quite some time. Thus, if you want the latest optimizations for AMD hardware, feel free to try AMD BLIS. However, please note that neither The University of Texas at Austin nor BLIS's developers can endorse or offer direct support for any outside fork of BLIS, including AMD BLIS.

### Who do I contact if I have a question about the AMD version of BLIS?

For questions or support regarding [AMD's fork of BLIS](https://github.com/amd/blis), please contact the [AMD Optimizing CPU Libraries](https://developer.amd.com/amd-aocl/) group at aoclsupport@amd.com.

### Does BLIS automatically detect my hardware?

On certain architectures (most notably x86_64), yes. In order to use auto-detection, you must specify `auto` as your configuration when running `configure` (Please see the BLIS [Build System](BuildSystem.md) guide for more info.) A runtime detection option is also available. (Please see the [Configuration Guide](ConfigurationHowTo.md) for a comprehensive walkthrough.)

If automatic hardware detection is requested at configure-time and the build process does not recognize your architecture, the `generic` configuration is selected.

### I understand that BLIS is mostly a tool for developers?

It is certainly the case that BLIS began as a tool targeted at developers. In order to achieve high performance, BLIS requires that hand-coded kernels and microkernels be written and referenced in a valid [BLIS configuration](ConfigurationHowTo.md). These components are usually written by developers and then included within BLIS for use by others.

The good news, however, is that BLIS has matured to the point where end-users can use it too! Once the aforementioned kernels are integrated into BLIS, they can be used without any developer-level knowledge, and many kernels have already been added! Usually, `./configure auto; make; make install` is sufficient for the typical users with typical hardware.

### How do I link against BLIS?

Linking against BLIS is easy! Most people can link to it as if it were a generic BLAS library. Please see the [Linking against BLIS](BuildSystem.md#linking-against-blis) section of the [Build System](BuildSystem.md) guide.

### Must I use git? Can I download a tarball?

We **strongly encourage** you to obtain the BLIS source code by cloning a `git` repository (via the [git clone](BuildSystem.md#obtaining-blis) command). The reason for this is that it will allow you to easily update your local copy of BLIS by executing `git pull`.

Tarballs and zip files may be obtained from the [releases](https://github.com/flame/blis/releases) page.

### What is a microkernel?

The microkernel (usually short for "`gemm` microkernel") is the basic unit of level-3 (matrix-matrix) computation within BLIS. It consists of one loop, where each iteration performs a very small outer product to update a very small matrix. The microkernel is typically the only piece of code that must be carefully optimized (via vector intrinsics or assembly code) to enable high performance in most of the level-3 operations such as `gemm`, `hemm`, `herk`, and `trmm`.

For a more thorough explanation of the microkernel and its role in the overall level-3 computations, please read our [ACM TOMS papers](https://github.com/flame/blis#citations). For API and technical reference, please see the [gemm microkernel section](KernelsHowTo.md#gemm-microkernel) of the BLIS [Kernels Guide](KernelsHowTo.md).

### What is a macrokernel?

The macrokernels are portable codes within the BLIS framework that implement relatively small subproblems within an overall level-3 operation. The overall problem (say, general matrix-matrix multiplication, or `gemm`) is partitioned down, according to cache blocksizes, such that its `A` and `B` operands are (1) a suitable size and (2) stored in a special packed format. At that time, the macrokernel is called. The macrokernel is implemented as two loops around the microkernel.

The macrokernels, along with the microkernel that they call, correspond to the so-called "inner kernels" (or simply "kernels") that formed the fundamental unit of computation in Kazushige Goto's GotoBLAS (and now in the successor library, OpenBLAS).

For more information on macrokernels, please read our [ACM TOMS papers](https://github.com/flame/blis#citations).

### What is a context?

As of 0.2.0, BLIS contains a new infrastructure for communicating runtime information (such as kernel addresses and blocksizes) from the highest levels of code all the way down the function stack, even into the kernels themselves. This new data structure is called a *context* (defined in code as a `cntx_t` type), and together with its API it helped us clean up some hacks and other awkwardness that existed in BLIS prior to 0.2.0. Contexts also lay the groundwork for managing kernels and related kernel information at runtime.

If you are a kernel developer, you can usually ignore the `cntx_t*` argument that is passed into each kernel, since the kernels already inherently "know" this information (such as register blocksizes). And if you are a user, and the function you want to call takes a `cntx_t*` argument, you can safely pass in `NULL` and BLIS will automatically build a suitable context for you at runtime.

### I'm used to thinking in terms of column-major/row-major storage and leading dimensions. What is a "row stride" / "column stride"?

Traditional BLAS assumes that matrices are stored in column-major order (or, as we often say, matrices that are "column-stored"), where a leading dimension measures the distance from one element to the next element in the same row. But column-major order is really just a special case of BLIS's more generalized storage scheme.

In generalized storage, we have a row stride and a column stride. The row stride measures the distance in memory between rows (within a single column) while the column stride measures the distance between columns (within a single row). Column-major storage corresponds to the situation where the row stride equals 1. Since the row stride is unit, you only have to track the column stride (i.e., the leading dimension). Similarly, in row-major order, the column stride is equal to 1 and only the row stride must be tracked.

BLIS also supports situations where both the row stride and column stride are non-unit. We call this situation "general stride".

### I'm somewhat new to this matrix stuff. Can you remind me, what is the difference between a matrix row and a matrix column?

Of course! (BLIS's primary author remembers what it was like to get columns and rows confused.)

Matrix columns consist of elements that are vertically aligned. Matrix rows consist of elements that are horizontally aligned. (One way to remember this distinction is that real-life columns are vertical structures that hold up buildings. A row of seats in a stadium, by contrast, is horizontal to the ground.)

Furthermore, it is helpful to know that the number of rows in a matrix constitutes its so-called *m* dimension, and the number of columns constitutes its *n* dimension.

Matrix dimension are always stated as *m x n*: the number of rows *by* the number of columns.

So, a *3 x 4* matrix contains three rows (each of length four) and four columns (each of length three).

### Why does BLIS have vector (level-1v) and matrix (level-1m) variations of most level-1 operations?

At first glance, it might appear that an element-wise operation such as `copym` or `axpym` would be sufficiently general purpose to cover the cases where the operands are vectors. After all, an *m x 1* matrix can be viewed as a vector of length m and vice versa. But in BLIS, operations on vectors are treated slightly differently than operations on matrices.

If an application wishes to perform an element-wise operation on two objects, and the application calls a level-1m operation, the dimensions of those objects must be conformal, or "match up" (after any transposition implied by the object properties). This includes situations where one of the dimensions is unit.

However, if an application instead decides to perform an element-wise operation on two objects, and the application calls a level-1v operation, the dimension constraints are slightly relaxed. In this scenario, BLIS only checks that the vector *lengths* are equal. This allows for the vectors to have different orientations (row vs column) while still being considered conformal. So, you could perform a `copyv` operation to copy from an *m x 1* vector to a *1 x m* vector. A `copym` operation on such objects would not be allowed (unless it was executed with the source object containing an implicit transposition).

### What does it mean when a matrix with general stride is column-tilted or row-tilted?

When a matrix is stored with general stride, both the row stride and column stride (let's call them `rs` and `cs`) are non-unit. When `rs` < `cs`, we call the general stride matrix "column-tilted" because it is "closer" to being column-stored (than row-stored). Similarly, when `rs` > `cs`, the matrix is "row-tilted" because it is closer to being row-stored.

### I'm not really interested in all of these newfangled features in BLIS. Can I just use BLIS as a BLAS library?

Absolutely! Just link your application to BLIS the same way you would link to a BLAS library. For a simple linking example, see the [Linking to BLIS](KernelsHowTo.md#linking-to-blis) section of the BLIS [Build System](BuildSystem.md) guide.

### What about CBLAS?

BLIS also contains an optional CBLAS compatibility layer, which leverages the BLAS compatibility layer to help map CBLAS function calls to the corresponding functionality in BLIS. Once BLIS is built with CBLAS support, your application can access CBLAS prototypes via either `cblas.h` or `blis.h`. At the time of this writing, CBLAS support is disabled by default, so be sure to enable it at configure-time. Please see `./configure --help` for the syntax for enabling CBLAS.

### Can I call the native BLIS API from Fortran-77/90/95/2000/C++/Python?

In principle, BLIS's native (and BLAS-like) [typed API](BLISTypedAPI) can be called from Fortran. However, you must ensure that the size of the integer in BLIS is equal to the size of integer used by your Fortran program/compiler/environment. The size of BLIS integers is determined at configure-time. Please see `./configure --help` for the syntax for options related to integer sizes.

You may also want to confirm that your Fortran compiler doesn't perform any name-mangling of called functions or subroutines (such as with additional underscores beyond the single trailing underscore found in the BLAS APIs), and if so, take steps to disable this additional name-mangling. For example, if your source code calls `dgemm()` but your Fortran compiler name-mangles that call to `_dgemm_()` or `dgemm__()`, your program will fail to link against BLIS since BLIS only defines `dgemm_()`.

As for bindings to other languages, please contact the [blis-devel](http://groups.google.com/group/blis-devel) mailing list.

### Do I need to call initialization/finalization functions before being able to use BLIS from my application?

Originally, BLIS did indeed require the application to explicitly setup (initialize) various internal data structures via `bli_init()`. Likewise, calling `bli_finalize()` was recommended to cleanup (finalize) the library. However, since commit `9804adf` (circa December 2017), BLIS has implemented self-initialization. These explicit calls to `bli_init()` and `bli_finalize()` are no longer necessary, though experts may still use them in special cases to control the allocation and freeing of resources. This topic is discussed in the BLIS [typed API reference](BLISTypedAPI.md#initialization-and-cleanup).

### Does BLIS support multithreading?

Yes! BLIS supports multithreading (via OpenMP, POSIX threads, or HPX) for all of its level-3 operations. For more information on enabling and controlling multithreading, please see the [Multithreading](Multithreading.md) guide.

BLIS is also thread-safe so that you can call BLIS from threads within a multithreaded library or application. BLIS derives its thread-safety via unconditional use of features present in POSIX threads (pthreads). These pthreads features are employed for thread-safety regardless of whether BLIS is configured for OpenMP multithreading, pthreads multithreading, or single-threaded execution.

### Does BLIS support NUMA environments?

We have integrated some early foundational support for NUMA *development*, but currently BLIS will execute sub-optimally on NUMA systems. If you are interested in adapting BLIS to a NUMA architecture, please contact us via the [blis-devel](http://groups.google.com/group/blis-devel) mailing list.

### Does BLIS work with GPUs?

BLIS does not currently support graphical processing units (GPUs). However, others have applied the BLIS approach towards frameworks that provide BLAS-like functionality on GPUs. To see how NVIDIA's implementation compares to an analogous approach based on the principles that underlie BLIS, please see a paper by some of our collaborators, ["Implementing Strassen’s Algorithm with CUTLASS on NVIDIA Volta GPUs"](https://apps.cs.utexas.edu/apps/sites/default/files/tech_reports/GPUStrassen.pdf).

### Does BLIS work on _(some architecture)_?

Please see the BLIS [Hardware Support](HardwareSupport.md) guide for a full list of supported architectures. If your favorite hardware is not listed and you have the expertise, please consider developing your own kernels and sharing them with the project! We will, of course, gratefully credit your contribution.

### What about distributed-memory parallelism?

No. BLIS is a framework for sequential and shared-memory/multicore implementations of BLAS-like operations. If you need distributed-memory dense linear algebra implementations, we recommend the [Elemental](http://libelemental.org/) library.

### Can I build BLIS on Mac OS X?

BLIS was designed for use in a GNU/Linux environment. However, we've gone to great lengths to keep BLIS compatible with other UNIX-like systems as well, such as BSD and OS X. System software requirements for UNIX-like systems are discussed in the BLIS [Build System](BuildSystem.md) guide.

### Can I build BLIS on Windows?

If all you need is a Windows DLL of BLIS, you may be in luck! BLIS uses [AppVeyor](https://ci.appveyor.com/) to automatically produces dynamically-linked libraries, which are preserved on the site as "artifacts". To try it out, just visit the [BLIS AppVeyor page](https://ci.appveyor.com/project/shpc/blis/), click on the `LIB_TYPE=shared` link for the most recent build, and then click on "Artifacts". If you would like to provide us feedback, you may do so by [opening an issue](http://github.com/flame/blis/issues), or you can join the [blis-devel](http://groups.google.com/group/blis-devel) mailing list and send us a message.

If you want to build on Windows, there are two options:

1. MSVC ABI compatible DLL with clang

   If you want BLIS to be compatible with DLLs built by MSVC, you need to use `clang.exe` to build BLIS as BLIS does not support building with Visual Studio C compiler (``cl.exe``). To build BLIS, you need a recent clang from [LLVM](https://releases.llvm.org/download.html), an [MSYS2](https://www.msys2.org/) environment (for build tools like `sed`, `bash`), a Visual Studio 2015 or later environment (for C standard library) and Windows SDK.
   To build `BLIS`,
     * Activate the Visual Studio environment from a command prompt
       Run `call C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvarsall.bat x64`
     * Start the bash shell from the same command prompt. (Run `bash.exe`)
     * Run `export AR=llvm-ar AS=llvm-as RANLIB=echo CC=clang CXX=clang++`
     * Run `./configure --prefix=/c/blis/ --disable-static --enable-shared auto`
     * Run `make -j install`

2. MinGW DLL

   This is the easiest option to compile BLIS on windows, but the DLL might not be compatible with other programs compiled with MSVC. To build `BLIS`, install [MSYS2](https://www.msys2.org) and `mingw-w64` compilers. Then start a `bash` shell from MSYS2 and follow the instructions for the Linux build.

### Can I build BLIS as a shared library?

Yes. By default, most configurations output only a static library archive (e.g. `.a` file). However, you can also request a shared object (e.g. `.so` file), sometimes also called a "dynamically-linked" library. For information on enabling shared library output, simply run `./configure --help`.

### Can I use the mixed domain / mixed precision support in BLIS?

Yes! As of 5fec95b (circa October 2018), BLIS supports mixed-datatype (mixed domain and/or mixed precision) computation via the `gemm` operation. Documentation on utilizing this new functionality is provided via the [MixedDatatype.md](MixedDatatypes.md) document in the source distribution.

If this feature is important or useful to your work, we would love to hear from you. Please contact us via the [blis-devel](http://groups.google.com/group/blis-devel) mailing list and tell us about your application and why you need/want support for BLAS-like operations with mixed-domain/mixed-precision operands.

### Who is involved in the project?

Lots of people! For a full list of those involved, see the
[CREDITS](https://github.com/flame/blis/blob/master/CREDITS) file within the BLIS framework source distribution.

### Who funded the development of BLIS?

BLIS was primarily funded by a variety of gifts/grants from industry and the National Science Foundation. Please see the "Funding" section of the [BLIS homepage](https://github.com/flame/blis#funding) for more details.

Reminder: _Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation (NSF)._

### I found a bug. How do I report it?

If you think you've found a bug, we request that you [open an issue](http://github.com/flame/blis/issues). Don't be shy! Really, it's the best and most convenient way for us to track your issues/bugs/concerns.

### How do I request a new feature?

Feature requests should also be submitted by [opening a new issue](http://github.com/flame/blis/issues).

### I'm a developer and I'd like to study the way matrix multiplication is implemented in BLIS. Where should I start?

Great question! The first thing you should know is that the core framework of [level-3 operations](BLISTypedAPI.md#operation-index) was *not* designed to be used to teach or explain a high-performance implementation of matrix multiplication. Rather, it was designed to encode the family of level-3 operations with as little code duplication as possible. Because of this, and also for historical/evolutionary reasons, it can be a little difficult to trace the execution of, say, `gemm` from within the core framework.

Thankfully, we have an alternative environment in which experts, application developers, and other curious individuals can study BLIS's matrix multiplication implementation. This so-called "sandbox" is a simplified collection of code that strips away much of the framework complexity while also maintaining local definitions for many of the interesting bits. You may find this `gemmlike` sandbox in `sandbox/gemmlike`.

Sandboxes go beyond the scope of this FAQ. For an introduction, please refer to the [Sandboxes](Sandboxes.md) document, and/or contact the BLIS developers for more information.

### Where did you get the photo for the BLIS logo / mascot?

The sleeping ["BLIS cat"](README.md) photo was taken by Petar Mitchev and is used with his permission.

