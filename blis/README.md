_Recipient of the **[2023 James H. Wilkinson Prize for Numerical Software](https://www.siam.org/prizes-recognition/major-prizes-lectures/detail/james-h-wilkinson-prize-for-numerical-software)**_

_Recipient of the **[2020 SIAM Activity Group on Supercomputing Best Paper Prize](https://www.siam.org/prizes-recognition/activity-group-prizes/detail/siag-sc-best-paper-prize)**_


![The BLIS cat is sleeping.](http://www.cs.utexas.edu/users/field/blis_cat.png)

[![Build Status (CircleCI)](https://dl.circleci.com/status-badge/img/gh/flame/blis/tree/master.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/flame/blis/tree/master)
[![Build Status (TravisCI)](https://api.travis-ci.com/flame/blis.svg?branch=master)](https://app.travis-ci.com/github/flame/blis)
[![Build Status (Appveyor)](https://ci.appveyor.com/api/projects/status/github/flame/blis?branch=master&svg=true)](https://ci.appveyor.com/project/shpc/blis/branch/master)

[<img alt="Discord logo" title="Join us on Discord!" height="32px" src="docs/images/discord.svg" />](docs/Discord.md)

Contents
--------

* **[Introduction](#introduction)**
* **[Education and Learning](#education-and-learning)**
* **[What's New](#whats-new)**
* **[What People Are Saying About BLIS](#what-people-are-saying-about-blis)**
* **[Key Features](#key-features)**
* **[How to Download BLIS](#how-to-download-blis)**
* **[Getting Started](#getting-started)**
* **[Example Code](#example-code)**
* **[Documentation](#documentation)**
* **[Performance](#performance)**
* **[External Packages](#external-packages)**
* **[Discussion](#discussion)**
* **[Contributing](#contributing)**
* **[Citations](#citations)**
* **[Awards](#awards)**
* **[Funding](#funding)**

Introduction
------------

BLIS is an [award-winning](#awards)
portable software framework for instantiating high-performance
BLAS-like dense linear algebra libraries. The framework was designed to isolate
essential kernels of computation that, when optimized, immediately enable
optimized implementations of most of its commonly used and computationally
intensive operations. BLIS is written in [ISO
C99](http://en.wikipedia.org/wiki/C99) and available under a
[new/modified/3-clause BSD
license](http://opensource.org/licenses/BSD-3-Clause). While BLIS exports a
[new BLAS-like API](docs/BLISTypedAPI.md),
it also includes a BLAS compatibility layer which gives application developers
access to BLIS implementations via traditional [BLAS routine
calls](http://www.netlib.org/lapack/lug/node145.html).
An [object-based API](docs/BLISObjectAPI.md) unique to BLIS is also available.

For a thorough presentation of our framework, please read our
[ACM Transactions on Mathematical Software (TOMS)](https://toms.acm.org/)
journal article, ["BLIS: A Framework for Rapidly Instantiating BLAS
Functionality"](http://dl.acm.org/authorize?N91172).
For those who just want an executive summary, please see the
[Key Features](#key-features) section below.

In a follow-up article (also in [ACM TOMS](https://toms.acm.org/)),
["The BLIS Framework: Experiments in
Portability"](http://dl.acm.org/authorize?N16240),
we investigate using BLIS to instantiate level-3 BLAS implementations on a
variety of general-purpose, low-power, and multicore architectures.

An IPDPS'14 conference paper titled ["Anatomy of High-Performance Many-Threaded
Matrix
Multiplication"](http://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf)
systematically explores the opportunities for parallelism within the five loops
that BLIS exposes in its matrix multiplication algorithm.

For other papers related to BLIS, please see the
[Citations section](#citations) below.

It is our belief that BLIS offers substantial benefits in productivity when
compared to conventional approaches to developing BLAS libraries, as well as a
much-needed refinement of the BLAS interface, and thus constitutes a major
advance in dense linear algebra computation. While BLIS remains a
work-in-progress, we are excited to continue its development and further
cultivate its use within the community.

The BLIS framework is primarily developed and maintained by individuals in the
[Science of High-Performance Computing](http://shpc.ices.utexas.edu/)
(SHPC) group in the
[Oden Institute for Computational Engineering and Sciences](https://www.oden.utexas.edu/)
at [The University of Texas at Austin](https://www.utexas.edu/)
and in the [Matthews Research Group](https://matthewsresearchgroup.webstarts.com/)
at [Southern Methodist University](https://www.smu.edu/).
Please visit the [SHPC](http://shpc.ices.utexas.edu/) website for more
information about our research group, such as a list of
[people](http://shpc.ices.utexas.edu/people.html)
and [collaborators](http://shpc.ices.utexas.edu/collaborators.html),
[funding sources](http://shpc.ices.utexas.edu/funding.html),
[publications](http://shpc.ices.utexas.edu/publications.html),
and [other educational projects](http://www.ulaff.net/) (such as MOOCs).

Support for BLIS development at SMU was provided by the Office of Information Technology Research Technology Services team and HPC System Administrators on computational resources provided in partnership with SMU’s [O’Donnell Data Science and Research Computing Institute](https://www.smu.edu/provost/odonnell-institute).

Education and Learning
----------------------

Want to understand what's under the hood?
Many of the same concepts and principles employed when developing BLIS are
introduced and taught in a basic pedagogical setting as part of
[LAFF-On Programming for High Performance (LAFF-On-PfHP)](http://www.ulaff.net/),
one of several massive open online courses (MOOCs) in the
[Linear Algebra: Foundations to Frontiers](http://www.ulaff.net/) series,
all of which are available for free via the [edX platform](http://www.edx.org/).

What's New
----------

 * **Plugin feature now available!** BLIS addons (see below) provided a way to
quickly extend BLIS's operation support or define new custom BLIS APIs for your application.
BLIS plugins extend this support to completely external code, needing only an installed BLIS
package (no source required). BLIS plugins also allow users to define their own kernels
and blocksizes, combined with the cross-architecture support provided by the BLIS framework.
Finally, user plugins can utilize the new API for modifying the BLIS "control tree" which
defines the mathematical operation to be computed, as well as information controlling packing,
partitioning, etc. Users can now modify the control tree to implement new linear algebra
operations not already included in BLIS. See the [documentation](docs/PluginHowTo.md) for
an overview of these features and a step-by-step guides for creating plugins and modifying
the control tree to implement an example operation "SYRKD".

 * **BLIS selected for the 2023 James H. Wilkinson Prize for Numerical Software!** We
are thrilled to announce that Field Van Zee and Devin Matthews were chosen to receive
the [2023 James H. Wilkinson Prize for Numerical Software](https://www.siam.org/prizes-recognition/major-prizes-lectures/detail/james-h-wilkinson-prize-for-numerical-software).
The selection committee sought to recognize the recipients "for the development of
BLIS, a portable open-source software framework that facilitates rapid instantiation
of high-performance BLAS and BLAS-like operations targeting modern CPUs." This prize
is awarded once every four years to the authors of an outstanding piece of numerical
software, or to individuals who have made an outstanding contribution to an existing
piece of numerical software. It is awarded to an entry that best addresses all phases
of the preparation of high-quality numerical software, and is intended to recognize
innovative software in scientific computing and to encourage researchers in the
earlier stages of their career. The prize will be awarded at the
[2023 SIAM Conference on Computational Science and Engineering](https://www.siam.org/conferences/cm/conference/cse23) in Amsterdam.

 * **Join us on Discord!** In 2021, we soft-launched our [Discord](https://discord.com/)
server by privately inviting current and former collaborators, attendees of our BLIS
Retreat, as well as other participants within the BLIS ecosystem. We've been thrilled
by the results thus far, and are happy to announce that our new community is now open
to the broader public! If you'd like to hang out with other BLIS users and developers,
ask a question, discuss future features, or just say hello, please feel free to join
us! We've put together a [step-by-step guide](docs/Discord.md) for creating an account
and joining our cozy enclave. We even have a monthly "BLIS happy hour" event where
people can casually come together for a video chat, Q&A, brainstorm session, or
whatever it happens to unfold into!

 * **Addons feature now available!** Have you ever wanted to quickly extend BLIS's
operation support or define new custom BLIS APIs for your application, but were
unsure of how to add your source code to BLIS? Do you want to isolate your custom
code so that it only gets enabled when the user requests it? Do you like
[sandboxes](docs/Sandboxes.md), but wish you didn't have to provide an
implementation of `gemm`? If so, you should check out our new
[addons](docs/Addons.md) feature. Addons act like optional extensions that can be
created, enabled, and combined to suit your application's needs, all without
formally integrating your code into the core BLIS framework.

 * **Multithreaded small/skinny matrix support for sgemm now available!** Thanks to
funding and hardware support from Oracle, we have now accelerated `gemm` for
single-precision real matrix problems where one or two dimensions is exceedingly
small. This work is similar to the `gemm` optimization announced last year.
For now, we have only gathered performance results on an AMD Epyc Zen2 system, but
we hope to publish additional graphs for other architectures in the future. You may
find these Zen2 graphs via the [PerformanceSmall](docs/PerformanceSmall.md) document.

 * **BLIS awarded SIAM Activity Group on Supercomputing Best Paper Prize for 2020!**
We are thrilled to announce that the paper that we internally refer to as the
second BLIS paper,

   "The BLIS Framework: Experiments in Portability." Field G. Van Zee, Tyler Smith, Bryan Marker, Tze Meng Low, Robert A. van de Geijn, Francisco Igual, Mikhail Smelyanskiy, Xianyi Zhang, Michael Kistler, Vernon Austel, John A. Gunnels, Lee Killough. ACM Transactions on Mathematical Software (TOMS), 42(2):12:1--12:19, 2016.

   was selected for the [SIAM Activity Group on Supercomputing Best Paper Prize](https://www.siam.org/prizes-recognition/activity-group-prizes/detail/siag-sc-best-paper-prize)
for 2020. The prize is awarded once every two years to a paper judged to be
the most outstanding paper in the field of parallel scientific and engineering
computing, and has only been awarded once before (in 2016) since its inception
in 2015 (the committee did not award the prize in 2018). The prize
[was awarded](https://www.oden.utexas.edu/about/news/ScienceHighPerfomanceComputingSIAMBestPaperPrize/)
at the [2020 SIAM Conference on Parallel Processing for Scientific Computing](https://www.siam.org/conferences/cm/conference/pp20) in Seattle. Robert was present at
the conference to give
[a talk on BLIS](https://meetings.siam.org/sess/dsp_programsess.cfm?SESSIONCODE=68266) and accept the prize alongside other coauthors.
The selection committee sought to recognize the paper, "which validates BLIS,
a framework relying on the notion of microkernels that enables both productivity
and high performance." Their statement continues, "The framework will continue
having an important influence on the design and the instantiation of dense linear
algebra libraries."

 * **Multithreaded small/skinny matrix support for dgemm now available!** Thanks to
contributions made possible by our partnership with AMD, we have dramatically
accelerated `gemm` for double-precision real matrix problems where one or two
dimensions is exceedingly small. A natural byproduct of this optimization is
that the traditional case of small _m = n = k_ (i.e. square matrices) is also
accelerated, even though it was not targeted specifically. And though only
`dgemm` was optimized for now, support for other datatypes and/or other operations
may be implemented in the future. We've also added new graphs to the
[PerformanceSmall](docs/PerformanceSmall.md) document to showcase multithreaded
performance when one or more matrix dimensions are small.

 * **Performance comparisons now available!** We recently measured the
performance of various level-3 operations on a variety of hardware architectures,
as implemented within BLIS and other BLAS libraries for all four of the standard
floating-point datatypes. The results speak for themselves! Check out our
extensive performance graphs and background info in our new
[Performance](docs/Performance.md) document.

 * **BLIS is now in Debian Unstable!** Thanks to Debian developer-maintainers
[M. Zhou](https://github.com/cdluminate) and
[Nico Schlömer](https://github.com/nschloe) for sponsoring our package in Debian.
Their participation, contributions, and advocacy were key to getting BLIS into
the second-most popular Linux distribution (behind Ubuntu, which Debian packages
feed into). The Debian tracker page may be found
[here](https://tracker.debian.org/pkg/blis).

 * **BLIS now supports mixed-datatype gemm!** The `gemm` operation may now be
executed on operands of mixed domains and/or mixed precisions. Any combination
of storage datatype for A, B, and C is now supported, along with a separate
computation precision that can differ from the storage precision of A and B.
And even the 1m method now supports mixed-precision computation.
For more details, please see our [ACM TOMS](https://toms.acm.org/) journal
article submission ([current
draft](http://www.cs.utexas.edu/users/flame/pubs/blis7_toms_rev0.pdf)).

 * **BLIS now implements the 1m method.** Let's face it: writing complex
assembly `gemm` microkernels for a new architecture is never a priority--and
now, it almost never needs to be. The 1m method leverages existing real domain
`gemm` microkernels to implement all complex domain level-3 operations. For
more details, please see our [ACM TOMS](https://toms.acm.org/) journal article
submission ([current
draft](http://www.cs.utexas.edu/users/flame/pubs/blis6_toms_rev2.pdf)).

What People Are Saying About BLIS
---------------------------------

*["I noticed a substantial increase in multithreaded performance on my own
machine, which was extremely satisfying."](https://groups.google.com/d/msg/blis-discuss/8iu9B5KCxpA/uftpjgIsBwAJ)* ... *["[I was] happy it worked so well!"](https://groups.google.com/d/msg/blis-discuss/8iu9B5KCxpA/uftpjgIsBwAJ)* (Justin Shea)

*["This is an awesome library."](https://github.com/flame/blis/issues/288#issuecomment-447488637)* ... *["I want to thank you and the blis team for your efforts."](https://github.com/flame/blis/issues/288#issuecomment-448074704)* ([@Lephar](https://github.com/Lephar))

*["Any time somebody outside Intel beats MKL by a nontrivial amount, I report it to the MKL team. It is fantastic for any open-source project to get within 10% of MKL... [T]his is why Intel funds BLIS development."](https://github.com/flame/blis/issues/264#issuecomment-428673275)* ([@jeffhammond](https://github.com/jeffhammond))

*["So BLIS is now a part of Elk."](https://github.com/flame/blis/issues/267#issuecomment-429303902)* ... *["We have found that zgemm applied to a 15000x15000 matrix with multi-threaded BLIS on a 32-core Ryzen 2990WX processor is about twice as fast as MKL"](https://github.com/flame/blis/issues/264#issuecomment-428373946)* ... *["I'm starting to like this a lot."](https://github.com/flame/blis/issues/264#issuecomment-428926191)* ([@jdk2016](https://github.com/jdk2016))

*["I [found] BLIS because I was looking for BLAS operations on C-ordered arrays for NumPy. BLIS has that, but even better is the fact that it's developed in the open using a more modern language than Fortran."](https://github.com/flame/blis/issues/254#issuecomment-423838345)* ([@nschloe](https://github.com/nschloe))

*["The specific reason to have BLIS included [in Linux distributions] is the KNL and SKX [AVX-512] BLAS support, which OpenBLAS doesn't have."](https://github.com/flame/blis/issues/210#issuecomment-393126303)* ([@loveshack](https://github.com/loveshack))

*["All tests pass without errors on OpenBSD. Thanks!"](https://github.com/flame/blis/issues/202#issuecomment-389691543)* ([@ararslan](https://github.com/ararslan))

*["Thank you very much for your great help!... Looking forward to benchmarking."](https://github.com/flame/blis/issues/180#issuecomment-375895449)* ([@mrader1248](https://github.com/mrader1248))

*["Thanks for the beautiful work."](https://github.com/flame/blis/issues/163#issue-286575452)* ([@mmrmo](https://github.com/mmrmo))

*["[M]y software currently uses BLIS for its BLAS interface..."](https://github.com/flame/blis/issues/129#issuecomment-302904805)* ([@ShadenSmith](https://github.com/ShadenSmith))

*["[T]hanks so much for your work on this! Excited to test."](https://github.com/flame/blis/issues/129#issuecomment-341565071)* ... *["[On AMD Excavator], BLIS is competitive to / slightly faster than OpenBLAS for dgemms in my tests."](https://github.com/flame/blis/issues/129#issuecomment-341608673)* ([@iotamudelta](https://github.com/iotamudelta))

*["BLIS provided the only viable option on KNL, whose ecosystem is at present dominated by blackbox toolchains. Thanks again. Keep on this great work."](https://github.com/flame/blis/issues/116#issuecomment-281225101)* ([@heroxbd](https://github.com/heroxbd))

*["I want to definitely try this out..."](https://github.com/flame/blis/issues/12#issuecomment-48086295)* ([@ViralBShah](https://github.com/ViralBShah))

Key Features
------------

BLIS offers several advantages over traditional BLAS libraries:

 * **Portability that doesn't impede high performance.** Portability was a top
priority of ours when creating BLIS. With virtually no additional effort on the
part of the developer, BLIS is configurable as a fully-functional reference
implementation. But more importantly, the framework identifies and isolates a
key set of computational kernels which, when optimized, immediately and
automatically optimize performance across virtually all level-2 and level-3
BLIS operations. In this way, the framework acts as a productivity multiplier.
And since the optimized (non-portable) code is compartmentalized within these
few kernels, instantiating a high-performance BLIS library on a new
architecture is a relatively straightforward endeavor.

 * **Generalized matrix storage.** The BLIS framework exports interfaces that
allow one to specify both the row stride and column stride of a matrix. This
allows one to compute with matrices stored in column-major order, row-major
order, or by general stride. (This latter storage format is important for those
seeking to implement tensor contractions on multidimensional arrays.)
Furthermore, since BLIS tracks stride information for each matrix, operands of
different storage formats can be used within the same operation invocation. By
contrast, BLAS requires column-major storage. And while the CBLAS interface
supports row-major storage, it does not allow mixing storage formats.

 * **Rich support for the complex domain.** BLIS operations are developed and
expressed in their most general form, which is typically in the complex domain.
These formulations then simplify elegantly down to the real domain, with
conjugations becoming no-ops. Unlike the BLAS, all input operands in BLIS that
allow transposition and conjugate-transposition also support conjugation
(without transposition), which obviates the need for thread-unsafe workarounds.
Also, where applicable, both complex symmetric and complex Hermitian forms are
supported. (BLAS omits some complex symmetric operations, such as `symv`,
`syr`, and `syr2`.) Another great example of BLIS serving as a portability
lever is its implementation of the 1m method for complex matrix multiplication,
a novel mechanism of providing high-performance complex level-3 operations using
only real domain microkernels. This new innovation guarantees automatic level-3
support in the complex domain even when the kernel developers entirely forgo
writing complex kernels.

 * **Advanced multithreading support.** BLIS allows multiple levels of
symmetric multithreading for nearly all level-3 operations. (Currently, users
may choose to obtain parallelism via OpenMP, POSIX threads, or HPX). This
means that matrices may be partitioned in multiple dimensions simultaneously to
attain scalable, high-performance parallelism on multicore and many-core
architectures. The key to this innovation is a thread-specific control tree
infrastructure which encodes information about the logical thread topology and
allows threads to query and communicate data amongst one another. BLIS also
employs so-called "quadratic partitioning" when computing dimension sub-ranges
for each thread, so that arbitrary diagonal offsets of structured matrices with
unreferenced regions are taken into account to achieve proper load balance.
More recently, BLIS introduced a runtime abstraction to specify parallelism on
a per-call basis, which is useful for applications that want to handle most of
the parallelism.

 * **Ease of use.** The BLIS framework, and the library of routines it
generates, are easy to use for end users, experts, and vendors alike. An
optional BLAS compatibility layer provides application developers with
backwards compatibility to existing BLAS-dependent codes. Or, one may adjust or
write their application to take advantage of new BLIS functionality (such as
generalized storage formats or additional complex operations) by calling one
of BLIS's native APIs directly. BLIS's typed API will feel familiar to many
veterans of BLAS since these interfaces use BLAS-like calling sequences. And
many will find BLIS's object-based APIs a delight to use when customizing
or writing their own BLIS operations. (Objects are relatively lightweight
`structs` and passed by address, which helps tame function calling overhead.)

 * **Multilayered API and exposed kernels.** The BLIS framework exposes its
implementations in various layers, allowing expert developers to access exactly
the functionality desired. This layered interface includes that of the
lowest-level kernels, for those who wish to bypass the bulk of the framework.
Optimizations can occur at various levels, in part thanks to exposed packing
and unpacking facilities, which by default are highly parameterized and
flexible.

 * **Functionality that grows with the community's needs.** As its name
suggests, the BLIS framework is not a single library or static API, but rather
a nearly-complete template for instantiating high-performance BLAS-like
libraries. Furthermore, the framework is extensible, allowing developers to
leverage existing components to support new operations as they are identified.
If such operations require new kernels for optimal efficiency, the framework
and its APIs will be adjusted and extended accordingly. Community developers
who wish to experiment with creating new operations or APIs in BLIS can quickly
and easily do so via the [Addons](docs/Addons.md) feature.

 * **Code re-use.** Auto-generation approaches to achieving the aforementioned
goals tend to quickly lead to code bloat due to the multiple dimensions of
variation supported: operation (i.e. `gemm`, `herk`, `trmm`, etc.); parameter
case (i.e. side, [conjugate-]transposition, upper/lower storage, unit/non-unit
diagonal); datatype (i.e. single-/double-precision real/complex); matrix
storage (i.e. row-major, column-major, generalized); and algorithm (i.e.
partitioning path and kernel shape). These "brute force" approaches often
consider and optimize each operation or case combination in isolation, which is
less than ideal when the goal is to provide entire libraries. BLIS was designed
to be a complete framework for implementing basic linear algebra operations,
but supporting this vast amount of functionality in a manageable way required a
holistic design that employed careful abstractions, layering, and recycling of
generic (highly parameterized) codes, subject to the constraint that high
performance remain attainable.

 * **A foundation for mixed domain and/or mixed precision operations.** BLIS
was designed with the hope of one day allowing computation on real and complex
operands within the same operation. Similarly, we wanted to allow mixing
operands' numerical domains, floating-point precisions, or both domain and
precision, and to optionally compute in a precision different than one or both
operands' storage precisions. This feature has been implemented for the general
matrix multiplication (`gemm`) operation, providing 128 different possible type
combinations, which, when combined with existing transposition, conjugation,
and storage parameters, enables 55,296 different `gemm` use cases. For more
details, please see the documentation on [mixed datatype](docs/MixedDatatypes.md)
support and/or our [ACM TOMS](https://toms.acm.org/) journal paper on
mixed-domain/mixed-precision `gemm` ([linked below](#citations)).

How to Download BLIS
--------------------

There are a few ways to download BLIS. We list the most common four ways below.
We **highly recommend** using either Option 1 or 2. Otherwise, we recommend
Option 3 (over Option 4) so your compiler can perform optimizations specific
to your hardware.

1. **Download a source repository with `git clone`.**
Generally speaking, we prefer using `git clone` to clone a `git` repository.
Having a repository allows the user to periodically pull in the latest changes,
try out release candidates when they become available, switch to older versions
easily, and quickly rebuild BLIS whenever they wish.
(Note that implicit in cloning a repository is that the repository defaults to
using the `master` branch, which, as of 1.0, is considered akin to a development
branch and likely contains improvements since the most recent release.)

   In order to clone a `git` repository of BLIS, please obtain a repository
URL by clicking on the green button above the file/directory listing near the
top of this page (as rendered by GitHub). Generally speaking, it will amount
to executing the following command in your terminal shell:
   ```
   git clone https://github.com/flame/blis.git
   ```
   At this point, you will have the latest commit of the `master` branch
checked out. If you wish to check out an official release version, say,
1.0, execute the following:
   ```
   git checkout 1.0
   ```
   `git` will then transform your working copy to match the state of the
commit associated with version 1.0. You can view a list of official
versiontags at any time by executing:
   ```
   git tag --list
   ```
   Note that pre-release versions, such as release candidates, are actually
branches rather than tags, and thus will not show up in the list of tagged
versions.

2. **Download a source release via a tarball/zip file.**
If you would like to stick to the code that is included in official releases
and don't need the convenience of pulling in the latest changes via `git`, you
may download either a tarball or zip file of BLIS's latest
[release](https://github.com/flame/blis/releases). (NOTE: Some older releases
are only available as [tagged](https://github.com/flame/blis/tags) commits.
Also note that downloading release x.y.z is equivalent to downloading, or
checking out, the `git` tag `x.y.z`.)
We consider this option to be less than ideal for some people since you will
not be able to update your code with a simple `git pull` command.

3. **Download a source repository via a zip file.**
If you are uncomfortable with using `git` but would still like the latest
stable commits, we recommend that you download BLIS as a zip file.

   In order to download a zip file of the BLIS source distribution, please
click on the green button above the file listing near the top of this page.
This should reveal a link for downloading the zip file.

4. **Download a binary package specific to your OS.**
While we don't recommend this as the first choice for most users, we provide
links to community members who generously maintain BLIS packages for various
Linux distributions such as Debian Unstable and EPEL/Fedora. Please see the
[External Packages](#external-packages) section below for more information.

Getting Started
---------------

*NOTE: This section assumes you've either cloned a BLIS source code repository
via `git`, downloaded the latest source code via a zip file, or downloaded the
source code for a tagged version release---Options 1, 2, or 3, respectively,
as discussed in [the previous section](#how-to-download-blis).*

If you just want to build a sequential (not parallelized) version of BLIS
in a hurry and come back and explore other topics later, you can configure
and build BLIS as follows:
```
$ ./configure auto
$ make [-j]
```
You can then verify your build by running BLAS- and BLIS-specific test
drivers via `make check`:
```
$ make check [-j]
```
And if you would like to install BLIS to the directory specified to `configure`
via the `--prefix` option, run the `install` target:
```
$ make install
```
Please read the output of `./configure --help` for a full list of configure-time
options.
If/when you have time, we *strongly* encourage you to read the detailed
walkthrough of the build system found in our [Build System](docs/BuildSystem.md)
guide.

If you are still having trouble, you are welcome to [join us on Discord](docs/Discord.md)
for further information and/or assistance.

Example Code
------------

The BLIS source distribution provides example code in the `examples` directory.
Example code focuses on using BLIS APIs (not BLAS or CBLAS), and resides in
two subdirectories: [examples/oapi](examples/oapi) (which demonstrates the
[object API](docs/BLISObjectAPI.md)) and [examples/tapi](examples/tapi) (which
demonstrates the [typed API](docs/BLISTypedAPI.md)).

Either directory contains several files, each containing various pieces of
code that exercise core functionality of the BLIS API in question (object or
typed). These example files should be thought of collectively like a tutorial,
and therefore it is recommended to start from the beginning (the file that
starts in `00`).

You can build all of the examples by simply running `make` from either example
subdirectory (`examples/oapi` or `examples/tapi`). (You can also run
`make clean`.) The local `Makefile` assumes that you've already configured and
built (but not necessarily installed) BLIS two directories up, in `../..`. If
you have already installed BLIS to some permanent directory, you may refer to
that installation by setting the environment variable `BLIS_INSTALL_PATH` prior
to running make:
```
export BLIS_INSTALL_PATH=/usr/local; make
```
or by setting the same variable as part of the make command:
```
make BLIS_INSTALL_PATH=/usr/local
```
**Once the executable files have been built, we recommend reading the code and
the corresponding executable output side by side. This will help you see the
effects of each section of code.**

This tutorial is not exhaustive or complete; several object API functions were
omitted (mostly for brevity's sake) and thus more examples could be written.

Documentation
-------------

We provide extensive documentation on the BLIS build system, APIs, test
infrastructure, and other important topics. All documentation is formatted in
markdown and included in the BLIS source distribution (usually in the `docs`
directory). Slightly longer descriptions of each document may be found via in
the project's [wiki](https://github.com/flame/blis/wiki) section.

**Documents for everyone:**

 * **[Build System](docs/BuildSystem.md).** This document covers the basics of
configuring and building BLIS libraries, as well as related topics.

 * **[Testsuite](docs/Testsuite.md).** This document describes how to run
BLIS's highly parameterized and configurable test suite, as well as the
included BLAS test drivers.

 * **[BLIS Typed API Reference](docs/BLISTypedAPI.md).** Here we document the
so-called "typed" (or BLAS-like) API. This is the API that many users who are
already familiar with the BLAS will likely want to use.

 * **[BLIS Object API Reference](docs/BLISObjectAPI.md).** Here we document
the object API. This is API abstracts away properties of vectors and matrices
within `obj_t` structs that can be queried with accessor functions. Many
developers and experts prefer this API over the typed API.

 * **[Hardware Support](docs/HardwareSupport.md).** This document maintains a
table of supported microarchitectures.

 * **[Multithreading](docs/Multithreading.md).** This document describes how to
use the multithreading features of BLIS.

 * **[Mixed-Datatypes](docs/MixedDatatypes.md).** This document provides an
overview of BLIS's mixed-datatype functionality and provides a brief example
of how to take advantage of this new code.

 * **[Extending BLIS functionality](docs/PluginHowTo.md).** This document provides an
overview of BLIS's mechanisms for extending functionality through user-defined code.
BLIS has a plugin infrastructure which allows users to define their own kernels,
blocksizes, and kernel preferences which are compiled and managed by the BLIS framework.
BLIS also provides an API for modifying the "control tree" which can be used to
implement novel linear algebra operations.

 * **[Performance](docs/Performance.md).** This document reports empirically
measured performance of a representative set of level-3 operations on a variety
of hardware architectures, as implemented within BLIS and other BLAS libraries
for all four of the standard floating-point datatypes.

 * **[PerformanceSmall](docs/PerformanceSmall.md).** This document reports
empirically measured performance of `gemm` on select hardware architectures
within BLIS and other BLAS libraries when performing matrix problems where one
or two dimensions is exceedingly small.

 * **[Discord](docs/Discord.md).** This document describes how to: create an
account on Discord (if you don't already have one); obtain a private invite
link; and use that invite link to join our BLIS server on Discord.

 * **[Release Notes](docs/ReleaseNotes.md).** This document tracks a summary of
changes included with each new version of BLIS, along with contributor credits
for key features.

 * **[Frequently Asked Questions](docs/FAQ.md).** If you have general questions
about BLIS, please read this FAQ. If you can't find the answer to your question,
please feel free to join the [blis-devel](https://groups.google.com/group/blis-devel)
mailing list and post a question. We also have a
[blis-discuss](https://groups.google.com/group/blis-discuss) mailing list that
anyone can post to (even without joining).

**Documents for github contributors:**

 * **[Contributing bug reports, feature requests, PRs, etc](CONTRIBUTING.md).**
Interested in contributing to BLIS? Please read this document before getting
started. It provides a general overview of how best to report bugs, propose new
features, and offer code patches.

 * **[Coding Conventions](docs/CodingConventions.md).** If you are interested or
planning on contributing code to BLIS, please read this document so that you can
format your code in accordance with BLIS's standards.

**Documents for BLIS developers:**

 * **[Kernels Guide](docs/KernelsHowTo.md).** If you would like to learn more
about the types of kernels that BLIS exposes, their semantics, the operations
that each kernel accelerates, and various implementation issues, please read
this guide.

 * **[Configuration Guide](docs/ConfigurationHowTo.md).** If you would like to
learn how to add new sub-configurations or configuration families, or are simply
interested in learning how BLIS organizes its configurations and kernel sets,
please read this thorough walkthrough of the configuration system.

 * **[Addon Guide](docs/Addons.md).** If you are interested in learning
about using BLIS addons--that is, enabling existing (or creating new) bundles
of operation or API code that are built into a BLIS library--please read this
document.

 * **[Sandbox Guide](docs/Sandboxes.md).** If you are interested in learning
about using sandboxes in BLIS--that is, providing alternative implementations
of the `gemm` operation--please read this document.

Performance
-----------

We provide graphs that report performance of several implementations across a
range of hardware types, multithreading configurations, problem sizes,
operations, and datatypes. These pages also document most of the details needed
to reproduce these experiments.

 * **[Performance](docs/Performance.md).** This document reports empirically
measured performance of a representative set of level-3 operations on a variety
of hardware architectures, as implemented within BLIS and other BLAS libraries
for all four of the standard floating-point datatypes.

 * **[PerformanceSmall](docs/PerformanceSmall.md).** This document reports
empirically measured performance of `gemm` on select hardware architectures
within BLIS and other BLAS libraries when performing matrix problems where one
or two dimensions is exceedingly small.

External Packages
-----------------

Generally speaking, we **highly recommend** building from source whenever
possible using the latest `git` clone. (Tarballs of each
[tagged release](https://github.com/flame/blis/releases) are also available, but
we consider them to be less ideal since they are not as easy to upgrade as
`git` clones.)

That said, some users may prefer binary and/or source packages through their
Linux distribution. Thanks to generous involvement/contributions from our
community members, the following BLIS packages are now available:

 * **Debian**. [M. Zhou](https://github.com/cdluminate) has volunteered to
sponsor and maintain BLIS packages within the Debian Linux distribution. The
Debian package tracker can be found [here](https://tracker.debian.org/pkg/blis).
(Also, thanks to [Nico Schlömer](https://github.com/nschloe) for previously
volunteering his time to set up a standalone PPA.)

 * **Gentoo**. [M. Zhou](https://github.com/cdluminate) also maintains the
[BLIS package](https://packages.gentoo.org/packages/sci-libs/blis) entry for
[Gentoo](https://www.gentoo.org/), a Linux distribution known for its
source-based [portage](https://wiki.gentoo.org/wiki/Portage) package manager
and distribution system.

 * **EPEL/Fedora**. There are official BLIS packages in Fedora and EPEL (for
RHEL7+ and compatible distributions) with versions for 64-bit integers, OpenMP,
and pthreads, and shims which can be dynamically linked instead of reference
BLAS. (NOTE: For architectures other than intel64, amd64, and maybe arm64, the
performance of packaged BLIS will be low because it uses unoptimized generic
kernels; for those architectures, [OpenBLAS](https://github.com/xianyi/OpenBLAS)
may be a better solution.) [Dave
Love](https://github.com/loveshack) provides additional packages for EPEL6 in a
[Fedora Copr](https://copr.fedorainfracloud.org/coprs/loveshack/blis/), and
possibly versions more recent than the official repo for other EPEL/Fedora
releases. The source packages may build on other rpm-based distributions.

 * **OpenSuSE**. The copr referred to above has rpms for some OpenSuSE releases;
the source rpms may build for others.

 * **GNU Guix**. Guix has BLIS packages, provides builds only for the generic
target and some specific `x86_64` micro-architectures.

 * **Conda**. conda channel [conda-forge](https://github.com/conda-forge/blis-feedstock)
has Linux, OSX and Windows binary packages for `x86_64`.

Discussion
----------

Most of the active discussions are now happening on our [Discord](https://discord.com/)
server. Users and developers alike are welcome! Please see the
[BLIS Discord guide](docs/Discord.md) for a walkthrough of how to join us.

You can also still stay in touch by using either of the following mailing lists:

 * [blis-devel](https://groups.google.com/group/blis-devel): Please join and
post to this mailing list if you are a BLIS developer, or if you are trying
to use BLIS beyond simply linking to it as a BLAS library.

 * [blis-discuss](https://groups.google.com/group/blis-discuss): Please join and
post to this mailing list if you have general questions or feedback regarding
BLIS. Application developers (end users) may wish to post here, unless they
have bug reports, in which case they should open a
[new issue](https://github.com/flame/blis/issues) on github.

Contributing
------------

For information on how to contribute to our project, including preferred
[coding conventions](docs/CodingConventions.md), please refer to the
[CONTRIBUTING](CONTRIBUTING.md) file at the top-level of the BLIS source
distribution.

Citations
---------

For those of you looking for the appropriate article to cite regarding BLIS, we
recommend citing our
[first ACM TOMS journal paper](https://dl.acm.org/doi/10.1145/2764454?cid=81314495332)
([unofficial backup link](https://www.cs.utexas.edu/users/flame/pubs/blis1_toms_rev3.pdf)):

```
@article{BLIS1,
   author      = {Field G. {V}an~{Z}ee and Robert A. {v}an~{d}e~{G}eijn},
   title       = {{BLIS}: A Framework for Rapidly Instantiating {BLAS} Functionality},
   journal     = {ACM Transactions on Mathematical Software},
   volume      = {41},
   number      = {3},
   pages       = {14:1--14:33},
   month       = {June},
   year        = {2015},
   issue_date  = {June 2015},
   url         = {https://doi.acm.org/10.1145/2764454},
}
```

You may also cite the
[second ACM TOMS journal paper](https://dl.acm.org/doi/10.1145/2755561?cid=81314495332)
([unofficial backup link](https://www.cs.utexas.edu/users/flame/pubs/blis2_toms_rev3.pdf)):

```
@article{BLIS2,
   author      = {Field G. {V}an~{Z}ee and Tyler Smith and Francisco D. Igual and
                  Mikhail Smelyanskiy and Xianyi Zhang and Michael Kistler and Vernon Austel and
                  John Gunnels and Tze Meng Low and Bryan Marker and Lee Killough and
                  Robert A. {v}an~{d}e~{G}eijn},
   title       = {The {BLIS} Framework: Experiments in Portability},
   journal     = {ACM Transactions on Mathematical Software},
   volume      = {42},
   number      = {2},
   pages       = {12:1--12:19},
   month       = {June},
   year        = {2016},
   issue_date  = {June 2016},
   url         = {https://doi.acm.org/10.1145/2755561},
}
```

We also have a third paper, submitted to IPDPS 2014, on achieving
[multithreaded parallelism in BLIS](https://dl.acm.org/doi/10.1109/IPDPS.2014.110)
([unofficial backup link](https://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf)):

```
@inproceedings{BLIS3,
   author      = {Tyler M. Smith and Robert A. {v}an~{d}e~{G}eijn and Mikhail Smelyanskiy and
                  Jeff R. Hammond and Field G. {V}an~{Z}ee},
   title       = {Anatomy of High-Performance Many-Threaded Matrix Multiplication},
   booktitle   = {28th IEEE International Parallel \& Distributed Processing Symposium
                  (IPDPS 2014)},
   year        = {2014},
   url         = {https://doi.org/10.1109/IPDPS.2014.110},
}
```

A fourth paper, submitted to ACM TOMS, also exists, which proposes an
[analytical model](https://dl.acm.org/doi/10.1145/2925987)
for determining blocksize parameters in BLIS
([unofficial backup link](https://www.cs.utexas.edu/users/flame/pubs/TOMS-BLIS-Analytical.pdf)):

```
@article{BLIS4,
   author      = {Tze Meng Low and Francisco D. Igual and Tyler M. Smith and
                  Enrique S. Quintana-Ort\'{\i}},
   title       = {Analytical Modeling Is Enough for High-Performance {BLIS}},
   journal     = {ACM Transactions on Mathematical Software},
   volume      = {43},
   number      = {2},
   pages       = {12:1--12:18},
   month       = {August},
   year        = {2016},
   issue_date  = {August 2016},
   url         = {https://doi.acm.org/10.1145/2925987},
}
```

A fifth paper, submitted to ACM TOMS, begins the study of so-called
[induced methods for complex matrix multiplication](https://dl.acm.org/doi/10.1145/3086466?cid=81314495332)
([unofficial backup link](https://www.cs.utexas.edu/users/flame/pubs/blis5_toms_rev2.pdf)):

```
@article{BLIS5,
   author      = {Field G. {V}an~{Z}ee and Tyler Smith},
   title       = {Implementing High-performance Complex Matrix Multiplication via the 3m and 4m Methods},
   journal     = {ACM Transactions on Mathematical Software},
   volume      = {44},
   number      = {1},
   pages       = {7:1--7:36},
   month       = {July},
   year        = {2017},
   issue_date  = {July 2017},
   url         = {https://doi.acm.org/10.1145/3086466},
}
```

A sixth paper, submitted to ACM TOMS, revisits the topic of the previous
article and derives a
[superior induced method](https://epubs.siam.org/doi/10.1137/19M1282040)
([unofficial backup link](https://www.cs.utexas.edu/users/flame/pubs/blis6_sisc_rev3.pdf)):

```
@article{BLIS6,
   author      = {Field G. {V}an~{Z}ee},
   title       = {Implementing High-Performance Complex Matrix Multiplication via the 1m Method},
   journal     = {SIAM Journal on Scientific Computing},
   volume      = {42},
   number      = {5},
   pages       = {C221--C244},
   month       = {September}
   year        = {2020},
   issue_date  = {September 2020},
   url         = {https://doi.org/10.1137/19M1282040}
}
```

A seventh paper, submitted to ACM TOMS, explores the implementation of `gemm` for
[mixed-domain and/or mixed-precision](https://dl.acm.org/doi/10.1145/3402225?cid=81314495332) operands
([unofficial backup link](https://www.cs.utexas.edu/users/flame/pubs/blis7_toms_rev0.pdf)):

```
@article{BLIS7,
   author      = {Field G. {V}an~{Z}ee and Devangi N. Parikh and Robert A. van~de~{G}eijn},
   title       = {Supporting Mixed-domain Mixed-precision Matrix Multiplication
within the BLIS Framework},
   journal     = {ACM Transactions on Mathematical Software},
   volume      = {47},
   number      = {2},
   pages       = {12:1--12:26},
   month       = {April},
   year        = {2021},
   issue_date  = {April 2021},
   url         = {https://doi.org/10.1145/3402225},
}
```

Awards
------

 * **[2023 James H. Wilkinson Prize for Numerical Software.](https://www.siam.org/prizes-recognition/major-prizes-lectures/detail/james-h-wilkinson-prize-for-numerical-software)**
This prize is awarded once every four years to the authors of an outstanding piece of
numerical software, or to individuals who have made an outstanding contribution to an
existing piece of numerical software. The selection committee sought to recognize the
recipients "for the development of [BLIS](https://github.com/flame/blis), a portable
open-source software framework that facilitates rapid instantiation of
high-performance BLAS and BLAS-like operations targeting modern CPUs." The prize will
be awarded at the
[2023 SIAM Conference on Computational Science and Engineering](https://www.siam.org/conferences/cm/conference/cse23) in Amsterdam.

 * **[2020 SIAM Activity Group on Supercomputing Best Paper Prize.](https://www.siam.org/prizes-recognition/activity-group-prizes/detail/siag-sc-best-paper-prize)**
This prize is awarded once every two years to the authors of the most outstanding
paper, as determined by the selection committee, in the field of parallel scientific
and engineering computing published within the four calendar years preceding the
award year. The prize was chosen for the paper ["The BLIS Framework: Experiments in
Portability."](#citations) and awarded at the [2020 SIAM Conference on Parallel Processing for Scientific Computing](https://www.siam.org/conferences/cm/conference/pp20) in Seattle where Robert van de Geijn delivered [a talk on BLIS](https://meetings.siam.org/sess/dsp_programsess.cfm?SESSIONCODE=68266) and accepted the prize alongside other coauthors.
See also:
   * [SIAM News | January 2020 Prize Spotlight](https://sinews.siam.org/Details-Page/january-2020-prize-spotlight#Field&Robert)
   * [Oden Institute's SHPC Group Win SIAM Best Paper Prize](https://www.oden.utexas.edu/about/news/ScienceHighPerfomanceComputingSIAMBestPaperPrize/)

Funding
-------

This project and its associated research were partially sponsored by grants from
[Microsoft](https://www.microsoft.com/),
[Intel](https://www.intel.com/),
[Texas Instruments](https://www.ti.com/),
[AMD](https://www.amd.com/),
[HPE](https://www.hpe.com/),
[Oracle](https://www.oracle.com/),
[Huawei](https://www.huawei.com/),
[Facebook](https://www.facebook.com/),
and
[ARM](https://www.arm.com/),
as well as grants from the
[National Science Foundation](https://www.nsf.gov/) (Awards
CCF-0917167, ACI-1148125/1340293, CCF-1320112, and ACI-1550493).

_Any opinions, findings and conclusions or recommendations expressed in this
material are those of the author(s) and do not necessarily reflect the views of
the National Science Foundation (NSF)._

