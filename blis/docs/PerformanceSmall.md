# Contents

* **[Contents](PerformanceSmall.md#contents)**
* **[Introduction](PerformanceSmall.md#introduction)**
* **[General information](PerformanceSmall.md#general-information)**
* **[Interpretation](PerformanceSmall.md#interpretation)**
* **[Reproduction](PerformanceSmall.md#reproduction)**
* **[Level-3 performance](PerformanceSmall.md#level-3-performance)**
  * **[Kaby Lake](PerformanceSmall.md#kaby-lake)**
    * **[Experiment details](PerformanceSmall.md#kaby-lake-experiment-details)**
    * **[Results](PerformanceSmall.md#kaby-lake-results)**
  * **[Haswell](PerformanceSmall.md#haswell)**
    * **[Experiment details](PerformanceSmall.md#haswell-experiment-details)**
    * **[Results](PerformanceSmall.md#haswell-results)**
  * **[Zen](PerformanceSmall.md#zen)**
    * **[Experiment details](PerformanceSmall.md#zen-experiment-details)**
    * **[Results](PerformanceSmall.md#zen-results)**
  * **[Zen2](PerformanceSmall.md#zen2)**
    * **[Experiment details](PerformanceSmall.md#zen2-experiment-details)**
    * **[Results](PerformanceSmall.md#zen2-results)**
* **[Feedback](PerformanceSmall.md#feedback)**

# Introduction

This document showcases performance results for the level-3 `gemm` operation
on small matrices with BLIS and BLAS for select hardware architectures.

# General information

Generally speaking, for level-3 operations on small matrices, we publish
two "panels" for each type of hardware, one that reflects performance on
row-stored matrices and another for column-stored matrices.
Each panel will consist of a 4x7 grid of graphs, with each row representing
a different transposition case (`nn`, `nt`, `tn`, `tt`)
complex) and each column representing a different shape scenario, usually
with one or two matrix dimensions bound to a fixed size for all problem
sizes tested.
Each of the 28 graphs within a panel will contain an x-axis that reports
problem size, with one, two, or all three matrix dimensions equal to the
problem size (e.g. _m_ = 6; _n_ = _k_, also encoded as `m6npkp`).
The y-axis will report in units GFLOPS (or billions of floating-point operations
per second) per core.

It's also worth pointing out that the top of some graphs (e.g. the maximum
y-axis value depicted) correspond to the theoretical peak performance
under the conditions associated with that graph, while in other graphs the
y-axis has been adjusted to better show the difference between the various
curves. (We *strongly* prefer to always use peak performance as the top of
the graph; however, this is one of the few exceptions where we feel some
scaling is warranted.)
Theoretical peak performance on a single core, in units of GFLOPS, is
calculated as the product of:
1. the maximum sustainable clock rate in GHz; and
2. the maximum number of floating-point operations (flops) that can be
executed per cycle.

Note that the maximum sustainable clock rate may change depending on the
conditions.
For example, on some systems the maximum clock rate is higher when only one
core is active (e.g. single-threaded performance) versus when all cores are
active (e.g. multithreaded performance).
The maximum number of flops executable per cycle (per core) is generally
computed as the product of:
1. the maximum number of fused multiply-add (FMA) vector instructions that
can be issued per cycle (per core);
2. the maximum number of elements that can be stored within a single vector
register (for the datatype in question); and
3. 2.0, since an FMA instruction fuses two operations (a multiply and an add).

Typically, organizations and individuals publish performance with square
matrices, which can miss the problem sizes of interest to many applications.
Here, in addition to square matrices (shown in the seventh column), we also
show six other scenarios where one or two `gemm` dimensions (of _m,_ _n_, and
_k_) is small. In these six columns, the constant small matrix dimensions were
chosen to be _very_ small--in the neighborhood of 8--intentionally to showcase
what happens when at least one of the matrices is abnormally "skinny."

The problem size range, represented on the x-axis, is sampled in
increments that vary. These increments (and the overall range) are generally
large for the cases where two dimensions are small (and constant), medium for
cases where one dimension is small (and constant), and small for cases where
all dimensions (e.g. _m_, _n_, and _k_) are variable and bound to the problem
size (i.e., square matrices).

The legend in each graph contains two entries for BLIS, corresponding to the
two black lines, one solid and one dotted. The dotted line, **"BLIS conv"**,
represents the conventional implementation that targets large matrices. This
was the only implementation available in BLIS prior to the addition to the
small/skinny matrix support. The solid line, **"BLIS sup"**, makes use of the
new small/skinny matrix implementation. Sometimes, the performance of
**"BLIS sup"** drops below that of **"BLIS conv"** for somewhat larger problems.
However, in practice, we use a threshold to determine when to switch from the
former to the latter, and therefore the goal is for the performance of
**"BLIS conv"** to serve as an approximate floor below which BLIS performance
never drops.

Finally, each point along each curve represents the best of three trials.

# Interpretation

In general, the the curves associated with higher-performing implementations
will appear higher in the graphs than lower-performing implementations.
Ideally, an implementation will climb in performance (as a function of problem
size) as quickly as possible and asymptotically approach some high fraction of
peak performance.

When corresponding with us, via email or when opening an
[issue](https://github.com/flame/blis/issues) on github, we kindly ask that
you specify as closely as possible (though a range is fine) your problem
size of interest so that we can better assist you.

# Reproduction

In general, we do not offer any step-by-step guide for how to reproduce the
performance graphs shown below.

That said, if you are keenly interested in running your own performance
benchmarks, either in an attempt to reproduce the results shown here or to
measure performance of different hardware, of different implementations (or
versions), and/or for different problem sizes, you should begin by studying
the source code, `Makefile`, and scripts in
the [test/sup](https://github.com/flame/blis/tree/master/test/sup) directory
of the BLIS source distribution. Then, you'll need to take time to build
and/or install some (or all) of the implementations shown (e.g.
[OpenBLAS](https://github.com/xianyi/OpenBLAS),
[MKL](https://software.intel.com/en-us/mkl/),
[Eigen](http://eigen.tuxfamily.org),
[BLASFEO](https://github.com/giaf/blasfeo), and
[libxsmm](https://github.com/hfp/libxsmm)), including BLIS. Be sure to consult
the detailed notes provided below; they should be *very* helpful in successfully
building the libraries. The `runme.sh` script in `test/sup` (or `test/supmt`)
will help you run
some (or all) of the test drivers produced by the `Makefile`, and the
Matlab/Octave function `plot_panel_trxsh()` defined in the `octave` directory
will help you turn the output of those test drivers into a PDF file of graphs.
The `runthese.m` file will contain example invocations of the function.

# Level-3 performance

## Kaby Lake

### Kaby Lake experiment details

* Location: undisclosed
* Processor model: Intel Core i5-7500 (Kaby Lake)
* Core topology: one socket, 4 cores total
* SMT status: unavailable
* Max clock rate: 3.8GHz (single-core)
* Max vector register length: 256 bits (AVX2)
* Max FMA vector IPC: 2
* Peak performance:
  * single-core: 57.6 GFLOPS (double-precision), 115.2 GFLOPS (single-precision)
  * multicore: 57.6 GFLOPS/core (double-precision), 115.2 GFLOPS/core (single-precision)
* Operating system: Gentoo Linux (Linux kernel 5.2.4)
* Page size: 4096 bytes
* Compiler: gcc 8.3.0
* Results gathered: 3 March 2020
* Implementations tested:
  * BLIS 90db88e (0.6.1-8)
    * configured with `./configure --enable-cblas auto` (single-threaded)
    * configured with `./configure --enable-cblas -t openmp auto` (multithreaded)
    * sub-configuration exercised: `haswell`
    * Multithreaded (4 cores) execution requested via `export BLIS_NUM_THREADS=4`
  * OpenBLAS 0.3.8
    * configured `Makefile.rule` with `BINARY=64 NO_LAPACK=1 NO_LAPACKE=1 USE_THREAD=0 USE_LOCKING=1` (single-threaded)
    * configured `Makefile.rule` with `BINARY=64 NO_LAPACK=1 NO_LAPACKE=1 USE_THREAD=1 NUM_THREADS=4` (multithreaded)
    * Multithreaded (4 cores) execution requested via `export OPENBLAS_NUM_THREADS=4`
  * BLASFEO f9b78c6
    * configured `Makefile.rule` with: `BLAS_API=1 FORTRAN_BLAS_API=1 CBLAS_API=1`.
  * Eigen 3.3.90
    * Obtained via the [Eigen git mirror](https://github.com/eigenteam/eigen-git-mirror) (36b9596)
    * Prior to compilation, modified top-level `CMakeLists.txt` to ensure that `-march=native` was added to `CXX_FLAGS` variable (h/t Sameer Agarwal):
         ```
         # These lines added after line 67.
         check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
         if(COMPILER_SUPPORTS_MARCH_NATIVE)
           set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
         endif()
         ```
    * configured and built BLAS library via `mkdir build; cd build; CC=gcc cmake ..; make blas`
    * installed headers via `cmake . -DCMAKE_INSTALL_PREFIX=$HOME/flame/eigen; make install`
    * The `gemm` implementation was pulled in at compile-time via Eigen headers; other operations were linked to Eigen's BLAS library.
    * Single-threaded (1 core) execution requested via `export OMP_NUM_THREADS=1`
    * Multithreaded (4 cores) execution requested via `export OMP_NUM_THREADS=4`
  * MKL 2020 initial release
    * Single-threaded (1 core) execution requested via `export MKL_NUM_THREADS=1`
    * Multithreaded (4 cores) execution requested via `export MKL_NUM_THREADS=4`
  * libxsmm a40a833 (post-1.14)
    * compiled with `make AVX=2`; linked with [netlib BLAS](http://www.netlib.org/blas/) 3.6.0 as the fallback library to better show where libxsmm stops handling the computation internally.
* Affinity:
  * Thread affinity for BLIS was specified manually via `GOMP_CPU_AFFINITY="0-3"`. However, multithreaded OpenBLAS appears to revert to single-threaded execution if `GOMP_CPU_AFFINITY` is set. Therefore, when measuring OpenBLAS performance, the `GOMP_CPU_AFFINITY` environment variable was unset.
* Frequency throttling (via `cpupower`):
  * Driver: intel_pstate
  * Governor: performance
  * Hardware limits: 800MHz - 3.8GHz
  * Adjusted minimum: 3.8GHz
* Comments:
  * libxsmm is highly competitive for very small problems, but quickly gives up once the "large" dimension exceeds about 180-240 (or 64 in the case where all operands are square). Also, libxsmm's `gemm` cannot handle a transposition on matrix A and similarly dispatches the fallback implementation for those cases. libxsmm also does not export CBLAS interfaces, and therefore only appears on the graphs for column-stored matrices.

### Kaby Lake results

#### pdf

* [Kaby Lake single-threaded row-stored](graphs/sup/dgemm_rrr_kbl_nt1.pdf)
* [Kaby Lake single-threaded column-stored](graphs/sup/dgemm_ccc_kbl_nt1.pdf)
* [Kaby Lake multithreaded (4 cores) row-stored](graphs/sup/dgemm_rrr_kbl_nt4.pdf)
* [Kaby Lake multithreaded (4 cores) column-stored](graphs/sup/dgemm_ccc_kbl_nt4.pdf)

#### png (inline)

* **Kaby Lake single-threaded row-stored**
![single-threaded row-stored](graphs/sup/dgemm_rrr_kbl_nt1.png)
* **Kaby Lake single-threaded column-stored**
![single-threaded column-stored](graphs/sup/dgemm_ccc_kbl_nt1.png)
* **Kaby Lake multithreaded (4 cores) row-stored**
![multithreaded row-stored](graphs/sup/dgemm_rrr_kbl_nt4.png)
* **Kaby Lake multithreaded (4 cores) column-stored**
![multithreaded column-stored](graphs/sup/dgemm_ccc_kbl_nt4.png)

---

## Haswell

### Haswell experiment details

* Location: TACC (Lonestar5)
* Processor model: Intel Xeon E5-2690 v3 (Haswell)
* Core topology: two sockets, 12 cores per socket, 24 cores total
* SMT status: enabled, but not utilized
* Max clock rate: 3.5GHz (single-core), 3.1GHz (multicore)
* Max vector register length: 256 bits (AVX2)
* Max FMA vector IPC: 2
* Peak performance:
  * single-core: 56 GFLOPS (double-precision), 112 GFLOPS (single-precision)
  * multicore: 49.6 GFLOPS/core (double-precision), 99.2 GFLOPS/core (single-precision)
* Operating system: Cray Linux Environment 6 (Linux kernel 4.4.103)
* Page size: 4096 bytes
* Compiler: gcc 7.3.0
* Results gathered: 3 March 2020
* Implementations tested:
  * BLIS 90db88e (0.6.1-8)
    * configured with `./configure --enable-cblas auto` (single-threaded)
    * configured with `./configure --enable-cblas -t openmp auto` (multithreaded)
    * sub-configuration exercised: `haswell`
    * Multithreaded (12 cores) execution requested via `export BLIS_NUM_THREADS=12`
  * OpenBLAS 0.3.8
    * configured `Makefile.rule` with `BINARY=64 NO_LAPACK=1 NO_LAPACKE=1 USE_THREAD=0 USE_LOCKING=1` (single-threaded)
    * configured `Makefile.rule` with `BINARY=64 NO_LAPACK=1 NO_LAPACKE=1 USE_THREAD=1 NUM_THREADS=12` (multithreaded)
    * Multithreaded (12 cores) execution requested via `export OPENBLAS_NUM_THREADS=12`
  * BLASFEO f9b78c6
    * configured `Makefile.rule` with: `BLAS_API=1 FORTRAN_BLAS_API=1 CBLAS_API=1`.
  * Eigen 3.3.90
    * Obtained via the [Eigen git mirror](https://github.com/eigenteam/eigen-git-mirror) (36b9596)
    * Prior to compilation, modified top-level `CMakeLists.txt` to ensure that `-march=native` was added to `CXX_FLAGS` variable (h/t Sameer Agarwal):
         ```
         # These lines added after line 67.
         check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
         if(COMPILER_SUPPORTS_MARCH_NATIVE)
           set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
         endif()
         ```
    * configured and built BLAS library via `mkdir build; cd build; CC=gcc cmake ..; make blas`
    * installed headers via `cmake . -DCMAKE_INSTALL_PREFIX=$HOME/flame/eigen; make install`
    * The `gemm` implementation was pulled in at compile-time via Eigen headers; other operations were linked to Eigen's BLAS library.
    * Single-threaded (1 core) execution requested via `export OMP_NUM_THREADS=1`
    * Multithreaded (12 cores) execution requested via `export OMP_NUM_THREADS=12`
  * MKL 2020 initial release
    * Single-threaded (1 core) execution requested via `export MKL_NUM_THREADS=1`
    * Multithreaded (12 cores) execution requested via `export MKL_NUM_THREADS=12`
  * libxsmm a40a833 (post-1.14)
    * compiled with `make AVX=2`; linked with [netlib BLAS](http://www.netlib.org/blas/) 3.6.0 as the fallback library to better show where libxsmm stops handling the computation internally.
* Affinity:
  * Thread affinity for BLIS was specified manually via `GOMP_CPU_AFFINITY="0-11"`. However, multithreaded OpenBLAS appears to revert to single-threaded execution if `GOMP_CPU_AFFINITY` is set. Therefore, when measuring OpenBLAS performance, the `GOMP_CPU_AFFINITY` environment variable was unset.
* Frequency throttling (via `cpupower`):
  * No changes made.
* Comments:
  * libxsmm is highly competitive for very small problems, but quickly gives up once the "large" dimension exceeds about 180-240 (or 64 in the case where all operands are square). Also, libxsmm's `gemm` cannot handle a transposition on matrix A and similarly dispatches the fallback implementation for those cases. libxsmm also does not export CBLAS interfaces, and therefore only appears on the graphs for column-stored matrices.

### Haswell results

#### pdf

* [Haswell single-threaded row-stored](graphs/sup/dgemm_rrr_has_nt1.pdf)
* [Haswell single-threaded column-stored](graphs/sup/dgemm_ccc_has_nt1.pdf)
* [Haswell multithreaded (12 cores) row-stored](graphs/sup/dgemm_rrr_has_nt12.pdf)
* [Haswell multithreaded (12 cores) column-stored](graphs/sup/dgemm_ccc_has_nt12.pdf)

#### png (inline)

* **Haswell single-threaded row-stored**
![single-threaded row-stored](graphs/sup/dgemm_rrr_has_nt1.png)
* **Haswell single-threaded column-stored**
![single-threaded column-stored](graphs/sup/dgemm_ccc_has_nt1.png)
* **Haswell multithreaded (12 cores) row-stored**
![multithreaded row-stored](graphs/sup/dgemm_rrr_has_nt12.png)
* **Haswell multithreaded (12 cores) column-stored**
![multithreaded column-stored](graphs/sup/dgemm_ccc_has_nt12.png)

---

## Zen

### Zen experiment details

* Location: Oracle cloud
* Processor model: AMD Epyc 7551 (Zen1)
* Core topology: two sockets, 4 dies per socket, 2 core complexes (CCX) per die, 4 cores per CCX, 64 cores total
* SMT status: enabled, but not utilized
* Max clock rate: 3.0GHz (single-core), 2.55GHz (multicore)
* Max vector register length: 256 bits (AVX2)
* Max FMA vector IPC: 1
  * Alternatively, FMA vector IPC is 2 when vectors are limited to 128 bits each.
* Peak performance:
  * single-core: 24 GFLOPS (double-precision), 48 GFLOPS (single-precision)
   * multicore: 20.4 GFLOPS/core (double-precision), 40.8 GFLOPS/core (single-precision)
* Operating system: Ubuntu 18.04 (Linux kernel 4.15.0)
* Page size: 4096 bytes
* Compiler: gcc 7.4.0
* Results gathered: 3 March 2020
* Implementations tested:
  * BLIS 90db88e (0.6.1-8)
    * configured with `./configure --enable-cblas auto` (single-threaded)
    * configured with `./configure --enable-cblas -t openmp auto` (multithreaded)
    * sub-configuration exercised: `zen`
    * Multithreaded (32 cores) execution requested via `export BLIS_NUM_THREADS=32`
  * OpenBLAS 0.3.8
    * configured `Makefile.rule` with `BINARY=64 NO_LAPACK=1 NO_LAPACKE=1 USE_THREAD=0 USE_LOCKING=1` (single-threaded)
    * configured `Makefile.rule` with `BINARY=64 NO_LAPACK=1 NO_LAPACKE=1 USE_THREAD=1 NUM_THREADS=32` (multithreaded)
    * Multithreaded (32 cores) execution requested via `export OPENBLAS_NUM_THREADS=32`
  * BLASFEO f9b78c6
    * configured `Makefile.rule` with: `BLAS_API=1 FORTRAN_BLAS_API=1 CBLAS_API=1`.
    * built BLAS library via `make CC=gcc`
  * Eigen 3.3.90
    * Obtained via the [Eigen git mirror](https://github.com/eigenteam/eigen-git-mirror) (36b9596)
    * Prior to compilation, modified top-level `CMakeLists.txt` to ensure that `-march=native` was added to `CXX_FLAGS` variable (h/t Sameer Agarwal):
         ```
         # These lines added after line 67.
         check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
         if(COMPILER_SUPPORTS_MARCH_NATIVE)
           set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
         endif()
         ```
    * configured and built BLAS library via `mkdir build; cd build; CC=gcc cmake ..; make blas`
    * installed headers via `cmake . -DCMAKE_INSTALL_PREFIX=$HOME/flame/eigen; make install`
    * The `gemm` implementation was pulled in at compile-time via Eigen headers; other operations were linked to Eigen's BLAS library.
    * Single-threaded (1 core) execution requested via `export OMP_NUM_THREADS=1`
    * Multithreaded (32 cores) execution requested via `export OMP_NUM_THREADS=32`
  * MKL 2020 initial release
    * Single-threaded (1 core) execution requested via `export MKL_NUM_THREADS=1`
    * Multithreaded (32 cores) execution requested via `export MKL_NUM_THREADS=32`
  * libxsmm a40a833 (post-1.14)
    * compiled with `make AVX=2`; linked with [netlib BLAS](http://www.netlib.org/blas/) 3.6.0 as the fallback library to better show where libxsmm stops handling the computation internally.
* Affinity:
  * Thread affinity for BLIS was specified manually via `GOMP_CPU_AFFINITY="0-31"`. However, multithreaded OpenBLAS appears to revert to single-threaded execution if `GOMP_CPU_AFFINITY` is set. Therefore, when measuring OpenBLAS performance, the `GOMP_CPU_AFFINITY` environment variable was unset.
* Frequency throttling (via `cpupower`):
  * Driver: acpi-cpufreq
  * Governor: performance
  * Hardware limits: 1.2GHz - 2.0GHz
  * Adjusted minimum: 2.0GHz
* Comments:
  * libxsmm is highly competitive for very small problems, but quickly gives up once the "large" dimension exceeds about 180-240 (or 64 in the case where all operands are square). Also, libxsmm's `gemm` cannot handle a transposition on matrix A and similarly dispatches the fallback implementation for those cases. libxsmm also does not export CBLAS interfaces, and therefore only appears on the graphs for column-stored matrices.

### Zen results

#### pdf

* [Zen single-threaded row-stored](graphs/sup/dgemm_rrr_zen_nt1.pdf)
* [Zen single-threaded column-stored](graphs/sup/dgemm_ccc_zen_nt1.pdf)
* [Zen multithreaded (32 cores) row-stored](graphs/sup/dgemm_rrr_zen_nt32.pdf)
* [Zen multithreaded (32 cores) column-stored](graphs/sup/dgemm_ccc_zen_nt32.pdf)

#### png (inline)

* **Zen single-threaded row-stored**
![single-threaded row-stored](graphs/sup/dgemm_rrr_zen_nt1.png)
* **Zen single-threaded column-stored**
![single-threaded column-stored](graphs/sup/dgemm_ccc_zen_nt1.png)
* **Zen multithreaded (32 cores) row-stored**
![multithreaded row-stored](graphs/sup/dgemm_rrr_zen_nt32.png)
* **Zen multithreaded (32 cores) column-stored**
![multithreaded column-stored](graphs/sup/dgemm_ccc_zen_nt32.png)

---

## Zen2

### Zen2 experiment details

* Location: Oracle cloud
* Processor model: AMD Epyc 7742 (Zen2 "Rome")
* Core topology: two sockets, 8 Core Complex Dies (CCDs) per socket, 2 Core Complexes (CCX) per CCD, 4 cores per CCX, 128 cores total
* SMT status: enabled, but not utilized
* Max clock rate: 2.25GHz (base, documented); 3.4GHz boost (single-core, documented); 2.6GHz boost (multicore, estimated)
* Max vector register length: 256 bits (AVX2)
* Max FMA vector IPC: 2
  * Alternatively, FMA vector IPC is 4 when vectors are limited to 128 bits each.
* Peak performance:
  * single-core: 54.4 GFLOPS (double-precision), 108.8 GFLOPS (single-precision)
  * multicore (estimated): 41.6 GFLOPS/core (double-precision), 83.2 GFLOPS/core (single-precision)
* Operating system: Ubuntu 18.04 (Linux kernel 4.15.0)
* Page size: 4096 bytes
* Compiler: gcc 9.3.0
* Results gathered: 8 October 2020
* Implementations tested:
  * BLIS a0849d3 (0.7.0-67)
    * configured with `./configure --enable-cblas auto` (single-threaded)
    * configured with `./configure --enable-cblas -t openmp auto` (multithreaded)
    * sub-configuration exercised: `zen2`
    * Multithreaded (32 cores) execution requested via `export BLIS_NUM_THREADS=32`
  * OpenBLAS 0.3.10
    * configured `Makefile.rule` with `BINARY=64 NO_LAPACK=1 NO_LAPACKE=1 USE_THREAD=0 USE_LOCKING=1` (single-threaded)
    * configured `Makefile.rule` with `BINARY=64 NO_LAPACK=1 NO_LAPACKE=1 USE_THREAD=1 NUM_THREADS=32` (multithreaded)
    * Multithreaded (32 cores) execution requested via `export OPENBLAS_NUM_THREADS=32`
  * BLASFEO 5b26d40
    * configured `Makefile.rule` with: `BLAS_API=1 FORTRAN_BLAS_API=1 CBLAS_API=1`.
    * built BLAS library via `make CC=gcc`
  * Eigen 3.3.90
    * Obtained via the [Eigen GitLab homepage](https://gitlab.com/libeigen/eigen) (24 September 2020)
    * Prior to compilation, modified top-level `CMakeLists.txt` to ensure that `-march=native` was added to `CXX_FLAGS` variable (h/t Sameer Agarwal):
         ```
         # These lines added after line 60.
         check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
         if(COMPILER_SUPPORTS_MARCH_NATIVE)
           set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
         endif()
         ```
    * configured and built BLAS library via `mkdir build; cd build; CC=gcc cmake ..; make blas`
    * installed headers via `cmake . -DCMAKE_INSTALL_PREFIX=$HOME/flame/eigen; make install`
    * The `gemm` implementation was pulled in at compile-time via Eigen headers; other operations were linked to Eigen's BLAS library.
    * Single-threaded (1 core) execution requested via `export OMP_NUM_THREADS=1`
    * Multithreaded (32 cores) execution requested via `export OMP_NUM_THREADS=32`
  * MKL 2020 update 3
    * Single-threaded (1 core) execution requested via `export MKL_NUM_THREADS=1`
    * Multithreaded (32 cores) execution requested via `export MKL_NUM_THREADS=32`
  * libxsmm f0ab9cb (post-1.16.1)
    * compiled with `make AVX=2`; linked with [netlib BLAS](http://www.netlib.org/blas/) 3.6.0 as the fallback library to better show where libxsmm stops handling the computation internally.
* Affinity:
  * Thread affinity for BLIS was specified manually via `GOMP_CPU_AFFINITY="0-31"`. However, multithreaded OpenBLAS appears to revert to single-threaded execution if `GOMP_CPU_AFFINITY` is set. Therefore, when measuring OpenBLAS performance, the `GOMP_CPU_AFFINITY` environment variable was unset.
  * All executables were run through `numactl --interleave=all`.
* Frequency throttling (via `cpupower`):
  * Driver: acpi-cpufreq
  * Governor: performance
  * Hardware limits (steps): 1.5GHz, 2.0GHz, 2.25GHz
  * Adjusted minimum: 2.25GHz
* Comments:
  * None.

### Zen2 results

#### pdf

* [Zen2 sgemm single-threaded row-stored](graphs/sup/sgemm_rrr_zen2_nt1.pdf)
* [Zen2 sgemm single-threaded column-stored](graphs/sup/sgemm_ccc_zen2_nt1.pdf)
* [Zen2 dgemm single-threaded row-stored](graphs/sup/dgemm_rrr_zen2_nt1.pdf)
* [Zen2 dgemm single-threaded column-stored](graphs/sup/dgemm_ccc_zen2_nt1.pdf)
* [Zen2 sgemm multithreaded (32 cores) row-stored](graphs/sup/sgemm_rrr_zen2_nt32.pdf)
* [Zen2 sgemm multithreaded (32 cores) column-stored](graphs/sup/sgemm_ccc_zen2_nt32.pdf)
* [Zen2 dgemm multithreaded (32 cores) row-stored](graphs/sup/dgemm_rrr_zen2_nt32.pdf)
* [Zen2 dgemm multithreaded (32 cores) column-stored](graphs/sup/dgemm_ccc_zen2_nt32.pdf)

#### png (inline)

* **Zen2 sgemm single-threaded row-stored**
![sgemm single-threaded row-stored](graphs/sup/sgemm_rrr_zen2_nt1.png)
* **Zen2 sgemm single-threaded column-stored**
![sgemm single-threaded column-stored](graphs/sup/sgemm_ccc_zen2_nt1.png)
* **Zen2 dgemm single-threaded row-stored**
![dgemm single-threaded row-stored](graphs/sup/dgemm_rrr_zen2_nt1.png)
* **Zen2 dgemm single-threaded column-stored**
![dgemm single-threaded column-stored](graphs/sup/dgemm_ccc_zen2_nt1.png)
* **Zen2 sgemm multithreaded (32 cores) row-stored**
![sgemm multithreaded row-stored](graphs/sup/sgemm_rrr_zen2_nt32.png)
* **Zen2 sgemm multithreaded (32 cores) column-stored**
![sgemm multithreaded column-stored](graphs/sup/sgemm_ccc_zen2_nt32.png)
* **Zen2 dgemm multithreaded (32 cores) row-stored**
![dgemm multithreaded row-stored](graphs/sup/dgemm_rrr_zen2_nt32.png)
* **Zen2 dgemm multithreaded (32 cores) column-stored**
![dgemm multithreaded column-stored](graphs/sup/dgemm_ccc_zen2_nt32.png)

---

# Feedback

Please let us know what you think of these performance results! Similarly, if you have any questions or concerns, or are interested in reproducing these performance experiments on your own hardware, we invite you to [open an issue](https://github.com/flame/blis/issues) and start a conversation with BLIS developers.

Thanks for your interest in BLIS!

