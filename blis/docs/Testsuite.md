# Contents

* **[Contents](Testsuite.md#contents)**
* **[BLIS testsuite](Testsuite.md#blis-testsuite)**
  * **[Introduction](Testsuite.md#introduction)**
  * **[Compiling](Testsuite.md#compiling)**
  * **[Setting test parameters](Testsuite.md#setting-test-parameters)**
    * [`input.general`](Testsuite.md#inputgeneral)
    * [`input.operations`](Testsuite.md#inputoperations)
  * **[Running tests](Testsuite.md#running-tests)**
  * **[Interpreting the results](Testsuite.md#interpreting-the-results)**
* **[BLAS test drivers](Testsuite.md#blas-test-drivers)**

# BLIS testsuite

## Introduction

This wiki explains how to use the test suite included with the BLIS framework.

The test suite exists in the `testsuite` directory within the top-level source distribution:
```
$ ls
CHANGELOG        Makefile   common.mk        examples     sandbox     version
CONTRIBUTING.md  README.md  config           frame        so_version  windows
CREDITS          RELEASING  config_registry  kernels      test
INSTALL          blastest   configure        mpi_test     testsuite
LICENSE          build      docs             ref_kernels  travis

```
There, you will find a `Makefile`, a script, several input files, and two directories:
```
$ cd testsuite
$ ls
Makefile            input.general.mixed    input.operations.mixed
check-blistest.sh   input.general.salt     input.operations.salt
input.general       input.operations       obj
input.general.fast  input.operations.fast  src
```
As you would expect, the test suite's source code lives in `src` and the object files, upon being built, are placed in `obj`. The two `input.*` files control how the test suite runs, while the `Makefile` controls how the test suite executable is compiled and linked. However, only two input files are used at any given time: one `input.general` and one `input.operations`. (We have several pairs so that Travis CI can run multiple variations of tests automatically when new commits are made to github.) You can focus your attention on the general-purpose input files `input.general` and `input.operations`.

## Compiling

Before running the test suite, you must first configure and compile BLIS. (Installing BLIS is not necessary to run the test suite, though it is supported.) For directions on how to build and install a BLIS library, please see the [Build System](BuildSystem.md) guide.

Once BLIS is installed, you are ready to compile the test suite.

When you are ready to compile, simply run `make` from within the `testsuite` directory. Running `make` will result in output similar to:
:
```
$ make
Compiling src/test_addm.c
Compiling src/test_addv.c
Compiling src/test_amaxv.c
Compiling src/test_axpbyv.c
Compiling src/test_axpy2v.c
Compiling src/test_axpyf.c
Compiling src/test_axpym.c
Compiling src/test_axpyv.c
Compiling src/test_copym.c
Compiling src/test_copyv.c
```
As with compiling a BLIS library, if you are working in a multicore environment, you may use the `-j<n>` option to compile source code in parallel with `<n>` parallel jobs:
```
$ make -j4
```
After `make` is complete, an executable named `test_libblis.x` is created:
```
$ ls
Makefile            input.general.mixed    input.operations.mixed  test_libblis.x
check-blistest.sh   input.general.salt     input.operations.salt
input.general       input.operations       obj
input.general.fast  input.operations.fast  src
```

### Compiling/linking aginst an installed copy of BLIS

By default, the `Makefile` in the `testsuite` directory is programmed to look in
`../include/<config_name>/` for `blis.h` and `../lib/<config_name>/` for the BLIS library. However, some users may wish to run the testsuite after installing BLIS and deleting the entire source tree. In this situation, it is necessary to point `make` to the location of your BLIS installation (i.e., the installation prefix). If you would like to compile with an installed header and link against an installed library, you have two options:
1. First, you may set the envrionment variable `BLIS_INSTALL_PATH` to the install prefix used when BLIS was installed, and then run `make`. In this example, we assume that BLIS was installed after running the `configure` script with the `--prefix=/usr/local` option.
   ```
   $ export BLIS_INSTALL_PATH=/usr/local
   $ make
   ```
2. Alternatively, you may set the `make` variable `BLIS_INSTALL_PATH` on the command line as you execute `make`:
   ```
   $ make BLIS_INSTALL_PATH=/usr/local
   ```
Both options result in the same outcome: `make` looks for the BLIS installation in `BLIS_INSTALL_PATH` when building the test suite.

## Setting test parameters

The BLIS test suite reads two input files, `input.general` and `input.operations`, to determine which tests to run and how those tests are run. Each file is contains comments and thus you may find them intuitive to use without formal instructions. However, for completeness and as a reference-of-last-resort, we describe each file and its contents in detail.

### `input.general`

The `input.general` input file, as its name suggests, contains parameters that control the general behavior of the test suite. These parameters (more or less) apply to all operations that get tested. Below is a representative example of the default contents of `input.general`.
```
# ----------------------------------------------------------------------
#
#  input.general
#  BLIS test suite
#
#  This file contains input values that control how BLIS operations are
#  tested. Comments explain the purpose of each parameter as well as
#  accepted values.
#

1       # Number of repeats per experiment (best result is reported)
c       # Matrix storage scheme(s) to test:
        #   'c' = col-major storage; 'g' = general stride storage;
        #   'r' = row-major storage
c       # Vector storage scheme(s) to test:
        #   'c' = colvec / unit stride; 'j' = colvec / non-unit stride;
        #   'r' = rowvec / unit stride; 'i' = rowvec / non-unit stride
0       # Test all combinations of storage schemes?
1       # Perform all tests with alignment?
        #   '0' = do NOT align buffers/ldims; '1' = align buffers/ldims
0       # Randomize vectors and matrices using:
        #   '0' = real values on [-1,1];
        #   '1' = powers of 2 in narrow precision range
32      # General stride spacing (for cases when testing general stride)
sdcz    # Datatype(s) to test:
        #   's' = single real; 'c' = single complex;
        #   'd' = double real; 'z' = double complex
0       # Test gemm with mixed-domain operands?
0       # Test gemm with mixed-precision operands?
100     # Problem size: first to test
300     # Problem size: maximum to test
100     # Problem size: increment between experiments
        # Complex level-3 implementations to test
1       #   1m   ('1' = enable; '0' = disable)
1       #   native ('1' = enable; '0' = disable)
1       # Simulate application-level threading:
        #   '1' = disable / use one testsuite thread;
        #   'n' = enable and use n testsuite threads
1       # Error-checking level:
        #   '0' = disable error checking; '1' = full error checking
i       # Reaction to test failure:
        #   'i' = ignore; 's' = sleep() and continue; 'a' = abort
0       # Output results in matlab/octave format? ('1' = yes; '0' = no)
0       # Output results to stdout AND files? ('1' = yes; '0' = no)
```
The remainder of this section explains each parameter switch in detail.

_**Number of repeats.**_ This is the number of times an operation is run for each result that is reported. The result with the best performance is reported.

_**Matrix storage scheme.**_ This string encodes all of the matrix storage schemes that are tested (for operations that contain matrix operands). There are three valid values: `'c'` for column storage, `'r'` for row storage, and `'g'` for general stride storage. You may choose one storage scheme, or combine more than one. The order of the characters determines the order in which the corresponding storage schemes are tested.

_**Vector storage scheme.**_ Similar to the matrix storage scheme string, this string determines which vector storage schemes are tested (for operations that contain vector operands). There are four valid values: `'c'` for column vectors with unit stride, `'r'` for row vectors with unit stride, `'j'` for column vectors with non-unit stride, and `'i'` for row vectors with non-unit stride. You may choose any one storage scheme, or combine more than one. The ordering behaves similarly to that of the matrix storage scheme string. Using `cj` will test both unit and non-unit vector strides, and since row and column vectors are logically equivalent, this should provide complete test coverage for operations with vector operands.

_**Test all combinations of storage schemes?**_ Enabling this option causes all combinations of storage schemes to be tested. For example, if the option is disabled, a matrix storage scheme string of `cr` would cause the `gemm` test module to test execution where all matrix operands are column-stored, and then where all matrix operands are row-stored. Enabling this option with the same matrix storage string (`cr`) would cause the test suite to test `gemm` under all eight scenarios where the three `gemm` matrix operands are either column-stored or row-stored.

_**Perform all tests with alignment?**_ Disabling this option causes the leading dimension (row or column stride) of test matrices to **not** be aligned according to `BLIS_HEAP_STRIDE_ALIGN_SIZE`, which defaults to `BLIS_SIMD_ALIGN_SIZE`, which defaults to `BLIS_SIMD_MAX_SIZE`, which defaults to 64 (bytes). (If any of these values is set to a non-default value, it would be in `bli_family_<arch>.h` where `<arch>` is the configuration family.) Sometimes it's useful to disable leading dimension alignment in order to test certain aspects of BLIS that need to handle computing with unaligned user data, such as level-1v and level-1f kernels.

_**Randomize vectors and matrices.**_ The default randomization method uses real values on the interval [-1,1]. However, we offer an alternate randomization using powers of two in a narrow precision range, which is more likely to result in test residuals exactly equal to zero. This method is somewhat niche/experimental and most people should use random values on the [-1,1] interval.

_**General stride spacing.**_ This value determines the simulated "inner" stride when testing general stride storage. For simplicity, the test suite always generates and tests general stride storage that is ["column-tilted"](FAQ.md#What_does_it_mean_when_a_matrix_with_general_stride_is_column-ti). If general stride storage is not being tested, then this value is ignored.

_**Datatype(s) to test.**_ This string determines which floating-point datatypes are tested. There are four valid values: `'s'` for single-precision real, `'d'` for double-precision real, `'c'` for single-precision complex, and `'z'` for double-precision complex. You may choose one datatype, or combine more than one. The order of the datatype characters determines the order in which they are tested.

_**Test gemm with mixed-domain operands?**_ This boolean determines whether `gemm` tests are performed that exercise the mixed-domain functionality within BLIS. (In other words, with precision held constant, all combinations of real and complex matrix operands will be tested.) If this option is set to 1 and the mixed-precision option is set to 0, then domain combinations will be varied for the precisions represented by the "Datatype(s) to test" option. If this option and the mixed-precision option are both set to 1, then _all_ datatype combinations will be tested, regardless of the datatypes indicated by the "Datatype(s) to test" option.

_**Test gemm with mixed-precision operands?**_ This boolean determines whether `gemm` tests are performed that exercise the mixed-precision functionality within BLIS. (In other words, with domain held constant, all combinations of supported precisions will be tested.) If this option is set to 1 and the mixed-domain option is set to 0, then precision combinations will be varied for the domains represented by the "Datatype(s) to test" option. If this option and the mixed-domain option are both set to 1, then _all_ datatype combinations will be tested, regardless of the datatypes indicated by the "Datatype(s) to test" option.

_**Problem size.**_ These values determine the first problem size to test, the maximum problem size to test, and the increment between problem sizes. Note that the maximum problem size only bounds the range of problem sizes; it is not guaranteed to be tested. Example: If the initial problem size is 128, the maximum is 1000, and the increment is 64, then the last problem size to be tested will be 960.

_**Complex level-3 implementations to test.**_ This section lists which complex domain implementations of level-3 operations are tested. If you don't know what these are, you can ignore them. The `native` switch corresponds to native execution of complex domain level-3 operations, which we test by default. We also test the `1m` method, since it is the induced method of choice when optimized complex microkernels are not available. Note that all of these induced method tests (including `native`) are automatically disabled if the `c` and `z` datatypes are disabled.

_**Simulate application-level threading.**_ This setting specifies the number of threads the testsuite will spawn, and is meant to allow the user to exercise BLIS as a multithreaded application might if it were to make multiple concurrent calls to BLIS operations. (Note that the threading controlled by this option is orthogonal to, and has no effect on, whatever multithreading may be employed _within_ BLIS, as specified by the environment variables described in the [Multithreading](Multithreading.md) documentation.) When this option is set to 1, the testsuite is run with only one thread. When set to n > 1 threads, the spawned threads will parallelize (in round-robin fashion) the total set of tests specified by the testsuite input files, executing them in roughly the same order as that of a sequential execution.

_**Error-checking level.**_ BLIS supports various "levels" of error checking prior to executing most operations. For now, only two error-checking levels are implemented: fully disabled (`'0'`) and fully enabled (`'1'`). Disabling error-checking may improve performance on some systems for small problem sizes, but generally speaking the cost is negligible.

_**Reaction to test failure.**_ If the test suite executes a test that results in a numerical result that is considered a "failure", this character determines how the test suite should proceed. There are three valid values: `'i'` will cause the test suite to ignore the failure and immediately continue with all remaining tests, `'s'` will cause the test suite to sleep for some short period of time before continuing, and `'a'` will cause the test suite to abort all remaining tests. The user must specify only **one** option via its character encoding.

_**Output results in Matlab/Octave format?**_ When this option is disabled, the test suite outputs results in a simple human-readable format of one experiment per line. When this option is enabled, the test suite similarly outputs results for one experiment per line, but in a format that may be read into Matlab or Octave. This is useful if the user intends to use the results of the test suite to plot performance data using one of these tools.

_**Output results to `stdout` AND files?**_ When this option is disabled, the test suite outputs only to standard output. When enabled, the test suite also writes its output to files, one for each operation tested. As with the Matlab/Octave option above, this option may be useful to some users who wish to gather and retain performance data for later use.


### `input.operations`

The `input.operations` input file determines **which** operations are tested, which parameter combinations are tested, and the relative sizes of the operation's dimensions. The file itself contains comments that explain various sections. However, we reproduce this information here for your convenience.

_**Enabling/disabling entire sections.**_ The values in the "Section overrides" section allow you to disable all operations in a given "level". Enabling a level here by itself does not enable every operation in that level; it simply means that the individual switches for each operation (in that level) determine whether or not the tests are executed. Use 1 to enable a section, or 0 to disable.

_**Enabling/disabling individual operation tests.**_ Given that an operation's section override switch is set to 1 (enabled), whether or not that operation will get tested is determined by its local switch. For example, if the level-1v section override is set to 1, and there is a 1 on the line marked `addv`, then the `addv` operation will be tested. NOTE: You may ignore the lines marked "test sequential front-end." These lines are for future use, to distinguish tests of the sequential implementation from tests of the multithreaded implementation. For now, BLIS does not contain separate APIs for multithreaded execution, even though multithreading is supported. So, these should be left set to 1.

_**Enabling only select operations**_ If you would like to enable just a few (or even just one) operation without adjusting any section overrides (or individual operation switches), change the desired operation switch(es) to 2. This will cause any operation that is not set to 2 to be disabled, regardless of section override values. For example, setting the `axpyv` and `gemv` operation switches to 2 will cause the test suite to test ONLY `axpyv` and `gemv`, even if all other sections and operations are set to 1. NOTE: As long as there is at least on operation switch set to 2, no other operations will be tested. When you are done testing your select operations, you should revert the operation switch(es) back to 1.

_**Changing the problem size/shapes tested.**_ The problem sizes tested by an operation are determined by the dimension specifiers on the line marked `dimensions: <spec_labels>`. If, for example, `<spec_labels>` contains two dimension labels (e.g. `m n`), then the line should begin with two dimension specifiers. Dimension specifiers of `-1` cause the corresponding dimension to be bound to the problem size, which is determined by values set in `input.general`. Positive values cause the corresponding dimension to be fixed to that value and held constant. Examples of dimension specifiers (where the dimensions are _m_ and _n_):
  * `-1 -1 `   ...Dimensions m and n grow with problem size (resulting in square matrices).
  * `-1 150 `   ...Dimension m grows with problem size and n is fixed at 150.
  * `-1 -2 `   ...Dimension m grows with problem size and n grows proportional to half the problem size.

_**Changing parameter combinations tested.**_ The parameter combinations tested by an operation are determined by the parameter specifier characters on the line marked `parameters: <param_labels>`. If, for example, `<param_labels>` contains two parameter labels (e.g. `transa conjx`), then the line should contain two parameter specifier characters. The `'?'` specifier character serves as a wildcard--it causes all possible values of that parameter to be tested. A character such as `'n'` or `'t'` causes only that value to be tested. Examples of parameter specifiers (where the parameters are `transa` and `conjx`):
  * `??`   ...All combinations of the `transa` and `conjx` parameters are tested: `nn, nc, tn, tc, cn, cc, hn, hc`.
  * `?n`   ...`conjx` is fixed to "no conjugate" but `transa` is allowed to vary: `nn, tn, cn, hn`.
  * `hc`   ...Only the case where `transa` is "Hermitian-transpose" and `conjx` is "conjugate" is tested.

Here is a full list of the parameter types used by the various BLIS operations along with their possible character encodings:
  * `side`: `l` = left,  `r` = right
  * `uplo`: `l` = lower-stored, `u` = upper-stored
  * `trans`: `n` = no transpose, `t` = transpose, `c` = conjugate, `h` = Hermitian-transpose (conjugate-transpose)
  * `conj`: `n` = no conjugate, `c` = conjugate
  * `diag`: `n` = non-unit diagonal, `u` = unit diagonal


## Running tests

Running the test suite is easy. Once `input.general` and `input.operations` have been tailored to your liking, simply run the test suit executable:
```
$ ./test_libblis.x
```
For sanity-checking purposes, the test suite begins by echoing the parameters it found in `input.general` to standard output. This is useful when troubleshooting the test suite if/when it exhibits strange behavior (such as seemingly skipped tests).

## Interpreting the results

The output to the test suite is more-or-less intuitive. Here is an snippet of output for the `gemm` test module when problem sizes of 100 to 300 in increments of 100 are tested.
```
% --- gemm ---
%
% test gemm seq front-end?    1
% gemm m n k                  -1 -1 -2
% gemm operand params         ??
%

% blis_<dt><oper>_<params>_<storage>           m     n     k   gflops  resid       result
blis_sgemm_nn_ccc                            100   100    50   1.447   1.14e-07    PASS
blis_sgemm_nn_ccc                            200   200   100   1.537   1.18e-07    PASS
blis_sgemm_nn_ccc                            300   300   150   1.532   1.38e-07    PASS
blis_sgemm_nc_ccc                            100   100    50   1.449   7.79e-08    PASS
blis_sgemm_nc_ccc                            200   200   100   1.540   1.23e-07    PASS
blis_sgemm_nc_ccc                            300   300   150   1.537   1.54e-07    PASS
blis_sgemm_nt_ccc                            100   100    50   1.479   7.40e-08    PASS
blis_sgemm_nt_ccc                            200   200   100   1.549   1.33e-07    PASS
blis_sgemm_nt_ccc                            300   300   150   1.534   1.44e-07    PASS
blis_sgemm_nh_ccc                            100   100    50   1.477   9.23e-08    PASS
blis_sgemm_nh_ccc                            200   200   100   1.547   1.13e-07    PASS
blis_sgemm_nh_ccc                            300   300   150   1.535   1.51e-07    PASS
blis_sgemm_cn_ccc                            100   100    50   1.477   9.62e-08    PASS
blis_sgemm_cn_ccc                            200   200   100   1.548   1.36e-07    PASS
blis_sgemm_cn_ccc                            300   300   150   1.539   1.51e-07    PASS
blis_sgemm_cc_ccc                            100   100    50   1.481   8.66e-08    PASS
blis_sgemm_cc_ccc                            200   200   100   1.549   1.41e-07    PASS
blis_sgemm_cc_ccc                            300   300   150   1.539   1.63e-07    PASS
blis_sgemm_ct_ccc                            100   100    50   1.484   7.09e-08    PASS
blis_sgemm_ct_ccc                            200   200   100   1.549   1.08e-07    PASS
blis_sgemm_ct_ccc                            300   300   150   1.539   1.33e-07    PASS
blis_sgemm_ch_ccc                            100   100    50   1.471   8.06e-08    PASS
blis_sgemm_ch_ccc                            200   200   100   1.546   1.24e-07    PASS
blis_sgemm_ch_ccc                            300   300   150   1.539   1.66e-07    PASS
```

Before each operation is tested, the test suite echos information it obtained from the `input.operations` file, such as the dimension specifier string (in this case, `"-1 -1 -2"`) and parameter specifier string (`"??"`).

Each line of output contains several sections. We will cover them now, from left to right.

_**Test identifier.**_ The left-most labels are strings which identify the specific test being performed. This string generally a concatenation of substrings, joined by underscores, which identify the operation being run, the parameter combination tested, and the storage scheme of each operand. When outputting to Matlab/Octave formatting is abled, these identifiers service as the names of the arrays in which the data are stored.

_**Dimensions.**_ The values near the middle of the output show the size of each dimension. Different operations have different dimension sets. For example, `gemv` only has two dimensions, _m_ and _n_, while `gemm` has an additional _k_ dimension. In the snippet above, you can see that the dimension specifier string, `"-1 -1 -2"`, explains the relative sizes of the dimensions for each test: _m_ and _n_ are bound to the problem size, while _k_ is always equal to half the problem size.

_**Performance.**_ The next value output is raw performance, reported in GFLOPS (billions of floating-point operations per second).

_**Residual.**_ The next value, which we loosely refer to as a "residual", reports the result of the numerical correctness test for the operation. The actual method of computing the residual (and hence its exact meaning) depends on the operation being tested. However, these residuals are always computed such that the result should be no more than 2-3 orders of magnitude away from machine precision for the datatype being tested. Thus, "good" results are typically in the neighborhood of `5e-06` for single precision and `1e-16` for double precision (preferrably less).

_**Test result.**_ The BLIS test suite compares the residual to internally-defined accuracy thresholds to categorize the test as either `PASS`, `MARGINAL`, or `FAIL`. The vast majority of tests should result in a `PASS` result, with perhaps a handful resulting in `MARGINAL`. Usually, a `MARGINAL` result is no cause for concern, especially when similar tests result in `PASS`.

Note that the various sections of output, which line up nicely as columns, are labeled on a line beginning with `%` immediately before the results:
```
% blis_<dt><oper>_<params>_<storage>           m     n     k   gflops  resid       result
blis_sgemm_nn_ccc                            100   100    50   1.447   1.14e-07    PASS
```
These labels are useful as concise reminders of the meaning of each column. They are especially useful in differentiating the various dimensions from each other for operations that contain two or three dimensions.

If you simply want to run the BLIS testsuite and know if there were any failures, you can do so via the `make check` and `make check-fast`. The former uses the `input.general` and `input.operations` files, while the latter uses the `input.general.fast` and `input.operations.fast`. (We generally recommend using the "fast" target since it usually finishes in much less time while still being relatively comprehensive.) A one-line characterization of the test results is output after the tests finish:
```
$ make check-fast
Running test_libblis.x (fast) with output redirected to 'output.testsuite'
check-blistest.sh: All BLIS tests passed!
```

# BLAS test drivers

In addition to the monolithic testsuite located in the `testsuite` directory, which exercises BLIS functionality in general (and via one of its native/preferred APIs), we also provide a C port of the netlib BLAS test drivers included in netlib LAPACK. These BLAS drivers are located in `blastest`, along with other files needed in order to build the drivers, such as a subset of `libf2c`. After configuring and compiling BLIS, the BLAS test drivers may be run from within `blastest`:
```
$ ./configure haswell
# Lots of configure output...
$ make -j4
# Lots of compilation output...
$ cd blastest
$ ls
Makefile  f2c  input  obj  src
```
Simply run `make`:
```
$ make
Compiling obj/abs.o
Compiling obj/acos.o
Compiling obj/asin.o
Compiling obj/atan.o
...
Compiling obj/wsfe.o
Compiling obj/wsle.o
Archiving libf2c.a
Compiling obj/cblat1.o
Linking cblat1.x against 'libf2c.a ../lib/haswell/libblis.a -lm -lpthread -lrt'
Compiling obj/cblat2.o
Linking cblat2.x against 'libf2c.a ../lib/haswell/libblis.a -lm -lpthread -lrt'
Compiling obj/cblat3.o
Linking cblat3.x against 'libf2c.a ../lib/haswell/libblis.a -lm -lpthread -lrt'
...
```
And then `make run`:
```
Running cblat1.x > 'out.cblat1'
Running cblat2.x < 'input/cblat2.in' (output to 'out.cblat2')
Running cblat3.x < 'input/cblat3.in' (output to 'out.cblat3')
Running dblat1.x > 'out.dblat1'
Running dblat2.x < 'input/dblat2.in' (output to 'out.dblat2')
Running dblat3.x < 'input/dblat3.in' (output to 'out.dblat3')
Running sblat1.x > 'out.sblat1'
Running sblat2.x < 'input/sblat2.in' (output to 'out.sblat2')
Running sblat3.x < 'input/sblat3.in' (output to 'out.sblat3')
Running zblat1.x > 'out.zblat1'
Running zblat2.x < 'input/zblat2.in' (output to 'out.zblat2')
Running zblat3.x < 'input/zblat3.in' (output to 'out.zblat3')
```
The results can quickly be checked via a script in the top-level `build` directory:
```
$ ../build/check-blastest.sh
All BLAS tests passed!
```
This is the message we expect when everything works as expected. You can also combine the `make`, `make run`, and script execution into one command: `make check`.

Alternatively, you can execute all of the steps described above (`make ; make run; ../build/check-blastest.sh`, or `make check`) from the top-level directory. After running `configure` and `make`, simply run `make checkblas`:
```
$ ./configure haswell
# Lots of configure output...
$ make -j4
# Lots of compilation output...
$ make check
```
This will build all of the necessary BLAS test driver object files, link them, and run the drivers. Output will go to the current directory (either the top-level directory of the source distribution, or the out-of-tree build directory from which you ran `configure`), with each output file (prefixed with `out.`) named according to the BLAS driver that generated its contents:
```
$ ls
CHANGELOG        blastest         docs      out.cblat1  out.sblat3   testsuite
CONTRIBUTING.md  bli_config.h     examples  out.cblat2  out.zblat1   travis
CREDITS          build            frame     out.cblat3  out.zblat2   version
INSTALL          common.mk        include   out.dblat1  out.zblat3   windows
LICENSE          config           kernels   out.dblat2  ref_kernels
Makefile         config.mk        lib       out.dblat3  sandbox
README.md        config_registry  mpi_test  out.sblat1  so_version
RELEASING        configure        obj       out.sblat2  test
```
If any of the tests fail, you'll instead see the message:
```
$ make check
At least one BLAS test failed. Please see out.* files for details.
```
As the message suggests, you should inspect the `out.*` files for more details about what went wrong.

