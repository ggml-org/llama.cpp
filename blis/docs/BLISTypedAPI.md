# Contents

* **[Contents](BLISTypedAPI.md#contents)**
* **[Operation index](BLISTypedAPI.md#operation-index)**
* **[Introduction](BLISTypedAPI.md#introduction)**
  * [BLIS types](BLISTypedAPI.md#blis-types)
    * [Integer-based types](BLISTypedAPI.md#integer-based-types)
    * [Floating-point types](BLISTypedAPI.md#floating-point-types)
    * [Enumerated parameter types](BLISTypedAPI.md#enumerated-parameter-types)
  * [Basic vs expert interfaces](BLISTypedAPI.md#basic-vs-expert-interfaces)
  * [Context type](BLISTypedAPI.md#context-type)
  * [Runtime type](BLISTypedAPI.md#runtime-type)
  * [BLIS header file](BLISTypedAPI.md#blis-header-file)
  * [Initialization and cleanup](BLISTypedAPI.md#initialization-and-cleanup)
* **[Computational function reference](BLISTypedAPI.md#computational-function-reference)**
  * [Level-1v operations](BLISTypedAPI.md#level-1v-operations)
  * [Level-1d operations](BLISTypedAPI.md#level-1d-operations)
  * [Level-1m operations](BLISTypedAPI.md#level-1m-operations)
  * [Level-1f operations](BLISTypedAPI.md#level-1f-operations)
  * [Level-2 operations](BLISTypedAPI.md#level-2-operations)
  * [Level-3 operations](BLISTypedAPI.md#level-3-operations)
  * [Utility operations](BLISTypedAPI.md#utility-operations)
  * [Level-3 microkernels](BLISTypedAPI.md#level-3-microkernels)
* **[Query function reference](BLISTypedAPI.md#query-function-reference)**
  * [General library information](BLISTypedAPI.md#general-library-information)
  * [Specific configuration](BLISTypedAPI.md#specific-configuration)
  * [General configuration](BLISTypedAPI.md#general-configuration)
  * [Kernel information](BLISTypedAPI.md#kernel-information)
  * [Clock functions](BLISTypedAPI.md#clock-functions)
* **[Example code](BLISTypedAPI.md#example-code)**



# Operation index

This index provides a quick way to jump directly to the description for each operation discussed later in the [Computational function reference](BLISTypedAPI.md#computational-function-reference) section:

  * **[Level-1v](BLISTypedAPI.md#level-1v-operations)**: Operations on vectors:
    * [addv](BLISTypedAPI.md#addv), [amaxv](BLISTypedAPI.md#amaxv), [axpyv](BLISTypedAPI.md#axpyv), [axpbyv](BLISTypedAPI.md#axpbyv), [copyv](BLISTypedAPI.md#copyv), [dotv](BLISTypedAPI.md#dotv), [dotxv](BLISTypedAPI.md#dotxv), [invertv](BLISTypedAPI.md#invertv), [invscalv](BLISTypedAPI.md#invscalv), [scalv](BLISTypedAPI.md#scalv), [scal2v](BLISTypedAPI.md#scal2v), [setv](BLISTypedAPI.md#setv), [subv](BLISTypedAPI.md#subv), [swapv](BLISTypedAPI.md#swapv), [xpbyv](BLISTypedAPI.md#xpbyv)
  * **[Level-1d](BLISTypedAPI.md#level-1d-operations)**: Element-wise operations on matrix diagonals:
    * [addd](BLISTypedAPI.md#addd), [axpyd](BLISTypedAPI.md#axpyd), [copyd](BLISTypedAPI.md#copyd), [invertd](BLISTypedAPI.md#invertd), [invscald](BLISTypedAPI.md#invscald), [scald](BLISTypedAPI.md#scald), [scal2d](BLISTypedAPI.md#scal2d), [setd](BLISTypedAPI.md#setd), [setid](BLISTypedAPI.md#setid), [shiftd](BLISTypedAPI.md#shiftd), [subd](BLISTypedAPI.md#subd), [xpbyd](BLISTypedAPI.md#xpbyd)
  * **[Level-1m](BLISTypedAPI.md#level-1m-operations)**: Element-wise operations on matrices:
    * [addm](BLISTypedAPI.md#addm), [axpym](BLISTypedAPI.md#axpym), [copym](BLISTypedAPI.md#copym), [invscalm](BLISTypedAPI.md#invscalm), [scalm](BLISTypedAPI.md#scalm), [scal2m](BLISTypedAPI.md#scal2m), [setm](BLISTypedAPI.md#setm), [subm](BLISTypedAPI.md#subm)
  * **[Level-1f](BLISTypedAPI.md#level-1f-operations)**: Fused operations on multiple vectors:
    * [axpy2v](BLISTypedAPI.md#axpy2v), [dotaxpyv](BLISTypedAPI.md#dotaxpyv), [axpyf](BLISTypedAPI.md#axpyf), [dotxf](BLISTypedAPI.md#dotxf), [dotxaxpyf](BLISTypedAPI.md#dotxaxpyf)
  * **[Level-2](BLISTypedAPI.md#level-2-operations)**: Operations with one matrix and (at least) one vector operand:
    * [gemv](BLISTypedAPI.md#gemv), [ger](BLISTypedAPI.md#ger), [hemv](BLISTypedAPI.md#hemv), [her](BLISTypedAPI.md#her), [her2](BLISTypedAPI.md#her2), [symv](BLISTypedAPI.md#symv), [syr](BLISTypedAPI.md#syr), [syr2](BLISTypedAPI.md#syr2), [trmv](BLISTypedAPI.md#trmv), [trsv](BLISTypedAPI.md#trsv)
  * **[Level-3](BLISTypedAPI.md#level-3-operations)**: Operations with matrices that are multiplication-like:
    * [gemm](BLISTypedAPI.md#gemm), [hemm](BLISTypedAPI.md#hemm), [herk](BLISTypedAPI.md#herk), [her2k](BLISTypedAPI.md#her2k), [symm](BLISTypedAPI.md#symm), [syrk](BLISTypedAPI.md#syrk), [syr2k](BLISTypedAPI.md#syr2k), [trmm](BLISTypedAPI.md#trmm), [trmm3](BLISTypedAPI.md#trmm3), [trsm](BLISTypedAPI.md#trsm)
  * **[Utility](BLISTypedAPI.md#Utility-operations)**: Miscellaneous operations on matrices and vectors:
    * [asumv](BLISTypedAPI.md#asumv), [norm1v](BLISTypedAPI.md#norm1v), [normfv](BLISTypedAPI.md#normfv), [normiv](BLISTypedAPI.md#normiv), [norm1m](BLISTypedAPI.md#norm1m), [normfm](BLISTypedAPI.md#normfm), [normim](BLISTypedAPI.md#normim), [mkherm](BLISTypedAPI.md#mkherm), [mksymm](BLISTypedAPI.md#mksymm), [mktrim](BLISTypedAPI.md#mktrim), [fprintv](BLISTypedAPI.md#fprintv), [fprintm](BLISTypedAPI.md#fprintm),[printv](BLISTypedAPI.md#printv), [printm](BLISTypedAPI.md#printm), [randv](BLISTypedAPI.md#randv), [randm](BLISTypedAPI.md#randm), [sumsqv](BLISTypedAPI.md#sumsqv), [getsc](BLISTypedAPI.md#getsc), [getijv](BLISTypedAPI.md#getijv), [getijm](BLISTypedAPI.md#getijm), [setsc](BLISTypedAPI.md#setsc), [setijv](BLISTypedAPI.md#setijv), [setijm](BLISTypedAPI.md#setijm), [eqsc](BLISTypedAPI.md#eqsc), [eqv](BLISTypedAPI.md#eqv), [eqm](BLISTypedAPI.md#eqm)



# Introduction

This document summarizes one of the primary native APIs in BLIS--the "typed" API. Here, we also discuss BLIS-specific type definitions, header files, and prototypes to auxiliary functions. This document also includes APIs to key kernels which are used to accelerate and optimize various level-2 and level-3 operations, though the [Kernels Guide](KernelsHowTo.md) goes into more detail, especially for level-3 microkernels.

There are many functions that BLIS implements that are not listed here, either because they are lower-level functions, or they are considered for use primarily by developers and experts.

For curious readers, the typed API was given its name (a) because it exposes the floating-point types in the names of its functions, and (b) to contrast it with the other native API in BLIS, the object API, which is [documented here](BLISObjectAPI.md). (The third API supported by BLIS is the BLAS compatibility layer, which mimics conventional Fortran-77 BLAS.)

In general, this document should be treated more as a reference than a place to learn how to use BLIS in your application. Thus, we highly encourage all readers to first study the [example code](BLISTypedAPI.md#example-code) provided within the BLIS source distribution.

## BLIS types

The following tables list various types used throughout the BLIS typed API.

### Integer-based types

| BLIS integer type | Type definition        | Used to represent...                                                 |
|:------------------|:-----------------------|:---------------------------------------------------------------------|
| `gint_t`          | `int32_t` or `int64_t` | general-purpose signed integer; used to define signed integer types. |
| `dim_t`           | `gint_t`               | matrix and vector dimensions.                                        |
| `inc_t`           | `gint_t`               | matrix row/column strides and vector increments.                     |
| `doff_t`          | `gint_t`               | matrix diagonal offset: if _k_ < 0, diagonal begins at element (-_k_,0); otherwise diagonal begins at element (0,_k_). |

### Floating-point types

| BLIS fp type | BLIS char | Type definition                        | Used to represent...              |
|:-------------|:----------|:---------------------------------------|:----------------------------------|
| `float`      | `s`       | _N/A_                                  | single-precision real numbers.    |
| `double`     | `d`       | _N/A_                                  | double-precision real numbers.    |
| `scomplex`   | `c`       | `struct { float real; float imag; }`   | single-precision complex numbers. |
| `dcomplex`   | `z`       | `struct { double real; double imag; }` | double-precision complex numbers. |

### Enumerated parameter types

| `trans_t`                | Semantic meaning: Matrix operand...             |
|:-------------------------|:------------------------------------------------|
| `BLIS_NO_TRANSPOSE`      | will be used as given.                          |
| `BLIS_TRANSPOSE`         | will be implicitly transposed.                  |
| `BLIS_CONJ_NO_TRANSPOSE` | will be implicitly conjugated.                  |
| `BLIS_CONJ_TRANSPOSE`    | will be implicitly transposed _and_ conjugated. |

| `conj_t`             | Semantic meaning: Matrix/vector operand... |
|:---------------------|:-------------------------------------------|
| `BLIS_NO_CONJUGATE`  | will be used as given.                     |
| `BLIS_CONJUGATE`     | will be implicitly conjugated.             |

| `side_t`     | Semantic meaning: Matrix operand... |
|:-------------|:------------------------------------|
| `BLIS_LEFT`  | appears on the left.                |
| `BLIS_RIGHT` | appears on the right.               |

| `uplo_t`     | Semantic meaning: Matrix operand...                               |
|:-------------|:------------------------------------------------------------------|
| `BLIS_LOWER` | is stored in (and will be accessed only from) the lower triangle. |
| `BLIS_UPPER` | is stored in (and will be accessed only from) the upper triangle. |
| `BLIS_DENSE` | is stored as a full matrix (ie: in both triangles).               |

| `diag_t`            | Semantic meaning: Matrix operand...                                        |
|:--------------------|:---------------------------------------------------------------------------|
| `BLIS_NONUNIT_DIAG` | has a non-unit diagonal that should be explicitly read from.               |
| `BLIS_UNIT_DIAG`    | has a unit diagonal that should be implicitly assumed (and not read from). |

### Basic vs expert interfaces

The functions listed in this document belong to the "basic" interface subset of the BLIS typed API. There is a companion "expert" interface that mirrors the basic interface, except that it also contains at least one additional parameter that is only of interest to experts and library developers. The expert interfaces use the same name as the basic function names, except for an additional "_ex" suffix. For example, the basic interface for `gemm` is
```c
void bli_?gemm
     (
             trans_t  transa,
             trans_t  transb,
             dim_t    m,
             dim_t    n,
             dim_t    k,
       const ctype*   alpha,
       const ctype*   a, inc_t rsa, inc_t csa,
       const ctype*   b, inc_t rsb, inc_t csb,
       const ctype*   beta,
             ctype*   c, inc_t rsc, inc_t csc
     );
```
while the expert interface is:
```c
void bli_?gemm_ex
     (
             trans_t  transa,
             trans_t  transb,
             dim_t    m,
             dim_t    n,
             dim_t    k,
       const ctype*   alpha,
       const ctype*   a, inc_t rsa, inc_t csa,
       const ctype*   b, inc_t rsb, inc_t csb,
       const ctype*   beta,
             ctype*   c, inc_t rsc, inc_t csc,
       const cntx_t*  cntx,
       const rntm_t*  rntm
     );
```
The expert interface contains two additional parameters: a `cntx_t*` and `rntm_t*`. Note that calling a function from the expert interface with the `cntx_t*` and `rntm_t*` arguments each set to `NULL` is equivalent to calling the corresponding basic interface. Specifically, a `NULL` value passed in for the `cntx_t*` results in a valid context being queried from BLIS, and a `NULL` value passed in for the `rntm_t*` results in the current global settings for multithreading to be used.

## Context type

In general, it is permissible to pass in `NULL` for a `cntx_t*` parameter when calling an expert interface such as `bli_dgemm_ex()`. However, there are cases where `NULL` values are not accepted and may result in a segmentation fault. Specifically, the `cntx_t*` argument appears in the interfaces to the `gemm`, `trsm`, and `gemmtrsm` [level-3 microkernels](KernelsHowTo.md#level-3) along with all [level-1v](KernelsHowTo.md#level-1v) and [level-1f](KernelsHowTo.md#level-1f) kernels. There, as a general rule, a valid pointer must be passed in. Whenever a valid context is needed, the developer may query a default context from the global kernel structure (if a context is not already available in the current scope):
```c
const cntx_t* bli_gks_query_cntx( void );
```
When BLIS is configured to target a configuration family (e.g. `intel64`, `x86_64`), `bli_gks_query_cntx()` will use `cpuid` or an equivalent heuristic to select and and return the appropriate context. When BLIS is configured to target a singleton sub-configuration (e.g. `haswell`, `skx`), `bli_gks_query_cntx()` will unconditionally return a pointer to the context appropriate for the targeted configuration.

## Runtime type

When calling one of the expert interfaces, a `rntm_t` (runtime) object can be used to convey a thread-local request for parallelism to the underlying implementation. Runtime objects are thread-safe by nature when they are declared statically as a stack variable (or allocated via `malloc()`), initialized, and then passed into the expert interface of interest.

Notice that runtime objects have no analogue in most BLAS libraries, where you are forced to specify parallelism at a global level (usually via environment variables).

For more information on using `rntm_t` objects, please read the [Multithreading](Multithreading.md) documentation, paying close attention to the section on [local setting of parallelism](Multithreading.md#locally-at-runtime).

## BLIS header file

All BLIS definitions and prototypes may be included in your C source file by including a single header file:

```c
#include "blis.h"
```


## Initialization and Cleanup

As of [9804adf](https://github.com/flame/blis/commit/9804adfd405056ec332bb8e13d68c7b52bd3a6c1), BLIS no longer requires explicit initialization and finalization at runtime. In other words, users do not need to call `bli_init()` before the application can make use of the library (and `bli_finalize()` after the application is finished with the library). Instead, all computational operations (and some non-computational functions) in BLIS will initialize the library on behalf of the user if it has not already been initialized. This change was made to simplify the user experience.

Application developers should keep in mind, however, that this new self-initialization regime implies the following: unless the library is *explicitly* finalized via `bli_finalize()`, it will, once initialized, remain initialized for the life of the application. This is likely not a problem in the vast majority of cases. However, a memory-constrained application that performs all of its DLA up-front, for example, may wish to explicitly finalize the library after BLIS is no longer needed in order to free up memory for other purposes.

Similarly, an expert user may call `bli_init()` manually in order to control when the overhead of library initialization is incurred, even though the library would have self-initialized.

The interfaces to `bli_init()` and `bli_finalize()` are quite simple; they require no arguments and return no values:
```c
void bli_init( void );
void bli_finalize( void );
```


# Computational function reference

Notes for interpreting the following prototypes:
  * Any occurrence of `?` should be replaced with `s`, `d`, `c`, or `z` to form an actual function name.
  * Any occurrence of `ctype` should be replaced with the actual C type corresponding to the datatype instance in question, while `rtype` should be replaced by the real projection of `ctype`. For example:
    * If we consider the prototype for `bli_zaxpyv()` below, `ctype` refers to `dcomplex`.
    * If we consider the prototype for `bli_znormfv()` below, `ctype` refers to `dcomplex` while `rtype` refers to `double`.
  * Any occurrence of `itype` should be replaced with the general-purpose signed integer type, `gint_t`.
  * All vector arguments have associated increments that proceed them, typically listed as `incX` for a given vector `x`. The semantic meaning of a vector increment is "the distance, in units of elements, between any two adjacent elements in the vector."
  * All matrix arguments have associated row and column strides arguments that proceed them, typically listed as `rsX` and `csX` for a given matrix `X`. Row strides are always listed first, and column strides are always listed second. The semantic meaning of a row stride is "the distance, in units of elements, to the next row (within a column)," and the meaning of a column stride is "the distance, in units of elements, to the next column (within a row)." Thus, unit row stride implies column-major storage and unit column stride implies row-major storage.

Notes for interpreting function descriptions:
  * `conjX()` and `transX()` should be interpreted as predicates that capture the operand X with any value of `conj_t` or `trans_t` applied. For example:
    * `conjx(x)` refers to a vector `x` that is either conjugated or used as given.
    * `transa(A)` refers to a matrix `A` that is either transposed, conjugated _and_ transposed, conjugated only, or used as given.
  * Any operand marked with `conj()` is unconditionally conjugated.
  * Any operand marked with `^T` is unconditionally transposed. Similarly, any operand that is marked with `^H` is unconditionally conjugate-transposed.
  * All occurrences of `alpha`, `beta`, and `rho` parameters are scalars.


---


## Level-1v operations

Level-1v operations perform various level-1 BLAS-like operations on vectors (hence the _v_).
**Note**: Most level-1v operations have a corresponding level-1v kernel through which it is primarily implemented.

---

#### addv
```c
void bli_?addv
     (
             conj_t  conjx,
             dim_t   n,
       const ctype*  x, inc_t incx,
             ctype*  y, inc_t incy
     );
```
Perform
```
  y := y + conjx(x)
```
where `x` and `y` are vectors of length _n_.

---

#### amaxv
```c
void bli_?amaxv
     (
             dim_t   n,
       const ctype*  x, inc_t incx,
             dim_t*  index
     );
```
Given a vector of length _n_, return the zero-based index `index` of the element of vector `x` that contains the largest absolute value (or, in the complex domain, the largest complex modulus).

If `NaN` is encountered, it is treated as if it were a valid value that was smaller than any other value in the vector. If more than one element contains the same maximum value, the index of the latter element is returned via `index`.

**Note:** This function attempts to mimic the algorithm for finding the element with the maximum absolute value in the netlib BLAS routines `i?amax()`.

---

#### axpyv
```c
void bli_?axpyv
     (
             conj_t  conjx,
             dim_t   n,
       const ctype*  alpha,
       const ctype*  x, inc_t incx,
             ctype*  y, inc_t incy
     );
```
Perform
```
  y := y + alpha * conjx(x)
```
where `x` and `y` are vectors of length _n_, and `alpha` is a scalar.

---

#### axpbyv
```c
void bli_?axpbyv
     (
             conj_t  conjx,
             dim_t   n,
       const ctype*  alpha,
       const ctype*  x, inc_t incx,
       const ctype*  beta,
             ctype*  y, inc_t incy
     );
```
Perform
```
  y := beta * y + alpha * conjx(x)
```
where `x` and `y` are vectors of length _n_, and `alpha` and `beta` are scalars.

---

#### copyv
```c
void bli_?copyv
     (
             conj_t  conjx,
             dim_t   n,
       const ctype*  x, inc_t incx,
             ctype*  y, inc_t incy
     );
```
Perform
```
  y := conjx(x)
```
where `x` and `y` are vectors of length _n_.

---

#### dotv
```c
void bli_?dotv
     (
             conj_t  conjx,
             conj_t  conjy,
             dim_t   n,
       const ctype*  x, inc_t incx,
       const ctype*  y, inc_t incy,
             ctype*  rho
     );
```
Perform
```
  rho := conjx(x)^T * conjy(y)
```
where `x` and `y` are vectors of length _n_, and `rho` is a scalar.

---

#### dotxv
```c
void bli_?dotxv
     (
             conj_t  conjx,
             conj_t  conjy,
             dim_t   n,
       const ctype*  alpha,
       const ctype*  x, inc_t incx,
       const ctype*  y, inc_t incy,
       const ctype*  beta,
             ctype*  rho
     );
```
Perform
```
  rho := beta * rho + alpha * conjx(x)^T * conjy(y)
```
where `x` and `y` are vectors of length _n_, and `alpha`, `beta`, and `rho` are scalars.

---

#### invertv
```c
void bli_?invertv
     (
       dim_t   n,
       ctype*  x, inc_t incx
     );
```
Invert all elements of an _n_-length vector `x`.

---

#### invscalv
```c
void bli_?invscalv
     (
             conj_t  conjalpha,
             dim_t   n,
       const ctype*  alpha,
             ctype*  x, inc_t incx
     );
```
Perform
```
  x := ( 1.0 / conjalpha(alpha) ) * x
```
where `x` is a vector of length _n_, and `alpha` is a scalar.

---

#### scalv
```c
void bli_?scalv
     (
             conj_t  conjalpha,
             dim_t   n,
       const ctype*  alpha,
             ctype*  x, inc_t incx
     );
```
Perform
```
  x := conjalpha(alpha) * x
```
where `x` is a vector of length _n_, and `alpha` is a scalar.

---

#### scal2v
```c
void bli_?scal2v
     (
             conj_t  conjx,
             dim_t   n,
       const ctype*  alpha,
       const ctype*  x, inc_t incx,
             ctype*  y, inc_t incy
     );
```
Perform
```
  y := alpha * conjx(x)
```
where `x` and `y` are vectors of length _n_, and `alpha` is a scalar.

---

#### setv
```c
void bli_?setv
     (
             conj_t  conjalpha,
             dim_t   n,
       const ctype*  alpha,
             ctype*  x, inc_t incx
     );
```
Perform
```
  x := conjalpha(alpha)
```
That is, set all elements of an _n_-length vector `x` to scalar `conjalpha(alpha)`.

---

#### subv
```c
void bli_?subv
     (
             conj_t  conjx,
             dim_t   n,
       const ctype*  x, inc_t incx,
             ctype*  y, inc_t incy
     );
```
Perform
```
  y := y - conjx(x)
```
where `x` and `y` are vectors of length _n_.

---

#### swapv
```c
void bli_?swapv
     (
       dim_t   n,
       ctype*  x, inc_t incx,
       ctype*  y, inc_t incy
     );
```
Swap corresponding elements of two _n_-length vectors `x` and `y`.

---

#### xpbyv
```c
void bli_?xpbyv
     (
             conj_t  conjx,
             dim_t   n,
       const ctype*  x, inc_t incx,
       const ctype*  beta,
             ctype*  y, inc_t incy
     );
```
Perform
```
  y := beta * y + conjx(x)
```
where `x` and `y` are vectors of length _n_, and `beta` is a scalar.

---





## Level-1d operations

Level-1d operations perform various level-1 BLAS-like operations on matrix diagonals (hence the _d_).

Most of these operations are similar to level-1m counterparts, except they only read and update matrix diagonals and therefore do not take any `uplo` arguments. Please see the descriptions for the corresponding level-1m operation for a description of the arguments.

---

#### addd
```c
void bli_?addd
     (
             doff_t   diagoffa,
             diag_t   diaga,
             trans_t  transa,
             dim_t    m,
             dim_t    n,
       const ctype*   a, inc_t rsa, inc_t csa,
             ctype*   b, inc_t rsb, inc_t csb
     );
```

---

#### axpyd
```c
void bli_?axpyd
     (
             doff_t   diagoffa,
             diag_t   diaga,
             trans_t  transa,
             dim_t    m,
             dim_t    n,
       const ctype*   alpha,
       const ctype*   a, inc_t rsa, inc_t csa,
             ctype*   b, inc_t rsb, inc_t csb
     );
```

---

#### copyd
```c
void bli_?copyd
     (
             doff_t   diagoffa,
             diag_t   diaga,
             trans_t  transa,
             dim_t    m,
             dim_t    n,
       const ctype*   a, inc_t rsa, inc_t csa,
             ctype*   b, inc_t rsb, inc_t csb
     );
```

---

#### invertd
```c
void bli_?invertd
     (
       doff_t  diagoffa,
       dim_t   m,
       dim_t   n,
       ctype*  a, inc_t rsa, inc_t csa
     );
```

---

#### invscald
```c
void bli_?invscald
     (
             conj_t  conjalpha,
             doff_t  diagoffa,
             dim_t   m,
             dim_t   n,
       const ctype*  alpha,
             ctype*  a, inc_t rsa, inc_t csa
     );
```

---

#### scald
```c
void bli_?scald
     (
             conj_t  conjalpha,
             doff_t  diagoffa,
             dim_t   m,
             dim_t   n,
       const ctype*  alpha,
             ctype*  a, inc_t rsa, inc_t csa
     );
```

---

#### scal2d
```c
void bli_?scal2d
     (
             doff_t   diagoffa,
             diag_t   diaga,
             trans_t  transa,
             dim_t    m,
             dim_t    n,
       const ctype*   alpha,
       const ctype*   a, inc_t rsa, inc_t csa,
             ctype*   b, inc_t rsb, inc_t csb
     );
```

---

#### setd
```c
void bli_?setd
     (
             conj_t  conjalpha,
             doff_t  diagoffa,
             dim_t   m,
             dim_t   n,
       const ctype*  alpha,
             ctype*  a, inc_t rsa, inc_t csa
     );
```

---

#### setid
```c
void bli_?setid
     (
             doff_t    diagoffa,
             dim_t     m,
             dim_t     n,
       const ctype_r*  alpha,
             ctype*    a, inc_t rsa, inc_t csa
     );
```
Set the imaginary components of every element along the diagonal of `a`, as
specified by `diagoffa`, to a scalar `alpha`.
Note that the datatype of `alpha` must be the real projection of the datatype
of `a`.

---

#### shiftd
```c
void bli_?shiftd
     (
             doff_t  diagoffa,
             dim_t   m,
             dim_t   n,
       const ctype*  alpha,
             ctype*  a, inc_t rsa, inc_t csa
     );
```
Add a constant value `alpha` to every element along the diagonal of `a`, as
specified by `diagoffa`.

---

#### subd
```c
void bli_?subd
     (
             doff_t   diagoffa,
             diag_t   diaga,
             trans_t  transa,
             dim_t    m,
             dim_t    n,
       const ctype*   a, inc_t rsa, inc_t csa,
             ctype*   b, inc_t rsb, inc_t csb
     );
```

---

#### xpbyd
```c
void bli_?xpbyd
     (
             doff_t   diagoffa,
             diag_t   diaga,
             trans_t  transa,
             dim_t    m,
             dim_t    n,
       const ctype*   a, inc_t rsa, inc_t csa,
       const ctype*   beta,
             ctype*   b, inc_t rsb, inc_t csb
     );
```

---



## Level-1m operations

Level-1m operations perform various level-1 BLAS-like operations on matrices (hence the _m_).

---

#### addm
```c
void bli_?addm
     (
             doff_t   diagoffa,
             diag_t   diaga,
             uplo_t   uploa,
             trans_t  transa,
             dim_t    m,
             dim_t    n,
       const ctype*   a, inc_t rsa, inc_t csa,
             ctype*   b, inc_t rsb, inc_t csb
     );
```
Perform
```
  B := B + transa(A)
```
where `B` is an _m x n_ matrix, `A` is stored as a dense matrix, or lower- or upper-triangular/trapezoidal matrix, as specified by `uploa`, with the diagonal offset of `A` specified by `diagoffa` and unit/non-unit nature of the diagonal specified by `diaga`. If `uploa` indicates lower or upper storage, only that part of matrix `A` will be referenced and used to update `B`.

---

#### axpym
```c
void bli_?axpym
     (
             doff_t   diagoffa,
             diag_t   diaga,
             uplo_t   uploa,
             trans_t  transa,
             dim_t    m,
             dim_t    n,
       const ctype*   alpha,
       const ctype*   a, inc_t rsa, inc_t csa,
             ctype*   b, inc_t rsb, inc_t csb
     );
```
Perform
```
  B := B + alpha * transa(A)
```
where `B` is an _m x n_ matrix, `A` is stored as a dense matrix, or lower- or upper-triangular/trapezoidal matrix, as specified by `uploa`, with the diagonal offset of `A` specified by `diagoffa` and unit/non-unit nature of the diagonal specified by `diaga`. If `uploa` indicates lower or upper storage, only that part of matrix `A` will be referenced and used to update `B`.

---

#### copym
```c
void bli_?copym
     (
             doff_t   diagoffa,
             diag_t   diaga,
             uplo_t   uploa,
             trans_t  transa,
             dim_t    m,
             dim_t    n,
       const ctype*   a, inc_t rsa, inc_t csa,
             ctype*   b, inc_t rsb, inc_t csb
     );
```
Perform
```
  B := transa(A)
```
where `B` is an _m x n_ matrix, `A` is stored as a dense matrix, or lower- or upper-triangular/trapezoidal matrix, as specified by `uploa`, with the diagonal offset of `A` specified by `diagoffa` and unit/non-unit nature of the diagonal specified by `diaga`. If `uploa` indicates lower or upper storage, only that part of matrix `A` will be referenced and used to update `B`.

---

#### invscalm
```c
void bli_?invscalm
     (
             conj_t  conjalpha,
             doff_t  diagoffa,
             uplo_t  uploa,
             dim_t   m,
             dim_t   n,
       const ctype*  alpha,
             ctype*  a, inc_t rsa, inc_t csa
     );
```
Perform
```
  A := ( 1.0 / conjalpha(alpha) ) * A
```
where `A` is an _m x n_ matrix stored as a dense matrix, or lower- or upper-triangular/trapezoidal matrix, as specified by `uploa`, with the diagonal offset of `A` specified by `diagoffa`. If `uploa` indicates lower or upper storage, only that part of matrix `A` will be updated.

---

#### scalm
```c
void bli_?scalm
     (
             conj_t  conjalpha,
             doff_t  diagoffa,
             uplo_t  uploa,
             dim_t   m,
             dim_t   n,
       const ctype*  alpha,
             ctype*  a, inc_t rsa, inc_t csa
     );
```
Perform
```
  A := conjalpha(alpha) * A
```
where `A` is an _m x n_ matrix stored as a dense matrix, or lower- or upper-triangular/trapezoidal matrix, as specified by `uploa`, with the diagonal offset of `A` specified by `diagoffa`. If `uploa` indicates lower or upper storage, only that part of matrix `A` will be updated.

---

#### scal2m
```c
void bli_?scal2m
     (
             doff_t   diagoffa,
             diag_t   diaga,
             uplo_t   uploa,
             trans_t  transa,
             dim_t    m,
             dim_t    n,
       const ctype*   alpha,
       const ctype*   a, inc_t rsa, inc_t csa,
             ctype*   b, inc_t rsb, inc_t csb
     );
```
Perform
```
  B := alpha * transa(A)
```
where `B` is an _m x n_ matrix, `A` is stored as a dense matrix, or lower- or upper-triangular/trapezoidal matrix, as specified by `uploa`, with the diagonal offset of `A` specified by `diagoffa` and unit/non-unit nature of the diagonal specified by `diaga`. If `uploa` indicates lower or upper storage, only that part of matrix `A` will be referenced and used to update `B`.

---

#### setm
```c
void bli_?setm
     (
             conj_t  conjalpha,
             doff_t  diagoffa,
             diag_t  diaga,
             uplo_t  uploa,
             dim_t   m,
             dim_t   n,
       const ctype*  alpha,
             ctype*  a, inc_t rsa, inc_t csa
     );
```
Set all elements of an _m x n_ matrix `A` to `conjalpha(alpha)`, where `A` is stored as a dense matrix, or lower- or upper- triangular/trapezoidal matrix, as specified by `uploa`, with the diagonal offset of `A` specified by `diagoffa` and unit/non-unit nature of the diagonal specified by `diaga`. If `uploa` indicates lower or upper storage, only that part of matrix `A` will be updated.

---

#### subm
```c
void bli_?subm
     (
             doff_t   diagoffa,
             diag_t   diaga,
             uplo_t   uploa,
             trans_t  transa,
             dim_t    m,
             dim_t    n,
       const ctype*   a, inc_t rsa, inc_t csa,
             ctype*   b, inc_t rsb, inc_t csb
     );
```
Perform
```
  B := B - transa(A)
```
where `B` is an _m x n_ matrix, `A` is stored as a dense matrix, or lower- or upper-triangular/trapezoidal matrix, as specified by `uploa`, with the diagonal offset of `A` specified by `diagoffa` and unit/non-unit nature of the diagonal specified by `diaga`. If `uploa` indicates lower or upper storage, only that part of matrix `A` will be referenced and used to update `B`.

---





## Level-1f operations

Level-1f operations implement various fused combinations of level-1 operations (hence the _f_).
**Note**: Each level-1f operation has a corresponding level-1f kernel through which it is primarily implemented.

Level-1f kernels are employed when optimizing level-2 operations.


---

#### axpy2v
```c
void bli_?axpy2v
     (
             conj_t  conjx,
             conj_t  conjy,
             dim_t   m,
       const ctype*  alphax,
       const ctype*  alphay,
       const ctype*  x, inc_t incx,
       const ctype*  y, inc_t incy,
             ctype*  z, inc_t incz
     );
```
Perform
```
  z := z + alphax * conjx(x) + alphay * conjy(y)
```
where `x`, `y`, and `z` are vectors of length _m_. The kernel, if optimized, is implemented as a fused pair of calls to [axpyv](BLISTypedAPI.md#axpyv).

---

#### dotaxpyv
```c
void bli_?dotaxpyv
     (
             conj_t  conjxt,
             conj_t  conjx,
             conj_t  conjy,
             dim_t   m,
       const ctype*  alpha,
       const ctype*  x, inc_t incx,
       const ctype*  y, inc_t incy,
             ctype*  rho,
             ctype*  z, inc_t incz
     );
```
Perform
```
  rho := conjxt(x^T) * conjy(y)
  z   := z + alpha * conjx(x)
```
where `x`, `y`, and `z` are vectors of length _m_ and `alpha` and `rho` are scalars. The kernel, if optimized, is implemented as a fusion of calls to [dotv](BLISTypedAPI.md#dotv) and [axpyv](BLISTypedAPI.md#axpyv).

---

#### axpyf
```c
void bli_?axpyf
     (
             conj_t  conja,
             conj_t  conjx,
             dim_t   m,
             dim_t   b,
       const ctype*  alpha,
       const ctype*  a, inc_t inca, inc_t lda,
       const ctype*  x, inc_t incx,
             ctype*  y, inc_t incy
     );
```
Perform
```
  y := y + alpha * conja(A) * conjx(x)
```
where `A` is an _m x b_ matrix, and `y` and `x` are vectors. The kernel, if optimized, is implemented as a fused series of calls to [axpyv](BLISTypedAPI.md#axpyv) where _b_ is less than or equal to an implementation-dependent fusing factor specific to `axpyf`.

---

#### dotxf
```c
void bli_?dotxf
     (
             conj_t  conjat,
             conj_t  conjx,
             dim_t   m,
             dim_t   b,
       const ctype*  alpha,
       const ctype*  a, inc_t inca, inc_t lda,
       const ctype*  x, inc_t incx,
       const ctype*  beta,
             ctype*  y, inc_t incy
     );
```
Perform
```
  y := beta * y + alpha * conjat(A^T) * conjx(x)
```
where `A` is an _m x b_ matrix, and `y` and `x` are vectors. The kernel, if optimized, is implemented as a fused series of calls to [dotxv](BLISTypedAPI.md#dotxv) where _b_ is less than or equal to an implementation-dependent fusing factor specific to `dotxf`.

---

#### dotxaxpyf
```c
void bli_?dotxaxpyf
     (
             conj_t  conjat,
             conj_t  conja,
             conj_t  conjw,
             conj_t  conjx,
             dim_t   m,
             dim_t   b,
       const ctype*  alpha,
       const ctype*  a, inc_t inca, inc_t lda,
       const ctype*  w, inc_t incw,
       const ctype*  x, inc_t incx,
       const ctype*  beta,
             ctype*  y, inc_t incy,
             ctype*  z, inc_t incz
     );
```
Perform
```
  y := beta * y + alpha * conjat(A^T) * conjw(w)
  z :=        z + alpha * conja(A)    * conjx(x)
```
where `A` is an _m x b_ matrix, `w` and `z` are vectors of length _m_, `x` and `y` are vectors of length `b`, and `alpha` and `beta` are scalars. The kernel, if optimized, is implemented as a fusion of calls to [dotxf](BLISTypedAPI.md#dotxf) and [axpyf](BLISTypedAPI.md#axpyf).



## Level-2 operations

Level-2 operations perform various level-2 BLAS-like operations.


---

#### gemv
```c
void bli_?gemv
     (
             trans_t  transa,
             conj_t   conjx,
             dim_t    m,
             dim_t    n,
       const ctype*   alpha,
       const ctype*   a, inc_t rsa, inc_t csa,
       const ctype*   x, inc_t incx,
       const ctype*   beta,
             ctype*   y, inc_t incy
     );
```
Perform
```
  y := beta * y + alpha * transa(A) * conjx(x)
```
where `transa(A)` is an _m x n_ matrix, and `y` and `x` are vectors.

---

#### ger
```c
void bli_?ger
     (
             conj_t  conjx,
             conj_t  conjy,
             dim_t   m,
             dim_t   n,
       const ctype*  alpha,
       const ctype*  x, inc_t incx,
       const ctype*  y, inc_t incy,
             ctype*  a, inc_t rsa, inc_t csa
     );
```
Perform
```
  A := A + alpha * conjx(x) * conjy(y)^T
```
where `A` is an _m x n_ matrix, and `x` and `y` are vectors of length _m_ and _n_, respectively.

---

#### hemv
```c
void bli_?hemv
     (
             uplo_t  uploa,
             conj_t  conja,
             conj_t  conjx,
             dim_t   m,
       const ctype*  alpha,
       const ctype*  a, inc_t rsa, inc_t csa,
       const ctype*  x, inc_t incx,
       const ctype*  beta,
             ctype*  y, inc_t incy
     );
```
Perform
```
  y := beta * y + alpha * conja(A) * conjx(x)
```
where `A` is an _m x m_ Hermitian matrix stored in the lower or upper triangle as specified by `uploa`, and `y` and `x` are vectors of length _m_.

---

#### her
```c
void bli_?her
     (
             uplo_t  uploa,
             conj_t  conjx,
             dim_t   m,
       const rtype*  alpha,
       const ctype*  x, inc_t incx,
             ctype*  a, inc_t rsa, inc_t csa
     );
```
Perform
```
  A := A + alpha * conjx(x) * conjx(x)^H
```
where `A` is an _m x m_ Hermitian matrix stored in the lower or upper triangle as specified by `uploa`, and `x` is a vector of length _m_.

**Note:** The floating-point type of `alpha` is always the real projection of the floating-point types of `x` and `A`.

---

#### her2
```c
void bli_?her2
     (
             uplo_t  uploa,
             conj_t  conjx,
             conj_t  conjy,
             dim_t   m,
       const ctype*  alpha,
       const ctype*  x, inc_t incx,
       const ctype*  y, inc_t incy,
             ctype*  a, inc_t rsa, inc_t csa
     );
```
Perform
```
  A := A + alpha * conjx(x) * conjy(y)^H + conj(alpha) * conjy(y) * conjx(x)^H
```
where `A` is an _m x m_ Hermitian matrix stored in the lower or upper triangle as specified by `uploa`, and `x` and `y` are vectors of length _m_.

---

#### symv
```c
void bli_?symv
     (
             uplo_t  uploa,
             conj_t  conja,
             conj_t  conjx,
             dim_t   m,
       const ctype*  alpha,
       const ctype*  a, inc_t rsa, inc_t csa,
       const ctype*  x, inc_t incx,
       const ctype*  beta,
             ctype*  y, inc_t incy
     );
```
Perform
```
  y := beta * y + alpha * conja(A) * conjx(x)
```
where `A` is an _m x m_ symmetric matrix stored in the lower or upper triangle as specified by `uploa`, and `y` and `x` are vectors of length _m_.

---

#### syr
```c
void bli_?syr
     (
             uplo_t  uploa,
             conj_t  conjx,
             dim_t   m,
       const ctype*  alpha,
       const ctype*  x, inc_t incx,
             ctype*  a, inc_t rsa, inc_t csa
     );
```
Perform
```
  A := A + alpha * conjx(x) * conjx(x)^T
```
where `A` is an _m x m_ symmetric matrix stored in the lower or upper triangle as specified by `uploa`, and `x` is a vector of length _m_.

---

#### syr2
```c
void bli_?syr2
     (
             uplo_t  uploa,
             conj_t  conjx,
             conj_t  conjy,
             dim_t   m,
       const ctype*  alpha,
       const ctype*  x, inc_t incx,
       const ctype*  y, inc_t incy,
             ctype*  a, inc_t rsa, inc_t csa
     );
```
Perform
```
  A := A + alpha * conjx(x) * conjy(y)^T + conj(alpha) * conjy(y) * conjx(x)^T
```
where `A` is an _m x m_ symmetric matrix stored in the lower or upper triangle as specified by `uploa`, and `x` and `y` are vectors of length _m_.

---

#### trmv
```c
void bli_?trmv
     (
             uplo_t   uploa,
             trans_t  transa,
             diag_t   diaga,
             dim_t    m,
       const ctype*   alpha,
       const ctype*   a, inc_t rsa, inc_t csa,
             ctype*   x, inc_t incx
     );
```
Perform
```
  x := alpha * transa(A) * x
```
where `A` is an _m x m_ triangular matrix stored in the lower or upper triangle as specified by `uploa` with unit/non-unit nature specified by `diaga`, and `x` is a vector of length _m_.

---

#### trsv
```c
void bli_?trsv
     (
             uplo_t   uploa,
             trans_t  transa,
             diag_t   diaga,
             dim_t    m,
       const ctype*   alpha,
       const ctype*   a, inc_t rsa, inc_t csa,
             ctype*   y, inc_t incy
     );
```
Solve the linear system
```
  transa(A) * x = alpha * y
```
where `A` is an _m x m_ triangular matrix stored in the lower or upper triangle as specified by `uploa` with unit/non-unit nature specified by `diaga`, and `x` and `y` are vectors of length _m_. The right-hand side vector operand `y` is overwritten with the solution vector `x`.

---



## Level-3 operations

Level-3 operations perform various level-3 BLAS-like operations.
**Note**: Each All level-3 operations are implemented through a handful of level-3 microkernels. Please see the [Kernels Guide](KernelsHowTo.md) for more details.


---

#### gemm
```c
void bli_?gemm
     (
             trans_t  transa,
             trans_t  transb,
             dim_t    m,
             dim_t    n,
             dim_t    k,
       const ctype*   alpha,
       const ctype*   a, inc_t rsa, inc_t csa,
       const ctype*   b, inc_t rsb, inc_t csb,
       const ctype*   beta,
             ctype*   c, inc_t rsc, inc_t csc
     );
```
Perform
```
  C := beta * C + alpha * transa(A) * transb(B)
```
where C is an _m x n_ matrix, `transa(A)` is an _m x k_ matrix, and `transb(B)` is a _k x n_ matrix.

---

#### gemmt
```c
void bli_?gemmt
     (
             uplo_t   uploc,
             trans_t  transa,
             trans_t  transb,
             dim_t    m,
             dim_t    k,
       const ctype*   alpha,
       const ctype*   a, inc_t rsa, inc_t csa,
       const ctype*   b, inc_t rsb, inc_t csb,
       const ctype*   beta,
             ctype*   c, inc_t rsc, inc_t csc
     );
```
Perform
```
  C := beta * C + alpha * transa(A) * transb(B)
```
where C is an _m x m_ matrix, `transa(A)` is an _m x k_ matrix, and `transb(B)` is a _k x m_ matrix. This operation is similar to `bli_?gemm()` except that it only updates the lower or upper triangle of `C` as specified by `uploc`.

---

#### hemm
```c
void bli_?hemm
     (
             side_t   sidea,
             uplo_t   uploa,
             conj_t   conja,
             trans_t  transb,
             dim_t    m,
             dim_t    n,
       const ctype*   alpha,
       const ctype*   a, inc_t rsa, inc_t csa,
       const ctype*   b, inc_t rsb, inc_t csb,
       const ctype*   beta,
             ctype*   c, inc_t rsc, inc_t csc
     );
```
Perform
```
  C := beta * C + alpha * conja(A) * transb(B)
```
if `sidea` is `BLIS_LEFT`, or
```
  C := beta * C + alpha * transb(B) * conja(A)
```
if `sidea` is `BLIS_RIGHT`, where `C` and `B` are _m x n_ matrices and `A` is a Hermitian matrix stored in the lower or upper triangle as specified by `uploa`. When `sidea` is `BLIS_LEFT`, `A` is _m x m_, and when `sidea` is `BLIS_RIGHT`, `A` is _n x n_.

---

#### herk
```c
void bli_?herk
     (
             uplo_t   uploc,
             trans_t  transa,
             dim_t    m,
             dim_t    k,
       const rtype*   alpha,
       const ctype*   a, inc_t rsa, inc_t csa,
       const rtype*   beta,
             ctype*   c, inc_t rsc, inc_t csc
     );
```
Perform
```
  C := beta * C + alpha * transa(A) * transa(A)^H
```
where C is an _m x m_ Hermitian matrix stored in the lower or upper triangle as specified by `uploc` and `transa(A)` is an _m x k_ matrix.

**Note:** The floating-point types of `alpha` and `beta` are always the real projection of the floating-point types of `A` and `C`.

---

#### her2k
```c
void bli_?her2k
     (
             uplo_t   uploc,
             trans_t  transa,
             trans_t  transb,
             dim_t    m,
             dim_t    k,
       const ctype*   alpha,
       const ctype*   a, inc_t rsa, inc_t csa,
       const ctype*   b, inc_t rsb, inc_t csb,
       const rtype*   beta,
             ctype*   c, inc_t rsc, inc_t csc
     );
```
Perform
```
  C := beta * C + alpha * transa(A) * transb(B)^H + conj(alpha) * transb(B) * transa(A)^H
```
where C is an _m x m_ Hermitian matrix stored in the lower or upper triangle as specified by `uploc` and `transa(A)` and `transb(B)` are _m x k_ matrices.

**Note:** The floating-point type of `beta` is always the real projection of the floating-point types of `A` and `C`.

---

#### symm
```c
void bli_?symm
     (
             side_t   sidea,
             uplo_t   uploa,
             conj_t   conja,
             trans_t  transb,
             dim_t    m,
             dim_t    n,
       const ctype*   alpha,
       const ctype*   a, inc_t rsa, inc_t csa,
       const ctype*   b, inc_t rsb, inc_t csb,
       const ctype*   beta,
             ctype*   c, inc_t rsc, inc_t csc
     );
```
Perform
```
  C := beta * C + alpha * conja(A) * transb(B)
```
if `sidea` is `BLIS_LEFT`, or
```
  C := beta * C + alpha * transb(B) * conja(A)
```
if `sidea` is `BLIS_RIGHT`, where `C` and `B` are _m x n_ matrices and `A` is a symmetric matrix stored in the lower or upper triangle as specified by `uploa`. When `sidea` is `BLIS_LEFT`, `A` is _m x m_, and when `sidea` is `BLIS_RIGHT`, `A` is _n x n_.

---

#### syrk
```c
void bli_?syrk
     (
             uplo_t   uploc,
             trans_t  transa,
             dim_t    m,
             dim_t    k,
       const ctype*   alpha,
       const ctype*   a, inc_t rsa, inc_t csa,
       const ctype*   beta,
             ctype*   c, inc_t rsc, inc_t csc
     );
```
Perform
```
  C := beta * C + alpha * transa(A) * transa(A)^T
```
where C is an _m x m_ symmetric matrix stored in the lower or upper triangle as specified by `uploa` and `transa(A)` is an _m x k_ matrix.

---

#### syr2k
```c
void bli_?syr2k
     (
             uplo_t   uploc,
             trans_t  transa,
             trans_t  transb,
             dim_t    m,
             dim_t    k,
       const ctype*   alpha,
       const ctype*   a, inc_t rsa, inc_t csa,
       const ctype*   b, inc_t rsb, inc_t csb,
       const ctype*   beta,
             ctype*   c, inc_t rsc, inc_t csc
     );
```
Perform
```
  C := beta * C + alpha * transa(A) * transb(B)^T + alpha * transb(B) * transa(A)^T
```
where C is an _m x m_ symmetric matrix stored in the lower or upper triangle as specified by `uploa` and `transa(A)` and `transb(B)` are _m x k_ matrices.

---

#### trmm
```c
void bli_?trmm
     (
             side_t   sidea,
             uplo_t   uploa,
             trans_t  transa,
             diag_t   diaga,
             dim_t    m,
             dim_t    n,
       const ctype*   alpha,
       const ctype*   a, inc_t rsa, inc_t csa,
             ctype*   b, inc_t rsb, inc_t csb
     );
```
Perform
```
  B := alpha * transa(A) * B
```
if `sidea` is `BLIS_LEFT`, or
```
  B := alpha * B * transa(A)
```
if `sidea` is `BLIS_RIGHT`, where `B` is an _m x n_ matrix and `A` is a triangular matrix stored in the lower or upper triangle as specified by `uploa` with unit/non-unit nature specified by `diaga`. When `sidea` is `BLIS_LEFT`, `A` is _m x m_, and when `sidea` is `BLIS_RIGHT`, `A` is _n x n_.

---

#### trmm3
```c
void bli_?trmm3
     (
             side_t   sidea,
             uplo_t   uploa,
             trans_t  transa,
             diag_t   diaga,
             trans_t  transb,
             dim_t    m,
             dim_t    n,
       const ctype*   alpha,
       const ctype*   a, inc_t rsa, inc_t csa,
       const ctype*   b, inc_t rsb, inc_t csb,
       const ctype*   beta,
             ctype*   c, inc_t rsc, inc_t csc
     );
```
Perform
```
  C := beta * C + alpha * transa(A) * transb(B)
```
if `sidea` is `BLIS_LEFT`, or
```
  C := beta * C + alpha * transb(B) * transa(A)
```
if `sidea` is `BLIS_RIGHT`, where `C` and `transb(B)` are _m x n_ matrices and `A` is a triangular matrix stored in the lower or upper triangle as specified by `uploa` with unit/non-unit nature specified by `diaga`. When `sidea` is `BLIS_LEFT`, `A` is _m x m_, and when `sidea` is `BLIS_RIGHT`, `A` is _n x n_.

---

#### trsm
```c
void bli_?trsm
     (
             side_t   sidea,
             uplo_t   uploa,
             trans_t  transa,
             diag_t   diaga,
             dim_t    m,
             dim_t    n,
       const ctype*   alpha,
       const ctype*   a, inc_t rsa, inc_t csa,
             ctype*   b, inc_t rsb, inc_t csb
     );
```
Solve the linear system with multiple right-hand sides
```
  transa(A) * X = alpha * B
```
if `sidea` is `BLIS_LEFT`, or
```
  X * transa(A) = alpha * B
```
if `sidea` is `BLIS_RIGHT`, where `X` and `B` are an _m x n_ matrices and `A` is a triangular matrix stored in the lower or upper triangle as specified by `uploa` with unit/non-unit nature specified by `diaga`. When `sidea` is `BLIS_LEFT`, `A` is _m x m_, and when `sidea` is `BLIS_RIGHT`, `A` is _n x n_. The right-hand side matrix operand `B` is overwritten with the solution matrix `X`.

---


## Utility operations

---

#### asumv
```c
void bli_?asumv
     (
             dim_t   n,
       const ctype*  x, inc_t incx,
             rtype*  asum
     );
```
Compute the sum of the absolute values of the fundamental elements of vector `x`. The resulting sum is stored to `asum`.

**Note:** The floating-point type of `asum` is always the real projection of the floating-point type of `x`.
**Note:** This function attempts to mimic the algorithm for computing the absolute vector sum in the netlib BLAS routines `*asum()`.

---

#### norm1m
#### normfm
#### normim
```c
void bli_?norm[1fi]m
     (
             doff_t  diagoffa,
             doff_t  diaga,
             uplo_t  uploa,
             dim_t   m,
             dim_t   n,
       const ctype*  a, inc_t rs_a, inc_t cs_a,
             rtype*  norm
     );
```
Compute the one-norm (`bli_?norm1m()`), Frobenius norm (`bli_?normfm()`), or infinity norm (`bli_?normim()`) of the elements in an _m x n_ matrix `A`. If `uploa` is `BLIS_LOWER` or `BLIS_UPPER` then `A` is assumed to be lower or upper triangular, respectively, with the main diagonal located at offset `diagoffa`. The resulting norm is stored to `norm`.

**Note:** The floating-point type of `norm` is always the real projection of the floating-point type of `x`.

---

#### norm1v
#### normfv
#### normiv
```c
void bli_?norm[1fi]v
     (
             dim_t   n,
       const ctype*  x, inc_t incx,
             rtype*  norm
     );
```
Compute the one-norm (`bli_?norm1v()`), Frobenius norm (`bli_?normfv()`), or infinity norm (`bli_?normiv()`) of the elements in a vector `x` of length _n_. The resulting norm is stored to `norm`.

**Note:** The floating-point type of `norm` is always the real projection of the floating-point type of `x`.

---

#### mkherm
```c
void bli_?mkherm
     (
       uplo_t  uploa,
       dim_t   m,
       ctype*  a, inc_t rs_a, inc_t cs_a
     );
```
Make an _m x m_ matrix `A` explicitly Hermitian by copying the conjugate of the triangle specified by `uploa` to the opposite triangle. Imaginary components of diagonal elements are explicitly set to zero. It is assumed that the diagonal offset of `A` is zero.

---

#### mksymm
```c
void bli_?mksymm
     (
       uplo_t  uploa,
       dim_t   m,
       ctype*  a, inc_t rs_a, inc_t cs_a
     );
```
Make an _m x m_ matrix `A` explicitly symmetric by copying the triangle specified by `uploa` to the opposite triangle. It is assumed that the diagonal offset of `A` is zero.

---

#### mktrim
```c
void bli_?mktrim
     (
       uplo_t  uploa,
       dim_t   m,
       ctype*  a, inc_t rs_a, inc_t cs_a
     );
```
Make an _m x m_ matrix `A` explicitly triangular by preserving the triangle specified by `uploa` and zeroing the elements in the opposite triangle. It is assumed that the diagonal offset of `A` is zero.

---

#### fprintv
```c
void bli_?fprintv
     (
             FILE*   file,
       const char*   s1,
             dim_t   m,
       const ctype*  x, inc_t incx,
       const char*   format,
       const char*   s2
     );
```
Print a vector `x` of length _m_ to file stream `file`, where `file` is a file pointer returned by the standard C library function `fopen()`. The caller may also pass in a global file pointer such as `stdout` or `stderr`. The strings `s1` and `s2` are printed immediately before and after the output (respectively), and the format specifier `format` is used to format the individual elements. For valid format specifiers, please see documentation for the standard C library function `printf()`.

**Note:** For complex datatypes, the format specifier is applied to both the real and imaginary components **individually**. Therefore, you should use format specifiers such as `"%5.2f"`, but **not** `"%5.2f + %5.2f"`.

---

#### fprintm
```c
void bli_?fprintm
     (
             FILE*   file,
       const char*   s1,
             dim_t   m,
             dim_t   n,
       const ctype*  a, inc_t rs_a, inc_t cs_a,
       const char*   format,
       const char*   s2
     );
```
Print an _m x n_ matrix `A` to file stream `file`, where `file` is a file pointer returned by the standard C library function `fopen()`. The caller may also pass in a global file pointer such as `stdout` or `stderr`. The strings `s1` and `s2` are printed immediately before and after the output (respectively), and the format specifier `format` is used to format the individual elements. For valid format specifiers, please see documentation for the standard C library function `printf()`.

**Note:** For complex datatypes, the format specifier is applied to both the real and imaginary components **individually**. Therefore, you should use format specifiers such as `"%5.2f"`, but **not** `"%5.2f + %5.2f"`.

---

#### printv
```c
void bli_?printv
     (
       const char*   s1,
             dim_t   m,
       const ctype*  x, inc_t incx,
       const char*   format,
       const char*   s2
     );
```
Print a vector `x` of length _m_ to standard output. This function call is equivalent to calling `bli_?fprintv()` with `stdout` as the file pointer.

---

#### printm
```c
void bli_?printm
     (
       const char*   s1,
             dim_t   m,
             dim_t   n,
       const ctype*  a, inc_t rs_a, inc_t cs_a,
       const char*   format,
       const char*   s2
     );
```
Print an _m x n_ matrix `a` to standard output. This function call is equivalent to calling `bli_?fprintm()` with `stdout` as the file pointer.

---

#### randv
```c
void bli_?randv
     (
       dim_t   n,
       ctype*  x, inc_t incx
     );
```
Set the elements of a vector `x` of length _n_ to random values on the interval `[-1,1)`.

**Note:** For complex datatypes, the real and imaginary components of each element are randomized individually and independently of one another.

---

#### randm
```c
void bli_?randm
     (
       doff_t  diagoffa,
       uplo_t  uploa,
       dim_t   m,
       dim_t   n,
       ctype*  a, inc_t rs_a, inc_t cs_a
     );
```
Set the elements of an _m x n_ matrix `A` to random values on the interval `[-1,1)`. If `uploa` is `BLIS_LOWER` or `BLIS_UPPER`, then additional scaling occurs so that the resulting matrix is diagonally dominant. Specifically, the diagonal elements (identified by diagonal offset `diagoffa`) are shifted so that they lie on the interval `[1,2)` and the off-diagonal elements (in the triangle specified by `uploa`) are scaled by `1.0/max(m,n)`.

**Note:** For complex datatypes, the real and imaginary components of each off-diagonal element are randomized individually and independently of one another.

---

#### sumsqv
```c
void bli_?sumsqv
     (
             dim_t   n,
       const ctype*  x, inc_t incx,
             rtype*  scale,
             rtype*  sumsq
     );
```
Compute the sum of the squares of the elements in a vector `x` of length _n_. The result is computed in scaled form, and in such a way that it may be used repeatedly to accumulate the sum of the squares of several vectors.

The function computes scale\_new and sumsq\_new such that
```
  scale_new^2 * sumsq_new = x[0]^2 + x[1]^2 + ... x[m-1]^2 + scale_old^2 * sumsq_old
```
where, on entry, `scale` and `sumsq` contain `scale_old` and `sumsq_old`, respectively, and on exit, `scale` and `sumsq` contain `scale_new` and `sumsq_new`, respectively.

**Note:** This function attempts to mimic the algorithm for computing the Frobenius norm in the netlib LAPACK routine `?lassq()`.

---

#### getsc
```c
void bli_?getsc
     (
       const ctype*   chi,
             double*  zeta_r,
             double*  zeta_i
     );
```
Copy the real and imaginary values from the scalar object `chi` to `zeta_r` and `zeta_i`. If `chi` is stored as a real type, then `zeta_i` is set to zero. (If `chi` is stored in single precision, the corresponding elements are typecast/promoted during the copy.)

---

#### getijv
```c
err_t bli_?getijv
     (
             dim_t    i,
       const void*    x, inc_t incx,
             double*  ar,
             double*  ai
     );
```
Copy the real and imaginary values at the `i`th element of vector `x` to `ar` and `ai`. For real domain invocations, only `ar` is overwritten and `ai` is left unchanged. (If `x` contains elements stored in single precision, the corresponding elements are typecast/promoted during the copy.)
Note that the object-based analogue of [getijv](BLISObjectAPI.md#getijv) does bounds checking of the vector element offset `i` against the vector length while the typed functions specified above do not (since the vector length is not given).

---

#### getijm
```c
err_t bli_?getijm
     (
             dim_t    i,
             dim_t    j,
       const void*    b, inc_t rs_b, inc_t cs_b,
             double*  ar,
             double*  ai
     );
```
Copy the real and imaginary values at the (`i`,`j`) element of object `b` to `ar` and `ai`. For real domain invocations, only `ar` is overwritten and `ai` is left unchanged. (If `b` contains elements stored in single precision, the corresponding elements are typecast/promoted during the copy.)
Note that the object-based analogue of [getijm](BLISObjectAPI.md#getijm) does bounds checking of the matrix element offsets (`i`,`j`) against the matrix dimensions while the typed functions specified above do not (since the matrix dimensions are not given).

---

#### setsc
```c
void bli_?setsc
     (
       double  zeta_r,
       double  zeta_i,
       ctype*  chi
     );
```
Copy real and imaginary values `zeta_r` and `zeta_i` to the scalar object `chi`. If `chi` is stored as a real type, then `zeta_i` is ignored. (If `chi` is stored in single precision, the contents are typecast/demoted during the copy.)

---

#### setijv
```c
err_t bli_?setijv
     (
       double  ar,
       double  ai,
       dim_t   i,
       void*   x, inc_t incx
     );
```
Copy real and imaginary values `ar` and `ai` to the `i`th element of vector object `x`. For real domain invocations, only `ar` is copied and `ai` is ignored. (If `x` contains elements stored in single precision, the corresponding elements are typecast/demoted during the copy.)
Note that the object-based analogue of [setijv](BLISObjectAPI.md#setijv) does bounds checking of the vector element offset `i` against the vector length while the typed functions specified above do not (since the vector length is not given).

---

#### setijm
```c
err_t bli_?setijm
     (
       double  ar,
       double  ai,
       dim_t   i,
       dim_t   j,
       void*   b, inc_t rs_b, inc_t cs_b
     );
```
Copy real and imaginary values `ar` and `ai` to the (`i`,`j`) element of object `b`. For real domain invocations, only `ar` is copied and `ai` is ignored. (If `b` contains elements stored in single precision, the corresponding elements are typecast/demoted during the copy.)
Note that the object-based analogue of [setijm](BLISObjectAPI.md#setijm) does bounds checking of the matrix element offsets (`i`,`j`) against the matrix dimensions while the typed functions specified above do not (since the matrix dimensions are not given).

---

#### eqsc
```c
void bli_?eqsc
     (
             conj_t  conjchi,
       const ctype*  chi,
       const ctype*  psi,
             bool*   is_eq
     );
```
Perform an element-wise comparison between scalars `chi` and `psi` and store the boolean result in the `bool` pointed to by `is_eq`.
If `conjchi` indicates a conjugation, `chi` will be implicitly conjugated for purposes of the comparision.

---

#### eqv
```c
void bli_?eqv
     (
             conj_t  conjx,
             dim_t   n,
       const ctype*  x, inc_t incx,
       const ctype*  y, inc_t incy,
             bool*   is_eq
     );
```
Perform an element-wise comparison between length _n_ vectors `x` and `y` and store the boolean result in the `bool` pointed to by `is_eq`.
If `conjx` indicates a conjugation, `x` will be implicitly conjugated for purposes of the comparision.

---

#### eqm
```c
void bli_?eqm
     (
             doff_t   diagoffa,
             diag_t   diaga,
             uplo_t   uploa,
             trans_t  transa,
             dim_t    m,
             dim_t    n,
       const ctype*   a, inc_t rs_a, inc_t cs_a,
       const ctype*   b, inc_t rs_b, inc_t cs_b,
             bool*    is_eq
     );
```
Perform an element-wise comparison between matrices `A` and `B` and store the boolean result in the `bool` pointed to by `is_eq`.
Here, `B` is an _m x n_ matrix, `A` is stored as a dense matrix, or lower- or upper-triangular/trapezoidal matrix with arbitrary diagonal offset and unit or non-unit diagonal.
If `diaga` indicates a unit diagonal, the diagonals of both matrices will be ignored for purposes of the comparision.
If `uploa` indicates lower or upper storage, only that part of matrix `A` will be referenced in the comparison.
If `transa` indicates a conjugation and/or transposition, then `A` will be conjugated and/or transposed for purposes of the comparison.





## Level-3 microkernels

**Note:** The `*` in level-3 microkernel function names shown below reflect that there is no exact naming convention required for the microkernels, except that they must begin with `bli_?`. We strongly recommend, however, that the microkernel function names include the name of the microkernel itself. For example, the `gemm` microkernel should be named with the prefix `bli_?gemm_` and the `trsm` microkernels should be named with the prefixes `bli_?trsm_l_` (lower triangular) and `bli_?trsm_u_` (upper triangular).

---

#### gemm microkernel
```c
void bli_?gemm_*
     (
             dim_t       m,
             dim_t       n,
             dim_t       k,
       const ctype*      alpha,
       const ctype*      a1,
       const ctype*      b1,
       const ctype*      beta,
             ctype*      c11, inc_t rsc, inc_t csc,
       const auxinfo_t*  data,
       const cntx_t*     cntx
     );
```
Perform
```
  C11 := beta * C11 + alpha * A1 * B1
```
where `C11` is an _MR x NR_ matrix, `A1` is an _MR x k_ "micropanel" matrix stored in packed (column-stored) format, `B1` is a _k x NR_ "micropanel" matrix in packed (row-stored) format, and alpha and beta are scalars. The storage of `C11` is specified by its row and column strides, `rsc` and `csc`.

Please see the [Kernel Guide](KernelsHowTo.md) for more information on the `gemm` microkernel.


---

#### trsm microkernels
```c
void bli_?trsm_l_*
     (
       const ctype*      a11,
             ctype*      b11,
             ctype*      c11, inc_t rsc, inc_t csc
       const auxinfo_t*  data,
       const cntx_t*     cntx
     );

void bli_?trsm_u_*
     (
       const ctype*      a11,
             ctype*      b11,
             ctype*      c11, inc_t rsc, inc_t csc
       const auxinfo_t*  data,
       const cntx_t*     cntx
     );
```
Perform
```
  B11 := inv(A11) * B11
  C11 := B11
```
where `A11` is an _MR x MR_ lower or upper triangular matrix stored in packed (column-stored) format, `B11` is an _MR x NR_ matrix stored in packed (row-stored) format, and `C11` is an _MR x NR_ matrix stored according to row and column strides `rsc` and `csc`.

Please see the [Kernel Guide](KernelsHowTo.md) for more information on the `trsm` microkernel.

---

#### gemmtrsm microkernels
```c
void bli_?gemmtrsm_l_*
     (
             dim_t       m,
             dim_t       n,
             dim_t       k,
       const ctype*      alpha,
       const ctype*      a10,
       const ctype*      a11,
       const ctype*      b01,
             ctype*      b11,
             ctype*      c11, inc_t rs_c, inc_t cs_c,
       const auxinfo_t*  data,
       const cntx_t*     cntx
     );

void bli_?gemmtrsm_u_*
     (
             dim_t       m,
             dim_t       n,
             dim_t       k,
       const ctype*      alpha,
       const ctype*      a12,
       const ctype*      a11,
       const ctype*      b21,
             ctype*      b11,
             ctype*      c11, inc_t rs_c, inc_t cs_c,
       const auxinfo_t*  data,
       const cntx_t*     cntx
     );
```
Perform
```
  B11 := alpha * B11 - A10 * B01
  B11 := inv(A11) * B11
  C11 := B11
```
if `A11` is lower triangular, or
```
  B11 := alpha * B11 - A12 * B21
  B11 := inv(A11) * B11
  C11 := B11
```
if `A11` is upper triangular.

Please see the [Kernel Guide](KernelsHowTo.md) for more information on the `gemmtrsm` microkernel.






# Query function reference

BLIS allows applications to query information about how BLIS was configured. The `bli_info_` API provides several categories of query routines. Most values are returned as a `gint_t`, which is a signed integer. The size of this integer can be queried through a special routine that returns the size in a character string:
```c
const char* bli_info_get_int_type_size_str( void );
```
**Note:** All of the `bli_info_` functions are **always** thread-safe, no matter how BLIS was configured.

## General library information

The following routine returns the address the full BLIS version string:
```c
const char* bli_info_get_version_str( void );
```

## Specific configuration

The following routine returns a unique ID of type `arch_t` that identifies the current current active configuration:
```c
arch_t bli_arch_query_id( void );
```
This is most useful when BLIS is configured with multiple configurations. (When linking to multi-configuration builds of BLIS, you don't know for sure which configuration will be used until runtime since the configuration-specific parameters are not loaded until after calling a hueristic to detect the hardware--usually based the `CPUID` instruction.)

Once the configuration's ID is known, it can be used to query a string that contains the name of the configuration:
```c
const char* bli_arch_string( arch_t id );
```

## General configuration

The following routines return various general-purpose constants that affect the entire framework. All of these settings default to sane values, which can then be overridden by the configuration in [bli\_config.h](ConfigurationHowTo#bli_configh). If they are absent from a particular configuration's `bli_config.h` header file, then the default value is used, as specified in [frame/include/bli_config_macro_defs.h](https://github.com/flame/blis/blob/master/frame/include/bli_config_macro_defs.h).

```c
gint_t bli_info_get_int_type_size( void );
gint_t bli_info_get_num_fp_types( void );
gint_t bli_info_get_max_type_size( void );
gint_t bli_info_get_page_size( void );
gint_t bli_info_get_simd_num_registers( void );
gint_t bli_info_get_simd_size( void );
gint_t bli_info_get_simd_align_size( void );
gint_t bli_info_get_stack_buf_max_size( void );
gint_t bli_info_get_stack_buf_align_size( void );
gint_t bli_info_get_heap_addr_align_size( void );
gint_t bli_info_get_heap_stride_align_size( void );
gint_t bli_info_get_pool_addr_align_size_a( void );
gint_t bli_info_get_pool_addr_align_size_b( void );
gint_t bli_info_get_pool_addr_align_size_c( void );
gint_t bli_info_get_pool_addr_align_size_gen( void );
gint_t bli_info_get_pool_addr_offset_size_a( void );
gint_t bli_info_get_pool_addr_offset_size_b( void );
gint_t bli_info_get_pool_addr_offset_size_c( void );
gint_t bli_info_get_pool_addr_offset_size_gen( void );
gint_t bli_info_get_enable_stay_auto_init( void );
gint_t bli_info_get_enable_blas( void );
gint_t bli_info_get_enable_cblas( void );
gint_t bli_info_get_blas_int_type_size( void );
gint_t bli_info_get_enable_pba_pools( void );
gint_t bli_info_get_enable_sba_pools( void );
gint_t bli_info_get_enable_threading( void );
gint_t bli_info_get_enable_openmp( void );
gint_t bli_info_get_enable_pthreads( void );
gint_t bli_info_get_enable_hpx( void );
gint_t bli_info_get_enable_openmp_as_default( void );
gint_t bli_info_get_enable_pthreads_as_default( void );
gint_t bli_info_get_enable_hpx_as_default( void );
gint_t bli_info_get_thread_jrir_slab( void );
gint_t bli_info_get_thread_jrir_rr( void );
gint_t bli_info_get_thread_jrir_tlb( void );
gint_t bli_info_get_enable_tls( void );
gint_t bli_info_get_enable_memkind( void );
gint_t bli_info_get_enable_sandbox( void );
```

## Kernel information

### Micro-kernel implementation type query

The following routines allow the caller to obtain a string that identifies the implementation type of each microkernel that is currently active (ie: part of the current active configuration, as identified bi `bli_arch_query_id()`).

```c
const char* bli_info_get_gemm_ukr_impl_string( ind_t method, num_t dt );
const char* bli_info_get_gemmtrsm_l_ukr_impl_string( ind_t method, num_t dt );
const char* bli_info_get_gemmtrsm_u_ukr_impl_string( ind_t method, num_t dt );
const char* bli_info_get_trsm_l_ukr_impl_string( ind_t method, num_t dt );
const char* bli_info_get_trsm_u_ukr_impl_string( ind_t method, num_t dt );
```

Possible implementation (ie: the `ind_t method` argument) types are:
 * `BLIS_1M`: Implementation based on the 1m method. (This is the default induced method when real domain kernels are present but complex kernels are missing.)
 * `BLIS_NAT`: Implementation based on "native" execution (ie: NOT an induced method).

Possible microkernel types (ie: the return values for `bli_info_get_*_ukr_impl_string()`) are:
 * `BLIS_REFERENCE_UKERNEL` (`"refrnce"`): This value is returned when the queried microkernel is provided by the reference implementation.
 * `BLIS_VIRTUAL_UKERNEL` (`"virtual"`): This value is returned when the queried microkernel is driven by a the "virtual" microkernel provided by an induced method. This happens for any `method` value that is not `BLIS_NAT` (ie: native), but only applies to the complex domain.
 * `BLIS_OPTIMIZED_UKERNEL` (`"optimzd"`): This value is returned when the queried microkernel is provided by an implementation that is neither reference nor virtual, and thus we assume the kernel author would deem it to be "optimized". Such a microkernel may not be optimal in the literal sense of the word, but nonetheless is _intended_ to be optimized, at least relative to the reference microkernels.
 * `BLIS_NOTAPPLIC_UKERNEL` (`"notappl"`): This value is returned usually when performing a `gemmtrsm` or `trsm` microkernel type query for any `method` value that is not `BLIS_NAT` (ie: native). That is, induced methods cannot be (purely) used on `trsm`-based microkernels because these microkernels perform more a triangular inversion, which is not matrix multiplication.


### Operation implementation type query

The following routines allow the caller to obtain a string that identifies the implementation (`ind_t`) that is currently active (ie: implemented and enabled) for each level-3 operation. Possible implementation types are listed in the section above covering [microkernel implemenation query](BLISTypedAPI.md#microkernel-implementation-type-query).
```c
const char* bli_info_get_gemm_impl_string( num_t dt );
const char* bli_info_get_gemmt_impl_string( num_t dt );
const char* bli_info_get_hemm_impl_string( num_t dt );
const char* bli_info_get_herk_impl_string( num_t dt );
const char* bli_info_get_her2k_impl_string( num_t dt );
const char* bli_info_get_symm_impl_string( num_t dt );
const char* bli_info_get_syrk_impl_string( num_t dt );
const char* bli_info_get_syr2k_impl_string( num_t dt );
const char* bli_info_get_trmm_impl_string( num_t dt );
const char* bli_info_get_trmm3_impl_string( num_t dt );
const char* bli_info_get_trsm_impl_string( num_t dt );
```


## Clock functions

---

#### clock
```c
double bli_clock( void );
```
Return the amount of time that has elapsed since some fixed time in the past. The return values of `bli_clock()` typically feature nanosecond precision, though this is not guaranteed.

**Note:** On Linux, `bli_clock()` is implemented in terms of `clock_gettime()` using the `clockid_t` value of `CLOCK_MONOTONIC`. On OS X, `bli_clock` is implemented in terms of `mach_absolute_time()`. And on Windows, `bli_clock` is implemented in terms of `QueryPerformanceFrequency()`. Please see [frame/base/bli_clock.c](https://github.com/flame/blis/blob/master/frame/base/bli_clock.c) for more details.
**Note:** This function is returns meaningless values when BLIS is configured with `--disable-system`.

---

#### clock_min_diff
```c
double bli_clock_min_diff
     (
       double  time_prev_min,
       double  time_start
     );
```
This function computes an intermediate value, `time_diff`, equal to `bli_clock() - time_start`, and then tentatively prepares to return the minimum value of `time_diff` and `time_min`. If that minimum value is extremely small (close to zero), the function returns `time_min` instead.

This function is meant to be used in conjuction with `bli_clock()` for
performance timing within applications--specifically in loops where only
the fastest timing is of interest. For example:
```c
double t_save = DBL_MAX;
for( i = 0; i < 3; ++i )
{
   double t = bli_clock();
   bli_gemm( ... );
   t_save = bli_clock_min_diff( t_save, t );
}
double gflops = ( 2.0 * m * k * n ) / ( t_save * 1.0e9 );
```
This code calls `bli_gemm()` three times and computes the performance, in GFLOPS, of the fastest of the three executions.

---



# Example code

BLIS provides lots of example code in the [examples/tapi](https://github.com/flame/blis/tree/master/examples/tapi) directory of the BLIS source distribution. The example code in this directory is set up like a tutorial, and so we recommend starting from the beginning. Topics include printing vectors and matrices and calling a representative subset of the computational level-1v, -1m, -2, -3, and utility operations documented above. Please read the `README` contained within the `examples/tapi` directory for further details.

