# Contents

* **[Contents](BLISObjectAPI.md#contents)**
* **[Operation index](BLISObjectAPI.md#operation-index)**
* **[Introduction](BLISObjectAPI.md#introduction)**
  * [BLIS types](BLISObjectAPI.md#blis-types)
    * [Integer-based types](BLISObjectAPI.md#integer-based-types)
    * [Floating-point types](BLISObjectAPI.md#floating-point-types)
    * [Enumerated parameter types](BLISObjectAPI.md#enumerated-parameter-types)
  * [Global scalar constants](BLISObjectAPI.md#global-scalar-constants)
  * [Basic vs expert interfaces](BLISObjectAPI.md#basic-vs-expert-interfaces)
  * [Context type](BLISObjectAPI.md#context-type)
  * [Runtime type](BLISObjectAPI.md#runtime-type)
  * [BLIS header file](BLISObjectAPI.md#blis-header-file)
  * [Initialization and cleanup](BLISObjectAPI.md#initialization-and-cleanup)
* **[Object management](BLISObjectAPI.md#object-management)**
  * [Object creation function reference](BLISObjectAPI.md#object-creation-function-reference)
  * [Object accessor function reference](BLISObjectAPI.md#object-accessor-function-reference)
  * [Object mutator function reference](BLISObjectAPI.md#object-mutator-function-reference)
  * [Other object function reference](BLISObjectAPI.md#other-object-function-reference)
* **[Computational function reference](BLISObjectAPI.md#computational-function-reference)**
  * [Level-1v operations](BLISObjectAPI.md#level-1v-operations)
  * [Level-1d operations](BLISObjectAPI.md#level-1d-operations)
  * [Level-1m operations](BLISObjectAPI.md#level-1m-operations)
  * [Level-1f operations](BLISObjectAPI.md#level-1f-operations)
  * [Level-2 operations](BLISObjectAPI.md#level-2-operations)
  * [Level-3 operations](BLISObjectAPI.md#level-3-operations)
  * [Utility operations](BLISObjectAPI.md#utility-operations)
* **[Query function reference](BLISObjectAPI.md#query-function-reference)**
  * [General library information](BLISObjectAPI.md#general-library-information)
  * [Specific configuration](BLISObjectAPI.md#specific-configuration)
  * [General configuration](BLISObjectAPI.md#general-configuration)
  * [Kernel information](BLISObjectAPI.md#kernel-information)
  * [Clock functions](BLISObjectAPI.md#clock-functions)
* **[Example code](BLISObjectAPI.md#example-code)**



# Operation index

This index provides a quick way to jump directly to the description for each operation discussed later in the [Computational function reference](BLISObjectAPI.md#computational-function-reference) section:

  * **[Level-1v](BLISObjectAPI.md#level-1v-operations)**: Operations on vectors:
    * [addv](BLISObjectAPI.md#addv), [amaxv](BLISObjectAPI.md#amaxv), [axpyv](BLISObjectAPI.md#axpyv), [axpbyv](BLISObjectAPI.md#axpbyv), [copyv](BLISObjectAPI.md#copyv), [dotv](BLISObjectAPI.md#dotv), [dotxv](BLISObjectAPI.md#dotxv), [invertv](BLISObjectAPI.md#invertv), [invscalv](BLISObjectAPI.md#invscalv), [scalv](BLISObjectAPI.md#scalv), [scal2v](BLISObjectAPI.md#scal2v), [setv](BLISObjectAPI.md#setv), [setrv](BLISObjectAPI.md#setrv), [setiv](BLISObjectAPI.md#setiv), [subv](BLISObjectAPI.md#subv), [swapv](BLISObjectAPI.md#swapv), [xpbyv](BLISObjectAPI.md#xpbyv)
  * **[Level-1d](BLISObjectAPI.md#level-1d-operations)**: Element-wise operations on matrix diagonals:
    * [addd](BLISObjectAPI.md#addd), [axpyd](BLISObjectAPI.md#axpyd), [copyd](BLISObjectAPI.md#copyd), [invertd](BLISObjectAPI.md#invertd), [invscald](BLISObjectAPI.md#invscald), [scald](BLISObjectAPI.md#scald), [scal2d](BLISObjectAPI.md#scal2d), [setd](BLISObjectAPI.md#setd), [setid](BLISObjectAPI.md#setid), [shiftd](BLISObjectAPI.md#shiftd), [subd](BLISObjectAPI.md#subd), [xpbyd](BLISObjectAPI.md#xpbyd)
  * **[Level-1m](BLISObjectAPI.md#level-1m-operations)**: Element-wise operations on matrices:
    * [addm](BLISObjectAPI.md#addm), [axpym](BLISObjectAPI.md#axpym), [copym](BLISObjectAPI.md#copym), [invscalm](BLISObjectAPI.md#invscalm), [scalm](BLISObjectAPI.md#scalm), [scal2m](BLISObjectAPI.md#scal2m), [setm](BLISObjectAPI.md#setm), [setrm](BLISObjectAPI.md#setrm), [setim](BLISObjectAPI.md#setim), [subm](BLISObjectAPI.md#subm)
  * **[Level-1f](BLISObjectAPI.md#level-1f-operations)**: Fused operations on multiple vectors:
    * [axpy2v](BLISObjectAPI.md#axpy2v), [dotaxpyv](BLISObjectAPI.md#dotaxpyv), [axpyf](BLISObjectAPI.md#axpyf), [dotxf](BLISObjectAPI.md#dotxf), [dotxaxpyf](BLISObjectAPI.md#dotxaxpyf)
  * **[Level-2](BLISObjectAPI.md#level-2-operations)**: Operations with one matrix and (at least) one vector operand:
    * [gemv](BLISObjectAPI.md#gemv), [ger](BLISObjectAPI.md#ger), [hemv](BLISObjectAPI.md#hemv), [her](BLISObjectAPI.md#her), [her2](BLISObjectAPI.md#her2), [symv](BLISObjectAPI.md#symv), [syr](BLISObjectAPI.md#syr), [syr2](BLISObjectAPI.md#syr2), [trmv](BLISObjectAPI.md#trmv), [trsv](BLISObjectAPI.md#trsv)
  * **[Level-3](BLISObjectAPI.md#level-3-operations)**: Operations with matrices that are multiplication-like:
    * [gemm](BLISObjectAPI.md#gemm), [hemm](BLISObjectAPI.md#hemm), [herk](BLISObjectAPI.md#herk), [her2k](BLISObjectAPI.md#her2k), [symm](BLISObjectAPI.md#symm), [syrk](BLISObjectAPI.md#syrk), [syr2k](BLISObjectAPI.md#syr2k), [trmm](BLISObjectAPI.md#trmm), [trmm3](BLISObjectAPI.md#trmm3), [trsm](BLISObjectAPI.md#trsm)
  * **[Utility](BLISObjectAPI.md#Utility-operations)**: Miscellaneous operations on matrices and vectors:
    * [asumv](BLISObjectAPI.md#asumv), [norm1v](BLISObjectAPI.md#norm1v), [normfv](BLISObjectAPI.md#normfv), [normiv](BLISObjectAPI.md#normiv), [norm1m](BLISObjectAPI.md#norm1m), [normfm](BLISObjectAPI.md#normfm), [normim](BLISObjectAPI.md#normim), [mkherm](BLISObjectAPI.md#mkherm), [mksymm](BLISObjectAPI.md#mksymm), [mktrim](BLISObjectAPI.md#mktrim), [fprintv](BLISObjectAPI.md#fprintv), [fprintm](BLISObjectAPI.md#fprintm),[printv](BLISObjectAPI.md#printv), [printm](BLISObjectAPI.md#printm), [randv](BLISObjectAPI.md#randv), [randm](BLISObjectAPI.md#randm), [sumsqv](BLISObjectAPI.md#sumsqv), [getsc](BLISObjectAPI.md#getsc), [getijv](BLISObjectAPI.md#getijv), [getijm](BLISObjectAPI.md#getijm), [setsc](BLISObjectAPI.md#setsc), [setijv](BLISObjectAPI.md#setijv), [setijm](BLISObjectAPI.md#setijm), [eqsc](BLISObjectAPI.md#eqsc), [eqv](BLISObjectAPI.md#eqv), [eqm](BLISObjectAPI.md#eqm)



# Introduction

This document summarizes one of the primary native APIs in BLIS--the object API. Here, we also discuss BLIS-specific type definitions, header files, and prototypes to auxiliary functions.

There are many functions that BLIS implements that are not listed here, either because they are lower-level functions, or they are considered for use primarily by developers and experts.

The object API was given its name (a) because it abstracts the floating-point types of its operands (along with many other properties) within a `typedef struct {...}` data structure, and (b) to contrast it with the other native API in BLIS, the typed API, which is [documented here](BLISTypedAPI.md). (The third API supported by BLIS is the BLAS compatibility layer, which mimics conventional Fortran-77 BLAS.)

In general, this document should be treated more as a reference than a place to learn how to use BLIS in your application. Thus, we highly encourage all readers to first study the [example code](BLISObjectAPI.md#example-code) provided within the BLIS source distribution.


## BLIS types

The following tables list various types used throughout the BLIS object API.

### Integer-based types

| BLIS integer type | Type definition          | Used to represent...                                                     |
|:------------------|:-------------------------|:-------------------------------------------------------------------------|
| `gint_t`          | `int32_t` or `int64_t`   | general-purpose signed integer; used to define signed integer types.     |
| `guint_t`         | `uint32_t` or `uint64_t` | general-purpose unsigned integer; used to define unsigned integer types. |
| `dim_t`           | `gint_t`                 | matrix and vector dimensions.                                            |
| `inc_t`           | `gint_t`                 | matrix row/column strides and vector increments.                         |
| `doff_t`          | `gint_t`                 | matrix diagonal offset: if _k_ < 0, diagonal begins at element (-_k_,0); otherwise diagonal begins at element (0,_k_). |
| `siz_t`           | `guint_t`                | a byte size or byte offset, unsigned integer.                            |
| `kerid_t`         | `uint32_t`               | a kernel, block size, operation, or kernel preference ID.                |

### Floating-point types

| BLIS fp type | Type definition                        | Used to represent...              |
|:-------------|:---------------------------------------|:----------------------------------|
| `float`      | _N/A_                                  | single-precision real numbers.    |
| `double`     | _N/A_                                  | double-precision real numbers.    |
| `scomplex`   | `struct { float real; float imag; }`   | single-precision complex numbers. |
| `dcomplex`   | `struct { double real; double imag; }` | double-precision complex numbers. |

### Enumerated parameter types

| `num_t`         | Semantic meaning: Matrix/vector operand...               |
|:----------------|:---------------------------------------------------------|
| `BLIS_FLOAT`    | contains single-precision real elements.                 |
| `BLIS_DOUBLE`   | contains double-precision real elements.                 |
| `BLIS_SCOMPLEX` | contains single-precision complex elements.              |
| `BLIS_DCOMPLEX` | contains double-precision complex elements.              |
| `BLIS_INT`      | contains integer elements of type `gint_t`.              |
| `BLIS_CONSTANT` | contains polymorphic representation of a constant value. |

| `dom_t`         | Semantic meaning: Matrix/vector operand... |
|:----------------|:-------------------------------------------|
| `BLIS_REAL`     | contains real domain elements.             |
| `BLIS_COMPLEX`  | contains complex domain elements.          |

| `prec_t`           | Semantic meaning: Matrix/vector operand... |
|:-------------------|:-------------------------------------------|
| `BLIS_SINGLE_PREC` | contains single-precision elements.        |
| `BLIS_DOUBLE_PREC` | contains double-precision elements.        |

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

| `struc_t`         | Semantic meaning: Matrix operand... |
|:------------------|:------------------------------------|
| `BLIS_GENERAL`    | has no structure.                   |
| `BLIS_HERMITIAN`  | has Hermitian structure.            |
| `BLIS_SYMMETRIC`  | has symmetric structure.            |
| `BLIS_TRIANGULAR` | has triangular structure.           |

| `uplo_t`     | Semantic meaning: Matrix operand...                               |
|:-------------|:------------------------------------------------------------------|
| `BLIS_LOWER` | is stored in (and will be accessed only from) the lower triangle. |
| `BLIS_UPPER` | is stored in (and will be accessed only from) the upper triangle. |
| `BLIS_DENSE` | is stored as a full matrix (ie: in both triangles).               |

| `diag_t`            | Semantic meaning: Matrix operand...                                        |
|:--------------------|:---------------------------------------------------------------------------|
| `BLIS_NONUNIT_DIAG` | has a non-unit diagonal that should be explicitly read from.               |
| `BLIS_UNIT_DIAG`    | has a unit diagonal that should be implicitly assumed (and not read from). |


## Global scalar constants

BLIS defines a handful of scalar objects that conveniently represent various constant values for all defined numerical type values (`num_t`). The following table lists the constants defined by BLIS.

| BLIS constant `obj_t` name | Numerical values |
|:---------------------------|:-----------------|
|  `BLIS_MINUS_TWO`          | `-2.0`           |
|  `BLIS_MINUS_ONE`          | `-1.0`           |
|  `BLIS_ZERO`               | ` 0.0`           |
|  `BLIS_ONE`                | ` 1.0`           |
|  `BLIS_TWO`                | ` 2.0`           |
|  `BLIS_MINUS_ONE_I`        | ` 0.0-1.0*i`     |
|  `BLIS_ONE_I`              | ` 0.0+1.0*i`     |
|  `BLIS_NAN`                | ` NaN`           |

These objects are polymorphic; each one contains a `float`, `double`, `scomplex`, `dcomplex`, and `gint_t` representation of the constant value in question. They can be used in place of any `obj_t*` operand in any object API function provided that the following criteria are met:
 * The object parameter requires unit dimensions (1x1). (In other words, the function expects a scalar for the operand in question.)
 * The object parameter is input-only. (In other words, the function is not trying to update the scalar.)
The correct representation is chosen by context, usually by inspecting the datatype of one of the other operands involved in an operation. For example, if we create and initialize objects `x` and `y` of `num_t` type `BLIS_DOUBLE`, the following call to `bli_axpyv()`
   ```c
   bli_axpyv( &BLIS_TWO, &x, &y );
   ```
   will use the `BLIS_DOUBLE` representation of `BLIS_TWO`.


## Basic vs expert interfaces

The functions listed in this document belong to the "basic" interface subset of the BLIS object API. There is a companion "expert" interface that mirrors the basic interface, except that it also contains two additional parameters that are only of interest to experts and library developers. The expert interfaces use the same name as the basic function names, except for an additional "_ex" suffix. For example, the basic interface for `gemm` is
```c
void bli_gemm
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c
     );
```
while the expert interface is:
```c
void bli_gemm_ex
     (
       const obj_t*   alpha,
       const obj_t*   a,
       const obj_t*   b,
       const obj_t*   beta,
       const obj_t*   c,
       const cntx_t*  cntx,
       const rntm_t*  rntm
     );
```
The expert interface contains two additional parameters: a `cntx_t*` and `rntm_t*`. Note that calling a function from the expert interface with the `cntx_t*` and `rntm_t*` arguments each set to `NULL` is equivalent to calling the corresponding basic interface. Specifically, a `NULL` value passed in for the `cntx_t*` results in a valid context being queried from BLIS, and a `NULL` value passed in for the `rntm_t*` results in the current global settings for multithreading to be used.

## Context type

In general, it is permissible to pass in `NULL` for a `cntx_t*` parameter when calling an expert interface such as `bli_gemm_ex()`. However, there are cases where `NULL` values are not accepted and may result in a segmentation fault. Specifically, the `cntx_t*` argument appears in the interfaces to the `gemm`, `trsm`, and `gemmtrsm` [level-3 microkernels](KernelsHowTo.md#level-3) along with all [level-1v](KernelsHowTo.md#level-1v) and [level-1f](KernelsHowTo.md#level-1f) kernels. There, as a general rule, a valid pointer must be passed in. Whenever a valid context is needed, the developer may query a default context from the global kernel structure (if a context is not already available in the current scope):
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


# Object management

## Introduction

Before using the object API, you must first create some objects to encapsulate your vector or matrix data. We provide examples code for creating matrix objects in the [examples/oapi](https://github.com/flame/blis/tree/master/examples/oapi) directory of the BLIS source distribution. However, we will provide API documentation for the most common functions for creating and freeing objects in the next section.

Generally speaking, an object is created when an `obj_t` structure is initialized with valid properties describing the object as well as a valid data buffer (to hold the elements of the vector or matrix). The valid data buffer can be allocated automatically on your behalf at the same time that the other object fields are initialized, or "attached" in a second step after the object is initialized with preliminary values. The former is useful when using the object API at the setup stage of an application (and if `malloc()` is an acceptable method of allocating memory). Similarly, the latter is useful when interfacing BLIS into the middle of an application after the allocation has already taken place, or when some function other than `malloc()` is desired for allocating the buffer.

Only objects that were created with automatic allocation must be freed via BLIS object API. Objects that were initialized with attached buffers can be freed in whatever manner is appropriate, based on how the application originally allocated the memory in question.

## Object creation function reference

```c
void bli_obj_create
     (
       num_t   dt,
       dim_t   m,
       dim_t   n,
       inc_t   rs,
       inc_t   cs,
       obj_t*  obj
     );
```
Initialize an _m x n_ object `obj` and allocate sufficient storage to hold _mn_ elements whose storage type is specified by `dt` and with row and column strides `rs` and `cs`, respectively. This function allocates enough space to enforce alignment of leading dimensions, where the alignment factor is specific to the configuration being used, though the alignment factor is almost always equal to the size of the hardware's SIMD registers.
The address `obj` must reference valid memory--typically an `obj_t` declared statically or allocated dynamically via `malloc()`.
After an object created via `bli_obj_create()` is no longer needed, it should be deallocated via `bli_obj_free()`.

---

```c
void bli_obj_free
     (
       obj_t*  obj
     );
```
Deallocate (release) an object `obj` that was previously created, typically via `bli_obj_create()`.

---

```c
void bli_obj_create_without_buffer
     (
       num_t   dt,
       dim_t   m,
       dim_t   n,
       obj_t*  obj
     );
```
Partially initialize an _m x n_ object `obj` that will eventually contain elements whose storage type is specified by `dt`. This function does not result in any memory allocation. Before `obj` can be used, the object must be fully initialized by attaching a buffer via `bli_obj_attach_buffer()`. This function is useful when the user wishes to encapsulate existing buffers into one or more `obj_t` objects.
An object (partially) initialized via this function should generally not be passed to `bli_obj_free()` even after a buffer is attached to it via `bli_obj_attach_buffer()`, unless the user wishes to pass that buffer into `free()`.

---

```c
void bli_obj_attach_buffer
     (
       void*   p,
       inc_t   rs,
       inc_t   cs,
       inc_t   is,
       obj_t*  obj
     );
```
Given a partially initialized object (i.e., one that has already been passed to `bli_obj_create_without_buffer()`), attach the buffer pointed to by `p` to the object referenced by `obj` and initialize `obj` as containing elements with row and column strides `rs` and `cs`, respectively. The function also initializes the imaginary stride as `is`, which is experimental and not consistently used by all parts of BLIS.

---

```c
void bli_obj_create_with_attached_buffer
     (
       num_t   dt,
       dim_t   m,
       dim_t   n,
       void*   p,
       inc_t   rs,
       inc_t   cs,
       obj_t*  obj
     );
```
Initialize an _m x n_ object `obj` as containing _mn_ elements whose storage type is specified by `dt` and with row and column strides `rs` and `cs`, respectively. The function does not allocate any memory and instead attaches the buffer pointed to by `p`. Note that calling this function is effectively equivalent to calling
```c
bli_obj_create_without_buffer( dt, m, n, obj );
bli_obj_attach_buffer( p, rs, cs, 1, obj );
```
Objects initialized via this function should generally not be passed to `bli_obj_free()`, unless the user wishes to pass `p` into `free()`.

---

```c
void bli_obj_alloc_buffer
     (
       inc_t   rs,
       inc_t   cs,
       inc_t   is,
       obj_t*  obj
     );
```
Given a partially initialized _m x n_ object, allocate and attach a buffer large enough to contain _mn_ elements with the row and column strides `rs` and `cs`, respectively. This function allocates enough space to enforce alignment of leading dimensions, where the alignment factor is specific to the configuration being used, though the alignment factor is almost always equal to the size of the hardware's SIMD registers.
Note that calling `bli_obj_create()` is effectively equivalent to calling
```c
bli_obj_create_without_buffer( dt, m, n, obj );
bli_obj_alloc_buffer( rs, cs, 1, obj );
```
Very few users will likely have a need to call this function. We provide documentation for it mostly so that others can manually access the alignment features of `bli_obj_create()` without also needing to initialize an `obj_t`.

---

```c
void bli_obj_create_1x1
     (
       num_t   dt,
       obj_t*  obj
     );
```
Initialize a _1 x 1_ object `obj` and allocate sufficient storage to hold one element whose storage type is specified by `dt`.
The address `obj` must reference valid memory--typically an `obj_t` declared statically or allocated dynamically via `malloc()`.
This function is useful any time the user wishes to create a scalar object with an allocated buffer.
Note that calling `bli_obj_create_1x1()` is effectively equivalent to calling
```c
bli_obj_create_without_buffer( dt, 1, 1, obj );
bli_obj_alloc_buffer( 1, 1, 1, obj );
```
After an object created via `bli_obj_create_1x1()` is no longer needed, it should be deallocated via `bli_obj_free()`.

---

```c
void bli_obj_create_1x1_with_attached_buffer
     (
       num_t   dt,
       void*   p,
       obj_t*  obj
     );
```
Initialize a _1 x 1_ object `obj` as containing one element whose storage type is specified by `dt`. The function does not allocate any memory and instead attaches the buffer pointed to by `p`. Note that calling this function is effectively equivalent to calling
```c
bli_obj_create_without_buffer( dt, 1, 1, obj );
bli_obj_attach_buffer( p, 1, 1, 1, obj );
```
Objects initialized via this function should generally not be passed to `bli_obj_free()`, unless the user wishes to pass `p` into `free()`.

---

```c
void bli_obj_create_conf_to
     (
       const obj_t*  s,
             obj_t*  d
     );
```
Initialize an object `d` with dimensions conformal to those of an existing object `s`. Object `d` is initialized with the same row and column strides as those of `s`. However, the structure, uplo, conjugation, and transposition properties of `s` are **not** inherited by `d`.
On entry, object `s` must be fully initialized and the address `d` must reference valid memory--typically an `obj_t` declared statically or allocated dynamically via `malloc()`.
Note that calling this function is effectively equivalent to calling
```c
num_t dt = bli_obj_dt( s );
dim_t m  = bli_obj_length( s );
dim_t n  = bli_obj_width( s );
inc_t rs = bli_obj_row_stride( s );
inc_t cs = bli_obj_col_stride( s );

bli_obj_create( dt, m, n, rs, cs, d );
```
After an object created via `bli_obj_create_conf_to()` is no longer needed, it should be deallocated via `bli_obj_free()`.

---

```c
void bli_obj_scalar_init_detached
     (
       num_t   dt,
       obj_t*  obj
     );
```
Initialize a _1 x 1_ object `obj` using internal storage sufficient to hold one element whose storage type is specified by `dt`. (Internal storage is present within every `obj_t` and is capable of holding on element of any supported type.) This function is similar to `bli_obj_create_1x1()`, except that the object does not trigger any dynamic memory allocation.
Objects initialized via this function should **never** be passed to `bli_obj_free()`.


## Object accessor function reference

Notes for interpreting function descriptions:
  * Object accessor functions allow the caller to query certain properties of objects.
  * These functions are only guaranteed to return meaningful values when called upon objects that have been fully initialized/created.
  * Many specialized functions are omitted from this section for brevity. For a full list of accessor functions, please see [frame/include/bli_obj_macro_defs.h](https://github.com/flame/blis/tree/master/frame/include/bli_obj_macro_defs.h), though most users will most likely not need methods beyond those documented below.

---

```c
num_t bli_obj_dt( const obj_t* obj );
```
Return the storage datatype property of `obj`.

---

```c
dom_t bli_obj_domain( const obj_t* obj );
```
Return the domain component of the storage datatype property of `obj`.

---

```c
prec_t bli_obj_prec( const obj_t* obj );
```
Return the precision component of the storage datatype property of `obj`.

---

```c
trans_t bli_obj_conjtrans_status( const obj_t* obj );
```
Return the `trans_t` property of `obj`, which may indicate transposition, conjugation, both, or neither. Thus, possible return values are `BLIS_NO_TRANSPOSE`, `BLIS_CONJ_NO_TRANSPOSE`, `BLIS_TRANSPOSE`, or `BLIS_CONJ_TRANSPOSE`.

---

```c
trans_t bli_obj_onlytrans_status( const obj_t* obj );
```
Return the transposition component of the `trans_t` property of `obj`, which may indicate transposition or no transposition.
Thus, possible return values are `BLIS_NO_TRANSPOSE` or `BLIS_TRANSPOSE`.

---

```c
conj_t bli_obj_conj_status( const obj_t* obj );
```
Return the conjugation component of the `trans_t` property of `obj`, which may indicate conjugation or no conjugation.
Thus, possible return values are `BLIS_NO_CONJUGATE` or `BLIS_CONJUGATE`.

---

```c
struc_t bli_obj_struc( const obj_t* obj );
```
Return the structure property of `obj`.

---

```c
uplo_t bli_obj_uplo( const obj_t* obj );
```
Return the uplo (i.e., storage) property of `obj`.

---

```c
diag_t bli_obj_diag( const obj_t* obj );
```
Return the diagonal property of `obj`.

---

```c
doff_t bli_obj_diag_offset( const obj_t* obj );
```
Return the diagonal offset of `obj`. Note that the diagonal offset will be negative, `-i`, if the diagonal begins at element `(-i,0)` and positive `j` if the diagonal begins at element `(0,j)`.

---

```c
dim_t bli_obj_length( const obj_t* obj );
```
Return the number of rows (or _m_ dimension) of `obj`. This value is the _m_ dimension **before** taking into account the transposition property as indicated by `bli_obj_onlytrans_status()` or `bli_obj_conjtrans_status()`.

---

```c
dim_t bli_obj_width( const obj_t* obj );
```
Return the number of columns (or _n_ dimension) of `obj`. This value is the _n_ dimension **before** taking into account the transposition property as indicated by `bli_obj_onlytrans_status()` or `bli_obj_conjtrans_status()`.

---

```c
dim_t bli_obj_length_after_trans( const obj_t* obj );
```
Return the number of rows (or _m_ dimension) of `obj` after taking into account the transposition property as indicated by `bli_obj_onlytrans_status()` or `bli_obj_conjtrans_status()`.

---

```c
dim_t bli_obj_width_after_trans( const obj_t* obj );
```
Return the number of columns (or _n_ dimension) of `obj` after taking into account the transposition property as indicated by `bli_obj_onlytrans_status()` or `bli_obj_conjtrans_status()`.

---

```c
inc_t bli_obj_row_stride( const obj_t* obj );
```
Return the row stride property of `obj`. When storing by columns, the row stride is 1. When storing by rows, the row stride is also sometimes called the _leading dimension_.

---

```c
inc_t bli_obj_col_stride( const obj_t* obj );
```
Return the column stride property of `obj`. When storing by rows, the column stride is 1. When storing by columns, the column stride is also sometimes called the _leading dimension_.

---

```c
dim_t bli_obj_vector_dim( const obj_t* obj );
```
Return the number of elements in a vector object `obj`.
This function assumes that at least one dimension of `obj` is unit, and that it therefore represents a vector.

---

```c
inc_t bli_obj_vector_inc( const obj_t* obj );
```
Return the storage increment of a vector object `obj`.
This function assumes that at least one dimension of `obj` is unit, and that it therefore represents a vector.

---

```c
void* bli_obj_buffer( const obj_t* obj );
```
Return the address to the data buffer associated with object `obj`.
**Note**: The address returned by this buffer will not take into account any subpartitioning. However, this will not be a problem for most casual users.

---

```c
siz_t bli_obj_elem_size( const obj_t* obj );
```
Return the size, in bytes, of the storage datatype as indicated by `bli_obj_dt()`.



## Object mutator function reference

Notes for interpreting function descriptions:
  * Object mutator functions allow the caller to modify certain properties of objects.
  * The user should be extra careful about modifying properties after objects are created. For typical use of these functions, please study the example code provided in [examples/oapi](https://github.com/flame/blis/tree/master/examples/oapi).
  * The list of mutators below is much shorter than the list of accessor functions provided in the previous section. Most mutator functions should *not* be called by users (unless you know what you are doing). For a full list of mutator functions, please see [frame/include/bli_obj_macro_defs.h](https://github.com/flame/blis/tree/master/frame/include/bli_obj_macro_defs.h), though most users will most likely not need methods beyond those documented below.

---

```c
void bli_obj_set_conjtrans( trans_t trans, obj_t* obj );
```
Set both conjugation and transposition properties of `obj` using the corresponding components of `trans`.

---

```c
void bli_obj_set_onlytrans( trans_t trans, obj_t* obj );
```
Set the transposition property of `obj` using the transposition component of `trans`. Leaves the conjugation property of `obj` unchanged.

---

```c
void bli_obj_set_conj( conj_t conj, obj_t* obj );
```
Set the conjugation property of `obj` using `conj`. Leaves the transposition property of `obj` unchanged.

---

```c
void bli_obj_apply_trans( trans_t trans, obj_t* obj );
```
Apply `trans` to the transposition property of `obj`. For example, applying `BLIS_TRANSPOSE` will toggle the transposition property of `obj` but leave the conjugation property unchanged; applying `BLIS_CONJ_TRANSPOSE` will toggle both the conjugation and transposition properties of `obj`.

---

```c
void bli_obj_apply_conj( conj_t conj, obj_t* obj );
```
Apply `conj` to the conjugation property of `obj`. Specifically, applying `BLIS_CONJUGATE` will toggle the conjugation property of `obj`; applying `BLIS_NO_CONJUGATE` will have no effect. Leaves the transposition property of `obj` unchanged.

---

```c
void bli_obj_set_struc( struc_t struc, obj_t* obj );
```
Set the structure property of `obj` to `struc`.

---

```c
void bli_obj_set_uplo( uplo_t uplo, obj_t* obj );
```
Set the uplo (i.e., storage) property of `obj` to `uplo`.

---

```c
void bli_obj_set_diag( diag_t diag, obj_t* obj );
```
Set the diagonal property of `obj` to `diag`.

---

```c
void bli_obj_set_diag_offset( doff_t doff, obj_t* obj );
```
Set the diagonal offset property of `obj` to `doff`. Note that `doff_t` may be typecast from any signed integer.

---


## Other object function reference

---

```c
void bli_obj_induce_trans( obj_t* obj );
```
Modify the properties of `obj` to induce a logical transposition. This function operates without regard to whether the transposition property is already set. Therefore, depending on the circumstance, the caller may or may not wish to clear the transposition property after calling this function.

---

```c
void bli_obj_alias_to( const obj_t* a, obj_t* b );
```
Initialize `b` to be a shallow copy, or alias, of `a`. For most people's purposes, this is equivalent to
```c
  b = a;
```
However, there is at least one field (one that only developers should be concerned with) that is not copied.

---

```c
void bli_obj_real_part( const obj_t* c, obj_t* r );
```
Initialize `r` to be a modified shallow copy of `c` that refers only to the real part of `c`.

---

```c
void bli_obj_imag_part( const obj_t* c, obj_t* i );
```
Initialize `i` to be a modified shallow copy of `c` that refers only to the imaginary part of `c`.


# Computational function reference

Notes for interpreting function descriptions:
  * `conj?(X)` and `trans?(X)` should be interpreted as predicates that capture the operand `X` with that object's `conj_t` or `trans_t` property applied. For example:
    * `conj?(x)` refers to a vector `x` that is either conjugated or used as given.
    * `trans?(A)` refers to a matrix `A` that is either transposed, conjugated _and_ transposed, conjugated only, or used as given.
  * Any operand marked with `conj()` is unconditionally conjugated.
  * Any operand marked with `^T` is unconditionally transposed. Similarly, any operand that is marked with `^H` is unconditionally conjugate-transposed.
  * All occurrences of `alpha`, `beta`, and `rho` parameters are scalars.
  * In general, unless otherwise noted, all object parameters must be stored using the same `num_t` datatype. In a few cases, one of the object parameters must be stored in the real projection of one of the other objects' types. (The real projection of a `num_t` datatype is the equivalent datatype in the real domain. So `BLIS_DOUBLE` is the real projection of `BLIS_DCOMPLEX`. `BLIS_DOUBLE` is also the real projection of itself.)
  * Many object API entries list the object properties that are honored/observed by the operation. For example, for `bli_gemv()`, the observed object properties are `trans?(A)` and `conj?(x)`. The former means that matrix `A` may be (optionally) marked for conjugation and/or tranaposition while the latter means that vector `x` may be (optionally) marked for conjugation. A function may also list `diagoff(A)` as an observe property, which means that it will accept general diagonal offsets. Similarly, `diag(A)` refers to recognizing the unit/non-unit structure of the diagonal and and `uplo(A)` refers to reading/updating only the stored triangle/trapezoid/region of `A`.


---


## Level-1v operations

Level-1v operations perform various level-1 BLAS-like operations on vectors (hence the _v_).
**Note**: Most level-1v operations have a corresponding level-1v kernel through which it is primarily implemented.

---

#### addv
```c
void bli_addv
     (
       const obj_t*  x,
       const obj_t*  y
     );
```
Perform
```
  y := y + conj?(x)
```
where `x` and `y` are vectors of length _n_.

Observed object properties: `conj?(x)`.

---

#### amaxv
```c
void bli_amaxv
     (
       const obj_t*  x,
       const obj_t*  index
     );
```
Given a vector of length _n_, return the zero-based index of the element of vector `x` that contains the largest absolute value (or, in the complex domain, the largest complex modulus). The object `index` must be created of type `BLIS_INT`.

If `NaN` is encountered, it is treated as if it were a valid value that was smaller than any other value in the vector. If more than one element contains the same maximum value, the index of the latter element is returned via `index`.

Observed object properties: none.

**Note:** This function attempts to mimic the algorithm for finding the element with the maximum absolute value in the netlib BLAS routines `i?amax()`.

---

#### axpyv
```c
void bli_axpyv
     (
       const obj_t*  alpha,
       const obj_t*  x,
       const obj_t*  y
     );
```
Perform
```
  y := y + conj?(alpha) * conj?(x)
```
where `x` and `y` are vectors of length _n_, and `alpha` is a scalar.

Observed object properties: `conj?(alpha)`, `conj?(x)`.

---

#### axpbyv
```c
void bli_axpbyv
     (
       const obj_t*  alpha,
       const obj_t*  x,
       const obj_t*  beta,
       const obj_t*  y
     );
```
Perform
```
  y := conj?(beta) * y + conj?(alpha) * conj?(x)
```
where `x` and `y` are vectors of length _n_, and `alpha` and `beta` are scalars.

Observed object properties: `conj?(alpha)`, `conj?(x)`.

---

#### copyv
```c
void bli_copyv
     (
       const obj_t*  x,
       const obj_t*  y
     );
```
Perform
```
  y := conj?(x)
```
where `x` and `y` are vectors of length _n_.

Observed object properties: `conj?(x)`.

---

#### dotv
```c
void bli_dotv
     (
       const obj_t*  x,
       const obj_t*  y,
       const obj_t*  rho
     );
```
Perform
```
  rho := conj?(x)^T * conj?(y)
```
where `x` and `y` are vectors of length _n_, and `rho` is a scalar.

Observed object properties: `conj?(x)`, `conj?(y)`.

---

#### dotxv
```c
void bli_dotxv
     (
       const obj_t*  alpha,
       const obj_t*  x,
       const obj_t*  y,
       const obj_t*  beta,
       const obj_t*  rho
     );
```
Perform
```
  rho := conj?(beta) * rho + conj?(alpha) * conj?(x)^T * conj?(y)
```
where `x` and `y` are vectors of length _n_, and `alpha`, `beta`, and `rho` are scalars.

Observed object properties: `conj?(alpha)`, `conj?(beta)`, `conj?(x)`, `conj?(y)`.

---

#### invertv
```c
void bli_invertv
     (
       const obj_t*  x
     );
```
Invert all elements of an _n_-length vector `x`.

---

#### invscalv
```c
void bli_invscalv
     (
       const obj_t*  alpha,
       const obj_t*  x
     );
```
Perform
```
  x := ( 1.0 / conj?(alpha) ) * x
```
where `x` is a vector of length _n_, and `alpha` is a scalar.

Observed object properties: `conj?(alpha)`.

---

#### scalv
```c
void bli_scalv
     (
       const obj_t*  alpha,
       const obj_t*  x
     );
```
Perform
```
  x := conj?(alpha) * x
```
where `x` is a vector of length _n_, and `alpha` is a scalar.

Observed object properties: `conj?(alpha)`.

---

#### scal2v
```c
void bli_scal2v
     (
       const obj_t*  alpha,
       const obj_t*  x,
       const obj_t*  y
     );
```
Perform
```
  y := conj?(alpha) * conj?(x)
```
where `x` and `y` are vectors of length _n_, and `alpha` is a scalar.

Observed object properties: `conj?(alpha)`, `conj?(x)`.

---

#### setv
```c
void bli_setv
     (
       const obj_t*  alpha,
       const obj_t*  x
     );
```
Perform
```
  x := conj?(alpha)
```
That is, set all elements of an _n_-length vector `x` to scalar `conj?(alpha)`.

Observed object properties: `conj?(alpha)`.

---

#### setrv
```c
void bli_setrv
     (
       const obj_t*  alpha,
       const obj_t*  x
     );
```
Perform
```
  real(x) := real(alpha)
```
That is, given an _n_-length vector `x`, set all elements' real components to the real component of scalar `alpha`. (If `alpha` is complex, the imaginary component is ignored.)
If `x` is real, this operation is equivalent to performing `setv` on `x` with the real component of scalar `alpha`.
**Note**: This operation is provided for convenience as an object wrapper to `setv`, and thus it has no analogue in the [BLIS typed API](BLISTypedAPI).

---

#### setiv
```c
void bli_setiv
     (
       const obj_t*  alpha,
       const obj_t*  x
     );
```
Perform
```
  imag(x) := real(alpha)
```
That is, given an _n_-length vector `x`, set all elements' imaginary components to the real component of scalar `alpha`. (If `alpha` is complex, the imaginary component is ignored.)
If `x` is real, this operation is equivalent to a no-op.
**Note**: This operation is provided for convenience as an object wrapper to `setv`, and thus it has no analogue in the [BLIS typed API](BLISTypedAPI).

---

#### subv
```c
void bli_subv
     (
       const obj_t*  x,
       const obj_t*  y
     );
```
Perform
```
  y := y - conj?(x)
```
where `x` and `y` are vectors of length _n_.

Observed object properties: `conj?(x)`.

---

#### swapv
```c
void bli_swapv
     (
       const obj_t*  x,
       const obj_t*  y
     );
```
Swap corresponding elements of two _n_-length vectors `x` and `y`.

---

#### xpbyv
```c
void bli_xpbyv
     (
       const obj_t*  x,
       const obj_t*  beta,
       const obj_t*  y
     );
```
Perform
```
  y := conj?(beta) * y + conj?(x)
```
where `x` and `y` are vectors of length _n_, and `beta` is a scalar.

Observed object properties: `conj?(beta)`, `conj?(x)`.

---





## Level-1d operations

Level-1d operations perform various level-1 BLAS-like operations on matrix diagonals (hence the _d_).

These operations are similar to their level-1m counterparts, except they only read and update matrix diagonals and therefore ignore the `uplo` property of their applicable input operands. Please see the descriptions for the corresponding level-1m operation for a description of the arguments.

---

#### addd
```c
void bli_addd
     (
       const obj_t*  a,
       const obj_t*  b
     );
```

Observed object properties: `diagoff(A)`, `diag(A)`, `trans?(A)`.

---

#### axpyd
```c
void bli_axpyd
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b
     );
```

Observed object properties: `conj?(alpha)`, `diagoff(A)`, `diag(A)`, `trans?(A)`.

---

#### copyd
```c
void bli_copyd
     (
       const obj_t*  a,
       const obj_t*  b
     );
```

Observed object properties: `diagoff(A)`, `diag(A)`, `trans?(A)`.

---

#### invertd
```c
void bli_invertd
     (
       const obj_t*  a
     );
```

Observed object properties: `diagoff(A)`.

---

#### invscald
```c
void bli_invscald
     (
       const obj_t*  alpha,
       const obj_t*  a
     );
```

Observed object properties: `conj?(alpha)`, `diagoff(A)`.

---

#### scald
```c
void bli_scald
     (
       const obj_t*  alpha,
       const obj_t*  a
     );
```

Observed object properties: `conj?(alpha)`, `diagoff(A)`.

---

#### scal2d
```c
void bli_scal2d
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b
     );
```

Observed object properties: `conj?(alpha)`, `diagoff(A)`, `diag(A)`, `trans?(A)`.

---

#### setd
```c
void bli_setd
     (
       const obj_t*  alpha,
       const obj_t*  a
     );
```

Observed object properties: `conj?(alpha)`, `diagoff(A)`.

---

#### setid
```c
void bli_setid
     (
       const obj_t*  alpha,
       const obj_t*  a
     );
```
Set the imaginary components of every element along the diagonal of `a`
to a scalar `alpha`.
Note that the datatype of `alpha` must be the real projection of the datatype
of `a`.

Observed object properties: `diagoff(A)`.

---

#### shiftd
```c
void bli_shiftd
     (
       const obj_t*  alpha,
       const obj_t*  a
     );
```
Add a constant value `alpha` to every element along the diagonal of `a`.

Observed object properties: `diagoff(A)`.

---

#### subd
```c
void bli_subd
     (
       const obj_t*  a,
       const obj_t*  b
     );
```

Observed object properties: `diagoff(A)`, `diag(A)`, `trans?(A)`.

---

#### xpbyd
```c
void bli_xpbyd
     (
       const obj_t*  a,
       const obj_t*  beta,
       const obj_t*  b
     );
```

Observed object properties: `conj?(beta)`, `diagoff(A)`, `diag(A)`, `trans?(A)`.

---



## Level-1m operations

Level-1m operations perform various level-1 BLAS-like operations on matrices (hence the _m_).

---

#### addm
```c
void bli_addm
     (
       const obj_t*  a,
       const obj_t*  b
     );
```
Perform
```
  B := B + trans?(A)
```
where `B` is an _m x n_ matrix, `A` is stored as a dense matrix, or lower- or upper-triangular/trapezoidal matrix with arbitrary diagonal offset and unit or non-unit diagonal.
If `uplo(A)` indicates lower or upper storage, only that part of matrix `A` will be referenced and used to update `B`.

Observed object properties: `diagoff(A)`, `diag(A)`, `uplo(A)`, `trans?(A)`.

---

#### axpym
```c
void bli_axpym
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b
     );
```
Perform
```
  B := B + conj?(alpha) * trans?(A)
```
where `B` is an _m x n_ matrix, `A` is stored as a dense matrix, or lower- or upper-triangular/trapezoidal matrix with arbitrary diagonal offset and unit or non-unit diagonal.
If `uplo(A)` indicates lower or upper storage, only that part of matrix `A` will be referenced and used to update `B`.

Observed object properties: `conj?(alpha)`, `diagoff(A)`, `diag(A)`, `uplo(A)`, `trans?(A)`.

---

#### copym
```c
void bli_copym
     (
       const obj_t*  a,
       const obj_t*  b
     );
```
Perform
```
  B := trans?(A)
```
where `B` is an _m x n_ matrix, `A` is stored as a dense matrix, or lower- or upper-triangular/trapezoidal matrix with arbitrary diagonal offset and unit or non-unit diagonal.
If `uplo(A)` indicates lower or upper storage, only that part of matrix `A` will be referenced and used to update `B`.

Observed object properties: `diagoff(A)`, `diag(A)`, `uplo(A)`, `trans?(A)`.

---

#### invscalm
```c
void bli_invscalm
     (
       const obj_t*  alpha,
       const obj_t*  a
     );
```
Perform
```
  A := ( 1.0 / conj?(alpha) ) * A
```
where `A` is an _m x n_ matrix stored as a dense matrix, or lower- or upper-triangular/trapezoidal matrix with arbitrary diagonal offset. If `uplo(A)` indicates lower or upper storage, only that part of matrix `A` will be updated.

Observed object properties: `conj?(alpha)`, `diagoff(A)`, `uplo(A)`.

---

#### scalm
```c
void bli_scalm
     (
       const obj_t*  alpha,
       const obj_t*  a
     );
```
Perform
```
  A := conj?(alpha) * A
```
where `A` is an _m x n_ matrix stored as a dense matrix, or lower- or upper-triangular/trapezoidal matrix with arbitrary diagonal offset. If `uplo(A)` indicates lower or upper storage, only that part of matrix `A` will be updated.

Observed object properties: `conj?(alpha)`, `diagoff(A)`, `uplo(A)`.

---

#### scal2m
```c
void bli_scal2m
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b
     );
```
Perform
```
  B := conj?(alpha) * trans?(A)
```
where `B` is an _m x n_ matrix, `A` is stored as a dense matrix, or lower- or upper-triangular/trapezoidal matrix with arbitrary diagonal offset and unit or non-unit diagonal.
If `uplo(A)` indicates lower or upper storage, only that part of matrix `A` will be referenced and used to update `B`.

Observed object properties: `conj?(alpha)`, `diagoff(A)`, `diag(A)`, `uplo(A)`, `trans?(A)`.

---

#### setm
```c
void bli_setm
     (
       const obj_t*  alpha,
       const obj_t*  a
     );
```
Perform
```
  A := conj?(alpha)
```
That is, set all elements of `A` to scalar `conj?(alpha)`, where `A` is an _m x n_ matrix stored as a dense matrix, or lower- or upper-triangular/trapezoidal matrix with arbitrary diagonal offset. If `uplo(A)` indicates lower or upper storage, only that part of matrix `A` will be updated.

Observed object properties: `conj?(alpha)`, `diagoff(A)`, `diag(A)`, `uplo(A)`.

---

#### setrm
```c
void bli_setrm
     (
       const obj_t*  alpha,
       const obj_t*  a
     );
```
Perform
```
  real(A) := real(alpha)
```
That is, given an _m x n_ matrix `A`, set all elements' real components to the real component of scalar `alpha`. (If `alpha` is complex, the imaginary component is ignored.)
If `A` is real, this operation is equivalent to performing `setm` on `A` with the real component of scalar `alpha`.
**Note**: This operation is provided for convenience as an object wrapper to `setm`, and thus it has no analogue in the [BLIS typed API](BLISTypedAPI).

Observed object properties: `diagoff(A)`, `diag(A)`, `uplo(A)`.

---

#### setim
```c
void bli_setim
     (
       const obj_t*  alpha,
       const obj_t*  a
     );
```
Perform
```
  imag(A) := real(alpha)
```
That is, given an _m x n_ matrix `A`, set all elements' imaginary components to the real component of scalar `alpha`. (If `alpha` is complex, the imaginary component is ignored.)
If `A` is real, this operation is equivalent to a no-op.
**Note**: This operation is provided for convenience as an object wrapper to `setm`, and thus it has no analogue in the [BLIS typed API](BLISTypedAPI).

Observed object properties: `diagoff(A)`, `diag(A)`, `uplo(A)`.

---

#### subm
```c
void bli_subm
     (
       const obj_t*  a,
       const obj_t*  b
     );
```
Perform
```
  B := B - trans?(A)
```
where `B` is an _m x n_ matrix, `A` is stored as a dense matrix, or lower- or upper-triangular/trapezoidal matrix with arbitrary diagonal offset and unit or non-unit diagonal.
If `uplo(A)` indicates lower or upper storage, only that part of matrix `A` will be referenced and used to update `B`.

Observed object properties: `diagoff(A)`, `diag(A)`, `uplo(A)`, `trans?(A)`.

---





## Level-1f operations

Level-1f operations implement various fused combinations of level-1 operations (hence the _f_).
**Note**: Each level-1f operation has a corresponding level-1f kernel through which it is primarily implemented.

Level-1f kernels are employed when optimizing level-2 operations.


---

#### axpy2v
```c
void bli_axpy2v
     (
       const obj_t*  alphax,
       const obj_t*  alphay,
       const obj_t*  x,
       const obj_t*  y,
       const obj_t*  z
     );
```
Perform
```
  z := z + conj?(alphax) * conj?(x) + conj?(alphay) * conj?(y)
```
where `x`, `y`, and `z` are vectors of length _m_. The kernel, if optimized, is implemented as a fused pair of calls to [axpyv](BLISObjectAPI.md#axpyv).

Observed object properties: `conj?(alphax)`, `conj?(x)`, `conj?(alphay)`, `conj?(y)`.

---

#### dotaxpyv
```c
void bli_dotaxpyv
     (
       const obj_t*  alpha,
       const obj_t*  xt,
       const obj_t*  x,
       const obj_t*  y,
       const obj_t*  rho,
       const obj_t*  z
     );
```
Perform
```
  rho := conj?(x)^T * conj?(y)
  z   := z + conj?(alpha) * conj?(x)
```
where `x`, `y`, and `z` are vectors of length _m_ and `alpha` and `rho` are scalars. The kernel, if optimized, is implemented as a fusion of calls to [dotv](BLISObjectAPI.md#dotv) and [axpyv](BLISObjectAPI.md#axpyv).

Observed object properties: `conj?(x)`, `conj?(y)`, `conj?(alpha)`.

---

#### axpyf
```c
void bli_axpyf
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  x,
       const obj_t*  y
     );
```
Perform
```
  y := y + alpha * conja(A) * conjx(x)
```
where `A` is an _m x b_ matrix, and `x` and `y` are vectors. The kernel, if optimized, is implemented as a fused series of calls to [axpyv](BLISObjectAPI.md#axpyv) where _b_ is less than or equal to an implementation-dependent fusing factor specific to `axpyf`.

Observed object properties: `conj?(alpha)`, `conj?(A)`, `conj?(x)`.

---

#### dotxf
```c
void bli_dotxf
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  x,
       const obj_t*  beta,
       const obj_t*  y
     );
```
Perform
```
  y := conj?(beta) * y + conj?(alpha) * conj?(A)^T * conj?(x)
```
where `A` is an _m x b_ matrix, and `x` and `y` are vectors. The kernel, if optimized, is implemented as a fused series of calls to [dotxv](BLISObjectAPI.md#dotxv) where _b_ is less than or equal to an implementation-dependent fusing factor specific to `dotxf`.

Observed object properties: `conj?(alpha)`, `conj?(beta)`, `conj?(A)`, `conj?(x)`.

---

#### dotxaxpyf
```c
void bli_dotxaxpyf
     (
       const obj_t*  alpha,
       const obj_t*  at,
       const obj_t*  a,
       const obj_t*  w,
       const obj_t*  x,
       const obj_t*  beta,
       const obj_t*  y,
       const obj_t*  z
     );
```
Perform
```
  y := conj?(beta) * y + conj?(alpha) * conj?(A)^T * conj?(w)
  z :=               z + conj?(alpha) * conj?(A)   * conj?(x)
```
where `A` is an _m x b_ matrix, `w` and `z` are vectors of length _m_, `x` and `y` are vectors of length `b`, and `alpha` and `beta` are scalars. The kernel, if optimized, is implemented as a fusion of calls to [dotxf](BLISObjectAPI.md#dotxf) and [axpyf](BLISObjectAPI.md#axpyf).

Observed object properties: `conj?(alpha)`, `conj?(beta)`, `conj?(A)`, `conj?(w)`, `conj?(x)`.



## Level-2 operations

Level-2 operations perform various level-2 BLAS-like operations.


---

#### gemv
```c
void bli_gemv
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  x,
       const obj_t*  beta,
       const obj_t*  y
     );
```
Perform
```
  y := conj?(beta) * y + conj?(alpha) * trans?(A) * conj?(x)
```
where `trans?(A)` is an _m x n_ matrix, and `x` and `y` are vectors.

Observed object properties: `conj?(alpha)`, `conj?(beta)`, `trans?(A)`, `conj?(x)`.

---

#### ger
```c
void bli_ger
     (
       const obj_t*  alpha,
       const obj_t*  x,
       const obj_t*  y,
       const obj_t*  a
     );
```
Perform
```
  A := A + conj?(alpha) * conj?(x) * conj?(y)^T
```
where `A` is an _m x n_ matrix, and `x` and `y` are vectors of length _m_ and _n_, respectively.

Observed object properties: `conj?(alpha)`, `conj?(x)`, `conj?(y)`.

---

#### hemv
```c
void bli_hemv
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  x,
       const obj_t*  beta,
       const obj_t*  y
     );
```
Perform
```
  y := conj?(beta) * y + conj?(alpha) * conj?(A) * conj?(x)
```
where `A` is an _m x m_ Hermitian matrix stored in the lower or upper triangle as specified by `uplo(A)`, and `x` and `y` are vectors of length _m_.

Observed object properties: `conj?(alpha)`, `conj?(beta)`, `conj?(A)`, `uplo(A)`, `conj?(x)`.

---

#### her
```c
void bli_her
     (
       const obj_t*  alpha,
       const obj_t*  x,
       const obj_t*  a
     );
```
Perform
```
  A := A + conj?(alpha) * conj?(x) * conj?(x)^H
```
where `A` is an _m x m_ Hermitian matrix stored in the lower or upper triangle as specified by `uplo(A)`, and `x` is a vector of length _m_.

Observed object properties: `conj?(alpha)`, `uplo(A)`, `conj?(x)`.

**Note:** The floating-point (`num_t`) type of `alpha` is always the real projection of the floating-point types of `x` and `A`.

---

#### her2
```c
void bli_her2
     (
       const obj_t*  alpha,
       const obj_t*  x,
       const obj_t*  y,
       const obj_t*  a
     );
```
Perform
```
  A := A + alpha * conj?(x) * conj?(y)^H + conj(alpha) * conj?(y) * conj?(x)^H
```
where `A` is an _m x m_ Hermitian matrix stored in the lower or upper triangle as specified by `uplo(A)`, and `x` and `y` are vectors of length _m_.

Observed object properties: `uplo(A)`, `conj?(x)`, `conj?(y)`.

---

#### symv
```c
void bli_symv
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  x,
       const obj_t*  beta,
       const obj_t*  y
     );
```
Perform
```
  y := conj?(beta) * y + conj?(alpha) * conj?(A) * conj?(x)
```
where `A` is an _m x m_ symmetric matrix stored in the lower or upper triangle as specified by `uplo(A)`, and `x` and `y` are vectors of length _m_.

Observed object properties: `conj?(alpha)`, `conj?(beta)`, `conj?(A)`, `uplo(A)`, `conj?(x)`.

---

#### syr
```c
void bli_syr
     (
       const obj_t*  alpha,
       const obj_t*  x,
       const obj_t*  a
     );
```
Perform
```
  A := A + conj?(alpha) * conj?(x) * conj?(x)^T
```
where `A` is an _m x m_ symmetric matrix stored in the lower or upper triangle as specified by `uploa`, and `x` is a vector of length _m_.

Observed object properties: `conj?(alpha)`, `conj?(x)`.

---

#### syr2
```c
void bli_syr2
     (
       const obj_t*  alpha,
       const obj_t*  x,
       const obj_t*  y,
       const obj_t*  a
     );
```
Perform
```
  A := A + alpha * conj?(x) * conj?(y)^T + conj(alpha) * conj?(y) * conj?(x)^T
```
where `A` is an _m x m_ symmetric matrix stored in the lower or upper triangle as specified by `uplo(A)`, and `x` and `y` are vectors of length _m_.

Observed object properties: `uplo(A)`, `conj?(x)`, `conj?(y)`.

---

#### trmv
```c
void bli_trmv
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  x
     );
```
Perform
```
  x := conj?(alpha) * transa(A) * x
```
where `A` is an _m x m_ triangular matrix stored in the lower or upper triangle as specified by `uplo(A)` with unit/non-unit nature specified by `diag(A)`, and `x` is a vector of length _m_.

Observed object properties: `conj?(alpha)`, `uplo(A)`, `trans?(A)`, `diag(A)`.

---

#### trsv
```c
void bli_trsv
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  y
     );
```
Solve the linear system
```
  transa(A) * x = alpha * y
```
where `A` is an _m x m_ triangular matrix stored in the lower or upper triangle as specified by `uplo(A)` with unit/non-unit nature specified by `diag(A)`, and `x` and `y` are vectors of length _m_. The right-hand side vector operand `y` is overwritten with the solution vector `x`.

Observed object properties: `conj?(alpha)`, `uplo(A)`, `trans?(A)`, `diag(A)`.

---



## Level-3 operations

Level-3 operations perform various level-3 BLAS-like operations.
**Note**: Each All level-3 operations are implemented through a handful of level-3 microkernels. Please see the [Kernels Guide](KernelsHowTo.md) for more details.


---

#### gemm
```c
void bli_gemm
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c
     );
```
Perform
```
  C := beta * C + alpha * trans?(A) * trans?(B)
```
where `C` is an _m x n_ matrix, `trans?(A)` is an _m x k_ matrix, and `trans?(B)` is a _k x n_ matrix.

Observed object properties: `trans?(A)`, `trans?(B)`.

---

#### gemmt
```c
void bli_gemmt
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c
     );
```
Perform
```
  C := beta * C + alpha * trans?(A) * trans?(B)
```
where `C` is an _m x m_ matrix, `trans?(A)` is an _m x k_ matrix, and `trans?(B)` is a _k x m_ matrix. This operation is similar to `bli_gemm()` except that it only updates the lower or upper triangle of `C` as specified by `uplo(C)`.

Observed object properties: `trans?(A)`, `trans?(B)`, `uplo(C)`.

---

#### hemm
```c
void bli_hemm
     (
             side_t  sidea,
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c
     );
```
Perform
```
  C := beta * C + alpha * conj?(A) * trans?(B)
```
if `sidea` is `BLIS_LEFT`, or
```
  C := beta * C + alpha * trans?(B) * conj?(A)
```
if `sidea` is `BLIS_RIGHT`, where `C` and `B` are _m x n_ matrices and `A` is a Hermitian matrix stored in the lower or upper triangle as specified by `uplo(A)`. When `sidea` is `BLIS_LEFT`, `A` is _m x m_, and when `sidea` is `BLIS_RIGHT`, `A` is _n x n_.

Observed object properties: `uplo(A)`, `conj?(A)`, `trans?(B)`.

---

#### herk
```c
void bli_herk
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  beta,
       const obj_t*  c
     );
```
Perform
```
  C := beta * C + alpha * trans?(A) * trans?(A)^H
```
where `C` is an _m x m_ Hermitian matrix stored in the lower or upper triangle as specified by `uplo(C)` and `trans?(A)` is an _m x k_ matrix.

Observed object properties: `trans?(A)`, `uplo(C)`.

**Note:** The floating-point (`num_t`) types of `alpha` and `beta` are always the real projection of the floating-point types of `A` and `C`.

---

#### her2k
```c
void bli_her2k
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c
     );
```
Perform
```
  C := beta * C + alpha * trans?(A) * trans?(B)^H + conj(alpha) * trans?(B) * trans?(A)^H
```
where `C` is an _m x m_ Hermitian matrix stored in the lower or upper triangle as specified by `uplo(C)` and `trans?(A)` and `trans?(B)` are _m x k_ matrices.

Observed object properties: `trans?(A)`, `trans?(B)`, `uplo(C)`.

**Note:** The floating-point (`num_t`) type of `beta` is always the real projection of the floating-point types of `A` and `C`.

---

#### symm
```c
void bli_symm
     (
             side_t  sidea,
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c
     );
```
Perform
```
  C := beta * C + alpha * conj?(A) * trans?(B)
```
if `sidea` is `BLIS_LEFT`, or
```
  C := beta * C + alpha * trans?(B) * conj?(A)
```
if `sidea` is `BLIS_RIGHT`, where `C` and `B` are _m x n_ matrices and `A` is a symmetric matrix stored in the lower or upper triangle as specified by `uplo(A)`. When `sidea` is `BLIS_LEFT`, `A` is _m x m_, and when `sidea` is `BLIS_RIGHT`, `A` is _n x n_.

Observed object properties: `uplo(A)`, `conj?(A)`, `trans?(B)`.

---

#### syrk
```c
void bli_syrk
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  beta,
       const obj_t*  c
     );
```
Perform
```
  C := beta * C + alpha * trans?(A) * trans?(A)^T
```
where `C` is an _m x m_ symmetric matrix stored in the lower or upper triangle as specified by `uplo(A)` and `trans?(A)` is an _m x k_ matrix.

Observed object properties: `trans?(A)`, `uplo(C)`.

---

#### syr2k
```c
void bli_syr2k
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c
     );
```
Perform
```
  C := beta * C + alpha * trans?(A) * trans?(B)^T + alpha * trans?(B) * trans?(A)^T
```
where `C` is an _m x m_ symmetric matrix stored in the lower or upper triangle as specified by `uplo(A)` and `trans?(A)` and `trans?(B)` are _m x k_ matrices.

Observed object properties: `trans?(A)`, `trans?(B)`, `uplo(C)`.

---

#### trmm
```c
void bli_trmm
     (
             side_t  sidea,
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b
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
if `sidea` is `BLIS_RIGHT`, where `B` is an _m x n_ matrix and `A` is a triangular matrix stored in the lower or upper triangle as specified by `uplo(A)` with unit/non-unit nature specified by `diag(A)`. When `sidea` is `BLIS_LEFT`, `A` is _m x m_, and when `sidea` is `BLIS_RIGHT`, `A` is _n x n_.

Observed object properties: `uplo(A)`, `trans?(A)`, `diag(A)`.

---

#### trmm3
```c
void bli_trmm3
     (
             side_t  sidea,
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c
     );
```
Perform
```
  C := beta * C + alpha * trans?(A) * trans?(B)
```
if `sidea` is `BLIS_LEFT`, or
```
  C := beta * C + alpha * trans?(B) * trans?(A)
```
if `sidea` is `BLIS_RIGHT`, where `C` and `trans?(B)` are _m x n_ matrices and `A` is a triangular matrix stored in the lower or upper triangle as specified by `uplo(A)` with unit/non-unit nature specified by `diag(A)`. When `sidea` is `BLIS_LEFT`, `A` is _m x m_, and when `sidea` is `BLIS_RIGHT`, `A` is _n x n_.

Observed object properties: `uplo(A)`, `trans?(A)`, `diag(A)`, `trans?(B)`.

---

#### trsm
```c
void bli_trsm
     (
             side_t  sidea,
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b
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
if `sidea` is `BLIS_RIGHT`, where `X` and `B` are an _m x n_ matrices and `A` is a triangular matrix stored in the lower or upper triangle as specified by `uplo(A)` with unit/non-unit nature specified by `diag(A)`. When `sidea` is `BLIS_LEFT`, `A` is _m x m_, and when `sidea` is `BLIS_RIGHT`, `A` is _n x n_. The right-hand side matrix operand `B` is overwritten with the solution matrix `X`.

Observed object properties: `uplo(A)`, `trans?(A)`, `diag(A)`.

---


## Utility operations

---

#### asumv
```c
void bli_asumv
     (
       const obj_t*  x,
       const obj_t*  asum
     );
```
Compute the sum of the absolute values of the fundamental elements of vector `x`. The resulting sum is stored to `asum`.

Observed object properties: none.

**Note:** The floating-point type of `asum` is always the real projection of the floating-point type of `x`.
**Note:** This function attempts to mimic the algorithm for computing the absolute vector sum in the netlib BLAS routines `*asum()`.

---

#### norm1m
#### normfm
#### normim
```c
void bli_norm[1fi]m
     (
       const obj_t*  a,
       const obj_t*  norm
     );
```
Compute the one-norm (`bli_norm1m()`), Frobenius norm (`bli_normfm()`), or infinity norm (`bli_normim()`) of the elements in an _m x n_ matrix `A`. If `uplo(A)` is `BLIS_LOWER` or `BLIS_UPPER` then `A` is assumed to be lower or upper triangular, respectively, with the main diagonal located at offset `diagoff(A)`. The resulting norm is stored to `norm`.

Observed object properties: `diagoff(A)`, `diag(A)`, `uplo(A)`.

**Note:** The floating-point (`num_t`) type of `norm` is always the real projection of the floating-point type of `x`.

---

#### norm1v
#### normfv
#### normiv
```c
void bli_norm[1fi]v
     (
       const obj_t*  x,
       const obj_t*  norm
     );
```
Compute the one-norm (`bli_norm1v()`), Frobenius norm (`bli_normfv()`), or infinity norm (`bli_normiv()`) of the elements in a vector `x` of length _n_. The resulting norm is stored to `norm`.

Observed object properties: `diagoff(A)`, `diag(A)`, `uplo(A)`.

**Note:** The floating-point (`num_t`) type of `norm` is always the real projection of the floating-point type of `x`.

---

#### mkherm
```c
void bli_mkherm
     (
       const obj_t*  a
     );
```
Make an _m x m_ matrix `A` explicitly Hermitian by copying the conjugate of the triangle specified by `uplo(A)` to the opposite triangle. Imaginary components of diagonal elements are explicitly set to zero. It is assumed that the diagonal offset of `A` is zero.

Observed object properties: `uplo(A)`.

---

#### mksymm
```c
void bli_mksymm
     (
       const obj_t*  a
     );
```
Make an _m x m_ matrix `A` explicitly symmetric by copying the triangle specified by `uplo(A)` to the opposite triangle. It is assumed that the diagonal offset of `A` is zero.

Observed object properties: `uplo(A)`.

---

#### mktrim
```c
void bli_mktrim
     (
       const obj_t*  a
     );
```
Make an _m x m_ matrix `A` explicitly triangular by preserving the triangle specified by `uplo(A)` and zeroing the elements in the opposite triangle. It is assumed that the diagonal offset of `A` is zero.

Observed object properties: `uplo(A)`.

---

#### fprintv
```c
void bli_fprintv
     (
             FILE*   file,
       const char*   s1,
       const obj_t*  x,
       const char*   format,
       const char*   s2
     );
```
Print a vector `x` of length _m_ to file stream `file`, where `file` is a file pointer returned by the standard C library function `fopen()`. The caller may also pass in a global file pointer such as `stdout` or `stderr`. The strings `s1` and `s2` are printed immediately before and after the output (respectively), and the format specifier `format` is used to format the individual elements. For valid format specifiers, please see documentation for the standard C library function `printf()`.

**Note:** For complex datatypes, the format specifier is applied to both the real and imaginary components **individually**. Therefore, you should use format specifiers such as `"%5.2f"`, but **not** `"%5.2f + %5.2f"`.

---

#### fprintm
```c
void bli_fprintm
     (
             FILE*   file,
       const char*   s1,
       const obj_t*  a,
       const char*   format,
       const char*   s2
     );
```
Print an _m x n_ matrix `A` to file stream `file`, where `file` is a file pointer returned by the standard C library function `fopen()`. The caller may also pass in a global file pointer such as `stdout` or `stderr`. The strings `s1` and `s2` are printed immediately before and after the output (respectively), and the format specifier `format` is used to format the individual elements. For valid format specifiers, please see documentation for the standard C library function `printf()`.

**Note:** For complex datatypes, the format specifier is applied to both the real and imaginary components **individually**. Therefore, you should use format specifiers such as `"%5.2f"`, but **not** `"%5.2f + %5.2f"`.

---

#### printv
```c
void bli_printv
     (
       const char*   s1,
       const obj_t*  x,
       const char*   format,
       const char*   s2
     );
```
Print a vector `x` of length _m_ to standard output. This function call is equivalent to calling `bli_fprintv()` with `stdout` as the file pointer.

---

#### printm
```c
void bli_printm
     (
       const char*   s1,
       const obj_t*  a,
       const char*   format,
       const char*   s2
     );
```
Print an _m x n_ matrix `a` to standard output. This function call is equivalent to calling `bli_fprintm()` with `stdout` as the file pointer.

---

#### randv
```c
void bli_randv
     (
       const obj_t*  x
     );
```
Set the elements of a vector `x` of length _n_ to random values on the interval `[-1,1)`.

**Note:** For complex datatypes, the real and imaginary components of each element are randomized individually and independently of one another.

---

#### randm
```c
void bli_randm
     (
       const obj_t*  a
     );
```
Set the elements of an _m x n_ matrix `A` to random values on the interval `[-1,1)`. Off-diagonal elements (in the triangle specified by `uplo(A)`) are scaled by `1.0/max(m,n)`.

Observed object properties: `diagoff(A)`, `uplo(A)`.

**Note:** For complex datatypes, the real and imaginary components of each off-diagonal element are randomized individually and independently of one another.
**Note:** If `uplo(A)` is `BLIS_LOWER` or `BLIS_UPPER` and you plan to use this matrix to test `trsv` or `trsm`, additional scaling of the diagonal is recommended to ensure that the matrix is invertible. In this case, try using the [addd](BLISObjectAPI.md#addd) operation to increase the magnitude to the diagonal elements.

---

#### sumsqv
```c
void bli_sumsqv
     (
       const obj_t*  x,
       const obj_t*  scale,
       const obj_t*  sumsq
     );
```
Compute the sum of the squares of the elements in a vector `x` of length _n_. The result is computed in scaled form, and in such a way that it may be used repeatedly to accumulate the sum of the squares of several vectors.

The function computes scale\_new and sumsq\_new such that
```
  scale_new^2 * sumsq_new = x[0]^2 + x[1]^2 + ... x[m-1]^2 + scale_old^2 * sumsq_old
```
where, on entry, `scale` and `sumsq` contain `scale_old` and `sumsq_old`, respectively, and on exit, `scale` and `sumsq` contain `scale_new` and `sumsq_new`, respectively.

**Note:** This function attempts to mimic the algorithm for computing the Frobenius norm in the netlib LAPACK routine `?lassq()`.
**Note:** The floating-point (`num_t`) types of `scale` and `sumsq` are always the real projection of the floating-point type of `x`.

---

#### getsc
```c
void bli_getsc
     (
       const obj_t*   chi,
             double*  zeta_r,
             double*  zeta_i
     );
```
Copy the real and imaginary values from the scalar object `chi` to `zeta_r` and `zeta_i`. If `chi` is stored as a real type, then `zeta_i` is set to zero. (If `chi` is stored in single precision, the corresponding elements are typecast/promoted during the copy.)

---

#### getijv
```c
err_t bli_getijv
     (
             dim_t    i,
       const obj_t*   x,
             double*  ar,
             double*  ai
     );
```
Copy the real and imaginary values at the `i`th element of vector object `x` to `ar` and `ai`. If elements of `x` are stored as real types, then only `ar` is overwritten and `ai` is left unchanged. (If `x` contains elements stored in single precision, the corresponding elements are typecast/promoted during the copy.)
If either the element offset `i` is beyond the vector dimension of `x` or less than zero, the function returns `BLIS_FAILURE` without taking any action. Similarly, if `x` is a global scalar constant such as `BLIS_ONE`, the function returns `BLIS_FAILURE`.

---

#### getijm
```c
err_t bli_getijm
     (
             dim_t    i,
             dim_t    j,
       const obj_t*   b,
             double*  ar,
             double*  ai
     );
```
Copy the real and imaginary values at the (`i`,`j`) element of object `b` to `ar` and `ai`. If elements of `b` are stored as real types, then only `ar` is overwritten and `ai` is left unchanged. (If `b` contains elements stored in single precision, the corresponding elements are typecast/promoted during the copy.)
If either the row offset `i` is beyond the _m_ dimension of `b` or less than zero, or column offset `j` is beyond the _n_ dimension of `b` or less than zero, the function returns `BLIS_FAILURE` without taking any action. Similarly, if `b` is a global scalar constant such as `BLIS_ONE`, the function returns `BLIS_FAILURE`.

---

#### setsc
```c
void bli_setsc
     (
             double  zeta_r,
             double  zeta_i,
       const obj_t*  chi
     );
```
Copy real and imaginary values `zeta_r` and `zeta_i` to the scalar object `chi`. If `chi` is stored as a real type, then `zeta_i` is ignored. (If `chi` is stored in single precision, the contents are typecast/demoted during the copy.)

---

#### setijv
```c
err_t bli_setijv
     (
             double  ar,
             double  ai,
             dim_t   i,
       const obj_t*  x
     );
```
Copy real and imaginary values `ar` and `ai` to the `i`th element of vector object `x`. If elements of `x` are stored as real types, then only `ar` is copied and `ai` is ignored. (If `x` contains elements stored in single precision, the corresponding elements are typecast/demoted during the copy.)
If the element offset `i` is beyond the vector dimension of `x` or less than zero, the function returns `BLIS_FAILURE` without taking any action. Similarly, if `x` is a global scalar constant such as `BLIS_ONE`, the function returns `BLIS_FAILURE`.

---

#### setijm
```c
err_t bli_setijm
     (
             double  ar,
             double  ai,
             dim_t   i,
             dim_t   j,
       const obj_t*  b
     );
```
Copy real and imaginary values `ar` and `ai` to the (`i`,`j`) element of object `b`. If elements of `b` are stored as real types, then only `ar` is copied and `ai` is ignored. (If `b` contains elements stored in single precision, the corresponding elements are typecast/demoted during the copy.)
If either the row offset `i` is beyond the _m_ dimension of `b` or less than zero, or column offset `j` is beyond the _n_ dimension of `b` or less than zero, the function returns `BLIS_FAILURE` without taking any action. Similarly, if `b` is a global scalar constant such as `BLIS_ONE`, the function returns `BLIS_FAILURE`.

---

#### eqsc
```c
void bli_eqsc
     (
       const obj_t*  chi,
       const obj_t*  psi,
             bool*   is_eq
     );
```
Perform an element-wise comparison between scalars `chi` and `psi` and store the boolean result in the `bool` pointed to by `is_eq`.
If exactly one of `conj(chi)` or `conj(psi)` (but not both) indicate a conjugation, then one of the scalars will be implicitly conjugated for purposes of the comparision.

Observed object properties: `conj?(chi)`, `conj?(psi)`.

---

#### eqv
```c
void bli_eqv
     (
       const obj_t*  x,
       const obj_t*  y,
             bool*   is_eq
     );
```
Perform an element-wise comparison between vectors `x` and `y` and store the boolean result in the `bool` pointed to by `is_eq`.
If exactly one of `conj(x)` or `conj(y)` (but not both) indicate a conjugation, then one of the vectors will be implicitly conjugated for purposes of the comparision.

Observed object properties: `conj?(x)`, `conj?(y)`.

---

#### eqm
```c
void bli_eqm
     (
       const obj_t*  a,
       const obj_t*  b,
             bool*   is_eq
     );
```
Perform an element-wise comparison between matrices `A` and `B` and store the boolean result in the `bool` pointed to by `is_eq`.
Here, `A` is stored as a dense matrix, or lower- or upper-triangular/trapezoidal matrix with arbitrary diagonal offset and unit or non-unit diagonal.
If `diag(A)` indicates a unit diagonal, the diagonals of both matrices will be ignored for purposes of the comparision.
If `uplo(A)` indicates lower or upper storage, only that part of both matrices `A` and `B` will be referenced.
If exactly one of `trans(A)` or `trans(B)` (but not both) indicate a transposition, then one of the matrices will be transposed for purposes of the comparison.
Similarly, if exactly one of `trans(A)` or `trans(B)` (but not both) indicate a conjugation, then one of the matrices will be implicitly conjugated for purposes of the comparision.

Observed object properties: `diagoff(A)`, `diag(A)`, `uplo(A)`, `trans?(A)`, `trans?(B)`.



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

BLIS provides lots of example code in the [examples/oapi](https://github.com/flame/blis/tree/master/examples/oapi) directory of the BLIS source distribution. The example code in this directory is set up like a tutorial, and so we recommend starting from the beginning. Topics include creating and managing objects, printing vectors and matrices, setting and querying object properties, and calling a representative subset of the computational level-1v, -1m, -2, -3, and utility operations documented above. Please read the `README` contained within the `examples/oapi` directory for further details.

