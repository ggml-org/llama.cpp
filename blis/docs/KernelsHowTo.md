## Contents

* **[Contents](KernelsHowTo.md#contents)**
* **[Introduction](KernelsHowTo.md#introduction)**
* **[BLIS kernels summary](KernelsHowTo.md#blis-kernels-summary)**
  * [Level-3](KernelsHowTo.md#level-3)
  * [Level-1f](KernelsHowTo.md#level-1f)
  * [Level-1v](KernelsHowTo.md#level-1v)
  * [Level-1v/-1f Dependencies for Level-2 operations](KernelsHowTo.md#level-1v-1f-dependencies-for-level-2-operations)
* **[Calling kernels](KernelsHowTo.md#calling-kernels)**
* **[BLIS kernels reference](KernelsHowTo.md#blis-kernels-reference)**
  * [Level-3 microkernels](KernelsHowTo.md#level-3-microkernels)
  * [Level-1f kernels](KernelsHowTo.md#level-1f-kernels)
  * [Level-1v kernels](KernelsHowTo.md#level-1v-kernels)


## Introduction

This wiki describes the computational kernels used by the BLIS framework.

One of the primary features of BLIS is that it provides a large set of dense linear algebra functionality while simultaneously minimizing the amount of kernel code that must be optimized for a given architecture. BLIS does this by isolating a handful of kernels which, when implemented, facilitate functionality and performance of several of the higher-level operations.

Presently, BLIS supports several groups of operations:
  * **[Level-1v](BLISTypedAPI.md#level-1v-operations)**: Operations on vectors:
    * [addv](BLISTypedAPI.md#addv), [amaxv](BLISTypedAPI.md#amaxv), [axpyv](BLISTypedAPI.md#axpyv), [copyv](BLISTypedAPI.md#copyv), [dotv](BLISTypedAPI.md#dotv), [dotxv](BLISTypedAPI.md#dotxv), [invertv](BLISTypedAPI.md#invertv), [invscalv](BLISTypedAPI.md#invscalv), [scalv](BLISTypedAPI.md#scalv), [scal2v](BLISTypedAPI.md#scal2v), [setv](BLISTypedAPI.md#setv), [subv](BLISTypedAPI.md#subv), [swapv](BLISTypedAPI.md#swapv)
  * **[Level-1d](BLISTypedAPI.md#level-1d-operations)**: Element-wise operations on matrix diagonals:
    * [addd](BLISTypedAPI.md#addd), [axpyd](BLISTypedAPI.md#axpyd), [copyd](BLISTypedAPI.md#copyd), [invertd](BLISTypedAPI.md#invertd), [invscald](BLISTypedAPI.md#invscald), [scald](BLISTypedAPI.md#scald), [scal2d](BLISTypedAPI.md#scal2d), [setd](BLISTypedAPI.md#setd), [setid](BLISTypedAPI.md#setid), [subd](BLISTypedAPI.md#subd)
  * **[Level-1m](BLISTypedAPI.md#level-1m-operations)**: Element-wise operations on matrices:
    * [addm](BLISTypedAPI.md#addm), [axpym](BLISTypedAPI.md#axpym), [copym](BLISTypedAPI.md#copym), [invscalm](BLISTypedAPI.md#invscalm), [scalm](BLISTypedAPI.md#scalm), [scal2m](BLISTypedAPI.md#scal2m), [setm](BLISTypedAPI.md#setm), [subm](BLISTypedAPI.md#subm)
  * **[Level-1f](BLISTypedAPI.md#level-1f-operations)**: Fused operations on multiple vectors:
    * [axpy2v](BLISTypedAPI.md#axpy2v), [dotaxpyv](BLISTypedAPI.md#dotaxpyv), [axpyf](BLISTypedAPI.md#axpyf), [dotxf](BLISTypedAPI.md#dotxf), [dotxaxpyf](BLISTypedAPI.md#dotxaxpyf)
  * **[Level-2](BLISTypedAPI.md#level-2-operations)**: Operations with one matrix and (at least) one vector operand:
    * [gemv](BLISTypedAPI.md#gemv), [ger](BLISTypedAPI.md#ger), [hemv](BLISTypedAPI.md#hemv), [her](BLISTypedAPI.md#her), [her2](BLISTypedAPI.md#her2), [symv](BLISTypedAPI.md#symv), [syr](BLISTypedAPI.md#syr), [syr2](BLISTypedAPI.md#syr2), [trmv](BLISTypedAPI.md#trmv), [trsv](BLISTypedAPI.md#trsv)
  * **[Level-3](BLISTypedAPI.md#level-3-operations)**: Operations with matrices that are multiplication-like:
    * [gemm](BLISTypedAPI.md#gemm), [hemm](BLISTypedAPI.md#hemm), [herk](BLISTypedAPI.md#herk), [her2k](BLISTypedAPI.md#her2k), [symm](BLISTypedAPI.md#symm), [syrk](BLISTypedAPI.md#syrk), [syr2k](BLISTypedAPI.md#syr2k), [trmm](BLISTypedAPI.md#trmm), [trmm3](BLISTypedAPI.md#trmm3), [trsm](BLISTypedAPI.md#trsm)
  * **[Utility](BLISTypedAPI.md#Utility-operations)**: Miscellaneous operations on matrices and vectors:
    * [asumv](BLISTypedAPI.md#asumv), [norm1v](BLISTypedAPI.md#norm1v), [normfv](BLISTypedAPI.md#normfv), [normiv](BLISTypedAPI.md#normiv), [norm1m](BLISTypedAPI.md#norm1m), [normfm](BLISTypedAPI.md#normfm), [normim](BLISTypedAPI.md#normim), [mkherm](BLISTypedAPI.md#mkherm), [mksymm](BLISTypedAPI.md#mksymm), [mktrim](BLISTypedAPI.md#mktrim), [fprintv](BLISTypedAPI.md#fprintv), [fprintm](BLISTypedAPI.md#fprintm),[printv](BLISTypedAPI.md#printv), [printm](BLISTypedAPI.md#printm), [randv](BLISTypedAPI.md#randv), [randm](BLISTypedAPI.md#randm), [sumsqv](BLISTypedAPI.md#sumsqv)

Most of the interest with BLAS libraries centers around level-3 operations because they exhibit favorable ratios of floating-point operations (flops) to memory operations (memops), which allows high performance. Some applications also require level-2 computation; however, these operations are at an inherent disadvantage on modern architectures due to their less favorable flop-to-memop ratio. The BLIS framework allows developers to quickly and easily build high performance level-3 operations, as well as relatively well-performing level-2 operations, simply by optimizing a small set of kernels. These kernels, and their relationship to the other higher-level operations supported by BLIS, are the subject of this wiki.

Some level-1v, level-1m, and level-1d operations may also be accelerated, but since they are memory-bound, optimization typically yields minor performance improvement.


---


## BLIS kernels summary

This section lists and briefly describes each of the main computational kernels supported by the BLIS framework. (Other kernels are supported, but they are not of interest to most developers.)

### Level-3

BLIS supports the following three level-3 microkernels. These microkernels are used to implement optimized level-3 operations.
  * **gemm**: The `gemm` microkernel performs a small matrix multiplication and is used by every level-3 operation.
  * **trsm**: The `trsm` microkernel performs a small triangular solve with multiple right-hand sides. It is not required for optimal performance and in fact is only needed when the developer opts to not implement the fused `gemmtrsm` kernel.
  * **gemmtrsm**: The `gemmtrsm` microkernel implements a fused operation whereby a `gemm` and a `trsm` subproblem are fused together in a single routine. This avoids redundant memory operations that would otherwise be incurred if the operations were executed separately.

The following shows the steps one would take to optimize, to varying degrees, the level-3 operations supported by BLIS:
  1. By implementing and optimizing the `gemm` microkernel, **all** level-3 operations **except** `trsm` are fully optimized. In this scenario, the `trsm` operation may achieve 60-90% of attainable peak performance, depending on the architecture and problem size.
  1. If one goes further and implements and optimizes the `trsm` microkernel, this kernel, when paired with an optimized `gemm` microkernel, results in a `trsm` implementation that is accelerated (but not optimized).
  1. Alternatively, if one implements and optimizes the fused `gemmtrsm` microkernel, this kernel, when paired with an optimized `gemm` microkernel, enables a fully optimized `trsm` implementation.

### Level-1f

BLIS supports the following five level-1f (fused) kernels. These kernels are used to implement optimized level-2 operations (as well as self-similar level-1f operations; that is, the `axpyf` kernel can be invoked indirectly via the `axpyf` operation).
  * **axpy2v**: Performs and fuses two [axpyv](BLISTypedAPI.md#axpyv) operations, accumulating to the same output vector.
  * **dotaxpyv**: Performs and fuses a [dotv](BLISTypedAPI.md#dotv) followed by an [axpyv](BLISTypedAPI.md#axpyv) operation with x.
  * **axpyf**: Performs and fuses some implementation-dependent number of [axpyv](BLISTypedAPI.md#axpyv) operations, accumulating to the same output vector. Can also be expressed as a [gemv](BLISTypedAPI.md#gemv) operation where matrix A is _m x nf_, where nf is the number of fused operations (fusing factor).
  * **dotxf**: Performs and fuses some implementation-dependent number of [dotxv](BLISTypedAPI.md#dotxv) operations, reusing the `y` vector for each [dotxv](BLISTypedAPI.md#dotxv).
  * **dotxaxpyf**: Performs and fuses a [dotxf](BLISTypedAPI.md#dotxf) and [axpyf](BLISTypedAPI.md#axpyf) in which the matrix operand is reused.


### Level-1v

BLIS supports the following 14 level-1v kernels. These kernels are used primarily to implement their self-similar operations. However, they are occasionally used to handle special cases of level-1f kernels or in situations where level-2 operations are partially optimized.
  * **addv**: Performs a [vector addition](BLISTypedAPI.md#addv) operation.
  * **amaxv**: Performs a [search for the index of the element with the largest absolute value (or complex modulus)](BLISTypedAPI.md#amaxv).
  * **axpyv**: Performs a [vector scale-and-accumulate](BLISTypedAPI.md#axpyv) operation.
  * **axpbyv**: Performs an [extended vector scale-and-accumulate](BLISTypedAPI.md#axpbyv) operation similar to axpyv except that the output vector is scaled by a second scalar.
  * **copyv**: Performs a [vector copy](BLISTypedAPI.md#copyv) operation
  * **dotv**: Performs a [dot product](BLISTypedAPI.md#dotv) where the output scalar is overwritten.
  * **dotxv**: Performs an [extended dot product](BLISTypedAPI.md#dotxv) operation where the dot product is first scaled and then accumulated into a scaled output scalar.
  * **invertv**: Performs an [element-wise vector inversion](BLISTypedAPI.md#invertv) operation.
  * **invscalv**: Performs an [in-place (destructive) vector inverse-scaling](BLISTypedAPI.md#invscalv) operation.
  * **scalv**: Performs an [in-place (destructive) vector scaling](BLISTypedAPI.md#scalv) operation.
  * **scal2v**: Performs an [out-of-place (non-destructive) vector scaling](BLISTypedAPI.md#scal2v) operation.
  * **setv**: Performs a [vector broadcast](BLISTypedAPI.md#setv) operation.
  * **subv**: Performs a [vector subtraction](BLISTypedAPI.md#subv) operation.
  * **swapv**: Performs a [vector swap](BLISTypedAPI.md#swapv) operation.
  * **xpbyv**: Performs a [alternate vector scale-and-accumulate](BLISTypedAPI.md#xpbyv) operation.


### Level-1v/-1f Dependencies for Level-2 operations

The table below shows dependencies between level-2 operations and each of the level-1v and level-1f kernels.

Kernels marked with a "1" for a given level-2 operation are preferred for optimization because they facilitate an optimal implementation on most architectures. Kernels marked with a "2", "3", or "4" denote those which need to be optimized for alternative implementations that would typically be second, third, or fourth choices, respectively, if the preferred kernels are not optimized.

| operation / kernel | effective storage   | `axpyv` | `dotxv` | `axpy2v` | `dotaxpyv` | `axpyf` | `dotxf` | `dotxaxpyf` |
|:-------------------|:--------------------|:--------|:--------|:---------|:-----------|:--------|:--------|:------------|
| `gemv, trmv, trsv` | row-wise            |         |   2     |          |            |         |   1     |             |
|                    | column-wise         |   2     |         |          |            |   1     |         |             |
| `hemv, symv`       | row- or column-wise |   4     |   4     |          |    3       |   2     |   2     |     1       |
| `ger, her, syr`    | row- or column-wise |   1     |         |          |            |         |         |             |
| `her2, syr2`       | row- or column-wise |   2     |         |    1     |            |         |         |             |

**Note:** The "effective storage" column reflects the orientation of the matrix operand **after** transposition via the corresponding `trans_t` parameter (if applicable). For example, calling `gemv` with a column-stored matrix `A` and the `transa` parameter equal to `BLIS_TRANSPOSE` would be effectively equivalent to row-wise storage.

---

## Calling kernels

Note that all kernels, whether they be reference implementations or based on fully optimized assembly code, use names that are architecture- and implementation-specific. (This appears as a `<suffix>` in the [kernel reference](KernelsHowTo.md#blis-kernels-reference) below.) Therefore, the easiest way to call the kernel is by querying a pointer from a valid context.

The first step is to obtain a valid context. Contexts store all of the information
specific to a particular sub-configuration (usually loosely specific to a
microarchitecture or group of closely-related microarchitectures). If a context is
not already available in your current scope, a default context for the hardware
for which BLIS was configured (or, in the case of multi-configuration builds, the
hardware on which BLIS is currently running) may be queried via:
```c
cntx_t* bli_gks_query_cntx( void );
```
Once this `cntx_t*` pointer is obtained, you may call one of three functions to query any of the computation kernels described in this document:
```c
void* bli_cntx_get_l3_nat_ukr_dt
     (
       num_t   dt,
       l3ukr_t ker_id,
       cntx_t* cntx
     );

void* bli_cntx_get_l1f_ker_dt
     (
       num_t   dt,
       l1fkr_t ker_id,
       cntx_t* cntx
     );

void* bli_cntx_get_l1v_ker_dt
     (
       num_t   dt,
       l1vkr_t ker_id,
       cntx_t* cntx
     );
```
The `dt` and `ker_id` parameters specify the floating-point datatype and the
kernel operation you wish to query, respectively.
Valid values for `dt` are `BLIS_FLOAT`, `BLIS_DOUBLE`, `BLIS_SCOMPLEX`, and
`BLIS_DCOMPLEX` for single- and double-precision real, and single- and
double-precision complex, respectively.
Valid values for `ker_id` are given in the tables below.

Also, note that the return values of `bli_cntx_get_l1v_ker_dt`
`bli_cntx_get_l1f_ker_dt()`, and `bli_cntx_get_l3_nat_ukr_dt()`,
will be `void*` and must be typecast to typed function pointers before being called.
As a convenience, BLIS defines function pointer types appropriate for usage in these
situations. The function pointer type for each operation is given in the third
columns of each table, with the `?` taking the place of one of the supported
datatype characters.

| kernel operation |  l3ukr_t              | function pointer type |
|:-----------------|:----------------------|:----------------------|
| gemm             | `BLIS_GEMM`           | `?gemm_ukr_ft`        |
| trsm_l           | `BLIS_TRSM_L_UKR`     | `?trsm_ukr_ft`        |
| trsm_u           | `BLIS_TRSM_U_UKR`     | `?trsm_ukr_ft`        |
| gemmtrsm_l       | `BLIS_GEMMTRSM_L_UKR` | `?gemmtrsm_ukr_ft`    |
| gemmtrsm_u       | `BLIS_GEMMTRSM_U_UKR` | `?gemmtrsm_ukr_ft`    |

| kernel operation |  l1fkr_t              | function pointer type |
|:-----------------|:----------------------|:----------------------|
| axpy2v           | `BLIS_AXPY2V_KER`     | `?axpy2v_ft`          |
| dotaxpyv         | `BLIS_DOTAXPYV_KER`   | `?dotaxpyv_ft`        |
| axpyf            | `BLIS_AXPYF_KER`      | `?axpyf_ft`           |
| dotxf            | `BLIS_DOTXF_KER`      | `?dotxf_ft`           |
| dotxaxpyf        | `BLIS_DOTXAXPYF_KER`  | `?dotxaxpyf_ft`       |

| kernel operation |  l1vkr_t              | function pointer type |
|:-----------------|:----------------------|:----------------------|
| addv             | `BLIS_ADDV_KER`       | `?addv_ft`            |
| amaxv            | `BLIS_AMAXV_KER`      | `?amaxv_ft`           |
| axpyv            | `BLIS_AXPYV_KER`      | `?axpyv_ft`           |
| axpbyv           | `BLIS_AXPBYV_KER`     | `?axpbyv_ft`          |
| dotaxpyv         | `BLIS_DOTAXPYV_KER`   | `?dotaxpyv_ft`        |
| copyv            | `BLIS_COPYV_KER`      | `?copyv_ft`           |
| dotxv            | `BLIS_DOTXV_KER`      | `?dotxv_ft`           |
| invertv          | `BLIS_INVERTV_KER`    | `?invertv_ft`         |
| invscalv         | `BLIS_INVSCALV_KER`   | `?invscalv_ft`        |
| scalv            | `BLIS_SCALV_KER`      | `?scalv_ft`           |
| scal2v           | `BLIS_SCAL2V_KER`     | `?scal2v_ft`          |
| setv             | `BLIS_SETV_KER`       | `?setv_ft`            |
| subv             | `BLIS_SUBV_KER`       | `?subv_ft`            |
| swapv            | `BLIS_SWAPV_KER`      | `?swapv_ft`           |
| xpybv            | `BLIS_XPBYV_KER`      | `?xpbyv_ft`           |

The specific information behind a queried function pointer is not typically available.
However, it is guaranteed that the function pointer will always be valid (usually either an optimized assembly implementation or a reference implementation).


---


## BLIS kernels reference

This section seeks to provide developers with a complete reference for each of the following BLIS kernels, including function prototypes, parameter descriptions, implementation notes, and diagrams:
  * [Level-3 microkernels](KernelsHowTo.md#level-3-microkernels)
    * [gemm](KernelsHowTo.md#gemm-microkernel)
    * [trsm](KernelsHowTo.md#trsm-microkernels)
    * [gemmtrsm](KernelsHowTo.md#gemmtrsm-microkernels)
  * [Level-1f kernels](KernelsHowTo.md#level-1f-kernels)
    * [axpy2v](KernelsHowTo.md#axpy2v-kernel)
    * [dotaxpyv](KernelsHowTo.md#dotaxpyv-kernel)
    * [axpyf](KernelsHowTo.md#axpyf-kernel)
    * [dotxf](KernelsHowTo.md#dotxf-kernel)
    * [dotxaxpyf](KernelsHowTo.md#dotxaxpyf-kernel)
  * [Level-1v kernels](KernelsHowTo.md#level-1v-kernels)
    * [addv](KernelsHowTo.md#addv-kernel)
    * [amaxv](KernelsHowTo.md#amaxv-kernel)
    * [axpyv](KernelsHowTo.md#axpyv-kernel)
    * [axpbyv](KernelsHowTo.md#axpbyv-kernel)
    * [copyv](KernelsHowTo.md#copyv-kernel)
    * [dotv](KernelsHowTo.md#dotv-kernel)
    * [dotxv](KernelsHowTo.md#dotxv-kernel)
    * [invertv](KernelsHowTo.md#invertv-kernel)
    * [invscalv](KernelsHowTo.md#invscalv-kernel)
    * [scalv](KernelsHowTo.md#scalv-kernel)
    * [scal2v](KernelsHowTo.md#scal2v-kernel)
    * [setv](KernelsHowTo.md#setv-kernel)
    * [subv](KernelsHowTo.md#subv-kernel)
    * [swapv](KernelsHowTo.md#swapv-kernel)
    * [xpbyv](KernelsHowTo.md#xpbyv-kernel)

The function prototypes in this section follow the same guidelines as those listed in the [BLIS typed API reference](BLISTypedAPI.md#Notes_for_using_this_reference). Namely:
  * Any occurrence of `?` should be replaced with `s`, `d`, `c`, or `z` to form an actual function name.
  * Any occurrence of `ctype` should be replaced with the actual C99 language type corresponding to the datatype instance in question.
  * Some matrix arguments have associated row and column strides arguments that proceed them, typically listed as `rsX` and `csX` for a given matrix `X`. Row strides are always listed first, and column strides are always listed second. The semantic meaning of a row stride is "the distance, in units of elements, from any given element to the corresponding element (within the same column) of the next row," and the meaning of a column stride is "the distance, in units of elements, from any given element to the corresponding element (within the same row) of the next column." Thus, unit row stride implies column-major storage and unit column stride implies row-major storage.
  * All occurrences of `alpha` and `beta` parameters are scalars.



### Level-3 microkernels

This section describes in detail the various level-3 microkernels supported by BLIS:
  * [gemm](KernelsHowTo.md#gemm-microkernel)
  * [trsm](KernelsHowTo.md#trsm-microkernels)
  * [gemmtrsm](KernelsHowTo.md#gemmtrsm-microkernels)


#### gemm microkernel

```c
void bli_?gemm_<suffix>
     (
       dim_t               m,
       dim_t               n,
       dim_t               k,
       ctype*     restrict alpha,
       ctype*     restrict a1,
       ctype*     restrict b1,
       ctype*     restrict beta,
       ctype*     restrict c11, inc_t rsc, inc_t csc,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     );
```

where `<suffix>` is implementation-dependent. (Recall that the precise `<suffix>` associated with the microkernel along with the rest of the function name doesn't matter if you are querying the function address from the context. See section on [calling kernels](KernelsHowTo.md#calling-kernels) for details.) The following (more portable) wrapper is also defined:

```c
void bli_?gemm_ukernel
     (
       dim_t               m,
       dim_t               n,
       dim_t               k,
       ctype*     restrict alpha,
       ctype*     restrict a1,
       ctype*     restrict b1,
       ctype*     restrict beta,
       ctype*     restrict c11, inc_t rsc, inc_t csc,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     );
```
This function simply queries a microkernel function pointer from the context specified by `cntx`. Note that in the case of either method of calling the microkernel, `cntx` must be a valid pointer. (Passing in `NULL` will *not* result in a default context being used.)

The `gemm` microkernel, sometimes simply referred to as "the BLIS microkernel" or "the microkernel", performs the following operation:

```
  C11 := beta * C11 + alpha * A1 * B1
```

where `A1` is an _m x k_ "micropanel" matrix stored in packed (column-wise) format, `B1` is a _k x n_ "micropanel" matrix stored in packed (row-wise) format, `C11` is an _m x n_ "microtile" matrix stored according to its row and column strides `rsc` and `csc`, and `alpha` and beta are scalars.

Here, _m <= MR_ and _n <= NR_, where _MR_ and _NR_ are the register blocksizes associated with the microkernel. They are chosen by the developer when the microkernel is written and then encoded into a BLIS configuration, which will reference the microkernel when the BLIS framework is instantiated into a library. For more information on setting register blocksizes and related constants, please see the [BLIS developer configuration guide](ConfigurationHowTo.md).

**Note:** For many years, BLIS defined its microkernel to operate on microtiles whose dimensions were *exactly* _MR x NR_. However, as of commit 54fa28b, we have augmented the `gemm` microkernel API to pass in _m_ and _n_ dimensions as well as _k_. This change was made as part of our decision to move edge-case handling into the microkernel, whereas previously it was handled outside of the microkernel, within the portable parts of BLIS framework. And while this does mean additional complexity for microkernel authors, adding generic edge-case handling can be done in a relatively painless manner by employing some pre-defined preprocessor macros (which are defined in `bli_edge_case_macro_defs.h`). For examples of how to use these macros, please see the beginning and end of existing microkernel functions residing within the `kernels` directory.

Parameters:

  * `m`:      The number of rows of `C11` and `A1`.
  * `n`:      The number of columns of `C11` and `B1`.
  * `k`:      The number of columns of `A1` and rows of `B1`.
  * `alpha`:  The address of a scalar to the `A1 * B1` product.
  * `a1`:     The address of a micropanel of matrix `A` of dimension _m x k_ (where _m <= MR_), stored by columns with leading dimension _PACKMR_, where typically _PACKMR_ = _MR_. (See [Implementation Notes for gemm](KernelsHowTo.md#implementation-notes-for-gemm) for a discussion of _PACKMR_.)
  * `b1`:     The address of a micropanel of matrix `B` of dimension _k x n_ (where _n <= NR_), stored by rows with leading dimension _PACKNR_, where typically _PACKNR_ = _NR_. (See [Implementation Notes for gemm](KernelsHowTo.md#implementation-notes-for-gemm) for a discussion of _PACKNR_.)
  * `beta`:   The address of a scalar to the input value of matrix `C11`.
  * `c11`:    The address of a matrix `C11` of dimension _MR x NR_, stored according to `rsc` and `csc`.
  * `rsc`:    The row stride of matrix `C11` (ie: the distance to the next row, in units of matrix elements).
  * `csc`:    The column stride of matrix `C11` (ie: the distance to the next column, in units of matrix elements).
  * `data`:   The address of an `auxinfo_t` object that contains auxiliary information that may be useful when optimizing the `gemm` microkernel implementation. (See [Using the auxinfo\_t object](KernelsHowTo.md#Using_the_auxinfo_t_object) for a discussion of the kinds of values available via `auxinfo_t`.)
  * `cntx`:   The address of the runtime context. The context can be queried for implementation-specific values such as cache and register blocksizes. However, most microkernels intrinsically "know" these values already, and thus the `cntx` argument usually can be safely ignored.

#### Diagram for gemm

The diagram below shows the packed micropanel operands and how elements of each would be stored when _MR_ = _NR_ = 4. The hex digits indicate the layout and order (but NOT the numeric contents) of the elements in memory. Note that the storage of `C11` is not shown since it is determined by the row and column strides of `C11`.

```
         c11:           a1:                        b1:
         _______        ______________________     _______
        |       |      |0 4 8 C               |   |0 1 2 3|
    MR  |       |      |1 5 9 D . . .         |   |4 5 6 7|
        |       |  +=  |2 6 A E               |   |8 9 A B|
        |_______|      |3_7_B_F_______________|   |C D E F|
                                                  |   .   |
            NR                    k               |   .   | k
                                                  |   .   |
                                                  |       |
                                                  |       |
                                                  |_______|

                                                      NR
```

#### Implementation Notes for gemm

  * **Register blocksizes.** The register blocksizes `MR` and `NR`, corresponding to the maximum number of *logical* rows in `a1` and columns in `b1`, respectively, are defined in the context and may be queried via `bli_cntx_get_blksz_def_dt()`. However, you shouldn't need to query these values since the implementation inherently "knows" them already.
  * **Leading dimensions of `a1` and `b1`: _PACKMR_ and _PACKNR_.** The packed micropanels `a1` and `b1` are simply stored in column-major and row-major order, respectively. Usually, the width of either micropanel (ie: the number of *logical* rows of `a1` and the number of columns of `b1`) is equal to that micropanel's so-called "leading dimension", or number of *physical* rows. Sometimes, it may be beneficial to specify a leading dimension that is larger than the panel width. This may be desirable because it allows each column of `a1` or row of `b1` to maintain a certain alignment in memory that would not otherwise be maintained by _MR_ and/or _NR_, which would othewise serve as the maximum value for each micropanel, respectively. If you want your microkernel to support _MR < PACKMR_ or _NR < PACKNR_, you should index through columns of `a1` and rows of `b1` using the values _PACKMR_ and _PACKNR_, respectively (which are stored in the context as the blocksize "maximums" associated with the `bszid_t` values `BLIS_MR` and `BLIS_NR`). These values are defined in the context and may be queried via `bli_cntx_get_blksz_max_dt()`. However, you shouldn't need to query these values since the microkernel implementation inherently must "know" them already.
  * **Storage preference of `c11`.** Usually, an optimized `gemm` microkernel will have a "preferred" storage format for `C11`--typically either contiguous row-storage (i.e. `cs_c` = 1) or contiguous column-storage (i.e. `rs_c` = 1). This preference comes from how the microkernel is most efficiently able to load/store elements of `C11` from/to memory. Most microkernels use vector instructions to access contiguous columns (or column segments) of `C11`. However, the developer may decide that accessing contiguous rows (or row segments) is more desirable. If this is the case, this preference should be indicated via the `bool` argument when registering microkernels via `bli_cntx_set_l3_nat_ukrs()`--`TRUE` indicating a row preference and `FALSE` indicating a column preference. Properly setting this property allows the framework to perform a runtime optimization that will ensure the microkernel preference is honored, if at all possible.
  * **Edge cases in _MR_, _NR_ dimensions.** Sometimes the microkernel will be called with micropanels `a1` and `b1` that correspond to edge cases, where only partial results are needed. This edge-case handling was once performed by the framework automatically. However, as of commit 54fa28b, edge-case handling is the responsiblity of the microkernel. This means that the kernel author will need to handle all possible values of _m_ and _n_ that are equal to **or** less than _MR_ and _NR_, respectively. Fortunately, this can be implemented outside of the assembly region of the microkernel with preprocessor macros. Please reference the existing microkernels in the `kernels` directory for examples of how this is done. (The macros that are now employed by most of BLIS's microkernels are defined in `bli_edge_case_macro_defs.h`.)
  * **Alignment of `a1` and `b1`.** By default, the alignment of addresses `a1` and `b1` are aligned to the page size (4096 bytes). These alignment factors are set by `BLIS_POOL_ADDR_ALIGN_SIZE_A` and `BLIS_POOL_ADDR_ALIGN_SIZE_B`, respectively. Note that these alignment factors control only the alignment of the *first* micropanel within a given packed blockof matrix `A` or packed row-panel of matrix `B`. Subsequent micropanels will only be aligned to `sizeof(type)`, or, if `BLIS_POOL_ADDR_ALIGN_SIZE_A` is a multiple of `PACKMR` and/or `BLIS_POOL_ADDR_ALIGN_SIZE_B` is a multiple of `PACKNR`, then subsequent micropanels `a1` and/or `b1` will be aligned to `PACKMR * sizeof(type)` and/or `PACKNR * sizeof(type)`, respectively.
  * **Unrolling loops.** As a general rule of thumb, the loop over _k_ is sometimes moderately unrolled; for example, in our experience, an unrolling factor of _u_ = 4 is fairly common. If unrolling is applied in the _k_ dimension, edge cases must be handled to support values of _k_ that are not multiples of _u_. It is nearly universally true that the microkernel should not contain loops in the _m_ or _n_ directions; in other words, iteration over these dimensions should always be fully unrolled (within the loop over _k_).
  * **Zero `beta`.** If `beta` = 0.0 (or 0.0 + 0.0i for complex datatypes), then the microkernel should NOT use it explicitly, as `C11` may contain uninitialized memory (including elements containing `NaN` or `Inf`). This case should be detected and handled separately by overwriting `C11` with the `alpha * A1 * B1` product.

#### Using the auxinfo\_t object

Each microkernel ([gemm](KernelsHowTo.md#gemm-microkernel), [trsm](KernelsHowTo.md#trsm_microkernels), and [gemmtrsm](KernelsHowTo.md#gemmtrsm-microkernels)) takes as its last argument a pointer of type `auxinfo_t`. This BLIS-defined type is defined as a `struct` whose fields contain auxiliary values that may be useful to some microkernel authors, particularly when implementing certain optimization techniques. BLIS provides kernel authors access to the fields of the `auxinfo_t` object via the following static inline functions. Each function takes a single argument, the `auxinfo_t` pointer, and returns one of the values stored within the object.

  * `bli_auxinfo_next_a()`. Returns the address (`void*`) of the micropanel of `A` that will be used the next time the microkernel will be called.
  * `bli_auxinfo_next_b()`. Returns the address (`void*`) of the micropanel of `B` that will be used the next time the microkernel will be called.
  * `bli_auxinfo_ps_a()`. Returns the panel stride (`inc_t`) of the current micropanel of `A`.
  * `bli_auxinfo_ps_b()`. Returns the panel stride (`inc_t`) of the current micropanel of `B`.

The addresses of the next micropanels of `A` and `B` may be used by the microkernel to perform prefetching, if prefetching is supported by the architecture. Similarly, it may be useful to know the precise distance in memory to the next micropanel. (Note that occasionally the next micropanel to be used is **not** the same as the next micropanel in memory.)

Any and all of these values may be safely ignored; they are completely optional. However, BLIS guarantees that all values accessed via the macros listed above will **always** be initialized and meaningful, for every invocation of each microkernel (`gemm`, `trsm`, and `gemmtrsm`).


#### Example code for gemm

An example implementation of the `gemm` microkernel may be found in the `template` configuration directory in:
  * [config/template/kernels/3/bli\_gemm_opt\_mxn.c](https://github.com/flame/blis/tree/master/config/template/kernels/3/bli_gemm_opt_mxn.c)

Note that this implementation is coded in C99 and lacks several kinds of optimization that are typical of real-world optimized microkernels, such as vector instructions (or intrinsics) and loop unrolling in the _m_ or _n_ dimensions. It is meant to serve only as a starting point for a microkernel developer.




---


#### trsm microkernels

```c
void bli_?trsm_l_<suffix>
     (
       ctype*     restrict a11,
       ctype*     restrict b11,
       ctype*     restrict c11, inc_t rsc, inc_t csc,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     );

void bli_?trsm_u_<suffix>
     (
       ctype*     restrict a11,
       ctype*     restrict b11,
       ctype*     restrict c11, inc_t rsc, inc_t csc,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     );
```

where `<suffix>` is implementation-dependent. (Recall that the precise `<suffix>` associated with the microkernel along with the rest of the function name doesn't matter if you are querying the function address from the context. See section on [calling kernels](KernelsHowTo.md#calling-kernels) for details.) The following (more portable) wrappers are also defined:

```c
void bli_?trsm_l_ukernel
     (
       ctype*     restrict a11,
       ctype*     restrict b11,
       ctype*     restrict c11, inc_t rsc, inc_t csc,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     );

void bli_?trsm_u_ukernel
     (
       ctype*     restrict a11,
       ctype*     restrict b11,
       ctype*     restrict c11, inc_t rsc, inc_t csc,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     );
```

The `trsm_l` and `trsm_u` microkernels perform the following operation:

```
  C11 := inv(A11) * B11
```

where `A11` is _MR x MR_ and lower (`trsm_l`) or upper (`trsm_u`) triangular, `B11` is _MR x NR_, and `C11` is _MR x NR_.

_MR_ and _NR_ are the register blocksizes associated with the microkernel. They are chosen by the developer when the microkernel is written and then encoded into a BLIS configuration, which will reference the microkernel when the BLIS framework is instantiated into a library. For more information on setting register blocksizes and related constants, please see the [BLIS developer configuration guide](ConfigurationHowTo.md).

**Note:** Although the `gemm` microkernel must handle edge-cases, and therefore must take _m_ and _n_ parameters, the `trsm` microkernels are simpler in that they still assume _m = MR_ and _n = NR_, and therefore do not need these _m_ and _n_ parameters passed in.

Parameters:

  * `a11`:    The address of `A11`, which is the _MR x MR_ lower (`trsm_l`) or upper (`trsm_u`) triangular submatrix within the packed micropanel of matrix `A`. `A11` is stored by columns with leading dimension _PACKMR_, where typically _PACKMR_ = _MR_. (See [Implementation Notes for gemm](KernelsHowTo.md#implementation-notes-for-gemm) for a discussion of _PACKMR_.) Note that `A11` contains elements in both triangles, though elements in the unstored triangle are not guaranteed to be zero and thus should not be referenced.
  * `b11`:    The address of `B11`, which is an _MR x NR_ submatrix of the packed micropanel of `B`. `B11` is stored by rows with leading dimension _PACKNR_, where typically _PACKNR_ = _NR_. (See [Implementation Notes for gemm](KernelsHowTo.md#implementation-notes-for-gemm) for a discussion of _PACKNR_.)
  * `c11`:    The address of `C11`, which is an _MR x NR_ submatrix of matrix `C`, stored according to `rsc` and `csc`. `C11` is the submatrix within `C` that corresponds to the elements which were packed into `B11`. Thus, `C` is the original input matrix `B` to the overall `trsm` operation.
  * `rsc`:    The row stride of matrix `C11` (ie: the distance to the next row, in units of matrix elements).
  * `csc`:    The column stride of matrix `C11` (ie: the distance to the next column, in units of matrix elements).
  * `data`:   The address of an `auxinfo_t` object that contains auxiliary information that may be useful when optimizing the `trsm` microkernel implementation. (See [Using the auxinfo\_t object](KernelsHowTo.md#Using_the_auxinfo_t_object) for a discussion of the kinds of values available via `auxinfo_t`, and also [Implementation Notes for trsm](KernelsHowTo.md#implementation-notes-for-trsm) for caveats.)
  * `cntx`:   The address of the runtime context. The context can be queried for implementation-specific values such as cache and register blocksizes. However, most microkernels intrinsically "know" these values already, and thus the `cntx` argument usually can be safely ignored.

#### Diagrams for trsm

Please see the diagram for [gemmtrsm\_l](KernelsHowTo.md#diagram-for-gemmtrsm-l) and [gemmtrsm\_u](KernelsHowTo.md#diagram-for-gemmtrsm-u) to see depictions of the `trsm_l` and `trsm_u` microkernel operations and where they fit in with their preceding `gemm` subproblems.

#### Implementation Notes for trsm

  * **Register blocksizes.** See [Implementation Notes for gemm](KernelsHowTo.md#implementation-notes-for-gemm).
  * **Leading dimensions of `a11` and `b11`: _PACKMR_ and _PACKNR_.** See [Implementation Notes for gemm](KernelsHowTo.md#implementation-notes-for-gemm).
  * **Edge cases in _MR_, _NR_ dimensions.** See [Implementation Notes for gemm](KernelsHowTo.md#implementation-notes-for-gemm).
  * **Alignment of `a11` and `b11`.** The addresses `a11` and `b11` are aligned according to `PACKMR * sizeof(type)` and `PACKNR * sizeof(type)`, respectively.
  * **Unrolling loops.** Most optimized implementations should unroll all three loops within the `trsm` microkernel.
  * **Prefetching next micropanels of `A` and `B`.** We advise against using the `bli_auxinfo_next_a()` and `bli_auxinfo_next_b()` macros from within the `trsm_l` and `trsm_u` microkernels, since the values returned usually only make sense in the context of the overall `gemmtrsm` subproblem.
  * **Diagonal elements of `A11`.** At the time this microkernel is called, the diagonal entries of triangular matrix `A11` contain the **_inverse_** of the original elements. This inversion is done during packing so that we can avoid expensive division instructions within the microkernel itself. If the `diag` parameter to the higher level `trsm` operation was equal to `BLIS_UNIT_DIAG`, the diagonal elements will be explicitly unit.
  * **Zero elements of `A11`.** Since `A11` is lower triangular (for `trsm_l`), the strictly upper triangle implicitly contains zeros. Similarly, the strictly lower triangle of `A11` implicitly contains zeros when `A11` is upper triangular (for `trsm_u`). However, the packing function may or may not actually write zeros to this region. Thus, the implementation should not reference these elements.
  * **Output.** This microkernel must write its result to two places: the submatrix `B11` of the current packed micropanel of `B` _and_ the submatrix `C11` of the output matrix `C`.

#### Example code for trsm

Example implementations of the `trsm` microkernels may be found in the `template` configuration directory in:
  * [config/template/kernels/3/bli\_trsm\_l\_opt\_mxn.c](https://github.com/flame/blis/tree/master/config/template/kernels/3/bli_trsm_l_opt_mxn.c)
  * [config/template/kernels/3/bli\_trsm\_u\_opt\_mxn.c](https://github.com/flame/blis/tree/master/config/template/kernels/3/bli_trsm_u_opt_mxn.c)

Note that these implementations are coded in C99 and lack several kinds of optimization that are typical of real-world optimized microkernels, such as vector instructions (or intrinsics) and loop unrolling in _MR_ or _NR_. They are meant to serve only as a starting point for a microkernel developer.


---


#### gemmtrsm microkernels

```c
void bli_?gemmtrsm_l_<suffix>
     (
       dim_t               m,
       dim_t               n,
       dim_t               k,
       ctype*     restrict alpha,
       ctype*     restrict a10,
       ctype*     restrict a11,
       ctype*     restrict b01,
       ctype*     restrict b11,
       ctype*     restrict c11, inc_t rsc, inc_t csc,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     );

void bli_?gemmtrsm_u_<suffix>
     (
       dim_t               m,
       dim_t               n,
       dim_t               k,
       ctype*     restrict alpha,
       ctype*     restrict a12,
       ctype*     restrict a11,
       ctype*     restrict b21,
       ctype*     restrict b11,
       ctype*     restrict c11, inc_t rsc, inc_t csc,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     );
```

where `<suffix>` is implementation-dependent. (Recall that the precise `<suffix>` associated with the microkernel along with the rest of the function name doesn't matter if you are querying the function address from the context. See section on [calling kernels](KernelsHowTo.md#calling-kernels) for details.) The following (more portable) wrappers are also defined:

```c
void bli_?gemmtrsm_l_ukernel
     (
       dim_t               m,
       dim_t               n,
       dim_t               k,
       ctype*     restrict alpha,
       ctype*     restrict a10,
       ctype*     restrict a11,
       ctype*     restrict b01,
       ctype*     restrict b11,
       ctype*     restrict c11, inc_t rsc, inc_t csc,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     );

void bli_?gemmtrsm_u_ukernel
     (
       dim_t               m,
       dim_t               n,
       dim_t               k,
       ctype*     restrict alpha,
       ctype*     restrict a12,
       ctype*     restrict a11,
       ctype*     restrict b21,
       ctype*     restrict b11,
       ctype*     restrict c11, inc_t rsc, inc_t csc,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     );
```

The `gemmtrsm_l` microkernel performs the following compound operation:

```
  B11 := alpha * B11 - A10 * B01
  B11 := inv(A11) * B11
  C11 := B11
```

where `A11` is _MR x MR_ and lower triangular, `A10` is _MR x k_, and `B01` is _k x NR_.
The `gemmtrsm_u` microkernel performs:

```
  B11 := alpha * B11 - A12 * B21
  B11 := inv(A11) * B11
  C11 := B11
```

where `A11` is _MR x MR_ and upper triangular, `A12` is _MR x k_, and `B21` is _k x NR_.
In both cases, `B11` is _MR x NR_ and `alpha` is a scalar. However, `C11` is _m x n_, and therefore the `C11 := B11` statements amount to a copy of only the top-leftmost _m x n_ elements of `B11`. (Recall that A11 and B11 are packed and therefore guaranteed to reside within fully-sized micropanels, whereas `C11` exists in the caller-provided output matrix and may represent a bottom-right edge case.) Here, `inv()` denotes matrix inverse.

_MR_ and _NR_ are the register blocksizes associated with the microkernel. They are chosen by the developer when the microkernel is written and then encoded into a BLIS configuration, which will reference the microkernel when the BLIS framework is instantiated into a library. For more information on setting register blocksizes and related constants, please see the [BLIS developer configuration guide](ConfigurationHowTo.md).

Parameters:

  * `m`:      The number of rows of `C11`.
  * `n`:      The number of columns of `C11`.
  * `k`:      The number of columns of `A10` and rows of `B01` (`trsm_l`); the number of columns of `A12` and rows of `B21` (`trsm_u`).
  * `alpha`:  The address of a scalar to be applied to `B11`.
  * `a10`, `a12`:    The address of `A10` or `A12`, which is the _MR x k_ submatrix of the packed micropanel of `A` that is situated to the left (`trsm_l`) or right (`trsm_u`) of the _MR x MR_ triangular submatrix `A11`. `A10` and `A12` are stored by columns with leading dimension _PACKMR_, where typically _PACKMR_ = _MR_. (See [Implementation Notes for gemm](KernelsHowTo.md#implementation-notes-for-gemm) for a discussion of _PACKMR_.)
  * `a11`:    The address of `A11`, which is the _MR x MR_ lower (`trsm_l`) or upper (`trsm_u`) triangular submatrix within the packed micropanel of matrix `A` that is situated to the right of `A10` (`trsm_l`) or the left of `A12` (`trsm_u`). `A11` is stored by columns with leading dimension _PACKMR_, where typically _PACKMR_ = _MR_. (See [Implementation Notes for gemm](KernelsHowTo.md#implementation-notes-for-gemm) for a discussion of _PACKMR_.) Note that `A11` contains elements in both triangles, though elements in the unstored triangle are not guaranteed to be zero and thus should not be referenced.
  * `b01`, `b21`:   The address of `B01` and `B21`, which is the _k x NR_ submatrix of the packed micropanel of `B` that is situated above (`trsm_l`) or below (`trsm_u`) the _MR x NR_ block `B11`. `B01` and `B21` are stored by rows with leading dimension _PACKNR_, where typically _PACKNR_ = _NR_. (See [Implementation Notes for gemm](KernelsHowTo.md#implementation-notes-for-gemm) for a discussion of _PACKNR_.)
  * `b11`:    The address of `B11`, which is the _MR x NR_ submatrix of the packed micropanel of `B`, situated below `B01` (`trsm_l`) or above `B21` (`trsm_u`). `B11` is stored by rows with leading dimension _PACKNR_, where typically _PACKNR_ = _NR_. (See [Implementation Notes for gemm](KernelsHowTo.md#implementation-notes-for-gemm) for a discussion of _PACKNR_.)
  * `c11`:    The address of `C11`, which is an _m x n_ submatrix of matrix `C`, stored according to `rsc` and `csc`, where _m <= MR_ and _n <= NR_. `C11` is the submatrix within `C` that corresponds to the elements which were packed into `B11`. Thus, `C` is the original input matrix `B` to the overall `trsm` operation.
  * `rsc`:    The row stride of matrix `C11` (ie: the distance to the next row, in units of matrix elements).
  * `csc`:    The column stride of matrix `C11` (ie: the distance to the next column, in units of matrix elements).
  * `data`:   The address of an `auxinfo_t` object that contains auxiliary information that may be useful when optimizing the `gemmtrsm` microkernel implementation. (See [Using the auxinfo\_t object](KernelsHowTo.md#Using_the_auxinfo_t_object) for a discussion of the kinds of values available via `auxinfo_t`, and also [Implementation Notes for gemmtrsm](KernelsHowTo.md#implementation-notes-for-gemmtrsm) for caveats.)
  * `cntx`:   The address of the runtime context. The context can be queried for implementation-specific values such as cache and register blocksizes. However, most microkernels intrinsically "know" these values already, and thus the `cntx` argument usually can be safely ignored.

#### Diagram for gemmtrsm\_l

The diagram below shows the packed micropanel operands for `trsm_l` and how elements of each would be stored when _MR_ = _NR_ = 4. (The hex digits indicate the layout and order (but NOT the numeric contents) in memory. Here, matrix `A11` (referenced by `a11`) is **lower triangular**. Matrix `A11` **does contain** elements corresponding to the strictly upper triangle, however, they are not guaranteed to contain zeros and thus these elements should not be referenced.

```
                                              NR
                                            _______
                                       b01:|0 1 2 3|
                                           |4 5 6 7|
                                           |8 9 A B|
                                           |C D E F|
                                         k |   .   |
                                           |   .   |
       a10:                a11:            |   .   |
       ___________________  _______        |_______|
      |0 4 8 C            |`.      |   b11:|       |
  MR  |1 5 9 D . . .      |  `.    |       |       |
      |2 6 A E            |    `.  |    MR |       |
      |3_7_B_F____________|______`.|       |_______|

                k             MR
```


#### Diagram for gemmtrsm\_u

The diagram below shows the packed micropanel operands for `trsm_u` and how elements of each would be stored when _MR_ = _NR_ = 4. (The hex digits indicate the layout and order (but NOT the numeric contents) in memory. Here, matrix `A11` (referenced by `a11`) is **upper triangular**. Matrix `A11` **does contain** elements corresponding to the strictly lower triangle, however, they are not guaranteed to contain zeros and thus these elements should not be referenced.

```
       a11:     a12:                          NR
       ________ ___________________         _______
      |`.      |0 4 8              |   b11:|0 1 2 3|
  MR  |  `.    |1 5 9 . . .        |       |4 5 6 7|
      |    `.  |2 6 A              |    MR |8 9 A B|
      |______`.|3_7_B______________|       |___.___|
                                       b21:|   .   |
          MR             k                 |   .   |
                                           |       |
                                           |       |
     NOTE: Storage digits are shown      k |       |
     starting with a12 to avoid            |       |
     obscuring triangular structure        |       |
     of a11.                               |_______|

```


#### Implementation Notes for gemmtrsm

  * **Register blocksizes.** See [Implementation Notes for gemm](KernelsHowTo.md#implementation-notes-for-gemm).
  * **Leading dimensions of `a1` and `b1`: _PACKMR_ and _PACKNR_.** See [Implementation Notes for gemm](KernelsHowTo.md#implementation-notes-for-gemm).
  * **Edge cases in _MR_, _NR_ dimensions.** See [Implementation Notes for gemm](KernelsHowTo.md#implementation-notes-for-gemm).
  * **Alignment of `a1` and `b1`.** See [Implementation Notes for gemm](KernelsHowTo.md#implementation-notes-for-gemm).
  * **Unrolling loops.** Most optimized implementations should unroll all three loops within the `trsm` subproblem of `gemmtrsm`. See [Implementation Notes for gemm](KernelsHowTo.md#implementation-notes-for-gemm) for remarks on unrolling the `gemm` subproblem.
  * **Prefetching next micropanels of `A` and `B`.** When invoked from within a `gemmtrsm_l` microkernel, the addresses accessible via `bli_auxinfo_next_a()` and `bli_auxinfo_next_b()` refer to the next invocation's `a10` and `b01`, respectively, while in `gemmtrsm_u`, the `_next_a()` and `_next_b()` macros return the addresses of the next invocation's `a11` and `b11` (since those submatrices precede `a12` and `b21`).
  * **Zero `alpha`.** The microkernel can safely assume that `alpha` is non-zero; "alpha equals zero" handling is performed at a much higher level, which means that, in such a scenario, the microkernel will never get called.
  * **Diagonal elements of `A11`.** See [Implementation Notes for trsm](KernelsHowTo.md#implementation-notes-for-trsm).
  * **Zero elements of `A11`.** See [Implementation Notes for trsm](KernelsHowTo.md#implementation-notes-for-trsm).
  * **Output.** See [Implementation Notes for trsm](KernelsHowTo.md#implementation-notes-for-trsm).
  * **Optimization.** Let's assume that the [gemm microkernel](KernelsHowTo.md#gemm-microkernel) has already been optimized. You have two options with regard to optimizing the fused `gemmtrsm` microkernels:
    1. Optimize only the [trsm microkernels](KernelsHowTo.md#trsm-microkernels). This will result in the `gemm` and `trsm_l` microkernels being called in sequence. (Likewise for `gemm` and `trsm_u`.)
    1. Fuse the implementation of the `gemm` microkernel with that of the `trsm` microkernels by inlining both into the `gemmtrsm_l` and `gemmtrsm_u` microkernel definitions. This option is more labor-intensive, but also more likely to yield higher performance because it avoids redundant memory operations on the packed _MR x NR_ submatrix `B11`.


#### Example code for gemmtrsm

Example implementations of the `gemmtrsm` microkernels may be found in the `template` configuration directory in:
  * [config/template/kernels/3/bli\_gemmtrsm\_l\_opt\_mxn.c](https://github.com/flame/blis/tree/master/config/template/kernels/3/bli_gemmtrsm_l_opt_mxn.c)
  * [config/template/kernels/3/bli\_gemmtrsm\_u\_opt\_mxn.c](https://github.com/flame/blis/tree/master/config/template/kernels/3/bli_gemmtrsm_u_opt_mxn.c)

Note that these implementations are coded in C99 and lack several kinds of optimization that are typical of real-world optimized microkernels, such as vector instructions (or intrinsics) and loop unrolling in _MR_ or _NR_. They are meant to serve only as a starting point for a microkernel developer.



### Level-1f kernels

---

#### axpy2v kernel
```c
void bli_?axpy2v_<suffix>
     (
       conj_t           conjx,
       conj_t           conjy,
       dim_t            n,
       ctype*  restrict alphax,
       ctype*  restrict alphay,
       ctype*  restrict x, inc_t incx,
       ctype*  restrict y, inc_t incy,
       ctype*  restrict z, inc_t incz,
       cntx_t* restrict cntx
     );
```
This kernel performs the following operation:
```
  z := z + alphax * conjx(x) + alphay * conjy(y)
```
where `x`, `y`, and `z` are vectors of length _n_ stored with strides `incx`, `incy`, and `incz`, respectively. This kernel is typically implemented as the fusion of two `axpyv` operations on different input vectors `x` and `y` and with different scalars `alphax` and `alpay` to update the same output vector `z`.

---

#### dotaxpyv kernel
```c
void bli_?dotaxpyv_<suffix>
     (
       conj_t           conjxt,
       conj_t           conjx,
       conj_t           conjy,
       dim_t            n,
       ctype*  restrict alpha,
       ctype*  restrict x, inc_t incx,
       ctype*  restrict y, inc_t incy,
       ctype*  restrict rho,
       ctype*  restrict z, inc_t incz,
       cntx_t* restrict cntx
     );
```
This kernel performs the following operation:
```
  rho := conjxt(x)^T * conjy(y)
  z   := z + alpha * conjx(x)
```
where `x`, `y`, and `z` are vectors of length _n_ stored with strides `incx`, `incy`, and `incz`, respectively, and `rho` is a scalar. This kernel is typically implemented as a `dotv` operation fused with an `axpyv` operation.

---

#### axpyf kernel
```c
void bli_?axpyf_<suffix>
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            b,
       ctype*  restrict alpha,
       ctype*  restrict a, inc_t inca, inc_t lda,
       ctype*  restrict x, inc_t incx,
       ctype*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     );
```
This kernel performs the following operation:
```
  y := y + alpha * conja(a) * conjy(x)
```
where `a` is an _m x b_ matrix, `x` is a vector of length _b_, and `y` is a vector of length _m_. Vectors `x` and `y` are stored with strides `incx` and `incy`, respectively. Matrix `a` is stored with row stride `inca` and column stride `lda`, though `inca` is most often (in practice) unit. This kernel is typically implemented as a fused series of _b_ `axpyv` operations updating the same vector `y` (with the elements of `x` serving as the scalars and the columns of `a` serving as the vectors to be scaled).

---

#### dotxf kernel
```c
void bli_?dotxf_<suffix>
     (
       conj_t           conjat,
       conj_t           conjx,
       dim_t            m,
       dim_t            b,
       ctype*  restrict alpha,
       ctype*  restrict a, inc_t inca, inc_t lda,
       ctype*  restrict x, inc_t incx,
       ctype*  restrict beta,
       ctype*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     );
```
This kernel performs the following operation:
```
  y := beta * y + alpha * conjat(a)^T conjx(x)
```
where `a` is an _m x b_ matrix, where `w` is a vector of length _m_, `y` is a vector of length _b_, and `alpha` is a scalar.
Vectors `x` and `y` are stored with strides `incx` and `incy`, respectively. Matrix `a` is stored with row stride `inca` and column stride `lda`, though `inca` is most often (in practice) unit.
This kernel is typically implemented as a series of _b_ `dotxv` operations with the same right-hand operand vector `x` (contracted with the rows of `a^T` and accumulating to the corresponding elements of vector `y`).

---

#### dotxaxpyf kernel
```c
void bli_?dotxaxpyf_<suffix>
     (
       conj_t           conjat,
       conj_t           conja,
       conj_t           conjw,
       conj_t           conjx,
       dim_t            m,
       dim_t            b,
       ctype*  restrict alpha,
       ctype*  restrict a, inc_t inca, inc_t lda,
       ctype*  restrict w, inc_t incw,
       ctype*  restrict x, inc_t incx,
       ctype*  restrict beta,
       ctype*  restrict y, inc_t incy,
       ctype*  restrict z, inc_t incz,
       cntx_t* restrict cntx
     );
```
This kernel performs the following operation:
```
  y := beta * y + alpha * conjat(a)^T conjw(w)
  z :=        z + alpha *  conja(a)   conjx(x)
```
where `a` is an _m x b_ matrix, `w` and `z` are vectors of length _m_, `x` and `y` are vectors of length _b_, and `alpha` and `beta` are scalars.
Vectors `w`, `z`, `x` and `y` are stored with strides `incw`, `incz`, `incx`, and `incy`, respectively. Matrix `a` is stored with row stride `inca` and column stride `lda`, though `inca` is most often (in practice) unit.
This kernel is typically implemented as a series of _b_ `dotxv` operations with the same right-hand operand vector `w` fused with a series of _b_ `axpyv` operations updating the same vector `z`.

---



### Level-1v kernels

---

#### addv kernel
```c
void bli_?addv_<suffix>
     (
       conj_t           conjx,
       dim_t            n,
       ctype*  restrict x, inc_t incx,
       ctype*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     );
```
This kernel performs the following operation:
```
  y := y + conjx(x)
```
where `x` and `y` are vectors of length _n_ stored with strides `incx` and `incy`, respectively.

---

#### amaxv kernel
```c
void bli_?amaxv_<suffix>
     (
       dim_t            n,
       ctype*  restrict x, inc_t incx,
       dim_t*  restrict index,
       cntx_t* restrict cntx
     );
```
Given a vector of length _n_, this kernel returns the zero-based index `index` of the element of vector `x` that contains the largest absolute value (or, in the complex domain, the largest complex modulus).

If `NaN` is encountered, it is treated as if it were a valid value that was smaller than any other value in the vector. If more than one element contains the same maximum value, the index of the latter element is returned via `index`.

---

#### axpyv kernel
```c
void bli_?axpyv_<suffix>
     (
       conj_t           conjx,
       dim_t            n,
       ctype*  restrict alpha,
       ctype*  restrict x, inc_t incx,
       ctype*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     );
```
This kernel performs the following operation:
```
  y := y + alpha * conjx(x)
```
where `x` and `y` are vectors of length _n_ stored with strides `incx` and `incy`, respectively, and `alpha` is a scalar.

---

#### axpbyv kernel
```c
void bli_?axpbyv_<suffix>
     (
       conj_t           conjx,
       dim_t            n,
       ctype*  restrict alpha,
       ctype*  restrict x, inc_t incx,
       ctype*  restrict beta,
       ctype*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     );
```
This kernel performs the following operation:
```
  y := beta * y + alpha * conjx(x)
```
where `x` and `y` are vectors of length _n_ stored with strides `incx` and `incy`, respectively, and `alpha` and `beta` are scalars.

---

#### copyv kernel
```c
void bli_?copyv_<suffix>
     (
       conj_t           conjx,
       dim_t            n,
       ctype*  restrict x, inc_t incx,
       ctype*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     );
```
This kernel performs the following operation:
```
  y := conjx(x)
```
where `x` and `y` are vectors of length _n_ stored with strides `incx` and `incy`, respectively.

---

#### dotv kernel
```c
void bli_?dotv_<suffix>
     (
       conj_t           conjx,
       conj_t           conjy,
       dim_t            n,
       ctype*  restrict x, inc_t incx,
       ctype*  restrict y, inc_t incy,
       ctype*  restrict rho,
       cntx_t* restrict cntx
     );
```
This kernel performs the following operation:
```
  rho := conjxt(x)^T * conjy(y)
```
where `x` and `y` are vectors of length _n_ stored with strides `incx` and `incy`, respectively, and `rho` is a scalar.

---

#### dotxv kernel
```c
void bli_?dotxv_<suffix>
     (
       conj_t           conjx,
       conj_t           conjy,
       dim_t            n,
       ctype*  restrict alpha,
       ctype*  restrict x, inc_t incx,
       ctype*  restrict y, inc_t incy,
       ctype*  restrict beta,
       ctype*  restrict rho,
       cntx_t* restrict cntx
     );
```
This kernel performs the following operation:
```
  rho := beta * rho + alpha * conjxt(x)^T * conjy(y)
```
where `x` and `y` are vectors of length _n_ stored with strides `incx` and `incy`, respectively, and `alpha`, `beta`, and `rho` are scalars.

---

#### invertv kernel
```c
void bli_?invertv_<suffix>
     (
       dim_t            n,
       ctype*  restrict x, inc_t incx,
       cntx_t* restrict cntx
     );
```
This kernel inverts all elements of an _n_-length vector `x`.

---

#### invscalv kernel
```c
void bli_?invscalv_<suffix>
     (
       conj_t           conjalpha,
       dim_t            n,
       ctype*  restrict alpha,
       ctype*  restrict x, inc_t incx,
       cntx_t* restrict cntx
     );
```
This kernel performs the following operation:
```
  x := ( 1.0 / conjalpha(alpha) ) * x
```
where `x` is a vector of length _n_ stored with stride `incx` and `alpha` is a scalar.

---

#### scalv kernel
```c
void bli_?scalv_<suffix>
     (
       conj_t           conjalpha,
       dim_t            n,
       ctype*  restrict alpha,
       ctype*  restrict x, inc_t incx,
       cntx_t* restrict cntx
     );
```
This kernel performs the following operation:
```
  x := conjalpha(alpha) * x
```
where `x` is a vector of length _n_ stored with stride `incx` and `alpha` is a scalar.

---

#### scal2v kernel
```c
void bli_?scal2v_<suffix>
     (
       conj_t           conjx,
       dim_t            n,
       ctype*  restrict alpha,
       ctype*  restrict x, inc_t incx,
       ctype*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     );
```
This kernel performs the following operation:
```
  y := alpha * conjx(x)
```
where `x` and `y` are vectors of length _n_ stored with strides `incx` and `incy`, respectively, and `alpha` is a scalar.

---

#### setv kernel
```c
void bli_?setv_<suffix>
     (
       conj_t           conjalpha,
       dim_t            n,
       ctype*  restrict alpha,
       ctype*  restrict x, inc_t incx,
       cntx_t* restrict cntx
     );
```
This kernel performs the following operation:
```
  x := conjalpha(alpha)
```
where `x` is a vector of length _n_ stored with stride `incx` and `alpha` is a scalar. Note that here, the `:=` operator represents a broadcast of `conjalpha(alpha)` to every element in `x`.

---

#### subv kernel
```c
void bli_?subv_<suffix>
     (
       conj_t           conjx,
       dim_t            n,
       ctype*  restrict x, inc_t incx,
       ctype*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     );
```
This kernel performs the following operation:
```
  y := y - conjx(x)
```
where `x` and `y` are vectors of length _n_.

---

#### swapv kernel
```c
void bli_?swapv_<suffix>
     (
       dim_t            n,
       ctype*  restrict x, inc_t incx,
       ctype*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     );
```
This kernel swaps corresponding elements of two _n_-length vectors `x` and `y` stored with strides `incx` and `incy`, respectively.

---

#### xpbyv kernel
```c
void bli_?xpbyv_<suffix>
     (
       conj_t           conjx,
       dim_t            n,
       ctype*  restrict x, inc_t incx,
       ctype*  restrict beta,
       ctype*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     );
```
This kernel performs the following operation:
```
  y := beta * y + conjx(x)
```
where `x` and `y` are vectors of length _n_ stored with strides `incx` and `incy`, respectively, and `beta` is a scalar.

