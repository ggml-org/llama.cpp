## Contents

* **[Contents](MixedDatatypes.md#contents)**
* **[Introduction](MixedDatatypes.md#introduction)**
* **[Categories of mixed datatypes](MixedDatatypes.md#categories-of-mixed-datatypes)**
  * **[Computation precision](MixedDatatypes.md#computation-precision)**
  * **[Computation domain](MixedDatatypes.md#computation-domain)**
* **[Performing gemm with mixed datatypes](MixedDatatypes.md#performing-gemm-with-mixed-datatypes)**
* **[Running the testsuite for gemm with mixed datatypes](MixedDatatypes.md#running-the-testsuite-for-gemm-with-mixed-datatypes)**
* **[Known issues](MixedDatatypes.md#known-issues)**
* **[Conclusion](MixedDatatypes.md#conclusion)**

## Introduction

This document serves as a guide to users interested in taking advantage of
BLIS's support for performing the `gemm` operation on operands of differing
datatypes (domain and/or precision). For further details on the implementation
present in BLIS, please see the latest draft of our paper
"Supporting Mixed-domain Mixed-precision Matrix Multiplication
within the BLIS Framework" available in the
[Citations section](https://github.com/flame/blis/#citations)
of the main [BLIS webpage](https://github.com/flame/blis).

## Categories of mixed datatypes

Before going any further, we find it useful to categorize mixed datatype
support into four categories:

1. **Fully identical datatypes.** This is what people generally think of when
they think about the `gemm` operation: all operands are stored in the same
datatype (precision and domain), and the matrix product computation is
performed in the arithmetic represented by that datatype. (This category
doesn't actually involve mixing datatypes, but it's still worthwhile to
define.)
Example: matrix C updated by the product of matrix A and matrix B
(all matrices double-precision real).

2. **Mixed domain with identical precisions.** This category includes all
combinations of datatypes where the domain (real or complex) of each
operand may vary while the precisions (single or double precision) are
held constant across all operands.
Example: complex matrix C updated by the product of real matrix A and
complex matrix B (all matrices single-precision).

3. **Mixed precision within a single domain.** Here, all operands are stored
in the same domain (real or complex), however, the precision of each operand
may vary.
Example: double-precision real matrix C updated by the product of
single-precision real matrix A and single-precision real matrix B.

4. **Mixed precision and mixed domain.** This category allows both domains and
precision of each matrix operand to vary.
Example: double-precision complex matrix C updated by the product of
single-precision complex matrix A and single-precision real matrix B.

BLIS's implementation of mixed-datatype `gemm` supports all combinations
within all four categories.

### Computation precision

Because categories 3 and 4 involve mixing precisions, they come with an added
parameter: the *computation precision*. This parameter specifies the precision
in which the matrix multiplication (product) takes place. This precision
can be different than the storage precision of matrices A or B, and/or the
storage precision of matrix C.

When the computation precision differs from the storage precision of matrix A,
it implies that a typecast must occur when BLIS packs matrix A to contiguous
storage. Similarly, B may also need to be typecast during packing.

When the computation precision differs from the storage precision of C, it
means the result of the matrix product A*B must be typecast just before it
is accumulated back into matrix C.

### Computation domain

In addition to the computation precision, we also track a computation domain.
(Together, they form the computation datatype.) However, for now we do not
allow the user to explicitly specify the computation domain. Instead, the
computation domain is implied by the domains of A, B, and C. The following
table enumerates the six cases where there is at least one operand of each
domain, along with the corresponding same-domain cases from category 1 for
reference. We also list the total number of floating-point operations
performed in each case.
In the table, an 'R' denotes a real domain matrix operand while a 'C' denotes
a matrix in the complex domain. The R's and C's appear in the following
format of C += A * B, where A, B, and C are the matrix operands of `gemm`.

| Case # | Mixed domain case | Implied computation domain | flops performed |
|--------|:-----------------:|:--------------------------:|:---------------:|
|   1    | R += R * R        |          real              |     2mnk        |
|   2    | R += R * C        |          real              |     2mnk        |
|   3    | R += C * R        |          real              |     2mnk        |
|   4    | R += C * C        |       complex              |     4mnk        |
|   5    | C += R * R        |          real              |     2mnk        |
|   6    | C += R * C        |       complex              |     4mnk        |
|   7    | C += C * R        |       complex              |     4mnk        |
|   8    | C += C * C        |       complex              |     8mnk        |

The computation domain is implied in cases 1 and 8 in the same way that
it would be if mixed datatype support were absent entirely. These
cases execute 2mnk and 8mnk flops, respectively, as any traditional
implementation would.

In cases 2 and 3, we assume the computation domain is real because only
B or A, respectively, is complex. Thus, in these cases, the imaginary
components of the complex matrix are ignored, allowing us to perform
only 2mnk flops.

In case 5, we take the computation domain to be real because A and B are
both real, and thus it makes no sense to compute in the complex domain.
This means that we need only update the real components of C, leaving
the imaginary components untouched. This also results in 2mnk flops
being performed.

In case 4, we have complex A and B, allowing us to compute a complex
product. However, we can only save the real part of that complex product
since the output matrix C is real. Since we cannot update the imaginary
component of C (since it is not stored), we avoid computing that half of
the update entirely, reducing the flops performed to 4mnk. (Alternatively,
one may wish to request real domain computation, in which case the
imaginary components of A and B were ignored *prior* to computing the
matrix product. This approach would result in only 2mnk flops being
performed.)

In case 6, we wish for both the real and imaginary parts of B to participate
in the multiplication by A, with the result updating the corresponding real
and imaginary parts of C. Granted, the imaginary part of A is zero, and this
is taken advantage of in the computation to optimize performance, as indicated
by the 4mnk flop count. But fundamentally this computation executes in the
complex domain because both the real and imaginary parts of C are updated.
A similar story can be told about case 7.

## Performing gemm with mixed datatypes

In BLIS, performing a mixed-datatype `gemm` operation is easy. However,
it will require that the user call `gemm` through BLIS's object API.
For a basic series of examples for using the object-based API, please
see the example codes in the `examples/oapi` directory of the BLIS source
distribution.

The first step is to ensure that BLIS is configured with mixed datatype support.
Please consult with your current distribution's `configure` script for the
current semantics:
```
$ ./configure --help
```
As of this writing, mixed datatype support is enabled by default, and thus
no additional options are needed.

With mixed datatype support enabled in BLIS, using the functionality is
simply a matter of creating and initializing matrices of different precisions
and/or domains.
```c
dim_t  m = 5, n = 4, k = 2;
obj_t  a, b, c;
obj_t* alpha;
obj_t* beta;

bli_obj_create( BLIS_DOUBLE,   m, k, 0, 0, &a );
bli_obj_create( BLIS_FLOAT,    k, n, 0, 0, &b );
bli_obj_create( BLIS_SCOMPLEX, m, n, 0, 0, &c );

alpha = &BLIS_ONE;
beta  = &BLIS_ONE;

bli_randm( &a );
bli_randm( &b );
bli_randm( &c );
```
Then, you specify the computation precision by setting the computation
precision property of matrix C.
```c
bli_obj_set_comp_prec( BLIS_DOUBLE_PREC, &c );
```
If you do not explicitly specify the computation precision, it will default
to the *storage* precision of C.

With the objects created and the computation precision specified, call
`bli_gemm()` just as you would if the datatypes were identical:
```c
bli_gemm( alpha, &a, &b, beta, &c );
```
For more examples of using BLIS's object-based API, including methods
of initializing an matrix object with arbitrary values, please review the
example code found in the `examples/oapi` directory of the BLIS source
distribution.

## Running the testsuite for gemm with mixed datatypes

The BLIS testsuite has been retrofitted to test all combinations of datatypes
for each matrix operand. For more information on enabling mixed-datatype tests
for the `gemm` operation, please see the explanations of the relevant options
in the [Testsuite](Testsuite.md) documentation.

## Known issues

There may be odd behavior in the current implementation of mixed-datatype `gemm`
that does not conform to the reader's expectations. Below is a list of issues
that BLIS developers are aware of. If any of these issues poses a problem for
your application, please contact us by
[opening an issue](https://github.com/flame/blis/issues).

* **alpha with non-zero imaginary components.** Currently, there are many cases
of mixed-datatype `gemm` that do not yet support computing with `alpha` scalars
that have non-zero imaginary components--in other words, values of `alpha` that
are not in the real domain. (By contrast, non-real values for `beta` are fully
supported.) In order to support these use cases, additional code complexity and
logic would be required. Thus, we have chosen, for now, to not implement them.
If mixed-datatype `gemm` is invoked with a non-real valued `alpha` scalar, a
runtime error message will be printed and the linked program will abort.

* **Manually specifying the computation domain.** As mentioned in the section
discussing the [computation domain](MixedDatatype.md#computation-domain),
the computation domain of any case of mixed domain `gemm` is implied by the
operands and thus fixed; the user may not specify a different computation
domain, even if the mixed-domain case would reasonably allow for computing
in either domain.

* **Sandboxes should be used with caution.** When building a `gemm` sandbox in
BLIS, please consider either (a) disabling mixed datatype support, or (b)
consciously **never** running the testsuite with mixed domain or precision
computation enabled. Even the reference `ref99` sandbox implementation in BLIS
does not support mixing datatypes. If you do choose to enable a sandbox while
also keeping mixed datatype support enabled in BLIS, make sure that the
mixing of datatypes is disabled in the testsuite's `input.general` file
(unless, of course, you decide to implement all mixed datatype cases within
your sandbox). This issue is also discussed in the documentation for
[Sandboxes](Sandboxes.md#known-issues).

## Conclusion

For more information and documentation on BLIS, please visit the [BLIS github page](https://github.com/flame/blis/).

If you found a bug or wish to request a feature, please [open an issue](https://github.com/flame/blis/issues).

For general discussion or questions, please join and post a message to the [blis-devel mailing list](http://groups.google.com/group/blis-devel).

Thanks for your interest in BLIS!

