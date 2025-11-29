/******************************************************************************
* Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

/*! @file blis.hh
 *  blis.hh defines all the BLAS CPP templated public interfaces
 *  */
#ifndef BLIS_HH
#define BLIS_HH

#include "cblas.hh"

namespace blis {

/*! \brief Construct plane rotation for arbitrary data types 

  \b Purpose:	

  ROTG  construct plane rotation that eliminates b for arbitrary data types, such that \n

  [ z ] = [  c  s ] [ a ] \n
  [ 0 ]   [ -s  c ] [ b ] \n
  Data precisions supported include SINGLE/DOUBLE PRECISION REAL
  
  \param[in, out] a
  SINGLE/DOUBLE PRECISION REAL
  On entry, scalar a. On exit, set to z.
 
  \param[in, out] b
  SINGLE/DOUBLE PRECISION REAL
  On entry, scalar b. On exit, set to s, 1/c, or 0.
 
  \param[out] c
  Cosine of rotation; SINGLE/DOUBLE PRECISION REAL.
 
  \param[out] s
  Sine of rotation; SINGLE/DOUBLE PRECISION REAL.
  */
template< typename T >
void rotg(
    T *a,
    T *b,
    T *c,
    T *s )
{
    cblas_rotg(a, b, c, s);
}

/*! \brief Construct the modified givens transformation matrix for arbitrary data types 

  \b Purpose:	

  ROTMG construct modified (fast) plane rotation, H, that eliminates b, such that \n
  [ z ] = H [ sqrt(d1)    0  ] [ a ] \n
  [ 0 ]     [  0    sqrt(d2) ] [ b ] \n
  Data precisions supported include SINGLE/DOUBLE PRECISION REAL

  \param[in, out] d1
  SINGLE/DOUBLE PRECISION REAL
  sqrt(d1) is scaling factor for vector x.
 
  \param[in, out] d2
  SINGLE/DOUBLE PRECISION REAL
  sqrt(d2) is scaling factor for vector y.
 
  \param[in, out] a
  On entry, scalar a. On exit, set to z. SINGLE/DOUBLE PRECISION REAL.
 
  \param[in, out] b
  On entry, scalar b. SINGLE/DOUBLE PRECISION REAL.  
  
  \param[out] param
  SINGLE/DOUBLE PRECISION REAL array, dimension (5),giving parameters 
  of modified plane rotation 
  param(1)=DFLAG
  param(2)=DH11
  param(3)=DH21
  param(4)=DH12
  param(5)=DH22
  */
template< typename T >
void rotmg(
    T *d1,
    T *d2,
    T *a,
    T  b,
    T  param[5] )
{
    cblas_rotmg(d1, d2, a, b, param );
}

/*! \brief Apply plane rotation for arbitrary data types 

  \b Purpose:	

  ROT applies a plane rotation:  \n
  [ x^T ]   [  c  s ] [ x^T ]  \n
  [ y^T ] = [ -s  c ] [ y^T ]  \n
  Data precisions supported include SINGLE/DOUBLE PRECISION REAL

  \param[in] n
  Number of elements in x and y. n >= 0.
 
  \param[in, out] x
  SINGLE/DOUBLE PRECISION REAL array
  The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
 
  \param[in] incx
  incx is INTEGER
  Stride between elements of x. incx must not be zero.
  If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
 
  \param[in, out] y
  SINGLE/DOUBLE PRECISION REAL array
  The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
 
  \param[in] incy
  incy is INTEGER
  Stride between elements of y. incy must not be zero.
  If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
 
  \param[in] c
  Cosine of rotation; SINGLE/DOUBLE PRECISION REAL.
 
  \param[in] s
  Sine of rotation; SINGLE/DOUBLE PRECISION REAL.
  */
template< typename T >
void rot(
    int64_t n,
    T *x, int64_t incx,
    T *y, int64_t incy,
    T c,
    T s )
{
    cblas_rot( n, x, incx, y, incy, c, s );
}

/*! \brief Apply the modified givens transformation for arbitrary data types 

  \b Purpose:	

  ROTM applies modified (fast) plane rotation, H:  \n
  [ x^T ] = H [ x^T ]  \n
  [ y^T ]     [ y^T ]  \n
  
  Data precisions supported include SINGLE/DOUBLE PRECISION REAL

  \param[in] n
  Number of elements in x and y. n >= 0.
 
  \param[in, out] x
  SINGLE/DOUBLE PRECISION REAL array
  The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
 
  \param[in] incx
  incx is INTEGER
  Stride between elements of x. incx must not be zero.
  If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
 
  \param[in, out] y
  SINGLE/DOUBLE PRECISION REAL array
  The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
 
  \param[in] incy
  incy is INTEGER
  Stride between elements of y. incy must not be zero.
  If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
 
  \param[in] P
  SINGLE/DOUBLE PRECISION REAL array, dimension (5),giving parameters 
  of modified plane rotation 
  param(1)=DFLAG
  param(2)=DH11
  param(3)=DH21
  param(4)=DH12
  param(5)=DH22
  */
template< typename T >
void rotm(
    int64_t n,
    T *x, int64_t incx,
    T *y, int64_t incy,
    const T *P)
{
    cblas_rotm( n, x, incx, y, incy, P );
}

/*! \brief Interchanges two vectors of arbitrary data types 

  \b Purpose:	

  SWAP interchanges two vectors uses unrolled loops for increments equal to 1.\n
  x <=> y  \n
  Data precisions supported include SINGLE/DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  \param[in] n
  n is INTEGER
  Number of elements in x and y. n >= 0.
 
  \param[in] x
  REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array.
  The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
 
  \param[in] incx
  incx is INTEGER.
  Stride between elements of x. incx must not be zero.
  If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
 
  \param[in, out] y
  REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array.
  The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
 
  \param[in] incy
  incy is INTEGER.
  Stride between elements of y. incy must not be zero.
  If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
  */
template< typename T >
void swap(
    int64_t n,
    T *x, int64_t incx,
    T *y, int64_t incy )
{
    cblas_swap( n, x, incx, y, incy );
}

/*! \brief Scales a vector of arbitrary data types by a constant.

  \b Purpose:	

  SCAL scales a vector by a constant, uses unrolled loops for increment equal to 1.\n
  x = alpha * x \n
  Data precisions of vector & constant include SINGLE/DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  \param[in] n
  n is INTEGER
  Number of elements in x. n >= 0.
 
  \param[in] alpha
  alpha is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha. 
  
  \param[in ,out] x
  REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array
  The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
 
  \param[in] incx
  incx is INTEGER
  Stride between elements of x. incx must not be zero.
  If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
  */
template< typename TA, typename TB >
void scal(
    int64_t n,
    TA alpha,
    TB* x, int64_t incx )
{
    cblas_scal( n, alpha, x, incx );
}

/*! \brief Copies a vector x to a vector y for arbitrary data types 

  \b Purpose:	

  COPY copies a vector x to a vector y.\n
  y = x  \n
  Data precisions supported include SINGLE/DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  \param[in] n
  n is INTEGER
  Number of elements in x and y. n >= 0.
 
  \param[in] x
  REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array.
  The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
 
  \param[in] incx
  incx is INTEGER.
  Stride between elements of x. incx must not be zero.
  If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
 
  \param[out] y
  REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array.
  The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
 
  \param[in] incy
  incy is INTEGER.
  Stride between elements of y. incy must not be zero.
  If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
  */
template< typename T >
void copy(
    int64_t n,
    T const *x, int64_t incx,
    T       *y, int64_t incy )
{
    cblas_copy( n, x, incx, y, incy );
}

/*! \brief Performs addition of scaled vector for arbitrary data types 

  \b Purpose:	

  AXPY constant times a vector plus a vector.\n
  y = alpha*x + y  \n
  Data precisions supported include SINGLE/DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  \param[in] n
  n is INTEGER
  Number of elements in x and y. n >= 0.
 
  \param[in] alpha
  alpha is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.\n
  If alpha is zero, y is not updated.
  
  \param[in] x
  REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array.
  The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
 
  \param[in] incx
  incx is INTEGER.
  Stride between elements of x. incx must not be zero.
  If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
 
  \param[out] y
  REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array.
  The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
 
  \param[in] incy
  incy is INTEGER.
  Stride between elements of y. incy must not be zero.
  If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
  */
template< typename T >
void axpy(
    int64_t n,
    T alpha,
    T const *x, int64_t incx,
    T       *y, int64_t incy )
{
    cblas_axpy( n, alpha, x, incx, y, incy );
}

/*! \brief Performs the dot product of two vectors for arbitrary data types 

  \b Purpose:	

  DOT forms the dot product of two vectors
  uses unrolled loops for increments equal to one.\n
  dot = x^T * y \n
  Data precisions supported include SINGLE/DOUBLE PRECISION REAL

  \param[in] n
  n is INTEGER
  Number of elements in x and y. n >= 0.
  
  \param[in] x
  REAL/DOUBLE PRECISION array.
  The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
 
  \param[in] incx
  incx is INTEGER.
  Stride between elements of x. incx must not be zero.
  If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
 
  \param[in] y
  REAL/DOUBLE PRECISION array.
  The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
 
  \param[in] incy
  incy is INTEGER.
  Stride between elements of y. incy must not be zero.
  If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
  
  \return Unconjugated dot product, x^T * y.
  REAL/DOUBLE PRECISION 
  */
template< typename T, typename TR >
TR dot(
    int64_t n,
    T const *x, int64_t incx,
    T const *y, int64_t incy )
{
    return cblas_dot( n, x, incx, y, incy );
}

/*! \brief Performs the dot product of two complex vectors 

  \b Purpose:	

  DOTU forms the dot product of two complex vectors. \n
  CDOTU = X^T * Y \n
  Data precisions supported include SINGLE/DOUBLE PRECISION COMPLEX

  \param[in] n
  n is INTEGER
  Number of elements in x and y. n >= 0.
  
  \param[in] x
  REAL/DOUBLE PRECISION COMPLEX array.
  The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
 
  \param[in] incx
  incx is INTEGER.
  Stride between elements of x. incx must not be zero.
  If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
 
  \param[in] y
  REAL/DOUBLE PRECISION COMPLEX array.
  The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
 
  \param[in] incy
  incy is INTEGER.
  Stride between elements of y. incy must not be zero.
  If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
  
  \return Unconjugated dot product, x^T * y.
  REAL/DOUBLE PRECISION COMPLEX
  */
template< typename T >
T dotu(
    int64_t n,
    T const *x, int64_t incx,
    T const *y, int64_t incy )
{
    return cblas_dotu( n, x, incx, y, incy );
}

/*! \brief Performs the dot product of two complex vectors 

  \b Purpose:	

  DOTC forms the dot product of two complex vectors. \n
  CDOTU = X^H * Y \n
  Data precisions supported include SINGLE/DOUBLE PRECISION COMPLEX

  \param[in] n
  n is INTEGER
  Number of elements in x and y. n >= 0.
  
  \param[in] x
  REAL/DOUBLE PRECISION COMPLEX array.
  The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
 
  \param[in] incx
  incx is INTEGER.
  Stride between elements of x. incx must not be zero.
  If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
 
  \param[in] y
  REAL/DOUBLE PRECISION COMPLEX array.
  The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
 
  \param[in] incy
  incy is INTEGER.
  Stride between elements of y. incy must not be zero.
  If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
  
  \return Conjugated dot product, x^H * y.
  REAL/DOUBLE PRECISION COMPLEX
  */
template< typename T >
T dotc(
    int64_t n,
    T const *x, int64_t incx,
    T const *y, int64_t incy )
{
    return cblas_dotc( n, x, incx, y, incy );
}

/*! \brief Performs inner product of two vectors with extended precision accumulation

  \b Purpose:	

  DOTC forms the inner product of two vectors with extended precision accumulation. \n
  Data precisions supported include SINGLE PRECISION REAL

  \param[in] n
  n is INTEGER\n
  number of elements in input vector(s)
  
  \param[in] alpha
  alpha is REAL\n
  single precision scalar to be added to inner product
  
  \param[in] x
  x is REAL array, dimension ( 1 + ( n - 1 )*abs( incx ) )\n
  single precision vector with n elements
  
  \param[in] incx
  incx is INTEGER\n
  storage spacing between elements of x
  
  \param[in] y
  y is REAL array, dimension ( 1 + ( n - 1 )*abs( incx ) )\n
  single precision vector with n elements
  
  \param[in] incy
  incy is INTEGER\n
  storage spacing between elements of y
  
  \return S.P. result with dot product accumulated in D.P.
  */
template< typename T >
T sdsdot(
    int64_t n,
    T alpha,
    T const *x, int64_t incx,
    T const *y, int64_t incy )
{
    return cblas_sdsdot( n, alpha, x, incx, y, incy );
}

/*! \brief return 2-norm of vectors of arbitrary data types

  \b Purpose:	

  NRM2 returns the euclidean norm of a vector via the function name, so that
  SNRM2 := sqrt( x'*x ). \n
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  \param[in] n
  n is INTEGER\n
  number of elements in input vector(s)
  
  \param[in] x
  x is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array, 
  dimension ( 1 + ( n - 1 )*abs( incx ) )\n
  single precision vector with n elements
  
  \param[in] incx
  incx is INTEGER\n
  storage spacing between elements of x
  
  \return 2-norm of vector
  REAL SINGLE/DOUBLE PRECISION
  */
template< typename T >
real_type<T>
nrm2(
    int64_t n,
    T const * x, int64_t incx )
{
    return cblas_nrm2( n, x, incx );
}

/*! \brief return 1-norm of vector of arbitrary data types

  \b Purpose:	

  ASUM takes the sum of the absolute values, uses unrolled loops for 
  increment equal to one. \n
  ASUM := || Re(x) ||_1 + || Im(x) ||_1. \n
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  \param[in] n
  n is INTEGER\n
  number of elements in input vector(s)
  
  \param[in] x
  x is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array, 
  dimension ( 1 + ( n - 1 )*abs( incx ) )\n
  single precision vector with n elements
  
  \param[in] incx
  incx is INTEGER\n
  storage spacing between elements of x
  
  \return 1-norm of vector
  REAL SINGLE/DOUBLE PRECISION
  */
template< typename T >
real_type<T>
asum(
    int64_t n,
    T const *x, int64_t incx )
{
    return cblas_asum( n, x, incx );
}

/*! \brief Return Index of infinity-norm of vectors of arbitrary types.

  \b Purpose:	

  IAMAX finds the index of the first element having maximum |Re(.)| + |Im(.)|. \n
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  \param[in] n
  n is INTEGER\n
  number of elements in input vector(s)
  
  \param[in] x
  x is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array, 
  dimension ( 1 + ( n - 1 )*abs( incx ) ) \n
  single precision vector with n elements
  
  \param[in] incx
  incx is INTEGER\n
  storage spacing between elements of x
  
  \return Index of infinity-norm of vector
  INTEGER
  */
template< typename T >
int64_t iamax(
    int64_t n,
    T const *x, int64_t incx )
{
    return cblas_iamax( n, x, incx );
}

/*! \brief Solve General matrix-vector multiply for arbitrary data types 

  \b Purpose:	

  GEMV  performs one of the matrix-vector operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)
 
     y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
 
  where alpha and beta are scalars, x and y are vectors and A is an
  m by n matrix.

  \param[in] layout
  layout is enum CBLAS_LAYOUT
  layout specifies Matrix storage as follows:
  layout = CBLAS_LAYOUT::CblasRowMajor or Layout::CblasColMajor.

  \param[in] trans
  trans is CBLAS_TRANSPOSE
  On entry, trans specifies the operation to be used as follows: \n
  trans = CBLAS_TRANSPOSE::CblasNoTrans,y := alpha*A*x + beta*y. \n
  trans = CBLAS_TRANSPOSE::CblasTrans,  y := alpha*A**T*x + beta*y. \n
  trans = CBLAS_TRANSPOSE::CblasConjTrans,  y := alpha*A**T*x + beta*y.

  \param[in] m
  m is INTEGER
  On entry,  m specifies the number of rows of the matrix A.
  m must be at least zero.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the number of columns of the matrix A.
  n must be at least zero.

  \param[in] alpha
  alpha is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.

  \param[in] A
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  m-by-n , stored in an lda-by-n array [RowMajor: m-by-lda].

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  lda >= max(1, m) [RowMajor: lda >= max(1, n)].

  \param[in] x
  x is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension : \n
  If trans = CblasNoTrans:
  at least ( 1 + ( n - 1 )*abs( incx ) ). \n
  Otherwise:
  at least ( 1 + ( m - 1 )*abs( incx ) ).

  \param[in] incx
  incx is INTEGER
  On entry, incx specifies the increment for the elements of x.
  incx must not be zero.

  \param[in] beta
  beta is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, beta specifies the scalar alpha.When  beta  is
  supplied as zero then y need not be set on input.

  \param[in,out] y
  y is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array, dimension : \n
  If trans = CblasNoTrans:
  at least ( 1 + ( m - 1 )*abs( incy ) ). \n
  Otherwise:
  at least ( 1 + ( n - 1 )*abs( incy ) ). \n
  Before entry with beta non-zero, the incremented array y
  must contain the vector y. On exit, y is overwritten by the
  updated vector y.

  \param[in] incy
  incy is INTEGER
  On entry, incy specifies the increment for the elements of y.
  incy must not be zero.
  */
template< typename T >
void gemv(
    CBLAS_ORDER layout,
    CBLAS_TRANSPOSE trans,
    int64_t m, int64_t n,
    T alpha,
    T const *A, int64_t lda,
    T const *x, int64_t incx,
    T beta,
    T *y, int64_t incy )
{
    cblas_gemv(layout, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

/*! \brief Solve General matrix-vector multiply for arbitrary data types 

  \b Purpose:	

  GBMV  performs one of the matrix-vector operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)
 
     y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,   or
	 
	 y := alpha*A**H*x + beta*y,
 
  where alpha and beta are scalars, x and y are vectors and A is an
  m by n matrix with kl sub-diagonals and ku super-diagonals.

  \param[in] layout
  layout is enum CBLAS_LAYOUT
  layout specifies Matrix storage as follows:
  layout = CBLAS_LAYOUT::CblasRowMajor or Layout::CblasColMajor.

  \param[in] trans
  trans is CBLAS_TRANSPOSE
  On entry, trans specifies the operation to be used as follows: \n
  trans = CBLAS_TRANSPOSE::CblasNoTrans,y := alpha*A*x + beta*y. \n
  trans = CBLAS_TRANSPOSE::CblasTrans,  y := alpha*A**T*x + beta*y. \n
  trans = CBLAS_TRANSPOSE::CblasConjTrans,  y := alpha*A**H*x + beta*y.

  \param[in] m
  m is INTEGER
  On entry,  m specifies the number of rows of the matrix A.
  m must be at least zero.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the number of columns of the matrix A.
  n must be at least zero.

  \param[in] kl
  kl is INTEGER
  On entry,  kl specifies the number of sub-diagonals of the matrix A.
  kl must be at least zero.

  \param[in] ku
  ku is INTEGER
  On entry,  ku specifies the number of super-diagonals of the matrix A.
  ku must be at least zero.

  \param[in] alpha
  alpha is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.

  \param[in] A
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension lda-by-n.
  Before entry, the leading ( kl + ku + 1 ) by n part of the
  array A must contain the matrix of coefficients, supplied
  column by column, with the leading diagonal of the matrix in
  row ( ku + 1 ) of the array, the first super-diagonal
  starting at position 2 in row ku, the first sub-diagonal
  starting at position 1 in row ( ku + 2 ), and so on.
  Elements in the array A that do not correspond to elements
  in the band matrix (such as the top left ku by ku triangle)
  are not referenced.

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  lda >= ( kl + ku + 1 )

  \param[in] x
  x is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension : \n
  If trans = CblasNoTrans:
  at least ( 1 + ( n - 1 )*abs( incx ) ). \n
  Otherwise:
  at least ( 1 + ( m - 1 )*abs( incx ) ). \n
  Before entry, the incremented array x must contain the
  vector x.  

  \param[in] incx
  incx is INTEGER
  On entry, incx specifies the increment for the elements of x.
  incx must not be zero.

  \param[in] beta
  beta is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, beta specifies the scalar alpha.When  beta  is
  supplied as zero then y need not be set on input.

  \param[in,out] y
  y is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array, dimension : \n
  If trans = CblasNoTrans:
  at least ( 1 + ( m - 1 )*abs( incy ) ). \n
  Otherwise:
  at least ( 1 + ( n - 1 )*abs( incy ) ). \n
  Before entry with beta non-zero, the incremented array y
  must contain the vector y. On exit, y is overwritten by the
  updated vector y.

  \param[in] incy
  incy is INTEGER
  On entry, incy specifies the increment for the elements of y.
  incy must not be zero.
  */
template< typename T >
void gbmv(
    CBLAS_ORDER layout,
    CBLAS_TRANSPOSE trans,
    int64_t m, int64_t n,
    int64_t kl, int64_t ku,
    T alpha,
    T const *A, int64_t lda,
    T const *x, int64_t incx,
    T beta,
    T *y, int64_t incy )
{
    cblas_gbmv(layout, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}

/*! \brief Solves Hermitian matrix-vector multiply for arbitrary data types 

  \b Purpose:	

  HEMV  performs one of the matrix-vector operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION COMPLEX, 
  DOUBLE PRECISION COMPLEX(COMPLEX*16)
 
     y := alpha*A*x + beta*y,
 
  where alpha and beta are scalars, x and y are  n element vectors and 
  A is an n by n hermitian matrix.

  \param[in] layout
  layout is enum CBLAS_LAYOUT
  layout specifies Matrix storage as follows:
  layout = CBLAS_LAYOUT::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO
  uplo specifies specifies whether the matrix A is an upper or
  lower triangular matrix as follows: \n
  uplo = CBLAS_UPLO::CblasUpper   A is an upper triangular matrix. \n
  uplo = CBLAS_UPLO::CblasLower   A is a lower triangular matrix.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix A.n must be at least zero.

  \param[in] alpha
  alpha is COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.

  \param[in] A
  A is COMPLEX/COMPLEX*16 array,dimension lda-by-n. \n
  Before entry with  UPLO = CblasUpper, the leading n by n
  upper triangular part of the array A must contain the upper
  triangular part of the hermitian matrix and the strictly
  lower triangular part of A is not referenced.
  Before entry with UPLO = CblasLower, the leading n by n
  lower triangular part of the array A must contain the lower
  triangular part of the hermitian matrix and the strictly
  upper triangular part of A is not referenced. \n
  Note that the imaginary parts of the diagonal elements need
  not be set and are assumed to be zero.

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  lda must be at least max( 1, n ).

  \param[in] x
  x is COMPLEX/COMPLEX*16 array,dimension : \n
  at least ( 1 + ( n - 1 )*abs( incx ) ). \n
  Before entry, the incremented array x must contain the
  vector x.

  \param[in] incx
  incx is INTEGER
  On entry, incx specifies the increment for the elements of x.
  incx must not be zero.

  \param[in] beta
  beta is COMPLEX/COMPLEX*16
  On entry, beta specifies the scalar alpha.When  beta  is
  supplied as zero then y need not be set on input.

  \param[in,out] y
  y is COMPLEX/COMPLEX*16 array, dimension : \n
  at least ( 1 + ( n - 1 )*abs( incy ) ). \n
  Before entry with beta non-zero, the incremented array y
  must contain the vector y. On exit, y is overwritten by the
  updated vector y.

  \param[in] incy
  incy is INTEGER
  On entry, incy specifies the increment for the elements of y.
  incy must not be zero.
  */
template< typename T >
void hemv(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n,
    T alpha,
    T const *A, int64_t lda,
    T const *x, int64_t incx,
    T beta,
    T *y, int64_t incy )
{
    cblas_hemv(layout, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

/*! \brief Solves Hermitian matrix-vector multiply for arbitrary data types 

  \b Purpose:	

  HBMV  performs one of the matrix-vector operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION COMPLEX, 
  DOUBLE PRECISION COMPLEX(COMPLEX*16)
 
     y := alpha*A*x + beta*y,
 
  where alpha and beta are scalars, x and y are  n element vectors and 
  A is an n by n hermitian matrix with k super-diagonals.

  \param[in] layout
  layout is enum CBLAS_LAYOUT
  layout specifies Matrix storage as follows:
  layout = CBLAS_LAYOUT::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO
  uplo specifies specifies whether the the upper or lower triangular
  part of the band matrix A is being supplied as follows: \n
  uplo = CBLAS_UPLO::CblasUpper   A is an upper triangular matrix. \n
  uplo = CBLAS_UPLO::CblasLower   A is a lower triangular matrix.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix A.n must be at least zero.

  \param[in] k
  k is INTEGER
  On entry,  k specifies the number of super-diagonals of the matrix A.
  k must be at least zero.

  \param[in] alpha
  alpha is COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.

  \param[in] A
  A is COMPLEX/COMPLEX*16 array,dimension lda-by-n. \n
  Before entry with UPLO = CblasUpper, the leading ( k + 1 )
  by n part of the array A must contain the upper triangular
  band part of the hermitian matrix, supplied column by
  column, with the leading diagonal of the matrix in row
  ( k + 1 ) of the array, the first super-diagonal starting at
  position 2 in row k, and so on. The top left k by k triangle
  of the array A is not referenced. \n
  Before entry with UPLO = CblasLower, the leading ( k + 1 )
  by n part of the array A must contain the lower triangular
  band part of the hermitian matrix, supplied column by
  column, with the leading diagonal of the matrix in row 1 of
  the array, the first sub-diagonal starting at position 1 in
  row 2, and so on. The bottom right k by k triangle of the
  array A is not referenced. \n
  Note that the imaginary parts of the diagonal elements need
  not be set and are assumed to be zero.

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  lda must be at least ( k + 1 ).

  \param[in] x
  x is COMPLEX/COMPLEX*16 array,dimension : \n
  at least ( 1 + ( n - 1 )*abs( incx ) ). \n
  Before entry, the incremented array x must contain the
  vector x.

  \param[in] incx
  incx is INTEGER
  On entry, incx specifies the increment for the elements of x.
  incx must not be zero.

  \param[in] beta
  beta is COMPLEX/COMPLEX*16
  On entry, beta specifies the scalar alpha.

  \param[in,out] y
  y is COMPLEX/COMPLEX*16 array, dimension : \n
  at least ( 1 + ( n - 1 )*abs( incy ) ). \n
  Before entry with beta non-zero, the incremented array y
  must contain the vector y. On exit, y is overwritten by the
  updated vector y.

  \param[in] incy
  incy is INTEGER
  On entry, incy specifies the increment for the elements of y.
  incy must not be zero.
  */
template< typename T >
void hbmv(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n, int64_t k,
    T alpha,
    T const *A, int64_t lda,
    T const *x, int64_t incx,
    T beta,
    T *y, int64_t incy )
{
    cblas_hbmv(layout, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

/*! \brief Solves Hermitian matrix-vector multiply for arbitrary data types 

  \b Purpose:	

  HPMV  performs one of the matrix-vector operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION COMPLEX, 
  DOUBLE PRECISION COMPLEX(COMPLEX*16)
 
     y := alpha*A*x + beta*y,
 
  where alpha and beta are scalars, x and y are  n element vectors and 
  A is an n by n hermitian matrix, supplied in packed form.

  \param[in] layout
  layout is enum CBLAS_LAYOUT
  layout specifies Matrix storage as follows:
  layout = CBLAS_LAYOUT::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO
  uplo specifies specifies whether the the upper or lower triangular
  part of the band matrix A is supplied in the packed array Ap as follows: \n
  uplo = CBLAS_UPLO::CblasUpper   A is an upper triangular matrix. \n
  uplo = CBLAS_UPLO::CblasLower   A is a lower triangular matrix.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix A.n must be at least zero.

  \param[in] alpha
  alpha is COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.

  \param[in] Ap
  Ap is COMPLEX/COMPLEX*16 array,dimension atleast ( ( n*( n + 1 ) )/2 ). \n
  Before entry with UPLO = CblasUpper, the array Ap must
  contain the upper triangular part of the hermitian matrix
  packed sequentially, column by column, so that Ap( 1 )
  contains a( 1, 1 ), Ap( 2 ) and Ap( 3 ) contain a( 1, 2 )
  and a( 2, 2 ) respectively, and so on. \n
  Before entry with UPLO = CblasLower, the array Ap must
  contain the lower triangular part of the hermitian matrix
  packed sequentially, column by column, so that Ap( 1 )
  contains a( 1, 1 ), Ap( 2 ) and Ap( 3 ) contain a( 2, 1 )
  and a( 3, 1 ) respectively, and so on. \n
  Note that the imaginary parts of the diagonal elements need
  not be set and are assumed to be zero.

  \param[in] x
  x is COMPLEX/COMPLEX*16 array,dimension : \n
  at least ( 1 + ( n - 1 )*abs( incx ) ). \n
  Before entry, the incremented array x must contain the
  vector x.

  \param[in] incx
  incx is INTEGER
  On entry, incx specifies the increment for the elements of x.
  incx must not be zero.

  \param[in] beta
  beta is COMPLEX/COMPLEX*16
  On entry, beta specifies the scalar alpha.When beta is
  supplied as zero then y need not be set on input.

  \param[in,out] y
  y is COMPLEX/COMPLEX*16 array, dimension : \n
  at least ( 1 + ( n - 1 )*abs( incy ) ). \n
  Before entry with beta non-zero, the incremented array y
  must contain the vector y. On exit, y is overwritten by the
  updated vector y.

  \param[in] incy
  incy is INTEGER
  On entry, incy specifies the increment for the elements of y.
  incy must not be zero.
  */
template< typename T >
void hpmv(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n,
    T alpha,
    T const *Ap,
    T const *x, int64_t incx,
    T beta,
    T *y, int64_t incy )
{
    cblas_hpmv(layout, uplo, n, alpha, Ap, x, incx, beta, y, incy);
}

/*! \brief Solves Symmetric matrix-vector multiply for arbitrary data types 

  \b Purpose:	

  SYMV  performs one of the matrix-vector operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL
 
     y := alpha*A*x + beta*y,
 
  where alpha and beta are scalars, x and y are  n element vectors and 
  A is an n by n symmetric matrix.

  \param[in] layout
  layout is enum CBLAS_LAYOUT
  layout specifies Matrix storage as follows:
  layout = CBLAS_LAYOUT::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO
  uplo specifies specifies whether the matrix A is an upper or
  lower triangular matrix as follows: \n
  uplo = CBLAS_UPLO::CblasUpper   A is an upper triangular matrix. \n
  uplo = CBLAS_UPLO::CblasLower   A is a lower triangular matrix.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix A.n must be at least zero.

  \param[in] alpha
  alpha is SINGLE/DOUBLE PRECISION REAL
  On entry, alpha specifies the scalar alpha.

  \param[in] A
  A is SINGLE/DOUBLE PRECISION REAL array,dimension lda-by-n. \n
  Before entry with  UPLO = CblasUpper, the leading n by n
  upper triangular part of the array A must contain the upper
  triangular part of the symmetric matrix and the strictly
  lower triangular part of A is not referenced.
  Before entry with UPLO = CblasLower, the leading n by n
  lower triangular part of the array A must contain the lower
  triangular part of the symmetric matrix and the strictly
  upper triangular part of A is not referenced. \n

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  lda must be at least max( 1, n ).

  \param[in] x
  x is SINGLE/DOUBLE PRECISION REAL array,dimension : \n
  at least ( 1 + ( n - 1 )*abs( incx ) ). \n
  Before entry, the incremented array x must contain the
  vector x.

  \param[in] incx
  incx is INTEGER
  On entry, incx specifies the increment for the elements of x.
  incx must not be zero.

  \param[in] beta
  beta is SINGLE/DOUBLE PRECISION REAL
  On entry, beta specifies the scalar alpha.When  beta  is
  supplied as zero then y need not be set on input.

  \param[in,out] y
  y is SINGLE/DOUBLE PRECISION REAL array, dimension : \n
  at least ( 1 + ( n - 1 )*abs( incy ) ). \n
  Before entry with beta non-zero, the incremented array y
  must contain the vector y. On exit, y is overwritten by the
  updated vector y.

  \param[in] incy
  incy is INTEGER
  On entry, incy specifies the increment for the elements of y.
  incy must not be zero.
  */
template< typename T >
void symv(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n,
    T alpha,
    T const *A, int64_t lda,
    T const *x, int64_t incx,
    T beta,
    T *y, int64_t incy )
{
    cblas_symv(layout, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

/*! \brief Solves symmetric matrix-vector multiply for arbitrary data types 

  \b Purpose:	

  SBMV  performs one of the matrix-vector operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL
 
     y := alpha*A*x + beta*y,
 
  where alpha and beta are scalars, x and y are  n element vectors and 
  A is an n by n symmetric matrix with k super-diagonals.

  \param[in] layout
  layout is enum CBLAS_LAYOUT
  layout specifies Matrix storage as follows:
  layout = CBLAS_LAYOUT::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO
  uplo specifies specifies whether the the upper or lower triangular
  part of the band matrix A is being supplied as follows: \n
  uplo = CBLAS_UPLO::CblasUpper   A is an upper triangular matrix. \n
  uplo = CBLAS_UPLO::CblasLower   A is a lower triangular matrix.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix A.n must be at least zero.

  \param[in] k
  k is INTEGER
  On entry,  k specifies the number of super-diagonals of the matrix A.
  k must be at least zero.

  \param[in] alpha
  alpha is SINGLE/DOUBLE PRECISION REAL
  On entry, alpha specifies the scalar alpha.

  \param[in] A
  A is SINGLE/DOUBLE PRECISION REAL array,dimension lda-by-n. \n
  Before entry with UPLO = CblasUpper, the leading ( k + 1 )
  by n part of the array A must contain the upper triangular
  band part of the symmetric matrix, supplied column by
  column, with the leading diagonal of the matrix in row
  ( k + 1 ) of the array, the first super-diagonal starting at
  position 2 in row k, and so on. The top left k by k triangle
  of the array A is not referenced. \n
  Before entry with UPLO = CblasLower, the leading ( k + 1 )
  by n part of the array A must contain the lower triangular
  band part of the symmetric matrix, supplied column by
  column, with the leading diagonal of the matrix in row 1 of
  the array, the first sub-diagonal starting at position 1 in
  row 2, and so on. The bottom right k by k triangle of the
  array A is not referenced. \n
  Note that the imaginary parts of the diagonal elements need
  not be set and are assumed to be zero.

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  lda must be at least ( k + 1 ).

  \param[in] x
  x is SINGLE/DOUBLE PRECISION REAL array,dimension : \n
  at least ( 1 + ( n - 1 )*abs( incx ) ). \n
  Before entry, the incremented array x must contain the
  vector x.

  \param[in] incx
  incx is INTEGER
  On entry, incx specifies the increment for the elements of x.
  incx must not be zero.

  \param[in] beta
  beta is SINGLE/DOUBLE PRECISION REAL
  On entry, beta specifies the scalar alpha.

  \param[in,out] y
  y is SINGLE/DOUBLE PRECISION REAL array, dimension : \n
  at least ( 1 + ( n - 1 )*abs( incy ) ). \n
  Before entry with beta non-zero, the incremented array y
  must contain the vector y. On exit, y is overwritten by the
  updated vector y.

  \param[in] incy
  incy is INTEGER
  On entry, incy specifies the increment for the elements of y.
  incy must not be zero.
  */
template< typename T >
void sbmv(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n, int64_t k,
    T alpha,
    T const *A, int64_t lda,
    T const *x, int64_t incx,
    T beta,
    T *y, int64_t incy )
{
    cblas_sbmv(layout, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

/*! \brief Solves symmetric matrix-vector multiply for arbitrary data types 

  \b Purpose:	

  SPMV  performs one of the matrix-vector operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL
 
     y := alpha*A*x + beta*y,
 
  where alpha and beta are scalars, x and y are  n element vectors and 
  A is an n by n symmetric matrix, supplied in packed form.

  \param[in] layout
  layout is enum CBLAS_LAYOUT
  layout specifies Matrix storage as follows:
  layout = CBLAS_LAYOUT::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO
  uplo specifies specifies whether the the upper or lower triangular
  part of the band matrix A is supplied in the packed array Ap as follows: \n
  uplo = CBLAS_UPLO::CblasUpper   A is an upper triangular matrix. \n
  uplo = CBLAS_UPLO::CblasLower   A is a lower triangular matrix.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix A.n must be at least zero.

  \param[in] alpha
  alpha is SINGLE/DOUBLE PRECISION REAL
  On entry, alpha specifies the scalar alpha.

  \param[in] Ap
  Ap is SINGLE/DOUBLE PRECISION REAL array,dimension atleast ( ( n*( n + 1 ) )/2 ). \n
  Before entry with UPLO = CblasUpper, the array Ap must
  contain the upper triangular part of the symmetric matrix
  packed sequentially, column by column, so that Ap( 1 )
  contains a( 1, 1 ), Ap( 2 ) and Ap( 3 ) contain a( 1, 2 )
  and a( 2, 2 ) respectively, and so on. \n
  Before entry with UPLO = CblasLower, the array Ap must
  contain the lower triangular part of the symmetric matrix
  packed sequentially, column by column, so that Ap( 1 )
  contains a( 1, 1 ), Ap( 2 ) and Ap( 3 ) contain a( 2, 1 )
  and a( 3, 1 ) respectively, and so on. \n
  Note that the imaginary parts of the diagonal elements need
  not be set and are assumed to be zero.

  \param[in] x
  x is SINGLE/DOUBLE PRECISION REAL array,dimension : \n
  at least ( 1 + ( n - 1 )*abs( incx ) ). \n
  Before entry, the incremented array x must contain the
  vector x.

  \param[in] incx
  incx is INTEGER
  On entry, incx specifies the increment for the elements of x.
  incx must not be zero.

  \param[in] beta
  beta is SINGLE/DOUBLE PRECISION REAL
  On entry, beta specifies the scalar alpha.When beta is
  supplied as zero then y need not be set on input.

  \param[in,out] y
  y is SINGLE/DOUBLE PRECISION REAL array, dimension : \n
  at least ( 1 + ( n - 1 )*abs( incy ) ). \n
  Before entry with beta non-zero, the incremented array y
  must contain the vector y. On exit, y is overwritten by the
  updated vector y.

  \param[in] incy
  incy is INTEGER
  On entry, incy specifies the increment for the elements of y.
  incy must not be zero.
  */
template< typename T >
void spmv(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n,
    T alpha,
    T const *Ap,
    T const *x, int64_t incx,
    T beta,
    T *y, int64_t incy )
{
    cblas_spmv(layout, uplo, n, alpha, Ap, x, incx, beta, y, incy);
}

/*! \brief Solve the one of the matrix-vector operations for arbitrary data types 

  \b Purpose:	

  TRMV  performs  one of the matrix-vector operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  x := A*x,   or   x := A**T*x,

  where x is an n element vector and  A is an n by n unit, or non-unit,
  upper or lower triangular matrix.

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO.
  uplo specifies specifies whether the matrix A is an upper or
  lower triangular matrix as follows: \n
  uplo = CBLAS_UPLO::CblasUpper   A is an upper triangular matrix. \n
  uplo = CBLAS_UPLO::CblasLower   A is a lower triangular matrix.
  
  \param[in] trans
  trans is CBLAS_TRANSPOSE
  On entry, trans specifies the operation to be performed as follows:
  trans = CBLAS_TRANSPOSE::CblasNoTrans,  x := A*x. \n
  trans = CBLAS_TRANSPOSE::CblasTrans,  x := A**T*x. \n
  trans = CBLAS_TRANSPOSE::CblasConjTrans,  x := A**T*x.
  
  \param[in] diag
  diag is enum CBLAS_DIAG
  diag specifies specifies whether or not A is unit triangular
  as follows: \n
  diag = CBLAS_DIAG::CblasUnit   A is assumed to be unit triangular.\n
  diag = CBLAS_DIAG::CblasNonUnit   A is not assumed to be unit
                                 triangular.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix A.n must be at least zero.

  \param[in] A
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension ( lda, n )\n
  Before entry with  UPLO = CblasUpper, the leading n by n
  upper triangular part of the array A must contain the upper
  triangular matrix and the strictly lower triangular part of
  A is not referenced. \n
  Before entry with UPLO = CblasLower, the leading n by n
  lower triangular part of the array A must contain the lower
  triangular matrix and the strictly upper triangular part of
  A is not referenced. \n
  Note that when  DIAG = CblasUnit, the diagonal elements of
  A are not referenced either, but are assumed to be unity.  

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  lda must be at least max( 1, n ).

  \param[in, out] x
  x is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension : \n
  at least ( 1 + ( n - 1 )*abs( incx ) ). \n
  Before entry, the incremented array x must contain the
  vector x.On exit, x is overwritten with the transformed vector x.

  \param[in] incx
  incx is INTEGER
  On entry, incx specifies the increment for the elements of x.
  incx must not be zero.
  */
template< typename T >
void trmv(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    int64_t n,
    T const *A, int64_t lda,
    T       *x, int64_t incx )
{
    cblas_trmv(layout, uplo, trans, diag, n, A, lda, x, incx);
}

/*! \brief Solve the one of the matrix-vector operations for arbitrary data types 

  \b Purpose:	

  TBMV  performs  one of the matrix-vector operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  x := A*x,   or   x := A**T*x,

  where x is an n element vector and  A is an n by n unit, or non-unit,
  upper or lower triangular band matrix, with ( k + 1 ) diagonals.

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO.
  uplo specifies specifies whether the matrix A is an upper or
  lower triangular matrix as follows: \n
  uplo = CBLAS_UPLO::CblasUpper   A is an upper triangular matrix. \n
  uplo = CBLAS_UPLO::CblasLower   A is a lower triangular matrix.
  
  \param[in] trans
  trans is CBLAS_TRANSPOSE
  On entry, trans specifies the operation to be performed as follows:
  trans = CBLAS_TRANSPOSE::CblasNoTrans,  x := A*x. \n
  trans = CBLAS_TRANSPOSE::CblasTrans,  x := A**T*x. \n
  trans = CBLAS_TRANSPOSE::CblasConjTrans,  x := A**T*x.
  
  \param[in] diag
  diag is enum CBLAS_DIAG
  diag specifies specifies whether or not A is unit triangular
  as follows: \n
  diag = CBLAS_DIAG::CblasUnit   A is assumed to be unit triangular.\n
  diag = CBLAS_DIAG::CblasNonUnit   A is not assumed to be unit
                                 triangular.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix A.n must be at least zero.

  \param[in] k
  k is INTEGER
  On entry with UPLO = CblasUpper, k specifies the number of
  super-diagonals of the matrix A.
  On entry with UPLO = CblasLower, k specifies the number of
  sub-diagonals of the matrix A.
  k must at least zero.
 
  \param[in] A
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension ( lda, n )\n
  Before entry with UPLO = CblasUpper, the leading ( k + 1 )
  by n part of the array A must contain the upper triangular
  band part of the matrix of coefficients, supplied column by
  column, with the leading diagonal of the matrix in row
  ( k + 1 ) of the array, the first super-diagonal starting at
  position 2 in row k, and so on. The top left k by k triangle
  of the array A is not referenced. \n 
  Before entry with UPLO = CblasLower, the leading ( k + 1 )
  by n part of the array A must contain the lower triangular
  band part of the matrix of coefficients, supplied column by
  column, with the leading diagonal of the matrix in row 1 of
  the array, the first sub-diagonal starting at position 1 in
  row 2, and so on. The bottom right k by k triangle of the
  array A is not referenced. \n
  Note that when DIAG = CblasUnit the elements of the array A
  corresponding to the diagonal elements of the matrix are not
  referenced, but are assumed to be unity.  

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  lda must be at least max( 1, ( k + 1 ) ).

  \param[in, out] x
  x is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension : \n
  at least ( 1 + ( n - 1 )*abs( incx ) ). \n
  Before entry, the incremented array x must contain the
  vector x.On exit, x is overwritten with the transformed vector x.

  \param[in] incx
  incx is INTEGER
  On entry, incx specifies the increment for the elements of x.
  incx must not be zero.
  */
template< typename T >
void tbmv(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    int64_t n, int64_t k,
    T const *A, int64_t lda,
    T       *x, int64_t incx )
{
    cblas_tbmv(layout, uplo, trans, diag, n, k, A, lda, x, incx);
}


/*! \brief Solve the one of the matrix-vector operations for arbitrary data types 

  \b Purpose:	

  TPMV  performs  one of the matrix-vector operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  x := A*x,   or   x := A**T*x,

  where x is an n element vector and  A is an n by n unit, or non-unit,
  upper or lower triangular matrix, supplied in packed form.

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO.
  uplo specifies specifies whether the matrix A is an upper or
  lower triangular matrix as follows: \n
  uplo = CBLAS_UPLO::CblasUpper   A is an upper triangular matrix. \n
  uplo = CBLAS_UPLO::CblasLower   A is a lower triangular matrix.
  
  \param[in] trans
  trans is CBLAS_TRANSPOSE
  On entry, trans specifies the operation to be performed as follows:
  trans = CBLAS_TRANSPOSE::CblasNoTrans,  x := A*x. \n
  trans = CBLAS_TRANSPOSE::CblasTrans,  x := A**T*x. \n
  trans = CBLAS_TRANSPOSE::CblasConjTrans,  x := A**T*x.
  
  \param[in] diag
  diag is enum CBLAS_DIAG
  diag specifies specifies whether or not A is unit triangular
  as follows: \n
  diag = CBLAS_DIAG::CblasUnit   A is assumed to be unit triangular.\n
  diag = CBLAS_DIAG::CblasNonUnit   A is not assumed to be unit
                                 triangular.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix A.n must be at least zero.
 
 \param[in] Ap
  Ap is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension 
  ( ( n*( n + 1 ) )/2 ). \n
  Before entry with  UPLO = CblasUpper, the array Ap must
  contain the upper triangular matrix packed sequentially,
  column by column, so that Ap( 1 ) contains a( 1, 1 ),
  Ap( 2 ) and Ap( 3 ) contain a( 1, 2 ) and a( 2, 2 )
  respectively, and so on. \n
  Before entry with UPLO = CblasLower, the array Ap must
  contain the lower triangular matrix packed sequentially,
  column by column, so that Ap( 1 ) contains a( 1, 1 ),
  Ap( 2 ) and Ap( 3 ) contain a( 2, 1 ) and a( 3, 1 )
  respectively, and so on. \n
  Note that when  DIAG = CblasUnit, the diagonal elements of
  A are not referenced, but are assumed to be unity.

  \param[in, out] x
  x is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension : \n
  at least ( 1 + ( n - 1 )*abs( incx ) ). \n
  Before entry, the incremented array x must contain the
  vector x.On exit, x is overwritten with the transformed vector x.

  \param[in] incx
  incx is INTEGER
  On entry, incx specifies the increment for the elements of x.
  incx must not be zero.
  */
template< typename T >
void tpmv(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    int64_t n,
    T const *Ap,
    T       *x, int64_t incx )
{
    cblas_tpmv(layout, uplo, trans, diag, n, Ap, x, incx);
}

/*! \brief Solve the one of the triangular matrix-vector equation for arbitrary data types 

  \b Purpose:	

  TRSV  solves one of the systems of equations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  A*x = b,   or   A**T*x = b,

  where b and x are n element vectors and A is an n by n unit, or
  non-unit, upper or lower triangular matrix

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO.
  uplo specifies specifies whether the matrix A is an upper or
  lower triangular matrix as follows: \n
  uplo = CBLAS_UPLO::CblasUpper   A is an upper triangular matrix. \n
  uplo = CBLAS_UPLO::CblasLower   A is a lower triangular matrix.
  
  \param[in] trans
  trans is CBLAS_TRANSPOSE
  On entry, trans specifies the operation to be performed as follows:
  trans = CBLAS_TRANSPOSE::CblasNoTrans,  A*x = b. \n
  trans = CBLAS_TRANSPOSE::CblasTrans,  A**T*x = b. \n
  trans = CBLAS_TRANSPOSE::CblasConjTrans,  A**T*x = b.
  
  \param[in] diag
  diag is enum CBLAS_DIAG
  diag specifies specifies whether or not A is unit triangular
  as follows: \n
  diag = CBLAS_DIAG::CblasUnit   A is assumed to be unit triangular.\n
  diag = CBLAS_DIAG::CblasNonUnit   A is not assumed to be unit
                                 triangular.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix A.n must be at least zero.

  \param[in] A
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension ( lda, n )\n
  Before entry with  UPLO = CblasUpper, the leading n by n
  upper triangular part of the array A must contain the upper
  triangular matrix and the strictly lower triangular part of
  A is not referenced. \n
  Before entry with UPLO = CblasLower, the leading n by n
  lower triangular part of the array A must contain the lower
  triangular matrix and the strictly upper triangular part of
  A is not referenced. \n
  Note that when  DIAG = CblasUnit, the diagonal elements of
  A are not referenced either, but are assumed to be unity.  

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  lda must be at least max( 1, n ).

  \param[in, out] x
  x is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension : 
  at least ( 1 + ( n - 1 )*abs( incx ) ). \n
  Before entry, the incremented array x must contain the
  element right-hand side vector b.On exit, x is overwritten
  with the transformed vector x.

  \param[in] incx
  incx is INTEGER
  On entry, incx specifies the increment for the elements of x.
  incx must not be zero.
  */
template< typename T >
void trsv(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    int64_t n,
    T const *A, int64_t lda,
    T       *x, int64_t incx )
{
    cblas_trsv(layout, uplo, trans, diag, n, A, lda, x, incx);
}

/*! \brief Solve the one of the triangular matrix-vector equation for arbitrary data types 

  \b Purpose:	

  TBSV  solves one of the systems of equations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  A*x = b,   or   A**T*x = b,

  where b and x are n element vectors and A is an n by n unit, or
  non-unit, upper or lower triangular band matrix, with ( k + 1 )
  diagonals.

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO.
  uplo specifies specifies whether the matrix A is an upper or
  lower triangular matrix as follows: \n
  uplo = CBLAS_UPLO::CblasUpper   A is an upper triangular matrix. \n
  uplo = CBLAS_UPLO::CblasLower   A is a lower triangular matrix.
  
  \param[in] trans
  trans is CBLAS_TRANSPOSE
  On entry, trans specifies the operation to be performed as follows:
  trans = CBLAS_TRANSPOSE::CblasNoTrans,  A*x = b. \n
  trans = CBLAS_TRANSPOSE::CblasTrans,  A**T*x = b. \n
  trans = CBLAS_TRANSPOSE::CblasConjTrans,  A**T*x = b.
  
  \param[in] diag
  diag is enum CBLAS_DIAG
  diag specifies specifies whether or not A is unit triangular
  as follows: \n
  diag = CBLAS_DIAG::CblasUnit   A is assumed to be unit triangular.\n
  diag = CBLAS_DIAG::CblasNonUnit   A is not assumed to be unit
                                 triangular.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix A.n must be at least zero.

  \param[in] k
  k is INTEGER
  On entry with UPLO = CblasUpper, k specifies the number of
  super-diagonals of the matrix A.
  On entry with UPLO = CblasLower, k specifies the number of
  sub-diagonals of the matrix A.
  k must at least zero.

  \param[in] A
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension ( lda, n )\n
  Before entry with  UPLO = CblasUpper, the leading ( k + 1 )
  by n part of the array A must contain the upper triangular
  band part of the matrix of coefficients, supplied column by
  column, with the leading diagonal of the matrix in row
  ( k + 1 ) of the array, the first super-diagonal starting at
  position 2 in row k, and so on. The top left k by k triangle
  of the array A is not referenced. \n
  Before entry with UPLO = CblasLower, the leading ( k + 1 )
  by n part of the array A must contain the lower triangular
  band part of the matrix of coefficients, supplied column by
  column, with the leading diagonal of the matrix in row 1 of
  the array, the first sub-diagonal starting at position 1 in
  row 2, and so on. The bottom right k by k triangle of the
  array A is not referenced. \n
  Note that when  DIAG = CblasUnit, the elements of the array A
  corresponding to the diagonal elements of the matrix are not
  referenced, but are assumed to be unity.  

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  lda must be at least max( 1, k+1 ).

  \param[in, out] x
  x is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension : 
  at least ( 1 + ( n - 1 )*abs( incx ) ). \n
  Before entry, the incremented array x must contain the
  element right-hand side vector b.On exit, x is overwritten
  with the solution vector x.

  \param[in] incx
  incx is INTEGER
  On entry, incx specifies the increment for the elements of x.
  incx must not be zero.
  */
template< typename T >
void tbsv(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    int64_t n, int64_t k,
    T const *A, int64_t lda,
    T       *x, int64_t incx )
{
    cblas_tbsv(layout, uplo, trans, diag, n, k, A, lda, x, incx);
}


/*! \brief Solve the one of the triangular matrix-vector equation for arbitrary data types 

  \b Purpose:	

  TPSV  solves one of the systems of equations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  A*x = b,   or   A**T*x = b,

  where b and x are n element vectors and A is an n by n unit, or
  non-unit, upper or lower triangular band matrix, supplied in packed form.

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO.
  uplo specifies specifies whether the matrix A is an upper or
  lower triangular matrix as follows: \n
  uplo = CBLAS_UPLO::CblasUpper   A is an upper triangular matrix. \n
  uplo = CBLAS_UPLO::CblasLower   A is a lower triangular matrix.
  
  \param[in] trans
  trans is CBLAS_TRANSPOSE
  On entry, trans specifies the operation to be performed as follows:
  trans = CBLAS_TRANSPOSE::CblasNoTrans,  A*x = b. \n
  trans = CBLAS_TRANSPOSE::CblasTrans,  A**T*x = b. \n
  trans = CBLAS_TRANSPOSE::CblasConjTrans,  A**T*x = b.
  
  \param[in] diag
  diag is enum CBLAS_DIAG
  diag specifies specifies whether or not A is unit triangular
  as follows: \n
  diag = CBLAS_DIAG::CblasUnit   A is assumed to be unit triangular.\n
  diag = CBLAS_DIAG::CblasNonUnit   A is not assumed to be unit
                                 triangular.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix A.n must be at least zero.

 \param[in] Ap
  Ap is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension 
  ( ( n*( n + 1 ) )/2 ). \n
  Before entry with  UPLO = CblasUpper, the array Ap must
  contain the upper triangular matrix packed sequentially,
  column by column, so that Ap( 1 ) contains a( 1, 1 ),
  Ap( 2 ) and Ap( 3 ) contain a( 1, 2 ) and a( 2, 2 )
  respectively, and so on. \n
  Before entry with UPLO = CblasLower, the array Ap must
  contain the lower triangular matrix packed sequentially,
  column by column, so that Ap( 1 ) contains a( 1, 1 ),
  Ap( 2 ) and Ap( 3 ) contain a( 2, 1 ) and a( 3, 1 )
  respectively, and so on. \n
  Note that when  DIAG = CblasUnit, the diagonal elements of
  A are not referenced, but are assumed to be unity.

  \param[in, out] x
  x is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension : 
  at least ( 1 + ( n - 1 )*abs( incx ) ). \n
  Before entry, the incremented array x must contain the
  element right-hand side vector b.On exit, x is overwritten
  with the solution vector x.

  \param[in] incx
  incx is INTEGER
  On entry, incx specifies the increment for the elements of x.
  incx must not be zero.
  */
template< typename T >
void tpsv(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    int64_t n,
    T const *Ap,
    T       *x, int64_t incx )
{
    cblas_tpsv(layout, uplo, trans, diag, n, Ap, x, incx);
}

/*! \brief Perform the General matrix rank-1 update for arbitrary data types 

  \b Purpose:	

  GER  performs the rank 1 operation for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,

  A := alpha*x*y**T + A,
 
  where alpha is a scalar, x is an m element vector, y is an n element
  vector and A is an m by n matrix.

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] m
  m is INTEGER
  On entry,  m specifies the number of rows of the matrix A.
  m must be at least zero.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the number of columns of the matrix A.
  n must be at least zero.

  \param[in] alpha
  alpha is REAL/DOUBLE PRECISION
  On entry, alpha specifies the scalar alpha.

  \param[in] x
  x is REAL/DOUBLE PRECISION array,dimension : 
  at least ( 1 + ( m - 1 )*abs( incx ) ). \n
  Before entry, the incremented array x must contain the m
  element vector x.

  \param[in] incx
  incx is INTEGER
  On entry, incx specifies the increment for the elements of x.
  incx must not be zero.

  \param[in] y
  y is REAL/DOUBLE PRECISION array,dimension : 
  at least ( 1 + ( n - 1 )*abs( incy ) ). \n
  Before entry, the incremented array y must contain the n
  element vector y.

  \param[in] incy
  incy is INTEGER
  On entry, incy specifies the increment for the elements of y.
  incy must not be zero.

  \param[in,out] A
  A is REAL/DOUBLE PRECISION array,dimension ( lda, n )\n
  Before entry, the leading m by n part of the array A must
  contain the matrix of coefficients. On exit, A is
  overwritten by the updated matrix.  

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  lda must be at least max( 1, m ).
  */
template< typename T >
void ger(
    CBLAS_ORDER layout,
    int64_t m, int64_t n,
    T alpha,
    T const *x, int64_t incx,
    T const *y, int64_t incy,
    T *A, int64_t lda )
{
    cblas_ger(layout, m, n, alpha, x, incx, y, incy, A, lda);
}

/*! \brief Perform the General matrix rank-1 update for arbitrary data types 

  \b Purpose:	

  GERU  performs the rank 1 operation for arbitrary data types
  Data precisions supported include SINGLE/DOUBLE PRECISION COMPLEX(COMPLEX*16)

  A := alpha*x*y**T + A,
 
  where alpha is a scalar, x is an m element vector, y is an n element
  vector and A is an m by n matrix.

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] m
  m is INTEGER
  On entry,  m specifies the number of rows of the matrix A.
  m must be at least zero.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the number of columns of the matrix A.
  n must be at least zero.

  \param[in] alpha
  alpha is SINGLE/DOUBLE PRECISION COMPLEX
  On entry, alpha specifies the scalar alpha.

  \param[in] x
  x is SINGLE/DOUBLE PRECISION COMPLEX array,dimension : 
  at least ( 1 + ( m - 1 )*abs( incx ) ). \n
  Before entry, the incremented array x must contain the m
  element vector x.

  \param[in] incx
  incx is INTEGER
  On entry, incx specifies the increment for the elements of x.
  incx must not be zero.

  \param[in] y
  y is SINGLE/DOUBLE PRECISION COMPLEX array,dimension : 
  at least ( 1 + ( n - 1 )*abs( incy ) ). \n
  Before entry, the incremented array y must contain the n
  element vector y.

  \param[in] incy
  incy is INTEGER
  On entry, incy specifies the increment for the elements of y.
  incy must not be zero.

  \param[in,out] A
  A is SINGLE/DOUBLE PRECISION COMPLEX array,dimension ( lda, n )\n
  Before entry, the leading m by n part of the array A must
  contain the matrix of coefficients. On exit, A is
  overwritten by the updated matrix.  

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  lda must be at least max( 1, m ).
  */
template< typename T >
void geru(
    CBLAS_ORDER layout,
    int64_t m, int64_t n,
    T alpha,
    T const *x, int64_t incx,
    T const *y, int64_t incy,
    T *A, int64_t lda )
{
    cblas_geru(layout, m, n, alpha, x, incx, y, incy, A, lda);
}

/*! \brief Perform the General matrix rank-1 update for arbitrary data types 

  \b Purpose:	

  GERC  performs the rank 1 operation for arbitrary data types
  Data precisions supported include SINGLE/DOUBLE PRECISION COMPLEX(COMPLEX*16)

  A := alpha*x*y**T + A,
 
  where alpha is a scalar, x is an m element vector, y is an n element
  vector and A is an m by n matrix.

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] m
  m is INTEGER
  On entry,  m specifies the number of rows of the matrix A.
  m must be at least zero.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the number of columns of the matrix A.
  n must be at least zero.

  \param[in] alpha
  alpha is SINGLE/DOUBLE PRECISION COMPLEX
  On entry, alpha specifies the scalar alpha.

  \param[in] x
  x is SINGLE/DOUBLE PRECISION COMPLEX array,dimension : 
  at least ( 1 + ( m - 1 )*abs( incx ) ). \n
  Before entry, the incremented array x must contain the m
  element vector x.

  \param[in] incx
  incx is INTEGER
  On entry, incx specifies the increment for the elements of x.
  incx must not be zero.

  \param[in] y
  y is SINGLE/DOUBLE PRECISION COMPLEX array,dimension : 
  at least ( 1 + ( n - 1 )*abs( incy ) ). \n
  Before entry, the incremented array y must contain the n
  element vector y.

  \param[in] incy
  incy is INTEGER
  On entry, incy specifies the increment for the elements of y.
  incy must not be zero.

  \param[in,out] A
  A is SINGLE/DOUBLE PRECISION COMPLEX array,dimension ( lda, n )\n
  Before entry, the leading m by n part of the array A must
  contain the matrix of coefficients. On exit, A is
  overwritten by the updated matrix.  

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  lda must be at least max( 1, m ).
  */
template< typename T >
void gerc(
    CBLAS_ORDER layout,
    int64_t m, int64_t n,
    T alpha,
    T const *x, int64_t incx,
    T const *y, int64_t incy,
    T *A, int64_t lda )
{
    cblas_gerc(layout, m, n, alpha, x, incx, y, incy, A, lda);
}

/*! \brief Perform the hermitian rank 1 operation for arbitrary data types 

  \b Purpose:	

  HER  performs the hermitian rank 1 operation for arbitrary data types
  Data precisions supported include SINGLE/DOUBLE PRECISION COMPLEX(COMPLEX*16)

  A := alpha*x*x**H + A,
 
  where alpha is a real scalar, x is an n element vector, A is an n by n
  hermitian matrix.

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO.
  uplo specifies specifies whether the upper or lower triangular 
  part of the array A is to be referenced as follows: \n
  uplo = CBLAS_UPLO::CblasUpper   A is an upper triangular matrix. \n
  uplo = CBLAS_UPLO::CblasLower   A is a lower triangular matrix.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix A.
  n must be at least zero.

  \param[in] alpha
  alpha is SINGLE/DOUBLE PRECISION REAL
  On entry, alpha specifies the scalar alpha.

  \param[in] x
  x is SINGLE/DOUBLE PRECISION COMPLEX array,dimension : 
  at least ( 1 + ( n - 1 )*abs( incx ) ). \n
  Before entry, the incremented array x must contain the n
  element vector x.

  \param[in] incx
  incx is INTEGER
  On entry, incx specifies the increment for the elements of x.
  incx must not be zero.

  \param[in,out] A
  A is SINGLE/DOUBLE PRECISION COMPLEX array,dimension ( lda, n )\n
  Before entry with  UPLO = CblasUpper, the leading n by n
  upper triangular part of the array A must contain the upper
  triangular part of the hermitian matrix and the strictly
  lower triangular part of A is not referenced. On exit, the
  upper triangular part of the array A is overwritten by the
  upper triangular part of the updated matrix. \n
  Before entry with UPLO = CblasLower, the leading n by n
  lower triangular part of the array A must contain the lower
  triangular part of the hermitian matrix and the strictly
  upper triangular part of A is not referenced. On exit, the
  lower triangular part of the array A is overwritten by the
  lower triangular part of the updated matrix. \n
  Note that the imaginary parts of the diagonal elements need
  not be set, they are assumed to be zero, and on exit they
  are set to zero.  

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  lda must be at least max( 1, n ).
  */
template< typename T >
void her(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n,
    real_type<T> alpha,  // zher takes double alpha; use real
    T const *x, int64_t incx,
    T       *A, int64_t lda )
{
    cblas_her(layout, uplo, n, alpha, x, incx, A, lda);
}

/*! \brief Perform the hermitian rank 1 operation for arbitrary data types 

  \b Purpose:	

  HPR  performs the hermitian rank 1 operation for arbitrary data types
  Data precisions supported include SINGLE/DOUBLE PRECISION COMPLEX(COMPLEX*16)

  A := alpha*x*x**H + A,
 
  where alpha is a real scalar, x is an n element vector, A is an n by n
  hermitian matrix, supplied in packed form.

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO.
  uplo specifies specifies whether the upper or lower triangular 
  part of the array A is to be referenced as follows: \n
  uplo = CBLAS_UPLO::CblasUpper   The upper triangular part of A is
                                  supplied in Ap. \n
  uplo = CBLAS_UPLO::CblasLower   The lower triangular part of A is
                                  supplied in Ap.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix A.
  n must be at least zero.

  \param[in] alpha
  alpha is SINGLE/DOUBLE PRECISION REAL
  On entry, alpha specifies the scalar alpha.

  \param[in] x
  x is SINGLE/DOUBLE PRECISION COMPLEX array,dimension : 
  at least ( 1 + ( n - 1 )*abs( incx ) ). \n
  Before entry, the incremented array x must contain the n
  element vector x.

  \param[in] incx
  incx is INTEGER
  On entry, incx specifies the increment for the elements of x.
  incx must not be zero.

  \param[in,out] Ap
  Ap is SINGLE/DOUBLE PRECISION COMPLEX array,dimension 
  atleast ( ( n*( n + 1 ) )/2 ).\n
  Before entry with  UPLO = CblasUpper, the array Ap must
  contain the upper triangular part of the hermitian matrix
  packed sequentially, column by column, so that Ap( 1 )
  contains a( 1, 1 ), Ap( 2 ) and Ap( 3 ) contain a( 1, 2 )
  and a( 2, 2 ) respectively, and so on. On exit, the array
  Ap is overwritten by the upper triangular part of the
  updated matrix. \n
  Before entry with UPLO = CblasLower, the array Ap must
  contain the lower triangular part of the hermitian matrix
  packed sequentially, column by column, so that Ap( 1 )
  contains a( 1, 1 ), Ap( 2 ) and Ap( 3 ) contain a( 2, 1 )
  and a( 3, 1 ) respectively, and so on. On exit, the array
  Ap is overwritten by the lower triangular part of the
  updated matrix. \n
  Note that the imaginary parts of the diagonal elements need
  not be set, they are assumed to be zero, and on exit they
  are set to zero.  
  */
template< typename T >
void hpr(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n,
    real_type<T> alpha,  // zher takes double alpha; use real
    T const *x, int64_t incx,
    T       *Ap )
{
    cblas_hpr(layout, uplo, n, alpha, x, incx, Ap);
}

/*! \brief Perform the hermitian rank 2 operation for arbitrary data types 

  \b Purpose:	

  HER2  performs the hermitian rank 2 operation for arbitrary data types
  Data precisions supported include SINGLE/DOUBLE PRECISION COMPLEX(COMPLEX*16)

  A := alpha*x*y**H + conjg( alpha )*y*x**H + A,
 
  where alpha is a scalar, x and y are n element vector, A is an n by n
  hermitian matrix.

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO.
  uplo specifies whether the upper or lower triangular part of the 
  array A is to be referenced as follows: \n
  UPLO = CblasUpper   Only the upper triangular part of A
                      is to be referenced. \n
  UPLO = CblasLower   Only the lower triangular part of A
                      is to be referenced.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix A.
  n must be at least zero.

  \param[in] alpha
  alpha is SINGLE/DOUBLE PRECISION COMPLEX
  On entry, alpha specifies the scalar alpha.

  \param[in] x
  x is SINGLE/DOUBLE PRECISION COMPLEX array,dimension : 
  at least ( 1 + ( n - 1 )*abs( incx ) ). \n
  Before entry, the incremented array x must contain the n
  element vector x.

  \param[in] incx
  incx is INTEGER
  On entry, incx specifies the increment for the elements of x.
  incx must not be zero.

  \param[in] y
  y is SINGLE/DOUBLE PRECISION COMPLEX array,dimension : 
  at least ( 1 + ( n - 1 )*abs( incy ) ). \n
  Before entry, the incremented array y must contain the n
  element vector y.

  \param[in] incy
  incy is INTEGER
  On entry, incy specifies the increment for the elements of y.
  incy must not be zero.
  
  \param[in,out] A
  A is SINGLE/DOUBLE PRECISION COMPLEX array,dimension ( lda, n )\n
  Before entry with  UPLO = CblasUpper, the leading n by n
  upper triangular part of the array A must contain the upper
  triangular part of the hermitian matrix and the strictly
  lower triangular part of A is not referenced. On exit, the
  upper triangular part of the array A is overwritten by the
  upper triangular part of the updated matrix. \n
  Before entry with UPLO = CblasLower, the leading n by n
  lower triangular part of the array A must contain the lower
  triangular part of the hermitian matrix and the strictly
  upper triangular part of A is not referenced. On exit, the
  lower triangular part of the array A is overwritten by the
  lower triangular part of the updated matrix. \n
  Note that the imaginary parts of the diagonal elements need
  not be set, they are assumed to be zero, and on exit they
  are set to zero.  

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  lda must be at least max( 1, n ).
  */
template< typename T >
void her2(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n,
    T alpha,  
    T const *x, int64_t incx,
    T const *y, int64_t incy,
    T       *A, int64_t lda )
{
    cblas_her2(layout, uplo, n, alpha, x, incx, y, incy, A, lda);
}

/*! \brief Perform the hermitian rank 2 operation for arbitrary data types 

  \b Purpose:	

  HPR2  performs the hermitian rank 2 operation for arbitrary data types
  Data precisions supported include SINGLE/DOUBLE PRECISION COMPLEX(COMPLEX*16)

  A := alpha*x*y**H + conjg( alpha )*y*x**H + A,
 
  where alpha is a scalar, x and y are n element vector, A is an n by n
  hermitian matrix, supplied in packed form.

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO.
  uplo specifies specifies whether the upper or lower triangular 
  part of the array A is to be referenced as follows: \n
  uplo = CBLAS_UPLO::CblasUpper   The upper triangular part of A is
                                  supplied in Ap. \n
  uplo = CBLAS_UPLO::CblasLower   The lower triangular part of A is
                                  supplied in Ap.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix A.
  n must be at least zero.

  \param[in] alpha
  alpha is SINGLE/DOUBLE PRECISION COMPLEX
  On entry, alpha specifies the scalar alpha.

  \param[in] x
  x is SINGLE/DOUBLE PRECISION COMPLEX array,dimension : 
  at least ( 1 + ( n - 1 )*abs( incx ) ). \n
  Before entry, the incremented array x must contain the n
  element vector x.

  \param[in] incx
  incx is INTEGER
  On entry, incx specifies the increment for the elements of x.
  incx must not be zero.

  \param[in] y
  y is SINGLE/DOUBLE PRECISION REAL array,dimension : 
  at least ( 1 + ( n - 1 )*abs( incy ) ). \n
  Before entry, the incremented array y must contain the n
  element vector y.

  \param[in] incy
  incy is INTEGER
  On entry, incy specifies the increment for the elements of y.
  incy must not be zero.
  
  \param[in,out] Ap
  Ap is SINGLE/DOUBLE PRECISION COMPLEX array,dimension 
  atleast ( ( n*( n + 1 ) )/2 ).\n
  Before entry with  UPLO = CblasUpper, the array Ap must
  contain the upper triangular part of the hermitian matrix
  packed sequentially, column by column, so that Ap( 1 )
  contains a( 1, 1 ), Ap( 2 ) and Ap( 3 ) contain a( 1, 2 )
  and a( 2, 2 ) respectively, and so on. On exit, the array
  Ap is overwritten by the upper triangular part of the
  updated matrix. \n
  Before entry with UPLO = CblasLower, the array Ap must
  contain the lower triangular part of the hermitian matrix
  packed sequentially, column by column, so that Ap( 1 )
  contains a( 1, 1 ), Ap( 2 ) and Ap( 3 ) contain a( 2, 1 )
  and a( 3, 1 ) respectively, and so on. On exit, the array
  Ap is overwritten by the lower triangular part of the
  updated matrix. \n
  Note that the imaginary parts of the diagonal elements need
  not be set, they are assumed to be zero, and on exit they
  are set to zero.  
  */
template< typename T >
void hpr2(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n,
    T alpha,  
    T const *x, int64_t incx,
    T const *y, int64_t incy,
    T       *Ap )
{
    cblas_hpr2(layout, uplo, n, alpha, x, incx, y, incy, Ap);
}

/*! \brief Perform the symmetric rank 1 operation for arbitrary data types 

  \b Purpose:	

  SYR performs the symmetric rank 1 operation for arbitrary data types
  Data precisions supported include SINGLE/DOUBLE PRECISION REAL

  A := alpha*x*x**T + A,
 
  where alpha is a real scalar, x is an n element vector, A is an n by n
  symmetric matrix.

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO.
  uplo specifies specifies whether the upper or lower triangular 
  part of the array A is to be referenced as follows: \n
  uplo = CBLAS_UPLO::CblasUpper   A is an upper triangular matrix. \n
  uplo = CBLAS_UPLO::CblasLower   A is a lower triangular matrix.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix A.
  n must be at least zero.

  \param[in] alpha
  alpha is SINGLE/DOUBLE PRECISION REAL
  On entry, alpha specifies the scalar alpha.

  \param[in] x
  x is SINGLE/DOUBLE PRECISION REAL array,dimension : 
  at least ( 1 + ( n - 1 )*abs( incx ) ). \n
  Before entry, the incremented array x must contain the n
  element vector x.

  \param[in] incx
  incx is INTEGER
  On entry, incx specifies the increment for the elements of x.
  incx must not be zero.

  \param[in,out] A
  A is SINGLE/DOUBLE PRECISION REAL array,dimension ( lda, n )\n
  Before entry with  UPLO = CblasUpper, the leading n by n
  upper triangular part of the array A must contain the upper
  triangular part of the symmetric matrix and the strictly
  lower triangular part of A is not referenced. On exit, the
  upper triangular part of the array A is overwritten by the
  upper triangular part of the updated matrix. \n
  Before entry with UPLO = CblasLower, the leading n by n
  lower triangular part of the array A must contain the lower
  triangular part of the symmetric matrix and the strictly
  upper triangular part of A is not referenced. On exit, the
  lower triangular part of the array A is overwritten by the
  lower triangular part of the updated matrix. \n

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  lda must be at least max( 1, n ).
  */
template< typename T >
void syr(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n,
    T alpha,  
    T const *x, int64_t incx,
    T       *A, int64_t lda )
{
    cblas_syr(layout, uplo, n, alpha, x, incx, A, lda);
}

/*! \brief Perform the symmetric rank 1 operation for arbitrary data types 

  \b Purpose:	

  SPR  performs the symmetric rank 1 operation for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL

  A := alpha*x*x**T + A,
 
  where alpha is a real scalar, x is an n element vector, A is an n by n
  symmetric matrix, supplied in packed form.

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO.
  uplo specifies specifies whether the upper or lower triangular 
  part of the array A is to be referenced as follows: \n
  uplo = CBLAS_UPLO::CblasUpper   The upper triangular part of A is
                                  supplied in Ap. \n
  uplo = CBLAS_UPLO::CblasLower   The lower triangular part of A is
                                  supplied in Ap.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix A.
  n must be at least zero.

  \param[in] alpha
  alpha is SINGLE/DOUBLE PRECISION REAL
  On entry, alpha specifies the scalar alpha.

  \param[in] x
  x is SINGLE/DOUBLE PRECISION REAL array,dimension : 
  at least ( 1 + ( n - 1 )*abs( incx ) ). \n
  Before entry, the incremented array x must contain the n
  element vector x.

  \param[in] incx
  incx is INTEGER
  On entry, incx specifies the increment for the elements of x.
  incx must not be zero.

  \param[in,out] Ap
  Ap is SINGLE/DOUBLE PRECISION REAL array,dimension 
  atleast ( ( n*( n + 1 ) )/2 ).\n
  Before entry with  UPLO = CblasUpper, the array Ap must
  contain the upper triangular part of the symmetric matrix
  packed sequentially, column by column, so that Ap( 1 )
  contains a( 1, 1 ), Ap( 2 ) and Ap( 3 ) contain a( 1, 2 )
  and a( 2, 2 ) respectively, and so on. On exit, the array
  Ap is overwritten by the upper triangular part of the
  updated matrix. \n
  Before entry with UPLO = CblasLower, the array Ap must
  contain the lower triangular part of the symmetric matrix
  packed sequentially, column by column, so that Ap( 1 )
  contains a( 1, 1 ), Ap( 2 ) and Ap( 3 ) contain a( 2, 1 )
  and a( 3, 1 ) respectively, and so on. On exit, the array
  Ap is overwritten by the lower triangular part of the
  updated matrix. \n  
  */
template< typename T >
void spr(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n,
    T alpha,  
    T const *x, int64_t incx,
    T       *Ap )
{
    cblas_spr(layout, uplo, n, alpha, x, incx, Ap);
}

/*! \brief Perform the symmetric rank 2 operation for arbitrary data types 

  \b Purpose:	

  SYR2  performs the symmetric rank 2 operation for arbitrary data types
  Data precisions supported include SINGLE/DOUBLE PRECISION REAL

  A := alpha*x*y**T + alpha*y*x**T + A,
 
  where alpha is a scalar, x and y are n element vector, A is an n by n
  symmetric matrix.

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO.
  uplo specifies whether the upper or lower triangular part of the 
  array A is to be referenced as follows: \n
  UPLO = CblasUpper   Only the upper triangular part of A
                      is to be referenced. \n
  UPLO = CblasLower   Only the lower triangular part of A
                      is to be referenced.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix A.
  n must be at least zero.

  \param[in] alpha
  alpha is SINGLE/DOUBLE PRECISION REAL
  On entry, alpha specifies the scalar alpha.

  \param[in] x
  x is SINGLE/DOUBLE PRECISION REAL array,dimension : 
  at least ( 1 + ( n - 1 )*abs( incx ) ). \n
  Before entry, the incremented array x must contain the n
  element vector x.

  \param[in] incx
  incx is INTEGER
  On entry, incx specifies the increment for the elements of x.
  incx must not be zero.

  \param[in] y
  y is SINGLE/DOUBLE PRECISION REAL array,dimension : 
  at least ( 1 + ( n - 1 )*abs( incy ) ). \n
  Before entry, the incremented array y must contain the n
  element vector y.

  \param[in] incy
  incy is INTEGER
  On entry, incy specifies the increment for the elements of y.
  incy must not be zero.
  
  \param[in,out] A
  A is SINGLE/DOUBLE PRECISION REAL array,dimension ( lda, n )\n
  Before entry with  UPLO = CblasUpper, the leading n by n
  upper triangular part of the array A must contain the upper
  triangular part of the symmetric matrix and the strictly
  lower triangular part of A is not referenced. On exit, the
  upper triangular part of the array A is overwritten by the
  upper triangular part of the updated matrix. \n
  Before entry with UPLO = CblasLower, the leading n by n
  lower triangular part of the array A must contain the lower
  triangular part of the symmetric matrix and the strictly
  upper triangular part of A is not referenced. On exit, the
  lower triangular part of the array A is overwritten by the
  lower triangular part of the updated matrix. \n

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  lda must be at least max( 1, n ).
  */
template< typename T >
void syr2(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n,
    T alpha,
    T const *x, int64_t incx,
    T const *y, int64_t incy,
    T       *A, int64_t lda )
{
    cblas_syr2(layout, uplo, n, alpha, x, incx, y, incy, A, lda);
}

/*! \brief Perform the symmetric rank 2 operation for arbitrary data types 

  \b Purpose:	

  SPR2  performs the symmetric rank 2 operation for arbitrary data types
  Data precisions supported include SINGLE/DOUBLE PRECISION REAL

  A := alpha*x*y**T + alpha*y*x**T + A,
 
  where alpha is a scalar, x and y are n element vector, A is an n by n
  symmetric matrix, supplied in packed form.

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO.
  uplo specifies specifies whether the upper or lower triangular 
  part of the array A is to be referenced as follows: \n
  uplo = CBLAS_UPLO::CblasUpper   The upper triangular part of A is
                                  supplied in Ap. \n
  uplo = CBLAS_UPLO::CblasLower   The lower triangular part of A is
                                  supplied in Ap.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix A.
  n must be at least zero.

  \param[in] alpha
  alpha is SINGLE/DOUBLE PRECISION REAL
  On entry, alpha specifies the scalar alpha.

  \param[in] x
  x is SINGLE/DOUBLE PRECISION REAL array,dimension : 
  at least ( 1 + ( n - 1 )*abs( incx ) ). \n
  Before entry, the incremented array x must contain the n
  element vector x.

  \param[in] incx
  incx is INTEGER
  On entry, incx specifies the increment for the elements of x.
  incx must not be zero.

  \param[in] y
  y is SINGLE/DOUBLE PRECISION REAL array,dimension : 
  at least ( 1 + ( n - 1 )*abs( incy ) ). \n
  Before entry, the incremented array y must contain the n
  element vector y.

  \param[in] incy
  incy is INTEGER
  On entry, incy specifies the increment for the elements of y.
  incy must not be zero.
  
  \param[in,out] Ap
  Ap is SINGLE/DOUBLE PRECISION REAL array,dimension 
  atleast ( ( n*( n + 1 ) )/2 ).\n
  Before entry with  UPLO = CblasUpper, the array Ap must
  contain the upper triangular part of the symmetric matrix
  packed sequentially, column by column, so that Ap( 1 )
  contains a( 1, 1 ), Ap( 2 ) and Ap( 3 ) contain a( 1, 2 )
  and a( 2, 2 ) respectively, and so on. On exit, the array
  Ap is overwritten by the upper triangular part of the
  updated matrix. \n
  Before entry with UPLO = CblasLower, the array Ap must
  contain the lower triangular part of the symmetric matrix
  packed sequentially, column by column, so that Ap( 1 )
  contains a( 1, 1 ), Ap( 2 ) and Ap( 3 ) contain a( 2, 1 )
  and a( 3, 1 ) respectively, and so on. On exit, the array
  Ap is overwritten by the lower triangular part of the
  updated matrix. \n  
  */
template< typename T >
void spr2(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    int64_t n,
    T alpha,
    T const *x, int64_t incx,
    T const *y, int64_t incy,
    T       *Ap )
{
    cblas_spr2(layout, uplo, n, alpha, x, incx, y, incy, Ap);
}

/*! \brief General matrix-matrix multiply for arbitrary data types

  \b Purpose:	

  GEMM  performs general matrix-matrix multiply for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  C := alpha*op( A )*op( B ) + beta*C,

  where  op( X ) is one of

  op( X ) = X   or   op( X ) = X**T   or   op( X ) = X**H,

  alpha and beta are scalars, and A, B and C are matrices, with op( A )
  an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] transA
  transA is CBLAS_TRANSPOSE
  On entry, transA specifies the form of op( A ) to be used in
  the matrix multiplication as follows:
  transA = CBLAS_TRANSPOSE::CblasNoTrans,  op( A ) = A.
  transA = CBLAS_TRANSPOSE::CblasTrans,  op( A ) = A**T.
  transA = CBLAS_TRANSPOSE::CblasConjTrans,  op( A ) = A**H.

  \param[in] transB
  transB is CBLAS_TRANSPOSE
  On entry, transB specifies the form of op( B ) to be used in
  the matrix multiplication as follows:
  transB = CBLAS_TRANSPOSE::CblasNoTrans,  op( B ) = B.
  transB = CBLAS_TRANSPOSE::CblasTrans,  op( B ) = B**T.
  transB = CBLAS_TRANSPOSE::CblasConjTrans,  op( B ) = B**H.

  \param[in] m
  m is INTEGER
  On entry,  m  specifies  the number  of rows  of the  matrix
  op( A )  and of the  matrix  C.  m  must  be at least  zero.

  \param[in] n
  n is INTEGER
  On entry,  n  specifies the number  of columns of the matrix
  op( B ) and the number of columns of the matrix C. n must be
  at least zero.

  \param[in] k
  k is INTEGER
  On entry,  k  specifies  the number of columns of the matrix
  op( A ) and the number of rows of the matrix op( B ). k must
  be at least  zero.

  \param[in] alpha
  alpha is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.

  \param[in] A
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If transA = CblasNoTrans:
  m-by-k , stored in an lda-by-k array [RowMajor: m-by-lda].
  Otherwise:
  k-by-m , stored in an lda-by-m array [RowMajor: k-by-lda].

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  If transA = CblasNoTrans: lda >= max(1, m) [RowMajor: lda >= max(1, k)].
  Otherwise:                lda >= max(1, k) [RowMajor: lda >= max(1, m)].

  \param[in] B
  B is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If transA = CblasNoTrans:
  k-by-n , stored in an ldb-by-n array [RowMajor: k-by-ldb].
  Otherwise:
  n-by-k , stored in an ldb-by-k array [RowMajor: n-by-ldb].

  \param[in] ldb
  ldb is INTEGER
  On entry, ldb specifies the Leading dimension of B
  If transA = CblasNoTrans: ldb >= max(1, k) [RowMajor: ldb >= max(1, n)].
  Otherwise:                ldb >= max(1, n) [RowMajor: ldb >= max(1, k)].

  \param[in] beta
  beta is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, beta specifies the scalar alpha.When  beta  is
  supplied as zero then C need not be set on input.

  \param[in,out] C
  C is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array, dimension :
  m-by-n stored in an ldc-by-n array [RowMajor: m-by-ldc].
  Before entry, the leading  m by n  part of the array  C must
  contain the matrix  C,  except when  beta  is zero, in which
  case C need not be set on entry.
  On exit, the array  C  is overwritten by the  m by n  matrix
  ( alpha*op( A )*op( B ) + beta*C ).

  \param[in] ldc
  ldc is INTEGER
  On entry, ldc specifies the first dimension of C
  ldc >= max(1, m) [RowMajor: ldc >= max(1, n)].
  */	
template< typename T >
void gemm(
    CBLAS_ORDER layout,
    CBLAS_TRANSPOSE transA,
    CBLAS_TRANSPOSE transB,
    int64_t m, int64_t n, int64_t k,
    T alpha,
    T const *A, int64_t lda,
    T const *B, int64_t ldb,
    T beta,
    T       *C, int64_t ldc )
{
    cblas_gemm(layout, transA, transB, m, n, k, alpha, A,lda, B, ldb, beta, C, ldc);
}

/*! \brief Solve the triangular matrix-matrix equation for arbitrary data types 

  \b Purpose:	

  TRSM  performs  one of the matrix equations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,

  where alpha is a scalar, X and B are m by n matrices, A is a unit, or
  non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
  where  op( X ) is one of
      
  op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H.

  The matrix X is overwritten on B.

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] side
  side is enum CBLAS_SIDE
  side specifies specifies whether op( A ) appears on the left
  or right of X as follows:
  side = CBLAS_SIDE::CblasLeft   op( A )*X = alpha*B.
  side = CBLAS_SIDE::CblasRight   op( A )*X = alpha*B.

  \param[in] uplo
  uplo is enum CBLAS_UPLO
  uplo specifies specifies whether the matrix A is an upper or
  lower triangular matrix as follows:
  uplo = CBLAS_UPLO::CblasUpper   A is an upper triangular matrix.
  uplo = CBLAS_UPLO::CblasLower   A is a lower triangular matrix.
  
  \param[in] trans
  trans is CBLAS_TRANSPOSE
  On entry, trans specifies the form of op( A ) to be used in
  the matrix multiplication as follows:
  trans = CBLAS_TRANSPOSE::CblasNoTrans,  op( A ) = A.
  trans = CBLAS_TRANSPOSE::CblasTrans,  op( A ) = A**T.
  trans = CBLAS_TRANSPOSE::CblasConjTrans,  op( A ) = A**H.
  
  \param[in] diag
  diag is enum CBLAS_DIAG
  diag specifies specifies whether or not A is unit triangular
  as follows:
  diag = CBLAS_DIAG::CblasUnit   A is assumed to be unit triangular.
  diag = CBLAS_DIAG::CblasNonUnit   A is not assumed to be unit
                                 triangular.

  \param[in] m
  m is INTEGER
  On entry,  m  specifies  the number  of rows  of the  matrix
  B.  m  must  be at least  zero.

  \param[in] n
  n is INTEGER
  On entry,  n  specifies the number  of columns of the matrix
  B. n must be at least zero.

  \param[in] alpha
  alpha is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.

  \param[in] A
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If side = CblasLeft:
  the m-by-m matrix A, stored in an lda-by-m array [RowMajor: m-by-lda].
  If side = CblasRight:
  the n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  If side = CblasLeft: lda >= max(1, m) .
  If side = CblasRight:lda >= max(1, k) .

  \param[in,out] B
  B is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  m-by-n , stored in an ldb-by-n array [RowMajor: m-by-ldb]. 
  on exit  is overwritten by the solution matrix  X.

  \param[in] ldb
  ldb is INTEGER
  On entry, ldb specifies the Leading dimension of B
  ldb >= max(1, m) [RowMajor: ldb >= max(1, n)].
  */
template< typename T >
void trsm(
    CBLAS_ORDER layout,
    CBLAS_SIDE side,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    int64_t m,
    int64_t n,
    T alpha,
    T const *A, int64_t lda,
    T       *B, int64_t ldb )
{
    cblas_trsm( layout, side, uplo, trans, diag, m, n,  alpha, A, lda, B, ldb);
}
/*! \brief Solve the Triangular matrix-matrix multiply for arbitrary data types 

  \b Purpose:	

  TRMM  performs solves one of the matrix equations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  B := alpha*op( A )*B,   or   B := alpha*B*op( A ),

  where alpha is a scalar, B is an m by n matrices, A is a unit, or
  non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
  op( A ) = A   or   op( A ) = A**T.

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] side
  side is enum CBLAS_SIDE
  side specifies whether op( A ) multiplies B from left or right of X
  as follows:
  side = CBLAS_SIDE::CblasLeft   B := alpha*op( A )*B.
  side = CBLAS_SIDE::CblasRight  B := alpha*B*op( A ).

  \param[in] uplo
  uplo is enum CBLAS_UPLO
  uplo specifies whether the matrix A is an upper or lower triangular
  matrix as follows:
  uplo = CBLAS_UPLO::CblasUpper   A is an upper triangular matrix.
  uplo = CBLAS_UPLO::CblasLower   A is a lower triangular matrix.

  \param[in] trans
  trans is CBLAS_TRANSPOSE
  On entry, trans specifies the form of op( A ) to be used in
  the matrix multiplication as follows:
  trans = CBLAS_TRANSPOSE::CblasNoTrans,  op( A ) = A.
  trans = CBLAS_TRANSPOSE::CblasTrans,  op( A ) = A**T.
  trans = CBLAS_TRANSPOSE::CblasConjTrans,  op( A ) = A**T.

  \param[in] diag
  diag is enum CBLAS_DIAG
  diag specifies specifies whether or not A is unit triangular
  as follows:
  diag = CBLAS_DIAG::CblasUnit   A is assumed to be unit triangular.
  diag = CBLAS_DIAG::CblasNonUnit   A is not assumed to be unit
                                 triangular.

  \param[in] m
  m is INTEGER
  On entry,  m  specifies  the number  of rows  of the  matrix
  B.  m  must  be at least  zero.

  \param[in] n
  n is INTEGER
  On entry,  n  specifies the number  of columns of the matrix
  B. n must be at least zero.

  \param[in] alpha
  alpha is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.When  alpha is
  zero then  A is not referenced and  B need not be set before
  entry.

  \param[in] A
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If side = CblasLeft:
  the m-by-m matrix A, stored in an lda-by-m array [RowMajor: m-by-lda].
  If side = CblasRight:
  the n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  If side = CblasLeft: lda >= max(1, m) .
  If side = CblasRight:lda >= max(1, n) .

  \param[in,out] B
  B is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  m-by-n , stored in an ldb-by-n array [RowMajor: m-by-ldb].

  \param[in] ldb
  ldb is INTEGER
  On entry, ldb specifies the Leading dimension of B
  ldb >= max(1, m) [RowMajor: ldb >= max(1, n)].
  */
template< typename T >
void trmm(
    CBLAS_ORDER layout,
    CBLAS_SIDE side,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    int64_t m,
    int64_t n,
    T alpha,
    T const *A, int64_t lda,
    T       *B, int64_t ldb )
{
    cblas_trmm( layout, side, uplo, trans, diag, m, n,  alpha, A, lda, B, ldb);
}

/*! \brief Solve the Hermitian matrix-matrix multiply for arbitrary data types 

  \b Purpose:	

  HEMM  performs solves one of the matrix-matrix operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  C := alpha*A*B + beta*C

  or

  C := alpha*B*A + beta*C,

  where alpha is a scalar,  A is an hermitian matrix
  C and B are m by n matrices

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] side
  side is enum CBLAS_SIDE
  side specifies specifies whether the  hermitian matrix  A
  appears on the  left or right  in the  operation as follows:
  side = CBLAS_SIDE::CblasLeft   C := alpha*A*B + beta*C,
  side = CBLAS_SIDE::CblasRight   C := alpha*B*A + beta*C

  \param[in] uplo
  uplo is enum CBLAS_UPLO
  uplo specifies specifies whether  the  upper  or  lower
  triangular  part  of  the  hermitian  matrix   A  is  to  be
  referenced as follows:
  uplo = CBLAS_UPLO::CblasUpper   Only the upper triangular part of the
                                  hermitian matrix is to be referenced.
  uplo = CBLAS_UPLO::CblasLower   Only the lower triangular part of the
                                  hermitian matrix is to be referenced.

  \param[in] m
  m is INTEGER
  On entry,  m  specifies  the number  of rows  of the  matrix
  C.  m  must  be at least  zero.

  \param[in] n
  n is INTEGER
  On entry,  n  specifies the number  of columns of the matrix
  C. n must be at least zero.

  \param[in] alpha
  alpha is COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.

  \param[in] A
  A is COMPLEX/COMPLEX*16 array,dimension :
  If side = CblasLeft:
  the m-by-m matrix A, stored in an lda-by-m array [RowMajor: m-by-lda].
  If side = CblasRight:
  the n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  If side = CblasLeft: lda >= max(1, m) .
  If side = CblasRight:lda >= max(1, k) .

  \param[in] B
  B is COMPLEX/COMPLEX*16 array,dimension :
  m-by-n , stored in an ldb-by-n array [RowMajor: m-by-ldb].

  \param[in] ldb
  ldb is INTEGER
  On entry, ldb specifies the Leading dimension of B
  ldb >= max(1, m) [RowMajor: ldb >= max(1, n)].

  \param[in] beta
  beta is COMPLEX/COMPLEX*16
  On entry, beta specifies the scalar beta.
  If beta is zero, C need not be set on input

  \param[in,out] C
  C is COMPLEX/COMPLEX*16 array,dimension :
  m-by-n , stored in an ldc-by-n array [RowMajor: m-by-ldc].

  \param[in] ldc
  ldc is INTEGER
  On entry, ldc specifies the Leading dimension of C
  ldc >= max(1, m) [RowMajor: ldc >= max(1, n)].
  */
template< typename T >
void hemm(
    CBLAS_ORDER layout,
    CBLAS_SIDE side,
    CBLAS_UPLO uplo,
    int64_t m, int64_t n,
    T alpha,
    T const *A, int64_t lda,
    T const *B, int64_t ldb,
    T beta,
    T       *C, int64_t ldc )
{
    cblas_hemm( layout, side, uplo, m, n,  alpha, A, lda, B, ldb, beta, C, ldc);
}

/*! \brief Solve the Symmetric matrix-matrix multiply for arbitrary data types 

  \b Purpose:	
  
  SYMM  performs solves one of the matrix-matrix operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  C := alpha*A*B + beta*C

  or

  C := alpha*B*A + beta*C,

  where alpha is a scalar,  A is an symmetric matrix
  C and B are m by n matrices

  \param[in] layout
  layout is enum CBLAS_ORDER
  layout specifies Matrix storage as follows:
  layout = CBLAS_ORDER::CblasRowMajor or Layout::CblasColMajor.

  \param[in] side
  side is enum CBLAS_SIDE
  side specifies specifies whether the  symmetric matrix  A
  appears on the  left or right  in the  operation as follows:
  side = CBLAS_SIDE::CblasLeft   C := alpha*A*B + beta*C,
  side = CBLAS_SIDE::CblasRight   C := alpha*B*A + beta*C

  \param[in] uplo
  uplo is enum CBLAS_UPLO
  uplo specifies specifies whether  the  upper  or  lower
  triangular  part  of  the  symmetric  matrix   A  is  to  be
  referenced as follows:
  uplo = CBLAS_UPLO::CblasUpper   Only the upper triangular part of the
                                  symmetric matrix is to be referenced.
  uplo = CBLAS_UPLO::CblasLower   Only the lower triangular part of the
                                  symmetric matrix is to be referenced.

  \param[in] m
  m is INTEGER
  On entry,  m  specifies  the number  of rows  of the  matrix
  C.  m  must  be at least  zero.

  \param[in] n
  n is INTEGER
  On entry,  n  specifies the number  of columns of the matrix
  C. n must be at least zero.

  \param[in] alpha
  alpha is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.

  \param[in] A
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If side = CblasLeft:
  the m-by-m matrix A, stored in an lda-by-m array [RowMajor: m-by-lda].
  If side = CblasRight:
  the n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  If side = CblasLeft: lda >= max(1, m) .
  If side = CblasRight:lda >= max(1, k) .

  \param[in] B
  B is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  m-by-n , stored in an ldb-by-n array [RowMajor: m-by-ldb].

  \param[in] ldb
  ldb is INTEGER
  On entry, ldb specifies the Leading dimension of B
  ldb >= max(1, m) [RowMajor: ldb >= max(1, n)].

  \param[in] beta
  beta is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, beta specifies the scalar beta.
  If beta is zero, C need not be set on input

  \param[in, out] C
  C is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  m-by-n , stored in an ldc-by-n array [RowMajor: m-by-ldc].

  \param[in] ldc
  ldc is INTEGER
  On entry, ldc specifies the Leading dimension of C
  ldc >= max(1, m) [RowMajor: ldc >= max(1, n)].
  */
template< typename T >
void symm(
    CBLAS_ORDER layout,
    CBLAS_SIDE side,
    CBLAS_UPLO uplo,
    int64_t m, int64_t n,
    T alpha,
    T const *A, int64_t lda,
    T const *B, int64_t ldb,
    T beta,
    T       *C, int64_t ldc )
{
    cblas_symm( layout, side, uplo, m, n,  alpha, A, lda, B, ldb, beta, C, ldc);
}

/*! \brief Solve the Symmetric rank-k operations for arbitrary data types 

  \b Purpose:	

  SYRK  performs one of the symmetric rank k operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  C := alpha*A*A**T + beta*C,

  or

  C := alpha*A**T*A + beta*C,

  where  alpha and beta  are scalars,  C is an  n by n symmetric matrix
  and  A  is an  n by k  matrix in the first case and a  k by n  matrix
  in the second case.

  \param[in] layout
  layout is enum CBLAS_LAYOUT
  layout specifies Matrix storage as follows:
  layout = CBLAS_LAYOUT::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO
  uplo specifies specifies whether  the  upper  or  lower
  triangular  part  of  the  array C  is  to  be  referenced
  as follows:
  uplo = CBLAS_UPLO::CblasUpper   Only the upper triangular part of C
                                  is to be referenced.
  uplo = CBLAS_UPLO::CblasLower   Only the lower triangular part of C
                                  is to be referenced.

  \param[in] trans
  trans is CBLAS_TRANSPOSE
  On entry, trans specifies the operation to be used as follows:
  trans = CBLAS_TRANSPOSE::CblasNoTrans,C := alpha*A*A**T + beta*C.
  trans = CBLAS_TRANSPOSE::CblasTrans,C := alpha*A**T*A + beta*C.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix C.  n must be
  at least zero.

  \param[in] k
  k is INTEGER
  If trans = CblasNoTrans: k is number of columns of the matrix A.
  Otherwise:               k is number of rows    of the matrix A.
  k must be at least  zero.

  \param[in] alpha
  alpha is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.

  \param[in] A
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If transA = CblasNoTrans:
  n-by-k , stored in an lda-by-k array [RowMajor: n-by-lda].
  Otherwise:
  k-by-n , stored in an lda-by-n array [RowMajor: k-by-lda].

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  If transA = CblasNoTrans: lda >= max(1, n) [RowMajor: lda >= max(1, k)].
  Otherwise:                lda >= max(1, k) [RowMajor: lda >= max(1, n)].

  \param[in] beta
  beta is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, beta specifies the scalar alpha.When  beta  is
  supplied as zero then C need not be set on input.

  \param[in,out] C
  C is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array, dimension :
  The n-by-n symmetric matrix C,
  stored in an ldc-by-n array [RowMajor: n-by-ldc].
  On exit, the array  C  is overwritten by the  lower/upper
  triangular part of the updated matrix.

  \param[in] ldc
  ldc is INTEGER
  On entry, ldc specifies the first dimension of C
  ldc >= max(1, n)
  */
template< typename T >
void syrk(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    int64_t n, int64_t k,
    T alpha,
    T const *A, int64_t lda,
    T beta,
    T       *C, int64_t ldc )
{
    cblas_syrk( layout, uplo, trans,  n, k,  alpha, A, lda, beta, C, ldc);
}

/*! \brief Solve the Symmetric rank 2k operations for arbitrary data types 

  \b Purpose:	

  SYR2K  performs one of the symmetric rank 2k operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION REAL, DOUBLE PRECISION REAL,
  SINGLE PRECISION COMPLEX, DOUBLE PRECISION COMPLEX(COMPLEX*16)

  C := alpha*A*B**T + alpha*B*A**T + beta*C,

  or

  C := alpha*A**T*B + alpha*B**T*A + beta*C,

  where  alpha and beta  are scalars,  C is an  n by n symmetric matrix
  and  A  and B are n by k  matrices in the first case and k by n  matrices
  in the second case.

  \param[in] layout
  layout is enum CBLAS_LAYOUT
  layout specifies Matrix storage as follows:
  layout = CBLAS_LAYOUT::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO
  uplo specifies specifies whether  the  upper  or  lower
  triangular  part  of  the  array C  is  to  be  referenced
  as follows:
  uplo = CBLAS_UPLO::CblasUpper   Only the upper triangular part of C
                                  is to be referenced.
  uplo = CBLAS_UPLO::CblasLower   Only the lower triangular part of C
                                  is to be referenced.

  \param[in] trans
  trans is CBLAS_TRANSPOSE
  On entry, trans specifies the operation to be used as follows:
  trans = CBLAS_TRANSPOSE::CblasNoTrans,C := alpha*A*B**T + alpha*B*A**T + beta*C.
  trans = CBLAS_TRANSPOSE::CblasTrans,  C := alpha*A**T*B + alpha*B**T*A + beta*C.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix C.  n must be
  at least zero.

  \param[in] k
  k is INTEGER
  If trans = CblasNoTrans: k is number of columns of the matrices A & B.
  Otherwise:               k is number of rows    of the matrices A & B.
  k must be at least  zero.

  \param[in] alpha
  alpha is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.

  \param[in] A
  A is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If trans = CblasNoTrans:
  n-by-k , stored in an lda-by-k array [RowMajor: n-by-lda].
  Otherwise:
  k-by-n , stored in an lda-by-n array [RowMajor: k-by-lda].

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  If trans = CblasNoTrans: lda >= max(1, n) [RowMajor: lda >= max(1, k)].
  Otherwise:                lda >= max(1, k) [RowMajor: lda >= max(1, n)].

  \param[in] B
  B is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array,dimension :
  If trans = CblasNoTrans:
  n-by-k , stored in an ldb-by-k array [RowMajor: n-by-ldb].
  Otherwise:
  k-by-n , stored in an ldb-by-n array [RowMajor: k-by-ldb]

  \param[in] ldb
  ldb is INTEGER
  On entry, ldb specifies the Leading dimension of B
  If trans = CblasNoTrans: ldb >= max(1, n) [RowMajor: ldb >= max(1, k)].
  Otherwise:               ldb >= max(1, k) [RowMajor: ldb >= max(1, n)].

  \param[in] beta
  beta is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16
  On entry, beta specifies the scalar alpha.When  beta  is
  supplied as zero then C need not be set on input.

  \param[in,out] C
  C is REAL/DOUBLE PRECISION/COMPLEX/COMPLEX*16 array, dimension :
  The n-by-n symmetric matrix C,
  stored in an ldc-by-n array [RowMajor: n-by-ldc].
  On exit, the array  C  is overwritten by the  lower/upper
  triangular part of the updated matrix.

  \param[in] ldc
  ldc is INTEGER
  On entry, ldc specifies the first dimension of C
  ldc >= max(1, n)
  */
template< typename T >
void syr2k(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    int64_t n, int64_t k,
    T alpha,
    T const *A, int64_t lda,
    T const *B, int64_t ldb,
    T beta,
    T       *C, int64_t ldc )
{
    cblas_syr2k( layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc );	
}

/*! \brief Solve the Hermitian rank k operations for arbitrary data types 

  \b Purpose:	

  HERK  performs one of the hermitian rank k operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION COMPLEX, 
  DOUBLE PRECISION COMPLEX(COMPLEX*16)

    C := alpha*A*B**H + conjg( alpha )*B*A**H + beta*C,
  
  or
  
    C := alpha*A**H*B + conjg( alpha )*B**H*A + beta*C,

  where  alpha and beta  are real scalars,  C is an  n by n  hermitian 
  matrix and  A  is an n by k  matrix in the first case and 
  k by n  matrix in the second case.

  \param[in] layout
  layout is enum CBLAS_LAYOUT
  layout specifies Matrix storage as follows:
  layout = CBLAS_LAYOUT::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO
  uplo specifies specifies whether  the  upper  or  lower
  triangular  part  of  the  array C  is  to  be  referenced 
  as follows:
  uplo = CBLAS_UPLO::CblasUpper   Only the upper triangular part of C
                                  is to be referenced.
  uplo = CBLAS_UPLO::CblasLower   Only the lower triangular part of C
                                  is to be referenced.

  \param[in] trans
  trans is CBLAS_TRANSPOSE
  On entry, trans specifies the operation to be used as follows:
  trans = CBLAS_TRANSPOSE::CblasNoTrans,  C := alpha*A*A**H + beta*C.
  trans = CBLAS_TRANSPOSE::CblasConjTrans,C := alpha*A**H*A + beta*C.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix C.  n must be
  at least zero.

  \param[in] k
  k is INTEGER
  If trans = CblasNoTrans: k is number of columns of the matrix   A. 
  Otherwise:               k is number of rows    of the matrix   A.
  k must be at least  zero.

  \param[in] alpha
  alpha is REAL/DOUBLE PRECISION
  On entry, alpha specifies the scalar alpha.

  \param[in] A
  A is COMPLEX/COMPLEX*16 array,dimension :
  If trans = CblasNoTrans:
  n-by-k , stored in an lda-by-k array [RowMajor: n-by-lda].
  Otherwise:
  k-by-n , stored in an lda-by-n array [RowMajor: k-by-lda].

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  If trans = CblasNoTrans: lda >= max(1, n) [RowMajor: lda >= max(1, k)].
  Otherwise:                lda >= max(1, k) [RowMajor: lda >= max(1, n)].
  
  \param[in] beta
  beta is REAL/DOUBLE PRECISION
  On entry, beta specifies the scalar alpha.When  beta  is
  supplied as zero then C need not be set on input.

  \param[in,out] C
  C is COMPLEX/COMPLEX*16 array, dimension :
  The n-by-n Hermitian matrix C,
  stored in an ldc-by-n array [RowMajor: n-by-ldc].
  On exit, the array  C  is overwritten by the  lower/upper 
  triangular part of the updated matrix.

  \param[in] ldc
  ldc is INTEGER
  On entry, ldc specifies the first dimension of C
  ldc >= max(1, n) 
  */
template< typename T >
void herk(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    int64_t n, int64_t k,
    real_type<T> alpha,
    T const *A, int64_t lda,
    real_type<T> beta,
    T       *C, int64_t ldc )
{
    cblas_herk( layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc );	
}

/*! \brief Solve the Hermitian rank 2k operations for arbitrary data types 

  \b Purpose:	

  HER2K  performs one of the hermitian rank 2k operations for arbitrary data types
  Data precisions supported include SINGLE PRECISION COMPLEX,
  DOUBLE PRECISION COMPLEX(COMPLEX*16)

    C := alpha*A*B**H + conjg( alpha )*B*A**H + beta*C,
  
  or
  
    C := alpha*A**H*B + conjg( alpha )*B**H*A + beta*C,

  where  alpha and beta  are scalars with  beta  real,  C is an  n by n 
  hermitian matrix and  A  and B are n by k  matrices in the first case 
  and k by n  matrices in the second case.

  \param[in] layout
  layout is enum CBLAS_LAYOUT
  layout specifies Matrix storage as follows:
  layout = CBLAS_LAYOUT::CblasRowMajor or Layout::CblasColMajor.

  \param[in] uplo
  uplo is enum CBLAS_UPLO
  uplo specifies specifies whether  the  upper  or  lower
  triangular  part  of  the  array C  is  to  be  referenced 
  as follows:
  uplo = CBLAS_UPLO::CblasUpper   Only the upper triangular part of C
                                  is to be referenced.
  uplo = CBLAS_UPLO::CblasLower   Only the lower triangular part of C
                                  is to be referenced.

  \param[in] trans
  trans is CBLAS_TRANSPOSE
  On entry, trans specifies the operation to be used as follows:
  trans = CBLAS_TRANSPOSE::CblasNoTrans,  C := alpha*A*B**H + conjg( alpha )*B*A**H + beta*C.
  trans = CBLAS_TRANSPOSE::CblasConjTrans,C := alpha*A**H*B + conjg( alpha )*B**H*A + beta*C.

  \param[in] n
  n is INTEGER
  On entry,  n specifies the order of the matrix C.  n must be
  at least zero.

  \param[in] k
  k is INTEGER
  If trans = CblasNoTrans: k is number of columns of the matrices A & B. 
  Otherwise:               k is number of rows    of the matrices A & B.
  k must be at least  zero.

  \param[in] alpha
  alpha is COMPLEX/COMPLEX*16
  On entry, alpha specifies the scalar alpha.

  \param[in] A
  A is COMPLEX/COMPLEX*16 array,dimension :
  If trans = CblasNoTrans:
  n-by-k , stored in an lda-by-k array [RowMajor: n-by-lda].
  Otherwise:
  k-by-n , stored in an lda-by-n array [RowMajor: k-by-lda].

  \param[in] lda
  lda is INTEGER
  On entry, lda specifies the Leading dimension of A
  If trans = CblasNoTrans: lda >= max(1, n) [RowMajor: lda >= max(1, k)].
  Otherwise:                lda >= max(1, k) [RowMajor: lda >= max(1, n)].

  \param[in] B
  B is COMPLEX/COMPLEX*16 array,dimension :
  If trans = CblasNoTrans:
  n-by-k , stored in an ldb-by-k array [RowMajor: n-by-ldb].
  Otherwise:
  k-by-n , stored in an ldb-by-n array [RowMajor: k-by-ldb]

  \param[in] ldb
  ldb is INTEGER
  On entry, ldb specifies the Leading dimension of B
  If trans = CblasNoTrans: ldb >= max(1, n) [RowMajor: ldb >= max(1, k)].
  Otherwise:               ldb >= max(1, k) [RowMajor: ldb >= max(1, n)].
  
  \param[in] beta
  beta is REAL/DOUBLE PRECISION
  On entry, beta specifies the scalar alpha.When  beta  is
  supplied as zero then C need not be set on input.

  \param[in,out] C
  C is COMPLEX/COMPLEX*16 array, dimension :
  The n-by-n Hermitian matrix C,
  stored in an ldc-by-n array [RowMajor: n-by-ldc].
  On exit, the array  C  is overwritten by the  lower/upper 
  triangular part of the updated matrix.

  \param[in] ldc
  ldc is INTEGER
  On entry, ldc specifies the first dimension of C
  ldc >= max(1, n) 
  */
template< typename T >
void her2k(
    CBLAS_ORDER layout,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    int64_t n, int64_t k,
    T alpha,
    T const *A, int64_t lda,
    T const *B, int64_t ldb,
    real_type<T> beta,
    T       *C, int64_t ldc )
{
    cblas_her2k( layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc );	
}

}  // namespace blis
#endif        //  #ifndef BLIS_HH
