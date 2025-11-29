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

/*! @file cblas.hh
 *  cblas.hh defines all the overloaded CPP functions to be invoked from 
 *  template interfaces
 *  */
#ifndef CBLAS_HH
#define CBLAS_HH

extern "C" {
#include <blis.h>
}

#include <complex>

namespace blis{

template< typename... Types > struct real_type_traits;

//define real_type<> type alias
template< typename... Types >
using real_type = typename real_type_traits< Types... >::real_t;

// for one type
template< typename T >
struct real_type_traits<T>
{
    using real_t = T;
};

// for one complex type, strip complex
template< typename T >
struct real_type_traits< std::complex<T> >
{
    using real_t = T;
};

// =============================================================================
// Level 1 BLAS
// -----------------------------------------------------------------------------
inline void
cblas_rotg(
    float *a, float *b,
    float *c, float *s )
{
    cblas_srotg( a, b, c, s );
}

inline void
cblas_rotg(
    double *a, double *b,
    double *c, double *s )
{
    cblas_drotg( a, b, c, s );
}

// -----------------------------------------------------------------------------
inline void
cblas_rotmg(
    float *d1, float *d2, float *x1, float y1, float param[5] )
{
    cblas_srotmg( d1, d2, x1, y1, param );
}

inline void
cblas_rotmg(
    double *d1, double *d2, double *x1, double y1, double param[5] )
{
    cblas_drotmg( d1, d2, x1, y1, param );
}

// -----------------------------------------------------------------------------
inline void
cblas_rot(
    int n,
    float *x, int incx,
    float *y, int incy,
    float c, float s )
{
    cblas_srot( n, x, incx, y, incy, c, s );
}

inline void
cblas_rot(
    int n,
    double *x, int incx,
    double *y, int incy,
    double c, double s )
{
    cblas_drot( n, x, incx, y, incy, c, s );
}

// -----------------------------------------------------------------------------
inline void
cblas_rotm(
    int n,
    float *x, int incx,
    float *y, int incy,
    const float  p[5] )
{
    cblas_srotm( n, x, incx, y, incy, p );
}

inline void
cblas_rotm(
    int n,
    double *x, int incx,
    double *y, int incy,
    const double  p[5] )
{
    cblas_drotm( n, x, incx, y, incy, p );
}

// -----------------------------------------------------------------------------
inline void
cblas_swap(
    int n,
    float* x, int incx,
    float* y, int incy )
{
    cblas_sswap( n, x, incx, y, incy );
}

inline void
cblas_swap(
    int n,
    double* x, int incx,
    double* y, int incy )
{
    cblas_dswap( n, x, incx, y, incy );
}

inline void
cblas_swap(
    int n,
    std::complex<float>* x, int incx,
    std::complex<float>* y, int incy )
{
    cblas_cswap( n, x, incx, y, incy );
}

inline void
cblas_swap(
    int n,
    std::complex<double>* x, int incx,
    std::complex<double>* y, int incy )
{
    cblas_zswap( n, x, incx, y, incy );
}

// -----------------------------------------------------------------------------
inline void
cblas_scal(
    int n, float alpha,
    float* x, int incx )
{
    cblas_sscal( n, alpha, x, incx );
}

inline void
cblas_scal(
    int n, double alpha,
    double* x, int incx )
{
    cblas_dscal( n, alpha, x, incx );
}

inline void
cblas_scal(
    int n, std::complex<float> alpha,
    std::complex<float>* x, int incx )
{
    cblas_cscal( n, &alpha, x, incx );
}

inline void
cblas_scal(
    int n, std::complex<double> alpha,
    std::complex<double>* x, int incx )
{
    cblas_zscal( n, &alpha, x, incx );
}

inline void
cblas_scal(
    int n, float alpha,
    std::complex<float>* x, int incx )
{
    cblas_csscal( n, alpha, x, incx );
}

inline void
cblas_scal(
    int n, double alpha,
    std::complex<double>* x, int incx )
{
    cblas_zdscal( n, alpha, x, incx );
}

// -----------------------------------------------------------------------------
inline void
cblas_copy(
    int n,
    float const *x, int incx,
    float*       y, int incy )
{
    cblas_scopy( n, x, incx, y, incy );
}

inline void
cblas_copy(
    int n,
    double const *x, int incx,
    double*       y, int incy )
{
    cblas_dcopy( n, x, incx, y, incy );
}

inline void
cblas_copy(
    int n,
    std::complex<float> const *x, int incx,
    std::complex<float>*       y, int incy )
{
    cblas_ccopy( n, x, incx, y, incy );
}

inline void
cblas_copy(
    int n,
    std::complex<double> const *x, int incx,
    std::complex<double>*       y, int incy )
{
    cblas_zcopy( n, x, incx, y, incy );
}

// -----------------------------------------------------------------------------
inline void
cblas_axpy(
    int n, float alpha,
    float const *x, int incx,
    float*       y, int incy )
{
    cblas_saxpy( n, alpha, x, incx, y, incy );
}

inline void
cblas_axpy(
    int n, double alpha,
    double const *x, int incx,
    double*       y, int incy )
{
    cblas_daxpy( n, alpha, x, incx, y, incy );
}

inline void
cblas_axpy(
    int n, std::complex<float> alpha,
    std::complex<float> const *x, int incx,
    std::complex<float>*       y, int incy )
{
    cblas_caxpy( n, &alpha, x, incx, y, incy );
}

inline void
cblas_axpy(
    int n, std::complex<double> alpha,
    std::complex<double> const *x, int incx,
    std::complex<double>*       y, int incy )
{
    cblas_zaxpy( n, &alpha, x, incx, y, incy );
}

// -----------------------------------------------------------------------------
inline float
cblas_dot(
    int n,
    float const *x, int incx,
    float const *y, int incy )
{
    return cblas_sdot( n, x, incx, y, incy );
}

inline double
cblas_dot(
    int n,
    double const *x, int incx,
    double const *y, int incy )
{
    return cblas_ddot( n, x, incx, y, incy );
}
// -----------------------------------------------------------------------------
inline std::complex<float>
cblas_dotu(
    int n,
    std::complex<float> const *x, int incx,
    std::complex<float> const *y, int incy )
{
    std::complex<float> result;
    cblas_cdotu_sub( n, x, incx, y, incy, &result );
    return result;
}

inline std::complex<double>
cblas_dotu(
    int n,
    std::complex<double> const *x, int incx,
    std::complex<double> const *y, int incy )
{
    std::complex<double> result;
    cblas_zdotu_sub( n, x, incx, y, incy, &result );
    return result;
}

// -----------------------------------------------------------------------------
inline std::complex<float>
cblas_dotc(
    int n,
    std::complex<float> const *x, int incx,
    std::complex<float> const *y, int incy )
{
    std::complex<float> result;
    cblas_cdotc_sub( n, x, incx, y, incy, &result );
    return result;
}

inline std::complex<double>
cblas_dotc(
    int n,
    std::complex<double> const *x, int incx,
    std::complex<double> const *y, int incy )
{
    std::complex<double> result;
    cblas_zdotc_sub( n, x, incx, y, incy, &result );
    return result;
}

// -----------------------------------------------------------------------------
inline int
cblas_iamax(
    int n, float const *x, int incx )
{
    return cblas_isamax( n, x, incx );
}

inline int
cblas_iamax(
    int n, double const *x, int incx )
{
    return cblas_idamax( n, x, incx );
}

inline int
cblas_iamax(
    int n, std::complex<float> const *x, int incx )
{
    return cblas_icamax( n, x, incx );
}

inline int
cblas_iamax(
    int n, std::complex<double> const *x, int incx )
{
    return cblas_izamax( n, x, incx );
}


// -----------------------------------------------------------------------------
inline float
cblas_nrm2(
    int n, float const *x, int incx )
{
    return cblas_snrm2( n, x, incx );
}

inline double
cblas_nrm2(
    int n, double const *x, int incx )
{
    return cblas_dnrm2( n, x, incx );
}

inline float
cblas_nrm2(
    int n, std::complex<float> const *x, int incx )
{
    return cblas_scnrm2( n, x, incx );
}

inline double
cblas_nrm2(
    int n, std::complex<double> const *x, int incx )
{
    return cblas_dznrm2( n, x, incx );
}

// -----------------------------------------------------------------------------
inline float
cblas_asum(
    int n, float const *x, int incx )
{
    return cblas_sasum( n, x, incx );
}

inline double
cblas_asum(
    int n, double const *x, int incx )
{
    return cblas_dasum( n, x, incx );
}

inline float
cblas_asum(
    int n, std::complex<float> const *x, int incx )
{
    return cblas_scasum( n, x, incx );
}

inline double
cblas_asum(
    int n, std::complex<double> const *x, int incx )
{
    return cblas_dzasum( n, x, incx );
}
// =============================================================================
// Level 2 BLAS

// -----------------------------------------------------------------------------
inline void
cblas_gemv(
    CBLAS_ORDER layout, CBLAS_TRANSPOSE trans, int m, int n,
    float  alpha,
    float const *A, int lda,
    float const *x, int incx,
    float  beta,
    float* y, int incy )
{
    cblas_sgemv( layout, trans, m, n,
                 alpha, A, lda, x, incx, beta, y, incy );
}

inline void
cblas_gemv(
    CBLAS_ORDER layout, CBLAS_TRANSPOSE trans, int m, int n,
    double  alpha,
    double const *A, int lda,
    double const *x, int incx,
    double  beta,
    double* y, int incy )
{
    cblas_dgemv( layout, trans, m, n,
                 alpha, A, lda, x, incx, beta, y, incy );
}

inline void
cblas_gemv(
    CBLAS_ORDER layout, CBLAS_TRANSPOSE trans, int m, int n,
    std::complex<float>  alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *x, int incx,
    std::complex<float>  beta,
    std::complex<float>* y, int incy )
{
    cblas_cgemv( layout, trans, m, n,
                 &alpha, A, lda, x, incx,
                 &beta, y, incy );
}

inline void
cblas_gemv(
    CBLAS_ORDER layout, CBLAS_TRANSPOSE trans, int m, int n,
    std::complex<double>  alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *x, int incx,
    std::complex<double>  beta,
    std::complex<double>* y, int incy )
{
    cblas_zgemv( layout, trans, m, n,
                 &alpha, A, lda, x, incx,
                 &beta, y, incy );
}
inline void
cblas_gbmv(
    CBLAS_ORDER layout, CBLAS_TRANSPOSE trans,
    int m, int n, int kl, int ku,
    float  alpha,
    float const *A, int lda,
    float const *x, int incx,
    float  beta,
    float* y, int incy )
{
    cblas_sgbmv( layout, trans, m, n, kl, ku,
                 alpha, A, lda, x, incx, beta, y, incy );
}

inline void
cblas_gbmv(
    CBLAS_ORDER layout, CBLAS_TRANSPOSE trans,
    int m, int n, int kl, int ku,
    double  alpha,
    double const *A, int lda,
    double const *x, int incx,
    double  beta,
    double* y, int incy )
{
    cblas_dgbmv( layout, trans, m, n, kl, ku,
                 alpha, A, lda, x, incx, beta, y, incy );
}

inline void
cblas_gbmv(
    CBLAS_ORDER layout, CBLAS_TRANSPOSE trans,
    int m, int n, int kl, int ku,
    std::complex<float>  alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *x, int incx,
    std::complex<float>  beta,
    std::complex<float>* y, int incy )
{
    cblas_cgbmv( layout, trans, m, n, kl, ku,
                 &alpha, A, lda, x, incx,
                 &beta, y, incy );
}

inline void
cblas_gbmv(
    CBLAS_ORDER layout, CBLAS_TRANSPOSE trans, 
    int m, int n, int kl, int ku,
    std::complex<double>  alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *x, int incx,
    std::complex<double>  beta,
    std::complex<double>* y, int incy )
{
    cblas_zgbmv( layout, trans, m, n, kl, ku,
                 &alpha, A, lda, x, incx,
                 &beta, y, incy );
}

// -----------------------------------------------------------------------------
inline void
cblas_hemv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n,
    std::complex<float>  alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *x, int incx,
    std::complex<float>  beta,
    std::complex<float>* y, int incy )
{
    cblas_chemv( layout, uplo, n,
                 &alpha, A, lda, x, incx,
                 &beta, y, incy );
}

inline void
cblas_hemv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n,
    std::complex<double>  alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *x, int incx,
    std::complex<double>  beta,
    std::complex<double>* y, int incy )
{
    cblas_zhemv( layout, uplo, n,
                 &alpha, A, lda, x, incx,
                 &beta, y, incy );
}

// -----------------------------------------------------------------------------
inline void
cblas_hbmv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n, int k,
    std::complex<float>  alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *x, int incx,
    std::complex<float>  beta,
    std::complex<float>* y, int incy )
{
    cblas_chbmv( layout, uplo, n, k,
                 &alpha, A, lda, x, incx,
                 &beta, y, incy );
}

inline void
cblas_hbmv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n, int k,
    std::complex<double>  alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *x, int incx,
    std::complex<double>  beta,
    std::complex<double>* y, int incy )
{
    cblas_zhbmv( layout, uplo, n, k,
                 &alpha, A, lda, x, incx,
                 &beta, y, incy );
}

// -----------------------------------------------------------------------------
inline void
cblas_hpmv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n,
    std::complex<float>  alpha,
    std::complex<float> const *Ap,
    std::complex<float> const *x, int incx,
    std::complex<float>  beta,
    std::complex<float>* y, int incy )
{
    cblas_chpmv( layout, uplo, n,
                 &alpha, Ap, x, incx,
                 &beta, y, incy );
}

inline void
cblas_hpmv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n,
    std::complex<double>  alpha,
    std::complex<double> const *Ap,
    std::complex<double> const *x, int incx,
    std::complex<double>  beta,
    std::complex<double>* y, int incy )
{
    cblas_zhpmv( layout, uplo, n,
                 &alpha, Ap, x, incx,
                 &beta, y, incy );
}

// -----------------------------------------------------------------------------
inline void
cblas_symv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n,
    float  alpha,
    float const *A, int lda,
    float const *x, int incx,
    float  beta,
    float* y, int incy )
{
    cblas_ssymv( layout, uplo, n,
                 alpha, A, lda, x, incx, beta, y, incy );
}

inline void
cblas_symv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n,
    double  alpha,
    double const *A, int lda,
    double const *x, int incx,
    double  beta,
    double* y, int incy )
{
    cblas_dsymv( layout, uplo, n,
                 alpha, A, lda, x, incx, beta, y, incy );
}

// -----------------------------------------------------------------------------
inline void
cblas_sbmv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n, int k,
    float  alpha,
    float const *A, int lda,
    float const *x, int incx,
    float  beta,
    float* y, int incy )
{
    cblas_ssbmv( layout, uplo, n, k,
                 alpha, A, lda, x, incx, beta, y, incy );
}

inline void
cblas_sbmv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n, int k,
    double  alpha,
    double const *A, int lda,
    double const *x, int incx,
    double  beta,
    double* y, int incy )
{
    cblas_dsbmv( layout, uplo, n, k,
                 alpha, A, lda, x, incx, beta, y, incy );
}

// -----------------------------------------------------------------------------
inline void
cblas_spmv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n,
    float  alpha,
    float const *Ap,
    float const *x, int incx,
    float  beta,
    float* y, int incy )
{
    cblas_sspmv( layout, uplo, n,
                 alpha, Ap, x, incx, beta, y, incy );
}

inline void
cblas_spmv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n,
    double  alpha,
    double const *Ap,
    double const *x, int incx,
    double  beta,
    double* y, int incy )
{
    cblas_dspmv( layout, uplo, n,
                 alpha, Ap, x, incx, beta, y, incy );
}

// -----------------------------------------------------------------------------
inline void
cblas_trmv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    float const *A, int lda,
    float* x, int incx )
{
    cblas_strmv( layout, uplo, trans, diag, n,
                 A, lda, x, incx );
}

inline void
cblas_trmv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    double const *A, int lda,
    double* x, int incx )
{
    cblas_dtrmv( layout, uplo, trans, diag, n,
                 A, lda, x, incx );
}

inline void
cblas_trmv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    std::complex<float> const *A, int lda,
    std::complex<float>* x, int incx )
{
    cblas_ctrmv( layout, uplo, trans, diag, n,
                 A, lda, x, incx );
}

inline void
cblas_trmv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    std::complex<double> const *A, int lda,
    std::complex<double>* x, int incx )
{
    cblas_ztrmv( layout, uplo, trans, diag, n,
                 A, lda, x, incx );
}

// -----------------------------------------------------------------------------
inline void
cblas_tbmv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, 
    int n, int k,
    float const *A, int lda,
    float* x, int incx )
{
    cblas_stbmv( layout, uplo, trans, diag, n, k,
                 A, lda, x, incx );
}

inline void
cblas_tbmv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
    int n, int k,
    double const *A, int lda,
    double* x, int incx )
{
    cblas_dtbmv( layout, uplo, trans, diag, n, k,
                 A, lda, x, incx );
}

inline void
cblas_tbmv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, 
    int n, int k,
    std::complex<float> const *A, int lda,
    std::complex<float>* x, int incx )
{
    cblas_ctbmv( layout, uplo, trans, diag, n, k,
                 A, lda, x, incx );
}

inline void
cblas_tbmv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
    int n, int k,
    std::complex<double> const *A, int lda,
    std::complex<double>* x, int incx )
{
    cblas_ztbmv( layout, uplo, trans, diag, n, k,
                 A, lda, x, incx );
}

// -----------------------------------------------------------------------------
inline void
cblas_tpmv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    float const *Ap,
    float* x, int incx )
{
    cblas_stpmv( layout, uplo, trans, diag, n,
                 Ap, x, incx );
}

inline void
cblas_tpmv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    double const *Ap,
    double* x, int incx )
{
    cblas_dtpmv( layout, uplo, trans, diag, n,
                 Ap, x, incx );
}

inline void
cblas_tpmv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    std::complex<float> const *Ap,
    std::complex<float>* x, int incx )
{
    cblas_ctpmv( layout, uplo, trans, diag, n,
                 Ap, x, incx );
}

inline void
cblas_tpmv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    std::complex<double> const *Ap,
    std::complex<double>* x, int incx )
{
    cblas_ztpmv( layout, uplo, trans, diag, n,
                 Ap, x, incx );
}

// -----------------------------------------------------------------------------
inline void
cblas_trsv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    float const *A, int lda,
    float* x, int incx )
{
    cblas_strsv( layout, uplo, trans, diag, n,
                 A, lda, x, incx );
}

inline void
cblas_trsv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    double const *A, int lda,
    double* x, int incx )
{
    cblas_dtrsv( layout, uplo, trans, diag, n,
                 A, lda, x, incx );
}

inline void
cblas_trsv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    std::complex<float> const *A, int lda,
    std::complex<float>* x, int incx )
{
    cblas_ctrsv( layout, uplo, trans, diag, n,
                 A, lda, x, incx );
}

inline void
cblas_trsv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    std::complex<double> const *A, int lda,
    std::complex<double>* x, int incx )
{
    cblas_ztrsv( layout, uplo, trans, diag, n,
                 A, lda, x, incx );
}

// -----------------------------------------------------------------------------
inline void
cblas_tbsv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
    int n, int k,
    float const *A, int lda,
    float* x, int incx )
{
    cblas_stbsv( layout, uplo, trans, diag, n, k,
                 A, lda, x, incx );
}

inline void
cblas_tbsv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
    int n, int k,
    double const *A, int lda,
    double* x, int incx )
{
    cblas_dtbsv( layout, uplo, trans, diag, n, k,
                 A, lda, x, incx );
}

inline void
cblas_tbsv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
    int n, int k,
    std::complex<float> const *A, int lda,
    std::complex<float>* x, int incx )
{
    cblas_ctbsv( layout, uplo, trans, diag, n, k,
                 A, lda, x, incx );
}

inline void
cblas_tbsv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
    int n, int k,
    std::complex<double> const *A, int lda,
    std::complex<double>* x, int incx )
{
    cblas_ztbsv( layout, uplo, trans, diag, n, k,
                 A, lda, x, incx );
}

// -----------------------------------------------------------------------------
inline void
cblas_tpsv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    float const *Ap,
    float* x, int incx )
{
    cblas_stpsv( layout, uplo, trans, diag, n,
                 Ap, x, incx );
}

inline void
cblas_tpsv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    double const *Ap,
    double* x, int incx )
{
    cblas_dtpsv( layout, uplo, trans, diag, n,
                 Ap, x, incx );
}

inline void
cblas_tpsv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    std::complex<float> const *Ap,
    std::complex<float>* x, int incx )
{
    cblas_ctpsv( layout, uplo, trans, diag, n,
                 Ap, x, incx );
}

inline void
cblas_tpsv(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    std::complex<double> const *Ap,
    std::complex<double>* x, int incx )
{
    cblas_ztpsv( layout, uplo, trans, diag, n,
                 Ap, x, incx );
}

// -----------------------------------------------------------------------------
inline void
cblas_ger(
    CBLAS_ORDER layout, int m, int n,
    float alpha,
    float const *x, int incx,
    float const *y, int incy,
    float* A, int lda )
{
    cblas_sger( layout, m, n, alpha, x, incx, y, incy, A, lda );
}

inline void
cblas_ger(
    CBLAS_ORDER layout, int m, int n,
    double alpha,
    double const *x, int incx,
    double const *y, int incy,
    double* A, int lda )
{
    cblas_dger( layout, m, n, alpha, x, incx, y, incy, A, lda );
}

// -----------------------------------------------------------------------------
inline void
cblas_geru(
    CBLAS_ORDER layout, int m, int n,
    std::complex<float> alpha,
    std::complex<float> const *x, int incx,
    std::complex<float> const *y, int incy,
    std::complex<float>* A, int lda )
{
    cblas_cgeru( layout, m, n, &alpha,
                 x, incx, y, incy, A, lda );
}

inline void
cblas_geru(
    CBLAS_ORDER layout, int m, int n,
    std::complex<double> alpha,
    std::complex<double> const *x, int incx,
    std::complex<double> const *y, int incy,
    std::complex<double>* A, int lda )
{
    cblas_zgeru( layout, m, n, &alpha,
                 x, incx, y, incy, A, lda );
}

// -----------------------------------------------------------------------------
inline void
cblas_gerc(
    CBLAS_ORDER layout, int m, int n,
    std::complex<float> alpha,
    std::complex<float> const *x, int incx,
    std::complex<float> const *y, int incy,
    std::complex<float>* A, int lda )
{
    cblas_cgerc( layout, m, n, &alpha,
                 x, incx, y, incy, A, lda );
}

inline void
cblas_gerc(
    CBLAS_ORDER layout, int m, int n,
    std::complex<double> alpha,
    std::complex<double> const *x, int incx,
    std::complex<double> const *y, int incy,
    std::complex<double>* A, int lda )
{
    cblas_zgerc( layout, m, n, &alpha,
                 x, incx, y, incy, A, lda );
}

// -----------------------------------------------------------------------------
inline void
cblas_her(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n,
    float alpha,
    std::complex<float> const *x, int incx,
    std::complex<float>* A, int lda )
{
    cblas_cher( layout, uplo, n, alpha, x, incx, A, lda );
}

inline void
cblas_her(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n,
    double alpha,
    std::complex<double> const *x, int incx,
    std::complex<double>* A, int lda )
{
    cblas_zher( layout, uplo, n, alpha, x, incx, A, lda );
}

// -----------------------------------------------------------------------------
inline void
cblas_hpr(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n,
    float alpha,
    std::complex<float> const *x, int incx,
    std::complex<float>* Ap )
{
    cblas_chpr( layout, uplo, n, alpha, x, incx, Ap );
}

inline void
cblas_hpr(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n,
    double alpha,
    std::complex<double> const *x, int incx,
    std::complex<double>* Ap )
{
    cblas_zhpr( layout, uplo, n, alpha, x, incx, Ap );
}

// -----------------------------------------------------------------------------
inline void
cblas_her2(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n,
    std::complex<float> alpha,
    std::complex<float> const *x, int incx,
    std::complex<float> const *y, int incy,
    std::complex<float>* A, int lda )
{
    cblas_cher2( layout, uplo, n, &alpha, x, incx, y, incy, A, lda );
}

inline void
cblas_her2(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n,
    std::complex<double> alpha,
    std::complex<double> const *x, int incx,
    std::complex<double> const *y, int incy,
    std::complex<double>* A, int lda )
{
    cblas_zher2( layout, uplo, n, &alpha, x, incx, y, incy, A, lda );
}

// -----------------------------------------------------------------------------
inline void
cblas_hpr2(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n,
    std::complex<float> alpha,
    std::complex<float> const *x, int incx,
    std::complex<float> const *y, int incy,
    std::complex<float>* Ap )
{
    cblas_chpr2( layout, uplo, n, &alpha, x, incx, y, incy, Ap );
}

inline void
cblas_hpr2(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n,
    std::complex<double> alpha,
    std::complex<double> const *x, int incx,
    std::complex<double> const *y, int incy,
    std::complex<double>* Ap )
{
    cblas_zhpr2( layout, uplo, n, &alpha, x, incx, y, incy, Ap );
}
// -----------------------------------------------------------------------------
inline void
cblas_syr(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n,
    float alpha,
    float const *x, int incx,
    float* A, int lda )
{
    cblas_ssyr( layout, uplo, n, alpha, x, incx, A, lda );
}

inline void
cblas_syr(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n,
    double alpha,
    double const *x, int incx,
    double* A, int lda )
{
    cblas_dsyr( layout, uplo, n, alpha, x, incx, A, lda );
}

// -----------------------------------------------------------------------------
inline void
cblas_spr(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n,
    float alpha,
    float const *x, int incx,
    float* Ap )
{
    cblas_sspr( layout, uplo, n, alpha, x, incx, Ap );
}

inline void
cblas_spr(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n,
    double alpha,
    double const *x, int incx,
    double* Ap )
{
    cblas_dspr( layout, uplo, n, alpha, x, incx, Ap );
}

// -----------------------------------------------------------------------------
inline void
cblas_syr2(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n,
    float alpha,
    float const *x, int incx,
    float const *y, int incy,
    float* A, int lda )
{
    cblas_ssyr2( layout, uplo, n, alpha, x, incx, y, incy, A, lda );
}

inline void
cblas_syr2(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n,
    double alpha,
    double const *x, int incx,
    double const *y, int incy,
    double* A, int lda )
{
    cblas_dsyr2( layout, uplo, n, alpha, x, incx, y, incy, A, lda );
}

// -----------------------------------------------------------------------------
inline void
cblas_spr2(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n,
    float alpha,
    float const *x, int incx,
    float const *y, int incy,
    float* Ap )
{
    cblas_sspr2( layout, uplo, n, alpha, x, incx, y, incy, Ap );
}

inline void
cblas_spr2(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, int n,
    double alpha,
    double const *x, int incx,
    double const *y, int incy,
    double* Ap )
{
    cblas_dspr2( layout, uplo, n, alpha, x, incx, y, incy, Ap );
}

// =============================================================================
// Level 3 BLAS

// -----------------------------------------------------------------------------
inline void
cblas_gemm(
    CBLAS_ORDER layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
    int m, int n, int k,
    float  alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float  beta,
    float* C, int ldc )
{
    cblas_sgemm( layout, transA, transB, m, n, k,
                 alpha, A, lda, B, ldb,
                 beta,  C, ldc );
}

inline void
cblas_gemm(
    CBLAS_ORDER layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
    int m, int n, int k,
    double  alpha,
    double const *A, int lda,
    double const *B, int ldb,
    double  beta,
    double* C, int ldc )
{
    cblas_dgemm( layout, transA, transB, m, n, k,
                 alpha, A, lda, B, ldb,
                 beta,  C, ldc );
}

inline void
cblas_gemm(
    CBLAS_ORDER layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
    int m, int n, int k,
    std::complex<float>  alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *B, int ldb,
    std::complex<float>  beta,
    std::complex<float>* C, int ldc )
{
    cblas_cgemm( layout, transA, transB, m, n, k,
                 &alpha, A, lda, B, ldb,
                 &beta,  C, ldc );
}

inline void
cblas_gemm(
    CBLAS_ORDER layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
    int m, int n, int k,
    std::complex<double>  alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *B, int ldb,
    std::complex<double>  beta,
    std::complex<double>* C, int ldc )
{
    cblas_zgemm( layout, transA, transB, m, n, k,
                 &alpha, A, lda, B, ldb,
                 &beta,  C, ldc );
}

// -----------------------------------------------------------------------------
inline void
cblas_trmm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
    int m, int n,
    float alpha,
    float const *A, int lda,
    float       *B, int ldb )
{
    cblas_strmm( layout, side, uplo, trans, diag, m, n,  alpha, A, lda, B, ldb);
}

inline void
cblas_trmm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
    int m, int n,
    double alpha,
    double const *A, int lda,
    double       *B, int ldb )
{
    cblas_dtrmm( layout, side, uplo, trans, diag, m, n,  alpha, A, lda, B, ldb);
}

inline void
cblas_trmm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
    int m, int n,
    std::complex<float> alpha,
    std::complex<float> const *A, int lda,
    std::complex<float>       *B, int ldb )
{
    cblas_ctrmm( layout, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb );
}

inline void
cblas_trmm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
    int m, int n,
    std::complex<double> alpha,
    std::complex<double> const *A, int lda,
    std::complex<double>       *B, int ldb )
{
    cblas_ztrmm( layout, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb );
}


// -----------------------------------------------------------------------------
inline void
cblas_trsm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
    int m, int n,
    float alpha,
    float const *A, int lda,
    float       *B, int ldb )
{
    cblas_strsm( layout, side, uplo, trans, diag, m, n,  alpha, A, lda, B, ldb);
}

inline void
cblas_trsm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
    int m, int n,
    double alpha,
    double const *A, int lda,
    double       *B, int ldb )
{
    cblas_dtrsm( layout, side, uplo, trans, diag, m, n,  alpha, A, lda, B, ldb);
}

inline void
cblas_trsm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
    int m, int n,
    std::complex<float> alpha,
    std::complex<float> const *A, int lda,
    std::complex<float>       *B, int ldb )
{
    cblas_ctrsm( layout, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb );
}

inline void
cblas_trsm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
    int m, int n,
    std::complex<double> alpha,
    std::complex<double> const *A, int lda,
    std::complex<double>       *B, int ldb )
{
    cblas_ztrsm( layout, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb );
}

// -----------------------------------------------------------------------------
inline void
cblas_hemm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    float  alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float  beta,
    float* C, int ldc )
{
    cblas_ssymm( layout, side, uplo, m, n,
                 alpha, A, lda, B, ldb,
                 beta,  C, ldc );
}

inline void
cblas_hemm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    double  alpha,
    double const *A, int lda,
    double const *B, int ldb,
    double  beta,
    double* C, int ldc )
{
    cblas_dsymm( layout, side, uplo, m, n,
                 alpha, A, lda, B, ldb,
                 beta,  C, ldc );
}

inline void
cblas_hemm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    std::complex<float>  alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *B, int ldb,
    std::complex<float>  beta,
    std::complex<float>* C, int ldc )
{
    cblas_chemm( layout, side, uplo, m, n,
                 &alpha, A, lda, B, ldb,
                 &beta,  C, ldc );
}

inline void
cblas_hemm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    std::complex<double>  alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *B, int ldb,
    std::complex<double>  beta,
    std::complex<double>* C, int ldc )
{
    cblas_zhemm( layout, side, uplo, m, n,
                 &alpha, A, lda, B, ldb,
                 &beta,  C, ldc );
}

// -----------------------------------------------------------------------------
inline void
cblas_symm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    float  alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float  beta,
    float* C, int ldc )
{
    cblas_ssymm( layout, side, uplo, m, n,
                 alpha, A, lda, B, ldb,
                 beta,  C, ldc );
}

inline void
cblas_symm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    double  alpha,
    double const *A, int lda,
    double const *B, int ldb,
    double  beta,
    double* C, int ldc )
{
    cblas_dsymm( layout, side, uplo, m, n,
                 alpha, A, lda, B, ldb,
                 beta,  C, ldc );
}

inline void
cblas_symm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    std::complex<float>  alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *B, int ldb,
    std::complex<float>  beta,
    std::complex<float>* C, int ldc )
{
    cblas_csymm( layout, side, uplo, m, n,
                 &alpha, A, lda, B, ldb,
                 &beta,  C, ldc );
}

inline void
cblas_symm(
    CBLAS_ORDER layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    std::complex<double>  alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *B, int ldb,
    std::complex<double>  beta,
    std::complex<double>* C, int ldc )
{
    cblas_zsymm( layout, side, uplo, m, n,
                 &alpha, A, lda, B, ldb,
                 &beta,  C, ldc );
}


// -----------------------------------------------------------------------------
inline void
cblas_syrk(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    float alpha,
    float const *A, int lda,
    float beta,
    float*       C, int ldc )
{
    cblas_ssyrk( layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc );
}

inline void
cblas_syrk(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    double alpha,
    double const *A, int lda,
    double beta,
    double*       C, int ldc )
{
    cblas_dsyrk( layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc );
}

inline void
cblas_syrk(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    std::complex<float> alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> beta,
    std::complex<float>*       C, int ldc )
{
    cblas_csyrk( layout, uplo, trans, n, k, &alpha, A, lda, &beta, C, ldc );
}

inline void
cblas_syrk(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    std::complex<double> alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> beta,
    std::complex<double>* C, int ldc )
{
    cblas_zsyrk( layout, uplo, trans, n, k, &alpha, A, lda, &beta, C, ldc );
}

// -----------------------------------------------------------------------------
inline void
cblas_herk(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    float alpha,
    float const *A, int lda,
    float beta,
    float*       C, int ldc )
{
    cblas_ssyrk( layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc );
}

inline void
cblas_herk(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    double alpha,
    double const *A, int lda,
    double beta,
    double*       C, int ldc )
{
    cblas_dsyrk( layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc );
}

inline void
cblas_herk(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    float alpha,  // note: real
    std::complex<float> const *A, int lda,
    float beta,   // note: real
    std::complex<float>*       C, int ldc )
{
    cblas_cherk( layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc );
}

inline void
cblas_herk(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    double alpha,  // note: real
    std::complex<double> const *A, int lda,
    double beta,   // note: real
    std::complex<double>* C, int ldc )
{
    cblas_zherk( layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc );
}

// -----------------------------------------------------------------------------
inline void
cblas_syr2k(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float*       C, int ldc )
{
    cblas_ssyr2k( layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
}

inline void
cblas_syr2k(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    double alpha,
    double const *A, int lda,
    double const *B, int ldb,
    double beta,
    double*       C, int ldc )
{
    cblas_dsyr2k( layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
}

inline void
cblas_syr2k(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    std::complex<float> alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *B, int ldb,
    std::complex<float> beta,
    std::complex<float>*       C, int ldc )
{
    cblas_csyr2k( layout, uplo, trans, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc );
}

inline void
cblas_syr2k(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    std::complex<double> alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *B, int ldb,
    std::complex<double> beta,
    std::complex<double>* C, int ldc )
{
    cblas_zsyr2k( layout, uplo, trans, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc );
}

// -----------------------------------------------------------------------------
inline void
cblas_her2k(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float*       C, int ldc )
{
    cblas_ssyr2k( layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
}

inline void
cblas_her2k(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    double alpha,
    double const *A, int lda,
    double const *B, int ldb,
    double beta,
    double*       C, int ldc )
{
    cblas_dsyr2k( layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
}

inline void
cblas_her2k(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    std::complex<float> alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *B, int ldb,
    float beta,  // note: real
    std::complex<float>*       C, int ldc )
{
    cblas_cher2k( layout, uplo, trans, n, k, &alpha, A, lda, B, ldb, beta, C, ldc );
}

inline void
cblas_her2k(
    CBLAS_ORDER layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    std::complex<double> alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *B, int ldb,
    double beta,  // note: real
    std::complex<double>* C, int ldc )
{
    cblas_zher2k( layout, uplo, trans, n, k, &alpha, A, lda, B, ldb, beta, C, ldc );
}
}//namespace blis

#endif        //  #ifndef CBLAS_HH
