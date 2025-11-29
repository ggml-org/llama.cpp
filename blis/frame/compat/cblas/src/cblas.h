
#ifndef CBLAS_H
#define CBLAS_H
#include <stddef.h>

// We need to #include "bli_type_defs.h" in order to pull in the definition of
// f77_int. But in order to #include that header, we also need to pull in the
// headers that precede it in blis.h.

// NOTE: bli_config.h must be included before any other BLIS header. It is
// bootstrapped by ./configure and does not depend on later headers. Moreover
// these configuration variables are necessary to change some default behaviors
// (e.g. disable OS detection in bli_system.h in case of --disable-system).
#include "bli_config.h"
#include "bli_system.h"
#include "bli_config_macro_defs.h"
#include "bli_type_defs.h"

/*
 * Enumerated and derived types
 */
enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE {CblasLeft=141, CblasRight=142};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS functions (complex are recast as routines)
 * ===========================================================================
 */
BLIS_EXPORT_BLAS float  cblas_sdsdot(f77_int N, float alpha, const float *X,
                    f77_int incX, const float *Y, f77_int incY);
BLIS_EXPORT_BLAS double cblas_dsdot(f77_int N, const float *X, f77_int incX, const float *Y,
                   f77_int incY);
BLIS_EXPORT_BLAS float  cblas_sdot(f77_int N, const float  *X, f77_int incX,
                  const float  *Y, f77_int incY);
BLIS_EXPORT_BLAS double cblas_ddot(f77_int N, const double *X, f77_int incX,
                  const double *Y, f77_int incY);

/*
 * Functions having prefixes Z and C only
 */
BLIS_EXPORT_BLAS void   cblas_cdotu_sub(f77_int N, const void *X, f77_int incX,
                       const void *Y, f77_int incY, void *dotu);
BLIS_EXPORT_BLAS void   cblas_cdotc_sub(f77_int N, const void *X, f77_int incX,
                       const void *Y, f77_int incY, void *dotc);

BLIS_EXPORT_BLAS void   cblas_zdotu_sub(f77_int N, const void *X, f77_int incX,
                       const void *Y, f77_int incY, void *dotu);
BLIS_EXPORT_BLAS void   cblas_zdotc_sub(f77_int N, const void *X, f77_int incX,
                       const void *Y, f77_int incY, void *dotc);


/*
 * Functions having prefixes S D SC DZ
 */
BLIS_EXPORT_BLAS float  cblas_snrm2(f77_int N, const float *X, f77_int incX);
BLIS_EXPORT_BLAS float  cblas_sasum(f77_int N, const float *X, f77_int incX);

BLIS_EXPORT_BLAS double cblas_dnrm2(f77_int N, const double *X, f77_int incX);
BLIS_EXPORT_BLAS double cblas_dasum(f77_int N, const double *X, f77_int incX);

BLIS_EXPORT_BLAS float  cblas_scnrm2(f77_int N, const void *X, f77_int incX);
BLIS_EXPORT_BLAS float  cblas_scasum(f77_int N, const void *X, f77_int incX);

BLIS_EXPORT_BLAS double cblas_dznrm2(f77_int N, const void *X, f77_int incX);
BLIS_EXPORT_BLAS double cblas_dzasum(f77_int N, const void *X, f77_int incX);


/*
 * Functions having standard 4 prefixes (S D C Z)
 */
BLIS_EXPORT_BLAS f77_int cblas_isamax(f77_int N, const float  *X, f77_int incX);
BLIS_EXPORT_BLAS f77_int cblas_idamax(f77_int N, const double *X, f77_int incX);
BLIS_EXPORT_BLAS f77_int cblas_icamax(f77_int N, const void   *X, f77_int incX);
BLIS_EXPORT_BLAS f77_int cblas_izamax(f77_int N, const void   *X, f77_int incX);

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (s, d, c, z)
 */
void BLIS_EXPORT_BLAS cblas_sswap(f77_int N, float *X, f77_int incX,
                 float *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_scopy(f77_int N, const float *X, f77_int incX,
                 float *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_saxpy(f77_int N, float alpha, const float *X,
                 f77_int incX, float *Y, f77_int incY);

void BLIS_EXPORT_BLAS cblas_dswap(f77_int N, double *X, f77_int incX,
                 double *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_dcopy(f77_int N, const double *X, f77_int incX,
                 double *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_daxpy(f77_int N, double alpha, const double *X,
                 f77_int incX, double *Y, f77_int incY);

void BLIS_EXPORT_BLAS cblas_cswap(f77_int N, void *X, f77_int incX,
                 void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_ccopy(f77_int N, const void *X, f77_int incX,
                 void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_caxpy(f77_int N, const void *alpha, const void *X,
                 f77_int incX, void *Y, f77_int incY);

void BLIS_EXPORT_BLAS cblas_zswap(f77_int N, void *X, f77_int incX,
                 void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_zcopy(f77_int N, const void *X, f77_int incX,
                 void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_zaxpy(f77_int N, const void *alpha, const void *X,
                 f77_int incX, void *Y, f77_int incY);


/*
 * Routines with S and D prefix only
 */
void BLIS_EXPORT_BLAS cblas_srotg(float *a, float *b, float *c, float *s);
void BLIS_EXPORT_BLAS cblas_srotmg(float *d1, float *d2, float *b1, const float b2, float *P);
void BLIS_EXPORT_BLAS cblas_srot(f77_int N, float *X, f77_int incX,
                float *Y, f77_int incY, const float c, const float s);
void BLIS_EXPORT_BLAS cblas_srotm(f77_int N, float *X, f77_int incX,
                float *Y, f77_int incY, const float *P);

void BLIS_EXPORT_BLAS cblas_drotg(double *a, double *b, double *c, double *s);
void BLIS_EXPORT_BLAS cblas_drotmg(double *d1, double *d2, double *b1, const double b2, double *P);
void BLIS_EXPORT_BLAS cblas_drot(f77_int N, double *X, f77_int incX,
                double *Y, f77_int incY, const double c, const double  s);
void BLIS_EXPORT_BLAS cblas_drotm(f77_int N, double *X, f77_int incX,
                double *Y, f77_int incY, const double *P);


/*
 * Routines with S D C Z CS and ZD prefixes
 */
void BLIS_EXPORT_BLAS cblas_sscal(f77_int N, float alpha, float *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_dscal(f77_int N, double alpha, double *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_cscal(f77_int N, const void *alpha, void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_zscal(f77_int N, const void *alpha, void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_csscal(f77_int N, float alpha, void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_zdscal(f77_int N, double alpha, void *X, f77_int incX);

/*
 * ===========================================================================
 * Prototypes for level 2 BLAS
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */
void BLIS_EXPORT_BLAS cblas_sgemv(enum CBLAS_ORDER order,
                 enum CBLAS_TRANSPOSE TransA, f77_int M, f77_int N,
                 float alpha, const float *A, f77_int lda,
                 const float *X, f77_int incX, float beta,
                 float *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_sgbmv(enum CBLAS_ORDER order,
                 enum CBLAS_TRANSPOSE TransA, f77_int M, f77_int N,
                 f77_int KL, f77_int KU, float alpha,
                 const float *A, f77_int lda, const float *X,
                 f77_int incX, float beta, float *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_strmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const float *A, f77_int lda,
                 float *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_stbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, f77_int K, const float *A, f77_int lda,
                 float *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_stpmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const float *Ap, float *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_strsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const float *A, f77_int lda, float *X,
                 f77_int incX);
void BLIS_EXPORT_BLAS cblas_stbsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, f77_int K, const float *A, f77_int lda,
                 float *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_stpsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const float *Ap, float *X, f77_int incX);

void BLIS_EXPORT_BLAS cblas_dgemv(enum CBLAS_ORDER order,
                 enum CBLAS_TRANSPOSE TransA, f77_int M, f77_int N,
                 double alpha, const double *A, f77_int lda,
                 const double *X, f77_int incX, double beta,
                 double *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_dgbmv(enum CBLAS_ORDER order,
                 enum CBLAS_TRANSPOSE TransA, f77_int M, f77_int N,
                 f77_int KL, f77_int KU, double alpha,
                 const double *A, f77_int lda, const double *X,
                 f77_int incX, double beta, double *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_dtrmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const double *A, f77_int lda,
                 double *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_dtbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, f77_int K, const double *A, f77_int lda,
                 double *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_dtpmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const double *Ap, double *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_dtrsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const double *A, f77_int lda, double *X,
                 f77_int incX);
void BLIS_EXPORT_BLAS cblas_dtbsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, f77_int K, const double *A, f77_int lda,
                 double *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_dtpsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const double *Ap, double *X, f77_int incX);

void BLIS_EXPORT_BLAS cblas_cgemv(enum CBLAS_ORDER order,
                 enum CBLAS_TRANSPOSE TransA, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 const void *X, f77_int incX, const void *beta,
                 void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_cgbmv(enum CBLAS_ORDER order,
                 enum CBLAS_TRANSPOSE TransA, f77_int M, f77_int N,
                 f77_int KL, f77_int KU, const void *alpha,
                 const void *A, f77_int lda, const void *X,
                 f77_int incX, const void *beta, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_ctrmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const void *A, f77_int lda,
                 void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_ctbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, f77_int K, const void *A, f77_int lda,
                 void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_ctpmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const void *Ap, void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_ctrsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const void *A, f77_int lda, void *X,
                 f77_int incX);
void BLIS_EXPORT_BLAS cblas_ctbsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, f77_int K, const void *A, f77_int lda,
                 void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_ctpsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const void *Ap, void *X, f77_int incX);

void BLIS_EXPORT_BLAS cblas_zgemv(enum CBLAS_ORDER order,
                 enum CBLAS_TRANSPOSE TransA, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 const void *X, f77_int incX, const void *beta,
                 void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_zgbmv(enum CBLAS_ORDER order,
                 enum CBLAS_TRANSPOSE TransA, f77_int M, f77_int N,
                 f77_int KL, f77_int KU, const void *alpha,
                 const void *A, f77_int lda, const void *X,
                 f77_int incX, const void *beta, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_ztrmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const void *A, f77_int lda,
                 void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_ztbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, f77_int K, const void *A, f77_int lda,
                 void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_ztpmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const void *Ap, void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_ztrsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const void *A, f77_int lda, void *X,
                 f77_int incX);
void BLIS_EXPORT_BLAS cblas_ztbsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, f77_int K, const void *A, f77_int lda,
                 void *X, f77_int incX);
void BLIS_EXPORT_BLAS cblas_ztpsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const void *Ap, void *X, f77_int incX);


/*
 * Routines with S and D prefixes only
 */
void BLIS_EXPORT_BLAS cblas_ssymv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, float alpha, const float *A,
                 f77_int lda, const float *X, f77_int incX,
                 float beta, float *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_ssbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, f77_int K, float alpha, const float *A,
                 f77_int lda, const float *X, f77_int incX,
                 float beta, float *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_sspmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, float alpha, const float *Ap,
                 const float *X, f77_int incX,
                 float beta, float *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_sger(enum CBLAS_ORDER order, f77_int M, f77_int N,
                float alpha, const float *X, f77_int incX,
                const float *Y, f77_int incY, float *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_ssyr(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, float alpha, const float *X,
                f77_int incX, float *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_sspr(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, float alpha, const float *X,
                f77_int incX, float *Ap);
void BLIS_EXPORT_BLAS cblas_ssyr2(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, float alpha, const float *X,
                f77_int incX, const float *Y, f77_int incY, float *A,
                f77_int lda);
void BLIS_EXPORT_BLAS cblas_sspr2(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, float alpha, const float *X,
                f77_int incX, const float *Y, f77_int incY, float *A);

void BLIS_EXPORT_BLAS cblas_dsymv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, double alpha, const double *A,
                 f77_int lda, const double *X, f77_int incX,
                 double beta, double *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_dsbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, f77_int K, double alpha, const double *A,
                 f77_int lda, const double *X, f77_int incX,
                 double beta, double *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_dspmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, double alpha, const double *Ap,
                 const double *X, f77_int incX,
                 double beta, double *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_dger(enum CBLAS_ORDER order, f77_int M, f77_int N,
                double alpha, const double *X, f77_int incX,
                const double *Y, f77_int incY, double *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_dsyr(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, double alpha, const double *X,
                f77_int incX, double *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_dspr(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, double alpha, const double *X,
                f77_int incX, double *Ap);
void BLIS_EXPORT_BLAS cblas_dsyr2(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, double alpha, const double *X,
                f77_int incX, const double *Y, f77_int incY, double *A,
                f77_int lda);
void BLIS_EXPORT_BLAS cblas_dspr2(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, double alpha, const double *X,
                f77_int incX, const double *Y, f77_int incY, double *A);


/*
 * Routines with C and Z prefixes only
 */
void BLIS_EXPORT_BLAS cblas_chemv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, const void *alpha, const void *A,
                 f77_int lda, const void *X, f77_int incX,
                 const void *beta, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_chbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, f77_int K, const void *alpha, const void *A,
                 f77_int lda, const void *X, f77_int incX,
                 const void *beta, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_chpmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, const void *alpha, const void *Ap,
                 const void *X, f77_int incX,
                 const void *beta, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_cgeru(enum CBLAS_ORDER order, f77_int M, f77_int N,
                 const void *alpha, const void *X, f77_int incX,
                 const void *Y, f77_int incY, void *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_cgerc(enum CBLAS_ORDER order, f77_int M, f77_int N,
                 const void *alpha, const void *X, f77_int incX,
                 const void *Y, f77_int incY, void *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_cher(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, float alpha, const void *X, f77_int incX,
                void *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_chpr(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, float alpha, const void *X,
                f77_int incX, void *A);
void BLIS_EXPORT_BLAS cblas_cher2(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo, f77_int N,
                const void *alpha, const void *X, f77_int incX,
                const void *Y, f77_int incY, void *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_chpr2(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo, f77_int N,
                const void *alpha, const void *X, f77_int incX,
                const void *Y, f77_int incY, void *Ap);

void BLIS_EXPORT_BLAS cblas_zhemv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, const void *alpha, const void *A,
                 f77_int lda, const void *X, f77_int incX,
                 const void *beta, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_zhbmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, f77_int K, const void *alpha, const void *A,
                 f77_int lda, const void *X, f77_int incX,
                 const void *beta, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_zhpmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 f77_int N, const void *alpha, const void *Ap,
                 const void *X, f77_int incX,
                 const void *beta, void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_zgeru(enum CBLAS_ORDER order, f77_int M, f77_int N,
                 const void *alpha, const void *X, f77_int incX,
                 const void *Y, f77_int incY, void *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_zgerc(enum CBLAS_ORDER order, f77_int M, f77_int N,
                 const void *alpha, const void *X, f77_int incX,
                 const void *Y, f77_int incY, void *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_zher(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, double alpha, const void *X, f77_int incX,
                void *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_zhpr(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, double alpha, const void *X,
                f77_int incX, void *A);
void BLIS_EXPORT_BLAS cblas_zher2(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo, f77_int N,
                const void *alpha, const void *X, f77_int incX,
                const void *Y, f77_int incY, void *A, f77_int lda);
void BLIS_EXPORT_BLAS cblas_zhpr2(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo, f77_int N,
                const void *alpha, const void *X, f77_int incX,
                const void *Y, f77_int incY, void *Ap);

/*
 * ===========================================================================
 * Prototypes for level 3 BLAS
 * ===========================================================================
 */

/*
 * Routines with standard 4 prefixes (S, D, C, Z)
 */
void BLIS_EXPORT_BLAS cblas_sgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, f77_int M, f77_int N,
                 f77_int K, float alpha, const float *A,
                 f77_int lda, const float *B, f77_int ldb,
                 float beta, float *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_ssymm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, f77_int M, f77_int N,
                 float alpha, const float *A, f77_int lda,
                 const float *B, f77_int ldb, float beta,
                 float *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_ssyrk(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                 float alpha, const float *A, f77_int lda,
                 float beta, float *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_ssyr2k(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                  float alpha, const float *A, f77_int lda,
                  const float *B, f77_int ldb, float beta,
                  float *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_strmm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, f77_int M, f77_int N,
                 float alpha, const float *A, f77_int lda,
                 float *B, f77_int ldb);
void BLIS_EXPORT_BLAS cblas_strsm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, f77_int M, f77_int N,
                 float alpha, const float *A, f77_int lda,
                 float *B, f77_int ldb);

void BLIS_EXPORT_BLAS cblas_dgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, f77_int M, f77_int N,
                 f77_int K, double alpha, const double *A,
                 f77_int lda, const double *B, f77_int ldb,
                 double beta, double *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_dsymm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, f77_int M, f77_int N,
                 double alpha, const double *A, f77_int lda,
                 const double *B, f77_int ldb, double beta,
                 double *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_dsyrk(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                 double alpha, const double *A, f77_int lda,
                 double beta, double *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_dsyr2k(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                  double alpha, const double *A, f77_int lda,
                  const double *B, f77_int ldb, double beta,
                  double *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_dtrmm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, f77_int M, f77_int N,
                 double alpha, const double *A, f77_int lda,
                 double *B, f77_int ldb);
void BLIS_EXPORT_BLAS cblas_dtrsm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, f77_int M, f77_int N,
                 double alpha, const double *A, f77_int lda,
                 double *B, f77_int ldb);

void BLIS_EXPORT_BLAS cblas_cgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, f77_int M, f77_int N,
                 f77_int K, const void *alpha, const void *A,
                 f77_int lda, const void *B, f77_int ldb,
                 const void *beta, void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_csymm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 const void *B, f77_int ldb, const void *beta,
                 void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_csyrk(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                 const void *alpha, const void *A, f77_int lda,
                 const void *beta, void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_csyr2k(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                  const void *alpha, const void *A, f77_int lda,
                  const void *B, f77_int ldb, const void *beta,
                  void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_ctrmm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 void *B, f77_int ldb);
void BLIS_EXPORT_BLAS cblas_ctrsm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 void *B, f77_int ldb);

void BLIS_EXPORT_BLAS cblas_zgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, f77_int M, f77_int N,
                 f77_int K, const void *alpha, const void *A,
                 f77_int lda, const void *B, f77_int ldb,
                 const void *beta, void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_zsymm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 const void *B, f77_int ldb, const void *beta,
                 void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_zsyrk(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                 const void *alpha, const void *A, f77_int lda,
                 const void *beta, void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_zsyr2k(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                  const void *alpha, const void *A, f77_int lda,
                  const void *B, f77_int ldb, const void *beta,
                  void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_ztrmm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 void *B, f77_int ldb);
void BLIS_EXPORT_BLAS cblas_ztrsm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 void *B, f77_int ldb);


/*
 * Routines with prefixes C and Z only
 */
void BLIS_EXPORT_BLAS cblas_chemm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 const void *B, f77_int ldb, const void *beta,
                 void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_cherk(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                 float alpha, const void *A, f77_int lda,
                 float beta, void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_cher2k(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                  const void *alpha, const void *A, f77_int lda,
                  const void *B, f77_int ldb, float beta,
                  void *C, f77_int ldc);

void BLIS_EXPORT_BLAS cblas_zhemm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, f77_int M, f77_int N,
                 const void *alpha, const void *A, f77_int lda,
                 const void *B, f77_int ldb, const void *beta,
                 void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_zherk(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                 double alpha, const void *A, f77_int lda,
                 double beta, void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_zher2k(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                  const void *alpha, const void *A, f77_int lda,
                  const void *B, f77_int ldb, double beta,
                  void *C, f77_int ldc);

void BLIS_EXPORT_BLAS cblas_xerbla(f77_int p, const char *rout, const char *form, ...);


/*
 * ===========================================================================
 * BLAS Extension prototypes
 * ===========================================================================
 */

// -- APIs to operations unique to BLIS --

void BLIS_EXPORT_BLAS cblas_saxpby(f77_int N, float alpha, const float *X,
                 f77_int incX, float beta, float *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_daxpby(f77_int N, double alpha, const double *X,
                 f77_int incX, double beta, double *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_caxpby(f77_int N, const void *alpha,
                const void *X, f77_int incX, const void* beta,
                void *Y, f77_int incY);
void BLIS_EXPORT_BLAS cblas_zaxpby(f77_int N, const void *alpha,
                const void *X, f77_int incX, const void *beta,
                void *Y, f77_int incY);

// -- APIs to level-3-like operations --

void BLIS_EXPORT_BLAS cblas_sgemmt(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
                 f77_int N, f77_int K, float alpha, const float *A,
                 f77_int lda, const float *B, f77_int ldb,
                 float beta, float *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_sgemmtr(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
                 f77_int N, f77_int K, float alpha, const float *A,
                 f77_int lda, const float *B, f77_int ldb,
                 float beta, float *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_dgemmt(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
                 f77_int N, f77_int K, double alpha, const double *A,
                 f77_int lda, const double *B, f77_int ldb,
                 double beta, double *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_dgemmtr(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
                 f77_int N, f77_int K, double alpha, const double *A,
                 f77_int lda, const double *B, f77_int ldb,
                 double beta, double *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_cgemmt(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
                 f77_int N, f77_int K, const void *alpha, const void *A,
                 f77_int lda, const void *B, f77_int ldb,
                 const void *beta, void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_cgemmtr(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
                 f77_int N, f77_int K, const void *alpha, const void *A,
                 f77_int lda, const void *B, f77_int ldb,
                 const void *beta, void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_zgemmt(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
                 f77_int N, f77_int K, const void *alpha, const void *A,
                 f77_int lda, const void *B, f77_int ldb,
                 const void *beta, void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_zgemmtr(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
                 f77_int N, f77_int K, const void *alpha, const void *A,
                 f77_int lda, const void *B, f77_int ldb,
                 const void *beta, void *C, f77_int ldc);

// -- Batch APIs --

void BLIS_EXPORT_BLAS cblas_sgemm_batch(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE *TransA_array,
                 enum CBLAS_TRANSPOSE *TransB_array,
                 f77_int *M_array, f77_int *N_array,
                 f77_int *K_array, const float *alpha_array, const float **A,
                 f77_int *lda_array, const float **B, f77_int *ldb_array,
                 const float *beta_array, float **C, f77_int *ldc_array,
                 f77_int group_count, f77_int *group_size);
void BLIS_EXPORT_BLAS cblas_dgemm_batch(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE *TransA_array,
                 enum CBLAS_TRANSPOSE *TransB_array,
                 f77_int *M_array, f77_int *N_array,
                 f77_int *K_array, const double *alpha_array,
                 const double **A,f77_int *lda_array,
                 const double **B, f77_int *ldb_array,
                 const double *beta_array, double **C, f77_int *ldc_array,
                 f77_int group_count, f77_int *group_size);
void BLIS_EXPORT_BLAS cblas_cgemm_batch(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE *TransA_array,
                 enum CBLAS_TRANSPOSE *TransB_array,
                 f77_int *M_array, f77_int *N_array,
                 f77_int *K_array, const void *alpha_array, const void **A,
                 f77_int *lda_array, const void **B, f77_int *ldb_array,
                 const void *beta_array, void **C, f77_int *ldc_array,
                 f77_int group_count, f77_int *group_size);
void BLIS_EXPORT_BLAS cblas_zgemm_batch(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE *TransA_array,
                 enum CBLAS_TRANSPOSE *TransB_array,
                 f77_int *M_array, f77_int *N_array,
                 f77_int *K_array, const void *alpha_array, const void **A,
                 f77_int *lda_array, const void **B, f77_int *ldb_array,
                 const void *beta_array, void **C, f77_int *ldc_array,
                 f77_int group_count, f77_int *group_size);

// -- 3m APIs --

void BLIS_EXPORT_BLAS cblas_cgemm3m(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, f77_int M, f77_int N,
                 f77_int K, const void *alpha, const void *A,
                 f77_int lda, const void *B, f77_int ldb,
                 const void *beta, void *C, f77_int ldc);
void BLIS_EXPORT_BLAS cblas_zgemm3m(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, f77_int M, f77_int N,
                 f77_int K, const void *alpha, const void *A,
                 f77_int lda, const void *B, f77_int ldb,
                 const void *beta, void *C, f77_int ldc);

#ifdef __cplusplus
}
#endif
#endif
