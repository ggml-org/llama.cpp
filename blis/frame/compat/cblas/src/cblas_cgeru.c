#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_cgeru.c
 * The program is a C interface to cgeru.
 * 
 * Keita Teranishi  5/20/98
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
void cblas_cgeru(enum CBLAS_ORDER order, f77_int M, f77_int N,
                 const void *alpha, const void *X, f77_int incX,
                 const void *Y, f77_int incY, void *A, f77_int lda)
{
#ifdef F77_INT
   F77_INT F77_M=M, F77_N=N, F77_lda=lda, F77_incX=incX, F77_incY=incY;
#else
   #define F77_M M
   #define F77_N N
   #define F77_incX incX
   #define F77_incY incY
   #define F77_lda lda
#endif

   extern int CBLAS_CallFromC;
   extern int RowMajorStrg;
   RowMajorStrg = 0;

   CBLAS_CallFromC = 1;

   if (order == CblasColMajor)
   {
      F77_cgeru( &F77_M, &F77_N, (scomplex*)alpha, (scomplex*)X, &F77_incX, (scomplex*)Y, &F77_incY, (scomplex*)A,
                      &F77_lda);
   }
   else if (order == CblasRowMajor)
   {
      RowMajorStrg = 1;
      F77_cgeru( &F77_N, &F77_M, (scomplex*)alpha, (scomplex*)Y, &F77_incY, (scomplex*)X, &F77_incX, (scomplex*)A,
                      &F77_lda);
   }
   else cblas_xerbla(1, "cblas_cgeru","Illegal Order setting, %d\n", order);
   CBLAS_CallFromC = 0;
   RowMajorStrg = 0;
   return;
}
#endif
