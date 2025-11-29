#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 *
 * cblas_dsbmv.c
 * This program is a C interface to dsbmv.
 * Written by Keita Teranishi
 * 4/6/1998
 *
 */

#include "cblas.h"
#include "cblas_f77.h"
void cblas_dsbmv(enum CBLAS_ORDER order,
                 enum CBLAS_UPLO Uplo, f77_int N, f77_int K,
                 double alpha, const double  *A, f77_int lda,
                 const double  *X, f77_int incX, double beta,
                 double  *Y, f77_int incY)
{
   char UL;
#ifdef F77_CHAR
   F77_CHAR F77_UL;
#else
   #define F77_UL &UL   
#endif
#ifdef F77_INT
   F77_INT F77_N=N, F77_K=K, F77_lda=lda, F77_incX=incX, F77_incY=incY;
#else
   #define F77_N N
   #define F77_K K
   #define F77_lda lda
   #define F77_incX incX
   #define F77_incY incY
#endif
   extern int CBLAS_CallFromC;
   extern int RowMajorStrg;
   RowMajorStrg = 0;

   CBLAS_CallFromC = 1;
   if (order == CblasColMajor)
   {
      if (Uplo == CblasUpper) UL = 'U';
      else if (Uplo == CblasLower) UL = 'L';
      else 
      {
         cblas_xerbla(2, "cblas_dsbmv","Illegal Uplo setting, %d\n",Uplo );
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
      #endif
      F77_dsbmv(F77_UL, &F77_N, &F77_K, &alpha, A, &F77_lda, X,  
                     &F77_incX, &beta, Y, &F77_incY);
   }
   else if (order == CblasRowMajor)
   {
      RowMajorStrg = 1;
      if (Uplo == CblasUpper) UL = 'L';
      else if (Uplo == CblasLower) UL = 'U';
      else 
      {
         cblas_xerbla(2, "cblas_dsbmv","Illegal Uplo setting, %d\n", Uplo);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
      #endif
      F77_dsbmv(F77_UL, &F77_N, &F77_K, &alpha, 
                     A ,&F77_lda, X,&F77_incX, &beta, Y, &F77_incY);
   }
   else cblas_xerbla(1, "cblas_dsbmv", "Illegal Order setting, %d\n", order);
   CBLAS_CallFromC = 0;
   RowMajorStrg = 0;
   return;
}
#endif
