#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 *
 * cblas_dsyr.c
 * This program is a C interface to dsyr.
 * Written by Keita Teranishi
 * 4/6/1998
 *
 */

#include "cblas.h"
#include "cblas_f77.h"
void cblas_dsyr(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, const double  alpha, const double  *X,
                f77_int incX, double  *A, f77_int lda)
{
   char UL;
#ifdef F77_CHAR
   F77_CHAR F77_UL;
#else
   #define F77_UL &UL
#endif

#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX, F77_lda=lda;
#else
   #define F77_N N
   #define F77_incX incX
   #define F77_lda  lda
#endif
   extern int CBLAS_CallFromC;
   extern int RowMajorStrg;
   RowMajorStrg = 0;
   CBLAS_CallFromC = 1;
   if (order == CblasColMajor)
   {
      if (Uplo == CblasLower) UL = 'L';
      else if (Uplo == CblasUpper) UL = 'U';
      else 
      {
         cblas_xerbla(2, "cblas_dsyr","Illegal Uplo setting, %d\n",Uplo );
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
      #endif

      F77_dsyr(F77_UL, &F77_N, &alpha, X, &F77_incX, A, &F77_lda);

   }  else if (order == CblasRowMajor) 
   {
      RowMajorStrg = 1;
      if (Uplo == CblasLower) UL = 'U';
      else if (Uplo == CblasUpper) UL = 'L';
      else 
      {
         cblas_xerbla(2, "cblas_dsyr","Illegal Uplo setting, %d\n",Uplo );
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
      #endif  
      F77_dsyr(F77_UL, &F77_N, &alpha, X, &F77_incX, A, &F77_lda); 
   } else cblas_xerbla(1, "cblas_dsyr", "Illegal Order setting, %d\n", order);
   CBLAS_CallFromC = 0;
   RowMajorStrg = 0;
   return;
} 
#endif
