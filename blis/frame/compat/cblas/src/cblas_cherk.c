#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 *
 * cblas_cherk.c
 * This program is a C interface to cherk.
 * Written by Keita Teranishi
 * 4/8/1998
 *
 */

#include "cblas.h"
#include "cblas_f77.h"
void cblas_cherk(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                 float alpha, const void *A, f77_int lda,
                 float beta, void *C, f77_int ldc)
{
   char UL, TR;   
#ifdef F77_CHAR
   F77_CHAR F77_TR, F77_UL;
#else
   #define F77_TR &TR  
   #define F77_UL &UL  
#endif

#ifdef F77_INT
   F77_INT F77_N=N, F77_K=K, F77_lda=lda;
   F77_INT F77_ldc=ldc;
#else
   #define F77_N N
   #define F77_K K
   #define F77_lda lda
   #define F77_ldc ldc
#endif

   extern int CBLAS_CallFromC;
   extern int RowMajorStrg;
   RowMajorStrg = 0;
   CBLAS_CallFromC = 1;

   if( Order == CblasColMajor )
   {
      if( Uplo == CblasUpper) UL='U';
      else if ( Uplo == CblasLower ) UL='L';
      else 
      {
         cblas_xerbla(2, "cblas_cherk", "Illegal Uplo setting, %d\n", Uplo);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }

      if( Trans == CblasTrans) TR ='T';
      else if ( Trans == CblasConjTrans ) TR='C';
      else if ( Trans == CblasNoTrans )   TR='N';
      else 
      {
         cblas_xerbla(3, "cblas_cherk", "Illegal Trans setting, %d\n", Trans);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }

      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
         F77_TR = C2F_CHAR(&TR);
      #endif

      F77_cherk(F77_UL, F77_TR, &F77_N, &F77_K, &alpha, (scomplex*)A, &F77_lda,
                     &beta, (scomplex*)C, &F77_ldc);
   } else if (Order == CblasRowMajor)
   {
      RowMajorStrg = 1;
      if( Uplo == CblasUpper) UL='L';
      else if ( Uplo == CblasLower ) UL='U';
      else 
      {
         cblas_xerbla(3, "cblas_cherk", "Illegal Uplo setting, %d\n", Uplo);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      if( Trans == CblasTrans) TR ='N';
      else if ( Trans == CblasConjTrans ) TR='N';
      else if ( Trans == CblasNoTrans )   TR='C';
      else 
      {
         cblas_xerbla(3, "cblas_cherk", "Illegal Trans setting, %d\n", Trans);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }

      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
         F77_SD = C2F_CHAR(&SD);
      #endif

      F77_cherk(F77_UL, F77_TR, &F77_N, &F77_K, &alpha, (scomplex*)A, &F77_lda,
                &beta, (scomplex*)C, &F77_ldc);
   } 
   else  cblas_xerbla(1, "cblas_cherk", "Illegal Order setting, %d\n", Order);
   CBLAS_CallFromC = 0;
   RowMajorStrg = 0;
   return;
}
#endif
