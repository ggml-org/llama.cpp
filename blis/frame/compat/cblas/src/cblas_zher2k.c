#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 *
 * cblas_zher2k.c
 * This program is a C interface to zher2k.
 * Written by Keita Teranishi
 * 4/8/1998
 *
 */

#include "cblas.h"
#include "cblas_f77.h"
void cblas_zher2k(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE Trans, f77_int N, f77_int K,
                  const void *alpha, const void *A, f77_int lda,
                  const void *B, f77_int ldb, double beta,
                  void *C, f77_int ldc)
{
   char UL, TR;   
#ifdef F77_CHAR
   F77_CHAR F77_TR, F77_UL;
#else
   #define F77_TR &TR  
   #define F77_UL &UL  
#endif

#ifdef F77_INT
   F77_INT F77_N=N, F77_K=K, F77_lda=lda, F77_ldb=ldb;
   F77_INT F77_ldc=ldc;
#else
   #define F77_N N
   #define F77_K K
   #define F77_lda lda
   #define F77_ldb ldb
   #define F77_ldc ldc
#endif

   extern int CBLAS_CallFromC;
   extern int RowMajorStrg;
   double ALPHA[2]; 
   const double *alp=(double *)alpha;

   CBLAS_CallFromC = 1;
   RowMajorStrg = 0;

   if( Order == CblasColMajor )
   {

      if( Uplo == CblasUpper) UL='U';
      else if ( Uplo == CblasLower ) UL='L';
      else 
      {
         cblas_xerbla(2, "cblas_zher2k", "Illegal Uplo setting, %d\n", Uplo);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }

      if( Trans == CblasTrans) TR ='T';
      else if ( Trans == CblasConjTrans ) TR='C';
      else if ( Trans == CblasNoTrans )   TR='N';
      else 
      {
         cblas_xerbla(3, "cblas_zher2k", "Illegal Trans setting, %d\n", Trans);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }

      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
         F77_TR = C2F_CHAR(&TR);
      #endif

      F77_zher2k(F77_UL, F77_TR, &F77_N, &F77_K, (dcomplex*)alpha, (dcomplex*)A, &F77_lda, (dcomplex*)B, &F77_ldb, &beta, (dcomplex*)C, &F77_ldc);
   } else if (Order == CblasRowMajor)
   {
      RowMajorStrg = 1;
      
      if( Uplo == CblasUpper) UL='L';
      else if ( Uplo == CblasLower ) UL='U';
      else 
      {
         cblas_xerbla(2, "cblas_zher2k", "Illegal Uplo setting, %d\n", Uplo);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      if( Trans == CblasTrans) TR ='N';
      else if ( Trans == CblasConjTrans ) TR='N';
      else if ( Trans == CblasNoTrans )   TR='C';
      else 
      {
         cblas_xerbla(3, "cblas_zher2k", "Illegal Trans setting, %d\n", Trans);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
         F77_TR = C2F_CHAR(&TR);
      #endif

      ALPHA[0]= *alp;
      ALPHA[1]= -alp[1];
      F77_zher2k(F77_UL,F77_TR, &F77_N, &F77_K, (dcomplex*)ALPHA, (dcomplex*)A, &F77_lda, (dcomplex*)B, &F77_ldb, &beta, (dcomplex*)C, &F77_ldc);
   } else  cblas_xerbla(1, "cblas_zher2k", "Illegal Order setting, %d\n", Order);
   CBLAS_CallFromC = 0;
   RowMajorStrg = 0;
   return;
}
#endif
