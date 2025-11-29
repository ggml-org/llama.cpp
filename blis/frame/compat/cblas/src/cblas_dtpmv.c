#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_dtpmv.c
 * The program is a C interface to dtpmv.
 *
 * Keita Teranishi  5/20/98
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
void cblas_dtpmv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const double  *Ap, double  *X, f77_int incX)
{
   char TA;
   char UL;
   char DI;
#ifdef F77_CHAR
   F77_CHAR F77_TA, F77_UL, F77_DI;
#else
   #define F77_TA &TA
   #define F77_UL &UL
   #define F77_DI &DI   
#endif
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX;
#else
   #define F77_N N
   #define F77_incX incX
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
         cblas_xerbla(2, "cblas_dtpmv","Illegal Uplo setting, %d\n", Uplo);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      if (TransA == CblasNoTrans) TA = 'N';
      else if (TransA == CblasTrans) TA = 'T';
      else if (TransA == CblasConjTrans) TA = 'C';
      else 
      {
         cblas_xerbla(3, "cblas_dtpmv","Illegal TransA setting, %d\n", TransA);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      if (Diag == CblasUnit) DI = 'U';
      else if (Diag == CblasNonUnit) DI = 'N';
      else 
      {
         cblas_xerbla(4, "cblas_dtpmv","Illegal Diag setting, %d\n", Diag);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
         F77_TA = C2F_CHAR(&TA);
         F77_DI = C2F_CHAR(&DI);
      #endif
      F77_dtpmv( F77_UL, F77_TA, F77_DI, &F77_N, Ap, X, &F77_incX);
   }
   else if (order == CblasRowMajor)
   {
      RowMajorStrg = 1;
      if (Uplo == CblasUpper) UL = 'L';
      else if (Uplo == CblasLower) UL = 'U';
      else 
      {
         cblas_xerbla(2, "cblas_dtpmv","Illegal Uplo setting, %d\n", Uplo);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }

      if (TransA == CblasNoTrans) TA = 'T';
      else if (TransA == CblasTrans) TA = 'N';
      else if (TransA == CblasConjTrans) TA = 'N';
      else 
      {
         cblas_xerbla(3, "cblas_dtpmv","Illegal TransA setting, %d\n", TransA);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }

      if (Diag == CblasUnit) DI = 'U';
      else if (Diag == CblasNonUnit) DI = 'N';
      else 
      {
         cblas_xerbla(4, "cblas_dtpmv","Illegal Diag setting, %d\n", Diag);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
         F77_TA = C2F_CHAR(&TA);
         F77_DI = C2F_CHAR(&DI);
      #endif

      F77_dtpmv( F77_UL, F77_TA, F77_DI, &F77_N, Ap, X,&F77_incX);
   }
   else cblas_xerbla(1, "cblas_dtpmv", "Illegal Order setting, %d\n", order);
   CBLAS_CallFromC = 0;
   RowMajorStrg = 0;
   return;
}
#endif
