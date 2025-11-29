#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_ctpsv.c
 * The program is a C interface to ctpsv.
 * 
 * Keita Teranishi  3/23/98
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
void cblas_ctpsv(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                 enum CBLAS_TRANSPOSE TransA, enum CBLAS_DIAG Diag,
                 f77_int N, const void  *Ap, void  *X, f77_int incX)
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
   int n, i=0, tincX; 
   float *st=0, *x=(float*)X;
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
         cblas_xerbla(2, "cblas_ctpsv","Illegal Uplo setting, %d\n", Uplo);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      if (TransA == CblasNoTrans) TA = 'N';
      else if (TransA == CblasTrans) TA = 'T';
      else if (TransA == CblasConjTrans) TA = 'C';
      else 
      {
         cblas_xerbla(3, "cblas_ctpsv","Illegal TransA setting, %d\n", TransA);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      if (Diag == CblasUnit) DI = 'U';
      else if (Diag == CblasNonUnit) DI = 'N';
      else 
      {
         cblas_xerbla(4, "cblas_ctpsv","Illegal Diag setting, %d\n", Diag);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
         F77_TA = C2F_CHAR(&TA);
         F77_DI = C2F_CHAR(&DI);
      #endif
      F77_ctpsv( F77_UL, F77_TA, F77_DI, &F77_N, (scomplex*)Ap, (scomplex*)X, &F77_incX);
   }
   else if (order == CblasRowMajor)
   {
      RowMajorStrg = 1;
      if (Uplo == CblasUpper) UL = 'L';
      else if (Uplo == CblasLower) UL = 'U';
      else 
      {
         cblas_xerbla(2, "cblas_ctpsv","Illegal Uplo setting, %d\n", Uplo);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }

      if (TransA == CblasNoTrans) TA = 'T';
      else if (TransA == CblasTrans) TA = 'N';
      else if (TransA == CblasConjTrans)
      {
         TA = 'N';
         if ( N > 0)
         {
            if ( incX > 0 )
               tincX = incX;
            else
               tincX = -incX;
 
            n = N*2*(tincX);
 
            x++;

            st=x+n;

            i = tincX << 1;
            do
            {
               *x = -(*x);
               x+=i;
            }
            while (x != st);
            x -= n;
         }
      }
      else 
      {
         cblas_xerbla(3, "cblas_ctpsv","Illegal TransA setting, %d\n", TransA);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }

      if (Diag == CblasUnit) DI = 'U';
      else if (Diag == CblasNonUnit) DI = 'N';
      else 
      {
         cblas_xerbla(4, "cblas_ctpsv","Illegal Diag setting, %d\n", Diag);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
         F77_TA = C2F_CHAR(&TA);
         F77_DI = C2F_CHAR(&DI);
      #endif

      F77_ctpsv( F77_UL, F77_TA, F77_DI, &F77_N, (scomplex*)Ap, (scomplex*)X,&F77_incX);

      if (TransA == CblasConjTrans)
      {
         if (N > 0)
         {
            do
            {
               *x = -(*x);
               x += i;
            }
            while (x != st);
         }
      }
   }
   else cblas_xerbla(1, "cblas_ctpsv", "Illegal Order setting, %d\n", order);
   CBLAS_CallFromC = 0;
   RowMajorStrg = 0;
   return;
}
#endif
