#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_cher.c
 * The program is a C interface to cher.
 * 
 * Keita Teranishi  5/20/98
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"
#include "cblas_f77.h"
void cblas_cher(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                f77_int N, float alpha, const void *X, f77_int incX
                ,void *A, f77_int lda)
{
   char UL;
#ifdef F77_CHAR
   F77_CHAR F77_UL;
#else
   #define F77_UL &UL
#endif

#ifdef F77_INT
   F77_INT F77_N=N, F77_lda=lda, F77_incX=incX;
#else
   #define F77_N N
   #define F77_lda lda
   #define F77_incX incX
#endif
   int n, i, tincx;
   float *x=(float *)X, *xx=(float *)X, *tx, *st;

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
         cblas_xerbla(2, "cblas_cher","Illegal Uplo setting, %d\n",Uplo );
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
      #endif

      F77_cher(F77_UL, &F77_N, &alpha, (scomplex*)X, &F77_incX, (scomplex*)A, &F77_lda);

   }  else if (order == CblasRowMajor)
   {
      RowMajorStrg = 1;
      if (Uplo == CblasUpper) UL = 'L';
      else if (Uplo == CblasLower) UL = 'U';
      else 
      {
         cblas_xerbla(2, "cblas_cher","Illegal Uplo setting, %d\n", Uplo);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
      #endif
      if (N > 0)
      {
         n = N << 1;
         x = malloc(n*sizeof(float));
         tx = x;
         if( incX > 0 ) {
            i = incX << 1 ;
            tincx = 2;
            st= x+n;
         } else { 
            i = incX *(-2);
            tincx = -2;
            st = x-2; 
            x +=(n-2); 
         }
         do
         {
            *x = *xx;
            x[1] = -xx[1];
            x += tincx ;
            xx += i;
         }
         while (x != st);
         x=tx;

         #ifdef F77_INT
           F77_incX = 1;
         #else
           incX = 1;
         #endif
      }
      else x = (float *) X;
      F77_cher(F77_UL, &F77_N, &alpha, (scomplex*)x, &F77_incX, (scomplex*)A, &F77_lda);
   } else 
   {
      cblas_xerbla(1, "cblas_cher","Illegal Order setting, %d\n", order);
      CBLAS_CallFromC = 0;
      RowMajorStrg = 0;
      return;
   }
   if(X!=x) 
      free(x);
   
   CBLAS_CallFromC = 0;
   RowMajorStrg = 0;
   return;
}
#endif
