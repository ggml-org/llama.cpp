#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_zhpr2.c
 * The program is a C interface to zhpr2.
 * 
 * Keita Teranishi  5/20/98
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"
#include "cblas_f77.h"
void cblas_zhpr2(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo,
                      f77_int N,const void *alpha, const void *X, 
                      f77_int incX,const void *Y, f77_int incY, void *Ap)

{
   char UL;
#ifdef F77_CHAR
   F77_CHAR F77_UL;
#else
   #define F77_UL &UL
#endif

#ifdef F77_INT
   F77_INT F77_N=N,  F77_incX=incX, F77_incY=incY;
#else
   #define F77_N N
   #define F77_incX incX
   #define F77_incY incY
#endif
   int n, i, j;
   double *x=(double *)X, *xx=(double *)X, *y=(double *)Y,
         *yy=(double *)Y, *stx, *sty;

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
         cblas_xerbla(2, "cblas_zhpr2","Illegal Uplo setting, %d\n",Uplo );
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
      #endif

      F77_zhpr2(F77_UL, &F77_N, (dcomplex*)alpha, (dcomplex*)X, &F77_incX, (dcomplex*)Y, &F77_incY, (dcomplex*)Ap);

   }  else if (order == CblasRowMajor)
   {
      RowMajorStrg = 1;
      if (Uplo == CblasUpper) UL = 'L';
      else if (Uplo == CblasLower) UL = 'U';
      else 
      {
         cblas_xerbla(2, "cblas_zhpr2","Illegal Uplo setting, %d\n", Uplo);
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
         x = malloc(n*sizeof(double));
         y = malloc(n*sizeof(double));         
         stx = x + n;
         sty = y + n;
         if( incX > 0 )
            i = incX << 1;
         else
            i = incX *(-2);
 
         if( incY > 0 )
            j = incY << 1;
         else
            j = incY *(-2);
         do
         {
            *x = *xx;
            x[1] = -xx[1];
            x += 2;
            xx += i;
         } while (x != stx);
         do
         {
            *y = *yy;
            y[1] = -yy[1];
            y += 2;
            yy += j;
         }
         while (y != sty);
         x -= n;
         y -= n;

         #ifdef F77_INT
            if(incX > 0 )
               F77_incX = 1;
            else
               F77_incX = -1;
 
            if(incY > 0 )
               F77_incY = 1;
            else
               F77_incY = -1;
 
         #else
            if(incX > 0 )
               incX = 1;
            else
               incX = -1;
 
            if(incY > 0 )
               incY = 1;
            else
               incY = -1;
         #endif

      }  else 
      {
         x = (double *) X;
         y = (void  *) Y;
      }
      F77_zhpr2(F77_UL, &F77_N, (dcomplex*)alpha, (dcomplex*)y, &F77_incY, (dcomplex*)x, &F77_incX, (dcomplex*)Ap);
   } 
   else 
   {
      cblas_xerbla(1, "cblas_zhpr2","Illegal Order setting, %d\n", order);
      CBLAS_CallFromC = 0;
      RowMajorStrg = 0;
      return;
   }
   if(X!=x)
      free(x);
   if(Y!=y)
      free(y);
   CBLAS_CallFromC = 0;
   RowMajorStrg = 0;
   return;
}
#endif
