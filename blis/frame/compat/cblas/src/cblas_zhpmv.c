#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_zhpmv.c
 * The program is a C interface of zhpmv
 * 
 * Keita Teranishi  5/18/98
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"
#include "cblas_f77.h"
void cblas_zhpmv(enum CBLAS_ORDER order,
                 enum CBLAS_UPLO Uplo,f77_int N,
                 const void *alpha, const void  *AP,
                 const void  *X, f77_int incX, const void *beta,
                 void  *Y, f77_int incY)
{
   char UL;
#ifdef F77_CHAR
   F77_CHAR F77_UL;
#else
   #define F77_UL &UL   
#endif
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX, F77_incY=incY;
#else
   #define F77_N N
   #define F77_incX incX
   #define F77_incY incY
#endif
   int n, i=0;
   const double *xx= (double *)X, *alp= (double *)alpha, *bet = (double *)beta;
   double ALPHA[2],BETA[2];
   int tincY, tincx;
   double *x=(double *)X, *y=(double *)Y, *st=0, *tx;
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
         cblas_xerbla(2, "cblas_zhpmv","Illegal Uplo setting, %d\n",Uplo );
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
      #endif
      F77_zhpmv(F77_UL, &F77_N, (dcomplex*)alpha, (dcomplex*)AP, (dcomplex*)X,
                     &F77_incX, (dcomplex*)beta, (dcomplex*)Y, &F77_incY);
   }
   else if (order == CblasRowMajor)
   {
      RowMajorStrg = 1;
      ALPHA[0]= *alp;
      ALPHA[1]= -alp[1];
      BETA[0]= *bet;
      BETA[1]= -bet[1];

      if (N > 0)
      {
         n = N << 1;
         x = malloc(n*sizeof(double));
 
         tx = x;
         if( incX > 0 ) {
           i = incX << 1;
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
 
         if(incY > 0)
           tincY = incY;
         else
           tincY = -incY;
         y++;

         i = tincY << 1;
         n = i * N ;
         st = y + n;
         do {
            *y = -(*y);
            y += i;
         } while(y != st);
         y -= n;
      }  else
         x = (double *) X;


      if (Uplo == CblasUpper) UL = 'L';
      else if (Uplo == CblasLower) UL = 'U';
      else 
      {
         cblas_xerbla(2, "cblas_zhpmv","Illegal Uplo setting, %d\n", Uplo );
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
      #endif

      F77_zhpmv(F77_UL, &F77_N, (dcomplex*)ALPHA,
                (dcomplex*)AP, (dcomplex*)x, &F77_incX, (dcomplex*)BETA, (dcomplex*)Y, &F77_incY);
   }
   else 
   {
      cblas_xerbla(1, "cblas_zhpmv","Illegal Order setting, %d\n", order);
      CBLAS_CallFromC = 0;
      RowMajorStrg = 0;
      return;
   }
   if ( order == CblasRowMajor ) 
   {
      RowMajorStrg = 1;
      if(X!=x)
         free(x);
      if (N > 0)
      {
         do
         {
            *y = -(*y);
            y += i;
         }
         while (y != st);
     }
  }

   CBLAS_CallFromC = 0;
   RowMajorStrg = 0;
   return;
}
#endif
