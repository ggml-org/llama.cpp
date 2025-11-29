#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_zgemv.c
 * The program is a C interface of zgemv
 * 
 * Keita Teranishi  5/20/98
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"
#include "cblas_f77.h"
void cblas_zgemv(enum CBLAS_ORDER order,
                 enum CBLAS_TRANSPOSE TransA, f77_int M, f77_int N,
                 const void *alpha, const void  *A, f77_int lda,
                 const void  *X, f77_int incX, const void *beta,
                 void  *Y, f77_int incY)
{
   char TA;
#ifdef F77_CHAR
   F77_CHAR F77_TA;
#else
   #define F77_TA &TA   
#endif
#ifdef F77_INT
   F77_INT F77_M=M, F77_N=N, F77_lda=lda, F77_incX=incX, F77_incY=incY;
#else
   #define F77_M M
   #define F77_N N
   #define F77_lda lda
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
      if (TransA == CblasNoTrans) TA = 'N';
      else if (TransA == CblasTrans) TA = 'T';
      else if (TransA == CblasConjTrans) TA = 'C';
      else 
      {
         cblas_xerbla(2, "cblas_zgemv","Illegal TransA setting, %d\n", TransA);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      #ifdef F77_CHAR
         F77_TA = C2F_CHAR(&TA);
      #endif
      F77_zgemv(F77_TA, &F77_M, &F77_N, (dcomplex*)alpha, (dcomplex*)A, &F77_lda, (dcomplex*)X, &F77_incX,
                (dcomplex*)beta, (dcomplex*)Y, &F77_incY);
   }
   else if (order == CblasRowMajor)
   {
      RowMajorStrg = 1;
         
      if (TransA == CblasNoTrans) TA = 'T';
      else if (TransA == CblasTrans) TA = 'N';
      else if (TransA == CblasConjTrans)
      {
         ALPHA[0]= *alp;
         ALPHA[1]= -alp[1];
         BETA[0]= *bet;
         BETA[1]= -bet[1];
         TA = 'N';
         if (M > 0)
         {
            n = M << 1;
            x = malloc(n*sizeof(double));
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

            if(incY > 0)
               tincY = incY; 
            else
               tincY = -incY; 

            y++;

            if (N > 0)
            {
               i = tincY << 1;
               n = i * N ;
               st = y + n;
               do {
                  *y = -(*y);
                  y += i;
               } while(y != st); 
               y -= n;
            }
         }
         else x = (double *) X;
      }
      else 
      {
         cblas_xerbla(2, "cblas_zgemv","Illegal TransA setting, %d\n", TransA);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      #ifdef F77_CHAR
         F77_TA = C2F_CHAR(&TA);
      #endif
      if (TransA == CblasConjTrans)
         F77_zgemv(F77_TA, &F77_N, &F77_M, (dcomplex*)ALPHA, (dcomplex*)A, &F77_lda, (dcomplex*)x,
                &F77_incX, (dcomplex*)BETA, (dcomplex*)Y, &F77_incY);
      else
         F77_zgemv(F77_TA, &F77_N, &F77_M, (dcomplex*)alpha, (dcomplex*)A, &F77_lda, (dcomplex*)x,
                &F77_incX, (dcomplex*)beta, (dcomplex*)Y, &F77_incY);

      if (TransA == CblasConjTrans)
      {
         if (x != (double *)X) free(x);
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
   }
   else cblas_xerbla(1, "cblas_zgemv", "Illegal Order setting, %d\n", order);
   CBLAS_CallFromC = 0;
   RowMajorStrg = 0;
   return;
}
#endif
