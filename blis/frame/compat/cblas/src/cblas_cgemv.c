#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_cgemv.c
 * The program is a C interface of cgemv
 * 
 * Keita Teranishi  5/20/98
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"
#include "cblas_f77.h"
void cblas_cgemv(enum CBLAS_ORDER order,
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

   int n=0, i=0;
   const float *xx= (const float *)X;
   float ALPHA[2],BETA[2];
   int tincY, tincx;
   float *x=(float *)X, *y=(float *)Y, *st=0, *tx=0;
   const float *stx = x;
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
         cblas_xerbla(2, "cblas_cgemv","Illegal TransA setting, %d\n", TransA);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      #ifdef F77_CHAR
         F77_TA = C2F_CHAR(&TA);
      #endif
      F77_cgemv(F77_TA, &F77_M, &F77_N, (scomplex*)alpha, (scomplex*)A, &F77_lda, (scomplex*)X, &F77_incX,
                (scomplex*)beta, (scomplex*)Y, &F77_incY);
   }
   else if (order == CblasRowMajor)
   {
      RowMajorStrg = 1;
         
      if (TransA == CblasNoTrans) TA = 'T';
      else if (TransA == CblasTrans) TA = 'N';
      else if (TransA == CblasConjTrans)
      {
         ALPHA[0]=    *( (const float *)  alpha    );
         ALPHA[1]= -( *( (const float *)  alpha+1) );
         BETA[0]=     *( (const float *)  beta     );
         BETA[1]= -(  *( (const float *)  beta+1 ) );
         TA = 'N';
         if (M > 0)
         {
            n = M << 1;
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

            F77_incX = 1;

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
            stx = x;
         }
         else stx = (const float *)X;
      }
      else 
      {
         cblas_xerbla(2, "cblas_cgemv","Illegal TransA setting, %d\n", TransA);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      #ifdef F77_CHAR
         F77_TA = C2F_CHAR(&TA);
      #endif
      if (TransA == CblasConjTrans)
         F77_cgemv(F77_TA, &F77_N, &F77_M, (scomplex*)ALPHA, (scomplex*)A, &F77_lda, (scomplex*)stx,
                &F77_incX, (scomplex*)BETA, (scomplex*)Y, &F77_incY);
      else
         F77_cgemv(F77_TA, &F77_N, &F77_M, (scomplex*)alpha, (scomplex*)A, &F77_lda, (scomplex*)x,
                &F77_incX, (scomplex*)beta, (scomplex*)Y, &F77_incY);

      if (TransA == CblasConjTrans)
      {
         if (x != (const float *)X) free(x);
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
   else cblas_xerbla(1, "cblas_cgemv", "Illegal Order setting, %d\n", order);
   CBLAS_CallFromC = 0;
   RowMajorStrg = 0;
   return;
}
#endif
