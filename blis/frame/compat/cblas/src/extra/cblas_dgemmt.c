#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
   cblas_dgemmt.c
   Based off of cblas_dgemm.c.
*/

/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2020, Advanced Micro Devices, Inc.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "cblas.h"
#include "cblas_f77.h"
void cblas_dgemmt(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE TransA,
                  enum CBLAS_TRANSPOSE TransB, f77_int M, f77_int K,
                  double alpha, const double  *A,
                  f77_int lda, const double  *B, f77_int ldb,
                  double beta, double  *C, f77_int ldc)
{
   char UL, TA, TB;
#ifdef F77_CHAR
   F77_CHAR F77_UL, F77_TA, F77_TB;
#else
   #define F77_UL &UL
   #define F77_TA &TA
   #define F77_TB &TB
#endif

#ifdef F77_INT
   F77_INT F77_M=M, F77_K=K, F77_lda=lda, F77_ldb=ldb;
   F77_INT F77_ldc=ldc;
#else
   #define F77_M M
   #define F77_K K
   #define F77_lda lda
   #define F77_ldb ldb
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
         cblas_xerbla(2, "cblas_dgemmt","Illegal Uplo setting, %d\n", Uplo);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }

      if(TransA == CblasTrans) TA='T';
      else if ( TransA == CblasConjTrans ) TA='C';
      else if ( TransA == CblasNoTrans )   TA='N';
      else
      {
         cblas_xerbla(3, "cblas_dgemmt","Illegal TransA setting, %d\n", TransA);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }

      if(TransB == CblasTrans) TB='T';
      else if ( TransB == CblasConjTrans ) TB='C';
      else if ( TransB == CblasNoTrans )   TB='N';
      else
      {
         cblas_xerbla(4, "cblas_dgemmt","Illegal TransB setting, %d\n", TransB);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }

      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
         F77_TA = C2F_CHAR(&TA);
         F77_TB = C2F_CHAR(&TB);
      #endif

      F77_dgemmt(F77_UL, F77_TA, F77_TB, &F77_M, &F77_K, &alpha, A,
       &F77_lda, B, &F77_ldb, &beta, C, &F77_ldc);
   } else if (Order == CblasRowMajor)
   {
      RowMajorStrg = 1;
      if( Uplo == CblasUpper) UL='L';
      else if ( Uplo == CblasLower ) UL='U';
      else
      {
         cblas_xerbla(2, "cblas_dgemmt","Illegal Uplo setting, %d\n", Uplo);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }

      if(TransA == CblasTrans) TB='T';
      else if ( TransA == CblasConjTrans ) TB='C';
      else if ( TransA == CblasNoTrans )   TB='N';
      else
      {
         cblas_xerbla(3, "cblas_dgemmt","Illegal TransA setting, %d\n", TransA);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      if(TransB == CblasTrans) TA='T';
      else if ( TransB == CblasConjTrans ) TA='C';
      else if ( TransB == CblasNoTrans )   TA='N';
      else
      {
         cblas_xerbla(4, "cblas_dgemmt","Illegal TransB setting, %d\n", TransB);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
         F77_TA = C2F_CHAR(&TA);
         F77_TB = C2F_CHAR(&TB);
      #endif

      F77_dgemmt(F77_UL, F77_TA, F77_TB, &F77_M, &F77_K, &alpha, B,
                  &F77_ldb, A, &F77_lda, &beta, C, &F77_ldc);
   }
   else  cblas_xerbla(1, "cblas_dgemmt", "Illegal Order setting, %d\n", Order);
   CBLAS_CallFromC = 0;
   RowMajorStrg = 0;
   return;
}

void cblas_dgemmtr(enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
                  enum CBLAS_TRANSPOSE TransA,
                  enum CBLAS_TRANSPOSE TransB, f77_int M, f77_int K,
                  double alpha, const double  *A,
                  f77_int lda, const double  *B, f77_int ldb,
                  double beta, double  *C, f77_int ldc)
#ifdef BLIS_OS_OSX
{ cblas_dgemmt(Order, Uplo, TransA, TransB, M, K, alpha, A, lda, B, ldb, beta, C, ldc); }
#else
__attribute__((alias("cblas_dgemmt")));
#endif

#endif
