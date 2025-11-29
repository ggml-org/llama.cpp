/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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

#include <unistd.h>
#include "blis.h"

/*
void     dgemm_ ( char* transa, char* transb, int* m, int* n, int* k, double*   alpha, double*   a, int* lda, double*   b, int* ldb, double*   beta, double*   c, int* ldc );
void     zhemm_ ( char* side, char* uplo, int* m, int* n, dcomplex* alpha, dcomplex* a, int* lda, dcomplex* b, int* ldb, dcomplex* beta, dcomplex* c, int* ldc );
void     zherk_ ( char* uplo, char* transa, int* n, int* k, double* alpha, dcomplex* a, int* lda, double* beta, dcomplex* c, int* ldc );
void     zher2k_( char* uplo, char* transa, int* n, int* k, dcomplex* alpha, dcomplex* a, int* lda, dcomplex* b, int* ldb, double* beta, dcomplex* c, int* ldc );
void     dsymm_ ( char* side, char* uplo, int* m, int* n, double*   alpha, double*   a, int* lda, double*   b, int* ldb, double*   beta, double*   c, int* ldc );
void     dsyrk_ ( char* uplo, char* transa, int* n, int* k, double*   alpha, double*   a, int* lda, double*   beta, double*   c, int* ldc );
void     dsyr2k_( char* uplo, char* transa, int* n, int* k, double*   alpha, double*   a, int* lda, double*   b, int* ldb, double*   beta, double*   c, int* ldc );
void     dtrmm_ ( char* side, char* uplo, char* transa, char* diag, int* m, int* n, double*   alpha, double*   a, int* lda, double*   b, int* ldb );
void     dtrsm_ ( char* side, char* uplo, char* transa, char* diag, int* m, int* n, double*   alpha, double*   a, int* lda, double*   b, int* ldb );
*/

int main( int argc, char** argv )
{
	obj_t a, b, c;
	obj_t x, y;
	obj_t alpha, beta;
	dim_t m;
	num_t dt_a, dt_b, dt_c;
	num_t dt_alpha, dt_beta;
	int   ii;

#ifdef NBLIS
	//bli_init();
#endif


	m = 4000;

	dt_a = BLIS_DOUBLE;
	dt_b = BLIS_DOUBLE;
	dt_c = BLIS_DOUBLE;
	dt_alpha = BLIS_DOUBLE;
	dt_beta = BLIS_DOUBLE;

	{


#ifdef NBLIS
		bli_obj_create( dt_alpha, 1, 1, 0, 0, &alpha );
		bli_obj_create( dt_beta,  1, 1, 0, 0, &beta );

		bli_obj_create( dt_a, m, 1, 0, 0, &x );
		bli_obj_create( dt_a, m, 1, 0, 0, &y );

		bli_obj_create( dt_a, m, m, 0, 0, &a );
		bli_obj_create( dt_b, m, m, 0, 0, &b );
		bli_obj_create( dt_c, m, m, 0, 0, &c );

		bli_randm( &a );
		bli_randm( &b );
		bli_randm( &c );

		bli_setsc(  (2.0/1.0), 0.0, &alpha );
		bli_setsc(  (1.0/1.0), 0.0, &beta );

#endif

#ifdef NBLAS
		x.buffer     = malloc( m * 1 * sizeof( double ) );
		y.buffer     = malloc( m * 1 * sizeof( double ) );

		alpha.buffer = malloc( 1 * sizeof( double ) );
		beta.buffer  = malloc( 1 * sizeof( double ) );

		a.buffer     = malloc( m * m * sizeof( double ) );
		bli_obj_set_dims( m, m, &a );
		bli_obj_set_strides( m, 1, &a );

		b.buffer     = malloc( m * m * sizeof( double ) );
		bli_obj_set_dims( m, m, &b );
		bli_obj_set_strides( m, 1, &b );

		c.buffer     = malloc( m * m * sizeof( double ) );
		bli_obj_set_dims( m, m, &c );
		bli_obj_set_strides( m, 1, &c );

		*((double*)alpha.buffer) =  2.0;
		*((double*)beta.buffer)  = -1.0;
#endif
	

#ifdef NBLIS

	#if NBLIS >= 1
		for ( ii = 0; ii < 2000000000; ++ii )
		{
			bli_gemm( &BLIS_ONE,
			          &a,
			          &b,
			          &BLIS_ONE,
			          &c );
		}
	#endif

	#if NBLIS >= 2
		{
			bli_hemm( BLIS_LEFT,
			          &BLIS_ONE,
			          &a,
			          &b,
			          &BLIS_ONE,
			          &c );
		}
	#endif

	#if NBLIS >= 3
		{
			bli_herk( &BLIS_ONE,
			          &a,
			          &BLIS_ONE,
			          &c );
		}
	#endif

	#if NBLIS >= 4
		{
			bli_her2k( &BLIS_ONE,
			           &a,
			           &b,
			           &BLIS_ONE,
			           &c );
		}
	#endif

	#if NBLIS >= 5
		{
			bli_trmm( BLIS_LEFT,
			          &BLIS_ONE,
			          &a,
			          &c );
		}
	#endif

	#if NBLIS >= 6
		{
			bli_trsm( BLIS_LEFT,
			          &BLIS_ONE,
			          &a,
			          &c );
		}
	#endif

#endif



#ifdef NBLAS

	#if NBLAS >= 1
		for ( ii = 0; ii < 2000000000; ++ii )
		{
			f77_char transa = 'N';
			f77_char transb = 'N';
			f77_int  mm     = bli_obj_length( &c );
			f77_int  kk     = bli_obj_width_after_trans( &a );
			f77_int  nn     = bli_obj_width( &c );
			f77_int  lda    = bli_obj_col_stride( &a );
			f77_int  ldb    = bli_obj_col_stride( &b );
			f77_int  ldc    = bli_obj_col_stride( &c );
			double*  alphap = bli_obj_buffer( &alpha );
			double*  ap     = bli_obj_buffer( &a );
			double*  bp     = bli_obj_buffer( &b );
			double*  betap  = bli_obj_buffer( &beta );
			double*  cp     = bli_obj_buffer( &c );

			dgemm_( &transa,
			        &transb,
			        &mm,
			        &nn,
			        &kk,
			        alphap,
			        ap, &lda,
			        bp, &ldb,
			        betap,
			        cp, &ldc );
		}
	#endif

	#if NBLAS >= 2
		{
			f77_char side   = 'L';
			f77_char uplo   = 'L';
			f77_int  mm     = bli_obj_length( &c );
			f77_int  nn     = bli_obj_width( &c );
			f77_int  lda    = bli_obj_col_stride( &a );
			f77_int  ldb    = bli_obj_col_stride( &b );
			f77_int  ldc    = bli_obj_col_stride( &c );
			double*  alphap = bli_obj_buffer( &alpha );
			double*  ap     = bli_obj_buffer( &a );
			double*  bp     = bli_obj_buffer( &b );
			double*  betap  = bli_obj_buffer( &beta );
			double*  cp     = bli_obj_buffer( &c );

			dsymm_( &side,
			        &uplo,
			        &mm,
			        &nn,
			        alphap,
			        ap, &lda,
			        bp, &ldb,
			        betap,
			        cp, &ldc );
		}
	#endif

	#if NBLAS >= 3
		{
			f77_char uplo   = 'L';
			f77_char trans  = 'N';
			f77_int  mm     = bli_obj_length( &c );
			f77_int  kk     = bli_obj_width( &a );
			f77_int  lda    = bli_obj_col_stride( &a );
			f77_int  ldc    = bli_obj_col_stride( &c );
			double*  alphap = bli_obj_buffer( &alpha );
			double*  ap     = bli_obj_buffer( &a );
			double*  betap  = bli_obj_buffer( &beta );
			double*  cp     = bli_obj_buffer( &c );

			dsyrk_( &uplo,
			        &trans,
			        &mm,
			        &kk,
			        alphap,
			        ap, &lda,
			        betap,
			        cp, &ldc );
		}
	#endif

	#if NBLAS >= 4
		{
			f77_char uplo   = 'L';
			f77_char trans  = 'N';
			f77_int  mm     = bli_obj_length( &c );
			f77_int  kk     = bli_obj_width( &a );
			f77_int  lda    = bli_obj_col_stride( &a );
			f77_int  ldb    = bli_obj_col_stride( &b );
			f77_int  ldc    = bli_obj_col_stride( &c );
			double*  alphap = bli_obj_buffer( &alpha );
			double*  ap     = bli_obj_buffer( &a );
			double*  bp     = bli_obj_buffer( &b );
			double*  betap  = bli_obj_buffer( &beta );
			double*  cp     = bli_obj_buffer( &c );

			dsyr2k_( &uplo,
			         &trans,
			         &mm,
			         &kk,
			         alphap,
			         ap, &lda,
			         bp, &ldb,
			         betap,
			         cp, &ldc );
		}
	#endif

	#if NBLAS >= 5
		{
			f77_char side   = 'L';
			f77_char uplo   = 'L';
			f77_char trans  = 'N';
			f77_char diag   = 'N';
			f77_int  mm     = bli_obj_length( &c );
			f77_int  nn     = bli_obj_width( &c );
			f77_int  lda    = bli_obj_col_stride( &a );
			f77_int  ldc    = bli_obj_col_stride( &c );
			double*  alphap = bli_obj_buffer( &alpha );
			double*  ap     = bli_obj_buffer( &a );
			double*  cp     = bli_obj_buffer( &c );

			dtrmm_( &side,
			        &uplo,
			        &trans,
			        &diag,
			        &mm,
			        &nn,
			        alphap,
			        ap, &lda,
			        cp, &ldc );
		}
	#endif

	#if NBLAS >= 6
		{
			f77_char side   = 'L';
			f77_char uplo   = 'L';
			f77_char trans  = 'N';
			f77_char diag   = 'N';
			f77_int  mm     = bli_obj_length( &c );
			f77_int  nn     = bli_obj_width( &c );
			f77_int  lda    = bli_obj_col_stride( &a );
			f77_int  ldc    = bli_obj_col_stride( &c );
			double*  alphap = bli_obj_buffer( &alpha );
			double*  ap     = bli_obj_buffer( &a );
			double*  cp     = bli_obj_buffer( &c );

			dtrsm_( &side,
			        &uplo,
			        &trans,
			        &diag,
			        &mm,
			        &nn,
			        alphap,
			        ap, &lda,
			        cp, &ldc );
		}
	#endif

	#if NBLAS >= 7
		{
			f77_char  transa = 'N';
			f77_char  transb = 'N';
			f77_int   mm     = bli_obj_length( &c );
			f77_int   kk     = bli_obj_width_after_trans( &a );
			f77_int   nn     = bli_obj_width( &c );
			f77_int   lda    = bli_obj_col_stride( &a );
			f77_int   ldb    = bli_obj_col_stride( &b );
			f77_int   ldc    = bli_obj_col_stride( &c );
			dcomplex* alphap = bli_obj_buffer( &alpha );
			dcomplex* ap     = bli_obj_buffer( &a );
			dcomplex* bp     = bli_obj_buffer( &b );
			dcomplex* betap  = bli_obj_buffer( &beta );
			dcomplex* cp     = bli_obj_buffer( &c );

			zgemm_( &transa,
			        &transb,
			        &mm,
			        &nn,
			        &kk,
			        alphap,
			        ap, &lda,
			        bp, &ldb,
			        betap,
			        cp, &ldc );
		}
	#endif

	#if NBLAS >= 8
		{
			f77_char  side   = 'L';
			f77_char  uplo   = 'L';
			f77_int   mm     = bli_obj_length( &c );
			f77_int   nn     = bli_obj_width( &c );
			f77_int   lda    = bli_obj_col_stride( &a );
			f77_int   ldb    = bli_obj_col_stride( &b );
			f77_int   ldc    = bli_obj_col_stride( &c );
			dcomplex* alphap = bli_obj_buffer( &alpha );
			dcomplex* ap     = bli_obj_buffer( &a );
			dcomplex* bp     = bli_obj_buffer( &b );
			dcomplex* betap  = bli_obj_buffer( &beta );
			dcomplex* cp     = bli_obj_buffer( &c );

			zhemm_( &side,
			        &uplo,
			        &mm,
			        &nn,
			        alphap,
			        ap, &lda,
			        bp, &ldb,
			        betap,
			        cp, &ldc );
		}
	#endif

	#if NBLAS >= 9
		{
			f77_char  uplo   = 'L';
			f77_char  trans  = 'N';
			f77_int   mm     = bli_obj_length( &c );
			f77_int   kk     = bli_obj_width( &a );
			f77_int   lda    = bli_obj_col_stride( &a );
			f77_int   ldc    = bli_obj_col_stride( &c );
			double*   alphap = bli_obj_buffer( &alpha );
			dcomplex* ap     = bli_obj_buffer( &a );
			double*   betap  = bli_obj_buffer( &beta );
			dcomplex* cp     = bli_obj_buffer( &c );

			zherk_( &uplo,
			        &trans,
			        &mm,
			        &kk,
			        alphap,
			        ap, &lda,
			        betap,
			        cp, &ldc );
		}
	#endif

	#if NBLAS >= 10
		{
			f77_char  uplo   = 'L';
			f77_char  trans  = 'N';
			f77_int   mm     = bli_obj_length( &c );
			f77_int   kk     = bli_obj_width( &a );
			f77_int   lda    = bli_obj_col_stride( &a );
			f77_int   ldb    = bli_obj_col_stride( &b );
			f77_int   ldc    = bli_obj_col_stride( &c );
			dcomplex* alphap = bli_obj_buffer( &alpha );
			dcomplex* ap     = bli_obj_buffer( &a );
			dcomplex* bp     = bli_obj_buffer( &b );
			double*   betap  = bli_obj_buffer( &beta );
			dcomplex* cp     = bli_obj_buffer( &c );

			zher2k_( &uplo,
			         &trans,
			         &mm,
			         &kk,
			         alphap,
			         ap, &lda,
			         bp, &ldb,
			         betap,
			         cp, &ldc );
		}
	#endif

	#if NBLAS >= 11
		{
			f77_char  side   = 'L';
			f77_char  uplo   = 'L';
			f77_char  trans  = 'N';
			f77_char  diag   = 'N';
			f77_int   mm     = bli_obj_length( &c );
			f77_int   nn     = bli_obj_width( &c );
			f77_int   lda    = bli_obj_col_stride( &a );
			f77_int   ldc    = bli_obj_col_stride( &c );
			dcomplex* alphap = bli_obj_buffer( &alpha );
			dcomplex* ap     = bli_obj_buffer( &a );
			dcomplex* cp     = bli_obj_buffer( &c );

			ztrmm_( &side,
			        &uplo,
			        &trans,
			        &diag,
			        &mm,
			        &nn,
			        alphap,
			        ap, &lda,
			        cp, &ldc );
		}
	#endif

	#if NBLAS >= 12
		{
			f77_char  side   = 'L';
			f77_char  uplo   = 'L';
			f77_char  trans  = 'N';
			f77_char  diag   = 'N';
			f77_int   mm     = bli_obj_length( &c );
			f77_int   nn     = bli_obj_width( &c );
			f77_int   lda    = bli_obj_col_stride( &a );
			f77_int   ldc    = bli_obj_col_stride( &c );
			dcomplex* alphap = bli_obj_buffer( &alpha );
			dcomplex* ap     = bli_obj_buffer( &a );
			dcomplex* cp     = bli_obj_buffer( &c );

			ztrsm_( &side,
			        &uplo,
			        &trans,
			        &diag,
			        &mm,
			        &nn,
			        alphap,
			        ap, &lda,
			        cp, &ldc );
		}
	#endif


#endif


#ifdef NBLIS
		bli_obj_free( &x );
		bli_obj_free( &y );

		bli_obj_free( &alpha );
		bli_obj_free( &beta );

		bli_obj_free( &a );
		bli_obj_free( &b );
		bli_obj_free( &c );
#endif

#ifdef NBLAS
		free( x.buffer );
		free( y.buffer );

		free( alpha.buffer );
		free( beta.buffer );

		free( a.buffer );
		free( b.buffer );
		free( c.buffer );
#endif
	}

#ifdef NBLIS
	//bli_finalize();
#endif

	return 0;
}

