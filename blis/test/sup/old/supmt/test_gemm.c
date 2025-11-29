/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2019, Advanced Micro Devices, Inc.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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
#ifdef EIGEN
  #define BLIS_DISABLE_BLAS_DEFS
  #include "blis.h"
  #include <Eigen/Core>
  //#include <Eigen/src/misc/blas.h>
  using namespace Eigen;
#else
  #include "blis.h"
#endif

//#define PRINT

int main( int argc, char** argv )
{
	rntm_t rntm_g;

	bli_init();

	// Copy the global rntm_t object so that we can use it later when disabling
	// sup. Starting with a copy of the global rntm_t is actually necessary;
	// if we start off with a locally-initialized rntm_t, it will not contain
	// the ways of parallelism that were conveyed via environment variables,
	// which is necessary when running this driver with multiple BLIS threads.
	bli_rntm_init_from_global( &rntm_g );

#ifndef ERROR_CHECK
	bli_error_checking_level_set( BLIS_NO_ERROR_CHECKING );
#endif


	dim_t n_trials = N_TRIALS;

	num_t dt      = DT;

#if 1
	dim_t p_begin = P_BEGIN;
	dim_t p_max   = P_MAX;
	dim_t p_inc   = P_INC;
#else
	dim_t p_begin = 4;
	dim_t p_max   = 40;
	dim_t p_inc   = 4;
#endif

#if 1
	dim_t m_input = M_DIM;
	dim_t n_input = N_DIM;
	dim_t k_input = K_DIM;
#else
	p_begin = p_inc = 32;
	dim_t m_input = 6;
	dim_t n_input = -1;
	dim_t k_input = -1;
#endif

#if 1
	trans_t transa = TRANSA;
	trans_t transb = TRANSB;
#else
	trans_t transa = BLIS_NO_TRANSPOSE;
	trans_t transb = BLIS_NO_TRANSPOSE;
#endif

#if 1
	stor3_t sc = STOR3;
#else
	stor3_t sc = BLIS_RRR;
#endif


	inc_t    rs_c, cs_c;
	inc_t    rs_a, cs_a;
	inc_t    rs_b, cs_b;

	if      ( sc == BLIS_RRR ) { rs_c = cs_c = -1; rs_a = cs_a = -1; rs_b = cs_b = -1; }
	else if ( sc == BLIS_RRC ) { rs_c = cs_c = -1; rs_a = cs_a = -1; rs_b = cs_b =  0; }
	else if ( sc == BLIS_RCR ) { rs_c = cs_c = -1; rs_a = cs_a =  0; rs_b = cs_b = -1; }
	else if ( sc == BLIS_RCC ) { rs_c = cs_c = -1; rs_a = cs_a =  0; rs_b = cs_b =  0; }
	else if ( sc == BLIS_CRR ) { rs_c = cs_c =  0; rs_a = cs_a = -1; rs_b = cs_b = -1; }
	else if ( sc == BLIS_CRC ) { rs_c = cs_c =  0; rs_a = cs_a = -1; rs_b = cs_b =  0; }
	else if ( sc == BLIS_CCR ) { rs_c = cs_c =  0; rs_a = cs_a =  0; rs_b = cs_b = -1; }
	else if ( sc == BLIS_CCC ) { rs_c = cs_c =  0; rs_a = cs_a =  0; rs_b = cs_b =  0; }
	else                       { bli_abort(); }

	f77_int cbla_storage;

	if      ( sc == BLIS_RRR ) cbla_storage = CblasRowMajor;
	else if ( sc == BLIS_CCC ) cbla_storage = CblasColMajor;
	else                       cbla_storage = -1;

	( void )cbla_storage;


	char dt_ch;

	// Choose the char corresponding to the requested datatype.
	if      ( bli_is_float( dt ) )    dt_ch = 's';
	else if ( bli_is_double( dt ) )   dt_ch = 'd';
	else if ( bli_is_scomplex( dt ) ) dt_ch = 'c';
	else                              dt_ch = 'z';

	f77_char f77_transa;
	f77_char f77_transb;
	char     transal, transbl;

	bli_param_map_blis_to_netlib_trans( transa, &f77_transa );
	bli_param_map_blis_to_netlib_trans( transb, &f77_transb );

	transal = tolower( f77_transa );
	transbl = tolower( f77_transb );

	f77_int cbla_transa = ( transal == 'n' ? CblasNoTrans : CblasTrans );
	f77_int cbla_transb = ( transbl == 'n' ? CblasNoTrans : CblasTrans );

	( void )cbla_transa;
	( void )cbla_transb;

	dim_t p;

	// Begin with initializing the last entry to zero so that
	// matlab allocates space for the entire array once up-front.
	for ( p = p_begin; p + p_inc <= p_max; p += p_inc ) ;

	printf( "data_%s_%cgemm_%c%c_%s", THR_STR, dt_ch,
	                                  transal, transbl, STR );
	printf( "( %2lu, 1:4 ) = [ %4lu %4lu %4lu %7.2f ];\n",
	        ( unsigned long )(p - p_begin)/p_inc + 1,
	        ( unsigned long )0,
	        ( unsigned long )0,
	        ( unsigned long )0, 0.0 );


	//for ( p = p_begin; p <= p_max; p += p_inc )
	for ( p = p_max; p_begin <= p; p -= p_inc )
	{
		obj_t  a, b, c;
		obj_t  c_save;
		obj_t  alpha, beta;
		dim_t  m, n, k;

		if ( m_input < 0 ) m = p / ( dim_t )abs(m_input);
		else               m =     ( dim_t )    m_input;
		if ( n_input < 0 ) n = p / ( dim_t )abs(n_input);
		else               n =     ( dim_t )    n_input;
		if ( k_input < 0 ) k = p / ( dim_t )abs(k_input);
		else               k =     ( dim_t )    k_input;

		bli_obj_create( dt, 1, 1, 0, 0, &alpha );
		bli_obj_create( dt, 1, 1, 0, 0, &beta );

		bli_obj_create( dt, m, n, rs_c, cs_c, &c );
		bli_obj_create( dt, m, n, rs_c, cs_c, &c_save );

		if ( bli_does_notrans( transa ) )
			bli_obj_create( dt, m, k, rs_a, cs_a, &a );
		else
			bli_obj_create( dt, k, m, rs_a, cs_a, &a );

		if ( bli_does_notrans( transb ) )
			bli_obj_create( dt, k, n, rs_b, cs_b, &b );
		else
			bli_obj_create( dt, n, k, rs_b, cs_b, &b );

		bli_randm( &a );
		bli_randm( &b );
		bli_randm( &c );

		bli_obj_set_conjtrans( transa, &a );
		bli_obj_set_conjtrans( transb, &b );

		bli_setsc(  (1.0/1.0), 0.0, &alpha );
		bli_setsc(  (1.0/1.0), 0.0, &beta );

		bli_copym( &c, &c_save );

#ifdef EIGEN
		double alpha_r, alpha_i;

		bli_getsc( &alpha, &alpha_r, &alpha_i );

		void* ap = bli_obj_buffer_at_off( &a );
		void* bp = bli_obj_buffer_at_off( &b );
		void* cp = bli_obj_buffer_at_off( &c );

		const int os_a = ( bli_obj_is_col_stored( &a ) ? bli_obj_col_stride( &a )
		                                               : bli_obj_row_stride( &a ) );
		const int os_b = ( bli_obj_is_col_stored( &b ) ? bli_obj_col_stride( &b )
		                                               : bli_obj_row_stride( &b ) );
		const int os_c = ( bli_obj_is_col_stored( &c ) ? bli_obj_col_stride( &c )
		                                               : bli_obj_row_stride( &c ) );

		Stride<Dynamic,1> stride_a( os_a, 1 );
		Stride<Dynamic,1> stride_b( os_b, 1 );
		Stride<Dynamic,1> stride_c( os_c, 1 );

		#if defined(IS_FLOAT)
		#elif defined (IS_DOUBLE)
			#ifdef A_STOR_R
			typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXd_A;
			#else
			typedef Matrix<double, Dynamic, Dynamic, ColMajor> MatrixXd_A;
			#endif
			#ifdef B_STOR_R
			typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXd_B;
			#else
			typedef Matrix<double, Dynamic, Dynamic, ColMajor> MatrixXd_B;
			#endif
			#ifdef C_STOR_R
			typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXd_C;
			#else
			typedef Matrix<double, Dynamic, Dynamic, ColMajor> MatrixXd_C;
			#endif

			#ifdef A_NOTRANS  // A is not transposed
			Map<MatrixXd_A,  0, Stride<Dynamic,1> > A( ( double* )ap, m, k, stride_a );
			#else // A is transposed
			Map<MatrixXd_A,  0, Stride<Dynamic,1> > A( ( double* )ap, k, m, stride_a );
			#endif

			#ifdef B_NOTRANS // B is not transposed
			Map<MatrixXd_B,  0, Stride<Dynamic,1> > B( ( double* )bp, k, n, stride_b );
			#else // B is transposed
			Map<MatrixXd_B,  0, Stride<Dynamic,1> > B( ( double* )bp, n, k, stride_b );
			#endif

			Map<MatrixXd_C,  0, Stride<Dynamic,1> > C( ( double* )cp, m, n, stride_c );
		#endif
#endif


		double dtime_save = DBL_MAX;

		for ( dim_t r = 0; r < n_trials; ++r )
		{
			bli_copym( &c_save, &c );


			double dtime = bli_clock();


#ifdef EIGEN

		#ifdef A_NOTRANS
			#ifdef B_NOTRANS
			C.noalias() += alpha_r * A * B;
			#else // B_TRANS
			C.noalias() += alpha_r * A * B.transpose();
			#endif
		#else // A_TRANS
			#ifdef B_NOTRANS
			C.noalias() += alpha_r * A.transpose() * B;
			#else // B_TRANS
			C.noalias() += alpha_r * A.transpose() * B.transpose();
			#endif
		#endif

#endif
#ifdef BLIS
	#ifdef SUP
			// Allow sup.
			bli_gemm( &alpha,
			          &a,
			          &b,
			          &beta,
			          &c );
	#else
			// NOTE: We can't use the static initializer and must instead
			// initialize the rntm_t with the copy from the global rntm_t we
			// made at the beginning of main(). Please see the comment there
			// for more info on why BLIS_RNTM_INITIALIZER doesn't work here.
			//rntm_t rntm = BLIS_RNTM_INITIALIZER;
			rntm_t rntm = rntm_g;

			// Disable sup and use the expert interface.
			bli_rntm_disable_l3_sup( &rntm );

			bli_gemm_ex( &alpha,
			             &a,
			             &b,
			             &beta,
			             &c, NULL, &rntm );
	#endif
#endif
#ifdef BLAS
			if ( bli_is_float( dt ) )
			{
				f77_int   mm     = bli_obj_length( &c );
				f77_int   kk     = bli_obj_width_after_trans( &a );
				f77_int   nn     = bli_obj_width( &c );
				f77_int   lda    = bli_obj_col_stride( &a );
				f77_int   ldb    = bli_obj_col_stride( &b );
				f77_int   ldc    = bli_obj_col_stride( &c );
				float*    alphap = ( float* )bli_obj_buffer( &alpha );
				float*    ap     = ( float* )bli_obj_buffer( &a );
				float*    bp     = ( float* )bli_obj_buffer( &b );
				float*    betap  = ( float* )bli_obj_buffer( &beta );
				float*    cp     = ( float* )bli_obj_buffer( &c );

				#ifdef XSMM
				libxsmm_sgemm( &f77_transa,
				#else
				sgemm_( &f77_transa,
				#endif
				        &f77_transb,
				        &mm,
				        &nn,
				        &kk,
				        alphap,
				        ap, &lda,
				        bp, &ldb,
				        betap,
				        cp, &ldc );
			}
			else if ( bli_is_double( dt ) )
			{
				f77_int   mm     = bli_obj_length( &c );
				f77_int   kk     = bli_obj_width_after_trans( &a );
				f77_int   nn     = bli_obj_width( &c );
				f77_int   lda    = bli_obj_col_stride( &a );
				f77_int   ldb    = bli_obj_col_stride( &b );
				f77_int   ldc    = bli_obj_col_stride( &c );
				double*   alphap = ( double* )bli_obj_buffer( &alpha );
				double*   ap     = ( double* )bli_obj_buffer( &a );
				double*   bp     = ( double* )bli_obj_buffer( &b );
				double*   betap  = ( double* )bli_obj_buffer( &beta );
				double*   cp     = ( double* )bli_obj_buffer( &c );

				#ifdef XSMM
				libxsmm_dgemm( &f77_transa,
				#else
				dgemm_( &f77_transa,
				#endif
				        &f77_transb,
				        &mm,
				        &nn,
				        &kk,
				        alphap,
				        ap, &lda,
				        bp, &ldb,
				        betap,
				        cp, &ldc );
			}
			else if ( bli_is_scomplex( dt ) )
			{
				f77_int   mm     = bli_obj_length( &c );
				f77_int   kk     = bli_obj_width_after_trans( &a );
				f77_int   nn     = bli_obj_width( &c );
				f77_int   lda    = bli_obj_col_stride( &a );
				f77_int   ldb    = bli_obj_col_stride( &b );
				f77_int   ldc    = bli_obj_col_stride( &c );
				scomplex* alphap = ( scomplex* )bli_obj_buffer( &alpha );
				scomplex* ap     = ( scomplex* )bli_obj_buffer( &a );
				scomplex* bp     = ( scomplex* )bli_obj_buffer( &b );
				scomplex* betap  = ( scomplex* )bli_obj_buffer( &beta );
				scomplex* cp     = ( scomplex* )bli_obj_buffer( &c );

				#ifdef XSMM
				libxsmm_cgemm( &f77_transa,
				#else
				cgemm_( &f77_transa,
				#endif
				        &f77_transb,
				        &mm,
				        &nn,
				        &kk,
				        alphap,
				        ap, &lda,
				        bp, &ldb,
				        betap,
				        cp, &ldc );
			}
			else if ( bli_is_dcomplex( dt ) )
			{
				f77_int   mm     = bli_obj_length( &c );
				f77_int   kk     = bli_obj_width_after_trans( &a );
				f77_int   nn     = bli_obj_width( &c );
				f77_int   lda    = bli_obj_col_stride( &a );
				f77_int   ldb    = bli_obj_col_stride( &b );
				f77_int   ldc    = bli_obj_col_stride( &c );
				dcomplex* alphap = ( dcomplex* )bli_obj_buffer( &alpha );
				dcomplex* ap     = ( dcomplex* )bli_obj_buffer( &a );
				dcomplex* bp     = ( dcomplex* )bli_obj_buffer( &b );
				dcomplex* betap  = ( dcomplex* )bli_obj_buffer( &beta );
				dcomplex* cp     = ( dcomplex* )bli_obj_buffer( &c );

				#ifdef XSMM
				libxsmm_zgemm( &f77_transa,
				#else
				zgemm_( &f77_transa,
				#endif
				        &f77_transb,
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
#ifdef CBLAS
			if ( bli_is_float( dt ) )
			{
				f77_int   mm     = bli_obj_length( &c );
				f77_int   kk     = bli_obj_width_after_trans( &a );
				f77_int   nn     = bli_obj_width( &c );
	#ifdef C_STOR_R
				f77_int   lda    = bli_obj_row_stride( &a );
				f77_int   ldb    = bli_obj_row_stride( &b );
				f77_int   ldc    = bli_obj_row_stride( &c );
	#else
				f77_int   lda    = bli_obj_col_stride( &a );
				f77_int   ldb    = bli_obj_col_stride( &b );
				f77_int   ldc    = bli_obj_col_stride( &c );
	#endif
				float*    alphap = bli_obj_buffer( &alpha );
				float*    ap     = bli_obj_buffer( &a );
				float*    bp     = bli_obj_buffer( &b );
				float*    betap  = bli_obj_buffer( &beta );
				float*    cp     = bli_obj_buffer( &c );

				cblas_sgemm( cbla_storage,
				             cbla_transa,
				             cbla_transb,
				             mm,
				             nn,
				             kk,
				             *alphap,
				             ap, lda,
				             bp, ldb,
				             *betap,
				             cp, ldc );
			}
			else if ( bli_is_double( dt ) )
			{
				f77_int   mm     = bli_obj_length( &c );
				f77_int   kk     = bli_obj_width_after_trans( &a );
				f77_int   nn     = bli_obj_width( &c );
	#ifdef C_STOR_R
				f77_int   lda    = bli_obj_row_stride( &a );
				f77_int   ldb    = bli_obj_row_stride( &b );
				f77_int   ldc    = bli_obj_row_stride( &c );
	#else
				f77_int   lda    = bli_obj_col_stride( &a );
				f77_int   ldb    = bli_obj_col_stride( &b );
				f77_int   ldc    = bli_obj_col_stride( &c );
	#endif
				double*   alphap = bli_obj_buffer( &alpha );
				double*   ap     = bli_obj_buffer( &a );
				double*   bp     = bli_obj_buffer( &b );
				double*   betap  = bli_obj_buffer( &beta );
				double*   cp     = bli_obj_buffer( &c );

				cblas_dgemm( cbla_storage,
				             cbla_transa,
				             cbla_transb,
				             mm,
				             nn,
				             kk,
				             *alphap,
				             ap, lda,
				             bp, ldb,
				             *betap,
				             cp, ldc );
			}
			else if ( bli_is_scomplex( dt ) )
			{
				f77_int   mm     = bli_obj_length( &c );
				f77_int   kk     = bli_obj_width_after_trans( &a );
				f77_int   nn     = bli_obj_width( &c );
	#ifdef C_STOR_R
				f77_int   lda    = bli_obj_row_stride( &a );
				f77_int   ldb    = bli_obj_row_stride( &b );
				f77_int   ldc    = bli_obj_row_stride( &c );
	#else
				f77_int   lda    = bli_obj_col_stride( &a );
				f77_int   ldb    = bli_obj_col_stride( &b );
				f77_int   ldc    = bli_obj_col_stride( &c );
	#endif
				scomplex* alphap = bli_obj_buffer( &alpha );
				scomplex* ap     = bli_obj_buffer( &a );
				scomplex* bp     = bli_obj_buffer( &b );
				scomplex* betap  = bli_obj_buffer( &beta );
				scomplex* cp     = bli_obj_buffer( &c );

				cblas_cgemm( cbla_storage,
				             cbla_transa,
				             cbla_transb,
				             mm,
				             nn,
				             kk,
				             alphap,
				             ap, lda,
				             bp, ldb,
				             betap,
				             cp, ldc );
			}
			else if ( bli_is_dcomplex( dt ) )
			{
				f77_int   mm     = bli_obj_length( &c );
				f77_int   kk     = bli_obj_width_after_trans( &a );
				f77_int   nn     = bli_obj_width( &c );
	#ifdef C_STOR_R
				f77_int   lda    = bli_obj_row_stride( &a );
				f77_int   ldb    = bli_obj_row_stride( &b );
				f77_int   ldc    = bli_obj_row_stride( &c );
	#else
				f77_int   lda    = bli_obj_col_stride( &a );
				f77_int   ldb    = bli_obj_col_stride( &b );
				f77_int   ldc    = bli_obj_col_stride( &c );
	#endif
				dcomplex* alphap = bli_obj_buffer( &alpha );
				dcomplex* ap     = bli_obj_buffer( &a );
				dcomplex* bp     = bli_obj_buffer( &b );
				dcomplex* betap  = bli_obj_buffer( &beta );
				dcomplex* cp     = bli_obj_buffer( &c );

				cblas_zgemm( cbla_storage,
				             cbla_transa,
				             cbla_transb,
				             mm,
				             nn,
				             kk,
				             alphap,
				             ap, lda,
				             bp, ldb,
				             betap,
				             cp, ldc );
			}
#endif

			dtime_save = bli_clock_min_diff( dtime_save, dtime );
		}

		double gflops = ( 2.0 * m * k * n ) / ( dtime_save * 1.0e9 );

		if ( bli_is_complex( dt ) ) gflops *= 4.0;

		printf( "data_%s_%cgemm_%c%c_%s", THR_STR, dt_ch,
		                                  transal, transbl, STR );
		printf( "( %2lu, 1:4 ) = [ %4lu %4lu %4lu %7.2f ];\n",
		        ( unsigned long )(p - p_begin)/p_inc + 1,
		        ( unsigned long )m,
		        ( unsigned long )n,
		        ( unsigned long )k, gflops );

		bli_obj_free( &alpha );
		bli_obj_free( &beta );

		bli_obj_free( &a );
		bli_obj_free( &b );
		bli_obj_free( &c );
		bli_obj_free( &c_save );
	}

	//bli_finalize();

	return 0;
}

