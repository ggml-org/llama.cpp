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
  #include <Eigen/src/misc/blas.h>
  using namespace Eigen;
#else
  #include "blis.h"
#endif

#define COL_STORAGE
//#define ROW_STORAGE

//#define PRINT

int main( int argc, char** argv )
{
	obj_t    a, b, c;
	obj_t    c_save;
	obj_t    alpha, beta;
	dim_t    m, n, k;
	dim_t    p;
	dim_t    p_begin, p_max, p_inc;
	int      m_input, n_input, k_input;
	ind_t    ind;
	num_t    dt;
	char     dt_ch;
	int      r, n_repeats;
	trans_t  transa;
	trans_t  transb;
	f77_char f77_transa;
	f77_char f77_transb;

	double   dtime;
	double   dtime_save;
	double   gflops;

	//bli_init();

	bli_error_checking_level_set( BLIS_NO_ERROR_CHECKING );

	n_repeats = 3;

	dt      = DT;

	ind     = IND;

#if 1
	p_begin = P_BEGIN;
	p_max   = P_MAX;
	p_inc   = P_INC;

	m_input = -1;
	n_input = -1;
	k_input = -1;
#else
	p_begin = 40;
	p_max   = 2000;
	p_inc   = 40;

	m_input = -1;
	n_input = -1;
	k_input = -1;
#endif


	// Supress compiler warnings about unused variable 'ind'.
	( void )ind;

#if 0

	cntx_t* cntx;

	ind_t ind_mod = ind;

	// Initialize a context for the current induced method and datatype.
	cntx = bli_gks_query_ind_cntx( ind_mod );

	// Set k to the kc blocksize for the current datatype.
	k_input = bli_cntx_get_blksz_def_dt( dt, BLIS_KC, cntx );

#elif 0

	#ifdef BLIS
	if      ( ind == BLIS_1M )   k_input = 128;
	else                         k_input = 256;
	#else
	k_input = 192;
	#endif

#endif

	// Choose the char corresponding to the requested datatype.
	if      ( bli_is_float( dt ) )    dt_ch = 's';
	else if ( bli_is_double( dt ) )   dt_ch = 'd';
	else if ( bli_is_scomplex( dt ) ) dt_ch = 'c';
	else                              dt_ch = 'z';

	transa = BLIS_NO_TRANSPOSE;
	transb = BLIS_NO_TRANSPOSE;

	bli_param_map_blis_to_netlib_trans( transa, &f77_transa );
	bli_param_map_blis_to_netlib_trans( transb, &f77_transb );

	// Begin with initializing the last entry to zero so that
	// matlab allocates space for the entire array once up-front.
	for ( p = p_begin; p + p_inc <= p_max; p += p_inc ) ;

	printf( "data_%s_%cgemm_%s", THR_STR, dt_ch, STR );
	printf( "( %2lu, 1:4 ) = [ %4lu %4lu %4lu %7.2f ];\n",
	        ( unsigned long )(p - p_begin)/p_inc + 1,
	        ( unsigned long )0,
	        ( unsigned long )0,
	        ( unsigned long )0, 0.0 );


	//for ( p = p_begin; p <= p_max; p += p_inc )
	for ( p = p_max; p_begin <= p; p -= p_inc )
	{

		if ( m_input < 0 ) m = p / ( dim_t )abs(m_input);
		else               m =     ( dim_t )    m_input;
		if ( n_input < 0 ) n = p / ( dim_t )abs(n_input);
		else               n =     ( dim_t )    n_input;
		if ( k_input < 0 ) k = p / ( dim_t )abs(k_input);
		else               k =     ( dim_t )    k_input;

		bli_obj_create( dt, 1, 1, 0, 0, &alpha );
		bli_obj_create( dt, 1, 1, 0, 0, &beta );

	#ifdef COL_STORAGE
		bli_obj_create( dt, m, k, 0, 0, &a );
		bli_obj_create( dt, k, n, 0, 0, &b );
		bli_obj_create( dt, m, n, 0, 0, &c );
		bli_obj_create( dt, m, n, 0, 0, &c_save );
	#else
		bli_obj_create( dt, m, k, k, 1, &a );
		bli_obj_create( dt, k, n, n, 1, &b );
		bli_obj_create( dt, m, n, n, 1, &c );
		bli_obj_create( dt, m, n, n, 1, &c_save );
	#endif

		bli_randm( &a );
		bli_randm( &b );
		bli_randm( &c );

		bli_obj_set_conjtrans( transa, &a );
		bli_obj_set_conjtrans( transb, &b );

		bli_setsc(  (1.0/1.0), 0.0, &alpha );
		bli_setsc(  (1.0/1.0), 0.0, &beta );

		bli_copym( &c, &c_save );
	
#ifdef BLIS
		bli_ind_disable_all_dt( dt );
		bli_ind_enable_dt( ind, dt );
#endif

#ifdef EIGEN
		double alpha_r, alpha_i;

		bli_getsc( &alpha, &alpha_r, &alpha_i );

		void* ap = bli_obj_buffer_at_off( &a );
		void* bp = bli_obj_buffer_at_off( &b );
		void* cp = bli_obj_buffer_at_off( &c );

	#ifdef COL_STORAGE
		const int os_a = bli_obj_col_stride( &a );
		const int os_b = bli_obj_col_stride( &b );
		const int os_c = bli_obj_col_stride( &c );
	#else
		const int os_a = bli_obj_row_stride( &a );
		const int os_b = bli_obj_row_stride( &b );
		const int os_c = bli_obj_row_stride( &c );
	#endif

		Stride<Dynamic,1> stride_a( os_a, 1 );
		Stride<Dynamic,1> stride_b( os_b, 1 );
		Stride<Dynamic,1> stride_c( os_c, 1 );

	#ifdef COL_STORAGE
		#if defined(IS_FLOAT)
		typedef Matrix<float,                Dynamic, Dynamic, ColMajor> MatrixXf_;
		#elif defined (IS_DOUBLE)
		typedef Matrix<double,               Dynamic, Dynamic, ColMajor> MatrixXd_;
		#elif defined (IS_SCOMPLEX)
		typedef Matrix<std::complex<float>,  Dynamic, Dynamic, ColMajor> MatrixXcf_;
		#elif defined (IS_DCOMPLEX)
		typedef Matrix<std::complex<double>, Dynamic, Dynamic, ColMajor> MatrixXcd_;
		#endif
	#else
		#if defined(IS_FLOAT)
		typedef Matrix<float,                Dynamic, Dynamic, RowMajor> MatrixXf_;
		#elif defined (IS_DOUBLE)
		typedef Matrix<double,               Dynamic, Dynamic, RowMajor> MatrixXd_;
		#elif defined (IS_SCOMPLEX)
		typedef Matrix<std::complex<float>,  Dynamic, Dynamic, RowMajor> MatrixXcf_;
		#elif defined (IS_DCOMPLEX)
		typedef Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor> MatrixXcd_;
		#endif
	#endif
	#if defined(IS_FLOAT)
		Map<MatrixXf_,  0, Stride<Dynamic,1> > A( ( float*  )ap, m, k, stride_a );
		Map<MatrixXf_,  0, Stride<Dynamic,1> > B( ( float*  )bp, k, n, stride_b );
		Map<MatrixXf_,  0, Stride<Dynamic,1> > C( ( float*  )cp, m, n, stride_c );
	#elif defined (IS_DOUBLE)
		Map<MatrixXd_,  0, Stride<Dynamic,1> > A( ( double* )ap, m, k, stride_a );
		Map<MatrixXd_,  0, Stride<Dynamic,1> > B( ( double* )bp, k, n, stride_b );
		Map<MatrixXd_,  0, Stride<Dynamic,1> > C( ( double* )cp, m, n, stride_c );
	#elif defined (IS_SCOMPLEX)
		Map<MatrixXcf_, 0, Stride<Dynamic,1> > A( ( std::complex<float>*  )ap, m, k, stride_a );
		Map<MatrixXcf_, 0, Stride<Dynamic,1> > B( ( std::complex<float>*  )bp, k, n, stride_b );
		Map<MatrixXcf_, 0, Stride<Dynamic,1> > C( ( std::complex<float>*  )cp, m, n, stride_c );
	#elif defined (IS_DCOMPLEX)
		Map<MatrixXcd_, 0, Stride<Dynamic,1> > A( ( std::complex<double>* )ap, m, k, stride_a );
		Map<MatrixXcd_, 0, Stride<Dynamic,1> > B( ( std::complex<double>* )bp, k, n, stride_b );
		Map<MatrixXcd_, 0, Stride<Dynamic,1> > C( ( std::complex<double>* )cp, m, n, stride_c );
	#endif
#endif

		dtime_save = DBL_MAX;

		for ( r = 0; r < n_repeats; ++r )
		{
			bli_copym( &c_save, &c );

			dtime = bli_clock();

#ifdef PRINT
			bli_printm( "a", &a, "%4.1f", "" );
			bli_printm( "b", &b, "%4.1f", "" );
			bli_printm( "c", &c, "%4.1f", "" );
#endif

#if defined(BLIS)

			bli_gemm( &alpha,
			          &a,
			          &b,
			          &beta,
			          &c );

#elif defined(EIGEN)

			C.noalias() += alpha_r * A * B;

#else // if defined(BLAS)

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

				sgemm_( &f77_transa,
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

				dgemm_( &f77_transa,
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

				cgemm_( &f77_transa,
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

				zgemm_( &f77_transa,
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

#ifdef PRINT
			bli_printm( "c after", &c, "%4.1f", "" );
			exit(1);
#endif

			dtime_save = bli_clock_min_diff( dtime_save, dtime );
		}

		gflops = ( 2.0 * m * k * n ) / ( dtime_save * 1.0e9 );

		if ( bli_is_complex( dt ) ) gflops *= 4.0;

		printf( "data_%s_%cgemm_%s", THR_STR, dt_ch, STR );
		printf( "( %2lu, 1:4 ) = [ %4lu %4lu %4lu %7.2f ];\n",
		        ( unsigned long )(p - p_begin)/p_inc + 1,
		        ( unsigned long )m,
		        ( unsigned long )k,
		        ( unsigned long )n, gflops );
		//fflush( stdout );

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

