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
  #include "test_utils.h"
  #include <Eigen/Core>
  #include <Eigen/src/misc/blas.h>
  using namespace Eigen;
#else
  #include "blis.h"
  #include "test_utils.h"
#endif

//#define PRINT

static const char* LOCAL_OPNAME_STR = "gemm";
static const char* LOCAL_PC_STR     = "nn";

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

	params_t params;

	// Supress compiler warnings about unused variable 'ind'.
	( void )ind;


	//bli_init();

	//bli_error_checking_level_set( BLIS_NO_ERROR_CHECKING );

	// Parse the command line options into strings, integers, enums,
	// and doubles, as appropriate.
	parse_cl_params( argc, argv, init_def_params, &params );

	dt        = params.dt;

	ind       = params.im;

	p_begin   = params.sta;
	p_max     = params.end;
	p_inc     = params.inc;

	m_input   = params.m;
	n_input   = params.n;
	k_input   = params.k;

	n_repeats = params.nr;


	// Map the datatype to its corresponding char.
	bli_param_map_blis_to_char_dt( dt, &dt_ch );

	// Map the parameter chars to their corresponding BLIS enum type values.
	bli_param_map_char_to_blis_trans( params.pc_str[0], &transa );
	bli_param_map_char_to_blis_trans( params.pc_str[1], &transb );

	// Map the BLIS enum type values to their corresponding BLAS chars.
	bli_param_map_blis_to_netlib_trans( transa, &f77_transa );
	bli_param_map_blis_to_netlib_trans( transb, &f77_transb );

	// Begin with initializing the last entry to zero so that
	// matlab allocates space for the entire array once up-front.
	for ( p = p_begin; p + p_inc <= p_max; p += p_inc ) ;

	printf( "data_%s_%cgemm_%s", THR_STR, dt_ch, IMPL_STR );
	printf( "( %4lu, 1:4 ) = [ %5lu %5lu %5lu %8.2f ];\n",
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

		// Choose the storage of each matrix based on the corresponding
		// char in the params_t struct. Note that the expected order of
		// storage specifers in sc_str is CAB (not ABC).
		if ( params.sc_str[1] == 'c' ) bli_obj_create( dt, m, k, 0, 0, &a );
		else                           bli_obj_create( dt, m, k, k, 1, &a );

		if ( params.sc_str[2] == 'c' ) bli_obj_create( dt, k, n, 0, 0, &b );
		else                           bli_obj_create( dt, k, n, n, 1, &b );

		if ( params.sc_str[0] == 'c' ) bli_obj_create( dt, m, n, 0, 0, &c );
		else                           bli_obj_create( dt, m, n, n, 1, &c );

		if ( params.sc_str[0] == 'c' ) bli_obj_create( dt, m, n, 0, 0, &c_save );
		else                           bli_obj_create( dt, m, n, n, 1, &c_save );

		bli_randm( &a );
		bli_randm( &b );
		bli_randm( &c );

		bli_obj_set_conjtrans( transa, &a );
		bli_obj_set_conjtrans( transb, &b );

		//bli_setsc(  (2.0/1.0), 0.0, &alpha );
		//bli_setsc(  (1.0/1.0), 0.0, &beta );
		bli_setsc( params.alpha, 0.0, &alpha );
		bli_setsc( params.beta,  0.0, &beta );

		//bli_printm( "alpha:", &alpha, "%7.4e", "" );
		//bli_printm( "beta: ", &beta,  "%7.4e", "" );

		bli_copym( &c, &c_save );

#ifdef BLIS
		// Switch to the induced method specified by ind, unless the 'auto'
		// option was given, in which case we leave the induced method
		// unchanged.
		if ( params.im_is_auto == FALSE )
		{
			bli_ind_disable_all_dt( dt );
			bli_ind_enable_dt( ind, dt );
		}
#endif

#ifdef EIGEN
		double alpha_r, alpha_i;

		bli_getsc( &alpha, &alpha_r, &alpha_i );

		void* ap = bli_obj_buffer_at_off( &a );
		void* bp = bli_obj_buffer_at_off( &b );
		void* cp = bli_obj_buffer_at_off( &c );

		int os_a, os_b, os_c;

		if ( params.sc_str[0] == 'c' )
		{
			os_a = bli_obj_col_stride( &a );
			os_b = bli_obj_col_stride( &b );
			os_c = bli_obj_col_stride( &c );
		}
		else
		{
			os_a = bli_obj_row_stride( &a );
			os_b = bli_obj_row_stride( &b );
			os_c = bli_obj_row_stride( &c );
		}

		Stride<Dynamic,1> stride_a( os_a, 1 );
		Stride<Dynamic,1> stride_b( os_b, 1 );
		Stride<Dynamic,1> stride_c( os_c, 1 );

		typedef Matrix<float,                Dynamic, Dynamic, ColMajor> MatrixXs_c;
		typedef Matrix<double,               Dynamic, Dynamic, ColMajor> MatrixXd_c;
		typedef Matrix<std::complex<float>,  Dynamic, Dynamic, ColMajor> MatrixXc_c;
		typedef Matrix<std::complex<double>, Dynamic, Dynamic, ColMajor> MatrixXz_c;

		typedef Matrix<float,                Dynamic, Dynamic, RowMajor> MatrixXs_r;
		typedef Matrix<double,               Dynamic, Dynamic, RowMajor> MatrixXd_r;
		typedef Matrix<std::complex<float>,  Dynamic, Dynamic, RowMajor> MatrixXc_r;
		typedef Matrix<std::complex<double>, Dynamic, Dynamic, RowMajor> MatrixXz_r;

		Map<MatrixXs_c, 0, Stride<Dynamic,1> > As_c(               ( float*  )ap, m, k, stride_a );
		Map<MatrixXs_c, 0, Stride<Dynamic,1> > Bs_c(               ( float*  )bp, k, n, stride_b );
		Map<MatrixXs_c, 0, Stride<Dynamic,1> > Cs_c(               ( float*  )cp, m, n, stride_c );

		Map<MatrixXd_c, 0, Stride<Dynamic,1> > Ad_c(               ( double* )ap, m, k, stride_a );
		Map<MatrixXd_c, 0, Stride<Dynamic,1> > Bd_c(               ( double* )bp, k, n, stride_b );
		Map<MatrixXd_c, 0, Stride<Dynamic,1> > Cd_c(               ( double* )cp, m, n, stride_c );

		Map<MatrixXc_c, 0, Stride<Dynamic,1> > Ac_c( ( std::complex<float>*  )ap, m, k, stride_a );
		Map<MatrixXc_c, 0, Stride<Dynamic,1> > Bc_c( ( std::complex<float>*  )bp, k, n, stride_b );
		Map<MatrixXc_c, 0, Stride<Dynamic,1> > Cc_c( ( std::complex<float>*  )cp, m, n, stride_c );

		Map<MatrixXz_c, 0, Stride<Dynamic,1> > Az_c( ( std::complex<double>* )ap, m, k, stride_a );
		Map<MatrixXz_c, 0, Stride<Dynamic,1> > Bz_c( ( std::complex<double>* )bp, k, n, stride_b );
		Map<MatrixXz_c, 0, Stride<Dynamic,1> > Cz_c( ( std::complex<double>* )cp, m, n, stride_c );

		Map<MatrixXs_r, 0, Stride<Dynamic,1> > As_r(               ( float*  )ap, m, k, stride_a );
		Map<MatrixXs_r, 0, Stride<Dynamic,1> > Bs_r(               ( float*  )bp, k, n, stride_b );
		Map<MatrixXs_r, 0, Stride<Dynamic,1> > Cs_r(               ( float*  )cp, m, n, stride_c );

		Map<MatrixXd_r, 0, Stride<Dynamic,1> > Ad_r(               ( double* )ap, m, k, stride_a );
		Map<MatrixXd_r, 0, Stride<Dynamic,1> > Bd_r(               ( double* )bp, k, n, stride_b );
		Map<MatrixXd_r, 0, Stride<Dynamic,1> > Cd_r(               ( double* )cp, m, n, stride_c );

		Map<MatrixXc_r, 0, Stride<Dynamic,1> > Ac_r( ( std::complex<float>*  )ap, m, k, stride_a );
		Map<MatrixXc_r, 0, Stride<Dynamic,1> > Bc_r( ( std::complex<float>*  )bp, k, n, stride_b );
		Map<MatrixXc_r, 0, Stride<Dynamic,1> > Cc_r( ( std::complex<float>*  )cp, m, n, stride_c );

		Map<MatrixXz_r, 0, Stride<Dynamic,1> > Az_r( ( std::complex<double>* )ap, m, k, stride_a );
		Map<MatrixXz_r, 0, Stride<Dynamic,1> > Bz_r( ( std::complex<double>* )bp, k, n, stride_b );
		Map<MatrixXz_r, 0, Stride<Dynamic,1> > Cz_r( ( std::complex<double>* )cp, m, n, stride_c );
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

			//C.noalias() += alpha_r * A * B;

			if ( params.sc_str[0] == 'c' )
			{
				if      ( params.dt_str[0] == 's' ) Cs_c.noalias() += alpha_r * As_c * Bs_c;
				else if ( params.dt_str[0] == 'd' ) Cd_c.noalias() += alpha_r * Ad_c * Bd_c;
				else if ( params.dt_str[0] == 'c' ) Cc_c.noalias() += alpha_r * Ac_c * Bc_c;
				else if ( params.dt_str[0] == 'z' ) Cz_c.noalias() += alpha_r * Az_c * Bz_c;
			}
			else // if ( params.sc_str[0] == 'r' )
			{
				if      ( params.dt_str[0] == 's' ) Cs_r.noalias() += alpha_r * As_r * Bs_r;
				else if ( params.dt_str[0] == 'd' ) Cd_r.noalias() += alpha_r * Ad_r * Bd_r;
				else if ( params.dt_str[0] == 'c' ) Cc_r.noalias() += alpha_r * Ac_r * Bc_r;
				else if ( params.dt_str[0] == 'z' ) Cz_r.noalias() += alpha_r * Az_r * Bz_r;
			}

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

		printf( "data_%s_%cgemm_%s", THR_STR, dt_ch, IMPL_STR );
		printf( "( %4lu, 1:4 ) = [ %5lu %5lu %5lu %8.2f ];\n",
		        ( unsigned long )(p - p_begin)/p_inc + 1,
		        ( unsigned long )m,
		        ( unsigned long )k,
		        ( unsigned long )n, gflops );
		fflush( stdout );

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

void init_def_params( params_t* params )
{
	params->opname    = LOCAL_OPNAME_STR;
	params->impl      = IMPL_STR;

	params->pc_str    = LOCAL_PC_STR;
	params->dt_str    = GLOB_DEF_DT_STR;
	params->sc_str    = GLOB_DEF_SC_STR;

	params->im_str    = GLOB_DEF_IM_STR;

	params->ps_str    = GLOB_DEF_PS_STR;
	params->m_str     = GLOB_DEF_M_STR;
	params->n_str     = GLOB_DEF_N_STR;
	params->k_str     = GLOB_DEF_K_STR;

	params->nr_str    = GLOB_DEF_NR_STR;

	params->alpha_str = GLOB_DEF_ALPHA_STR;
	params->beta_str  = GLOB_DEF_BETA_STR;
}

