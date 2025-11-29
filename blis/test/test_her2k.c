/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.

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

#ifdef WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#include "blis.h"


//#define PRINT

int main( int argc, char** argv )
{
	obj_t a, b, c;
	obj_t c_save;
	obj_t alpha, beta;
	dim_t m, k;
	dim_t p;
	dim_t p_begin, p_end, p_inc;
	int   m_input, k_input;
	num_t dt;
	int   r, n_repeats;
	uplo_t uploc;
	trans_t transa;
	f77_char f77_uploc;
	f77_char f77_transa;

	double dtime;
	double dtime_save;
	double gflops;

	//bli_init();

	//bli_error_checking_level_set( BLIS_NO_ERROR_CHECKING );

	n_repeats = 3;

#ifndef PRINT
	p_begin = 200;
	p_end   = 2000;
	p_inc   = 200;

	m_input = -1;
	k_input = -1;
#else
	p_begin = 16;
	p_end   = 16;
	p_inc   = 1;

	m_input = 3;
	k_input = 1;
#endif

	// her2k supports complex and double complex 
	//dt = BLIS_SCOMPLEX;
	dt = BLIS_DCOMPLEX;


	uploc = BLIS_LOWER;
	//uploc = BLIS_UPPER;

	transa = BLIS_NO_TRANSPOSE;

	bli_param_map_blis_to_netlib_uplo( uploc, &f77_uploc );
	bli_param_map_blis_to_netlib_trans( transa, &f77_transa );

	// Begin with initializing the last entry to zero so that
	// matlab allocates space for the entire array once up-front.
	for ( p = p_begin; p + p_inc <= p_end; p += p_inc ) ;
#ifdef BLIS
	printf( "data_her2k_blis" );
#else
	printf( "data_her2k_%s", BLAS );
#endif
	printf( "( %2lu, 1:3 ) = [ %4lu %4lu %7.2f ];\n",
	        ( unsigned long )(p - p_begin)/p_inc + 1,
	        ( unsigned long )0,
	        ( unsigned long )0, 0.0 );

	//for ( p = p_begin; p <= p_end; p += p_inc )
	for ( p = p_end; p_begin <= p; p -= p_inc )
	{
		if ( m_input < 0 ) m = p * ( dim_t )abs(m_input);
		else               m =     ( dim_t )    m_input;
		if ( k_input < 0 ) k = p * ( dim_t )abs(k_input);
		else               k =     ( dim_t )    k_input;

		bli_obj_create( dt, 1, 1, 0, 0, &alpha );
		bli_obj_create( dt, 1, 1, 0, 0, &beta );

		if ( bli_does_trans( transa ) )
		{
			bli_obj_create( dt, k, m, 0, 0, &a );
			bli_obj_create( dt, k, m, 0, 0, &b );
		}
		else
		{
			bli_obj_create( dt, m, k, 0, 0, &a );
			bli_obj_create( dt, m, k, 0, 0, &b );
		}
		bli_obj_create( dt, m, m, 0, 0, &c );
		bli_obj_create( dt, m, m, 0, 0, &c_save );

		bli_randm( &a );
		bli_randm( &b );
		bli_randm( &c );

		bli_obj_set_struc( BLIS_HERMITIAN, &c );
		bli_obj_set_uplo( uploc, &c );

		bli_obj_set_conjtrans( transa, &a );
		bli_obj_set_conjtrans( transa, &b );


		bli_setsc(  (2.0/1.0), 0.0, &alpha );
		bli_setsc( -(1.0/1.0), 0.0, &beta );


		bli_copym( &c, &c_save );

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

#ifdef BLIS

			bli_her2k( &alpha,
			           &a,
			           &b,
			           &beta,
			           &c );

#else
		if ( bli_is_float( dt ) )
		{
			f77_int mm     = bli_obj_length( &c );
			f77_int kk     = bli_obj_width_after_trans( &a );
			f77_int lda    = bli_obj_col_stride( &a );
			f77_int ldb    = bli_obj_col_stride( &b );
			f77_int ldc    = bli_obj_col_stride( &c );
			float*  alphap = bli_obj_buffer( &alpha );
			float*  ap     = bli_obj_buffer( &a );
			float*  bp     = bli_obj_buffer( &b );
			float*  betap  = bli_obj_buffer( &beta );
			float*  cp     = bli_obj_buffer( &c );

			ssyr2k_( &f77_uploc,
			         &f77_transa,
			         &mm,
			         &kk,
			         alphap,
			         ap, &lda,
			         bp, &ldb,
			         betap,
			         cp, &ldc );
		}
		else if ( bli_is_double( dt ) )
		{
			f77_int mm     = bli_obj_length( &c );
			f77_int kk     = bli_obj_width_after_trans( &a );
			f77_int lda    = bli_obj_col_stride( &a );
			f77_int ldb    = bli_obj_col_stride( &b );
			f77_int ldc    = bli_obj_col_stride( &c );
			double* alphap = bli_obj_buffer( &alpha );
			double* ap     = bli_obj_buffer( &a );
			double* bp     = bli_obj_buffer( &b );
			double* betap  = bli_obj_buffer( &beta );
			double* cp     = bli_obj_buffer( &c );

			dsyr2k_( &f77_uploc,
			         &f77_transa,
			         &mm,
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
			f77_int   lda    = bli_obj_col_stride( &a );
			f77_int   ldb    = bli_obj_col_stride( &b );
			f77_int   ldc    = bli_obj_col_stride( &c );
			scomplex* alphap = bli_obj_buffer( &alpha );
			scomplex* ap     = bli_obj_buffer( &a );
			scomplex* bp     = bli_obj_buffer( &b );
			float*    betap  = bli_obj_buffer( &beta );
			scomplex* cp     = bli_obj_buffer( &c );

			cher2k_( &f77_uploc,
			         &f77_transa,
			         &mm,
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
			f77_int   lda    = bli_obj_col_stride( &a );
			f77_int   ldb    = bli_obj_col_stride( &b );
			f77_int   ldc    = bli_obj_col_stride( &c );
			dcomplex* alphap = bli_obj_buffer( &alpha );
			dcomplex* ap     = bli_obj_buffer( &a );
			dcomplex* bp     = bli_obj_buffer( &b );
			double*   betap  = bli_obj_buffer( &beta );
			dcomplex* cp     = bli_obj_buffer( &c );

			zher2k_( &f77_uploc,
			         &f77_transa,
			         &mm,
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

		gflops = ( 2.0 * m * k * m ) / ( dtime_save * 1.0e9 );

		if ( bli_is_complex( dt ) ) gflops *= 4.0;

#ifdef BLIS
		printf( "data_her2k_blis" );
#else
		printf( "data_her2k_%s", BLAS );
#endif
		printf( "( %2lu, 1:3 ) = [ %4lu %4lu %7.2f ];\n",
		        ( unsigned long )(p - p_begin)/p_inc + 1,
		        ( unsigned long )m,
		        ( unsigned long )k, gflops );


		bli_obj_free( &alpha );
		bli_obj_free( &beta );

		bli_obj_free( &a );
		bli_obj_free( &c );
		bli_obj_free( &c_save );
	}

	//bli_finalize();

	return 0;
}

