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
	dim_t m, n;
	dim_t p;
	dim_t p_begin, p_end, p_inc;
	int   m_input, n_input;
	num_t dt;
	int   r, n_repeats;
	side_t side;
	uplo_t uploa;
	f77_char f77_side;
	f77_char f77_uploa;

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
	n_input = -1;
#else
	p_begin = 16;
	p_end   = 16;
	p_inc   = 1;

	m_input = 4;
	n_input = 4;
#endif

#if 1
	//dt = BLIS_FLOAT;
	dt = BLIS_DOUBLE;
#else
	//dt = BLIS_SCOMPLEX;
	dt = BLIS_DCOMPLEX;
#endif

	side = BLIS_LEFT;
	//side = BLIS_RIGHT;

	uploa = BLIS_LOWER;
	//uploa = BLIS_UPPER;

	bli_param_map_blis_to_netlib_side( side, &f77_side );
	bli_param_map_blis_to_netlib_uplo( uploa, &f77_uploa );

	// Begin with initializing the last entry to zero so that
	// matlab allocates space for the entire array once up-front.
	for ( p = p_begin; p + p_inc <= p_end; p += p_inc ) ;
#ifdef BLIS
	printf( "data_hemm_blis" );
#else
	printf( "data_hemm_%s", BLAS );
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
		if ( n_input < 0 ) n = p * ( dim_t )abs(n_input);
		else               n =     ( dim_t )    n_input;

		bli_obj_create( dt, 1, 1, 0, 0, &alpha );
		bli_obj_create( dt, 1, 1, 0, 0, &beta );

		if ( bli_is_left( side ) )
			bli_obj_create( dt, m, m, 0, 0, &a );
		else
			bli_obj_create( dt, n, n, 0, 0, &a );
		bli_obj_create( dt, m, n, 0, 0, &b );
		bli_obj_create( dt, m, n, 0, 0, &c );
		bli_obj_create( dt, m, n, 0, 0, &c_save );

		bli_randm( &a );
		bli_randm( &b );
		bli_randm( &c );

		bli_obj_set_struc( BLIS_HERMITIAN, &a );
		bli_obj_set_uplo( uploa, &a );

		// Randomize A, make it densely Hermitian, and zero the unstored
		// triangle to ensure the implementation reads only from the stored
		// region.
		bli_randm( &a );
		bli_mkherm( &a );
		bli_mktrim( &a );
/*
		bli_obj_toggle_uplo( &a );
		bli_obj_inc_diag_offset( 1, &a );
		bli_setm( &BLIS_ZERO, &a );
		bli_obj_inc_diag_offset( -1, &a );
		bli_obj_toggle_uplo( &a );
		bli_obj_set_diag( BLIS_NONUNIT_DIAG, &a );
		bli_scalm( &BLIS_TWO, &a );
		bli_scalm( &BLIS_TWO, &a );
*/

		bli_setsc(  (2.0/1.0), 1.0, &alpha );
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

			bli_hemm( side,
			          &alpha,
			          &a,
			          &b,
			          &beta,
			          &c );
#else

		if ( bli_is_float( dt ) )
		{
			f77_int  mm     = bli_obj_length( &c );
			f77_int  nn     = bli_obj_width( &c );
			f77_int  lda    = bli_obj_col_stride( &a );
			f77_int  ldb    = bli_obj_col_stride( &b );
			f77_int  ldc    = bli_obj_col_stride( &c );
			float*   alphap = bli_obj_buffer( &alpha );
			float*   ap     = bli_obj_buffer( &a );
			float*   bp     = bli_obj_buffer( &b );
			float*   betap  = bli_obj_buffer( &beta );
			float*   cp     = bli_obj_buffer( &c );

			ssymm_( &f77_side,
			        &f77_uploa,
			        &mm,
			        &nn,
			        alphap,
			        ap, &lda,
			        bp, &ldb,
			        betap,
			        cp, &ldc );
		}
		else if ( bli_is_double( dt ) )
		{
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

			dsymm_( &f77_side,
			        &f77_uploa,
			        &mm,
			        &nn,
			        alphap,
			        ap, &lda,
			        bp, &ldb,
			        betap,
			        cp, &ldc );
		}
		else if ( bli_is_scomplex( dt ) )
		{
			f77_int  mm     = bli_obj_length( &c );
			f77_int  nn     = bli_obj_width( &c );
			f77_int  lda    = bli_obj_col_stride( &a );
			f77_int  ldb    = bli_obj_col_stride( &b );
			f77_int  ldc    = bli_obj_col_stride( &c );
			scomplex*  alphap = bli_obj_buffer( &alpha );
			scomplex*  ap     = bli_obj_buffer( &a );
			scomplex*  bp     = bli_obj_buffer( &b );
			scomplex*  betap  = bli_obj_buffer( &beta );
			scomplex*  cp     = bli_obj_buffer( &c );

			chemm_( &f77_side,
			        &f77_uploa,
			        &mm,
			        &nn,
			        alphap,
			        ap, &lda,
			        bp, &ldb,
			        betap,
			        cp, &ldc );
		}
		else if ( bli_is_dcomplex( dt ) )
		{
			f77_int  mm     = bli_obj_length( &c );
			f77_int  nn     = bli_obj_width( &c );
			f77_int  lda    = bli_obj_col_stride( &a );
			f77_int  ldb    = bli_obj_col_stride( &b );
			f77_int  ldc    = bli_obj_col_stride( &c );
			dcomplex*  alphap = bli_obj_buffer( &alpha );
			dcomplex*  ap     = bli_obj_buffer( &a );
			dcomplex*  bp     = bli_obj_buffer( &b );
			dcomplex*  betap  = bli_obj_buffer( &beta );
			dcomplex*  cp     = bli_obj_buffer( &c );

			zhemm_( &f77_side,
			        &f77_uploa,
			        &mm,
			        &nn,
			        alphap,
			        ap, &lda,
			        bp, &ldb,
			        betap,
			        cp, &ldc );
		}
#endif

#ifdef PRINT
			bli_printm( "c after", &c, "%9.5f", "" );
			exit(1);
#endif

			dtime_save = bli_clock_min_diff( dtime_save, dtime );
		}

		if ( bli_is_left( side ) )
			gflops = ( 2.0 * m * m * n ) / ( dtime_save * 1.0e9 );
		else
			gflops = ( 2.0 * m * n * n ) / ( dtime_save * 1.0e9 );

		if ( bli_is_complex( dt ) ) gflops *= 4.0;

#ifdef BLIS
		printf( "data_hemm_blis" );
#else
		printf( "data_hemm_%s", BLAS );
#endif
		printf( "( %2lu, 1:3 ) = [ %4lu %4lu %7.2f ];\n",
		        ( unsigned long )(p - p_begin)/p_inc + 1,
		        ( unsigned long )m,
		        ( unsigned long )n, gflops );

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

