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

//           uplo   m     alpha    x        incx  y        incy  a        lda
//void dsyr2_( char*, int*, double*, double*, int*, double*, int*, double*, int* );

//#define PRINT

int main( int argc, char** argv )
{
	obj_t a, x, y;
	obj_t a_save;
	obj_t alpha;
	dim_t m;
	dim_t p;
	dim_t p_begin, p_end, p_inc;
	int   m_input;
	num_t dt_a, dt_x, dt_y;
	num_t dt_alpha;
	int   r, n_repeats;
	uplo_t uplo;

	double dtime;
	double dtime_save;
	double gflops;

	//bli_init();

	n_repeats = 3;

#ifndef PRINT
	p_begin = 40;
	p_end   = 2000;
	p_inc   = 40;

	m_input = -1;
#else
	p_begin = 16;
	p_end   = 16;
	p_inc   = 1;

	m_input = 6;
#endif

	// her2 supports complex and double complex
	dt_alpha = dt_x = dt_y = dt_a = BLIS_SCOMPLEX;

	uplo = BLIS_LOWER;

	// Begin with initializing the last entry to zero so that
	// matlab allocates space for the entire array once up-front.
	for ( p = p_begin; p + p_inc <= p_end; p += p_inc ) ;
#ifdef BLIS
	printf( "data_her2_blis" );
#else
	printf( "data_her2_%s", BLAS );
#endif
	printf( "( %2lu, 1:2 ) = [ %4lu %7.2f ];\n",
	        ( unsigned long )(p - p_begin)/p_inc + 1,
	        ( unsigned long )0, 0.0 );

	//for ( p = p_begin; p <= p_end; p += p_inc )
	for ( p = p_end; p_begin <= p; p -= p_inc )
	{

		if ( m_input < 0 ) m = p * ( dim_t )abs(m_input);
		else               m =     ( dim_t )    m_input;


		bli_obj_create( dt_alpha, 1, 1, 0, 0, &alpha );

		bli_obj_create( dt_x, m, 1, 0, 0, &x );
		bli_obj_create( dt_y, m, 1, 0, 0, &y );
		bli_obj_create( dt_a, m, m, 0, 0, &a );
		bli_obj_create( dt_a, m, m, 0, 0, &a_save );

		bli_randm( &x );
		bli_randm( &y );
		bli_randm( &a );

		bli_obj_set_struc( BLIS_HERMITIAN, &a );
		//bli_obj_set_struc( BLIS_SYMMETRIC, &a );
		bli_obj_set_uplo( uplo, &a );


		bli_setsc(  (2.0/1.0), 0.0, &alpha );


		bli_copym( &a, &a_save );

		dtime_save = DBL_MAX;

		for ( r = 0; r < n_repeats; ++r )
		{
			bli_copym( &a_save, &a );


			dtime = bli_clock();

#ifdef PRINT
			bli_printm( "x", &x, "%4.1f", "" );
			bli_printm( "y", &y, "%4.1f", "" );
			bli_printm( "a", &a, "%4.1f", "" );
#endif

#ifdef BLIS

			bli_her2( &alpha,
			          &x,
			          &y,
			          &a );

#else
			if ( bli_is_float( dt_a ) )
			{
				f77_char uplo   = 'L';
				f77_int  mm     = bli_obj_length( &a );
				f77_int  incx   = bli_obj_vector_inc( &x );
				f77_int  incy   = bli_obj_vector_inc( &y );
				f77_int  lda    = bli_obj_col_stride( &a );
				float*   alphap = bli_obj_buffer( &alpha );
				float*   xp     = bli_obj_buffer( &x );
				float*   yp     = bli_obj_buffer( &y );
				float*   ap     = bli_obj_buffer( &a );

				ssyr2_( &uplo,
				        &mm,
				        alphap,
				        xp, &incx,
				        yp, &incy,
				        ap, &lda );
			}
			else if ( bli_is_double( dt_a ) )
			{
				f77_char uplo   = 'L';
				f77_int  mm     = bli_obj_length( &a );
				f77_int  incx   = bli_obj_vector_inc( &x );
				f77_int  incy   = bli_obj_vector_inc( &y );
				f77_int  lda    = bli_obj_col_stride( &a );
				double*  alphap = bli_obj_buffer( &alpha );
				double*  xp     = bli_obj_buffer( &x );
				double*  yp     = bli_obj_buffer( &y );
				double*  ap     = bli_obj_buffer( &a );

				dsyr2_( &uplo,
				        &mm,
				        alphap,
				        xp, &incx,
				        yp, &incy,
				        ap, &lda );
			}
			else if ( bli_is_scomplex( dt_a ) )
			{
				f77_char  uplo   = 'L';
				f77_int   mm     = bli_obj_length( &a );
				f77_int   incx   = bli_obj_vector_inc( &x );
				f77_int   incy   = bli_obj_vector_inc( &y );
				f77_int   lda    = bli_obj_col_stride( &a );
				scomplex* alphap = bli_obj_buffer( &alpha );
				scomplex* xp     = bli_obj_buffer( &x );
				scomplex* yp     = bli_obj_buffer( &y );
				scomplex* ap     = bli_obj_buffer( &a );

				cher2_( &uplo,
				        &mm,
				        alphap,
				        xp, &incx,
				        yp, &incy,
				        ap, &lda );
			}
			else if ( bli_is_dcomplex( dt_a ) )
			{
				f77_char  uplo   = 'L';
				f77_int   mm     = bli_obj_length( &a );
				f77_int   incx   = bli_obj_vector_inc( &x );
				f77_int   incy   = bli_obj_vector_inc( &y );
				f77_int   lda    = bli_obj_col_stride( &a );
				dcomplex* alphap = bli_obj_buffer( &alpha );
				dcomplex* xp     = bli_obj_buffer( &x );
				dcomplex* yp     = bli_obj_buffer( &y );
				dcomplex* ap     = bli_obj_buffer( &a );

				zher2_( &uplo,
				        &mm,
				        alphap,
				        xp, &incx,
				        yp, &incy,
				        ap, &lda );
			}

#endif

#ifdef PRINT
			bli_printm( "a after", &a, "%4.1f", "" );
			exit(1);
#endif

			dtime_save = bli_clock_min_diff( dtime_save, dtime );
		}

		gflops = ( 2.0 * m * m ) / ( dtime_save * 1.0e9 );

#ifdef BLIS
		printf( "data_her2_blis" );
#else
		printf( "data_her2_%s", BLAS );
#endif
		printf( "( %2lu, 1:2 ) = [ %4lu %7.2f ];\n",
		        ( unsigned long )(p - p_begin)/p_inc + 1,
		        ( unsigned long )m, gflops );

		bli_obj_free( &alpha );

		bli_obj_free( &x );
		bli_obj_free( &y );
		bli_obj_free( &a );
		bli_obj_free( &a_save );
	}

	//bli_finalize();

	return 0;
}

