/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

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

//        res             n     x      incx      y        incy
//double  res =  ddotv_( int*, double*, int*, double*,   int* );
//#define PRINT

int main( int argc, char** argv )
{
	obj_t x, y;
	obj_t res;
	dim_t n;
	dim_t p;
	dim_t p_begin, p_end, p_inc;
	int   n_input;
	num_t dt_x, dt_y, dt_res;
	int   r, n_repeats;
	num_t dt;

	double dtime;
	double dtime_save;
	double gflops;

	bli_init();

	n_repeats = 10;

#ifndef PRINT
	p_begin = 40;
	p_end   = 4000;
	p_inc   = 40;

	n_input = -1;
#else
	p_begin = 16;
	p_end   = 16;
	p_inc   = 1;

	n_input = 16;
#endif

#if 1
	dt = BLIS_FLOAT;
	//dt = BLIS_DOUBLE;
#else
	//dt = BLIS_SCOMPLEX;
	dt = BLIS_DCOMPLEX;
#endif

	dt_x = dt_y  = dt_res = dt;

	// Begin with initializing the last entry to zero so that
	// matlab allocates space for the entire array once up-front.
	for ( p = p_begin; p + p_inc <= p_end; p += p_inc ) ;
#ifdef BLIS
	printf( "data_dotv_blis" );
#else
	printf( "data_dotv_%s", BLAS );
#endif
	printf( "( %2lu, 1:2 ) = [ %4lu %7.2f ];\n",
	        ( unsigned long )(p - p_begin)/p_inc + 1,
	        ( unsigned long )0, 0.0 );

	//for ( p = p_begin; p <= p_end; p += p_inc )
	for ( p = p_end; p_begin <= p; p -= p_inc )
	{

		if ( n_input < 0 ) n = p * ( dim_t )abs(n_input);
		else               n =     ( dim_t )    n_input;

		bli_obj_create( dt_x, n, 1, 0, 0, &x );
		bli_obj_create( dt_y, n, 1, 0, 0, &y );
		bli_obj_create( dt_res, 1, 1, 0, 0, &res );

		bli_randm( &x );
		bli_randm( &y );

		dtime_save = 1.0e9;

		for ( r = 0; r < n_repeats; ++r )
		{

			dtime = bli_clock();

#ifdef PRINT
			bli_printm( "x", &x, "%4.1f", "" );
			bli_printm( "y", &y, "%4.1f", "" );
#endif

#ifdef BLIS
			bli_dotv( &x,
			          &y,
			          &res);
#else
			if ( bli_is_float( dt ) )
			{
				f77_int nn     = bli_obj_length( &x );
				f77_int incx   = bli_obj_vector_inc( &x );
				f77_int incy   = bli_obj_vector_inc( &y );
				float*  xp     = bli_obj_buffer( &x );
				float*  yp     = bli_obj_buffer( &y );
				float*  resp   = bli_obj_buffer( &res );

				*resp = sdot_( &nn,
				               xp, &incx,
				               yp, &incy );

			}
			else if ( bli_is_double( dt ) )
			{

				f77_int  nn     = bli_obj_length( &x );
				f77_int  incx   = bli_obj_vector_inc( &x );
				f77_int  incy   = bli_obj_vector_inc( &y );
				double*  xp     = bli_obj_buffer( &x );
				double*  yp     = bli_obj_buffer( &y );
				double*  resp   = bli_obj_buffer( &res );

				*resp = ddot_( &nn,
				               xp, &incx,
				               yp, &incy );
			}
#endif

#ifdef PRINT
			bli_printm( "res after", &res, "%4.1f", "" );
			exit(1);
#endif

			dtime_save = bli_clock_min_diff( dtime_save, dtime );
		}

		gflops = ( 2.0 * n ) / ( dtime_save * 1.0e9 );

#ifdef BLIS
		printf( "data_dotv_blis" );
#else
		printf( "data_dotv_%s", BLAS );
#endif
		printf( "( %2lu, 1:2 ) = [ %4lu %7.2f ];\n",
		        ( unsigned long )(p - p_begin)/p_inc + 1,
		        ( unsigned long )n, gflops );

		bli_obj_free( &x );
		bli_obj_free( &y );
		bli_obj_free( &res );
	}

	bli_finalize();

	return 0;
}
