/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2019 - 2022, Advanced Micro Devices, Inc.

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

//#define BLIS_ACCURACY_TEST
#ifdef BLIS_ACCURACY_TEST

bool scompare_result( int n, float *x, int incx, float *y, int incy )
{
	for ( int i = 0; i < n; i++ )
	{
		if ( (*x) != (*y) )
		{
			printf( "%4f != %4f at location %d\n", *x, *y, i );
			return FALSE;
		}
		x += incx;
		y += incy;
	}
	return TRUE;
}

bool dcompare_result( int n, double *x, int incx, double *y, int incy )
{
	for ( int i = 0; i < n; i++ )
	{
		if ( (*x) != (*y) )
		{
			printf( "%4f != %4f at location %d\n", *x, *y, i );
			return FALSE;
		}
		x += incx;
		y += incy;
	}
	return TRUE;
}

#endif


int main( int argc, char** argv )
{
	obj_t x, y;
	dim_t n;
	dim_t p;
	dim_t p_begin, p_end, p_inc;
	int   n_input, sizeof_dt;
	int   r, n_repeats;
	num_t dt;

	double dtime;
	double dtime_save;
	double gbps;

	//bli_init();

	n_repeats = 100000;

#ifndef PRINT
	p_begin = 200;
	p_end = 100000;
	p_inc = 200;

	n_input = -1;
#else
	p_begin = 16;
	p_end = 16;
	p_inc = 1;

	n_input = 16;
#endif

#if 1
	 // dt = BLIS_FLOAT;
	dt = BLIS_DOUBLE;
#else
	//dt = BLIS_SCOMPLEX;
	dt = BLIS_DCOMPLEX;
#endif

	if      ( dt == BLIS_FLOAT  ) sizeof_dt = sizeof( float );
	else if ( dt == BLIS_DOUBLE ) sizeof_dt = sizeof( double );

	printf( "executable\t n\t GBs per sec\n" );

	for ( p = p_begin; p <= p_end; p += p_inc )
	{

		if ( n_input < 0 ) n = p * ( dim_t )abs( n_input );
		else               n =     ( dim_t )     n_input;

		bli_obj_create( dt, n, 1, 0, 0, &x );
		bli_obj_create( dt, n, 1, 0, 0, &y );

		bli_randm( &x );

		dtime_save = DBL_MAX;

		for ( r = 0; r < n_repeats; ++r )
		{

			dtime = bli_clock();

#ifdef BLIS
			bli_copyv( &x,
			           &y );
#else
			if ( bli_is_float( dt ) )
			{
				f77_int nn   = bli_obj_length( &x );
				f77_int incx = bli_obj_vector_inc( &x );
				float*  xp   = bli_obj_buffer( &x );
				f77_int incy = bli_obj_vector_inc( &y );
				float*  yp   = bli_obj_buffer( &y );

				scopy_( &nn,
				        xp, &incx,
				        yp, &incy );

			}
			else if ( bli_is_double( dt ) )
			{

				f77_int nn   = bli_obj_length( &x );
				f77_int incx = bli_obj_vector_inc( &x );
				double* xp   = bli_obj_buffer( &x );
				f77_int incy = bli_obj_vector_inc( &y );
				double* yp   = bli_obj_buffer( &y );

				dcopy_( &nn,
				        xp, &incx,
				        yp, &incy );
			}
#endif
			dtime_save = bli_clock_min_diff( dtime_save, dtime );

#ifdef BLIS_ACCURACY_TEST
			if ( dt == BLIS_FLOAT )
			{
				int     nn   = bli_obj_length( &x );
				int     incx = bli_obj_vector_inc( &x );
				float*  xp   = bli_obj_buffer( &x );
				int     incy = bli_obj_vector_inc( &y );
				float*  yp   = bli_obj_buffer( &y );
				if ( scompare_result( nn, xp, incx, yp, incy ) )
					printf( "Copy Successful\n" );
				else
					printf( "ALERT!!! Copy Failed\n" );
			}
			if ( dt == BLIS_DOUBLE )
			{
				int     nn   = bli_obj_length( &x );
				int     incx = bli_obj_vector_inc( &x );
				double* xp   = bli_obj_buffer( &x );
				int     incy = bli_obj_vector_inc( &y );
				double* yp   = bli_obj_buffer( &y );
				if ( dcompare_result( nn, xp, incx, yp, incy ) )
					printf( "Copy Successful\n" );
				else
					printf( "ALERT!!! Copy Failed\n" );
			}
#endif
		}

		// Size of the vectors are incrementd by 1000, to test wide range of inputs.
		if ( p >= 1000  ) p_inc = 1000;
		if ( p >= 10000 ) p_inc = 10000;
		gbps = ( n * sizeof_dt ) / ( dtime_save * 1.0e9 );

#ifdef BLIS
		printf( "data_copyv_blis\t" );
#else
		printf( "data_copyv_%s\t", BLAS );
#endif
		printf( "%4lu\t %7.2f\n",
		        ( unsigned long )n, gbps );

		bli_obj_free( &x );
		bli_obj_free( &y );
	}

	//bli_finalize();

	return 0;
}
