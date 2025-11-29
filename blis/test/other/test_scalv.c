/*

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
	- Neither the name of The University of Texas at Austin nor the names
	  of its contributors may be used to endorse or promote products
	  derived from this software without specific prior written permission.

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

int main(int argc, char** argv)
{
	obj_t a, alpha;
	dim_t n, p;
	dim_t p_begin, p_end, p_inc;
	int   n_input;
	num_t dt;
	int   r, n_repeats;

	double dtime;
	double dtime_save;
	double gflops;

	//bli_init();
	//bli_error_checking_level_set( BLIS_NO_ERROR_CHECKING );

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

	n_input = 4;
#endif

#if 1
    dt = BLIS_FLOAT;
    //dt = BLIS_DOUBLE;
#else
    //dt = BLIS_SCOMPLEX;
      dt = BLIS_DCOMPLEX;
#endif
#ifdef BLIS
	printf( "data_scalv_blis\t n\t gflops\n" );
#else
	printf( "data_scalv_%s\t n\t gflops\n", BLAS );
#endif

	for (p = p_begin; p <= p_end; p += p_inc)
	{
		if (n_input < 0) n = p * (dim_t)abs(n_input);
		else               n = (dim_t)n_input;


		bli_obj_create(dt, 1, 1, 0, 0, &alpha);
		bli_obj_create(dt, 1, n, 0, 0, &a);

		bli_randm(&a);
		bli_setsc((2.0), 0.0, &alpha);
		dtime_save = DBL_MAX;

		for (r = 0; r < n_repeats; ++r)
		{
			dtime = bli_clock();
#ifdef BLIS
		bli_scalm(&BLIS_TWO, &a);
#else
		if ( bli_is_float( dt ) )
		{
			f77_int nn     = bli_obj_length( &a );
			f77_int inca   = bli_obj_vector_inc( &a );
			float*  scalar = bli_obj_buffer( &alpha );
			float*  ap     = bli_obj_buffer( &a );

			sscal_( &nn, scalar,
				 ap, &inca );
		}
		else if ( bli_is_double( dt ) )
		{
			f77_int  nn     = bli_obj_length( &a );
			f77_int  inca   = bli_obj_vector_inc( &a );
			double*  scalar = bli_obj_buffer( &alpha );
			double*  ap     = bli_obj_buffer( &a );

			dscal_( &nn, scalar,
				 ap, &inca );
			}
#endif
			dtime_save = bli_clock_min_diff(dtime_save, dtime);
		}
// Size of the vectors are incrementd by 1000, to test wide range of inputs.
		if (p == 10000)
		    p_inc = 10000;

		if (p == 1000)
		    p_inc = 1000;

		gflops = n / (dtime_save * 1.0e9);
#ifdef BLIS
	printf( "data_scalv_blis\t" );
#else
	printf( "data_scalv_%s\t", BLAS );
#endif
		printf(" %4lu\t %7.2f \n",
			(unsigned long)n, gflops);

		bli_obj_free(&alpha);
		bli_obj_free(&a);
	}
	return 0;
}



