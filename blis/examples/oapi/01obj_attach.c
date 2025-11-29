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

#include <stdio.h>
#include <stdlib.h>
#include "blis.h"

void init_dmatrix_by_rows( dim_t m, dim_t n, double* a, inc_t rs, inc_t cs );
void init_dmatrix_by_cols( dim_t m, dim_t n, double* a, inc_t rs, inc_t cs );

int main( int argc, char** argv )
{
	obj_t a1, a2;
	num_t dt;
	dim_t m, n;
	inc_t rs, cs;


	//
	// This file demonstrates interfacing external or existing buffers
	// with BLIS objects.
	//


	//
	// Example 1: Create a bufferless object and then attach an external
	//            buffer to it, specifying column storage.
	//

	printf( "\n#\n#  -- Example 1 --\n#\n\n" );

	// We'll use these parameters for the following examples.
	dt = BLIS_DOUBLE;
	m = 4; n = 5; rs = 1; cs = m;

	// First we allocate and initialize a matrix by columns.
	double* p1 = malloc( m * n * sizeof( double ) );
	init_dmatrix_by_cols( m, n, p1, rs, cs );

	// bli_obj_create() automatically allocates an array large enough to hold
	// of the elements. We can also create a "bufferless" object and then
	// "attach" our own buffer to that object. This is useful when interfacing
	// BLIS objects to an existing application that produces its own matrix
	// arrays/buffers.
	bli_obj_create_without_buffer( dt, m, n, &a1 );

	// Note that the fourth argument of bli_obj_attach_buffer() is the so-called
	// "imaginary stride". First of all, this stride only has meaning in the
	// complex domain. Secondly, it is a somewhat experimental property of the
	// obj_t, and one that is not fully recognized/utilized throughout BLIS.
	// Thus, the safe thing to do is to always pass in a 0, which is a request
	// for the default (which is actually 1). Please don't use any other value
	// unless you really know what you are doing.
	bli_obj_attach_buffer( p1, rs, cs, 0, &a1 );

	// Now let's print the matrix so we can see how the element values were
	// assigned.
	bli_printm( "matrix 'a1', initialized by columns:", &a1, "%5.1f", "" );


	//
	// Example 2: Create a bufferless object and then attach an external
	//            buffer to it, specifying row storage.
	//

	printf( "\n#\n#  -- Example 2 --\n#\n\n" );

	// Now let's allocate another buffer, but this time we'll initialize it by
	// rows instead of by columns. We'll use the same values for m, n, rs, cs.
	double* p2 = malloc( m * n * sizeof( double ) );
	init_dmatrix_by_rows( m, n, p2, rs, cs );

	// Create a new bufferless object and attach the new buffer. This time,
	// instead of calling bli_obj_create_without_buffer() followed by
	// bli_obj_attach_buffer(), we call bli_obj_create_with_attached_buffer(),
	// which is just a convenience wrapper around the former two functions.
	// (Note that the wrapper function omits the imaginary stride argument.)
#if 1
	bli_obj_create_with_attached_buffer( dt, m, n, p2, rs, cs, &a2 );
#else
	bli_obj_create_without_buffer( dt, m, n, &a2 );
	bli_obj_attach_buffer( p2, rs, cs, 0, &a2 );
#endif

	// Print the matrix so we can compare it to the first matrix output.
	bli_printm( "matrix 'a2', initialized by rows:", &a2, "%5.1f", "" );

	// Please note that after creating an object via either of:
	// - bli_obj_create_without_buffer(), or
	// - bli_obj_create_with_attached_buffer()
	// we do NOT free it! That's because these functions merely initialize the
	// object and do not actually allocate any memory.


	// Free the memory arrays we allocated.
	free( p1 );
	free( p2 );

	return 0;
}

// -----------------------------------------------------------------------------

void init_dmatrix_by_rows( dim_t m, dim_t n, double* a, inc_t rs, inc_t cs )
{
	dim_t  i, j;

	double alpha = 0.0;

	// Step through a matrix by rows, assigning each element a unique
	// value, starting at 0.
	for ( i = 0; i < m; ++i )
	{
		for ( j = 0; j < n; ++j )
		{
			double* a_ij = a + i*rs + j*cs;

			*a_ij = alpha;

			alpha += 1.0;
		}
	}
}

void init_dmatrix_by_cols( dim_t m, dim_t n, double* a, inc_t rs, inc_t cs )
{
	dim_t  i, j;

	double alpha = 0.0;

	// Step through a matrix by columns, assigning each element a unique
	// value, starting at 0.
	for ( j = 0; j < n; ++j )
	{
		for ( i = 0; i < m; ++i )
		{
			double* a_ij = a + i*rs + j*cs;

			*a_ij = alpha;

			alpha += 1.0;
		}
	}
}

