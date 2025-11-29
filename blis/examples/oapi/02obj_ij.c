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
void init_dobj_by_cols( obj_t* a );
void init_zobj_by_cols( obj_t* a );

int main( int argc, char** argv )
{
	obj_t a1, a2, a3;
	num_t dt;
	dim_t m, n;
	inc_t rs, cs;
	dim_t i, j;


	//
	// This file demonstrates accessing and updating individual matrix elements
	// through the BLIS object API.
	//


	//
	// Example 1: Create an object and then individually access/view some of
	//            its elements.
	//

	printf( "\n#\n#  -- Example 1 --\n#\n\n" );

	// We'll use these parameters for the following examples.
	dt = BLIS_DOUBLE;
	m = 4; n = 5; rs = 1; cs = m;

	// Create a object with known elements using the same approach as the
	// previous example file.
	double* p1 = malloc( m * n * sizeof( double ) );
	init_dmatrix_by_cols( m, n, p1, rs, cs );
	bli_obj_create_with_attached_buffer( dt, m, n, p1, rs, cs, &a1 );

	bli_printm( "matrix 'a1' (initial state)", &a1, "%5.1f", "" );

	// Regardless of how we create our object--whether via bli_obj_create() or
	// via attaching an existing buffer to a bufferless object--we can access
	// individual elements by specifying their offsets. The output value is
	// broken up by real and imaginary component. (When accessing real matrices,
	// the imaginary component will always be zero.)
	i = 1; j = 3;
	double alpha_r, alpha_i;
	bli_getijm( i, j, &a1, &alpha_r, &alpha_i );

	// Here, we print out the element "returned" by bli_getijm().
	printf( "element (%2d,%2d) of matrix 'a1' (real + imag): %5.1f + %5.1f\n", ( int )i, ( int )j, alpha_r, alpha_i );

	// Let's query a few more elements.
	i = 0; j = 2;
	bli_getijm( i, j, &a1, &alpha_r, &alpha_i );

	printf( "element (%2d,%2d) of matrix 'a1' (real + imag): %5.1f + %5.1f\n", ( int )i, ( int )j, alpha_r, alpha_i );

	i = 3; j = 4;
	bli_getijm( i, j, &a1, &alpha_r, &alpha_i );

	printf( "element (%2d,%2d) of matrix 'a1' (real + imag): %5.1f + %5.1f\n", ( int )i, ( int )j, alpha_r, alpha_i );

	printf( "\n" );


	//
	// Example 2: Modify individual elements of an existing matrix.
	//

	printf( "\n#\n#  -- Example 2 --\n#\n\n" );

	// Now let's change a few elements. Even if we set the imaginary
	// argument to a non-zero value, argument is ignored since we're
	// modifying a real matrix. If a1 were a complex object, those
	// values would be stored verbatim into the appropriate matrix
	// elements (see example for a3 below).
	alpha_r = -3.0; alpha_i =  0.0; i = 1; j = 3;
	bli_setijm( alpha_r, alpha_i, i, j, &a1 );

	alpha_r = -9.0; alpha_i = -1.0; i = 0; j = 2;
	bli_setijm( alpha_r, alpha_i, i, j, &a1 );

	alpha_r = -7.0; alpha_i =  2.0; i = 3; j = 4;
	bli_setijm( alpha_r, alpha_i, i, j, &a1 );

	// Print the matrix again so we can see the update elements.
	bli_printm( "matrix 'a1' (modified state)", &a1, "%5.1f", "" );

	// Next, let's create a regular object (with a buffer) and then
	// initialize its elements using bli_setijm().
	bli_obj_create( dt, m, n, rs, cs, &a2 );

	// See definition of init_dobj_by_cols() below.
	init_dobj_by_cols( &a2 );

	// Because we initialized a2 in the same manner as a1 (by columns),
	// it should contain the same initial state as a1.
	bli_printm( "matrix 'a2'", &a2, "%5.1f", "" );


	//
	// Example 3: Modify individual elements of an existing complex matrix.
	//

	printf( "\n#\n#  -- Example 3 --\n#\n\n" );

	// Create and initialize a complex object.
	dt = BLIS_DCOMPLEX;
	bli_obj_create( dt, m, n, rs, cs, &a3 );

	// Initialize the matrix elements. (See definition of init_dobj_by_cols()
	// below).
	init_zobj_by_cols( &a3 );

	// Print the complex matrix.
	bli_printm( "matrix 'a3' (initial state)", &a3, "%5.1f", "" );

	i = 3; j = 0;
	bli_getijm( i, j, &a3, &alpha_r, &alpha_i );
	alpha_r *= -1.0; alpha_i *= -1.0;
	bli_setijm( alpha_r, alpha_i, i, j, &a3 );

	i = 3; j = 4;
	bli_getijm( i, j, &a3, &alpha_r, &alpha_i );
	alpha_r *= -1.0; alpha_i *= -1.0;
	bli_setijm( alpha_r, alpha_i, i, j, &a3 );

	i = 0; j = 4;
	bli_getijm( i, j, &a3, &alpha_r, &alpha_i );
	alpha_r *= -1.0; alpha_i *= -1.0;
	bli_setijm( alpha_r, alpha_i, i, j, &a3 );

	// Print the matrix again so we can see the update elements.
	bli_printm( "matrix 'a3' (modified state)", &a3, "%5.1f", "" );

	// Free the memory arrays we allocated.
	free( p1 );


	// Free the objects we created.
	bli_obj_free( &a2 );
	bli_obj_free( &a3 );

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

void init_dobj_by_cols( obj_t* a )
{
	dim_t m = bli_obj_length( a );
	dim_t n = bli_obj_width( a );
	dim_t i, j;

	double alpha = 0.0;

	// Step through a matrix by columns, assigning each element a unique
	// value, starting at 0.
	for ( j = 0; j < n; ++j )
	{
		for ( i = 0; i < m; ++i )
		{
			bli_setijm( alpha, 0.0, i, j, a );

			alpha += 1.0;
		}
	}
}

void init_zobj_by_cols( obj_t* a )
{
	dim_t m = bli_obj_length( a );
	dim_t n = bli_obj_width( a );
	dim_t i, j;

	double alpha = 0.0;

	// Step through a matrix by columns, assigning each real and imaginary
	// element a unique value, starting at 0.
	for ( j = 0; j < n; ++j )
	{
		for ( i = 0; i < m; ++i )
		{
			bli_setijm( alpha, alpha + 1.0, i, j, a );

			alpha += 2.0;
		}
	}
}

