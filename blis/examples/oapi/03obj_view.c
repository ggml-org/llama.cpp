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
	obj_t a1, a2;
	obj_t v1, v2, v3, v4, v5;
	num_t dt;
	dim_t m, n;
	inc_t rs, cs;
	dim_t i, j;
	dim_t mv, nv;


	//
	// This file demonstrates creating and submatrix views into existing matrices.
	//


	//
	// Example 1: Create an object and then create a submatrix view.
	//

	printf( "\n#\n#  -- Example 1 --\n#\n\n" );

	// We'll use these parameters for the following examples.
	dt = BLIS_DOUBLE;
	m = 6; n = 7; rs = 1; cs = m;

	// Create an object a1 using bli_obj_create().
	bli_obj_create( dt, m, n, rs, cs, &a1 );

 	// Initialize a1 to contain known values.
	init_dobj_by_cols( &a1 );

	bli_printm( "matrix 'a1' (initial state)", &a1, "%5.1f", "" );

	// Acquire a 4x3 submatrix view into a1 at (i,j) offsets (1,2).
	i = 1; j = 2; mv = 4; nv = 3;
	bli_acquire_mpart( i, j, mv, nv, &a1, &v1 );

	bli_printm( "4x3 submatrix 'v1' at offsets (1,2)", &v1, "%5.1f", "" );

	// NOTE: Submatrix views should never be passed to bli_obj_free(). It
	// will not cause an immediate error, but it is bad practice. Instead,
	// you should only release the objects that were created directy via
	// bli_obj_create(). In the above example, that means only object a1
	// would be passed to bli_obj_free().


	//
	// Example 2: Modify the contents of a submatrix view.
	//

	printf( "\n#\n#  -- Example 2 --\n#\n\n" );

	// Modify the first three elements of the first column.
	bli_setijm( -3.0, 0.0, 0, 0, &v1 );
	bli_setijm( -4.0, 0.0, 1, 0, &v1 );
	bli_setijm( -5.0, 0.0, 2, 0, &v1 );

	// Modify the first three elements of the second column.
	bli_setijm( -6.0, 0.0, 0, 1, &v1 );
	bli_setijm( -7.0, 0.0, 1, 1, &v1 );
	bli_setijm( -8.0, 0.0, 2, 1, &v1 );

	// Print the matrix again so we can see the update elements.
	bli_printm( "submatrix view 'v1' (modified state)", &v1, "%5.1f", "" );
	bli_printm( "matrix 'a1' (indirectly modified due to changes to 'v1')", &a1, "%5.1f", "" );


	//
	// Example 3: Create a submatrix view that is "too big".
	//

	printf( "\n#\n#  -- Example 3 --\n#\n\n" );

	// bli_acquire_mpart() will safely truncate your requested submatrix
	// view dimensions (or even the offsets) if they extend beyond the
	// bounds of the parent object.

	bli_printm( "matrix 'a1' (current state)", &a1, "%5.1f", "" );

	// Acquire a 4x3 submatrix view into a1 at offsets (4,2). Notice how
	// the requested view contains four rows, but the view is created with
	// only two rows because the starting m offset of 4 leaves only two rows
	// left in the parent matrix.
	bli_acquire_mpart( 4, 2, 4, 3, &a1, &v2 );

	bli_printm( "4x3 submatrix 'v2' at offsets (4,2) -- two rows truncated for safety", &v2, "%5.1f", "" );


	//
	// Example 4: Create a bufferless object, attach an external buffer, and
	//            then create a submatrix view.
	//

	printf( "\n#\n#  -- Example 4 --\n#\n\n" );

	// Create a object with known elements using the same approach as the
	// previous example file.
	double* p1 = malloc( m * n * sizeof( double ) );
	init_dmatrix_by_cols( m, n, p1, rs, cs );
	bli_obj_create_with_attached_buffer( dt, m, n, p1, rs, cs, &a2 );

	bli_printm( "matrix 'a2' (initial state)", &a2, "%5.1f", "" );

	// Acquire a 3x4 submatrix view at offset (2,3).
	bli_acquire_mpart( 2, 3, 3, 4, &a2, &v3 );

	bli_printm( "3x4 submatrix view 'v3' at offsets (2,3)", &v3, "%5.1f", "" );


	//
	// Example 5: Use a submatrix view to set a region of a larger matrix to
	//            zero.
	//

	printf( "\n#\n#  -- Example 5 --\n#\n\n" );

	bli_printm( "3x4 submatrix view 'v3' at offsets (2,3)", &v3, "%5.1f", "" );

	bli_setm( &BLIS_ZERO, &v3 );

	bli_printm( "3x4 submatrix view 'v3' (zeroed out)", &v3, "%5.1f", "" );

	bli_printm( "matrix 'a2' (modified state)", &a2, "%5.1f", "" );


	//
	// Example 6: Obtain a submatrix view into a submatrix view.
	//

	printf( "\n#\n#  -- Example 6 --\n#\n\n" );

	bli_acquire_mpart( 1, 1, 5, 6, &a2, &v4 );

	bli_printm( "5x6 submatrix view 'v4' at offsets (1,1) of 'a2'", &v4, "%5.1f", "" );

	bli_acquire_mpart( 1, 0, 4, 5, &v4, &v5 );

	bli_printm( "4x5 submatrix view 'v5' at offsets (1,0) of 'v4'", &v5, "%5.1f", "" );


	// Free the memory arrays we allocated.
	free( p1 );

	// Free the objects we created.
	bli_obj_free( &a1 );

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

