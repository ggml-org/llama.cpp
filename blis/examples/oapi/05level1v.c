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
#include "blis.h"

int main( int argc, char** argv )
{
	obj_t alpha, beta, gamma;
	obj_t x, y, z, w, a;
	num_t dt;
	dim_t m, n;
	inc_t rs, cs;


	//
	// This file demonstrates working with vector objects and the level-1v
	// operations.
	//


	//
	// Example 1: Create vector objects and then broadcast (copy) scalar
	//            values to all elements.
	//

	printf( "\n#\n#  -- Example 1 --\n#\n\n" );

	// Create a few vectors to work with. We make them all of the same length
	// so that we can perform operations between them.
	// NOTE: We've chosen to use row vectors here (1x4) instead of column
	// vectors (4x1) to allow for easier reading of standard output (less
	// scrolling).
	dt = BLIS_DOUBLE;
	m = 1; n = 4; rs = 0; cs = 0;
	bli_obj_create( dt, m, n, rs, cs, &x );
	bli_obj_create( dt, m, n, rs, cs, &y );
	bli_obj_create( dt, m, n, rs, cs, &z );
	bli_obj_create( dt, m, n, rs, cs, &w );
	bli_obj_create( dt, m, n, rs, cs, &a );

	// Let's also create and initialize some scalar objects.
	bli_obj_create_1x1( dt, &alpha );
	bli_obj_create_1x1( dt, &beta );
	bli_obj_create_1x1( dt, &gamma );

	bli_setsc( 2.0, 0.0, &alpha );
	bli_setsc( 0.2, 0.0, &beta );
	bli_setsc( 3.0, 0.0, &gamma );

	bli_printm( "alpha:", &alpha, "% 4.3f", "" );
	bli_printm( "beta:", &beta, "% 4.3f", "" );
	bli_printm( "gamma:", &gamma, "% 4.3f", "" );

	// Vectors can set by "broadcasting" a constant to every element.
	bli_setv( &BLIS_ONE, &x );
	bli_setv( &alpha, &y );
	bli_setv( &BLIS_ZERO, &z );

	// Note that we can use printv or printm to print vectors since vectors
	// are also matrices. We choose to use printm because it honors the
	// orientation of the vector (row or column) when printing, whereas
	// printv always prints vectors as column vectors regardless of their
	// they are 1 x n or n x 1.
	bli_printm( "x := 1.0", &x, "% 4.3f", "" );
	bli_printm( "y := alpha", &y, "% 4.3f", "" );
	bli_printm( "z := 0.0", &z, "% 4.3f", "" );


	//
	// Example 2: Randomize a vector object.
	//

	printf( "\n#\n#  -- Example 2 --\n#\n\n" );

	// Set a vector to random values.
	bli_randv( &w );

	bli_printm( "w := randv()", &w, "% 4.3f", "" );


	//
	// Example 3: Perform various element-wise operations on vector objects.
	//

	printf( "\n#\n#  -- Example 3 --\n#\n\n" );

	// Copy a vector.
	bli_copyv( &w, &a );
	bli_printm( "a := w", &a, "% 4.3f", "" );

	// Add and subtract vectors.
	bli_addv( &y, &a );
	bli_printm( "a := a + y", &a, "% 4.3f", "" );

	bli_subv( &w, &a );
	bli_printm( "a := a - w", &a, "% 4.3f", "" );

	// Scale a vector (destructive).
	bli_scalv( &beta, &a );
	bli_printm( "a := beta * a", &a, "% 4.3f", "" );

	// Scale a vector (non-destructive).
	bli_scal2v( &gamma, &a, &z );
	bli_printm( "z := gamma * a", &z, "% 4.3f", "" );

	// Scale and accumulate between vectors.
	bli_axpyv( &alpha, &w, &x );
	bli_printm( "x := x + alpha * w", &x, "% 4.3f", "" );

	bli_xpbyv( &w, &BLIS_MINUS_ONE, &x );
	bli_printm( "x := -1.0 * x + w", &x, "% 4.3f", "" );

	// Invert a vector element-wise.
	bli_invertv( &y );
	bli_printm( "y := 1 / y", &y, "% 4.3f", "" );

	// Swap two vectors.
	bli_swapv( &x, &y );
	bli_printm( "x (after swapping with y)", &x, "% 4.3f", "" );
	bli_printm( "y (after swapping with x)", &y, "% 4.3f", "" );


	//
	// Example 4: Perform contraction-like operations on vector objects.
	//

	printf( "\n#\n#  -- Example 4 --\n#\n\n" );

	// Perform a dot product.
	bli_dotv( &a, &z, &gamma );
	bli_printm( "gamma := a * z (dot product)", &gamma, "%5.2f", "" );

	// Perform an extended dot product.
	bli_dotxv( &alpha, &a, &z, &BLIS_ONE, &gamma );
	bli_printm( "gamma := 1.0 * gamma + alpha * a * z (accumulate scaled dot product)", &gamma, "%5.2f", "" );


	// Free the objects.
	bli_obj_free( &alpha );
	bli_obj_free( &beta );
	bli_obj_free( &gamma );
	bli_obj_free( &x );
	bli_obj_free( &y );
	bli_obj_free( &z );
	bli_obj_free( &w );
	bli_obj_free( &a );

	return 0;
}

// -----------------------------------------------------------------------------

