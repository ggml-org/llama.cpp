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
	double* x;
	double* y;
	double* z;
	double* w;
	double* a;
	double  alpha, beta, gamma;
	dim_t m, n;
	inc_t rs, cs;

	// Initialize some basic constants.
	double zero      = 0.0;
	double one       = 1.0;
	double minus_one = -1.0;


	//
	// This file demonstrates working with vectors and the level-1v
	// operations.
	//


	//
	// Example 1: Create vectors and then broadcast (copy) scalar
	//            values to all elements.
	//

	printf( "\n#\n#  -- Example 1 --\n#\n\n" );

	// Create a few vectors to work with. We make them all of the same length
	// so that we can perform operations between them.
	// NOTE: We've chosen to use row vectors here (1x4) instead of column
	// vectors (4x1) to allow for easier reading of standard output (less
	// scrolling).
	m = 1; n = 4; rs = n; cs = 1;
	x = malloc( m * n * sizeof( double ) );
	y = malloc( m * n * sizeof( double ) );
	z = malloc( m * n * sizeof( double ) );
	w = malloc( m * n * sizeof( double ) );
	a = malloc( m * n * sizeof( double ) );
	
	// Let's initialize some scalars.
	alpha = 2.0;
	beta  = 0.2;
	gamma = 3.0;

	printf( "alpha:\n% 4.3f\n\n", alpha );
	printf( "beta:\n% 4.3f\n\n", beta );
	printf( "gamma:\n% 4.3f\n\n", gamma );
	printf( "\n" );

	bli_dsetv( BLIS_NO_CONJUGATE, n, &one, x, 1 );
	bli_dsetv( BLIS_NO_CONJUGATE, n, &alpha, y, 1 );
	bli_dsetv( BLIS_NO_CONJUGATE, n, &zero, z, 1 );

	// Note that we can use printv or printm to print vectors since vectors
	// are also matrices. We choose to use printm because it honors the
	// orientation of the vector (row or column) when printing, whereas
	// printv always prints vectors as column vectors regardless of their
	// they are 1 x n or n x 1.
	bli_dprintm( "x := 1.0", m, n, x, rs, cs, "% 4.3f", "" );
	bli_dprintm( "y := alpha", m, n, y, rs, cs, "% 4.3f", "" );
	bli_dprintm( "z := 0.0", m, n, z, rs, cs, "% 4.3f", "" );


	//
	// Example 2: Randomize a vector.
	//

	printf( "\n#\n#  -- Example 2 --\n#\n\n" );

	// Set a vector to random values.
	bli_drandv( n, w, 1 );

	bli_dprintm( "x := randv()", m, n, w, rs, cs, "% 4.3f", "" );


	//
	// Example 3: Perform various element-wise operations on vectors.
	//

	printf( "\n#\n#  -- Example 3 --\n#\n\n" );

	// Copy a vector.
	bli_dcopyv( BLIS_NO_CONJUGATE, n, w, 1, a, 1 );
	bli_dprintm( "a := w", m, n, a, rs, cs, "% 4.3f", "" );

	// Add and subtract vectors.
	bli_daddv( BLIS_NO_CONJUGATE, n, y, 1, a, 1 );
	bli_dprintm( "a := a + y", m, n, a, rs, cs, "% 4.3f", "" );

	bli_dsubv( BLIS_NO_CONJUGATE, n, w, 1, a, 1 );
	bli_dprintm( "a := a + w", m, n, a, rs, cs, "% 4.3f", "" );

	// Scale a vector (destructive).
	bli_dscalv( BLIS_NO_CONJUGATE, n, &beta, a, 1 );
	bli_dprintm( "a := beta * a", m, n, a, rs, cs, "% 4.3f", "" );

	// Scale a vector (non-destructive).
	bli_dscal2v( BLIS_NO_CONJUGATE, n, &gamma, a, 1, z, 1 );
	bli_dprintm( "z := gamma * a", m, n, z, rs, cs, "% 4.3f", "" );

	// Scale and accumulate between vectors.
	bli_daxpyv( BLIS_NO_CONJUGATE, n, &alpha, w, 1, x, 1 );
	bli_dprintm( "x := x + alpha * w", m, n, x, rs, cs, "% 4.3f", "" );

	bli_dxpbyv( BLIS_NO_CONJUGATE, n, w, 1, &minus_one, x, 1 );
	bli_dprintm( "x := -1.0 * x + w", m, n, x, rs, cs, "% 4.3f", "" );

	// Invert a vector element-wise.
	bli_dinvertv( n, y, 1 );
	bli_dprintm( "y := 1 / y", m, n, y, rs, cs, "% 4.3f", "" );

	// Swap two vectors.
	bli_dswapv( n, x, 1, y, 1 );
	bli_dprintm( "x (after swapping with y)", m, n, x, rs, cs, "% 4.3f", "" );
	bli_dprintm( "y (after swapping with x)", m, n, y, rs, cs, "% 4.3f", "" );


	//
	// Example 4: Perform contraction-like operations on vectors.
	//

	printf( "\n#\n#  -- Example 4 --\n#\n\n" );

	// Perform a dot product.
	bli_ddotv( BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, n, a, 1, z, 1, &gamma );
	printf( "gamma := a * z (dot product):\n%5.2f\n\n", gamma );

	// Perform an extended dot product.
	bli_ddotxv( BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, n, &alpha, a, 1, z, 1, &one, &gamma );
	printf( "gamma := 1.0 * gamma + alpha * a * z (accumulate scaled dot product):\n%5.2f\n\n", gamma );


	// Free the memory obtained via malloc().
	free( x );
	free( y );
	free( z );
	free( w );
	free( a );

	return 0;
}

// -----------------------------------------------------------------------------

