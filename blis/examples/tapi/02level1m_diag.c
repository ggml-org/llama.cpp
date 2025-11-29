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
	double* a;
	double* b;
	double* c;
	double* d;
	double* e;
	double* h;
	dim_t m, n;
	inc_t rs, cs;

	// Initialize some basic constants.
	double zero      =   0.0;
	double minus_one =  -1.0;


	//
	// This file demonstrates level-1m operations on structured matrices.
	//


	//
	// Example 1: Initialize the upper triangle of a matrix to random values.
	//

	printf( "\n#\n#  -- Example 1 --\n#\n\n" );

	// Create a matrix to work with.
	m = 5; n = 5; rs = 1; cs = m;
	a = malloc( m * n * sizeof( double ) );

	// Set the upper triangle to random values.
	bli_drandm( 0, BLIS_UPPER, m, n, a, rs, cs );

	bli_dprintm( "a: randomize upper part (lower part may contain garbage)",
	             m, n, a, rs, cs, "% 4.3f", "" );


	//
	// Example 2: Initialize the upper triangle of a matrix to random values
	//            but also explicitly set the strictly lower triangle to zero.
	//

	printf( "\n#\n#  -- Example 2 --\n#\n\n" );

	// Create a matrix to work with.
	m = 5; n = 5; rs = 1; cs = m;
	b = malloc( m * n * sizeof( double ) );

	// Set the upper triangle to random values.
	bli_drandm( 0, BLIS_UPPER, m, n, b, rs, cs );

	// Set the strictly lower triangle of 'b' to zero (by setting the lower
	// triangle of 'bl' to zero).
	bli_dsetm( BLIS_NO_CONJUGATE, -1, BLIS_NONUNIT_DIAG, BLIS_LOWER,
	           m, n, &zero, b, rs, cs );

	bli_dprintm( "b: randomize upper part; set strictly lower part to 0.0)",
	             m, n, b, rs, cs, "% 4.3f", "" );

	// You may not see the effect of setting the strictly lower part to zero,
	// since those values may already be zero (instead of random junk). So
	// let's set it to something you'll notice, like -1.0.
	bli_dsetm( BLIS_NO_CONJUGATE, -1, BLIS_NONUNIT_DIAG, BLIS_LOWER,
	           m, n, &minus_one, b, rs, cs );

	bli_dprintm( "b: randomize upper part; set strictly lower part to -1.0)",
	             m, n, b, rs, cs, "% 4.3f", "" );


	//
	// Example 3: Copy the lower triangle of an existing matrix to a newly
	//            created (but otherwise uninitialized) matrix.
	//

	printf( "\n#\n#  -- Example 3 --\n#\n\n" );

	// Create a matrix to work with.
	m = 5; n = 5; rs = 1; cs = m;
	c = malloc( m * n * sizeof( double ) );

	bli_dcopym( 0, BLIS_NONUNIT_DIAG, BLIS_LOWER, BLIS_NO_TRANSPOSE,
	            m, n, b, rs, cs, c, rs, cs );

	bli_dprintm( "c: copy lower part of b (upper part may contain garbage)",
	             m, n, c, rs, cs, "% 4.3f", "" );

	bli_dcopym( 0, BLIS_NONUNIT_DIAG, BLIS_LOWER, BLIS_NO_TRANSPOSE,
	            m, n, b, rs, cs, a, rs, cs );

	bli_dprintm( "a: copy lower triangle of b to upper triangular a",
	             m, n, a, rs, cs, "% 4.3f", "" );


	//
	// Example 4: Copy the lower triangle of an existing object into the
	//            upper triangle of an existing object.
	//

	printf( "\n#\n#  -- Example 4 --\n#\n\n" );

	// Create a matrix to work with.
	m = 5; n = 5; rs = 1; cs = m;
	d = malloc( m * n * sizeof( double ) );

	// Let's start by setting entire destination matrix to zero.
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
	           m, n, &zero, d, rs, cs );

	bli_dprintm( "d: initial value (all zeros)",
	             m, n, d, rs, cs, "% 4.3f", "" );

	// Let's change a few values of b manually so we can later see the full
	// effect of the transposition.
	bli_dsetijm( 2.0, 0.0, 2, 0, b, rs, cs );
	bli_dsetijm( 3.0, 0.0, 3, 0, b, rs, cs );
	bli_dsetijm( 4.0, 0.0, 4, 0, b, rs, cs );
	bli_dsetijm( 3.1, 0.0, 2, 1, b, rs, cs );
	bli_dsetijm( 3.2, 0.0, 3, 2, b, rs, cs );

	bli_dprintm( "b:",
	             m, n, b, rs, cs, "% 4.3f", "" );

	bli_dcopym( 0, BLIS_NONUNIT_DIAG, BLIS_LOWER, BLIS_TRANSPOSE,
	            m, n, b, rs, cs, d, rs, cs );

	bli_dprintm( "d: transpose of lower triangle of b copied to d",
	             m, n, d, rs, cs, "% 4.3f", "" );


	//
	// Example 5: Create a rectangular matrix (m > n) with a lower trapezoid
	//            containing random values, then set the strictly upper
	//            triangle to zeros.
	//

	printf( "\n#\n#  -- Example 5 --\n#\n\n" );

	// Create a matrix to work with.
	m = 6; n = 4; rs = 1; cs = m;
	e = malloc( m * n * sizeof( double ) );

	// Initialize the entire matrix to -1.0 to simulate junk values.
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
	           m, n, &minus_one, e, rs, cs );

	bli_dprintm( "e: initial value (all -1.0)",
	             m, n, e, rs, cs, "% 4.3f", "" );

	// Randomize the lower trapezoid.
	bli_drandm( 0, BLIS_LOWER, m, n, e, rs, cs );

	bli_dprintm( "e: after lower trapezoid randomized",
	             m, n, e, rs, cs, "% 4.3f", "" );

	// Set the upper triangle to zero.
	bli_dsetm( BLIS_NO_CONJUGATE, 1, BLIS_NONUNIT_DIAG, BLIS_UPPER,
	           m, n, &zero, e, rs, cs );

	bli_dprintm( "e: after upper triangle set to zero",
	             m, n, e, rs, cs, "% 4.3f", "" );


	//
	// Example 6: Create an upper Hessenberg matrix of random values and then
	//            set the "unstored" values to zero.
	//

	printf( "\n#\n#  -- Example 6 --\n#\n\n" );

	// Create a matrix to work with.
	m = 5; n = 5; rs = 1; cs = m;
	h = malloc( m * n * sizeof( double ) );

	// Initialize the entire matrix to -1.0 to simulate junk values.
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
	           m, n, &minus_one, h, rs, cs );

	bli_dprintm( "h: initial value (all -1.0)",
	             m, n, h, rs, cs, "% 4.3f", "" );

	// Randomize the elements on and above the first subdiagonal.
	bli_drandm( -1, BLIS_UPPER, m, n, h, rs, cs );

	bli_dprintm( "h: after randomizing above first subdiagonal",
	             m, n, h, rs, cs, "% 4.3f", "" );

	// Set the region strictly below the first subdiagonal (on or below
	// the second subdiagonal) to zero.
	bli_dsetm( BLIS_NO_CONJUGATE, -2, BLIS_NONUNIT_DIAG, BLIS_LOWER,
	           m, n, &zero, h, rs, cs );

	bli_dprintm( "h: after setting elements below first subdiagonal to zero",
	             m, n, h, rs, cs, "% 4.3f", "" );


	// Free the memory obtained via malloc().
	free( a );
	free( b );
	free( c );
	free( d );
	free( e );
	free( h );

	return 0;
}

// -----------------------------------------------------------------------------

