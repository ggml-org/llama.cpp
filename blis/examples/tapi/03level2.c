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
	double* x;
	double* y;
	double* b;
	double  alpha, beta;
	dim_t   m, n;
	inc_t   rs, cs;

	// Initialize some basic constants.
    double   zero        =   0.0;
    double   one         =   1.0;
    double   two         =   2.0;
    double   minus_one   =  -1.0;


	//
	// This file demonstrates level-2 operations.
	//


	//
	// Example 1: Perform a general rank-1 update (ger) operation.
	//

	printf( "\n#\n#  -- Example 1 --\n#\n\n" );

	// Create some matrix and vector operands to work with.
	m = 4; n = 5; rs = 1; cs = m;
	a = malloc( m * n * sizeof( double ) );
	x = malloc( m * 1 * sizeof( double ) );
	y = malloc( 1 * n * sizeof( double ) );

	// Let's initialize some scalars.
	alpha = 1.0;

	// Initialize vectors 'x' and 'y'.
	bli_drandv( m, x, 1 );
	bli_dsetv( BLIS_NO_CONJUGATE, n, &minus_one, y, 1 );

	// Initialize 'a' to 1.0.
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
	           m, n, &one, a, rs, cs );

	bli_dprintm( "x: set to random values", m, 1, x, 1, m, "% 4.3f", "" );
	bli_dprintm( "y: set to -1.0", 1, n, y, n, 1, "% 4.3f", "" );
	bli_dprintm( "a: intial value", m, n, a, rs, cs, "% 4.3f", "" );

	// a := a + alpha * x * y, where 'a' is general.
	bli_dger( BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE,
	          m, n, &alpha, x, 1, y, 1, a, rs, cs );

	bli_dprintm( "a: after ger", m, n, a, rs, cs, "% 4.3f", "" );

	// Free the memory obtained via malloc().
	free( a );
	free( x );
	free( y );


	//
	// Example 2: Perform a general matrix-vector multiply (gemv) operation.
	//

	printf( "\n#\n#  -- Example 2 --\n#\n\n" );

	// Create some matrix and vector operands to work with.
	m = 4; n = 5; rs = 1; cs = m;
	a = malloc( m * n * sizeof( double ) );
	x = malloc( 1 * n * sizeof( double ) );
	y = malloc( 1 * m * sizeof( double ) );

	// Set the scalars to use.
	alpha = 1.0;
	beta  = 1.0;

	// Initialize vectors 'x' and 'y'.
	bli_dsetv( BLIS_NO_CONJUGATE, n, &one,  x, 1 );
	bli_dsetv( BLIS_NO_CONJUGATE, m, &zero, y, 1 );

	// Randomize 'a'.
	bli_drandm( 0, BLIS_DENSE, m, n, a, rs, cs );

	bli_dprintm( "a: randomized", m, n, a, rs, cs, "% 4.3f", "" );
	bli_dprintm( "x: set to 1.0", 1, n, x, n, 1, "% 4.3f", "" );
	bli_dprintm( "y: intial value", 1, m, y, m, 1, "% 4.3f", "" );

	// y := beta * y + alpha * a * x, where 'a' is general.
	bli_dgemv( BLIS_NO_TRANSPOSE, BLIS_NO_CONJUGATE,
	           m, n, &alpha, a, rs, cs, x, 1, &beta, y, 1 );

	bli_dprintm( "y: after gemv", 1, m, y, m, 1, "% 4.3f", "" );

	// Free the memory obtained via malloc().
	free( a );
	free( x );
	free( y );


	//
	// Example 3: Perform a symmetric rank-1 update (syr) operation.
	//

	printf( "\n#\n#  -- Example 3 --\n#\n\n" );

	// Create some matrix and vector operands to work with.
	m = 5; rs = 1; cs = 5;
	a = malloc( m * m * sizeof( double ) );
	x = malloc( 1 * m * sizeof( double ) );

	// Set alpha.
	alpha = 1.0;

	// Initialize vector 'x'.
	bli_drandv( m, x, 1 );

	// Zero out all of matrix 'a'. This is optional, but will avoid possibly
	// displaying junk values in the unstored triangle.
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
	           m, m, &zero, a, rs, cs );

	// Randomize the lower triangle of 'a'.
	bli_drandm( 0, BLIS_LOWER, m, m, a, rs, cs );

	bli_dprintm( "x: set to random values", 1, m, x, m, 1, "% 4.3f", "" );
	bli_dprintm( "a: initial value (zeros in upper triangle)", m, m, a, 1, m, "% 4.3f", "" );

	// a := a + alpha * x * x^T, where 'a' is symmetric and lower-stored.
	bli_dsyr( BLIS_LOWER, BLIS_NO_CONJUGATE, m, &alpha, x, 1, a, rs, cs );

	bli_dprintm( "a: after syr", m, m, a, 1, m, "% 4.3f", "" );

	// Free the memory obtained via malloc().
	free( a );
	free( x );


	//
	// Example 4: Perform a symmetric matrix-vector multiply (symv) operation.
	//

	printf( "\n#\n#  -- Example 4 --\n#\n\n" );

	// Create some matrix and vector operands to work with.
	m = 5;; rs = 1; cs = m;
	a = malloc( m * m * sizeof( double ) );
	x = malloc( 1 * m * sizeof( double ) );
	y = malloc( 1 * m * sizeof( double ) );

	// Set the scalars to use.
	alpha = 1.0;
	beta  = 1.0;

	// Initialize vectors 'x' and 'y'.
	bli_dsetv( BLIS_NO_CONJUGATE, m, &one,  x, 1 );
	bli_dsetv( BLIS_NO_CONJUGATE, m, &zero, y, 1 );

	// Zero out all of matrix 'a'. This is optional, but will avoid possibly
	// displaying junk values in the unstored triangle.
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
	           m, m, &zero, a, rs, cs );

	// Randomize 'a'.
	bli_drandm( 0, BLIS_UPPER, m, m, a, rs, cs );

	bli_dprintm( "a: randomized (zeros in lower triangle)", m, m, a, rs, cs, "% 4.3f", "" );
	bli_dprintm( "x: set to 1.0", 1, m, x, m, 1, "% 4.3f", "" );
	bli_dprintm( "y: intial value", 1, m, y, m, 1, "% 4.3f", "" );

	// y := beta * y + alpha * a * x, where 'a' is symmetric and upper-stored.
	bli_dsymv( BLIS_UPPER, BLIS_NO_TRANSPOSE, BLIS_NO_CONJUGATE,
	           m, &alpha, a, rs, cs, x, 1, &beta, y, 1 );

	bli_dprintm( "y: after symv", 1, m, y, m, 1, "% 4.3f", "" );

	// Free the memory obtained via malloc().
	free( a );
	free( x );
	free( y );


	//
	// Example 5: Perform a triangular matrix-vector multiply (trmv) operation.
	//

	printf( "\n#\n#  -- Example 5 --\n#\n\n" );

	// Create some matrix and vector operands to work with.
	m = 5;; rs = 1; cs = m;
	a = malloc( m * m * sizeof( double ) );
	x = malloc( 1 * m * sizeof( double ) );

	// Set the scalars to use.
	alpha = 1.0;

	// Initialize vector 'x'.
	bli_dsetv( BLIS_NO_CONJUGATE, m, &one, x, 1 );

	// Zero out all of matrix 'a'. This is optional, but will avoid possibly
	// displaying junk values in the unstored triangle.
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
	           m, m, &zero, a, rs, cs );

	// Randomize 'a'.
	bli_drandm( 0, BLIS_LOWER, m, m, a, rs, cs );

	bli_dprintm( "a: randomized (zeros in upper triangle)", m, m, a, rs, cs, "% 4.3f", "" );
	bli_dprintm( "x: intial value", 1, m, x, m, 1, "% 4.3f", "" );

	// x := alpha * a * x, where 'a' is triangular and lower-stored.
	bli_dtrmv( BLIS_LOWER, BLIS_NO_TRANSPOSE, BLIS_NONUNIT_DIAG,
	           m, &alpha, a, rs, cs, x, 1 );

	bli_dprintm( "x: after trmv", 1, m, x, m, 1, "% 4.3f", "" );

	// Free the memory obtained via malloc().
	free( a );
	free( x );


	//
	// Example 6: Perform a triangular solve (trsv) operation.
	//

	printf( "\n#\n#  -- Example 6 --\n#\n\n" );

	// Create some matrix and vector operands to work with.
	m = 5;; rs = 1; cs = m;
	a = malloc( m * m * sizeof( double ) );
	b = malloc( 1 * m * sizeof( double ) );
	y = malloc( 1 * m * sizeof( double ) );

	// Set the scalars to use.
	alpha = 1.0;

	// Initialize vector 'x'.
	bli_dsetv( BLIS_NO_CONJUGATE, m, &one, b, 1 );

	// Zero out all of matrix 'a'. This is optional, but will avoid possibly
	// displaying junk values in the unstored triangle.
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
	           m, m, &zero, a, rs, cs );

	// Randomize 'a'.
	bli_drandm( 0, BLIS_LOWER, m, m, a, rs, cs );

	// Load the diagonal. By setting the diagonal to something of greater
	// absolute value than the off-diagonal elements, we increase the odds
	// that the matrix is not singular (singular matrices have no inverse).
	bli_dshiftd( 0, m, m, &two, a, rs, cs );

	bli_dprintm( "a: randomized (zeros in upper triangle)", m, m, a, rs, cs, "% 4.3f", "" );
	bli_dprintm( "b: intial value", 1, m, b, m, 1, "% 4.3f", "" );

	// x := alpha * a * x, where 'a' is triangular and lower-stored.
	bli_dtrsv( BLIS_LOWER, BLIS_NO_TRANSPOSE, BLIS_NONUNIT_DIAG,
	           m, &alpha, a, rs, cs, x, 1 );

	bli_dprintm( "b: after trsv", 1, m, b, m, 1, "% 4.3f", "" );

	// We can confirm the solution by comparing the product of a and x to the
	// original value of b.
	bli_dcopyv( BLIS_NO_TRANSPOSE, m, b, 1, y, 1 );
	bli_dtrmv( BLIS_LOWER, BLIS_NO_TRANSPOSE, BLIS_NONUNIT_DIAG,
	           m, &alpha, a, rs, cs, y, 1 );

	bli_dprintm( "y: should equal initial value of b", 1, m, y, m, 1, "% 4.3f", "" );

	// Free the memory obtained via malloc().
	free( a );
	free( b );
	free( y );


	return 0;
}

