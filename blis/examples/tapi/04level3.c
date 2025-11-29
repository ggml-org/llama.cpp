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
	dim_t m, n, k;
	inc_t rsa, csa;
	inc_t rsb, csb;
	inc_t rsc, csc;

	double* a;
	double* b;
	double* c;
	double  alpha, beta;

	// Initialize some basic constants.
	double zero = 0.0;
	double one  = 1.0;
	double two  = 2.0;


	//
	// This file demonstrates level-3 operations.
	//


	//
	// Example 1: Perform a general matrix-matrix multiply (gemm) operation.
	//

	printf( "\n#\n#  -- Example 1 --\n#\n\n" );

	// Create some matrix and vector operands to work with.
	m = 4; n = 5; k = 3;
	rsc = 1; csc = m;
	rsa = 1; csa = m;
	rsb = 1; csb = k;
	c = malloc( m * n * sizeof( double ) );
	a = malloc( m * k * sizeof( double ) );
	b = malloc( k * n * sizeof( double ) );

	// Set the scalars to use.
	alpha = 1.0;
	beta  = 1.0;

	// Initialize the matrix operands.
	bli_drandm( 0, BLIS_DENSE, m, k, a, rsa, csa );
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
               k, n, &one, b, rsb, csb );
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
               m, n, &zero, c, rsc, csc );

	bli_dprintm( "a: randomized", m, k, a, rsa, csa, "% 4.3f", "" );
	bli_dprintm( "b: set to 1.0", k, n, b, rsb, csb, "% 4.3f", "" );
	bli_dprintm( "c: initial value", m, n, c, rsc, csc, "% 4.3f", "" );

	// c := beta * c + alpha * a * b, where 'a', 'b', and 'c' are general.
	bli_dgemm( BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,
	           m, n, k, &alpha, a, rsa, csa, b, rsb, csb,
	                     &beta, c, rsc, csc );

	bli_dprintm( "c: after gemm", m, n, c, rsc, csc, "% 4.3f", "" );

	// Free the memory obtained via malloc().
	free( a );
	free( b );
	free( c );


	//
	// Example 1b: Perform a general matrix-matrix multiply (gemm) operation
	//             with the left input operand (matrix A) transposed.
	//

	printf( "\n#\n#  -- Example 1b --\n#\n\n" );

	// Create some matrix and vector operands to work with.
	m = 4; n = 5; k = 3;
	rsc = 1; csc = m;
	rsa = 1; csa = k;
	rsb = 1; csb = k;
	c = malloc( m * n * sizeof( double ) );
	a = malloc( k * m * sizeof( double ) );
	b = malloc( k * n * sizeof( double ) );

	// Set the scalars to use.
	alpha = 1.0;
	beta  = 1.0;

	// Initialize the matrix operands.
	bli_drandm( 0, BLIS_DENSE, k, m, a, rsa, csa );
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
               k, n, &one, b, rsb, csb );
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
               m, n, &zero, c, rsc, csc );

	bli_dprintm( "a: randomized", k, m, a, rsa, csa, "% 4.3f", "" );
	bli_dprintm( "b: set to 1.0", k, n, b, rsb, csb, "% 4.3f", "" );
	bli_dprintm( "c: initial value", m, n, c, rsc, csc, "% 4.3f", "" );

	// c := beta * c + alpha * a^T * b, where 'a', 'b', and 'c' are general.
	bli_dgemm( BLIS_TRANSPOSE, BLIS_NO_TRANSPOSE,
	           m, n, k, &alpha, a, rsa, csa, b, rsb, csb,
	                     &beta, c, rsc, csc );

	bli_dprintm( "c: after gemm", m, n, c, rsc, csc, "% 4.3f", "" );

	// Free the memory obtained via malloc().
	free( a );
	free( b );
	free( c );


	//
	// Example 2: Perform a symmetric rank-k update (syrk) operation.
	//

	printf( "\n#\n#  -- Example 2 --\n#\n\n" );

	// Create some matrix and vector operands to work with.
	m = 5; k = 3;
	rsc = 1; csc = m;
	rsa = 1; csa = m;
	c = malloc( m * m * sizeof( double ) );
	a = malloc( m * k * sizeof( double ) );

	// Set the scalars to use.
	alpha = 1.0;

	// Initialize the matrix operands.
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
               m, m, &zero, c, rsc, csc );
	bli_drandm( 0, BLIS_DENSE, m, k, a, rsa, csa );

	// Randomize the lower triangle of 'c'.
	bli_drandm( 0, BLIS_LOWER, m, n, c, rsc, csc );

	bli_dprintm( "a: set to random values", m, k, a, rsa, csa, "% 4.3f", "" );
	bli_dprintm( "c: initial value (zeros in upper triangle)", m, m, c, rsc, csc, "% 4.3f", "" );

	// c := c + alpha * a * a^T, where 'c' is symmetric and lower-stored.
	bli_dsyrk( BLIS_LOWER, BLIS_NO_TRANSPOSE,
	           m, k, &alpha, a, rsa, csa,
	                  &beta, c, rsc, csc );

	bli_dprintm( "c: after syrk", m, m, c, rsc, csc, "% 4.3f", "" );

	// Free the memory obtained via malloc().
	free( a );
	free( c );


	//
	// Example 3: Perform a symmetric matrix-matrix multiply (symm) operation.
	//

	printf( "\n#\n#  -- Example 3 --\n#\n\n" );

	// Create some matrix and vector operands to work with.
	m = 5; n = 6;
	rsc = 1; csc = m;
	rsa = 1; csa = m;
	rsb = 1; csb = m;
	c = malloc( m * n * sizeof( double ) );
	a = malloc( m * m * sizeof( double ) );
	b = malloc( m * n * sizeof( double ) );

	// Set the scalars to use.
	alpha = 1.0;
	beta  = 1.0;

	// Initialize matrices 'b' and 'c'.
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
               m, n, &one, b, rsb, csb );
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
               m, n, &zero, c, rsc, csc );

	// Zero out all of matrix 'a'. This is optional, but will avoid possibly
	// displaying junk values in the unstored triangle.
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
               m, m, &zero, a, rsa, csa );

	// Randomize the upper triangle of 'a'.
	bli_drandm( 0, BLIS_UPPER, m, m, a, rsa, csa );

	bli_dprintm( "a: randomized (zeros in lower triangle)", m, m, a, rsa, csa, "% 4.3f", "" );
	bli_dprintm( "b: set to 1.0", m, n, b, rsb, csb, "% 4.3f", "" );
	bli_dprintm( "c: initial value", m, n, c, rsc, csc, "% 4.3f", "" );

	// c := beta * c + alpha * a * b, where 'a' is symmetric and upper-stored.
	bli_dsymm( BLIS_LEFT, BLIS_UPPER, BLIS_NO_CONJUGATE, BLIS_NO_TRANSPOSE,
	           m, n, &alpha, a, rsa, csa, b, rsb, csb,
	                  &beta, c, rsc, csc );

	bli_dprintm( "c: after symm", m, n, c, rsc, csc, "% 4.3f", "" );

	// Free the memory obtained via malloc().
	free( a );
	free( b );
	free( c );


	//
	// Example 4: Perform a triangular matrix-matrix multiply (trmm) operation.
	//

	printf( "\n#\n#  -- Example 4 --\n#\n\n" );

	// Create some matrix and vector operands to work with.
	m = 5; n = 4;
	rsa = 1; csa = m;
	rsb = 1; csb = m;
	a = malloc( m * m * sizeof( double ) );
	b = malloc( m * n * sizeof( double ) );

	// Set the scalars to use.
	alpha = 1.0;

	// Initialize matrix 'b'.
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
               m, n, &one, b, rsb, csb );

	// Zero out all of matrix 'a'. This is optional, but will avoid possibly
	// displaying junk values in the unstored triangle.
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
               m, m, &zero, a, rsa, csa );

	// Randomize the lower triangle of 'a'.
	bli_drandm( 0, BLIS_LOWER, m, m, a, rsa, csa );

	bli_dprintm( "a: randomized (zeros in upper triangle)", m, m, a, rsa, csa, "% 4.3f", "" );
	bli_dprintm( "b: initial value", m, n, b, rsb, csb, "% 4.3f", "" );

	// b := alpha * a * b, where 'a' is triangular and lower-stored.
	bli_dtrmm( BLIS_LEFT, BLIS_LOWER, BLIS_NONUNIT_DIAG, BLIS_NO_TRANSPOSE,
	           m, n, &alpha, a, rsa, csa, b, rsb, csb );

	bli_dprintm( "b: after trmm", m, n, b, rsb, csb, "% 4.3f", "" );

	// Free the memory obtained via malloc().
	free( a );
	free( b );


	//
	// Example 5: Perform a triangular solve with multiple right-hand sides
	//            (trsm) operation.
	//

	printf( "\n#\n#  -- Example 5 --\n#\n\n" );

	// Create some matrix and vector operands to work with.
	m = 5; n = 4;
	rsa = 1; csa = m;
	rsb = 1; csb = m;
	rsc = 1; csc = m;
	a = malloc( m * m * sizeof( double ) );
	b = malloc( m * n * sizeof( double ) );
	c = malloc( m * n * sizeof( double ) );

	// Set the scalars to use.
	alpha = 1.0;

	// Initialize matrix 'b'.
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
               m, n, &one, b, rsb, csb );

	// Zero out all of matrix 'a'. This is optional, but will avoid possibly
	// displaying junk values in the unstored triangle.
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
               m, m, &zero, a, rsa, csa );

	// Randomize the lower triangle of 'a'.
	bli_drandm( 0, BLIS_LOWER, m, m, a, rsa, csa );

	// Load the diagonal. By setting the diagonal to something of greater
	// absolute value than the off-diagonal elements, we increase the odds
	// that the matrix is not singular (singular matrices have no inverse).
	bli_dshiftd( 0, m, m, &two, a, rsa, csa );

	bli_dprintm( "a: randomized (zeros in upper triangle)", m, m, a, rsa, csa, "% 4.3f", "" );
	bli_dprintm( "b: initial value", m, n, b, rsb, csb, "% 4.3f", "" );

	// solve a * x = alpha * b, where 'a' is triangular and lower-stored, and
	// overwrite b with the solution matrix x.
	bli_dtrsm( BLIS_LEFT, BLIS_LOWER, BLIS_NONUNIT_DIAG, BLIS_NO_TRANSPOSE,
	           m, n, &alpha, a, rsa, csa, b, rsb, csb );

	bli_dprintm( "b: after trmm", m, n, b, rsb, csb, "% 4.3f", "" );

	// We can confirm the solution by comparing the product of a and x to the
	// original value of b.
	bli_dcopym( 0, BLIS_NONUNIT_DIAG, BLIS_DENSE, BLIS_NO_TRANSPOSE,
	            m, n, b, rsb, csb, c, rsc, csc );
	bli_dtrmm( BLIS_LEFT, BLIS_LOWER, BLIS_NONUNIT_DIAG, BLIS_NO_TRANSPOSE,
	           m, n, &alpha, a, rsa, csa, c, rsc, csc );

	bli_dprintm( "c: should equal initial value of b", m, n, c, rsc, csc, "% 4.3f", "" );

	// Free the memory obtained via malloc().
	free( a );
	free( b );
	free( c );


	return 0;
}

// -----------------------------------------------------------------------------

