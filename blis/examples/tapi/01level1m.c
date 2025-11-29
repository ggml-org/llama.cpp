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
	double*   a;
	double*   b;
	double*   c;
	double*   d;
	double*   e;
	double*   f;
	dcomplex* g;
	dcomplex* h;
	double    alpha, beta, gamma;
	dim_t     m, n;
	inc_t     rs, cs;

	// Initialize some basic constants.
	double   zero        =   0.0;
	double   one         =   1.0;
	double   minus_one   =  -1.0;
	dcomplex minus_one_z = {-1.0, 0.0};


	//
	// This file demonstrates working with matrices and the level-1m
	// operations.
	//


	//
	// Example 1: Create matrices and then broadcast (copy) scalar
	//            values to all elements.
	//

	printf( "\n#\n#  -- Example 1 --\n#\n\n" );

	// Create a few matrices to work with. We make them all of the same
	// dimensions so that we can perform operations between them.
	m = 2; n = 3; rs = 1; cs = m;
	a = malloc( m * n * sizeof( double ) );
	b = malloc( m * n * sizeof( double ) );
	c = malloc( m * n * sizeof( double ) );
	d = malloc( m * n * sizeof( double ) );
	e = malloc( m * n * sizeof( double ) );

	// Let's initialize some scalars.
	alpha = 2.0;
	beta  = 0.2;
	gamma = 3.0;

	printf( "alpha:\n% 4.3f\n\n", alpha );
	printf( "beta:\n% 4.3f\n\n", beta );
	printf( "gamma:\n% 4.3f\n\n", gamma );
	printf( "\n" );

	// Matrices, like vectors, can set by "broadcasting" a constant to every
	// element. Note that the second argument (0) is the diagonal offset.
	// The diagonal offset is only used when the uplo value is something other
	// than BLIS_DENSE (e.g. BLIS_LOWER or BLIS_UPPER).
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
	           m, n, &one, a, rs, cs );
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
	           m, n, &alpha, b, rs, cs );
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
	           m, n, &zero, c, rs, cs );

	bli_dprintm( "a := 1.0", m, n, a, rs, cs, "% 4.3f", "" );
	bli_dprintm( "b := alpha", m, n, b, rs, cs, "% 4.3f", "" );
	bli_dprintm( "c := 0.0", m, n, c, rs, cs, "% 4.3f", "" );


	//
	// Example 2: Randomize a matrix object.
	//

	printf( "\n#\n#  -- Example 2 --\n#\n\n" );

	bli_drandm( 0, BLIS_DENSE, m, n, e, rs, cs );

	bli_dprintm( "e (randomized):", m, n, e, rs, cs, "% 4.3f", "" );


	//
	// Example 3: Perform element-wise operations on matrices.
	//

	printf( "\n#\n#  -- Example 3 --\n#\n\n" );

	// Copy a matrix.
	bli_dcopym( 0, BLIS_NONUNIT_DIAG, BLIS_DENSE, BLIS_NO_TRANSPOSE,
	            m, n, e, rs, cs, d, rs, cs );
	bli_dprintm( "d := e", m, n, d, rs, cs, "% 4.3f", "" );

	// Add and subtract vectors.
	bli_daddm( 0, BLIS_NONUNIT_DIAG, BLIS_DENSE, BLIS_NO_TRANSPOSE,
	           m, n, a, rs, cs, d, rs, cs );
	bli_dprintm( "d := d + a", m, n, d, rs, cs, "% 4.3f", "" );

	bli_dsubm( 0, BLIS_NONUNIT_DIAG, BLIS_DENSE, BLIS_NO_TRANSPOSE,
	           m, n, a, rs, cs, e, rs, cs );
	bli_dprintm( "e := e - a", m, n, e, rs, cs, "% 4.3f", "" );

	// Scale a matrix (destructive).
	bli_dscalm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
	            m, n, &alpha, e, rs, cs );
	bli_dprintm( "e := alpha * e", m, n, e, rs, cs, "% 4.3f", "" );

	// Scale a matrix (non-destructive).
	bli_dscal2m( 0, BLIS_NONUNIT_DIAG, BLIS_DENSE, BLIS_NO_TRANSPOSE,
	             m, n, &beta, e, rs, cs, c, rs, cs );
	bli_dprintm( "c := beta * e", m, n, c, rs, cs, "% 4.3f", "" );

	// Scale and accumulate between matrices.
	bli_daxpym( 0, BLIS_NONUNIT_DIAG, BLIS_DENSE, BLIS_NO_TRANSPOSE,
	            m, n, &alpha, a, rs, cs, c, rs, cs );
	bli_dprintm( "c := alpha * a", m, n, c, rs, cs, "% 4.3f", "" );


	//
	// Example 4: Copy and transpose a matrix.
	//

	printf( "\n#\n#  -- Example 4 --\n#\n\n" );

	// Create an n-by-m matrix into which we can copy-transpose an m-by-n
	// matrix.
	f = malloc( n * m * sizeof( double ) );
	dim_t rsf = 1, csf = n;

	// Initialize all of 'f' to -1.0 to simulate junk values.
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
	           n, m, &minus_one, f, rsf, csf );

	bli_dprintm( "e:", m, n, e, rs, cs, "% 4.3f", "" );
	bli_dprintm( "f (initial value):", n, m, f, rsf, csf, "% 4.3f", "" );


	// Copy 'e' to 'f', transposing 'e' in the process. Notice that we haven't
	// modified any properties of 'f'. It's the source operand that matters
	// when marking an operand for transposition, not the destination.
	bli_dcopym( 0, BLIS_NONUNIT_DIAG, BLIS_DENSE, BLIS_TRANSPOSE,
	            n, m, e, rs, cs, f, rsf, csf );

	bli_dprintm( "f (copied value):", n, m, f, rsf, csf, "% 4.3f", "" );


	//
	// Example 5: Copy and Hermitian-transpose a matrix.
	//

	printf( "\n#\n#  -- Example 5 --\n#\n\n" );

	g = malloc( m * n * sizeof(dcomplex) );
	h = malloc( n * m * sizeof(dcomplex) );

	bli_zrandm( 0, BLIS_DENSE, m, n, g, rs, cs );

	bli_zsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
	           n, m, &minus_one_z, h, rsf, csf );

	bli_zprintm( "g:", m, n, g, rs, cs, "% 4.3f", "" );
	bli_zprintm( "h (initial value):", n, m, h, rsf, csf, "% 4.3f", "" );

	bli_zcopym( 0, BLIS_NONUNIT_DIAG, BLIS_DENSE, BLIS_CONJ_TRANSPOSE,
	            n, m, g, rs, cs, h, rsf, csf );

	bli_zprintm( "h (copied value):", n, m, h, rsf, csf, "% 4.3f", "" );


	// Free the memory obtained via malloc().
	free( a );
	free( b );
	free( c );
	free( d );
	free( e );
	free( f );
	free( g );
	free( h );

	return 0;
}

// -----------------------------------------------------------------------------

