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
	double*   x;
	dcomplex* y;
	double*   a;
	dcomplex* b;
	double*   c;
	double*   d;
	dcomplex* e;
	dcomplex* f;
	double*   g;
	double    norm1, normi, normf;
	dim_t     m, n;
	inc_t     rs, cs;

	// Initialize some basic constants.
	double   minus_one   =   -1.0;
	dcomplex minus_one_z = { -1.0, 0.0 };


	//
	// This file demonstrates working with vector and matrices in the
	// context of various utility operations.
	//


	//
	// Example 1: Compute various vector norms.
	//

	printf( "\n#\n#  -- Example 1 --\n#\n\n" );

	// Create a few matrices to work with.
	m = 1; n = 5; rs = 5; cs = 1;
	x = malloc( m * n * sizeof( double ) );
	y = malloc( m * n * sizeof( dcomplex ) );

	// Initialize the vectors to random values.
	bli_drandv( n, x, 1 );
	bli_zrandv( n, y, 1 );

	bli_dprintm( "x", m, n, x, rs, cs, "% 4.3f", "" );

	// Compute the one, infinity, and frobenius norms of 'x'. Note that when
	// computing the norm alpha of a vector 'x', the datatype of alpha must be
	// equal to the real projection of the datatype of 'x'.
	bli_dnorm1v( n, x, 1, &norm1 );
	bli_dnormiv( n, x, 1, &normi );
	bli_dnormfv( n, x, 1, &normf );

	bli_dprintm( "x: 1-norm:", 1, 1, &norm1, rs, cs, "% 4.3f", "" );
	bli_dprintm( "x: infinity norm:", 1, 1, &normi, rs, cs, "% 4.3f", "" );
	bli_dprintm( "x: frobenius norm:", 1, 1, &normf, rs, cs, "% 4.3f", "" );

	bli_zprintm( "y", m, n, y, rs, cs, "% 4.3f", "" );

	// Compute the one, infinity, and frobenius norms of 'y'. Note that we
	// can reuse the same scalars from before for computing norms of
	// dcomplex matrices, since the real projection of dcomplex is double.
	bli_znorm1v( n, y, 1, &norm1 );
	bli_znormiv( n, y, 1, &normi );
	bli_znormfv( n, y, 1, &normf );

	bli_dprintm( "y: 1-norm:", 1, 1, &norm1, 1, 1, "% 4.3f", "" );
	bli_dprintm( "y: infinity norm:", 1, 1, &normi, 1, 1, "% 4.3f", "" );
	bli_dprintm( "y: frobenius norm:", 1, 1, &normf, 1, 1, "% 4.3f", "" );


	//
	// Example 2: Compute various matrix norms.
	//

	printf( "\n#\n#  -- Example 2 --\n#\n\n" );

	// Create a few matrices to work with.
	m = 5; n = 6; rs = 1; cs = m;
	a = malloc( m * n * sizeof( double ) );
	b = malloc( m * n * sizeof( dcomplex ) );

	// Initialize the matrices to random values.
	bli_drandm( 0, BLIS_DENSE, m, n, a, rs, cs );
	bli_zrandm( 0, BLIS_DENSE, m, n, b, rs, cs );

	bli_dprintm( "a:", m, n, a, rs, cs, "% 4.3f", "" );

	// Compute the one-norm of 'a'.
	bli_dnorm1m( 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
	             m, n, a, rs, cs, &norm1 );
	bli_dnormim( 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
	             m, n, a, rs, cs, &normi );
	bli_dnormfm( 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
	             m, n, a, rs, cs, &normf );

	bli_dprintm( "a: 1-norm:", 1, 1, &norm1, 1, 1, "% 4.3f", "" );
	bli_dprintm( "a: infinity norm:", 1, 1, &normi, 1, 1, "% 4.3f", "" );
	bli_dprintm( "a: frobenius norm:", 1, 1, &normf, 1, 1, "% 4.3f", "" );

	bli_zprintm( "b:", m, n, b, rs, cs, "% 4.3f", "" );

	// Compute the one-norm of 'b'.
	bli_znorm1m( 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
	             m, n, b, rs, cs, &norm1 );
	bli_znormim( 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
	             m, n, b, rs, cs, &normi );
	bli_znormfm( 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
	             m, n, b, rs, cs, &normf );

	bli_dprintm( "a: 1-norm:", 1, 1, &norm1, 1, 1, "% 4.3f", "" );
	bli_dprintm( "a: infinity norm:", 1, 1, &normi, 1, 1, "% 4.3f", "" );
	bli_dprintm( "a: frobenius norm:", 1, 1, &normf, 1, 1, "% 4.3f", "" );


	//
	// Example 3: Make a real matrix explicitly symmetric (or Hermitian).
	//

	printf( "\n#\n#  -- Example 3 --\n#\n\n" );

	// Create a few matrices to work with.
	m = 4; n = 4; rs = 1; cs = m;
	c = malloc( m * m * sizeof( double ) );
	d = malloc( m * m * sizeof( double ) );

	// Initialize all of 'c' to -1.0 to simulate junk values.
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
	           m, m, &minus_one, c, rs, cs );

	// Randomize the lower triangle of 'c'.
	bli_drandm( 0, BLIS_LOWER, m, m, c, rs, cs );

	bli_dprintm( "c (initial state):", m, m, c, rs, cs, "% 4.3f", "" );

	// mksymm on a real matrix transposes the stored triangle into the
	// unstored triangle, making the matrix densely symmetric.
	bli_dmksymm( BLIS_LOWER, m, c, rs, cs );

	bli_dprintm( "c (after mksymm on lower triangle):", m, m, c, rs, cs, "% 4.3f", "" );

	// Digression: Most people think only of complex matrices as being able
	// to be complex. However, in BLIS, we define Hermitian operations on
	// real matrices, too--they are simply equivalent to the corresponding
	// symmetric operation. For example, when we make a real matrix explicitly
	// Hermitian, the result is indistinguishable from making it symmetric.

	// Initialize all of 'd' to -1.0 to simulate junk values.
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
	           m, m, &minus_one, d, rs, cs );

	// Randomize the lower triangle of 'd'.
	bli_drandm( 0, BLIS_LOWER, m, m, d, rs, cs );

	bli_dprintm( "d (initial state):", m, m, d, rs, cs, "% 4.3f", "" );

	// mkherm on a real matrix behaves the same as mksymm, as there are no
	// imaginary elements to conjugate.
	bli_dmkherm( BLIS_LOWER, m, d, rs, cs );

	bli_dprintm( "c (after mkherm on lower triangle):", m, m, d, rs, cs, "% 4.3f", "" );


	//
	// Example 4: Make a complex matrix explicitly symmetric or Hermitian.
	//

	printf( "\n#\n#  -- Example 4 --\n#\n\n" );

	// Create a few matrices to work with.
	m = 4; n = 4; rs = 1; cs = m;
	e = malloc( m * m * sizeof( dcomplex ) );
	f = malloc( m * m * sizeof( dcomplex ) );

	// Initialize all of 'e' to -1.0 to simulate junk values.
	bli_zsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
	           m, m, &minus_one_z, e, rs, cs );

	// Randomize the upper triangle of 'e'.
	bli_zrandm( 0, BLIS_UPPER, m, m, e, rs, cs );

	bli_zprintm( "e (initial state):", m, m, e, rs, cs, "% 4.3f", "" );

	// mksymm on a complex matrix transposes the stored triangle into the
	// unstored triangle.
	bli_zmksymm( BLIS_UPPER, m, e, rs, cs );

	bli_zprintm( "e (after mksymm on lower triangle):", m, m, e, rs, cs, "% 4.3f", "" );

	// Initialize all of 'f' to -1.0 to simulate junk values.
	bli_zsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
	           m, m, &minus_one_z, f, rs, cs );

	// Randomize the upper triangle of 'd'.
	bli_zrandm( 0, BLIS_UPPER, m, m, f, rs, cs );

	bli_zprintm( "f (initial state):", m, m, f, rs, cs, "% 4.3f", "" );

	// mkherm on a real matrix behaves the same as mksymm, as there are no
	// imaginary elements to conjugate.
	bli_zmkherm( BLIS_UPPER, m, f, rs, cs );

	bli_zprintm( "f (after mkherm on lower triangle):", m, m, f, rs, cs, "% 4.3f", "" );


	//
	// Example 5: Make a real matrix explicitly triangular.
	//

	printf( "\n#\n#  -- Example 5 --\n#\n\n" );

	// Create a few matrices to work with.
	m = 5; n = 5; rs = 1; cs = m;
	g = malloc( m * m * sizeof( double ) );

	// Initialize all of 'g' to -1.0 to simulate junk values.
	bli_dsetm( BLIS_NO_CONJUGATE, 0, BLIS_NONUNIT_DIAG, BLIS_DENSE,
	           m, m, &minus_one, g, rs, cs );

	// Randomize the lower triangle of 'g'.
	bli_drandm( 0, BLIS_LOWER, m, m, g, rs, cs );

	bli_dprintm( "g (initial state):", m, m, g, rs, cs, "% 4.3f", "" );

	// mktrim does not explicitly copy any data, since presumably the stored
	// triangle already contains the data of interest. However, mktrim does
	// explicitly writes zeros to the unstored region.
	bli_dmktrim( BLIS_LOWER, m, g, rs, cs );

	bli_dprintm( "g (after mktrim):", m, m, g, rs, cs, "% 4.3f", "" );


	// Free the memory obtained via malloc().
	free( x );
	free( y );
	free( a );
	free( b );
	free( c );
	free( d );
	free( e );
	free( f );
	free( g );

	return 0;
}

// -----------------------------------------------------------------------------

