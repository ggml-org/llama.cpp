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
	obj_t norm1, normi, normf;
	obj_t x, y, a, b, c, d, e, f, g;
	num_t dt;
	dim_t m, n;
	inc_t rs, cs;


	//
	// This file demonstrates working with vector and matrix objects in the
	// context of various utility operations.
	//


	//
	// Example 1: Compute various vector norms.
	//

	printf( "\n#\n#  -- Example 1 --\n#\n\n" );

	// Create a few matrices to work with.
	m = 1; n = 5; rs = 0; cs = 0;
	bli_obj_create( BLIS_DOUBLE,   m, n, rs, cs, &x );
	bli_obj_create( BLIS_DCOMPLEX, m, n, rs, cs, &y );

	// Let's also create some scalar objects to hold the norms. Note that when
	// computing the norm alpha of a vector 'x', the datatype of alpha must be
	// equal to the real projection of the datatype of 'x'.
	dt = BLIS_DOUBLE;
	bli_obj_create_1x1( dt, &norm1 );
	bli_obj_create_1x1( dt, &normi );
	bli_obj_create_1x1( dt, &normf );

	// Initialize the vectors to random values.
	bli_randv( &x );
	bli_randv( &y );

	bli_printm( "x:", &x, "% 4.3f", "" );

	// Compute the one, infinity, and frobenius norms of 'x'.
	bli_norm1v( &x, &norm1 );
	bli_normiv( &x, &normi );
	bli_normfv( &x, &normf );

	bli_printm( "x: 1-norm:", &norm1, "% 4.3f", "" );
	bli_printm( "x: infinity norm:", &normi, "% 4.3f", "" );
	bli_printm( "x: frobenius norm:", &normf, "% 4.3f", "" );

	bli_printm( "y:", &y, "% 4.3f", "" );

	// Compute the one, infinity, and frobenius norms of 'y'. Note that we
	// can reuse the same scalars from before for computing norms of
	// dcomplex matrices, since the real projection of dcomplex is double.
	bli_norm1v( &y, &norm1 );
	bli_normiv( &y, &normi );
	bli_normfv( &y, &normf );

	bli_printm( "y: 1-norm:", &norm1, "% 4.3f", "" );
	bli_printm( "y: infinity norm:", &normi, "% 4.3f", "" );
	bli_printm( "y: frobenius norm:", &normf, "% 4.3f", "" );


	//
	// Example 2: Compute various matrix norms.
	//

	printf( "\n#\n#  -- Example 2 --\n#\n\n" );

	// Create a few matrices to work with.
	m = 5; n = 6; rs = 0; cs = 0;
	bli_obj_create( BLIS_DOUBLE,   m, n, rs, cs, &a );
	bli_obj_create( BLIS_DCOMPLEX, m, n, rs, cs, &b );

	// Initialize the matrices to random values.
	bli_randm( &a );
	bli_randm( &b );

	bli_printm( "a:", &a, "% 4.3f", "" );

	// Compute the one, infinity, and frobenius norms of 'a'.
	bli_norm1m( &a, &norm1 );
	bli_normim( &a, &normi );
	bli_normfm( &a, &normf );

	bli_printm( "a: 1-norm:", &norm1, "% 4.3f", "" );
	bli_printm( "a: infinity norm:", &normi, "% 4.3f", "" );
	bli_printm( "a: frobenius norm:", &normf, "% 4.3f", "" );

	bli_printm( "b:", &b, "% 4.3f", "" );

	// Compute the one-norm of 'b'.
	bli_norm1m( &b, &norm1 );
	bli_normim( &b, &normi );
	bli_normfm( &b, &normf );

	bli_printm( "b: 1-norm:", &norm1, "% 4.3f", "" );
	bli_printm( "b: infinity norm:", &normi, "% 4.3f", "" );
	bli_printm( "b: frobenius norm:", &normf, "% 4.3f", "" );


	//
	// Example 3: Make a real matrix explicitly symmetric (or Hermitian).
	//

	printf( "\n#\n#  -- Example 3 --\n#\n\n" );

	// Create a few matrices to work with.
	m = 4; n = 4; rs = 0; cs = 0;
	bli_obj_create( BLIS_DOUBLE, m, n, rs, cs, &c );
	bli_obj_create( BLIS_DOUBLE, m, n, rs, cs, &d );

	// Initialize all of 'c' to -1.0 to simulate junk values.
	bli_setm( &BLIS_MINUS_ONE, &c );

	// Set the structure and uplo of 'c'.
	bli_obj_set_struc( BLIS_SYMMETRIC, &c );
	bli_obj_set_uplo( BLIS_LOWER, &c );

	// Randomize the lower triangle of 'c'.
	bli_randm( &c );

	bli_printm( "c (initial state):", &c, "% 4.3f", "" );

	// mksymm on a real matrix transposes the stored triangle into the
	// unstored triangle, making the matrix densely symmetric.
	bli_mksymm( &c );

	bli_printm( "c (after mksymm on lower triangle):", &c, "% 4.3f", "" );

	// Digression: Most people think only of complex matrices as being able
	// to be complex. However, in BLIS, we define Hermitian operations on
	// real matrices, too--they are simply equivalent to the corresponding
	// symmetric operation. For example, when we make a real matrix explicitly
	// Hermitian, the result is indistinguishable from making it symmetric.

	// Initialize all of 'd' to -1.0 to simulate junk values.
	bli_setm( &BLIS_MINUS_ONE, &d );

	bli_obj_set_struc( BLIS_HERMITIAN, &d );
	bli_obj_set_uplo( BLIS_LOWER, &d );

	// Randomize the lower triangle of 'd'.
	bli_randm( &d );

	bli_printm( "d (initial state):", &d, "% 4.3f", "" );

	// mkherm on a real matrix behaves the same as mksymm, as there are no
	// imaginary elements to conjugate.
	bli_mkherm( &d );

	bli_printm( "d (after mkherm on lower triangle):", &d, "% 4.3f", "" );


	//
	// Example 4: Make a complex matrix explicitly symmetric or Hermitian.
	//

	printf( "\n#\n#  -- Example 4 --\n#\n\n" );

	// Create a few matrices to work with.
	m = 4; n = 4; rs = 0; cs = 0;
	bli_obj_create( BLIS_DCOMPLEX, m, n, rs, cs, &e );
	bli_obj_create( BLIS_DCOMPLEX, m, n, rs, cs, &f );

	// Initialize all of 'e' to -1.0 to simulate junk values.
	bli_setm( &BLIS_MINUS_ONE, &e );

	// Set the structure and uplo of 'e'.
	bli_obj_set_struc( BLIS_SYMMETRIC, &e );
	bli_obj_set_uplo( BLIS_UPPER, &e );

	// Randomize the upper triangle of 'e'.
	bli_randm( &e );

	bli_printm( "e (initial state):", &e, "% 4.3f", "" );

	// mksymm on a complex matrix transposes the stored triangle into the
	// unstored triangle.
	bli_mksymm( &e );

	bli_printm( "e (after mksymm):", &e, "% 4.3f", "" );

	// Initialize all of 'f' to -1.0 to simulate junk values.
	bli_setm( &BLIS_MINUS_ONE, &f );

	// Set the structure and uplo of 'f'.
	bli_obj_set_struc( BLIS_HERMITIAN, &f );
	bli_obj_set_uplo( BLIS_UPPER, &f );

	// Randomize the upper triangle of 'f'.
	bli_randm( &f );

	bli_printm( "f (initial state):", &f, "% 4.3f", "" );

	// mkherm on a complex matrix transposes and conjugates the stored
	// triangle into the unstored triangle.
	bli_mkherm( &f );

	bli_printm( "f (after mkherm):", &f, "% 4.3f", "" );


	//
	// Example 5: Make a real matrix explicitly triangular.
	//

	printf( "\n#\n#  -- Example 5 --\n#\n\n" );

	// Create a few matrices to work with.
	m = 5; n = 5; rs = 0; cs = 0;
	bli_obj_create( BLIS_DOUBLE, m, n, rs, cs, &g );

	// Initialize all of 'g' to -1.0 to simulate junk values.
	bli_setm( &BLIS_MINUS_ONE, &g );

	// Set the structure and uplo of 'g'.
	bli_obj_set_struc( BLIS_TRIANGULAR, &g );
	bli_obj_set_uplo( BLIS_LOWER, &g );

	// Randomize the lower triangle of 'g'.
	bli_randm( &g );

	bli_printm( "g (initial state):", &g, "% 4.3f", "" );

	// mktrim does not explicitly copy any data, since presumably the stored
	// triangle already contains the data of interest. However, mktrim does
	// explicitly writes zeros to the unstored region.
	bli_mktrim( &g );

	bli_printm( "g (after mktrim):", &g, "% 4.3f", "" );


	// Free the objects.
	bli_obj_free( &norm1 );
	bli_obj_free( &normi );
	bli_obj_free( &normf );
	bli_obj_free( &x );
	bli_obj_free( &y );
	bli_obj_free( &a );
	bli_obj_free( &b );
	bli_obj_free( &c );
	bli_obj_free( &d );
	bli_obj_free( &e );
	bli_obj_free( &f );
	bli_obj_free( &g );

	return 0;
}

// -----------------------------------------------------------------------------

