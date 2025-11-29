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
	obj_t a, b, c, d, e, f, g, h;
	num_t dt;
	dim_t m, n;
	inc_t rs, cs;


	//
	// This file demonstrates working with matrix objects and the level-1m
	// operations.
	//


	//
	// Example 1: Create matrix objects and then broadcast (copy) scalar
	//            values to all elements.
	//

	printf( "\n#\n#  -- Example 1 --\n#\n\n" );

	// Create a few matrices to work with. We make them all of the same
	// dimensions so that we can perform operations between them.
	dt = BLIS_DOUBLE;
	m = 2; n = 3; rs = 0; cs = 0;
	bli_obj_create( dt, m, n, rs, cs, &a );
	bli_obj_create( dt, m, n, rs, cs, &b );
	bli_obj_create( dt, m, n, rs, cs, &c );
	bli_obj_create( dt, m, n, rs, cs, &d );
	bli_obj_create( dt, m, n, rs, cs, &e );

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

	// Matrices, like vectors, can set by "broadcasting" a constant to every
	// element.
	bli_setm( &BLIS_ONE, &a );
	bli_setm( &alpha, &b );
	bli_setm( &BLIS_ZERO, &c );

	bli_printm( "a := 1.0", &a, "% 4.3f", "" );
	bli_printm( "b := alpha", &b, "% 4.3f", "" );
	bli_printm( "c := 0.0", &c, "% 4.3f", "" );


	//
	// Example 2: Randomize a matrix object.
	//

	printf( "\n#\n#  -- Example 2 --\n#\n\n" );

	// Set a matrix to random values.
	bli_randm( &e );

	bli_printm( "e (randomized):", &e, "% 4.3f", "" );


	//
	// Example 3: Perform element-wise operations on matrices.
	//

	printf( "\n#\n#  -- Example 3 --\n#\n\n" );

	// Copy a matrix.
	bli_copym( &e, &d );
	bli_printm( "d := e", &d, "% 4.3f", "" );

	// Add and subtract vectors.
	bli_addm( &a, &d );
	bli_printm( "d := d + a", &d, "% 4.3f", "" );

	bli_subm( &a, &e );
	bli_printm( "e := e - a", &e, "% 4.3f", "" );

	// Scale a matrix (destructive).
	bli_scalm( &alpha, &e );
	bli_printm( "e := alpha * e", &e, "% 4.3f", "" );

	// Scale a matrix (non-destructive).
	bli_scal2m( &beta, &e, &c );
	bli_printm( "c := beta * e", &c, "% 4.3f", "" );

	// Scale and accumulate between matrices.
	bli_axpym( &alpha, &a, &c );
	bli_printm( "c := c + alpha * a", &c, "% 4.3f", "" );


	//
	// Example 4: Copy and transpose a matrix.
	//

	printf( "\n#\n#  -- Example 4 --\n#\n\n" );

	// Create an n-by-m matrix into which we can copy-transpose an m-by-n
	// matrix.
	bli_obj_create( dt, n, m, rs, cs, &f );

	// Initialize all of 'f' to -1.0 to simulate junk values.
	bli_setm( &BLIS_MINUS_ONE, &f );

	bli_printm( "e:", &e, "% 4.3f", "" );
	bli_printm( "f (initial value):", &f, "% 4.3f", "" );

	// Since we are going to copy 'e' to 'f', we need to indicate a transpose
	// on 'e', the input operand. Transposition can be indicated by setting a
	// bit in the object. Since it always starts out as "no transpose", we can
	// simply toggle the bit.
	bli_obj_toggle_trans( &e );

	// Another way to mark and object for transposition is to set it directly.
	//bli_obj_set_onlytrans( BLIS_TRANSPOSE, &e );

	// A third way is to "apply" a transposition. This is equivalent to toggling
	// the transposition when the value being applied is BLIS_TRANSPOSE. If
	// the value applied is BLIS_NO_TRANSPOSE, the transposition bit in the
	// targeted object is unaffected. (Applying transposes is more useful in
	// practice when the 'trans' argument is a variable and not a constant
	// literal.)
	//bli_obj_apply_trans( BLIS_TRANSPOSE, &e );
	//bli_obj_apply_trans( BLIS_NO_TRANSPOSE, &e );
	//bli_obj_apply_trans( trans, &e );

	// Copy 'e' to 'f', transposing 'e' in the process. Notice that we haven't
	// modified any properties of 'f'. It's the source operand that matters
	// when marking an operand for transposition, not the destination.
	bli_copym( &e, &f );

	bli_printm( "f (copied value):", &f, "% 4.3f", "" );


	//
	// Example 5: Copy and Hermitian-transpose a matrix.
	//

	printf( "\n#\n#  -- Example 5 --\n#\n\n" );

	// Create an n-by-m complex matrix into which we can Hermitian-transpose
	// (or, conjugate-transpose) another complex (m-by-n) matrix.
	dt = BLIS_DCOMPLEX;
	bli_obj_create( dt, m, n, rs, cs, &g );
	bli_obj_create( dt, n, m, rs, cs, &h );

	// Randomize 'g', the input operand.
	bli_randm( &g );

	// Initialize all of 'h' to -1.0 to simulate junk values.
	bli_setm( &BLIS_MINUS_ONE, &h );

	bli_printm( "g:", &g, "% 4.3f", "" );
	bli_printm( "h (initial value):", &h, "% 4.3f", "" );

	// Set both the transpose and conjugation bits.
	bli_obj_toggle_trans( &g );
	bli_obj_toggle_conj( &g );

	// Copy 'g' to 'h', conjugating and transposing 'g' in the process.
	// Once again, notice that it's the source operand that we've marked for
	// conjugation.
	bli_copym( &g, &h );

	bli_printm( "h (copied value):", &h, "% 4.3f", "" );


	// Free the objects.
	bli_obj_free( &alpha );
	bli_obj_free( &beta );
	bli_obj_free( &gamma );
	bli_obj_free( &a );
	bli_obj_free( &b );
	bli_obj_free( &c );
	bli_obj_free( &d );
	bli_obj_free( &e );
	bli_obj_free( &f );
	bli_obj_free( &g );
	bli_obj_free( &h );

	return 0;
}

// -----------------------------------------------------------------------------

