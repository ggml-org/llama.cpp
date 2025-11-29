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
	num_t dt;
	dim_t m, n;
	inc_t rs, cs;


	//
	// This file demonstrates level-1m operations on structured matrices.
	//


	//
	// Example 1: Initialize the upper triangle of a matrix to random values.
	//

	printf( "\n#\n#  -- Example 1 --\n#\n\n" );

	obj_t a;

	// Create a matrix to work with.
	dt = BLIS_DOUBLE;
	m = 5; n = 5; rs = 0; cs = 0;
	bli_obj_create( dt, m, n, rs, cs, &a );

	// First, we mark the matrix structure as triangular.
	bli_obj_set_struc( BLIS_TRIANGULAR, &a );

	// Next, we specify whether the lower part or the upper part is to be
	// recognized as the "stored" region (which we call the uplo field). The
	// strictly opposite part (in this case, the strictly lower region) will
	// be *assumed* to be zero during computation. However, when printed out,
	// the strictly lower part may contain junk values.
	bli_obj_set_uplo( BLIS_UPPER, &a );

	// Now set the upper triangle to random values.
	bli_randm( &a );

	bli_printm( "a: randomize upper part (lower part may contain garbage)", &a, "% 4.3f", "" );


	//
	// Example 2: Initialize the upper triangle of a matrix to random values
	//            but also explicitly set the strictly lower triangle to zero.
	//

	printf( "\n#\n#  -- Example 2 --\n#\n\n" );

	obj_t b, bl;

	// Create a matrix to work with.
	dt = BLIS_DOUBLE;
	m = 5; n = 5; rs = 0; cs = 0;
	bli_obj_create( dt, m, n, rs, cs, &b );

	// Set structure and uplo.
	bli_obj_set_struc( BLIS_TRIANGULAR, &b );
	bli_obj_set_uplo( BLIS_UPPER, &b );

	// Create an alias, 'bl', of the original object 'b'. Both objects will
	// refer to the same underlying matrix elements, but now we will have two
	// different "views" into the matrix. Aliases are simply "shallow copies"
	// of the objects, meaning no additional memory allocation takes place.
	// Therefore it is up to the API user (you) to make sure that you only
	// free the original object (or exactly one of the aliases).
	bli_obj_alias_to( &b, &bl );

	// Digression: Each object contains a diagonal offset (even vectors),
	// even if it is never needed. The diagonal offset for a newly-created
	// object (ie: objects created via bli_obj_create*()) defaults to 0,
	// meaning it intersects element (0,0), but it can be changed. When the
	// diagonal offset delta is positive, the diagonal intersects element
	// (0,delta). When the diagonal offset is negative, the diagonal
	// intersects element (-delta,0). In other words, think of element (0,0)
	// as the origin of a coordinate plane, with the diagonal being the
	// x-axis value.

	// Set the diagonal offset of 'bl' to -1.
	bli_obj_set_diag_offset( -1, &bl );
	
	// Set the uplo field of 'bl' to "lower".
	bli_obj_set_uplo( BLIS_LOWER, &bl );

	// Set the upper triangle of 'b' to random values.
	bli_randm( &b );

	// Set the strictly lower triangle of 'b' to zero (by setting the lower
	// triangle of 'bl' to zero).
	bli_setm( &BLIS_ZERO, &bl );

	bli_printm( "b: randomize upper part; set strictly lower part to 0.0", &b, "% 4.3f", "" );

	// You may not see the effect of setting the strictly lower part to zero,
	// since those values may already be zero (instead of random junk). So
	// let's set it to something you'll notice, like -1.0.
	bli_setm( &BLIS_MINUS_ONE, &bl );

	bli_printm( "b: randomize upper part; set strictly lower part to -1.0", &b, "% 4.3f", "" );


	//
	// Example 3: Copy the lower triangle of an existing object to a newly
	//            created (but otherwise uninitialized) object.
	//

	printf( "\n#\n#  -- Example 3 --\n#\n\n" );

	obj_t c;

	// Create a matrix to work with.
	dt = BLIS_DOUBLE;
	m = 5; n = 5; rs = 0; cs = 0;
	bli_obj_create( dt, m, n, rs, cs, &c );

	// Reset the diagonal offset of 'bl' to 0.
	bli_obj_set_diag_offset( 0, &bl );

	// Copy the lower triangle of matrix 'b' from Example 2 to object 'c'.
	// This should give us -1.0 in the strictly lower part and some non-zero
	// random values along the diagonal. Note that since 'c' is starting out
	// uninitialized, the strictly upper part could contain junk.
	bli_copym( &bl, &c );

	bli_printm( "c: copy lower part of b (upper part may contain garbage)", &c, "% 4.3f", "" );

	// Notice that the structure and uplo properties of 'c' were set to their
	// default values, BLIS_GENERAL and BLIS_DENSE, respectively. Thus, it is
	// the structure and uplo of the *source* operand that controls what gets
	// copied, regardless of the structure/uplo of the destination. To
	// demonstrate this further, let's see what happens when we copy 'bl'
	// (which is lower triangular) to 'a' (which is upper triangular).

	bli_copym( &bl, &a );

	// The result is that the lower part (diagonal and strictly lower part) is
	// copied into 'a', but the elements in the strictly upper part of 'a' are
	// unaffected. Note, however, that 'a' is still marked as upper triangular
	// and so in future computations where 'a' is an input operand, the -1.0
	// values that were copied from 'bl' into the lower triangle will be
	// ignored. Generally speaking, level-1m operations on triangular matrices
	// ignore the "unstored" regions of input operands because they are assumed
	// to be zero).

	bli_printm( "a: copy lower triangular bl to upper triangular a", &a, "% 4.3f", "" );


	//
	// Example 4: Copy the lower triangle of an existing object into the
	//            upper triangle of an existing object.
	//

	printf( "\n#\n#  -- Example 4 --\n#\n\n" );

	obj_t d;

	// Create a matrix to work with.
	dt = BLIS_DOUBLE;
	m = 5; n = 5; rs = 0; cs = 0;
	bli_obj_create( dt, m, n, rs, cs, &d );

	// Let's start by setting entire destination matrix to zero.
	bli_setm( &BLIS_ZERO, &d );

	bli_printm( "d: initial value (all zeros)", &d, "% 4.3f", "" );

	// Recall that 'bl' is marked as lower triangular with a diagonal offset
	// of 0. Also recall that 'bl' is an alias of 'b', which is now fully
	// initialized. But let's change a few values manually so we can later
	// see the full effect of the transposition.
	bli_setijm( 2.0, 0.0, 2, 0, &bl );
	bli_setijm( 3.0, 0.0, 3, 0, &bl );
	bli_setijm( 4.0, 0.0, 4, 0, &bl );
	bli_setijm( 3.1, 0.0, 3, 1, &bl );
	bli_setijm( 3.2, 0.0, 3, 2, &bl );

	bli_printm( "bl: lower triangular bl is aliased to b", &bl, "% 4.3f", "" );

	// We want to pluck out the lower triangle and transpose it into the upper
	// triangle of 'd'.
	bli_obj_toggle_trans( &bl );

	// Now we copy the transpose of the lower part of 'bl' into the upper
	// part of 'd'. (Again, notice that we haven't modified any properties of
	// 'd'. It's the source operand that matters, not the destination!)
	bli_copym( &bl, &d );

	bli_printm( "d: transpose of lower triangular of bl copied to d", &d, "% 4.3f", "" );


	//
	// Example 5: Create a rectangular matrix (m > n) with a lower trapezoid
	//            containing random values, then set the strictly upper
	//            triangle to zeros.
	//

	printf( "\n#\n#  -- Example 5 --\n#\n\n" );

	obj_t e, el;

	// Create a matrix to work with.
	dt = BLIS_DOUBLE;
	m = 6; n = 4; rs = 0; cs = 0;
	bli_obj_create( dt, m, n, rs, cs, &e );

	// Initialize the entire matrix to -1.0 to simulate junk values.
	bli_setm( &BLIS_MINUS_ONE, &e );

	bli_printm( "e: initial value (all -1.0)", &e, "% 4.3f", "" );

	// Create an alias to work with.
	bli_obj_alias_to( &e, &el );

	// Set structure and uplo of 'el'.
	bli_obj_set_struc( BLIS_TRIANGULAR, &el );
	bli_obj_set_uplo( BLIS_LOWER, &el );

	// Digression: Notice that "triangular" structure does not require that
	// the matrix be square. Rather, it simply means that either the part above
	// or below the diagonal will be assumed to be zero.

	// Randomize the lower trapezoid.
	bli_randm( &el );

	bli_printm( "e: after lower trapezoid randomized", &e, "% 4.3f", "" );

	// Move the diagonal offset of 'el' to 1 and flip the uplo field to
	// "upper".
	bli_obj_set_diag_offset( 1, &el );
	bli_obj_set_uplo( BLIS_UPPER, &el );

	// Set the upper triangle to zero.
	bli_setm( &BLIS_ZERO, &el );

	bli_printm( "e: after upper triangle set to zero", &e, "% 4.3f", "" );


	//
	// Example 6: Create an upper Hessenberg matrix of random values and then
	//            set the "unstored" values to zero.
	//

	printf( "\n#\n#  -- Example 6 --\n#\n\n" );

	obj_t h, hl;

	// Create a matrix to work with.
	dt = BLIS_DOUBLE;
	m = 5; n = 5; rs = 0; cs = 0;
	bli_obj_create( dt, m, n, rs, cs, &h );

	// Initialize the entire matrix to -1.0 to simulate junk values.
	bli_setm( &BLIS_MINUS_ONE, &h );

	bli_printm( "h: initial value (all -1.0)", &h, "% 4.3f", "" );

	// Set the diagonal offset of 'h' to -1.
	bli_obj_set_diag_offset( -1, &h );

	// Set the structure and uplo of 'h'.
	bli_obj_set_struc( BLIS_TRIANGULAR, &h );
	bli_obj_set_uplo( BLIS_UPPER, &h );

	// Randomize the elements on and above the first subdiagonal.
	bli_randm( &h );

	bli_printm( "h: after randomizing above first subdiagonal", &h, "% 4.3f", "" );

	// Create an alias to work with.
	bli_obj_alias_to( &h, &hl );

	// Flip the uplo of 'hl' and move the diagonal down by one.
	bli_obj_set_uplo( BLIS_LOWER, &hl );
	bli_obj_set_diag_offset( -2, &hl );

	// Set the region strictly below the first subdiagonal (on or below
	// the second subdiagonal) to zero.
	bli_setm( &BLIS_ZERO, &hl );

	bli_printm( "h: after setting elements below first subdiagonal to zero", &h, "% 4.3f", "" );


	// Free the objects.
	bli_obj_free( &a );
	bli_obj_free( &b );
	bli_obj_free( &c );
	bli_obj_free( &d );
	bli_obj_free( &e );
	bli_obj_free( &h );

	return 0;
}

// -----------------------------------------------------------------------------

