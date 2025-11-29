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

	obj_t a, x, y, b;
	obj_t* alpha;
	obj_t* beta;


	//
	// This file demonstrates level-2 operations.
	//


	//
	// Example 1: Perform a general rank-1 update (ger) operation.
	//

	printf( "\n#\n#  -- Example 1 --\n#\n\n" );

	// Create some matrix and vector operands to work with.
	dt = BLIS_DOUBLE;
	m = 4; n = 5; rs = 0; cs = 0;
	bli_obj_create( dt, m, n, rs, cs, &a );
	bli_obj_create( dt, m, 1, rs, cs, &x );
	bli_obj_create( dt, 1, n, rs, cs, &y );

	// Set alpha.
	alpha = &BLIS_ONE;

	// Initialize vectors 'x' and 'y'.
	bli_randv( &x );
	bli_setv( &BLIS_MINUS_ONE, &y );

	// Initialize 'a' to 1.0.
	bli_setm( &BLIS_ONE, &a );

	bli_printm( "x: set to random values", &x, "% 4.3f", "" );
	bli_printm( "y: set to -1.0", &y, "% 4.3f", "" );
	bli_printm( "a: initial value", &a, "% 4.3f", "" );

	// a := a + alpha * x * y, where 'a' is general.
	bli_ger( alpha, &x, &y, &a );

	bli_printm( "a: after ger", &a, "% 4.3f", "" );

	// Free the objects.
	bli_obj_free( &a );
	bli_obj_free( &x );
	bli_obj_free( &y );


	//
	// Example 2: Perform a general matrix-vector multiply (gemv) operation.
	//

	printf( "\n#\n#  -- Example 2 --\n#\n\n" );

	// Create some matrix and vector operands to work with.
	dt = BLIS_DOUBLE;
	m = 4; n = 5; rs = 0; cs = 0;
	bli_obj_create( dt, m, n, rs, cs, &a );
	bli_obj_create( dt, 1, n, rs, cs, &x );
	bli_obj_create( dt, 1, m, rs, cs, &y );

	// Notice that we created vectors 'x' and 'y' as row vectors, even though
	// we often think of them as column vectors so that the overall problem
	// dimensions remain conformal. Note that this flexibility only comes
	// from the fact that the operation requires those operands to be vectors.
	// If we were instead looking at an operation where the operands were of
	// general shape (such as with the gemm operation), then typically the
	// dimensions matter, and column vectors would not be interchangeable with
	// row vectors and vice versa.

	// Set the scalars to use.
	alpha = &BLIS_ONE;
	beta  = &BLIS_ONE;

	// Initialize vectors 'x' and 'y'.
	bli_setv( &BLIS_ONE,  &x );
	bli_setv( &BLIS_ZERO, &y );

	// Randomize 'a'.
	bli_randm( &a );

	bli_printm( "a: randomized", &a, "% 4.3f", "" );
	bli_printm( "x: set to 1.0", &x, "% 4.3f", "" );
	bli_printm( "y: initial value", &y, "% 4.3f", "" );

	// y := beta * y + alpha * a * x, where 'a' is general.
	bli_gemv( alpha, &a, &x, beta, &y );

	bli_printm( "y: after gemv", &y, "% 4.3f", "" );

	// Free the objects.
	bli_obj_free( &a );
	bli_obj_free( &x );
	bli_obj_free( &y );


	//
	// Example 3: Perform a symmetric rank-1 update (syr) operation.
	//

	printf( "\n#\n#  -- Example 3 --\n#\n\n" );

	// Create some matrix and vector operands to work with.
	dt = BLIS_DOUBLE;
	m = 5; rs = 0; cs = 0;
	bli_obj_create( dt, m, m, rs, cs, &a );
	bli_obj_create( dt, 1, m, rs, cs, &x );

	// Set alpha.
	alpha = &BLIS_ONE;

	// Initialize vector 'x'.
	bli_randv( &x );

	// Zero out all of matrix 'a'. This is optional, but will avoid possibly
	// displaying junk values in the unstored triangle.
	bli_setm( &BLIS_ZERO, &a );

	// Mark matrix 'a' as symmetric and stored in the lower triangle, and
	// then randomize that lower triangle.
	bli_obj_set_struc( BLIS_SYMMETRIC, &a );
	bli_obj_set_uplo( BLIS_LOWER, &a );
	bli_randm( &a );

	bli_printm( "x: set to random values", &x, "% 4.3f", "" );
	bli_printm( "a: initial value (zeros in upper triangle)", &a, "% 4.3f", "" );

	// a := a + alpha * x * x^T, where 'a' is symmetric and lower-stored.
	bli_syr( alpha, &x, &a );

	bli_printm( "a: after syr", &a, "% 4.3f", "" );

	// Free the objects.
	bli_obj_free( &a );
	bli_obj_free( &x );


	//
	// Example 4: Perform a symmetric matrix-vector multiply (symv) operation.
	//

	printf( "\n#\n#  -- Example 4 --\n#\n\n" );

	// Create some matrix and vector operands to work with.
	dt = BLIS_DOUBLE;
	m = 5; rs = 0; cs = 0;
	bli_obj_create( dt, m, m, rs, cs, &a );
	bli_obj_create( dt, 1, m, rs, cs, &x );
	bli_obj_create( dt, 1, m, rs, cs, &y );

	// Set the scalars to use.
	alpha = &BLIS_ONE;
	beta  = &BLIS_ONE;

	// Initialize vectors 'x' and 'y'.
	bli_setv( &BLIS_ONE,  &x );
	bli_setv( &BLIS_ZERO, &y );

	// Zero out all of matrix 'a'. This is optional, but will avoid possibly
	// displaying junk values in the unstored triangle.
	bli_setm( &BLIS_ZERO, &a );

	// Mark matrix 'a' as symmetric and stored in the upper triangle, and
	// then randomize that upper triangle.
	bli_obj_set_struc( BLIS_SYMMETRIC, &a );
	bli_obj_set_uplo( BLIS_UPPER, &a );
	bli_randm( &a );

	bli_printm( "a: randomized (zeros in lower triangle)", &a, "% 4.3f", "" );
	bli_printm( "x: set to 1.0", &x, "% 4.3f", "" );
	bli_printm( "y: initial value", &y, "% 4.3f", "" );

	// y := beta * y + alpha * a * x, where 'a' is symmetric and upper-stored.
	bli_symv( alpha, &a, &x, beta, &y );

	bli_printm( "y: after symv", &y, "% 4.3f", "" );

	// Free the objects.
	bli_obj_free( &a );
	bli_obj_free( &x );
	bli_obj_free( &y );


	//
	// Example 5: Perform a triangular matrix-vector multiply (trmv) operation.
	//

	printf( "\n#\n#  -- Example 5 --\n#\n\n" );

	// Create some matrix and vector operands to work with.
	dt = BLIS_DOUBLE;
	m = 5; rs = 0; cs = 0;
	bli_obj_create( dt, m, m, rs, cs, &a );
	bli_obj_create( dt, 1, m, rs, cs, &x );

	// Set the scalars to use.
	alpha = &BLIS_ONE;

	// Initialize vector 'x'.
	bli_setv( &BLIS_ONE, &x );

	// Zero out all of matrix 'a'. This is optional, but will avoid possibly
	// displaying junk values in the unstored triangle.
	bli_setm( &BLIS_ZERO, &a );

	// Mark matrix 'a' as triangular, stored in the lower triangle, and
	// having a non-unit diagonal. Then randomize that lower triangle.
	bli_obj_set_struc( BLIS_TRIANGULAR, &a );
	bli_obj_set_uplo( BLIS_LOWER, &a );
	bli_obj_set_diag( BLIS_NONUNIT_DIAG, &a );
	bli_randm( &a );

	bli_printm( "a: randomized (zeros in upper triangle)", &a, "% 4.3f", "" );
	bli_printm( "x: initial value", &x, "% 4.3f", "" );

	// x := alpha * a * x, where 'a' is triangular and lower-stored.
	bli_trmv( alpha, &a, &x );

	bli_printm( "x: after trmv", &x, "% 4.3f", "" );

	// Free the objects.
	bli_obj_free( &a );
	bli_obj_free( &x );


	//
	// Example 6: Perform a triangular solve (trsv) operation.
	//

	printf( "\n#\n#  -- Example 6 --\n#\n\n" );

	// Create some matrix and vector operands to work with.
	dt = BLIS_DOUBLE;
	m = 5; rs = 0; cs = 0;
	bli_obj_create( dt, m, m, rs, cs, &a );
	bli_obj_create( dt, 1, m, rs, cs, &b );
	bli_obj_create( dt, 1, m, rs, cs, &y );

	// Set the scalars to use.
	alpha = &BLIS_ONE;

	// Initialize vector 'x'.
	bli_setv( &BLIS_ONE, &b );

	// Zero out all of matrix 'a'. This is optional, but will avoid possibly
	// displaying junk values in the unstored triangle.
	bli_setm( &BLIS_ZERO, &a );

	// Mark matrix 'a' as triangular, stored in the lower triangle, and
	// having a non-unit diagonal. Then randomize that lower triangle.
	bli_obj_set_struc( BLIS_TRIANGULAR, &a );
	bli_obj_set_uplo( BLIS_LOWER, &a );
	bli_obj_set_diag( BLIS_NONUNIT_DIAG, &a );
	bli_randm( &a );

	// Load the diagonal. By setting the diagonal to something of greater
	// absolute value than the off-diagonal elements, we increase the odds
	// that the matrix is not singular (singular matrices have no inverse).
	bli_shiftd( &BLIS_TWO, &a );

	bli_printm( "a: randomized (zeros in upper triangle)", &a, "% 4.3f", "" );
	bli_printm( "b: initial value", &b, "% 4.3f", "" );

	// solve a * x = alpha * b, where 'a' is triangular and lower-stored, and
	// overwrite b with the solution vector x.
	bli_trsv( alpha, &a, &b );

	bli_printm( "b: after trsv", &b, "% 4.3f", "" );

	// We can confirm the solution by comparing the product of a and x to the
	// original value of b.
	bli_copyv( &b, &y );
	bli_trmv( alpha, &a, &y );

	bli_printm( "y: should equal initial value of b", &y, "% 4.3f", "" );

	// Free the objects.
	bli_obj_free( &a );
	bli_obj_free( &b );


	return 0;
}

// -----------------------------------------------------------------------------

