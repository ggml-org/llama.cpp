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
	dim_t m, n, k;
	inc_t rs, cs;
	side_t side;

	obj_t a, b, c;
	obj_t* alpha;
	obj_t* beta;


	//
	// This file demonstrates level-3 operations.
	//


	//
	// Example 1: Perform a general matrix-matrix multiply (gemm) operation.
	//

	printf( "\n#\n#  -- Example 1 --\n#\n\n" );

	// Create some matrix operands to work with.
	dt = BLIS_DOUBLE;
	m = 4; n = 5; k = 3; rs = 0; cs = 0;
	bli_obj_create( dt, m, n, rs, cs, &c );
	bli_obj_create( dt, m, k, rs, cs, &a );
	bli_obj_create( dt, k, n, rs, cs, &b );

	// Set the scalars to use.
	alpha = &BLIS_ONE;
	beta  = &BLIS_ONE;

	// Initialize the matrix operands.
	bli_randm( &a );
	bli_setm( &BLIS_ONE, &b );
	bli_setm( &BLIS_ZERO, &c );

	bli_printm( "a: randomized", &a, "% 4.3f", "" );
	bli_printm( "b: set to 1.0", &b, "% 4.3f", "" );
	bli_printm( "c: initial value", &c, "% 4.3f", "" );

	// c := beta * c + alpha * a * b, where 'a', 'b', and 'c' are general.
	bli_gemm( alpha, &a, &b, beta, &c );

	bli_printm( "c: after gemm", &c, "% 4.3f", "" );

	// Free the objects.
	bli_obj_free( &a );
	bli_obj_free( &b );
	bli_obj_free( &c );


	//
	// Example 1b: Perform a general matrix-matrix multiply (gemm) operation
	//             with the left input operand (matrix A) transposed.
	//

	printf( "\n#\n#  -- Example 1b --\n#\n\n" );

	// Create some matrix operands to work with.
	dt = BLIS_DOUBLE;
	m = 4; n = 5; k = 3; rs = 0; cs = 0;
	bli_obj_create( dt, m, n, rs, cs, &c );
	bli_obj_create( dt, k, m, rs, cs, &a );
	bli_obj_create( dt, k, n, rs, cs, &b );

	// Set the scalars to use.
	alpha = &BLIS_ONE;
	beta  = &BLIS_ONE;

	// Initialize the matrix operands.
	bli_randm( &a );
	bli_setm( &BLIS_ONE, &b );
	bli_setm( &BLIS_ZERO, &c );

	// Set the transpose bit in 'a'.
	bli_obj_toggle_trans( &a );

	bli_printm( "a: randomized", &a, "% 4.3f", "" );
	bli_printm( "b: set to 1.0", &b, "% 4.3f", "" );
	bli_printm( "c: initial value", &c, "% 4.3f", "" );

	// c := beta * c + alpha * a^T * b, where 'a', 'b', and 'c' are general.
	bli_gemm( alpha, &a, &b, beta, &c );

	bli_printm( "c: after gemm", &c, "% 4.3f", "" );

	// Free the objects.
	bli_obj_free( &a );
	bli_obj_free( &b );
	bli_obj_free( &c );


	//
	// Example 2: Perform a symmetric rank-k update (syrk) operation.
	//

	printf( "\n#\n#  -- Example 2 --\n#\n\n" );

	// Create some matrix and vector operands to work with.
	dt = BLIS_DOUBLE;
	m = 5; k = 3; rs = 0; cs = 0;
	bli_obj_create( dt, m, m, rs, cs, &c );
	bli_obj_create( dt, m, k, rs, cs, &a );

	// Set alpha.
	alpha = &BLIS_ONE;

	// Initialize matrix operands.
	bli_setm( &BLIS_ZERO, &c );
	bli_randm( &a );

	// Mark matrix 'c' as symmetric and stored in the lower triangle, and
	// then randomize that lower triangle.
	bli_obj_set_struc( BLIS_SYMMETRIC, &c );
	bli_obj_set_uplo( BLIS_LOWER, &c );
	bli_randm( &c );

	bli_printm( "a: set to random values", &a, "% 4.3f", "" );
	bli_printm( "c: initial value (zeros in upper triangle)", &c, "% 4.3f", "" );

	// c := c + alpha * a * a^T, where 'c' is symmetric and lower-stored.
	bli_syrk( alpha, &a, beta, &c );

	bli_printm( "c: after syrk", &c, "% 4.3f", "" );

	// Free the objects.
	bli_obj_free( &c );
	bli_obj_free( &a );


	//
	// Example 3: Perform a symmetric matrix-matrix multiply (symm) operation.
	//

	printf( "\n#\n#  -- Example 3 --\n#\n\n" );

	// Create some matrix and vector operands to work with.
	dt = BLIS_DOUBLE;
	m = 5; n = 6; rs = 0; cs = 0;
	bli_obj_create( dt, m, m, rs, cs, &a );
	bli_obj_create( dt, m, n, rs, cs, &b );
	bli_obj_create( dt, m, n, rs, cs, &c );

	// Set the scalars to use.
	alpha = &BLIS_ONE;
	beta  = &BLIS_ONE;

	// Set the side operand.
	side = BLIS_LEFT;

	// Initialize matrices 'b' and 'c'.
	bli_setm( &BLIS_ONE,  &b );
	bli_setm( &BLIS_ZERO, &c );

	// Zero out all of matrix 'a'. This is optional, but will avoid possibly
	// displaying junk values in the unstored triangle.
	bli_setm( &BLIS_ZERO, &a );

	// Mark matrix 'a' as symmetric and stored in the upper triangle, and
	// then randomize that upper triangle.
	bli_obj_set_struc( BLIS_SYMMETRIC, &a );
	bli_obj_set_uplo( BLIS_UPPER, &a );
	bli_randm( &a );

	bli_printm( "a: randomized (zeros in lower triangle)", &a, "% 4.3f", "" );
	bli_printm( "b: set to 1.0", &b, "% 4.3f", "" );
	bli_printm( "c: initial value", &c, "% 4.3f", "" );

	// c := beta * c + alpha * a * b, where 'a' is symmetric and upper-stored.
	// Note that the first 'side' operand indicates the side from which matrix
	// 'a' is multiplied into 'b'.
	bli_symm( side, alpha, &a, &b, beta, &c );

	bli_printm( "c: after symm", &c, "% 4.3f", "" );

	// Free the objects.
	bli_obj_free( &a );
	bli_obj_free( &b );
	bli_obj_free( &c );


	//
	// Example 4: Perform a triangular matrix-matrix multiply (trmm) operation.
	//

	printf( "\n#\n#  -- Example 4 --\n#\n\n" );

	// Create some matrix and vector operands to work with.
	dt = BLIS_DOUBLE;
	m = 5; n = 4; rs = 0; cs = 0;
	bli_obj_create( dt, m, m, rs, cs, &a );
	bli_obj_create( dt, m, n, rs, cs, &b );

	// Set the scalars to use.
	alpha = &BLIS_ONE;

	// Set the side operand.
	side = BLIS_LEFT;

	// Initialize matrix 'b'.
	bli_setm( &BLIS_ONE, &b );

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
	bli_printm( "b: initial value", &b, "% 4.3f", "" );

	// b := alpha * a * b, where 'a' is triangular and lower-stored.
	bli_trmm( side, alpha, &a, &b );

	bli_printm( "x: after trmm", &b, "% 4.3f", "" );

	// Free the objects.
	bli_obj_free( &a );
	bli_obj_free( &b );


	//
	// Example 5: Perform a triangular solve with multiple right-hand sides
	//            (trsm) operation.
	//

	printf( "\n#\n#  -- Example 5 --\n#\n\n" );

	// Create some matrix and vector operands to work with.
	dt = BLIS_DOUBLE;
	m = 5; n = 4; rs = 0; cs = 0;
	bli_obj_create( dt, m, m, rs, cs, &a );
	bli_obj_create( dt, m, n, rs, cs, &b );
	bli_obj_create( dt, m, n, rs, cs, &c );

	// Set the scalars to use.
	alpha = &BLIS_ONE;

	// Set the side operand.
	side = BLIS_LEFT;

	// Initialize matrix 'b'.
	bli_setm( &BLIS_ONE, &b );

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
	// overwrite b with the solution matrix x.
	bli_trsm( side, alpha, &a, &b );

	bli_printm( "b: after trsm", &b, "% 4.3f", "" );

	// We can confirm the solution by comparing the product of a and x to the
	// original value of b.
	bli_copym( &b, &c );
	bli_trmm( side, alpha, &a, &c );

	bli_printm( "c: should equal initial value of b", &c, "% 4.3f", "" );

	// Free the objects.
	bli_obj_free( &a );
	bli_obj_free( &b );
	bli_obj_free( &c );


	return 0;
}

// -----------------------------------------------------------------------------

