/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas

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
	num_t dt_r, dt_c;
	num_t dt_s, dt_d;
	num_t dt_a, dt_b;
	dim_t m, n, k;
	inc_t rs, cs;

	obj_t a, b, c;
	obj_t* alpha;
	obj_t* beta;

	//
	// This file demonstrates mixing datatypes in gemm.
	//
	// NOTE: Please make sure that mixed datatype support is enabled in BLIS
	// before proceeding to build and run the example binaries. If you're not
	// sure whether mixed datatype support is enabled in BLIS, please refer
	// to './configure --help' for the relevant options.
	//

	//
	// Example 1: Perform a general matrix-matrix multiply (gemm) operation
	//            with operands of different domains (but identical precisions).
	//

	printf( "\n#\n#  -- Example 1 --\n#\n\n" );

	// Create some matrix operands to work with.
	dt_r = BLIS_DOUBLE;
	dt_c = BLIS_DCOMPLEX;
	m = 4; n = 5; k = 1; rs = 0; cs = 0;
	bli_obj_create( dt_c, m, n, rs, cs, &c );
	bli_obj_create( dt_r, m, k, rs, cs, &a );
	bli_obj_create( dt_c, k, n, rs, cs, &b );

	// Set the scalars to use.
	alpha = &BLIS_ONE;
	beta  = &BLIS_ONE;

	// Initialize the matrix operands.
	bli_randm( &a );
	bli_randm( &b );
	bli_setm( &BLIS_ZERO, &c );

	bli_printm( "a (double real):    randomized", &a, "% 4.3f", "" );
	bli_printm( "b (double complex): randomized", &b, "% 4.3f", "" );
	bli_printm( "c (double complex): initial value", &c, "% 4.3f", "" );

	// c := beta * c + alpha * a * b, where 'a' is real, and 'b' and 'c' are
	// complex.
	bli_gemm( alpha, &a, &b, beta, &c );

	bli_printm( "c (double complex): after gemm", &c, "% 4.3f", "" );

	// Free the objects.
	bli_obj_free( &a );
	bli_obj_free( &b );
	bli_obj_free( &c );

	//
	// Example 2: Perform a general matrix-matrix multiply (gemm) operation
	//            with operands of different precisions (but identical domains).
	//

	printf( "\n#\n#  -- Example 2 --\n#\n\n" );

	// Create some matrix operands to work with.
	dt_s = BLIS_FLOAT;
	dt_d = BLIS_DOUBLE;
	m = 4; n = 5; k = 1; rs = 0; cs = 0;
	bli_obj_create( dt_d, m, n, rs, cs, &c );
	bli_obj_create( dt_s, m, k, rs, cs, &a );
	bli_obj_create( dt_s, k, n, rs, cs, &b );

	// Notice that we've chosen C to be double-precision real and A and B to be
	// single-precision real.

	// Since we are mixing precisions, we will also need to specify the
	// so-called "computation precision." That is, we need to signal to
	// bli_gemm() whether we want the A*B product to be computed in single
	// precision or double precision (prior to the result being accumulated
	// back to C). To specify the computation precision, we need to set the
	// corresponding bit in the C object. Here, we specify double-precision
	// computation.
	// NOTE: If you do not explicitly specify the computation precision, it
	// will default to the storage precision of the C object.
	bli_obj_set_comp_prec( BLIS_DOUBLE_PREC, &c );

	// Initialize the matrix operands.
	bli_randm( &a );
	bli_randm( &b );
	bli_setm( &BLIS_ZERO, &c );

	bli_printm( "a (single real): randomized", &a, "% 4.3f", "" );
	bli_printm( "b (single real): randomized", &b, "% 4.3f", "" );
	bli_printm( "c (double real): initial value", &c, "% 4.3f", "" );

	// c := beta * c + alpha * a * b, where 'a' and 'b' are single-precision
	// real, 'c' is double-precision real, and the matrix product is performed
	// in double-precision arithmetic.
	bli_gemm( alpha, &a, &b, beta, &c );

	bli_printm( "c (double real): after gemm (exec prec = double precision)", &c, "% 4.3f", "" );

	// Free the objects.
	bli_obj_free( &a );
	bli_obj_free( &b );
	bli_obj_free( &c );

	//
	// Example 3: Perform a general matrix-matrix multiply (gemm) operation
	//            with operands of different domains AND precisions.
	//

	printf( "\n#\n#  -- Example 3 --\n#\n\n" );

	// Create some matrix operands to work with.
	dt_a = BLIS_FLOAT;
	dt_b = BLIS_DCOMPLEX;
	dt_c = BLIS_SCOMPLEX;
	m = 4; n = 5; k = 1; rs = 0; cs = 0;
	bli_obj_create( dt_c, m, n, rs, cs, &c );
	bli_obj_create( dt_a, m, k, rs, cs, &a );
	bli_obj_create( dt_b, k, n, rs, cs, &b );

	// Notice that we've chosen C to be single-precision complex, and A to be
	// single-precision real, and B to be double-precision complex.

	// Set the computation precision to single precision this time.
	bli_obj_set_comp_prec( BLIS_SINGLE_PREC, &c );

	// Initialize the matrix operands.
	bli_randm( &a );
	bli_randm( &b );
	bli_setm( &BLIS_ZERO, &c );

	bli_printm( "a (single real): randomized", &a, "% 4.3f", "" );
	bli_printm( "b (double complex): randomized", &b, "% 4.3f", "" );
	bli_printm( "c (single complex): initial value", &c, "% 4.3f", "" );

	// c := beta * c + alpha * a * b, where 'a' is single-precision real, 'b'
	// is double-precision complex, 'c' is single-precision complex, and the
	// matrix product is performed in single-precision arithmetic.
	bli_gemm( alpha, &a, &b, beta, &c );

	bli_printm( "c (single complex): after gemm (exec prec = single precision)", &c, "% 4.3f", "" );

	// Free the objects.
	bli_obj_free( &a );
	bli_obj_free( &b );
	bli_obj_free( &c );

	//
	// Example 4: Project objects between the real and complex domains.
	//

	printf( "\n#\n#  -- Example 4 --\n#\n\n" );

	// Create some matrix operands to work with.
	dt_r = BLIS_DOUBLE;
	dt_c = BLIS_DCOMPLEX;
	m = 4; n = 5; rs = 0; cs = 0;
	bli_obj_create( dt_r, m, n, rs, cs, &a );
	bli_obj_create( dt_c, m, n, rs, cs, &b );

	// Initialize a real matrix A.
	bli_randm( &a );

	bli_printm( "a (double real): randomized", &a, "% 4.3f", "" );

	// Project real matrix A to the complex domain (in B).
	bli_projm( &a, &b );

	bli_printm( "b (double complex): projected from 'a'", &b, "% 4.3f", "" );

	// Notice how the imaginary components in B are zero since any real
	// matrix implicitly has imaginary values that are equal to zero.

	// Now let's project in the other direction.

	// Initialize the complex matrix B.
	bli_randm( &b );

	bli_printm( "b (double complex): randomized", &b, "% 4.3f", "" );

	// Project complex matrix B to the real domain (in A).
	bli_projm( &b, &a );

	bli_printm( "a (double real): projected from 'b'", &a, "% 4.3f", "" );

	// Notice how the imaginary components are lost in the projection from
	// the complex domain to the real domain.

	// Free the objects.
	bli_obj_free( &a );
	bli_obj_free( &b );

	//
	// Example 5: Typecast objects between the single and double precisions.
	//

	printf( "\n#\n#  -- Example 5 --\n#\n\n" );

	// Create some matrix operands to work with.
	dt_s = BLIS_FLOAT;
	dt_d = BLIS_DOUBLE;
	m = 4; n = 3; rs = 0; cs = 0;
	bli_obj_create( dt_d, m, n, rs, cs, &a );
	bli_obj_create( dt_s, m, n, rs, cs, &b );

	// Initialize a double-precision real matrix A.
	bli_randm( &a );

	bli_printm( "a (double real): randomized", &a, "%23.16e", "" );

	// Typecast A to single precision.
	bli_castm( &a, &b );

	bli_printm( "b (single real): typecast from 'a'", &b, "%23.16e", "" );

	// Notice how the values in B are only accurate to the 6th or 7th decimal
	// place relative to the true values in A.

	// Free the objects.
	bli_obj_free( &a );
	bli_obj_free( &b );


	return 0;
}

