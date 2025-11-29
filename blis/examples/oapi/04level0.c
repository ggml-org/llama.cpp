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
	obj_t alpha, beta, gamma, kappa, zeta;
	num_t dt;
	double gamma_d;


	//
	// This file demonstrates working with scalar objects.
	//


	//
	// Example 1: Create a scalar (1x1) object.
	//

	dt = BLIS_DOUBLE;

	// The easiest way to create a scalar object is with the following
	// convenience function.
	bli_obj_create_1x1( dt, &alpha );

	// We could, of course, create an object using our more general-purpose
	// function, using m = n = 1.
	bli_obj_create( dt, 1, 1, 0, 0, &beta );

	// We can even attach an external scalar. This function, unlike
	// bli_obj_create_1x1() and bli_obj_create(), does not result in any
	// memory allocation.
	bli_obj_create_1x1_with_attached_buffer( dt, &gamma_d, &gamma );

	// There is one more way to create an object. Like the previous method,
	// it also avoids memory allocation by referencing a special "internal"
	// scalar that is invisibly part of every object.
	bli_obj_scalar_init_detached( dt, &kappa );

	// Digression: In the most common cases, there is no need to create scalar
	// objects to begin with. That's because BLIS comes with three ready-to-use
	// globally-scoped scalar objects:
	//
	//   obj_t BLIS_MINUS_ONE;
	//   obj_t BLIS_ZERO;
	//   obj_t BLIS ONE;
	//
	// Each of these special objects is provided by blis.h. They can be used
	// wherever a scalar object is expected as an input operand regardless of
	// the datatype of your other operands. Note that you should never try to
	// modify these global scalar objects directly, nor should you ever try to
	// perform an operation *on* the objects (that is, you should never try to
	// update their values, though you can always perform operations *with*
	// them--that's the whole point!).


	//
	// Example 2: Set the value of an existing scalar object.
	//

	printf( "\n#\n#  -- Example 2 --\n#\n\n" );

	// Once you've created an object, you can set its value via setsc. As with
	// setijm, setsc takes a real and imaginary value, but you can ignore the
	// imaginary argument if your object is real. And even if you pass in a
	// non-zero value, it is ignored for real objects.
	bli_setsc( -4.0, 0.0, &alpha );
	bli_setsc(  3.0, 1.0, &beta );
	bli_setsc(  0.5, 0.0, &kappa );
	bli_setsc( 10.0, 0.0, &gamma );

	// BLIS does not have a special print function for scalars, but since a
	// 1x1 is also a vector and a matrix, we can use printv or printm.
	bli_printm( "alpha:", &alpha, "% 4.3f", "" );
	bli_printm( "beta:", &beta, "% 4.3f", "" );
	bli_printm( "kappa:", &kappa, "% 4.3f", "" );
	bli_printm( "gamma:", &gamma, "% 4.3f", "" );


	//
	// Example 3: Create and set the value of a complex scalar object.
	//

	printf( "\n#\n#  -- Example 3 --\n#\n\n" );

	// Create one more scalar, this time a complex scalar, to show how it
	// can be used.
	bli_obj_create_1x1( BLIS_DCOMPLEX, &zeta );
	bli_setsc( 3.3, -4.4, &zeta );
	bli_printm( "zeta (complex):", &zeta, "% 4.3f", "" );


	//
	// Example 4: Copy scalar objects.
	//

	printf( "\n#\n#  -- Example 4 --\n#\n\n" );

	// We can copy scalars amongst one another, and we can use the global
	// scalar constants for input operands.
	bli_copysc( &beta, &gamma );
	bli_printm( "gamma (overwritten with beta):", &gamma, "% 4.3f", "" );

	bli_copysc( &BLIS_ONE, &gamma );
	bli_printm( "gamma (overwritten with BLIS_ONE):", &gamma, "% 4.3f", "" );


	//
	// Example 5: Perform other operations on scalar objects.
	//

	printf( "\n#\n#  -- Example 5 --\n#\n\n" );

	// BLIS defines a range of basic floating-point operations on scalars.
	bli_addsc( &beta, &gamma );
	bli_printm( "gamma := gamma + beta", &gamma, "% 4.3f", "" );

	bli_subsc( &alpha, &gamma );
	bli_printm( "gamma := gamma - alpha", &gamma, "% 4.3f", "" );

	bli_divsc( &kappa, &gamma );
	bli_printm( "gamma := gamma / kappa", &gamma, "% 4.3f", "" );

	bli_sqrtsc( &gamma, &gamma );
	bli_printm( "gamma := sqrt( gamma )", &gamma, "% 4.3f", "" );

	bli_normfsc( &alpha, &alpha );
	bli_printm( "alpha := normf( alpha ) # normf() = abs() in real domain.", &alpha, "% 4.3f", "" );

	// Note that normfsc() allows complex input objects, but requires that the
	// output operand (the second operand) be a real object.
	bli_normfsc( &zeta, &alpha );
	bli_printm( "alpha := normf( zeta )  # normf() = complex modulus in complex domain.", &alpha, "% 4.3f", "" );

	bli_invertsc( &gamma, &gamma );
	bli_printm( "gamma := 1.0 / gamma", &gamma, "%4.2f", "" );


	// Only free the objects that resulted in actual allocation.
	bli_obj_free( &alpha );
	bli_obj_free( &beta );
	bli_obj_free( &zeta );

	return 0;
}

// -----------------------------------------------------------------------------

