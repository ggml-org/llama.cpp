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
    - Neither the name(s) of the copyright holder(s) nor the names of its
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

#include "blis.h"

void bli_projv
     (
       const obj_t* x,
       const obj_t* y
     )
{
	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_projv_check( x, y );

	if ( ( bli_obj_is_real( x )    && bli_obj_is_real( y )    ) ||
	     ( bli_obj_is_complex( x ) && bli_obj_is_complex( y ) ) )
	{
		// If x and y are both real or both complex, we can simply use
		// copyv.
		bli_copyv( x, y );
	}
	else
	{
		// This branch handles the case where one operand is real and
		// the other is complex.

		if ( bli_obj_is_real( x ) /* && bli_obj_is_complex( y ) */ )
		{
			// If x is real and y is complex, we must obtain the real part
			// of y so that we can copy x into the real part (after
			// initializing all of y, including imaginary components, to
			// zero).

			obj_t yr;

			bli_obj_real_part( y, &yr );

			bli_setv( &BLIS_ZERO, y );
			bli_copyv( x, &yr );
		}
		else // bli_obj_is_complex( x ) && bli_obj_is_real( y )
		{
			// If x is complex and y is real, we can simply copy the
			// real part of x into y.

			obj_t xr;

			bli_obj_real_part( x, &xr );

			bli_copyv( &xr, y );
		}
	}
}

// -----------------------------------------------------------------------------

void bli_projv_check
     (
       const obj_t* x,
       const obj_t* y
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_floating_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( y );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_precisions( x, y );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_vector_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( y );
	bli_check_error_code( e_val );

	e_val = bli_check_equal_vector_lengths( x, y );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( x );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( y );
	bli_check_error_code( e_val );
}

