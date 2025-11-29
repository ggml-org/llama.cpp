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

void bli_projm
     (
       const obj_t* a,
       const obj_t* b
     )
{
	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_projm_check( a, b );

	if ( ( bli_obj_is_real( a )    && bli_obj_is_real( b )    ) ||
	     ( bli_obj_is_complex( a ) && bli_obj_is_complex( b ) ) )
	{
		// If a and b are both real or both complex, we can simply use
		// copym.
		bli_copym( a, b );
	}
	else
	{
		// This branch handles the case where one operand is real and
		// the other is complex.

		if ( bli_obj_is_real( a ) /* && bli_obj_is_complex( b ) */ )
		{
			// If a is real and b is complex, we must obtain the real part
			// of b so that we can copy a into the real part (after
			// initializing all of b, including imaginary components, to
			// zero).

			obj_t br;

			bli_obj_real_part( b, &br );

			bli_setm( &BLIS_ZERO, b );
			bli_copym( a, &br );
		}
		else // bli_obj_is_complex( a ) && bli_obj_is_real( b )
		{
			// If a is complex and b is real, we can simply copy the
			// real part of a into b.

			obj_t ar;

			bli_obj_real_part( a, &ar );

			bli_copym( &ar, b );
		}
	}
}

// -----------------------------------------------------------------------------

void bli_projm_check
     (
       const obj_t* a,
       const obj_t* b
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_floating_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( b );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_precisions( a, b );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_matrix_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_matrix_object( b );
	bli_check_error_code( e_val );

	e_val = bli_check_conformal_dims( a, b );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( a );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( b );
	bli_check_error_code( e_val );
}

