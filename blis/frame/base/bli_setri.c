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

// -- setr ---------------------------------------------------------------------

void bli_setrm
     (
       const obj_t* alpha,
       const obj_t* b
     )
{
	obj_t alpha_real;
	obj_t br;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_setm_check( alpha, b );

	// Initialize a local scalar, alpha_real, using the real projection
	// of the datatype of b.
	bli_obj_scalar_init_detached( bli_obj_dt_proj_to_real( b ),
	                              &alpha_real );

	// Copy/typecast alpha to alpha_real. This discards the imaginary
	// part of alpha (if it is complex).
	bli_copysc( alpha, &alpha_real );

	// Acquire an alias to the real part of b.
	bli_obj_real_part( b, &br );

	// Use setm to set the real part of b to alpha_real.
	bli_setm( &alpha_real, &br );
}

void bli_setrv
     (
       const obj_t* alpha,
       const obj_t* x
     )
{
	obj_t alpha_real;
	obj_t xr;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_setv_check( alpha, x );

	// Initialize a local scalar, alpha_real, using the real projection
	// of the datatype of x.
	bli_obj_scalar_init_detached( bli_obj_dt_proj_to_real( x ),
	                              &alpha_real );

	// Copy/typecast alpha to alpha_real. This discards the imaginary
	// part of alpha (if it is complex).
	bli_copysc( alpha, &alpha_real );

	// Acquire an alias to the real part of x.
	bli_obj_real_part( x, &xr );

	// Use setv to set the real part of x to alpha_real.
	bli_setv( &alpha_real, &xr );
}

// -- seti ---------------------------------------------------------------------

void bli_setim
     (
       const obj_t* alpha,
       const obj_t* b
     )
{
	obj_t alpha_real;
	obj_t bi;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_setm_check( alpha, b );

	// If the object is real, return early.
	if ( bli_obj_is_real( b ) ) return;

	// Initialize a local scalar, alpha_real, using the real projection
	// of the datatype of b.
	bli_obj_scalar_init_detached( bli_obj_dt_proj_to_real( b ),
	                              &alpha_real );

	// Copy/typecast alpha to alpha_real. This discards the imaginary
	// part of alpha (if it is complex).
	bli_copysc( alpha, &alpha_real );

	// Acquire an alias to the imaginary part of b.
	bli_obj_imag_part( b, &bi );

	// Use setm to set the imaginary part of b to alpha_real.
	bli_setm( &alpha_real, &bi );
}

void bli_setiv
     (
       const obj_t* alpha,
       const obj_t* x
     )
{
	obj_t alpha_real;
	obj_t xi;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_setv_check( alpha, x );

	// If the object is real, return early.
	if ( bli_obj_is_real( x ) ) return;

	// Initialize a local scalar, alpha_real, using the real projection
	// of the datatype of x.
	bli_obj_scalar_init_detached( bli_obj_dt_proj_to_real( x ),
	                              &alpha_real );

	// Copy/typecast alpha to alpha_real. This discards the imaginary
	// part of alpha (if it is complex).
	bli_copysc( alpha, &alpha_real );

	// Acquire an alias to the imaginary part of x.
	bli_obj_imag_part( x, &xi );

	// Use setm to set the imaginary part of x to alpha_real.
	bli_setm( &alpha_real, &xi );
}

