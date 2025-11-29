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

//
// Define object-based check functions.
//

void bli_axpy2v_check
     (
       const obj_t* alphax,
       const obj_t* alphay,
       const obj_t* x,
       const obj_t* y,
       const obj_t* z
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_noninteger_object( alphax );
	bli_check_error_code( e_val );

	e_val = bli_check_noninteger_object( alphay );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( y );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( z );
	bli_check_error_code( e_val );

	// Check for consistent datatypes.

	e_val = bli_check_consistent_object_datatypes( x, y );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_datatypes( x, z );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_scalar_object( alphax );
	bli_check_error_code( e_val );

	e_val = bli_check_scalar_object( alphay );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( y );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( z );
	bli_check_error_code( e_val );

	e_val = bli_check_equal_vector_lengths( x, y );
	bli_check_error_code( e_val );

	e_val = bli_check_equal_vector_lengths( x, z );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( alphax );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( alphay );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( x );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( y );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( z );
	bli_check_error_code( e_val );
}


void bli_axpyf_check
     (
       const obj_t* alpha,
       const obj_t* a,
       const obj_t* x,
       const obj_t* y
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_noninteger_object( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( y );
	bli_check_error_code( e_val );

	// Check for consistent datatypes.

	e_val = bli_check_consistent_object_datatypes( a, x );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_datatypes( a, y );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_scalar_object( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_matrix_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( y );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_dim_equals( x, bli_obj_width_after_trans( a ) );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_dim_equals( y, bli_obj_length_after_trans( a ) );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( a );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( x );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( y );
	bli_check_error_code( e_val );
}


void bli_dotaxpyv_check
     (
       const obj_t* alpha,
       const obj_t* xt,
       const obj_t* x,
       const obj_t* y,
       const obj_t* rho,
       const obj_t* z
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_noninteger_object( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( xt );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( y );
	bli_check_error_code( e_val );

	e_val = bli_check_noninteger_object( rho );
	bli_check_error_code( e_val );

	e_val = bli_check_nonconstant_object( rho );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( z );
	bli_check_error_code( e_val );

	// Check for consistent datatypes.

	e_val = bli_check_consistent_object_datatypes( x, xt );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_datatypes( x, y );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_datatypes( x, z );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_scalar_object( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( xt );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( y );
	bli_check_error_code( e_val );

	e_val = bli_check_scalar_object( rho );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( z );
	bli_check_error_code( e_val );

	e_val = bli_check_equal_vector_lengths( x, xt );
	bli_check_error_code( e_val );

	e_val = bli_check_equal_vector_lengths( x, y );
	bli_check_error_code( e_val );

	e_val = bli_check_equal_vector_lengths( x, z );
	bli_check_error_code( e_val );

	// Check object aliases.

	e_val = bli_check_object_alias_of( xt, x );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( xt );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( x );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( y );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( rho );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( z );
	bli_check_error_code( e_val );
}


void bli_dotxaxpyf_check
     (
       const obj_t* alpha,
       const obj_t* at,
       const obj_t* a,
       const obj_t* w,
       const obj_t* x,
       const obj_t* beta,
       const obj_t* y,
       const obj_t* z
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_noninteger_object( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( at );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( w );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_noninteger_object( beta );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( y );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( z );
	bli_check_error_code( e_val );

	// Check for consistent datatypes.

	e_val = bli_check_consistent_object_datatypes( a, at );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_datatypes( a, w );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_datatypes( a, x );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_datatypes( a, y );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_datatypes( a, z );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_scalar_object( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_matrix_object( at );
	bli_check_error_code( e_val );

	e_val = bli_check_matrix_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( w );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_scalar_object( beta );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( y );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( z );
	bli_check_error_code( e_val );

	e_val = bli_check_equal_vector_lengths( w, z );
	bli_check_error_code( e_val );

	e_val = bli_check_equal_vector_lengths( x, y );
	bli_check_error_code( e_val );

	e_val = bli_check_conformal_dims( at, a );
	bli_check_error_code( e_val );

	e_val = bli_check_object_length_equals( at, bli_obj_vector_dim( w ) );
	bli_check_error_code( e_val );

	e_val = bli_check_object_width_equals( at, bli_obj_vector_dim( y ) );
	bli_check_error_code( e_val );

	e_val = bli_check_object_length_equals( a, bli_obj_vector_dim( z ) );
	bli_check_error_code( e_val );

	e_val = bli_check_object_width_equals( a, bli_obj_vector_dim( x ) );
	bli_check_error_code( e_val );

	// Check object aliases.

	e_val = bli_check_object_alias_of( at, a );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( at );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( a );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( w );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( x );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( beta );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( y );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( z );
	bli_check_error_code( e_val );
}


void bli_dotxf_check
     (
       const obj_t* alpha,
       const obj_t* a,
       const obj_t* x,
       const obj_t* beta,
       const obj_t* y
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_noninteger_object( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_noninteger_object( beta );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( y );
	bli_check_error_code( e_val );

	// Check for consistent datatypes.

	e_val = bli_check_consistent_object_datatypes( a, x );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_datatypes( a, y );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_scalar_object( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_matrix_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_scalar_object( beta );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( y );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_dim_equals( x, bli_obj_length_after_trans( a ) );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_dim_equals( y, bli_obj_width_after_trans( a ) );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( a );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( x );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( beta );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( y );
	bli_check_error_code( e_val );
}

