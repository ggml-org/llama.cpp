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

void bao_gemmd_check
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  d,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_noninteger_object( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_noninteger_object( beta );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( d );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( b );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( c );
	bli_check_error_code( e_val );

	// Check scalar/vector/matrix type.

	e_val = bli_check_scalar_object( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_scalar_object( beta );
	bli_check_error_code( e_val );

	e_val = bli_check_matrix_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( d );
	bli_check_error_code( e_val );

	e_val = bli_check_matrix_object( b );
	bli_check_error_code( e_val );

	e_val = bli_check_matrix_object( c );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( a );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( d );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( b );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( beta );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( c );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_level3_dims( a, b, c );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_dim_equals( d, bli_obj_width_after_trans( a ) );
	bli_check_error_code( e_val );

	// Check for consistent datatypes.
	// NOTE: We only perform these tests when mixed datatype support is
	// disabled.

	e_val = bli_check_consistent_object_datatypes( c, a );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_datatypes( c, d );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_datatypes( c, b );
	bli_check_error_code( e_val );
}

