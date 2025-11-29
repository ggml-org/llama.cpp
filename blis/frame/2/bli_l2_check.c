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

void bli_gemv_check
     (
       const obj_t* alpha,
       const obj_t* a,
       const obj_t* x,
       const obj_t* beta,
       const obj_t* y
     )
{
	err_t e_val;

	// Perform checks common to gemv/hemv/symv/trmv/trsv.

	bli_xxmv_check( alpha, a, x, beta, y );

	// Check object structure.

	e_val = bli_check_general_object( a );
	bli_check_error_code( e_val );

	// Check for consistent datatypes.

	e_val = bli_check_consistent_object_datatypes( a, x );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_datatypes( a, y );
	bli_check_error_code( e_val );
}


void bli_hemv_check
     (
       const obj_t* alpha,
       const obj_t* a,
       const obj_t* x,
       const obj_t* beta,
       const obj_t* y
     )
{
	err_t e_val;

	// Perform checks common to gemv/hemv/symv/trmv/trsv.

	bli_xxmv_check( alpha, a, x, beta, y );

	// Check squareness.

	e_val = bli_check_square_object( a );
	bli_check_error_code( e_val );

	// Check object structure.

	e_val = bli_check_hermitian_object( a );
	bli_check_error_code( e_val );

	// Check for consistent datatypes.

	e_val = bli_check_consistent_object_datatypes( a, x );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_datatypes( a, y );
	bli_check_error_code( e_val );
}


void bli_symv_check
     (
       const obj_t* alpha,
       const obj_t* a,
       const obj_t* x,
       const obj_t* beta,
       const obj_t* y
     )
{
	err_t e_val;

	// Perform checks common to gemv/hemv/symv/trmv/trsv.

	bli_xxmv_check( alpha, a, x, beta, y );

	// Check squareness.

	e_val = bli_check_square_object( a );
	bli_check_error_code( e_val );

	// Check object structure.

	e_val = bli_check_symmetric_object( a );
	bli_check_error_code( e_val );

	// Check for consistent datatypes.

	e_val = bli_check_consistent_object_datatypes( a, x );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_datatypes( a, y );
	bli_check_error_code( e_val );
}


void bli_trmv_check
     (
       const obj_t* alpha,
       const obj_t* a,
       const obj_t* x
     )
{
	err_t e_val;

	// Perform checks common to gemv/hemv/symv/trmv/trsv.

	bli_xxmv_check( alpha, a, x, alpha, x );

	// Check squareness.

	e_val = bli_check_square_object( a );
	bli_check_error_code( e_val );

	// Check object structure.

	e_val = bli_check_triangular_object( a );
	bli_check_error_code( e_val );

	// Check for consistent datatypes.

	e_val = bli_check_consistent_object_datatypes( a, x );
	bli_check_error_code( e_val );
}


void bli_trsv_check
     (
       const obj_t* alpha,
       const obj_t* a,
       const obj_t* x
     )
{
	err_t e_val;

	// Perform checks common to gemv/hemv/symv/trmv/trsv.

	bli_xxmv_check( alpha, a, x, alpha, x );

	// Check squareness.

	e_val = bli_check_square_object( a );
	bli_check_error_code( e_val );

	// Check object structure.

	e_val = bli_check_triangular_object( a );
	bli_check_error_code( e_val );

	// Check for consistent datatypes.

	e_val = bli_check_consistent_object_datatypes( a, x );
	bli_check_error_code( e_val );
}


void bli_ger_check
     (
       const obj_t* alpha,
       const obj_t* x,
       const obj_t* y,
       const obj_t* a
     )
{
	err_t e_val;

	// Perform checks common to ger/her/her2/syr/syr2.

	bli_xxr_check( alpha, x, y, a );

	// Check object structure.

	e_val = bli_check_general_object( a );
	bli_check_error_code( e_val );

	// Check for consistent datatypes.

	e_val = bli_check_consistent_object_datatypes( a, x );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_datatypes( a, y );
	bli_check_error_code( e_val );
}


void bli_her_check
     (
       const obj_t* alpha,
       const obj_t* x,
       const obj_t* a
     )
{
	err_t e_val;

	// Perform checks common to ger/her/her2/syr/syr2.

	bli_xxr_check( alpha, x, x, a );

	// Check squareness.

	e_val = bli_check_square_object( a );
	bli_check_error_code( e_val );

	// Check object structure.

	e_val = bli_check_hermitian_object( a );
	bli_check_error_code( e_val );

	// Check for consistent datatypes.

	e_val = bli_check_consistent_object_datatypes( a, x );
	bli_check_error_code( e_val );
}


void bli_her2_check
     (
       const obj_t* alpha,
       const obj_t* x,
       const obj_t* y,
       const obj_t* a
     )
{
	err_t e_val;

	// Perform checks common to ger/her/her2/syr/syr2.

	bli_xxr_check( alpha, x, y, a );

	// Check squareness.

	e_val = bli_check_square_object( a );
	bli_check_error_code( e_val );

	// Check object structure.

	e_val = bli_check_hermitian_object( a );
	bli_check_error_code( e_val );

	// Check for consistent datatypes.

	e_val = bli_check_consistent_object_datatypes( a, x );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_datatypes( a, y );
	bli_check_error_code( e_val );
}


void bli_syr_check
     (
       const obj_t* alpha,
       const obj_t* x,
       const obj_t* a
     )
{
	err_t e_val;

	// Perform checks common to ger/her/her2/syr/syr2.

	bli_xxr_check( alpha, x, x, a );

	// Check squareness.

	e_val = bli_check_square_object( a );
	bli_check_error_code( e_val );

	// Check object structure.

	e_val = bli_check_symmetric_object( a );
	bli_check_error_code( e_val );

	// Check for consistent datatypes.

	e_val = bli_check_consistent_object_datatypes( a, x );
	bli_check_error_code( e_val );
}


void bli_syr2_check
     (
       const obj_t* alpha,
       const obj_t* x,
       const obj_t* y,
       const obj_t* a
     )
{
	err_t e_val;

	// Perform checks common to ger/her/her2/syr/syr2.

	bli_xxr_check( alpha, x, y, a );

	// Check squareness.

	e_val = bli_check_square_object( a );
	bli_check_error_code( e_val );

	// Check object structure.

	e_val = bli_check_symmetric_object( a );
	bli_check_error_code( e_val );

	// Check for consistent datatypes.

	e_val = bli_check_consistent_object_datatypes( a, x );
	bli_check_error_code( e_val );

	e_val = bli_check_consistent_object_datatypes( a, y );
	bli_check_error_code( e_val );
}


// -----------------------------------------------------------------------------

void bli_xxmv_check
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

	e_val = bli_check_noninteger_object( beta );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( y );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_scalar_object( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_scalar_object( beta );
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

	e_val = bli_check_object_buffer( beta );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( y );
	bli_check_error_code( e_val );
}

void bli_xxr_check
     (
       const obj_t* alpha,
       const obj_t* x,
       const obj_t* y,
       const obj_t* a
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_noninteger_object( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( y );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( a );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_scalar_object( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( y );
	bli_check_error_code( e_val );

	e_val = bli_check_matrix_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_dim_equals( x, bli_obj_length_after_trans( a ) );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_dim_equals( y, bli_obj_width_after_trans( a ) );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( x );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( y );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( a );
	bli_check_error_code( e_val );
}

