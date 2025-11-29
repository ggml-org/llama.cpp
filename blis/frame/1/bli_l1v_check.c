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

#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* x, \
       const obj_t* y  \
     ) \
{ \
	bli_l1v_xy_check( x, y ); \
}

GENFRONT( addv )
GENFRONT( copyv )
GENFRONT( subv )
GENFRONT( swapv )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* x, \
       const obj_t* index  \
     ) \
{ \
	bli_l1v_xi_check( x, index ); \
}

GENFRONT( amaxv )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* alpha, \
       const obj_t* x, \
       const obj_t* beta, \
       const obj_t* y  \
     ) \
{ \
	bli_l1v_axby_check( alpha, x, beta, y ); \
}

GENFRONT( axpbyv )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* alpha, \
       const obj_t* x, \
       const obj_t* y  \
     ) \
{ \
	bli_l1v_axy_check( alpha, x, y ); \
}

GENFRONT( axpyv )
GENFRONT( scal2v )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* x, \
       const obj_t* y, \
       const obj_t* rho  \
     ) \
{ \
	bli_l1v_dot_check( &BLIS_ONE, x, y, &BLIS_ONE, rho ); \
}

GENFRONT( dotv )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* alpha, \
       const obj_t* x, \
       const obj_t* y, \
       const obj_t* beta, \
       const obj_t* rho  \
     ) \
{ \
	bli_l1v_dot_check( alpha, x, y, beta, rho ); \
}

GENFRONT( dotxv )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* x  \
     ) \
{ \
	bli_l1v_x_check( x ); \
}

GENFRONT( invertv )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* alpha, \
       const obj_t* x  \
     ) \
{ \
	bli_l1v_ax_check( alpha, x ); \
}

GENFRONT( invscalv )
GENFRONT( scalv )
GENFRONT( setv )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* x, \
       const obj_t* beta, \
       const obj_t* y  \
     ) \
{ \
	bli_l1v_xby_check( x, beta, y ); \
}

GENFRONT( xpbyv )


// -----------------------------------------------------------------------------

void bli_l1v_xy_check
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

	// Check for consistent datatypes.

	e_val = bli_check_consistent_object_datatypes( x, y );
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

void bli_l1v_axy_check
     (
       const obj_t* alpha,
       const obj_t* x,
       const obj_t* y
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

	// Check for consistent datatypes.

	e_val = bli_check_consistent_object_datatypes( x, y );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_scalar_object( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( y );
	bli_check_error_code( e_val );

	e_val = bli_check_equal_vector_lengths( x, y );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( x );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( y );
	bli_check_error_code( e_val );
}

void bli_l1v_xby_check
     (
       const obj_t* x,
       const obj_t* beta,
       const obj_t* y
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_noninteger_object( beta );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( y );
	bli_check_error_code( e_val );

	// Check for consistent datatypes.

	e_val = bli_check_consistent_object_datatypes( x, y );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_scalar_object( beta );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( y );
	bli_check_error_code( e_val );

	e_val = bli_check_equal_vector_lengths( x, y );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( beta );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( x );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( y );
	bli_check_error_code( e_val );
}

void bli_l1v_axby_check
     (
       const obj_t* alpha,
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

	e_val = bli_check_floating_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( y );
	bli_check_error_code( e_val );

	// Check for consistent datatypes.

	e_val = bli_check_consistent_object_datatypes( x, y );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_scalar_object( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_scalar_object( beta );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( y );
	bli_check_error_code( e_val );

	e_val = bli_check_equal_vector_lengths( x, y );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( beta );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( x );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( y );
	bli_check_error_code( e_val );
}

void bli_l1v_dot_check
     (
       const obj_t* alpha,
       const obj_t* x,
       const obj_t* y,
       const obj_t* beta,
       const obj_t* rho
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

	e_val = bli_check_noninteger_object( beta );
	bli_check_error_code( e_val );

	e_val = bli_check_noninteger_object( rho );
	bli_check_error_code( e_val );

	e_val = bli_check_nonconstant_object( rho );
	bli_check_error_code( e_val );

	// Check for consistent datatypes.

	e_val = bli_check_consistent_object_datatypes( x, y );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_scalar_object( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( y );
	bli_check_error_code( e_val );

	e_val = bli_check_scalar_object( beta );
	bli_check_error_code( e_val );

	e_val = bli_check_scalar_object( rho );
	bli_check_error_code( e_val );

	e_val = bli_check_equal_vector_lengths( x, y );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( x );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( y );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( beta );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( rho );
	bli_check_error_code( e_val );
}

void bli_l1v_x_check
     (
       const obj_t* x
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_floating_object( x );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_vector_object( x );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( x );
	bli_check_error_code( e_val );
}

void bli_l1v_ax_check
     (
       const obj_t* alpha,
       const obj_t* x
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_noninteger_object( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( x );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_scalar_object( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( x );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( alpha );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( x );
	bli_check_error_code( e_val );
}

void bli_l1v_xi_check
     (
       const obj_t* x,
       const obj_t* index
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_floating_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_integer_object( index );
	bli_check_error_code( e_val );

	e_val = bli_check_nonconstant_object( index );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_vector_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_scalar_object( index );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( x );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( index );
	bli_check_error_code( e_val );
}

