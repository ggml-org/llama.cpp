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
       const obj_t* asum  \
     ) \
{ \
	bli_utilv_xa_check( x, asum ); \
}

GENFRONT( asumv )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* x  \
     ) \
{ \
	bli_utilm_mkhst_check( x ); \
}

GENFRONT( mkherm )
GENFRONT( mksymm )
GENFRONT( mktrim )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* x, \
       const obj_t* norm  \
     ) \
{ \
	bli_utilv_norm_check( x, norm ); \
}

GENFRONT( norm1v )
GENFRONT( normfv )
GENFRONT( normiv )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* x, \
       const obj_t* norm  \
     ) \
{ \
	bli_utilm_norm_check( x, norm ); \
}

GENFRONT( norm1m )
GENFRONT( normfm )
GENFRONT( normim )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* x  \
     ) \
{ \
	bli_utilm_rand_check( x ); \
}

GENFRONT( randv )
GENFRONT( randnv )
GENFRONT( randm )
GENFRONT( randnm )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* x, \
       const obj_t* scale, \
       const obj_t* sumsq  \
     ) \
{ \
	bli_utilv_sumsqv_check( x, scale, sumsq ); \
}

GENFRONT( sumsqv )

// -----------------------------------------------------------------------------

#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* chi, \
       const obj_t* psi, \
       const bool*  is  \
     ) \
{ \
	bli_l0_xxbsc_check( chi, psi, is ); \
}

GENFRONT( eqsc )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* chi, \
       const obj_t* psi, \
       const bool*  is  \
     ) \
{ \
	bli_l0_xxbsc_check( chi, psi, is ); \
\
	err_t e_val; \
\
	e_val = bli_check_real_datatype( bli_obj_dt( chi ) ); \
	bli_check_error_code( e_val ); \
\
	e_val = bli_check_real_datatype( bli_obj_dt( psi ) ); \
	bli_check_error_code( e_val ); \
}

GENFRONT( ltsc )
GENFRONT( lesc )
GENFRONT( gtsc )
GENFRONT( gesc )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* x, \
       const obj_t* y, \
       const bool*  is  \
     ) \
{ \
	bli_l1v_xy_check( x, y ); \
}

GENFRONT( eqv )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* x, \
       const obj_t* y, \
       const bool*  is  \
     ) \
{ \
	bli_l1m_xy_check( x, y ); \
}

GENFRONT( eqm )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const FILE*  file, \
       const char*  s1, \
       const obj_t* x, \
       const char*  format, \
       const char*  s2  \
     ) \
{ \
	bli_utilm_fprint_check( file, s1, x, format, s2 ); \
}

GENFRONT( fprintv )
GENFRONT( fprintm )

// -----------------------------------------------------------------------------

void bli_utilv_xa_check
     (
       const obj_t* x,
       const obj_t* asum
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_floating_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_nonconstant_object( asum );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_vector_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_scalar_object( asum );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( x );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( asum );
	bli_check_error_code( e_val );
}

void bli_utilm_mkhst_check
     (
       const obj_t* a
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_floating_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_nonconstant_object( a );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_matrix_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_square_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_object_diag_offset_equals( a, 0 );
	bli_check_error_code( e_val );

	// Check matrix storage.

	e_val = bli_check_upper_or_lower_object( a );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( a );
	bli_check_error_code( e_val );
}

void bli_utilv_norm_check
     (
       const obj_t* x,
       const obj_t* norm
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_floating_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_noninteger_object( norm );
	bli_check_error_code( e_val );

	e_val = bli_check_nonconstant_object( norm );
	bli_check_error_code( e_val );

	e_val = bli_check_object_real_proj_of( x, norm );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_vector_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_scalar_object( norm );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( x );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( norm );
	bli_check_error_code( e_val );
}


void bli_utilm_norm_check
     (
       const obj_t* x,
       const obj_t* norm
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_floating_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_noninteger_object( norm );
	bli_check_error_code( e_val );

	e_val = bli_check_nonconstant_object( norm );
	bli_check_error_code( e_val );

	e_val = bli_check_object_real_proj_of( x, norm );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_matrix_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_scalar_object( norm );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( x );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( norm );
	bli_check_error_code( e_val );
}

void bli_utilm_fprint_check
     (
       const FILE*  file,
       const char*  s1,
       const obj_t* x,
       const char*  format,
       const char*  s2
     )
{
	err_t e_val;

	// Check argument pointers.

	e_val = bli_check_null_pointer( file );
	bli_check_error_code( e_val );

	e_val = bli_check_null_pointer( s1 );
	bli_check_error_code( e_val );

	e_val = bli_check_null_pointer( s2 );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( x );
	bli_check_error_code( e_val );
}

void bli_utilm_rand_check
     (
       const obj_t* x
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_noninteger_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_nonconstant_object( x );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( x );
	bli_check_error_code( e_val );
}

void bli_utilv_sumsqv_check
     (
       const obj_t* x,
       const obj_t* scale,
       const obj_t* sumsq
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_floating_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_nonconstant_object( scale );
	bli_check_error_code( e_val );

	e_val = bli_check_nonconstant_object( sumsq );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_vector_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_scalar_object( scale );
	bli_check_error_code( e_val );

	e_val = bli_check_scalar_object( sumsq );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( x );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( scale );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( sumsq );
	bli_check_error_code( e_val );
}

