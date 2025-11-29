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
       const obj_t* chi, \
       const obj_t* psi  \
     ) \
{ \
	bli_l0_xxsc_check( chi, psi ); \
}

GENFRONT( addsc )
GENFRONT( copysc )
GENFRONT( divsc )
GENFRONT( mulsc )
GENFRONT( sqrtsc )
GENFRONT( sqrtrsc )
GENFRONT( subsc )
GENFRONT( invertsc )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* chi, \
       const obj_t* norm  \
     ) \
{ \
	bli_l0_xx2sc_check( chi, norm ); \
}

GENFRONT( absqsc )
GENFRONT( normfsc )

// -----------------------------------------------------------------------------

void bli_getsc_check
     (
       const obj_t*  chi,
       const double* zeta_r,
       const double* zeta_i
     )
{
	err_t e_val;

	// Check object datatypes.

	//e_val = bli_check_noninteger_object( chi );
	//bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_scalar_object( chi );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( chi );
	bli_check_error_code( e_val );
}


void bli_setsc_check
     (
       double       zeta_r,
       double       zeta_i,
       const obj_t* chi
     )
{
	err_t e_val;

	// Check object datatypes.

	//e_val = bli_check_floating_object( chi );
	//bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_scalar_object( chi );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( chi );
	bli_check_error_code( e_val );
}


void bli_unzipsc_check
     (
       const obj_t* chi,
       const obj_t* zeta_r,
       const obj_t* zeta_i
     )
{
	err_t e_val;

	// Check object datatypes.

    e_val = bli_check_noninteger_object( chi );
    bli_check_error_code( e_val );

    e_val = bli_check_real_object( zeta_r );
    bli_check_error_code( e_val );

    e_val = bli_check_real_object( zeta_i );
    bli_check_error_code( e_val );

    e_val = bli_check_nonconstant_object( zeta_r );
    bli_check_error_code( e_val );

    e_val = bli_check_nonconstant_object( zeta_i );
    bli_check_error_code( e_val );

    e_val = bli_check_object_real_proj_of( chi, zeta_r );
    bli_check_error_code( e_val );

    e_val = bli_check_object_real_proj_of( chi, zeta_i );
    bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_scalar_object( chi );
	bli_check_error_code( e_val );

	e_val = bli_check_scalar_object( zeta_r );
	bli_check_error_code( e_val );

	e_val = bli_check_scalar_object( zeta_i );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( chi );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( zeta_r );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( zeta_i );
	bli_check_error_code( e_val );
}


void bli_zipsc_check
     (
       const obj_t* zeta_r,
       const obj_t* zeta_i,
       const obj_t* chi
     )
{
	err_t e_val;

	// Check object datatypes.

    e_val = bli_check_real_object( zeta_r );
    bli_check_error_code( e_val );

    e_val = bli_check_real_object( zeta_i );
    bli_check_error_code( e_val );

    e_val = bli_check_noninteger_object( chi );
    bli_check_error_code( e_val );

    e_val = bli_check_nonconstant_object( chi );
    bli_check_error_code( e_val );

    e_val = bli_check_object_real_proj_of( chi, zeta_r );
    bli_check_error_code( e_val );

    e_val = bli_check_object_real_proj_of( chi, zeta_i );
    bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_scalar_object( zeta_r );
	bli_check_error_code( e_val );

	e_val = bli_check_scalar_object( zeta_i );
	bli_check_error_code( e_val );

	e_val = bli_check_scalar_object( chi );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( zeta_r );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( zeta_i );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( chi );
	bli_check_error_code( e_val );
}


// -----------------------------------------------------------------------------

void bli_l0_xsc_check
     (
       const obj_t* chi
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_noninteger_object( chi );
	bli_check_error_code( e_val );

	e_val = bli_check_nonconstant_object( chi );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_scalar_object( chi );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( chi );
	bli_check_error_code( e_val );
}

void bli_l0_xxsc_check
     (
       const obj_t* chi,
       const obj_t* psi
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_noninteger_object( chi );
	bli_check_error_code( e_val );

	e_val = bli_check_noninteger_object( psi );
	bli_check_error_code( e_val );

	e_val = bli_check_nonconstant_object( psi );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_scalar_object( chi );
	bli_check_error_code( e_val );

	e_val = bli_check_scalar_object( psi );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( chi );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( psi );
	bli_check_error_code( e_val );
}

void bli_l0_xx2sc_check
     (
       const obj_t* chi,
       const obj_t* absq
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_noninteger_object( chi );
	bli_check_error_code( e_val );

	e_val = bli_check_nonconstant_object( absq );
	bli_check_error_code( e_val );

	e_val = bli_check_real_object( absq );
	bli_check_error_code( e_val );

	e_val = bli_check_object_real_proj_of( chi, absq );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_scalar_object( chi );
	bli_check_error_code( e_val );

	e_val = bli_check_scalar_object( absq );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( chi );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( absq );
	bli_check_error_code( e_val );
}

void bli_l0_xxbsc_check
     (
       const obj_t* chi,
       const obj_t* psi,
       const bool*  is
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_noninteger_object( chi );
	bli_check_error_code( e_val );

	e_val = bli_check_noninteger_object( psi );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_scalar_object( chi );
	bli_check_error_code( e_val );

	e_val = bli_check_scalar_object( psi );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( chi );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( psi );
	bli_check_error_code( e_val );
}

