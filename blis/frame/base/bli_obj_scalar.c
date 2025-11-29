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


void bli_obj_scalar_init_detached
     (
       num_t  dt,
       obj_t* beta
     )
{
	// Initialize beta without a buffer and then attach its internal buffer.
	// NOTE: This initializes both the storage datatype and scalar datatype
	// bitfields within beta to dt.
	bli_obj_create_without_buffer( dt, 1, 1, beta );

	// Query the address of the object's internal scalar buffer.
	void* p = bli_obj_internal_scalar_buffer( beta );

	// Update the object.
	bli_obj_set_buffer( p, beta );
	bli_obj_set_strides( 1, 1, beta );
	bli_obj_set_imag_stride( 1, beta );
}

void bli_obj_scalar_init_detached_copy_of
     (
             num_t  dt,
             conj_t conj,
       const obj_t* alpha,
             obj_t* beta
     )
{
	obj_t alpha_local;

	// Make a local copy of alpha so we can apply the conj parameter.
	bli_obj_alias_to( alpha, &alpha_local );
	bli_obj_apply_conj( conj, &alpha_local );

	// Initialize beta without a buffer and then attach its internal buffer.
	bli_obj_scalar_init_detached( dt, beta );

	// Copy the scalar value in a to object b, conjugating and/or
	// typecasting if needed.
	bli_copysc( &alpha_local, beta );
}

void bli_obj_scalar_detach
     (
       const obj_t* a,
             obj_t* alpha
     )
{
	// Use the scalar datatype of A as the storage datatype of the detached
	// object alpha.
	num_t dt_a = bli_obj_scalar_dt( a );

	// Initialize alpha to be a bufferless internal scalar of the same
	// datatype as the scalar attached to A.
	bli_obj_scalar_init_detached( dt_a, alpha );

	// Copy the internal scalar in A to alpha.
	// NOTE: This is simply a field-to-field copy with no typecasting. But
	// that's okay since bli_obj_scalar_init_detached() initializes the
	// storage datatype of alpha to be the same as the datatype of the
	// scalar queried from bli_obj_scalar_dt() above.
	bli_obj_copy_internal_scalar( a, alpha );
}

void bli_obj_scalar_attach
     (
             conj_t conj,
       const obj_t* alpha,
             obj_t* a
     )
{
	obj_t alpha_cast;

	// Use the target datatype of A as the datatype to which we cast
	// alpha locally.
	const num_t dt = bli_obj_dt( alpha );

	// Make a copy-cast of alpha to the target datatype of A, queried
	// above. This step gives us the opportunity to conjugate and/or
	// typecast alpha.
	bli_obj_scalar_init_detached_copy_of( dt,
	                                      conj,
	                                      alpha,
	                                      &alpha_cast );

	// Copy the internal scalar in alpha_cast to A.
	bli_obj_copy_internal_scalar( &alpha_cast, a );

	// Update the scalar datatype of A.
	bli_obj_set_scalar_dt( dt, a );
}

void bli_obj_scalar_cast_to
     (
       num_t  dt,
       obj_t* a
     )
{
	obj_t alpha;
	obj_t alpha_cast;

	// Initialize an object alpha to be a bufferless scalar whose
	// storage datatype is equal to the scalar datatype of A.
	bli_obj_scalar_init_detached( bli_obj_scalar_dt( a ), &alpha );

	// Copy the internal scalar in A to alpha.
	// NOTE: Since alpha was initialized with the scalar datatype of A,
	// a simple field-to-field copy is sufficient (no casting is needed
	// here).
	bli_obj_copy_internal_scalar( a, &alpha );

	// Make a copy-cast of alpha, alpha_cast, with the datatype given by
	// the caller. (This is where the typecasting happens.)
	bli_obj_scalar_init_detached_copy_of( dt,
	                                      BLIS_NO_CONJUGATE,
	                                      &alpha,
	                                      &alpha_cast );

	// Copy the newly-typecasted value in alpha_cast back to A.
	bli_obj_copy_internal_scalar( &alpha_cast, a );

	// Update the scalar datatype of A to reflect to new datatype used
	// in the typecast.
	bli_obj_set_scalar_dt( dt, a );
}

void bli_obj_scalar_apply_scalar
     (
       const obj_t* alpha,
             obj_t* a
     )
{
	obj_t alpha_cast;
	obj_t scalar_a;

	// Make a copy of alpha, alpha_cast, with the same datatype as the
	// scalar datatype of A. (This is where the typecasting happens.)
	bli_obj_scalar_init_detached_copy_of( bli_obj_scalar_dt( a ),
	                                      BLIS_NO_CONJUGATE,
	                                      alpha,
	                                      &alpha_cast );
	// Detach the scalar from A.
	bli_obj_scalar_detach( a, &scalar_a );

	// Scale the detached scalar by alpha.
	bli_mulsc( &alpha_cast, &scalar_a );

	// Copy the internal scalar in scalar_a to A.
	bli_obj_copy_internal_scalar( &scalar_a, a );
}

void bli_obj_scalar_reset
     (
       obj_t* a
     )
{
	num_t       dt       = bli_obj_scalar_dt( a );
	void*       scalar_a = bli_obj_internal_scalar_buffer( a );
	const void* one      = bli_obj_buffer_for_const( dt, &BLIS_ONE );

	if      ( bli_is_float( dt )    ) *(( float*    )scalar_a) = *(( float*    )one);
	else if ( bli_is_double( dt )   ) *(( double*   )scalar_a) = *(( double*   )one);
	else if ( bli_is_scomplex( dt ) ) *(( scomplex* )scalar_a) = *(( scomplex* )one);
	else if ( bli_is_dcomplex( dt ) ) *(( dcomplex* )scalar_a) = *(( dcomplex* )one);

	// Alternate implementation:
	//bli_obj_scalar_attach( BLIS_NO_CONJUGATE, &BLIS_ONE, a );
}

bool bli_obj_scalar_has_nonzero_imag
     (
       obj_t* a
     )
{
	bool  r_val    = FALSE;
	num_t dt       = bli_obj_scalar_dt( a );
	void* scalar_a = bli_obj_internal_scalar_buffer( a );

	// FGVZ: Reimplement by using bli_obj_imag_part() and then
	// bli_obj_equals( &BLIS_ZERO, ... ).

	if      ( bli_is_real( dt ) )
	{
		r_val = FALSE;
	}
	else if ( bli_is_scomplex( dt ) )
	{
		r_val = ( bli_cimag( *(( scomplex* )scalar_a) ) != 0.0F );
	}
	else if ( bli_is_dcomplex( dt ) )
	{
		r_val = ( bli_zimag( *(( dcomplex* )scalar_a) ) != 0.0  );
	}

	return r_val;
}

bool bli_obj_scalar_equals
     (
       const obj_t* a,
       const obj_t* beta
     )
{
	obj_t scalar_a;

	bli_obj_scalar_detach( a, &scalar_a );

	bool r_val = bli_obj_equals( &scalar_a, beta );

	return r_val;
}

