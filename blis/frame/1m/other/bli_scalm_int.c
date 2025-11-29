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

#define FUNCPTR_T scalm_fp

typedef void (*FUNCPTR_T)( obj_t*  alpha,
                           obj_t*  x,
                           cntx_t* cntx );

static FUNCPTR_T vars[1][3] =
{
	// unblocked          optimized unblocked    blocked
	{ bli_scalm_ex,       bli_scalm_ex,          NULL }
};

void bli_scalm_int( obj_t*   alpha,
                    obj_t*   x,
                    cntx_t*  cntx,
                    scalm_t* cntl )
{
	//obj_t     x_local;
	varnum_t  n;
	impl_t    i;
	FUNCPTR_T f;

	// Return early if one of the matrix operands has a zero dimension.
	if ( bli_obj_has_zero_dim( x ) ) return;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_scalm_check( alpha, x );

	// First check if we are to skip this operation.
	if ( bli_cntl_is_noop( cntl ) ) return;

	// Return early if both alpha and the scalar attached to x are unit.
	if ( bli_obj_equals( alpha, &BLIS_ONE ) &&
	     bli_obj_scalar_equals( x, &BLIS_ONE ) ) return;

	//
	// This code has been disabled since we've now added the alpha
	// parameter back to the object interface to the underlying
	// scalm variant.
	//
	// Alias x to x_local so we can apply alpha if it is non-unit.
	//bli_obj_alias_to( *x, x_local );

	// If alpha is non-unit, apply it to the scalar attached to x.
	//if ( !bli_obj_equals( alpha, &BLIS_ONE ) )
	//{
	//	bli_obj_scalar_apply_scalar( alpha, &x_local );
	//}

	// Extract the variant number and implementation type.
	n = bli_cntl_var_num( cntl );
	i = bli_cntl_impl_type( cntl );

	// Index into the variant array to extract the correct function pointer.
	f = vars[n][i];

	// Invoke the variant.
	f( alpha,
	   x,
	   cntx );
}

