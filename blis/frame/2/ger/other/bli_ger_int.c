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

#define FUNCPTR_T ger_fp

typedef void (*FUNCPTR_T)( obj_t*  alpha,
                           obj_t*  x,
                           obj_t*  y,
                           obj_t*  a,
                           cntx_t* cntx,
                           ger_t*  cntl );

static FUNCPTR_T vars[4][3] =
{
	// unblocked          unblocked with fusing   blocked
	{ bli_ger_unb_var1,   NULL,                   bli_ger_blk_var1, },
	{ bli_ger_unb_var2,   NULL,                   bli_ger_blk_var2, },
	{ NULL,               NULL,                   NULL,             },
	{ NULL,               NULL,                   NULL,             },
};

void bli_ger_int( conj_t  conjx,
                  conj_t  conjy,
                  obj_t*  alpha,
                  obj_t*  x,
                  obj_t*  y,
                  obj_t*  a,
                  cntx_t* cntx,
                  ger_t*  cntl )
{
	varnum_t  n;
	impl_t    i;
	FUNCPTR_T f;
	obj_t     alpha_local;
	obj_t     x_local;
	obj_t     y_local;
	obj_t     a_local;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_ger_check( alpha, x, y, a );

	// If A has a zero dimension, return early.
	if ( bli_obj_has_zero_dim( a ) ) return;

	// If x or y has a zero dimension, return early.
	if ( bli_obj_has_zero_dim( x ) ||
	     bli_obj_has_zero_dim( y ) ) return;

	// Alias the objects, applying conjx and conjy to x and y, respectively.
	bli_obj_alias_with_conj( conjx, x, &x_local );
	bli_obj_alias_with_conj( conjy, y, &y_local );
	bli_obj_alias_to( a, &a_local );

	// If matrix A is marked for conjugation, we interpret this as a request
	// to apply a conjugation to the other operands.
	if ( bli_obj_has_conj( &a_local ) )
	{
		bli_obj_toggle_conj( &a_local );

		bli_obj_toggle_conj( &x_local );
		bli_obj_toggle_conj( &y_local );

		bli_obj_scalar_init_detached_copy_of( bli_obj_dt( alpha ),
		                                      BLIS_CONJUGATE,
		                                      alpha,
		                                      &alpha_local );
	}
	else
	{
		bli_obj_alias_to( *alpha, alpha_local );
	}

	// If we are about the call a leaf-level implementation, and matrix A
	// still needs a transposition, then we must induce one by swapping the
	// strides and dimensions.
	if ( bli_cntl_is_leaf( cntl ) && bli_obj_has_trans( &a_local ) )
	{
		bli_obj_induce_trans( &a_local );
		bli_obj_set_onlytrans( BLIS_NO_TRANSPOSE, &a_local );
	}

	// Extract the variant number and implementation type.
	n = bli_cntl_var_num( cntl );
	i = bli_cntl_impl_type( cntl );

	// Index into the variant array to extract the correct function pointer.
	f = vars[n][i];

	// Invoke the variant.
	f( &alpha_local,
	   &x_local,
	   &y_local,
	   &a_local,
	   cntx,
	   cntl );
}

