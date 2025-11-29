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

#define FUNCPTR_T her_fp

typedef void (*FUNCPTR_T)( conj_t  conjh,
                           obj_t*  alpha,
                           obj_t*  x,
                           obj_t*  c,
                           cntx_t* cntx,
                           her_t*  cntl );

static FUNCPTR_T vars[4][3] =
{
	// unblocked          unblocked with fusing   blocked
	{ bli_her_unb_var1,   NULL,                   bli_her_blk_var1, },
	{ bli_her_unb_var2,   NULL,                   bli_her_blk_var2, },
	{ NULL,               NULL,                   NULL,             },
	{ NULL,               NULL,                   NULL,             },
};

void bli_her_int( conj_t  conjh,
                  obj_t*  alpha,
                  obj_t*  x,
                  obj_t*  c,
                  cntx_t* cntx,
                  her_t*  cntl )
{
	varnum_t  n;
	impl_t    i;
	FUNCPTR_T f;
	obj_t     x_local;
	obj_t     c_local;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
	{
		if ( bli_is_conj( conjh ) ) bli_her_check( alpha, x, c );
		else                        bli_syr_check( alpha, x, c );
	}

	// If C or x has a zero dimension, return early.
	if ( bli_obj_has_zero_dim( c ) ) return;
	if ( bli_obj_has_zero_dim( x ) ) return;

	// Alias the operands in case we need to apply conjugations.
	bli_obj_alias_to( x, &x_local );
	bli_obj_alias_to( c, &c_local );

	// If matrix C is marked for conjugation, we interpret this as a request
	// to apply a conjugation to the other operands.
	if ( bli_obj_has_conj( &c_local ) )
	{
		bli_obj_toggle_conj( &c_local );

		// Notice that we don't need to conjugate alpha since it is guaranteed
		// to be real.
		bli_obj_toggle_conj( &x_local );
	}

	// Extract the variant number and implementation type.
	n = bli_cntl_var_num( cntl );
	i = bli_cntl_impl_type( cntl );

	// Index into the variant array to extract the correct function pointer.
	f = vars[n][i];

	// Invoke the variant.
	f( conjh,
	   alpha,
	   &x_local,
	   &c_local,
	   cntx,
	   cntl );
}

