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

#define FUNCPTR_T gemv_fp

typedef void (*FUNCPTR_T)( obj_t*  alpha,
                           obj_t*  a,
                           obj_t*  x,
                           obj_t*  beta,
                           obj_t*  y,
                           cntx_t* cntx,
                           gemv_t* cntl );

static FUNCPTR_T vars[3][3] =
{
	// unblocked          unblocked with fusing   blocked
	{ bli_gemv_unb_var1,  bli_gemv_unf_var1,      bli_gemv_blk_var1 },
	{ bli_gemv_unb_var2,  bli_gemv_unf_var2,      bli_gemv_blk_var2 },
	{ NULL,               NULL,                   NULL              },
};

void bli_gemv_int( trans_t transa,
                   conj_t  conjx,
                   obj_t*  alpha,
                   obj_t*  a,
                   obj_t*  x,
                   obj_t*  beta,
                   obj_t*  y,
                   cntx_t* cntx,
                   gemv_t* cntl )
{
	varnum_t  n;
	impl_t    i;
	FUNCPTR_T f;
	obj_t     a_local;
	obj_t     x_local;

	// Apply the trans and/or conj parameters to aliases of the objects.
	bli_obj_alias_with_trans( transa, a, &a_local );
	bli_obj_alias_with_conj( conjx, x, &x_local );

	// Check parameters. We use the aliased copy of A so the transa parameter
	// is taken into account for dimension checking.
	if ( bli_error_checking_is_enabled() )
		bli_gemv_check( alpha, &a_local, &x_local, beta, y );

	// If y has a zero dimension, return early.
	if ( bli_obj_has_zero_dim( y ) ) return;

	// If x has a zero dimension, scale y by beta and return early.
	if ( bli_obj_has_zero_dim( x ) )
	{
		bli_scalm( beta, y );
		return;
	}

	// Extract the variant number and implementation type.
	n = bli_cntl_var_num( cntl );
	i = bli_cntl_impl_type( cntl );

	// Index into the variant array to extract the correct function pointer.
	f = vars[n][i];

	// Invoke the variant.
	f( alpha,
	   &a_local,
	   &x_local,
	   beta,
	   y,
	   cntx,
	   cntl );
}

