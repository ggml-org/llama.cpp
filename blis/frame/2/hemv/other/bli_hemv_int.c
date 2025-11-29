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

#define FUNCPTR_T hemv_fp

typedef void (*FUNCPTR_T)( conj_t  conjh,
                           obj_t*  alpha,
                           obj_t*  a,
                           obj_t*  x,
                           obj_t*  beta,
                           obj_t*  y,
                           cntx_t* cntx,
                           hemv_t* cntl );

static FUNCPTR_T vars[4][3] =
{
	// unblocked          unblocked with fusing   blocked
	{ bli_hemv_unb_var1,  bli_hemv_unf_var1,      bli_hemv_blk_var1, },
	{ bli_hemv_unb_var2,  NULL,                   bli_hemv_blk_var2, },
	{ bli_hemv_unb_var3,  bli_hemv_unf_var3,      bli_hemv_blk_var3, },
	{ bli_hemv_unb_var4,  NULL,                   bli_hemv_blk_var4, },
};

void bli_hemv_int( conj_t  conjh,
                   obj_t*  alpha,
                   obj_t*  a,
                   obj_t*  x,
                   obj_t*  beta,
                   obj_t*  y,
                   cntx_t* cntx,
                   hemv_t* cntl )
{
	varnum_t  n;
	impl_t    i;
	FUNCPTR_T f;
	obj_t     a_local;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
	{
		if ( bli_is_conj( conjh ) ) bli_hemv_check( alpha, a, x, beta, y );
		else                        bli_symv_check( alpha, a, x, beta, y );
	}

	// If y has a zero dimension, return early.
	if ( bli_obj_has_zero_dim( y ) ) return;

	// If x has a zero dimension, scale y by beta and return early.
	if ( bli_obj_has_zero_dim( x ) )
	{
		bli_scalm( beta, y );
		return;
	}

	// Alias A in case we need to induce the upper triangular case.
	bli_obj_alias_to( a, &a_local );

/*
	// Our blocked algorithms only [explicitly] implement the lower triangular
	// case, so if matrix A is stored as upper triangular, we must toggle the
	// transposition (and conjugation) bits so that the diagonal partitioning
	// routines grab the correct partitions corresponding to the upper
	// triangular case. But we only need to do this for blocked algorithms,
	// since unblocked algorithms are responsible for handling the upper case
	// explicitly (and they should not be inspecting the transposition bit anyway).
	if ( bli_cntl_is_blocked( cntl ) && bli_obj_is_upper( a ) )
	{
		bli_obj_toggle_conj( &a_local );
		bli_obj_toggle_trans( &a_local );
	}
*/

	// Extract the variant number and implementation type.
	n = bli_cntl_var_num( cntl );
	i = bli_cntl_impl_type( cntl );

	// Index into the variant array to extract the correct function pointer.
	f = vars[n][i];

	// Invoke the variant.
	f( conjh,
	   alpha,
	   &a_local,
	   x,
	   beta,
	   y,
	   cntx,
	   cntl );
}

