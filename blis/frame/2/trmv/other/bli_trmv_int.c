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

#define FUNCPTR_T trmv_fp

typedef void (*FUNCPTR_T)( obj_t*  alpha,
                           obj_t*  a,
                           obj_t*  x,
                           cntx_t* cntx,
                           trmv_t* cntl );

static FUNCPTR_T vars[2][3][3] =
{
	// lower triangular
	{
		// unblocked         unblocked with fusing  blocked
		{ bli_trmv_unb_var1, bli_trmv_unf_var1,     bli_trmv_l_blk_var1 },
		{ bli_trmv_unb_var2, bli_trmv_unf_var2,     bli_trmv_l_blk_var2 },
		{ NULL,              NULL,                  NULL                },
	},
	// upper triangular
	{
		// unblocked         unblocked with fusing  blocked
		{ bli_trmv_unb_var1, bli_trmv_unf_var1,     bli_trmv_u_blk_var1 },
		{ bli_trmv_unb_var2, bli_trmv_unf_var2,     bli_trmv_u_blk_var2 },
		{ NULL,              NULL,                  NULL                },
	}
};

void bli_trmv_int( obj_t*  alpha,
                   obj_t*  a,
                   obj_t*  x,
                   cntx_t* cntx,
                   trmv_t* cntl )
{
	varnum_t  n;
	impl_t    i;
	bool      uplo;
	FUNCPTR_T f;
	obj_t     a_local;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_trmv_check( alpha, a, x );

	// If A or x has a zero dimension, return early.
	if ( bli_obj_has_zero_dim( a ) ) return;
	if ( bli_obj_has_zero_dim( x ) ) return;

	// Alias A in case we need to induce a transformation (ie: transposition).
	bli_obj_alias_to( a, &a_local );

	// NOTE: to support cases where B is complex and A is real, we will
	// need to have the default side case be BLIS_RIGHT and then express
	// the left case in terms of it, rather than the other way around.

	// Determine uplo (for indexing to the correct function pointer).
	if ( bli_obj_is_lower( &a_local ) ) uplo = 0;
	else                               uplo = 1;

	// We do not explicitly implement the cases where A is transposed.
	// However, we can still handle them. Specifically, if A is marked as
	// needing a transposition, we simply toggle the uplo value to cause the
	// correct algorithm to be induced. When that algorithm partitions into
	// A, it will grab the correct subpartitions, which will inherit A's
	// transposition bit and thus downstream subproblems will do the right
	// thing. Alternatively, we could accomplish the same end goal by
	// inducing a transposition, via bli_obj_induce_trans(), in the code
	// block below. That macro function swaps dimensions, strides, and
	// offsets. As an example, given a lower triangular, column-major matrix
	// that needs a transpose, we would induce that transposition by recasting
	// the object as an upper triangular, row-major matrix (with no transpose
	// needed). Note that how we choose to handle transposition here does NOT
	// affect the optimal choice of kernel (ie: a column-major column panel
	// matrix with transpose times a vector would use the same kernel as a
	// row-major row panel matrix with no transpose times a vector).
	if ( bli_obj_has_trans( &a_local ) )
	{
		//bli_obj_induce_trans( &a_local );
		//bli_obj_set_onlytrans( BLIS_NO_TRANSPOSE, &a_local );
		if ( uplo == 1 ) uplo = 0;
		else             uplo = 1;
	}

	// Extract the variant number and implementation type.
	n = bli_cntl_var_num( cntl );
	i = bli_cntl_impl_type( cntl );

	// Index into the variant array to extract the correct function pointer.
	f = vars[uplo][n][i];

	// Invoke the variant.
	f( alpha,
	   &a_local,
	   x,
	   cntx,
	   cntl );
}

