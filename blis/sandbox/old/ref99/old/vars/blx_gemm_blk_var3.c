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
#include "blix.h"

void blx_gemm_blk_var3
     (
       obj_t*  a,
       obj_t*  b,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm,
       cntl_t* cntl,
       thrinfo_t* thread
     )
{
	obj_t a1, b1;
	dim_t i;
	dim_t b_alg;
	dim_t k_trans;

	// Query dimension in partitioning direction.
	k_trans = bli_obj_width_after_trans( a );

	// Partition along the k dimension.
	for ( i = 0; i < k_trans; i += b_alg )
	{
		// Determine the current algorithmic blocksize.
		b_alg = blx_determine_blocksize_f( i, k_trans, c,
		                                   bli_cntl_bszid( cntl ), cntx );

		// Acquire partitions for A1 and B1.
		bli_acquire_mpart_ndim( BLIS_FWD, BLIS_SUBPART1, i, b_alg, a, &a1 );
		bli_acquire_mpart_mdim( BLIS_FWD, BLIS_SUBPART1, i, b_alg, b, &b1 );

		// Perform gemm subproblem.
		blx_gemm_int
		(
		  &a1, &b1, c, cntx, rntm,
		  bli_cntl_sub_node( cntl ),
		  bli_thrinfo_sub_node( thread )
		);

		bli_thrinfo_barrier( bli_thrinfo_sub_node( thread ) );

		// This variant executes multiple rank-k updates. Therefore, if the
		// internal beta scalar on matrix C is non-zero, we must use it
		// only for the first iteration (and then BLIS_ONE for all others).
		// And since c is a locally aliased obj_t, we can simply overwrite
		// the internal beta scalar with BLIS_ONE once it has been used in
		// the first iteration.
		if ( i == 0 ) bli_obj_scalar_reset( c );
	}
}

