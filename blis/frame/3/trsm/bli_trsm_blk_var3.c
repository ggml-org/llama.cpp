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

void bli_trsm_blk_var3
     (
       const obj_t*     a,
       const obj_t*     b,
       const obj_t*     c,
       const cntx_t*    cntx,
       const cntl_t*    cntl,
             thrinfo_t* thread_par
     )
{
	obj_t ap, bp, cs;
	bli_obj_alias_to( a, &ap );
	bli_obj_alias_to( b, &bp );
	bli_obj_alias_to( c, &cs );

	thrinfo_t* thread = bli_thrinfo_sub_node( 0, thread_par );

	// Determine the direction in which to partition (forwards or backwards).
	const dir_t direct = bli_part_cntl_direct( cntl );

	// Prune any zero region that exists along the partitioning dimension.
	bli_l3_prune_unref_mparts_k( &ap, &bp, &cs );

	// Query dimension in partitioning direction.
	dim_t k_trans = bli_obj_width_after_trans( &ap );

	// Partition along the k dimension.
	dim_t b_alg;
	for ( dim_t i = 0; i < k_trans; i += b_alg )
	{
		// Determine the current algorithmic blocksize.
		b_alg = bli_determine_blocksize( direct, i, k_trans,
		                                 bli_part_cntl_blksz_alg( cntl ),
		                                 bli_part_cntl_blksz_max( cntl ) );

		// Acquire partitions for A1 and B1.
		obj_t a1, b1;
		bli_acquire_mpart_ndim( direct, BLIS_SUBPART1,
		                        i, b_alg, &ap, &a1 );
		bli_acquire_mpart_mdim( direct, BLIS_SUBPART1,
		                        i, b_alg, &bp, &b1 );

		// Perform trsm subproblem.
		bli_l3_int
		(
		  &a1,
		  &b1,
		  &cs,
		  cntx,
		  bli_cntl_sub_node( 0, cntl ),
		  thread
		);

		// This variant executes multiple rank-k updates. Therefore, if the
		// internal alpha scalars on A/B and C are non-zero, we must ensure
		// that they are only used in the first iteration.
		if ( i == 0 )
		{
			bli_obj_scalar_reset( &ap );
			bli_obj_scalar_reset( &bp );
			bli_obj_scalar_reset( &cs );
		}
	}
}

