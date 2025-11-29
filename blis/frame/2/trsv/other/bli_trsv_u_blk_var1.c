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

void bli_trsv_u_blk_var1( obj_t*  alpha,
                          obj_t*  a,
                          obj_t*  x,
                          cntx_t* cntx,
                          trsv_t* cntl )
{
	obj_t   a11, a11_pack;
	obj_t   a12;
	obj_t   x1, x1_pack;
	obj_t   x2;

	dim_t   mn;
	dim_t   ij;
	dim_t   b_alg;

	// Initialize objects for packing.
	bli_obj_init_pack( &a11_pack );
	bli_obj_init_pack( &x1_pack );

	// Query dimension.
	mn = bli_obj_length( a );

	// x = alpha * x;
	bli_scalv_int( alpha,
	               x,
	               cntx, bli_cntl_sub_scalv( cntl ) );

	// Partition diagonally.
	for ( ij = 0; ij < mn; ij += b_alg )
	{
		// Determine the current algorithmic blocksize.
		b_alg = bli_determine_blocksize_b( ij, mn, a,
		                                   bli_cntl_bszid( cntl ), cntx );

		// Acquire partitions for A11, A12, x1, and x2.
		bli_acquire_mpart_br2tl( BLIS_SUBPART11,
		                         ij, b_alg, a, &a11 );
		bli_acquire_mpart_br2tl( BLIS_SUBPART12,
		                         ij, b_alg, a, &a12 );
		bli_acquire_vpart_b2f( BLIS_SUBPART1,
		                       ij, b_alg, x, &x1 );
		bli_acquire_vpart_b2f( BLIS_SUBPART2,
		                       ij, b_alg, x, &x2 );

		// Initialize objects for packing A11 and x1 (if needed).
		bli_packm_init( &a11, &a11_pack,
		                cntx, bli_cntl_sub_packm_a11( cntl ) );
		bli_packv_init( &x1, &x1_pack,
		                cntx, bli_cntl_sub_packv_x1( cntl ) );

		// Copy/pack A11, x1 (if needed).
		bli_packm_int( &a11, &a11_pack,
		               cntx, bli_cntl_sub_packm_a11( cntl ),
                       &BLIS_PACKM_SINGLE_THREADED );
		bli_packv_int( &x1, &x1_pack,
		               cntx, bli_cntl_sub_packv_x1( cntl ) );

		// x1 = x1 - A12 * x2;
		bli_gemv_int( BLIS_NO_TRANSPOSE,
		              BLIS_NO_CONJUGATE,
	                  &BLIS_MINUS_ONE,
		              &a12,
		              &x2,
		              &BLIS_ONE,
		              &x1_pack,
		              cntx,
		              bli_cntl_sub_gemv_rp( cntl ) );

		// x1 = x1 / tril( A11 );
		bli_trsv_int( &BLIS_ONE,
		              &a11_pack,
		              &x1_pack,
		              cntx,
		              bli_cntl_sub_trsv( cntl ) );

		// Copy/unpack x1 (if x1 was packed).
		bli_unpackv_int( &x1_pack, &x1,
		                 cntx, bli_cntl_sub_unpackv_x1( cntl ) );
	}

	// If any packing buffers were acquired within packm, release them back
	// to the memory manager.
	bli_packm_release( &a11_pack, bli_cntl_sub_packm_a11( cntl ) );
	bli_packv_release( &x1_pack, bli_cntl_sub_packv_x1( cntl ) );
}

