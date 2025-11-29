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

void bli_her2_blk_var1( conj_t  conjh,
                        obj_t*  alpha,
                        obj_t*  alpha_conj,
                        obj_t*  x,
                        obj_t*  y,
                        obj_t*  c,
                        cntx_t* cntx,
                        her2_t* cntl )
{
	obj_t   c11, c11_pack;
	obj_t   c10;
	obj_t   x1, x1_pack;
	obj_t   x0;
	obj_t   y1, y1_pack;
	obj_t   y0;

	dim_t   mn;
	dim_t   ij;
	dim_t   b_alg;

	// Even though this blocked algorithm is expressed only in terms of the
	// lower triangular case, the upper triangular case is still supported:
	// when bli_acquire_mpart_tl2br() is passed a matrix that is stored in
	// in the upper triangle, and the requested subpartition resides in the
	// lower triangle (as is the case for this algorithm), the routine fills
	// the request as if the caller had actually requested the corresponding
	// "mirror" subpartition in the upper triangle, except that it marks the
	// subpartition for transposition (and conjugation).

	// Initialize objects for packing.
	bli_obj_init_pack( &c11_pack );
	bli_obj_init_pack( &x1_pack );
	bli_obj_init_pack( &y1_pack );

	// Query dimension.
	mn = bli_obj_length( c );

	// Partition diagonally.
	for ( ij = 0; ij < mn; ij += b_alg )
	{
		// Determine the current algorithmic blocksize.
		b_alg = bli_determine_blocksize_f( ij, mn, c,
		                                   bli_cntl_bszid( cntl ), cntx );

		// Acquire partitions for C11, C10, x1, x0, y1, and y0.
		bli_acquire_mpart_tl2br( BLIS_SUBPART11,
		                         ij, b_alg, c, &c11 );
		bli_acquire_mpart_tl2br( BLIS_SUBPART10,
		                         ij, b_alg, c, &c10 );
		bli_acquire_vpart_f2b( BLIS_SUBPART1,
		                       ij, b_alg, x, &x1 );
		bli_acquire_vpart_f2b( BLIS_SUBPART0,
		                       ij, b_alg, x, &x0 );
		bli_acquire_vpart_f2b( BLIS_SUBPART1,
		                       ij, b_alg, y, &y1 );
		bli_acquire_vpart_f2b( BLIS_SUBPART0,
		                       ij, b_alg, y, &y0 );

		// Initialize objects for packing C11, x1, and y1 (if needed).
		bli_packm_init( &c11, &c11_pack,
		                cntx, bli_cntl_sub_packm_c11( cntl ) );
		bli_packv_init( &x1, &x1_pack,
		                cntx, bli_cntl_sub_packv_x1( cntl ) );
		bli_packv_init( &y1, &y1_pack,
		                cntx, bli_cntl_sub_packv_y1( cntl ) );

		// Copy/pack C11, x1, y1 (if needed).
		bli_packm_int( &c11, &c11_pack,
		               cntx, bli_cntl_sub_packm_c11( cntl ),
                       &BLIS_PACKM_SINGLE_THREADED );
		bli_packv_int( &x1, &x1_pack,
		               cntx, bli_cntl_sub_packv_x1( cntl ) );
		bli_packv_int( &y1, &y1_pack,
		               cntx, bli_cntl_sub_packv_y1( cntl ) );

		// C10 = C10 + alpha * x1 * y0';
		bli_ger_int( BLIS_NO_CONJUGATE,
		             conjh,
		             alpha,
		             &x1_pack,
		             &y0,
		             &c10,
		             cntx,
		             bli_cntl_sub_ger_rp( cntl ) );

		// C10 = C10 + conj(alpha) * y1 * x0';
		bli_ger_int( BLIS_NO_CONJUGATE,
		             conjh,
		             alpha_conj,
		             &y1_pack,
		             &x0,
		             &c10,
		             cntx,
		             bli_cntl_sub_ger_rp( cntl ) );

		// C11 = C11 + alpha * x1 * y1' + conj(alpha) * y1 * x1';
		bli_her2_int( conjh,
		              alpha,
		              alpha_conj,
		              &x1_pack,
		              &y1_pack,
		              &c11_pack,
		              cntx,
		              bli_cntl_sub_her2( cntl ) );

		// Copy/unpack C11 (if C11 was packed).
		bli_unpackm_int( &c11_pack, &c11,
		                 cntx, bli_cntl_sub_unpackm_c11( cntl ),
                         &BLIS_PACKM_SINGLE_THREADED );
	}

	// If any packing buffers were acquired within packm, release them back
	// to the memory manager.
	bli_packm_release( &c11_pack, bli_cntl_sub_packm_c11( cntl ) );
	bli_packv_release( &x1_pack, bli_cntl_sub_packv_x1( cntl ) );
	bli_packv_release( &y1_pack, bli_cntl_sub_packv_y1( cntl ) );
}

