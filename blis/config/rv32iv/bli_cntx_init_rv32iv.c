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

#include "../../kernels/rviv/3/bli_rviv_utils.h"

void bli_cntx_init_rv32iv( cntx_t* cntx )
{
	blksz_t blkszs[ BLIS_NUM_BLKSZS ];

	// Set default kernel blocksizes and functions.
	bli_cntx_init_rv32iv_ref( cntx );

	// -------------------------------------------------------------------------

	// A reasonable assumptions for application cores is VLEN >= 128 bits, i.e.,
	// v >= 4. Embedded cores, however, may implement the minimal configuration,
	// which allows VLEN = 32 bits. Here, we assume VLEN >= 128 and otherwise
	// fall back to the reference kernels.
	const uint32_t v = get_vlenb() / sizeof(float);

	if ( v >= 4 )
	{
		const uint32_t mr_s = 4 * v;
		const uint32_t mr_d = 2 * v;
		const uint32_t mr_c = 2 * v;
		const uint32_t mr_z = v;

		// Update the context with optimized native gemm micro-kernels.
		bli_cntx_set_ukrs
		(
		  cntx,

		  // level-3
		  BLIS_GEMM_UKR, BLIS_FLOAT,    bli_sgemm_rviv_4vx4,
		  BLIS_GEMM_UKR, BLIS_DOUBLE,   bli_dgemm_rviv_4vx4,
		  BLIS_GEMM_UKR, BLIS_SCOMPLEX, bli_cgemm_rviv_4vx4,
		  BLIS_GEMM_UKR, BLIS_DCOMPLEX, bli_zgemm_rviv_4vx4,

		  BLIS_VA_END
		);

		// Update the context with storage preferences.
		bli_cntx_set_ukr_prefs
		(
		  cntx,

		  // level-3
		  BLIS_GEMM_UKR_ROW_PREF, BLIS_FLOAT,    FALSE,
		  BLIS_GEMM_UKR_ROW_PREF, BLIS_DOUBLE,   FALSE,
		  BLIS_GEMM_UKR_ROW_PREF, BLIS_SCOMPLEX, FALSE,
		  BLIS_GEMM_UKR_ROW_PREF, BLIS_DCOMPLEX, FALSE,

		  BLIS_VA_END
		);

		// Initialize level-3 blocksize objects with architecture-specific values.
		//                                              s        d        c        z
		bli_blksz_init_easy( &blkszs[ BLIS_MR ],     mr_s,    mr_d,    mr_c,    mr_z );
		bli_blksz_init_easy( &blkszs[ BLIS_NR ],        4,       4,       4,       4 );
		bli_blksz_init_easy( &blkszs[ BLIS_MC ],  20*mr_s, 20*mr_d, 60*mr_c, 30*mr_z );
		bli_blksz_init_easy( &blkszs[ BLIS_KC ],      640,     320,     320,     160 );
		bli_blksz_init_easy( &blkszs[ BLIS_NC ],     3072,    3072,    3072,    3072 );

		bli_cntx_set_blkszs
		(
		  cntx,

		  // level-3
		  BLIS_NC, &blkszs[ BLIS_NC ], BLIS_NR,
		  BLIS_KC, &blkszs[ BLIS_KC ], BLIS_KR,
		  BLIS_MC, &blkszs[ BLIS_MC ], BLIS_MR,
		  BLIS_NR, &blkszs[ BLIS_NR ], BLIS_NR,
		  BLIS_MR, &blkszs[ BLIS_MR ], BLIS_MR,

		  BLIS_VA_END
		);
	}
}
