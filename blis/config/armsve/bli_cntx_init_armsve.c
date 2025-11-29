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

void bli_cntx_init_armsve( cntx_t* cntx )
{
	// Set default kernel blocksizes and functions.
	bli_cntx_init_armsve_ref( cntx );

	// If we are autodetecting the correct aarch64 config, then we have to make sure
	// that SVE instructions are actually available since these are used in determining
	// the register blocksizes.
	#ifdef BLIS_FAMILY_ARM64
	uint32_t family, model, features = 0;
	bli_cpuid_query( &family, &model, &features );

	if ( ! bli_cpuid_has_features( features, FEATURE_SVE ) )
		return;
	#endif

	blksz_t blkszs[ BLIS_NUM_BLKSZS ];

	// -------------------------------------------------------------------------

	// Block size.
	dim_t m_r_s, n_r_s, k_c_s, m_c_s, n_c_s;
	dim_t m_r_d, n_r_d, k_c_d, m_c_d, n_c_d;
	dim_t m_r_c, n_r_c, k_c_c, m_c_c, n_c_c;
	dim_t m_r_z, n_r_z, k_c_z, m_c_z, n_c_z;
	bli_s_blksz_armsve(&m_r_s, &n_r_s, &k_c_s, &m_c_s, &n_c_s);
	bli_d_blksz_armsve(&m_r_d, &n_r_d, &k_c_d, &m_c_d, &n_c_d);
	bli_c_blksz_armsve(&m_r_c, &n_r_c, &k_c_c, &m_c_c, &n_c_c);
	bli_z_blksz_armsve(&m_r_z, &n_r_z, &k_c_z, &m_c_z, &n_c_z);

	// Update the context with optimized native gemm micro-kernels.
	bli_cntx_set_ukrs
	(
	  cntx,

	  // level-3
	  // These are vector-length agnostic kernels. Yet knowing mr is required at runtime.
	  BLIS_GEMM_UKR, BLIS_FLOAT,    bli_sgemm_armsve_asm_2vx10_unindexed,
	  BLIS_GEMM_UKR, BLIS_DOUBLE,   bli_dgemm_armsve_asm_2vx10_unindexed,
	  BLIS_GEMM_UKR, BLIS_SCOMPLEX, bli_cgemm_armsve_asm_2vx10_unindexed,
	  BLIS_GEMM_UKR, BLIS_DCOMPLEX, bli_zgemm_armsve_asm_2vx10_unindexed,

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

	// Set VL-specific packing routines if applicable.
	if ( m_r_d == 16 )
	{
	  bli_cntx_set_ukrs
	  (
		cntx,
		BLIS_PACKM_KER, BLIS_DOUBLE, bli_dpackm_armsve512_asm_16x10,
		BLIS_VA_END
	  );
	}
	else if ( m_r_d == 8 )
	{
	  bli_cntx_set_ukrs
	  (
		cntx,
		BLIS_PACKM_KER, BLIS_DOUBLE, bli_dpackm_armsve256_int_8x10,
		BLIS_VA_END
	  );
	}

	// Initialize level-3 blocksize objects with architecture-specific values.
	//                                           s      d      c      z
	bli_blksz_init_easy( &blkszs[ BLIS_MR ], m_r_s, m_r_d, m_r_c, m_r_z );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ], n_r_s, n_r_d, n_r_c, n_r_z );
	bli_blksz_init_easy( &blkszs[ BLIS_MC ], m_c_s, m_c_d, m_c_c, m_c_z );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ], k_c_s, k_c_d, k_c_c, k_c_z );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ], n_c_s, n_c_d, n_c_c, n_c_z );

	// Update the context with the current architecture's register and cache
	// blocksizes (and multiples) for native execution.
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

