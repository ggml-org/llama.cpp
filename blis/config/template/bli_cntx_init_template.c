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

void bli_cntx_init_template( cntx_t* cntx )
{
	blksz_t blkszs[ BLIS_NUM_BLKSZS ];

	// Set default kernel blocksizes and functions.
	bli_cntx_init_template_ref( cntx );

	// -------------------------------------------------------------------------

	// Update the context with optimized native gemm micro-kernels and
	// their storage preferences.
	bli_cntx_set_ukrs
	(
	  cntx,

	  // level-3
	  BLIS_GEMM_UKR,       BLIS_DCOMPLEX, bli_zgemm_template_noopt,
	  BLIS_GEMMTRSM_L_UKR, BLIS_DCOMPLEX, bli_zgemmtrsm_l_template_noopt,
	  BLIS_GEMMTRSM_U_UKR, BLIS_DCOMPLEX, bli_zgemmtrsm_u_template_noopt,
	  BLIS_TRSM_L_UKR,     BLIS_DCOMPLEX, bli_ztrsm_l_template_noopt,
	  BLIS_TRSM_U_UKR,     BLIS_DCOMPLEX, bli_ztrsm_u_template_noopt,

	  // level-1f
	  BLIS_AXPY2V_KER,    BLIS_DCOMPLEX, bli_zaxpy2v_template_noopt,
	  BLIS_DOTAXPYV_KER,  BLIS_DCOMPLEX, bli_zdotaxpyv_template_noopt,
	  BLIS_AXPYF_KER,     BLIS_DCOMPLEX, bli_zaxpyf_template_noopt,
	  BLIS_DOTXF_KER,     BLIS_DCOMPLEX, bli_zdotxf_template_noopt,
	  BLIS_DOTXAXPYF_KER, BLIS_DCOMPLEX, bli_zdotxaxpyf_template_noopt,

	  // level-1v
	  BLIS_AXPYV_KER, BLIS_DCOMPLEX, bli_zaxpyv_template_noopt,
	  BLIS_DOTV_KER,  BLIS_DCOMPLEX, bli_zdotv_template_noopt,

	  BLIS_VA_END
	);

	// Update the context with storage preferences.
	bli_cntx_set_ukr_prefs
	(
	  cntx,

	  // level-3
	  BLIS_GEMM_UKR_ROW_PREF,       BLIS_DCOMPLEX, FALSE,
	  BLIS_GEMMTRSM_L_UKR_ROW_PREF, BLIS_DCOMPLEX, FALSE,
	  BLIS_GEMMTRSM_U_UKR_ROW_PREF, BLIS_DCOMPLEX, FALSE,
	  BLIS_TRSM_L_UKR_ROW_PREF,     BLIS_DCOMPLEX, FALSE,
	  BLIS_TRSM_U_UKR_ROW_PREF,     BLIS_DCOMPLEX, FALSE,

	  BLIS_VA_END
	);

	// Initialize level-3 blocksize objects with architecture-specific values.
	//                                           s      d      c      z
	bli_blksz_init_easy( &blkszs[ BLIS_MR ],    -1,    -1,    -1,     4 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],    -1,    -1,    -1,     4 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],    -1,    -1,    -1,   128 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],    -1,    -1,    -1,   256 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],    -1,    -1,    -1,  4096 );

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

