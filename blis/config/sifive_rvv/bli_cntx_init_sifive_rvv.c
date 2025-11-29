/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, SiFive, Inc.

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
#include <riscv_vector.h>

void bli_cntx_init_sifive_rvv( cntx_t* cntx )
{
	blksz_t blkszs[ BLIS_NUM_BLKSZS ];

	// Set default kernel blocksizes and functions.
	bli_cntx_init_sifive_rvv_ref( cntx );

	// -------------------------------------------------------------------------

        unsigned vlenb = __riscv_vlenb();

	// Update the context with optimized native kernels.
	bli_cntx_set_ukrs
	(
	  cntx,

	  // Level 1
	  BLIS_ADDV_KER,       BLIS_FLOAT,    bli_saddv_sifive_rvv_intr,
	  BLIS_ADDV_KER,       BLIS_DOUBLE,   bli_daddv_sifive_rvv_intr,
	  BLIS_ADDV_KER,       BLIS_SCOMPLEX, bli_caddv_sifive_rvv_intr,
	  BLIS_ADDV_KER,       BLIS_DCOMPLEX, bli_zaddv_sifive_rvv_intr,

	  BLIS_AMAXV_KER,      BLIS_FLOAT,    bli_samaxv_sifive_rvv_intr,
	  BLIS_AMAXV_KER,      BLIS_DOUBLE,   bli_damaxv_sifive_rvv_intr,
	  BLIS_AMAXV_KER,      BLIS_SCOMPLEX, bli_camaxv_sifive_rvv_intr,
	  BLIS_AMAXV_KER,      BLIS_DCOMPLEX, bli_zamaxv_sifive_rvv_intr,

	  BLIS_AXPBYV_KER,     BLIS_FLOAT,    bli_saxpbyv_sifive_rvv_intr,
	  BLIS_AXPBYV_KER,     BLIS_DOUBLE,   bli_daxpbyv_sifive_rvv_intr,
	  BLIS_AXPBYV_KER,     BLIS_SCOMPLEX, bli_caxpbyv_sifive_rvv_intr,
	  BLIS_AXPBYV_KER,     BLIS_DCOMPLEX, bli_zaxpbyv_sifive_rvv_intr,

	  BLIS_AXPYV_KER,      BLIS_FLOAT,    bli_saxpyv_sifive_rvv_intr,
	  BLIS_AXPYV_KER,      BLIS_DOUBLE,   bli_daxpyv_sifive_rvv_intr,
	  BLIS_AXPYV_KER,      BLIS_SCOMPLEX, bli_caxpyv_sifive_rvv_intr,
	  BLIS_AXPYV_KER,      BLIS_DCOMPLEX, bli_zaxpyv_sifive_rvv_intr,

	  BLIS_COPYV_KER,      BLIS_FLOAT,    bli_scopyv_sifive_rvv_intr,
	  BLIS_COPYV_KER,      BLIS_DOUBLE,   bli_dcopyv_sifive_rvv_intr,
	  BLIS_COPYV_KER,      BLIS_SCOMPLEX, bli_ccopyv_sifive_rvv_intr,
	  BLIS_COPYV_KER,      BLIS_DCOMPLEX, bli_zcopyv_sifive_rvv_intr,

	  BLIS_DOTV_KER,       BLIS_FLOAT,    bli_sdotv_sifive_rvv_intr,
	  BLIS_DOTV_KER,       BLIS_DOUBLE,   bli_ddotv_sifive_rvv_intr,
	  BLIS_DOTV_KER,       BLIS_SCOMPLEX, bli_cdotv_sifive_rvv_intr,
	  BLIS_DOTV_KER,       BLIS_DCOMPLEX, bli_zdotv_sifive_rvv_intr,

	  BLIS_DOTXV_KER,      BLIS_FLOAT,    bli_sdotxv_sifive_rvv_intr,
	  BLIS_DOTXV_KER,      BLIS_DOUBLE,   bli_ddotxv_sifive_rvv_intr,
	  BLIS_DOTXV_KER,      BLIS_SCOMPLEX, bli_cdotxv_sifive_rvv_intr,
	  BLIS_DOTXV_KER,      BLIS_DCOMPLEX, bli_zdotxv_sifive_rvv_intr,

	  BLIS_INVERTV_KER,    BLIS_FLOAT,    bli_sinvertv_sifive_rvv_intr,
	  BLIS_INVERTV_KER,    BLIS_DOUBLE,   bli_dinvertv_sifive_rvv_intr,
	  BLIS_INVERTV_KER,    BLIS_SCOMPLEX, bli_cinvertv_sifive_rvv_intr,
	  BLIS_INVERTV_KER,    BLIS_DCOMPLEX, bli_zinvertv_sifive_rvv_intr,

	  BLIS_INVSCALV_KER,   BLIS_FLOAT,    bli_sinvscalv_sifive_rvv_intr,
	  BLIS_INVSCALV_KER,   BLIS_DOUBLE,   bli_dinvscalv_sifive_rvv_intr,
	  BLIS_INVSCALV_KER,   BLIS_SCOMPLEX, bli_cinvscalv_sifive_rvv_intr,
	  BLIS_INVSCALV_KER,   BLIS_DCOMPLEX, bli_zinvscalv_sifive_rvv_intr,

	  BLIS_SCAL2V_KER,     BLIS_FLOAT,    bli_sscal2v_sifive_rvv_intr,
	  BLIS_SCAL2V_KER,     BLIS_DOUBLE,   bli_dscal2v_sifive_rvv_intr,
	  BLIS_SCAL2V_KER,     BLIS_SCOMPLEX, bli_cscal2v_sifive_rvv_intr,
	  BLIS_SCAL2V_KER,     BLIS_DCOMPLEX, bli_zscal2v_sifive_rvv_intr,

	  BLIS_SCALV_KER,      BLIS_FLOAT,    bli_sscalv_sifive_rvv_intr,
	  BLIS_SCALV_KER,      BLIS_DOUBLE,   bli_dscalv_sifive_rvv_intr,
	  BLIS_SCALV_KER,      BLIS_SCOMPLEX, bli_cscalv_sifive_rvv_intr,
	  BLIS_SCALV_KER,      BLIS_DCOMPLEX, bli_zscalv_sifive_rvv_intr,

	  BLIS_SETV_KER,       BLIS_FLOAT,    bli_ssetv_sifive_rvv_intr,
	  BLIS_SETV_KER,       BLIS_DOUBLE,   bli_dsetv_sifive_rvv_intr,
	  BLIS_SETV_KER,       BLIS_SCOMPLEX, bli_csetv_sifive_rvv_intr,
	  BLIS_SETV_KER,       BLIS_DCOMPLEX, bli_zsetv_sifive_rvv_intr,

	  BLIS_SUBV_KER,       BLIS_FLOAT,    bli_ssubv_sifive_rvv_intr,
	  BLIS_SUBV_KER,       BLIS_DOUBLE,   bli_dsubv_sifive_rvv_intr,
	  BLIS_SUBV_KER,       BLIS_SCOMPLEX, bli_csubv_sifive_rvv_intr,
	  BLIS_SUBV_KER,       BLIS_DCOMPLEX, bli_zsubv_sifive_rvv_intr,

	  BLIS_SWAPV_KER,      BLIS_FLOAT,    bli_sswapv_sifive_rvv_intr,
	  BLIS_SWAPV_KER,      BLIS_DOUBLE,   bli_dswapv_sifive_rvv_intr,
	  BLIS_SWAPV_KER,      BLIS_SCOMPLEX, bli_cswapv_sifive_rvv_intr,
	  BLIS_SWAPV_KER,      BLIS_DCOMPLEX, bli_zswapv_sifive_rvv_intr,

	  BLIS_XPBYV_KER,      BLIS_FLOAT,    bli_sxpbyv_sifive_rvv_intr,
	  BLIS_XPBYV_KER,      BLIS_DOUBLE,   bli_dxpbyv_sifive_rvv_intr,
	  BLIS_XPBYV_KER,      BLIS_SCOMPLEX, bli_cxpbyv_sifive_rvv_intr,
	  BLIS_XPBYV_KER,      BLIS_DCOMPLEX, bli_zxpbyv_sifive_rvv_intr,

	  // Level 1f
	  BLIS_AXPY2V_KER,     BLIS_FLOAT,    bli_saxpy2v_sifive_rvv_intr,
	  BLIS_AXPY2V_KER,     BLIS_DOUBLE,   bli_daxpy2v_sifive_rvv_intr,
	  BLIS_AXPY2V_KER,     BLIS_SCOMPLEX, bli_caxpy2v_sifive_rvv_intr,
	  BLIS_AXPY2V_KER,     BLIS_DCOMPLEX, bli_zaxpy2v_sifive_rvv_intr,

	  BLIS_AXPYF_KER,      BLIS_FLOAT,    bli_saxpyf_sifive_rvv_intr,
	  BLIS_AXPYF_KER,      BLIS_DOUBLE,   bli_daxpyf_sifive_rvv_intr,
	  BLIS_AXPYF_KER,      BLIS_SCOMPLEX, bli_caxpyf_sifive_rvv_intr,
	  BLIS_AXPYF_KER,      BLIS_DCOMPLEX, bli_zaxpyf_sifive_rvv_intr,

	  BLIS_DOTXF_KER,      BLIS_FLOAT,    bli_sdotxf_sifive_rvv_intr,
	  BLIS_DOTXF_KER,      BLIS_DOUBLE,   bli_ddotxf_sifive_rvv_intr,
	  BLIS_DOTXF_KER,      BLIS_SCOMPLEX, bli_cdotxf_sifive_rvv_intr,
	  BLIS_DOTXF_KER,      BLIS_DCOMPLEX, bli_zdotxf_sifive_rvv_intr,

	  BLIS_DOTAXPYV_KER,   BLIS_FLOAT,    bli_sdotaxpyv_sifive_rvv_intr,
	  BLIS_DOTAXPYV_KER,   BLIS_DOUBLE,   bli_ddotaxpyv_sifive_rvv_intr,
	  BLIS_DOTAXPYV_KER,   BLIS_SCOMPLEX, bli_cdotaxpyv_sifive_rvv_intr,
	  BLIS_DOTAXPYV_KER,   BLIS_DCOMPLEX, bli_zdotaxpyv_sifive_rvv_intr,

	  BLIS_DOTXAXPYF_KER,  BLIS_FLOAT,    bli_sdotxaxpyf_sifive_rvv_intr,
	  BLIS_DOTXAXPYF_KER,  BLIS_DOUBLE,   bli_ddotxaxpyf_sifive_rvv_intr,
	  BLIS_DOTXAXPYF_KER,  BLIS_SCOMPLEX, bli_cdotxaxpyf_sifive_rvv_intr,
	  BLIS_DOTXAXPYF_KER,  BLIS_DCOMPLEX, bli_zdotxaxpyf_sifive_rvv_intr,

	  // Level 1m
	  BLIS_PACKM_KER,      BLIS_FLOAT,    bli_spackm_sifive_rvv_intr,
	  BLIS_PACKM_KER,      BLIS_DOUBLE,   bli_dpackm_sifive_rvv_intr,
	  BLIS_PACKM_KER,      BLIS_SCOMPLEX, bli_cpackm_sifive_rvv_intr,
	  BLIS_PACKM_KER,      BLIS_DCOMPLEX, bli_zpackm_sifive_rvv_intr,

	  // Level 3
	  BLIS_GEMM_UKR,       BLIS_FLOAT,    bli_sgemm_sifive_rvv_intr,
	  BLIS_GEMM_UKR,       BLIS_DOUBLE,   bli_dgemm_sifive_rvv_intr,
	  BLIS_GEMM_UKR,       BLIS_SCOMPLEX, bli_cgemm_sifive_rvv_intr,
	  BLIS_GEMM_UKR,       BLIS_DCOMPLEX, bli_zgemm_sifive_rvv_intr,

	  BLIS_GEMMTRSM_L_UKR, BLIS_FLOAT,    bli_sgemmtrsm_l_sifive_rvv_intr,
	  BLIS_GEMMTRSM_L_UKR, BLIS_DOUBLE,   bli_dgemmtrsm_l_sifive_rvv_intr,
	  BLIS_GEMMTRSM_L_UKR, BLIS_SCOMPLEX, bli_cgemmtrsm_l_sifive_rvv_intr,
	  BLIS_GEMMTRSM_L_UKR, BLIS_DCOMPLEX, bli_zgemmtrsm_l_sifive_rvv_intr,
	  BLIS_GEMMTRSM_U_UKR, BLIS_FLOAT,    bli_sgemmtrsm_u_sifive_rvv_intr,
	  BLIS_GEMMTRSM_U_UKR, BLIS_DOUBLE,   bli_dgemmtrsm_u_sifive_rvv_intr,
	  BLIS_GEMMTRSM_U_UKR, BLIS_SCOMPLEX, bli_cgemmtrsm_u_sifive_rvv_intr,
	  BLIS_GEMMTRSM_U_UKR, BLIS_DCOMPLEX, bli_zgemmtrsm_u_sifive_rvv_intr,

	  BLIS_VA_END
	);

	// Update the context with storage preferences.
	bli_cntx_set_ukr_prefs
	(
	  cntx,

	  BLIS_GEMM_UKR_ROW_PREF,             BLIS_FLOAT,    TRUE,
	  BLIS_GEMM_UKR_ROW_PREF,             BLIS_DOUBLE,   TRUE,
	  BLIS_GEMM_UKR_ROW_PREF,             BLIS_SCOMPLEX, TRUE,
	  BLIS_GEMM_UKR_ROW_PREF,             BLIS_DCOMPLEX, TRUE,

	  BLIS_VA_END
	);

	// Initialize level-3 blocksize objects with architecture-specific values.
	//                                           s      d      c      z
	bli_blksz_init     ( &blkszs[ BLIS_MR ],     7,     7,     6,     6,
	                                             8,     8,     8,     8 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ], 4 * vlenb / 4, 4 * vlenb / 8, 2 * vlenb / 4, 2 * vlenb / 8 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],     7,     7,     6,     6 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ], 4 * vlenb / 4, 4 * vlenb / 8, 2 * vlenb / 4, 2 * vlenb / 8 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],    64,    64,    64,    64 );
	// Default BLIS_BBM_s = 1, but set here to ensure it's correct
	bli_blksz_init_easy( &blkszs[ BLIS_BBM ],    1,     1,     1,     1 );
	bli_blksz_init_easy( &blkszs[ BLIS_BBN ],    1,     1,     1,     1 );

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

	  // level-1m
	  BLIS_BBM, &blkszs[ BLIS_BBM ], BLIS_BBM,
	  BLIS_BBN, &blkszs[ BLIS_BBN ], BLIS_BBN,

	  BLIS_VA_END
	);
}

