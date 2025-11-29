/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020-2022, Advanced Micro Devices, Inc.

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

//GEMMSUP_KER_PROT( double,   d, gemmsup_r_haswell_ref )

void bli_cntx_init_zen( cntx_t* cntx )
{
	blksz_t blkszs[ BLIS_NUM_BLKSZS ];

	// Set default kernel blocksizes and functions.
	bli_cntx_init_zen_ref( cntx );

	// -------------------------------------------------------------------------

	// Update the context with optimized native gemm micro-kernels.
	bli_cntx_set_ukrs
	(
	  cntx,

	  // gemm
	  BLIS_GEMM_UKR,       BLIS_FLOAT,    bli_sgemm_haswell_asm_6x16,
	  BLIS_GEMM_UKR,       BLIS_DOUBLE,   bli_dgemm_haswell_asm_6x8,
	  BLIS_GEMM_UKR,       BLIS_SCOMPLEX, bli_cgemm_haswell_asm_3x8,
	  BLIS_GEMM_UKR,       BLIS_DCOMPLEX, bli_zgemm_haswell_asm_3x4,

	  // gemmtrsm_l
	  BLIS_GEMMTRSM_L_UKR, BLIS_FLOAT,    bli_sgemmtrsm_l_haswell_asm_6x16,
	  BLIS_GEMMTRSM_L_UKR, BLIS_DOUBLE,   bli_dgemmtrsm_l_haswell_asm_6x8,

	  // gemmtrsm_u

	  BLIS_GEMMTRSM_U_UKR, BLIS_FLOAT,    bli_sgemmtrsm_u_haswell_asm_6x16,
	  BLIS_GEMMTRSM_U_UKR, BLIS_DOUBLE,   bli_dgemmtrsm_u_haswell_asm_6x8,

	  // gemmsup
	  BLIS_GEMMSUP_RRR_UKR, BLIS_DOUBLE, bli_dgemmsup_rv_haswell_asm_6x8m,
	  BLIS_GEMMSUP_RRC_UKR, BLIS_DOUBLE, bli_dgemmsup_rd_haswell_asm_6x8m,
	  BLIS_GEMMSUP_RCR_UKR, BLIS_DOUBLE, bli_dgemmsup_rv_haswell_asm_6x8m,
	  BLIS_GEMMSUP_RCC_UKR, BLIS_DOUBLE, bli_dgemmsup_rv_haswell_asm_6x8n,
	  BLIS_GEMMSUP_CRR_UKR, BLIS_DOUBLE, bli_dgemmsup_rv_haswell_asm_6x8m,
	  BLIS_GEMMSUP_CRC_UKR, BLIS_DOUBLE, bli_dgemmsup_rd_haswell_asm_6x8n,
	  BLIS_GEMMSUP_CCR_UKR, BLIS_DOUBLE, bli_dgemmsup_rv_haswell_asm_6x8n,
	  BLIS_GEMMSUP_CCC_UKR, BLIS_DOUBLE, bli_dgemmsup_rv_haswell_asm_6x8n,

	  BLIS_GEMMSUP_RRR_UKR, BLIS_FLOAT, bli_sgemmsup_rv_haswell_asm_6x16m,
	  BLIS_GEMMSUP_RRC_UKR, BLIS_FLOAT, bli_sgemmsup_rd_haswell_asm_6x16m,
	  BLIS_GEMMSUP_RCR_UKR, BLIS_FLOAT, bli_sgemmsup_rv_haswell_asm_6x16m,
	  BLIS_GEMMSUP_RCC_UKR, BLIS_FLOAT, bli_sgemmsup_rv_haswell_asm_6x16n,
	  BLIS_GEMMSUP_CRR_UKR, BLIS_FLOAT, bli_sgemmsup_rv_haswell_asm_6x16m,
	  BLIS_GEMMSUP_CRC_UKR, BLIS_FLOAT, bli_sgemmsup_rd_haswell_asm_6x16n,
	  BLIS_GEMMSUP_CCR_UKR, BLIS_FLOAT, bli_sgemmsup_rv_haswell_asm_6x16n,
	  BLIS_GEMMSUP_CCC_UKR, BLIS_FLOAT, bli_sgemmsup_rv_haswell_asm_6x16n,
#if 0
	  BLIS_GEMMSUP_RRR_UKR, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x16m,
	  BLIS_GEMMSUP_RRC_UKR, BLIS_FLOAT, bli_sgemmsup_rd_zen_asm_6x16m,
	  BLIS_GEMMSUP_RCR_UKR, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x16m,
	  BLIS_GEMMSUP_RCC_UKR, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x16n,
	  BLIS_GEMMSUP_CRR_UKR, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x16m,
	  BLIS_GEMMSUP_CRC_UKR, BLIS_FLOAT, bli_sgemmsup_rd_zen_asm_6x16n,
	  BLIS_GEMMSUP_CCR_UKR, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x16n,
	  BLIS_GEMMSUP_CCC_UKR, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x16n,
#endif

#if 0
	  // NOTE: This set of kernels is likely broken and therefore disabled.
	  BLIS_GEMMSUP_RRR_UKR, BLIS_SCOMPLEX, bli_cgemmsup_rv_zen_asm_3x8m,
	  BLIS_GEMMSUP_RCR_UKR, BLIS_SCOMPLEX, bli_cgemmsup_rv_zen_asm_3x8m,
	  BLIS_GEMMSUP_CRR_UKR, BLIS_SCOMPLEX, bli_cgemmsup_rv_zen_asm_3x8m,
	  BLIS_GEMMSUP_RCC_UKR, BLIS_SCOMPLEX, bli_cgemmsup_rv_zen_asm_3x8n,
	  BLIS_GEMMSUP_CCR_UKR, BLIS_SCOMPLEX, bli_cgemmsup_rv_zen_asm_3x8n,
	  BLIS_GEMMSUP_CCC_UKR, BLIS_SCOMPLEX, bli_cgemmsup_rv_zen_asm_3x8n,

	  BLIS_GEMMSUP_RRR_UKR, BLIS_DCOMPLEX, bli_zgemmsup_rv_zen_asm_3x4m,
	  BLIS_GEMMSUP_RCR_UKR, BLIS_DCOMPLEX, bli_zgemmsup_rv_zen_asm_3x4m,
	  BLIS_GEMMSUP_CRR_UKR, BLIS_DCOMPLEX, bli_zgemmsup_rv_zen_asm_3x4m,
	  BLIS_GEMMSUP_RCC_UKR, BLIS_DCOMPLEX, bli_zgemmsup_rv_zen_asm_3x4n,
	  BLIS_GEMMSUP_CCR_UKR, BLIS_DCOMPLEX, bli_zgemmsup_rv_zen_asm_3x4n,
	  BLIS_GEMMSUP_CCC_UKR, BLIS_DCOMPLEX, bli_zgemmsup_rv_zen_asm_3x4n,
#endif

	  // packm
	  BLIS_PACKM_KER, BLIS_FLOAT,    bli_spackm_haswell_asm_6x16,
	  BLIS_PACKM_KER, BLIS_DOUBLE,   bli_dpackm_haswell_asm_6x8,
	  BLIS_PACKM_KER, BLIS_SCOMPLEX, bli_cpackm_haswell_asm_3x8,
	  BLIS_PACKM_KER, BLIS_DCOMPLEX, bli_zpackm_haswell_asm_3x4,

	  // axpyf
	  BLIS_AXPYF_KER,  BLIS_FLOAT,  bli_saxpyf_zen_int_8,
	  BLIS_AXPYF_KER,  BLIS_DOUBLE, bli_daxpyf_zen_int_8,
	  BLIS_AXPYF_KER,  BLIS_SCOMPLEX, bli_caxpyf_zen_int_5,
	  BLIS_AXPYF_KER,  BLIS_DCOMPLEX, bli_zaxpyf_zen_int_5,

	  // dotxf
	  BLIS_DOTXF_KER,  BLIS_FLOAT,  bli_sdotxf_zen_int_8,
	  BLIS_DOTXF_KER,  BLIS_DOUBLE, bli_ddotxf_zen_int_8,

	  // amaxv
	  BLIS_AMAXV_KER,  BLIS_FLOAT,  bli_samaxv_zen_int,
	  BLIS_AMAXV_KER,  BLIS_DOUBLE, bli_damaxv_zen_int,

	  // axpyv
	  BLIS_AXPYV_KER,  BLIS_FLOAT,  bli_saxpyv_zen_int_10,
	  BLIS_AXPYV_KER,  BLIS_DOUBLE, bli_daxpyv_zen_int_10,
	  BLIS_AXPYV_KER,  BLIS_SCOMPLEX,  bli_caxpyv_zen_int_5,
	  BLIS_AXPYV_KER,  BLIS_DCOMPLEX, bli_zaxpyv_zen_int_5,

	  // copyv
	  BLIS_COPYV_KER,  BLIS_FLOAT,  bli_scopyv_zen_int,
	  BLIS_COPYV_KER,  BLIS_DOUBLE, bli_dcopyv_zen_int,

	  // dotv
	  BLIS_DOTV_KER,   BLIS_FLOAT,  bli_sdotv_zen_int,
	  BLIS_DOTV_KER,   BLIS_DOUBLE, bli_ddotv_zen_int,

	  // dotxv
	  BLIS_DOTXV_KER,  BLIS_FLOAT,  bli_sdotxv_zen_int,
	  BLIS_DOTXV_KER,  BLIS_DOUBLE, bli_ddotxv_zen_int,

	  // scalv
	  BLIS_SCALV_KER,  BLIS_FLOAT,  bli_sscalv_zen_int10,
	  BLIS_SCALV_KER,  BLIS_DOUBLE, bli_dscalv_zen_int10,

	  // setv
	  BLIS_SETV_KER,   BLIS_FLOAT,  bli_ssetv_zen_int,
	  BLIS_SETV_KER,   BLIS_DOUBLE, bli_dsetv_zen_int,

	  // swapv
	  BLIS_SWAPV_KER,  BLIS_FLOAT,  bli_sswapv_zen_int8,
	  BLIS_SWAPV_KER,  BLIS_DOUBLE, bli_dswapv_zen_int8,

	  BLIS_VA_END
	);

	// Update the context with storage preferences.
	bli_cntx_set_ukr_prefs
	(
	  cntx,

	  // gemm
	  BLIS_GEMM_UKR_ROW_PREF,       BLIS_FLOAT,    TRUE,
	  BLIS_GEMM_UKR_ROW_PREF,       BLIS_DOUBLE,   TRUE,
	  BLIS_GEMM_UKR_ROW_PREF,       BLIS_SCOMPLEX, TRUE,
	  BLIS_GEMM_UKR_ROW_PREF,       BLIS_DCOMPLEX, TRUE,

	  // gemmtrsm_l
	  BLIS_GEMMTRSM_L_UKR_ROW_PREF, BLIS_FLOAT,    TRUE,
	  BLIS_GEMMTRSM_L_UKR_ROW_PREF, BLIS_DOUBLE,   TRUE,

	  // gemmtrsm_u
	  BLIS_GEMMTRSM_U_UKR_ROW_PREF, BLIS_FLOAT,    TRUE,
	  BLIS_GEMMTRSM_U_UKR_ROW_PREF, BLIS_DOUBLE,   TRUE,

	  // gemmsup
	  BLIS_GEMMSUP_RRR_UKR_ROW_PREF, BLIS_DOUBLE, TRUE,
	  BLIS_GEMMSUP_RRC_UKR_ROW_PREF, BLIS_DOUBLE, TRUE,
	  BLIS_GEMMSUP_RCR_UKR_ROW_PREF, BLIS_DOUBLE, TRUE,
	  BLIS_GEMMSUP_RCC_UKR_ROW_PREF, BLIS_DOUBLE, TRUE,
	  BLIS_GEMMSUP_CRR_UKR_ROW_PREF, BLIS_DOUBLE, TRUE,
	  BLIS_GEMMSUP_CRC_UKR_ROW_PREF, BLIS_DOUBLE, TRUE,
	  BLIS_GEMMSUP_CCR_UKR_ROW_PREF, BLIS_DOUBLE, TRUE,
	  BLIS_GEMMSUP_CCC_UKR_ROW_PREF, BLIS_DOUBLE, TRUE,

	  BLIS_GEMMSUP_RRR_UKR_ROW_PREF, BLIS_FLOAT,  TRUE,
	  BLIS_GEMMSUP_RRC_UKR_ROW_PREF, BLIS_FLOAT,  TRUE,
	  BLIS_GEMMSUP_RCR_UKR_ROW_PREF, BLIS_FLOAT,  TRUE,
	  BLIS_GEMMSUP_RCC_UKR_ROW_PREF, BLIS_FLOAT,  TRUE,
	  BLIS_GEMMSUP_CRR_UKR_ROW_PREF, BLIS_FLOAT,  TRUE,
	  BLIS_GEMMSUP_CRC_UKR_ROW_PREF, BLIS_FLOAT,  TRUE,
	  BLIS_GEMMSUP_CCR_UKR_ROW_PREF, BLIS_FLOAT,  TRUE,
	  BLIS_GEMMSUP_CCC_UKR_ROW_PREF, BLIS_FLOAT,  TRUE,

#if 0
	  // NOTE: This set of kernels is likely broken and therefore disabled.
	  BLIS_GEMMSUP_RRR_UKR_ROW_PREF, BLIS_SCOMPLEX, TRUE,
	  BLIS_GEMMSUP_RCR_UKR_ROW_PREF, BLIS_SCOMPLEX, TRUE,
	  BLIS_GEMMSUP_CRR_UKR_ROW_PREF, BLIS_SCOMPLEX, TRUE,
	  BLIS_GEMMSUP_RCC_UKR_ROW_PREF, BLIS_SCOMPLEX, TRUE,
	  BLIS_GEMMSUP_CCR_UKR_ROW_PREF, BLIS_SCOMPLEX, TRUE,
	  BLIS_GEMMSUP_CCC_UKR_ROW_PREF, BLIS_SCOMPLEX, TRUE,

	  BLIS_GEMMSUP_RRR_UKR_ROW_PREF, BLIS_DCOMPLEX, TRUE,
	  BLIS_GEMMSUP_RCR_UKR_ROW_PREF, BLIS_DCOMPLEX, TRUE,
	  BLIS_GEMMSUP_CRR_UKR_ROW_PREF, BLIS_DCOMPLEX, TRUE,
	  BLIS_GEMMSUP_RCC_UKR_ROW_PREF, BLIS_DCOMPLEX, TRUE,
	  BLIS_GEMMSUP_CCR_UKR_ROW_PREF, BLIS_DCOMPLEX, TRUE,
	  BLIS_GEMMSUP_CCC_UKR_ROW_PREF, BLIS_DCOMPLEX, TRUE,
#endif

	  BLIS_VA_END
	);

	// Initialize level-3 blocksize objects with architecture-specific values.
	//                                           s      d      c      z
	bli_blksz_init_easy( &blkszs[ BLIS_MR ],     6,     6,     3,     3 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],    16,     8,     8,     4 );

/*
	Multi Instance performance improvement of DGEMM when binded to a CCX
	In Multi instance each thread runs a sequential DGEMM.

	a)  If BLIS is run in a multi-instance mode with
	    CPU freq 2.6/2.2 Ghz
	    DDR4 clock frequency 2400Mhz
	    mc = 240, kc = 512, and nc = 2040
	    has better performance on EPYC server, over the default block sizes.

	b)  If BLIS is run in Single Instance mode
	    mc = 510, kc = 1024 and nc = 4080
*/

#ifdef BLIS_ENABLE_ZEN_BLOCK_SIZES
	// Zen optmized level 3 cache block sizes
	#if BLIS_ENABLE_SINGLE_INSTANCE_BLOCK_SIZES
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],  1020,   510,   510,   255 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],  1024,  1024,  1024,  1024 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],  8160,  4080,  4080,  3056 );
	#else
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],   144,   240,   144,    72 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],   256,   512,   256,   256 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],  4080,  2040,  2040,  1528 );
	#endif
#else
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],   144,    72,   144,    72 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],   256,   256,   256,   256 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],  8160,  4080,  4080,  3056 );
#endif
	bli_blksz_init_easy( &blkszs[ BLIS_AF ],     8,     8,    5,    5 );
	bli_blksz_init_easy( &blkszs[ BLIS_DF ],     8,     8,    -1,    -1 );

	// Initialize sup thresholds with architecture-appropriate values.
	//                                           s      d      c      z
	bli_blksz_init_easy( &blkszs[ BLIS_MT ],   512,   256,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NT ],   512,   256,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_KT ],   440,   220,    -1,    -1 );

	// Initialize level-3 sup blocksize objects with architecture-specific
	// values.
	//                                               s      d      c      z
	bli_blksz_init     ( &blkszs[ BLIS_MR_SUP ],     6,     6,    -1,    -1,
	                                                 9,     9,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR_SUP ],    16,     8,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC_SUP ],   144,    72,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC_SUP ],   256,   256,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC_SUP ],  8160,  4080,    -1,    -1 );
#if 0
	bli_blksz_init     ( &blkszs[ BLIS_MR_SUP ],     6,     6,     3,     3,
	                                                 9,     9,     3,     3 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR_SUP ],    16,     8,     8,     4 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC_SUP ],   144,    72,    72,    36 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC_SUP ],   512,   256,   128,    64 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC_SUP ],  8160,  4080,  2040,  1020 );
#endif

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

	  // level-1f
	  BLIS_AF, &blkszs[ BLIS_AF ], BLIS_AF,
	  BLIS_DF, &blkszs[ BLIS_DF ], BLIS_DF,

	  // sup thresholds
	  BLIS_MT, &blkszs[ BLIS_MT ], BLIS_MT,
	  BLIS_NT, &blkszs[ BLIS_NT ], BLIS_NT,
	  BLIS_KT, &blkszs[ BLIS_KT ], BLIS_KT,

	  // gemmsup
	  BLIS_NC_SUP, &blkszs[ BLIS_NC_SUP ], BLIS_NR_SUP,
	  BLIS_KC_SUP, &blkszs[ BLIS_KC_SUP ], BLIS_KR_SUP,
	  BLIS_MC_SUP, &blkszs[ BLIS_MC_SUP ], BLIS_MR_SUP,
	  BLIS_NR_SUP, &blkszs[ BLIS_NR_SUP ], BLIS_NR_SUP,
	  BLIS_MR_SUP, &blkszs[ BLIS_MR_SUP ], BLIS_MR_SUP,

	  BLIS_VA_END
	);

	// -------------------------------------------------------------------------

#if 0
	// Initialize the context with the sup handlers.
	bli_cntx_set_l3_sup_handlers
	(
	  cntx,

	  BLIS_GEMM, bli_gemmsup_ref,
	  //BLIS_GEMMT, bli_gemmtsup_ref,

	  BLIS_VA_END
	);
#endif
}
