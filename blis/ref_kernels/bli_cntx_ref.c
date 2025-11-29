/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2020, Advanced Micro Devices, Inc.
   Copyright (C) 2022-2024, Southern Methodist University

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

// -- Instantiate kernel prototypes for the current architecture ---------------

// Define macros to construct the full symbol name from the operation name.
#undef  GENARNAME             // opname, architecture, _ref (no bli_)
#define GENARNAME(opname)     PASTECH(opname,BLIS_CNAME_INFIX,BLIS_REF_SUFFIX)
#undef  GENTARNAME            // bli, ch, opname, architecture, _ref
#define GENTARNAME(ch,opname) PASTEMAC(ch,opname,BLIS_CNAME_INFIX,BLIS_REF_SUFFIX)
#undef  GENBARNAME            // bli_, opname, architecture, _ref
#define GENBARNAME(opname)    PASTEMAC(opname,BLIS_CNAME_INFIX,BLIS_REF_SUFFIX)

// Define a prototype-inserting template that uses an arbitrary prototype-
// generating macro.

#undef  INSERT_PROTMAC_BASIC
#define INSERT_PROTMAC_BASIC( protmac, kername ) \
\
protmac( float,    s, kername ) \
protmac( double,   d, kername ) \
protmac( scomplex, c, kername ) \
protmac( dcomplex, z, kername )

#undef  INSERT_PROTMAC_BASIC_CO
#define INSERT_PROTMAC_BASIC_CO( protmac, kername ) \
\
protmac( scomplex, c, kername ) \
protmac( dcomplex, z, kername )

#undef  INSERT_PROTMAC_MIX_RO
#define INSERT_PROTMAC_MIX_RO( protmac, kername ) \
\
protmac( float,  float,  s, s, kername ) \
protmac( double, float,  d, s, kername ) \
protmac( float,  double, s, d, kername ) \
protmac( double, double, d, d, kername )

#undef  INSERT_PROTMAC_MIX_CO
#define INSERT_PROTMAC_MIX_CO( protmac, kername ) \
\
protmac( scomplex, scomplex, c, c, kername ) \
protmac( dcomplex, scomplex, z, c, kername ) \
protmac( scomplex, dcomplex, c, z, kername ) \
protmac( dcomplex, dcomplex, z, z, kername )

#undef  INSERT_PROTMAC_MIX_P
#define INSERT_PROTMAC_MIX_P( protmac, kername ) \
\
protmac( float,    float,    s, s, kername ) \
protmac( double,   float,    d, s, kername ) \
protmac( float,    double,   s, d, kername ) \
protmac( double,   double,   d, d, kername ) \
\
protmac( scomplex, scomplex, c, c, kername ) \
protmac( dcomplex, scomplex, z, c, kername ) \
protmac( scomplex, dcomplex, c, z, kername ) \
protmac( dcomplex, dcomplex, z, z, kername )

#undef  INSERT_PROTMAC_MIX_DP
#define INSERT_PROTMAC_MIX_DP( protmac, kername ) \
\
protmac( float,    float,    s, s, kername ) \
protmac( double,   float,    d, s, kername ) \
protmac( scomplex, float,    c, s, kername ) \
protmac( dcomplex, float,    z, s, kername ) \
\
protmac( float,    double,   s, d, kername ) \
protmac( double,   double,   d, d, kername ) \
protmac( scomplex, double,   c, d, kername ) \
protmac( dcomplex, double,   z, d, kername ) \
\
protmac( float,    scomplex, s, c, kername ) \
protmac( double,   scomplex, d, c, kername ) \
protmac( scomplex, scomplex, c, c, kername ) \
protmac( dcomplex, scomplex, z, c, kername ) \
\
protmac( float,    dcomplex, s, z, kername ) \
protmac( double,   dcomplex, d, z, kername ) \
protmac( scomplex, dcomplex, c, z, kername ) \
protmac( dcomplex, dcomplex, z, z, kername )


// -- Level-3 native micro-kernel prototype redefinitions ----------------------

// -- Construct arch-specific names for reference level-3 microkernels --

#define gemm_ukr_name       GENARNAME(gemm)
#define gemmtrsm_l_ukr_name GENARNAME(gemmtrsm_l)
#define gemmtrsm_u_ukr_name GENARNAME(gemmtrsm_u)
#define trsm_l_ukr_name     GENARNAME(trsm_l)
#define trsm_u_ukr_name     GENARNAME(trsm_u)

// Instantiate prototypes for above functions using the pre-defined level-3
// microkernel prototype-generating macros.

INSERT_PROTMAC_MIX_P( GEMM_UKR2_PROT,    gemm_ukr_name )
INSERT_PROTMAC_BASIC( GEMMTRSM_UKR_PROT, gemmtrsm_l_ukr_name )
INSERT_PROTMAC_BASIC( GEMMTRSM_UKR_PROT, gemmtrsm_u_ukr_name )
INSERT_PROTMAC_BASIC( TRSM_UKR_PROT,     trsm_l_ukr_name )
INSERT_PROTMAC_BASIC( TRSM_UKR_PROT,     trsm_u_ukr_name )


// -- Level-3 virtual micro-kernel prototype redefinitions ---------------------

// -- Construct arch-specific names for reference virtual level-3 microkernels --

#define gemm1m_ukr_name        GENARNAME(gemm1m)
#define gemm_ccr_ukr_name       GENARNAME(gemm_ccr)
#define gemm_rcc_ukr_name       GENARNAME(gemm_rcc)
#define gemm_crr_ukr_name        GENARNAME(gemm_crr)
#define gemmtrsm1m_l_ukr_name  GENARNAME(gemmtrsm1m_l)
#define gemmtrsm1m_u_ukr_name  GENARNAME(gemmtrsm1m_u)

// Instantiate prototypes for above functions using the pre-defined level-3
// microkernel prototype-generating macros.

INSERT_PROTMAC_MIX_RO( GEMM_UKR2_PROT,    gemm1m_ukr_name )
INSERT_PROTMAC_MIX_RO( GEMM_UKR2_PROT,    gemm_ccr_ukr_name )
INSERT_PROTMAC_MIX_RO( GEMM_UKR2_PROT,    gemm_rcc_ukr_name )
INSERT_PROTMAC_MIX_RO( GEMM_UKR2_PROT,    gemm_crr_ukr_name )
INSERT_PROTMAC_BASIC(  GEMMTRSM_UKR_PROT, gemmtrsm1m_l_ukr_name )
INSERT_PROTMAC_BASIC(  GEMMTRSM_UKR_PROT, gemmtrsm1m_u_ukr_name )
INSERT_PROTMAC_BASIC(  TRSM_UKR_PROT,     trsm1m_l_ukr_name )
INSERT_PROTMAC_BASIC(  TRSM_UKR_PROT,     trsm1m_u_ukr_name )


// -- Level-3 small/unpacked micro-kernel prototype definitions ----------------

// -- Construct arch-specific names for reference gemmsup kernels --

#define gemmsup_rv_ukr_name  GENARNAME(gemmsup_r)
#define gemmsup_rg_ukr_name  GENARNAME(gemmsup_r)
#define gemmsup_cv_ukr_name  GENARNAME(gemmsup_c)
#define gemmsup_cg_ukr_name  GENARNAME(gemmsup_c)
#define gemmsup_gx_ukr_name  GENARNAME(gemmsup_g)

// Instantiate prototypes for above functions using the pre-defined gemmsup
// kernel prototype-generating macros.

INSERT_PROTMAC_BASIC( GEMMSUP_KER_PROT, gemmsup_rv_ukr_name )
INSERT_PROTMAC_BASIC( GEMMSUP_KER_PROT, gemmsup_rg_ukr_name )
INSERT_PROTMAC_BASIC( GEMMSUP_KER_PROT, gemmsup_cv_ukr_name )
INSERT_PROTMAC_BASIC( GEMMSUP_KER_PROT, gemmsup_cg_ukr_name )
INSERT_PROTMAC_BASIC( GEMMSUP_KER_PROT, gemmsup_gx_ukr_name )


// -- Level-1m (packm/unpackm) kernel prototype redefinitions ------------------

// -- Construct arch-specific names for reference packm kernels --

#define packm_ker_name           GENARNAME(packm)
#define packm_1er_ker_name       GENARNAME(packm_1er)
#define packm_ro_ker_name       GENARNAME(packm_ro)
#define packm_diag_ker_name      GENARNAME(packm_diag)
#define packm_diag_1er_ker_name  GENARNAME(packm_diag_1er)
#define packm_diag_ro_ker_name  GENARNAME(packm_diag_ro)
#define unpackm_ker_name         GENARNAME(unpackm)

// Instantiate prototypes for above functions using the pre-defined packm
// kernel prototype-generating macros.

INSERT_PROTMAC_MIX_P ( PACKM_KER_PROT2,      packm_ker_name )
INSERT_PROTMAC_MIX_CO( PACKM_KER_PROT2,      packm_1er_ker_name )
INSERT_PROTMAC_MIX_CO( PACKM_KER_PROT2,      packm_ro_ker_name )
INSERT_PROTMAC_MIX_P ( PACKM_DIAG_KER_PROT2, packm_diag_ker_name )
INSERT_PROTMAC_MIX_CO( PACKM_DIAG_KER_PROT2, packm_diag_1er_ker_name )
INSERT_PROTMAC_MIX_CO( PACKM_DIAG_KER_PROT2, packm_diag_ro_ker_name )
INSERT_PROTMAC_MIX_P ( UNPACKM_KER_PROT2,    unpackm_ker_name )


// -- Level-1f kernel prototype redefinitions ----------------------------------

// -- Construct arch-specific names for reference level-1f kernels --

#define axpy2v_ker_name     GENARNAME(axpy2v)
#define axpyf_ker_name      GENARNAME(axpyf)
#define dotaxpyv_ker_name   GENARNAME(dotaxpyv)
#define dotxaxpyf_ker_name  GENARNAME(dotxaxpyf)
#define dotxf_ker_name      GENARNAME(dotxf)

// Instantiate prototypes for above functions using the pre-defined level-1f
// kernel prototype-generating macros.

INSERT_PROTMAC_BASIC( AXPY2V_KER_PROT,     axpy2v_ker_name )
INSERT_PROTMAC_BASIC( AXPYF_KER_PROT,      axpyf_ker_name )
INSERT_PROTMAC_BASIC( DOTAXPYV_KER_PROT,   dotaxpyv_ker_name )
INSERT_PROTMAC_BASIC( DOTXAXPYF_KER_PROT,  dotxaxpyf_ker_name )
INSERT_PROTMAC_BASIC( DOTXF_KER_PROT,      dotxf_ker_name )


// -- Level-1v kernel prototype redefinitions ----------------------------------

// -- Construct arch-specific names for reference level-1v kernels --

#define addv_ker_name      GENARNAME(addv)
#define amaxv_ker_name     GENARNAME(amaxv)
#define axpbyv_ker_name    GENARNAME(axpbyv)
#define axpyv_ker_name     GENARNAME(axpyv)
#define copyv_ker_name     GENARNAME(copyv)
#define dotv_ker_name      GENARNAME(dotv)
#define dotxv_ker_name     GENARNAME(dotxv)
#define invertv_ker_name   GENARNAME(invertv)
#define invscalv_ker_name  GENARNAME(invscalv)
#define scalv_ker_name     GENARNAME(scalv)
#define scal2v_ker_name    GENARNAME(scal2v)
#define setv_ker_name      GENARNAME(setv)
#define subv_ker_name      GENARNAME(subv)
#define swapv_ker_name     GENARNAME(swapv)
#define xpbyv_ker_name     GENARNAME(xpbyv)

// Instantiate prototypes for above functions using the pre-defined level-1v
// kernel prototype-generating macros.

INSERT_PROTMAC_BASIC( ADDV_KER_PROT,     addv_ker_name )
INSERT_PROTMAC_BASIC( AMAXV_KER_PROT,    amaxv_ker_name )
INSERT_PROTMAC_BASIC( AXPBYV_KER_PROT,   axpbyv_ker_name )
INSERT_PROTMAC_BASIC( AXPYV_KER_PROT,    axpyv_ker_name )
INSERT_PROTMAC_BASIC( COPYV_KER_PROT,    copyv_ker_name )
INSERT_PROTMAC_BASIC( DOTV_KER_PROT,     dotv_ker_name )
INSERT_PROTMAC_BASIC( DOTXV_KER_PROT,    dotxv_ker_name )
INSERT_PROTMAC_BASIC( INVERTV_KER_PROT,  invertv_ker_name )
INSERT_PROTMAC_BASIC( INVSCALV_KER_PROT, invscalv_ker_name )
INSERT_PROTMAC_BASIC( SCALV_KER_PROT,    scalv_ker_name )
INSERT_PROTMAC_BASIC( SCAL2V_KER_PROT,   scal2v_ker_name )
INSERT_PROTMAC_BASIC( SETV_KER_PROT,     setv_ker_name )
INSERT_PROTMAC_BASIC( SUBV_KER_PROT,     subv_ker_name )
INSERT_PROTMAC_BASIC( SWAPV_KER_PROT,    swapv_ker_name )
INSERT_PROTMAC_BASIC( XPBYV_KER_PROT,    xpbyv_ker_name )



// -- Macros to help concisely instantiate bli_func_init() ---------------------

#define gen_func_init_ro( func_p, opname ) \
do { \
	bli_func_init( func_p, PASTEMAC(s,opname), PASTEMAC(d,opname), \
	                       NULL,               NULL ); \
} while (0)

#define gen_func_init_co( func_p, opname ) \
do { \
	bli_func_init( func_p, NULL,               NULL, \
	                       PASTEMAC(c,opname), PASTEMAC(z,opname) ); \
} while (0)

#define gen_func_init_mix_co( func_p, opname ) \
do { \
	bli_func2_init( func_p, NULL,                  NULL, \
	                        NULL,                  NULL, \
	                        NULL,                  NULL, \
	                        NULL,                  NULL, \
	                        NULL,                  NULL, \
	                        PASTEMAC(c,c,opname), PASTEMAC(c,z,opname), \
	                        NULL,                  NULL, \
	                        PASTEMAC(z,c,opname), PASTEMAC(z,z,opname) ); \
} while (0)

#define gen_func_init( func_p, opname ) \
do { \
	bli_func_init( func_p, PASTEMAC(s,opname), PASTEMAC(d,opname), \
	                       PASTEMAC(c,opname), PASTEMAC(z,opname) ); \
} while (0)

#define gen_func_init_ro_mix_p( func_p, opname ) \
do { \
	bli_func2_init( func_p, PASTEMAC(s,s,opname), PASTEMAC(s,d,opname), \
	                        NULL,                  NULL, \
	                        PASTEMAC(d,s,opname), PASTEMAC(d,d,opname), \
	                        NULL,                  NULL, \
	                        NULL,                  NULL, \
	                        NULL,                  NULL, \
	                        NULL,                  NULL, \
	                        NULL,                  NULL ); \
} while (0)

#define gen_func_init_mix_p( func_p, opname ) \
do { \
	bli_func2_init( func_p, PASTEMAC(s,s,opname), PASTEMAC(s,d,opname), \
	                        NULL,                  NULL, \
	                        PASTEMAC(d,s,opname), PASTEMAC(d,d,opname), \
	                        NULL,                  NULL, \
	                        NULL,                  NULL, \
	                        PASTEMAC(c,c,opname), PASTEMAC(c,z,opname), \
	                        NULL,                  NULL, \
	                        PASTEMAC(z,c,opname), PASTEMAC(z,z,opname) ); \
} while (0)

#define gen_func_init_mix_dp( func_p, opname ) \
do { \
	bli_func2_init( func_p, PASTEMAC(s,s,opname), PASTEMAC(s,d,opname), \
	                        PASTEMAC(s,c,opname), PASTEMAC(s,z,opname), \
	                        PASTEMAC(d,s,opname), PASTEMAC(d,d,opname), \
	                        PASTEMAC(d,c,opname), PASTEMAC(d,z,opname), \
	                        PASTEMAC(c,s,opname), PASTEMAC(c,d,opname), \
	                        PASTEMAC(c,c,opname), PASTEMAC(c,z,opname), \
	                        PASTEMAC(z,s,opname), PASTEMAC(z,d,opname), \
	                        PASTEMAC(z,c,opname), PASTEMAC(z,z,opname) ); \
} while (0)

#define gen_sup_func_init( func0_p, func1_p, opname ) \
do { \
	bli_func_init( func0_p, PASTEMAC(s,opname), PASTEMAC(d,opname), \
	                        PASTEMAC(c,opname), PASTEMAC(z,opname) ); \
	bli_func_init( func1_p, PASTEMAC(s,opname), PASTEMAC(d,opname), \
	                        PASTEMAC(c,opname), PASTEMAC(z,opname) ); \
} while (0)

// -----------------------------------------------------------------------------

void GENBARNAME(cntx_init)
     (
       cntx_t* cntx
     )
{
	blksz_t blkszs[ BLIS_NUM_BLKSZS ];
	func_t  funcs [ BLIS_NUM_UKRS ];
	func2_t func2s[ BLIS_NUM_UKR2S ];
	mbool_t mbools[ BLIS_NUM_UKR_PREFS ];
	void_fp vfuncs[ BLIS_NUM_LEVEL3_OPS ];

	// Make sure any unset function pointers or block sizes are NULL/0
	memset( blkszs, 0, sizeof(blkszs) );
	memset( funcs,  0, sizeof(funcs)  );
	memset( func2s, 0, sizeof(func2s) );
	memset( mbools, 0, sizeof(mbools) );
	memset( vfuncs, 0, sizeof(vfuncs) );

	// -- Initialize the context -----------------------------------------------

	bli_cntx_init( cntx );


	// -- Set blocksizes -------------------------------------------------------

	// NOTE: The macro values for register blocksizes and packm broadcast factors are
	// used here as defined in the bli_kernel_defs_<family>.h or generic values from
	// bli_kernel_macro_defs.h otherwise. Configurations should also initialize the
	// blocksizes in the context explicitly, but using the correct values here helps
	// to prevent accidents.
	//                                                    s              d              c              z
	bli_blksz_init_easy( &blkszs[ BLIS_KR  ],             1,             1,             1,             1 );
	bli_blksz_init     ( &blkszs[ BLIS_MR  ],     BLIS_MR_s,     BLIS_MR_d,     BLIS_MR_c,     BLIS_MR_z,
	                                          BLIS_PACKMR_s, BLIS_PACKMR_d, BLIS_PACKMR_c, BLIS_PACKMR_z );
	bli_blksz_init     ( &blkszs[ BLIS_NR  ],     BLIS_NR_s,     BLIS_NR_d,     BLIS_NR_c,     BLIS_NR_z,
	                                          BLIS_PACKNR_s, BLIS_PACKNR_d, BLIS_PACKNR_c, BLIS_PACKNR_z );
	bli_blksz_init_easy( &blkszs[ BLIS_MC  ],           256,           128,           128,            64 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC  ],           256,           256,           256,           256 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC  ],          4096,          4096,          4096,          4096 );
	bli_blksz_init_easy( &blkszs[ BLIS_M2  ],          1000,          1000,          1000,          1000 );
	bli_blksz_init_easy( &blkszs[ BLIS_N2  ],          1000,          1000,          1000,          1000 );
	bli_blksz_init_easy( &blkszs[ BLIS_AF  ],             8,             8,             8,             8 );
	bli_blksz_init_easy( &blkszs[ BLIS_DF  ],             6,             6,             6,             6 );
	bli_blksz_init_easy( &blkszs[ BLIS_XF  ],             4,             4,             4,             4 );
	bli_blksz_init_easy( &blkszs[ BLIS_BBM ],    BLIS_BBM_s,    BLIS_BBM_d,    BLIS_BBM_c,    BLIS_BBM_z );
	bli_blksz_init_easy( &blkszs[ BLIS_BBN ],    BLIS_BBN_s,    BLIS_BBN_d,    BLIS_BBN_c,    BLIS_BBN_z );

	// -- Set level-3 small/unpacked thresholds --------------------------------

	// NOTE: The default thresholds are set to zero so that the sup framework
	// does not activate by default. Note that the semantic meaning of the
	// thresholds is that the sup code path is executed if a dimension is
	// strictly less than its corresponding threshold. So actually, the
	// thresholds specify the minimum dimension size that will still dispatch
	// the non-sup/large code path. This "strictly less than" behavior was
	// chosen over "less than or equal to" so that threshold values of 0 would
	// effectively disable sup (even for matrix dimensions of 0).
	//                                          s     d     c     z
	bli_blksz_init_easy( &blkszs[ BLIS_MT ],    0,    0,    0,    0 );
	bli_blksz_init_easy( &blkszs[ BLIS_NT ],    0,    0,    0,    0 );
	bli_blksz_init_easy( &blkszs[ BLIS_KT ],    0,    0,    0,    0 );

	// Initialize the context with the default blocksize objects and their
	// multiples.
	bli_cntx_set_blkszs
	(
	  cntx,
	  BLIS_NC,  &blkszs[ BLIS_NC  ], BLIS_NR,
	  BLIS_KC,  &blkszs[ BLIS_KC  ], BLIS_KR,
	  BLIS_MC,  &blkszs[ BLIS_MC  ], BLIS_MR,
	  BLIS_NR,  &blkszs[ BLIS_NR  ], BLIS_NR,
	  BLIS_MR,  &blkszs[ BLIS_MR  ], BLIS_MR,
	  BLIS_KR,  &blkszs[ BLIS_KR  ], BLIS_KR,
	  BLIS_M2,  &blkszs[ BLIS_M2  ], BLIS_M2,
	  BLIS_N2,  &blkszs[ BLIS_N2  ], BLIS_N2,
	  BLIS_AF,  &blkszs[ BLIS_AF  ], BLIS_AF,
	  BLIS_DF,  &blkszs[ BLIS_DF  ], BLIS_DF,
	  BLIS_XF,  &blkszs[ BLIS_XF  ], BLIS_XF,
	  BLIS_MT,  &blkszs[ BLIS_MT  ], BLIS_MT,
	  BLIS_NT,  &blkszs[ BLIS_NT  ], BLIS_NT,
	  BLIS_KT,  &blkszs[ BLIS_KT  ], BLIS_KT,
	  BLIS_BBM, &blkszs[ BLIS_BBM ], BLIS_BBM,
	  BLIS_BBN, &blkszs[ BLIS_BBN ], BLIS_BBN,
	  BLIS_VA_END
	);


	// -- Set level-3 native micro-kernels and preferences ---------------------

	gen_func_init_mix_p( &func2s[ bli_ker_idx( BLIS_GEMM_UKR ) ], gemm_ukr_name );

	gen_func_init_ro_mix_p( &func2s[ bli_ker_idx( BLIS_GEMM1M_UKR ) ],  gemm1m_ukr_name  );
	gen_func_init_ro_mix_p( &func2s[ bli_ker_idx( BLIS_GEMM_CCR_UKR ) ], gemm_ccr_ukr_name );
	gen_func_init_ro_mix_p( &func2s[ bli_ker_idx( BLIS_GEMM_RCC_UKR ) ], gemm_rcc_ukr_name );
	gen_func_init_ro_mix_p( &func2s[ bli_ker_idx( BLIS_GEMM_CRR_UKR ) ],  gemm_crr_ukr_name  );

	gen_func_init( &funcs[ bli_ker_idx( BLIS_GEMMTRSM_L_UKR ) ], gemmtrsm_l_ukr_name );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_GEMMTRSM_U_UKR ) ], gemmtrsm_u_ukr_name );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_TRSM_L_UKR ) ],     trsm_l_ukr_name     );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_TRSM_U_UKR ) ],     trsm_u_ukr_name     );

	gen_func_init_ro( &funcs[ bli_ker_idx( BLIS_GEMMTRSM1M_L_UKR ) ], gemmtrsm1m_l_ukr_name );
	gen_func_init_ro( &funcs[ bli_ker_idx( BLIS_GEMMTRSM1M_U_UKR ) ], gemmtrsm1m_u_ukr_name );

	//                                                           s      d      c      z
	bli_mbool_init( &mbools[ BLIS_GEMM_UKR_ROW_PREF ],        TRUE,  TRUE,  TRUE,  TRUE );
	bli_mbool_init( &mbools[ BLIS_GEMMTRSM_L_UKR_ROW_PREF ], FALSE, FALSE, FALSE, FALSE );
	bli_mbool_init( &mbools[ BLIS_GEMMTRSM_U_UKR_ROW_PREF ], FALSE, FALSE, FALSE, FALSE );
	bli_mbool_init( &mbools[ BLIS_TRSM_L_UKR_ROW_PREF ],     FALSE, FALSE, FALSE, FALSE );
	bli_mbool_init( &mbools[ BLIS_TRSM_U_UKR_ROW_PREF ],     FALSE, FALSE, FALSE, FALSE );


	// -- Set level-3 small/unpacked micro-kernels and preferences -------------

	gen_func_init( &funcs[ bli_ker_idx( BLIS_GEMMSUP_RRR_UKR ) ], gemmsup_rv_ukr_name );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_GEMMSUP_RRC_UKR ) ], gemmsup_rv_ukr_name );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_GEMMSUP_RCR_UKR ) ], gemmsup_rv_ukr_name );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_GEMMSUP_RCC_UKR ) ], gemmsup_rv_ukr_name );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_GEMMSUP_CRR_UKR ) ], gemmsup_rv_ukr_name );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_GEMMSUP_CRC_UKR ) ], gemmsup_rv_ukr_name );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_GEMMSUP_CCR_UKR ) ], gemmsup_rv_ukr_name );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_GEMMSUP_CCC_UKR ) ], gemmsup_rv_ukr_name );

	// Register the general-stride/generic ukernel to the "catch-all" slot
	// associated with the BLIS_XXX enum value. This slot will be queried if
	// *any* operand is stored with general stride.
	gen_func_init( &funcs[ bli_ker_idx( BLIS_GEMMSUP_XXX_UKR ) ], gemmsup_gx_ukr_name );


	// Set the l3 sup ukernel storage preferences.
	//                                                            s      d      c      z
	bli_mbool_init( &mbools[ BLIS_GEMMSUP_RRR_UKR_ROW_PREF ],  TRUE,  TRUE,  TRUE,  TRUE );
	bli_mbool_init( &mbools[ BLIS_GEMMSUP_RRC_UKR_ROW_PREF ],  TRUE,  TRUE,  TRUE,  TRUE );
	bli_mbool_init( &mbools[ BLIS_GEMMSUP_RCR_UKR_ROW_PREF ],  TRUE,  TRUE,  TRUE,  TRUE );
	bli_mbool_init( &mbools[ BLIS_GEMMSUP_RCC_UKR_ROW_PREF ],  TRUE,  TRUE,  TRUE,  TRUE );
	bli_mbool_init( &mbools[ BLIS_GEMMSUP_CRR_UKR_ROW_PREF ],  TRUE,  TRUE,  TRUE,  TRUE );
	bli_mbool_init( &mbools[ BLIS_GEMMSUP_CRC_UKR_ROW_PREF ],  TRUE,  TRUE,  TRUE,  TRUE );
	bli_mbool_init( &mbools[ BLIS_GEMMSUP_CCR_UKR_ROW_PREF ],  TRUE,  TRUE,  TRUE,  TRUE );
	bli_mbool_init( &mbools[ BLIS_GEMMSUP_CCC_UKR_ROW_PREF ],  TRUE,  TRUE,  TRUE,  TRUE );

	bli_mbool_init( &mbools[ BLIS_GEMMSUP_XXX_UKR_ROW_PREF ],  TRUE,  TRUE,  TRUE,  TRUE );


	// -- Set level-1f kernels -------------------------------------------------

	gen_func_init( &funcs[ bli_ker_idx( BLIS_AXPY2V_KER ) ],    axpy2v_ker_name    );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_DOTAXPYV_KER ) ],  dotaxpyv_ker_name  );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_AXPYF_KER ) ],     axpyf_ker_name     );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_DOTXF_KER ) ],     dotxf_ker_name     );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_DOTXAXPYF_KER ) ], dotxaxpyf_ker_name );


	// -- Set level-1v kernels -------------------------------------------------

	gen_func_init( &funcs[ bli_ker_idx( BLIS_ADDV_KER ) ],     addv_ker_name     );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_AMAXV_KER ) ],    amaxv_ker_name    );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_AXPBYV_KER ) ],   axpbyv_ker_name   );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_AXPYV_KER ) ],    axpyv_ker_name    );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_COPYV_KER ) ],    copyv_ker_name    );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_DOTV_KER ) ],     dotv_ker_name     );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_DOTXV_KER ) ],    dotxv_ker_name    );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_INVERTV_KER ) ],  invertv_ker_name  );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_INVSCALV_KER ) ], invscalv_ker_name );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_SCALV_KER ) ],    scalv_ker_name    );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_SCAL2V_KER ) ],   scal2v_ker_name   );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_SETV_KER ) ],     setv_ker_name     );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_SUBV_KER ) ],     subv_ker_name     );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_SWAPV_KER ) ],    swapv_ker_name    );
	gen_func_init( &funcs[ bli_ker_idx( BLIS_XPBYV_KER ) ],    xpbyv_ker_name    );


	// -- Set level-1m (packm/unpackm) kernels ---------------------------------

	gen_func_init_mix_p ( &func2s[ bli_ker_idx( BLIS_PACKM_KER ) ],           packm_ker_name );
	gen_func_init_mix_co( &func2s[ bli_ker_idx( BLIS_PACKM_1ER_KER ) ],       packm_1er_ker_name );
	gen_func_init_mix_co( &func2s[ bli_ker_idx( BLIS_PACKM_RO_KER ) ],        packm_ro_ker_name );
	gen_func_init_mix_p ( &func2s[ bli_ker_idx( BLIS_PACKM_DIAG_KER ) ],      packm_diag_ker_name );
	gen_func_init_mix_co( &func2s[ bli_ker_idx( BLIS_PACKM_DIAG_1ER_KER ) ],  packm_diag_1er_ker_name );
	gen_func_init_mix_co( &func2s[ bli_ker_idx( BLIS_PACKM_DIAG_RO_KER ) ],   packm_diag_ro_ker_name );
	gen_func_init_mix_p ( &func2s[ bli_ker_idx( BLIS_UNPACKM_KER ) ],         unpackm_ker_name );


	// -- Put the default kernels and their preferences into the context -------

	for ( dim_t i = 0; i < BLIS_NUM_UKRS; i++ )
		bli_cntx_set_ukr( BLIS_1TYPE_KER + i, &funcs[ i ], cntx );

	for ( dim_t i = 0; i < BLIS_NUM_UKR2S; i++ )
		bli_cntx_set_ukr2( BLIS_2TYPE_KER + i, &func2s[ i ], cntx );

	for ( dim_t i = 0; i < BLIS_NUM_UKR_PREFS; i++ )
		bli_cntx_set_ukr_pref( i, &mbools[ i ], cntx );


	// -- Set level-3 small/unpacked handlers ----------------------------------

	// Initialize all of the function pointers to NULL;
	for ( dim_t i = 0; i < BLIS_NUM_LEVEL3_OPS; ++i )
		vfuncs[ i ] = NULL;

	// The level-3 sup handlers are oapi-based, so we only set one slot per
	// operation.

	// Set the gemm slot to the default gemm sup handler.
	vfuncs[ BLIS_GEMM ]  = bli_gemmsup_ref;
	vfuncs[ BLIS_GEMMT ] = bli_gemmtsup_ref;

	for ( dim_t i = 0; i < BLIS_NUM_LEVEL3_OPS; i++ )
		bli_cntx_set_l3_sup_handler( i, vfuncs[ i ], cntx );
}

