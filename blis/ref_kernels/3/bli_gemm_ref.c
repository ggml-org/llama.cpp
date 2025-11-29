/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2022-2025, Southern Methodist University

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

// Completely generic gemm ukr implementation which checks MR/NR at
// runtime. Very slow, but has to be used in certain cases.

#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, opname, arch, suf ) \
\
static void PASTEMAC(ch,ch,opname,arch,suf) \
     ( \
             dim_t      m, \
             dim_t      n, \
             dim_t      k, \
       const void*      alpha0, \
       const void*      a0, \
       const void*      b0, \
       const void*      beta0, \
             void*      c0, inc_t rs_c, inc_t cs_c, \
       const auxinfo_t* data, \
       const cntx_t*    cntx  \
     ) \
{ \
	const ctype* alpha = alpha0; \
	const ctype* a     = a0; \
	const ctype* b     = b0; \
	const ctype* beta  = beta0; \
	      ctype* c     = c0; \
\
	const num_t dt     = PASTEMAC(ch,type); \
\
	const inc_t packmr = bli_cntx_get_blksz_max_dt( dt, BLIS_MR, cntx ); \
	const inc_t packnr = bli_cntx_get_blksz_max_dt( dt, BLIS_NR, cntx ); \
\
	const inc_t rs_a   = bli_cntx_get_blksz_def_dt( dt, BLIS_BBM, cntx ); \
	const inc_t cs_a   = packmr; \
\
	const inc_t rs_b   = packnr; \
	const inc_t cs_b   = bli_cntx_get_blksz_def_dt( dt, BLIS_BBN, cntx ); \
\
	ctype       ab[ BLIS_STACK_BUF_MAX_SIZE \
	                / sizeof( ctype ) ] \
	                __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE))); \
	const inc_t rs_ab  = 1; \
	const inc_t cs_ab  = m; \
\
	/* Initialize the accumulator elements in ab to zero. */ \
	for ( dim_t i = 0; i < m * n; ++i ) \
	{ \
		bli_tset0s( ch, *(ab + i) ); \
	} \
\
	/* Perform a series of k rank-1 updates into ab. */ \
	for ( dim_t l = 0; l < k; ++l ) \
	{ \
		ctype* restrict abij = ab; \
\
		/* In an optimized implementation, these two loops over MR and NR
		   are typically fully unrolled. */ \
		for ( dim_t j = 0; j < n; ++j ) \
		{ \
			ctype bj; \
			const ctype_r* b_r = (const ctype_r*)(b + j*cs_b); \
			const ctype_r* b_i = b_r + cs_b; (void)b_i; \
			bli_tsets( ch,ch, *b_r, *b_i, bj ); \
\
			for ( dim_t i = 0; i < m; ++i ) \
			{ \
				ctype ai; \
				const ctype_r* a_r = (const ctype_r*)(a + i*rs_a); \
				const ctype_r* a_i = a_r + rs_a; (void)a_i; \
				bli_tsets( ch,ch, *a_r, *a_i, ai ); \
\
				bli_tdots( ch,ch,ch,ch, ai, bj, *abij ); \
\
				abij += rs_ab; \
			} \
		} \
\
		a += cs_a; \
		b += rs_b; \
	} \
\
	/* Scale the result in ab by alpha. */ \
	for ( dim_t i = 0; i < m * n; ++i ) \
	{ \
		bli_tscals( ch,ch,ch, *alpha, *(ab + i) ); \
	} \
\
	/* If beta is zero, overwrite c with the scaled result in ab. Otherwise,
	   scale by beta and then add the scaled redult in ab. */ \
	if ( bli_teq0s( ch, *beta ) ) \
	{ \
		bli_tcopys_mxn \
		( \
		  ch, \
		  ch, \
		  m, \
		  n, \
		  ab, rs_ab, cs_ab, \
		  c,  rs_c,  cs_c \
		); \
	} \
	else \
	{ \
		bli_txpbys_mxn \
		( \
		  ch, \
		  ch, \
		  ch, \
		  ch, \
		  m, \
		  n, \
		  ab, rs_ab, cs_ab, \
		  beta, \
		  c,  rs_c,  cs_c \
		); \
	} \
}

INSERT_GENTFUNCR_BASIC( gemm_gen, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )

// An implementation that attempts to facilitate emission of vectorized
// instructions via constant loop bounds + #pragma omp simd directives.
// If compile-time MR/NR are not available (indicated by BLIS_[MN]R_x = -1),
// then the non-unrolled version (above) is used.

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, arch, suf ) \
\
void PASTEMAC(ch,ch,opname,arch,suf) \
     ( \
             dim_t      m, \
             dim_t      n, \
             dim_t      k, \
       const void*      alpha0, \
       const void*      a0, \
       const void*      b0, \
       const void*      beta0, \
             void*      c0, inc_t rs_c, inc_t cs_c, \
       const auxinfo_t* data, \
       const cntx_t*    cntx  \
     ) \
{ \
	const ctype* alpha = alpha0; \
	const ctype* a     = a0; \
	const ctype* b     = b0; \
	const ctype* beta  = beta0; \
	      ctype* c     = c0; \
\
	const dim_t mr = PASTECH(BLIS_MR_,ch); \
	const dim_t nr = PASTECH(BLIS_NR_,ch); \
\
	const inc_t rs_a   = PASTECH(BLIS_BBM_,ch); \
	const inc_t cs_a   = PASTECH(BLIS_PACKMR_,ch); \
	const inc_t rs_b   = PASTECH(BLIS_PACKNR_,ch); \
	const inc_t cs_b   = PASTECH(BLIS_BBN_,ch); \
\
	/* If either BLIS_MR_? or BLIS_NR_? was left undefined by the subconfig,
	   the compiler can't fully unroll the MR and NR loop iterations below,
	   which means there's no benefit to using this kernel over a general-
	   purpose implementation instead. */ \
	if ( mr == -1 || nr == -1 || rs_a != 1 || cs_b != 1 ) \
	{ \
		PASTEMAC(ch,ch,gemm_gen,arch,suf) \
		( \
		  m, \
		  n, \
		  k, \
		  alpha, \
		  a, \
		  b, \
		  beta, \
		  c, rs_c, cs_c, \
		  data, \
		  cntx \
		); \
		return; \
	} \
\
	      char   ab_[ BLIS_STACK_BUF_MAX_SIZE ] __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE))) = { 0 }; \
	      ctype* ab    = (ctype*)ab_; \
	const inc_t  rs_ab = nr; \
	const inc_t  cs_ab = 1; \
\
\
	/* Initialize the accumulator elements in ab to zero. */ \
	PRAGMA_SIMD \
	for ( dim_t i = 0; i < mr * nr; ++i ) \
	{ \
		bli_tset0s( ch, ab[ i ] ); \
	} \
\
	/* Perform a series of k rank-1 updates into ab. */ \
	for ( dim_t l = 0; l < k; ++l ) \
	{ \
		for ( dim_t i = 0; i < mr; ++i ) \
		{ \
			PRAGMA_SIMD \
			for ( dim_t j = 0; j < nr; ++j ) \
			{ \
				bli_tdots \
				( \
				  ch,ch,ch,ch, \
				  a[ i*rs_a ], \
				  b[ j*cs_b ], \
				  ab[ i*rs_ab + j*cs_ab ]  \
				); \
			} \
		} \
\
		a += cs_a; \
		b += rs_b; \
	} \
\
	/* Scale the result in ab by alpha. */ \
	PRAGMA_SIMD \
	for ( dim_t i = 0; i < mr * nr; ++i ) \
	{ \
		bli_tscals( ch,ch,ch, *alpha, ab[ i ] ); \
	} \
\
	/* Output/accumulate intermediate result ab based on the storage
	   of c and the value of beta. */ \
	if ( cs_c == 1 ) \
	{ \
		/* C is row-stored. */ \
\
		if ( bli_teq0s( ch, *beta ) ) \
		{ \
			for ( dim_t i = 0; i < m; ++i ) \
			for ( dim_t j = 0; j < n; ++j ) \
			bli_tcopys \
			( \
			  ch,ch, \
			  ab[ i*rs_ab + j*cs_ab ], \
			  c [ i*rs_c  + j*1     ]  \
			); \
		} \
		else \
		{ \
			for ( dim_t i = 0; i < m; ++i ) \
			for ( dim_t j = 0; j < n; ++j ) \
			bli_txpbys \
			( \
			  ch,ch,ch,ch, \
			  ab[ i*rs_ab + j*cs_ab ], \
			  *beta, \
			  c [ i*rs_c  + j*1     ]  \
			); \
		} \
	} \
	else \
	{ \
		/* C is column-stored or general-stored. */ \
\
		if (bli_teq0s( ch, *beta ) ) \
		{ \
			for ( dim_t j = 0; j < n; ++j ) \
			for ( dim_t i = 0; i < m; ++i ) \
			bli_tcopys \
			( \
			  ch,ch, \
			  ab[ i*rs_ab + j*cs_ab ], \
			  c [ i*rs_c  + j*cs_c  ]  \
			); \
		} \
		else \
		{ \
			for ( dim_t j = 0; j < n; ++j ) \
			for ( dim_t i = 0; i < m; ++i ) \
			bli_txpbys \
			( \
			  ch,ch,ch,ch, \
			  ab[ i*rs_ab + j*cs_ab ], \
			  *beta, \
			  c [ i*rs_c  + j*cs_c  ]  \
			); \
		} \
	} \
}

INSERT_GENTFUNC_BASIC( gemm, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )

// Mixed-precision implementation (does not handle mixed-domain cases)

#undef  GENTFUNC2
#define GENTFUNC2( ctype_ab, ctype_c, chab, chc, opname, arch, suf ) \
\
void PASTEMAC(chab,chc,opname,arch,suf) \
     ( \
             dim_t      m, \
             dim_t      n, \
             dim_t      k, \
       const void*      alpha, \
       const void*      a, \
       const void*      b, \
       const void*      beta0, \
             void*      c0, inc_t rs_c, inc_t cs_c, \
       const auxinfo_t* auxinfo, \
       const cntx_t*    cntx  \
     ) \
{ \
	const ctype_c*    beta      = beta0; \
	      ctype_c*    c         = c0; \
\
	const cntl_t*     params    = bli_auxinfo_params( auxinfo ); \
\
	const gemm_ukr_ft rgemm_ukr = bli_gemm_var_cntl_real_ukr( params ); \
	const bool        row_pref  = bli_gemm_var_cntl_row_pref( params ); \
	const void*       params_r  = bli_gemm_var_cntl_real_params( params ); \
\
	const dim_t       mr        = bli_gemm_var_cntl_mr( params ); \
	const dim_t       nr        = bli_gemm_var_cntl_nr( params ); \
\
	      ctype_ab    ct[ BLIS_STACK_BUF_MAX_SIZE / sizeof( ctype_ab ) ] \
	                  __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE))); \
	const inc_t       rs_ct     = row_pref ? nr : 1; \
	const inc_t       cs_ct     = row_pref ? 1 : mr; \
\
	const ctype_ab*   zero      = PASTEMAC(chab,0); \
\
	auxinfo_t auxinfo_r = *auxinfo; \
	bli_auxinfo_set_params( params_r, &auxinfo_r ); \
\
	/* ab = alpha * a * b; */ \
	rgemm_ukr \
	( \
	  mr, \
	  nr, \
	  k, \
	  alpha, \
	  a, \
	  b, \
	  zero, \
	  ct, rs_ct, cs_ct, \
	  &auxinfo_r, \
	  cntx  \
	); \
\
	bli_txpbys_mxn \
	( \
	  chab,chc,chc,chc, \
	  m, n, \
	  ct, rs_ct, cs_ct, \
	  beta, \
	  c, rs_c, cs_c \
	); \
}

INSERT_GENTFUNC2_MIX_P( gemm, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )


