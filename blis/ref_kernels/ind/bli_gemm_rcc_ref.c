/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Southern Methodist University

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

#undef  GENTFUNC2RO
#define GENTFUNC2RO( ctype_ab, ctype_abc, ctype_c, ctype_cc, chab, chabc, chc, chcc, opname, arch, suf ) \
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
	      inc_t       rs_ct; \
	      inc_t       cs_ct; \
\
	const ctype_ab* restrict zero_r = PASTEMAC(chab,0); \
\
	auxinfo_t auxinfo_r = *auxinfo; \
	bli_auxinfo_set_params( params_r, &auxinfo_r ); \
\
	if ( !PASTEMAC(chab,chc,same) ) \
	{ \
		/* In the atypical cases, we compute the result into temporary
		   workspace ct and then accumulated it back to c at the end. */ \
\
		/* Set the strides of ct based on the preference of the underlying
		   native real domain gemm micro-kernel. */ \
		if ( !row_pref ) { rs_ct = 1;  cs_ct = mr; } \
		else             { rs_ct = nr; cs_ct = 1; } \
\
		/* Complex values of alpha will have already been applied during packing */ \
		/* c = beta * c + alpha_r * a * b; */ \
		rgemm_ukr \
		( \
		  mr, \
		  nr, \
		  2*k, \
		  alpha, \
		  a, \
		  b, \
		  zero_r, \
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
	} \
	else \
	{ \
		/* Complex values of alpha will have already been applied during packing */ \
		/* c = beta * c + alpha_r * a * b; */ \
		rgemm_ukr \
		( \
		  m, \
		  n, \
		  2*k, \
		  alpha, \
		  a, \
		  b, \
		  beta, \
		  c, rs_c, cs_c, \
		  &auxinfo_r, \
		  cntx  \
		); \
	} \
}

INSERT_GENTFUNC2RO( gemm_rcc, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
INSERT_GENTFUNC2RO_MIX_P( gemm_rcc, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )

