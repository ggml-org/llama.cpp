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

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       uplo_t  uplo, \
       conj_t  conjx, \
       conj_t  conjh, \
       dim_t   m, \
       ctype*  alpha, /* complex alpha allows her variants to also perform syr. */ \
       ctype*  x, inc_t incx, \
       ctype*  c, inc_t rs_c, inc_t cs_c, \
       cntx_t* cntx  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	ctype*  x0; \
	ctype*  chi1; \
	ctype*  c10t; \
	ctype*  gamma11; \
	ctype   alpha_local; \
	ctype   alpha_chi1; \
	ctype   alpha_chi1_chi1; \
	ctype   conjx0_chi1; \
	ctype   conjx1_chi1; \
	dim_t   i; \
	dim_t   n_behind; \
	inc_t   rs_ct, cs_ct; \
	conj_t  conj0, conj1; \
\
	/* Eliminate unused variable warnings. */ \
	( void )conj0; \
\
	/* Make a local copy of alpha and zero out the imaginary component if
	   we are being invoked as her, since her requires alpha to be real. */ \
	bli_tcopys( ch,ch, *alpha, alpha_local ); \
	if ( bli_is_conj( conjh ) ) \
	{ \
		bli_tseti0s( ch, alpha_local ); \
	} \
\
	/* The algorithm will be expressed in terms of the lower triangular case;
	   the upper triangular case is supported by swapping the row and column
	   strides of A and toggling some conj parameters. */ \
	if      ( bli_is_lower( uplo ) ) \
	{ \
		rs_ct = rs_c; \
		cs_ct = cs_c; \
	} \
	else /* if ( bli_is_upper( uplo ) ) */ \
	{ \
		rs_ct = cs_c; \
		cs_ct = rs_c; \
\
		/* Toggle conjugation of conjx, but only if we are being invoked
		   as her; for syr, conjx is unchanged. */ \
		conjx = bli_apply_conj( conjh, conjx ); \
	} \
\
	/* Apply conjh (which carries the conjugation component of the Hermitian
       transpose, if applicable) to conjx as needed to arrive at the effective
	   conjugation for the scalar and vector subproblems. */ \
	conj0 = conjx; \
	conj1 = bli_apply_conj( conjh, conjx ); \
\
	/* Query the context for the kernel function pointer. */ \
	axpyv_ker_ft kfp_av = bli_cntx_get_ukr_dt( dt, BLIS_AXPYV_KER, cntx ); \
\
	for ( i = 0; i < m; ++i ) \
	{ \
		n_behind = i; \
		x0       = x + (0  )*incx; \
		chi1     = x + (i  )*incx; \
		c10t     = c + (i  )*rs_ct + (0  )*cs_ct; \
		gamma11  = c + (i  )*rs_ct + (i  )*cs_ct; \
\
		/* Apply conjx to chi1. */ \
		bli_tcopycjs( ch,ch, conj0, *chi1, conjx0_chi1 ); \
		bli_tcopycjs( ch,ch, conj1, *chi1, conjx1_chi1 ); \
\
		/* Compute scalar for vector subproblem. */ \
		bli_tscal2s( ch,ch,ch,ch, alpha_local, conjx0_chi1, alpha_chi1 ); \
\
		/* Compute alpha * chi1 * conj(chi1) after chi1 has already been
		   conjugated, if needed, by conjx. */ \
		bli_tscal2s( ch,ch,ch,ch, alpha_chi1, conjx1_chi1, alpha_chi1_chi1 ); \
\
		/* c10t = c10t + alpha * chi1 * x0'; */ \
		kfp_av \
		( \
		  conj1, \
		  n_behind, \
		  &alpha_chi1, \
		  x0,   incx, \
		  c10t, cs_ct, \
		  cntx  \
		); \
\
		/* gamma11 = gamma11 + alpha * chi1 * conj(chi1); */ \
		bli_tadds( ch,ch,ch, alpha_chi1_chi1, *gamma11 ); \
\
		/* For her2, explicitly set the imaginary component of gamma11 to
		   zero. */ \
		if ( bli_is_conj( conjh ) ) \
			bli_tseti0s( ch, *gamma11 ); \
	} \
}

INSERT_GENTFUNC_BASIC( her_unb_var1 )

