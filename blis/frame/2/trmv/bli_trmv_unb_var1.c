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
       uplo_t  uploa, \
       trans_t transa, \
       diag_t  diaga, \
       dim_t   m, \
       ctype*  alpha, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       ctype*  x, inc_t incx, \
       cntx_t* cntx  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	ctype*  a10t; \
	ctype*  alpha11; \
	ctype*  a12t; \
	ctype*  x0; \
	ctype*  chi1; \
	ctype*  x2; \
	ctype   alpha_alpha11_conj; \
	ctype   rho; \
	dim_t   iter, i; \
	dim_t   n_ahead; \
	inc_t   rs_at, cs_at; \
	uplo_t  uploa_trans; \
	conj_t  conja; \
\
	if      ( bli_does_notrans( transa ) ) \
	{ \
		rs_at = rs_a; \
		cs_at = cs_a; \
		uploa_trans = uploa; \
	} \
	else /* if ( bli_does_trans( transa ) ) */ \
	{ \
		rs_at = cs_a; \
		cs_at = rs_a; \
		uploa_trans = bli_uplo_toggled( uploa ); \
	} \
\
	conja = bli_extract_conj( transa ); \
\
	/* Query the context for the kernel function pointer. */ \
	dotv_ker_ft kfp_dv = bli_cntx_get_ukr_dt( dt, BLIS_DOTV_KER, cntx ); \
\
	/* We reduce all of the possible cases down to just lower/upper. */ \
	if      ( bli_is_upper( uploa_trans ) ) \
	{ \
		for ( iter = 0; iter < m; ++iter ) \
		{ \
			i        = iter; \
			n_ahead  = m - iter - 1; \
			alpha11  = a + (i  )*rs_at + (i  )*cs_at; \
			a12t     = a + (i  )*rs_at + (i+1)*cs_at; \
			chi1     = x + (i  )*incx; \
			x2       = x + (i+1)*incx; \
\
			/* chi1 = alpha * alpha11 * chi1; */ \
			bli_tcopys( ch,ch, *alpha, alpha_alpha11_conj ); \
			if ( bli_is_nonunit_diag( diaga ) ) \
				bli_tscalcjs( ch,ch,ch, conja, *alpha11, alpha_alpha11_conj ); \
			bli_tscals( ch,ch,ch, alpha_alpha11_conj, *chi1 ); \
\
			/* chi1 = chi1 + alpha * a12t * x2; */ \
			kfp_dv \
			( \
			  conja, \
			  BLIS_NO_CONJUGATE, \
			  n_ahead, \
			  a12t, cs_at, \
			  x2,   incx, \
			  &rho, \
			  cntx  \
			); \
			bli_taxpys( ch,ch,ch,ch, *alpha, rho, *chi1 ); \
		} \
	} \
	else /* if ( bli_is_lower( uploa_trans ) ) */ \
	{ \
		for ( iter = 0; iter < m; ++iter ) \
		{ \
			i        = m - iter - 1; \
			n_ahead  = i; \
			alpha11  = a + (i  )*rs_at + (i  )*cs_at; \
			a10t     = a + (i  )*rs_at + (0  )*cs_at; \
			chi1     = x + (i  )*incx; \
			x0       = x + (0  )*incx; \
\
			/* chi1 = alpha * alpha11 * chi1; */ \
			bli_tcopys( ch,ch, *alpha, alpha_alpha11_conj ); \
			if ( bli_is_nonunit_diag( diaga ) ) \
				bli_tscalcjs( ch,ch,ch, conja, *alpha11, alpha_alpha11_conj ); \
			bli_tscals( ch,ch,ch, alpha_alpha11_conj, *chi1 ); \
\
			/* chi1 = chi1 + alpha * a10t * x0; */ \
			kfp_dv \
			( \
			  conja, \
			  BLIS_NO_CONJUGATE, \
			  n_ahead, \
			  a10t, cs_at, \
			  x0,   incx, \
			  &rho, \
			  cntx  \
			); \
			bli_taxpys( ch,ch,ch,ch, *alpha, rho, *chi1 ); \
		} \
	} \
}

INSERT_GENTFUNC_BASIC( trmv_unb_var1 )

