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
       conj_t  conjy, \
       conj_t  conjh, \
       dim_t   m, \
       ctype*  alpha, \
       ctype*  x, inc_t incx, \
       ctype*  y, inc_t incy, \
       ctype*  c, inc_t rs_c, inc_t cs_c, \
       cntx_t* cntx  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	ctype*  chi1; \
	ctype*  y0; \
	ctype*  psi1; \
	ctype*  y2; \
	ctype*  c10t; \
	ctype*  gamma11; \
	ctype*  c21; \
	ctype   alpha0; \
	ctype   alpha1; \
	ctype   alpha0_chi1; \
	ctype   alpha1_chi1; \
	ctype   alpha0_chi1_psi1; \
	ctype   conjx0_chi1; \
	ctype   conjx1_chi1; \
	ctype   conjy0_psi1; \
	dim_t   i; \
	dim_t   n_behind; \
	dim_t   n_ahead; \
	inc_t   rs_ct, cs_ct; \
	conj_t  conj0, conj1; \
	conj_t  conjh_conjx; \
\
	/* Eliminate unused variable warnings. */ \
	( void )conjh_conjx; \
\
	/* The algorithm will be expressed in terms of the lower triangular case;
	   the upper triangular case is supported by swapping the row and column
	   strides of A and toggling some conj parameters. */ \
	if      ( bli_is_lower( uplo ) ) \
	{ \
		rs_ct = rs_c; \
		cs_ct = cs_c; \
\
		bli_tcopys( ch,ch, *alpha, alpha0 ); \
		bli_tcopycjs( ch,ch, conjh, *alpha, alpha1 ); \
	} \
	else /* if ( bli_is_upper( uplo ) ) */ \
	{ \
		rs_ct = cs_c; \
		cs_ct = rs_c; \
\
		/* Toggle conjugation of conjx/conjy, but only if we are being invoked
		   as her2; for syr2, conjx/conjy are unchanged. */ \
		conjx = bli_apply_conj( conjh, conjx ); \
		conjy = bli_apply_conj( conjh, conjy ); \
\
		bli_tcopycjs( ch,ch, conjh, *alpha, alpha0 ); \
		bli_tcopys( ch,ch, *alpha, alpha1 ); \
	} \
\
	/* Apply conjh (which carries the conjugation component of the Hermitian
	   transpose, if applicable) to conjx and/or conjy as needed to arrive at
	   the effective conjugation for the vector subproblems. */ \
	conj0       = bli_apply_conj( conjh, conjy ); \
	conj1       = conjy; \
	conjh_conjx = bli_apply_conj( conjh, conjx ); \
\
	/* Query the context for the kernel function pointer. */ \
	axpyv_ker_ft kfp_av = bli_cntx_get_ukr_dt( dt, BLIS_AXPYV_KER, cntx ); \
\
	for ( i = 0; i < m; ++i ) \
	{ \
		n_behind = i; \
		n_ahead  = m - i - 1; \
		chi1     = x + (i  )*incx; \
		y0       = y + (0  )*incy; \
		psi1     = y + (i  )*incy; \
		y2       = y + (i+1)*incy; \
		c10t     = c + (i  )*rs_ct + (0  )*cs_ct; \
		gamma11  = c + (i  )*rs_ct + (i  )*cs_ct; \
		c21      = c + (i+1)*rs_ct + (i  )*cs_ct; \
\
		/* Apply conjx and/or conjy to chi1 and/or psi1. */ \
		bli_tcopycjs( ch,ch, conjx,       *chi1, conjx0_chi1 ); \
		bli_tcopycjs( ch,ch, conjh_conjx, *chi1, conjx1_chi1 ); \
		bli_tcopycjs( ch,ch, conj0,       *psi1, conjy0_psi1 ); \
\
		/* Compute scalars for vector subproblems. */ \
		bli_tscal2s( ch,ch,ch,ch, alpha0, conjx0_chi1, alpha0_chi1 ); \
		bli_tscal2s( ch,ch,ch,ch, alpha1, conjx1_chi1, alpha1_chi1 ); \
\
		/* Compute alpha * chi1 * conj(psi1) after both chi1 and psi1 have
		   already been conjugated, if needed, by conjx and conjy. */ \
		bli_tscal2s( ch,ch,ch,ch, alpha0_chi1, conjy0_psi1, alpha0_chi1_psi1 ); \
\
		/* c10t = c10t + alpha * chi1 * y0'; */ \
		kfp_av \
		( \
		  conj0, \
		  n_behind, \
		  &alpha0_chi1, \
		  y0,   incy, \
		  c10t, cs_ct, \
		  cntx  \
		); \
\
		/* c21 = c21 + conj(alpha) * y2 * conj(chi1); */ \
		kfp_av \
		( \
		  conj1, \
		  n_ahead, \
		  &alpha1_chi1, \
		  y2,  incy, \
		  c21, rs_ct, \
		  cntx  \
		); \
\
		/* gamma11 = gamma11 +      alpha  * chi1 * conj(psi1) \
		                     + conj(alpha) * psi1 * conj(chi1); */ \
		bli_tadds( ch,ch,ch, alpha0_chi1_psi1, *gamma11 ); \
		bli_tadds( ch,ch,ch, alpha0_chi1_psi1, *gamma11 ); \
\
		/* For her2, explicitly set the imaginary component of gamma11 to
           zero. */ \
		if ( bli_is_conj( conjh ) ) \
			bli_tseti0s( ch, *gamma11 ); \
	} \
}

INSERT_GENTFUNC_BASIC( her2_unb_var3 )

