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
       conj_t  conjx, \
       conj_t  conjy, \
       dim_t   m, \
       dim_t   n, \
       ctype*  alpha, \
       ctype*  x, inc_t incx, \
       ctype*  y, inc_t incy, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       cntx_t* cntx  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	ctype*  a1; \
	ctype*  x1; \
	ctype*  psi1; \
	ctype   alpha_psi1; \
	dim_t   j; \
\
	/* Query the context for the kernel function pointer. */ \
	axpyv_ker_ft kfp_av = bli_cntx_get_ukr_dt( dt, BLIS_AXPYV_KER, cntx ); \
\
	for ( j = 0; j < n; ++j ) \
	{ \
		a1   = a + (0  )*rs_a + (j  )*cs_a; \
		x1   = x + (0  )*incx; \
		psi1 = y + (j  )*incy; \
\
		/* a1 = a1 + alpha * psi1 * x; */ \
		bli_tcopycjs( ch,ch, conjy, *psi1, alpha_psi1 ); \
		bli_tscals( ch,ch,ch, *alpha, alpha_psi1 ); \
\
		kfp_av \
		( \
		  conjx, \
		  m, \
		  &alpha_psi1, \
		  x1, incx, \
		  a1, rs_a, \
		  cntx  \
		); \
	} \
}

INSERT_GENTFUNC_BASIC( ger_unb_var2 )

