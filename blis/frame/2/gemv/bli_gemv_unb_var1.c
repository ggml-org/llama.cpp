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
       trans_t transa, \
       conj_t  conjx, \
       dim_t   m, \
       dim_t   n, \
       ctype*  alpha, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       ctype*  x, inc_t incx, \
       ctype*  beta, \
       ctype*  y, inc_t incy, \
       cntx_t* cntx  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	ctype*  a1t; \
	ctype*  x1; \
	ctype*  psi1; \
	dim_t   i; \
	dim_t   n_elem, n_iter; \
	inc_t   rs_at, cs_at; \
	conj_t  conja; \
\
	bli_set_dims_incs_with_trans( transa, \
	                              m, n, rs_a, cs_a, \
	                              &n_iter, &n_elem, &rs_at, &cs_at ); \
\
	conja = bli_extract_conj( transa ); \
\
	/* Query the context for the kernel function pointer. */ \
	dotxv_ker_ft kfp_dv = bli_cntx_get_ukr_dt( dt, BLIS_DOTXV_KER, cntx ); \
\
	for ( i = 0; i < n_iter; ++i ) \
	{ \
		a1t  = a + (i  )*rs_at + (0  )*cs_at; \
		x1   = x + (0  )*incy; \
		psi1 = y + (i  )*incy; \
\
		/* psi1 = beta * psi1 + alpha * a1t * x1; */ \
		kfp_dv \
		( \
		  conja, \
		  conjx, \
		  n_elem, \
		  alpha, \
		  a1t, cs_at, \
		  x1,  incx, \
		  beta, \
		  psi1, \
		  cntx  \
		); \
	} \
}

INSERT_GENTFUNC_BASIC( gemv_unb_var1 )

