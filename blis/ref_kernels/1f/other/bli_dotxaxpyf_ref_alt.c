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
       conj_t          conjat, \
       conj_t          conja, \
       conj_t          conjw, \
       conj_t          conjx, \
       dim_t           m, \
       dim_t           b_n, \
       ctype* restrict alpha, \
       ctype* restrict a, inc_t inca, inc_t lda, \
       ctype* restrict w, inc_t incw, \
       ctype* restrict x, inc_t incx, \
       ctype* restrict beta, \
       ctype* restrict y, inc_t incy, \
       ctype* restrict z, inc_t incz, \
       cntx_t*         cntx  \
     ) \
{ \
	ctype* a1; \
	ctype* chi1; \
	ctype* w1; \
	ctype* psi1; \
	ctype* z1; \
	ctype  conjx_chi1; \
	ctype  alpha_chi1; \
	dim_t  i; \
\
	/* Query the context for the kernel function pointer. */ \
	const num_t          dt     = PASTEMAC(ch,type); \
	PASTECH(ch,dotxv_ft) kfp_dv = bli_cntx_get_ukr_dt( dt, BLIS_DOTXV_KER, cntx ); \
	PASTECH(ch,axpyv_ft) kfp_av = bli_cntx_get_ukr_dt( dt, BLIS_AXPYV_KER, cntx ); \
\
	/* A is m x n.                   */ \
	/* y = beta * y + alpha * A^T w; */ \
	/* z =        z + alpha * A   x; */ \
	for ( i = 0; i < b_n; ++i ) \
	{ \
		a1   = a + (0  )*inca + (i  )*lda; \
		w1   = w + (0  )*incw; \
		psi1 = y + (i  )*incy; \
\
		kfp_dv \
		( \
		  conjat, \
		  conjw, \
		  m, \
		  alpha, \
		  a1, inca, \
		  w1, incw, \
		  beta, \
		  psi1, \
		  cntx  \
		); \
	} \
\
	for ( i = 0; i < b_n; ++i ) \
	{ \
		a1   = a + (0  )*inca + (i  )*lda; \
		chi1 = x + (i  )*incx; \
		z1   = z + (0  )*incz; \
\
		PASTEMAC(ch,copycjs)( conjx, *chi1, conjx_chi1 ); \
		PASTEMAC(ch,scal2s)( *alpha, conjx_chi1, alpha_chi1 ); \
\
		kfp_av \
		( \
		  conja, \
		  m, \
		  &alpha_chi1, \
		  a1, inca, \
		  z1, incz, \
		  cntx  \
		); \
	} \
}

INSERT_GENTFUNC_BASIC0( dotxaxpyf_ref_var1 )

