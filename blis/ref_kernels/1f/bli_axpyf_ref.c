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
#define GENTFUNC( ctype, ch, opname, arch, suf, ff ) \
\
void PASTEMAC(ch,opname,arch,suf) \
     ( \
             conj_t  conja, \
             conj_t  conjx, \
             dim_t   m, \
             dim_t   b_n, \
       const void*   alpha0, \
       const void*   a0, inc_t inca, inc_t lda, \
       const void*   x0, inc_t incx, \
             void*   y0, inc_t incy, \
       const cntx_t* cntx  \
     ) \
{ \
	if ( bli_zero_dim1( m ) ) return; \
\
	const ctype* restrict alpha = alpha0; \
	const ctype* restrict a     = a0; \
	const ctype* restrict x     = x0; \
	      ctype* restrict y     = y0; \
\
	if ( inca == 1 && incx == 1 && incy == 1 && b_n == ff ) \
	{ \
		ctype ax[ ff ]; \
\
		/* Scale x by alpha, storing to a temporary array ax. */ \
		if ( bli_is_conj( conjx ) ) \
		{ \
			PRAGMA_SIMD \
			for ( dim_t j = 0; j < ff; ++j ) \
				bli_tscal2js( ch,ch,ch,ch, *alpha, x[j], ax[j] ); \
		} \
		else \
		{ \
			PRAGMA_SIMD \
			for ( dim_t j = 0; j < ff; ++j ) \
				bli_tscal2s( ch,ch,ch,ch, *alpha, x[j], ax[j] ); \
		} \
\
		/* Accumulate ff separate axpyv's into y. */ \
		if ( bli_is_noconj( conja ) ) \
		{ \
			PRAGMA_SIMD \
			for ( dim_t i = 0; i < m; ++i ) \
			for ( dim_t j = 0; j < ff; ++j ) \
			{ \
				bli_taxpys( ch,ch,ch,ch, ax[j], a[i + j*lda], y[i] ); \
			} \
		} \
		else \
		{ \
			PRAGMA_SIMD \
			for ( dim_t i = 0; i < m; ++i ) \
			for ( dim_t j = 0; j < ff; ++j ) \
			{ \
				bli_taxpyjs( ch,ch,ch,ch, ax[j], a[i + j*lda], y[i] ); \
			} \
		} \
	} \
	else \
	{ \
		/* Query the context for the kernel function pointer. */ \
		const num_t  dt     = PASTEMAC(ch,type); \
		axpyv_ker_ft kfp_av = bli_cntx_get_ukr_dt( dt, BLIS_AXPYV_KER, cntx ); \
\
		for ( dim_t i = 0; i < b_n; ++i ) \
		{ \
			const ctype* restrict a1   = a + (0  )*inca + (i  )*lda; \
			const ctype* restrict chi1 = x + (i  )*incx; \
			      ctype* restrict y1   = y + (0  )*incy; \
\
			ctype alpha_chi1; \
\
			bli_tcopycjs( ch,ch, conjx, *chi1, alpha_chi1 ); \
			bli_tscals( ch,ch,ch, *alpha, alpha_chi1 ); \
\
			kfp_av \
			( \
			  conja, \
			  m, \
			  &alpha_chi1, \
			  a1, inca, \
			  y1, incy, \
			  cntx  \
			); \
		} \
	} \
}

//INSERT_GENTFUNC_BASIC( axpyf, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
GENTFUNC( float,    s, axpyf, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, 8 )
GENTFUNC( double,   d, axpyf, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, 8 )
GENTFUNC( scomplex, c, axpyf, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, 8 )
GENTFUNC( dcomplex, z, axpyf, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, 8 )

