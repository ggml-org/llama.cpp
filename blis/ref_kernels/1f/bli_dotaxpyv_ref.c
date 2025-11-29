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
#define GENTFUNC( ctype, ch, opname, arch, suf ) \
\
void PASTEMAC(ch,opname,arch,suf) \
     ( \
             conj_t  conjxt, \
             conj_t  conjx, \
             conj_t  conjy, \
             dim_t   m, \
       const void*   alpha0, \
       const void*   x0, inc_t incx, \
       const void*   y0, inc_t incy, \
             void*   rho0, \
             void*   z0, inc_t incz, \
       const cntx_t* cntx  \
     ) \
{ \
	if ( bli_zero_dim1( m ) ) return; \
\
	const ctype* restrict alpha = alpha0; \
	const ctype* restrict x     = x0; \
	const ctype* restrict y     = y0; \
	      ctype* restrict rho   = rho0; \
	      ctype* restrict z     = z0; \
\
	if ( incz == 1 && incx == 1 && incy == 1 ) \
	{ \
		if ( bli_is_noconj( conjx ) ) \
		{ \
			conj_t conjxt_use = conjxt; \
			ctype  dotxy; \
\
			bli_tset0s( ch, dotxy ); \
\
			if ( bli_is_conj( conjy ) ) \
				bli_toggle_conj( &conjxt_use ); \
\
			if ( bli_is_noconj( conjxt_use ) ) \
			{ \
				PRAGMA_SIMD \
				for ( dim_t i = 0; i < m; ++i ) \
				{ \
					bli_tdots( ch,ch,ch,ch, x[i], y[i], dotxy ); \
					bli_taxpys( ch,ch,ch,ch, *alpha, x[i], z[i] ); \
				} \
			} \
			else /* bli_is_conj( conjxt_use ) ) */ \
			{ \
				PRAGMA_SIMD \
				for ( dim_t i = 0; i < m; ++i ) \
				{ \
					bli_tdotjs( ch,ch,ch,ch, x[i], y[i], dotxy ); \
					bli_taxpys( ch,ch,ch,ch, *alpha, x[i], z[i] ); \
				} \
			} \
\
			if ( bli_is_conj( conjy ) ) \
				bli_tconjs( ch, dotxy ); \
\
			bli_tcopys( ch,ch, dotxy, *rho ); \
		} \
		else /* bli_is_conj( conjx ) ) */ \
		{ \
			conj_t conjxt_use = conjxt; \
			ctype  dotxy; \
\
			bli_tset0s( ch, dotxy ); \
\
			if ( bli_is_conj( conjy ) ) \
				bli_toggle_conj( &conjxt_use ); \
\
			if ( bli_is_noconj( conjxt_use ) ) \
			{ \
				PRAGMA_SIMD \
				for ( dim_t i = 0; i < m; ++i ) \
				{ \
					bli_tdots( ch,ch,ch,ch, x[i], y[i], dotxy ); \
					bli_taxpyjs( ch,ch,ch,ch, *alpha, x[i], z[i] ); \
				} \
			} \
			else /* bli_is_conj( conjxt_use ) ) */ \
			{ \
				PRAGMA_SIMD \
				for ( dim_t i = 0; i < m; ++i ) \
				{ \
					bli_tdotjs( ch,ch,ch,ch, x[i], y[i], dotxy ); \
					bli_taxpyjs( ch,ch,ch,ch, *alpha, x[i], z[i] ); \
				} \
			} \
\
			if ( bli_is_conj( conjy ) ) \
				bli_tconjs( ch, dotxy ); \
\
			bli_tcopys( ch,ch, dotxy, *rho ); \
		} \
	} \
	else \
	{ \
\
		/* Query the context for the kernel function pointer. */ \
		const num_t  dt     = PASTEMAC(ch,type); \
		dotv_ker_ft  kfp_dv = bli_cntx_get_ukr_dt( dt, BLIS_DOTV_KER, cntx ); \
		axpyv_ker_ft kfp_av = bli_cntx_get_ukr_dt( dt, BLIS_AXPYV_KER, cntx ); \
\
		kfp_dv \
		( \
		  conjxt, \
		  conjy, \
		  m, \
		  x, incx, \
		  y, incy, \
		  rho, \
		  cntx  \
		); \
\
		kfp_av \
		( \
		  conjx, \
		  m, \
		  alpha, \
		  x, incx, \
		  z, incz, \
		  cntx  \
		); \
	} \
}

INSERT_GENTFUNC_BASIC( dotaxpyv, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )

