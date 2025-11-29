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
             conj_t  conjat, \
             conj_t  conjx, \
             dim_t   m, \
             dim_t   b_n, \
       const void*   alpha0, \
       const void*   a0, inc_t inca, inc_t lda, \
       const void*   x0, inc_t incx, \
       const void*   beta0, \
             void*   y0, inc_t incy, \
       const cntx_t* cntx  \
     ) \
{ \
	const ctype* restrict alpha = alpha0; \
	const ctype* restrict a     = a0; \
	const ctype* restrict x     = x0; \
	const ctype* restrict beta  = beta0; \
	      ctype* restrict y     = y0; \
\
	if ( inca == 1 && incx == 1 && incy == 1 && b_n == ff ) \
	{ \
		ctype r[ ff ]; \
\
		/* If beta is zero, clear y. Otherwise, scale by beta. */ \
		if ( bli_teq0s( ch, *beta ) ) \
		{ \
			for ( dim_t i = 0; i < ff; ++i ) bli_tset0s( ch, y[i] ); \
		} \
		else \
		{ \
			for ( dim_t i = 0; i < ff; ++i ) bli_tscals( ch,ch,ch, *beta, y[i] ); \
		} \
\
		/* If the vectors are empty or if alpha is zero, return early. */ \
		if ( bli_zero_dim1( m ) || bli_teq0s( ch, *alpha ) ) return; \
\
		/* Initialize r vector to 0. */ \
		for ( dim_t i = 0; i < ff; ++i ) bli_tset0s( ch, r[i] ); \
\
		/* If a must be conjugated, we do so indirectly by first toggling the
		   effective conjugation of x and then conjugating the resulting dot
		   products. */ \
		conj_t conjx_use = conjx; \
\
		if ( bli_is_conj( conjat ) ) \
			bli_toggle_conj( &conjx_use ); \
\
		if ( bli_is_noconj( conjx_use ) ) \
		{ \
			PRAGMA_SIMD \
			for ( dim_t p = 0; p < m; ++p ) \
			for ( dim_t i = 0; i < ff; ++i ) \
			{ \
				bli_taxpys( ch,ch,ch,ch, a[p + i*lda], x[p], r[i] ); \
			} \
		} \
		else \
		{ \
			PRAGMA_SIMD \
			for ( dim_t p = 0; p < m; ++p ) \
			for ( dim_t i = 0; i < ff; ++i ) \
			{ \
				bli_taxpyjs( ch,ch,ch,ch, a[p + i*lda], x[p], r[i] ); \
			} \
		} \
\
		if ( bli_is_conj( conjat ) ) \
			for ( dim_t i = 0; i < ff; ++i ) bli_tconjs( ch, r[i] ); \
\
		for ( dim_t i = 0; i < ff; ++i ) \
		{ \
			bli_taxpys( ch,ch,ch,ch, *alpha, r[i], y[i] ); \
		} \
	} \
	else \
	{ \
		/* Query the context for the kernel function pointer. */ \
		const num_t  dt     = PASTEMAC(ch,type); \
		dotxv_ker_ft kfp_dv = bli_cntx_get_ukr_dt( dt, BLIS_DOTXV_KER, cntx ); \
\
		for ( dim_t i = 0; i < b_n; ++i ) \
		{ \
			const ctype* restrict a1   = a + (0  )*inca + (i  )*lda; \
			const ctype* restrict x1   = x + (0  )*incx; \
			      ctype* restrict psi1 = y + (i  )*incy; \
\
			kfp_dv \
			( \
			  conjat, \
			  conjx, \
			  m, \
			  alpha, \
			  a1, inca, \
			  x1, incx, \
			  beta, \
			  psi1, \
			  cntx  \
			); \
		} \
	} \
}

//INSERT_GENTFUNC_BASIC( dotxf, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
GENTFUNC( float,    s, dotxf, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, 6 )
GENTFUNC( double,   d, dotxf, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, 6 )
GENTFUNC( scomplex, c, dotxf, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, 6 )
GENTFUNC( dcomplex, z, dotxf, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, 6 )

