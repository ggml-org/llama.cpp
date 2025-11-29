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

#if 1

// An implementation that attempts to facilitate emission of vectorized
// instructions via constant loop bounds + #pragma omp simd directives.

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, arch, suf, mr, nr ) \
\
void PASTEMAC3(ch,opname,arch,suf) \
     ( \
       ctype*     restrict a, \
       ctype*     restrict b, \
       ctype*     restrict c, inc_t rs_c, inc_t cs_c, \
       auxinfo_t* restrict data, \
       cntx_t*    restrict cntx  \
     ) \
{ \
	const inc_t     rs_a   = 1; \
	const inc_t     cs_a   = mr; \
\
	const inc_t     rs_b   = nr; \
	const inc_t     cs_b   = 1; \
\
	PRAGMA_SIMD \
	for ( dim_t i = 0; i < mr; ++i ) \
	{ \
		/* b1 = b1 - a10t * B0; */ \
		/* b1 = b1 / alpha11; */ \
		for ( dim_t j = 0; j < nr; ++j ) \
		{ \
			ctype beta11c = b[i*rs_b + j*cs_b]; \
			ctype rho11; \
\
			/* beta11 = beta11 - a10t * b01; */ \
			PASTEMAC(ch,set0s)( rho11 ); \
			for ( dim_t l = 0; l < i; ++l ) \
			{ \
				PASTEMAC(ch,axpys)( a[i*rs_a + l*cs_a], \
				                    b[l*rs_b + j*cs_b], rho11 ); \
			} \
			PASTEMAC(ch,subs)( rho11, beta11c ); \
\
			/* beta11 = beta11 / alpha11; */ \
			/* NOTE: The INVERSE of alpha11 (1.0/alpha11) is stored instead
			   of alpha11, so we can multiply rather than divide. We store
			   the inverse of alpha11 intentionally to avoid expensive
			   division instructions within the micro-kernel. */ \
			PASTEMAC(ch,scals)( a[i*rs_a + i*cs_a], beta11c ); \
\
			/* Output final result to matrix c. */ \
			PASTEMAC(ch,copys)( beta11c, c[i*rs_c + j*cs_c] ); \
\
			/* Store the local value back to b11. */ \
			PASTEMAC(ch,copys)( beta11c, b[i*rs_b + j*cs_b] ); \
		} \
	} \
}

//INSERT_GENTFUNC_BASIC2( trsm_l, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
GENTFUNC( float,    s, trsm_l, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, 4, 16 )
GENTFUNC( double,   d, trsm_l, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, 4, 8 )
GENTFUNC( scomplex, c, trsm_l, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, 4, 8 )
GENTFUNC( dcomplex, z, trsm_l, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, 4, 4 )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, arch, suf, mr, nr ) \
\
void PASTEMAC3(ch,opname,arch,suf) \
     ( \
       ctype*     restrict a, \
       ctype*     restrict b, \
       ctype*     restrict c, inc_t rs_c, inc_t cs_c, \
       auxinfo_t* restrict data, \
       cntx_t*    restrict cntx  \
     ) \
{ \
	const inc_t     rs_a   = 1; \
	const inc_t     cs_a   = mr; \
\
	const inc_t     rs_b   = nr; \
	const inc_t     cs_b   = 1; \
\
	PRAGMA_SIMD \
	for ( dim_t iter = 0; iter < mr; ++iter ) \
	{ \
		dim_t i = mr - iter - 1; \
\
		/* b1 = b1 - a12t * B2; */ \
		/* b1 = b1 / alpha11; */ \
		for ( dim_t j = 0; j < nr; ++j ) \
		{ \
			ctype beta11c = b[i*rs_b + j*cs_b]; \
			ctype rho11; \
\
			/* beta11 = beta11 - a12t * b21; */ \
			PASTEMAC(ch,set0s)( rho11 ); \
			for ( dim_t l = 0; l < iter; ++l ) \
			{ \
				PASTEMAC(ch,axpys)( a[i*rs_a + (i+1+l)*cs_a], \
				                    b[(i+1+l)*rs_b + j*cs_b], rho11 ); \
			} \
			PASTEMAC(ch,subs)( rho11, beta11c ); \
\
			/* beta11 = beta11 / alpha11; */ \
			/* NOTE: The INVERSE of alpha11 (1.0/alpha11) is stored instead
			   of alpha11, so we can multiply rather than divide. We store
			   the inverse of alpha11 intentionally to avoid expensive
			   division instructions within the micro-kernel. */ \
			PASTEMAC(ch,scals)( a[i*rs_a + i*cs_a], beta11c ); \
\
			/* Output final result to matrix c. */ \
			PASTEMAC(ch,copys)( beta11c, c[i*rs_c + j*cs_c] ); \
\
			/* Store the local value back to b11. */ \
			PASTEMAC(ch,copys)( beta11c, b[i*rs_b + j*cs_b] ); \
		} \
	} \
}

//INSERT_GENTFUNC_BASIC2( trsm_u, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
GENTFUNC( float,    s, trsm_u, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, 4, 16 )
GENTFUNC( double,   d, trsm_u, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, 4, 8 )
GENTFUNC( scomplex, c, trsm_u, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, 4, 8 )
GENTFUNC( dcomplex, z, trsm_u, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, 4, 4 )

#else

#endif
