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
#define GENTFUNC( ctype, ch, kername ) \
\
void PASTEMAC(ch,kername) \
     ( \
       dim_t               k, \
       ctype*     restrict alpha, \
       ctype*     restrict a, \
       ctype*     restrict b, \
       ctype*     restrict beta, \
       ctype*     restrict c, inc_t rs_c, inc_t cs_c, \
       auxinfo_t* restrict data, \
       cntx_t*    restrict cntx  \
     ) \
{ \
	ctype  a0; \
	ctype  a1; \
	ctype  a2; \
	ctype  a3; \
\
	ctype  b0, b1, b2, b3; \
\
	ctype  ab00, ab01, ab02, ab03; \
	ctype  ab10, ab11, ab12, ab13; \
	ctype  ab20, ab21, ab22, ab23; \
	ctype  ab30, ab31, ab32, ab33; \
\
	ctype* c00, * c01, * c02, * c03; \
	ctype* c10, * c11, * c12, * c13; \
	ctype* c20, * c21, * c22, * c23; \
	ctype* c30, * c31, * c32, * c33; \
\
	dim_t  i; \
\
\
	c00 = (c + 0*rs_c + 0*cs_c); \
	c10 = (c + 1*rs_c + 0*cs_c); \
	c20 = (c + 2*rs_c + 0*cs_c); \
	c30 = (c + 3*rs_c + 0*cs_c); \
\
	c01 = (c + 0*rs_c + 1*cs_c); \
	c11 = (c + 1*rs_c + 1*cs_c); \
	c21 = (c + 2*rs_c + 1*cs_c); \
	c31 = (c + 3*rs_c + 1*cs_c); \
\
	c02 = (c + 0*rs_c + 2*cs_c); \
	c12 = (c + 1*rs_c + 2*cs_c); \
	c22 = (c + 2*rs_c + 2*cs_c); \
	c32 = (c + 3*rs_c + 2*cs_c); \
\
	c03 = (c + 0*rs_c + 3*cs_c); \
	c13 = (c + 1*rs_c + 3*cs_c); \
	c23 = (c + 2*rs_c + 3*cs_c); \
	c33 = (c + 3*rs_c + 3*cs_c); \
\
	PASTEMAC(ch,set0s)( ab00 ); \
	PASTEMAC(ch,set0s)( ab10 ); \
	PASTEMAC(ch,set0s)( ab20 ); \
	PASTEMAC(ch,set0s)( ab30 ); \
\
	PASTEMAC(ch,set0s)( ab01 ); \
	PASTEMAC(ch,set0s)( ab11 ); \
	PASTEMAC(ch,set0s)( ab21 ); \
	PASTEMAC(ch,set0s)( ab31 ); \
\
	PASTEMAC(ch,set0s)( ab02 ); \
	PASTEMAC(ch,set0s)( ab12 ); \
	PASTEMAC(ch,set0s)( ab22 ); \
	PASTEMAC(ch,set0s)( ab32 ); \
\
	PASTEMAC(ch,set0s)( ab03 ); \
	PASTEMAC(ch,set0s)( ab13 ); \
	PASTEMAC(ch,set0s)( ab23 ); \
	PASTEMAC(ch,set0s)( ab33 ); \
\
	for ( i = 0; i < k; ++i ) \
	{ \
		a0 = *(a + 0); \
		a1 = *(a + 1); \
		a2 = *(a + 2); \
		a3 = *(a + 3); \
\
		b0 = *(b + 0); \
		b1 = *(b + 1); \
		b2 = *(b + 2); \
		b3 = *(b + 3); \
\
		PASTEMAC(ch,dots)( a0, b0, ab00 ); \
		PASTEMAC(ch,dots)( a1, b0, ab10 ); \
		PASTEMAC(ch,dots)( a2, b0, ab20 ); \
		PASTEMAC(ch,dots)( a3, b0, ab30 ); \
\
		PASTEMAC(ch,dots)( a0, b1, ab01 ); \
		PASTEMAC(ch,dots)( a1, b1, ab11 ); \
		PASTEMAC(ch,dots)( a2, b1, ab21 ); \
		PASTEMAC(ch,dots)( a3, b1, ab31 ); \
\
		PASTEMAC(ch,dots)( a0, b2, ab02 ); \
		PASTEMAC(ch,dots)( a1, b2, ab12 ); \
		PASTEMAC(ch,dots)( a2, b2, ab22 ); \
		PASTEMAC(ch,dots)( a3, b2, ab32 ); \
\
		PASTEMAC(ch,dots)( a0, b3, ab03 ); \
		PASTEMAC(ch,dots)( a1, b3, ab13 ); \
		PASTEMAC(ch,dots)( a2, b3, ab23 ); \
		PASTEMAC(ch,dots)( a3, b3, ab33 ); \
\
		a += 4; \
		b += 4; \
	} \
\
	if ( PASTEMAC(ch,eq0)( *beta ) ) \
	{ \
		PASTEMAC(ch,set0s)( *c00 ); \
		PASTEMAC(ch,set0s)( *c10 ); \
		PASTEMAC(ch,set0s)( *c20 ); \
		PASTEMAC(ch,set0s)( *c30 ); \
\
		PASTEMAC(ch,set0s)( *c01 ); \
		PASTEMAC(ch,set0s)( *c11 ); \
		PASTEMAC(ch,set0s)( *c21 ); \
		PASTEMAC(ch,set0s)( *c31 ); \
\
		PASTEMAC(ch,set0s)( *c02 ); \
		PASTEMAC(ch,set0s)( *c12 ); \
		PASTEMAC(ch,set0s)( *c22 ); \
		PASTEMAC(ch,set0s)( *c32 ); \
\
		PASTEMAC(ch,set0s)( *c03 ); \
		PASTEMAC(ch,set0s)( *c13 ); \
		PASTEMAC(ch,set0s)( *c23 ); \
		PASTEMAC(ch,set0s)( *c33 ); \
	} \
	else \
	{ \
		PASTEMAC(ch,scals)( *beta, *c00 ); \
		PASTEMAC(ch,scals)( *beta, *c10 ); \
		PASTEMAC(ch,scals)( *beta, *c20 ); \
		PASTEMAC(ch,scals)( *beta, *c30 ); \
\
		PASTEMAC(ch,scals)( *beta, *c01 ); \
		PASTEMAC(ch,scals)( *beta, *c11 ); \
		PASTEMAC(ch,scals)( *beta, *c21 ); \
		PASTEMAC(ch,scals)( *beta, *c31 ); \
\
		PASTEMAC(ch,scals)( *beta, *c02 ); \
		PASTEMAC(ch,scals)( *beta, *c12 ); \
		PASTEMAC(ch,scals)( *beta, *c22 ); \
		PASTEMAC(ch,scals)( *beta, *c32 ); \
\
		PASTEMAC(ch,scals)( *beta, *c03 ); \
		PASTEMAC(ch,scals)( *beta, *c13 ); \
		PASTEMAC(ch,scals)( *beta, *c23 ); \
		PASTEMAC(ch,scals)( *beta, *c33 ); \
	} \
\
	PASTEMAC(ch,dots)( *alpha, ab00, *c00 ); \
	PASTEMAC(ch,dots)( *alpha, ab10, *c10 ); \
	PASTEMAC(ch,dots)( *alpha, ab20, *c20 ); \
	PASTEMAC(ch,dots)( *alpha, ab30, *c30 ); \
\
	PASTEMAC(ch,dots)( *alpha, ab01, *c01 ); \
	PASTEMAC(ch,dots)( *alpha, ab11, *c11 ); \
	PASTEMAC(ch,dots)( *alpha, ab21, *c21 ); \
	PASTEMAC(ch,dots)( *alpha, ab31, *c31 ); \
\
	PASTEMAC(ch,dots)( *alpha, ab02, *c02 ); \
	PASTEMAC(ch,dots)( *alpha, ab12, *c12 ); \
	PASTEMAC(ch,dots)( *alpha, ab22, *c22 ); \
	PASTEMAC(ch,dots)( *alpha, ab32, *c32 ); \
\
	PASTEMAC(ch,dots)( *alpha, ab03, *c03 ); \
	PASTEMAC(ch,dots)( *alpha, ab13, *c13 ); \
	PASTEMAC(ch,dots)( *alpha, ab23, *c23 ); \
	PASTEMAC(ch,dots)( *alpha, ab33, *c33 ); \
}

INSERT_GENTFUNC_BASIC0( gemm_c99_4x4 )

