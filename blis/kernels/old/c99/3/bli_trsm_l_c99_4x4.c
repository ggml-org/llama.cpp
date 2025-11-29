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
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       ctype*     restrict a, \
       ctype*     restrict b, \
       ctype*     restrict c, inc_t rs_c, inc_t cs_c, \
       auxinfo_t* restrict data, \
       cntx_t*    restrict cntx  \
     ) \
{ \
	const dim_t rs_a = 1; \
	const dim_t cs_a = 4; \
\
	const dim_t rs_b = 4; \
	const dim_t cs_b = 1; \
\
	ctype  a00; \
	ctype  a10, a11; \
	ctype  a20, a21, a22; \
	ctype  a30, a31, a32, a33; \
\
	ctype  b00, b01, b02, b03; \
	ctype  b10, b11, b12, b13; \
	ctype  b20, b21, b22, b23; \
	ctype  b30, b31, b32, b33; \
\
\
	/* Load contents of B. */ \
\
	b00 = *(b + 0*rs_b + 0*cs_b); \
	b01 = *(b + 0*rs_b + 1*cs_b); \
	b02 = *(b + 0*rs_b + 2*cs_b); \
	b03 = *(b + 0*rs_b + 3*cs_b); \
\
	b10 = *(b + 1*rs_b + 0*cs_b); \
	b11 = *(b + 1*rs_b + 1*cs_b); \
	b12 = *(b + 1*rs_b + 2*cs_b); \
	b13 = *(b + 1*rs_b + 3*cs_b); \
\
	b20 = *(b + 2*rs_b + 0*cs_b); \
	b21 = *(b + 2*rs_b + 1*cs_b); \
	b22 = *(b + 2*rs_b + 2*cs_b); \
	b23 = *(b + 2*rs_b + 3*cs_b); \
\
	b30 = *(b + 3*rs_b + 0*cs_b); \
	b31 = *(b + 3*rs_b + 1*cs_b); \
	b32 = *(b + 3*rs_b + 2*cs_b); \
	b33 = *(b + 3*rs_b + 3*cs_b); \
\
\
	/* iteration 0 */ \
\
	a00 = *(a + 0*rs_a + 0*cs_a); \
\
	PASTEMAC(ch,scals)( a00, b00 ); \
	PASTEMAC(ch,scals)( a00, b01 ); \
	PASTEMAC(ch,scals)( a00, b02 ); \
	PASTEMAC(ch,scals)( a00, b03 ); \
\
	*(b + 0*rs_b + 0*cs_b) = b00; \
	*(b + 0*rs_b + 1*cs_b) = b01; \
	*(b + 0*rs_b + 2*cs_b) = b02; \
	*(b + 0*rs_b + 3*cs_b) = b03; \
\
	*(c + 0*rs_c + 0*cs_c) = b00; \
	*(c + 0*rs_c + 1*cs_c) = b01; \
	*(c + 0*rs_c + 2*cs_c) = b02; \
	*(c + 0*rs_c + 3*cs_c) = b03; \
\
\
	/* iteration 1 */ \
\
	a10 = *(a + 1*rs_a + 0*cs_a); \
	a11 = *(a + 1*rs_a + 1*cs_a); \
\
	PASTEMAC(ch,axmys)( a10, b00, b10 ); \
	PASTEMAC(ch,axmys)( a10, b01, b11 ); \
	PASTEMAC(ch,axmys)( a10, b02, b12 ); \
	PASTEMAC(ch,axmys)( a10, b03, b13 ); \
\
	PASTEMAC(ch,scals)( a11, b10 ); \
	PASTEMAC(ch,scals)( a11, b11 ); \
	PASTEMAC(ch,scals)( a11, b12 ); \
	PASTEMAC(ch,scals)( a11, b13 ); \
\
	*(b + 1*rs_b + 0*cs_b) = b10; \
	*(b + 1*rs_b + 1*cs_b) = b11; \
	*(b + 1*rs_b + 2*cs_b) = b12; \
	*(b + 1*rs_b + 3*cs_b) = b13; \
\
	*(c + 1*rs_c + 0*cs_c) = b10; \
	*(c + 1*rs_c + 1*cs_c) = b11; \
	*(c + 1*rs_c + 2*cs_c) = b12; \
	*(c + 1*rs_c + 3*cs_c) = b13; \
\
\
	/* iteration 2 */ \
\
	a20 = *(a + 2*rs_a + 0*cs_a); \
	a21 = *(a + 2*rs_a + 1*cs_a); \
	a22 = *(a + 2*rs_a + 2*cs_a); \
\
	PASTEMAC(ch,axmys)( a20, b00, b20 ); \
	PASTEMAC(ch,axmys)( a20, b01, b21 ); \
	PASTEMAC(ch,axmys)( a20, b02, b22 ); \
	PASTEMAC(ch,axmys)( a20, b03, b23 ); \
\
	PASTEMAC(ch,axmys)( a21, b10, b20 ); \
	PASTEMAC(ch,axmys)( a21, b11, b21 ); \
	PASTEMAC(ch,axmys)( a21, b12, b22 ); \
	PASTEMAC(ch,axmys)( a21, b13, b23 ); \
\
	PASTEMAC(ch,scals)( a22, b20 ); \
	PASTEMAC(ch,scals)( a22, b21 ); \
	PASTEMAC(ch,scals)( a22, b22 ); \
	PASTEMAC(ch,scals)( a22, b23 ); \
\
	*(b + 2*rs_b + 0*cs_b) = b20; \
	*(b + 2*rs_b + 1*cs_b) = b21; \
	*(b + 2*rs_b + 2*cs_b) = b22; \
	*(b + 2*rs_b + 3*cs_b) = b23; \
\
	*(c + 2*rs_c + 0*cs_c) = b20; \
	*(c + 2*rs_c + 1*cs_c) = b21; \
	*(c + 2*rs_c + 2*cs_c) = b22; \
	*(c + 2*rs_c + 3*cs_c) = b23; \
\
\
	/* iteration 3 */ \
\
	a30 = *(a + 3*rs_a + 0*cs_a); \
	a31 = *(a + 3*rs_a + 1*cs_a); \
	a32 = *(a + 3*rs_a + 2*cs_a); \
	a33 = *(a + 3*rs_a + 3*cs_a); \
\
	PASTEMAC(ch,axmys)( a30, b00, b30 ); \
	PASTEMAC(ch,axmys)( a30, b01, b31 ); \
	PASTEMAC(ch,axmys)( a30, b02, b32 ); \
	PASTEMAC(ch,axmys)( a30, b03, b33 ); \
\
	PASTEMAC(ch,axmys)( a31, b10, b30 ); \
	PASTEMAC(ch,axmys)( a31, b11, b31 ); \
	PASTEMAC(ch,axmys)( a31, b12, b32 ); \
	PASTEMAC(ch,axmys)( a31, b13, b33 ); \
\
	PASTEMAC(ch,axmys)( a32, b20, b30 ); \
	PASTEMAC(ch,axmys)( a32, b21, b31 ); \
	PASTEMAC(ch,axmys)( a32, b22, b32 ); \
	PASTEMAC(ch,axmys)( a32, b23, b33 ); \
\
	PASTEMAC(ch,scals)( a33, b30 ); \
	PASTEMAC(ch,scals)( a33, b31 ); \
	PASTEMAC(ch,scals)( a33, b32 ); \
	PASTEMAC(ch,scals)( a33, b33 ); \
\
	*(b + 3*rs_b + 0*cs_b) = b30; \
	*(b + 3*rs_b + 1*cs_b) = b31; \
	*(b + 3*rs_b + 2*cs_b) = b32; \
	*(b + 3*rs_b + 3*cs_b) = b33; \
\
	*(c + 3*rs_c + 0*cs_c) = b30; \
	*(c + 3*rs_c + 1*cs_c) = b31; \
	*(c + 3*rs_c + 2*cs_c) = b32; \
	*(c + 3*rs_c + 3*cs_c) = b33; \
}

INSERT_GENTFUNC_BASIC0( trsm_l_c99_4x4 )

