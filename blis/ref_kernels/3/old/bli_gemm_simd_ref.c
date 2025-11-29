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
#define GENTFUNC( ctype, ch, opname, arch, suf, mr, nr ) \
\
void PASTEMAC4(ch,opname,arch,_simd,suf) \
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
	ctype           ab[ BLIS_STACK_BUF_MAX_SIZE \
	                    / sizeof( ctype ) ] \
	                    __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE))); \
	const inc_t     rs_ab  = nr; \
	const inc_t     cs_ab  = 1; \
\
	const inc_t     cs_a   = mr; \
	const inc_t     rs_b   = nr; \
\
\
	/* Initialize the accumulator elements in ab to zero. */ \
	PRAGMA_SIMD \
	for ( dim_t i = 0; i < mr * nr; ++i ) \
	{ \
		PASTEMAC(ch,set0s)( ab[ i ] ); \
	} \
\
/*
	const dim_t pre = 16; \
	dim_t k16; \
	if ( k >= pre ) { k16 = k - pre; k = pre; } \
	else            { k16 = 0;                } \
\
	for ( dim_t l = 0; l < k16; ++l ) \
	{ \
		for ( dim_t i = 0; i < mr; ++i ) \
		{ \
			PRAGMA_SIMD \
			for ( dim_t j = 0; j < nr; ++j ) \
			{ \
				PASTEMAC(ch,dots) \
				( \
				  a[ i ], \
				  b[ j ], \
				  ab[ i*rs_ab + j*cs_ab ]  \
				); \
			} \
		} \
\
		a += cs_a; \
		b += rs_b; \
	} \
\
	__builtin_prefetch( c + 0*cs_c, 1, 0 ); \
	__builtin_prefetch( c + 1*cs_c, 1, 0 ); \
	__builtin_prefetch( c + 2*cs_c, 1, 0 ); \
	__builtin_prefetch( c + 3*cs_c, 1, 0 ); \
*/ \
\
	/* Perform a series of k rank-1 updates into ab. */ \
	for ( dim_t l = 0; l < k; ++l ) \
	{ \
		for ( dim_t i = 0; i < mr; ++i ) \
		{ \
			PRAGMA_SIMD \
			for ( dim_t j = 0; j < nr; ++j ) \
			{ \
				PASTEMAC(ch,dots) \
				( \
				  a[ i ], \
				  b[ j ], \
				  ab[ i*rs_ab + j*cs_ab ]  \
				); \
			} \
		} \
\
		a += cs_a; \
		b += rs_b; \
	} \
\
	/* Scale the result in ab by alpha. */ \
	PRAGMA_SIMD \
	for ( dim_t i = 0; i < mr * nr; ++i ) \
	{ \
		PASTEMAC(ch,scals)( *alpha, ab[ i ] ); \
	} \
\
	/* Output/accumulate intermediate result ab based on the storage
	   of c and the value of beta. */ \
	if ( cs_c == 1 ) \
	{ \
		/* C is row-stored. */ \
\
		if ( PASTEMAC(ch,eq0)( *beta ) ) \
		{ \
			for ( dim_t i = 0; i < mr; ++i ) \
			for ( dim_t j = 0; j < nr; ++j ) \
			PASTEMAC(ch,copys) \
			( \
			  ab[ i*rs_ab + j*cs_ab ], \
			  c [ i*rs_c  + j*1     ]  \
			); \
		} \
		else \
		{ \
			for ( dim_t i = 0; i < mr; ++i ) \
			for ( dim_t j = 0; j < nr; ++j ) \
			PASTEMAC(ch,xpbys) \
			( \
			  ab[ i*rs_ab + j*cs_ab ], \
			  *beta, \
			  c [ i*rs_c  + j*1     ]  \
			); \
		} \
	} \
	else \
	{ \
		/* C is column-stored or general-stored. */ \
\
		if ( PASTEMAC(ch,eq0)( *beta ) ) \
		{ \
			for ( dim_t j = 0; j < nr; ++j ) \
			for ( dim_t i = 0; i < mr; ++i ) \
			PASTEMAC(ch,copys) \
			( \
			  ab[ i*rs_ab + j*cs_ab ], \
			  c [ i*rs_c  + j*1     ]  \
			); \
		} \
		else \
		{ \
			for ( dim_t j = 0; j < nr; ++j ) \
			for ( dim_t i = 0; i < mr; ++i ) \
			PASTEMAC(ch,xpbys) \
			( \
			  ab[ i*rs_ab + j*cs_ab ], \
			  *beta, \
			  c [ i*rs_c  + j*1     ]  \
			); \
		} \
	} \
}

//INSERT_GENTFUNC_BASIC2( gemm, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
GENTFUNC( float,    s, gemm, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, 4, 16 )
GENTFUNC( double,   d, gemm, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, 4, 8 )
GENTFUNC( scomplex, c, gemm, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, 4, 8 )
GENTFUNC( dcomplex, z, gemm, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, 4, 4 )

