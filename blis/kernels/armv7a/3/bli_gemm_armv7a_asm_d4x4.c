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

#undef  GENTPROT
#define GENTPROT( ctype, ch, opname, suf ) \
\
extern \
void PASTEMAC(ch,opname,suf) \
     ( \
             uint32_t   k, \
       const ctype*     alpha, \
       const ctype*     a, \
       const ctype*     b, \
       const ctype*     beta, \
             ctype*     c, uint32_t rs_c, uint32_t cs_c, \
       const auxinfo_t* data  \
     );

GENTPROT( float,    s, gemm_armv7a_ker_, 4x4 )
GENTPROT( double,   d, gemm_armv7a_ker_, 4x4 )
GENTPROT( scomplex, c, gemm_armv7a_ker_, 2x2 )
GENTPROT( dcomplex, z, gemm_armv7a_ker_, 2x2 )




void bli_sgemm_armv7a_asm_4x4
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c, inc_t cs_c,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	GEMM_UKR_SETUP_CT_ANY( s, 4, 4, false );
	bli_sgemm_armv7a_ker_4x4( k, alpha, a, b, beta, c, rs_c, cs_c, data );
	GEMM_UKR_FLUSH_CT( s );
}


void bli_dgemm_armv7a_asm_4x4
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c, inc_t cs_c,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	GEMM_UKR_SETUP_CT_ANY( d, 4, 4, false );
	bli_dgemm_armv7a_ker_4x4( k, alpha, a, b, beta, c, rs_c, cs_c, data );
	GEMM_UKR_FLUSH_CT( d );
}


void bli_cgemm_armv7a_asm_2x2
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c, inc_t cs_c,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	GEMM_UKR_SETUP_CT_ANY( c, 2, 2, false );
	bli_cgemm_armv7a_ker_2x2( k, alpha, a, b, beta, c, rs_c, cs_c, data );
	GEMM_UKR_FLUSH_CT( c );
}

void bli_zgemm_armv7a_asm_2x2
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c, inc_t cs_c,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	GEMM_UKR_SETUP_CT_ANY( z, 2, 2, false );
	bli_zgemm_armv7a_ker_2x2( k, alpha, a, b, beta, c, rs_c, cs_c, data );
	GEMM_UKR_FLUSH_CT( z );
}

