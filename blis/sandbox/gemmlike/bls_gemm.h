/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, The University of Texas at Austin

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

//
// -- Prototype the gemm-like operation's object API ---------------------------
//

void bls_gemm
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c
     );

void bls_gemm_ex
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     );

//
// -- Prototype the gemm-like operation's thread entry point -------------------
//

err_t bls_gemm_int
     (
       const obj_t*     alpha,
       const obj_t*     a,
       const obj_t*     b,
       const obj_t*     beta,
       const obj_t*     c,
       const cntx_t*    cntx,
       const rntm_t*    rntm,
             thrinfo_t* thread
     );

//
// -- Prototype the gemm-like operation's typed API ----------------------------
//

#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
void PASTECH(bls_,ch,opname) \
     ( \
       trans_t transa, \
       trans_t transb, \
       dim_t   m, \
       dim_t   n, \
       dim_t   k, \
       ctype*  alpha, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       ctype*  b, inc_t rs_b, inc_t cs_b, \
       ctype*  beta, \
       ctype*  c, inc_t rs_c, inc_t cs_c  \
     );

//INSERT_GENTPROT_BASIC( gemm )
GENTPROT( float,    s, gemm )
GENTPROT( double,   d, gemm )
GENTPROT( scomplex, c, gemm )
GENTPROT( dcomplex, z, gemm )

