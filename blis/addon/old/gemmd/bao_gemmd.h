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
// -- Prototype the gemmd operation's object API -------------------------------
//

BLIS_EXPORT_ADDON void bao_gemmd
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  d,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c
     );

BLIS_EXPORT_ADDON void bao_gemmd_ex
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  d,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm
     );

//
// -- Prototype the gemmd operation's thread entry point -----------------------
//

void bao_gemmd_int
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  d,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm,
       thrinfo_t* thread
     );

//
// -- Prototype the gemmd operation's typed API --------------------------------
//

#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
BLIS_EXPORT_ADDON void PASTECH2(bao_,ch,opname) \
     ( \
       trans_t transa, \
       trans_t transb, \
       dim_t   m, \
       dim_t   n, \
       dim_t   k, \
       ctype*  alpha, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       ctype*  d, inc_t incd, \
       ctype*  b, inc_t rs_b, inc_t cs_b, \
       ctype*  beta, \
       ctype*  c, inc_t rs_c, inc_t cs_c  \
     );

//INSERT_GENTPROT_BASIC0( gemmd )
GENTPROT( float,    s, gemmd )
GENTPROT( double,   d, gemmd )
GENTPROT( scomplex, c, gemmd )
GENTPROT( dcomplex, z, gemmd )

