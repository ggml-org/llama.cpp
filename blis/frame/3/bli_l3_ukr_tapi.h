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


//
// Generate prototypes for level-3 micro-kernel wrappers.
//

#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
             dim_t      m, \
             dim_t      n, \
             dim_t      k, \
       const ctype*     alpha, \
       const ctype*     a, \
       const ctype*     b, \
       const ctype*     beta, \
             ctype*     c, inc_t rs_c, inc_t cs_c, \
       const auxinfo_t* data, \
       const cntx_t*    cntx  \
     );

INSERT_GENTPROT_BASIC( gemm_ukernel )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
             dim_t      m, \
             dim_t      n, \
             dim_t      k, \
       const ctype*     alpha, \
       const ctype*     a1x, \
       const ctype*     a11, \
       const ctype*     bx1, \
             ctype*     b11, \
             ctype*     c11, inc_t rs_c, inc_t cs_c, \
       const auxinfo_t* data, \
       const cntx_t*    cntx  \
     );

INSERT_GENTPROT_BASIC( gemmtrsm_l_ukernel )
INSERT_GENTPROT_BASIC( gemmtrsm_u_ukernel )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       const ctype*     a, \
             ctype*     b, \
             ctype*     c, inc_t rs_c, inc_t cs_c, \
       const auxinfo_t* data, \
       const cntx_t*    cntx  \
     );

INSERT_GENTPROT_BASIC( trsm_l_ukernel )
INSERT_GENTPROT_BASIC( trsm_u_ukernel )

