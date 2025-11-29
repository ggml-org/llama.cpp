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
// Prototype BLAS-like interfaces with typed operands.
//

#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             doff_t  diagoffx, \
             diag_t  diagx, \
             uplo_t  uplox, \
             trans_t transx, \
             dim_t   m, \
             dim_t   n, \
       const ctype*  x, inc_t rs_x, inc_t cs_x, \
             ctype*  y, inc_t rs_y, inc_t cs_y  \
       BLIS_TAPI_EX_PARAMS  \
     );

INSERT_GENTPROT_BASIC( addm )
INSERT_GENTPROT_BASIC( copym )
INSERT_GENTPROT_BASIC( subm )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             doff_t  diagoffx, \
             diag_t  diagx, \
             uplo_t  uplox, \
             trans_t transx, \
             dim_t   m, \
             dim_t   n, \
       const ctype*  alpha, \
       const ctype*  x, inc_t rs_x, inc_t cs_x, \
             ctype*  y, inc_t rs_y, inc_t cs_y  \
       BLIS_TAPI_EX_PARAMS  \
     );

INSERT_GENTPROT_BASIC( axpym )
INSERT_GENTPROT_BASIC( scal2m )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             conj_t conjalpha, \
             doff_t diagoffx, \
             diag_t diagx, \
             uplo_t uplox, \
             dim_t  m, \
             dim_t  n, \
       const ctype* alpha, \
             ctype* x, inc_t rs_x, inc_t cs_x  \
       BLIS_TAPI_EX_PARAMS  \
     );

INSERT_GENTPROT_BASIC( invscalm )
INSERT_GENTPROT_BASIC( scalm )
INSERT_GENTPROT_BASIC( setm )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname,EX_SUF) \
     ( \
             doff_t  diagoffx, \
             diag_t  diagx, \
             uplo_t  uplox, \
             trans_t transx, \
             dim_t   m, \
             dim_t   n, \
       const ctype*  x, inc_t rs_x, inc_t cs_x, \
       const ctype*  beta, \
             ctype*  y, inc_t rs_y, inc_t cs_y  \
       BLIS_TAPI_EX_PARAMS  \
     );

INSERT_GENTPROT_BASIC( xpbym )


#undef  GENTPROT2
#define GENTPROT2( ctype_x, ctype_y, chx, chy, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(chx,chy,opname,EX_SUF) \
     ( \
             doff_t   diagoffx, \
             diag_t   diagx, \
             uplo_t   uplox, \
             trans_t  transx, \
             dim_t    m, \
             dim_t    n, \
       const ctype_x* x, inc_t rs_x, inc_t cs_x, \
       const ctype_y* beta, \
             ctype_y* y, inc_t rs_y, inc_t cs_y  \
       BLIS_TAPI_EX_PARAMS  \
     );

INSERT_GENTPROT2_BASIC( xpbym_md )
INSERT_GENTPROT2_MIX_DP( xpbym_md )

