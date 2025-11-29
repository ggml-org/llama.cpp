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

#ifndef BLIS_L3_FT_EX_H
#define BLIS_L3_FT_EX_H


//
// -- Level-3 expert function types --------------------------------------------
//

// gemm

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH3(ch,opname,BLIS_TAPI_EX_SUF,tsuf)) \
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
       ctype*  c, inc_t rs_c, inc_t cs_c, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     );

INSERT_GENTDEF( gemm )


// hemm, symm

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH3(ch,opname,BLIS_TAPI_EX_SUF,tsuf)) \
     ( \
       side_t  side, \
       uplo_t  uploa, \
       conj_t  conja, \
       trans_t transb, \
       dim_t   m, \
       dim_t   n, \
       ctype*  alpha, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       ctype*  b, inc_t rs_b, inc_t cs_b, \
       ctype*  beta, \
       ctype*  c, inc_t rs_c, inc_t cs_c, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     );

INSERT_GENTDEF( hemm )
INSERT_GENTDEF( symm )


// herk

#undef  GENTDEFR
#define GENTDEFR( ctype, ctype_r, ch, chr, opname, tsuf ) \
\
typedef void (*PASTECH3(ch,opname,BLIS_TAPI_EX_SUF,tsuf)) \
     ( \
       uplo_t   uploc, \
       trans_t  transa, \
       dim_t    m, \
       dim_t    k, \
       ctype_r* alpha, \
       ctype*   a, inc_t rs_a, inc_t cs_a, \
       ctype_r* beta, \
       ctype*   c, inc_t rs_c, inc_t cs_c, \
       cntx_t*  cntx, \
       rntm_t*  rntm  \
     );

INSERT_GENTDEFR( herk )


// her2k

#undef  GENTDEFR
#define GENTDEFR( ctype, ctype_r, ch, chr, opname, tsuf ) \
\
typedef void (*PASTECH3(ch,opname,BLIS_TAPI_EX_SUF,tsuf)) \
     ( \
       uplo_t   uploc, \
       trans_t  transa, \
       trans_t  transb, \
       dim_t    m, \
       dim_t    k, \
       ctype*   alpha, \
       ctype*   a, inc_t rs_a, inc_t cs_a, \
       ctype*   b, inc_t rs_b, inc_t cs_b, \
       ctype_r* beta, \
       ctype*   c, inc_t rs_c, inc_t cs_c, \
       cntx_t*  cntx, \
       rntm_t*  rntm  \
     );

INSERT_GENTDEFR( her2k )


// syrk

#undef  GENTDEFR
#define GENTDEFR( ctype, ctype_r, ch, chr, opname, tsuf ) \
\
typedef void (*PASTECH3(ch,opname,BLIS_TAPI_EX_SUF,tsuf)) \
     ( \
       uplo_t  uploc, \
       trans_t transa, \
       dim_t   m, \
       dim_t   k, \
       ctype*  alpha, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       ctype*  beta, \
       ctype*  c, inc_t rs_c, inc_t cs_c, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     );

INSERT_GENTDEFR( syrk )


// syr2k

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH3(ch,opname,BLIS_TAPI_EX_SUF,tsuf)) \
     ( \
       uplo_t  uploc, \
       trans_t transa, \
       trans_t transb, \
       dim_t   m, \
       dim_t   k, \
       ctype*  alpha, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       ctype*  b, inc_t rs_b, inc_t cs_b, \
       ctype*  beta, \
       ctype*  c, inc_t rs_c, inc_t cs_c, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     );

INSERT_GENTDEF( syr2k )


// trmm3

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH3(ch,opname,BLIS_TAPI_EX_SUF,tsuf)) \
     ( \
       side_t  side, \
       uplo_t  uploa, \
       trans_t transa, \
       diag_t  diaga, \
       trans_t transb, \
       dim_t   m, \
       dim_t   n, \
       ctype*  alpha, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       ctype*  b, inc_t rs_b, inc_t cs_b, \
       ctype*  beta, \
       ctype*  c, inc_t rs_c, inc_t cs_c, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     );

INSERT_GENTDEF( trmm3 )


// trmm

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH3(ch,opname,BLIS_TAPI_EX_SUF,tsuf)) \
     ( \
       side_t  side, \
       uplo_t  uploa, \
       trans_t transa, \
       diag_t  diaga, \
       dim_t   m, \
       dim_t   n, \
       ctype*  alpha, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       ctype*  b, inc_t rs_b, inc_t cs_b, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     );

INSERT_GENTDEF( trmm )
INSERT_GENTDEF( trsm )


#endif

