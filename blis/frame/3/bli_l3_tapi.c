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

#include "blis.h"

//
// Define BLAS-like interfaces with typed operands (basic).
//

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
             trans_t transa, \
             trans_t transb, \
             dim_t   m, \
             dim_t   n, \
             dim_t   k, \
       const ctype*  alpha, \
       const ctype*  a, inc_t rs_a, inc_t cs_a, \
       const ctype*  b, inc_t rs_b, inc_t cs_b, \
       const ctype*  beta, \
             ctype*  c, inc_t rs_c, inc_t cs_c  \
     ) \
{ \
	/* Invoke the expert interface and request default cntx_t and rntm_t
	   objects. */ \
	PASTEMAC(ch,opname,BLIS_TAPI_EX_SUF) \
	( \
	  transa, \
	  transb, \
	  m, n, k, \
	  alpha, \
	  a, rs_a, cs_a, \
	  b, rs_b, cs_b, \
	  beta, \
	  c, rs_c, cs_c, \
	  NULL, \
	  NULL  \
	); \
}

INSERT_GENTFUNC_BASIC( gemm )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
             uplo_t  uploc, \
             trans_t transa, \
             trans_t transb, \
             dim_t   m, \
             dim_t   k, \
       const ctype*  alpha, \
       const ctype*  a, inc_t rs_a, inc_t cs_a, \
       const ctype*  b, inc_t rs_b, inc_t cs_b, \
       const ctype*  beta, \
             ctype*  c, inc_t rs_c, inc_t cs_c  \
     ) \
{ \
	/* Invoke the expert interface and request default cntx_t and rntm_t
	   objects. */ \
	PASTEMAC(ch,opname,BLIS_TAPI_EX_SUF) \
	( \
	  uploc, \
	  transa, \
	  transb, \
	  m, k, \
	  alpha, \
	  a, rs_a, cs_a, \
	  b, rs_b, cs_b, \
	  beta, \
	  c, rs_c, cs_c, \
	  NULL, \
	  NULL  \
	); \
}

INSERT_GENTFUNC_BASIC( gemmt )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, struca ) \
\
void PASTEMAC(ch,opname) \
     ( \
             side_t  side, \
             uplo_t  uploa, \
             conj_t  conja, \
             trans_t transb, \
             dim_t   m, \
             dim_t   n, \
       const ctype*  alpha, \
       const ctype*  a, inc_t rs_a, inc_t cs_a, \
       const ctype*  b, inc_t rs_b, inc_t cs_b, \
       const ctype*  beta, \
             ctype*  c, inc_t rs_c, inc_t cs_c  \
     ) \
{ \
	/* Invoke the expert interface and request default cntx_t and rntm_t
	   objects. */ \
	PASTEMAC(ch,opname,BLIS_TAPI_EX_SUF) \
	( \
	  side, \
	  uploa, \
	  conja, \
	  transb, \
	  m, n, \
	  alpha, \
	  a, rs_a, cs_a, \
	  b, rs_b, cs_b, \
	  beta, \
	  c, rs_c, cs_c, \
	  NULL, \
	  NULL  \
	); \
}

INSERT_GENTFUNC_BASIC( hemm, BLIS_HERMITIAN )
INSERT_GENTFUNC_BASIC( symm, BLIS_SYMMETRIC )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
             uplo_t   uploc, \
             trans_t  transa, \
             dim_t    m, \
             dim_t    k, \
       const ctype_r* alpha, \
       const ctype*   a, inc_t rs_a, inc_t cs_a, \
       const ctype_r* beta, \
             ctype*   c, inc_t rs_c, inc_t cs_c  \
     ) \
{ \
	/* Invoke the expert interface and request default cntx_t and rntm_t
	   objects. */ \
	PASTEMAC(ch,opname,BLIS_TAPI_EX_SUF) \
	( \
	  uploc, \
	  transa, \
	  m, k, \
	  alpha, \
	  a, rs_a, cs_a, \
	  beta, \
	  c, rs_c, cs_c, \
	  NULL, \
	  NULL  \
	); \
}

INSERT_GENTFUNCR_BASIC( herk )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
             uplo_t   uploc, \
             trans_t  transa, \
             trans_t  transb, \
             dim_t    m, \
             dim_t    k, \
       const ctype*   alpha, \
       const ctype*   a, inc_t rs_a, inc_t cs_a, \
       const ctype*   b, inc_t rs_b, inc_t cs_b, \
       const ctype_r* beta, \
             ctype*   c, inc_t rs_c, inc_t cs_c  \
     ) \
{ \
	/* Invoke the expert interface and request default cntx_t and rntm_t
	   objects. */ \
	PASTEMAC(ch,opname,BLIS_TAPI_EX_SUF) \
	( \
	  uploc, \
	  transa, \
	  transb, \
	  m, k, \
	  alpha, \
	  a, rs_a, cs_a, \
	  b, rs_b, cs_b, \
	  beta, \
	  c, rs_c, cs_c, \
	  NULL, \
	  NULL  \
	); \
}

INSERT_GENTFUNCR_BASIC( her2k )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
             uplo_t  uploc, \
             trans_t transa, \
             dim_t   m, \
             dim_t   k, \
       const ctype*  alpha, \
       const ctype*  a, inc_t rs_a, inc_t cs_a, \
       const ctype*  beta, \
             ctype*  c, inc_t rs_c, inc_t cs_c  \
     ) \
{ \
	/* Invoke the expert interface and request default cntx_t and rntm_t
	   objects. */ \
	PASTEMAC(ch,opname,BLIS_TAPI_EX_SUF) \
	( \
	  uploc, \
	  transa, \
	  m, k, \
	  alpha, \
	  a, rs_a, cs_a, \
	  beta, \
	  c, rs_c, cs_c, \
	  NULL, \
	  NULL  \
	); \
}

INSERT_GENTFUNC_BASIC( syrk )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
             uplo_t  uploc, \
             trans_t transa, \
             trans_t transb, \
             dim_t   m, \
             dim_t   k, \
       const ctype*  alpha, \
       const ctype*  a, inc_t rs_a, inc_t cs_a, \
       const ctype*  b, inc_t rs_b, inc_t cs_b, \
       const ctype*  beta, \
             ctype*  c, inc_t rs_c, inc_t cs_c  \
     ) \
{ \
	/* Invoke the expert interface and request default cntx_t and rntm_t
	   objects. */ \
	PASTEMAC(ch,opname,BLIS_TAPI_EX_SUF) \
	( \
	  uploc, \
	  transa, \
	  transb, \
	  m, k, \
	  alpha, \
	  a, rs_a, cs_a, \
	  b, rs_b, cs_b, \
	  beta, \
	  c, rs_c, cs_c, \
	  NULL, \
	  NULL  \
	); \
}

INSERT_GENTFUNC_BASIC( syr2k )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
             side_t  side, \
             uplo_t  uploa, \
             trans_t transa, \
             diag_t  diaga, \
             trans_t transb, \
             dim_t   m, \
             dim_t   n, \
       const ctype*  alpha, \
       const ctype*  a, inc_t rs_a, inc_t cs_a, \
       const ctype*  b, inc_t rs_b, inc_t cs_b, \
       const ctype*  beta, \
             ctype*  c, inc_t rs_c, inc_t cs_c  \
     ) \
{ \
	/* Invoke the expert interface and request default cntx_t and rntm_t
	   objects. */ \
	PASTEMAC(ch,opname,BLIS_TAPI_EX_SUF) \
	( \
	  side, \
	  uploa, \
	  transa, \
	  diaga, \
	  transb, \
	  m, n, \
	  alpha, \
	  a, rs_a, cs_a, \
	  b, rs_b, cs_b, \
	  beta, \
	  c, rs_c, cs_c, \
	  NULL, \
	  NULL  \
	); \
}

INSERT_GENTFUNC_BASIC( trmm3 )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
             side_t  side, \
             uplo_t  uploa, \
             trans_t transa, \
             diag_t  diaga, \
             dim_t   m, \
             dim_t   n, \
       const ctype*  alpha, \
       const ctype*  a, inc_t rs_a, inc_t cs_a, \
             ctype*  b, inc_t rs_b, inc_t cs_b  \
     ) \
{ \
	/* Invoke the expert interface and request default cntx_t and rntm_t
	   objects. */ \
	PASTEMAC(ch,opname,BLIS_TAPI_EX_SUF) \
	( \
	  side, \
	  uploa, \
	  transa, \
	  diaga, \
	  m, n, \
	  alpha, \
	  a, rs_a, cs_a, \
	  b, rs_b, cs_b, \
	  NULL, \
	  NULL  \
	); \
}

INSERT_GENTFUNC_BASIC( trmm )
INSERT_GENTFUNC_BASIC( trsm )

