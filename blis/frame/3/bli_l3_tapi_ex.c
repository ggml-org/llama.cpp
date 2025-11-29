/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020, Advanced Micro Devices, Inc.

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
// Define BLAS-like interfaces with typed operands (expert).
//

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname,BLIS_OAPI_EX_SUF) \
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
             ctype*  c, inc_t rs_c, inc_t cs_c, \
       const cntx_t* cntx, \
       const rntm_t* rntm  \
     ) \
{ \
	bli_init_once(); \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	obj_t       alphao = BLIS_OBJECT_INITIALIZER_1X1; \
	obj_t       ao     = BLIS_OBJECT_INITIALIZER; \
	obj_t       bo     = BLIS_OBJECT_INITIALIZER; \
	obj_t       betao  = BLIS_OBJECT_INITIALIZER_1X1; \
	obj_t       co     = BLIS_OBJECT_INITIALIZER; \
\
	dim_t       m_a, n_a; \
	dim_t       m_b, n_b; \
\
	bli_set_dims_with_trans( transa, m, k, &m_a, &n_a ); \
	bli_set_dims_with_trans( transb, k, n, &m_b, &n_b ); \
\
	bli_obj_init_finish_1x1( dt, ( void* )alpha, &alphao ); \
	bli_obj_init_finish_1x1( dt, ( void* )beta,  &betao  ); \
\
	bli_obj_init_finish( dt, m_a, n_a, ( void* )a, rs_a, cs_a, &ao ); \
	bli_obj_init_finish( dt, m_b, n_b, ( void* )b, rs_b, cs_b, &bo ); \
	bli_obj_init_finish( dt, m,   n,            c, rs_c, cs_c, &co ); \
\
	bli_obj_set_conjtrans( transa, &ao ); \
	bli_obj_set_conjtrans( transb, &bo ); \
\
	PASTEMAC(opname,BLIS_OAPI_EX_SUF) \
	( \
	  &alphao, \
	  &ao, \
	  &bo, \
	  &betao, \
	  &co, \
	  cntx, \
	  rntm  \
	); \
}

INSERT_GENTFUNC_BASIC( gemm )

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, struca ) \
\
void PASTEMAC(ch,opname,BLIS_OAPI_EX_SUF) \
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
             ctype*  c, inc_t rs_c, inc_t cs_c, \
       const cntx_t* cntx, \
       const rntm_t* rntm  \
     ) \
{ \
	bli_init_once(); \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	obj_t       alphao = BLIS_OBJECT_INITIALIZER_1X1; \
	obj_t       ao     = BLIS_OBJECT_INITIALIZER; \
	obj_t       bo     = BLIS_OBJECT_INITIALIZER; \
	obj_t       betao  = BLIS_OBJECT_INITIALIZER_1X1; \
	obj_t       co     = BLIS_OBJECT_INITIALIZER; \
\
	dim_t       mn_a; \
	dim_t       m_b, n_b; \
\
	bli_set_dim_with_side(   side,   m, n, &mn_a ); \
	bli_set_dims_with_trans( transb, m, n, &m_b, &n_b ); \
\
	bli_obj_init_finish_1x1( dt, ( void* )alpha, &alphao ); \
	bli_obj_init_finish_1x1( dt, ( void* )beta,  &betao  ); \
\
	bli_obj_init_finish( dt, mn_a, mn_a, ( void* )a, rs_a, cs_a, &ao ); \
	bli_obj_init_finish( dt, m_b,  n_b,  ( void* )b, rs_b, cs_b, &bo ); \
	bli_obj_init_finish( dt, m,    n,             c, rs_c, cs_c, &co ); \
\
	bli_obj_set_uplo( uploa, &ao ); \
	bli_obj_set_conj( conja, &ao ); \
	bli_obj_set_conjtrans( transb, &bo ); \
\
	bli_obj_set_struc( struca, &ao ); \
\
	PASTEMAC(opname,BLIS_OAPI_EX_SUF) \
	( \
	  side, \
	  &alphao, \
	  &ao, \
	  &bo, \
	  &betao, \
	  &co, \
	  cntx, \
	  rntm  \
	); \
}

INSERT_GENTFUNC_BASIC( hemm, BLIS_HERMITIAN )
INSERT_GENTFUNC_BASIC( symm, BLIS_SYMMETRIC )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, opname ) \
\
void PASTEMAC(ch,opname,BLIS_OAPI_EX_SUF) \
     ( \
             uplo_t   uploc, \
             trans_t  transa, \
             dim_t    m, \
             dim_t    k, \
       const ctype_r* alpha, \
       const ctype*   a, inc_t rs_a, inc_t cs_a, \
       const ctype_r* beta, \
             ctype*   c, inc_t rs_c, inc_t cs_c, \
       const cntx_t*  cntx, \
       const rntm_t*  rntm  \
     ) \
{ \
	bli_init_once(); \
\
	const num_t dt_r = PASTEMAC(chr,type); \
	const num_t dt   = PASTEMAC(ch,type); \
\
	obj_t       alphao = BLIS_OBJECT_INITIALIZER_1X1; \
	obj_t       ao     = BLIS_OBJECT_INITIALIZER; \
	obj_t       betao  = BLIS_OBJECT_INITIALIZER_1X1; \
	obj_t       co     = BLIS_OBJECT_INITIALIZER; \
\
	dim_t       m_a, n_a; \
\
	bli_set_dims_with_trans( transa, m, k, &m_a, &n_a ); \
\
	bli_obj_init_finish_1x1( dt_r, ( void* )alpha, &alphao ); \
	bli_obj_init_finish_1x1( dt_r, ( void* )beta,  &betao  ); \
\
	bli_obj_init_finish( dt, m_a, n_a, ( void* )a, rs_a, cs_a, &ao ); \
	bli_obj_init_finish( dt, m,   m,            c, rs_c, cs_c, &co ); \
\
	bli_obj_set_uplo( uploc, &co ); \
	bli_obj_set_conjtrans( transa, &ao ); \
\
	bli_obj_set_struc( BLIS_HERMITIAN, &co ); \
\
	PASTEMAC(opname,BLIS_OAPI_EX_SUF) \
	( \
	  &alphao, \
	  &ao, \
	  &betao, \
	  &co, \
	  cntx, \
	  rntm  \
	); \
}

INSERT_GENTFUNCR_BASIC( herk )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, opname ) \
\
void PASTEMAC(ch,opname,BLIS_OAPI_EX_SUF) \
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
             ctype*   c, inc_t rs_c, inc_t cs_c, \
       const cntx_t*  cntx, \
       const rntm_t*  rntm  \
     ) \
{ \
	bli_init_once(); \
\
	const num_t dt_r = PASTEMAC(chr,type); \
	const num_t dt   = PASTEMAC(ch,type); \
\
	obj_t       alphao = BLIS_OBJECT_INITIALIZER_1X1; \
	obj_t       ao     = BLIS_OBJECT_INITIALIZER; \
	obj_t       bo     = BLIS_OBJECT_INITIALIZER; \
	obj_t       betao  = BLIS_OBJECT_INITIALIZER_1X1; \
	obj_t       co     = BLIS_OBJECT_INITIALIZER; \
\
	dim_t       m_a, n_a; \
	dim_t       m_b, n_b; \
\
	bli_set_dims_with_trans( transa, m, k, &m_a, &n_a ); \
	bli_set_dims_with_trans( transb, m, k, &m_b, &n_b ); \
\
	bli_obj_init_finish_1x1( dt,   ( void* )alpha, &alphao ); \
	bli_obj_init_finish_1x1( dt_r, ( void* )beta,  &betao  ); \
\
	bli_obj_init_finish( dt, m_a, n_a, ( void* )a, rs_a, cs_a, &ao ); \
	bli_obj_init_finish( dt, m_b, n_b, ( void* )b, rs_b, cs_b, &bo ); \
	bli_obj_init_finish( dt, m,   m,            c, rs_c, cs_c, &co ); \
\
	bli_obj_set_uplo( uploc, &co ); \
	bli_obj_set_conjtrans( transa, &ao ); \
	bli_obj_set_conjtrans( transb, &bo ); \
\
	bli_obj_set_struc( BLIS_HERMITIAN, &co ); \
\
	PASTEMAC(opname,BLIS_OAPI_EX_SUF) \
	( \
	  &alphao, \
	  &ao, \
	  &bo, \
	  &betao, \
	  &co, \
	  cntx, \
	  rntm  \
	); \
}

INSERT_GENTFUNCR_BASIC( her2k )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname,BLIS_OAPI_EX_SUF) \
     ( \
             uplo_t  uploc, \
             trans_t transa, \
             dim_t   m, \
             dim_t   k, \
       const ctype*  alpha, \
       const ctype*  a, inc_t rs_a, inc_t cs_a, \
       const ctype*  beta, \
             ctype*  c, inc_t rs_c, inc_t cs_c, \
       const cntx_t* cntx, \
       const rntm_t* rntm  \
     ) \
{ \
	bli_init_once(); \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	obj_t       alphao = BLIS_OBJECT_INITIALIZER_1X1; \
	obj_t       ao     = BLIS_OBJECT_INITIALIZER; \
	obj_t       betao  = BLIS_OBJECT_INITIALIZER_1X1; \
	obj_t       co     = BLIS_OBJECT_INITIALIZER; \
\
	dim_t       m_a, n_a; \
\
	bli_set_dims_with_trans( transa, m, k, &m_a, &n_a ); \
\
	bli_obj_init_finish_1x1( dt, ( void* )alpha, &alphao ); \
	bli_obj_init_finish_1x1( dt, ( void* )beta,  &betao  ); \
\
	bli_obj_init_finish( dt, m_a, n_a, ( void* )a, rs_a, cs_a, &ao ); \
	bli_obj_init_finish( dt, m,   m,            c, rs_c, cs_c, &co ); \
\
	bli_obj_set_uplo( uploc, &co ); \
	bli_obj_set_conjtrans( transa, &ao ); \
\
	bli_obj_set_struc( BLIS_SYMMETRIC, &co ); \
\
	PASTEMAC(opname,BLIS_OAPI_EX_SUF) \
	( \
	  &alphao, \
	  &ao, \
	  &betao, \
	  &co, \
	  cntx, \
	  rntm  \
	); \
}

INSERT_GENTFUNC_BASIC( syrk )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname,BLIS_OAPI_EX_SUF) \
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
             ctype*  c, inc_t rs_c, inc_t cs_c, \
       const cntx_t* cntx, \
       const rntm_t* rntm  \
     ) \
{ \
	bli_init_once(); \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	obj_t       alphao = BLIS_OBJECT_INITIALIZER_1X1; \
	obj_t       ao     = BLIS_OBJECT_INITIALIZER; \
	obj_t       bo     = BLIS_OBJECT_INITIALIZER; \
	obj_t       betao  = BLIS_OBJECT_INITIALIZER_1X1; \
	obj_t       co     = BLIS_OBJECT_INITIALIZER; \
\
	dim_t       m_a, n_a; \
	dim_t       m_b, n_b; \
\
	bli_set_dims_with_trans( transa, m, k, &m_a, &n_a ); \
	bli_set_dims_with_trans( transb, m, k, &m_b, &n_b ); \
\
	bli_obj_init_finish_1x1( dt, ( void* )alpha, &alphao ); \
	bli_obj_init_finish_1x1( dt, ( void* )beta,  &betao  ); \
\
	bli_obj_init_finish( dt, m_a, n_a, ( void* )a, rs_a, cs_a, &ao ); \
	bli_obj_init_finish( dt, m_b, n_b, ( void* )b, rs_b, cs_b, &bo ); \
	bli_obj_init_finish( dt, m,   m,            c, rs_c, cs_c, &co ); \
\
	bli_obj_set_uplo( uploc, &co ); \
	bli_obj_set_conjtrans( transa, &ao ); \
	bli_obj_set_conjtrans( transb, &bo ); \
\
	bli_obj_set_struc( BLIS_SYMMETRIC, &co ); \
\
	PASTEMAC(opname,BLIS_OAPI_EX_SUF) \
	( \
	  &alphao, \
	  &ao, \
	  &bo, \
	  &betao, \
	  &co, \
	  cntx, \
	  rntm  \
	); \
}

INSERT_GENTFUNC_BASIC( syr2k )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname,BLIS_OAPI_EX_SUF) \
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
             ctype*  c, inc_t rs_c, inc_t cs_c, \
       const cntx_t* cntx, \
       const rntm_t* rntm  \
     ) \
{ \
	bli_init_once(); \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	obj_t       alphao = BLIS_OBJECT_INITIALIZER_1X1; \
	obj_t       ao     = BLIS_OBJECT_INITIALIZER; \
	obj_t       bo     = BLIS_OBJECT_INITIALIZER; \
	obj_t       betao  = BLIS_OBJECT_INITIALIZER_1X1; \
	obj_t       co     = BLIS_OBJECT_INITIALIZER; \
\
	dim_t       m_a, n_a; \
	dim_t       m_b, n_b; \
\
	bli_set_dims_with_trans( transa, m, k, &m_a, &n_a ); \
	bli_set_dims_with_trans( transb, k, m, &m_b, &n_b ); \
\
	bli_obj_init_finish_1x1( dt, ( void* )alpha, &alphao ); \
	bli_obj_init_finish_1x1( dt, ( void* )beta,  &betao  ); \
\
	bli_obj_init_finish( dt, m_a, n_a, ( void* )a, rs_a, cs_a, &ao ); \
	bli_obj_init_finish( dt, m_b, n_b, ( void* )b, rs_b, cs_b, &bo ); \
	bli_obj_init_finish( dt, m,   m,            c, rs_c, cs_c, &co ); \
\
	bli_obj_set_uplo( uploc, &co ); \
	bli_obj_set_conjtrans( transa, &ao ); \
	bli_obj_set_conjtrans( transb, &bo ); \
\
	PASTEMAC(opname,BLIS_OAPI_EX_SUF) \
	( \
	  &alphao, \
	  &ao, \
	  &bo, \
	  &betao, \
	  &co, \
	  cntx, \
	  rntm  \
	); \
}

INSERT_GENTFUNC_BASIC( gemmt )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname,BLIS_OAPI_EX_SUF) \
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
             ctype*  c, inc_t rs_c, inc_t cs_c, \
       const cntx_t* cntx, \
       const rntm_t* rntm  \
     ) \
{ \
	bli_init_once(); \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	obj_t       alphao = BLIS_OBJECT_INITIALIZER_1X1; \
	obj_t       ao     = BLIS_OBJECT_INITIALIZER; \
	obj_t       bo     = BLIS_OBJECT_INITIALIZER; \
	obj_t       betao  = BLIS_OBJECT_INITIALIZER_1X1; \
	obj_t       co     = BLIS_OBJECT_INITIALIZER; \
\
	dim_t       mn_a; \
	dim_t       m_b, n_b; \
\
	bli_set_dim_with_side(   side,   m, n, &mn_a ); \
	bli_set_dims_with_trans( transb, m, n, &m_b, &n_b ); \
\
	bli_obj_init_finish_1x1( dt, ( void* )alpha, &alphao ); \
	bli_obj_init_finish_1x1( dt, ( void* )beta,  &betao  ); \
\
	bli_obj_init_finish( dt, mn_a, mn_a, ( void* )a, rs_a, cs_a, &ao ); \
	bli_obj_init_finish( dt, m_b,  n_b,  ( void* )b, rs_b, cs_b, &bo ); \
	bli_obj_init_finish( dt, m,    n,             c, rs_c, cs_c, &co ); \
\
	bli_obj_set_uplo( uploa, &ao ); \
	bli_obj_set_diag( diaga, &ao ); \
	bli_obj_set_conjtrans( transa, &ao ); \
	bli_obj_set_conjtrans( transb, &bo ); \
\
	bli_obj_set_struc( BLIS_TRIANGULAR, &ao ); \
\
	PASTEMAC(opname,BLIS_OAPI_EX_SUF) \
	( \
	  side, \
	  &alphao, \
	  &ao, \
	  &bo, \
	  &betao, \
	  &co, \
	  cntx, \
	  rntm  \
	); \
}

INSERT_GENTFUNC_BASIC( trmm3 )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname,BLIS_OAPI_EX_SUF) \
     ( \
             side_t  side, \
             uplo_t  uploa, \
             trans_t transa, \
             diag_t  diaga, \
             dim_t   m, \
             dim_t   n, \
       const ctype*  alpha, \
       const ctype*  a, inc_t rs_a, inc_t cs_a, \
             ctype*  b, inc_t rs_b, inc_t cs_b, \
       const cntx_t* cntx, \
       const rntm_t* rntm  \
     ) \
{ \
	bli_init_once(); \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	obj_t       alphao = BLIS_OBJECT_INITIALIZER_1X1; \
	obj_t       ao     = BLIS_OBJECT_INITIALIZER; \
	obj_t       bo     = BLIS_OBJECT_INITIALIZER; \
\
	dim_t       mn_a; \
\
	bli_set_dim_with_side( side, m, n, &mn_a ); \
\
	bli_obj_init_finish_1x1( dt, ( void* )alpha, &alphao ); \
\
	bli_obj_init_finish( dt, mn_a, mn_a, ( void* )a, rs_a, cs_a, &ao ); \
	bli_obj_init_finish( dt, m,    n,             b, rs_b, cs_b, &bo ); \
\
	bli_obj_set_uplo( uploa, &ao ); \
	bli_obj_set_diag( diaga, &ao ); \
	bli_obj_set_conjtrans( transa, &ao ); \
\
	bli_obj_set_struc( BLIS_TRIANGULAR, &ao ); \
\
	PASTEMAC(opname,BLIS_OAPI_EX_SUF) \
	( \
	  side, \
	  &alphao, \
	  &ao, \
	  &bo, \
	  cntx, \
	  rntm  \
	); \
}

INSERT_GENTFUNC_BASIC( trmm )
INSERT_GENTFUNC_BASIC( trsm )

