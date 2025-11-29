/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.

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
// Define BLAS-to-BLIS interfaces.
//

#ifdef BLIS_BLAS3_CALLS_TAPI

#undef  GENTFUNCCO
#define GENTFUNCCO( ftype, ftype_r, ch, chr, blasname, blisname ) \
\
void PASTEF77(ch,blasname) \
     ( \
       const f77_char* transa, \
       const f77_char* transb, \
       const f77_int*  m, \
       const f77_int*  n, \
       const f77_int*  k, \
       const ftype*    alpha, \
       const ftype*    a, const f77_int* lda, \
       const ftype*    b, const f77_int* ldb, \
       const ftype*    beta, \
             ftype*    c, const f77_int* ldc  \
     ) \
{ \
	trans_t blis_transa; \
	trans_t blis_transb; \
	dim_t   m0, n0, k0; \
\
	/* Initialize BLIS. */ \
	bli_init_auto(); \
\
	/* Perform BLAS parameter checking. */ \
	PASTEBLACHK(blisname) \
	( \
	  MKSTR(ch), \
	  MKSTR(blasname), \
	  transa, \
	  transb, \
	  m, \
	  n, \
	  k, \
	  lda, \
	  ldb, \
	  ldc  \
	); \
\
	/* Map BLAS chars to their corresponding BLIS enumerated type value. */ \
	bli_param_map_netlib_to_blis_trans( *transa, &blis_transa ); \
	bli_param_map_netlib_to_blis_trans( *transb, &blis_transb ); \
\
	/* Typecast BLAS integers to BLIS integers. */ \
	bli_convert_blas_dim1( *m, m0 ); \
	bli_convert_blas_dim1( *n, n0 ); \
	bli_convert_blas_dim1( *k, k0 ); \
\
	/* Set the row and column strides of the matrix operands. */ \
	const inc_t rs_a = 1; \
	const inc_t cs_a = *lda; \
	const inc_t rs_b = 1; \
	const inc_t cs_b = *ldb; \
	const inc_t rs_c = 1; \
	const inc_t cs_c = *ldc; \
\
	/* As a placeholder, invoke 1m since BLIS does no longer contains an
	   official 3m implementation. Note that we do this by inlining an
	   abbreviated version of bli_gemm_ex() so that we can bypass
	   consideration of sup, which doesn't make sense in this context. */ \
	{ \
		cntx_t* cntx = ( cntx_t* )bli_gks_query_ind_cntx( BLIS_1M ); \
\
		rntm_t  rntm_l; \
		rntm_t* rntm = &rntm_l; \
		bli_rntm_init_from_global( rntm ); \
\
		/* Note that we MUST disable sup handling since it could redirect
		   execution for some problem sizes to a non-3m implementation. */ \
		bli_rntm_disable_l3_sup( rntm ); \
\
		/* Call BLIS interface. */ \
		PASTEMAC(ch,blisname,BLIS_TAPI_EX_SUF) \
		( \
		  blis_transa, \
		  blis_transb, \
		  m0, \
		  n0, \
		  k0, \
		  (ftype*)alpha, \
		  (ftype*)a, rs_a, cs_a, \
		  (ftype*)b, rs_b, cs_b, \
		  (ftype*)beta, \
		  (ftype*)c, rs_c, cs_c, \
		  cntx, \
		  rntm  \
		); \
	} \
\
	/* Finalize BLIS. */ \
	bli_finalize_auto(); \
}

#else

#undef  GENTFUNCCO
#define GENTFUNCCO( ftype, ftype_r, ch, chr, blasname, blisname ) \
\
void PASTEF77(ch,blasname) \
     ( \
       const f77_char* transa, \
       const f77_char* transb, \
       const f77_int*  m, \
       const f77_int*  n, \
       const f77_int*  k, \
       const ftype*    alpha, \
       const ftype*    a, const f77_int* lda, \
       const ftype*    b, const f77_int* ldb, \
       const ftype*    beta, \
             ftype*    c, const f77_int* ldc  \
     ) \
{ \
	trans_t blis_transa; \
	trans_t blis_transb; \
	dim_t   m0, n0, k0; \
\
	/* Initialize BLIS. */ \
	bli_init_auto(); \
\
	/* Perform BLAS parameter checking. */ \
	PASTEBLACHK(blisname) \
	( \
	  MKSTR(ch), \
	  MKSTR(blasname), \
	  transa, \
	  transb, \
	  m, \
	  n, \
	  k, \
	  lda, \
	  ldb, \
	  ldc  \
	); \
\
	/* Map BLAS chars to their corresponding BLIS enumerated type value. */ \
	bli_param_map_netlib_to_blis_trans( *transa, &blis_transa ); \
	bli_param_map_netlib_to_blis_trans( *transb, &blis_transb ); \
\
	/* Typecast BLAS integers to BLIS integers. */ \
	bli_convert_blas_dim1( *m, m0 ); \
	bli_convert_blas_dim1( *n, n0 ); \
	bli_convert_blas_dim1( *k, k0 ); \
\
	/* Set the row and column strides of the matrix operands. */ \
	const inc_t rs_a = 1; \
	const inc_t cs_a = *lda; \
	const inc_t rs_b = 1; \
	const inc_t cs_b = *ldb; \
	const inc_t rs_c = 1; \
	const inc_t cs_c = *ldc; \
\
	const num_t dt     = PASTEMAC(ch,type); \
\
	obj_t       alphao = BLIS_OBJECT_INITIALIZER_1X1; \
	obj_t       ao     = BLIS_OBJECT_INITIALIZER; \
	obj_t       bo     = BLIS_OBJECT_INITIALIZER; \
	obj_t       betao  = BLIS_OBJECT_INITIALIZER_1X1; \
	obj_t       co     = BLIS_OBJECT_INITIALIZER; \
\
	dim_t       m0_a, n0_a; \
	dim_t       m0_b, n0_b; \
\
	bli_set_dims_with_trans( blis_transa, m0, k0, &m0_a, &n0_a ); \
	bli_set_dims_with_trans( blis_transb, k0, n0, &m0_b, &n0_b ); \
\
	bli_obj_init_finish_1x1( dt, (ftype*)alpha, &alphao ); \
	bli_obj_init_finish_1x1( dt, (ftype*)beta,  &betao  ); \
\
	bli_obj_init_finish( dt, m0_a, n0_a, (ftype*)a, rs_a, cs_a, &ao ); \
	bli_obj_init_finish( dt, m0_b, n0_b, (ftype*)b, rs_b, cs_b, &bo ); \
	bli_obj_init_finish( dt, m0,   n0,   (ftype*)c, rs_c, cs_c, &co ); \
\
	bli_obj_set_conjtrans( blis_transa, &ao ); \
	bli_obj_set_conjtrans( blis_transb, &bo ); \
\
	/* As a placeholder, invoke bli_gemm_ex(). */ \
	PASTEMAC(blisname,BLIS_OAPI_EX_SUF) \
	( \
	  &alphao, \
	  &ao, \
	  &bo, \
	  &betao, \
	  &co, \
	  NULL, \
	  NULL \
	); \
\
	/* Finalize BLIS. */ \
	bli_finalize_auto(); \
}

#endif

#ifdef BLIS_ENABLE_BLAS
INSERT_GENTFUNCCO_BLAS( gemm3m, gemm )
#endif

