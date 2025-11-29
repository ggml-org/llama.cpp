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


//
// Define BLAS-to-BLIS interfaces.
//
#undef  GENTFUNC
#define GENTFUNC( ftype, ch, blasname, blisname ) \
\
void PASTEF77(ch,blasname) \
     ( \
       const f77_char* uploa, \
       const f77_char* transa, \
       const f77_char* diaga, \
       const f77_int*  m, \
       const ftype*    a, const f77_int* lda, \
             ftype*    x, const f77_int* incx  \
     ) \
{ \
	uplo_t  blis_uploa; \
	trans_t blis_transa; \
	diag_t  blis_diaga; \
	dim_t   m0; \
	ftype*  x0; \
	inc_t   incx0; \
	ftype*  one_p; \
\
	/* Initialize BLIS. */ \
	bli_init_auto(); \
\
	/* Perform BLAS parameter checking. */ \
	PASTEBLACHK(blasname) \
	( \
	  MKSTR(ch), \
	  MKSTR(blasname), \
	  uploa, \
	  transa, \
	  diaga, \
	  m, \
	  lda, \
	  incx  \
	); \
\
	/* Map BLAS chars to their corresponding BLIS enumerated type value. */ \
	bli_param_map_netlib_to_blis_uplo( *uploa, &blis_uploa ); \
	bli_param_map_netlib_to_blis_trans( *transa, &blis_transa ); \
	bli_param_map_netlib_to_blis_diag( *diaga, &blis_diaga ); \
\
	/* Convert/typecast negative values of m to zero. */ \
	bli_convert_blas_dim1( *m, m0 ); \
\
	/* If the input increments are negative, adjust the pointers so we can
	   use positive increments instead. */ \
	bli_convert_blas_incv( m0, (ftype*)x, *incx, x0, incx0 ); \
\
	/* Set the row and column strides of A. */ \
	const inc_t rs_a = 1; \
	const inc_t cs_a = *lda; \
\
	/* Acquire a pointer to the global scalar constant BLIS_ONE. */ \
	one_p = PASTEMAC(ch,1); \
\
	/* Call BLIS interface. */ \
	PASTEMAC(ch,blisname,BLIS_TAPI_EX_SUF) \
	( \
	  blis_uploa, \
	  blis_transa, \
	  blis_diaga, \
	  m0, \
	  one_p, \
	  (ftype*)a,  rs_a, cs_a, \
	  x0, incx0, \
	  NULL, \
	  NULL  \
	); \
\
	/* Finalize BLIS. */ \
	bli_finalize_auto(); \
}

#ifdef BLIS_ENABLE_BLAS
INSERT_GENTFUNC_BLAS( trsv, trsv )
#endif

