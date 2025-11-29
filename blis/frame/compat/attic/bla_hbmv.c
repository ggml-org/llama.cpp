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
#undef  GENTFUNCCO
#define GENTFUNCCO( ftype, ftype_r, ch, chr, blasname, blisname ) \
\
void PASTEF77(ch,blasname)( \
                            f77_char* uploa, \
                            f77_int*  m, \
                            f77_int*  k, \
                            ftype*    alpha, \
                            ftype*    a, f77_int* lda, \
                            ftype*    x, f77_int* incx, \
                            ftype*    beta, \
                            ftype*    y, f77_int* incy  \
                          ) \
{ \
	uplo_t  blis_uploa; \
	dim_t   m0; \
	ftype*  x0; \
	ftype*  y0; \
	inc_t   incx0; \
	inc_t   incy0; \
\
	/* Map BLAS chars to their corresponding BLIS enumerated type value. */ \
	bli_param_map_netlib_to_blis_uplo( *uploa, &blis_uploa ); \
\
	/* Convert/typecast negative values of m to zero. */ \
	bli_convert_blas_dim1( *m, m0 ); \
\
	/* If the input increments are negative, adjust the pointers so we can
	   use positive increments instead. */ \
	bli_convert_blas_incv( m0, x, *incx, x0, incx0 ); \
	bli_convert_blas_incv( m0, y, *incy, y0, incy0 ); \
\
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED ); \
}

#ifdef BLIS_ENABLE_BLAS
INSERT_GENTFUNCCO_BLAS( hbmv, hbmv )
#endif

