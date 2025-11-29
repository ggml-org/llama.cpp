/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2022, Advanced Micro Devices, Inc.

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
#define GENTFUNC( ftype, ch, blasname, blisname, isuf ) \
\
void PASTEF77(ch,blasname) \
     ( \
       const f77_int* n, \
       const ftype*   x, const f77_int* incx, \
             ftype*   y, const f77_int* incy  \
     ) \
{ \
	dim_t  n0; \
	ftype* x0; \
	ftype* y0; \
	inc_t  incx0; \
	inc_t  incy0; \
\
	/* Initialize BLIS. */ \
	/*bli_init_auto()*/; \
\
	/* Convert/typecast negative values of n to zero. */ \
	bli_convert_blas_dim1( *n, n0 ); \
\
	/* If the input increments are negative, adjust the pointers so we can
	   use positive increments instead. */ \
	bli_convert_blas_incv( n0, (ftype*)x, *incx, x0, incx0 ); \
	bli_convert_blas_incv( n0, (ftype*)y, *incy, y0, incy0 ); \
\
	/* Call BLIS interface. */ \
	/* NOTE: While we skip explicit initialization for real domain instances
	   since we call the microkernel directly, the complex domain instances
	   still need initialization so that they can query valid contexts from
	   gks. However, the expert API will self-initialize before attempting
	   to query a context, so the complex domain cases should work fine. */ \
	PASTEMAC(ch,blisname,isuf) \
	( \
	  BLIS_NO_CONJUGATE, \
	  n0, \
	  x0, incx0, \
	  y0, incy0, \
	  NULL  \
	); \
\
	/* Finalize BLIS. */ \
	/*bli_finalize_auto();*/ \
}

#ifdef BLIS_ENABLE_BLAS
//INSERT_GENTFUNC_BLAS( copy, copyv )
GENTFUNC( float,    s, copy, copyv, _zen_int )
GENTFUNC( double,   d, copy, copyv, _zen_int )
#endif


#undef  GENTFUNC
#define GENTFUNC( ftype, ch, blasname, blisname, isuf ) \
\
void PASTEF77(ch,blasname) \
     ( \
       const f77_int* n, \
       const ftype*   x, const f77_int* incx, \
             ftype*   y, const f77_int* incy  \
     ) \
{ \
	dim_t  n0; \
	ftype* x0; \
	ftype* y0; \
	inc_t  incx0; \
	inc_t  incy0; \
\
	/* Initialize BLIS. */ \
	/*bli_init_auto()*/; \
\
	/* Convert/typecast negative values of n to zero. */ \
	bli_convert_blas_dim1( *n, n0 ); \
\
	/* If the input increments are negative, adjust the pointers so we can
	   use positive increments instead. */ \
	bli_convert_blas_incv( n0, (ftype*)x, *incx, x0, incx0 ); \
	bli_convert_blas_incv( n0, (ftype*)y, *incy, y0, incy0 ); \
\
	/* Call BLIS interface. */ \
	/* NOTE: While we skip explicit initialization for real domain instances
	   since we call the microkernel directly, the complex domain instances
	   still need initialization so that they can query valid contexts from
	   gks. However, the expert API will self-initialize before attempting
	   to query a context, so the complex domain cases should work fine. */ \
	PASTEMAC(ch,blisname,isuf) \
	( \
	  BLIS_NO_CONJUGATE, \
	  n0, \
	  x0, incx0, \
	  y0, incy0, \
	  NULL, \
	  NULL  \
	); \
\
	/* Finalize BLIS. */ \
	/*bli_finalize_auto();*/ \
}

#ifdef BLIS_ENABLE_BLAS
//INSERT_GENTFUNC_BLAS( copy, copyv )
GENTFUNC( scomplex, c, copy, copyv, _ex )
GENTFUNC( dcomplex, z, copy, copyv, _ex )
#endif

