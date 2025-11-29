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

#if 1

//
// Prototype BLAS-to-BLIS interfaces.
//
#undef  GENTPROTDOT
#define GENTPROTDOT( ftype, ch, chc, blasname ) \
\
BLIS_EXPORT_BLAS ftype PASTEF77(ch,blasname,chc) \
     ( \
       const f77_int* n, \
       const ftype*   x, const f77_int* incx, \
       const ftype*   y, const f77_int* incy  \
     );

INSERT_GENTPROTDOTR_BLAS( dot )

#ifdef BLIS_DISABLE_COMPLEX_RETURN_INTEL

INSERT_GENTPROTDOTC_BLAS( dot )

#else

// For the "intel" complex return type, we use a hidden parameter (passed by
// address) to return the result.
#undef  GENTPROTDOT
#define GENTPROTDOT( ftype, ch, chc, blasname ) \
\
BLIS_EXPORT_BLAS void PASTEF77(ch,blasname,chc) \
     ( \
       ftype*         rhop, \
       const f77_int* n, \
       const ftype*   x, const f77_int* incx, \
       const ftype*   y, const f77_int* incy  \
     );

INSERT_GENTPROTDOTC_BLAS( dot )

#endif


// -- "Black sheep" dot product function prototypes --

BLIS_EXPORT_BLAS float PASTEF77(sd,sdot)
     (
       const f77_int* n,
       const float*   sb,
       const float*   x, const f77_int* incx,
       const float*   y, const f77_int* incy
     );

BLIS_EXPORT_BLAS double PASTEF77(d,sdot)
     (
         const f77_int* n,
         const float*   x, const f77_int* incx,
         const float*   y, const f77_int* incy
     );

#endif
