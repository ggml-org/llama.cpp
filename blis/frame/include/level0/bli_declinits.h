/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2024, Southern Methodist University

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

#ifndef BLIS_DECLINITS_H
#define BLIS_DECLINITS_H

// declinits

// Notes:
// - The first char encodes the domain of output yr + yi.
// - The pxy precision character encodes the precision of x AND y (they
//   are assumed to be the same).
// - These macros are used to declare AND initialize variables corresponding
//   to the real and imaginary parts of (presumably) temporary variables.
//   If the domain is real, only the real part is declared and initialized.

#define bli_rdeclinits( pxy, xr, xi, yr, yi ) PASTEMAC(pxy,ctype) yr = xr; (void)yr; \
                                              PASTEMAC(pxy,ctype) yi;      (void)yi;
#define bli_cdeclinits( pxy, xr, xi, yr, yi ) PASTEMAC(pxy,ctype) yr = xr; (void)yr; \
                                              PASTEMAC(pxy,ctype) yi = xi; (void)yi;

// An extra definition for situations where we only need a real value declared
// and initialized (e.g. when explicitly implementing in the complex domain).
#define bli_rodeclinits( pxy, xr, yr ) PASTEMAC(pxy,ctype) yr = xr; (void)yr;


#endif

