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

#ifndef BLIS_COMPLEX_TERMS_H
#define BLIS_COMPLEX_TERMS_H


// -- Complex term-zeroing macros ----------------------------------------------

// Note:
// - pab is the precision of the A_[ri] * B_[ri] product. It is only used in
//   certain cases where we need to decide which precision of zero to substitute
//   into the calling expression.

// ar * br term
#define bli_rrtermrr( pab, ab )  ab
#define bli_rctermrr( pab, ab )  ab
#define bli_crtermrr( pab, ab )  ab
#define bli_cctermrr( pab, ab )  ab

// ai * bi term
#define bli_rrtermii( pab, ab )  ( (void)(ab), PASTEMAC(pab,zero) )
#define bli_rctermii( pab, ab )  ( (void)(ab), PASTEMAC(pab,zero) )
#define bli_crtermii( pab, ab )  ( (void)(ab), PASTEMAC(pab,zero) )
#define bli_cctermii( pab, ab )  ab

// ai * br term
#define bli_rrtermir( pab, ab )  ( (void)(ab), PASTEMAC(pab,zero) )
#define bli_rctermir( pab, ab )  ( (void)(ab), PASTEMAC(pab,zero) )
#define bli_crtermir( pab, ab )  ab
#define bli_cctermir( pab, ab )  ab

// ar * bi term
#define bli_rrtermri( pab, ab )  ( (void)(ab), PASTEMAC(pab,zero) )
#define bli_rctermri( pab, ab )  ab
#define bli_crtermri( pab, ab )  ( (void)(ab), PASTEMAC(pab,zero) )
#define bli_cctermri( pab, ab )  ab



#endif
