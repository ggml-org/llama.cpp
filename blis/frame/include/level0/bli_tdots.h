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

#ifndef BLIS_TDOTS_H
#define BLIS_TDOTS_H

// -- Implementation macro -----------------------------------------------------

// (cr) += (ar) * (br) - (ai) * (bi);
// (ci) += (ai) * (br) + (ar) * (bi);

#define bli_tdotims bli_taxpyims

// -- API macros ---------------------------------------------------------------

// NOTE: When defining the tdots macros, we can recycle taxpys since they both
// perform c += a * b. However, when invoking taxpys, the first two operands
// passed in must be swapped because in BLIS axpy is set up to conjugate its
// second operand (ie: the second operand to the a*x product) while dot
// is set up to conjugate its first operand (ie: the first operand to the x*y
// product).

// -- Consolidated --

// tdots
#define  bli_tdots( chx,chy,cha,chc, x, y, a ) \
        bli_taxpys( chy,chx,cha,chc, y, x, a )

// tdotjs
#define  bli_tdotjs( chx,chy,cha,chc, x, y, a ) \
        bli_taxpyjs( chy,chx,cha,chc, y, x, a )

// -- Exposed real/imaginary --

// tdotris
#define  bli_tdotris( chx,chy,cha,chc, xr, xi, yr, yi, ar, ai ) \
        bli_taxpyris( chy,chx,cha,chc, yr, yi, xr, yx, ar, ai )

// tdotjris
#define  bli_tdotjris( chx,chy,cha,chc, xr, xi, yr, yi, ar, ai ) \
        bli_taxpyjris( chy,chx,cha,chc, yr, yi, xr, yx, ar, ai )

// -- Higher-level static functions --------------------------------------------

// -- Legacy macros ------------------------------------------------------------

#define bli_sdots( x, y, a ) bli_tdots( s,s,s,s, x, y, a )
#define bli_ddots( x, y, a ) bli_tdots( d,d,d,d, x, y, a )
#define bli_cdots( x, y, a ) bli_tdots( c,c,c,s, x, y, a )
#define bli_zdots( x, y, a ) bli_tdots( z,z,z,d, x, y, a )

// -- Notes --------------------------------------------------------------------

// -- Domain cases --

//   r       r      r
// (yr) += (ar) * (xr) -   0  *   0 ;
// (yi) xx   0  * (xr) + (ar) *   0 ;

//   r       r      c
// (yr) += (ar) * (xr) -   0  * (xi);
// (yi) xx   0  * (xr) + (ar) * (xi);

//   r       c      r
// (yr) += (ar) * (xr) - (ai) *   0 ;
// (yi) xx (ai) * (xr) + (ar) *   0 ;

//   r       c      c
// (yr) += (ar) * (xr) - (ai) * (xi);
// (yi) xx (ai) * (xr) + (ar) * (xi);

//   c       r      r
// (yr) += (ar) * (xr) -   0  *   0 ;
// (yi) +=   0  * (xr) + (ar) *   0 ;

//   c       r      c
// (yr) += (ar) * (xr) -   0  * (xi);
// (yi) +=   0  * (xr) + (ar) * (xi);

//   c       c      r
// (yr) += (ar) * (xr) - (ai) *   0 ;
// (yi) += (ai) * (xr) + (ar) *   0 ;

//   c       c      c
// (yr) += (ar) * (xr) - (ai) * (xi);
// (yi) += (ai) * (xr) + (ar) * (xi);

#endif

