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

#ifndef BLIS_TCOPYNZS_H
#define BLIS_TCOPYNZS_H

// -- Implementation macro -----------------------------------------------------

// (yr) := (xr);
// if ( is_complex( x ) )
//     (yi) := (xi);

#define bli_tcopynzims( \
          \
          dx, px, xr, xi, \
          dy, py, yr, yi \
        ) \
{ \
	PASTEMAC(dx,dy,copynzims) \
	( \
	  dx, px, xr, xi, \
	  dy, py, yr, yi \
	); \
}

// -- real-to-real domain implementation --

#define bli_rrcopynzims( \
          \
          dx, px, xr, xi, \
          dy, py, yr, yi \
        ) \
{ \
	PASTEMAC(r,assigns) \
	( \
	  PASTEMAC(px,py,tcast)(xr), \
	  PASTEMAC(px,py,tcast)(xi), \
	  yr, \
	  yi \
	); \
}

// -- complex-to-real domain implementation --
// -- real-to-complex domain implementation --

// NOTE: Normally, the real-to-complex case would take place in the complex
// domain (in that an implicit zero would be copied to y.imag), but since
// this is copynz, we avoid updating the imaginary parts of complex y when
// x is real. Thus, real-to-complex ends up getting implemented the same as
// real-to-real (and complex-to-real).

#define bli_rccopynzims bli_rrcopynzims
#define bli_crcopynzims bli_rrcopynzims

// -- complex-to-complex domain implementation --

#define bli_cccopynzims( \
          \
          dx, px, xr, xi, \
          dy, py, yr, yi \
        ) \
{ \
	PASTEMAC(c,assigns) \
	( \
	  PASTEMAC(px,py,tcast)(xr), \
	  PASTEMAC(px,py,tcast)(xi), \
	  yr, \
	  yi \
	); \
}

// -- API macros ---------------------------------------------------------------

// -- Consolidated --

// tcopynzs
#define bli_tcopynzs( chx, chy, x, y ) \
        bli_tcopynzims \
        ( \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
          PASTEMAC(chx,real)(x), \
          PASTEMAC(chx,imag)(x), \
          PASTEMAC(chy,dom),  \
          PASTEMAC(chy,prec), \
          PASTEMAC(chy,real)(y), \
          PASTEMAC(chy,imag)(y) \
        )

// tcopyjnzs
#define bli_tcopyjnzs( chx, chy, x, y ) \
        bli_tcopynzims \
        ( \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
          PASTEMAC(chx,real)(x), \
          PASTEMAC(PASTEMAC(chx,prec),neg)( \
            PASTEMAC(chx,imag)(x)  \
          ), \
          PASTEMAC(chy,dom),  \
          PASTEMAC(chy,prec), \
          PASTEMAC(chy,real)(y), \
          PASTEMAC(chy,imag)(y) \
        )

// -- Exposed real/imaginary --

// tcopynzris
#define bli_tcopynzris( chx, chy, xr, xi, yr, yi ) \
        bli_tcopynzims \
        ( \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
                     xr, \
                     xi, \
          PASTEMAC(chy,dom),  \
          PASTEMAC(chy,prec), \
                     yr, \
                     yi \
        )

// tcopyjnzris
#define bli_tcopyjnzris( chx, chy, xr, xi, yr, yi ) \
        bli_tcopynzims \
        ( \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
                     xr, \
          PASTEMAC(PASTEMAC(chx,prec),neg)( \
                     xi ), \
          PASTEMAC(chy,dom),  \
          PASTEMAC(chy,prec), \
                     yr, \
                     yi \
        )

// -- Higher-level static functions --------------------------------------------

// -- Notes --------------------------------------------------------------------

// -- Domain cases --

//   r       r
// (yr) := (xr);
// (yi) xx   0 ;

//   r       c
// (yr) := (xr);
// (yi) xx (xi);

//   c       r
// (yr) := (xr);
// (yi) xx   0 ;    // NOTE: This is what copynzs does differently from copys.

//   c       c
// (yr) := (xr);
// (yi) := (xi);

#endif

