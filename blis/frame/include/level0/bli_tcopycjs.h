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

#ifndef BLIS_TCOPYCJS_H
#define BLIS_TCOPYCJS_H

// -- Implementation macro -----------------------------------------------------

// (yr) := (xr);
// (yi) := ( is_conj( conj ) ? -(xi) : (xi) );

#define bli_tcopycjims( \
          \
          conj, \
          dx, px, xr, xi, \
          dy, py, yr, yi \
        ) \
{ \
	PASTEMAC(dy,assigns) \
	( \
	  PASTEMAC(px,py,tcast)(xr), \
	  ( bli_is_conj( conj ) ? PASTEMAC(px,neg)( \
	                            PASTEMAC(px,py,tcast)(xi) \
	                          ) \
	                        :   PASTEMAC(px,py,tcast)(xi) \
	  ), \
	  yr, \
	  yi \
	); \
}

// -- API macros ---------------------------------------------------------------

// -- Consolidated --

// tcopycjs
#define bli_tcopycjs( chx, chy, conj, x, y ) \
        bli_tcopycjims \
        ( \
          conj, \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
          PASTEMAC(chx,real)(x), \
          PASTEMAC(chx,imag)(x), \
          PASTEMAC(chy,dom),  \
          PASTEMAC(chy,prec), \
          PASTEMAC(chy,real)(y), \
          PASTEMAC(chy,imag)(y) \
        )

// -- Exposed real/imaginary --

// tcopyris
#define bli_tcopycjris( chx, chy, conj, xr, xi, yr, yi ) \
        bli_tcopycjims \
        ( \
          conj, \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
                     xr, \
                     xi, \
          PASTEMAC(chy,dom),  \
          PASTEMAC(chy,prec), \
                     yr, \
                     yi \
        )

// -- Higher-level static functions --------------------------------------------

// -- Legacy macros ------------------------------------------------------------

#define bli_scopycjs( conj, x, y ) bli_tcopycjs( s,s, conj, x, y )
#define bli_dcopycjs( conj, x, y ) bli_tcopycjs( d,d, conj, x, y )
#define bli_ccopycjs( conj, x, y ) bli_tcopycjs( c,c, conj, x, y )
#define bli_zcopycjs( conj, x, y ) bli_tcopycjs( z,z, conj, x, y )

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
// (yi) :=   0 ;

//   c       c
// (yr) := (xr);
// (yi) := (xi);

#endif

