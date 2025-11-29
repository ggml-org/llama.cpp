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

#ifndef BLIS_TSWAPS_H
#define BLIS_TSWAPS_H

// -- Implementation macro -----------------------------------------------------

// (tr) := (yr);
// (ti) := (yi);
// (yr) := (xr);
// (yi) := (xi);
// (xr) := (tr);
// (xi) := (ti);

#define bli_tswapims( \
          \
          dx, px, xr, xi, \
          dy, py, yr, yi \
        ) \
{ \
	/* It is important to use dx (or, alternatively, 'c') in the declinits macro
	   so that in the case of chy chx = r c, ti gets set to zero. The c r case
	   also works since ti, while unset by declinits, is unused by assigns. */ \
	PASTEMAC(dx,declinits)( py, yr, yi, tr, ti ) \
	PASTEMAC(dy,assigns) \
	( \
	  PASTEMAC(px,py,tcast)(xr),\
	  PASTEMAC(px,py,tcast)(xi), \
	  yr, \
	  yi \
	); \
	PASTEMAC(dx,assigns) \
	( \
	  PASTEMAC(py,px,tcast)(tr),\
	  PASTEMAC(py,px,tcast)(ti), \
	  xr, \
	  xi \
	); \
}

// -- API macros ---------------------------------------------------------------

// -- Consolidated --

// tswaps
#define bli_tswaps( chx, chy, x, y ) \
        bli_tswapims \
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

// -- Exposed real/imaginary --

// tswapris
#define bli_tswapris( chx, chy, xr, xi, yr, yi ) \
        bli_tswapims \
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

// -- Higher-level static functions --------------------------------------------

// -- Legacy macros ------------------------------------------------------------


#define bli_sswaps( x, y ) bli_tswaps( s,s, x, y )
#define bli_dswaps( x, y ) bli_tswaps( d,d, x, y )
#define bli_cswaps( x, y ) bli_tswaps( c,c, x, y )
#define bli_zswaps( x, y ) bli_tswaps( z,z, x, y )

// -- Notes --------------------------------------------------------------------

// -- Domain cases --

// chy chx: r  r
// (tr) := (yr);
// (ti) :=   0 ;
// (yr) := (xr);
// (yi) xx (xi);
// (xr) := (tr);
// (xi) xx (ti);

// chy chx: r  c
// (tr) := (yr);
// (ti) :=   0 ;
// (yr) := (xr);
// (yi) xx (xi);
// (xr) := (tr);
// (xi) := (ti);

// chy chx: c  r
// (tr) := (yr);
// (ti) xx (yi);
// (yr) := (xr);
// (yi) :=   0 ;
// (xr) := (tr);
// (xi) xx (ti);

// chy chx: c  c
// (tr) := (yr);
// (ti) := (yi);
// (yr) := (xr);
// (yi) := (xi);
// (xr) := (tr);
// (xi) := (ti);

#endif

