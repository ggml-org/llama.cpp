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

#ifndef BLIS_TABVAL2S_H
#define BLIS_TABVAL2S_H

// -- Implementation macro -----------------------------------------------------

#define bli_tabval2ims( \
          \
          dx, px, xr, xi, \
          dy, py, yr, yi, \
          chc  \
        ) \
{ \
	PASTEMAC(dx,abval2ims) \
	( \
	  dx, px, xr, xi, \
	  dy, py, yr, yi, \
	  chc  \
	); \
}

// -- real-to-real domain implementation --
// -- real-to-complex domain implementation --

// (yr) = abs( xr );
// (yi) = 0;

#define bli_rabval2ims( \
          \
          dx, px, xr, xi, \
          dy, py, yr, yi, \
          chc  \
        ) \
{ \
	PASTEMAC(dy,assigns) \
	( \
	  PASTEMAC(chc,py,tcast)( \
	    PASTEMAC(chc,abs)( \
	      PASTEMAC(px,chc,tcast)(xr) \
	    ) \
	  ), \
	  PASTEMAC(py,zero), \
	  yr, \
	  yi \
	) \
} \

// -- complex-to-real domain implementation --
// -- complex-to-complex domain implementation --

// NOTE: Instead of defining abval2 in terms of bli_?hypot(), we use an
// alternate definition that can avoid overflow in the final result due
// to overflow in the intermediate results (e.g. xr * xr and xi * xi).

// xmaxr = maxabs( xr, xi );
// if ( s == 0.0 ) mag = 0.0;
// else            mag = sqrt( xmaxr ) *
//                       sqrt( ( xr / xmaxr ) * xr +
//                             ( xi / xmaxr ) * xi );
// yr = mag;
// yi = 0.0;

#define bli_cabval2ims( \
          \
          dx, px, xr, xi, \
          dy, py, yr, yi, \
          chc  \
        ) \
{ \
	PASTEMAC(ro,declinits) \
	( \
	  px, \
	  PASTEMAC(px,maxabs)(xr,xi), \
	  xmaxr  \
	) \
	PASTEMAC(dy,assigns) \
	( \
	  ( PASTEMAC(teq0s)(px,xmaxr) && \
	    !PASTEMAC(px,isnan)(xi) && \
	    !PASTEMAC(px,isnan)(xr) \
	    ? PASTEMAC(py,zero) \
	    : PASTEMAC(chc,py,tcast)( \
	        PASTEMAC(chc,mul)( \
	          PASTEMAC(chc,sqrt)( \
	            PASTEMAC(px,chc,tcast)(xmaxr) \
	          ), \
	          PASTEMAC(chc,sqrt)( \
	            PASTEMAC(chc,add)( \
	              PASTEMAC(chc,mul)( \
	                PASTEMAC(px,chc,tcast)(xr), \
	                PASTEMAC(chc,div)( \
	                  PASTEMAC(px,chc,tcast)(xr), \
	                  PASTEMAC(px,chc,tcast)(xmaxr) \
	                ) \
	              ), \
	              PASTEMAC(chc,mul)( \
	                PASTEMAC(px,chc,tcast)(xi), \
	                PASTEMAC(chc,div)( \
	                  PASTEMAC(px,chc,tcast)(xi), \
	                  PASTEMAC(px,chc,tcast)(xmaxr) \
                    ) \
	              ) \
	            ) \
	          ) \
	        ) \
	      ) \
	  ), \
	  PASTEMAC(py,zero), \
	  yr, \
	  yi \
	) \
}

// -- API macros ---------------------------------------------------------------

// -- Consolidated --

// tabval2s
#define bli_tabval2s( chx, chy, chc, x, y ) \
        bli_tabval2ims \
        ( \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
          PASTEMAC(chx,real)(x), \
          PASTEMAC(chx,imag)(x), \
          PASTEMAC(chy,dom),  \
          PASTEMAC(chy,prec), \
          PASTEMAC(chy,real)(y), \
          PASTEMAC(chy,imag)(y), \
          PASTEMAC(chc,prec)  \
        )

// -- Exposed real/imaginary --

// tabval2ris
#define bli_tabval2ris( chx, chy, chc, xr, xi, yr, yi ) \
        bli_tabval2ims \
        ( \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
                     xr, \
                     xi, \
          PASTEMAC(chy,dom),  \
          PASTEMAC(chy,prec), \
                     yr, \
                     yi, \
          PASTEMAC(chc,prec)  \
        )

// -- Higher-level static functions --------------------------------------------

// -- Notes --------------------------------------------------------------------

// -- Domain cases --

//   r       r
// (yr) := abs(xr);
// (yi) xx   0 ;

//   r       c
// (yr) := sqrt(s) * sqrt( ( xr / s ) * xr + ( xi / s ) * xi );
// (yi) xx   0 ;

//   c       r
// (yr) := abs(xr);
// (yi) :=   0 ;

//   c       c
// (yr) := sqrt(s) * sqrt( ( xr / s ) * xr + ( xi / s ) * xi );
// (yi) :=   0 ;

#endif

