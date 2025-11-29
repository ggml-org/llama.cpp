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

#ifndef BLIS_TSQRT2S_H
#define BLIS_TSQRT2S_H

// -- Implementation macro -----------------------------------------------------

#define bli_tsqrt2ims( \
          \
          dx, px, xr, xi, \
          dy, py, yr, yi, \
          chc  \
        ) \
{ \
	PASTEMAC(dx,sqrtims) \
	( \
	  dx, px, xr, xi, \
	  dy, py, yr, yi, \
	  chc  \
	); \
}

// -- real domain implementation --

// yr = sqrt( xr );

#define bli_rsqrtims( \
          \
          dx, px, xr, xi, \
          dy, py, yr, yi, \
          chc  \
        ) \
{ \
	PASTEMAC(dy,assigns) \
	( \
	  PASTEMAC(chc,py,tcast)( \
	    PASTEMAC(chc,sqrt)( \
	      PASTEMAC(px,chc,tcast)( xr )  \
	    ) \
	  ), \
	  PASTEMAC(py,zero), \
	  yr, \
	  yi \
	); \
}

// -- complex domain implementation --

// mag = hypot( xr, xi );
// tr  = sqrt( ( mag + xr ) / 2.0 );
// ti  = sqrt( ( mag - xr ) / 2.0 );
// yr  =           tr;
// yi  = copysign( ti, xi );

#define bli_csqrtims( \
          \
          dx, px, xr, xi, \
          dy, py, yr, yi, \
          chc  \
        ) \
{ \
	PASTEMAC(ro,declinits) \
	( \
	  chc, \
	  PASTEMAC(chc,hypot)( \
	    PASTEMAC(px,chc,tcast)( xr ), \
	    PASTEMAC(px,chc,tcast)( xi )  \
	  ), \
	  mag  \
	) \
	PASTEMAC(c,declinits) \
	( \
	  chc, \
	  PASTEMAC(chc,sqrt)( \
	    PASTEMAC(chc,div)( \
	      PASTEMAC(chc,add)( \
	        mag, \
	        PASTEMAC(px,chc,tcast)( xr ) \
	      ), \
	      PASTEMAC(chc,two) \
	    ) \
	  ), \
	  PASTEMAC(chc,sqrt)( \
	    PASTEMAC(chc,div)( \
	      PASTEMAC(chc,sub)( \
	        mag, \
	        PASTEMAC(px,chc,tcast)( xr ) \
	      ), \
	      PASTEMAC(chc,two) \
	    ) \
	  ), \
	  tr, \
	  ti \
	) \
	PASTEMAC(dy,assigns) \
	( \
	  PASTEMAC(chc,py,tcast)( tr ), \
	  PASTEMAC(chc,py,tcast)( \
	    PASTEMAC(chc,copysign)( ti, xi ) \
	  ), \
	  yr, \
	  yi \
	); \
}

// -- API macros ---------------------------------------------------------------

// -- Consolidated --

// tsqrt2s
#define bli_tsqrt2s( chx, chy, chc, x, y ) \
        bli_tsqrt2ims \
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

// tsqrt2ris
#define bli_tsqrt2ris( chx, chy, chc, xr, xi, yr, yi ) \
        bli_tsqrt2ims \
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
// (yr) := sqrt(xr);
// (yi) xx       0 ;

//   r       c
// (yr) := sqrt(xr ...);
// (yi) xx copysign(xi ...);

//   c       r
// (yr) := sqrt(xr);
// (yi) :=       0 ;

//   c       c
// (yr) := sqrt(xr ... );
// (yi) := copysign(xi ...);

#endif

