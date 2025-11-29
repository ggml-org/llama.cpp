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

#ifndef BLIS_TINVERTS_H
#define BLIS_TINVERTS_H

// -- Implementation macro -----------------------------------------------------

#define bli_tinvertims( \
          \
          dx, px, xr, xi, \
          chc  \
        ) \
{ \
	PASTEMAC(dx,invertims) \
	( \
	  dx, px, xr, xi, \
	  chc  \
	); \
}

// -- real domain implementation --

// (xr) = 1.0 / (xr);

#define bli_rinvertims( \
          \
          dx, px, xr, xi, \
          chc  \
        ) \
{ \
	PASTEMAC(r,assigns) \
	( \
	  PASTEMAC(chc,px,tcast)( \
	    PASTEMAC(chc,div)( \
	      PASTEMAC(chc,one), \
	      PASTEMAC(px,chc,tcast)(xr)  \
	    ) \
	  ),\
	  PASTEMAC(px,zero), \
	  xr, \
	  xi \
	); \
}

// -- complex domain implementation --

// sr    = maxabs( xr, xi );
// xsr   = xr / sr;
// xsi   = xi / sr;
// tempr = xr * xsr + xi * xsi
// xr    =  xsr / tempr;
// xi    = -xsi / tempr;

#define bli_cinvertims( \
          \
          dx, px, xr, xi, \
          chc  \
        ) \
{ \
	PASTEMAC(ro,declinits) \
	( \
	  chc, \
	  PASTEMAC(chc,maxabs)( \
	    PASTEMAC(px,chc,tcast)(xr), \
	    PASTEMAC(px,chc,tcast)(xi)  \
	  ), \
	  sr  \
	) \
	PASTEMAC(c,declinits) \
	( \
	  chc, \
	  PASTEMAC(chc,div)( \
	    PASTEMAC(px,chc,tcast)(xr), \
	    sr  \
	  ), \
	  PASTEMAC(chc,div)( \
	    PASTEMAC(px,chc,tcast)(xi), \
	    sr  \
	  ), \
	  xsr, \
	  xsi \
	) \
	PASTEMAC(ro,declinits) \
	( \
	  chc, \
	  PASTEMAC(chc,add)( \
	    PASTEMAC(chc,mul)( \
	      PASTEMAC(px,chc,tcast)(xr), \
	      xsr  \
	    ), \
	    PASTEMAC(chc,mul)( \
	      PASTEMAC(px,chc,tcast)(xi), \
	      xsi  \
	    ) \
	  ), \
	  tempr  \
	) \
	PASTEMAC(c,assigns) \
	( \
	  PASTEMAC(chc,px,tcast)( \
	    PASTEMAC(chc,div)( \
	      xsr, \
	      tempr  \
	    ) \
	  ),\
	  PASTEMAC(chc,px,tcast)( \
	    PASTEMAC(chc,div)( \
	      PASTEMAC(PASTEMAC(chc,prec),neg)(xsi), \
	      tempr  \
	    ) \
	  ),\
	  xr, \
	  xi \
	); \
}

// -- API macros ---------------------------------------------------------------

// -- Consolidated --

// tinverts
#define bli_tinverts( chx, chc, x ) \
        bli_tinvertims \
        ( \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
          PASTEMAC(chx,real)(x), \
          PASTEMAC(chx,imag)(x), \
          PASTEMAC(chc,prec)  \
        )

// -- Exposed real/imaginary --

// tinvertris
#define bli_tinvertris( chx, chc, xr, xi ) \
        bli_tinvertims \
        ( \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
                     xr, \
                     xi, \
          PASTEMAC(chc,prec)  \
        )

// -- 1e / 1r --

// invert1es
#define bli_tinvert1es( chx, chc, xri, xir ) \
        bli_tinvertims \
        ( \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
          PASTEMAC(chx,real)(xri), \
          PASTEMAC(chx,imag)(xri), \
          PASTEMAC(chc,prec)  \
        ); \
        bli_tcopyims \
        ( \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
          PASTEMAC(PASTEMAC(chx,prec),neg)( \
            PASTEMAC(chx,imag)(xri)  \
          ), \
          PASTEMAC(chx,real)(xri), \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
          PASTEMAC(chx,real)(xir), \
          PASTEMAC(chx,imag)(xir) \
        )

// invert1rs
#define bli_tinvert1rs( chx, chc, xr, xi ) \
        bli_tinvertims \
        ( \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
          xr, \
          xi, \
          PASTEMAC(chc,prec)  \
        )

// -- Higher-level static functions --------------------------------------------

// -- Legacy macros ------------------------------------------------------------

#define bli_sinverts( x ) bli_tinverts( s,s, x )
#define bli_dinverts( x ) bli_tinverts( d,d, x )
#define bli_cinverts( x ) bli_tinverts( c,c, x )
#define bli_zinverts( x ) bli_tinverts( z,z, x )

// -- Notes --------------------------------------------------------------------

#endif

