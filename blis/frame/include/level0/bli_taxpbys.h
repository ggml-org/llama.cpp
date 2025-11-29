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

#ifndef BLIS_TAXPBYS_H
#define BLIS_TAXPBYS_H

// -- Implementation macro -----------------------------------------------------

// (yorigr) := (yr)
// (yorigi) := (yi)
// (yr) := (ar) * (xr) - (ai) * (xi) + (br) * (yorigr) - (bi) * (yorigi);
// (yi) := (ai) * (xr) + (ar) * (xi) + (bi) * (yorigr) + (br) * (yorigi);

#define bli_taxpbyims( \
          \
          da, pa, ar, ai, \
          dx, px, xr, xi, \
          db, pb, br, bi, \
          dy, py, yr, yi, \
          chc  \
        ) \
{ \
	PASTEMAC(c,declinits) \
	( \
	  py, \
	  PASTEMAC(chc,py,tcast)( \
	    PASTEMAC(chc,add)( \
	      PASTEMAC(chc,sub)( \
	        PASTEMAC(da,dx,termrr)( \
	          chc, \
	          PASTEMAC(chc,mul)( \
	            PASTEMAC(pa,chc,tcast)(ar), \
	            PASTEMAC(px,chc,tcast)(xr) \
	          )  \
	        ), \
	        PASTEMAC(da,dx,termii)( \
	          chc, \
	          PASTEMAC(chc,mul)( \
	            PASTEMAC(pa,chc,tcast)(ai), \
	            PASTEMAC(px,chc,tcast)(xi) \
	          )  \
	        )  \
	      ), \
	      PASTEMAC(chc,sub)( \
	        PASTEMAC(db,dy,termrr)( \
	          chc, \
	          PASTEMAC(chc,mul)( \
	            PASTEMAC(pb,chc,tcast)(br), \
	            PASTEMAC(py,chc,tcast)(yr) \
	          )  \
	        ), \
	        PASTEMAC(db,dy,termii)( \
	          chc, \
	          PASTEMAC(chc,mul)( \
	            PASTEMAC(pb,chc,tcast)(bi), \
	            PASTEMAC(py,chc,tcast)(yi) \
	          ) \
	        ) \
	      ) \
	    ) \
	  ),\
	  PASTEMAC(chc,py,tcast)( \
	    PASTEMAC(chc,add)( \
	      PASTEMAC(chc,add)( \
	        PASTEMAC(da,dx,termir)( \
	          chc, \
	          PASTEMAC(chc,mul)( \
	            PASTEMAC(pa,chc,tcast)(ai), \
	            PASTEMAC(px,chc,tcast)(xr) \
	          )  \
	        ), \
	        PASTEMAC(da,dx,termri)( \
	          chc, \
	          PASTEMAC(chc,mul)( \
	            PASTEMAC(pa,chc,tcast)(ar), \
	            PASTEMAC(px,chc,tcast)(xi) \
	          ) \
	        ) \
	      ), \
	      PASTEMAC(chc,add)( \
	        PASTEMAC(db,dy,termir)( \
	          chc, \
	          PASTEMAC(chc,mul)( \
	            PASTEMAC(pb,chc,tcast)(bi), \
	            PASTEMAC(py,chc,tcast)(yr) \
	          )  \
	        ), \
	        PASTEMAC(db,dy,termri)( \
	          chc, \
	          PASTEMAC(chc,mul)( \
	            PASTEMAC(pb,chc,tcast)(br), \
	            PASTEMAC(py,chc,tcast)(yi) \
	          ) \
	        ) \
	      ) \
	    ) \
	  ), \
	  tr, \
	  ti \
	); \
	PASTEMAC(dy,assigns) \
	( \
	  tr, \
	  ti, \
	  yr, \
	  yi \
	); \
}

// -- API macros ---------------------------------------------------------------

// -- Consolidated --

// taxpbys
#define bli_taxpbys( cha, chx, chb, chy, chc, a, x, b, y ) \
        bli_taxpbyims \
        ( \
          PASTEMAC(cha,dom),  \
          PASTEMAC(cha,prec), \
          PASTEMAC(cha,real)(a), \
          PASTEMAC(cha,imag)(a), \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
          PASTEMAC(chx,real)(x), \
          PASTEMAC(chx,imag)(x), \
          PASTEMAC(chb,dom),  \
          PASTEMAC(chb,prec), \
          PASTEMAC(chb,real)(b), \
          PASTEMAC(chb,imag)(b), \
          PASTEMAC(chy,dom),  \
          PASTEMAC(chy,prec), \
          PASTEMAC(chy,real)(y), \
          PASTEMAC(chy,imag)(y), \
          PASTEMAC(chc,prec)  \
        )

// taxpbyjs
#define bli_taxpbyjs( cha, chx, chb, chy, chc, a, x, b, y ) \
        bli_taxpbyims \
        ( \
          PASTEMAC(cha,dom),  \
          PASTEMAC(cha,prec), \
          PASTEMAC(cha,real)(a), \
          PASTEMAC(cha,imag)(a), \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
          PASTEMAC(chx,real)(x), \
          PASTEMAC(PASTEMAC(chx,prec),neg)( \
            PASTEMAC(chx,imag)(x)  \
          ), \
          PASTEMAC(chb,dom),  \
          PASTEMAC(chb,prec), \
          PASTEMAC(chb,real)(b), \
          PASTEMAC(chb,imag)(b), \
          PASTEMAC(chy,dom),  \
          PASTEMAC(chy,prec), \
          PASTEMAC(chy,real)(y), \
          PASTEMAC(chy,imag)(y), \
          PASTEMAC(chc,prec)  \
        )

// -- Exposed real/imaginary --

// taxpbyris
#define bli_taxpbyris( cha, chx, chb, chy, chc, ar, ai, xr, xi, br, bi, yr, yi ) \
        bli_taxpbyims \
        ( \
          PASTEMAC(cha,dom),  \
          PASTEMAC(cha,prec), \
                     ar, \
                     ai, \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
                     xr, \
                     xi, \
          PASTEMAC(chb,dom),  \
          PASTEMAC(chb,prec), \
                     br, \
                     bi, \
          PASTEMAC(chy,dom),  \
          PASTEMAC(chy,prec), \
                     yr, \
                     yi, \
          PASTEMAC(chc,prec)  \
        )

// taxpbyjris
#define bli_taxpbyjris( cha, chx, chb, chy, chc, ar, ai, xr, xi, br, bi, yr, yi ) \
        bli_taxpbyims \
        ( \
          PASTEMAC(cha,dom),  \
          PASTEMAC(cha,prec), \
                     ar, \
                     ai, \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
                     xr, \
          PASTEMAC(PASTEMAC(chx,prec),neg)( \
                     xi ), \
          PASTEMAC(chb,dom),  \
          PASTEMAC(chb,prec), \
                     br, \
                     bi, \
          PASTEMAC(chy,dom),  \
          PASTEMAC(chy,prec), \
                     yr, \
                     yi, \
          PASTEMAC(chc,prec)  \
        )

// -- Higher-level static functions --------------------------------------------

// -- mxn --

// axpbys_mxn
#define bli_taxpbys_mxn( cha, chx, chb, chy, chc, m, n, alpha, x, rs_x, cs_x, beta, y, rs_y, cs_y ) \
{ \
\
	/* If beta is zero, overwrite y with x (in case y has infs or NaNs). */ \
	if ( bli_teq0s( chb, *(beta) ) ) \
	{ \
		bli_tscal2s_mxn( cha, chx, chy, chc, BLIS_NO_CONJUGATE, m, n, alpha, x, rs_x, cs_x, y, rs_y, cs_y ); \
	} \
	else \
	{ \
		for ( dim_t jj = 0; jj < n; ++jj ) \
		for ( dim_t ii = 0; ii < m; ++ii ) \
		{ \
			PASTEMAC(chx,ctype)* restrict xij = ( PASTEMAC(chx,ctype)* )(x) + ii*(rs_x) + jj*(cs_x); \
			PASTEMAC(chy,ctype)* restrict yij = ( PASTEMAC(chy,ctype)* )(y) + ii*(rs_y) + jj*(cs_y); \
\
			bli_taxpbys( cha,chx,chb,chy,chc, *(alpha), *xij, *(beta), *yij ); \
		} \
	} \
}

// -- Legacy macros ------------------------------------------------------------

#define bli_saxpbys( a, x, b, y ) bli_taxpbys( s,s,s,s,s, a, x, b, y )
#define bli_daxpbys( a, x, b, y ) bli_taxpbys( d,d,d,d,d, a, x, b, y )
#define bli_caxpbys( a, x, b, y ) bli_taxpbys( c,c,c,c,s, a, x, b, y )
#define bli_zaxpbys( a, x, b, y ) bli_taxpbys( z,z,z,z,d, a, x, b, y )

// -- Notes --------------------------------------------------------------------

#endif

