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

#ifndef BLIS_TADDS_H
#define BLIS_TADDS_H

// -- Implementation macro -----------------------------------------------------

// (yr) = (yr) + (xr);
// (yi) = (yi) + (xi);

#define bli_taddims( \
          \
          dx, px, xr, xi, \
          dy, py, yr, yi, \
          chc  \
        ) \
{ \
	PASTEMAC(dy,assigns) \
	( \
	  PASTEMAC(chc,py,tcast)( \
	    PASTEMAC(chc,add)( \
	      PASTEMAC(py,chc,tcast)(yr), \
	      PASTEMAC(px,chc,tcast)(xr)  \
	    ) \
	  ),\
	  PASTEMAC(chc,py,tcast)( \
	    PASTEMAC(chc,add)( \
	      PASTEMAC(py,chc,tcast)(yi), \
	      PASTEMAC(px,chc,tcast)(xi) \
	    ) \
	  ), \
	  yr, \
	  yi \
	); \
}

// -- API macros ---------------------------------------------------------------

// -- Consolidated --

// tadds
#define bli_tadds( chx, chy, chc, x, y ) \
        bli_taddims \
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

// taddjs
#define bli_taddjs( chx, chy, chc, x, y ) \
        bli_taddims \
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
          PASTEMAC(chy,imag)(y), \
          PASTEMAC(chc,prec)  \
        )

// -- Exposed real/imaginary --

// taddris
#define bli_taddris( chx, chy, chc, xr, xi, yr, yi ) \
        bli_taddims \
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

// taddjris
#define bli_taddjris( chx, chy, chc, xr, xi, yr, yi ) \
        bli_taddims \
        ( \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
                     xr, \
          PASTEMAC(PASTEMAC(chx,prec),neg)( \
                     xi ), \
          PASTEMAC(chy,dom),  \
          PASTEMAC(chy,prec), \
                     yr, \
                     yi, \
          PASTEMAC(chc,prec)  \
        )

// -- Higher-level static functions --------------------------------------------

// -- mxn --

// tadds_mxn
#define bli_tadds_mxn( chx, chy, chc, m, n, x, rs_x, cs_x, y, rs_y, cs_y ) \
{ \
	for ( dim_t jj = 0; jj < (n); ++jj ) \
	for ( dim_t ii = 0; ii < (m); ++ii ) \
	{ \
		PASTEMAC(chx,ctype)* restrict xij = ( PASTEMAC(chx,ctype)* )(x) + ii*(rs_x) + jj*(cs_x); \
		PASTEMAC(chy,ctype)* restrict yij = ( PASTEMAC(chy,ctype)* )(y) + ii*(rs_y) + jj*(cs_y); \
\
		bli_tadds( chx,chy,chc, *xij, *yij ); \
	} \
}

// -- Notes --------------------------------------------------------------------

// -- Domain cases --

//   r       r
// (yr) += (xr);
// (yi) xx   0 ;

//   r       c
// (yr) += (xr);
// (yi) xx (xi);

//   c       r
// (yr) += (xr);
// (yi) +=   0 ;

//   c       c
// (yr) += (xr);
// (yi) += (xi);

#endif

