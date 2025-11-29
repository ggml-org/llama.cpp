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

#ifndef BLIS_TAXPYS_H
#define BLIS_TAXPYS_H

// -- Implementation macro -----------------------------------------------------

// (tr) += (ar) * (xr) - (ai) * (xi);
// (ti) += (ai) * (xr) + (ar) * (xi);
// (yr) += (tr);
// (yi) += (ti);

#define bli_taxpyims( \
          \
          da, pa, ar, ai, \
          dx, px, xr, xi, \
          dy, py, yr, yi, \
          chc  \
        ) \
{ \
	PASTEMAC(c,declinits) \
	( \
	  py, \
	  PASTEMAC(chc,py,tcast)( \
	    PASTEMAC(chc,add)( \
	      PASTEMAC(py,chc,tcast)(yr), \
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
	          ) \
	        ) \
	      ) \
	    ) \
	  ),\
	  PASTEMAC(chc,py,tcast)( \
	    PASTEMAC(chc,add)( \
	      PASTEMAC(py,chc,tcast)(yi), \
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

// taxpys
#define bli_taxpys( cha, chx, chy, chc, a, x, y ) \
        bli_taxpyims \
        ( \
          PASTEMAC(cha,dom),  \
          PASTEMAC(cha,prec), \
          PASTEMAC(cha,real)(a), \
          PASTEMAC(cha,imag)(a), \
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

// taxpyjs
#define bli_taxpyjs( cha, chx, chy, chc, a, x, y ) \
        bli_taxpyims \
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
          PASTEMAC(chy,dom),  \
          PASTEMAC(chy,prec), \
          PASTEMAC(chy,real)(y), \
          PASTEMAC(chy,imag)(y), \
          PASTEMAC(chc,prec)  \
        )

// -- Exposed real/imaginary --

// taxpyris
#define bli_taxpyris( cha, chx, chy, chc, ar, ai, xr, xi, yr, yi ) \
        bli_taxpyims \
        ( \
          PASTEMAC(cha,dom),  \
          PASTEMAC(cha,prec), \
                     ar, \
                     ai, \
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

// taxpyjris
#define bli_taxpyjris( cha, chx, chy, chc, ar, ai, xr, xi, yr, yi ) \
        bli_taxpyims \
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
          PASTEMAC(chy,dom),  \
          PASTEMAC(chy,prec), \
                     yr, \
                     yi, \
          PASTEMAC(chc,prec)  \
        )

// -- Higher-level static functions --------------------------------------------

// -- Legacy macros ------------------------------------------------------------

#define bli_saxpys( a, x, y ) bli_taxpys( s,s,s,s, a, x, y )
#define bli_daxpys( a, x, y ) bli_taxpys( d,d,d,d, a, x, y )
#define bli_caxpys( a, x, y ) bli_taxpys( c,c,c,s, a, x, y )
#define bli_zaxpys( a, x, y ) bli_taxpys( z,z,z,d, a, x, y )

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

