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

#ifndef BLIS_TSCALCJS_H
#define BLIS_TSCALCJS_H

// -- Implementation macro -----------------------------------------------------

// (tr) := (ar) * (xr) - ( is_conj( conj ) ? -(ai) : (ai) ) * (xi);
// (ti) := ( is_conj( conj ) ? -(ai) : (ai) ) * (xr) + (ar) * (xi);
// (xr) := (tr);
// (xi) := (ti);

#define bli_tscalcjims( \
        \
          conj, \
          da, pa, ar, ai, \
          dx, px, xr, xi, \
          chc  \
        ) \
{ \
	PASTEMAC(c,declinits) \
	( \
	  chc, \
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
	        ( bli_is_conj( conj ) ? PASTEMAC(chc,neg)( \
	                                  PASTEMAC(pa,chc,tcast)(ai) \
	                                ) \
	                              :   PASTEMAC(pa,chc,tcast)(ai) \
	        ), \
	        PASTEMAC(px,chc,tcast)(xi) \
	      ) \
	    ) \
	  ),\
	  PASTEMAC(chc,add)( \
	    PASTEMAC(da,dx,termir)( \
	      chc, \
	      PASTEMAC(chc,mul)( \
	        ( bli_is_conj( conj ) ? PASTEMAC(chc,neg)( \
	                                  PASTEMAC(pa,chc,tcast)(ai) \
	                                ) \
	                              :   PASTEMAC(pa,chc,tcast)(ai) \
	        ), \
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
	  tr, \
	  ti \
	) \
	PASTEMAC(dx,assigns) \
	( \
	  PASTEMAC(chc,px,tcast)(tr), \
	  PASTEMAC(chc,px,tcast)(ti), \
	  xr, \
	  xi \
	); \
}

// -- API macros ---------------------------------------------------------------

// -- Consolidated --

// tscals
#define bli_tscalcjs( cha, chx, chc, conj, a, x ) \
        bli_tscalcjims \
        ( \
          conj, \
          PASTEMAC(cha,dom),  \
          PASTEMAC(cha,prec), \
          PASTEMAC(cha,real)(a), \
          PASTEMAC(cha,imag)(a), \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
          PASTEMAC(chx,real)(x), \
          PASTEMAC(chx,imag)(x), \
          PASTEMAC(chc,prec)  \
        )

#endif

