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

#ifndef BLIS_TINVSCALS_H
#define BLIS_TINVSCALS_H

// -- Implementation macro -----------------------------------------------------

#define bli_tinvscalims( \
          \
          da, pa, ar, ai, \
          dx, px, xr, xi, \
          chc  \
        ) \
{ \
	PASTEMAC(da,dx,invscalims) \
	( \
	  da, pa, ar, ai, \
	  dx, px, xr, xi, \
	  chc  \
	); \
}

// -- real-real domain implementation --
// -- real-complex domain implementation --

// (xr) = (xr) / (ar);
// (xi) = (xi) / (ar);

#define bli_rrinvscalims bli_rcinvscalims

#define bli_rcinvscalims( \
          \
          da, pa, ar, ai, \
          dx, px, xr, xi, \
          chc  \
        ) \
{ \
	PASTEMAC(dx,assigns) \
	( \
	  PASTEMAC(chc,px,tcast)( \
	    PASTEMAC(chc,div)( \
	      PASTEMAC(px,chc,tcast)(xr),  \
	      PASTEMAC(pa,chc,tcast)(ar)  \
	    ) \
	  ),\
	  PASTEMAC(chc,px,tcast)( \
	    PASTEMAC(chc,div)( \
	      PASTEMAC(px,chc,tcast)(xi),  \
	      PASTEMAC(pa,chc,tcast)(ar)  \
	    ) \
	  ),\
	  xr, \
	  xi \
	); \
}

// -- complex-real domain implementation --
// -- complex-complex domain implementation --

// sr    = maxabs( ar, ai );
// asr   = ar / sr;
// asi   = ai / sr;
// xrt   = xr;
// tempr = ar * asr + ai * asi
// xr    = ( asr * xrt + asi * xi  ) / tempr;
// xi    = ( asr * xi  - asi * xrt ) / tempr;

#define bli_crinvscalims bli_ccinvscalims

#define bli_ccinvscalims( \
          \
          da, pa, ar, ai, \
          dx, px, xr, xi, \
          chc  \
        ) \
{ \
	PASTEMAC(ro,declinits) \
	( \
	  chc, \
	  PASTEMAC(chc,maxabs)( \
	    PASTEMAC(pa,chc,tcast)(ar), \
	    PASTEMAC(pa,chc,tcast)(ai)  \
	  ), \
	  sr  \
	) \
	PASTEMAC(c,declinits) \
	( \
	  chc, \
	  PASTEMAC(chc,div)( \
	    PASTEMAC(pa,chc,tcast)(ar), \
	    sr  \
	  ), \
	  PASTEMAC(chc,div)( \
	    PASTEMAC(pa,chc,tcast)(ai), \
	    sr  \
	  ), \
	  asr, \
	  asi \
	) \
	PASTEMAC(ro,declinits) \
	( \
	  chc, \
	  PASTEMAC(px,chc,tcast)(xr), \
	  xrt  \
	) \
	PASTEMAC(ro,declinits) \
	( \
	  chc, \
	  PASTEMAC(chc,add)( \
	    PASTEMAC(chc,mul)( \
	      PASTEMAC(pa,chc,tcast)(ar), \
	      asr  \
	    ), \
	    PASTEMAC(chc,mul)( \
	      PASTEMAC(pa,chc,tcast)(ai), \
	      asi  \
	    ) \
	  ), \
	  tempr  \
	) \
	PASTEMAC(dx,assigns) \
	( \
	  PASTEMAC(chc,px,tcast)( \
	    PASTEMAC(chc,div)( \
	      PASTEMAC(chc,add)( \
	        PASTEMAC(chc,mul)( \
	          asr, \
              xrt  \
            ), \
	        PASTEMAC(chc,mul)( \
	          asi, \
	          PASTEMAC(px,chc,tcast)(xi)  \
            )  \
          ), \
	      tempr  \
	    ) \
	  ),\
	  PASTEMAC(chc,px,tcast)( \
	    PASTEMAC(chc,div)( \
	      PASTEMAC(chc,sub)( \
	        PASTEMAC(chc,mul)( \
	          asr, \
	          PASTEMAC(px,chc,tcast)(xi)  \
            ), \
	        PASTEMAC(chc,mul)( \
	          asi, \
              xrt  \
            )  \
          ), \
	      tempr  \
	    ) \
	  ),\
	  xr, \
	  xi \
	); \
}

// -- API macros ---------------------------------------------------------------

// -- Consolidated --

// tinvscals
#define bli_tinvscals( cha, chx, chc, a, x ) \
        bli_tinvscalims \
        ( \
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

// tinvscaljs
#define bli_tinvscaljs( cha, chx, chc, a, x ) \
        bli_tinvscalims \
        ( \
          PASTEMAC(cha,dom),  \
          PASTEMAC(cha,prec), \
          PASTEMAC(cha,real)(a), \
          PASTEMAC(PASTEMAC(cha,prec),neg)( \
            PASTEMAC(cha,imag)(a)  \
          ), \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
          PASTEMAC(chx,real)(x), \
          PASTEMAC(chx,imag)(x), \
          PASTEMAC(chc,prec)  \
        )

// -- Exposed real/imaginary --

// tinvscalris
#define bli_tinvscalris( cha, chx, chc, ar, ai, xr, xi ) \
        bli_tinvscalims \
        ( \
          PASTEMAC(cha,dom),  \
          PASTEMAC(cha,prec), \
                     ar, \
                     ai, \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
                     xr, \
                     xi, \
          PASTEMAC(chc,prec)  \
        )

// tinvscaljris
#define bli_tinvscaljris( cha, chx, chc, ar, ai, xr, xi ) \
        bli_tinvscalims \
        ( \
          PASTEMAC(cha,dom),  \
          PASTEMAC(cha,prec), \
                     ar, \
          PASTEMAC(PASTEMAC(cha,prec),neg)( \
                     ai ), \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
                     xr, \
                     xi, \
          PASTEMAC(chc,prec)  \
        )

// -- Higher-level static functions --------------------------------------------

// -- Notes --------------------------------------------------------------------

#endif

