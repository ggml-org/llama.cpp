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

#ifndef BLIS_TSCALS_H
#define BLIS_TSCALS_H

// -- Implementation macro -----------------------------------------------------

// (tr) := (ar) * (xr) - (ai) * (xi);
// (ti) := (ai) * (xr) + (ar) * (xi);
// (xr) := (tr);
// (xi) := (ti);

#define bli_tscalims( \
        \
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
	        PASTEMAC(pa,chc,tcast)(ai), \
	        PASTEMAC(px,chc,tcast)(xi) \
	      ) \
	    ) \
	  ),\
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
#define bli_tscals( cha, chx, chc, a, x ) \
        bli_tscalims \
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

// tscaljs
#define bli_tscaljs( cha, chx, chc, a, x ) \
        bli_tscalims \
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

// tscalris
#define bli_tscalris( cha, chx, chc, ar, ai, xr, xi ) \
        bli_tscalims \
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

// tscaljris
#define bli_tscaljris( cha, chx, chc, ar, ai, xr, xi ) \
        bli_tscalims \
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

// -- 1e / 1r --

// scal1es
#define bli_tscal1es( cha, chx, chc, a, xri, xir ) \
        bli_tscalims \
        ( \
          PASTEMAC(cha,dom),  \
          PASTEMAC(cha,prec), \
          PASTEMAC(cha,real)(a), \
          PASTEMAC(cha,imag)(a), \
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

// scal1rs
#define bli_tscal1rs( cha, chx, chc, a, xr, xi ) \
        bli_tscalims \
        ( \
          PASTEMAC(cha,dom),  \
          PASTEMAC(cha,prec), \
          PASTEMAC(cha,real)(a), \
          PASTEMAC(cha,imag)(a), \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
          xr, \
          xi, \
          PASTEMAC(chc,prec)  \
        )

// -- Higher-level static functions --------------------------------------------

// -- mxn_uplo --

#define bli_tscalris_mxn_uplo( cha, chx, chc, uplo, diagoff, m, n, ar, ai, xr, xi, rs_x, cs_x ) \
{ \
	if ( bli_is_upper( uplo ) ) \
	{ \
		for ( dim_t jj = 0; jj < n; ++jj ) \
		for ( dim_t ii = 0; ii < m; ++ii ) \
		{ \
			if ( (doff_t)jj - (doff_t)ii >= diagoff ) \
			{ \
				PASTEMAC(chx,ctyper)* restrict xij_r = (xr) + ii*(rs_x) + jj*(cs_x); \
				PASTEMAC(chx,ctyper)* restrict xij_i = (xi) + ii*(rs_x) + jj*(cs_x); \
				(void)xij_i; \
\
				bli_tscalris( cha,chx,chc, *(ar), *(ai), *xij_r, *xij_i ); \
			} \
		} \
	} \
	else \
	{ \
		for ( dim_t jj = 0; jj < n; ++jj ) \
		for ( dim_t ii = 0; ii < m; ++ii ) \
		{ \
			if ( (doff_t)jj - (doff_t)ii <= diagoff ) \
			{ \
				PASTEMAC(chx,ctyper)* restrict xij_r = (xr) + ii*(rs_x) + jj*(cs_x); \
				PASTEMAC(chx,ctyper)* restrict xij_i = (xi) + ii*(rs_x) + jj*(cs_x); \
				(void)xij_i; \
\
				bli_tscalris( cha,chx,chc, *(ar), *(ai), *xij_r, *xij_i ); \
			} \
		} \
	} \
}

// -- Legacy macros ------------------------------------------------------------

#define bli_sscals( a, x ) bli_tscals( s,s,s, a, x )
#define bli_dscals( a, x ) bli_tscals( d,d,d, a, x )
#define bli_cscals( a, x ) bli_tscals( c,c,s, a, x )
#define bli_zscals( a, x ) bli_tscals( z,z,d, a, x )

#define bli_ssscals( a, x ) bli_tscals( s,s,s, a, x )
#define bli_ddscals( a, x ) bli_tscals( d,d,d, a, x )
#define bli_ccscals( a, x ) bli_tscals( c,c,s, a, x )
#define bli_zzscals( a, x ) bli_tscals( z,z,d, a, x )

// -- Notes --------------------------------------------------------------------

// -- Domain cases --

//   r       r
// (xr) := (ar) * (xr) -   0  *   0 ;
// (xi) xx   0  * (xr) + (ar) *   0 ;

//   r       c
// (xr) := (ar) * (xr) - (ai) *   0 ;
// (xi) xx (ai) * (xr) + (ar) *   0 ;

//   c       r
// (xr) := (ar) * (xr) -   0  * (xi);
// (xi) :=   0  * (xr) + (ar) * (xi);

//   c       c
// (xr) := (ar) * (xr) - (ai) * (xi);
// (xi) := (ai) * (xr) + (ar) * (xi);

#endif

