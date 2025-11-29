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

#ifndef BLIS_TSCAL2S_H
#define BLIS_TSCAL2S_H

// -- Implementation macro -----------------------------------------------------

// (tr) := (ar) * (xr) - (ai) * (xi);
// (ti) := (ai) * (xr) + (ar) * (xi);
// (yr) := (tr);
// (yi) := (ti);

#define bli_tscal2ims( \
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
	  ), \
	  PASTEMAC(chc,py,tcast)( \
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

// tscal2s
#define bli_tscal2s( cha, chx, chy, chc, a, x, y ) \
        bli_tscal2ims \
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

// tscal2js
#define bli_tscal2js( cha, chx, chy, chc, a, x, y ) \
        bli_tscal2ims \
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

// tscal2ris
#define bli_tscal2ris( cha, chx, chy, chc, ar, ai, xr, xi, yr, yi ) \
        bli_tscal2ims \
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

// tscal2jris
#define bli_tscal2jris( cha, chx, chy, chc, ar, ai, xr, xi, yr, yi ) \
        bli_tscal2ims \
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

// -- 1e / 1r --

// tscal21es
#define bli_tscal21es( cha, chx, chy, chc, a, x, yri, yir ) \
        bli_tscal2ims \
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
          PASTEMAC(chy,real)(yri), \
          PASTEMAC(chy,imag)(yri), \
          PASTEMAC(chc,prec)  \
        ); \
        bli_tcopyims \
        ( \
          PASTEMAC(chy,dom),  \
          PASTEMAC(chy,prec), \
          PASTEMAC(PASTEMAC(chy,prec),neg)( \
            PASTEMAC(chy,imag)(yri)  \
          ), \
          PASTEMAC(chy,real)(yri), \
          PASTEMAC(chy,dom),  \
          PASTEMAC(chy,prec), \
          PASTEMAC(chy,real)(yir), \
          PASTEMAC(chy,imag)(yir) \
        )

// tscal2j1es
#define bli_tscal2j1es( cha, chx, chy, chc, a, x, yri, yir ) \
        bli_tscal2ims \
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
          PASTEMAC(chy,real)(yri), \
          PASTEMAC(chy,imag)(yri), \
          PASTEMAC(chc,prec)  \
        ); \
        bli_tcopyims \
        ( \
          PASTEMAC(chy,dom),  \
          PASTEMAC(chy,prec), \
          PASTEMAC(PASTEMAC(chy,prec),neg)( \
            PASTEMAC(chy,imag)(yri)  \
          ), \
          PASTEMAC(chy,real)(yri), \
          PASTEMAC(chy,dom),  \
          PASTEMAC(chy,prec), \
          PASTEMAC(chy,real)(yir), \
          PASTEMAC(chy,imag)(yir) \
        )

// tscal21rs
#define bli_tscal21rs( cha, chx, chy, chc, a, x, yr, yi ) \
        bli_tscal2ims \
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
          yr, \
          yi, \
          PASTEMAC(chc,prec)  \
        )

// tscal2j1rs
#define bli_tscal2j1rs( cha, chx, chy, chc, a, x, yr, yi ) \
        bli_tscal2ims \
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
          yr, \
          yi, \
          PASTEMAC(chc,prec)  \
        )

// -- Higher-level static functions --------------------------------------------

// -- mxn --

// scal2bbs_mxn

#define bli_tscal2bbs_mxn_r( \
                             cha,chx,chy,chc, \
                             ctypea, ctypea_r, \
                             ctypex, ctypex_r, \
                             ctypey, ctypey_r, \
                             conjx, \
                             m, \
                             n, \
                             alpha, \
                             x, incx, ldx, \
                             y, incy, ldy  \
                           ) \
{ \
	/* Assume that the duplication factor is the row stride of y. */ \
	const dim_t d    = incy; \
	const dim_t ds_y = 1; \
\
	if ( bli_is_conj( conjx ) ) \
	{ \
		for ( dim_t j = 0; j < (n); ++j ) \
		{ \
			ctypex* restrict xj = (ctypex*)(x) + j*(ldx); \
			ctypey* restrict yj = (ctypey*)(y) + j*(ldy); \
	\
			for ( dim_t i = 0; i < (m); ++i ) \
			{ \
				ctypex* restrict xij = xj + i*(incx); \
				ctypey* restrict yij = yj + i*(incy); \
	\
				bli_tscal2js( cha,chx,chy,chc, *(const ctypea* restrict)(alpha), *xij, *yij ); \
	\
				for ( dim_t p = 1; p < d; ++p ) \
				{ \
					ctypey* restrict yijd = yij + p*ds_y; \
	\
					bli_tcopys( chy,chy, *yij, *yijd ); \
				} \
			} \
		} \
	} \
	else \
	{ \
		for ( dim_t j = 0; j < (n); ++j ) \
		{ \
			ctypex* restrict xj = (ctypex*)(x) + j*(ldx); \
			ctypey* restrict yj = (ctypey*)(y) + j*(ldy); \
	\
			for ( dim_t i = 0; i < (m); ++i ) \
			{ \
				ctypex* restrict xij = xj + i*(incx); \
				ctypey* restrict yij = yj + i*(incy); \
	\
				bli_tscal2s( cha,chx,chy,chc, *(const ctypea* restrict)(alpha), *xij, *yij ); \
	\
				for ( dim_t p = 1; p < d; ++p ) \
				{ \
					ctypey* restrict yijd = yij + p*ds_y; \
	\
					bli_tcopys( chy,chy, *yij, *yijd ); \
				} \
			} \
		} \
	} \
}

#define bli_tscal2bbs_mxn_c( \
                             cha,chx,chy,chc, \
                             ctypea, ctypea_r, \
                             ctypex, ctypex_r, \
                             ctypey, ctypey_r, \
                             conjx, \
                             m, \
                             n, \
                             alpha, \
                             x, incx, ldx, \
                             y, incy, ldy  \
                           ) \
{ \
	/* Assume that the duplication factor is the row stride of y. */ \
	const dim_t       d          = incy; \
	const dim_t       ds_y       = 1; \
\
	const inc_t       incx2      = sizeof(ctypex) / sizeof(ctypex_r) * (incx); \
	const inc_t       ldx2       = sizeof(ctypex) / sizeof(ctypex_r) * (ldx); \
\
	const inc_t       incy2      = sizeof(ctypey) / sizeof(ctypey_r) * (incy); \
	const inc_t       ldy2       = sizeof(ctypey) / sizeof(ctypey_r) * (ldy); \
\
	ctypea_r* restrict alpha_r    = ( ctypea_r* )(alpha); \
	ctypea_r* restrict alpha_i    = ( ctypea_r* )(alpha) + 1; (void)alpha_i; \
	ctypex_r* restrict chi_r      = ( ctypex_r* )(x); \
	ctypex_r* restrict chi_i      = ( ctypex_r* )(x) + 1; (void)chi_i; \
	ctypey_r* restrict psi_r      = ( ctypey_r* )(y); \
	ctypey_r* restrict psi_i      = ( ctypey_r* )(y) + 1*d; (void)psi_i; \
\
	if ( bli_is_conj( conjx ) ) \
	{ \
		for ( dim_t j = 0; j < (n); ++j ) \
		{ \
			ctypex_r* restrict chij_r = chi_r + j*ldx2; \
			ctypex_r* restrict chij_i = chi_i + j*ldx2; \
			ctypey_r* restrict psij_r = psi_r + j*ldy2; \
			ctypey_r* restrict psij_i = psi_i + j*ldy2; \
\
			for ( dim_t i = 0; i < (m); ++i ) \
			{ \
				ctypex_r* restrict chiij_r = chij_r + i*incx2; \
				ctypex_r* restrict chiij_i = chij_i + i*incx2; (void)chiij_i; \
				ctypey_r* restrict psiij_r = psij_r + i*incy2; \
				ctypey_r* restrict psiij_i = psij_i + i*incy2; (void)psiij_i; \
\
				bli_tscal2jris( cha,chx,chy,chc, \
				                *alpha_r, *alpha_i, \
				                *chiij_r, *chiij_i, \
				                *psiij_r, *psiij_i ); \
\
				for ( dim_t p = 1; p < d; ++p ) \
				{ \
					ctypey_r* restrict psiijd_r = psiij_r + p*ds_y; \
					ctypey_r* restrict psiijd_i = psiij_i + p*ds_y; (void)psiijd_i; \
\
					bli_tcopyris( chy,chy, *psiij_r,  *psiij_i, \
					                       *psiijd_r, *psiijd_i ); \
				} \
			} \
		} \
	} \
	else /* if ( bli_is_noconj( conjx ) ) */ \
	{ \
		for ( dim_t j = 0; j < (n); ++j ) \
		{ \
			ctypex_r* restrict chij_r = chi_r + j*ldx2; \
			ctypex_r* restrict chij_i = chi_i + j*ldx2; \
			ctypey_r* restrict psij_r = psi_r + j*ldy2; \
			ctypey_r* restrict psij_i = psi_i + j*ldy2; \
\
			for ( dim_t i = 0; i < (m); ++i ) \
			{ \
				ctypex_r* restrict chiij_r = chij_r + i*incx2; \
				ctypex_r* restrict chiij_i = chij_i + i*incx2; (void)chiij_i; \
				ctypey_r* restrict psiij_r = psij_r + i*incy2; \
				ctypey_r* restrict psiij_i = psij_i + i*incy2; (void)psiij_i; \
\
				bli_tscal2ris( cha,chx,chy,chc, \
				               *alpha_r, *alpha_i, \
				               *chiij_r, *chiij_i, \
				               *psiij_r, *psiij_i ); \
\
				for ( dim_t p = 1; p < d; ++p ) \
				{ \
					ctypey_r* restrict psiijd_r = psiij_r + p*ds_y; \
					ctypey_r* restrict psiijd_i = psiij_i + p*ds_y; (void)psiijd_i; \
\
					bli_tcopyris( chy,chy, *psiij_r,  *psiij_i, \
					                       *psiijd_r, *psiijd_i ); \
				} \
			} \
		} \
	} \
}

#define bli_tscal2bbs_mxn( \
                           cha,chx,chy,chc, \
                           conjx, \
                           m, \
                           n, \
                           alpha, \
                           x, incx, ldx, \
                           y, incy, ldy  \
                         ) \
PASTECH(bli_tscal2bbs_mxn_,PASTEMAC(chy,dom)) \
( \
  cha,chx,chy,chc, \
  PASTEMAC(cha,ctype),PASTEMAC(cha,ctyper), \
  PASTEMAC(chx,ctype),PASTEMAC(chx,ctyper), \
  PASTEMAC(chy,ctype),PASTEMAC(chy,ctyper), \
  conjx, \
  m, \
  n, \
  alpha, \
  x, incx, ldx, \
  y, incy, ldy \
)

#define bli_tscal2s_mxn( cha, chx, chy, chc, conjx, m, n, alpha, x, rs_x, cs_x, y, rs_y, cs_y ) \
{ \
	if ( bli_is_conj( conjx ) ) \
	{ \
		for ( dim_t jj = 0; jj < (n); ++jj ) \
		for ( dim_t ii = 0; ii < (m); ++ii ) \
		{ \
			PASTEMAC(chx,ctype)* restrict xij = ( PASTEMAC(chx,ctype)* )(x) + ii*(rs_x) + jj*(cs_x); \
			PASTEMAC(chy,ctype)* restrict yij = ( PASTEMAC(chy,ctype)* )(y) + ii*(rs_y) + jj*(cs_y); \
\
			bli_tscal2js( cha,chx,chy,chc, *(alpha), *xij, *yij ); \
		} \
	} \
	else \
	{ \
		for ( dim_t jj = 0; jj < (n); ++jj ) \
		for ( dim_t ii = 0; ii < (m); ++ii ) \
		{ \
			PASTEMAC(chx,ctype)* restrict xij = ( PASTEMAC(chx,ctype)* )(x) + ii*(rs_x) + jj*(cs_x); \
			PASTEMAC(chy,ctype)* restrict yij = ( PASTEMAC(chy,ctype)* )(y) + ii*(rs_y) + jj*(cs_y); \
\
			bli_tscal2s( cha,chx,chy,chc, *(alpha), *xij, *yij ); \
		} \
	} \
}

#define bli_tscal2ris_mxn( cha, chx, chy, chc, conjx, m, n, alpha, x, rs_x, cs_x, y, rs_y, cs_y, is_y ) \
{ \
	PASTEMAC(cha,ctyper)* restrict alpha_r = ( PASTEMAC(cha,ctyper)* )(alpha);     (void)alpha_r; \
	PASTEMAC(cha,ctyper)* restrict alpha_i = ( PASTEMAC(cha,ctyper)* )(alpha) + 1; (void)alpha_i; \
	PASTEMAC(chx,ctyper)* restrict x_r     = ( PASTEMAC(chx,ctyper)* )(x); \
	PASTEMAC(chx,ctyper)* restrict x_i     = ( PASTEMAC(chx,ctyper)* )(x) + 1; \
	PASTEMAC(chy,ctyper)* restrict y_r     = ( PASTEMAC(chy,ctyper)* )(y); \
	PASTEMAC(chy,ctyper)* restrict y_i     = ( PASTEMAC(chy,ctyper)* )(y) + (is_y); \
	const dim_t incx2                      = 2*(rs_x); \
	const dim_t ldx2                       = 2*(cs_x); \
\
	if ( bli_is_conj( conjx ) ) \
	{ \
		for ( dim_t jj = 0; jj < (n); ++jj ) \
		for ( dim_t ii = 0; ii < (m); ++ii ) \
		{ \
			PASTEMAC(chx,ctyper)* restrict chi11_r = x_r + ii*incx2  + jj*ldx2;   (void)chi11_r; \
			PASTEMAC(chx,ctyper)* restrict chi11_i = x_i + ii*incx2  + jj*ldx2;   (void)chi11_i; \
			PASTEMAC(chy,ctyper)* restrict psi11_r = y_r + ii*(rs_y) + jj*(cs_y); (void)psi11_r; \
			PASTEMAC(chy,ctyper)* restrict psi11_i = y_i + ii*(rs_y) + jj*(cs_y); (void)psi11_i; \
\
			bli_tscal2jris \
			( \
			  cha,chx,chy,chc, \
			  *alpha_r, *alpha_i, \
			  *chi11_r, *chi11_i, \
			  *psi11_r, *psi11_i  \
			); \
		} \
	} \
	else \
	{ \
		for ( dim_t jj = 0; jj < (n); ++jj ) \
		for ( dim_t ii = 0; ii < (m); ++ii ) \
		{ \
			PASTEMAC(chx,ctyper)* restrict chi11_r = x_r + ii*incx2  + jj*ldx2;   (void)chi11_r; \
			PASTEMAC(chx,ctyper)* restrict chi11_i = x_i + ii*incx2  + jj*ldx2;   (void)chi11_i; \
			PASTEMAC(chy,ctyper)* restrict psi11_r = y_r + ii*(rs_y) + jj*(cs_y); (void)psi11_r; \
			PASTEMAC(chy,ctyper)* restrict psi11_i = y_i + ii*(rs_y) + jj*(cs_y); (void)psi11_i; \
\
			bli_tscal2ris \
			( \
			  cha,chx,chy,chc, \
			  *alpha_r, *alpha_i, \
			  *chi11_r, *chi11_i, \
			  *psi11_r, *psi11_i  \
			); \
		} \
	} \
}

// -- Legacy macros ------------------------------------------------------------

#define bli_sscal2bbs_mxn( conjx, m, n, alpha, x, incx, ldx, y, incy, ldy ) \
  bli_tscal2bbs_mxn( \
                     s,s,s,s, \
                     conjx, \
                     m, \
                     n, \
                     alpha, \
                     x, incx, ldx, \
                     y, incy, ldy  \
                   ) \

#define bli_dscal2bbs_mxn( conjx, m, n, alpha, x, incx, ldx, y, incy, ldy ) \
  bli_tscal2bbs_mxn( \
                     d,d,d,d, \
                     conjx, \
                     m, \
                     n, \
                     alpha, \
                     x, incx, ldx, \
                     y, incy, ldy  \
                   ) \

#define bli_cscal2bbs_mxn( conjx, m, n, alpha, x, incx, ldx, y, incy, ldy ) \
  bli_tscal2bbs_mxn( \
                     c,c,c,c, \
                     conjx, \
                     m, \
                     n, \
                     alpha, \
                     x, incx, ldx, \
                     y, incy, ldy  \
                   ) \

#define bli_zscal2bbs_mxn( conjx, m, n, alpha, x, incx, ldx, y, incy, ldy ) \
  bli_tscal2bbs_mxn( \
                     z,z,z,z, \
                     conjx, \
                     m, \
                     n, \
                     alpha, \
                     x, incx, ldx, \
                     y, incy, ldy  \
                   ) \

// -- Legacy macros ------------------------------------------------------------

#define bli_sscal2s( a, x, y ) bli_tscal2s( s,s,s,s, a, x, y )
#define bli_dscal2s( a, x, y ) bli_tscal2s( d,d,d,d, a, x, y )
#define bli_cscal2s( a, x, y ) bli_tscal2s( c,c,c,s, a, x, y )
#define bli_zscal2s( a, x, y ) bli_tscal2s( z,z,z,d, a, x, y )

// -- Notes --------------------------------------------------------------------

// -- Domain cases --

//   r       r      r
// (yr) := (ar) * (xr) -   0  *   0 ;
// (yi) xx   0  * (xr) + (ar) *   0 ;

//   r       r      c
// (yr) := (ar) * (xr) -   0  * (xi);
// (yi) xx   0  * (xr) + (ar) * (xi);

//   r       c      r
// (yr) := (ar) * (xr) - (ai) *   0 ;
// (yi) xx (ai) * (xr) + (ar) *   0 ;

//   r       c      c
// (yr) := (ar) * (xr) - (ai) * (xi);
// (yi) xx (ai) * (xr) + (ar) * (xi);

//   c       r      r
// (yr) := (ar) * (xr) -   0  *   0 ;
// (yi) :=   0  * (xr) + (ar) *   0 ;

//   c       r      c
// (yr) := (ar) * (xr) -   0  * (xi);
// (yi) :=   0  * (xr) + (ar) * (xi);

//   c       c      r
// (yr) := (ar) * (xr) - (ai) *   0 ;
// (yi) := (ai) * (xr) + (ar) *   0 ;

//   c       c      c
// (yr) := (ar) * (xr) - (ai) * (xi);
// (yi) := (ai) * (xr) + (ar) * (xi);

#endif

