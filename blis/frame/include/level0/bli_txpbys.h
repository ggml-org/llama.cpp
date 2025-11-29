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

#ifndef BLIS_TXPBYS_H
#define BLIS_TXPBYS_H

// -- Implementation macro -----------------------------------------------------

// (yr) := (xr) + (br) * (yr) - (bi) * (yi);
// (yi) := (xi) + (bi) * (yr) + (br) * (yi);

#define bli_txpbyims( \
	      \
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
	      PASTEMAC(px,chc,tcast)(xr), \
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
	      PASTEMAC(px,chc,tcast)(xi), \
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

// txpbys
#define bli_txpbys( chx, chb, chy, chc, x, b, y ) \
	    bli_txpbyims \
	    ( \
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

// txpbyjs
#define bli_txpbyjs( chx, chb, chy, chc, x, b, y ) \
	    bli_txpbyims \
	    ( \
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

// txpbyris
#define bli_txpbyris( chx, chb, chy, chc, xr, xi, br, bi, yr, yi ) \
	    bli_txpbyims \
	    ( \
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

// txpbyjris
#define bli_txpbyjris( chx, chb, chy, chc, xr, xi, br, bi, yr, yi ) \
	    bli_txpbyims \
	    ( \
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

// xpbys_mxn
#define bli_txpbys_mxn( chx, chb, chy, chc, m, n, x, rs_x, cs_x, beta, y, rs_y, cs_y ) \
{ \
\
	/* If beta is zero, overwrite y with x (in case y has infs or NaNs). */ \
	if ( bli_teq0s( chb, *(beta) ) ) \
	{ \
		bli_tcopys_mxn( chx, chy, m, n, x, rs_x, cs_x, y, rs_y, cs_y ); \
	} \
	else \
	{ \
		for ( dim_t jj = 0; jj < n; ++jj ) \
		for ( dim_t ii = 0; ii < m; ++ii ) \
		{ \
			PASTEMAC(chx,ctype)* restrict xij = ( PASTEMAC(chx,ctype)* )(x) + ii*(rs_x) + jj*(cs_x); \
			PASTEMAC(chy,ctype)* restrict yij = ( PASTEMAC(chy,ctype)* )(y) + ii*(rs_y) + jj*(cs_y); \
\
			bli_txpbys( chx,chb,chy,chc, *xij, *(beta), *yij ); \
		} \
	} \
}

// xpbys_mxn_uplo
#define bli_txpbys_mxn_uplo( chx, chb, chy, chc, diagoff, uplo, m, n, x, rs_x, cs_x, beta, y, rs_y, cs_y ) \
{ \
	if ( bli_is_upper( uplo ) ) \
	{ \
		/* If beta is zero, overwrite y with x (in case y has infs or NaNs). */ \
		if ( bli_teq0s( chb, *(beta) ) ) \
		{ \
			for ( dim_t jj = 0; jj < n; ++jj ) \
			for ( dim_t ii = 0; ii < m; ++ii ) \
			{ \
				if ( (doff_t)jj - (doff_t)ii >= (diagoff) ) \
				{ \
					PASTEMAC(chx,ctype)* restrict xij = ( PASTEMAC(chx,ctype)* )(x) + ii*(rs_x) + jj*(cs_x); \
					PASTEMAC(chy,ctype)* restrict yij = ( PASTEMAC(chy,ctype)* )(y) + ii*(rs_y) + jj*(cs_y); \
\
					bli_tcopys( chx,chy, *xij, *yij ); \
				} \
			} \
		} \
		else \
		{ \
			for ( dim_t jj = 0; jj < n; ++jj ) \
			for ( dim_t ii = 0; ii < m; ++ii ) \
			{ \
				if ( (doff_t)jj - (doff_t)ii >= (diagoff) ) \
				{ \
					PASTEMAC(chx,ctype)* restrict xij = ( PASTEMAC(chx,ctype)* )(x) + ii*(rs_x) + jj*(cs_x); \
					PASTEMAC(chy,ctype)* restrict yij = ( PASTEMAC(chy,ctype)* )(y) + ii*(rs_y) + jj*(cs_y); \
\
					bli_txpbys( chx,chb,chy,chc, *xij, *(beta), *yij ); \
				} \
			} \
		} \
	} \
	else /* if ( bli_is_lower( uplo ) ) */ \
	{ \
		/* If beta is zero, overwrite y with x (in case y has infs or NaNs). */ \
		if ( bli_teq0s( chb, *(beta) ) ) \
		{ \
			for ( dim_t jj = 0; jj < n; ++jj ) \
			for ( dim_t ii = 0; ii < m; ++ii ) \
			{ \
				if ( (doff_t)jj - (doff_t)ii <= (diagoff) ) \
				{ \
					PASTEMAC(chx,ctype)* restrict xij = ( PASTEMAC(chx,ctype)* )(x) + ii*(rs_x) + jj*(cs_x); \
					PASTEMAC(chy,ctype)* restrict yij = ( PASTEMAC(chy,ctype)* )(y) + ii*(rs_y) + jj*(cs_y); \
\
					bli_tcopys( chx,chy, *xij, *yij ); \
				} \
			} \
		} \
		else \
		{ \
			for ( dim_t jj = 0; jj < n; ++jj ) \
			for ( dim_t ii = 0; ii < m; ++ii ) \
			{ \
				if ( (doff_t)jj - (doff_t)ii <= (diagoff) ) \
				{ \
					PASTEMAC(chx,ctype)* restrict xij = ( PASTEMAC(chx,ctype)* )(x) + ii*(rs_x) + jj*(cs_x); \
					PASTEMAC(chy,ctype)* restrict yij = ( PASTEMAC(chy,ctype)* )(y) + ii*(rs_y) + jj*(cs_y); \
\
					bli_txpbys( chx,chb,chy,chc, *xij, *(beta), *yij ); \
				} \
			} \
		} \
	} \
}

// -- Notes --------------------------------------------------------------------

// -- Domain cases --

//   r       r      r
// (yr) := (xr) + (br) * (yr) -   0  *   0 ;
// (yi) xx   0  +   0  * (yr) + (br) *   0 ;

//   r       r      c
// (yr) := (xr) + (br) * (yr) - (bi) *   0 ;
// (yi) xx   0  + (bi) * (yr) + (br) *   0 ;

//   r       c      r
// (yr) := (xr) + (br) * (yr) -   0  *   0 ;
// (yi) xx (xi) +   0  * (yr) + (br) *   0 ;

//   r       c      c
// (yr) := (xr) + (br) * (yr) - (bi) *   0 ;
// (yi) xx (xi) + (bi) * (yr) + (br) *   0 ;

//   c       r      r
// (yr) := (xr) + (br) * (yr) -   0  * (yi);
// (yi) :=   0  +   0  * (yr) + (br) * (yi);

//   c       r      c
// (yr) := (xr) + (br) * (yr) - (bi) * (yi);
// (yi) :=   0  + (bi) * (yr) + (br) * (yi);

//   c       c      r
// (yr) := (xr) + (br) * (yr) -   0  * (yi);
// (yi) := (xi) +   0  * (yr) + (br) * (yi);

//   c       c      c
// (yr) := (xr) + (br) * (yr) - (bi) * (yi);
// (yi) := (xi) + (bi) * (yr) + (br) * (yi);

#endif

