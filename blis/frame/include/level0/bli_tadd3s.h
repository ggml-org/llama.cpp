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

#ifndef BLIS_TADD3S_H
#define BLIS_TADD3S_H

// -- Implementation macro -----------------------------------------------------

// (zr) = (yr) + (xr);
// (zi) = (yi) + (xi);

#define bli_tadd3ims( \
          \
          dx, px, xr, xi, \
          dy, py, yr, yi, \
          dz, pz, zr, zi, \
          chc  \
        ) \
{ \
	PASTEMAC(dz,assigns) \
	( \
	  PASTEMAC(chc,pz,tcast)( \
	    PASTEMAC(chc,add)( \
	      PASTEMAC(py,chc,tcast)(yr), \
	      PASTEMAC(px,chc,tcast)(xr)  \
	    ) \
	  ),\
	  PASTEMAC(chc,pz,tcast)( \
	    PASTEMAC(chc,add)( \
	      PASTEMAC(py,chc,tcast)(yi), \
	      PASTEMAC(px,chc,tcast)(xi) \
	    ) \
	  ), \
	  zr, \
	  zi \
	); \
}

// -- API macros ---------------------------------------------------------------

// -- Consolidated --

// tadd3s
#define bli_tadd3s( chx, chy, chz, chc, x, y, z ) \
        bli_tadd3ims \
        ( \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
          PASTEMAC(chx,real)(x), \
          PASTEMAC(chx,imag)(x), \
          PASTEMAC(chy,dom),  \
          PASTEMAC(chy,prec), \
          PASTEMAC(chy,real)(y), \
          PASTEMAC(chy,imag)(y), \
          PASTEMAC(chz,dom),  \
          PASTEMAC(chz,prec), \
          PASTEMAC(chz,real)(z), \
          PASTEMAC(chz,imag)(z), \
          PASTEMAC(chc,prec)  \
        )

#undef GENTFUNC
#define GENTFUNC( ctypex, chx, ctypey, chy, ctypez, chz, ctypec, chc, opname ) \
UNIT_TEST(chx,chy,chz,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto z0 = convert<ctypez>( convert_prec<ctypec>( x ) + \
		                           convert_prec<ctypec>( y ) ); \
\
		INFO( "x: " << x ); \
		INFO( "y: " << y ); \
\
		ctypez z; \
		bli_tadds( chx,chy,chz,chc, x, y, z ); \
\
		INFO( "z (C++):  " << z0 ); \
		INFO( "z (BLIS): " << z ); \
\
		check<ctypec>( z, z0 ); \
	} \
)

// tadd3js
#define bli_tadd3js( chx, chy, chz, chc, x, y, z ) \
        bli_tadd3ims \
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
          PASTEMAC(chz,dom),  \
          PASTEMAC(chz,prec), \
          PASTEMAC(chz,real)(z), \
          PASTEMAC(chz,imag)(z), \
          PASTEMAC(chc,prec)  \
        )

// -- Exposed real/imaginary --

// tadd3ris
#define bli_tadd3ris( chx, chy, chz, chc, xr, xi, yr, yi, zr, zi ) \
        bli_tadd3ims \
        ( \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
                     xr, \
                     xi, \
          PASTEMAC(chy,dom),  \
          PASTEMAC(chy,prec), \
                     yr, \
                     yi, \
          PASTEMAC(chz,dom),  \
          PASTEMAC(chz,prec), \
                     zr, \
                     zi, \
          PASTEMAC(chc,prec)  \
        )

// tadd3jris
#define bli_tadd3jris( chx, chy, chz, chc, xr, xi, yr, yi, zr, zi ) \
        bli_tadd3ims \
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
          PASTEMAC(chz,dom),  \
          PASTEMAC(chz,prec), \
                     zr, \
                     zi, \
          PASTEMAC(chc,prec)  \
        )

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

