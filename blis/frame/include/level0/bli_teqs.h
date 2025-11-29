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

#ifndef BLIS_TEQS_H
#define BLIS_TEQS_H

// -- Implementation macro -----------------------------------------------------

// (xr) == (yr) && (xi) == (yi)

#define bli_teqims( \
          \
          dx, px, xr, xi, \
          dy, py, yr, yi, \
          chc  \
        ) \
    ( PASTEMAC(PASTEMAC(chc,prec),eq)( PASTEMAC(px,chc,tcast)(xr), \
                                       PASTEMAC(py,chc,tcast)(yr) ) && \
      PASTEMAC(PASTEMAC(chc,prec),eq)( PASTEMAC(px,chc,tcast)(xi), \
                                       PASTEMAC(py,chc,tcast)(yi) ) )

// -- API macros ---------------------------------------------------------------

// -- Consolidated --

// teqs
#define bli_teqs( chx, chy, chc, x, y ) \
        bli_teqims \
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

// -- Exposed real/imaginary --

// teqris
#define bli_teqris( chx, chy, chc, xr, xi, yr, yi ) \
        bli_teqims \
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

// -- Convenience macros -------------------------------------------------------

// -- Exposed real/imaginary --

#define bli_teq1ris( chx, xr, xi ) \
        bli_teqris \
        ( \
          chx, chx, chx, \
                     xr, \
                     xi, \
          PASTEMAC(PASTEMAC(chx,prec),one), \
          PASTEMAC(PASTEMAC(chx,prec),zero) \
        )

#define bli_teq0ris( chx, xr, xi ) \
        bli_teqris \
        ( \
          chx, chx, chx, \
                     xr, \
                     xi, \
          PASTEMAC(PASTEMAC(chx,prec),zero), \
          PASTEMAC(PASTEMAC(chx,prec),zero) \
        )

#define bli_teqm1ris( chx, xr, xi ) \
        bli_teqris \
        ( \
          chx, chx, chx, \
                     xr, \
                     xi, \
          PASTEMAC(PASTEMAC(chx,prec),mone), \
          PASTEMAC(PASTEMAC(chx,prec),zero) \
        )

// -- Consolidated --

#define bli_teq1s( chx, x ) \
        bli_teq1ris \
        ( \
          chx, \
          PASTEMAC(chx,real)(x), \
          PASTEMAC(chx,imag)(x) \
        )

#define bli_teq0s( chx, x ) \
        bli_teq0ris \
        ( \
          chx, \
          PASTEMAC(chx,real)(x), \
          PASTEMAC(chx,imag)(x) \
        )

#define bli_teqm1s( chx, x ) \
        bli_teqm1ris \
        ( \
          chx, \
          PASTEMAC(chx,real)(x), \
          PASTEMAC(chx,imag)(x)  \
        )

// -- Higher-level static functions --------------------------------------------

// -- Legacy macros ------------------------------------------------------------

#define bli_seqs( x, y ) bli_teqs( s,s,s, x, y )
#define bli_deqs( x, y ) bli_teqs( d,d,d, x, y )
#define bli_ceqs( x, y ) bli_teqs( c,c,c, x, y )
#define bli_zeqs( x, y ) bli_teqs( z,z,z, x, y )

#define bli_seq1( x ) bli_teq1s( s, x )
#define bli_deq1( x ) bli_teq1s( d, x )
#define bli_ceq1( x ) bli_teq1s( c, x )
#define bli_zeq1( x ) bli_teq1s( z, x )

#define bli_seq0( x ) bli_teq0s( s, x )
#define bli_deq0( x ) bli_teq0s( d, x )
#define bli_ceq0( x ) bli_teq0s( c, x )
#define bli_zeq0( x ) bli_teq0s( z, x )

// -- Notes --------------------------------------------------------------------

#endif

