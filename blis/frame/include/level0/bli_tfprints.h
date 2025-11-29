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

#ifndef BLIS_TFPRINTS_H
#define BLIS_TFPRINTS_H

// -- Implementation macro -----------------------------------------------------

// -- real domain implementation --

#define bli_rfprintims( \
          \
          file, spec, \
          dx, px, xr, xi \
        ) \
{ \
	fprintf( file, spec, xr ); \
}

// -- complex domain implementation --

#define bli_cfprintims( \
          \
          file, spec, \
          dx, px, xr, xi \
        ) \
{ \
	fprintf( file, spec, xr ); \
	fprintf( file, " + " ); \
	fprintf( file, spec, xi ); \
	fprintf( file, "i" ); \
}

// -- general implementation --

#define bli_tfprintims( \
          \
          file, spec, \
          dx, px, xr, xi \
        ) \
{ \
	PASTEMAC(dx,fprintims) \
	( \
	  file, spec, \
	  dx, px, xr, xi \
	); \
}

// -- API macros ---------------------------------------------------------------

// -- Consolidated --

// tfprints
#define bli_tfprints( chx, file, spec, x ) \
        bli_tfprintims \
        ( \
          file, spec, \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
          PASTEMAC(chx,real)(x), \
          PASTEMAC(chx,imag)(x) \
        )

// -- Exposed real/imaginary --

// tfprintris
#define bli_tfprintris( chx, file, spec, xr, xi ) \
        bli_tfprintims \
        ( \
          file, spec, \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
                     xr, \
                     xi \
        )

// -- Higher-level static functions --------------------------------------------

// -- Notes --------------------------------------------------------------------

#endif

