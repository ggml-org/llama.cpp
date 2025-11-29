/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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

#ifndef BLIS_GENTDEF_MACRO_DEFS_H
#define BLIS_GENTDEF_MACRO_DEFS_H

//
// -- MACROS TO INSERT TYPEDEF-GENERATING MACROS -------------------------------
//


// -- function typedef macro (both typed and void) --

#define INSERT_GENTDEF( opname ) \
\
GENTDEF( float,    s, opname, _ft ) \
GENTDEF( double,   d, opname, _ft ) \
GENTDEF( scomplex, c, opname, _ft ) \
GENTDEF( dcomplex, z, opname, _ft ) \
\
GENTDEF( void,     s, opname, _vft ) \
GENTDEF( void,     d, opname, _vft ) \
GENTDEF( void,     c, opname, _vft ) \
GENTDEF( void,     z, opname, _vft ) \
\
GENTDEF( void,      , opname, _vft )

// -- function typedef macro (both typed and void) with real projection --

#define INSERT_GENTDEFR( opname ) \
\
GENTDEFR( float,    float,    s, s, opname, _ft ) \
GENTDEFR( double,   double,   d, d, opname, _ft ) \
GENTDEFR( scomplex, float,    c, s, opname, _ft ) \
GENTDEFR( dcomplex, double,   z, d, opname, _ft ) \
\
GENTDEFR( void,     void,     s, s, opname, _vft ) \
GENTDEFR( void,     void,     d, d, opname, _vft ) \
GENTDEFR( void,     void,     c, s, opname, _vft ) \
GENTDEFR( void,     void,     z, d, opname, _vft ) \
\
GENTDEFR( void,     void,      ,  , opname, _vft )


#endif
