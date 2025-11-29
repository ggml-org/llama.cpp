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


#ifndef BLIS_GENTPROT_MACRO_DEFS_H
#define BLIS_GENTPROT_MACRO_DEFS_H

//
// -- MACROS TO INSERT PROTOTYPE-GENERATING MACROS -----------------------------
//



// -- Macros for generating BLAS routines --------------------------------------


// -- Basic one-operand macro --


#define INSERT_GENTPROT_BLAS( blasname ) \
\
GENTPROT( float,    s, blasname ) \
GENTPROT( double,   d, blasname ) \
GENTPROT( scomplex, c, blasname ) \
GENTPROT( dcomplex, z, blasname )


// -- Basic one-operand macro with real domain only --


#define INSERT_GENTPROTRO_BLAS( blasname ) \
\
GENTPROTRO( float,    s, blasname ) \
GENTPROTRO( double,   d, blasname )


// -- Basic one-operand macro with complex domain only and real projection --


#define INSERT_GENTPROTCO_BLAS( blasname ) \
\
GENTPROTCO( scomplex, float,  c, s, blasname ) \
GENTPROTCO( dcomplex, double, z, d, blasname )


// -- Basic one-operand macro with conjugation (real funcs only, used only for dot, ger) --


#define INSERT_GENTPROTDOTR_BLAS( blasname ) \
\
GENTPROTDOT( float,    s,  , blasname ) \
GENTPROTDOT( double,   d,  , blasname )


// -- Basic one-operand macro with conjugation (complex funcs only, used only for dot, ger) --


#define INSERT_GENTPROTDOTC_BLAS( blasname ) \
\
GENTPROTDOT( scomplex, c, c, blasname ) \
GENTPROTDOT( scomplex, c, u, blasname ) \
GENTPROTDOT( dcomplex, z, c, blasname ) \
GENTPROTDOT( dcomplex, z, u, blasname )


// -- Basic one-operand macro with conjugation (used only for dot, ger) --


#define INSERT_GENTPROTDOT_BLAS( blasname ) \
\
INSERT_GENTPROTDOTR_BLAS( blasname ) \
INSERT_GENTPROTDOTC_BLAS( blasname )


// -- Basic one-operand macro with real projection --


#define INSERT_GENTPROTR_BLAS( rblasname, cblasname ) \
\
GENTPROTR( float,    float,  s, s, rblasname ) \
GENTPROTR( double,   double, d, d, rblasname ) \
GENTPROTR( scomplex, float,  c, s, cblasname ) \
GENTPROTR( dcomplex, double, z, d, cblasname )


// -- Alternate two-operand macro (one char for complex, one for real proj) --


#define INSERT_GENTPROTR2_BLAS( blasname ) \
\
GENTPROTR2( float,    float,   , s, blasname ) \
GENTPROTR2( double,   double,  , d, blasname ) \
GENTPROTR2( scomplex, float,  c, s, blasname ) \
GENTPROTR2( dcomplex, double, z, d, blasname )


// -- Extended two-operand macro (used only for scal) --


#define INSERT_GENTPROTSCAL_BLAS( blasname ) \
\
GENTPROTSCAL( float,    float,     , s, blasname ) \
GENTPROTSCAL( double,   double,    , d, blasname ) \
GENTPROTSCAL( scomplex, scomplex,  , c, blasname ) \
GENTPROTSCAL( dcomplex, dcomplex,  , z, blasname ) \
GENTPROTSCAL( float,    scomplex, s, c, blasname ) \
GENTPROTSCAL( double,   dcomplex, d, z, blasname )




// -- Macros for functions with one operand ------------------------------------


// -- Basic one-operand macro --

#define INSERT_GENTPROT_BASIC( ... ) \
\
GENTPROT( float,    s, __VA_ARGS__ ) \
GENTPROT( double,   d, __VA_ARGS__ ) \
GENTPROT( scomplex, c, __VA_ARGS__ ) \
GENTPROT( dcomplex, z, __VA_ARGS__ )



// -- Basic one-operand with real projection --

#define INSERT_GENTPROTR_BASIC( ... ) \
\
GENTPROTR( float,    float,  s, s, __VA_ARGS__ ) \
GENTPROTR( double,   double, d, d, __VA_ARGS__ ) \
GENTPROTR( scomplex, float,  c, s, __VA_ARGS__ ) \
GENTPROTR( dcomplex, double, z, d, __VA_ARGS__ )



// -- Basic one-operand macro with real domain only --

#define INSERT_GENTPROTRO_BASIC( ... ) \
\
GENTPROTRO( float,  s, __VA_ARGS__ ) \
GENTPROTRO( double, d, __VA_ARGS__ )



// -- Basic one-operand macro with complex domain only and real projection --

#define INSERT_GENTPROTCO_BASIC( ... ) \
\
GENTPROTCO( scomplex, float,  c, s, __VA_ARGS__ ) \
GENTPROTCO( dcomplex, double, z, d, __VA_ARGS__ )



// -- Basic one-operand macro with integer instance --

#define INSERT_GENTPROT_BASIC_I( ... ) \
\
GENTPROT( float,    s, __VA_ARGS__ ) \
GENTPROT( double,   d, __VA_ARGS__ ) \
GENTPROT( scomplex, c, __VA_ARGS__ ) \
GENTPROT( dcomplex, z, __VA_ARGS__ ) \
GENTPROT( gint_t,   i, __VA_ARGS__ )



// -- Basic one-operand with integer projection --

#define INSERT_GENTPROTI_BASIC( ... ) \
\
GENTPROTI( float,    gint_t, s, i, __VA_ARGS__ ) \
GENTPROTI( double,   gint_t, d, i, __VA_ARGS__ ) \
GENTPROTI( scomplex, gint_t, c, i, __VA_ARGS__ ) \
GENTPROTI( dcomplex, gint_t, z, i, __VA_ARGS__ )



// -- Basic one-operand with real and integer projections --

#define INSERT_GENTPROTRI_BASIC( funcname ) \
\
GENTPROTRI( float,    float,  gint_t, s, s, i, funcname ) \
GENTPROTRI( double,   double, gint_t, d, d, i, funcname ) \
GENTPROTRI( scomplex, float,  gint_t, c, s, i, funcname ) \
GENTPROTRI( dcomplex, double, gint_t, z, d, i, funcname )




// -- Macros for functions with two primary operands ---------------------------


// -- Basic two-operand macro --

#define INSERT_GENTPROT2_BASIC( ... ) \
\
GENTPROT2( float,    float,    s, s, __VA_ARGS__ ) \
GENTPROT2( double,   double,   d, d, __VA_ARGS__ ) \
GENTPROT2( scomplex, scomplex, c, c, __VA_ARGS__ ) \
GENTPROT2( dcomplex, dcomplex, z, z, __VA_ARGS__ )



// -- Mixed domain two-operand macro --

#define INSERT_GENTPROT2_MIX_D( ... ) \
\
GENTPROT2( float,    scomplex, s, c, __VA_ARGS__ ) \
GENTPROT2( scomplex, float,    c, s, __VA_ARGS__ ) \
\
GENTPROT2( double,   dcomplex, d, z, __VA_ARGS__ ) \
GENTPROT2( dcomplex, double,   z, d, __VA_ARGS__ )



// -- Mixed precision two-operand macro --

#define INSERT_GENTPROT2_MIX_P( ... ) \
\
GENTPROT2( float,    double,   s, d, __VA_ARGS__ ) \
GENTPROT2( float,    dcomplex, s, z, __VA_ARGS__ ) \
\
GENTPROT2( double,   float,    d, s, __VA_ARGS__ ) \
GENTPROT2( double,   scomplex, d, c, __VA_ARGS__ ) \
\
GENTPROT2( scomplex, double,   c, d, __VA_ARGS__ ) \
GENTPROT2( scomplex, dcomplex, c, z, __VA_ARGS__ ) \
\
GENTPROT2( dcomplex, float,    z, s, __VA_ARGS__ ) \
GENTPROT2( dcomplex, scomplex, z, c, __VA_ARGS__ ) \



// -- Mixed domain/precision (all) two-operand macro --

#define INSERT_GENTPROT2_MIX_DP( ... ) \
\
GENTPROT2( float,    double,   s, d, __VA_ARGS__ ) \
GENTPROT2( float,    scomplex, s, c, __VA_ARGS__ ) \
GENTPROT2( float,    dcomplex, s, z, __VA_ARGS__ ) \
\
GENTPROT2( double,   float,    d, s, __VA_ARGS__ ) \
GENTPROT2( double,   scomplex, d, c, __VA_ARGS__ ) \
GENTPROT2( double,   dcomplex, d, z, __VA_ARGS__ ) \
\
GENTPROT2( scomplex, float,    c, s, __VA_ARGS__ ) \
GENTPROT2( scomplex, double,   c, d, __VA_ARGS__ ) \
GENTPROT2( scomplex, dcomplex, c, z, __VA_ARGS__ ) \
\
GENTPROT2( dcomplex, float,    z, s, __VA_ARGS__ ) \
GENTPROT2( dcomplex, double,   z, d, __VA_ARGS__ ) \
GENTPROT2( dcomplex, scomplex, z, c, __VA_ARGS__ )



// -- Basic two-operand with real projection of first operand --

#define INSERT_GENTPROT2R_BASIC( ... ) \
\
GENTPROT2R( float,    float,    float,    s, s, s, __VA_ARGS__ ) \
GENTPROT2R( double,   double,   double,   d, d, d, __VA_ARGS__ ) \
GENTPROT2R( scomplex, scomplex, float,    c, c, s, __VA_ARGS__ ) \
GENTPROT2R( dcomplex, dcomplex, double,   z, z, d, __VA_ARGS__ )



// -- Mixed domain two-operand with real projection of first operand --

#define INSERT_GENTPROT2R_MIX_D( ... ) \
\
GENTPROT2R( float,    scomplex, float,    s, c, s, __VA_ARGS__ ) \
GENTPROT2R( scomplex, float,    float,    c, s, s, __VA_ARGS__ ) \
\
GENTPROT2R( double,   dcomplex, double,   d, z, d, __VA_ARGS__ ) \
GENTPROT2R( dcomplex, double,   double,   z, d, d, __VA_ARGS__ )



// -- Mixed precision two-operand with real projection of first operand --

#define INSERT_GENTPROT2R_MIX_P( ... ) \
\
GENTPROT2R( float,    double,   float,    s, d, s, __VA_ARGS__ ) \
GENTPROT2R( float,    dcomplex, float,    s, z, s, __VA_ARGS__ ) \
\
GENTPROT2R( double,   float,    double,   d, s, d, __VA_ARGS__ ) \
GENTPROT2R( double,   scomplex, double,   d, c, d, __VA_ARGS__ ) \
\
GENTPROT2R( scomplex, double,   float,    c, d, s, __VA_ARGS__ ) \
GENTPROT2R( scomplex, dcomplex, float,    c, z, s, __VA_ARGS__ ) \
\
GENTPROT2R( dcomplex, float,    double,   z, s, d, __VA_ARGS__ ) \
GENTPROT2R( dcomplex, scomplex, double,   z, c, d, __VA_ARGS__ )



// -- Macros for functions with three primary operands -------------------------


// -- Basic three-operand macro --


#define INSERT_GENTPROT3_BASIC( funcname ) \
\
GENTPROT3( float,    float,    float,    s, s, s, funcname ) \
GENTPROT3( double,   double,   double,   d, d, d, funcname ) \
GENTPROT3( scomplex, scomplex, scomplex, c, c, c, funcname ) \
GENTPROT3( dcomplex, dcomplex, dcomplex, z, z, z, funcname )


// -- Mixed domain three-operand macro --


#define INSERT_GENTPROT3_MIX_D( funcname ) \
\
GENTPROT3( float,    float,    scomplex, s, s, c, funcname ) \
GENTPROT3( float,    scomplex, float,    s, c, s, funcname ) \
GENTPROT3( float,    scomplex, scomplex, s, c, c, funcname ) \
\
GENTPROT3( double,   double,   dcomplex, d, d, z, funcname ) \
GENTPROT3( double,   dcomplex, double,   d, z, d, funcname ) \
GENTPROT3( double,   dcomplex, dcomplex, d, z, z, funcname ) \
\
GENTPROT3( scomplex, float,    float,    c, s, s, funcname ) \
GENTPROT3( scomplex, float,    scomplex, c, s, c, funcname ) \
GENTPROT3( scomplex, scomplex, float,    c, c, s, funcname ) \
\
GENTPROT3( dcomplex, double,   double,   z, d, d, funcname ) \
GENTPROT3( dcomplex, double,   dcomplex, z, d, z, funcname ) \
GENTPROT3( dcomplex, dcomplex, double,   z, z, d, funcname )


// -- Mixed precision three-operand macro --


#define INSERT_GENTPROT3_MIX_P( funcname ) \
\
GENTPROT3( float,    float,    double,   s, s, d, funcname ) \
GENTPROT3( float,    float,    dcomplex, s, s, z, funcname ) \
\
GENTPROT3( float,    double,   float,    s, d, s, funcname ) \
GENTPROT3( float,    double,   double,   s, d, d, funcname ) \
GENTPROT3( float,    double,   scomplex, s, d, c, funcname ) \
GENTPROT3( float,    double,   dcomplex, s, d, z, funcname ) \
\
GENTPROT3( float,    scomplex, double,   s, c, d, funcname ) \
GENTPROT3( float,    scomplex, dcomplex, s, c, z, funcname ) \
\
GENTPROT3( float,    dcomplex, float,    s, z, s, funcname ) \
GENTPROT3( float,    dcomplex, double,   s, z, d, funcname ) \
GENTPROT3( float,    dcomplex, scomplex, s, z, c, funcname ) \
GENTPROT3( float,    dcomplex, dcomplex, s, z, z, funcname ) \
\
\
GENTPROT3( double,   float,    float,    d, s, s, funcname ) \
GENTPROT3( double,   float,    double,   d, s, d, funcname ) \
GENTPROT3( double,   float,    scomplex, d, s, c, funcname ) \
GENTPROT3( double,   float,    dcomplex, d, s, z, funcname ) \
\
GENTPROT3( double,   double,   float,    d, d, s, funcname ) \
GENTPROT3( double,   double,   scomplex, d, d, c, funcname ) \
\
GENTPROT3( double,   scomplex, float,    d, c, s, funcname ) \
GENTPROT3( double,   scomplex, double,   d, c, d, funcname ) \
GENTPROT3( double,   scomplex, scomplex, d, c, c, funcname ) \
GENTPROT3( double,   scomplex, dcomplex, d, c, z, funcname ) \
\
GENTPROT3( double,   dcomplex, float,    d, z, s, funcname ) \
GENTPROT3( double,   dcomplex, scomplex, d, z, c, funcname ) \
\
\
GENTPROT3( scomplex, float,    double,   c, s, d, funcname ) \
GENTPROT3( scomplex, float,    dcomplex, c, s, z, funcname ) \
\
GENTPROT3( scomplex, double,   float,    c, d, s, funcname ) \
GENTPROT3( scomplex, double,   double,   c, d, d, funcname ) \
GENTPROT3( scomplex, double,   scomplex, c, d, c, funcname ) \
GENTPROT3( scomplex, double,   dcomplex, c, d, z, funcname ) \
\
GENTPROT3( scomplex, scomplex, double,   c, c, d, funcname ) \
GENTPROT3( scomplex, scomplex, dcomplex, c, c, z, funcname ) \
\
GENTPROT3( scomplex, dcomplex, float,    c, z, s, funcname ) \
GENTPROT3( scomplex, dcomplex, double,   c, z, d, funcname ) \
GENTPROT3( scomplex, dcomplex, scomplex, c, z, c, funcname ) \
GENTPROT3( scomplex, dcomplex, dcomplex, c, z, z, funcname ) \
\
\
GENTPROT3( dcomplex, float,    float,    z, s, s, funcname ) \
GENTPROT3( dcomplex, float,    double,   z, s, d, funcname ) \
GENTPROT3( dcomplex, float,    scomplex, z, s, c, funcname ) \
GENTPROT3( dcomplex, float,    dcomplex, z, s, z, funcname ) \
\
GENTPROT3( dcomplex, double,   float,    z, d, s, funcname ) \
GENTPROT3( dcomplex, double,   scomplex, z, d, c, funcname ) \
\
GENTPROT3( dcomplex, scomplex, float,    z, c, s, funcname ) \
GENTPROT3( dcomplex, scomplex, double,   z, c, d, funcname ) \
GENTPROT3( dcomplex, scomplex, scomplex, z, c, c, funcname ) \
GENTPROT3( dcomplex, scomplex, dcomplex, z, c, z, funcname ) \
\
GENTPROT3( dcomplex, dcomplex, float,    z, z, s, funcname ) \
GENTPROT3( dcomplex, dcomplex, scomplex, z, z, c, funcname ) \



// -- Basic three-operand with union of operands 1 and 2 --


#define INSERT_GENTPROT3U12_BASIC( funcname ) \
\
GENTPROT3U12( float,    float,    float,    float,    s, s, s, s, funcname ) \
GENTPROT3U12( double,   double,   double,   double,   d, d, d, d, funcname ) \
GENTPROT3U12( scomplex, scomplex, scomplex, scomplex, c, c, c, c, funcname ) \
GENTPROT3U12( dcomplex, dcomplex, dcomplex, dcomplex, z, z, z, z, funcname )


// -- Mixed domain three-operand with union of operands 1 and 2 --


#define INSERT_GENTPROT3U12_MIX_D( funcname ) \
\
GENTPROT3U12( float,    float,    scomplex, float,    s, s, c, s, funcname ) \
GENTPROT3U12( float,    scomplex, float,    scomplex, s, c, s, c, funcname ) \
GENTPROT3U12( float,    scomplex, scomplex, scomplex, s, c, c, c, funcname ) \
\
GENTPROT3U12( double,   double,   dcomplex, double,   d, d, z, d, funcname ) \
GENTPROT3U12( double,   dcomplex, double,   dcomplex, d, z, d, z, funcname ) \
GENTPROT3U12( double,   dcomplex, dcomplex, dcomplex, d, z, z, z, funcname ) \
\
GENTPROT3U12( scomplex, float,    float,    scomplex, c, s, s, c, funcname ) \
GENTPROT3U12( scomplex, float,    scomplex, scomplex, c, s, c, c, funcname ) \
GENTPROT3U12( scomplex, scomplex, float,    scomplex, c, c, s, c, funcname ) \
\
GENTPROT3U12( dcomplex, double,   double,   dcomplex, z, d, d, z, funcname ) \
GENTPROT3U12( dcomplex, double,   dcomplex, dcomplex, z, d, z, z, funcname ) \
GENTPROT3U12( dcomplex, dcomplex, double,   dcomplex, z, z, d, z, funcname )


// -- Mixed precision three-operand with union of operands 1 and 2 --


#define INSERT_GENTPROT3U12_MIX_P( funcname ) \
\
GENTPROT3U12( float,    float,    double,   float,    s, s, d, s, funcname ) \
GENTPROT3U12( float,    float,    dcomplex, float,    s, s, z, s, funcname ) \
\
GENTPROT3U12( float,    double,   float,    double,   s, d, s, d, funcname ) \
GENTPROT3U12( float,    double,   double,   double,   s, d, d, d, funcname ) \
GENTPROT3U12( float,    double,   scomplex, double,   s, d, c, d, funcname ) \
GENTPROT3U12( float,    double,   dcomplex, double,   s, d, z, d, funcname ) \
\
GENTPROT3U12( float,    scomplex, double,   scomplex, s, c, d, c, funcname ) \
GENTPROT3U12( float,    scomplex, dcomplex, scomplex, s, c, z, c, funcname ) \
\
GENTPROT3U12( float,    dcomplex, float,    dcomplex, s, z, s, z, funcname ) \
GENTPROT3U12( float,    dcomplex, double,   dcomplex, s, z, d, z, funcname ) \
GENTPROT3U12( float,    dcomplex, scomplex, dcomplex, s, z, c, z, funcname ) \
GENTPROT3U12( float,    dcomplex, dcomplex, dcomplex, s, z, z, z, funcname ) \
\
\
GENTPROT3U12( double,   float,    float,    double,   d, s, s, d, funcname ) \
GENTPROT3U12( double,   float,    double,   double,   d, s, d, d, funcname ) \
GENTPROT3U12( double,   float,    scomplex, double,   d, s, c, d, funcname ) \
GENTPROT3U12( double,   float,    dcomplex, double,   d, s, z, d, funcname ) \
\
GENTPROT3U12( double,   double,   float,    double,   d, d, s, d, funcname ) \
GENTPROT3U12( double,   double,   scomplex, double,   d, d, c, d, funcname ) \
\
GENTPROT3U12( double,   scomplex, float,    dcomplex, d, c, s, z, funcname ) \
GENTPROT3U12( double,   scomplex, double,   dcomplex, d, c, d, z, funcname ) \
GENTPROT3U12( double,   scomplex, scomplex, dcomplex, d, c, c, z, funcname ) \
GENTPROT3U12( double,   scomplex, dcomplex, dcomplex, d, c, z, z, funcname ) \
\
GENTPROT3U12( double,   dcomplex, float,    dcomplex, d, z, s, z, funcname ) \
GENTPROT3U12( double,   dcomplex, scomplex, dcomplex, d, z, c, z, funcname ) \
\
\
GENTPROT3U12( scomplex, float,    double,   scomplex, c, s, d, c, funcname ) \
GENTPROT3U12( scomplex, float,    dcomplex, scomplex, c, s, z, c, funcname ) \
\
GENTPROT3U12( scomplex, double,   float,    dcomplex, c, d, s, z, funcname ) \
GENTPROT3U12( scomplex, double,   double,   dcomplex, c, d, d, z, funcname ) \
GENTPROT3U12( scomplex, double,   scomplex, dcomplex, c, d, c, z, funcname ) \
GENTPROT3U12( scomplex, double,   dcomplex, dcomplex, c, d, z, z, funcname ) \
\
GENTPROT3U12( scomplex, scomplex, double,   scomplex, c, c, d, c, funcname ) \
GENTPROT3U12( scomplex, scomplex, dcomplex, scomplex, c, c, z, c, funcname ) \
\
GENTPROT3U12( scomplex, dcomplex, float,    dcomplex, c, z, s, z, funcname ) \
GENTPROT3U12( scomplex, dcomplex, double,   dcomplex, c, z, d, z, funcname ) \
GENTPROT3U12( scomplex, dcomplex, scomplex, dcomplex, c, z, c, z, funcname ) \
GENTPROT3U12( scomplex, dcomplex, dcomplex, dcomplex, c, z, z, z, funcname ) \
\
\
GENTPROT3U12( dcomplex, float,    float,    dcomplex, z, s, s, z, funcname ) \
GENTPROT3U12( dcomplex, float,    double,   dcomplex, z, s, d, z, funcname ) \
GENTPROT3U12( dcomplex, float,    scomplex, dcomplex, z, s, c, z, funcname ) \
GENTPROT3U12( dcomplex, float,    dcomplex, dcomplex, z, s, z, z, funcname ) \
\
GENTPROT3U12( dcomplex, double,   float,    dcomplex, z, d, s, z, funcname ) \
GENTPROT3U12( dcomplex, double,   scomplex, dcomplex, z, d, c, z, funcname ) \
\
GENTPROT3U12( dcomplex, scomplex, float,    dcomplex, z, c, s, z, funcname ) \
GENTPROT3U12( dcomplex, scomplex, double,   dcomplex, z, c, d, z, funcname ) \
GENTPROT3U12( dcomplex, scomplex, scomplex, dcomplex, z, c, c, z, funcname ) \
GENTPROT3U12( dcomplex, scomplex, dcomplex, dcomplex, z, c, z, z, funcname ) \
\
GENTPROT3U12( dcomplex, dcomplex, float,    dcomplex, z, z, s, z, funcname ) \
GENTPROT3U12( dcomplex, dcomplex, scomplex, dcomplex, z, z, c, z, funcname )


#endif
