/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2023, Southern Methodist University

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


#ifndef BLIS_GENTFUNC_MACRO_DEFS_H
#define BLIS_GENTFUNC_MACRO_DEFS_H

//
// -- MACROS TO INSERT FUNCTION-GENERATING MACROS ------------------------------
//



// -- Macros for generating BLAS routines --------------------------------------


// -- Basic one-operand macro --


#define INSERT_GENTFUNC_BLAS( blasname, blisname ) \
\
GENTFUNC( float,    s, blasname, blisname ) \
GENTFUNC( double,   d, blasname, blisname ) \
GENTFUNC( scomplex, c, blasname, blisname ) \
GENTFUNC( dcomplex, z, blasname, blisname )


// -- Basic one-operand macro with real domain only --


#define INSERT_GENTFUNCRO_BLAS( blasname, blisname ) \
\
GENTFUNCRO( float,    s, blasname, blisname ) \
GENTFUNCRO( double,   d, blasname, blisname )


// -- Basic one-operand macro with complex domain only and real projection --


#define INSERT_GENTFUNCCO_BLAS( blasname, blisname ) \
\
GENTFUNCCO( scomplex, float,  c, s, blasname, blisname ) \
GENTFUNCCO( dcomplex, double, z, d, blasname, blisname )


// -- Basic one-operand macro with conjugation (real funcs only, used only for dot, ger) --


#define INSERT_GENTFUNCDOTR_BLAS( blasname, blisname ) \
\
GENTFUNCDOT( float,    s,  , BLIS_NO_CONJUGATE, blasname, blisname ) \
GENTFUNCDOT( double,   d,  , BLIS_NO_CONJUGATE, blasname, blisname )


// -- Basic one-operand macro with conjugation (complex funcs only, used only for dot, ger) --


#define INSERT_GENTFUNCDOTC_BLAS( blasname, blisname ) \
\
GENTFUNCDOT( scomplex, c, c, BLIS_CONJUGATE,    blasname, blisname ) \
GENTFUNCDOT( scomplex, c, u, BLIS_NO_CONJUGATE, blasname, blisname ) \
GENTFUNCDOT( dcomplex, z, c, BLIS_CONJUGATE,    blasname, blisname ) \
GENTFUNCDOT( dcomplex, z, u, BLIS_NO_CONJUGATE, blasname, blisname )


// -- Basic one-operand macro with conjugation (used only for dot, ger) --


#define INSERT_GENTFUNCDOT_BLAS( blasname, blisname ) \
\
INSERT_GENTFUNCDOTR_BLAS( blasname, blisname ) \
INSERT_GENTFUNCDOTC_BLAS( blasname, blisname )


// -- Basic one-operand macro with real projection --


#define INSERT_GENTFUNCR_BLAS( rblasname, cblasname, blisname ) \
\
GENTFUNCR( float,    float,  s, s, rblasname, blisname ) \
GENTFUNCR( double,   double, d, d, rblasname, blisname ) \
GENTFUNCR( scomplex, float,  c, s, cblasname, blisname ) \
GENTFUNCR( dcomplex, double, z, d, cblasname, blisname )


// -- Alternate two-operand macro (one char for complex, one for real proj) --


#define INSERT_GENTFUNCR2_BLAS( blasname, blisname ) \
\
GENTFUNCR2( float,    float,  s,  , blasname, blisname ) \
GENTFUNCR2( double,   double, d,  , blasname, blisname ) \
GENTFUNCR2( scomplex, float,  c, s, blasname, blisname ) \
GENTFUNCR2( dcomplex, double, z, d, blasname, blisname )


// -- Extended two-operand macro (used only for scal) --


#define INSERT_GENTFUNCSCAL_BLAS( blasname, blisname ) \
\
GENTFUNCSCAL( float,    float,    s,  , s, blasname, blisname ) \
GENTFUNCSCAL( double,   double,   d,  , d, blasname, blisname ) \
GENTFUNCSCAL( scomplex, scomplex, c,  , c, blasname, blisname ) \
GENTFUNCSCAL( dcomplex, dcomplex, z,  , z, blasname, blisname ) \
GENTFUNCSCAL( scomplex, float,    c, s, s, blasname, blisname ) \
GENTFUNCSCAL( dcomplex, double,   z, d, d, blasname, blisname )




// -- Macros for functions with one operand ------------------------------------


// -- Basic one-operand macro --

#define INSERT_GENTFUNC_BASIC( ... ) \
\
GENTFUNC( float,    s, __VA_ARGS__ ) \
GENTFUNC( double,   d, __VA_ARGS__ ) \
GENTFUNC( scomplex, c, __VA_ARGS__ ) \
GENTFUNC( dcomplex, z, __VA_ARGS__ )



// -- Basic one-operand with real projection --

#define INSERT_GENTFUNCR_BASIC( ... ) \
\
GENTFUNCR( float,    float,  s, s, __VA_ARGS__ ) \
GENTFUNCR( double,   double, d, d, __VA_ARGS__ ) \
GENTFUNCR( scomplex, float,  c, s, __VA_ARGS__ ) \
GENTFUNCR( dcomplex, double, z, d, __VA_ARGS__ )



// -- Basic one-operand macro with real domain only --

#define INSERT_GENTFUNCRO_BASIC( ... ) \
\
GENTFUNCRO( float,  s, __VA_ARGS__ ) \
GENTFUNCRO( double, d, __VA_ARGS__ )

// -- Basic one-operand macro with complex domain only --

#define INSERT_GENTFUNCCO_BASIC( ... ) \
\
GENTFUNCCO( scomplex, c, __VA_ARGS__ ) \
GENTFUNCCO( dcomplex, z, __VA_ARGS__ )

// -- Basic one-operand macro with real domain only and complex projection --

#define INSERT_GENTFUNCRO( ... ) \
\
GENTFUNCRO( float,  scomplex, s, c, __VA_ARGS__ ) \
GENTFUNCRO( double, dcomplex, d, z, __VA_ARGS__ )

// -- Basic one-operand macro with complex domain only and real projection --

#define INSERT_GENTFUNCCO( ... ) \
\
GENTFUNCCO( scomplex, float,  c, s, __VA_ARGS__ ) \
GENTFUNCCO( dcomplex, double, z, d, __VA_ARGS__ )



// -- Basic one-operand macro with integer instance --

#define INSERT_GENTFUNC_BASIC_I( ... ) \
\
GENTFUNC( float,    s, __VA_ARGS__ ) \
GENTFUNC( double,   d, __VA_ARGS__ ) \
GENTFUNC( scomplex, c, __VA_ARGS__ ) \
GENTFUNC( dcomplex, z, __VA_ARGS__ ) \
GENTFUNC( gint_t,   i, __VA_ARGS__ )



// -- Basic one-operand with integer projection --

#define INSERT_GENTFUNCI_BASIC( ... ) \
\
GENTFUNCI( float,    gint_t, s, i, __VA_ARGS__ ) \
GENTFUNCI( double,   gint_t, d, i, __VA_ARGS__ ) \
GENTFUNCI( scomplex, gint_t, c, i, __VA_ARGS__ ) \
GENTFUNCI( dcomplex, gint_t, z, i, __VA_ARGS__ )



// -- Basic one-operand with real and integer projections --

#define INSERT_GENTFUNCRI_BASIC( ... ) \
\
GENTFUNCRI( float,    float,  gint_t, s, s, i, __VA_ARGS__ ) \
GENTFUNCRI( double,   double, gint_t, d, d, i, __VA_ARGS__ ) \
GENTFUNCRI( scomplex, float,  gint_t, c, s, i, __VA_ARGS__ ) \
GENTFUNCRI( dcomplex, double, gint_t, z, d, i, __VA_ARGS__ )




// -- Macros for functions with two primary operands ---------------------------


// -- Basic two-operand macro --

#define INSERT_GENTFUNC2_BASIC( ... ) \
\
GENTFUNC2( float,    float,    s, s, __VA_ARGS__ ) \
GENTFUNC2( double,   double,   d, d, __VA_ARGS__ ) \
GENTFUNC2( scomplex, scomplex, c, c, __VA_ARGS__ ) \
GENTFUNC2( dcomplex, dcomplex, z, z, __VA_ARGS__ )



// -- Mixed domain two-operand macro --

#define INSERT_GENTFUNC2_MIX_D( ... ) \
\
GENTFUNC2( float,    scomplex, s, c, __VA_ARGS__ ) \
GENTFUNC2( scomplex, float,    c, s, __VA_ARGS__ ) \
\
GENTFUNC2( double,   dcomplex, d, z, __VA_ARGS__ ) \
GENTFUNC2( dcomplex, double,   z, d, __VA_ARGS__ )



// -- Mixed precision two-operand macro --

#define INSERT_GENTFUNC2_MIX_P( ... ) \
\
GENTFUNC2( float,    double,   s, d, __VA_ARGS__ ) \
GENTFUNC2( double,   float,    d, s, __VA_ARGS__ ) \
\
GENTFUNC2( scomplex, dcomplex, c, z, __VA_ARGS__ ) \
GENTFUNC2( dcomplex, scomplex, z, c, __VA_ARGS__ )



// -- Mixed domain/precision (all) two-operand macro --

#define INSERT_GENTFUNC2_MIX_DP( ... ) \
\
GENTFUNC2( float,    double,   s, d, __VA_ARGS__ ) \
GENTFUNC2( float,    scomplex, s, c, __VA_ARGS__ ) \
GENTFUNC2( float,    dcomplex, s, z, __VA_ARGS__ ) \
\
GENTFUNC2( double,   float,    d, s, __VA_ARGS__ ) \
GENTFUNC2( double,   scomplex, d, c, __VA_ARGS__ ) \
GENTFUNC2( double,   dcomplex, d, z, __VA_ARGS__ ) \
\
GENTFUNC2( scomplex, float,    c, s, __VA_ARGS__ ) \
GENTFUNC2( scomplex, double,   c, d, __VA_ARGS__ ) \
GENTFUNC2( scomplex, dcomplex, c, z, __VA_ARGS__ ) \
\
GENTFUNC2( dcomplex, float,    z, s, __VA_ARGS__ ) \
GENTFUNC2( dcomplex, double,   z, d, __VA_ARGS__ ) \
GENTFUNC2( dcomplex, scomplex, z, c, __VA_ARGS__ )



// -- Basic two-operand with real projection of second operand --

#define INSERT_GENTFUNC2R_BASIC( ... ) \
\
GENTFUNC2R( float,    float,    float,    s, s, s, __VA_ARGS__ ) \
GENTFUNC2R( double,   double,   double,   d, d, d, __VA_ARGS__ ) \
GENTFUNC2R( scomplex, scomplex, float,    c, c, s, __VA_ARGS__ ) \
GENTFUNC2R( dcomplex, dcomplex, double,   z, z, d, __VA_ARGS__ )



// -- Mixed domain two-operand with real projection of second operand --

#define INSERT_GENTFUNC2R_MIX_D( ... ) \
\
GENTFUNC2R( float,    scomplex, float,    s, c, s, __VA_ARGS__ ) \
GENTFUNC2R( scomplex, float,    float,    c, s, s, __VA_ARGS__ ) \
\
GENTFUNC2R( double,   dcomplex, double,   d, z, d, __VA_ARGS__ ) \
GENTFUNC2R( dcomplex, double,   double,   z, d, d, __VA_ARGS__ )



// -- Mixed precision two-operand with real projection of second operand --

#define INSERT_GENTFUNC2R_MIX_P( ... ) \
\
GENTFUNC2R( float,    double,   double,   s, d, d, __VA_ARGS__ ) \
GENTFUNC2R( double,   float,    float,    d, s, s, __VA_ARGS__ ) \
\
GENTFUNC2R( scomplex, dcomplex, double,   c, z, d, __VA_ARGS__ ) \
GENTFUNC2R( dcomplex, scomplex, float,    z, c, s, __VA_ARGS__ )



// -- Real-only two-operand with complex projection of both operands --

#define INSERT_GENTFUNC2RO( ... ) \
\
GENTFUNC2RO( float,  scomplex, float,  scomplex, s, c, s, c, __VA_ARGS__ ) \
GENTFUNC2RO( double, dcomplex, double, dcomplex, d, z, d, z, __VA_ARGS__ )



// -- Mixed precision real-only two-operand with complex projection of both operands --

#define INSERT_GENTFUNC2RO_MIX_P( ... ) \
\
GENTFUNC2RO( float,  scomplex, double, dcomplex, s, c, d, z, __VA_ARGS__ ) \
GENTFUNC2RO( double, dcomplex, float,  scomplex, d, z, s, c, __VA_ARGS__ )



// -- Mixed domain/precision (all) two-operand macro with real projection of second operand --

#define INSERT_GENTFUNC2R_MIX_DP( ... ) \
\
GENTFUNC2R( float,    double,   double,   s, d, d, __VA_ARGS__ ) \
GENTFUNC2R( float,    scomplex, float,    s, c, s, __VA_ARGS__ ) \
GENTFUNC2R( float,    dcomplex, double,   s, z, d, __VA_ARGS__ ) \
\
GENTFUNC2R( double,   float,    float,    d, s, s, __VA_ARGS__ ) \
GENTFUNC2R( double,   scomplex, float,    d, c, s, __VA_ARGS__ ) \
GENTFUNC2R( double,   dcomplex, double,   d, z, d, __VA_ARGS__ ) \
\
GENTFUNC2R( scomplex, float,    float,    c, s, s, __VA_ARGS__ ) \
GENTFUNC2R( scomplex, double,   double,   c, d, d, __VA_ARGS__ ) \
GENTFUNC2R( scomplex, dcomplex, double,   c, z, d, __VA_ARGS__ ) \
\
GENTFUNC2R( dcomplex, float,    float,    z, s, s, __VA_ARGS__ ) \
GENTFUNC2R( dcomplex, double,   double,   z, d, d, __VA_ARGS__ ) \
GENTFUNC2R( dcomplex, scomplex, float,    z, c, s, __VA_ARGS__ )




// -- Macros for functions with three primary operands -------------------------


// -- Basic three-operand macro --

#define INSERT_GENTFUNC3_BASIC( ... ) \
\
GENTFUNC3( float,    float,    float,    s, s, s, __VA_ARGS__ ) \
GENTFUNC3( double,   double,   double,   d, d, d, __VA_ARGS__ ) \
GENTFUNC3( scomplex, scomplex, scomplex, c, c, c, __VA_ARGS__ ) \
GENTFUNC3( dcomplex, dcomplex, dcomplex, z, z, z, __VA_ARGS__ )



// -- Mixed domain three-operand macro --

#define INSERT_GENTFUNC3_MIX_D( ... ) \
\
GENTFUNC3( float,    float,    scomplex, s, s, c, __VA_ARGS__ ) \
GENTFUNC3( float,    scomplex, float,    s, c, s, __VA_ARGS__ ) \
GENTFUNC3( float,    scomplex, scomplex, s, c, c, __VA_ARGS__ ) \
\
GENTFUNC3( double,   double,   dcomplex, d, d, z, __VA_ARGS__ ) \
GENTFUNC3( double,   dcomplex, double,   d, z, d, __VA_ARGS__ ) \
GENTFUNC3( double,   dcomplex, dcomplex, d, z, z, __VA_ARGS__ ) \
\
GENTFUNC3( scomplex, float,    float,    c, s, s, __VA_ARGS__ ) \
GENTFUNC3( scomplex, float,    scomplex, c, s, c, __VA_ARGS__ ) \
GENTFUNC3( scomplex, scomplex, float,    c, c, s, __VA_ARGS__ ) \
\
GENTFUNC3( dcomplex, double,   double,   z, d, d, __VA_ARGS__ ) \
GENTFUNC3( dcomplex, double,   dcomplex, z, d, z, __VA_ARGS__ ) \
GENTFUNC3( dcomplex, dcomplex, double,   z, z, d, __VA_ARGS__ )



// -- Mixed precision three-operand macro --

#define INSERT_GENTFUNC3_MIX_P( ... ) \
\
GENTFUNC3( float,    float,    double,   s, s, d, __VA_ARGS__ ) \
GENTFUNC3( float,    float,    dcomplex, s, s, z, __VA_ARGS__ ) \
\
GENTFUNC3( float,    double,   float,    s, d, s, __VA_ARGS__ ) \
GENTFUNC3( float,    double,   double,   s, d, d, __VA_ARGS__ ) \
GENTFUNC3( float,    double,   scomplex, s, d, c, __VA_ARGS__ ) \
GENTFUNC3( float,    double,   dcomplex, s, d, z, __VA_ARGS__ ) \
\
GENTFUNC3( float,    scomplex, double,   s, c, d, __VA_ARGS__ ) \
GENTFUNC3( float,    scomplex, dcomplex, s, c, z, __VA_ARGS__ ) \
\
GENTFUNC3( float,    dcomplex, float,    s, z, s, __VA_ARGS__ ) \
GENTFUNC3( float,    dcomplex, double,   s, z, d, __VA_ARGS__ ) \
GENTFUNC3( float,    dcomplex, scomplex, s, z, c, __VA_ARGS__ ) \
GENTFUNC3( float,    dcomplex, dcomplex, s, z, z, __VA_ARGS__ ) \
\
\
GENTFUNC3( double,   float,    float,    d, s, s, __VA_ARGS__ ) \
GENTFUNC3( double,   float,    double,   d, s, d, __VA_ARGS__ ) \
GENTFUNC3( double,   float,    scomplex, d, s, c, __VA_ARGS__ ) \
GENTFUNC3( double,   float,    dcomplex, d, s, z, __VA_ARGS__ ) \
\
GENTFUNC3( double,   double,   float,    d, d, s, __VA_ARGS__ ) \
GENTFUNC3( double,   double,   scomplex, d, d, c, __VA_ARGS__ ) \
\
GENTFUNC3( double,   scomplex, float,    d, c, s, __VA_ARGS__ ) \
GENTFUNC3( double,   scomplex, double,   d, c, d, __VA_ARGS__ ) \
GENTFUNC3( double,   scomplex, scomplex, d, c, c, __VA_ARGS__ ) \
GENTFUNC3( double,   scomplex, dcomplex, d, c, z, __VA_ARGS__ ) \
\
GENTFUNC3( double,   dcomplex, float,    d, z, s, __VA_ARGS__ ) \
GENTFUNC3( double,   dcomplex, scomplex, d, z, c, __VA_ARGS__ ) \
\
\
GENTFUNC3( scomplex, float,    double,   c, s, d, __VA_ARGS__ ) \
GENTFUNC3( scomplex, float,    dcomplex, c, s, z, __VA_ARGS__ ) \
\
GENTFUNC3( scomplex, double,   float,    c, d, s, __VA_ARGS__ ) \
GENTFUNC3( scomplex, double,   double,   c, d, d, __VA_ARGS__ ) \
GENTFUNC3( scomplex, double,   scomplex, c, d, c, __VA_ARGS__ ) \
GENTFUNC3( scomplex, double,   dcomplex, c, d, z, __VA_ARGS__ ) \
\
GENTFUNC3( scomplex, scomplex, double,   c, c, d, __VA_ARGS__ ) \
GENTFUNC3( scomplex, scomplex, dcomplex, c, c, z, __VA_ARGS__ ) \
\
GENTFUNC3( scomplex, dcomplex, float,    c, z, s, __VA_ARGS__ ) \
GENTFUNC3( scomplex, dcomplex, double,   c, z, d, __VA_ARGS__ ) \
GENTFUNC3( scomplex, dcomplex, scomplex, c, z, c, __VA_ARGS__ ) \
GENTFUNC3( scomplex, dcomplex, dcomplex, c, z, z, __VA_ARGS__ ) \
\
\
GENTFUNC3( dcomplex, float,    float,    z, s, s, __VA_ARGS__ ) \
GENTFUNC3( dcomplex, float,    double,   z, s, d, __VA_ARGS__ ) \
GENTFUNC3( dcomplex, float,    scomplex, z, s, c, __VA_ARGS__ ) \
GENTFUNC3( dcomplex, float,    dcomplex, z, s, z, __VA_ARGS__ ) \
\
GENTFUNC3( dcomplex, double,   float,    z, d, s, __VA_ARGS__ ) \
GENTFUNC3( dcomplex, double,   scomplex, z, d, c, __VA_ARGS__ ) \
\
GENTFUNC3( dcomplex, scomplex, float,    z, c, s, __VA_ARGS__ ) \
GENTFUNC3( dcomplex, scomplex, double,   z, c, d, __VA_ARGS__ ) \
GENTFUNC3( dcomplex, scomplex, scomplex, z, c, c, __VA_ARGS__ ) \
GENTFUNC3( dcomplex, scomplex, dcomplex, z, c, z, __VA_ARGS__ ) \
\
GENTFUNC3( dcomplex, dcomplex, float,    z, z, s, __VA_ARGS__ ) \
GENTFUNC3( dcomplex, dcomplex, scomplex, z, z, c, __VA_ARGS__ )



// -- Basic three-operand with union of operands 1 and 2 --

#define INSERT_GENTFUNC3U12_BASIC( ... ) \
\
GENTFUNC3U12( float,    float,    float,    float,    s, s, s, s, __VA_ARGS__ ) \
GENTFUNC3U12( double,   double,   double,   double,   d, d, d, d, __VA_ARGS__ ) \
GENTFUNC3U12( scomplex, scomplex, scomplex, scomplex, c, c, c, c, __VA_ARGS__ ) \
GENTFUNC3U12( dcomplex, dcomplex, dcomplex, dcomplex, z, z, z, z, __VA_ARGS__ )



// -- Mixed domain three-operand with union of operands 1 and 2 --

#define INSERT_GENTFUNC3U12_MIX_D( ... ) \
\
GENTFUNC3U12( float,    float,    scomplex, float,    s, s, c, s, __VA_ARGS__ ) \
GENTFUNC3U12( float,    scomplex, float,    scomplex, s, c, s, c, __VA_ARGS__ ) \
GENTFUNC3U12( float,    scomplex, scomplex, scomplex, s, c, c, c, __VA_ARGS__ ) \
\
GENTFUNC3U12( double,   double,   dcomplex, double,   d, d, z, d, __VA_ARGS__ ) \
GENTFUNC3U12( double,   dcomplex, double,   dcomplex, d, z, d, z, __VA_ARGS__ ) \
GENTFUNC3U12( double,   dcomplex, dcomplex, dcomplex, d, z, z, z, __VA_ARGS__ ) \
\
GENTFUNC3U12( scomplex, float,    float,    scomplex, c, s, s, c, __VA_ARGS__ ) \
GENTFUNC3U12( scomplex, float,    scomplex, scomplex, c, s, c, c, __VA_ARGS__ ) \
GENTFUNC3U12( scomplex, scomplex, float,    scomplex, c, c, s, c, __VA_ARGS__ ) \
\
GENTFUNC3U12( dcomplex, double,   double,   dcomplex, z, d, d, z, __VA_ARGS__ ) \
GENTFUNC3U12( dcomplex, double,   dcomplex, dcomplex, z, d, z, z, __VA_ARGS__ ) \
GENTFUNC3U12( dcomplex, dcomplex, double,   dcomplex, z, z, d, z, __VA_ARGS__ )



// -- Mixed precision three-operand with union of operands 1 and 2 --

#define INSERT_GENTFUNC3U12_MIX_P( ... ) \
\
GENTFUNC3U12( float,    float,    double,   float,    s, s, d, s, __VA_ARGS__ ) \
GENTFUNC3U12( float,    float,    dcomplex, float,    s, s, z, s, __VA_ARGS__ ) \
\
GENTFUNC3U12( float,    double,   float,    double,   s, d, s, d, __VA_ARGS__ ) \
GENTFUNC3U12( float,    double,   double,   double,   s, d, d, d, __VA_ARGS__ ) \
GENTFUNC3U12( float,    double,   scomplex, double,   s, d, c, d, __VA_ARGS__ ) \
GENTFUNC3U12( float,    double,   dcomplex, double,   s, d, z, d, __VA_ARGS__ ) \
\
GENTFUNC3U12( float,    scomplex, double,   scomplex, s, c, d, c, __VA_ARGS__ ) \
GENTFUNC3U12( float,    scomplex, dcomplex, scomplex, s, c, z, c, __VA_ARGS__ ) \
\
GENTFUNC3U12( float,    dcomplex, float,    dcomplex, s, z, s, z, __VA_ARGS__ ) \
GENTFUNC3U12( float,    dcomplex, double,   dcomplex, s, z, d, z, __VA_ARGS__ ) \
GENTFUNC3U12( float,    dcomplex, scomplex, dcomplex, s, z, c, z, __VA_ARGS__ ) \
GENTFUNC3U12( float,    dcomplex, dcomplex, dcomplex, s, z, z, z, __VA_ARGS__ ) \
\
\
GENTFUNC3U12( double,   float,    float,    double,   d, s, s, d, __VA_ARGS__ ) \
GENTFUNC3U12( double,   float,    double,   double,   d, s, d, d, __VA_ARGS__ ) \
GENTFUNC3U12( double,   float,    scomplex, double,   d, s, c, d, __VA_ARGS__ ) \
GENTFUNC3U12( double,   float,    dcomplex, double,   d, s, z, d, __VA_ARGS__ ) \
\
GENTFUNC3U12( double,   double,   float,    double,   d, d, s, d, __VA_ARGS__ ) \
GENTFUNC3U12( double,   double,   scomplex, double,   d, d, c, d, __VA_ARGS__ ) \
\
GENTFUNC3U12( double,   scomplex, float,    dcomplex, d, c, s, z, __VA_ARGS__ ) \
GENTFUNC3U12( double,   scomplex, double,   dcomplex, d, c, d, z, __VA_ARGS__ ) \
GENTFUNC3U12( double,   scomplex, scomplex, dcomplex, d, c, c, z, __VA_ARGS__ ) \
GENTFUNC3U12( double,   scomplex, dcomplex, dcomplex, d, c, z, z, __VA_ARGS__ ) \
\
GENTFUNC3U12( double,   dcomplex, float,    dcomplex, d, z, s, z, __VA_ARGS__ ) \
GENTFUNC3U12( double,   dcomplex, scomplex, dcomplex, d, z, c, z, __VA_ARGS__ ) \
\
\
GENTFUNC3U12( scomplex, float,    double,   scomplex, c, s, d, c, __VA_ARGS__ ) \
GENTFUNC3U12( scomplex, float,    dcomplex, scomplex, c, s, z, c, __VA_ARGS__ ) \
\
GENTFUNC3U12( scomplex, double,   float,    dcomplex, c, d, s, z, __VA_ARGS__ ) \
GENTFUNC3U12( scomplex, double,   double,   dcomplex, c, d, d, z, __VA_ARGS__ ) \
GENTFUNC3U12( scomplex, double,   scomplex, dcomplex, c, d, c, z, __VA_ARGS__ ) \
GENTFUNC3U12( scomplex, double,   dcomplex, dcomplex, c, d, z, z, __VA_ARGS__ ) \
\
GENTFUNC3U12( scomplex, scomplex, double,   scomplex, c, c, d, c, __VA_ARGS__ ) \
GENTFUNC3U12( scomplex, scomplex, dcomplex, scomplex, c, c, z, c, __VA_ARGS__ ) \
\
GENTFUNC3U12( scomplex, dcomplex, float,    dcomplex, c, z, s, z, __VA_ARGS__ ) \
GENTFUNC3U12( scomplex, dcomplex, double,   dcomplex, c, z, d, z, __VA_ARGS__ ) \
GENTFUNC3U12( scomplex, dcomplex, scomplex, dcomplex, c, z, c, z, __VA_ARGS__ ) \
GENTFUNC3U12( scomplex, dcomplex, dcomplex, dcomplex, c, z, z, z, __VA_ARGS__ ) \
\
\
GENTFUNC3U12( dcomplex, float,    float,    dcomplex, z, s, s, z, __VA_ARGS__ ) \
GENTFUNC3U12( dcomplex, float,    double,   dcomplex, z, s, d, z, __VA_ARGS__ ) \
GENTFUNC3U12( dcomplex, float,    scomplex, dcomplex, z, s, c, z, __VA_ARGS__ ) \
GENTFUNC3U12( dcomplex, float,    dcomplex, dcomplex, z, s, z, z, __VA_ARGS__ ) \
\
GENTFUNC3U12( dcomplex, double,   float,    dcomplex, z, d, s, z, __VA_ARGS__ ) \
GENTFUNC3U12( dcomplex, double,   scomplex, dcomplex, z, d, c, z, __VA_ARGS__ ) \
\
GENTFUNC3U12( dcomplex, scomplex, float,    dcomplex, z, c, s, z, __VA_ARGS__ ) \
GENTFUNC3U12( dcomplex, scomplex, double,   dcomplex, z, c, d, z, __VA_ARGS__ ) \
GENTFUNC3U12( dcomplex, scomplex, scomplex, dcomplex, z, c, c, z, __VA_ARGS__ ) \
GENTFUNC3U12( dcomplex, scomplex, dcomplex, dcomplex, z, c, z, z, __VA_ARGS__ ) \
\
GENTFUNC3U12( dcomplex, dcomplex, float,    dcomplex, z, z, s, z, __VA_ARGS__ ) \
GENTFUNC3U12( dcomplex, dcomplex, scomplex, dcomplex, z, z, c, z, __VA_ARGS__ )


#endif
