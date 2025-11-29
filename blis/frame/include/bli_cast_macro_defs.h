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

#ifndef BLIS_CAST_MACRO_DEFS_H
#define BLIS_CAST_MACRO_DEFS_H

// -- Typecast { bfloat16 | float | double } to bfloat16 -----------------------

#ifdef BFLOAT
BLIS_INLINE bfloat bli_bbcast( bfloat b )
{
	return b;
}
#endif

#ifdef BFLOAT
BLIS_INLINE bfloat bli_sbcast( float s )
{
	bfloat b;

	// View the float as a char array.
	char* s_ch = ( char* )&s;

	// Copy upper two bytes of float to a local bfloat16.
	memcpy( &b, &s_ch[2], 2 );

	return b;
}
#endif

#ifdef BFLOAT
BLIS_INLINE bfloat bli_dbcast( double d )
{
	bfloat b;

	// Typecast double input argument to a local float.
	float s = ( float )d;

	// View the float as a char array.
	char* s_ch = ( char* )&s;

	// Copy upper two bytes of float to a local bfloat16.
	memcpy( &b, &s_ch[2], 2 );

	return b;
}
#endif

// -- Typecast { bfloat16 | float | double | int } to float --------------------------

#ifdef BFLOAT
BLIS_INLINE float bli_bscast( bfloat b )
{
	// Initialize all bits in a local float to zero.
	float s = 0.0F;

	// View the float as a char array.
	char* s_ch = ( char* )&s;

	// Copy bfloat16 to the upper two bytes of a local float.
	memcpy( &s_ch[2], &b, 2 );

	return s;
}
#endif

BLIS_INLINE float bli_sscast( float s )
{
	return s;
}

BLIS_INLINE float bli_dscast( double d )
{
	return ( float )d;
}

BLIS_INLINE float bli_iscast( dim_t i )
{
	return ( float )i;
}

// -- Typecast { bfloat16 | float | double | int } to double -------------------------

#ifdef BFLOAT
BLIS_INLINE double bli_bdcast( bfloat b )
{
	// Initialize all bits in a local float to zero.
	float s = 0.0F;

	// View the float as a char array.
	char* s_ch = ( char* )&s;

	// Copy bfloat16 to the upper two bytes of a local float.
	memcpy( &s_ch[2], &b, 2 );

	return ( double )s;
}
#endif

BLIS_INLINE double bli_sdcast( float s )
{
	return ( double )s;
}

//#if 1
BLIS_INLINE double bli_ddcast( double d )
{
	return d;
}
//#else
//#define bli_ddcast( d )  ( d )
//#endif

BLIS_INLINE double bli_idcast( dim_t i )
{
	return ( double )i;
}

// -- Typecast { float | double | int } to int -------------------------

BLIS_INLINE dim_t bli_sicast( float s )
{
	return ( dim_t )s;
}

BLIS_INLINE dim_t bli_dicast( double d )
{
	return ( dim_t )d;
}

BLIS_INLINE dim_t bli_iicast( dim_t i )
{
	return i;
}

#if 0
// -- Fused real/imag accessor + typecast --------------------------------------

// Generate static functions that fuse two operations:
// - accessing the real and imaginary components of all datatypes (real
//   and complex)
// - typecasting a real (or imaginary) component to any real datatype
// Examples:
// static float  bli_dreals( double   a ) { return bli_dscast( bli_dreal( a ) ); }
// static double bli_sreald( float    a ) { return bli_sdcast( bli_sreal( a ) ); }
// static float  bli_creals( scomplex a ) { return bli_sscast( bli_creal( a ) ); }
// static double bli_cimagd( scomplex a ) { return bli_sdcast( bli_cimag( a ) ); }

#undef  GENTFUNC
#define GENTFUNC( chi, cho ) \
\
BLIS_INLINE PASTEMAC(cho,ctype) PASTEMAC2(chi,real,cho)( PASTEMAC(chi,ctype) a ) \
{ \
	return PASTEMAC2(chi,cho,cast)( PASTEMAC(chi,real)( a ) ); \
} \
BLIS_INLINE PASTEMAC(cho,ctype) PASTEMAC2(chi,imag,cho)( PASTEMAC(chi,ctype) a ) \
{ \
	return PASTEMAC2(chi,cho,cast)( PASTEMAC(chi,imag)( a ) ); \
}

// NOTE: We only have to generate functions that output to types [bsd] because
// these macros only need to output real types. The composition that allows
// complex types will be handled by the consumers to these bli_?[real|imag]?()
// functions.

// [bsdkcz][bsd]

GENTFUNC( b, b )
GENTFUNC( s, b )
GENTFUNC( d, b )
GENTFUNC( k, b )
GENTFUNC( c, b )
GENTFUNC( z, b )

GENTFUNC( b, s )
GENTFUNC( s, s )
GENTFUNC( d, s )
GENTFUNC( k, s )
GENTFUNC( c, s )
GENTFUNC( z, s )

GENTFUNC( b, d )
GENTFUNC( s, d )
GENTFUNC( d, d )
GENTFUNC( k, d )
GENTFUNC( c, d )
GENTFUNC( z, d )
#endif

// bli_xytcast() macros are only used in the definitions of level0 scalar
// macros. There, we use a different name from the actual cast functions--
// which are named using the format bli_xycast()--so that we can optionally
// replace them as part of the optimization below without distrubing any
// other uses of bli_xycast() that should not be changed.

#define bli_bbtcast  bli_bbcast
#define bli_sbtcast  bli_sbcast
#define bli_dbtcast  bli_dbcast
#define bli_kbtcast  bli_kbcast
#define bli_cbtcast  bli_cbcast
#define bli_zbtcast  bli_zbcast

#define bli_bstcast  bli_bscast
#define bli_sstcast  bli_sscast
#define bli_dstcast  bli_dscast
#define bli_kstcast  bli_kscast
#define bli_cstcast  bli_cscast
#define bli_zstcast  bli_zscast
#define bli_istcast  bli_iscast

#define bli_bdtcast  bli_bdcast
#define bli_sdtcast  bli_sdcast
#define bli_ddtcast  bli_ddcast
#define bli_kdtcast  bli_kdcast
#define bli_cdtcast  bli_cdcast
#define bli_zdtcast  bli_zdcast
#define bli_idtcast  bli_idcast

#define bli_sitcast  bli_sicast
#define bli_ditcast  bli_dicast
#define bli_iitcast  bli_iicast

// An optimization. In situations where computations would normally occur
// in bfloat, redundant typecasting may occur. For example, in the case of
// performing ssbbaxpy (a and x stored in type s; y stored in type b;
// compute in b), a and x would normally be typecast to b so that all
// operands are in the computation precision (namely, bfloat), but since
// our reference implementation implements bfloat flops in terms of float
// flops, all operands would need to be typecast back to s anyway just so
// the computation can take place. This means that a and x were truncated
// down to bfloat (and thus lost precision) somewhat unnecessarily. Instead,
// what could happen is that a and x remain in s, y is typecast to s,
// computation would take place in s, and then the result is truncated to
// bfloat on output to y. These macros substitute certain static function
// calls to be the equivalent calls that would cast to float instead of
// bfloat.
#ifdef BLIS_OPTIMIZE_BFLOAT_AS_FLOAT

#undef  bli_bbcast
#define bli_bbcast  bli_bscast
#undef  bli_sbcast
#define bli_sbcast  bli_sscast
#undef  bli_dbcast
#define bli_dbcast  bli_dscast
#undef  bli_kbcast
#define bli_kbcast  bli_kscast
#undef  bli_cbcast
#define bli_cbcast  bli_cscast
#undef  bli_zbcast
#define bli_zbcast  bli_zscast

#endif


// -- Basic constants (per precision) ------------------------------------------

#ifdef BLIS_OPTIMIZE_BFLOAT_AS_FLOAT

#define bli_btwo                bli_stwo
#define bli_bone                bli_sone
#define bli_bzero               bli_szero
#define bli_bmone               bli_smone
#define bli_bmtwo               bli_smtwo

#else

#define bli_btwo    bli_sbcast( bli_stwo )
#define bli_bone    bli_sbcast( bli_sone )
#define bli_bzero   bli_sbcast( bli_szero )
#define bli_bmone   bli_sbcast( bli_smone )
#define bli_bmtwo   bli_sbcast( bli_smtwo )

#endif

#define bli_stwo    2.0F
#define bli_sone    1.0F
#define bli_szero   0.0F
#define bli_smone  -1.0F
#define bli_smtwo  -2.0F

#define bli_dtwo    2.0
#define bli_done    1.0
#define bli_dzero   0.0
#define bli_dmone  -1.0
#define bli_dmtwo  -2.0

// -- Basic arithmetic operations (per precision) ------------------------------

#ifdef BLIS_OPTIMIZE_BFLOAT_AS_FLOAT

#define bli_bmul( a, b )                  bli_smul(             a,             b  )
#define bli_bdiv( a, b )                  bli_sdiv(             a,             b  )
#define bli_badd( a, b )                  bli_sadd(             a,             b  )
#define bli_bsub( a, b )                  bli_ssub(             a,             b  )
#define bli_bneg( a )                     bli_sneg(             a                 )
#define bli_bsqrt( a )                    bli_ssqrt(            a                 )
#define bli_bhypot( a, b )                bli_shypot(            a,             b  )

#else

#define bli_bmul( a, b )      bli_sbcast( bli_smul(  bli_bscast(a), bli_bscast(b) ) )
#define bli_bdiv( a, b )      bli_sbcast( bli_sdiv(  bli_bscast(a), bli_bscast(b) ) )
#define bli_badd( a, b )      bli_sbcast( bli_sadd(  bli_bscast(a), bli_bscast(b) ) )
#define bli_bsub( a, b )      bli_sbcast( bli_ssub(  bli_bscast(a), bli_bscast(b) ) )
#define bli_bneg( a )         bli_sbcast( bli_sneg(  bli_bscast(a)                ) )
#define bli_bsqrt( a )        bli_sbcast( bli_ssqrt( bli_bscast(a)                ) )
#define bli_bhypot( a, b )    bli_sbcast( bli_shypot( bli_bscast(a), bli_bscast(b) ) )

#endif

#define bli_smul( a, b )       (a) * (b)
#define bli_sdiv( a, b )       (a) / (b)
#define bli_sadd( a, b )       (a) + (b)
#define bli_ssub( a, b )       (a) - (b)
#define bli_sneg( a )          -(a)
#define bli_ssqrt( a )         sqrtf(a)
#define bli_shypot( a, b )     hypotf(a,b)

#define bli_dmul( a, b )       (a) * (b)
#define bli_ddiv( a, b )       (a) / (b)
#define bli_dadd( a, b )       (a) + (b)
#define bli_dsub( a, b )       (a) - (b)
#define bli_dneg( a )          -(a)
#define bli_dsqrt( a )         sqrt(a)
#define bli_dhypot( a, b )     hypot(a,b)

// -- Basic compare operations (per precision) ---------------------------------

#ifdef BLIS_OPTIMIZE_BFLOAT_AS_FLOAT

#define bli_beq( a, b )                  bli_seq(            a,             b  )
#define bli_blt( a, b )                  bli_slt(            a,             b  )
#define bli_ble( a, b )                  bli_sle(            a,             b  )
#define bli_bgt( a, b )                  bli_sgt(            a,             b  )
#define bli_bge( a, b )                  bli_sge(            a,             b  )

#else

#define bli_beq( a, b )      bli_sbcast( bli_seq( bli_bscast(a), bli_bscast(b) ) )
#define bli_blt( a, b )      bli_sbcast( bli_slt( bli_bscast(a), bli_bscast(b) ) )
#define bli_ble( a, b )      bli_sbcast( bli_sle( bli_bscast(a), bli_bscast(b) ) )
#define bli_bgt( a, b )      bli_sbcast( bli_sgt( bli_bscast(a), bli_bscast(b) ) )
#define bli_bge( a, b )      bli_sbcast( bli_sge( bli_bscast(a), bli_bscast(b) ) )

#endif

#define bli_seq( a, b )  ( a == b )
#define bli_slt( a, b )  ( a <  b )
#define bli_sle( a, b )  ( a <= b )
#define bli_sgt( a, b )  ( a >  b )
#define bli_sge( a, b )  ( a >= b )

#define bli_deq( a, b )  ( a == b )
#define bli_dlt( a, b )  ( a <  b )
#define bli_dle( a, b )  ( a <= b )
#define bli_dgt( a, b )  ( a >  b )
#define bli_dge( a, b )  ( a >= b )

#define bli_ieq( a, b )  ( a == b )
#define bli_ilt( a, b )  ( a <  b )
#define bli_ile( a, b )  ( a <= b )
#define bli_igt( a, b )  ( a >  b )
#define bli_ige( a, b )  ( a >= b )

// -- Min/max/abs/etc. operations (per precision) ------------------------------

#ifdef BLIS_OPTIMIZE_BFLOAT_AS_FLOAT

#define bli_bmin( a, b )                   bli_smin(               a,             b  )
#define bli_bmax( a, b )                   bli_smax(               a,             b  )
#define bli_babs( a )                      bli_sabs(               a                 )
#define bli_bminabs( a, b )                bli_sminabs(            a              b  )
#define bli_bmaxabs( a, b )                bli_smaxabs(            a              b  )
#define bli_bcopysign( a, b )            ( bli_slt(            b , bli_szero ) \
                                           ? bli_sneg( bli_sabs(            a  ) ) \
                                           :           bli_sabs(            a  )   )

#else

#define bli_bmin( a, b )       bli_sbcast(    bli_smin( bli_bscast(a), bli_bscast(b) ) )
#define bli_bmax( a, b )       bli_sbcast(    bli_smax( bli_bscast(a), bli_bscast(b) ) )
#define bli_babs( a )          bli_sbcast(    bli_sabs( bli_bscast(a)                ) )
#define bli_bminabs( a, b )    bli_sbcast( bli_sminabs( bli_bscast(a), bli_bscast(b) ) )
#define bli_bmaxabs( a, b )    bli_sbcast( bli_smaxabs( bli_bscast(a), bli_bscast(b) ) )
#define bli_bcopysign( a, b )  bli_sbcast( bli_slt( bli_bscast(b), bli_szero ) \
                                           ? bli_sneg( bli_sabs( bli_bscast(a) ) ) \
                                           :           bli_sabs( bli_bscast(a) )   )

#endif

#define bli_smin( a, b )       ( bli_slt( a, b ) ? a : b )
#define bli_smax( a, b )       ( bli_sgt( a, b ) ? a : b )
//#define bli_sabs( a )          ( bli_slt( a, PASTEMAC(s,zero) ) ? -(a) : a )
#define bli_sabs( a )          ( fabsf(a) )
#define bli_sminabs( a, b )    bli_smin( bli_sabs( a ), bli_sabs( b ) )
#define bli_smaxabs( a, b )    bli_smax( bli_sabs( a ), bli_sabs( b ) )
#define bli_scopysign( a, b )  ( copysignf( a, b ) ) \

#define bli_dmin( a, b )       ( bli_dlt( a, b ) ? a : b )
#define bli_dmax( a, b )       ( bli_dgt( a, b ) ? a : b )
//#define bli_dabs( a )          ( bli_dlt( a, PASTEMAC(d,zero) ) ? -(a) : a )
#define bli_dabs( a )          ( fabs(a) )
#define bli_dminabs( a, b )    bli_dmin( bli_dabs( a ), bli_dabs( b ) )
#define bli_dmaxabs( a, b )    bli_dmax( bli_dabs( a ), bli_dabs( b ) )
#define bli_dcopysign( a, b )  ( copysign( a, b ) ) \

// -- Infinity/NaN check (per precision) ---------------------------------------

#ifdef BLIS_OPTIMIZE_BFLOAT_AS_FLOAT

#define bli_bisinf( a )        bli_sisinf(            a  )
#define bli_bisnan( a )        bli_sisnan(            a  )

#else

#define bli_bisinf( a )        bli_sisinf( bli_bscast(a) )
#define bli_bisnan( a )        bli_sisnan( bli_bscast(a) )

#endif

#define bli_sisinf( a )        isinf( a )
#define bli_sisnan( a )        isnan( a )

#define bli_disinf( a )        isinf( a )
#define bli_disnan( a )        isnan( a )

// -- Randomization operations (per precision) ---------------------------------

#define bli_brand              bli_dbcast( bli_rand() )
#define bli_srand              bli_dscast( bli_rand() )
#define bli_drand              bli_ddcast( bli_rand() )

// Randomize a real number on the interval [-1.0,1.0] and return it as a double.
BLIS_INLINE double bli_rand( void )
{
	return ( ( ( double ) rand()         ) /
             ( ( double ) RAND_MAX / 2.0 )
           ) - 1.0;
}

#define bli_brandnp2           bli_dbcast( bli_randnp2s() )
#define bli_srandnp2           bli_dscast( bli_randnp2s() )
#define bli_drandnp2           bli_ddcast( bli_randnp2s() )

// Randomize a power of two on a narrow range and return it as a double.
BLIS_INLINE double bli_randnp2s( void )
{
	const double m_max  = 6.0;
	const double m_max2 = m_max + 2.0;
	double       t;
	double       r_val;

	// Compute a narrow-range power of two.
	//
	// For the purposes of commentary, we'll assume that m_max = 4. This
	// represents the largest power of two we will use to generate the
	// random numbers.

	do
	{
		// Generate a random real number t on the interval: [0.0, 6.0].
		t = ( ( double ) rand() / ( double ) RAND_MAX ) * m_max2;

		// Transform the interval into the set of integers, {0,1,2,3,4,5}.
		// Note that 6 is prohibited by the loop guard below.
		t = floor( t );
	}
	// If t is ever equal to m_max2, we re-randomize. The guard against
	// m_max2 < t is for sanity and shouldn't happen, unless perhaps there
	// is weirdness in the typecasting to double when computing t above.
	while ( m_max2 <= t );

	// Map values of t == 0 to a final value of 0.
	if ( t == 0.0 ) r_val = 0.0;
	else
	{
		// This case handles values of t = {1,2,3,4,5}.

		// Compute r_val = 2^s where s = -(t-1) = {-4,-3,-2,-1,0}.
		r_val = pow( 2.0, -(t - 1.0) );

		// Compute a random number to determine the sign of the final
		// result.
		const double s_val = PASTEMAC(d,rand);

		// If our sign value is negative, our random power of two will
		// be negative.
		if ( s_val < 0.0 ) r_val = -r_val;
	}

	// r_val = 0, or +/-{2^0, 2^-1, 2^-2, 2^-3, 2^-4}.
	return r_val;
}



#endif

