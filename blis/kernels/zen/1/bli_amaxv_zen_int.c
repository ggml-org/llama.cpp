/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2016 - 2018 - 2019, Advanced Micro Devices, Inc.
   Copyright (C) 2018, The University of Texas at Austin

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

#include "immintrin.h"
#include "blis.h"


/* Union data structure to access AVX registers
   One 256-bit AVX register holds 8 SP elements. */
typedef union
{
	__m256  v;
	float   f[8] __attribute__((aligned(64)));
} v8sf_t;

typedef union
{
	__m128  v;
	float   f[4];
} v4sf_t;

/* Union data structure to access AVX registers
   One 256-bit AVX register holds 4 DP elements. */
typedef union
{
	__m256d v;
	double  d[4] __attribute__((aligned(64)));
}v4df_t;

typedef union
{
	__m128d v;
	double  d[2];
}v2dd_t;

// return a mask which indicates either:
// - v1 > v2
// - v1 is NaN and v2 is not
// assumes that idx(v1) > idx(v2)
// all "OQ" comparisons false if either operand NaN
#define CMP256( dt, v1, v2 ) \
	_mm256_or_p##dt( _mm256_cmp_p##dt( v1, v2, _CMP_GT_OQ ),                        /* v1 > v2  ||     */ \
	                 _mm256_andnot_p##dt( _mm256_cmp_p##dt( v2, v2, _CMP_UNORD_Q ), /* ( !isnan(v2) && */ \
	                                      _mm256_cmp_p##dt( v1, v1, _CMP_UNORD_Q )  /*    isnan(v1) )  */ \
	                                    ) \
	               );

// return a mask which indicates either:
// - v1 > v2
// - v1 is NaN and v2 is not
// - v1 == v2 (maybe == NaN) and i1 < i2
// all "OQ" comparisons false if either operand NaN
#define CMP128( dt, v1, v2, i1, i2 ) \
	_mm_or_p##dt( _mm_or_p##dt( _mm_cmp_p##dt( v1, v2, _CMP_GT_OQ ),                      /* ( v1 > v2 ||           */ \
	                            _mm_andnot_p##dt( _mm_cmp_p##dt( v2, v2, _CMP_UNORD_Q ),  /*   ( !isnan(v2) &&      */ \
	                                              _mm_cmp_p##dt( v1, v1, _CMP_UNORD_Q )   /*      isnan(v1) ) ) ||  */ \
	                                            ) \
	                          ), \
	              _mm_and_p##dt( _mm_or_p##dt( _mm_cmp_p##dt( v1, v2, _CMP_EQ_OQ ),                  /* ( ( v1 == v2 ||        */ \
	                                           _mm_and_p##dt( _mm_cmp_p##dt( v1, v1, _CMP_UNORD_Q ), /*     ( isnan(v1) &&     */ \
	                                                          _mm_cmp_p##dt( v2, v2, _CMP_UNORD_Q )  /*       isnan(v2) ) ) && */ \
	                                                        ) \
	                                         ), \
	                             _mm_cmp_p##dt( i1, i2, _CMP_LT_OQ )                                 /*   i1 < i2 )            */ \
	                           ) \
	            );

// -----------------------------------------------------------------------------

void bli_samaxv_zen_int
     (
             dim_t   n,
       const void*   x0, inc_t incx,
             dim_t*  index,
       const cntx_t* cntx
     )
{
	const float* restrict x = x0;

	const float* restrict minus_one = PASTEMAC(s,m1);
	const dim_t* restrict zero_i    = PASTEMAC(i,0);

	float   chi1_r;
	//float   chi1_i;
	float   abs_chi1;
	float   abs_chi1_max;
	dim_t   i_max_l;
	dim_t   i;

	/* If the vector length is zero, return early. This directly emulates
	   the behavior of netlib BLAS's i?amax() routines. */
	if ( bli_zero_dim1( n ) )
	{
		PASTEMAC(i,copys)( *zero_i, *index );
		return;
	}

	/* Initialize the index of the maximum absolute value to zero. */
	PASTEMAC(i,copys)( *zero_i, i_max_l );

	/* Initialize the maximum absolute value search candidate with
	   -1, which is guaranteed to be less than all values we will
	   compute. */
	PASTEMAC(s,copys)( *minus_one, abs_chi1_max );

	// For non-unit strides, or very small vector lengths, compute with
	// scalar code.
	if ( incx != 1 || n < 8 )
	{
		for ( i = 0; i < n; ++i )
		{
			const float* restrict chi1 = x + (i  )*incx;

			/* Get the real and imaginary components of chi1. */
			chi1_r = *chi1;

			/* Replace chi1_r and chi1_i with their absolute values. */
			chi1_r = fabsf( chi1_r );

			/* Add the real and imaginary absolute values together. */
			abs_chi1 = chi1_r;

			/* If the absolute value of the current element exceeds that of
			   the previous largest, save it and its index. If NaN is
			   encountered, then treat it the same as if it were a valid
			   value that was smaller than any previously seen. This
			   behavior mimics that of LAPACK's i?amax(). */
			if ( abs_chi1_max < abs_chi1 || ( isnan( abs_chi1 ) && !isnan( abs_chi1_max ) ) )
			{
				abs_chi1_max = abs_chi1;
				i_max_l      = i;
			}
		}
	}
	else
	{
		dim_t  n_iter, n_left;
		dim_t  num_vec_elements = 8;
		v8sf_t x_vec, max_vec, maxInx_vec, mask_vec;
		v8sf_t idx_vec, inc_vec;
		v8sf_t sign_mask;

		v4sf_t max_vec_lo, max_vec_hi, mask_vec_lo;
		v4sf_t maxInx_vec_lo, maxInx_vec_hi;

		n_iter = n / num_vec_elements;
		n_left = n % num_vec_elements;

		idx_vec.v    = _mm256_set_ps( 7, 6, 5, 4, 3, 2, 1, 0 );
		inc_vec.v    = _mm256_set1_ps( 8 );
		max_vec.v    = _mm256_set1_ps( -1 );
		maxInx_vec.v = _mm256_setzero_ps();
		sign_mask.v  = _mm256_set1_ps( -0.f );

		for ( i = 0; i < n_iter; ++i )
		{
			x_vec.v      = _mm256_loadu_ps( x );

			// Get the absolute value of the vector element.
			x_vec.v      = _mm256_andnot_ps( sign_mask.v, x_vec.v );

			mask_vec.v   = CMP256( s, x_vec.v, max_vec.v );

			max_vec.v    = _mm256_blendv_ps( max_vec.v, x_vec.v, mask_vec.v );
			maxInx_vec.v = _mm256_blendv_ps( maxInx_vec.v, idx_vec.v, mask_vec.v );

			idx_vec.v += inc_vec.v;
			x         += num_vec_elements;
		}

		max_vec_lo.v    = _mm256_extractf128_ps( max_vec.v, 0 );
		max_vec_hi.v    = _mm256_extractf128_ps( max_vec.v, 1 );
		maxInx_vec_lo.v = _mm256_extractf128_ps( maxInx_vec.v, 0 );
		maxInx_vec_hi.v = _mm256_extractf128_ps( maxInx_vec.v, 1 );

		mask_vec_lo.v = CMP128( s, max_vec_hi.v, max_vec_lo.v, maxInx_vec_hi.v, maxInx_vec_lo.v );

		max_vec_lo.v    = _mm_blendv_ps( max_vec_lo.v, max_vec_hi.v, mask_vec_lo.v );
		maxInx_vec_lo.v = _mm_blendv_ps( maxInx_vec_lo.v, maxInx_vec_hi.v, mask_vec_lo.v );

		max_vec_hi.v    = _mm_permute_ps( max_vec_lo.v, 14 );
		maxInx_vec_hi.v = _mm_permute_ps( maxInx_vec_lo.v, 14 );

		mask_vec_lo.v = CMP128( s, max_vec_hi.v, max_vec_lo.v, maxInx_vec_hi.v, maxInx_vec_lo.v );

		max_vec_lo.v    = _mm_blendv_ps( max_vec_lo.v, max_vec_hi.v, mask_vec_lo.v );
		maxInx_vec_lo.v = _mm_blendv_ps( maxInx_vec_lo.v, maxInx_vec_hi.v, mask_vec_lo.v );

		max_vec_hi.v    = _mm_permute_ps( max_vec_lo.v, 1 );
		maxInx_vec_hi.v = _mm_permute_ps( maxInx_vec_lo.v, 1 );

		mask_vec_lo.v = CMP128( s, max_vec_hi.v, max_vec_lo.v, maxInx_vec_hi.v, maxInx_vec_lo.v );

		max_vec_lo.v    = _mm_blendv_ps( max_vec_lo.v, max_vec_hi.v, mask_vec_lo.v );
		maxInx_vec_lo.v = _mm_blendv_ps( maxInx_vec_lo.v, maxInx_vec_hi.v, mask_vec_lo.v );

		abs_chi1_max = max_vec_lo.f[0];
		i_max_l      = maxInx_vec_lo.f[0];

		for ( i = n - n_left; i < n; i++ )
		{
			const float* restrict chi1 = x;

			/* Get the real and imaginary components of chi1. */
			chi1_r = *chi1;

			/* Replace chi1_r and chi1_i with their absolute values. */
			abs_chi1 = fabsf( chi1_r );

			/* If the absolute value of the current element exceeds that of
			   the previous largest, save it and its index. If NaN is
			   encountered, then treat it the same as if it were a valid
			   value that was smaller than any previously seen. This
			   behavior mimics that of LAPACK's i?amax(). */
			if ( abs_chi1_max < abs_chi1 || ( isnan( abs_chi1 ) && !isnan( abs_chi1_max ) ) )
			{
				abs_chi1_max = abs_chi1;
				i_max_l      = i;
			}

			x += 1;
		}
	}

	// Issue vzeroupper instruction to clear upper lanes of ymm registers.
	// This avoids a performance penalty caused by false dependencies when
	// transitioning from from AVX to SSE instructions (which may occur
	// later, especially if BLIS is compiled with -mfpmath=sse).
	_mm256_zeroupper();

	/* Store final index to output variable. */
	*index = i_max_l;
}

// -----------------------------------------------------------------------------

void bli_damaxv_zen_int
     (
             dim_t   n,
       const void*   x0, inc_t incx,
             dim_t*  index,
       const cntx_t* cntx
     )
{
	const double* restrict x = x0;

	const double* restrict minus_one = PASTEMAC(d,m1);
	const dim_t*  restrict zero_i    = PASTEMAC(i,0);

	double  chi1_r;
	//double  chi1_i;
	double  abs_chi1;
	double  abs_chi1_max;
	dim_t   i_max_l;
	dim_t   i;

	/* If the vector length is zero, return early. This directly emulates
	   the behavior of netlib BLAS's i?amax() routines. */
	if ( bli_zero_dim1( n ) )
	{
		PASTEMAC(i,copys)( *zero_i, *index );
		return;
	}

	/* Initialize the index of the maximum absolute value to zero. */ \
	PASTEMAC(i,copys)( *zero_i, i_max_l );

	/* Initialize the maximum absolute value search candidate with
	   -1, which is guaranteed to be less than all values we will
	   compute. */
	PASTEMAC(d,copys)( *minus_one, abs_chi1_max );

	// For non-unit strides, or very small vector lengths, compute with
	// scalar code.
	if ( incx != 1 || n < 4 )
	{
		for ( i = 0; i < n; ++i )
		{
			const double* restrict chi1 = x + (i  )*incx;

			/* Get the real and imaginary components of chi1. */
			chi1_r = *chi1;

			/* Replace chi1_r and chi1_i with their absolute values. */
			chi1_r = fabs( chi1_r );

			/* Add the real and imaginary absolute values together. */
			abs_chi1 = chi1_r;

			/* If the absolute value of the current element exceeds that of
			   the previous largest, save it and its index. If NaN is
			   encountered, then treat it the same as if it were a valid
			   value that was smaller than any previously seen. This
			   behavior mimics that of LAPACK's i?amax(). */
			if ( abs_chi1_max < abs_chi1 || ( isnan( abs_chi1 ) && !isnan( abs_chi1_max ) ) )
			{
				abs_chi1_max = abs_chi1;
				i_max_l      = i;
			}
		}
	}
	else
	{
		dim_t  n_iter, n_left;
		dim_t  num_vec_elements = 4;
		v4df_t x_vec, max_vec, maxInx_vec, mask_vec;
		v4df_t idx_vec, inc_vec;
		v4df_t sign_mask;

		v2dd_t max_vec_lo, max_vec_hi, mask_vec_lo;
		v2dd_t maxInx_vec_lo, maxInx_vec_hi;

		n_iter = n / num_vec_elements;
		n_left = n % num_vec_elements;

		idx_vec.v    = _mm256_set_pd( 3, 2, 1, 0 );
		inc_vec.v    = _mm256_set1_pd( 4 );
		max_vec.v    = _mm256_set1_pd( -1 );
		maxInx_vec.v = _mm256_setzero_pd();
		sign_mask.v  = _mm256_set1_pd( -0.f );

		for ( i = 0; i < n_iter; ++i )
		{
			x_vec.v      = _mm256_loadu_pd( x );

			// Get the absolute value of the vector element.
			x_vec.v      = _mm256_andnot_pd( sign_mask.v, x_vec.v );

			mask_vec.v   = CMP256( d, x_vec.v, max_vec.v );

			max_vec.v    = _mm256_blendv_pd( max_vec.v, x_vec.v, mask_vec.v );
			maxInx_vec.v = _mm256_blendv_pd( maxInx_vec.v, idx_vec.v, mask_vec.v );

			idx_vec.v += inc_vec.v;
			x         += num_vec_elements;
		}

		max_vec_lo.v    = _mm256_extractf128_pd( max_vec.v, 0 );
		max_vec_hi.v    = _mm256_extractf128_pd( max_vec.v, 1 );
		maxInx_vec_lo.v = _mm256_extractf128_pd( maxInx_vec.v, 0 );
		maxInx_vec_hi.v = _mm256_extractf128_pd( maxInx_vec.v, 1 );

		mask_vec_lo.v = CMP128( d, max_vec_hi.v, max_vec_lo.v, maxInx_vec_hi.v, maxInx_vec_lo.v );

		max_vec_lo.v    = _mm_blendv_pd( max_vec_lo.v, max_vec_hi.v, mask_vec_lo.v );
		maxInx_vec_lo.v = _mm_blendv_pd( maxInx_vec_lo.v, maxInx_vec_hi.v, mask_vec_lo.v );

		max_vec_hi.v    = _mm_permute_pd( max_vec_lo.v, 1 );
		maxInx_vec_hi.v = _mm_permute_pd( maxInx_vec_lo.v, 1 );

		mask_vec_lo.v = CMP128( d, max_vec_hi.v, max_vec_lo.v, maxInx_vec_hi.v, maxInx_vec_lo.v );

		max_vec_lo.v    = _mm_blendv_pd( max_vec_lo.v, max_vec_hi.v, mask_vec_lo.v );
		maxInx_vec_lo.v = _mm_blendv_pd( maxInx_vec_lo.v, maxInx_vec_hi.v, mask_vec_lo.v );

		abs_chi1_max = max_vec_lo.d[0];
		i_max_l      = maxInx_vec_lo.d[0];

		for ( i = n - n_left; i < n; i++ )
		{
			const double* restrict chi1 = x;

			/* Get the real and imaginary components of chi1. */
			chi1_r = *chi1;

			/* Replace chi1_r and chi1_i with their absolute values. */
			abs_chi1 = fabs( chi1_r );

			/* If the absolute value of the current element exceeds that of
			   the previous largest, save it and its index. If NaN is
			   encountered, return the index of the first NaN. This
			   behavior mimics that of LAPACK's i?amax(). */
			if ( abs_chi1_max < abs_chi1 || ( isnan( abs_chi1 ) && !isnan( abs_chi1_max ) ) )
			{
				abs_chi1_max = abs_chi1;
				i_max_l      = i;
			}

			x += 1;
		}
	}

	// Issue vzeroupper instruction to clear upper lanes of ymm registers.
	// This avoids a performance penalty caused by false dependencies when
	// transitioning from from AVX to SSE instructions (which may occur
	// later, especially if BLIS is compiled with -mfpmath=sse).
	_mm256_zeroupper();

	/* Store final index to output variable. */
	*index = i_max_l;
}

// -----------------------------------------------------------------------------

#if 0
#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       dim_t    n, \
       ctype*   x, inc_t incx, \
       dim_t*   index, \
       cntx_t*  cntx  \
     ) \
{ \
	ctype_r* minus_one = PASTEMAC(chr,m1); \
	dim_t*   zero_i    = PASTEMAC(i,0); \
\
	ctype_r  chi1_r; \
	ctype_r  chi1_i; \
	ctype_r  abs_chi1; \
	ctype_r  abs_chi1_max; \
	dim_t    i; \
\
	/* Initialize the index of the maximum absolute value to zero. */ \
	PASTEMAC(i,copys)( zero_i, *index ); \
\
	/* If the vector length is zero, return early. This directly emulates
	   the behavior of netlib BLAS's i?amax() routines. */ \
	if ( bli_zero_dim1( n ) ) return; \
\
	/* Initialize the maximum absolute value search candidate with
	   -1, which is guaranteed to be less than all values we will
	   compute. */ \
	PASTEMAC(chr,copys)( *minus_one, abs_chi1_max ); \
\
	if ( incx == 1 ) \
	{ \
		for ( i = 0; i < n; ++i ) \
		{ \
			/* Get the real and imaginary components of chi1. */ \
			PASTEMAC(ch,chr,gets)( x[i], chi1_r, chi1_i ); \
\
			/* Replace chi1_r and chi1_i with their absolute values. */ \
			PASTEMAC(chr,abval2s)( chi1_r, chi1_r ); \
			PASTEMAC(chr,abval2s)( chi1_i, chi1_i ); \
\
			/* Add the real and imaginary absolute values together. */ \
			PASTEMAC(chr,set0s)( abs_chi1 ); \
			PASTEMAC(chr,adds)( chi1_r, abs_chi1 ); \
			PASTEMAC(chr,adds)( chi1_i, abs_chi1 ); \
\
			/* If the absolute value of the current element exceeds that of
			   the previous largest, save it and its index. If NaN is
			   encountered, then treat it the same as if it were a valid
			   value that was smaller than any previously seen. This
			   behavior mimics that of LAPACK's ?lange(). */ \
			if ( abs_chi1_max < abs_chi1 || bli_isnan( abs_chi1 ) ) \
			{ \
				abs_chi1_max = abs_chi1; \
				*index       = i; \
			} \
		} \
	} \
	else \
	{ \
		for ( i = 0; i < n; ++i ) \
		{ \
			ctype* chi1 = x + (i  )*incx; \
\
			/* Get the real and imaginary components of chi1. */ \
			PASTEMAC(ch,chr,gets)( *chi1, chi1_r, chi1_i ); \
\
			/* Replace chi1_r and chi1_i with their absolute values. */ \
			PASTEMAC(chr,abval2s)( chi1_r, chi1_r ); \
			PASTEMAC(chr,abval2s)( chi1_i, chi1_i ); \
\
			/* Add the real and imaginary absolute values together. */ \
			PASTEMAC(chr,set0s)( abs_chi1 ); \
			PASTEMAC(chr,adds)( chi1_r, abs_chi1 ); \
			PASTEMAC(chr,adds)( chi1_i, abs_chi1 ); \
\
			/* If the absolute value of the current element exceeds that of
			   the previous largest, save it and its index. If NaN is
			   encountered, then treat it the same as if it were a valid
			   value that was smaller than any previously seen. This
			   behavior mimics that of LAPACK's ?lange(). */ \
			if ( abs_chi1_max < abs_chi1 || bli_isnan( abs_chi1 ) ) \
			{ \
				abs_chi1_max = abs_chi1; \
				*index       = i; \
			} \
		} \
	} \
}
GENTFUNCR( scomplex, float,  c, s, amaxv_zen_int )
GENTFUNCR( dcomplex, double, z, d, amaxv_zen_int )
#endif
