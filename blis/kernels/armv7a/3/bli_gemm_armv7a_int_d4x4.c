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

#include "blis.h"
#include "arm_neon.h"

void bli_sgemm_armv7a_int_4x4
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha0,
       const void*      a0,
       const void*      b0,
       const void*      beta0,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	const float* alpha = alpha0;
	const float* a     = a0;
	const float* b     = b0;
	const float* beta  = beta0;
	      float* c     = c0;

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint32_t k_iter = k / 4;
	uint32_t k_left = k % 4;
	uint32_t rs_c   = rs_c0;
	uint32_t cs_c   = cs_c0;
	uint32_t i;

    GEMM_UKR_SETUP_CT( s, 4, 4, false );

	const void* a_next = bli_auxinfo_next_a( data );
	const void* b_next = bli_auxinfo_next_b( data );

	float32x4_t alphav;
	alphav = vmovq_n_f32( *alpha );

	float32x4_t av1;
	float32x4_t av2;
	float32x4_t av3;
	float32x4_t av4;

	float32x4_t bv1;
	float32x4_t bv2;
	float32x4_t bv3;
	float32x4_t bv4;

	// Vector for column 0
	float32x4_t cv0;
	// Vector for column 1
	float32x4_t cv1;
	// Vector for column 2
	float32x4_t cv2;
	// Vector for column 3
	float32x4_t cv3;

	if ( *beta != 0.0F )
	{
		// Load column 0
		cv0 = vld1q_f32( c + 0*cs_c );

		// Load column 1
		cv1 = vld1q_f32( c + 1*cs_c );

		// Load column 2
		cv2 = vld1q_f32( c + 2*cs_c );

		// Load column 3
		cv3 = vld1q_f32( c + 3*cs_c );
	}
	else
	{
		cv0 = vmovq_n_f32( 0.0 );
		cv1 = vmovq_n_f32( 0.0 );
		cv2 = vmovq_n_f32( 0.0 );
		cv3 = vmovq_n_f32( 0.0 );
	}

	// Vector for accummulating column 0
	float32x4_t abv0;
	// Initialize vector to 0.0
	abv0 = vmovq_n_f32( 0.0 );

	// Vector for accummulating column 1
	float32x4_t abv1;
	// Initialize vector to 0.0
	abv1 = vmovq_n_f32( 0.0 );

	// Vector for accummulating column 2
	float32x4_t abv2;
	// Initialize vector to 0.0
	abv2 = vmovq_n_f32( 0.0 );

	// Vector for accummulating column 3
	float32x4_t abv3;
	// Initialize vector to 0.0
	abv3 = vmovq_n_f32( 0.0 );

	for ( i = 0; i < k_iter; ++i )
	{
		// Begin iter 0
			av1 = vld1q_f32( a );

		__builtin_prefetch( a + 224 );
		__builtin_prefetch( b + 224 );

		bv1 = vld1q_f32( b );

		abv0 = vmlaq_lane_f32( abv0, av1, vget_low_f32(bv1), 0 );
		abv1 = vmlaq_lane_f32( abv1, av1, vget_low_f32(bv1), 1 );
		abv2 = vmlaq_lane_f32( abv2, av1, vget_high_f32(bv1), 0 );
		abv3 = vmlaq_lane_f32( abv3, av1, vget_high_f32(bv1), 1 );


		av2 = vld1q_f32( a+4 );

		//__builtin_prefetch( a + 116 );
		//__builtin_prefetch( b + 116 );

		bv2 = vld1q_f32( b+4 );

		abv0 = vmlaq_lane_f32( abv0, av2, vget_low_f32(bv2), 0 );
		abv1 = vmlaq_lane_f32( abv1, av2, vget_low_f32(bv2), 1 );
		abv2 = vmlaq_lane_f32( abv2, av2, vget_high_f32(bv2), 0 );
		abv3 = vmlaq_lane_f32( abv3, av2, vget_high_f32(bv2), 1 );

		av3 = vld1q_f32( a+8 );

		//__builtin_prefetch( a + 120 );
		//__builtin_prefetch( b + 120 );

		bv3 = vld1q_f32( b+8 );

		abv0 = vmlaq_lane_f32( abv0, av3, vget_low_f32(bv3), 0 );
		abv1 = vmlaq_lane_f32( abv1, av3, vget_low_f32(bv3), 1 );
		abv2 = vmlaq_lane_f32( abv2, av3, vget_high_f32(bv3), 0 );
		abv3 = vmlaq_lane_f32( abv3, av3, vget_high_f32(bv3), 1 );


		av4 = vld1q_f32( a+12);

		//__builtin_prefetch( a + 124 );
		//__builtin_prefetch( b + 124 );

		bv4 = vld1q_f32( b+12);

		abv0 = vmlaq_lane_f32( abv0, av4, vget_low_f32(bv4), 0 );
		abv1 = vmlaq_lane_f32( abv1, av4, vget_low_f32(bv4), 1 );
		abv2 = vmlaq_lane_f32( abv2, av4, vget_high_f32(bv4), 0 );
		abv3 = vmlaq_lane_f32( abv3, av4, vget_high_f32(bv4), 1 );



		a += 16;
		b += 16;
	}

	for ( i = 0; i < k_left; ++i )
	{
		av1 = vld1q_f32( a );

		__builtin_prefetch( a + 112 );
		__builtin_prefetch( b + 112 );

		bv1 = vld1q_f32( b );

		abv0 = vmlaq_lane_f32( abv0, av1, vget_low_f32(bv1), 0 );
		abv1 = vmlaq_lane_f32( abv1, av1, vget_low_f32(bv1), 1 );
		abv2 = vmlaq_lane_f32( abv2, av1, vget_high_f32(bv1), 0 );
		abv3 = vmlaq_lane_f32( abv3, av1, vget_high_f32(bv1), 1 );

		a += 4;
		b += 4;
	}

	__builtin_prefetch( a_next );
	__builtin_prefetch( b_next );

	if ( *beta != 0.0F )
	{
		// Multiply C by beta and then accumulate alpha * A * B.
		cv0 = vmulq_n_f32( cv0, *beta );
		cv1 = vmulq_n_f32( cv1, *beta );
		cv2 = vmulq_n_f32( cv2, *beta );
		cv3 = vmulq_n_f32( cv3, *beta );

		cv0 = vmlaq_f32( cv0, abv0, alphav );
		cv1 = vmlaq_f32( cv1, abv1, alphav );
		cv2 = vmlaq_f32( cv2, abv2, alphav );
		cv3 = vmlaq_f32( cv3, abv3, alphav );
	}
	else
	{
		// Since beta = 0, skip straight to accumulating alpha * A * B.
		// Note: C (cv?) was initialized to zero above.
		cv0 = vmlaq_f32( cv0, abv0, alphav );
		cv1 = vmlaq_f32( cv1, abv1, alphav );
		cv2 = vmlaq_f32( cv2, abv2, alphav );
		cv3 = vmlaq_f32( cv3, abv3, alphav );
	}

	// Store column 0
	vst1q_f32( c + 0*cs_c, cv0 );
	// Store column 1
	vst1q_f32( c + 1*cs_c, cv1 );
	// Store column 2
	vst1q_f32( c + 2*cs_c, cv2 );
	// Store column 3
	vst1q_f32( c + 3*cs_c, cv3 );

    GEMM_UKR_FLUSH_CT( s );
}

void bli_dgemm_armv7a_int_4x4
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha_,
       const void*      a_,
       const void*      b_,
       const void*      beta_,
             void*      c_, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	const double* alpha = alpha_;
	const double* a     = a_;
	const double* b     = b_;
	const double* beta  = beta_;
	      double* c     = c_;

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	//uint32_t k_iter = k0 / 4;
	uint32_t k_left = k % 4;
	uint32_t rs_c   = rs_c0;
	uint32_t cs_c   = cs_c0;
	uint32_t i;

    GEMM_UKR_SETUP_CT_ANY( d, 4, 4, false );

	//void* a_next = bli_auxinfo_next_a( data );
	//void* b_next = bli_auxinfo_next_b( data );

	register double a0;
	register double a1;
	register double a2;
	register double a3;

	register double A0;
	register double A1;
	register double A2;
	register double A3;

	double b0, b1, b2, b3;
	double B0, B1, B2, B3;

	double  ab00, ab01, ab02, ab03;
	double  ab10, ab11, ab12, ab13;
	double  ab20, ab21, ab22, ab23;
	double  ab30, ab31, ab32, ab33;

	double* restrict c00, * restrict c01, * restrict c02, * restrict c03;
	double* restrict c10, * restrict c11, * restrict c12, * restrict c13;
	double* restrict c20, * restrict c21, * restrict c22, * restrict c23;
	double* restrict c30, * restrict c31, * restrict c32, * restrict c33;

	double* restrict ap = a;
        double* restrict bp = b;

	double* restrict Ap = a + 4;
        double* restrict Bp = b + 4;

	c00 = (c + 0*rs_c + 0*cs_c);
	c10 = (c + 1*rs_c + 0*cs_c);
	c20 = (c + 2*rs_c + 0*cs_c);
	c30 = (c + 3*rs_c + 0*cs_c);

	c01 = (c + 0*rs_c + 1*cs_c);
	c11 = (c + 1*rs_c + 1*cs_c);
	c21 = (c + 2*rs_c + 1*cs_c);
	c31 = (c + 3*rs_c + 1*cs_c);

	c02 = (c + 0*rs_c + 2*cs_c);
	c12 = (c + 1*rs_c + 2*cs_c);
	c22 = (c + 2*rs_c + 2*cs_c);
	c32 = (c + 3*rs_c + 2*cs_c);

	c03 = (c + 0*rs_c + 3*cs_c);
	c13 = (c + 1*rs_c + 3*cs_c);
	c23 = (c + 2*rs_c + 3*cs_c);
	c33 = (c + 3*rs_c + 3*cs_c);

	ab00 = 0.0; ab10 = 0.0; ab20 = 0.0; ab30 = 0.0;
	ab01 = 0.0; ab11 = 0.0; ab21 = 0.0; ab31 = 0.0;
	ab02 = 0.0; ab12 = 0.0; ab22 = 0.0; ab32 = 0.0;
	ab03 = 0.0; ab13 = 0.0; ab23 = 0.0; ab33 = 0.0;

	A0 = *(Ap + 0);
	A1 = *(Ap + 1);
	A2 = *(Ap + 2);
	A3 = *(Ap + 3);

	a0 = *(ap + 0);
	a1 = *(ap + 1);
 	a2 = *(ap + 2);

	B0 = *(Bp + 0);
	B1 = *(Bp + 1);
	B2 = *(Bp + 2);
	B3 = *(Bp + 3);

	b0 = *(bp + 0);
	b1 = *(bp + 1);
	b2 = *(bp + 2);

	double *Aplast = (Ap + 4*(k-k_left));

	//for ( i = 0; i < k_iter; ++i ) // Unroll by factor 4.
	for ( ; Ap != Aplast ; ) // Unroll by factor 4.
	{
		/* Prefetch */
		//__asm__ ("pld\t[%0],#100\n\t" : :"r"(Ap) : );
		__builtin_prefetch( ap + 112 );
		__builtin_prefetch( Ap + 112 );
		__builtin_prefetch( bp + 112 );
		__builtin_prefetch( Bp + 112 );
		// Iteration 0.
		ab00 += A0 * B0;
		a3 = *(ap + 3);
		ab10 += A1 * B0;
		b3 = *(bp + 3);
		ab20 += A2 * B0;
		ab30 += A3 * B0;

		ab01 += A0 * B1;
		ab11 += A1 * B1;
		B0 = *(Bp + 8);  // Prefetch.
		ab21 += A2 * B1;
		ab31 += A3 * B1;

		ab02 += A0 * B2;
		B1 = *(Bp + 9);
		ab12 += A1 * B2;
		ab22 += A2 * B2;
		ab32 += A3 * B2;
		B2 = *(Bp + 10);

		ab03 += A0 * B3;
		A0 = *(Ap + 8);  // Prefetch.
		ab13 += A1 * B3;
		A1 = *(Ap + 9);  // Prefetch.
		ab23 += A2 * B3;
		ab33 += A3 * B3;
		A2 = *(Ap + 10); // Prefetch.

		// Iteration 1.
		//__asm__ ("pld\t[%0],#200\n\t" : :"r"(Ap) : );
		ab00 += a0 * b0;
		ab10 += a1 * b0;
		A3 = *(Ap + 11); // Prefetch.
		ab20 += a2 * b0;
		ab30 += a3 * b0;
		B3 = *(Bp + 11);

		ab01 += a0 * b1;
		b0 = *(bp + 8);
		ab11 += a1 * b1;
		ab21 += a2 * b1;
		ab31 += a3 * b1;
		b1 = *(bp + 9);

		ab02 += a0 * b2;
		ab12 += a1 * b2;
		ab22 += a2 * b2;
		ab32 += a3 * b2;
		b2 = *(bp + 10);

		ab03 += a0 * b3;
		a0 = *(ap + 8);
		ab13 += a1 * b3;
		a1 = *(ap + 9);
		ab23 += a2 * b3;
		a2 = *(ap + 10);
		ab33 += a3 * b3;
		//a3 = *(ap + 11);

		ap += 8;
		Ap += 8;
		bp += 8;
		Bp += 8;

	}


	for ( i = 0; i < k_left; ++i )
	{
		a0 = *(ap + 0);
		a1 = *(ap + 1);
		a2 = *(ap + 2);
		a3 = *(ap + 3);

		b0 = *(bp + 0);
		b1 = *(bp + 1);
		b2 = *(bp + 2);
		b3 = *(bp + 3);

		ab00 += a0 * b0;
		ab10 += a1 * b0;
		ab20 += a2 * b0;
		ab30 += a3 * b0;

		ab01 += a0 * b1;
		ab11 += a1 * b1;
		ab21 += a2 * b1;
		ab31 += a3 * b1;

		ab02 += a0 * b2;
		ab12 += a1 * b2;
		ab22 += a2 * b2;
		ab32 += a3 * b2;

		ab03 += a0 * b3;
		ab13 += a1 * b3;
		ab23 += a2 * b3;
		ab33 += a3 * b3;

		ap += 4;
		bp += 4;
	}

    if ( *beta == 0.0 )
    {
    	*c00 = ab00 * *alpha;
    	*c10 = ab10 * *alpha;
    	*c20 = ab20 * *alpha;
    	*c30 = ab30 * *alpha;

    	*c01 = ab01 * *alpha;
    	*c11 = ab11 * *alpha;
    	*c21 = ab21 * *alpha;
    	*c31 = ab31 * *alpha;

    	*c02 = ab02 * *alpha;
    	*c12 = ab12 * *alpha;
    	*c22 = ab22 * *alpha;
    	*c32 = ab32 * *alpha;

    	*c03 = ab03 * *alpha;
    	*c13 = ab13 * *alpha;
    	*c23 = ab23 * *alpha;
    	*c33 = ab33 * *alpha;
    }
    else
    {
    	*c00 = *c00 * *beta;
    	*c10 = *c10 * *beta;
    	*c20 = *c20 * *beta;
    	*c30 = *c30 * *beta;

    	*c01 = *c01 * *beta;
    	*c11 = *c11 * *beta;
    	*c21 = *c21 * *beta;
    	*c31 = *c31 * *beta;

    	*c02 = *c02 * *beta;
    	*c12 = *c12 * *beta;
    	*c22 = *c22 * *beta;
    	*c32 = *c32 * *beta;

    	*c03 = *c03 * *beta;
    	*c13 = *c13 * *beta;
    	*c23 = *c23 * *beta;
    	*c33 = *c33 * *beta;

    	*c00 += ab00 * *alpha;
    	*c10 += ab10 * *alpha;
    	*c20 += ab20 * *alpha;
    	*c30 += ab30 * *alpha;

    	*c01 += ab01 * *alpha;
    	*c11 += ab11 * *alpha;
    	*c21 += ab21 * *alpha;
    	*c31 += ab31 * *alpha;

    	*c02 += ab02 * *alpha;
    	*c12 += ab12 * *alpha;
    	*c22 += ab22 * *alpha;
    	*c32 += ab32 * *alpha;

    	*c03 += ab03 * *alpha;
    	*c13 += ab13 * *alpha;
    	*c23 += ab23 * *alpha;
    	*c33 += ab33 * *alpha;
    }

    GEMM_UKR_FLUSH_CT( d );
}

