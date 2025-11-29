/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2021, The University of Tokyo

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

// Supplimentary dynamic-size gemmsup.

#include "blis.h"
#include "assert.h"
#include <arm_neon.h>

#if defined(__clang__)
#define PRAGMA_NOUNROLL _Pragma("nounroll")
#define PRAGMA_UNROLL   _Pragma("unroll")
#elif defined(__GNUC__)
#define PRAGMA_NOUNROLL _Pragma("GCC unroll 1")
#define PRAGMA_UNROLL   _Pragma("GCC unroll 2")
#else
#define PRAGMA_NOUNROLL
#define PRAGMA_UNROLL
#endif

/*
 * As these kernels requires num. of vregs about half of the total 32,
 *  it should be all right to implement w/ intrinsics.
 *
 * c.f. https://www.youtube.com/watch?v=R2hQOVjRwVE .
 */
void bli_dgemmsup_rd_armv8a_int_2x8
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a, inc_t rs_a, inc_t cs_a,
       const void*      b, inc_t rs_b, inc_t cs_b,
       const void*      beta,
             void*      c, inc_t rs_c, inc_t cs_c,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
  assert( m0 <= 2 );
  assert( n0 <= 8 );

  const double *a_loc = a;
  const double *b_loc = b;
        double *c_loc = c;

  uint64_t k_mker = k0 / 2;
  uint64_t k_left = k0 % 2;
  uint64_t b_iszr = ( *(( double* )beta) == 0.0 );

  assert( cs_a == 1 );
  assert( rs_b == 1 );

  // Registers used to store a 2x8x2 block of C (summing the last dimension).
  // Total: 22 specified.
  float64x2_t vc_00, vc_01, vc_02, vc_03, vc_04, vc_05, vc_06, vc_07;
  float64x2_t vc_10, vc_11, vc_12, vc_13, vc_14, vc_15, vc_16, vc_17;
  float64x2_t va_0, va_1;
  float64x2_t vb_0, vb_1, vb_2, vb_3;

  vc_00 = (float64x2_t)vdupq_n_f64( 0 );
  vc_01 = (float64x2_t)vdupq_n_f64( 0 );
  vc_02 = (float64x2_t)vdupq_n_f64( 0 );
  vc_03 = (float64x2_t)vdupq_n_f64( 0 );
  vc_04 = (float64x2_t)vdupq_n_f64( 0 );
  vc_05 = (float64x2_t)vdupq_n_f64( 0 );
  vc_06 = (float64x2_t)vdupq_n_f64( 0 );
  vc_07 = (float64x2_t)vdupq_n_f64( 0 );
  vc_10 = (float64x2_t)vdupq_n_f64( 0 );
  vc_11 = (float64x2_t)vdupq_n_f64( 0 );
  vc_12 = (float64x2_t)vdupq_n_f64( 0 );
  vc_13 = (float64x2_t)vdupq_n_f64( 0 );
  vc_14 = (float64x2_t)vdupq_n_f64( 0 );
  vc_15 = (float64x2_t)vdupq_n_f64( 0 );
  vc_16 = (float64x2_t)vdupq_n_f64( 0 );
  vc_17 = (float64x2_t)vdupq_n_f64( 0 );

  PRAGMA_UNROLL
  for ( ; k_mker > 0; --k_mker )
  {
    // if ( m0 > 0 )
                  va_0 = vld1q_f64( a_loc + rs_a * 0 );
    if ( m0 > 1 ) va_1 = vld1q_f64( a_loc + rs_a * 1 );
    // if ( n0 > 0 )
                  vb_0 = vld1q_f64( b_loc + cs_b * 0 );
    if ( n0 > 1 ) vb_1 = vld1q_f64( b_loc + cs_b * 1 );
    if ( n0 > 2 ) vb_2 = vld1q_f64( b_loc + cs_b * 2 );
    if ( n0 > 3 ) vb_3 = vld1q_f64( b_loc + cs_b * 3 );

    vc_00 = vfmaq_f64( vc_00, va_0, vb_0 );
    vc_01 = vfmaq_f64( vc_01, va_0, vb_1 );
    vc_02 = vfmaq_f64( vc_02, va_0, vb_2 );
    vc_03 = vfmaq_f64( vc_03, va_0, vb_3 );
    if ( m0 > 1 )
    {
      vc_10 = vfmaq_f64( vc_10, va_1, vb_0 );
      vc_11 = vfmaq_f64( vc_11, va_1, vb_1 );
      vc_12 = vfmaq_f64( vc_12, va_1, vb_2 );
      vc_13 = vfmaq_f64( vc_13, va_1, vb_3 );
    }

    if ( n0 > 4 ) {
                    vb_0 = vld1q_f64( b_loc + cs_b * 4 );
      if ( n0 > 5 ) vb_1 = vld1q_f64( b_loc + cs_b * 5 );
      if ( n0 > 6 ) vb_2 = vld1q_f64( b_loc + cs_b * 6 );
      if ( n0 > 7 ) vb_3 = vld1q_f64( b_loc + cs_b * 7 );

      vc_04 = vfmaq_f64( vc_04, va_0, vb_0 );
      vc_05 = vfmaq_f64( vc_05, va_0, vb_1 );
      if ( n0 > 6 )
      {
        vc_06 = vfmaq_f64( vc_06, va_0, vb_2 );
        vc_07 = vfmaq_f64( vc_07, va_0, vb_3 );
      }
      if ( m0 > 1 )
      {
        vc_14 = vfmaq_f64( vc_14, va_1, vb_0 );
        vc_15 = vfmaq_f64( vc_15, va_1, vb_1 );
        if ( n0 > 6 )
        {
          vc_16 = vfmaq_f64( vc_16, va_1, vb_2 );
          vc_17 = vfmaq_f64( vc_17, va_1, vb_3 );
        }
      }
    }

    a_loc += 2;
    b_loc += 2;
  }

  // Pay no care for O(1) details.
  va_0 = (float64x2_t)vdupq_n_f64( 0 );
  va_1 = (float64x2_t)vdupq_n_f64( 0 );
  vb_0 = (float64x2_t)vdupq_n_f64( 0 );
  vb_1 = (float64x2_t)vdupq_n_f64( 0 );
  vb_2 = (float64x2_t)vdupq_n_f64( 0 );
  vb_3 = (float64x2_t)vdupq_n_f64( 0 );
  PRAGMA_NOUNROLL
  for ( ; k_left > 0; --k_left )
  {
    // if ( m0 > 0 )
                  va_0 = vld1q_lane_f64( a_loc + rs_a * 0, va_0, 0 );
    if ( m0 > 1 ) va_1 = vld1q_lane_f64( a_loc + rs_a * 1, va_1, 0 );
    // if ( n0 > 0 )
                  vb_0 = vld1q_lane_f64( b_loc + cs_b * 0, vb_0, 0 );
    if ( n0 > 1 ) vb_1 = vld1q_lane_f64( b_loc + cs_b * 1, vb_1, 0 );
    if ( n0 > 2 ) vb_2 = vld1q_lane_f64( b_loc + cs_b * 2, vb_2, 0 );
    if ( n0 > 3 ) vb_3 = vld1q_lane_f64( b_loc + cs_b * 3, vb_3, 0 );

    vc_00 = vfmaq_f64( vc_00, va_0, vb_0 );
    vc_01 = vfmaq_f64( vc_01, va_0, vb_1 );
    vc_02 = vfmaq_f64( vc_02, va_0, vb_2 );
    vc_03 = vfmaq_f64( vc_03, va_0, vb_3 );
    vc_10 = vfmaq_f64( vc_10, va_1, vb_0 );
    vc_11 = vfmaq_f64( vc_11, va_1, vb_1 );
    vc_12 = vfmaq_f64( vc_12, va_1, vb_2 );
    vc_13 = vfmaq_f64( vc_13, va_1, vb_3 );

    if ( n0 > 4 ) vb_0 = vld1q_lane_f64( b_loc + cs_b * 4, vb_0, 0 );
    if ( n0 > 5 ) vb_1 = vld1q_lane_f64( b_loc + cs_b * 5, vb_1, 0 );
    if ( n0 > 6 ) vb_2 = vld1q_lane_f64( b_loc + cs_b * 6, vb_2, 0 );
    if ( n0 > 7 ) vb_3 = vld1q_lane_f64( b_loc + cs_b * 7, vb_3, 0 );

    vc_04 = vfmaq_f64( vc_04, va_0, vb_0 );
    vc_05 = vfmaq_f64( vc_05, va_0, vb_1 );
    vc_06 = vfmaq_f64( vc_06, va_0, vb_2 );
    vc_07 = vfmaq_f64( vc_07, va_0, vb_3 );
    vc_14 = vfmaq_f64( vc_14, va_1, vb_0 );
    vc_15 = vfmaq_f64( vc_15, va_1, vb_1 );
    vc_16 = vfmaq_f64( vc_16, va_1, vb_2 );
    vc_17 = vfmaq_f64( vc_17, va_1, vb_3 );

    a_loc += 1;
    b_loc += 1;
  }

  // Load alpha and beta.
  // Note that here vb is used for alpha, in contrast to other kernels.
  vb_0 = vld1q_dup_f64( alpha );
  va_0 = vld1q_dup_f64( beta );

  // Scale.
  vc_00 = vmulq_f64( vc_00, vb_0 );
  vc_01 = vmulq_f64( vc_01, vb_0 );
  vc_02 = vmulq_f64( vc_02, vb_0 );
  vc_03 = vmulq_f64( vc_03, vb_0 );
  vc_04 = vmulq_f64( vc_04, vb_0 );
  vc_05 = vmulq_f64( vc_05, vb_0 );
  vc_06 = vmulq_f64( vc_06, vb_0 );
  vc_07 = vmulq_f64( vc_07, vb_0 );
  vc_10 = vmulq_f64( vc_10, vb_0 );
  vc_11 = vmulq_f64( vc_11, vb_0 );
  vc_12 = vmulq_f64( vc_12, vb_0 );
  vc_13 = vmulq_f64( vc_13, vb_0 );
  vc_14 = vmulq_f64( vc_14, vb_0 );
  vc_15 = vmulq_f64( vc_15, vb_0 );
  vc_16 = vmulq_f64( vc_16, vb_0 );
  vc_17 = vmulq_f64( vc_17, vb_0 );

  if ( cs_c == 1 )
  {
    // Row-storage.
    vc_00 = vpaddq_f64( vc_00, vc_01 );
    vc_02 = vpaddq_f64( vc_02, vc_03 );
    vc_04 = vpaddq_f64( vc_04, vc_05 );
    vc_06 = vpaddq_f64( vc_06, vc_07 );
    vc_10 = vpaddq_f64( vc_10, vc_11 );
    vc_12 = vpaddq_f64( vc_12, vc_13 );
    vc_14 = vpaddq_f64( vc_14, vc_15 );
    vc_16 = vpaddq_f64( vc_16, vc_17 );

    if      ( n0 > 1 ) vb_0 = vld1q_f64     ( c_loc + 0 * rs_c + 0 );
    else if ( n0 > 0 ) vb_0 = vld1q_lane_f64( c_loc + 0 * rs_c + 0, vb_0, 0 );
    if      ( n0 > 3 ) vb_1 = vld1q_f64     ( c_loc + 0 * rs_c + 2 );
    else if ( n0 > 2 ) vb_1 = vld1q_lane_f64( c_loc + 0 * rs_c + 2, vb_1, 0 );
    if      ( n0 > 5 ) vb_2 = vld1q_f64     ( c_loc + 0 * rs_c + 4 );
    else if ( n0 > 4 ) vb_2 = vld1q_lane_f64( c_loc + 0 * rs_c + 4, vb_2, 0 );
    if      ( n0 > 7 ) vb_3 = vld1q_f64     ( c_loc + 0 * rs_c + 6 );
    else if ( n0 > 6 ) vb_3 = vld1q_lane_f64( c_loc + 0 * rs_c + 6, vb_3, 0 );
    if ( !b_iszr )
    {
      vc_00 = vfmaq_f64( vc_00, va_0, vb_0 );
      vc_02 = vfmaq_f64( vc_02, va_0, vb_1 );
      vc_04 = vfmaq_f64( vc_04, va_0, vb_2 );
      vc_06 = vfmaq_f64( vc_06, va_0, vb_3 );
    }
    if      ( n0 > 1 ) vst1q_f64     ( c_loc + 0 * rs_c + 0, vc_00 );
    else if ( n0 > 0 ) vst1q_lane_f64( c_loc + 0 * rs_c + 0, vc_00, 0 );
    if      ( n0 > 3 ) vst1q_f64     ( c_loc + 0 * rs_c + 2, vc_02 );
    else if ( n0 > 2 ) vst1q_lane_f64( c_loc + 0 * rs_c + 2, vc_02, 0 );
    if      ( n0 > 5 ) vst1q_f64     ( c_loc + 0 * rs_c + 4, vc_04 );
    else if ( n0 > 4 ) vst1q_lane_f64( c_loc + 0 * rs_c + 4, vc_04, 0 );
    if      ( n0 > 7 ) vst1q_f64     ( c_loc + 0 * rs_c + 6, vc_06 );
    else if ( n0 > 6 ) vst1q_lane_f64( c_loc + 0 * rs_c + 6, vc_06, 0 );

    if ( m0 > 1 )
    {
      if      ( n0 > 1 ) vb_0 = vld1q_f64     ( c_loc + 1 * rs_c + 0 );
      else if ( n0 > 0 ) vb_0 = vld1q_lane_f64( c_loc + 1 * rs_c + 0, vb_0, 0 );
      if      ( n0 > 3 ) vb_1 = vld1q_f64     ( c_loc + 1 * rs_c + 2 );
      else if ( n0 > 2 ) vb_1 = vld1q_lane_f64( c_loc + 1 * rs_c + 2, vb_1, 0 );
      if      ( n0 > 5 ) vb_2 = vld1q_f64     ( c_loc + 1 * rs_c + 4 );
      else if ( n0 > 4 ) vb_2 = vld1q_lane_f64( c_loc + 1 * rs_c + 4, vb_2, 0 );
      if      ( n0 > 7 ) vb_3 = vld1q_f64     ( c_loc + 1 * rs_c + 6 );
      else if ( n0 > 6 ) vb_3 = vld1q_lane_f64( c_loc + 1 * rs_c + 6, vb_3, 0 );
      if ( !b_iszr )
      {
        vc_10 = vfmaq_f64( vc_10, va_0, vb_0 );
        vc_12 = vfmaq_f64( vc_12, va_0, vb_1 );
        vc_14 = vfmaq_f64( vc_14, va_0, vb_2 );
        vc_16 = vfmaq_f64( vc_16, va_0, vb_3 );
      }
      if      ( n0 > 1 ) vst1q_f64     ( c_loc + 1 * rs_c + 0, vc_10 );
      else if ( n0 > 0 ) vst1q_lane_f64( c_loc + 1 * rs_c + 0, vc_10, 0 );
      if      ( n0 > 3 ) vst1q_f64     ( c_loc + 1 * rs_c + 2, vc_12 );
      else if ( n0 > 2 ) vst1q_lane_f64( c_loc + 1 * rs_c + 2, vc_12, 0 );
      if      ( n0 > 5 ) vst1q_f64     ( c_loc + 1 * rs_c + 4, vc_14 );
      else if ( n0 > 4 ) vst1q_lane_f64( c_loc + 1 * rs_c + 4, vc_14, 0 );
      if      ( n0 > 7 ) vst1q_f64     ( c_loc + 1 * rs_c + 6, vc_16 );
      else if ( n0 > 6 ) vst1q_lane_f64( c_loc + 1 * rs_c + 6, vc_16, 0 );
    }
  }
  else
  {
    // Column-storage.
    vc_00 = vpaddq_f64( vc_00, vc_10 );
    vc_01 = vpaddq_f64( vc_01, vc_11 );
    vc_02 = vpaddq_f64( vc_02, vc_12 );
    vc_03 = vpaddq_f64( vc_03, vc_13 );
    vc_04 = vpaddq_f64( vc_04, vc_14 );
    vc_05 = vpaddq_f64( vc_05, vc_15 );
    vc_06 = vpaddq_f64( vc_06, vc_16 );
    vc_07 = vpaddq_f64( vc_07, vc_17 );

    if ( m0 > 1 )
    {
      // if ( n0 > 0 )
                    vb_0 = vld1q_f64( c_loc + 0 + 0 * cs_c );
      if ( n0 > 1 ) vb_1 = vld1q_f64( c_loc + 0 + 1 * cs_c );
      if ( n0 > 2 ) vb_2 = vld1q_f64( c_loc + 0 + 2 * cs_c );
      if ( n0 > 3 ) vb_3 = vld1q_f64( c_loc + 0 + 3 * cs_c );
      if ( !b_iszr )
      {
        vc_00 = vfmaq_f64( vc_00, va_0, vb_0 );
        vc_01 = vfmaq_f64( vc_01, va_0, vb_1 );
        vc_02 = vfmaq_f64( vc_02, va_0, vb_2 );
        vc_03 = vfmaq_f64( vc_03, va_0, vb_3 );
      }
                    vst1q_f64( c_loc + 0 + 0 * cs_c, vc_00 );
      if ( n0 > 1 ) vst1q_f64( c_loc + 0 + 1 * cs_c, vc_01 );
      if ( n0 > 2 ) vst1q_f64( c_loc + 0 + 2 * cs_c, vc_02 );
      if ( n0 > 3 ) vst1q_f64( c_loc + 0 + 3 * cs_c, vc_03 );

      if ( n0 > 4 ) vb_0 = vld1q_f64( c_loc + 0 + 4 * cs_c );
      if ( n0 > 5 ) vb_1 = vld1q_f64( c_loc + 0 + 5 * cs_c );
      if ( n0 > 6 ) vb_2 = vld1q_f64( c_loc + 0 + 6 * cs_c );
      if ( n0 > 7 ) vb_3 = vld1q_f64( c_loc + 0 + 7 * cs_c );
      if ( !b_iszr )
      {
        vc_04 = vfmaq_f64( vc_04, va_0, vb_0 );
        vc_05 = vfmaq_f64( vc_05, va_0, vb_1 );
        vc_06 = vfmaq_f64( vc_06, va_0, vb_2 );
        vc_07 = vfmaq_f64( vc_07, va_0, vb_3 );
      }
      if ( n0 > 4 ) vst1q_f64( c_loc + 0 + 4 * cs_c, vc_04 );
      if ( n0 > 5 ) vst1q_f64( c_loc + 0 + 5 * cs_c, vc_05 );
      if ( n0 > 6 ) vst1q_f64( c_loc + 0 + 6 * cs_c, vc_06 );
      if ( n0 > 7 ) vst1q_f64( c_loc + 0 + 7 * cs_c, vc_07 );
    }
    else
    {
      // if ( n0 > 0 )
                    vb_0 = vld1q_lane_f64( c_loc + 0 + 0 * cs_c, vb_0, 0 );
      if ( n0 > 1 ) vb_1 = vld1q_lane_f64( c_loc + 0 + 1 * cs_c, vb_1, 0 );
      if ( n0 > 2 ) vb_2 = vld1q_lane_f64( c_loc + 0 + 2 * cs_c, vb_2, 0 );
      if ( n0 > 3 ) vb_3 = vld1q_lane_f64( c_loc + 0 + 3 * cs_c, vb_3, 0 );
      if ( !b_iszr )
      {
        vc_00 = vfmaq_f64( vc_00, va_0, vb_0 );
        vc_01 = vfmaq_f64( vc_01, va_0, vb_1 );
        vc_02 = vfmaq_f64( vc_02, va_0, vb_2 );
        vc_03 = vfmaq_f64( vc_03, va_0, vb_3 );
      }
                    vst1q_lane_f64( c_loc + 0 + 0 * cs_c, vc_00, 0 );
      if ( n0 > 1 ) vst1q_lane_f64( c_loc + 0 + 1 * cs_c, vc_01, 0 );
      if ( n0 > 2 ) vst1q_lane_f64( c_loc + 0 + 2 * cs_c, vc_02, 0 );
      if ( n0 > 3 ) vst1q_lane_f64( c_loc + 0 + 3 * cs_c, vc_03, 0 );

      if ( n0 > 4 ) vb_0 = vld1q_lane_f64( c_loc + 0 + 4 * cs_c, vb_0, 0 );
      if ( n0 > 5 ) vb_1 = vld1q_lane_f64( c_loc + 0 + 5 * cs_c, vb_1, 0 );
      if ( n0 > 6 ) vb_2 = vld1q_lane_f64( c_loc + 0 + 6 * cs_c, vb_2, 0 );
      if ( n0 > 7 ) vb_3 = vld1q_lane_f64( c_loc + 0 + 7 * cs_c, vb_3, 0 );
      if ( !b_iszr )
      {
        vc_04 = vfmaq_f64( vc_04, va_0, vb_0 );
        vc_05 = vfmaq_f64( vc_05, va_0, vb_1 );
        vc_06 = vfmaq_f64( vc_06, va_0, vb_2 );
        vc_07 = vfmaq_f64( vc_07, va_0, vb_3 );
      }
      if ( n0 > 4 ) vst1q_lane_f64( c_loc + 0 + 4 * cs_c, vc_04, 0 );
      if ( n0 > 5 ) vst1q_lane_f64( c_loc + 0 + 5 * cs_c, vc_05, 0 );
      if ( n0 > 6 ) vst1q_lane_f64( c_loc + 0 + 6 * cs_c, vc_06, 0 );
      if ( n0 > 7 ) vst1q_lane_f64( c_loc + 0 + 7 * cs_c, vc_07, 0 );
    }
  }

}
