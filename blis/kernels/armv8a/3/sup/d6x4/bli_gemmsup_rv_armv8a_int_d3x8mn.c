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
void bli_dgemmsup_rv_armv8a_int_3x8mn
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a, inc_t cs_a,
       const void*      b0, inc_t rs_b, inc_t cs_b,
       const void*      beta,
             void*      c0, inc_t rs_c, inc_t cs_c,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
  // Unlike the rd case, this rv case does not impose restriction upon
  //  maximal m & n.

  const double *a_loc;
  const double *b_loc;
        double *c_loc;

  const double *b_in;
        double *c_in;

  dim_t n;
  dim_t k;
  uint64_t ps_a   = bli_auxinfo_ps_a( data );
  uint64_t ps_b   = bli_auxinfo_ps_b( data );
  uint64_t b_iszr = ( *(( double* )beta) == 0.0 );
  assert( cs_b == 1 );

  // Registers used to store a 3x8 block of C.
  float64x2_t vc_00, vc_01, vc_02, vc_03;
  float64x2_t vc_10, vc_11, vc_12, vc_13;
  float64x2_t vc_20, vc_21, vc_22, vc_23;
  float64x2_t va_0, va_1;
  float64x2_t vb_0, vb_1, vb_2, vb_3;

  PRAGMA_NOUNROLL
  for ( ; m0 > 0; m0 -= 3 )
  {
    n = n0;
    b_in = b0;
    c_in = c0;

    PRAGMA_NOUNROLL
    for ( ; n > 0; n -= 8 )
    {
      a_loc = a0;
      b_loc = b_in;
      c_loc = c_in;
      k = k0;

      vc_00 = (float64x2_t)vdupq_n_f64( 0 );
      vc_01 = (float64x2_t)vdupq_n_f64( 0 );
      vc_02 = (float64x2_t)vdupq_n_f64( 0 );
      vc_03 = (float64x2_t)vdupq_n_f64( 0 );
      vc_10 = (float64x2_t)vdupq_n_f64( 0 );
      vc_11 = (float64x2_t)vdupq_n_f64( 0 );
      vc_12 = (float64x2_t)vdupq_n_f64( 0 );
      vc_13 = (float64x2_t)vdupq_n_f64( 0 );
      vc_20 = (float64x2_t)vdupq_n_f64( 0 );
      vc_21 = (float64x2_t)vdupq_n_f64( 0 );
      vc_22 = (float64x2_t)vdupq_n_f64( 0 );
      vc_23 = (float64x2_t)vdupq_n_f64( 0 );

      PRAGMA_UNROLL
      for ( ; k > 0; --k )
      {
        // A columns.
        // if ( m0 > 0 )
                      va_0 = vld1q_lane_f64( a_loc + rs_a * 0, va_0, 0 );
        if ( m0 > 1 ) va_0 = vld1q_lane_f64( a_loc + rs_a * 1, va_0, 1 );
        if ( m0 > 2 ) va_1 = vld1q_lane_f64( a_loc + rs_a * 2, va_1, 0 );
        // B rows.
        if      ( n > 1 ) vb_0 = vld1q_f64     ( b_loc + 0 );
        else              vb_0 = vld1q_lane_f64( b_loc + 0, vb_0, 0 );
        if      ( n > 3 ) vb_1 = vld1q_f64     ( b_loc + 2 );
        else if ( n > 2 ) vb_1 = vld1q_lane_f64( b_loc + 2, vb_1, 0 );
        if      ( n > 5 ) vb_2 = vld1q_f64     ( b_loc + 4 );
        else if ( n > 4 ) vb_2 = vld1q_lane_f64( b_loc + 4, vb_2, 0 );
        if      ( n > 7 ) vb_3 = vld1q_f64     ( b_loc + 6 );
        else if ( n > 6 ) vb_3 = vld1q_lane_f64( b_loc + 6, vb_3, 0 );
        a_loc += cs_a;
        b_loc += rs_b;

        // if ( m0 > 0 )
        {
          vc_00 = vfmaq_laneq_f64( vc_00, vb_0, va_0, 0 );
          vc_01 = vfmaq_laneq_f64( vc_01, vb_1, va_0, 0 );
          vc_02 = vfmaq_laneq_f64( vc_02, vb_2, va_0, 0 );
          vc_03 = vfmaq_laneq_f64( vc_03, vb_3, va_0, 0 );
        }
        if ( m0 > 1 )
        {
          vc_10 = vfmaq_laneq_f64( vc_10, vb_0, va_0, 1 );
          vc_11 = vfmaq_laneq_f64( vc_11, vb_1, va_0, 1 );
          vc_12 = vfmaq_laneq_f64( vc_12, vb_2, va_0, 1 );
          vc_13 = vfmaq_laneq_f64( vc_13, vb_3, va_0, 1 );
        }
        if ( m0 > 2 )
        {
          vc_20 = vfmaq_laneq_f64( vc_20, vb_0, va_1, 0 );
          vc_21 = vfmaq_laneq_f64( vc_21, vb_1, va_1, 0 );
          vc_22 = vfmaq_laneq_f64( vc_22, vb_2, va_1, 0 );
          vc_23 = vfmaq_laneq_f64( vc_23, vb_3, va_1, 0 );
        }
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
      vc_10 = vmulq_f64( vc_10, vb_0 );
      vc_11 = vmulq_f64( vc_11, vb_0 );
      vc_12 = vmulq_f64( vc_12, vb_0 );
      vc_13 = vmulq_f64( vc_13, vb_0 );
      vc_20 = vmulq_f64( vc_20, vb_0 );
      vc_21 = vmulq_f64( vc_21, vb_0 );
      vc_22 = vmulq_f64( vc_22, vb_0 );
      vc_23 = vmulq_f64( vc_23, vb_0 );

      if ( cs_c == 1 )
      {
        // Store in rows.
        //
        // if ( m0 > 0 )
        {
          // Load.
          if      ( n > 1 ) vb_0 = vld1q_f64     ( c_loc + 0 * rs_c + 0 );
          else              vb_0 = vld1q_lane_f64( c_loc + 0 * rs_c + 0, vb_0, 0 );
          if      ( n > 3 ) vb_1 = vld1q_f64     ( c_loc + 0 * rs_c + 2 );
          else if ( n > 2 ) vb_1 = vld1q_lane_f64( c_loc + 0 * rs_c + 2, vb_1, 0 );
          if      ( n > 5 ) vb_2 = vld1q_f64     ( c_loc + 0 * rs_c + 4 );
          else if ( n > 4 ) vb_2 = vld1q_lane_f64( c_loc + 0 * rs_c + 4, vb_2, 0 );
          if      ( n > 7 ) vb_3 = vld1q_f64     ( c_loc + 0 * rs_c + 6 );
          else if ( n > 6 ) vb_3 = vld1q_lane_f64( c_loc + 0 * rs_c + 6, vb_3, 0 );

          // Scale.
          if ( !b_iszr )
          {
            vc_00 = vfmaq_f64( vc_00, vb_0, va_0 );
            vc_01 = vfmaq_f64( vc_01, vb_1, va_0 );
            vc_02 = vfmaq_f64( vc_02, vb_2, va_0 );
            vc_03 = vfmaq_f64( vc_03, vb_3, va_0 );
          }

          // Store.
          if      ( n > 1 ) vst1q_f64     ( c_loc + 0 * rs_c + 0, vc_00 );
          else              vst1q_lane_f64( c_loc + 0 * rs_c + 0, vc_00, 0 );
          if      ( n > 3 ) vst1q_f64     ( c_loc + 0 * rs_c + 2, vc_01 );
          else if ( n > 2 ) vst1q_lane_f64( c_loc + 0 * rs_c + 2, vc_01, 0 );
          if      ( n > 5 ) vst1q_f64     ( c_loc + 0 * rs_c + 4, vc_02 );
          else if ( n > 4 ) vst1q_lane_f64( c_loc + 0 * rs_c + 4, vc_02, 0 );
          if      ( n > 7 ) vst1q_f64     ( c_loc + 0 * rs_c + 6, vc_03 );
          else if ( n > 6 ) vst1q_lane_f64( c_loc + 0 * rs_c + 6, vc_03, 0 );
        }
        if ( m0 > 1 )
        {
          // Load.
          if      ( n > 1 ) vb_0 = vld1q_f64     ( c_loc + 1 * rs_c + 0 );
          else              vb_0 = vld1q_lane_f64( c_loc + 1 * rs_c + 0, vb_0, 0 );
          if      ( n > 3 ) vb_1 = vld1q_f64     ( c_loc + 1 * rs_c + 2 );
          else if ( n > 2 ) vb_1 = vld1q_lane_f64( c_loc + 1 * rs_c + 2, vb_1, 0 );
          if      ( n > 5 ) vb_2 = vld1q_f64     ( c_loc + 1 * rs_c + 4 );
          else if ( n > 4 ) vb_2 = vld1q_lane_f64( c_loc + 1 * rs_c + 4, vb_2, 0 );
          if      ( n > 7 ) vb_3 = vld1q_f64     ( c_loc + 1 * rs_c + 6 );
          else if ( n > 6 ) vb_3 = vld1q_lane_f64( c_loc + 1 * rs_c + 6, vb_3, 0 );

          // Scale.
          if ( !b_iszr )
          {
            vc_10 = vfmaq_f64( vc_10, vb_0, va_0 );
            vc_11 = vfmaq_f64( vc_11, vb_1, va_0 );
            vc_12 = vfmaq_f64( vc_12, vb_2, va_0 );
            vc_13 = vfmaq_f64( vc_13, vb_3, va_0 );
          }

          // Store.
          if      ( n > 1 ) vst1q_f64     ( c_loc + 1 * rs_c + 0, vc_10 );
          else              vst1q_lane_f64( c_loc + 1 * rs_c + 0, vc_10, 0 );
          if      ( n > 3 ) vst1q_f64     ( c_loc + 1 * rs_c + 2, vc_11 );
          else if ( n > 2 ) vst1q_lane_f64( c_loc + 1 * rs_c + 2, vc_11, 0 );
          if      ( n > 5 ) vst1q_f64     ( c_loc + 1 * rs_c + 4, vc_12 );
          else if ( n > 4 ) vst1q_lane_f64( c_loc + 1 * rs_c + 4, vc_12, 0 );
          if      ( n > 7 ) vst1q_f64     ( c_loc + 1 * rs_c + 6, vc_13 );
          else if ( n > 6 ) vst1q_lane_f64( c_loc + 1 * rs_c + 6, vc_13, 0 );
        }
        if ( m0 > 2 )
        {
          // Load.
          if      ( n > 1 ) vb_0 = vld1q_f64     ( c_loc + 2 * rs_c + 0 );
          else              vb_0 = vld1q_lane_f64( c_loc + 2 * rs_c + 0, vb_0, 0 );
          if      ( n > 3 ) vb_1 = vld1q_f64     ( c_loc + 2 * rs_c + 2 );
          else if ( n > 2 ) vb_1 = vld1q_lane_f64( c_loc + 2 * rs_c + 2, vb_1, 0 );
          if      ( n > 5 ) vb_2 = vld1q_f64     ( c_loc + 2 * rs_c + 4 );
          else if ( n > 4 ) vb_2 = vld1q_lane_f64( c_loc + 2 * rs_c + 4, vb_2, 0 );
          if      ( n > 7 ) vb_3 = vld1q_f64     ( c_loc + 2 * rs_c + 6 );
          else if ( n > 6 ) vb_3 = vld1q_lane_f64( c_loc + 2 * rs_c + 6, vb_3, 0 );

          // Scale.
          if ( !b_iszr )
          {
            vc_20 = vfmaq_f64( vc_20, vb_0, va_0 );
            vc_21 = vfmaq_f64( vc_21, vb_1, va_0 );
            vc_22 = vfmaq_f64( vc_22, vb_2, va_0 );
            vc_23 = vfmaq_f64( vc_23, vb_3, va_0 );
          }

          // Store.
          if      ( n > 1 ) vst1q_f64     ( c_loc + 2 * rs_c + 0, vc_20 );
          else              vst1q_lane_f64( c_loc + 2 * rs_c + 0, vc_20, 0 );
          if      ( n > 3 ) vst1q_f64     ( c_loc + 2 * rs_c + 2, vc_21 );
          else if ( n > 2 ) vst1q_lane_f64( c_loc + 2 * rs_c + 2, vc_21, 0 );
          if      ( n > 5 ) vst1q_f64     ( c_loc + 2 * rs_c + 4, vc_22 );
          else if ( n > 4 ) vst1q_lane_f64( c_loc + 2 * rs_c + 4, vc_22, 0 );
          if      ( n > 7 ) vst1q_f64     ( c_loc + 2 * rs_c + 6, vc_23 );
          else if ( n > 6 ) vst1q_lane_f64( c_loc + 2 * rs_c + 6, vc_23, 0 );
        }
      }
      else
      {
        // Store in columns.
        // No in-reg transpose here.
        //
        // if ( m0 > 0 )
        {
          // Load.
          if ( n > 0 ) vb_0 = vld1q_lane_f64( c_loc + 0 + 0 * cs_c, vb_0, 0 );
          if ( n > 1 ) vb_0 = vld1q_lane_f64( c_loc + 0 + 1 * cs_c, vb_0, 1 );
          if ( n > 2 ) vb_1 = vld1q_lane_f64( c_loc + 0 + 2 * cs_c, vb_1, 0 );
          if ( n > 3 ) vb_1 = vld1q_lane_f64( c_loc + 0 + 3 * cs_c, vb_1, 1 );
          if ( n > 4 ) vb_2 = vld1q_lane_f64( c_loc + 0 + 4 * cs_c, vb_2, 0 );
          if ( n > 5 ) vb_2 = vld1q_lane_f64( c_loc + 0 + 5 * cs_c, vb_2, 1 );
          if ( n > 6 ) vb_3 = vld1q_lane_f64( c_loc + 0 + 6 * cs_c, vb_3, 0 );
          if ( n > 7 ) vb_3 = vld1q_lane_f64( c_loc + 0 + 7 * cs_c, vb_3, 1 );

          // Scale.
          if ( !b_iszr )
          {
            vc_00 = vfmaq_f64( vc_00, vb_0, va_0 );
            vc_01 = vfmaq_f64( vc_01, vb_1, va_0 );
            vc_02 = vfmaq_f64( vc_02, vb_2, va_0 );
            vc_03 = vfmaq_f64( vc_03, vb_3, va_0 );
          }

          // Store.
          if ( n > 0 ) vst1q_lane_f64( c_loc + 0 + 0 * cs_c, vc_00, 0 );
          if ( n > 1 ) vst1q_lane_f64( c_loc + 0 + 1 * cs_c, vc_00, 1 );
          if ( n > 2 ) vst1q_lane_f64( c_loc + 0 + 2 * cs_c, vc_01, 0 );
          if ( n > 3 ) vst1q_lane_f64( c_loc + 0 + 3 * cs_c, vc_01, 1 );
          if ( n > 4 ) vst1q_lane_f64( c_loc + 0 + 4 * cs_c, vc_02, 0 );
          if ( n > 5 ) vst1q_lane_f64( c_loc + 0 + 5 * cs_c, vc_02, 1 );
          if ( n > 6 ) vst1q_lane_f64( c_loc + 0 + 6 * cs_c, vc_03, 0 );
          if ( n > 7 ) vst1q_lane_f64( c_loc + 0 + 7 * cs_c, vc_03, 1 );
        }
        if ( m0 > 1 )
        {
          // Load.
          if ( n > 0 ) vb_0 = vld1q_lane_f64( c_loc + 1 + 0 * cs_c, vb_0, 0 );
          if ( n > 1 ) vb_0 = vld1q_lane_f64( c_loc + 1 + 1 * cs_c, vb_0, 1 );
          if ( n > 2 ) vb_1 = vld1q_lane_f64( c_loc + 1 + 2 * cs_c, vb_1, 0 );
          if ( n > 3 ) vb_1 = vld1q_lane_f64( c_loc + 1 + 3 * cs_c, vb_1, 1 );
          if ( n > 4 ) vb_2 = vld1q_lane_f64( c_loc + 1 + 4 * cs_c, vb_2, 0 );
          if ( n > 5 ) vb_2 = vld1q_lane_f64( c_loc + 1 + 5 * cs_c, vb_2, 1 );
          if ( n > 6 ) vb_3 = vld1q_lane_f64( c_loc + 1 + 6 * cs_c, vb_3, 0 );
          if ( n > 7 ) vb_3 = vld1q_lane_f64( c_loc + 1 + 7 * cs_c, vb_3, 1 );

          // Scale.
          if ( !b_iszr )
          {
            vc_10 = vfmaq_f64( vc_10, vb_0, va_0 );
            vc_11 = vfmaq_f64( vc_11, vb_1, va_0 );
            vc_12 = vfmaq_f64( vc_12, vb_2, va_0 );
            vc_13 = vfmaq_f64( vc_13, vb_3, va_0 );
          }

          // Store.
          if ( n > 0 ) vst1q_lane_f64( c_loc + 1 + 0 * cs_c, vc_10, 0 );
          if ( n > 1 ) vst1q_lane_f64( c_loc + 1 + 1 * cs_c, vc_10, 1 );
          if ( n > 2 ) vst1q_lane_f64( c_loc + 1 + 2 * cs_c, vc_11, 0 );
          if ( n > 3 ) vst1q_lane_f64( c_loc + 1 + 3 * cs_c, vc_11, 1 );
          if ( n > 4 ) vst1q_lane_f64( c_loc + 1 + 4 * cs_c, vc_12, 0 );
          if ( n > 5 ) vst1q_lane_f64( c_loc + 1 + 5 * cs_c, vc_12, 1 );
          if ( n > 6 ) vst1q_lane_f64( c_loc + 1 + 6 * cs_c, vc_13, 0 );
          if ( n > 7 ) vst1q_lane_f64( c_loc + 1 + 7 * cs_c, vc_13, 1 );
        }
        if ( m0 > 2 )
        {
          // Load.
          if ( n > 0 ) vb_0 = vld1q_lane_f64( c_loc + 2 + 0 * cs_c, vb_0, 0 );
          if ( n > 1 ) vb_0 = vld1q_lane_f64( c_loc + 2 + 1 * cs_c, vb_0, 1 );
          if ( n > 2 ) vb_1 = vld1q_lane_f64( c_loc + 2 + 2 * cs_c, vb_1, 0 );
          if ( n > 3 ) vb_1 = vld1q_lane_f64( c_loc + 2 + 3 * cs_c, vb_1, 1 );
          if ( n > 4 ) vb_2 = vld1q_lane_f64( c_loc + 2 + 4 * cs_c, vb_2, 0 );
          if ( n > 5 ) vb_2 = vld1q_lane_f64( c_loc + 2 + 5 * cs_c, vb_2, 1 );
          if ( n > 6 ) vb_3 = vld1q_lane_f64( c_loc + 2 + 6 * cs_c, vb_3, 0 );
          if ( n > 7 ) vb_3 = vld1q_lane_f64( c_loc + 2 + 7 * cs_c, vb_3, 1 );

          // Scale.
          if ( !b_iszr )
          {
            vc_20 = vfmaq_f64( vc_20, vb_0, va_0 );
            vc_21 = vfmaq_f64( vc_21, vb_1, va_0 );
            vc_22 = vfmaq_f64( vc_22, vb_2, va_0 );
            vc_23 = vfmaq_f64( vc_23, vb_3, va_0 );
          }

          // Store.
          if ( n > 0 ) vst1q_lane_f64( c_loc + 2 + 0 * cs_c, vc_20, 0 );
          if ( n > 1 ) vst1q_lane_f64( c_loc + 2 + 1 * cs_c, vc_20, 1 );
          if ( n > 2 ) vst1q_lane_f64( c_loc + 2 + 2 * cs_c, vc_21, 0 );
          if ( n > 3 ) vst1q_lane_f64( c_loc + 2 + 3 * cs_c, vc_21, 1 );
          if ( n > 4 ) vst1q_lane_f64( c_loc + 2 + 4 * cs_c, vc_22, 0 );
          if ( n > 5 ) vst1q_lane_f64( c_loc + 2 + 5 * cs_c, vc_22, 1 );
          if ( n > 6 ) vst1q_lane_f64( c_loc + 2 + 6 * cs_c, vc_23, 0 );
          if ( n > 7 ) vst1q_lane_f64( c_loc + 2 + 7 * cs_c, vc_23, 1 );
        }
      }

      b_in += ps_b;
      c_in += 8 * cs_c;
    }

    a0 = ( double* )a0 + ps_a;
    c0 = ( double* )c0 + 3 * rs_c;
  }
}

