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
void bli_dgemmsup_rv_armv8a_int_6x4mn
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

  // Registers used to store a 6x4 block of C.
  float64x2_t vc_00, vc_01;
  float64x2_t vc_10, vc_11;
  float64x2_t vc_20, vc_21;
  float64x2_t vc_30, vc_31;
  float64x2_t vc_40, vc_41;
  float64x2_t vc_50, vc_51;
  float64x2_t va_0, va_1, va_2;
  float64x2_t vb_0, vb_1;

  PRAGMA_NOUNROLL
  for ( ; m0 > 0; m0 -= 6 )
  {
    n = n0;
    b_in = b0;
    c_in = c0;

    PRAGMA_NOUNROLL
    for ( ; n > 0; n -= 4 )
    {
      a_loc = a0;
      b_loc = b_in;
      c_loc = c_in;
      k = k0;

      vc_00 = (float64x2_t)vdupq_n_f64( 0 ); vc_01 = (float64x2_t)vdupq_n_f64( 0 );
      vc_10 = (float64x2_t)vdupq_n_f64( 0 ); vc_11 = (float64x2_t)vdupq_n_f64( 0 );
      vc_20 = (float64x2_t)vdupq_n_f64( 0 ); vc_21 = (float64x2_t)vdupq_n_f64( 0 );
      vc_30 = (float64x2_t)vdupq_n_f64( 0 ); vc_31 = (float64x2_t)vdupq_n_f64( 0 );
      vc_40 = (float64x2_t)vdupq_n_f64( 0 ); vc_41 = (float64x2_t)vdupq_n_f64( 0 );
      vc_50 = (float64x2_t)vdupq_n_f64( 0 ); vc_51 = (float64x2_t)vdupq_n_f64( 0 );

      PRAGMA_UNROLL
      for ( ; k > 0; --k )
      {
        // A columns.
        // if ( m0 > 0 )
                      va_0 = vld1q_lane_f64( a_loc + rs_a * 0, va_0, 0 );
        if ( m0 > 1 ) va_0 = vld1q_lane_f64( a_loc + rs_a * 1, va_0, 1 );
        if ( m0 > 2 ) va_1 = vld1q_lane_f64( a_loc + rs_a * 2, va_1, 0 );
        if ( m0 > 3 ) va_1 = vld1q_lane_f64( a_loc + rs_a * 3, va_1, 1 );
        if ( m0 > 4 ) va_2 = vld1q_lane_f64( a_loc + rs_a * 4, va_2, 0 );
        if ( m0 > 5 ) va_2 = vld1q_lane_f64( a_loc + rs_a * 5, va_2, 1 );
        // B rows.
        if      ( n > 1 ) vb_0 = vld1q_f64     ( b_loc + 0 );
        else              vb_0 = vld1q_lane_f64( b_loc + 0, vb_0, 0 );
        if      ( n > 3 ) vb_1 = vld1q_f64     ( b_loc + 2 );
        else if ( n > 2 ) vb_1 = vld1q_lane_f64( b_loc + 2, vb_1, 0 );
        a_loc += cs_a;
        b_loc += rs_b;

        // One or two-column case.
        if ( n <= 2 )
        {
          // if ( m0 > 0 )
          {
            vc_00 = vfmaq_laneq_f64( vc_00, vb_0, va_0, 0 );
            vc_10 = vfmaq_laneq_f64( vc_10, vb_0, va_0, 1 );
            vc_20 = vfmaq_laneq_f64( vc_20, vb_0, va_1, 0 );
          }
          if ( m0 > 3 )
          {
            vc_30 = vfmaq_laneq_f64( vc_30, vb_0, va_1, 1 );
            vc_40 = vfmaq_laneq_f64( vc_40, vb_0, va_2, 0 );
            vc_50 = vfmaq_laneq_f64( vc_50, vb_0, va_2, 1 );
          }
          continue;
        }

        // Three or four-column case. Moderately decrease num. of FMLA instructions
        //  according to m and n.
        // if ( m0 > 0 )
        {
          vc_00 = vfmaq_laneq_f64( vc_00, vb_0, va_0, 0 );
          vc_01 = vfmaq_laneq_f64( vc_01, vb_1, va_0, 0 );
          vc_10 = vfmaq_laneq_f64( vc_10, vb_0, va_0, 1 );
          vc_11 = vfmaq_laneq_f64( vc_11, vb_1, va_0, 1 );
        }
        if ( m0 > 2 )
        {
          vc_20 = vfmaq_laneq_f64( vc_20, vb_0, va_1, 0 );
          vc_21 = vfmaq_laneq_f64( vc_21, vb_1, va_1, 0 );
          vc_30 = vfmaq_laneq_f64( vc_30, vb_0, va_1, 1 );
          vc_31 = vfmaq_laneq_f64( vc_31, vb_1, va_1, 1 );
        }
        if ( m0 > 4 )
        {
          vc_40 = vfmaq_laneq_f64( vc_40, vb_0, va_2, 0 );
          vc_41 = vfmaq_laneq_f64( vc_41, vb_1, va_2, 0 );
          vc_50 = vfmaq_laneq_f64( vc_50, vb_0, va_2, 1 );
          vc_51 = vfmaq_laneq_f64( vc_51, vb_1, va_2, 1 );
        }
      }

      // Load alpha and beta.
      va_0 = vld1q_dup_f64( alpha );
      vb_0 = vld1q_dup_f64( beta );

      // Scale.
      vc_00 = vmulq_f64( vc_00, va_0 ); vc_01 = vmulq_f64( vc_01, va_0 );
      vc_10 = vmulq_f64( vc_10, va_0 ); vc_11 = vmulq_f64( vc_11, va_0 );
      vc_20 = vmulq_f64( vc_20, va_0 ); vc_21 = vmulq_f64( vc_21, va_0 );
      vc_30 = vmulq_f64( vc_30, va_0 ); vc_31 = vmulq_f64( vc_31, va_0 );
      vc_40 = vmulq_f64( vc_40, va_0 ); vc_41 = vmulq_f64( vc_41, va_0 );
      vc_50 = vmulq_f64( vc_50, va_0 ); vc_51 = vmulq_f64( vc_51, va_0 );

      if ( cs_c == 1 )
      {
        // Store in rows.
        // if ( m0 > 0 )
        {
          // Load.
          if      ( n > 1 ) va_0 = vld1q_f64     ( c_loc + 0 * rs_c + 0 );
          else              va_0 = vld1q_lane_f64( c_loc + 0 * rs_c + 0, va_0, 0 );
          if      ( n > 3 ) va_1 = vld1q_f64     ( c_loc + 0 * rs_c + 2 );
          else if ( n > 2 ) va_1 = vld1q_lane_f64( c_loc + 0 * rs_c + 2, va_1, 0 );

          // Scale.
          if ( !b_iszr )
          {
            vc_00 = vfmaq_f64( vc_00, va_0, vb_0 );
            vc_01 = vfmaq_f64( vc_01, va_1, vb_0 );
          }

          // Store.
          if      ( n > 1 ) vst1q_f64     ( c_loc + 0 * rs_c + 0, vc_00 );
          else              vst1q_lane_f64( c_loc + 0 * rs_c + 0, vc_00, 0 );
          if      ( n > 3 ) vst1q_f64     ( c_loc + 0 * rs_c + 2, vc_01 );
          else if ( n > 2 ) vst1q_lane_f64( c_loc + 0 * rs_c + 2, vc_01, 0 );
        }
        if ( m0 > 1 )
        {
          // Load.
          if      ( n > 1 ) va_0 = vld1q_f64     ( c_loc + 1 * rs_c + 0 );
          else              va_0 = vld1q_lane_f64( c_loc + 1 * rs_c + 0, va_0, 0 );
          if      ( n > 3 ) va_1 = vld1q_f64     ( c_loc + 1 * rs_c + 2 );
          else if ( n > 2 ) va_1 = vld1q_lane_f64( c_loc + 1 * rs_c + 2, va_1, 0 );

          // Scale.
          if ( !b_iszr )
          {
            vc_10 = vfmaq_f64( vc_10, va_0, vb_0 );
            vc_11 = vfmaq_f64( vc_11, va_1, vb_0 );
          }

          // Store.
          if      ( n > 1 ) vst1q_f64     ( c_loc + 1 * rs_c + 0, vc_10 );
          else              vst1q_lane_f64( c_loc + 1 * rs_c + 0, vc_10, 0 );
          if      ( n > 3 ) vst1q_f64     ( c_loc + 1 * rs_c + 2, vc_11 );
          else if ( n > 2 ) vst1q_lane_f64( c_loc + 1 * rs_c + 2, vc_11, 0 );
        }
        if ( m0 > 2 )
        {
          // Load.
          if      ( n > 1 ) va_0 = vld1q_f64     ( c_loc + 2 * rs_c + 0 );
          else              va_0 = vld1q_lane_f64( c_loc + 2 * rs_c + 0, va_0, 0 );
          if      ( n > 3 ) va_1 = vld1q_f64     ( c_loc + 2 * rs_c + 2 );
          else if ( n > 2 ) va_1 = vld1q_lane_f64( c_loc + 2 * rs_c + 2, va_1, 0 );

          // Scale.
          if ( !b_iszr )
          {
            vc_20 = vfmaq_f64( vc_20, va_0, vb_0 );
            vc_21 = vfmaq_f64( vc_21, va_1, vb_0 );
          }

          // Store.
          if      ( n > 1 ) vst1q_f64     ( c_loc + 2 * rs_c + 0, vc_20 );
          else              vst1q_lane_f64( c_loc + 2 * rs_c + 0, vc_20, 0 );
          if      ( n > 3 ) vst1q_f64     ( c_loc + 2 * rs_c + 2, vc_21 );
          else if ( n > 2 ) vst1q_lane_f64( c_loc + 2 * rs_c + 2, vc_21, 0 );
        }
        if ( m0 > 3 )
        {
          // Load.
          if      ( n > 1 ) va_0 = vld1q_f64     ( c_loc + 3 * rs_c + 0 );
          else              va_0 = vld1q_lane_f64( c_loc + 3 * rs_c + 0, va_0, 0 );
          if      ( n > 3 ) va_1 = vld1q_f64     ( c_loc + 3 * rs_c + 2 );
          else if ( n > 2 ) va_1 = vld1q_lane_f64( c_loc + 3 * rs_c + 2, va_1, 0 );

          // Scale.
          if ( !b_iszr )
          {
            vc_30 = vfmaq_f64( vc_30, va_0, vb_0 );
            vc_31 = vfmaq_f64( vc_31, va_1, vb_0 );
          }

          // Store.
          if      ( n > 1 ) vst1q_f64     ( c_loc + 3 * rs_c + 0, vc_30 );
          else              vst1q_lane_f64( c_loc + 3 * rs_c + 0, vc_30, 0 );
          if      ( n > 3 ) vst1q_f64     ( c_loc + 3 * rs_c + 2, vc_31 );
          else if ( n > 2 ) vst1q_lane_f64( c_loc + 3 * rs_c + 2, vc_31, 0 );
        }
        if ( m0 > 4 )
        {
          // Load.
          if      ( n > 1 ) va_0 = vld1q_f64     ( c_loc + 4 * rs_c + 0 );
          else              va_0 = vld1q_lane_f64( c_loc + 4 * rs_c + 0, va_0, 0 );
          if      ( n > 3 ) va_1 = vld1q_f64     ( c_loc + 4 * rs_c + 2 );
          else if ( n > 2 ) va_1 = vld1q_lane_f64( c_loc + 4 * rs_c + 2, va_1, 0 );

          // Scale.
          if ( !b_iszr )
          {
            vc_40 = vfmaq_f64( vc_40, va_0, vb_0 );
            vc_41 = vfmaq_f64( vc_41, va_1, vb_0 );
          }

          // Store.
          if      ( n > 1 ) vst1q_f64     ( c_loc + 4 * rs_c + 0, vc_40 );
          else              vst1q_lane_f64( c_loc + 4 * rs_c + 0, vc_40, 0 );
          if      ( n > 3 ) vst1q_f64     ( c_loc + 4 * rs_c + 2, vc_41 );
          else if ( n > 2 ) vst1q_lane_f64( c_loc + 4 * rs_c + 2, vc_41, 0 );
        }
        if ( m0 > 5 )
        {
          // Load.
          if      ( n > 1 ) va_0 = vld1q_f64     ( c_loc + 5 * rs_c + 0 );
          else              va_0 = vld1q_lane_f64( c_loc + 5 * rs_c + 0, va_0, 0 );
          if      ( n > 3 ) va_1 = vld1q_f64     ( c_loc + 5 * rs_c + 2 );
          else if ( n > 2 ) va_1 = vld1q_lane_f64( c_loc + 5 * rs_c + 2, va_1, 0 );

          // Scale.
          if ( !b_iszr )
          {
            vc_50 = vfmaq_f64( vc_50, va_0, vb_0 );
            vc_51 = vfmaq_f64( vc_51, va_1, vb_0 );
          }

          // Store.
          if      ( n > 1 ) vst1q_f64     ( c_loc + 5 * rs_c + 0, vc_50 );
          else              vst1q_lane_f64( c_loc + 5 * rs_c + 0, vc_50, 0 );
          if      ( n > 3 ) vst1q_f64     ( c_loc + 5 * rs_c + 2, vc_51 );
          else if ( n > 2 ) vst1q_lane_f64( c_loc + 5 * rs_c + 2, vc_51, 0 );
        }
      }
      else
      {
        // Store in columns.

        // Rename some vectors.
#define VCOL0 va_0
#define VCOL1 va_1
#define VCOL2 va_2
#define VCOL3 vb_1
#define VTMP0 vc_00
#define VTMP1 vc_01
#define VTMP2 vc_10
#define VTMP3 vc_11
        // if ( m0 > 0 )
        {
          VCOL0 = vtrn1q_f64(vc_00, vc_10);
          VCOL1 = vtrn2q_f64(vc_00, vc_10);
          VCOL2 = vtrn1q_f64(vc_01, vc_11);
          VCOL3 = vtrn2q_f64(vc_01, vc_11);

          if ( m0 > 1 )
          {
            if ( n > 0 ) VTMP0 = vld1q_f64( c_loc + 0 * cs_c + 0 );
            if ( n > 1 ) VTMP1 = vld1q_f64( c_loc + 1 * cs_c + 0 );
            if ( n > 2 ) VTMP2 = vld1q_f64( c_loc + 2 * cs_c + 0 );
            if ( n > 3 ) VTMP3 = vld1q_f64( c_loc + 3 * cs_c + 0 );
            if ( !b_iszr )
            {
              VCOL0 = vfmaq_f64( VCOL0, VTMP0, vb_0 );
              VCOL1 = vfmaq_f64( VCOL1, VTMP1, vb_0 );
              VCOL2 = vfmaq_f64( VCOL2, VTMP2, vb_0 );
              VCOL3 = vfmaq_f64( VCOL3, VTMP3, vb_0 );
            }
            if ( n > 0 ) vst1q_f64( c_loc + 0 * cs_c + 0, VCOL0 );
            if ( n > 1 ) vst1q_f64( c_loc + 1 * cs_c + 0, VCOL1 );
            if ( n > 2 ) vst1q_f64( c_loc + 2 * cs_c + 0, VCOL2 );
            if ( n > 3 ) vst1q_f64( c_loc + 3 * cs_c + 0, VCOL3 );
          }
          else
          {
            if ( n > 0 ) VTMP0 = vld1q_lane_f64( c_loc + 0 * cs_c + 0, VTMP0, 0 );
            if ( n > 1 ) VTMP1 = vld1q_lane_f64( c_loc + 1 * cs_c + 0, VTMP1, 0 );
            if ( n > 2 ) VTMP2 = vld1q_lane_f64( c_loc + 2 * cs_c + 0, VTMP2, 0 );
            if ( n > 3 ) VTMP3 = vld1q_lane_f64( c_loc + 3 * cs_c + 0, VTMP3, 0 );
            if ( !b_iszr )
            {
              VCOL0 = vfmaq_f64( VCOL0, VTMP0, vb_0 );
              VCOL1 = vfmaq_f64( VCOL1, VTMP1, vb_0 );
              VCOL2 = vfmaq_f64( VCOL2, VTMP2, vb_0 );
              VCOL3 = vfmaq_f64( VCOL3, VTMP3, vb_0 );
            }
            if ( n > 0 ) vst1q_lane_f64( c_loc + 0 * cs_c + 0, VCOL0, 0 );
            if ( n > 1 ) vst1q_lane_f64( c_loc + 1 * cs_c + 0, VCOL1, 0 );
            if ( n > 2 ) vst1q_lane_f64( c_loc + 2 * cs_c + 0, VCOL2, 0 );
            if ( n > 3 ) vst1q_lane_f64( c_loc + 3 * cs_c + 0, VCOL3, 0 );
          }
        }
        if ( m0 > 2 )
        {
          VCOL0 = vtrn1q_f64(vc_20, vc_30);
          VCOL1 = vtrn2q_f64(vc_20, vc_30);
          VCOL2 = vtrn1q_f64(vc_21, vc_31);
          VCOL3 = vtrn2q_f64(vc_21, vc_31);

          if ( m0 > 3 )
          {
            if ( n > 0 ) VTMP0 = vld1q_f64( c_loc + 0 * cs_c + 2 );
            if ( n > 1 ) VTMP1 = vld1q_f64( c_loc + 1 * cs_c + 2 );
            if ( n > 2 ) VTMP2 = vld1q_f64( c_loc + 2 * cs_c + 2 );
            if ( n > 3 ) VTMP3 = vld1q_f64( c_loc + 3 * cs_c + 2 );
            if ( !b_iszr )
            {
              VCOL0 = vfmaq_f64( VCOL0, VTMP0, vb_0 );
              VCOL1 = vfmaq_f64( VCOL1, VTMP1, vb_0 );
              VCOL2 = vfmaq_f64( VCOL2, VTMP2, vb_0 );
              VCOL3 = vfmaq_f64( VCOL3, VTMP3, vb_0 );
            }
            if ( n > 0 ) vst1q_f64( c_loc + 0 * cs_c + 2, VCOL0 );
            if ( n > 1 ) vst1q_f64( c_loc + 1 * cs_c + 2, VCOL1 );
            if ( n > 2 ) vst1q_f64( c_loc + 2 * cs_c + 2, VCOL2 );
            if ( n > 3 ) vst1q_f64( c_loc + 3 * cs_c + 2, VCOL3 );
          }
          else
          {
            if ( n > 0 ) VTMP0 = vld1q_lane_f64( c_loc + 0 * cs_c + 2, VTMP0, 0 );
            if ( n > 1 ) VTMP1 = vld1q_lane_f64( c_loc + 1 * cs_c + 2, VTMP1, 0 );
            if ( n > 2 ) VTMP2 = vld1q_lane_f64( c_loc + 2 * cs_c + 2, VTMP2, 0 );
            if ( n > 3 ) VTMP3 = vld1q_lane_f64( c_loc + 3 * cs_c + 2, VTMP3, 0 );
            if ( !b_iszr )
            {
              VCOL0 = vfmaq_f64( VCOL0, VTMP0, vb_0 );
              VCOL1 = vfmaq_f64( VCOL1, VTMP1, vb_0 );
              VCOL2 = vfmaq_f64( VCOL2, VTMP2, vb_0 );
              VCOL3 = vfmaq_f64( VCOL3, VTMP3, vb_0 );
            }
            if ( n > 0 ) vst1q_lane_f64( c_loc + 0 * cs_c + 2, VCOL0, 0 );
            if ( n > 1 ) vst1q_lane_f64( c_loc + 1 * cs_c + 2, VCOL1, 0 );
            if ( n > 2 ) vst1q_lane_f64( c_loc + 2 * cs_c + 2, VCOL2, 0 );
            if ( n > 3 ) vst1q_lane_f64( c_loc + 3 * cs_c + 2, VCOL3, 0 );
          }
        }
        if ( m0 > 4 )
        {
          VCOL0 = vtrn1q_f64(vc_40, vc_50);
          VCOL1 = vtrn2q_f64(vc_40, vc_50);
          VCOL2 = vtrn1q_f64(vc_41, vc_51);
          VCOL3 = vtrn2q_f64(vc_41, vc_51);

          if ( m0 > 5 )
          {
            if ( n > 0 ) VTMP0 = vld1q_f64( c_loc + 0 * cs_c + 4 );
            if ( n > 1 ) VTMP1 = vld1q_f64( c_loc + 1 * cs_c + 4 );
            if ( n > 2 ) VTMP2 = vld1q_f64( c_loc + 2 * cs_c + 4 );
            if ( n > 3 ) VTMP3 = vld1q_f64( c_loc + 3 * cs_c + 4 );
            if ( !b_iszr )
            {
              VCOL0 = vfmaq_f64( VCOL0, VTMP0, vb_0 );
              VCOL1 = vfmaq_f64( VCOL1, VTMP1, vb_0 );
              VCOL2 = vfmaq_f64( VCOL2, VTMP2, vb_0 );
              VCOL3 = vfmaq_f64( VCOL3, VTMP3, vb_0 );
            }
            if ( n > 0 ) vst1q_f64( c_loc + 0 * cs_c + 4, VCOL0 );
            if ( n > 1 ) vst1q_f64( c_loc + 1 * cs_c + 4, VCOL1 );
            if ( n > 2 ) vst1q_f64( c_loc + 2 * cs_c + 4, VCOL2 );
            if ( n > 3 ) vst1q_f64( c_loc + 3 * cs_c + 4, VCOL3 );
          }
          else
          {
            if ( n > 0 ) VTMP0 = vld1q_lane_f64( c_loc + 0 * cs_c + 4, VTMP0, 0 );
            if ( n > 1 ) VTMP1 = vld1q_lane_f64( c_loc + 1 * cs_c + 4, VTMP1, 0 );
            if ( n > 2 ) VTMP2 = vld1q_lane_f64( c_loc + 2 * cs_c + 4, VTMP2, 0 );
            if ( n > 3 ) VTMP3 = vld1q_lane_f64( c_loc + 3 * cs_c + 4, VTMP3, 0 );
            if ( !b_iszr )
            {
              VCOL0 = vfmaq_f64( VCOL0, VTMP0, vb_0 );
              VCOL1 = vfmaq_f64( VCOL1, VTMP1, vb_0 );
              VCOL2 = vfmaq_f64( VCOL2, VTMP2, vb_0 );
              VCOL3 = vfmaq_f64( VCOL3, VTMP3, vb_0 );
            }
            if ( n > 0 ) vst1q_lane_f64( c_loc + 0 * cs_c + 4, VCOL0, 0 );
            if ( n > 1 ) vst1q_lane_f64( c_loc + 1 * cs_c + 4, VCOL1, 0 );
            if ( n > 2 ) vst1q_lane_f64( c_loc + 2 * cs_c + 4, VCOL2, 0 );
            if ( n > 3 ) vst1q_lane_f64( c_loc + 3 * cs_c + 4, VCOL3, 0 );
          }
        }
      }

      b_in += ps_b;
      c_in += 4 * cs_c;
    }

    a0 = ( double* )a0 + ps_a;
    c0 = ( double* )c0 + 6 * rs_c;
  }
}

