/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020, Linaro Limited

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
#include <arm_neon.h>

#if defined(__clang__)
#define PRAGMA_NOUNROLL _Pragma("nounroll")
#define PRAGMA_UNROLL_2 _Pragma("unroll 2")
#define PRAGMA_UNROLL_4 _Pragma("unroll 4")
#elif defined(__GNUC__)
#define PRAGMA_NOUNROLL _Pragma("GCC unroll 1")
#define PRAGMA_UNROLL_2 _Pragma("GCC unroll 2")
#define PRAGMA_UNROLL_4 _Pragma("GCC unroll 4")
#else
#define PRAGMA_NOUNROLL
#define PRAGMA_UNROLL_2
#define PRAGMA_UNROLL_4
#endif

void bli_spackm_armv8a_int_8x12
     (
             conj_t  conja,
             pack_t  schema,
             dim_t   cdim0,
             dim_t   cdim_max,
             dim_t   cdim_bcast,
             dim_t   k0,
             dim_t   k0_max,
       const void*   kappa,
       const void*   a, inc_t inca0, inc_t lda0,
             void*   p,              inc_t ldp0,
       const void*   params,
       const cntx_t* cntx
     )
{
  // This is the panel dimension assumed by the packm kernel.
  const dim_t    mr     = 8;
  const dim_t    nr     = 12;

  // Typecast local copies of integers in case dim_t and inc_t are a
  // different size than is expected by load instructions.
  uint64_t       k_iter = k0 / 4;
  uint64_t       k_left = k0 % 4;

  const float*   a_loc  = a;
        float*   p_loc  = p;

  // NOTE: For the purposes of the comments in this packm kernel, we
  // interpret inca and lda as rs_a and cs_a, respectively, and similarly
  // interpret ldp as cs_p (with rs_p implicitly unit). Thus, when reading
  // this packm kernel, you should think of the operation as packing an
  // m x n micropanel, where m and n are tiny and large, respectively, and
  // where elements of each column of the packed matrix P are contiguous.
  // (This packm kernel can still be used to pack micropanels of matrix B
  // in a gemm operation.)
  const uint64_t inca   = inca0;
  const uint64_t lda    = lda0;
  const uint64_t ldp    = ldp0;

  const bool     gs     = ( inca0 != 1 && lda0 != 1 );

  // NOTE: If/when this kernel ever supports scaling by kappa within the
  // assembly region, this constraint should be lifted.
  const bool     unitk  = bli_seq1( *(( float* )kappa) );


  // -------------------------------------------------------------------------

  if ( cdim0 == mr && cdim_bcast == 1 && !gs )
  {
    if ( unitk )
    {
      if ( inca == 1 )
      {
        // No need to use k-loops here.
        // Simply let compiler to expand loops.
        PRAGMA_UNROLL_4
        for ( dim_t ik = k_iter * 4 + k_left; ik > 0; --ik )
        {
          float32x4_t v0 = vld1q_f32( a_loc +  0 );
          float32x4_t v1 = vld1q_f32( a_loc +  4 );

          vst1q_f32( p_loc +  0, v0 );
          vst1q_f32( p_loc +  4, v1 );

          a_loc += lda;
          p_loc += ldp;
        }
      }
      else // if ( lda == 1 )
      {
        float32x4_t v0  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v1  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v2  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v3  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v4  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v5  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v6  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v7  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t vt0;
        float32x4_t vt1;
        float32x4_t vt2;
        float32x4_t vt3;

        PRAGMA_NOUNROLL
        for ( ; k_iter > 0; --k_iter )
        {
          v0 = vld1q_f32( a_loc + inca * 0 );
          v1 = vld1q_f32( a_loc + inca * 1 );
          v2 = vld1q_f32( a_loc + inca * 2 );
          v3 = vld1q_f32( a_loc + inca * 3 );
          v4 = vld1q_f32( a_loc + inca * 4 );
          v5 = vld1q_f32( a_loc + inca * 5 );
          v6 = vld1q_f32( a_loc + inca * 6 );
          v7 = vld1q_f32( a_loc + inca * 7 );

          // In-register transpose.
          //
          // Column 0-3
          vt0 = vtrn1q_f32( v0, v1 );
          vt1 = vtrn2q_f32( v0, v1 );
          vt2 = vtrn1q_f32( v2, v3 );
          vt3 = vtrn2q_f32( v2, v3 );
          v0 = (float32x4_t)vtrn1q_f64( (float64x2_t)vt0, (float64x2_t)vt2 );
          v1 = (float32x4_t)vtrn1q_f64( (float64x2_t)vt1, (float64x2_t)vt3 );
          v2 = (float32x4_t)vtrn2q_f64( (float64x2_t)vt0, (float64x2_t)vt2 );
          v3 = (float32x4_t)vtrn2q_f64( (float64x2_t)vt1, (float64x2_t)vt3 );
          // Column 4-7
          vt0 = vtrn1q_f32( v4, v5 );
          vt1 = vtrn2q_f32( v4, v5 );
          vt2 = vtrn1q_f32( v6, v7 );
          vt3 = vtrn2q_f32( v6, v7 );
          v4 = (float32x4_t)vtrn1q_f64( (float64x2_t)vt0, (float64x2_t)vt2 );
          v5 = (float32x4_t)vtrn1q_f64( (float64x2_t)vt1, (float64x2_t)vt3 );
          v6 = (float32x4_t)vtrn2q_f64( (float64x2_t)vt0, (float64x2_t)vt2 );
          v7 = (float32x4_t)vtrn2q_f64( (float64x2_t)vt1, (float64x2_t)vt3 );

          vst1q_f32( p_loc + 0,  v0  );
          vst1q_f32( p_loc + 4,  v4  );
          p_loc += ldp;

          vst1q_f32( p_loc + 0,  v1  );
          vst1q_f32( p_loc + 4,  v5  );
          p_loc += ldp;

          vst1q_f32( p_loc + 0,  v2  );
          vst1q_f32( p_loc + 4,  v6  );
          p_loc += ldp;

          vst1q_f32( p_loc + 0,  v3  );
          vst1q_f32( p_loc + 4,  v7  );
          p_loc += ldp;
          a_loc += 4 * lda; // 4;
        }
        for ( ; k_left > 0; --k_left )
        {
          v0 = vld1q_lane_f32( a_loc + inca * 0 , v0, 0 );
          v0 = vld1q_lane_f32( a_loc + inca * 1 , v0, 1 );
          v0 = vld1q_lane_f32( a_loc + inca * 2 , v0, 2 );
          v0 = vld1q_lane_f32( a_loc + inca * 3 , v0, 3 );
          v1 = vld1q_lane_f32( a_loc + inca * 4 , v1, 0 );
          v1 = vld1q_lane_f32( a_loc + inca * 5 , v1, 1 );
          v1 = vld1q_lane_f32( a_loc + inca * 6 , v1, 2 );
          v1 = vld1q_lane_f32( a_loc + inca * 7 , v1, 3 );

          vst1q_f32( p_loc + 0,  v0 );
          vst1q_f32( p_loc + 4,  v1 );
          p_loc += ldp;
          a_loc += lda; // 1;
        }
      }
    }
    else // if ( !unitk )
    {
      float32x4_t vkappa = vld1q_dup_f32( kappa );

      if ( inca == 1 )
      {
        // No need to use k-loops here.
        // Simply let compiler to expand loops.
        PRAGMA_UNROLL_4
        for ( dim_t ik = k_iter * 4 + k_left; ik > 0; --ik )
        {
          float32x4_t v0 = vld1q_f32( a_loc + 0 );
          float32x4_t v1 = vld1q_f32( a_loc + 4 );

          // Scale by kappa.
          v0 = vmulq_f32( v0, vkappa );
          v1 = vmulq_f32( v1, vkappa );

          vst1q_f32( p_loc + 0, v0 );
          vst1q_f32( p_loc + 4, v1 );

          a_loc += lda;
          p_loc += ldp;
        }
      }
      else // if ( lda == 1 )
      {
        float32x4_t v0  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v1  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v2  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v3  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v4  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v5  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v6  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v7  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t vt0;
        float32x4_t vt1;
        float32x4_t vt2;
        float32x4_t vt3;

        PRAGMA_NOUNROLL
        for ( ; k_iter > 0; --k_iter )
        {
          v0  = vld1q_f32( a_loc + inca * 0  );
          v1  = vld1q_f32( a_loc + inca * 1  );
          v2  = vld1q_f32( a_loc + inca * 2  );
          v3  = vld1q_f32( a_loc + inca * 3  );
          v4  = vld1q_f32( a_loc + inca * 4  );
          v5  = vld1q_f32( a_loc + inca * 5  );
          v6  = vld1q_f32( a_loc + inca * 6  );
          v7  = vld1q_f32( a_loc + inca * 7  );

          // Scale by kappa.
          v0  = vmulq_f32( v0,  vkappa );
          v1  = vmulq_f32( v1,  vkappa );
          v2  = vmulq_f32( v2,  vkappa );
          v3  = vmulq_f32( v3,  vkappa );
          v4  = vmulq_f32( v4,  vkappa );
          v5  = vmulq_f32( v5,  vkappa );
          v6  = vmulq_f32( v6,  vkappa );
          v7  = vmulq_f32( v7,  vkappa );

          // In-register transpose.
          //
          // Column 0-3
          vt0 = vtrn1q_f32( v0, v1 );
          vt1 = vtrn2q_f32( v0, v1 );
          vt2 = vtrn1q_f32( v2, v3 );
          vt3 = vtrn2q_f32( v2, v3 );
          v0 = (float32x4_t)vtrn1q_f64( (float64x2_t)vt0, (float64x2_t)vt2 );
          v1 = (float32x4_t)vtrn1q_f64( (float64x2_t)vt1, (float64x2_t)vt3 );
          v2 = (float32x4_t)vtrn2q_f64( (float64x2_t)vt0, (float64x2_t)vt2 );
          v3 = (float32x4_t)vtrn2q_f64( (float64x2_t)vt1, (float64x2_t)vt3 );
          // Column 4-7
          vt0 = vtrn1q_f32( v4, v5 );
          vt1 = vtrn2q_f32( v4, v5 );
          vt2 = vtrn1q_f32( v6, v7 );
          vt3 = vtrn2q_f32( v6, v7 );
          v4 = (float32x4_t)vtrn1q_f64( (float64x2_t)vt0, (float64x2_t)vt2 );
          v5 = (float32x4_t)vtrn1q_f64( (float64x2_t)vt1, (float64x2_t)vt3 );
          v6 = (float32x4_t)vtrn2q_f64( (float64x2_t)vt0, (float64x2_t)vt2 );
          v7 = (float32x4_t)vtrn2q_f64( (float64x2_t)vt1, (float64x2_t)vt3 );

          vst1q_f32( p_loc + 0,  v0  );
          vst1q_f32( p_loc + 4,  v4  );
          p_loc += ldp;

          vst1q_f32( p_loc + 0,  v1  );
          vst1q_f32( p_loc + 4,  v5  );
          p_loc += ldp;

          vst1q_f32( p_loc + 0,  v2  );
          vst1q_f32( p_loc + 4,  v6  );
          p_loc += ldp;

          vst1q_f32( p_loc + 0,  v3  );
          vst1q_f32( p_loc + 4,  v7  );
          p_loc += ldp;
          a_loc += 4 * lda; // 4;
        }
        for ( ; k_left > 0; --k_left )
        {
          v0 = vld1q_lane_f32( a_loc + inca * 0 , v0, 0 );
          v0 = vld1q_lane_f32( a_loc + inca * 1 , v0, 1 );
          v0 = vld1q_lane_f32( a_loc + inca * 2 , v0, 2 );
          v0 = vld1q_lane_f32( a_loc + inca * 3 , v0, 3 );
          v1 = vld1q_lane_f32( a_loc + inca * 4 , v1, 0 );
          v1 = vld1q_lane_f32( a_loc + inca * 5 , v1, 1 );
          v1 = vld1q_lane_f32( a_loc + inca * 6 , v1, 2 );
          v1 = vld1q_lane_f32( a_loc + inca * 7 , v1, 3 );

          // Scale by kappa.
          v0 = vmulq_f32( v0, vkappa );
          v1 = vmulq_f32( v1, vkappa );

          vst1q_f32( p_loc + 0,  v0 );
          vst1q_f32( p_loc + 4,  v1 );
          p_loc += ldp;
          a_loc += lda; // 1;
        }
      }
    }
  }
  else if ( cdim0 == nr && cdim_bcast == 1 && !gs )
  {
    if ( unitk )
    {
      if ( inca == 1 )
      {
        // No need to use k-loops here.
        // Simply let compiler to expand loops.
        PRAGMA_UNROLL_2
        for ( dim_t ik = k_iter * 4 + k_left; ik > 0; --ik )
        {
          float32x4_t v0 = vld1q_f32( a_loc +  0 );
          float32x4_t v1 = vld1q_f32( a_loc +  4 );
          float32x4_t v2 = vld1q_f32( a_loc +  8 );

          vst1q_f32( p_loc +  0, v0 );
          vst1q_f32( p_loc +  4, v1 );
          vst1q_f32( p_loc +  8, v2 );

          a_loc += lda;
          p_loc += ldp;
        }
      }
      else // if ( lda == 1 )
      {
        float32x4_t v0  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v1  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v2  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v3  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v4  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v5  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v6  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v7  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v8  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v9  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v10 = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v11 = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t vt0;
        float32x4_t vt1;
        float32x4_t vt2;
        float32x4_t vt3;

        PRAGMA_NOUNROLL
        for ( ; k_iter > 0; --k_iter )
        {
          v0  = vld1q_f32( a_loc + inca * 0  );
          v1  = vld1q_f32( a_loc + inca * 1  );
          v2  = vld1q_f32( a_loc + inca * 2  );
          v3  = vld1q_f32( a_loc + inca * 3  );
          v4  = vld1q_f32( a_loc + inca * 4  );
          v5  = vld1q_f32( a_loc + inca * 5  );
          v6  = vld1q_f32( a_loc + inca * 6  );
          v7  = vld1q_f32( a_loc + inca * 7  );
          v8  = vld1q_f32( a_loc + inca * 8  );
          v9  = vld1q_f32( a_loc + inca * 9  );
          v10 = vld1q_f32( a_loc + inca * 10 );
          v11 = vld1q_f32( a_loc + inca * 11 );

          // In-register transpose.
          //
          // Column 0-3
          vt0 = vtrn1q_f32( v0, v1 );
          vt1 = vtrn2q_f32( v0, v1 );
          vt2 = vtrn1q_f32( v2, v3 );
          vt3 = vtrn2q_f32( v2, v3 );
          v0 = (float32x4_t)vtrn1q_f64( (float64x2_t)vt0, (float64x2_t)vt2 );
          v1 = (float32x4_t)vtrn1q_f64( (float64x2_t)vt1, (float64x2_t)vt3 );
          v2 = (float32x4_t)vtrn2q_f64( (float64x2_t)vt0, (float64x2_t)vt2 );
          v3 = (float32x4_t)vtrn2q_f64( (float64x2_t)vt1, (float64x2_t)vt3 );
          // Column 4-7
          vt0 = vtrn1q_f32( v4, v5 );
          vt1 = vtrn2q_f32( v4, v5 );
          vt2 = vtrn1q_f32( v6, v7 );
          vt3 = vtrn2q_f32( v6, v7 );
          v4 = (float32x4_t)vtrn1q_f64( (float64x2_t)vt0, (float64x2_t)vt2 );
          v5 = (float32x4_t)vtrn1q_f64( (float64x2_t)vt1, (float64x2_t)vt3 );
          v6 = (float32x4_t)vtrn2q_f64( (float64x2_t)vt0, (float64x2_t)vt2 );
          v7 = (float32x4_t)vtrn2q_f64( (float64x2_t)vt1, (float64x2_t)vt3 );
          // Column 8-11
          vt0 = vtrn1q_f32( v8,  v9  );
          vt1 = vtrn2q_f32( v8,  v9  );
          vt2 = vtrn1q_f32( v10, v11 );
          vt3 = vtrn2q_f32( v10, v11 );
          v8  = (float32x4_t)vtrn1q_f64( (float64x2_t)vt0, (float64x2_t)vt2 );
          v9  = (float32x4_t)vtrn1q_f64( (float64x2_t)vt1, (float64x2_t)vt3 );
          v10 = (float32x4_t)vtrn2q_f64( (float64x2_t)vt0, (float64x2_t)vt2 );
          v11 = (float32x4_t)vtrn2q_f64( (float64x2_t)vt1, (float64x2_t)vt3 );

          vst1q_f32( p_loc + 0,  v0  );
          vst1q_f32( p_loc + 4,  v4  );
          vst1q_f32( p_loc + 8,  v8  );
          p_loc += ldp;

          vst1q_f32( p_loc + 0,  v1  );
          vst1q_f32( p_loc + 4,  v5  );
          vst1q_f32( p_loc + 8,  v9  );
          p_loc += ldp;

          vst1q_f32( p_loc + 0,  v2  );
          vst1q_f32( p_loc + 4,  v6  );
          vst1q_f32( p_loc + 8,  v10 );
          p_loc += ldp;

          vst1q_f32( p_loc + 0,  v3  );
          vst1q_f32( p_loc + 4,  v7  );
          vst1q_f32( p_loc + 8,  v11 );
          p_loc += ldp;
          a_loc += 4 * lda; // 4;
        }
        for ( ; k_left > 0; --k_left )
        {
          v0 = vld1q_lane_f32( a_loc + inca * 0 , v0, 0 );
          v0 = vld1q_lane_f32( a_loc + inca * 1 , v0, 1 );
          v0 = vld1q_lane_f32( a_loc + inca * 2 , v0, 2 );
          v0 = vld1q_lane_f32( a_loc + inca * 3 , v0, 3 );
          v1 = vld1q_lane_f32( a_loc + inca * 4 , v1, 0 );
          v1 = vld1q_lane_f32( a_loc + inca * 5 , v1, 1 );
          v1 = vld1q_lane_f32( a_loc + inca * 6 , v1, 2 );
          v1 = vld1q_lane_f32( a_loc + inca * 7 , v1, 3 );
          v2 = vld1q_lane_f32( a_loc + inca * 8 , v2, 0 );
          v2 = vld1q_lane_f32( a_loc + inca * 9 , v2, 1 );
          v2 = vld1q_lane_f32( a_loc + inca * 10, v2, 2 );
          v2 = vld1q_lane_f32( a_loc + inca * 11, v2, 3 );

          vst1q_f32( p_loc + 0,  v0 );
          vst1q_f32( p_loc + 4,  v1 );
          vst1q_f32( p_loc + 8,  v2 );
          p_loc += ldp;
          a_loc += lda; // 1;
        }
      }
    }
    else // if ( !unitk )
    {
      float32x4_t vkappa = vld1q_dup_f32( kappa );

      if ( inca == 1 )
      {
        // No need to use k-loops here.
        // Simply let compiler to expand loops.
        PRAGMA_UNROLL_2
        for ( dim_t ik = k_iter * 4 + k_left; ik > 0; --ik )
        {
          float32x4_t v0 = vld1q_f32( a_loc + 0 );
          float32x4_t v1 = vld1q_f32( a_loc + 4 );
          float32x4_t v2 = vld1q_f32( a_loc + 8 );

          // Scale by kappa.
          v0 = vmulq_f32( v0, vkappa );
          v1 = vmulq_f32( v1, vkappa );
          v2 = vmulq_f32( v2, vkappa );

          vst1q_f32( p_loc + 0, v0 );
          vst1q_f32( p_loc + 4, v1 );
          vst1q_f32( p_loc + 8, v2 );

          a_loc += lda;
          p_loc += ldp;
        }
      }
      else // if ( lda == 1 )
      {
        float32x4_t v0  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v1  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v2  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v3  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v4  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v5  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v6  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v7  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v8  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v9  = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v10 = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t v11 = (float32x4_t)vdupq_n_u32( 0 );
        float32x4_t vt0;
        float32x4_t vt1;
        float32x4_t vt2;
        float32x4_t vt3;

        PRAGMA_NOUNROLL
        for ( ; k_iter > 0; --k_iter )
        {
          v0  = vld1q_f32( a_loc + inca * 0  );
          v1  = vld1q_f32( a_loc + inca * 1  );
          v2  = vld1q_f32( a_loc + inca * 2  );
          v3  = vld1q_f32( a_loc + inca * 3  );
          v4  = vld1q_f32( a_loc + inca * 4  );
          v5  = vld1q_f32( a_loc + inca * 5  );
          v6  = vld1q_f32( a_loc + inca * 6  );
          v7  = vld1q_f32( a_loc + inca * 7  );
          v8  = vld1q_f32( a_loc + inca * 8  );
          v9  = vld1q_f32( a_loc + inca * 9  );
          v10 = vld1q_f32( a_loc + inca * 10 );
          v11 = vld1q_f32( a_loc + inca * 11 );

          // Scale by kappa.
          v0  = vmulq_f32( v0,  vkappa );
          v1  = vmulq_f32( v1,  vkappa );
          v2  = vmulq_f32( v2,  vkappa );
          v3  = vmulq_f32( v3,  vkappa );
          v4  = vmulq_f32( v4,  vkappa );
          v5  = vmulq_f32( v5,  vkappa );
          v6  = vmulq_f32( v6,  vkappa );
          v7  = vmulq_f32( v7,  vkappa );
          v8  = vmulq_f32( v8,  vkappa );
          v9  = vmulq_f32( v9,  vkappa );
          v10 = vmulq_f32( v10, vkappa );
          v11 = vmulq_f32( v11, vkappa );

          // In-register transpose.
          //
          // Column 0-3
          vt0 = vtrn1q_f32( v0, v1 );
          vt1 = vtrn2q_f32( v0, v1 );
          vt2 = vtrn1q_f32( v2, v3 );
          vt3 = vtrn2q_f32( v2, v3 );
          v0 = (float32x4_t)vtrn1q_f64( (float64x2_t)vt0, (float64x2_t)vt2 );
          v1 = (float32x4_t)vtrn1q_f64( (float64x2_t)vt1, (float64x2_t)vt3 );
          v2 = (float32x4_t)vtrn2q_f64( (float64x2_t)vt0, (float64x2_t)vt2 );
          v3 = (float32x4_t)vtrn2q_f64( (float64x2_t)vt1, (float64x2_t)vt3 );
          // Column 4-7
          vt0 = vtrn1q_f32( v4, v5 );
          vt1 = vtrn2q_f32( v4, v5 );
          vt2 = vtrn1q_f32( v6, v7 );
          vt3 = vtrn2q_f32( v6, v7 );
          v4 = (float32x4_t)vtrn1q_f64( (float64x2_t)vt0, (float64x2_t)vt2 );
          v5 = (float32x4_t)vtrn1q_f64( (float64x2_t)vt1, (float64x2_t)vt3 );
          v6 = (float32x4_t)vtrn2q_f64( (float64x2_t)vt0, (float64x2_t)vt2 );
          v7 = (float32x4_t)vtrn2q_f64( (float64x2_t)vt1, (float64x2_t)vt3 );
          // Column 8-11
          vt0 = vtrn1q_f32( v8,  v9  );
          vt1 = vtrn2q_f32( v8,  v9  );
          vt2 = vtrn1q_f32( v10, v11 );
          vt3 = vtrn2q_f32( v10, v11 );
          v8  = (float32x4_t)vtrn1q_f64( (float64x2_t)vt0, (float64x2_t)vt2 );
          v9  = (float32x4_t)vtrn1q_f64( (float64x2_t)vt1, (float64x2_t)vt3 );
          v10 = (float32x4_t)vtrn2q_f64( (float64x2_t)vt0, (float64x2_t)vt2 );
          v11 = (float32x4_t)vtrn2q_f64( (float64x2_t)vt1, (float64x2_t)vt3 );

          vst1q_f32( p_loc + 0,  v0  );
          vst1q_f32( p_loc + 4,  v4  );
          vst1q_f32( p_loc + 8,  v8  );
          p_loc += ldp;

          vst1q_f32( p_loc + 0,  v1  );
          vst1q_f32( p_loc + 4,  v5  );
          vst1q_f32( p_loc + 8,  v9  );
          p_loc += ldp;

          vst1q_f32( p_loc + 0,  v2  );
          vst1q_f32( p_loc + 4,  v6  );
          vst1q_f32( p_loc + 8,  v10 );
          p_loc += ldp;

          vst1q_f32( p_loc + 0,  v3  );
          vst1q_f32( p_loc + 4,  v7  );
          vst1q_f32( p_loc + 8,  v11 );
          p_loc += ldp;
          a_loc += 4 * lda; // 4;
        }
        for ( ; k_left > 0; --k_left )
        {
          v0 = vld1q_lane_f32( a_loc + inca * 0 , v0, 0 );
          v0 = vld1q_lane_f32( a_loc + inca * 1 , v0, 1 );
          v0 = vld1q_lane_f32( a_loc + inca * 2 , v0, 2 );
          v0 = vld1q_lane_f32( a_loc + inca * 3 , v0, 3 );
          v1 = vld1q_lane_f32( a_loc + inca * 4 , v1, 0 );
          v1 = vld1q_lane_f32( a_loc + inca * 5 , v1, 1 );
          v1 = vld1q_lane_f32( a_loc + inca * 6 , v1, 2 );
          v1 = vld1q_lane_f32( a_loc + inca * 7 , v1, 3 );
          v2 = vld1q_lane_f32( a_loc + inca * 8 , v2, 0 );
          v2 = vld1q_lane_f32( a_loc + inca * 9 , v2, 1 );
          v2 = vld1q_lane_f32( a_loc + inca * 10, v2, 2 );
          v2 = vld1q_lane_f32( a_loc + inca * 11, v2, 3 );

          // Scale by kappa.
          v0 = vmulq_f32( v0, vkappa );
          v1 = vmulq_f32( v1, vkappa );
          v2 = vmulq_f32( v2, vkappa );

          vst1q_f32( p_loc + 0,  v0 );
          vst1q_f32( p_loc + 4,  v1 );
          vst1q_f32( p_loc + 8,  v2 );
          p_loc += ldp;
          a_loc += lda; // 1;
        }
      }
    }
  }
	else
	{
		bli_sscal2bbs_mxn
		(
		  conja,
		  cdim0,
		  k0,
		  kappa,
		  a,       inca, lda,
		  p, cdim_bcast, ldp
		);
	}

	bli_sset0s_edge
	(
	  cdim0*cdim_bcast, cdim_max*cdim_bcast,
	  k0, k0_max,
	  p, ldp
	);
}

