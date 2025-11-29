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

#ifdef BLIS_SANDBOX_POWER10

#include "vector_int_macros.h"

#define I4_ACCUMULATE \
    __builtin_mma_xvi4ger8pp (&acc0, ca[0], rb[0]); \
    __builtin_mma_xvi4ger8pp (&acc1, ca[0], rb[1]); \
    __builtin_mma_xvi4ger8pp (&acc2, ca[0], rb[2]); \
    __builtin_mma_xvi4ger8pp (&acc3, ca[0], rb[3]); \
    __builtin_mma_xvi4ger8pp (&acc4, ca[1], rb[0]); \
    __builtin_mma_xvi4ger8pp (&acc5, ca[1], rb[1]); \
    __builtin_mma_xvi4ger8pp (&acc6, ca[1], rb[2]); \
    __builtin_mma_xvi4ger8pp (&acc7, ca[1], rb[3]);

#define I4_INCREMENT \
    A0+=32; \
    B0+=64;

#define I4_AB_PRODUCT \
    LOAD_VECTORS \
    I4_INCREMENT \
    I4_ACCUMULATE

void bli_i4gemm_power10_mma_8x16
    (
              dim_t      m,
              dim_t      n,
              dim_t      k,
        //const int32_t*   alpha,
        //const nibbles*   a,
        //const nibbles*   b,
        //const int32_t*   beta,
        //      int32_t*   c, inc_t rs_c0, inc_t cs_c0,
        const void*      alpha,
        const void*      a,
        const void*      b,
        const void*      beta,
              void*      c, inc_t rs_c0, inc_t cs_c0,
        const auxinfo_t* data,
        const cntx_t*    cntx
    )
{

    uint64_t k_iter = (k-1) / 4;
    uint64_t k_left = (k-1) % 4;

    uint64_t rs_c   = rs_c0;
    //uint64_t cs_c   = cs_c0;

    const nibbles* restrict A0 = a;
    const nibbles* restrict B0 = b;
          int*     restrict C0 = c;

    int alpha_ = *((int32_t*)alpha),
        beta_  = *((int32_t*)beta);

    iv4sf_t result[4];
    iv4sf_t *rowC;

    // accumulators that will hold the matrix product
    __vector_quad acc0, acc1, acc2, acc3,
                  acc4, acc5, acc6, acc7;

    vec_t *ca = (vec_t *) A0;
    vec_t *rb = (vec_t *) B0;

    __builtin_mma_xvi4ger8 (&acc0, ca[0], rb[0]);
    __builtin_mma_xvi4ger8 (&acc1, ca[0], rb[1]);
    __builtin_mma_xvi4ger8 (&acc2, ca[0], rb[2]);
    __builtin_mma_xvi4ger8 (&acc3, ca[0], rb[3]);
    __builtin_mma_xvi4ger8 (&acc4, ca[1], rb[0]);
    __builtin_mma_xvi4ger8 (&acc5, ca[1], rb[1]);
    __builtin_mma_xvi4ger8 (&acc6, ca[1], rb[2]);
    __builtin_mma_xvi4ger8 (&acc7, ca[1], rb[3]);

    I4_INCREMENT

    // k loop (unrolled by 4)
    for (int k = 0; k<k_iter; k++)
    {
        I4_AB_PRODUCT
        I4_AB_PRODUCT
        I4_AB_PRODUCT
        I4_AB_PRODUCT
    }

    // edge loop
    for (int k = 0; k<k_left; k++)
    {
        I4_AB_PRODUCT
    }

    // handle beta cases
    if (beta_ != 0.0)
    {
        SAVE_ACC(iv4sf_t, &acc0, rs_c,  0     );
        SAVE_ACC(iv4sf_t, &acc1, rs_c,  4     );
        SAVE_ACC(iv4sf_t, &acc2, rs_c,  8     );
        SAVE_ACC(iv4sf_t, &acc3, rs_c, 12     );
        SAVE_ACC(iv4sf_t, &acc4, rs_c,    4*rs_c);
        SAVE_ACC(iv4sf_t, &acc5, rs_c,  4+4*rs_c);
        SAVE_ACC(iv4sf_t, &acc6, rs_c,  8+4*rs_c);
        SAVE_ACC(iv4sf_t, &acc7, rs_c, 12+4*rs_c);
    }
    else
    {
        SAVE_ACC_bz(iv4sf_t, &acc0, rs_c,  0     );
        SAVE_ACC_bz(iv4sf_t, &acc1, rs_c,  4     );
        SAVE_ACC_bz(iv4sf_t, &acc2, rs_c,  8     );
        SAVE_ACC_bz(iv4sf_t, &acc3, rs_c, 12     );
        SAVE_ACC_bz(iv4sf_t, &acc4, rs_c,    4*rs_c);
        SAVE_ACC_bz(iv4sf_t, &acc5, rs_c,  4+4*rs_c);
        SAVE_ACC_bz(iv4sf_t, &acc6, rs_c,  8+4*rs_c);
        SAVE_ACC_bz(iv4sf_t, &acc7, rs_c, 12+4*rs_c);
    }
}
#endif // BLIS_SANDBOX_POWER10
