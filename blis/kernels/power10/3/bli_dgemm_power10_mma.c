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


#include "vector_int_macros.h"

#define D_ASSEMBLE_VEC_PAIR \
        __builtin_mma_assemble_pair (&colA_1, ca[1], ca[0]); \
        __builtin_mma_assemble_pair (&colA_2, ca[3], ca[2]);

#define D_ACCUMULATE \
        __builtin_mma_xvf64gerpp (&acc0, colA_1, rb[0]); \
        __builtin_mma_xvf64gerpp (&acc1, colA_1, rb[1]); \
        __builtin_mma_xvf64gerpp (&acc2, colA_1, rb[2]); \
        __builtin_mma_xvf64gerpp (&acc3, colA_1, rb[3]); \
        __builtin_mma_xvf64gerpp (&acc4, colA_2, rb[0]); \
        __builtin_mma_xvf64gerpp (&acc5, colA_2, rb[1]); \
        __builtin_mma_xvf64gerpp (&acc6, colA_2, rb[2]); \
        __builtin_mma_xvf64gerpp (&acc7, colA_2, rb[3]);

#define D_INCREMENT \
        A0+=8; \
        B0+=8;

#define D_AB_PRODUCT \
        LOAD_VECTORS \
        D_ASSEMBLE_VEC_PAIR \
        D_INCREMENT \
        D_ACCUMULATE


void bli_dgemm_power10_mma_8x8
    (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
    )
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k / 4;
    uint64_t k_left = k % 4;

    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    GEMM_UKR_SETUP_CT( d, 8, 8, true );

    const double* restrict A0 = a;
    const double* restrict B0 = b;
          double* restrict C0 = c;

    double alpha_ = *((double*)alpha),
           beta_  = *((double*)beta);

    dv4sf_t result[4];
    dv4sf_t *rowC;

    /* 8 accumulator registers that will be used to store the result.

       Each accumulator register is mapped to 4 vector registers.
       Illustration:

            acc0 = [  vs0
                      vs1
                      vs3
                      vs4  ]

        These registers are used to store the result of an outer product
        instruction (general outer product instruction syntax: xv???ger??). */
    __vector_quad acc0, acc1, acc2, acc3,
                  acc4, acc5, acc6, acc7;

    // initialize the accumulators to zeros
    __builtin_mma_xxsetaccz(&acc0);
    __builtin_mma_xxsetaccz(&acc1);
    __builtin_mma_xxsetaccz(&acc2);
    __builtin_mma_xxsetaccz(&acc3);
    __builtin_mma_xxsetaccz(&acc4);
    __builtin_mma_xxsetaccz(&acc5);
    __builtin_mma_xxsetaccz(&acc6);
    __builtin_mma_xxsetaccz(&acc7);

    /* 2 vector pairs are necessary for a double precision outer product
       instruction. */
    __vector_pair colA_1,
                  colA_2;

    /* Prefetch C so that it stays in cache */
    PREFETCH1 (C0, 0);
    PREFETCH1 (C0 + rs_c, 0);
    PREFETCH1 (C0 + rs_c + rs_c, 0);
    PREFETCH1 (C0 + rs_c + rs_c + rs_c, 0);
    PREFETCH1 (C0, 128);
    PREFETCH1 (C0 + rs_c, 128);
    PREFETCH1 (C0 + rs_c + rs_c, 128);
    PREFETCH1 (C0 + rs_c + rs_c + rs_c, 128);

    /* Load elements into vector registers */
    vec_t *ca = (vec_t *) A0;
    vec_t *rb = (vec_t *) B0;

    /* Each accumulator represents a matrix of size
       4 x ( 16 / (datatype size in bytes) )  (vector register size = 16B)

       Thus in the case of double, the accumulate registers represent a 4x2
       matrix. However, a vector register can hold at most 2 doubles. Thus, if
       we performed an outer product using 2 vector register, we can only get a
       2x2 matrix. Therefore, we must create a vector register pair in order
       to get the desired 4x2 matrix.

    */
    D_ASSEMBLE_VEC_PAIR

    // k loop (unrolled by 4)
    for (int k = 0; k<k_iter; k++)
    {
        D_AB_PRODUCT
        D_AB_PRODUCT
        D_AB_PRODUCT
        D_AB_PRODUCT
    }

    // edge loop
    for (int k = 0; k<k_left; k++)
    {
        D_AB_PRODUCT
    }

    // handle beta cases
    if (beta_ != 0.0)
    {
        SAVE_ACC(dv4sf_t, &acc0, rs_c, 0       );
        SAVE_ACC(dv4sf_t, &acc1, rs_c, 2       );
        SAVE_ACC(dv4sf_t, &acc2, rs_c, 4       );
        SAVE_ACC(dv4sf_t, &acc3, rs_c, 6       );
        SAVE_ACC(dv4sf_t, &acc4, rs_c,   4*rs_c);
        SAVE_ACC(dv4sf_t, &acc5, rs_c, 2+4*rs_c);
        SAVE_ACC(dv4sf_t, &acc6, rs_c, 4+4*rs_c);
        SAVE_ACC(dv4sf_t, &acc7, rs_c, 6+4*rs_c);
    }
    else
    {
        SAVE_ACC_bz(dv4sf_t, &acc0, rs_c, 0       );
        SAVE_ACC_bz(dv4sf_t, &acc1, rs_c, 2       );
        SAVE_ACC_bz(dv4sf_t, &acc2, rs_c, 4       );
        SAVE_ACC_bz(dv4sf_t, &acc3, rs_c, 6       );
        SAVE_ACC_bz(dv4sf_t, &acc4, rs_c,   4*rs_c);
        SAVE_ACC_bz(dv4sf_t, &acc5, rs_c, 2+4*rs_c);
        SAVE_ACC_bz(dv4sf_t, &acc6, rs_c, 4+4*rs_c);
        SAVE_ACC_bz(dv4sf_t, &acc7, rs_c, 6+4*rs_c);
    }

    GEMM_UKR_FLUSH_CT( d );
}
