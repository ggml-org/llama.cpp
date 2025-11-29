/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, SiFive, Inc.

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

// clang-format off
#ifdef DOTXF

#define DOTXF_SIFIVE_RVV_LOAD_ACOL(i)                                         \
    do {                                                                       \
        acol_vec = VLE_V_F(PREC, LMUL)(a_tmp + i * lda, vl);                   \
    } while (0)

#define DOTXF_SIFIVE_RVV_LOAD_ACOL_STRIDED(i)                                 \
    do {                                                                       \
        acol_vec = VLSE_V_F(PREC, LMUL)(a_tmp + i * lda, FLT_SIZE * inca, vl); \
    } while (0)

#define DOTXF_SIFIVE_RVV_LOOP_BODY_FIRST(LOAD_SUF)      \
    do {                                                 \
        DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(0);        \
        acc0 = VFMUL_VV(PREC, LMUL)(acol_vec, xvec, vl); \
        DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(1);        \
        acc1 = VFMUL_VV(PREC, LMUL)(acol_vec, xvec, vl); \
        DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(2);        \
        acc2 = VFMUL_VV(PREC, LMUL)(acol_vec, xvec, vl); \
        DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(3);        \
        acc3 = VFMUL_VV(PREC, LMUL)(acol_vec, xvec, vl); \
        DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(4);        \
        acc4 = VFMUL_VV(PREC, LMUL)(acol_vec, xvec, vl); \
        DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(5);        \
        acc5 = VFMUL_VV(PREC, LMUL)(acol_vec, xvec, vl); \
    } while (0)

#define DOTXF_SIFIVE_RVV_LOOP_BODY(LOAD_SUF)                      \
    do {                                                           \
        DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(0);                  \
        acc0 = VFMACC_VV_TU(PREC, LMUL)(acc0, acol_vec, xvec, vl); \
        DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(1);                  \
        acc1 = VFMACC_VV_TU(PREC, LMUL)(acc1, acol_vec, xvec, vl); \
        DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(2);                  \
        acc2 = VFMACC_VV_TU(PREC, LMUL)(acc2, acol_vec, xvec, vl); \
        DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(3);                  \
        acc3 = VFMACC_VV_TU(PREC, LMUL)(acc3, acol_vec, xvec, vl); \
        DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(4);                  \
        acc4 = VFMACC_VV_TU(PREC, LMUL)(acc4, acol_vec, xvec, vl); \
        DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(5);                  \
        acc5 = VFMACC_VV_TU(PREC, LMUL)(acc5, acol_vec, xvec, vl); \
    } while (0)

#define DOTXF_SIFIVE_RVV_CLEANUP_BODY_FIRST(LOAD_SUF)           \
    do {                                                         \
        switch (b) {                                             \
            case 5:                                              \
                DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(4);        \
                acc4 = VFMUL_VV(PREC, LMUL)(acol_vec, xvec, vl); \
            case 4:                                              \
                DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(3);        \
                acc3 = VFMUL_VV(PREC, LMUL)(acol_vec, xvec, vl); \
            case 3:                                              \
                DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(2);        \
                acc2 = VFMUL_VV(PREC, LMUL)(acol_vec, xvec, vl); \
            case 2:                                              \
                DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(1);        \
                acc1 = VFMUL_VV(PREC, LMUL)(acol_vec, xvec, vl); \
            case 1:                                              \
                DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(0);        \
                acc0 = VFMUL_VV(PREC, LMUL)(acol_vec, xvec, vl); \
        }                                                        \
    } while (0)

#define DOTXF_SIFIVE_RVV_CLEANUP_BODY(LOAD_SUF)                           \
    do {                                                                   \
        switch (b) {                                                       \
            case 5:                                                        \
                DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(4);                  \
                acc4 = VFMACC_VV_TU(PREC, LMUL)(acc4, acol_vec, xvec, vl); \
            case 4:                                                        \
                DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(3);                  \
                acc3 = VFMACC_VV_TU(PREC, LMUL)(acc3, acol_vec, xvec, vl); \
            case 3:                                                        \
                DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(2);                  \
                acc2 = VFMACC_VV_TU(PREC, LMUL)(acc2, acol_vec, xvec, vl); \
            case 2:                                                        \
                DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(1);                  \
                acc1 = VFMACC_VV_TU(PREC, LMUL)(acc1, acol_vec, xvec, vl); \
            case 1:                                                        \
                DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(0);                  \
                acc0 = VFMACC_VV_TU(PREC, LMUL)(acc0, acol_vec, xvec, vl); \
        }                                                                  \
    } while (0)

#define DOTXF_SIFIVE_RVV_REDUCE(i)                                         \
    do {                                                                    \
        RVV_TYPE_F(PREC, m1) dot##i = VFMV_S_F(PREC, m1)(0., 1);            \
        dot##i = VF_REDUSUM_VS(PREC, LMUL)(acc##i, dot##i, m);              \
        if (PASTEMAC(PRECISION_CHAR, eq0)(*beta)) {                         \
            dot##i = VFMUL_VF(PREC, m1)(dot##i, *alpha, 1);                 \
            y[i * incy] = VFMV_F_S(PREC)(dot##i);                           \
        }                                                                   \
        else {                                                              \
            y[i * incy] *= *beta;                                           \
            RVV_TYPE_F(PREC, m1) y##i = VFMV_S_F(PREC, m1)(y[i * incy], 1); \
            y##i = VFMACC_VF(PREC, m1)(y##i, *alpha, dot##i, 1);            \
            y[i * incy] = VFMV_F_S(PREC)(y##i);                             \
        }                                                                   \
    } while (0)

DOTXF(PRECISION_CHAR, void)
{
    // Computes y := beta * y + alpha * conjat(A^T) * conjx(x)
    
    (void) conjat; // Suppress unused parameter warnings
    (void) conjx;
    (void) cntx;
    const DATATYPE* restrict alpha = alpha_;
    const DATATYPE* restrict a = a_;
    const DATATYPE* restrict x = x_;
    const DATATYPE* restrict beta = beta_;
    DATATYPE* restrict y = y_;

    if (b == 0) return;
    if (m == 0 || PASTEMAC(PRECISION_CHAR, eq0)(*alpha)) {
        if (PASTEMAC(PRECISION_CHAR, eq0)(*beta))
            SETV(PRECISION_CHAR)(BLIS_NO_CONJUGATE, b, beta, y, incy, NULL);
        else
            SCALV(PRECISION_CHAR)(BLIS_NO_CONJUGATE, b, beta, y, incy, NULL);
        return;
    }

    while (b >= 6) {
        // Compute dot product of x with 6 columns of a.
        const DATATYPE* restrict a_tmp = a;
        const DATATYPE* restrict x_tmp = x;
        RVV_TYPE_F(PREC, LMUL) acc0, acc1, acc2, acc3, acc4, acc5;
        RVV_TYPE_F(PREC, LMUL) xvec, acol_vec;
        bool first = true;
        size_t avl = m;
        while (avl) {
            size_t vl = VSETVL(PREC, LMUL)(avl);
            if (incx == 1)
                xvec = VLE_V_F(PREC, LMUL)(x_tmp, vl);
            else
                xvec = VLSE_V_F(PREC, LMUL)(x_tmp, FLT_SIZE * incx, vl);
            if (first) {
                if (inca == 1)
                    DOTXF_SIFIVE_RVV_LOOP_BODY_FIRST();
                else
                    DOTXF_SIFIVE_RVV_LOOP_BODY_FIRST(_STRIDED);
                first = false;
            }
            else {
                if (inca == 1)
                    DOTXF_SIFIVE_RVV_LOOP_BODY();
                else
                    DOTXF_SIFIVE_RVV_LOOP_BODY(_STRIDED);
            }
              
            a_tmp += vl * inca;
            x_tmp += vl * incx;
            avl -= vl;
        }

        DOTXF_SIFIVE_RVV_REDUCE(0);
        DOTXF_SIFIVE_RVV_REDUCE(1);
        DOTXF_SIFIVE_RVV_REDUCE(2);
        DOTXF_SIFIVE_RVV_REDUCE(3);
        DOTXF_SIFIVE_RVV_REDUCE(4);
        DOTXF_SIFIVE_RVV_REDUCE(5);

        a += 6 * lda;
        y += 6 * incy;
        b -= 6;
    }

    if (b > 0) {
        const DATATYPE* restrict a_tmp = a;
        const DATATYPE* restrict x_tmp = x;
        RVV_TYPE_F(PREC, LMUL) acc0, acc1, acc2, acc3, acc4;
        RVV_TYPE_F(PREC, LMUL) xvec, acol_vec;
        bool first = true;
        size_t avl = m;
        while (avl) {
            size_t vl = VSETVL(PREC, LMUL)(avl);
            if (incx == 1)
                xvec = VLE_V_F(PREC, LMUL)(x_tmp, vl);
            else
                xvec = VLSE_V_F(PREC, LMUL)(x_tmp, FLT_SIZE * incx, vl);
            if (first) {
                if (inca == 1)
                    DOTXF_SIFIVE_RVV_CLEANUP_BODY_FIRST();
                else
                    DOTXF_SIFIVE_RVV_CLEANUP_BODY_FIRST(_STRIDED);
                first = false;
            }
            else {
                if (inca == 1)
                    DOTXF_SIFIVE_RVV_CLEANUP_BODY();
                else
                    DOTXF_SIFIVE_RVV_CLEANUP_BODY(_STRIDED);
            }

            a_tmp += vl * inca;
            x_tmp += vl * incx;
            avl -= vl;
        }

        switch (b) {
            case 5:
                DOTXF_SIFIVE_RVV_REDUCE(4);
            case 4:
                DOTXF_SIFIVE_RVV_REDUCE(3);
            case 3:
                DOTXF_SIFIVE_RVV_REDUCE(2);
            case 2:
                DOTXF_SIFIVE_RVV_REDUCE(1);
            case 1:
                DOTXF_SIFIVE_RVV_REDUCE(0);
        }

    }
    return;
}

#undef DOTXF_SIFIVE_RVV_LOAD_ACOL
#undef DOTXF_SIFIVE_RVV_LOAD_ACOL_STRIDED
#undef DOTXF_SIFIVE_RVV_LOOP_BODY_FIRST
#undef DOTXF_SIFIVE_RVV_LOOP_BODY
#undef DOTXF_SIFIVE_RVV_CLEANUP_BODY_FIRST
#undef DOTXF_SIFIVE_RVV_CLEANUP_BODY
#undef DOTXF_SIFIVE_RVV_REDUCE

#endif // DOTXF
