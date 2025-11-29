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

#define DOTXF_SIFIVE_RVV_LOAD_ACOL(i)                                          \
    do {                                                                        \
        acol_vec = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (a_tmp + i * lda), vl); \
        acol_vec_r = VGET_V_F(PREC, LMUL, 2)(acol_vec, 0);                      \
        acol_vec_i = VGET_V_F(PREC, LMUL, 2)(acol_vec, 1);                      \
    } while (0)

#define DOTXF_SIFIVE_RVV_LOAD_ACOL_STRIDED(i)                                                        \
    do {                                                                                              \
        acol_vec = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (a_tmp + i * lda), 2 * FLT_SIZE * inca, vl); \
        acol_vec_r = VGET_V_F(PREC, LMUL, 2)(acol_vec, 0);                                            \
        acol_vec_i = VGET_V_F(PREC, LMUL, 2)(acol_vec, 1);                                            \
    } while (0)

#define DOTXF_SIFIVE_RVV_LOOP_BODY_FIRST(LOAD_SUF, CONJ_SUF)                                       \
    do {                                                                                            \
        DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(0);                                                   \
        VCMUL_VV##CONJ_SUF(PREC, LMUL, acc0_r, acc0_i, acol_vec_r, acol_vec_i, xvec_r, xvec_i, vl); \
        DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(1);                                                   \
        VCMUL_VV##CONJ_SUF(PREC, LMUL, acc1_r, acc1_i, acol_vec_r, acol_vec_i, xvec_r, xvec_i, vl); \
        DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(2);                                                   \
        VCMUL_VV##CONJ_SUF(PREC, LMUL, acc2_r, acc2_i, acol_vec_r, acol_vec_i, xvec_r, xvec_i, vl); \
        DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(3);                                                   \
        VCMUL_VV##CONJ_SUF(PREC, LMUL, acc3_r, acc3_i, acol_vec_r, acol_vec_i, xvec_r, xvec_i, vl); \
        DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(4);                                                   \
        VCMUL_VV##CONJ_SUF(PREC, LMUL, acc4_r, acc4_i, acol_vec_r, acol_vec_i, xvec_r, xvec_i, vl); \
        DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(5);                                                   \
        VCMUL_VV##CONJ_SUF(PREC, LMUL, acc5_r, acc5_i, acol_vec_r, acol_vec_i, xvec_r, xvec_i, vl); \
    } while (0)

#define DOTXF_SIFIVE_RVV_LOOP_BODY(LOAD_SUF, CONJ_SUF)                                                   \
    do {                                                                                                  \
        DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(0);                                                         \
        VCMACC_VV##CONJ_SUF##_TU(PREC, LMUL, acc0_r, acc0_i, xvec_r, xvec_i, acol_vec_r, acol_vec_i, vl); \
        DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(1);                                                         \
        VCMACC_VV##CONJ_SUF##_TU(PREC, LMUL, acc1_r, acc1_i, xvec_r, xvec_i, acol_vec_r, acol_vec_i, vl); \
        DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(2);                                                         \
        VCMACC_VV##CONJ_SUF##_TU(PREC, LMUL, acc2_r, acc2_i, xvec_r, xvec_i, acol_vec_r, acol_vec_i, vl); \
        DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(3);                                                         \
        VCMACC_VV##CONJ_SUF##_TU(PREC, LMUL, acc3_r, acc3_i, xvec_r, xvec_i, acol_vec_r, acol_vec_i, vl); \
        DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(4);                                                         \
        VCMACC_VV##CONJ_SUF##_TU(PREC, LMUL, acc4_r, acc4_i, xvec_r, xvec_i, acol_vec_r, acol_vec_i, vl); \
        DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(5);                                                         \
        VCMACC_VV##CONJ_SUF##_TU(PREC, LMUL, acc5_r, acc5_i, xvec_r, xvec_i, acol_vec_r, acol_vec_i, vl); \
    } while (0)

#define DOTXF_SIFIVE_RVV_CLEANUP_BODY_FIRST(LOAD_SUF, CONJ_SUF)                                            \
    do {                                                                                                    \
        switch (b) {                                                                                        \
            case 5:                                                                                         \
                DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(4);                                                   \
                VCMUL_VV##CONJ_SUF(PREC, LMUL, acc4_r, acc4_i, acol_vec_r, acol_vec_i, xvec_r, xvec_i, vl); \
            case 4:                                                                                         \
                DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(3);                                                   \
                VCMUL_VV##CONJ_SUF(PREC, LMUL, acc3_r, acc3_i, acol_vec_r, acol_vec_i, xvec_r, xvec_i, vl); \
            case 3:                                                                                         \
                DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(2);                                                   \
                VCMUL_VV##CONJ_SUF(PREC, LMUL, acc2_r, acc2_i, acol_vec_r, acol_vec_i, xvec_r, xvec_i, vl); \
            case 2:                                                                                         \
                DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(1);                                                   \
                VCMUL_VV##CONJ_SUF(PREC, LMUL, acc1_r, acc1_i, acol_vec_r, acol_vec_i, xvec_r, xvec_i, vl); \
            case 1:                                                                                         \
                DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(0);                                                   \
                VCMUL_VV##CONJ_SUF(PREC, LMUL, acc0_r, acc0_i, acol_vec_r, acol_vec_i, xvec_r, xvec_i, vl); \
        }                                                                                                   \
    } while (0)

#define DOTXF_SIFIVE_RVV_CLEANUP_BODY(LOAD_SUF, CONJ_SUF)                                                        \
    do {                                                                                                          \
        switch (b) {                                                                                              \
            case 5:                                                                                               \
                DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(4);                                                         \
                VCMACC_VV##CONJ_SUF##_TU(PREC, LMUL, acc4_r, acc4_i, xvec_r, xvec_i, acol_vec_r, acol_vec_i, vl); \
            case 4:                                                                                               \
                DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(3);                                                         \
                VCMACC_VV##CONJ_SUF##_TU(PREC, LMUL, acc3_r, acc3_i, xvec_r, xvec_i, acol_vec_r, acol_vec_i, vl); \
            case 3:                                                                                               \
                DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(2);                                                         \
                VCMACC_VV##CONJ_SUF##_TU(PREC, LMUL, acc2_r, acc2_i, xvec_r, xvec_i, acol_vec_r, acol_vec_i, vl); \
            case 2:                                                                                               \
                DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(1);                                                         \
                VCMACC_VV##CONJ_SUF##_TU(PREC, LMUL, acc1_r, acc1_i, xvec_r, xvec_i, acol_vec_r, acol_vec_i, vl); \
            case 1:                                                                                               \
                DOTXF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(0);                                                         \
                VCMACC_VV##CONJ_SUF##_TU(PREC, LMUL, acc0_r, acc0_i, xvec_r, xvec_i, acol_vec_r, acol_vec_i, vl); \
        }                                                                                                         \
    } while (0)

#define DOTXF_SIFIVE_RVV_REDUCE(i)                                                                                \
    do {                                                                                                           \
        RVV_TYPE_F(PREC, m1) dot##i##_r = VFMV_S_F(PREC, m1)(0., 1);                                               \
        RVV_TYPE_F(PREC, m1) dot##i##_i = VFMV_S_F(PREC, m1)(0., 1);                                               \
        dot##i##_r = VF_REDUSUM_VS(PREC, LMUL)(acc##i##_r, dot##i##_r, m);                                         \
        dot##i##_i = VF_REDUSUM_VS(PREC, LMUL)(acc##i##_i, dot##i##_i, m);                                         \
        RVV_TYPE_F(PREC, m1) y##i##_r, y##i##_i;                                                                   \
        if (PASTEMAC(PRECISION_CHAR, eq0)(*beta)) {                                                                \
            if (bli_is_conj(conjatx))                                                                              \
                VCMUL_VF_CONJ(PREC, m1, y##i##_r, y##i##_i, dot##i##_r, dot##i##_i, alpha->real, alpha->imag, 1);  \
            else                                                                                                   \
                VCMUL_VF(PREC, m1, y##i##_r, y##i##_i, dot##i##_r, dot##i##_i, alpha->real, alpha->imag, 1);       \
            y[i * incy].real = VFMV_F_S(PREC)(y##i##_r);                                                           \
            y[i * incy].imag = VFMV_F_S(PREC)(y##i##_i);                                                           \
        }                                                                                                          \
        else {                                                                                                     \
            PASTEMAC(PRECISION_CHAR, scals)(*beta, y[i * incy])                                                    \
            y##i##_r = VFMV_S_F(PREC, m1)(y[i * incy].real, 1);                                                    \
            y##i##_i = VFMV_S_F(PREC, m1)(y[i * incy].imag, 1);                                                    \
            if (bli_is_conj(conjatx))                                                                              \
                VCMACC_VF_CONJ(PREC, m1, y##i##_r, y##i##_i, alpha->real, alpha->imag, dot##i##_r, dot##i##_i, 1); \
            else                                                                                                   \
                VCMACC_VF(PREC, m1, y##i##_r, y##i##_i, alpha->real, alpha->imag, dot##i##_r, dot##i##_i, 1);      \
            y[i * incy].real = VFMV_F_S(PREC)(y##i##_r);                                                           \
            y[i * incy].imag = VFMV_F_S(PREC)(y##i##_i);                                                           \
        }                                                                                                          \
    } while (0)

DOTXF(PRECISION_CHAR, void)
{
    // Computes y := beta * y + alpha * conjat(A^T) * conjx(x)
    
    (void) cntx; // Suppress unused parameter warnings
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

    conj_t conjatx = BLIS_NO_CONJUGATE;
    if (bli_is_conj(conjx)) {
        bli_toggle_conj(&conjat);
        bli_toggle_conj(&conjx);
        bli_toggle_conj(&conjatx);
    }

    while (b >= 6) {
        // Compute dot product of x with 6 columns of a.
        const DATATYPE* restrict a_tmp = a;
        const DATATYPE* restrict x_tmp = x;
        RVV_TYPE_F(PREC, LMUL) acc0_r, acc0_i, acc1_r, acc1_i, acc2_r, acc2_i,
                               acc3_r, acc3_i, acc4_r, acc4_i, acc5_r, acc5_i;
        RVV_TYPE_FX(PREC, LMUL, 2) xvec, acol_vec;
        RVV_TYPE_F(PREC, LMUL) xvec_r, xvec_i, acol_vec_r, acol_vec_i;
        bool first = true;
        size_t avl = m;
        while (avl) {
            size_t vl = VSETVL(PREC, LMUL)(avl);
            if (incx == 1)
                xvec = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) x_tmp, vl);
            else
                xvec = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) x_tmp, 2 * FLT_SIZE * incx, vl);
            xvec_r = VGET_V_F(PREC, LMUL, 2)(xvec, 0);
            xvec_i = VGET_V_F(PREC, LMUL, 2)(xvec, 1);

            if (first) {
                if (bli_is_conj(conjat)) {
                    if (inca == 1)
                        DOTXF_SIFIVE_RVV_LOOP_BODY_FIRST( , _CONJ);
                    else
                        DOTXF_SIFIVE_RVV_LOOP_BODY_FIRST(_STRIDED, _CONJ);
                }
                else {
                    if (inca == 1)
                        DOTXF_SIFIVE_RVV_LOOP_BODY_FIRST( , );
                    else
                        DOTXF_SIFIVE_RVV_LOOP_BODY_FIRST(_STRIDED, );
                }
                first = false;
            }
            else {
                if (bli_is_conj(conjat)) {
                    if (inca == 1)
                        DOTXF_SIFIVE_RVV_LOOP_BODY( , _CONJ);
                    else
                        DOTXF_SIFIVE_RVV_LOOP_BODY(_STRIDED, _CONJ);
                }
                else {
                    if (inca == 1)
                        DOTXF_SIFIVE_RVV_LOOP_BODY( , );
                    else
                        DOTXF_SIFIVE_RVV_LOOP_BODY(_STRIDED, );
                }
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
        RVV_TYPE_F(PREC, LMUL) acc0_r, acc0_i, acc1_r, acc1_i, acc2_r, acc2_i,
                               acc3_r, acc3_i, acc4_r, acc4_i;
        RVV_TYPE_FX(PREC, LMUL, 2) xvec, acol_vec;
        RVV_TYPE_F(PREC, LMUL) xvec_r, xvec_i, acol_vec_r, acol_vec_i;
        bool first = true;
        size_t avl = m;
        while (avl) {
            size_t vl = VSETVL(PREC, LMUL)(avl);
            if (incx == 1)
                xvec = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) x_tmp, vl);
            else
                xvec = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) x_tmp, 2 * FLT_SIZE * incx, vl);
            xvec_r = VGET_V_F(PREC, LMUL, 2)(xvec, 0);
            xvec_i = VGET_V_F(PREC, LMUL, 2)(xvec, 1);

            if (first) {
                if (bli_is_conj(conjat)) {
                    if (inca == 1)
                        DOTXF_SIFIVE_RVV_CLEANUP_BODY_FIRST( , _CONJ);
                    else
                        DOTXF_SIFIVE_RVV_CLEANUP_BODY_FIRST(_STRIDED, _CONJ);
                }
                else {
                    if (inca == 1)
                        DOTXF_SIFIVE_RVV_CLEANUP_BODY_FIRST( , );
                    else
                        DOTXF_SIFIVE_RVV_CLEANUP_BODY_FIRST(_STRIDED, );
                }
                first = false;
            }
            else {
                if (bli_is_conj(conjat)) {
                    if (inca == 1)
                        DOTXF_SIFIVE_RVV_CLEANUP_BODY( , _CONJ);
                    else
                        DOTXF_SIFIVE_RVV_CLEANUP_BODY(_STRIDED, _CONJ);
                }
                else {
                    if (inca == 1)
                        DOTXF_SIFIVE_RVV_CLEANUP_BODY( , );
                    else
                        DOTXF_SIFIVE_RVV_CLEANUP_BODY(_STRIDED, );
                }
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
