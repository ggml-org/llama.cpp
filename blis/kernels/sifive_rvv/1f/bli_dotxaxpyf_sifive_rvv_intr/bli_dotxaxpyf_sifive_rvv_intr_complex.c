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
#ifdef DOTXAXPYF

#define DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL(i)                                      \
    do {                                                                        \
        acol_vec = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (a_tmp + i * lda), vl); \
        acol_vec_r = VGET_V_F(PREC, LMUL, 2)(acol_vec, 0);                      \
        acol_vec_i = VGET_V_F(PREC, LMUL, 2)(acol_vec, 1);                      \
    } while (0)

#define DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL_STRIDED(i)                                                    \
    do {                                                                                              \
        acol_vec = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (a_tmp + i * lda), 2 * FLT_SIZE * inca, vl); \
        acol_vec_r = VGET_V_F(PREC, LMUL, 2)(acol_vec, 0);                                            \
        acol_vec_i = VGET_V_F(PREC, LMUL, 2)(acol_vec, 1);                                            \
    } while (0)

#define DOTXAXPYF_SIFIVE_RVV_LOOP_BODY_FIRST(LOAD_SUF, DF_CONJ_SUF, AF_CONJ_SUF)                                           \
    do {                                                                                                                    \
        DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(0);                                                                       \
        VCMUL_VV##DF_CONJ_SUF(PREC, LMUL, yacc0_r, yacc0_i, acol_vec_r, acol_vec_i, wvec_r, wvec_i, vl);                    \
        VCMUL_VF##AF_CONJ_SUF(PREC, LMUL, zacc_r, zacc_i, acol_vec_r, acol_vec_i, x[0 * incx].real, x[0 * incx].imag, vl);  \
        DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(1);                                                                       \
        VCMUL_VV##DF_CONJ_SUF(PREC, LMUL, yacc1_r, yacc1_i, acol_vec_r, acol_vec_i, wvec_r, wvec_i, vl);                    \
        VCMACC_VF##AF_CONJ_SUF(PREC, LMUL, zacc_r, zacc_i, x[1 * incx].real, x[1 * incx].imag, acol_vec_r, acol_vec_i, vl); \
        DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(2);                                                                       \
        VCMUL_VV##DF_CONJ_SUF(PREC, LMUL, yacc2_r, yacc2_i, acol_vec_r, acol_vec_i, wvec_r, wvec_i, vl);                    \
        VCMACC_VF##AF_CONJ_SUF(PREC, LMUL, zacc_r, zacc_i, x[2 * incx].real, x[2 * incx].imag, acol_vec_r, acol_vec_i, vl); \
        DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(3);                                                                       \
        VCMUL_VV##DF_CONJ_SUF(PREC, LMUL, yacc3_r, yacc3_i, acol_vec_r, acol_vec_i, wvec_r, wvec_i, vl);                    \
        VCMACC_VF##AF_CONJ_SUF(PREC, LMUL, zacc_r, zacc_i, x[3 * incx].real, x[3 * incx].imag, acol_vec_r, acol_vec_i, vl); \
    } while (0)

#define DOTXAXPYF_SIFIVE_RVV_LOOP_BODY(LOAD_SUF, DF_CONJ_SUF, AF_CONJ_SUF)                                                 \
    do {                                                                                                                    \
        DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(0);                                                                       \
        VCMACC_VV##DF_CONJ_SUF##_TU(PREC, LMUL, yacc0_r, yacc0_i, wvec_r, wvec_i, acol_vec_r, acol_vec_i, vl);              \
        VCMUL_VF##AF_CONJ_SUF(PREC, LMUL, zacc_r, zacc_i, acol_vec_r, acol_vec_i, x[0 * incx].real, x[0 * incx].imag, vl);  \
        DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(1);                                                                       \
        VCMACC_VV##DF_CONJ_SUF##_TU(PREC, LMUL, yacc1_r, yacc1_i, wvec_r, wvec_i, acol_vec_r, acol_vec_i, vl);              \
        VCMACC_VF##AF_CONJ_SUF(PREC, LMUL, zacc_r, zacc_i, x[1 * incx].real, x[1 * incx].imag, acol_vec_r, acol_vec_i, vl); \
        DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(2);                                                                       \
        VCMACC_VV##DF_CONJ_SUF##_TU(PREC, LMUL, yacc2_r, yacc2_i, wvec_r, wvec_i, acol_vec_r, acol_vec_i, vl);              \
        VCMACC_VF##AF_CONJ_SUF(PREC, LMUL, zacc_r, zacc_i, x[2 * incx].real, x[2 * incx].imag, acol_vec_r, acol_vec_i, vl); \
        DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(3);                                                                       \
        VCMACC_VV##DF_CONJ_SUF##_TU(PREC, LMUL, yacc3_r, yacc3_i, wvec_r, wvec_i, acol_vec_r, acol_vec_i, vl);              \
        VCMACC_VF##AF_CONJ_SUF(PREC, LMUL, zacc_r, zacc_i, x[3 * incx].real, x[3 * incx].imag, acol_vec_r, acol_vec_i, vl); \
    } while (0)

#define DOTXAXPYF_SIFIVE_RVV_CLEANUP_BODY_FIRST(LOAD_SUF, DF_CONJ_SUF, AF_CONJ_SUF)                                            \
    do {                                                                                                                        \
        switch (b) {                                                                                                            \
        case 3:                                                                                                                 \
            DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(2);                                                                       \
            VCMUL_VV##DF_CONJ_SUF(PREC, LMUL, yacc2_r, yacc2_i, acol_vec_r, acol_vec_i, wvec_r, wvec_i, vl);                    \
            VCMACC_VF##AF_CONJ_SUF(PREC, LMUL, zacc_r, zacc_i, x[2 * incx].real, x[2 * incx].imag, acol_vec_r, acol_vec_i, vl); \
        case 2:                                                                                                                 \
            DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(1);                                                                       \
            VCMUL_VV##DF_CONJ_SUF(PREC, LMUL, yacc1_r, yacc1_i, acol_vec_r, acol_vec_i, wvec_r, wvec_i, vl);                    \
            VCMACC_VF##AF_CONJ_SUF(PREC, LMUL, zacc_r, zacc_i, x[1 * incx].real, x[1 * incx].imag, acol_vec_r, acol_vec_i, vl); \
        case 1:                                                                                                                 \
            DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(0);                                                                       \
            VCMUL_VV##DF_CONJ_SUF(PREC, LMUL, yacc0_r, yacc0_i, acol_vec_r, acol_vec_i, wvec_r, wvec_i, vl);                    \
            VCMACC_VF##AF_CONJ_SUF(PREC, LMUL, zacc_r, zacc_i, x[0 * incx].real, x[0 * incx].imag, acol_vec_r, acol_vec_i, vl); \
        }                                                                                                                       \
    } while (0)

#define DOTXAXPYF_SIFIVE_RVV_CLEANUP_BODY(LOAD_SUF, DF_CONJ_SUF, AF_CONJ_SUF)                                                  \
    do {                                                                                                                        \
        switch (b) {                                                                                                            \
        case 3:                                                                                                                 \
            DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(2);                                                                       \
            VCMACC_VV##DF_CONJ_SUF##_TU(PREC, LMUL, yacc2_r, yacc2_i, wvec_r, wvec_i, acol_vec_r, acol_vec_i, vl);              \
            VCMACC_VF##AF_CONJ_SUF(PREC, LMUL, zacc_r, zacc_i, x[2 * incx].real, x[2 * incx].imag, acol_vec_r, acol_vec_i, vl); \
        case 2:                                                                                                                 \
            DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(1);                                                                       \
            VCMACC_VV##DF_CONJ_SUF##_TU(PREC, LMUL, yacc1_r, yacc1_i, wvec_r, wvec_i, acol_vec_r, acol_vec_i, vl);              \
            VCMACC_VF##AF_CONJ_SUF(PREC, LMUL, zacc_r, zacc_i, x[1 * incx].real, x[1 * incx].imag, acol_vec_r, acol_vec_i, vl); \
        case 1:                                                                                                                 \
            DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(0);                                                                       \
            VCMACC_VV##DF_CONJ_SUF##_TU(PREC, LMUL, yacc0_r, yacc0_i, wvec_r, wvec_i, acol_vec_r, acol_vec_i, vl);              \
            VCMACC_VF##AF_CONJ_SUF(PREC, LMUL, zacc_r, zacc_i, x[0 * incx].real, x[0 * incx].imag, acol_vec_r, acol_vec_i, vl); \
        }                                                                                                                       \
    } while (0)

#define DOTXAXPYF_SIFIVE_RVV_REDUCE(i)                                                                            \
    do {                                                                                                           \
        RVV_TYPE_F(PREC, m1) dot##i##_r = VFMV_S_F(PREC, m1)(0., 1);                                               \
        RVV_TYPE_F(PREC, m1) dot##i##_i = VFMV_S_F(PREC, m1)(0., 1);                                               \
        dot##i##_r = VF_REDUSUM_VS(PREC, LMUL)(yacc##i##_r, dot##i##_r, m);                                        \
        dot##i##_i = VF_REDUSUM_VS(PREC, LMUL)(yacc##i##_i, dot##i##_i, m);                                        \
        RVV_TYPE_F(PREC, m1) y##i##_r, y##i##_i;                                                                   \
        if (PASTEMAC(PRECISION_CHAR, eq0)(*beta)) {                                                                \
            if (bli_is_conj(conjatw))                                                                              \
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
            if (bli_is_conj(conjatw))                                                                              \
                VCMACC_VF_CONJ(PREC, m1, y##i##_r, y##i##_i, alpha->real, alpha->imag, dot##i##_r, dot##i##_i, 1); \
            else                                                                                                   \
                VCMACC_VF(PREC, m1, y##i##_r, y##i##_i, alpha->real, alpha->imag, dot##i##_r, dot##i##_i, 1);      \
            y[i * incy].real = VFMV_F_S(PREC)(y##i##_r);                                                           \
            y[i * incy].imag = VFMV_F_S(PREC)(y##i##_i);                                                           \
        }                                                                                                          \
    } while (0)

DOTXAXPYF(PRECISION_CHAR, void)
{
    // Computes y := beta * y + alpha * conjat(A^T) * conjx(x)
    
    (void) cntx; // Suppress unused parameter warnings
    const DATATYPE* restrict alpha = alpha_;
    const DATATYPE* restrict a = a_;
    const DATATYPE* restrict w = w_;
    const DATATYPE* restrict x = x_;
    const DATATYPE* restrict beta = beta_;
    DATATYPE* restrict y = y_;
    DATATYPE* restrict z = z_;

    if (b == 0) return;
    if (m == 0 || PASTEMAC(PRECISION_CHAR, eq0)(*alpha)) {
        if (PASTEMAC(PRECISION_CHAR, eq0)(*beta))
            SETV(PRECISION_CHAR)(BLIS_NO_CONJUGATE, b, beta, y, incy, NULL);
        else
            SCALV(PRECISION_CHAR)(BLIS_NO_CONJUGATE, b, beta, y, incy, NULL);
        return;
    }

    conj_t conjatw = BLIS_NO_CONJUGATE;
    conj_t conjax = BLIS_NO_CONJUGATE;
    if (bli_is_conj(conjw)) {
        bli_toggle_conj(&conjat);
        bli_toggle_conj(&conjw);
        bli_toggle_conj(&conjatw);
    }
    if (bli_is_conj(conjx)) {
        bli_toggle_conj(&conja);
        bli_toggle_conj(&conjx);
        bli_toggle_conj(&conjax);
    }

    while (b >= 4) {
        // Compute dot product of w with 4 columns of a.
        const DATATYPE* restrict a_tmp = a;
        const DATATYPE* restrict w_tmp = w;
        DATATYPE* restrict z_tmp = z;
        RVV_TYPE_F(PREC, LMUL) yacc0_r, yacc0_i, yacc1_r, yacc1_i,
                               yacc2_r, yacc2_i, yacc3_r, yacc3_i;
        bool first = true;
        size_t avl = m;
        while (avl) {
            size_t vl = VSETVL(PREC, LMUL)(avl);
            RVV_TYPE_FX(PREC, LMUL, 2) wvec, acol_vec;
            RVV_TYPE_F(PREC, LMUL) wvec_r, wvec_i, acol_vec_r, acol_vec_i;
            RVV_TYPE_F(PREC, LMUL) zacc_r, zacc_i;
            if (incw == 1)
                wvec = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) w_tmp, vl);
            else
                wvec = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) w_tmp, 2 * FLT_SIZE * incw, vl);
            wvec_r = VGET_V_F(PREC, LMUL, 2)(wvec, 0);
            wvec_i = VGET_V_F(PREC, LMUL, 2)(wvec, 1);

            if (first) {
                if (bli_is_conj(conjat)) {
                    if (bli_is_conj(conja)) {
                        if (inca == 1)
                            DOTXAXPYF_SIFIVE_RVV_LOOP_BODY_FIRST( , _CONJ, _CONJ);
                        else
                            DOTXAXPYF_SIFIVE_RVV_LOOP_BODY_FIRST(_STRIDED, _CONJ, _CONJ);
                    }
                    else {
                        if (inca == 1)
                            DOTXAXPYF_SIFIVE_RVV_LOOP_BODY_FIRST( , _CONJ, );
                        else
                            DOTXAXPYF_SIFIVE_RVV_LOOP_BODY_FIRST(_STRIDED, _CONJ, );
                    }
                }
                else {
                    if (bli_is_conj(conja)) {
                        if (inca == 1)
                            DOTXAXPYF_SIFIVE_RVV_LOOP_BODY_FIRST( , , _CONJ);
                        else
                            DOTXAXPYF_SIFIVE_RVV_LOOP_BODY_FIRST(_STRIDED, , _CONJ);
                    }
                    else {
                        if (inca == 1)
                            DOTXAXPYF_SIFIVE_RVV_LOOP_BODY_FIRST( , , );
                        else
                            DOTXAXPYF_SIFIVE_RVV_LOOP_BODY_FIRST(_STRIDED, , );
                    }
                }
                first = false;
            }
            else {
                if (bli_is_conj(conjat)) {
                    if (bli_is_conj(conja)) {
                        if (inca == 1)
                            DOTXAXPYF_SIFIVE_RVV_LOOP_BODY( , _CONJ, _CONJ);
                        else
                            DOTXAXPYF_SIFIVE_RVV_LOOP_BODY(_STRIDED, _CONJ, _CONJ);
                    }
                    else {
                        if (inca == 1)
                            DOTXAXPYF_SIFIVE_RVV_LOOP_BODY( , _CONJ, );
                        else
                            DOTXAXPYF_SIFIVE_RVV_LOOP_BODY(_STRIDED, _CONJ, );
                    }
                }
                else {
                    if (bli_is_conj(conja)) {
                        if (inca == 1)
                            DOTXAXPYF_SIFIVE_RVV_LOOP_BODY( , , _CONJ);
                        else
                            DOTXAXPYF_SIFIVE_RVV_LOOP_BODY(_STRIDED, , _CONJ);
                    }
                    else {
                        if (inca == 1)
                            DOTXAXPYF_SIFIVE_RVV_LOOP_BODY( , , );
                        else
                            DOTXAXPYF_SIFIVE_RVV_LOOP_BODY(_STRIDED, , );
                    }
                }
            }
              
            RVV_TYPE_FX(PREC, LMUL, 2) zvec;
            if (incz == 1)
                zvec = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) z_tmp, vl);
            else
                zvec = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) z_tmp, 2 * FLT_SIZE * incz, vl);
            RVV_TYPE_F(PREC, LMUL) zvec_r = VGET_V_F(PREC, LMUL, 2)(zvec, 0);
            RVV_TYPE_F(PREC, LMUL) zvec_i = VGET_V_F(PREC, LMUL, 2)(zvec, 1);
            if (bli_is_conj(conjax))
                VCMACC_VF_CONJ(PREC, LMUL, zvec_r, zvec_i, alpha->real, alpha->imag, zacc_r, zacc_i, vl);
            else
                VCMACC_VF(PREC, LMUL, zvec_r, zvec_i, alpha->real, alpha->imag, zacc_r, zacc_i, vl);
            zvec = VSET_V_F(PREC, LMUL, 2)(zvec, 0, zvec_r);
            zvec = VSET_V_F(PREC, LMUL, 2)(zvec, 1, zvec_i);
            if (incz == 1)
                VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) z_tmp, zvec, vl);
            else
                VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) z_tmp, 2 * FLT_SIZE * incz, zvec, vl);

            a_tmp += vl * inca;
            w_tmp += vl * incw;
            z_tmp += vl * incz;
            avl -= vl;
        }

        DOTXAXPYF_SIFIVE_RVV_REDUCE(0);
        DOTXAXPYF_SIFIVE_RVV_REDUCE(1);
        DOTXAXPYF_SIFIVE_RVV_REDUCE(2);
        DOTXAXPYF_SIFIVE_RVV_REDUCE(3);

        a += 4 * lda;
        x += 4 * incx;
        y += 4 * incy;
        b -= 4;
    }

    if (b > 0) {
        const DATATYPE* restrict a_tmp = a;
        const DATATYPE* restrict w_tmp = w;
        DATATYPE* restrict z_tmp = z;
        RVV_TYPE_F(PREC, LMUL) yacc0_r, yacc0_i, yacc1_r, yacc1_i, yacc2_r, yacc2_i;
        bool first = true;
        size_t avl = m;
        while (avl) {
            size_t vl = VSETVL(PREC, LMUL)(avl);
            RVV_TYPE_FX(PREC, LMUL, 2) wvec, acol_vec;
            RVV_TYPE_F(PREC, LMUL) wvec_r, wvec_i, acol_vec_r, acol_vec_i;
            RVV_TYPE_F(PREC, LMUL) zacc_r = VFMV_V_F(PREC, LMUL)(0, vl);
            RVV_TYPE_F(PREC, LMUL) zacc_i = VFMV_V_F(PREC, LMUL)(0, vl);
            if (incw == 1)
                wvec = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) w_tmp, vl);
            else
                wvec = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) w_tmp, 2 * FLT_SIZE * incw, vl);
            wvec_r = VGET_V_F(PREC, LMUL, 2)(wvec, 0);
            wvec_i = VGET_V_F(PREC, LMUL, 2)(wvec, 1);

            if (first) {
                if (bli_is_conj(conjat)) {
                    if (bli_is_conj(conja)) {
                        if (inca == 1)
                            DOTXAXPYF_SIFIVE_RVV_CLEANUP_BODY_FIRST( , _CONJ, _CONJ);
                        else
                            DOTXAXPYF_SIFIVE_RVV_CLEANUP_BODY_FIRST(_STRIDED, _CONJ, _CONJ);
                    }
                    else {
                        if (inca == 1)
                            DOTXAXPYF_SIFIVE_RVV_CLEANUP_BODY_FIRST( , _CONJ, );
                        else
                            DOTXAXPYF_SIFIVE_RVV_CLEANUP_BODY_FIRST(_STRIDED, _CONJ, );
                    }
                }
                else {
                    if (bli_is_conj(conja)) {
                        if (inca == 1)
                            DOTXAXPYF_SIFIVE_RVV_CLEANUP_BODY_FIRST( , , _CONJ);
                        else
                            DOTXAXPYF_SIFIVE_RVV_CLEANUP_BODY_FIRST(_STRIDED, , _CONJ);
                    }
                    else {
                        if (inca == 1)
                            DOTXAXPYF_SIFIVE_RVV_CLEANUP_BODY_FIRST( , , );
                        else
                            DOTXAXPYF_SIFIVE_RVV_CLEANUP_BODY_FIRST(_STRIDED, , );
                    }
                }
                first = false;
            }
            else {
                if (bli_is_conj(conjat)) {
                    if (bli_is_conj(conja)) {
                        if (inca == 1)
                            DOTXAXPYF_SIFIVE_RVV_CLEANUP_BODY( , _CONJ, _CONJ);
                        else
                            DOTXAXPYF_SIFIVE_RVV_CLEANUP_BODY(_STRIDED, _CONJ, _CONJ);
                    }
                    else {
                        if (inca == 1)
                            DOTXAXPYF_SIFIVE_RVV_CLEANUP_BODY( , _CONJ, );
                        else
                            DOTXAXPYF_SIFIVE_RVV_CLEANUP_BODY(_STRIDED, _CONJ, );
                    }
                }
                else {
                    if (bli_is_conj(conja)) {
                        if (inca == 1)
                            DOTXAXPYF_SIFIVE_RVV_CLEANUP_BODY( , , _CONJ);
                        else
                            DOTXAXPYF_SIFIVE_RVV_CLEANUP_BODY(_STRIDED, , _CONJ);
                    }
                    else {
                        if (inca == 1)
                            DOTXAXPYF_SIFIVE_RVV_CLEANUP_BODY( , , );
                        else
                            DOTXAXPYF_SIFIVE_RVV_CLEANUP_BODY(_STRIDED, , );
                    }
                }
            }
              
            RVV_TYPE_FX(PREC, LMUL, 2) zvec;
            if (incz == 1)
                zvec = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) z_tmp, vl);
            else
                zvec = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) z_tmp, 2 * FLT_SIZE * incz, vl);
            RVV_TYPE_F(PREC, LMUL) zvec_r = VGET_V_F(PREC, LMUL, 2)(zvec, 0);
            RVV_TYPE_F(PREC, LMUL) zvec_i = VGET_V_F(PREC, LMUL, 2)(zvec, 1);
            if (bli_is_conj(conjax))
                VCMACC_VF_CONJ(PREC, LMUL, zvec_r, zvec_i, alpha->real, alpha->imag, zacc_r, zacc_i, vl);
            else
                VCMACC_VF(PREC, LMUL, zvec_r, zvec_i, alpha->real, alpha->imag, zacc_r, zacc_i, vl);
            zvec = VSET_V_F(PREC, LMUL, 2)(zvec, 0, zvec_r);
            zvec = VSET_V_F(PREC, LMUL, 2)(zvec, 1, zvec_i);
            if (incz == 1)
                VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) z_tmp, zvec, vl);
            else
                VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) z_tmp, 2 * FLT_SIZE * incz, zvec, vl);

            a_tmp += vl * inca;
            w_tmp += vl * incw;
            z_tmp += vl * incz;
            avl -= vl;
        }

        switch (b) {
        case 3:
            DOTXAXPYF_SIFIVE_RVV_REDUCE(2);
        case 2:
            DOTXAXPYF_SIFIVE_RVV_REDUCE(1);
        case 1:
            DOTXAXPYF_SIFIVE_RVV_REDUCE(0);
        }
    }
    return;
}

#undef DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL
#undef DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL_STRIDED
#undef DOTXAXPYF_SIFIVE_RVV_LOOP_BODY_FIRST
#undef DOTXAXPYF_SIFIVE_RVV_LOOP_BODY
#undef DOTXAXPYF_SIFIVE_RVV_CLEANUP_BODY_FIRST
#undef DOTXAXPYF_SIFIVE_RVV_CLEANUP_BODY
#undef DOTXAXPYF_SIFIVE_RVV_REDUCE

#endif // DOTXAXPYF
