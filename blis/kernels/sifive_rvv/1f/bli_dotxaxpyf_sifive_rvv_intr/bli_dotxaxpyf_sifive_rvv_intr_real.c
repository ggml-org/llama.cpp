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

#define DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL(i)                   \
    do {                                                     \
        acol_vec = VLE_V_F(PREC, LMUL)(a_tmp + i * lda, vl); \
    } while (0)

#define DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL_STRIDED(i)                             \
    do {                                                                       \
        acol_vec = VLSE_V_F(PREC, LMUL)(a_tmp + i * lda, FLT_SIZE * inca, vl); \
    } while (0)

#define DOTXAXPYF_SIFIVE_RVV_LOOP_BODY_FIRST(LOAD_SUF)                \
    do {                                                               \
        DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(0);                  \
        yacc0 = VFMUL_VV(PREC, LMUL)(acol_vec, wvec, vl);              \
        zacc = VFMUL_VF(PREC, LMUL)(acol_vec, x[0 * incx], vl);        \
        DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(1);                  \
        yacc1 = VFMUL_VV(PREC, LMUL)(acol_vec, wvec, vl);              \
        zacc = VFMACC_VF(PREC, LMUL)(zacc, x[1 * incx], acol_vec, vl); \
        DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(2);                  \
        yacc2 = VFMUL_VV(PREC, LMUL)(acol_vec, wvec, vl);              \
        zacc = VFMACC_VF(PREC, LMUL)(zacc, x[2 * incx], acol_vec, vl); \
        DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(3);                  \
        yacc3 = VFMUL_VV(PREC, LMUL)(acol_vec, wvec, vl);              \
        zacc = VFMACC_VF(PREC, LMUL)(zacc, x[3 * incx], acol_vec, vl); \
    } while (0)

#define DOTXAXPYF_SIFIVE_RVV_LOOP_BODY(LOAD_SUF)                      \
    do {                                                               \
        DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(0);                  \
        yacc0 = VFMACC_VV_TU(PREC, LMUL)(yacc0, acol_vec, wvec, vl);   \
        zacc = VFMUL_VF(PREC, LMUL)(acol_vec, x[0 * incx], vl);        \
        DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(1);                  \
        yacc1 = VFMACC_VV_TU(PREC, LMUL)(yacc1, acol_vec, wvec, vl);   \
        zacc = VFMACC_VF(PREC, LMUL)(zacc, x[1 * incx], acol_vec, vl); \
        DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(2);                  \
        yacc2 = VFMACC_VV_TU(PREC, LMUL)(yacc2, acol_vec, wvec, vl);   \
        zacc = VFMACC_VF(PREC, LMUL)(zacc, x[2 * incx], acol_vec, vl); \
        DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(3);                  \
        yacc3 = VFMACC_VV_TU(PREC, LMUL)(yacc3, acol_vec, wvec, vl);   \
        zacc = VFMACC_VF(PREC, LMUL)(zacc, x[3 * incx], acol_vec, vl); \
    } while (0)

#define DOTXAXPYF_SIFIVE_RVV_CLEANUP_BODY_FIRST(LOAD_SUF)                 \
    do {                                                                   \
        switch (b) {                                                       \
        case 3:                                                            \
            DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(2);                  \
            yacc2 = VFMUL_VV(PREC, LMUL)(acol_vec, wvec, vl);              \
            zacc = VFMACC_VF(PREC, LMUL)(zacc, x[2 * incx], acol_vec, vl); \
        case 2:                                                            \
            DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(1);                  \
            yacc1 = VFMUL_VV(PREC, LMUL)(acol_vec, wvec, vl);              \
            zacc = VFMACC_VF(PREC, LMUL)(zacc, x[1 * incx], acol_vec, vl); \
        case 1:                                                            \
            DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(0);                  \
            yacc0 = VFMUL_VV(PREC, LMUL)(acol_vec, wvec, vl);              \
            zacc = VFMACC_VF(PREC, LMUL)(zacc, x[0 * incx], acol_vec, vl); \
        }                                                                  \
    } while (0)

#define DOTXAXPYF_SIFIVE_RVV_CLEANUP_BODY(LOAD_SUF)                       \
    do {                                                                   \
        switch (b) {                                                       \
        case 3:                                                            \
            DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(2);                  \
            yacc2 = VFMACC_VV_TU(PREC, LMUL)(yacc2, acol_vec, wvec, vl);   \
            zacc = VFMACC_VF(PREC, LMUL)(zacc, x[2 * incx], acol_vec, vl); \
        case 2:                                                            \
            DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(1);                  \
            yacc1 = VFMACC_VV_TU(PREC, LMUL)(yacc1, acol_vec, wvec, vl);   \
            zacc = VFMACC_VF(PREC, LMUL)(zacc, x[1 * incx], acol_vec, vl); \
        case 1:                                                            \
            DOTXAXPYF_SIFIVE_RVV_LOAD_ACOL##LOAD_SUF(0);                  \
            yacc0 = VFMACC_VV_TU(PREC, LMUL)(yacc0, acol_vec, wvec, vl);   \
            zacc = VFMACC_VF(PREC, LMUL)(zacc, x[0 * incx], acol_vec, vl); \
        }                                                                  \
    } while (0)

#define DOTXAXPYF_SIFIVE_RVV_REDUCE(i)                                     \
    do {                                                                    \
        RVV_TYPE_F(PREC, m1) dot##i = VFMV_S_F(PREC, m1)(0., 1);            \
        dot##i = VF_REDUSUM_VS(PREC, LMUL)(yacc##i, dot##i, m);             \
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

DOTXAXPYF(PRECISION_CHAR, void)
{
    // Computes y := beta * y + alpha * conjat(A^T) * conjw(w)
    //          z :=        z + alpha * conja(A)    * conjx(x)
    
    (void) conjat; // Suppress unused parameter warnings
    (void) conja;
    (void) conjw;
    (void) conjx;
    (void) cntx;
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

    while (b >= 4) {
        // Process 4 columns of a at a time.
        const DATATYPE* restrict a_tmp = a;
        const DATATYPE* restrict w_tmp = w;
        DATATYPE* restrict z_tmp = z;
        RVV_TYPE_F(PREC, LMUL) yacc0, yacc1, yacc2, yacc3;
        bool first = true;
        size_t avl = m;
        while (avl) {
            size_t vl = VSETVL(PREC, LMUL)(avl);
            RVV_TYPE_F(PREC, LMUL) wvec, acol_vec;
            RVV_TYPE_F(PREC, LMUL) zacc;
            if (incw == 1)
                wvec = VLE_V_F(PREC, LMUL)(w_tmp, vl);
            else
                wvec = VLSE_V_F(PREC, LMUL)(w_tmp, FLT_SIZE * incw, vl);
            if (first) {
                if (inca == 1)
                    DOTXAXPYF_SIFIVE_RVV_LOOP_BODY_FIRST( );
                else
                    DOTXAXPYF_SIFIVE_RVV_LOOP_BODY_FIRST(_STRIDED);
                first = false;
            }
            else {
                if (inca == 1)
                    DOTXAXPYF_SIFIVE_RVV_LOOP_BODY( );
                else
                    DOTXAXPYF_SIFIVE_RVV_LOOP_BODY(_STRIDED);
            }

            RVV_TYPE_F(PREC, LMUL) zvec;
            if (incz == 1)
                zvec = VLE_V_F(PREC, LMUL)(z_tmp, vl);
            else
                zvec = VLSE_V_F(PREC, LMUL)(z_tmp, FLT_SIZE * incz, vl);
            zvec = VFMACC_VF(PREC, LMUL)(zvec, *alpha, zacc, vl);
            if (incz == 1)
                VSE_V_F(PREC, LMUL)(z_tmp, zvec, vl);
            else
                VSSE_V_F(PREC, LMUL)(z_tmp, FLT_SIZE * incz, zvec, vl);
              
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
        RVV_TYPE_F(PREC, LMUL) yacc0, yacc1, yacc2;
        bool first = true;
        size_t avl = m;
        while (avl) {
            size_t vl = VSETVL(PREC, LMUL)(avl);
            RVV_TYPE_F(PREC, LMUL) wvec, acol_vec;
            RVV_TYPE_F(PREC, LMUL) zacc = VFMV_V_F(PREC, LMUL)(0, vl);
            if (incw == 1)
                wvec = VLE_V_F(PREC, LMUL)(w_tmp, vl);
            else
                wvec = VLSE_V_F(PREC, LMUL)(w_tmp, FLT_SIZE * incw, vl);
            if (first) {
                if (inca == 1)
                    DOTXAXPYF_SIFIVE_RVV_CLEANUP_BODY_FIRST( );
                else
                    DOTXAXPYF_SIFIVE_RVV_CLEANUP_BODY_FIRST(_STRIDED);
                first = false;
            }
            else {
                if (inca == 1)
                    DOTXAXPYF_SIFIVE_RVV_CLEANUP_BODY( );
                else
                    DOTXAXPYF_SIFIVE_RVV_CLEANUP_BODY(_STRIDED);
            }

            RVV_TYPE_F(PREC, LMUL) zvec;
            if (incz == 1)
                zvec = VLE_V_F(PREC, LMUL)(z_tmp, vl);
            else
                zvec = VLSE_V_F(PREC, LMUL)(z_tmp, FLT_SIZE * incz, vl);
            zvec = VFMACC_VF(PREC, LMUL)(zvec, *alpha, zacc, vl);
            if (incz == 1)
                VSE_V_F(PREC, LMUL)(z_tmp, zvec, vl);
            else
                VSSE_V_F(PREC, LMUL)(z_tmp, FLT_SIZE * incz, zvec, vl);
              
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
