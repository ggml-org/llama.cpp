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
#ifdef GEMMTRSM

#define GEMMTRSM_IMPL_NAME_(PRECISION_CHAR) bli_##PRECISION_CHAR##gemmtrsm_sifive_rvv_intr
#define GEMMTRSM_IMPL_NAME(PRECISION_CHAR) GEMMTRSM_IMPL_NAME_(PRECISION_CHAR)

static void GEMMTRSM_IMPL_NAME(PRECISION_CHAR)
     (
             dim_t              m,
             dim_t              n,
             dim_t              k,
       const DATATYPE* restrict beta,
       const DATATYPE* restrict a, inc_t rsa, inc_t csa,
       const DATATYPE* restrict b, inc_t rsb,
             DATATYPE* restrict c, inc_t rsc,
       const DATATYPE* restrict a11, inc_t rsa11, inc_t csa11,
             DATATYPE* restrict c11, inc_t rsc11, inc_t csc11
     )
{
    // This function computes inv(a11) * (beta * c - a * b)
    // and stores the result in c and c11.
    
    RVV_TYPE_F(PREC, LMUL) ab0, ab1, ab2, ab3, ab4, ab5, ab6;
    // gemm step
    if (m <= 0 || n <= 0 || k < 0)
        return;
    else if (k == 0) {
        if (PASTEMAC(PRECISION_CHAR, eq0)(*beta)) {
            RVV_TYPE_F(PREC, LMUL) zero_splat = VFMV_V_F(PREC, LMUL)(0., n);
            switch (m) {
            case 7:
                ab6 = zero_splat;
            case 6:
                ab5 = zero_splat;
            case 5:
                ab4 = zero_splat;
            case 4:
                ab3 = zero_splat;
            case 3:
                ab2 = zero_splat;
            case 2:
                ab1 = zero_splat;
            case 1:
                ab0 = zero_splat;
            }
        }
        else {
            RVV_TYPE_F(PREC, LMUL) c0;
            switch (m) {
            case 7:
                c0 = VLE_V_F(PREC, LMUL)(c + 6 * rsc, n);
                ab6 = VFMUL_VF(PREC, LMUL)(c0, *beta, n);
            case 6:
                c0 = VLE_V_F(PREC, LMUL)(c + 5 * rsc, n);
                ab5 = VFMUL_VF(PREC, LMUL)(c0, *beta, n);
            case 5:
                c0 = VLE_V_F(PREC, LMUL)(c + 4 * rsc, n);
                ab4 = VFMUL_VF(PREC, LMUL)(c0, *beta, n);
            case 4:
                c0 = VLE_V_F(PREC, LMUL)(c + 3 * rsc, n);
                ab3 = VFMUL_VF(PREC, LMUL)(c0, *beta, n);
            case 3:
                c0 = VLE_V_F(PREC, LMUL)(c + 2 * rsc, n);
                ab2 = VFMUL_VF(PREC, LMUL)(c0, *beta, n);
            case 2:
                c0 = VLE_V_F(PREC, LMUL)(c + 1 * rsc, n);
                ab1 = VFMUL_VF(PREC, LMUL)(c0, *beta, n);
            case 1:
                c0 = VLE_V_F(PREC, LMUL)(c + 0 * rsc, n);
                ab0 = VFMUL_VF(PREC, LMUL)(c0, *beta, n);
            }
        }
    }
    else {
        bool first = true;
        for (dim_t i = 0; i < k; ++i) {
            RVV_TYPE_F(PREC, LMUL) b0 = VLE_V_F(PREC, LMUL)(b, n);
            if (first) {
                switch (m) {
                case 7:
                    ab6 = VFMUL_VF(PREC, LMUL)(b0, a[6 * rsa], n);
                case 6:
                    ab5 = VFMUL_VF(PREC, LMUL)(b0, a[5 * rsa], n);
                case 5:
                    ab4 = VFMUL_VF(PREC, LMUL)(b0, a[4 * rsa], n);
                case 4:
                    ab3 = VFMUL_VF(PREC, LMUL)(b0, a[3 * rsa], n);
                case 3:
                    ab2 = VFMUL_VF(PREC, LMUL)(b0, a[2 * rsa], n);
                case 2:
                    ab1 = VFMUL_VF(PREC, LMUL)(b0, a[1 * rsa], n);
                case 1:
                    ab0 = VFMUL_VF(PREC, LMUL)(b0, a[0 * rsa], n);
                }
                first = false;
            }
            else {
                switch (m) {
                case 7:
                    ab6 = VFMACC_VF(PREC, LMUL)(ab6, a[6 * rsa], b0, n);
                case 6:
                    ab5 = VFMACC_VF(PREC, LMUL)(ab5, a[5 * rsa], b0, n);
                case 5:
                    ab4 = VFMACC_VF(PREC, LMUL)(ab4, a[4 * rsa], b0, n);
                case 4:
                    ab3 = VFMACC_VF(PREC, LMUL)(ab3, a[3 * rsa], b0, n);
                case 3:
                    ab2 = VFMACC_VF(PREC, LMUL)(ab2, a[2 * rsa], b0, n);
                case 2:
                    ab1 = VFMACC_VF(PREC, LMUL)(ab1, a[1 * rsa], b0, n);
                case 1:
                    ab0 = VFMACC_VF(PREC, LMUL)(ab0, a[0 * rsa], b0, n);
                }
            }

            a += csa;
            b += rsb;
        }

        if (PASTEMAC(PRECISION_CHAR, eq0)(*beta)) {
            switch (m) {
            case 7:
                ab6 = VFNEG_VF(PREC, LMUL)(ab6, n);
            case 6:
                ab5 = VFNEG_VF(PREC, LMUL)(ab5, n);
            case 5:
                ab4 = VFNEG_VF(PREC, LMUL)(ab4, n);
            case 4:
                ab3 = VFNEG_VF(PREC, LMUL)(ab3, n);
            case 3:
                ab2 = VFNEG_VF(PREC, LMUL)(ab2, n);
            case 2:
                ab1 = VFNEG_VF(PREC, LMUL)(ab1, n);
            case 1:
                ab0 = VFNEG_VF(PREC, LMUL)(ab0, n);
            }
        }
        else {
            RVV_TYPE_F(PREC, LMUL) c0;
            switch (m) {
            case 7:
                c0 = VLE_V_F(PREC, LMUL)(c + 6 * rsc, n);
                ab6 = VFMSAC_VF(PREC, LMUL)(ab6, *beta, c0, n);
            case 6:
                c0 = VLE_V_F(PREC, LMUL)(c + 5 * rsc, n);
                ab5 = VFMSAC_VF(PREC, LMUL)(ab5, *beta, c0, n);
            case 5:
                c0 = VLE_V_F(PREC, LMUL)(c + 4 * rsc, n);
                ab4 = VFMSAC_VF(PREC, LMUL)(ab4, *beta, c0, n);
            case 4:
                c0 = VLE_V_F(PREC, LMUL)(c + 3 * rsc, n);
                ab3 = VFMSAC_VF(PREC, LMUL)(ab3, *beta, c0, n);
            case 3:
                c0 = VLE_V_F(PREC, LMUL)(c + 2 * rsc, n);
                ab2 = VFMSAC_VF(PREC, LMUL)(ab2, *beta, c0, n);
            case 2:
                c0 = VLE_V_F(PREC, LMUL)(c + 1 * rsc, n);
                ab1 = VFMSAC_VF(PREC, LMUL)(ab1, *beta, c0, n);
            case 1:
                c0 = VLE_V_F(PREC, LMUL)(c + 0 * rsc, n);
                ab0 = VFMSAC_VF(PREC, LMUL)(ab0, *beta, c0, n);
            }
        }
    }
    
    // trsm step
    ab0 = VFMUL_VF(PREC, LMUL)(ab0, a11[0 * rsa11], n);
    VSE_V_F(PREC, LMUL)(c + 0 * rsc, ab0, n);
    if (csc11 == 1)
        VSE_V_F(PREC, LMUL)(c11 + 0 * rsc11, ab0, n);
    else
        VSSE_V_F(PREC, LMUL)(c11 + 0 * rsc11, FLT_SIZE * csc11, ab0, n);
    if (m == 1) return;
    switch (m) {
    case 7:
        ab6 = VFNMSAC_VF(PREC, LMUL)(ab6, a11[6 * rsa11], ab0, n);
    case 6:
        ab5 = VFNMSAC_VF(PREC, LMUL)(ab5, a11[5 * rsa11], ab0, n);
    case 5:
        ab4 = VFNMSAC_VF(PREC, LMUL)(ab4, a11[4 * rsa11], ab0, n);
    case 4:
        ab3 = VFNMSAC_VF(PREC, LMUL)(ab3, a11[3 * rsa11], ab0, n);
    case 3:
        ab2 = VFNMSAC_VF(PREC, LMUL)(ab2, a11[2 * rsa11], ab0, n);
    case 2:
        ab1 = VFNMSAC_VF(PREC, LMUL)(ab1, a11[1 * rsa11], ab0, n);
    }
    a11 += csa11;

    ab1 = VFMUL_VF(PREC, LMUL)(ab1, a11[1 * rsa11], n);
    VSE_V_F(PREC, LMUL)(c + 1 * rsc, ab1, n);
    if (csc11 == 1)
        VSE_V_F(PREC, LMUL)(c11 + 1 * rsc11, ab1, n);
    else
        VSSE_V_F(PREC, LMUL)(c11 + 1 * rsc11, FLT_SIZE * csc11, ab1, n);
    if (m == 2) return;
    switch (m) {
    case 7:
        ab6 = VFNMSAC_VF(PREC, LMUL)(ab6, a11[6 * rsa11], ab1, n);
    case 6:
        ab5 = VFNMSAC_VF(PREC, LMUL)(ab5, a11[5 * rsa11], ab1, n);
    case 5:
        ab4 = VFNMSAC_VF(PREC, LMUL)(ab4, a11[4 * rsa11], ab1, n);
    case 4:
        ab3 = VFNMSAC_VF(PREC, LMUL)(ab3, a11[3 * rsa11], ab1, n);
    case 3:
        ab2 = VFNMSAC_VF(PREC, LMUL)(ab2, a11[2 * rsa11], ab1, n);
    }
    a11 += csa11;

    ab2 = VFMUL_VF(PREC, LMUL)(ab2, a11[2 * rsa11], n);
    VSE_V_F(PREC, LMUL)(c + 2 * rsc, ab2, n);
    if (csc11 == 1)
        VSE_V_F(PREC, LMUL)(c11 + 2 * rsc11, ab2, n);
    else
        VSSE_V_F(PREC, LMUL)(c11 + 2 * rsc11, FLT_SIZE * csc11, ab2, n);
    if (m == 3) return;
    switch (m) {
    case 7:
        ab6 = VFNMSAC_VF(PREC, LMUL)(ab6, a11[6 * rsa11], ab2, n);
    case 6:
        ab5 = VFNMSAC_VF(PREC, LMUL)(ab5, a11[5 * rsa11], ab2, n);
    case 5:
        ab4 = VFNMSAC_VF(PREC, LMUL)(ab4, a11[4 * rsa11], ab2, n);
    case 4:
        ab3 = VFNMSAC_VF(PREC, LMUL)(ab3, a11[3 * rsa11], ab2, n);
    }
    a11 += csa11;

    ab3 = VFMUL_VF(PREC, LMUL)(ab3, a11[3 * rsa11], n);
    VSE_V_F(PREC, LMUL)(c + 3 * rsc, ab3, n);
    if (csc11 == 1)
        VSE_V_F(PREC, LMUL)(c11 + 3 * rsc11, ab3, n);
    else
        VSSE_V_F(PREC, LMUL)(c11 + 3 * rsc11, FLT_SIZE * csc11, ab3, n);
    if (m == 4) return;
    switch (m) {
    case 7:
        ab6 = VFNMSAC_VF(PREC, LMUL)(ab6, a11[6 * rsa11], ab3, n);
    case 6:
        ab5 = VFNMSAC_VF(PREC, LMUL)(ab5, a11[5 * rsa11], ab3, n);
    case 5:
        ab4 = VFNMSAC_VF(PREC, LMUL)(ab4, a11[4 * rsa11], ab3, n);
    }
    a11 += csa11;

    ab4 = VFMUL_VF(PREC, LMUL)(ab4, a11[4 * rsa11], n);
    VSE_V_F(PREC, LMUL)(c + 4 * rsc, ab4, n);
    if (csc11 == 1)
        VSE_V_F(PREC, LMUL)(c11 + 4 * rsc11, ab4, n);
    else
        VSSE_V_F(PREC, LMUL)(c11 + 4 * rsc11, FLT_SIZE * csc11, ab4, n);
    if (m == 5) return;
    switch (m) {
    case 7:
        ab6 = VFNMSAC_VF(PREC, LMUL)(ab6, a11[6 * rsa11], ab4, n);
    case 6:
        ab5 = VFNMSAC_VF(PREC, LMUL)(ab5, a11[5 * rsa11], ab4, n);
    }
    a11 += csa11;

    ab5 = VFMUL_VF(PREC, LMUL)(ab5, a11[5 * rsa11], n);
    VSE_V_F(PREC, LMUL)(c + 5 * rsc, ab5, n);
    if (csc11 == 1)
        VSE_V_F(PREC, LMUL)(c11 + 5 * rsc11, ab5, n);
    else
        VSSE_V_F(PREC, LMUL)(c11 + 5 * rsc11, FLT_SIZE * csc11, ab5, n);
    if (m == 6) return;
    ab6 = VFNMSAC_VF(PREC, LMUL)(ab6, a11[6 * rsa11], ab5, n);
    a11 += csa11;

    ab6 = VFMUL_VF(PREC, LMUL)(ab6, a11[6 * rsa11], n);
    VSE_V_F(PREC, LMUL)(c + 6 * rsc, ab6, n);
    if (csc11 == 1)
        VSE_V_F(PREC, LMUL)(c11 + 6 * rsc11, ab6, n);
    else
        VSSE_V_F(PREC, LMUL)(c11 + 6 * rsc11, FLT_SIZE * csc11, ab6, n);
    return;
}

GEMMTRSM(GEMMTRSM_L, PRECISION_CHAR, void)
{
    const DATATYPE* restrict alpha = alpha_;
    const DATATYPE* restrict a10 = a10_;
    const DATATYPE* restrict a11 = a11_;
    const DATATYPE* restrict b01 = b01_;
    DATATYPE* restrict b11 = b11_;
    DATATYPE* restrict c11 = c11_;

    GEMMTRSM_IMPL_NAME(PRECISION_CHAR)
    (
      m, n, k,
      alpha,
      a10, 1, PACKMR,
      b01, PACKNR,
      b11, PACKNR,
      a11, 1, PACKMR,
      c11, rsc, csc
    );

    return;
}

GEMMTRSM(GEMMTRSM_U, PRECISION_CHAR, void)
{
    const DATATYPE* restrict alpha = alpha_;
    const DATATYPE* restrict a12 = a12_;
    const DATATYPE* restrict a11 = a11_;
    const DATATYPE* restrict b21 = b21_;
    DATATYPE* restrict b11 = b11_;
    DATATYPE* restrict c11 = c11_;

    GEMMTRSM_IMPL_NAME(PRECISION_CHAR)
    (
      m, n, k,
      alpha,
      a12 + (m - 1), -1, PACKMR,
      b21, PACKNR,
      b11 + (m - 1) * PACKNR, -PACKNR,
      a11 + (m - 1) + (m - 1) * PACKMR, -1, -PACKMR,
      c11 + (m - 1) * rsc, -rsc, csc
    );

    return;
}

#undef GEMMTRSM_IMPL_NAME_
#undef GEMMTRSM_IMPL_NAME

#endif // GEMMTRSM
