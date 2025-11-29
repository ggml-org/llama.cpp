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
#ifdef GEMM

GEMM(PRECISION_CHAR, void)
{
    (void) data; // Suppress unused parameter warnings
    (void) cntx;
    const DATATYPE* restrict alpha = alpha_;
    const DATATYPE* restrict a = a_;
    const DATATYPE* restrict b = b_;
    const DATATYPE* restrict beta = beta_;
    DATATYPE* restrict c = c_;

    if (m <= 0 || n <= 0 || k < 0)
        return;
    else if (k == 0) {
        if (PASTEMAC(PRECISION_CHAR, eq0)(*beta)) {
            RVV_TYPE_F(PREC, LMUL) zero_splat = VFMV_V_F(PREC, LMUL)(0., n);
            for (dim_t i = 0; i < m; ++i) {
                if (csc == 1)
                    VSE_V_F(PREC, LMUL)(c + i * rsc, zero_splat, n);
                else
                    VSSE_V_F(PREC, LMUL)(c + i * rsc, FLT_SIZE * csc, zero_splat, n);
            }
        }
        else {
            for (dim_t i = 0; i < m; ++i) {
                RVV_TYPE_F(PREC, LMUL) c0;
                if (csc == 1)
                    c0 = VLE_V_F(PREC, LMUL)(c + i * rsc, n);
                else
                    c0 = VLSE_V_F(PREC, LMUL)(c + i * rsc, FLT_SIZE * csc, n);
                c0 = VFMUL_VF(PREC, LMUL)(c0, *beta, n);
                if (csc == 1)
                    VSE_V_F(PREC, LMUL)(c + i * rsc, c0, n);
                else
                    VSSE_V_F(PREC, LMUL)(c + i * rsc, FLT_SIZE * csc, c0, n);
            }
        }
    }
    else if (m == 7) {
        RVV_TYPE_F(PREC, LMUL) ab0, ab1, ab2, ab3, ab4, ab5, ab6;
        bool first = true;
        for (dim_t i = 0; i < k; ++i) {
            RVV_TYPE_F(PREC, LMUL) b0 = VLE_V_F(PREC, LMUL)(b, n);
            if (first) {
                ab0 = VFMUL_VF(PREC, LMUL)(b0, a[0], n);
                ab1 = VFMUL_VF(PREC, LMUL)(b0, a[1], n);
                ab2 = VFMUL_VF(PREC, LMUL)(b0, a[2], n);
                ab3 = VFMUL_VF(PREC, LMUL)(b0, a[3], n);
                ab4 = VFMUL_VF(PREC, LMUL)(b0, a[4], n);
                ab5 = VFMUL_VF(PREC, LMUL)(b0, a[5], n);
                ab6 = VFMUL_VF(PREC, LMUL)(b0, a[6], n);
                first = false;
            }
            else {
                ab0 = VFMACC_VF(PREC, LMUL)(ab0, a[0], b0, n);
                ab1 = VFMACC_VF(PREC, LMUL)(ab1, a[1], b0, n);
                ab2 = VFMACC_VF(PREC, LMUL)(ab2, a[2], b0, n);
                ab3 = VFMACC_VF(PREC, LMUL)(ab3, a[3], b0, n);
                ab4 = VFMACC_VF(PREC, LMUL)(ab4, a[4], b0, n);
                ab5 = VFMACC_VF(PREC, LMUL)(ab5, a[5], b0, n);
                ab6 = VFMACC_VF(PREC, LMUL)(ab6, a[6], b0, n);
            }

            a += PACKMR;
            b += PACKNR;
        }

        if (PASTEMAC(PRECISION_CHAR, eq0)(*beta)) {
            ab0 = VFMUL_VF(PREC, LMUL)(ab0, *alpha, n);
            ab1 = VFMUL_VF(PREC, LMUL)(ab1, *alpha, n);
            ab2 = VFMUL_VF(PREC, LMUL)(ab2, *alpha, n);
            ab3 = VFMUL_VF(PREC, LMUL)(ab3, *alpha, n);
            ab4 = VFMUL_VF(PREC, LMUL)(ab4, *alpha, n);
            ab5 = VFMUL_VF(PREC, LMUL)(ab5, *alpha, n);
            ab6 = VFMUL_VF(PREC, LMUL)(ab6, *alpha, n);
        }
        else {
            RVV_TYPE_F(PREC, LMUL) c0;
            if (csc == 1) {
                c0 = VLE_V_F(PREC, LMUL)(c + 0 * rsc, n);
                ab0 = VFMUL_VF(PREC, LMUL)(ab0, *alpha, n);
                ab0 = VFMACC_VF(PREC, LMUL)(ab0, *beta, c0, n);
                c0 = VLE_V_F(PREC, LMUL)(c + 1 * rsc, n);
                ab1 = VFMUL_VF(PREC, LMUL)(ab1, *alpha, n);
                ab1 = VFMACC_VF(PREC, LMUL)(ab1, *beta, c0, n);
                c0 = VLE_V_F(PREC, LMUL)(c + 2 * rsc, n);
                ab2 = VFMUL_VF(PREC, LMUL)(ab2, *alpha, n);
                ab2 = VFMACC_VF(PREC, LMUL)(ab2, *beta, c0, n);
                c0 = VLE_V_F(PREC, LMUL)(c + 3 * rsc, n);
                ab3 = VFMUL_VF(PREC, LMUL)(ab3, *alpha, n);
                ab3 = VFMACC_VF(PREC, LMUL)(ab3, *beta, c0, n);
                c0 = VLE_V_F(PREC, LMUL)(c + 4 * rsc, n);
                ab4 = VFMUL_VF(PREC, LMUL)(ab4, *alpha, n);
                ab4 = VFMACC_VF(PREC, LMUL)(ab4, *beta, c0, n);
                c0 = VLE_V_F(PREC, LMUL)(c + 5 * rsc, n);
                ab5 = VFMUL_VF(PREC, LMUL)(ab5, *alpha, n);
                ab5 = VFMACC_VF(PREC, LMUL)(ab5, *beta, c0, n);
                c0 = VLE_V_F(PREC, LMUL)(c + 6 * rsc, n);
                ab6 = VFMUL_VF(PREC, LMUL)(ab6, *alpha, n);
                ab6 = VFMACC_VF(PREC, LMUL)(ab6, *beta, c0, n);
            }
            else {
                c0 = VLSE_V_F(PREC, LMUL)(c + 0 * rsc, FLT_SIZE * csc, n);
                ab0 = VFMUL_VF(PREC, LMUL)(ab0, *alpha, n);
                ab0 = VFMACC_VF(PREC, LMUL)(ab0, *beta, c0, n);
                c0 = VLSE_V_F(PREC, LMUL)(c + 1 * rsc, FLT_SIZE * csc, n);
                ab1 = VFMUL_VF(PREC, LMUL)(ab1, *alpha, n);
                ab1 = VFMACC_VF(PREC, LMUL)(ab1, *beta, c0, n);
                c0 = VLSE_V_F(PREC, LMUL)(c + 2 * rsc, FLT_SIZE * csc, n);
                ab2 = VFMUL_VF(PREC, LMUL)(ab2, *alpha, n);
                ab2 = VFMACC_VF(PREC, LMUL)(ab2, *beta, c0, n);
                c0 = VLSE_V_F(PREC, LMUL)(c + 3 * rsc, FLT_SIZE * csc, n);
                ab3 = VFMUL_VF(PREC, LMUL)(ab3, *alpha, n);
                ab3 = VFMACC_VF(PREC, LMUL)(ab3, *beta, c0, n);
                c0 = VLSE_V_F(PREC, LMUL)(c + 4 * rsc, FLT_SIZE * csc, n);
                ab4 = VFMUL_VF(PREC, LMUL)(ab4, *alpha, n);
                ab4 = VFMACC_VF(PREC, LMUL)(ab4, *beta, c0, n);
                c0 = VLSE_V_F(PREC, LMUL)(c + 5 * rsc, FLT_SIZE * csc, n);
                ab5 = VFMUL_VF(PREC, LMUL)(ab5, *alpha, n);
                ab5 = VFMACC_VF(PREC, LMUL)(ab5, *beta, c0, n);
                c0 = VLSE_V_F(PREC, LMUL)(c + 6 * rsc, FLT_SIZE * csc, n);
                ab6 = VFMUL_VF(PREC, LMUL)(ab6, *alpha, n);
                ab6 = VFMACC_VF(PREC, LMUL)(ab6, *beta, c0, n);
            }
        }

        if (csc == 1) {
            VSE_V_F(PREC, LMUL)(c + 0 * rsc, ab0, n);
            VSE_V_F(PREC, LMUL)(c + 1 * rsc, ab1, n);
            VSE_V_F(PREC, LMUL)(c + 2 * rsc, ab2, n);
            VSE_V_F(PREC, LMUL)(c + 3 * rsc, ab3, n);
            VSE_V_F(PREC, LMUL)(c + 4 * rsc, ab4, n);
            VSE_V_F(PREC, LMUL)(c + 5 * rsc, ab5, n);
            VSE_V_F(PREC, LMUL)(c + 6 * rsc, ab6, n);
        }
        else {
            VSSE_V_F(PREC, LMUL)(c + 0 * rsc, FLT_SIZE * csc, ab0, n);
            VSSE_V_F(PREC, LMUL)(c + 1 * rsc, FLT_SIZE * csc, ab1, n);
            VSSE_V_F(PREC, LMUL)(c + 2 * rsc, FLT_SIZE * csc, ab2, n);
            VSSE_V_F(PREC, LMUL)(c + 3 * rsc, FLT_SIZE * csc, ab3, n);
            VSSE_V_F(PREC, LMUL)(c + 4 * rsc, FLT_SIZE * csc, ab4, n);
            VSSE_V_F(PREC, LMUL)(c + 5 * rsc, FLT_SIZE * csc, ab5, n);
            VSSE_V_F(PREC, LMUL)(c + 6 * rsc, FLT_SIZE * csc, ab6, n);
        }
    }
    else {
        // 0 < m < 7
        RVV_TYPE_F(PREC, LMUL) ab0, ab1, ab2, ab3, ab4, ab5;
        bool first = true;
        for (dim_t i = 0; i < k; ++i) {
            RVV_TYPE_F(PREC, LMUL) b0 = VLE_V_F(PREC, LMUL)(b, n);
            if (first) {
                switch (m) {
                case 6:
                    ab5 = VFMUL_VF(PREC, LMUL)(b0, a[5], n);
                case 5:
                    ab4 = VFMUL_VF(PREC, LMUL)(b0, a[4], n);
                case 4:
                    ab3 = VFMUL_VF(PREC, LMUL)(b0, a[3], n);
                case 3:
                    ab2 = VFMUL_VF(PREC, LMUL)(b0, a[2], n);
                case 2:
                    ab1 = VFMUL_VF(PREC, LMUL)(b0, a[1], n);
                case 1:
                    ab0 = VFMUL_VF(PREC, LMUL)(b0, a[0], n);
                }
                first = false;
            }
            else {
                switch (m) {
                case 6:
                    ab5 = VFMACC_VF(PREC, LMUL)(ab5, a[5], b0, n);
                case 5:
                    ab4 = VFMACC_VF(PREC, LMUL)(ab4, a[4], b0, n);
                case 4:
                    ab3 = VFMACC_VF(PREC, LMUL)(ab3, a[3], b0, n);
                case 3:
                    ab2 = VFMACC_VF(PREC, LMUL)(ab2, a[2], b0, n);
                case 2:
                    ab1 = VFMACC_VF(PREC, LMUL)(ab1, a[1], b0, n);
                case 1:
                    ab0 = VFMACC_VF(PREC, LMUL)(ab0, a[0], b0, n);
                }
            }

            a += PACKMR;
            b += PACKNR;
        }

        if (PASTEMAC(PRECISION_CHAR, eq0)(*beta)) {
            switch (m) {
            case 6:
                ab5 = VFMUL_VF(PREC, LMUL)(ab5, *alpha, n);
            case 5:
                ab4 = VFMUL_VF(PREC, LMUL)(ab4, *alpha, n);
            case 4:
                ab3 = VFMUL_VF(PREC, LMUL)(ab3, *alpha, n);
            case 3:
                ab2 = VFMUL_VF(PREC, LMUL)(ab2, *alpha, n);
            case 2:
                ab1 = VFMUL_VF(PREC, LMUL)(ab1, *alpha, n);
            case 1:
                ab0 = VFMUL_VF(PREC, LMUL)(ab0, *alpha, n);
            }
        }
        else {
            RVV_TYPE_F(PREC, LMUL) c0;
            if (csc == 1) {
                switch (m) {
                case 6:
                    c0 = VLE_V_F(PREC, LMUL)(c + 5 * rsc, n);
                    ab5 = VFMUL_VF(PREC, LMUL)(ab5, *alpha, n);
                    ab5 = VFMACC_VF(PREC, LMUL)(ab5, *beta, c0, n);
                case 5:
                    c0 = VLE_V_F(PREC, LMUL)(c + 4 * rsc, n);
                    ab4 = VFMUL_VF(PREC, LMUL)(ab4, *alpha, n);
                    ab4 = VFMACC_VF(PREC, LMUL)(ab4, *beta, c0, n);
                case 4:
                    c0 = VLE_V_F(PREC, LMUL)(c + 3 * rsc, n);
                    ab3 = VFMUL_VF(PREC, LMUL)(ab3, *alpha, n);
                    ab3 = VFMACC_VF(PREC, LMUL)(ab3, *beta, c0, n);
                case 3:
                    c0 = VLE_V_F(PREC, LMUL)(c + 2 * rsc, n);
                    ab2 = VFMUL_VF(PREC, LMUL)(ab2, *alpha, n);
                    ab2 = VFMACC_VF(PREC, LMUL)(ab2, *beta, c0, n);
                case 2:
                    c0 = VLE_V_F(PREC, LMUL)(c + 1 * rsc, n);
                    ab1 = VFMUL_VF(PREC, LMUL)(ab1, *alpha, n);
                    ab1 = VFMACC_VF(PREC, LMUL)(ab1, *beta, c0, n);
                case 1:
                    c0 = VLE_V_F(PREC, LMUL)(c + 0 * rsc, n);
                    ab0 = VFMUL_VF(PREC, LMUL)(ab0, *alpha, n);
                    ab0 = VFMACC_VF(PREC, LMUL)(ab0, *beta, c0, n);
                }
            }
            else {
                switch (m) {
                case 6:
                    c0 = VLSE_V_F(PREC, LMUL)(c + 5 * rsc, FLT_SIZE * csc, n);
                    ab5 = VFMUL_VF(PREC, LMUL)(ab5, *alpha, n);
                    ab5 = VFMACC_VF(PREC, LMUL)(ab5, *beta, c0, n);
                case 5:
                    c0 = VLSE_V_F(PREC, LMUL)(c + 4 * rsc, FLT_SIZE * csc, n);
                    ab4 = VFMUL_VF(PREC, LMUL)(ab4, *alpha, n);
                    ab4 = VFMACC_VF(PREC, LMUL)(ab4, *beta, c0, n);
                case 4:
                    c0 = VLSE_V_F(PREC, LMUL)(c + 3 * rsc, FLT_SIZE * csc, n);
                    ab3 = VFMUL_VF(PREC, LMUL)(ab3, *alpha, n);
                    ab3 = VFMACC_VF(PREC, LMUL)(ab3, *beta, c0, n);
                case 3:
                    c0 = VLSE_V_F(PREC, LMUL)(c + 2 * rsc, FLT_SIZE * csc, n);
                    ab2 = VFMUL_VF(PREC, LMUL)(ab2, *alpha, n);
                    ab2 = VFMACC_VF(PREC, LMUL)(ab2, *beta, c0, n);
                case 2:
                    c0 = VLSE_V_F(PREC, LMUL)(c + 1 * rsc, FLT_SIZE * csc, n);
                    ab1 = VFMUL_VF(PREC, LMUL)(ab1, *alpha, n);
                    ab1 = VFMACC_VF(PREC, LMUL)(ab1, *beta, c0, n);
                case 1:
                    c0 = VLSE_V_F(PREC, LMUL)(c + 0 * rsc, FLT_SIZE * csc, n);
                    ab0 = VFMUL_VF(PREC, LMUL)(ab0, *alpha, n);
                    ab0 = VFMACC_VF(PREC, LMUL)(ab0, *beta, c0, n);
                }
            }
        }

        if (csc == 1) {
            switch (m) {
            case 6:
                VSE_V_F(PREC, LMUL)(c + 5 * rsc, ab5, n);
            case 5:
                VSE_V_F(PREC, LMUL)(c + 4 * rsc, ab4, n);
            case 4:
                VSE_V_F(PREC, LMUL)(c + 3 * rsc, ab3, n);
            case 3:
                VSE_V_F(PREC, LMUL)(c + 2 * rsc, ab2, n);
            case 2:
                VSE_V_F(PREC, LMUL)(c + 1 * rsc, ab1, n);
            case 1:
                VSE_V_F(PREC, LMUL)(c + 0 * rsc, ab0, n);
            }
        }
        else {
            switch (m) {
            case 6:
                VSSE_V_F(PREC, LMUL)(c + 5 * rsc, FLT_SIZE * csc, ab5, n);
            case 5:
                VSSE_V_F(PREC, LMUL)(c + 4 * rsc, FLT_SIZE * csc, ab4, n);
            case 4:
                VSSE_V_F(PREC, LMUL)(c + 3 * rsc, FLT_SIZE * csc, ab3, n);
            case 3:
                VSSE_V_F(PREC, LMUL)(c + 2 * rsc, FLT_SIZE * csc, ab2, n);
            case 2:
                VSSE_V_F(PREC, LMUL)(c + 1 * rsc, FLT_SIZE * csc, ab1, n);
            case 1:
                VSSE_V_F(PREC, LMUL)(c + 0 * rsc, FLT_SIZE * csc, ab0, n);
            }
        }
    }

    return;
}

#endif // GEMM
