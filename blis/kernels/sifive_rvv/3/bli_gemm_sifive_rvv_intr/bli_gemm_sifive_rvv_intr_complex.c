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
            RVV_TYPE_FX(PREC, LMUL, 2) zero_splat = VUNDEFINED_FX(PREC, LMUL, 2)();
            zero_splat = VSET_V_F(PREC, LMUL, 2)(zero_splat, 0, VFMV_V_F(PREC, LMUL)(0., n));
            zero_splat = VSET_V_F(PREC, LMUL, 2)(zero_splat, 1, VFMV_V_F(PREC, LMUL)(0., n));

            for (dim_t i = 0; i < m; ++i) {
                if (csc == 1)
                    VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c + i * rsc), zero_splat, n);
                else
                    VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c + i * rsc), 2 * FLT_SIZE * csc, zero_splat, n);
            }
        }
        else {
            for (dim_t i = 0; i < m; ++i) {
                RVV_TYPE_FX(PREC, LMUL, 2) c0;
                RVV_TYPE_F(PREC, LMUL) c0_r, c0_i;
                RVV_TYPE_F(PREC, LMUL) beta_c0_r, beta_c0_i;

                if (csc == 1)
                    c0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c + i * rsc), n);
                else
                    c0 = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c + i * rsc), 2 * FLT_SIZE * csc, n);
                c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                VCMUL_VF(PREC, LMUL, beta_c0_r, beta_c0_i, c0_r, c0_i, beta->real, beta->imag, n); 
                c0 = VSET_V_F(PREC, LMUL, 2)(c0, 0, beta_c0_r);
                c0 = VSET_V_F(PREC, LMUL, 2)(c0, 1, beta_c0_i);
                if (csc == 1)
                    VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c + i * rsc), c0, n);
                else
                    VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c + i * rsc), 2 * FLT_SIZE * csc, c0, n);
            }
        }
    }
    else if (m == 6) {
        RVV_TYPE_F(PREC, LMUL) ab0_r, ab1_r, ab2_r, ab3_r, ab4_r, ab5_r;
        RVV_TYPE_F(PREC, LMUL) ab0_i, ab1_i, ab2_i, ab3_i, ab4_i, ab5_i;
        RVV_TYPE_FX(PREC, LMUL, 2) b0, b1;
        RVV_TYPE_F(PREC, LMUL) b0_r, b1_r;
        RVV_TYPE_F(PREC, LMUL) b0_i, b1_i;

        b0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) b, n);
        b0_r = VGET_V_F(PREC, LMUL, 2)(b0, 0);
        b0_i = VGET_V_F(PREC, LMUL, 2)(b0, 1);
        b += PACKNR;
        if (k >= 2) {
            b1 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) b, n);
            b1_r = VGET_V_F(PREC, LMUL, 2)(b1, 0);
            b1_i = VGET_V_F(PREC, LMUL, 2)(b1, 1);
            b += PACKNR;
        }

        VCMUL_VF(PREC, LMUL, ab0_r, ab0_i, b0_r, b0_i, a[0].real, a[0].imag, n);
        VCMUL_VF(PREC, LMUL, ab1_r, ab1_i, b0_r, b0_i, a[1].real, a[1].imag, n);
        VCMUL_VF(PREC, LMUL, ab2_r, ab2_i, b0_r, b0_i, a[2].real, a[2].imag, n);
        VCMUL_VF(PREC, LMUL, ab3_r, ab3_i, b0_r, b0_i, a[3].real, a[3].imag, n);
        VCMUL_VF(PREC, LMUL, ab4_r, ab4_i, b0_r, b0_i, a[4].real, a[4].imag, n);
        VCMUL_VF(PREC, LMUL, ab5_r, ab5_i, b0_r, b0_i, a[5].real, a[5].imag, n);
        
        a += PACKMR;
        k -= 1;
        
        if (k >= 2) {
            b0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) b, n);
            b0_r = VGET_V_F(PREC, LMUL, 2)(b0, 0);
            b0_i = VGET_V_F(PREC, LMUL, 2)(b0, 1);
            b += PACKNR;
        }

        while (k > 0) {
            VCMACC_VF(PREC, LMUL, ab0_r, ab0_i, a[0].real, a[0].imag, b1_r, b1_i, n);
            VCMACC_VF(PREC, LMUL, ab1_r, ab1_i, a[1].real, a[1].imag, b1_r, b1_i, n);
            VCMACC_VF(PREC, LMUL, ab2_r, ab2_i, a[2].real, a[2].imag, b1_r, b1_i, n);
            VCMACC_VF(PREC, LMUL, ab3_r, ab3_i, a[3].real, a[3].imag, b1_r, b1_i, n);
            VCMACC_VF(PREC, LMUL, ab4_r, ab4_i, a[4].real, a[4].imag, b1_r, b1_i, n);
            VCMACC_VF(PREC, LMUL, ab5_r, ab5_i, a[5].real, a[5].imag, b1_r, b1_i, n);
             
            a += PACKMR;
            k -= 1;

            if (k == 0) { break; }

            if (k >= 2) {
                b1 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) b, n);
                b1_r = VGET_V_F(PREC, LMUL, 2)(b1, 0);
                b1_i = VGET_V_F(PREC, LMUL, 2)(b1, 1);
                b += PACKNR;
            }

            VCMACC_VF(PREC, LMUL, ab0_r, ab0_i, a[0].real, a[0].imag, b0_r, b0_i, n);
            VCMACC_VF(PREC, LMUL, ab1_r, ab1_i, a[1].real, a[1].imag, b0_r, b0_i, n);
            VCMACC_VF(PREC, LMUL, ab2_r, ab2_i, a[2].real, a[2].imag, b0_r, b0_i, n);
            VCMACC_VF(PREC, LMUL, ab3_r, ab3_i, a[3].real, a[3].imag, b0_r, b0_i, n);
            VCMACC_VF(PREC, LMUL, ab4_r, ab4_i, a[4].real, a[4].imag, b0_r, b0_i, n);
            VCMACC_VF(PREC, LMUL, ab5_r, ab5_i, a[5].real, a[5].imag, b0_r, b0_i, n);
             
            a += PACKMR;
            k -= 1;

            if (k == 0) { break; }

            if (k >= 2) {
                b0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) b, n);
                b0_r = VGET_V_F(PREC, LMUL, 2)(b0, 0);
                b0_i = VGET_V_F(PREC, LMUL, 2)(b0, 1);
                b += PACKNR;
            }
        }

        RVV_TYPE_F(PREC, LMUL) temp0_r, temp1_r;
        RVV_TYPE_F(PREC, LMUL) temp0_i, temp1_i;
        temp0_r = VFMUL_VF(PREC, LMUL)(ab0_i, alpha->imag, n);
        temp0_i = VFMUL_VF(PREC, LMUL)(ab0_r, alpha->imag, n);
        temp1_r = VFMUL_VF(PREC, LMUL)(ab1_i, alpha->imag, n);
        temp1_i = VFMUL_VF(PREC, LMUL)(ab1_r, alpha->imag, n);

        ab0_r = VFMSUB_VF(PREC, LMUL)(ab0_r, alpha->real, temp0_r, n);
        ab0_i = VFMADD_VF(PREC, LMUL)(ab0_i, alpha->real, temp0_i, n);
        ab1_r = VFMSUB_VF(PREC, LMUL)(ab1_r, alpha->real, temp1_r, n);
        ab1_i = VFMADD_VF(PREC, LMUL)(ab1_i, alpha->real, temp1_i, n);

        temp0_r = VFMUL_VF(PREC, LMUL)(ab2_i, alpha->imag, n);
        temp0_i = VFMUL_VF(PREC, LMUL)(ab2_r, alpha->imag, n);
        temp1_r = VFMUL_VF(PREC, LMUL)(ab3_i, alpha->imag, n);
        temp1_i = VFMUL_VF(PREC, LMUL)(ab3_r, alpha->imag, n);

        ab2_r = VFMSUB_VF(PREC, LMUL)(ab2_r, alpha->real, temp0_r, n);
        ab2_i = VFMADD_VF(PREC, LMUL)(ab2_i, alpha->real, temp0_i, n);
        ab3_r = VFMSUB_VF(PREC, LMUL)(ab3_r, alpha->real, temp1_r, n);
        ab3_i = VFMADD_VF(PREC, LMUL)(ab3_i, alpha->real, temp1_i, n);

        temp0_r = VFMUL_VF(PREC, LMUL)(ab4_i, alpha->imag, n);
        temp0_i = VFMUL_VF(PREC, LMUL)(ab4_r, alpha->imag, n);
        temp1_r = VFMUL_VF(PREC, LMUL)(ab5_i, alpha->imag, n);
        temp1_i = VFMUL_VF(PREC, LMUL)(ab5_r, alpha->imag, n);

        ab4_r = VFMSUB_VF(PREC, LMUL)(ab4_r, alpha->real, temp0_r, n);
        ab4_i = VFMADD_VF(PREC, LMUL)(ab4_i, alpha->real, temp0_i, n);
        ab5_r = VFMSUB_VF(PREC, LMUL)(ab5_r, alpha->real, temp1_r, n);
        ab5_i = VFMADD_VF(PREC, LMUL)(ab5_i, alpha->real, temp1_i, n);

        if (!PASTEMAC(PRECISION_CHAR, eq0)(*beta)) {
            RVV_TYPE_FX(PREC, LMUL, 2) c0;
            RVV_TYPE_F(PREC, LMUL) c0_r, c0_i;
            if (csc == 1) {
                c0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 0 * rsc), n);
                c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                VCMACC_VF(PREC, LMUL, ab0_r, ab0_i, beta->real, beta->imag, c0_r, c0_i, n);
                
                c0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 1 * rsc), n);
                c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                VCMACC_VF(PREC, LMUL, ab1_r, ab1_i, beta->real, beta->imag, c0_r, c0_i, n);
                
                c0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 2 * rsc), n);
                c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                VCMACC_VF(PREC, LMUL, ab2_r, ab2_i, beta->real, beta->imag, c0_r, c0_i, n);
                
                c0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 3 * rsc), n);
                c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                VCMACC_VF(PREC, LMUL, ab3_r, ab3_i, beta->real, beta->imag, c0_r, c0_i, n);
                
                c0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 4 * rsc), n);
                c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                VCMACC_VF(PREC, LMUL, ab4_r, ab4_i, beta->real, beta->imag, c0_r, c0_i, n);
                
                c0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 5 * rsc), n);
                c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                VCMACC_VF(PREC, LMUL, ab5_r, ab5_i, beta->real, beta->imag, c0_r, c0_i, n);
            }
            else {
                c0 = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 0 * rsc), 2 * FLT_SIZE * csc, n);
                c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                VCMACC_VF(PREC, LMUL, ab0_r, ab0_i, beta->real, beta->imag, c0_r, c0_i, n);
                
                c0 = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 1 * rsc), 2 * FLT_SIZE * csc, n);
                c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                VCMACC_VF(PREC, LMUL, ab1_r, ab1_i, beta->real, beta->imag, c0_r, c0_i, n);
                
                c0 = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 2 * rsc), 2 * FLT_SIZE * csc, n);
                c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                VCMACC_VF(PREC, LMUL, ab2_r, ab2_i, beta->real, beta->imag, c0_r, c0_i, n);
                
                c0 = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 3 * rsc), 2 * FLT_SIZE * csc, n);
                c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                VCMACC_VF(PREC, LMUL, ab3_r, ab3_i, beta->real, beta->imag, c0_r, c0_i, n);
                
                c0 = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 4 * rsc), 2 * FLT_SIZE * csc, n);
                c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                VCMACC_VF(PREC, LMUL, ab4_r, ab4_i, beta->real, beta->imag, c0_r, c0_i, n);
                
                c0 = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 5 * rsc), 2 * FLT_SIZE * csc, n);
                c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                VCMACC_VF(PREC, LMUL, ab5_r, ab5_i, beta->real, beta->imag, c0_r, c0_i, n);
            }
        }

        RVV_TYPE_FX(PREC, LMUL, 2) ab0 = VCREATE_V_FX(PREC, LMUL, 2)(ab0_r, ab0_i);
        RVV_TYPE_FX(PREC, LMUL, 2) ab1 = VCREATE_V_FX(PREC, LMUL, 2)(ab1_r, ab1_i);
        RVV_TYPE_FX(PREC, LMUL, 2) ab2 = VCREATE_V_FX(PREC, LMUL, 2)(ab2_r, ab2_i);
        RVV_TYPE_FX(PREC, LMUL, 2) ab3 = VCREATE_V_FX(PREC, LMUL, 2)(ab3_r, ab3_i);
        RVV_TYPE_FX(PREC, LMUL, 2) ab4 = VCREATE_V_FX(PREC, LMUL, 2)(ab4_r, ab4_i);
        RVV_TYPE_FX(PREC, LMUL, 2) ab5 = VCREATE_V_FX(PREC, LMUL, 2)(ab5_r, ab5_i);

        if (csc == 1) {
            VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 0 * rsc), ab0, n);
            VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 1 * rsc), ab1, n);
            VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 2 * rsc), ab2, n);
            VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 3 * rsc), ab3, n);
            VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 4 * rsc), ab4, n);
            VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 5 * rsc), ab5, n);
        }
        else {
            VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 0 * rsc), 2 * FLT_SIZE * csc, ab0, n);
            VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 1 * rsc), 2 * FLT_SIZE * csc, ab1, n);
            VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 2 * rsc), 2 * FLT_SIZE * csc, ab2, n);
            VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 3 * rsc), 2 * FLT_SIZE * csc, ab3, n);
            VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 4 * rsc), 2 * FLT_SIZE * csc, ab4, n);
            VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 5 * rsc), 2 * FLT_SIZE * csc, ab5, n);
        }
    }
    else {
        RVV_TYPE_F(PREC, LMUL) ab0_r, ab1_r, ab2_r, ab3_r, ab4_r;
        RVV_TYPE_F(PREC, LMUL) ab0_i, ab1_i, ab2_i, ab3_i, ab4_i;
        RVV_TYPE_FX(PREC, LMUL, 2) b0, b1;
        RVV_TYPE_F(PREC, LMUL) b0_r, b1_r;
        RVV_TYPE_F(PREC, LMUL) b0_i, b1_i;

        b0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) b, n);
        b0_r = VGET_V_F(PREC, LMUL, 2)(b0, 0);
        b0_i = VGET_V_F(PREC, LMUL, 2)(b0, 1);
        b += PACKNR;
        if (k >= 2) {
            b1 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) b, n);
            b1_r = VGET_V_F(PREC, LMUL, 2)(b1, 0);
            b1_i = VGET_V_F(PREC, LMUL, 2)(b1, 1);
            b += PACKNR;
        }

        switch (m) {
        case 5:
            VCMUL_VF(PREC, LMUL, ab4_r, ab4_i, b0_r, b0_i, a[4].real, a[4].imag, n);
        case 4:
            VCMUL_VF(PREC, LMUL, ab3_r, ab3_i, b0_r, b0_i, a[3].real, a[3].imag, n);
        case 3:
            VCMUL_VF(PREC, LMUL, ab2_r, ab2_i, b0_r, b0_i, a[2].real, a[2].imag, n);
        case 2:
            VCMUL_VF(PREC, LMUL, ab1_r, ab1_i, b0_r, b0_i, a[1].real, a[1].imag, n);
        case 1:
            VCMUL_VF(PREC, LMUL, ab0_r, ab0_i, b0_r, b0_i, a[0].real, a[0].imag, n);
        }
        
        a += PACKMR;
        k -= 1;
        
        if (k >= 2) {
            b0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) b, n);
            b0_r = VGET_V_F(PREC, LMUL, 2)(b0, 0);
            b0_i = VGET_V_F(PREC, LMUL, 2)(b0, 1);
            b += PACKNR;
        }

        while (k > 0) {
            switch (m) {
            case 5:
                VCMACC_VF(PREC, LMUL, ab4_r, ab4_i, a[4].real, a[4].imag, b1_r, b1_i, n);
            case 4:
                VCMACC_VF(PREC, LMUL, ab3_r, ab3_i, a[3].real, a[3].imag, b1_r, b1_i, n);
            case 3:
                VCMACC_VF(PREC, LMUL, ab2_r, ab2_i, a[2].real, a[2].imag, b1_r, b1_i, n);
            case 2:
                VCMACC_VF(PREC, LMUL, ab1_r, ab1_i, a[1].real, a[1].imag, b1_r, b1_i, n);
            case 1:
                VCMACC_VF(PREC, LMUL, ab0_r, ab0_i, a[0].real, a[0].imag, b1_r, b1_i, n);
            }
             
            a += PACKMR;
            k -= 1;

            if (k == 0) { break; }

            if (k >= 2) {
                b1 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) b, n);
                b1_r = VGET_V_F(PREC, LMUL, 2)(b1, 0);
                b1_i = VGET_V_F(PREC, LMUL, 2)(b1, 1);
                b += PACKNR;
            }

            switch (m) {
            case 5:
                VCMACC_VF(PREC, LMUL, ab4_r, ab4_i, a[4].real, a[4].imag, b0_r, b0_i, n);
            case 4:
                VCMACC_VF(PREC, LMUL, ab3_r, ab3_i, a[3].real, a[3].imag, b0_r, b0_i, n);
            case 3:
                VCMACC_VF(PREC, LMUL, ab2_r, ab2_i, a[2].real, a[2].imag, b0_r, b0_i, n);
            case 2:
                VCMACC_VF(PREC, LMUL, ab1_r, ab1_i, a[1].real, a[1].imag, b0_r, b0_i, n);
            case 1:
                VCMACC_VF(PREC, LMUL, ab0_r, ab0_i, a[0].real, a[0].imag, b0_r, b0_i, n);
            }
             
            a += PACKMR;
            k -= 1;

            if (k == 0) { break; }

            if (k >= 2) {
                b0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) b, n);
                b0_r = VGET_V_F(PREC, LMUL, 2)(b0, 0);
                b0_i = VGET_V_F(PREC, LMUL, 2)(b0, 1);
                b += PACKNR;
            }
        }

        RVV_TYPE_F(PREC, LMUL) temp0_r, temp1_r;
        RVV_TYPE_F(PREC, LMUL) temp0_i, temp1_i;
        switch (m) {
        case 5:
            temp0_r = VFMUL_VF(PREC, LMUL)(ab4_i, alpha->imag, n);
            temp0_i = VFMUL_VF(PREC, LMUL)(ab4_r, alpha->imag, n);
            ab4_r = VFMSUB_VF(PREC, LMUL)(ab4_r, alpha->real, temp0_r, n);
            ab4_i = VFMADD_VF(PREC, LMUL)(ab4_i, alpha->real, temp0_i, n);
        case 4:
            temp1_r = VFMUL_VF(PREC, LMUL)(ab3_i, alpha->imag, n);
            temp1_i = VFMUL_VF(PREC, LMUL)(ab3_r, alpha->imag, n);
            ab3_r = VFMSUB_VF(PREC, LMUL)(ab3_r, alpha->real, temp1_r, n);
            ab3_i = VFMADD_VF(PREC, LMUL)(ab3_i, alpha->real, temp1_i, n);
        case 3:
            temp0_r = VFMUL_VF(PREC, LMUL)(ab2_i, alpha->imag, n);
            temp0_i = VFMUL_VF(PREC, LMUL)(ab2_r, alpha->imag, n);
            ab2_r = VFMSUB_VF(PREC, LMUL)(ab2_r, alpha->real, temp0_r, n);
            ab2_i = VFMADD_VF(PREC, LMUL)(ab2_i, alpha->real, temp0_i, n);
        case 2:
            temp1_r = VFMUL_VF(PREC, LMUL)(ab1_i, alpha->imag, n);
            temp1_i = VFMUL_VF(PREC, LMUL)(ab1_r, alpha->imag, n);
            ab1_r = VFMSUB_VF(PREC, LMUL)(ab1_r, alpha->real, temp1_r, n);
            ab1_i = VFMADD_VF(PREC, LMUL)(ab1_i, alpha->real, temp1_i, n);
        case 1:
            temp0_r = VFMUL_VF(PREC, LMUL)(ab0_i, alpha->imag, n);
            temp0_i = VFMUL_VF(PREC, LMUL)(ab0_r, alpha->imag, n);
            ab0_r = VFMSUB_VF(PREC, LMUL)(ab0_r, alpha->real, temp0_r, n);
            ab0_i = VFMADD_VF(PREC, LMUL)(ab0_i, alpha->real, temp0_i, n);
        }

        if (!PASTEMAC(PRECISION_CHAR, eq0)(*beta)) {
            RVV_TYPE_FX(PREC, LMUL, 2) c0;
            RVV_TYPE_F(PREC, LMUL) c0_r, c0_i;
            if (csc == 1) {
                switch (m) {
                case 5:
                    c0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 4 * rsc), n);
                    c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                    c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                    VCMACC_VF(PREC, LMUL, ab4_r, ab4_i, beta->real, beta->imag, c0_r, c0_i, n);
                case 4:
                    c0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 3 * rsc), n);
                    c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                    c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                    VCMACC_VF(PREC, LMUL, ab3_r, ab3_i, beta->real, beta->imag, c0_r, c0_i, n);
                case 3:
                    c0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 2 * rsc), n);
                    c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                    c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                    VCMACC_VF(PREC, LMUL, ab2_r, ab2_i, beta->real, beta->imag, c0_r, c0_i, n);
                case 2:
                    c0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 1 * rsc), n);
                    c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                    c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                    VCMACC_VF(PREC, LMUL, ab1_r, ab1_i, beta->real, beta->imag, c0_r, c0_i, n);
                case 1:
                    c0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 0 * rsc), n);
                    c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                    c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                    VCMACC_VF(PREC, LMUL, ab0_r, ab0_i, beta->real, beta->imag, c0_r, c0_i, n);
                }
                
            }
            else {
                switch (m) {
                case 5:
                    c0 = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 4 * rsc), 2 * FLT_SIZE * csc, n);
                    c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                    c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                    VCMACC_VF(PREC, LMUL, ab4_r, ab4_i, beta->real, beta->imag, c0_r, c0_i, n);
                case 4:
                    c0 = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 3 * rsc), 2 * FLT_SIZE * csc, n);
                    c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                    c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                    VCMACC_VF(PREC, LMUL, ab3_r, ab3_i, beta->real, beta->imag, c0_r, c0_i, n);
                case 3:
                    c0 = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 2 * rsc), 2 * FLT_SIZE * csc, n);
                    c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                    c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                    VCMACC_VF(PREC, LMUL, ab2_r, ab2_i, beta->real, beta->imag, c0_r, c0_i, n);
                case 2:
                    c0 = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 1 * rsc), 2 * FLT_SIZE * csc, n);
                    c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                    c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                    VCMACC_VF(PREC, LMUL, ab1_r, ab1_i, beta->real, beta->imag, c0_r, c0_i, n);
                case 1:
                    c0 = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 0 * rsc), 2 * FLT_SIZE * csc, n);
                    c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                    c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                    VCMACC_VF(PREC, LMUL, ab0_r, ab0_i, beta->real, beta->imag, c0_r, c0_i, n);
                }
            }
        }

        RVV_TYPE_FX(PREC, LMUL, 2) ab0, ab1, ab2, ab3, ab4;
        switch (m) {
        case 5:
            ab4 = VCREATE_V_FX(PREC, LMUL, 2)(ab4_r, ab4_i);
        case 4:
            ab3 = VCREATE_V_FX(PREC, LMUL, 2)(ab3_r, ab3_i);
        case 3:
            ab2 = VCREATE_V_FX(PREC, LMUL, 2)(ab2_r, ab2_i);
        case 2:
            ab1 = VCREATE_V_FX(PREC, LMUL, 2)(ab1_r, ab1_i);
        case 1:
            ab0 = VCREATE_V_FX(PREC, LMUL, 2)(ab0_r, ab0_i);
        }

        if (csc == 1) {
            switch (m) {
            case 5:
                VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 4 * rsc), ab4, n);
            case 4:
                VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 3 * rsc), ab3, n);
            case 3:
                VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 2 * rsc), ab2, n);
            case 2:
                VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 1 * rsc), ab1, n);
            case 1:
                VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 0 * rsc), ab0, n);
            }
        }
        else {
            switch (m) {
            case 5:
                VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 4 * rsc), 2 * FLT_SIZE * csc, ab4, n);
            case 4:
                VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 3 * rsc), 2 * FLT_SIZE * csc, ab3, n);
            case 3:
                VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 2 * rsc), 2 * FLT_SIZE * csc, ab2, n);
            case 2:
                VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 1 * rsc), 2 * FLT_SIZE * csc, ab1, n);
            case 1:
                VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 0 * rsc), 2 * FLT_SIZE * csc, ab0, n);
            }
        }
    }

    return;
}

#endif // GEMM
