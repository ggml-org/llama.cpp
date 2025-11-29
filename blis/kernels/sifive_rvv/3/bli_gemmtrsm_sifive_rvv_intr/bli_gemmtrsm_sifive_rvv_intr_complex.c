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
    
    RVV_TYPE_F(PREC, LMUL) ab0_r, ab1_r, ab2_r, ab3_r, ab4_r, ab5_r;
    RVV_TYPE_F(PREC, LMUL) ab0_i, ab1_i, ab2_i, ab3_i, ab4_i, ab5_i;
    // gemm step
    if (m <= 0 || n <= 0 || k < 0)
        return;
    else if (k == 0) {
        if (PASTEMAC(PRECISION_CHAR, eq0)(*beta)) {
            RVV_TYPE_F(PREC, LMUL) zero_splat = VFMV_V_F(PREC, LMUL)(0., n);
            switch (m) {
            case 6:
                ab5_r = zero_splat;
                ab5_i = zero_splat;
            case 5:
                ab4_r = zero_splat;
                ab4_i = zero_splat;
            case 4:
                ab3_r = zero_splat;
                ab3_i = zero_splat;
            case 3:
                ab2_r = zero_splat;
                ab2_i = zero_splat;
            case 2:
                ab1_r = zero_splat;
                ab1_i = zero_splat;
            case 1:
                ab0_r = zero_splat;
                ab0_i = zero_splat;
            }
        }
        else {
            RVV_TYPE_FX(PREC, LMUL, 2) c0;
            RVV_TYPE_F(PREC, LMUL) c0_r, c0_i;

            switch (m) {
            case 6:
                c0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c + 5 * rsc), n);
                c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                VCMUL_VF(PREC, LMUL, ab5_r, ab5_i, c0_r, c0_i, beta->real, beta->imag, n); 
            case 5:
                c0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c + 4 * rsc), n);
                c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                VCMUL_VF(PREC, LMUL, ab4_r, ab4_i, c0_r, c0_i, beta->real, beta->imag, n); 
            case 4:
                c0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c + 3 * rsc), n);
                c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                VCMUL_VF(PREC, LMUL, ab3_r, ab3_i, c0_r, c0_i, beta->real, beta->imag, n); 
            case 3:
                c0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c + 2 * rsc), n);
                c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                VCMUL_VF(PREC, LMUL, ab2_r, ab2_i, c0_r, c0_i, beta->real, beta->imag, n); 
            case 2:
                c0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c + 1 * rsc), n);
                c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                VCMUL_VF(PREC, LMUL, ab1_r, ab1_i, c0_r, c0_i, beta->real, beta->imag, n); 
            case 1:
                c0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c + 0 * rsc), n);
                c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
                c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
                VCMUL_VF(PREC, LMUL, ab0_r, ab0_i, c0_r, c0_i, beta->real, beta->imag, n); 
            }
        }
    }
    else {
        RVV_TYPE_FX(PREC, LMUL, 2) b0, b1;
        RVV_TYPE_F(PREC, LMUL) b0_r, b1_r;
        RVV_TYPE_F(PREC, LMUL) b0_i, b1_i;

        b0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) b, n);
        b0_r = VGET_V_F(PREC, LMUL, 2)(b0, 0);
        b0_i = VGET_V_F(PREC, LMUL, 2)(b0, 1);
        b += rsb;
        if (k >= 2) {
            b1 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) b, n);
            b1_r = VGET_V_F(PREC, LMUL, 2)(b1, 0);
            b1_i = VGET_V_F(PREC, LMUL, 2)(b1, 1);
            b += rsb;
        }

        switch (m) {
        case 6:
            VCMUL_VF(PREC, LMUL, ab5_r, ab5_i, b0_r, b0_i, a[5 * rsa].real, a[5 * rsa].imag, n);
        case 5:
            VCMUL_VF(PREC, LMUL, ab4_r, ab4_i, b0_r, b0_i, a[4 * rsa].real, a[4 * rsa].imag, n);
        case 4:
            VCMUL_VF(PREC, LMUL, ab3_r, ab3_i, b0_r, b0_i, a[3 * rsa].real, a[3 * rsa].imag, n);
        case 3:
            VCMUL_VF(PREC, LMUL, ab2_r, ab2_i, b0_r, b0_i, a[2 * rsa].real, a[2 * rsa].imag, n);
        case 2:
            VCMUL_VF(PREC, LMUL, ab1_r, ab1_i, b0_r, b0_i, a[1 * rsa].real, a[1 * rsa].imag, n);
        case 1:
            VCMUL_VF(PREC, LMUL, ab0_r, ab0_i, b0_r, b0_i, a[0 * rsa].real, a[0 * rsa].imag, n);
        }
        
        a += csa;
        k -= 1;
        
        if (k >= 2) {
            b0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) b, n);
            b0_r = VGET_V_F(PREC, LMUL, 2)(b0, 0);
            b0_i = VGET_V_F(PREC, LMUL, 2)(b0, 1);
            b += rsb;
        }

        while (k > 0) {
            switch (m) {
            case 6:
                VCMACC_VF(PREC, LMUL, ab5_r, ab5_i, a[5 * rsa].real, a[5 * rsa].imag, b1_r, b1_i, n);
            case 5:
                VCMACC_VF(PREC, LMUL, ab4_r, ab4_i, a[4 * rsa].real, a[4 * rsa].imag, b1_r, b1_i, n);
            case 4:
                VCMACC_VF(PREC, LMUL, ab3_r, ab3_i, a[3 * rsa].real, a[3 * rsa].imag, b1_r, b1_i, n);
            case 3:
                VCMACC_VF(PREC, LMUL, ab2_r, ab2_i, a[2 * rsa].real, a[2 * rsa].imag, b1_r, b1_i, n);
            case 2:
                VCMACC_VF(PREC, LMUL, ab1_r, ab1_i, a[1 * rsa].real, a[1 * rsa].imag, b1_r, b1_i, n);
            case 1:
                VCMACC_VF(PREC, LMUL, ab0_r, ab0_i, a[0 * rsa].real, a[0 * rsa].imag, b1_r, b1_i, n);
            }
             
            a += csa;
            k -= 1;

            if (k == 0) { break; }

            if (k >= 2) {
                b1 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) b, n);
                b1_r = VGET_V_F(PREC, LMUL, 2)(b1, 0);
                b1_i = VGET_V_F(PREC, LMUL, 2)(b1, 1);
                b += rsb;
            }

            switch (m) {
            case 6:
                VCMACC_VF(PREC, LMUL, ab5_r, ab5_i, a[5 * rsa].real, a[5 * rsa].imag, b0_r, b0_i, n);
            case 5:
                VCMACC_VF(PREC, LMUL, ab4_r, ab4_i, a[4 * rsa].real, a[4 * rsa].imag, b0_r, b0_i, n);
            case 4:
                VCMACC_VF(PREC, LMUL, ab3_r, ab3_i, a[3 * rsa].real, a[3 * rsa].imag, b0_r, b0_i, n);
            case 3:
                VCMACC_VF(PREC, LMUL, ab2_r, ab2_i, a[2 * rsa].real, a[2 * rsa].imag, b0_r, b0_i, n);
            case 2:
                VCMACC_VF(PREC, LMUL, ab1_r, ab1_i, a[1 * rsa].real, a[1 * rsa].imag, b0_r, b0_i, n);
            case 1:
                VCMACC_VF(PREC, LMUL, ab0_r, ab0_i, a[0 * rsa].real, a[0 * rsa].imag, b0_r, b0_i, n);
            }
             
            a += csa;
            k -= 1;

            if (k == 0) { break; }

            if (k >= 2) {
                b0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) b, n);
                b0_r = VGET_V_F(PREC, LMUL, 2)(b0, 0);
                b0_i = VGET_V_F(PREC, LMUL, 2)(b0, 1);
                b += rsb;
            }
        }

        if (PASTEMAC(PRECISION_CHAR, eq0)(*beta)) {
            switch (m) {
            case 6:
                ab5_r = VFNEG_VF(PREC, LMUL)(ab5_r, n);
                ab5_i = VFNEG_VF(PREC, LMUL)(ab5_i, n);
            case 5:
                ab4_r = VFNEG_VF(PREC, LMUL)(ab4_r, n);
                ab4_i = VFNEG_VF(PREC, LMUL)(ab4_i, n);
            case 4:
                ab3_r = VFNEG_VF(PREC, LMUL)(ab3_r, n);
                ab3_i = VFNEG_VF(PREC, LMUL)(ab3_i, n);
            case 3:
                ab2_r = VFNEG_VF(PREC, LMUL)(ab2_r, n);
                ab2_i = VFNEG_VF(PREC, LMUL)(ab2_i, n);
            case 2:
                ab1_r = VFNEG_VF(PREC, LMUL)(ab1_r, n);
                ab1_i = VFNEG_VF(PREC, LMUL)(ab1_i, n);
            case 1:
                ab0_r = VFNEG_VF(PREC, LMUL)(ab0_r, n);
                ab0_i = VFNEG_VF(PREC, LMUL)(ab0_i, n);
            }
        }
        else {
            RVV_TYPE_FX(PREC, LMUL, 2) c0;
            RVV_TYPE_F(PREC, LMUL) c0_r, c0_i;
	    switch (m) {
	    case 6:
		c0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 5 * rsc), n);
		c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
		c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
		VCMSAC_VF(PREC, LMUL, ab5_r, ab5_i, beta->real, beta->imag, c0_r, c0_i, n);
	    case 5:
		c0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 4 * rsc), n);
		c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
		c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
		VCMSAC_VF(PREC, LMUL, ab4_r, ab4_i, beta->real, beta->imag, c0_r, c0_i, n);
	    case 4:
		c0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 3 * rsc), n);
		c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
		c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
		VCMSAC_VF(PREC, LMUL, ab3_r, ab3_i, beta->real, beta->imag, c0_r, c0_i, n);
	    case 3:
		c0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 2 * rsc), n);
		c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
		c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
		VCMSAC_VF(PREC, LMUL, ab2_r, ab2_i, beta->real, beta->imag, c0_r, c0_i, n);
	    case 2:
		c0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 1 * rsc), n);
		c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
		c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
		VCMSAC_VF(PREC, LMUL, ab1_r, ab1_i, beta->real, beta->imag, c0_r, c0_i, n);
	    case 1:
		c0 = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) (c + 0 * rsc), n);
		c0_r = VGET_V_F(PREC, LMUL, 2)(c0, 0);
		c0_i = VGET_V_F(PREC, LMUL, 2)(c0, 1);
		VCMSAC_VF(PREC, LMUL, ab0_r, ab0_i, beta->real, beta->imag, c0_r, c0_i, n);
	    }
        }
    }   

    // trsm step
    RVV_TYPE_FX(PREC, LMUL, 2) temp = VUNDEFINED_FX(PREC, LMUL, 2)();
    RVV_TYPE_F(PREC, LMUL) temp_r, temp_i;

    VCMUL_VF(PREC, LMUL, temp_r, temp_i, ab0_r, ab0_i, a11[0 * rsa11].real, a11[0 * rsa11].imag, n);
    temp = VSET_V_F(PREC, LMUL, 2)(temp, 0, temp_r);
    temp = VSET_V_F(PREC, LMUL, 2)(temp, 1, temp_i);
    VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c + 0 * rsc), temp, n);
    if (csc11 == 1)
        VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c11 + 0 * rsc11), temp, n);
    else
        VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c11 + 0 * rsc11), 2 * FLT_SIZE * csc11, temp, n);
    if (m == 1) return;
    switch (m) {
    case 6:
        VCNMSAC_VF(PREC, LMUL, ab5_r, ab5_i, a11[5 * rsa11].real, a11[5 * rsa11].imag, temp_r, temp_i, n);
    case 5:
        VCNMSAC_VF(PREC, LMUL, ab4_r, ab4_i, a11[4 * rsa11].real, a11[4 * rsa11].imag, temp_r, temp_i, n);
    case 4:
        VCNMSAC_VF(PREC, LMUL, ab3_r, ab3_i, a11[3 * rsa11].real, a11[3 * rsa11].imag, temp_r, temp_i, n);
    case 3:
        VCNMSAC_VF(PREC, LMUL, ab2_r, ab2_i, a11[2 * rsa11].real, a11[2 * rsa11].imag, temp_r, temp_i, n);
    case 2:
        VCNMSAC_VF(PREC, LMUL, ab1_r, ab1_i, a11[1 * rsa11].real, a11[1 * rsa11].imag, temp_r, temp_i, n);
    }
    a11 += csa11;

    VCMUL_VF(PREC, LMUL, temp_r, temp_i, ab1_r, ab1_i, a11[1 * rsa11].real, a11[1 * rsa11].imag, n);
    temp = VSET_V_F(PREC, LMUL, 2)(temp, 0, temp_r);
    temp = VSET_V_F(PREC, LMUL, 2)(temp, 1, temp_i);
    VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c + 1 * rsc), temp, n);
    if (csc11 == 1)
        VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c11 + 1 * rsc11), temp, n);
    else
        VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c11 + 1 * rsc11), 2 * FLT_SIZE * csc11, temp, n);
    if (m == 2) return;
    switch (m) {
    case 6:
        VCNMSAC_VF(PREC, LMUL, ab5_r, ab5_i, a11[5 * rsa11].real, a11[5 * rsa11].imag, temp_r, temp_i, n);
    case 5:
        VCNMSAC_VF(PREC, LMUL, ab4_r, ab4_i, a11[4 * rsa11].real, a11[4 * rsa11].imag, temp_r, temp_i, n);
    case 4:
        VCNMSAC_VF(PREC, LMUL, ab3_r, ab3_i, a11[3 * rsa11].real, a11[3 * rsa11].imag, temp_r, temp_i, n);
    case 3:
        VCNMSAC_VF(PREC, LMUL, ab2_r, ab2_i, a11[2 * rsa11].real, a11[2 * rsa11].imag, temp_r, temp_i, n);
    }
    a11 += csa11;

    VCMUL_VF(PREC, LMUL, temp_r, temp_i, ab2_r, ab2_i, a11[2 * rsa11].real, a11[2 * rsa11].imag, n);
    temp = VSET_V_F(PREC, LMUL, 2)(temp, 0, temp_r);
    temp = VSET_V_F(PREC, LMUL, 2)(temp, 1, temp_i);
    VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c + 2 * rsc), temp, n);
    if (csc11 == 1)
        VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c11 + 2 * rsc11), temp, n);
    else
        VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c11 + 2 * rsc11), 2 * FLT_SIZE * csc11, temp, n);
    if (m == 3) return;
    switch (m) {
    case 6:
        VCNMSAC_VF(PREC, LMUL, ab5_r, ab5_i, a11[5 * rsa11].real, a11[5 * rsa11].imag, temp_r, temp_i, n);
    case 5:
        VCNMSAC_VF(PREC, LMUL, ab4_r, ab4_i, a11[4 * rsa11].real, a11[4 * rsa11].imag, temp_r, temp_i, n);
    case 4:
        VCNMSAC_VF(PREC, LMUL, ab3_r, ab3_i, a11[3 * rsa11].real, a11[3 * rsa11].imag, temp_r, temp_i, n);
    }
    a11 += csa11;

    VCMUL_VF(PREC, LMUL, temp_r, temp_i, ab3_r, ab3_i, a11[3 * rsa11].real, a11[3 * rsa11].imag, n);
    temp = VSET_V_F(PREC, LMUL, 2)(temp, 0, temp_r);
    temp = VSET_V_F(PREC, LMUL, 2)(temp, 1, temp_i);
    VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c + 3 * rsc), temp, n);
    if (csc11 == 1)
        VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c11 + 3 * rsc11), temp, n);
    else
        VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c11 + 3 * rsc11), 2 * FLT_SIZE * csc11, temp, n);
    if (m == 4) return;
    switch (m) {
    case 6:
        VCNMSAC_VF(PREC, LMUL, ab5_r, ab5_i, a11[5 * rsa11].real, a11[5 * rsa11].imag, temp_r, temp_i, n);
    case 5:
        VCNMSAC_VF(PREC, LMUL, ab4_r, ab4_i, a11[4 * rsa11].real, a11[4 * rsa11].imag, temp_r, temp_i, n);
    }
    a11 += csa11;

    VCMUL_VF(PREC, LMUL, temp_r, temp_i, ab4_r, ab4_i, a11[4 * rsa11].real, a11[4 * rsa11].imag, n);
    temp = VSET_V_F(PREC, LMUL, 2)(temp, 0, temp_r);
    temp = VSET_V_F(PREC, LMUL, 2)(temp, 1, temp_i);
    VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c + 4 * rsc), temp, n);
    if (csc11 == 1)
        VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c11 + 4 * rsc11), temp, n);
    else
        VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c11 + 4 * rsc11), 2 * FLT_SIZE * csc11, temp, n);
    if (m == 5) return;
    VCNMSAC_VF(PREC, LMUL, ab5_r, ab5_i, a11[5 * rsa11].real, a11[5 * rsa11].imag, temp_r, temp_i, n);
    a11 += csa11;

    VCMUL_VF(PREC, LMUL, temp_r, temp_i, ab5_r, ab5_i, a11[5 * rsa11].real, a11[5 * rsa11].imag, n);
    temp = VSET_V_F(PREC, LMUL, 2)(temp, 0, temp_r);
    temp = VSET_V_F(PREC, LMUL, 2)(temp, 1, temp_i);
    VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c + 5 * rsc), temp, n);
    if (csc11 == 1)
        VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c11 + 5 * rsc11), temp, n);
    else
        VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(c11 + 5 * rsc11), 2 * FLT_SIZE * csc11, temp, n);
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
