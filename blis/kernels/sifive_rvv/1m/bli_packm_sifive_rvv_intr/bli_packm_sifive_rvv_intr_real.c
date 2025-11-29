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
#ifdef PACKM

PACKM(PRECISION_CHAR, void)
{
    (void) schema; // Suppress unused parameter warnings
    (void) params;
    (void) cntx;
    const DATATYPE* restrict kappa = kappa_;
    const DATATYPE* restrict a = a_;
    DATATYPE* restrict p = p_;

    // MRxk kernel
    if (cdim <= MR && cdim_max == MR && cdim_bcast == 1)
    {
        if (lda == 1) {
            // a is "row major"
            // pad the lower edge with zeros
            RVV_TYPE_F(PREC, LMUL) arow0, arow1, arow2, arow3, arow4, arow5, arow6;
            switch (cdim) {
            case 0:
                arow0 = VFMV_V_F(PREC, LMUL)(0., n);
            case 1:
                arow1 = VFMV_V_F(PREC, LMUL)(0., n);
            case 2:
                arow2 = VFMV_V_F(PREC, LMUL)(0., n);
            case 3:
                arow3 = VFMV_V_F(PREC, LMUL)(0., n);
            case 4:
                arow4 = VFMV_V_F(PREC, LMUL)(0., n);
            case 5:
                arow5 = VFMV_V_F(PREC, LMUL)(0., n);
            case 6:
                arow6 = VFMV_V_F(PREC, LMUL)(0., n);
            }

            size_t avl = n;
            while (avl) {
                size_t vl = VSETVL(PREC, LMUL)(avl);
                switch (cdim) {
                    case 7:
                        arow6 = VLE_V_F(PREC, LMUL)(a + 6 * inca, vl);
                    case 6:
                        arow5 = VLE_V_F(PREC, LMUL)(a + 5 * inca, vl);
                    case 5:
                        arow4 = VLE_V_F(PREC, LMUL)(a + 4 * inca, vl);
                    case 4:
                        arow3 = VLE_V_F(PREC, LMUL)(a + 3 * inca, vl);
                    case 3:
                        arow2 = VLE_V_F(PREC, LMUL)(a + 2 * inca, vl);
                    case 2:
                        arow1 = VLE_V_F(PREC, LMUL)(a + 1 * inca, vl);
                    case 1:
                        arow0 = VLE_V_F(PREC, LMUL)(a + 0 * inca, vl);
                }
                
                if (!PASTEMAC(PRECISION_CHAR, eq1)(*kappa)) {
                    switch (cdim) {
                        case 7:
                            arow6 = VFMUL_VF(PREC, LMUL)(arow6, *kappa, vl);
                        case 6:
                            arow5 = VFMUL_VF(PREC, LMUL)(arow5, *kappa, vl);
                        case 5:
                            arow4 = VFMUL_VF(PREC, LMUL)(arow4, *kappa, vl);
                        case 4:
                            arow3 = VFMUL_VF(PREC, LMUL)(arow3, *kappa, vl);
                        case 3:
                            arow2 = VFMUL_VF(PREC, LMUL)(arow2, *kappa, vl);
                        case 2:
                            arow1 = VFMUL_VF(PREC, LMUL)(arow1, *kappa, vl);
                        case 1:
                            arow0 = VFMUL_VF(PREC, LMUL)(arow0, *kappa, vl);
                    }
                }

                RVV_TYPE_FX(PREC, LMUL, 7) ablock = VUNDEFINED_FX(PREC, LMUL, 7)();
                ablock = VSET_V_F(PREC, LMUL, 7)(ablock, 0, arow0); 
                ablock = VSET_V_F(PREC, LMUL, 7)(ablock, 1, arow1); 
                ablock = VSET_V_F(PREC, LMUL, 7)(ablock, 2, arow2); 
                ablock = VSET_V_F(PREC, LMUL, 7)(ablock, 3, arow3); 
                ablock = VSET_V_F(PREC, LMUL, 7)(ablock, 4, arow4); 
                ablock = VSET_V_F(PREC, LMUL, 7)(ablock, 5, arow5); 
                ablock = VSET_V_F(PREC, LMUL, 7)(ablock, 6, arow6); 
                VSSSEG7_V_F(PREC, LMUL, 7)(p, FLT_SIZE * ldp, ablock, vl);

                a += vl;
                p += vl * ldp;
                avl -= vl;
            }

            RVV_TYPE_F(PREC, LMUL_MR) zero_padding = VFMV_V_F(PREC, LMUL_MR)(0., cdim_max);
            for (size_t i = n; i < n_max; ++i) {
                VSE_V_F(PREC, LMUL_MR)(p, zero_padding, cdim_max);
                p += ldp;
            }
        }
        else {
            RVV_TYPE_F(PREC, LMUL_MR) zero_padding = VFMV_V_F(PREC, LMUL_MR)(0., cdim_max);
            for (size_t i = 0; i < n; ++i) {
                RVV_TYPE_F(PREC, LMUL_MR) acol_vec;
                if (inca == 1)
                    acol_vec = VLE_V_F_TU(PREC, LMUL_MR)(zero_padding, a, cdim);
                else
                    acol_vec = VLSE_V_F_TU(PREC, LMUL_MR)(zero_padding, a, FLT_SIZE * inca, cdim);

                if (!PASTEMAC(PRECISION_CHAR, eq1)(*kappa))
                    acol_vec = VFMUL_VF_TU(PREC, LMUL_MR)(acol_vec, acol_vec, *kappa, cdim);

                VSE_V_F(PREC, LMUL_MR)(p, acol_vec, cdim_max);
                 
                a += lda;
                p += ldp;
            }

            for (size_t i = n; i < n_max; ++i) {
                VSE_V_F(PREC, LMUL_MR)(p, zero_padding, cdim_max);
                p += ldp;
            }
        }
    }
    // NRxk kernel
    else if (cdim <= NR && cdim_max == NR && cdim_bcast == 1)
    {
        if (lda == 1) {
            // a is "row major"
            RVV_TYPE_F(PREC, LMUL_NR) zero_padding = VFMV_V_F(PREC, LMUL_NR)(0., cdim_max);
            size_t avl = n;
            while (avl) {
                size_t vl = VSETVL(PREC, LMUL)(avl);
                dim_t cdim_tmp = cdim;
                const DATATYPE* restrict a_tmp = a;
                DATATYPE* restrict p_tmp = p;
                while (cdim_tmp >= 8) {
                    RVV_TYPE_F(PREC, LMUL) arow0, arow1, arow2, arow3, arow4, arow5, arow6, arow7;
                    arow0 = VLE_V_F(PREC, LMUL)(a_tmp + 0 * inca, vl);
                    arow1 = VLE_V_F(PREC, LMUL)(a_tmp + 1 * inca, vl);
                    arow2 = VLE_V_F(PREC, LMUL)(a_tmp + 2 * inca, vl);
                    arow3 = VLE_V_F(PREC, LMUL)(a_tmp + 3 * inca, vl);
                    arow4 = VLE_V_F(PREC, LMUL)(a_tmp + 4 * inca, vl);
                    arow5 = VLE_V_F(PREC, LMUL)(a_tmp + 5 * inca, vl);
                    arow6 = VLE_V_F(PREC, LMUL)(a_tmp + 6 * inca, vl);
                    arow7 = VLE_V_F(PREC, LMUL)(a_tmp + 7 * inca, vl);
                    
                    if (!PASTEMAC(PRECISION_CHAR, eq1)(*kappa)) {
                        arow0 = VFMUL_VF(PREC, LMUL)(arow0, *kappa, vl);
                        arow1 = VFMUL_VF(PREC, LMUL)(arow1, *kappa, vl);
                        arow2 = VFMUL_VF(PREC, LMUL)(arow2, *kappa, vl);
                        arow3 = VFMUL_VF(PREC, LMUL)(arow3, *kappa, vl);
                        arow4 = VFMUL_VF(PREC, LMUL)(arow4, *kappa, vl);
                        arow5 = VFMUL_VF(PREC, LMUL)(arow5, *kappa, vl);
                        arow6 = VFMUL_VF(PREC, LMUL)(arow6, *kappa, vl);
                        arow7 = VFMUL_VF(PREC, LMUL)(arow7, *kappa, vl);
                    }
                    
                    RVV_TYPE_FX(PREC, LMUL, 8) ablock = VUNDEFINED_FX(PREC, LMUL, 8)();
                    ablock = VSET_V_F(PREC, LMUL, 8)(ablock, 0, arow0); 
                    ablock = VSET_V_F(PREC, LMUL, 8)(ablock, 1, arow1); 
                    ablock = VSET_V_F(PREC, LMUL, 8)(ablock, 2, arow2); 
                    ablock = VSET_V_F(PREC, LMUL, 8)(ablock, 3, arow3); 
                    ablock = VSET_V_F(PREC, LMUL, 8)(ablock, 4, arow4); 
                    ablock = VSET_V_F(PREC, LMUL, 8)(ablock, 5, arow5); 
                    ablock = VSET_V_F(PREC, LMUL, 8)(ablock, 6, arow6); 
                    ablock = VSET_V_F(PREC, LMUL, 8)(ablock, 7, arow7); 
                    VSSSEG8_V_F(PREC, LMUL, 8)(p_tmp, FLT_SIZE * ldp, ablock, vl);
                    
                    a_tmp += 8 * inca;
                    p_tmp += 8;
                    cdim_tmp -= 8;
                }

                if (cdim_tmp > 0) {
                    RVV_TYPE_F(PREC, LMUL) arow0, arow1, arow2, arow3, arow4, arow5, arow6;
                    switch (cdim_tmp) {
                    case 7:
                        arow6 = VLE_V_F(PREC, LMUL)(a_tmp + 6 * inca, vl);
                    case 6:
                        arow5 = VLE_V_F(PREC, LMUL)(a_tmp + 5 * inca, vl);
                    case 5:
                        arow4 = VLE_V_F(PREC, LMUL)(a_tmp + 4 * inca, vl);
                    case 4:
                        arow3 = VLE_V_F(PREC, LMUL)(a_tmp + 3 * inca, vl);
                    case 3:
                        arow2 = VLE_V_F(PREC, LMUL)(a_tmp + 2 * inca, vl);
                    case 2:
                        arow1 = VLE_V_F(PREC, LMUL)(a_tmp + 1 * inca, vl);
                    case 1:
                        arow0 = VLE_V_F(PREC, LMUL)(a_tmp + 0 * inca, vl);
                    }

                    if (!PASTEMAC(PRECISION_CHAR, eq1)(*kappa)) {
                        switch (cdim_tmp) {
                        case 7:
                            arow6 = VFMUL_VF(PREC, LMUL)(arow6, *kappa, vl);
                        case 6:
                            arow5 = VFMUL_VF(PREC, LMUL)(arow5, *kappa, vl);
                        case 5:
                            arow4 = VFMUL_VF(PREC, LMUL)(arow4, *kappa, vl);
                        case 4:
                            arow3 = VFMUL_VF(PREC, LMUL)(arow3, *kappa, vl);
                        case 3:
                            arow2 = VFMUL_VF(PREC, LMUL)(arow2, *kappa, vl);
                        case 2:
                            arow1 = VFMUL_VF(PREC, LMUL)(arow1, *kappa, vl);
                        case 1:
                            arow0 = VFMUL_VF(PREC, LMUL)(arow0, *kappa, vl);
                        }
                    }

                    RVV_TYPE_FX(PREC, LMUL, 7) ablock7 = VUNDEFINED_FX(PREC, LMUL, 7)();
                    RVV_TYPE_FX(PREC, LMUL, 6) ablock6 = VUNDEFINED_FX(PREC, LMUL, 6)();
                    RVV_TYPE_FX(PREC, LMUL, 5) ablock5 = VUNDEFINED_FX(PREC, LMUL, 5)();
                    RVV_TYPE_FX(PREC, LMUL, 4) ablock4 = VUNDEFINED_FX(PREC, LMUL, 4)();
                    RVV_TYPE_FX(PREC, LMUL, 3) ablock3 = VUNDEFINED_FX(PREC, LMUL, 3)();
                    RVV_TYPE_FX(PREC, LMUL, 2) ablock2 = VUNDEFINED_FX(PREC, LMUL, 2)();
                    switch (cdim_tmp) {
                    case 7:
                        ablock7 = VSET_V_F(PREC, LMUL, 7)(ablock7, 0, arow0); 
                        ablock7 = VSET_V_F(PREC, LMUL, 7)(ablock7, 1, arow1); 
                        ablock7 = VSET_V_F(PREC, LMUL, 7)(ablock7, 2, arow2); 
                        ablock7 = VSET_V_F(PREC, LMUL, 7)(ablock7, 3, arow3); 
                        ablock7 = VSET_V_F(PREC, LMUL, 7)(ablock7, 4, arow4); 
                        ablock7 = VSET_V_F(PREC, LMUL, 7)(ablock7, 5, arow5); 
                        ablock7 = VSET_V_F(PREC, LMUL, 7)(ablock7, 6, arow6); 
                        VSSSEG7_V_F(PREC, LMUL, 7)(p_tmp, FLT_SIZE * ldp, ablock7, vl);
                        break;
                    case 6:
                        ablock6 = VSET_V_F(PREC, LMUL, 6)(ablock6, 0, arow0); 
                        ablock6 = VSET_V_F(PREC, LMUL, 6)(ablock6, 1, arow1); 
                        ablock6 = VSET_V_F(PREC, LMUL, 6)(ablock6, 2, arow2); 
                        ablock6 = VSET_V_F(PREC, LMUL, 6)(ablock6, 3, arow3); 
                        ablock6 = VSET_V_F(PREC, LMUL, 6)(ablock6, 4, arow4); 
                        ablock6 = VSET_V_F(PREC, LMUL, 6)(ablock6, 5, arow5); 
                        VSSSEG6_V_F(PREC, LMUL, 6)(p_tmp, FLT_SIZE * ldp, ablock6, vl);
                        break;
                    case 5:
                        ablock5 = VSET_V_F(PREC, LMUL, 5)(ablock5, 0, arow0); 
                        ablock5 = VSET_V_F(PREC, LMUL, 5)(ablock5, 1, arow1); 
                        ablock5 = VSET_V_F(PREC, LMUL, 5)(ablock5, 2, arow2); 
                        ablock5 = VSET_V_F(PREC, LMUL, 5)(ablock5, 3, arow3); 
                        ablock5 = VSET_V_F(PREC, LMUL, 5)(ablock5, 4, arow4); 
                        VSSSEG5_V_F(PREC, LMUL, 5)(p_tmp, FLT_SIZE * ldp, ablock5, vl);
                        break;
                    case 4:
                        ablock4 = VSET_V_F(PREC, LMUL, 4)(ablock4, 0, arow0); 
                        ablock4 = VSET_V_F(PREC, LMUL, 4)(ablock4, 1, arow1); 
                        ablock4 = VSET_V_F(PREC, LMUL, 4)(ablock4, 2, arow2); 
                        ablock4 = VSET_V_F(PREC, LMUL, 4)(ablock4, 3, arow3); 
                        VSSSEG4_V_F(PREC, LMUL, 4)(p_tmp, FLT_SIZE * ldp, ablock4, vl);
                        break;
                    case 3:
                        ablock3 = VSET_V_F(PREC, LMUL, 3)(ablock3, 0, arow0); 
                        ablock3 = VSET_V_F(PREC, LMUL, 3)(ablock3, 1, arow1); 
                        ablock3 = VSET_V_F(PREC, LMUL, 3)(ablock3, 2, arow2); 
                        VSSSEG3_V_F(PREC, LMUL, 3)(p_tmp, FLT_SIZE * ldp, ablock3, vl);
                        break;
                    case 2:
                        ablock2 = VSET_V_F(PREC, LMUL, 2)(ablock2, 0, arow0); 
                        ablock2 = VSET_V_F(PREC, LMUL, 2)(ablock2, 1, arow1); 
                        VSSSEG2_V_F(PREC, LMUL, 2)(p_tmp, FLT_SIZE * ldp, ablock2, vl);
                        break;
                    case 1:
                        VSSE_V_F(PREC, LMUL)(p_tmp, FLT_SIZE * ldp, arow0, vl);
                        break;
                    }
                    p_tmp += cdim_tmp;
                }

                for (size_t i = 0; i < vl; ++i) {
                    VSE_V_F(PREC, LMUL_NR)(p_tmp, zero_padding, cdim_max - cdim);
                    p_tmp += ldp;
                }

                a += vl;
                p += vl * ldp;
                avl -= vl;
            }

            for (size_t i = n; i < n_max; ++i) {
                VSE_V_F(PREC, LMUL_NR)(p, zero_padding, cdim_max);
                p += ldp;
            }
        } else {
            RVV_TYPE_F(PREC, LMUL_NR) zero_padding = VFMV_V_F(PREC, LMUL_NR)(0., cdim_max);
            for (size_t i = 0; i < n; ++i) {
                RVV_TYPE_F(PREC, LMUL_NR) acol_vec;
                if (inca == 1)
                    acol_vec = VLE_V_F_TU(PREC, LMUL_NR)(zero_padding, a, cdim);
                else
                    acol_vec = VLSE_V_F_TU(PREC, LMUL_NR)(zero_padding, a, FLT_SIZE * inca, cdim);

                if (!PASTEMAC(PRECISION_CHAR, eq1)(*kappa))
                    acol_vec = VFMUL_VF_TU(PREC, LMUL_NR)(acol_vec, acol_vec, *kappa, cdim);

                VSE_V_F(PREC, LMUL_NR)(p, acol_vec, cdim_max);
                 
                a += lda;
                p += ldp;
            }

            for (size_t i = n; i < n_max; ++i) {
                VSE_V_F(PREC, LMUL_NR)(p, zero_padding, cdim_max);
                p += ldp;
            }
        }
    }
    // generic kernel
    else
    {
        BLI_SCAL2BBS_MXN(PRECISION_CHAR)
        (
          conja,
          cdim,
          n,
          kappa,
          a,       inca, lda,
          p, cdim_bcast, ldp
        );

        BLI_SET0S_EDGE(PRECISION_CHAR)
        (
          cdim*cdim_bcast, cdim_max*cdim_bcast,
          n, n_max,
          p, ldp
        );
    }

    return;
}

#endif // PACKM
