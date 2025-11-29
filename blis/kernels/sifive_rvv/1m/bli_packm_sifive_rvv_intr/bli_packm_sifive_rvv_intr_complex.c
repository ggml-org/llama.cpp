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
            RVV_TYPE_F(PREC, LMUL) arow0_r, arow1_r, arow2_r, arow3_r, arow4_r, arow5_r;
            RVV_TYPE_F(PREC, LMUL) arow0_i, arow1_i, arow2_i, arow3_i, arow4_i, arow5_i;
            RVV_TYPE_F(PREC, LMUL) kappa_arow0_r, kappa_arow1_r, kappa_arow2_r,
                                   kappa_arow3_r, kappa_arow4_r, kappa_arow5_r;
            RVV_TYPE_F(PREC, LMUL) kappa_arow0_i, kappa_arow1_i, kappa_arow2_i,
                                   kappa_arow3_i, kappa_arow4_i, kappa_arow5_i;
            // pad lower edge
            if (PASTEMAC(PRECISION_CHAR, eq1)(*kappa)) {
                switch (cdim) {
                case 0:
                    arow0_r = VFMV_V_F(PREC, LMUL)(0., n);
                    arow0_i = VFMV_V_F(PREC, LMUL)(0., n);
                case 1:
                    arow1_r = VFMV_V_F(PREC, LMUL)(0., n);
                    arow1_i = VFMV_V_F(PREC, LMUL)(0., n);
                case 2:
                    arow2_r = VFMV_V_F(PREC, LMUL)(0., n);
                    arow2_i = VFMV_V_F(PREC, LMUL)(0., n);
                case 3:
                    arow3_r = VFMV_V_F(PREC, LMUL)(0., n);
                    arow3_i = VFMV_V_F(PREC, LMUL)(0., n);
                case 4:
                    arow4_r = VFMV_V_F(PREC, LMUL)(0., n);
                    arow4_i = VFMV_V_F(PREC, LMUL)(0., n);
                case 5:
                    arow5_r = VFMV_V_F(PREC, LMUL)(0., n);
                    arow5_i = VFMV_V_F(PREC, LMUL)(0., n);
                }
            } else {
                switch (cdim) {
                case 0:
                    kappa_arow0_r = VFMV_V_F(PREC, LMUL)(0., n);
                    kappa_arow0_i = VFMV_V_F(PREC, LMUL)(0., n);
                case 1:
                    kappa_arow1_r = VFMV_V_F(PREC, LMUL)(0., n);
                    kappa_arow1_i = VFMV_V_F(PREC, LMUL)(0., n);
                case 2:
                    kappa_arow2_r = VFMV_V_F(PREC, LMUL)(0., n);
                    kappa_arow2_i = VFMV_V_F(PREC, LMUL)(0., n);
                case 3:
                    kappa_arow3_r = VFMV_V_F(PREC, LMUL)(0., n);
                    kappa_arow3_i = VFMV_V_F(PREC, LMUL)(0., n);
                case 4:
                    kappa_arow4_r = VFMV_V_F(PREC, LMUL)(0., n);
                    kappa_arow4_i = VFMV_V_F(PREC, LMUL)(0., n);
                case 5:
                    kappa_arow5_r = VFMV_V_F(PREC, LMUL)(0., n);
                    kappa_arow5_i = VFMV_V_F(PREC, LMUL)(0., n);
                }
            }

            size_t avl = n;
            while (avl) {
                size_t vl = VSETVL(PREC, LMUL)(avl);
                RVV_TYPE_FX(PREC, LMUL, 2) arow_vec;
                switch (cdim) {
                case 6:
                    arow_vec = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(a + 5 * inca), vl);
                    arow5_r = VGET_V_F(PREC, LMUL, 2)(arow_vec, 0);
                    arow5_i = VGET_V_F(PREC, LMUL, 2)(arow_vec, 1);
                case 5:
                    arow_vec = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(a + 4 * inca), vl);
                    arow4_r = VGET_V_F(PREC, LMUL, 2)(arow_vec, 0);
                    arow4_i = VGET_V_F(PREC, LMUL, 2)(arow_vec, 1);
                case 4:
                    arow_vec = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(a + 3 * inca), vl);
                    arow3_r = VGET_V_F(PREC, LMUL, 2)(arow_vec, 0);
                    arow3_i = VGET_V_F(PREC, LMUL, 2)(arow_vec, 1);
                case 3:
                    arow_vec = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(a + 2 * inca), vl);
                    arow2_r = VGET_V_F(PREC, LMUL, 2)(arow_vec, 0);
                    arow2_i = VGET_V_F(PREC, LMUL, 2)(arow_vec, 1);
                case 2:
                    arow_vec = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(a + 1 * inca), vl);
                    arow1_r = VGET_V_F(PREC, LMUL, 2)(arow_vec, 0);
                    arow1_i = VGET_V_F(PREC, LMUL, 2)(arow_vec, 1);
                case 1:
                    arow_vec = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(a + 0 * inca), vl);
                    arow0_r = VGET_V_F(PREC, LMUL, 2)(arow_vec, 0);
                    arow0_i = VGET_V_F(PREC, LMUL, 2)(arow_vec, 1);
                }

                if (PASTEMAC(PRECISION_CHAR, eq1)(*kappa)) {
                    if (bli_is_conj(conja)) {
                        switch (cdim) {
                        case 6:
                            arow5_i = VFNEG_VF(PREC, LMUL)(arow5_i, vl);
                        case 5:
                            arow4_i = VFNEG_VF(PREC, LMUL)(arow4_i, vl);
                        case 4:
                            arow3_i = VFNEG_VF(PREC, LMUL)(arow3_i, vl);
                        case 3:
                            arow2_i = VFNEG_VF(PREC, LMUL)(arow2_i, vl);
                        case 2:
                            arow1_i = VFNEG_VF(PREC, LMUL)(arow1_i, vl);
                        case 1:
                            arow0_i = VFNEG_VF(PREC, LMUL)(arow0_i, vl);
                        }
                    }

                    RVV_TYPE_FX(PREC, LMUL, 6) ablock = VUNDEFINED_FX(PREC, LMUL, 6)();
                    ablock = VSET_V_F(PREC, LMUL, 6)(ablock, 0, arow0_r);
                    ablock = VSET_V_F(PREC, LMUL, 6)(ablock, 1, arow0_i);
                    ablock = VSET_V_F(PREC, LMUL, 6)(ablock, 2, arow1_r);
                    ablock = VSET_V_F(PREC, LMUL, 6)(ablock, 3, arow1_i);
                    ablock = VSET_V_F(PREC, LMUL, 6)(ablock, 4, arow2_r);
                    ablock = VSET_V_F(PREC, LMUL, 6)(ablock, 5, arow2_i);
                    VSSSEG6_V_F(PREC, LMUL, 6)((BASE_DT*) p, 2 * FLT_SIZE * ldp, ablock, vl);
                    ablock = VSET_V_F(PREC, LMUL, 6)(ablock, 0, arow3_r);
                    ablock = VSET_V_F(PREC, LMUL, 6)(ablock, 1, arow3_i);
                    ablock = VSET_V_F(PREC, LMUL, 6)(ablock, 2, arow4_r);
                    ablock = VSET_V_F(PREC, LMUL, 6)(ablock, 3, arow4_i);
                    ablock = VSET_V_F(PREC, LMUL, 6)(ablock, 4, arow5_r);
                    ablock = VSET_V_F(PREC, LMUL, 6)(ablock, 5, arow5_i);
                    VSSSEG6_V_F(PREC, LMUL, 6)((BASE_DT*)(p + 3), 2 * FLT_SIZE * ldp, ablock, vl);
                } else {
                    if (bli_is_conj(conja)) {
                        switch (cdim) {
                        case 6:
                            VCMUL_VF_CONJ(PREC, LMUL, kappa_arow5_r, kappa_arow5_i, arow5_r, arow5_i, kappa->real, kappa->imag, vl);
                        case 5:
                            VCMUL_VF_CONJ(PREC, LMUL, kappa_arow4_r, kappa_arow4_i, arow4_r, arow4_i, kappa->real, kappa->imag, vl);
                        case 4:
                            VCMUL_VF_CONJ(PREC, LMUL, kappa_arow3_r, kappa_arow3_i, arow3_r, arow3_i, kappa->real, kappa->imag, vl);
                        case 3:
                            VCMUL_VF_CONJ(PREC, LMUL, kappa_arow2_r, kappa_arow2_i, arow2_r, arow2_i, kappa->real, kappa->imag, vl);
                        case 2:
                            VCMUL_VF_CONJ(PREC, LMUL, kappa_arow1_r, kappa_arow1_i, arow1_r, arow1_i, kappa->real, kappa->imag, vl);
                        case 1:
                            VCMUL_VF_CONJ(PREC, LMUL, kappa_arow0_r, kappa_arow0_i, arow0_r, arow0_i, kappa->real, kappa->imag, vl);
                        }
                    } else {
                        switch (cdim) {
                        case 6:
                            VCMUL_VF(PREC, LMUL, kappa_arow5_r, kappa_arow5_i, arow5_r, arow5_i, kappa->real, kappa->imag, vl);
                        case 5:
                            VCMUL_VF(PREC, LMUL, kappa_arow4_r, kappa_arow4_i, arow4_r, arow4_i, kappa->real, kappa->imag, vl);
                        case 4:
                            VCMUL_VF(PREC, LMUL, kappa_arow3_r, kappa_arow3_i, arow3_r, arow3_i, kappa->real, kappa->imag, vl);
                        case 3:
                            VCMUL_VF(PREC, LMUL, kappa_arow2_r, kappa_arow2_i, arow2_r, arow2_i, kappa->real, kappa->imag, vl);
                        case 2:
                            VCMUL_VF(PREC, LMUL, kappa_arow1_r, kappa_arow1_i, arow1_r, arow1_i, kappa->real, kappa->imag, vl);
                        case 1:
                            VCMUL_VF(PREC, LMUL, kappa_arow0_r, kappa_arow0_i, arow0_r, arow0_i, kappa->real, kappa->imag, vl);
                        }
                    }

                    RVV_TYPE_FX(PREC, LMUL, 6) ablock = VUNDEFINED_FX(PREC, LMUL, 6)();
                    ablock = VSET_V_F(PREC, LMUL, 6)(ablock, 0, kappa_arow0_r);
                    ablock = VSET_V_F(PREC, LMUL, 6)(ablock, 1, kappa_arow0_i);
                    ablock = VSET_V_F(PREC, LMUL, 6)(ablock, 2, kappa_arow1_r);
                    ablock = VSET_V_F(PREC, LMUL, 6)(ablock, 3, kappa_arow1_i);
                    ablock = VSET_V_F(PREC, LMUL, 6)(ablock, 4, kappa_arow2_r);
                    ablock = VSET_V_F(PREC, LMUL, 6)(ablock, 5, kappa_arow2_i);
                    VSSSEG6_V_F(PREC, LMUL, 6)((BASE_DT*) p, 2 * FLT_SIZE * ldp, ablock, vl);
                    ablock = VSET_V_F(PREC, LMUL, 6)(ablock, 0, kappa_arow3_r);
                    ablock = VSET_V_F(PREC, LMUL, 6)(ablock, 1, kappa_arow3_i);
                    ablock = VSET_V_F(PREC, LMUL, 6)(ablock, 2, kappa_arow4_r);
                    ablock = VSET_V_F(PREC, LMUL, 6)(ablock, 3, kappa_arow4_i);
                    ablock = VSET_V_F(PREC, LMUL, 6)(ablock, 4, kappa_arow5_r);
                    ablock = VSET_V_F(PREC, LMUL, 6)(ablock, 5, kappa_arow5_i);
                    VSSSEG6_V_F(PREC, LMUL, 6)((BASE_DT*)(p + 3), 2 * FLT_SIZE * ldp, ablock, vl);
                }

                a += vl;
                p += vl * ldp;
                avl -= vl;
            }
            
            RVV_TYPE_FX(PREC, LMUL_MR, 2) zero_padding = VUNDEFINED_FX(PREC, LMUL_MR, 2)();
            zero_padding = VSET_V_F(PREC, LMUL_MR, 2)(zero_padding, 0, VFMV_V_F(PREC, LMUL_MR)(0., cdim_max));
            zero_padding = VSET_V_F(PREC, LMUL_MR, 2)(zero_padding, 1, VFMV_V_F(PREC, LMUL_MR)(0., cdim_max));
            for (size_t i = n; i < n_max; ++i) {
                VSSEG2_V_F(PREC, LMUL_MR, 2)((BASE_DT*) p, zero_padding, cdim_max);
                p += ldp;
            }
        }
        else {
            RVV_TYPE_FX(PREC, LMUL_MR, 2) zero_padding = VUNDEFINED_FX(PREC, LMUL_MR, 2)();
            zero_padding = VSET_V_F(PREC, LMUL_MR, 2)(zero_padding, 0, VFMV_V_F(PREC, LMUL_MR)(0., cdim_max));
            zero_padding = VSET_V_F(PREC, LMUL_MR, 2)(zero_padding, 1, VFMV_V_F(PREC, LMUL_MR)(0., cdim_max));

            for (size_t i = 0; i < n; ++i) {
                RVV_TYPE_FX(PREC, LMUL_MR, 2) acol;
                if (inca == 1)
                    acol = VLSEG2_V_F_TU(PREC, LMUL_MR, 2)(zero_padding, (BASE_DT*) a, cdim);
                else
                    acol = VLSSEG2_V_F_TU(PREC, LMUL_MR, 2)(zero_padding, (BASE_DT*) a, 2 * FLT_SIZE * inca, cdim);
                RVV_TYPE_F(PREC, LMUL_MR) acol_r = VGET_V_F(PREC, LMUL_MR, 2)(acol, 0); 
                RVV_TYPE_F(PREC, LMUL_MR) acol_i = VGET_V_F(PREC, LMUL_MR, 2)(acol, 1); 

                if (PASTEMAC(PRECISION_CHAR, eq1)(*kappa)) {
                    if (bli_is_conj(conja)) {
                        acol_i = VFNEG_VF_TU(PREC, LMUL_MR)(acol_i, acol_i, cdim);
                        acol = VSET_V_F(PREC, LMUL_MR, 2)(acol, 0, acol_r);
                        acol = VSET_V_F(PREC, LMUL_MR, 2)(acol, 1, acol_i);
                    }
                } else {
                    RVV_TYPE_F(PREC, LMUL_MR) kappa_acol_r, kappa_acol_i;
                    if (bli_is_conj(conja))
                        VCMUL_VF_CONJ_TU(PREC, LMUL_MR, kappa_acol_r, kappa_acol_i, acol_r, acol_i, kappa->real, kappa->imag, cdim);
                    else
                        VCMUL_VF_TU(PREC, LMUL_MR, kappa_acol_r, kappa_acol_i, acol_r, acol_i, kappa->real, kappa->imag, cdim);
                    acol = VSET_V_F(PREC, LMUL_MR, 2)(acol, 0, kappa_acol_r);
                    acol = VSET_V_F(PREC, LMUL_MR, 2)(acol, 1, kappa_acol_i);
                }

                VSSEG2_V_F(PREC, LMUL_MR, 2)((BASE_DT*) p, acol, cdim_max);
                 
                a += lda;
                p += ldp;
            }

            for (size_t i = n; i < n_max; ++i) {
                VSSEG2_V_F(PREC, LMUL_MR, 2)((BASE_DT*) p, zero_padding, cdim_max);
                p += ldp;
            }
        }
    }
    // NRxk kernel
    else if (cdim <= NR && cdim_max == NR && cdim_bcast == 1)
    {
        if (lda == 1) {
            // a is "row major"
            RVV_TYPE_FX(PREC, LMUL_NR, 2) zero_padding = VUNDEFINED_FX(PREC, LMUL_NR, 2)();
            zero_padding = VSET_V_F(PREC, LMUL_NR, 2)(zero_padding, 0, VFMV_V_F(PREC, LMUL_NR)(0., cdim_max));
            zero_padding = VSET_V_F(PREC, LMUL_NR, 2)(zero_padding, 1, VFMV_V_F(PREC, LMUL_NR)(0., cdim_max));
            size_t avl = n;
            while (avl) {
                size_t vl = VSETVL(PREC, LMUL)(avl);
                dim_t cdim_tmp = cdim;
                const DATATYPE* restrict a_tmp = a;
                DATATYPE* restrict p_tmp = p;
                while (cdim_tmp >= 4) {
                    RVV_TYPE_FX(PREC, LMUL, 2) arow_vec;
                    RVV_TYPE_F(PREC, LMUL) arow0_r, arow1_r, arow2_r, arow3_r;
                    RVV_TYPE_F(PREC, LMUL) arow0_i, arow1_i, arow2_i, arow3_i;
                    arow_vec = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(a_tmp + 0 * inca), vl);
                    arow0_r = VGET_V_F(PREC, LMUL, 2)(arow_vec, 0);
                    arow0_i = VGET_V_F(PREC, LMUL, 2)(arow_vec, 1);
                    arow_vec = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(a_tmp + 1 * inca), vl);
                    arow1_r = VGET_V_F(PREC, LMUL, 2)(arow_vec, 0);
                    arow1_i = VGET_V_F(PREC, LMUL, 2)(arow_vec, 1);
                    arow_vec = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(a_tmp + 2 * inca), vl);
                    arow2_r = VGET_V_F(PREC, LMUL, 2)(arow_vec, 0);
                    arow2_i = VGET_V_F(PREC, LMUL, 2)(arow_vec, 1);
                    arow_vec = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(a_tmp + 3 * inca), vl);
                    arow3_r = VGET_V_F(PREC, LMUL, 2)(arow_vec, 0);
                    arow3_i = VGET_V_F(PREC, LMUL, 2)(arow_vec, 1);

                    if (PASTEMAC(PRECISION_CHAR, eq1)(*kappa)) {
                        if (bli_is_conj(conja)) {
                            arow0_i = VFNEG_VF(PREC, LMUL)(arow0_i, vl);
                            arow1_i = VFNEG_VF(PREC, LMUL)(arow1_i, vl);
                            arow2_i = VFNEG_VF(PREC, LMUL)(arow2_i, vl);
                            arow3_i = VFNEG_VF(PREC, LMUL)(arow3_i, vl);
                        }

                        RVV_TYPE_FX(PREC, LMUL, 8) ablock = VUNDEFINED_FX(PREC, LMUL, 8)();
                        ablock = VSET_V_F(PREC, LMUL, 8)(ablock, 0, arow0_r);
                        ablock = VSET_V_F(PREC, LMUL, 8)(ablock, 1, arow0_i);
                        ablock = VSET_V_F(PREC, LMUL, 8)(ablock, 2, arow1_r);
                        ablock = VSET_V_F(PREC, LMUL, 8)(ablock, 3, arow1_i);
                        ablock = VSET_V_F(PREC, LMUL, 8)(ablock, 4, arow2_r);
                        ablock = VSET_V_F(PREC, LMUL, 8)(ablock, 5, arow2_i);
                        ablock = VSET_V_F(PREC, LMUL, 8)(ablock, 6, arow3_r);
                        ablock = VSET_V_F(PREC, LMUL, 8)(ablock, 7, arow3_i);
                        VSSSEG8_V_F(PREC, LMUL, 8)((BASE_DT*) p_tmp, 2 * FLT_SIZE * ldp, ablock, vl);
                    } else {
                        RVV_TYPE_F(PREC, LMUL) kappa_arow0_r, kappa_arow1_r, kappa_arow2_r, kappa_arow3_r;
                        RVV_TYPE_F(PREC, LMUL) kappa_arow0_i, kappa_arow1_i, kappa_arow2_i, kappa_arow3_i;
                        if (bli_is_conj(conja)) {
                            VCMUL_VF_CONJ(PREC, LMUL, kappa_arow0_r, kappa_arow0_i, arow0_r, arow0_i, kappa->real, kappa->imag, vl);
                            VCMUL_VF_CONJ(PREC, LMUL, kappa_arow1_r, kappa_arow1_i, arow1_r, arow1_i, kappa->real, kappa->imag, vl);
                            VCMUL_VF_CONJ(PREC, LMUL, kappa_arow2_r, kappa_arow2_i, arow2_r, arow2_i, kappa->real, kappa->imag, vl);
                            VCMUL_VF_CONJ(PREC, LMUL, kappa_arow3_r, kappa_arow3_i, arow3_r, arow3_i, kappa->real, kappa->imag, vl);
                        } else {
                            VCMUL_VF(PREC, LMUL, kappa_arow0_r, kappa_arow0_i, arow0_r, arow0_i, kappa->real, kappa->imag, vl);
                            VCMUL_VF(PREC, LMUL, kappa_arow1_r, kappa_arow1_i, arow1_r, arow1_i, kappa->real, kappa->imag, vl);
                            VCMUL_VF(PREC, LMUL, kappa_arow2_r, kappa_arow2_i, arow2_r, arow2_i, kappa->real, kappa->imag, vl);
                            VCMUL_VF(PREC, LMUL, kappa_arow3_r, kappa_arow3_i, arow3_r, arow3_i, kappa->real, kappa->imag, vl);
                        }

                        RVV_TYPE_FX(PREC, LMUL, 8) ablock = VUNDEFINED_FX(PREC, LMUL, 8)();
                        ablock = VSET_V_F(PREC, LMUL, 8)(ablock, 0, kappa_arow0_r);
                        ablock = VSET_V_F(PREC, LMUL, 8)(ablock, 1, kappa_arow0_i);
                        ablock = VSET_V_F(PREC, LMUL, 8)(ablock, 2, kappa_arow1_r);
                        ablock = VSET_V_F(PREC, LMUL, 8)(ablock, 3, kappa_arow1_i);
                        ablock = VSET_V_F(PREC, LMUL, 8)(ablock, 4, kappa_arow2_r);
                        ablock = VSET_V_F(PREC, LMUL, 8)(ablock, 5, kappa_arow2_i);
                        ablock = VSET_V_F(PREC, LMUL, 8)(ablock, 6, kappa_arow3_r);
                        ablock = VSET_V_F(PREC, LMUL, 8)(ablock, 7, kappa_arow3_i);
                        VSSSEG8_V_F(PREC, LMUL, 8)((BASE_DT*) p_tmp, 2 * FLT_SIZE * ldp, ablock, vl);
                    }
                    
                    a_tmp += 4 * inca;
                    p_tmp += 4;
                    cdim_tmp -= 4;
                }

                if (cdim_tmp > 0) {
                    RVV_TYPE_FX(PREC, LMUL, 2) arow_vec;
                    RVV_TYPE_F(PREC, LMUL) arow0_r, arow1_r, arow2_r;
                    RVV_TYPE_F(PREC, LMUL) arow0_i, arow1_i, arow2_i;
                    switch (cdim_tmp) {
                    case 3:
                        arow_vec = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(a_tmp + 2 * inca), vl);
                        arow2_r = VGET_V_F(PREC, LMUL, 2)(arow_vec, 0);
                        arow2_i = VGET_V_F(PREC, LMUL, 2)(arow_vec, 1);
                    case 2:
                        arow_vec = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(a_tmp + 1 * inca), vl);
                        arow1_r = VGET_V_F(PREC, LMUL, 2)(arow_vec, 0);
                        arow1_i = VGET_V_F(PREC, LMUL, 2)(arow_vec, 1);
                    case 1:
                        arow_vec = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*)(a_tmp + 0 * inca), vl);
                        arow0_r = VGET_V_F(PREC, LMUL, 2)(arow_vec, 0);
                        arow0_i = VGET_V_F(PREC, LMUL, 2)(arow_vec, 1);
                    }

                    if (PASTEMAC(PRECISION_CHAR, eq1)(*kappa)) {
                        if (bli_is_conj(conja)) {
                            switch (cdim_tmp) {
                            case 3:
                                arow2_i = VFNEG_VF(PREC, LMUL)(arow2_i, vl);
                            case 2:
                                arow1_i = VFNEG_VF(PREC, LMUL)(arow1_i, vl);
                            case 1:
                                arow0_i = VFNEG_VF(PREC, LMUL)(arow0_i, vl);
                            }
                        }

                        RVV_TYPE_FX(PREC, LMUL, 6) ablock3 = VUNDEFINED_FX(PREC, LMUL, 6)();
                        RVV_TYPE_FX(PREC, LMUL, 4) ablock2 = VUNDEFINED_FX(PREC, LMUL, 4)();
                        RVV_TYPE_FX(PREC, LMUL, 2) ablock1 = VUNDEFINED_FX(PREC, LMUL, 2)();
                        switch (cdim_tmp) {
                        case 3:
                            ablock3 = VSET_V_F(PREC, LMUL, 6)(ablock3, 0, arow0_r);
                            ablock3 = VSET_V_F(PREC, LMUL, 6)(ablock3, 1, arow0_i);
                            ablock3 = VSET_V_F(PREC, LMUL, 6)(ablock3, 2, arow1_r);
                            ablock3 = VSET_V_F(PREC, LMUL, 6)(ablock3, 3, arow1_i);
                            ablock3 = VSET_V_F(PREC, LMUL, 6)(ablock3, 4, arow2_r);
                            ablock3 = VSET_V_F(PREC, LMUL, 6)(ablock3, 5, arow2_i);
                            VSSSEG6_V_F(PREC, LMUL, 6)((BASE_DT*) p_tmp, 2 * FLT_SIZE * ldp, ablock3, vl);
                            break;
                        case 2:
                            ablock2 = VSET_V_F(PREC, LMUL, 4)(ablock2, 0, arow0_r);
                            ablock2 = VSET_V_F(PREC, LMUL, 4)(ablock2, 1, arow0_i);
                            ablock2 = VSET_V_F(PREC, LMUL, 4)(ablock2, 2, arow1_r);
                            ablock2 = VSET_V_F(PREC, LMUL, 4)(ablock2, 3, arow1_i);
                            VSSSEG4_V_F(PREC, LMUL, 4)((BASE_DT*) p_tmp, 2 * FLT_SIZE * ldp, ablock2, vl);
                            break;
                        case 1:
                            ablock1 = VSET_V_F(PREC, LMUL, 2)(ablock1, 0, arow0_r);
                            ablock1 = VSET_V_F(PREC, LMUL, 2)(ablock1, 1, arow0_i);
                            VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) p_tmp, 2 * FLT_SIZE * ldp, ablock1, vl);
                            break;
                        }
                    } else {
                        RVV_TYPE_F(PREC, LMUL) kappa_arow0_r, kappa_arow1_r, kappa_arow2_r;
                        RVV_TYPE_F(PREC, LMUL) kappa_arow0_i, kappa_arow1_i, kappa_arow2_i;
                        if (bli_is_conj(conja)) {
                            switch (cdim_tmp) {
                            case 3:
                                VCMUL_VF_CONJ(PREC, LMUL, kappa_arow2_r, kappa_arow2_i, arow2_r, arow2_i, kappa->real, kappa->imag, vl);
                            case 2:
                                VCMUL_VF_CONJ(PREC, LMUL, kappa_arow1_r, kappa_arow1_i, arow1_r, arow1_i, kappa->real, kappa->imag, vl);
                            case 1:
                                VCMUL_VF_CONJ(PREC, LMUL, kappa_arow0_r, kappa_arow0_i, arow0_r, arow0_i, kappa->real, kappa->imag, vl);
                            }
                        } else {
                            switch (cdim_tmp) {
                            case 3:
                                VCMUL_VF(PREC, LMUL, kappa_arow2_r, kappa_arow2_i, arow2_r, arow2_i, kappa->real, kappa->imag, vl);
                            case 2:
                                VCMUL_VF(PREC, LMUL, kappa_arow1_r, kappa_arow1_i, arow1_r, arow1_i, kappa->real, kappa->imag, vl);
                            case 1:
                                VCMUL_VF(PREC, LMUL, kappa_arow0_r, kappa_arow0_i, arow0_r, arow0_i, kappa->real, kappa->imag, vl);
                            }
                        }

                        RVV_TYPE_FX(PREC, LMUL, 6) ablock3 = VUNDEFINED_FX(PREC, LMUL, 6)();
                        RVV_TYPE_FX(PREC, LMUL, 4) ablock2 = VUNDEFINED_FX(PREC, LMUL, 4)();
                        RVV_TYPE_FX(PREC, LMUL, 2) ablock1 = VUNDEFINED_FX(PREC, LMUL, 2)();
                        switch (cdim_tmp) {
                        case 3:
                            ablock3 = VSET_V_F(PREC, LMUL, 6)(ablock3, 0, kappa_arow0_r);
                            ablock3 = VSET_V_F(PREC, LMUL, 6)(ablock3, 1, kappa_arow0_i);
                            ablock3 = VSET_V_F(PREC, LMUL, 6)(ablock3, 2, kappa_arow1_r);
                            ablock3 = VSET_V_F(PREC, LMUL, 6)(ablock3, 3, kappa_arow1_i);
                            ablock3 = VSET_V_F(PREC, LMUL, 6)(ablock3, 4, kappa_arow2_r);
                            ablock3 = VSET_V_F(PREC, LMUL, 6)(ablock3, 5, kappa_arow2_i);
                            VSSSEG6_V_F(PREC, LMUL, 6)((BASE_DT*) p_tmp, 2 * FLT_SIZE * ldp, ablock3, vl);
                            break;
                        case 2:
                            ablock2 = VSET_V_F(PREC, LMUL, 4)(ablock2, 0, kappa_arow0_r);
                            ablock2 = VSET_V_F(PREC, LMUL, 4)(ablock2, 1, kappa_arow0_i);
                            ablock2 = VSET_V_F(PREC, LMUL, 4)(ablock2, 2, kappa_arow1_r);
                            ablock2 = VSET_V_F(PREC, LMUL, 4)(ablock2, 3, kappa_arow1_i);
                            VSSSEG4_V_F(PREC, LMUL, 4)((BASE_DT*) p_tmp, 2 * FLT_SIZE * ldp, ablock2, vl);
                            break;
                        case 1:
                            ablock1 = VSET_V_F(PREC, LMUL, 2)(ablock1, 0, kappa_arow0_r);
                            ablock1 = VSET_V_F(PREC, LMUL, 2)(ablock1, 1, kappa_arow0_i);
                            VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) p_tmp, 2 * FLT_SIZE * ldp, ablock1, vl);
                            break;
                        }
                    }
                    
                    p_tmp += cdim_tmp;
                }

                // pad lower edge
                for (size_t i = 0; i < vl; ++i) {
                    VSSEG2_V_F(PREC, LMUL_NR, 2)((BASE_DT*) p_tmp, zero_padding, cdim_max - cdim);
                    p_tmp += ldp;
                }

                a += vl;
                p += vl * ldp;
                avl -= vl;
            }
            
            // pad right edge
            for (size_t i = n; i < n_max; ++i) {
                VSSEG2_V_F(PREC, LMUL_NR, 2)((BASE_DT*) p, zero_padding, cdim_max);
                p += ldp;
            }
        } else {
            RVV_TYPE_FX(PREC, LMUL_NR, 2) zero_padding = VUNDEFINED_FX(PREC, LMUL_NR, 2)();
            zero_padding = VSET_V_F(PREC, LMUL_NR, 2)(zero_padding, 0, VFMV_V_F(PREC, LMUL_NR)(0., cdim_max));
            zero_padding = VSET_V_F(PREC, LMUL_NR, 2)(zero_padding, 1, VFMV_V_F(PREC, LMUL_NR)(0., cdim_max));

            for (size_t i = 0; i < n; ++i) {
                RVV_TYPE_FX(PREC, LMUL_NR, 2) acol;
                if (inca == 1)
                    acol = VLSEG2_V_F_TU(PREC, LMUL_NR, 2)(zero_padding, (BASE_DT*) a, cdim);
                else
                    acol = VLSSEG2_V_F_TU(PREC, LMUL_NR, 2)(zero_padding, (BASE_DT*) a, 2 * FLT_SIZE * inca, cdim);
                RVV_TYPE_F(PREC, LMUL_NR) acol_r = VGET_V_F(PREC, LMUL_NR, 2)(acol, 0); 
                RVV_TYPE_F(PREC, LMUL_NR) acol_i = VGET_V_F(PREC, LMUL_NR, 2)(acol, 1); 

                if (PASTEMAC(PRECISION_CHAR, eq1)(*kappa)) {
                    if (bli_is_conj(conja)) {
                        acol_i = VFNEG_VF_TU(PREC, LMUL_NR)(acol_i, acol_i, cdim);
                        acol = VSET_V_F(PREC, LMUL_NR, 2)(acol, 0, acol_r);
                        acol = VSET_V_F(PREC, LMUL_NR, 2)(acol, 1, acol_i);
                    }
                } else {
                    RVV_TYPE_F(PREC, LMUL_NR) kappa_acol_r, kappa_acol_i;
                    if (bli_is_conj(conja))
                        VCMUL_VF_CONJ_TU(PREC, LMUL_NR, kappa_acol_r, kappa_acol_i, acol_r, acol_i, kappa->real, kappa->imag, cdim);
                    else
                        VCMUL_VF_TU(PREC, LMUL_NR, kappa_acol_r, kappa_acol_i, acol_r, acol_i, kappa->real, kappa->imag, cdim);
                    acol = VSET_V_F(PREC, LMUL_NR, 2)(acol, 0, kappa_acol_r);
                    acol = VSET_V_F(PREC, LMUL_NR, 2)(acol, 1, kappa_acol_i);
                }

                VSSEG2_V_F(PREC, LMUL_NR, 2)((BASE_DT*) p, acol, cdim_max);
                 
                a += lda;
                p += ldp;
            }

            for (size_t i = n; i < n_max; ++i) {
                VSSEG2_V_F(PREC, LMUL_NR, 2)((BASE_DT*) p, zero_padding, cdim_max);
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
