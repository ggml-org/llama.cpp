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
#ifdef AMAXV

AMAXV(PRECISION_CHAR, void)
{
    (void)cntx;
    const DATATYPE* restrict x = x_;

    if (n <= 1) {
        *index = 0;
        return;
    }

    RVV_TYPE_F(PREC_X, LMUL_X) xacc;
    // Indices will be unsigned and of the same width as dim_t.
    RVV_TYPE_U(PREC_I, LMUL_I) iacc;
    RVV_TYPE_U(PREC_I, LMUL_I) vid_vec = VID_V(PREC_I, LMUL_I)(n);
    bool first = true;
    guint_t offset = 0;
    size_t avl = n;
    while (avl) {
        size_t vl = VSETVL(PREC_X, LMUL_X)(avl);
        RVV_TYPE_FX(PREC_X, LMUL_X, 2) xvec;

        if (incx == 1)
            xvec = VLSEG2_V_F(PREC_X, LMUL_X, 2)((BASE_DT*) x, vl);
        else
            xvec = VLSSEG2_V_F(PREC_X, LMUL_X, 2)((BASE_DT*) x, 2 * FLT_SIZE * incx, vl);

        RVV_TYPE_F(PREC_X, LMUL_X) xvec_real = VGET_V_F(PREC_X, LMUL_X, 2)(xvec, 0);
        RVV_TYPE_F(PREC_X, LMUL_X) xvec_imag = VGET_V_F(PREC_X, LMUL_X, 2)(xvec, 1);
        RVV_TYPE_F(PREC_X, LMUL_X) xvec_real_abs = VFABS_V(PREC_X, LMUL_X)(xvec_real, vl);
        RVV_TYPE_F(PREC_X, LMUL_X) xvec_imag_abs = VFABS_V(PREC_X, LMUL_X)(xvec_imag, vl);
        RVV_TYPE_F(PREC_X, LMUL_X) xvec_abs = VFADD_VV(PREC_X, LMUL_X)(xvec_real_abs, xvec_imag_abs, vl);

        RVV_TYPE_B(RATIO) is_nan = VMFNE_VV(PREC_X, LMUL_X, RATIO)(xvec_abs, xvec_abs, vl);
        int nan_index = VFIRST_M(RATIO)(is_nan, vl);
        if (nan_index != -1) {
            *index = (guint_t) nan_index + offset;
            return;
        }

        if (first) {
            xacc = xvec_abs; 
            iacc = vid_vec;
            first = false;
        }
        else {
            RVV_TYPE_B(RATIO) mask = VMFGT_VV(PREC_X, LMUL_X, RATIO)(xvec_abs, xacc, vl);
            xacc = VFMAX_VV_TU(PREC_X, LMUL_X)(xacc, xvec_abs, xacc, vl);
            RVV_TYPE_U(PREC_I, LMUL_I) ivec = VADD_VX_U(PREC_I, LMUL_I)(vid_vec, offset, vl);
            iacc = VMERGE_VVM_TU_U(PREC_I, LMUL_I)(iacc, iacc, ivec, mask, vl);
        }

        x += vl * incx;
        offset += vl;
        avl -= vl;
    }

    RVV_TYPE_F(PREC_X, m1) xmax = VFMV_S_F(PREC_X, m1)(0., 1);
    xmax = VFREDMAX_VS(PREC_X, LMUL_X)(xacc, xmax, n);
    RVV_TYPE_F(PREC_X, LMUL_X) xmax_splat = VLMUL_EXT_V_F_M1(PREC_X, LMUL_X)(xmax);
    xmax_splat = VRGATHER_VX_F(PREC_X, LMUL_X)(xmax_splat, 0, n);
    RVV_TYPE_B(RATIO) mask = VMFEQ_VV(PREC_X, LMUL_X, RATIO)(xacc, xmax_splat, n);
    RVV_TYPE_U(PREC_I, m1) imax = VMV_S_X_U(PREC_I, m1)(-1, 1);
    imax = VREDMINU_VS_M(PREC_I, LMUL_I)(mask, iacc, imax, n);
    *index = VMV_X_S_U(PREC_I)(imax);
    return;
}

#endif // AMAXV
