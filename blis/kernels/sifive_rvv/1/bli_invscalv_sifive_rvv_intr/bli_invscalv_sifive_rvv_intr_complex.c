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
#ifdef INVSCALV

INVSCALV(PRECISION_CHAR, void)
{
    (void)cntx;
    const DATATYPE* restrict alpha = alpha_;
    DATATYPE* restrict x = x_;

    if (n <= 0) return;
    if (PASTEMAC(PRECISION_CHAR, eq1)(*alpha)) return;
    if (PASTEMAC(PRECISION_CHAR, eq0)(*alpha)) return;

    DATATYPE alpha_conj_inv;
    PASTEMAC(PRECISION_CHAR, copycjs)(conjalpha, *alpha, alpha_conj_inv); 
    PASTEMAC(PRECISION_CHAR, inverts)(alpha_conj_inv);

    size_t avl = n;
    while (avl) {
        size_t vl = VSETVL(PREC, LMUL)(avl);
        RVV_TYPE_FX(PREC, LMUL, 2) xvec;

        if (incx == 1)
            xvec = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) x, vl);
        else
            xvec = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) x, 2 * FLT_SIZE * incx, vl);

        RVV_TYPE_F(PREC, LMUL) xvec_real = VGET_V_F(PREC, LMUL, 2)(xvec, 0);
        RVV_TYPE_F(PREC, LMUL) xvec_imag = VGET_V_F(PREC, LMUL, 2)(xvec, 1);
        RVV_TYPE_F(PREC, LMUL) yvec_real, yvec_imag;

        VCMUL_VF(PREC, LMUL, yvec_real, yvec_imag, xvec_real, xvec_imag, alpha_conj_inv.real, alpha_conj_inv.imag, vl);

        RVV_TYPE_FX(PREC, LMUL, 2) yvec = VUNDEFINED_FX(PREC, LMUL, 2)();
        yvec = VSET_V_F(PREC, LMUL, 2)(yvec, 0, yvec_real);
        yvec = VSET_V_F(PREC, LMUL, 2)(yvec, 1, yvec_imag);

        if (incx == 1)
            VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) x, yvec, vl);
        else
            VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) x, 2 * FLT_SIZE * incx, yvec, vl);

        x += vl * incx;
        avl -= vl;
    }
    return;
}

#endif // INVSCALV
