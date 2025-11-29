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
#ifdef SETV

SETV(PRECISION_CHAR, void)
{
    (void)cntx;
    const DATATYPE* restrict alpha = alpha_;
    DATATYPE* restrict x = x_;

    if (n <= 0) return;

    DATATYPE alpha_conj;
    PASTEMAC(PRECISION_CHAR, copycjs)(conjalpha, *alpha, alpha_conj);

    RVV_TYPE_F(PREC, LMUL) alpha_conj_real_vec = VFMV_V_F(PREC, LMUL)(alpha_conj.real, n); 
    RVV_TYPE_F(PREC, LMUL) alpha_conj_imag_vec = VFMV_V_F(PREC, LMUL)(alpha_conj.imag, n); 

    RVV_TYPE_FX(PREC, LMUL, 2) alpha_conj_vec = VUNDEFINED_FX(PREC, LMUL, 2)();
    alpha_conj_vec = VSET_V_F(PREC, LMUL, 2)(alpha_conj_vec, 0, alpha_conj_real_vec);
    alpha_conj_vec = VSET_V_F(PREC, LMUL, 2)(alpha_conj_vec, 1, alpha_conj_imag_vec);

    size_t avl = n;
    while (avl) {
        size_t vl = VSETVL(PREC, LMUL)(avl);

        if (incx == 1)
            VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) x, alpha_conj_vec, vl);
        else
            VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) x, 2 * FLT_SIZE * incx, alpha_conj_vec, vl);

        x += vl * incx;
        avl -= vl;
    }
    return;
}

#endif // SETV
