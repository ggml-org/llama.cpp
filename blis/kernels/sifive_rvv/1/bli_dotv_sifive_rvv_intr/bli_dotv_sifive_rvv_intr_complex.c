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
#ifdef DOTV

DOTV(PRECISION_CHAR, void)
{
    // Computes rho = conjxt(x)^T * conjy(y)
    (void) cntx;
    DATATYPE* restrict rho = rho_;
    const DATATYPE* restrict x = x_;
    const DATATYPE* restrict y = y_;
    
    if (n <= 0) {
        rho->real = 0;
        rho->imag = 0;
        return;
    }

    // Instead of conjugating x, switch conjugation on y
    //  and conjugate rho at the end
    conj_t conjrho = conjxt;
    if (conjxt == BLIS_CONJUGATE)
        bli_toggle_conj(&conjy); // Switch conjugation of y

    RVV_TYPE_F(PREC, LMUL) acc_real, acc_imag;
    size_t avl = n;
    bool first = true;
    while (avl) {
        size_t vl = VSETVL(PREC, LMUL)(avl);
        RVV_TYPE_FX(PREC, LMUL, 2) xvec, yvec;
        RVV_TYPE_F(PREC, LMUL) xvec_real, xvec_imag, yvec_real, yvec_imag;

        if (incx == 1)
            xvec = VLSEG2_V_F(PREC, LMUL, 2)( (BASE_DT*) x, vl);
        else
            xvec = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) x, 2*FLT_SIZE*incx, vl);
        
        if (incy == 1)
            yvec = VLSEG2_V_F(PREC, LMUL, 2)( (BASE_DT*) y, vl);
        else
            yvec = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) y, 2*FLT_SIZE*incy, vl);

        xvec_real = VGET_V_F(PREC, LMUL, 2)(xvec, 0);
        xvec_imag = VGET_V_F(PREC, LMUL, 2)(xvec, 1);
        yvec_real = VGET_V_F(PREC, LMUL, 2)(yvec, 0);
        yvec_imag = VGET_V_F(PREC, LMUL, 2)(yvec, 1);

        if (first) {
            acc_real = VFMUL_VV(PREC, LMUL)(xvec_real, yvec_real, vl);
            acc_imag = VFMUL_VV(PREC, LMUL)(xvec_imag, yvec_real, vl);
            first = false;
        } else {
            acc_real = VFMACC_VV_TU(PREC, LMUL)(acc_real, xvec_real, yvec_real, vl);
            acc_imag = VFMACC_VV_TU(PREC, LMUL)(acc_imag, xvec_imag, yvec_real, vl);
        }
        if (conjy == BLIS_NO_CONJUGATE) {
            acc_real = VFNMSAC_VV_TU(PREC, LMUL)(acc_real, xvec_imag, yvec_imag, vl);
            acc_imag = VFMACC_VV_TU(PREC, LMUL)( acc_imag, xvec_real, yvec_imag, vl);
        } else {
            acc_real = VFMACC_VV_TU(PREC, LMUL)( acc_real, xvec_imag, yvec_imag, vl);
            acc_imag = VFNMSAC_VV_TU(PREC, LMUL)(acc_imag, xvec_real, yvec_imag, vl);
        }

        x += vl*incx;
        y += vl*incy;
        avl -= vl;
    }


    RVV_TYPE_F(PREC, m1) sum_real = VFMV_S_F(PREC, m1)(0.f, 1);
    RVV_TYPE_F(PREC, m1) sum_imag = VFMV_S_F(PREC, m1)(0.f, 1);
    sum_real = VF_REDUSUM_VS(PREC, LMUL)(acc_real, sum_real, n);
    sum_imag = VF_REDUSUM_VS(PREC, LMUL)(acc_imag, sum_imag, n);

    if (conjrho == BLIS_CONJUGATE) {
        sum_imag = VFNEG_VF(PREC, m1)(sum_imag, 1);
    }
    rho->real = VFMV_F_S(PREC)(sum_real);
    rho->imag = VFMV_F_S(PREC)(sum_imag);

}

#endif // DOTV
