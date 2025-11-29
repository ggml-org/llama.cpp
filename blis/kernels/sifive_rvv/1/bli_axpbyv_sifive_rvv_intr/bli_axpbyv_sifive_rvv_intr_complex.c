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
#ifdef AXPBYV

AXPBYV(PRECISION_CHAR, void)
{
    // Computes y := beta * y + alpha * conjx(x)
    
    if (n <= 0) return;

    const DATATYPE* restrict alpha = alpha_;
    const DATATYPE* restrict beta = beta_;
    const DATATYPE* restrict x = x_;
    DATATYPE* restrict y = y_;

    if (alpha->real == 0 && alpha->imag == 0 && beta->real == 0 && beta->imag == 0){
        SETV(PRECISION_CHAR)(BLIS_NO_CONJUGATE, n, alpha, y, incy, cntx);
        return;
    }
    if (alpha->real == 0 && alpha->imag == 0){
        SCALV(PRECISION_CHAR)(BLIS_NO_CONJUGATE, n, beta, y, incy, cntx);
        return;
    }
    if (beta->real == 0 && beta->imag == 0){
        SCAL2V(PRECISION_CHAR)(conjx, n, alpha, x, incx, y, incy, cntx);
        return;
    }

    // Note: in the cases alpha = 0 && beta = 1, or alpha = 1 && beta = 0, we 
    // will canonicalize NaNs whereas the reference code will propagate NaN payloads.

    // TO DO (optimization): special cases for alpha = +-1, +-i, beta = +-1, +-i

    // alpha and beta are both nonzero
    size_t avl = n;
    while (avl) {
        size_t vl = VSETVL(PREC, LMUL)(avl);
        RVV_TYPE_FX(PREC, LMUL, 2) xvec, yvec;
        RVV_TYPE_F(PREC, LMUL) xvec_real, xvec_imag, yvec_real, yvec_imag, temp_real, temp_imag;

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

        // Computed as:
        // y.real = beta.real * y.real - beta.imag * y.imag + alpha.real * x.real - alpha.imag * conj(x.imag)
        // y.imag = beta.real * y.imag + beta.imag * y.real + alpha.imag * x.real + alpha.real * conj(x.imag)
        temp_real = VFMUL_VF(PREC, LMUL)  (yvec_real, beta->real, vl);
        temp_imag = VFMUL_VF(PREC, LMUL)  (yvec_imag, beta->real, vl);
        temp_real = VFNMSAC_VF(PREC, LMUL)(temp_real, beta->imag, yvec_imag, vl);
        temp_imag = VFMACC_VF(PREC, LMUL) (temp_imag, beta->imag, yvec_real, vl);
        yvec_real = VFMACC_VF(PREC, LMUL) (temp_real, alpha->real, xvec_real, vl);
        yvec_imag = VFMACC_VF(PREC, LMUL) (temp_imag, alpha->imag, xvec_real, vl);
        if (conjx == BLIS_NO_CONJUGATE) {
            yvec_real = VFNMSAC_VF(PREC, LMUL)(yvec_real, alpha->imag, xvec_imag, vl);
            yvec_imag = VFMACC_VF(PREC, LMUL) (yvec_imag, alpha->real, xvec_imag, vl);
        } else {
            yvec_real = VFMACC_VF(PREC, LMUL) (yvec_real, alpha->imag, xvec_imag, vl);
            yvec_imag = VFNMSAC_VF(PREC, LMUL)(yvec_imag, alpha->real, xvec_imag, vl);
        }

        yvec = VSET_V_F(PREC, LMUL, 2)(yvec, 0, yvec_real);
        yvec = VSET_V_F(PREC, LMUL, 2)(yvec, 1, yvec_imag);

        if (incy == 1)
            VSSEG2_V_F(PREC, LMUL, 2)( (BASE_DT*) y, yvec, vl);
        else
            VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) y, 2*FLT_SIZE*incy, yvec, vl);

        x += vl*incx;
        y += vl*incy;
        avl -= vl;
    }

}

#endif // AXPBYV
