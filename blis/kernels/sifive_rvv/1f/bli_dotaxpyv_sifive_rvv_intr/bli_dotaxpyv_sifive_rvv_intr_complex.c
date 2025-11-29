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
#ifdef DOTAXPYV

DOTAXPYV(PRECISION_CHAR, void)
{
    // Computes z := z + alpha * conjx(x)
    //   and  rho := conjxt(x)^T * conjy(y)
    const DATATYPE* restrict alpha = alpha_;
    const DATATYPE* restrict x = x_;
    const DATATYPE* restrict y = y_;
    DATATYPE* restrict rho = rho_;
    DATATYPE* restrict z = z_;

    if (n <= 0)
        return;

    size_t avl = n;
    bool first = true;
    RVV_TYPE_F(PREC, LMUL) acc_real, acc_imag;

    while (avl) {
        size_t vl = VSETVL(PREC, LMUL)(avl);
        RVV_TYPE_FX(PREC, LMUL, 2) xvec, yvec, zvec;
        RVV_TYPE_F(PREC, LMUL) xvec_real, xvec_imag, yvec_real, yvec_imag, zvec_real, zvec_imag;

        // Loads
        if (incx == 1)
            xvec = VLSEG2_V_F(PREC, LMUL, 2)( (BASE_DT*) x, vl);
        else
            xvec = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) x, 2*FLT_SIZE*incx, vl);
        
        if (incy == 1)
            yvec = VLSEG2_V_F(PREC, LMUL, 2)( (BASE_DT*) y, vl);
        else
            yvec = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) y, 2*FLT_SIZE*incy, vl);
        
        if (incz == 1)
            zvec = VLSEG2_V_F(PREC, LMUL, 2)( (BASE_DT*) z, vl);
        else
            zvec = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) z, 2*FLT_SIZE*incz, vl);

        xvec_real = VGET_V_F(PREC, LMUL, 2)(xvec, 0);
        xvec_imag = VGET_V_F(PREC, LMUL, 2)(xvec, 1);
        yvec_real = VGET_V_F(PREC, LMUL, 2)(yvec, 0);
        yvec_imag = VGET_V_F(PREC, LMUL, 2)(yvec, 1);
        zvec_real = VGET_V_F(PREC, LMUL, 2)(zvec, 0);
        zvec_imag = VGET_V_F(PREC, LMUL, 2)(zvec, 1);

        // z := z + alpha * conjx(x)
        zvec_real = VFMACC_VF(PREC, LMUL)( zvec_real, alpha->real, xvec_real, vl);
        zvec_imag = VFMACC_VF(PREC, LMUL)( zvec_imag, alpha->imag, xvec_real, vl);
        if (conjx == BLIS_NO_CONJUGATE){
            zvec_real = VFNMSAC_VF(PREC, LMUL)(zvec_real, alpha->imag, xvec_imag, vl);
            zvec_imag = VFMACC_VF(PREC, LMUL)( zvec_imag, alpha->real, xvec_imag, vl);
        } else {
            zvec_real = VFMACC_VF(PREC, LMUL)( zvec_real, alpha->imag, xvec_imag, vl);
            zvec_imag = VFNMSAC_VF(PREC, LMUL)(zvec_imag, alpha->real, xvec_imag, vl);
        }
        
        // rho := conjxt(x)^T * conjy(y)
        // We accumulate the current term of the dot product as (a*c-b*d) + (a*d+b*c)*i,
        // conjugating when necessary
        if (first) {
            // Initialize real part:      a*c
            acc_real = VFMUL_VV(PREC, LMUL)( xvec_real, yvec_real, vl);
            // Initialize imaginary part: a*d
            acc_imag = VFMUL_VV(PREC, LMUL)( xvec_real, yvec_imag, vl);
            if (conjy == BLIS_CONJUGATE)
                acc_imag = VFNEG_VF(PREC, LMUL)(acc_imag, vl); // TO DO: eliminate this negation
            first = false;
        } else {
            // Accumulate real part:      a*c
            acc_real = VFMACC_VV_TU(PREC, LMUL)( acc_real, xvec_real, yvec_real, vl);
            // Accumulate imaginary part: a*d
            if (conjy == BLIS_NO_CONJUGATE)
                acc_imag = VFMACC_VV_TU(PREC, LMUL)(acc_imag, xvec_real, yvec_imag, vl);
            else
                acc_imag = VFNMSAC_VV_TU(PREC, LMUL)(acc_imag, xvec_real, yvec_imag, vl);
        }
        // Finish real part:      b*d
        if (conjxt == BLIS_NO_CONJUGATE ^ conjy == BLIS_NO_CONJUGATE)
            // Exactly one is conjugated => add
            acc_real = VFMACC_VV_TU(PREC, LMUL)(acc_real, xvec_imag, yvec_imag, vl);
        else
            acc_real = VFNMSAC_VV_TU(PREC,LMUL)(acc_real, xvec_imag, yvec_imag, vl);
        // Finish imaginary part: b*c
        if (conjxt == BLIS_NO_CONJUGATE)
            acc_imag = VFMACC_VV_TU(PREC, LMUL)( acc_imag, xvec_imag, yvec_real, vl);
        else
            acc_imag = VFNMSAC_VV_TU(PREC, LMUL)( acc_imag, xvec_imag, yvec_real, vl);

        // Stores
        zvec = VSET_V_F(PREC, LMUL, 2)(zvec, 0, zvec_real);
        zvec = VSET_V_F(PREC, LMUL, 2)(zvec, 1, zvec_imag);

        if (incz == 1)
            VSSEG2_V_F(PREC, LMUL, 2)( (BASE_DT*) z, zvec, vl);
        else
            VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) z, 2*FLT_SIZE*incz, zvec, vl);

        x += vl*incx;
        y += vl*incy;
        z += vl*incz;
        avl -= vl;
    }

    // Compute rho
    RVV_TYPE_F(PREC, m1) sum_real = VFMV_S_F(PREC, m1)( 0.f, 1);
    RVV_TYPE_F(PREC, m1) sum_imag = VFMV_S_F(PREC, m1)( 0.f, 1);
    sum_real = VF_REDUSUM_VS(PREC, LMUL)(acc_real, sum_real, n);
    sum_imag = VF_REDUSUM_VS(PREC, LMUL)(acc_imag, sum_imag, n);
    rho->real = VFMV_F_S(PREC)(sum_real);
    rho->imag = VFMV_F_S(PREC)(sum_imag);

}

#endif // ifdef DOTAXPYV
