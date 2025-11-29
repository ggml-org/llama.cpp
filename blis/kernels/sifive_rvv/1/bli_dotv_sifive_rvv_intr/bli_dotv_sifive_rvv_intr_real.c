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
    //             ==     x^T     *    y       (real case)
    (void) cntx;
    (void) conjxt; // Suppress unused parameter warnings
    (void) conjy;
    DATATYPE* restrict rho = rho_;
    const DATATYPE* restrict x = x_;
    const DATATYPE* restrict y = y_;

    if (n <= 0) {
        *rho = 0;
        return;
    }

    RVV_TYPE_F(PREC, LMUL) acc;
    size_t avl = n;
    bool first = true;
    while (avl) {
        size_t vl = VSETVL(PREC, LMUL)(avl);
        RVV_TYPE_F(PREC, LMUL) xvec, yvec;

        if (incx == 1)
            xvec = VLE_V_F(PREC, LMUL) (x, vl);
        else
            xvec = VLSE_V_F(PREC, LMUL)(x, FLT_SIZE * incx, vl);

        if (incy == 1)
            yvec = VLE_V_F(PREC, LMUL) (y, vl);
        else
            yvec = VLSE_V_F(PREC, LMUL)(y, FLT_SIZE * incy, vl);
        
        if (first) {
            acc = VFMUL_VV(PREC, LMUL)(xvec, yvec, vl);
            first = false;
        } else
            acc = VFMACC_VV_TU(PREC, LMUL)(acc, xvec, yvec, vl);

        x += vl * incx;
        y += vl * incy;
        avl -= vl;
    }

    RVV_TYPE_F(PREC, m1) sum = VFMV_S_F(PREC, m1)(0.f, 1);
    sum = VF_REDUSUM_VS(PREC, LMUL)(acc, sum, n);
    *rho = VFMV_F_S(PREC)(sum);
}

#endif // DOTV
