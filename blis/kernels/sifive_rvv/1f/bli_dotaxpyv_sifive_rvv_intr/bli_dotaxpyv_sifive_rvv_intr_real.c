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
    //            == z + alphax *    x      (real case)
    //   and  rho := conjxt(x)^T * conjy(y)
    //            == x^T * y                (real case)
    
    (void) conjx; // Suppress unused parameter warnings
    (void) conjxt;
    (void) conjy;
    const DATATYPE* restrict alpha = alpha_;
    const DATATYPE* restrict x = x_;
    const DATATYPE* restrict y = y_;
    DATATYPE* restrict rho = rho_;
    DATATYPE* restrict z = z_;

    if (n <= 0)
        return;

    size_t avl = n;
    bool first = true;
    RVV_TYPE_F(PREC, LMUL) acc;

    while (avl) {
        size_t vl = VSETVL(PREC, LMUL)(avl);
        RVV_TYPE_F(PREC, LMUL) xvec, yvec, zvec;

        // Loads
        if (incx == 1)
            xvec = VLE_V_F(PREC, LMUL)(x, vl);
        else
            xvec = VLSE_V_F(PREC, LMUL)(x, FLT_SIZE * incx, vl);
        
        if (incy == 1)
            yvec = VLE_V_F(PREC, LMUL)(y, vl);
        else
            yvec = VLSE_V_F(PREC, LMUL)(y, FLT_SIZE * incy, vl);
        
        if (incz == 1)
            zvec = VLE_V_F(PREC, LMUL)(z, vl);
        else
            zvec = VLSE_V_F(PREC, LMUL)(z, FLT_SIZE * incz, vl);

        // z := z + alphax * x
        zvec = VFMACC_VF(PREC, LMUL)(zvec, *alpha, xvec, vl);

        // rho := x^T * y
        if (first){
            acc = VFMUL_VV(PREC, LMUL)( xvec, yvec, vl);
            first = false;
        } else {
            acc  = VFMACC_VV_TU(PREC, LMUL)( acc,   xvec, yvec, vl);
        }

        // Store
        if (incz == 1)
            VSE_V_F(PREC, LMUL)(z, zvec, vl);
        else
            VSSE_V_F(PREC, LMUL)(z, FLT_SIZE * incz, zvec, vl);
        
        x += vl*incx;
        y += vl*incy;
        z += vl*incz;
        avl -= vl;
    }

    // Compute rho
    RVV_TYPE_F(PREC, m1) sum = VFMV_S_F(PREC, m1)( 0.f, 1);
    sum = VF_REDUSUM_VS(PREC, LMUL)(acc, sum, n);
    *rho = VFMV_F_S(PREC)(sum);

}

#endif // ifdef DOTAXPYV
