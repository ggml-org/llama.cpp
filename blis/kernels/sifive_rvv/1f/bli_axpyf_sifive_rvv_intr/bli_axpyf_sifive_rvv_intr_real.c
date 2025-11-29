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
#ifdef AXPYF

AXPYF(PRECISION_CHAR, void)
{
    // Computes y := y + alpha * conja(A) * conjx(x)
    
    (void) conja; // Suppress unused parameter warnings
    (void) conjx;
    (void) cntx;
    const DATATYPE* restrict alpha = alpha_;
    const DATATYPE* restrict a = a_;
    const DATATYPE* restrict x = x_;
    DATATYPE* restrict y = y_;

    if (m <= 0 || b <= 0 || PASTEMAC(PRECISION_CHAR, eq0)(*alpha))
        return;

    size_t avl = m;
    while (avl) {
        size_t vl = VSETVL(PREC, LMUL)(avl);
        const DATATYPE* restrict a_tmp = a;
        const DATATYPE* restrict x_tmp = x;
        RVV_TYPE_F(PREC, LMUL) ax_vec;

        for (size_t i = 0; i < b; ++i) {
            RVV_TYPE_F(PREC, LMUL) acol_vec;
            if (inca == 1)
                acol_vec = VLE_V_F(PREC, LMUL)(a_tmp, vl);
            else
                acol_vec = VLSE_V_F(PREC, LMUL)(a_tmp, FLT_SIZE * inca, vl);

            if (i == 0)
                ax_vec = VFMUL_VF(PREC, LMUL)(acol_vec, *x_tmp, vl);
            else
                ax_vec = VFMACC_VF(PREC, LMUL)(ax_vec, *x_tmp, acol_vec, vl);

            a_tmp += lda;
            x_tmp += incx;
        }
        
        RVV_TYPE_F(PREC, LMUL) yvec;
        if (incy == 1)
            yvec = VLE_V_F(PREC, LMUL)(y, vl);
        else
            yvec = VLSE_V_F(PREC, LMUL)(y, FLT_SIZE * incy, vl);

        yvec = VFMACC_VF(PREC, LMUL)(yvec, *alpha, ax_vec, vl);

        if (incy == 1)
            VSE_V_F(PREC, LMUL)(y, yvec, vl);
        else
            VSSE_V_F(PREC, LMUL)(y, FLT_SIZE * incy, yvec, vl);

        a += vl * inca;
        y += vl * incy;
        avl -= vl;
    }
    return;
}

#endif // AXPYF
