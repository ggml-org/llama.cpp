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
    
    (void) cntx; // Suppress unused parameter warnings
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
        RVV_TYPE_F(PREC, LMUL) ax_vec_real, ax_vec_imag;

        for (size_t i = 0; i < b; ++i) {
            DATATYPE x_tmp_conj;
            PASTEMAC(PRECISION_CHAR, copycjs)(conjx, *x_tmp, x_tmp_conj);

            RVV_TYPE_FX(PREC, LMUL, 2) acol_vec;
            if (inca == 1)
                acol_vec = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) a_tmp, vl);
            else
                acol_vec = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) a_tmp, 2 * FLT_SIZE * inca, vl);

            RVV_TYPE_F(PREC, LMUL) acol_vec_real = VGET_V_F(PREC, LMUL, 2)(acol_vec, 0);
            RVV_TYPE_F(PREC, LMUL) acol_vec_imag = VGET_V_F(PREC, LMUL, 2)(acol_vec, 1);

            if (bli_is_conj(conja)) {
                if (i == 0)
                    VCMUL_VF_CONJ
                    (
                      PREC, LMUL,
                      ax_vec_real, ax_vec_imag,
                      acol_vec_real, acol_vec_imag,
                      x_tmp_conj.real, x_tmp_conj.imag,
                      vl
                    );
                else
                    VCMACC_VF_CONJ
                    (
                      PREC, LMUL,
                      ax_vec_real, ax_vec_imag,
                      x_tmp_conj.real, x_tmp_conj.imag,
                      acol_vec_real, acol_vec_imag,
                      vl
                    );
            }
            else {
                if (i == 0)
                    VCMUL_VF
                    (
                      PREC, LMUL,
                      ax_vec_real, ax_vec_imag,
                      acol_vec_real, acol_vec_imag,
                      x_tmp_conj.real, x_tmp_conj.imag,
                      vl
                    );
                else
                    VCMACC_VF
                    (
                      PREC, LMUL,
                      ax_vec_real, ax_vec_imag,
                      x_tmp_conj.real, x_tmp_conj.imag,
                      acol_vec_real, acol_vec_imag,
                      vl
                    );
            }

            a_tmp += lda;
            x_tmp += incx;
        }
        
        RVV_TYPE_FX(PREC, LMUL, 2) yvec;
	if (incy == 1)
	    yvec = VLSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) y, vl);
	else
	    yvec = VLSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) y, 2 * FLT_SIZE * incy, vl);

        RVV_TYPE_F(PREC, LMUL) yvec_real = VGET_V_F(PREC, LMUL, 2)(yvec, 0);
        RVV_TYPE_F(PREC, LMUL) yvec_imag = VGET_V_F(PREC, LMUL, 2)(yvec, 1);

        VCMACC_VF
        (
          PREC, LMUL,
          yvec_real, yvec_imag,
          alpha->real, alpha->imag,
          ax_vec_real, ax_vec_imag,
          vl
        );

        yvec = VSET_V_F(PREC, LMUL, 2)(yvec, 0, yvec_real);
        yvec = VSET_V_F(PREC, LMUL, 2)(yvec, 1, yvec_imag);

	if (incy == 1)
	    VSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) y, yvec, vl);
	else
	    VSSSEG2_V_F(PREC, LMUL, 2)((BASE_DT*) y, 2 * FLT_SIZE * incy, yvec, vl);

        a += vl * inca;
        y += vl * incy;
        avl -= vl;
    }
    return;
}

#endif // AXPYF
