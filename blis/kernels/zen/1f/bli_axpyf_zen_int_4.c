/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#include "immintrin.h"
#include "blis.h"

/* Union data structure to access AVX registers
   One 256-bit AVX register holds 8 SP elements. */
typedef union
{
    __m256  v;
    float   f[8] __attribute__((aligned(64)));
} v8sf_t;

/* Union data structure to access AVX registers
*  One 256-bit AVX register holds 4 DP elements. */
typedef union
{
    __m256d v;
    double  d[4] __attribute__((aligned(64)));
} v4df_t;


void bli_caxpyf_zen_int_4
     (
             conj_t  conja,
             conj_t  conjx,
             dim_t   m,
             dim_t   b_n,
       const void*   alpha0,
       const void*   a0, inc_t inca, inc_t lda,
       const void*   x0, inc_t incx,
             void*   y0, inc_t incy,
       const cntx_t* cntx
     )
{
	const scomplex* restrict alpha = (scomplex *)alpha0;
	scomplex* restrict a     = (scomplex *)a0;
	scomplex* restrict x     = (scomplex *)x0;
	scomplex* restrict y     = (scomplex *)y0;

    inc_t fuse_fac = 4;
    inc_t i;

    __m256 ymm0, ymm1, ymm2, ymm3;
    __m256 ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm10;
    __m256 ymm12, ymm13;

    float* ap[4];
    float* yp = (float*)y;

    scomplex            chi0;
    scomplex            chi1;
    scomplex            chi2;
    scomplex            chi3;


    dim_t setPlusOne = 1;

    if ( bli_is_conj(conja) )
    {
        setPlusOne = -1;
    }

    // If either dimension is zero, or if alpha is zero, return early.
    if ( bli_zero_dim2( m, b_n ) || bli_ceq0( *alpha ) ) return;

    // If b_n is not equal to the fusing factor, then perform the entire
    // operation as a loop over axpyv.
    if ( b_n != fuse_fac )
    {
        if ( cntx == NULL ) cntx = ( cntx_t* )bli_gks_query_cntx();

        axpyv_ker_ft f = bli_cntx_get_ukr_dt( BLIS_SCOMPLEX, BLIS_AXPYV_KER, cntx );

        for ( i = 0; i < b_n; ++i )
        {
            const scomplex* restrict a1   = a + (0  )*inca + (i  )*lda;
            const scomplex* restrict chi1 = x + (i  )*incx;
                  scomplex* restrict y1   = y + (0  )*incy;
                  scomplex           alpha_chi1;

            bli_ccopycjs( conjx, *chi1, alpha_chi1 );
            bli_cscals( *alpha, alpha_chi1 );

            f
            (
              conja,
              m,
              &alpha_chi1,
              a1, inca,
              y1, incy,
              cntx
            );
        }

        return;
    }


    // At this point, we know that b_n is exactly equal to the fusing factor.
    if(bli_is_noconj(conjx))
    {
        chi0 = *( x + 0*incx );
        chi1 = *( x + 1*incx );
        chi2 = *( x + 2*incx );
        chi3 = *( x + 3*incx );
    }
    else
    {
        const scomplex* restrict pchi0 = x + 0*incx ;
        const scomplex* restrict pchi1 = x + 1*incx ;
        const scomplex* restrict pchi2 = x + 2*incx ;
        const scomplex* restrict pchi3 = x + 3*incx ;

        bli_ccopycjs( conjx, *pchi0, chi0 );
        bli_ccopycjs( conjx, *pchi1, chi1 );
        bli_ccopycjs( conjx, *pchi2, chi2 );
        bli_ccopycjs( conjx, *pchi3, chi3 );
    }

    // Scale each chi scalar by alpha.
    bli_cscals( *alpha, chi0 );
    bli_cscals( *alpha, chi1 );
    bli_cscals( *alpha, chi2 );
    bli_cscals( *alpha, chi3 );

    lda *= 2;
    incx *= 2;
    incy *= 2;
    inca *= 2;

    ap[0] = (float*)a;
    ap[1] = (float*)a + lda;
    ap[2] = ap[1] + lda;
    ap[3] = ap[2] + lda;

    if( inca == 2 && incy == 2 )
    {
        inc_t n1 = m/4;
        inc_t n2 = m%4;

        ymm12 = _mm256_setzero_ps();
        ymm13 = _mm256_setzero_ps();

            // broadcast real & imag parts of 4 elements of x
        ymm0 = _mm256_broadcast_ss(&chi0.real); // real part of x0
        ymm1 = _mm256_broadcast_ss(&chi0.imag); // imag part of x0
        ymm2 = _mm256_broadcast_ss(&chi1.real); // real part of x1
        ymm3 = _mm256_broadcast_ss(&chi1.imag); // imag part of x1
        ymm4 = _mm256_broadcast_ss(&chi2.real); // real part of x2
        ymm5 = _mm256_broadcast_ss(&chi2.imag); // imag part of x2
        ymm6 = _mm256_broadcast_ss(&chi3.real); // real part of x3
        ymm7 = _mm256_broadcast_ss(&chi3.imag); // imag part of x3

        for(i = 0; i < n1; i++)
        {
            //load first two columns of A
     	    ymm8  = _mm256_loadu_ps(ap[0] + 0);
            ymm10 = _mm256_loadu_ps(ap[1] + 0);

            ymm12 = _mm256_mul_ps(ymm8, ymm0);
            ymm13 = _mm256_mul_ps(ymm8, ymm1);

            ymm12 = _mm256_fmadd_ps(ymm10, ymm2, ymm12);
            ymm13 = _mm256_fmadd_ps(ymm10, ymm3, ymm13);

	    //load 3rd and 4th columns of A
            ymm8  = _mm256_loadu_ps(ap[2] + 0);
            ymm10 = _mm256_loadu_ps(ap[3] + 0);

            ymm12 = _mm256_fmadd_ps(ymm8, ymm4, ymm12);
            ymm13 = _mm256_fmadd_ps(ymm8, ymm5, ymm13);

            ymm12 = _mm256_fmadd_ps(ymm10, ymm6, ymm12);
            ymm13 = _mm256_fmadd_ps(ymm10, ymm7, ymm13);

	    //load Y vector
            ymm10 = _mm256_loadu_ps(yp + 0);

            if(bli_is_noconj(conja))
            {
                //printf("Inside no conj if\n");
                ymm13 = _mm256_permute_ps(ymm13, 0xB1);
                ymm8 = _mm256_addsub_ps(ymm12, ymm13);
            }
            else
            {
                ymm12 = _mm256_permute_ps(ymm12, 0xB1);
                ymm8 = _mm256_addsub_ps(ymm13, ymm12);
                ymm8 = _mm256_permute_ps(ymm8, 0xB1);
            }

            ymm12 = _mm256_add_ps(ymm8, ymm10);

            _mm256_storeu_ps((float*)(yp), ymm12);

            yp += 8;
            ap[0] += 8;
            ap[1] += 8;
            ap[2] += 8;
            ap[3] += 8;
        }

        // If there are leftover iterations, perform them with scalar code.

        for ( i = 0; (i + 0) < n2 ; ++i )
        {

            scomplex       y0c = *(scomplex*)yp;

            const scomplex a0c = *(scomplex*)ap[0];
            const scomplex a1c = *(scomplex*)ap[1];
            const scomplex a2c = *(scomplex*)ap[2];
            const scomplex a3c = *(scomplex*)ap[3];

            y0c.real += chi0.real * a0c.real - chi0.imag * a0c.imag * setPlusOne;
            y0c.real += chi1.real * a1c.real - chi1.imag * a1c.imag * setPlusOne;
            y0c.real += chi2.real * a2c.real - chi2.imag * a2c.imag * setPlusOne;
            y0c.real += chi3.real * a3c.real - chi3.imag * a3c.imag * setPlusOne;

            y0c.imag += chi0.imag * a0c.real + chi0.real * a0c.imag * setPlusOne;
            y0c.imag += chi1.imag * a1c.real + chi1.real * a1c.imag * setPlusOne;
            y0c.imag += chi2.imag * a2c.real + chi2.real * a2c.imag * setPlusOne;
            y0c.imag += chi3.imag * a3c.real + chi3.real * a3c.imag * setPlusOne;

            *(scomplex*)yp = y0c;

            ap[0] += 2;
            ap[1] += 2;
            ap[2] += 2;
            ap[3] += 2;
            yp += 2;
        }
    //PASTEMAC(c,fprintm)(stdout, "Y after A*x in axpyf",m, 1, (scomplex*)y, 1, 1, "%4.1f", "");

    }
    else
    {
        for (i = 0 ; (i + 0) < m ; ++i )
        {
            scomplex       y0c = *(scomplex*)yp;
            const scomplex a0c = *(scomplex*)ap[0];
            const scomplex a1c = *(scomplex*)ap[1];
            const scomplex a2c = *(scomplex*)ap[2];
            const scomplex a3c = *(scomplex*)ap[3];

            y0c.real += chi0.real * a0c.real - chi0.imag * a0c.imag * setPlusOne;
            y0c.real += chi1.real * a1c.real - chi1.imag * a1c.imag * setPlusOne;
            y0c.real += chi2.real * a2c.real - chi2.imag * a2c.imag * setPlusOne;
            y0c.real += chi3.real * a3c.real - chi3.imag * a3c.imag * setPlusOne;

            y0c.imag += chi0.imag * a0c.real + chi0.real * a0c.imag * setPlusOne;
            y0c.imag += chi1.imag * a1c.real + chi1.real * a1c.imag * setPlusOne;
            y0c.imag += chi2.imag * a2c.real + chi2.real * a2c.imag * setPlusOne;
            y0c.imag += chi3.imag * a3c.real + chi3.real * a3c.imag * setPlusOne;

            *(scomplex*)yp = y0c;

            ap[0] += inca;
            ap[1] += inca;
            ap[2] += inca;
            ap[3] += inca;
            yp += incy;
        }
    }
}


void bli_zaxpyf_zen_int_4
     (
             conj_t  conja,
             conj_t  conjx,
             dim_t   m,
             dim_t   b_n,
       const void*   alpha0,
       const void*   a00, inc_t inca, inc_t lda,
       const void*   x00, inc_t incx,
             void*   y00, inc_t incy,
       const cntx_t* cntx
     )
{
	const dcomplex* restrict alpha = (dcomplex *)alpha0;
	dcomplex* restrict a     = (dcomplex *)a00;
	dcomplex* restrict x     = (dcomplex *)x00;
	dcomplex* restrict y     = (dcomplex *)y00;

    dim_t fuse_fac = 4;
    // If either dimension is zero, or if alpha is zero, return early.
    if ( bli_zero_dim2( m, b_n ) || bli_zeq0( *alpha ) ) return;

    // If b_n is not equal to the fusing factor, then perform the entire
    // operation as a loop over axpyv.
    if (b_n != fuse_fac)
    {
        __m128d x_vec, alpha_real, alpha_imag, temp[2];

        alpha_real = _mm_set1_pd(((*alpha).real));
        alpha_imag = _mm_set1_pd(((*alpha).imag));

        for (dim_t i = 0; i < b_n; ++i)
        {
            dcomplex *a1 = a + (0) * inca + (i)*lda;
            dcomplex *chi1 = x + (i)*incx;
            dcomplex *y1 = y + (0) * incy;
            dcomplex alpha_chi1;

            // Vectorization of scaling X by alpha
            x_vec = _mm_loadu_pd((double *)chi1);

            if (bli_is_conj(conjx))
            {
                __m128d identity;

                identity = _mm_setr_pd(1, -1);

                x_vec = _mm_mul_pd(x_vec, identity);
            }

            temp[0] = _mm_mul_pd(x_vec, alpha_real);
            temp[1] = _mm_mul_pd(x_vec, alpha_imag);

            temp[1] = _mm_permute_pd(temp[1], 0b01);

            temp[0] = _mm_addsub_pd(temp[0], temp[1]);

            _mm_storeu_pd((double *)&alpha_chi1, temp[0]);

            bli_zaxpyv_zen_int_5
            (
                conja,
                m,
                &alpha_chi1,
                a1, inca,
                y1, incy,
                cntx
            );
        }

        return;
    }

    // A prefetch distance used inside the main loop
    const dim_t distance = 32;

    dcomplex chi0 = *(x + 0 * incx);
    dcomplex chi1 = *(x + 1 * incx);
    dcomplex chi2 = *(x + 2 * incx);
    dcomplex chi3 = *(x + 3 * incx);

    /* Alpha scaling of X can be vectorized
       irrespective of the incx  and should
       be avoided when alpha is 1*/
    __m128d x_vec[8], alpha_real, alpha_imag, temp[8];

    x_vec[0] = _mm_loadu_pd((double *)&chi0);
    x_vec[1] = _mm_loadu_pd((double *)&chi1);
    x_vec[2] = _mm_loadu_pd((double *)&chi2);
    x_vec[3] = _mm_loadu_pd((double *)&chi3);

    if (bli_is_conj(conjx))
    {
        __m128d identity;

        identity = _mm_setr_pd(1, -1);

        x_vec[0] = _mm_mul_pd(x_vec[0], identity);
        x_vec[1] = _mm_mul_pd(x_vec[1], identity);
        x_vec[2] = _mm_mul_pd(x_vec[2], identity);
        x_vec[3] = _mm_mul_pd(x_vec[3], identity);
    }

    if (!(bli_zeq1(*alpha)))
    {
        alpha_real = _mm_set1_pd(((*alpha).real));
        alpha_imag = _mm_set1_pd(((*alpha).imag));

        temp[0] = _mm_mul_pd(x_vec[0], alpha_real);
        temp[1] = _mm_mul_pd(x_vec[0], alpha_imag);
        temp[2] = _mm_mul_pd(x_vec[1], alpha_real);
        temp[3] = _mm_mul_pd(x_vec[1], alpha_imag);
        temp[4] = _mm_mul_pd(x_vec[2], alpha_real);
        temp[5] = _mm_mul_pd(x_vec[2], alpha_imag);
        temp[6] = _mm_mul_pd(x_vec[3], alpha_real);
        temp[7] = _mm_mul_pd(x_vec[3], alpha_imag);

        temp[1] = _mm_permute_pd(temp[1], 0b01);
        temp[3] = _mm_permute_pd(temp[3], 0b01);
        temp[5] = _mm_permute_pd(temp[5], 0b01);
        temp[7] = _mm_permute_pd(temp[7], 0b01);

        temp[0] = _mm_addsub_pd(temp[0], temp[1]);
        temp[2] = _mm_addsub_pd(temp[2], temp[3]);
        temp[4] = _mm_addsub_pd(temp[4], temp[5]);
        temp[6] = _mm_addsub_pd(temp[6], temp[7]);

        _mm_storeu_pd((double *)&chi0, temp[0]);
        _mm_storeu_pd((double *)&chi1, temp[2]);
        _mm_storeu_pd((double *)&chi2, temp[4]);
        _mm_storeu_pd((double *)&chi3, temp[6]);
    }
    else
    {
        _mm_storeu_pd((double *)&chi0, x_vec[0]);
        _mm_storeu_pd((double *)&chi1, x_vec[1]);
        _mm_storeu_pd((double *)&chi2, x_vec[2]);
        _mm_storeu_pd((double *)&chi3, x_vec[3]);
    }

    dim_t i = 0;

    double *a_ptr[4];
    double *y0 = (double *)y;

    a_ptr[0] = (double *)a;
    a_ptr[1] = (double *)a + 2 * lda;
    a_ptr[2] = a_ptr[1] + 2 * lda;
    a_ptr[3] = a_ptr[2] + 2 * lda;


    // Prefetching the elements of A to the L1 cache.
    // These will be used even if SSE instructions are used
    _mm_prefetch((char const*)(a_ptr[0]), _MM_HINT_T1);
    _mm_prefetch((char const*)(a_ptr[1]), _MM_HINT_T1);
    _mm_prefetch((char const*)(a_ptr[2]), _MM_HINT_T1);
    _mm_prefetch((char const*)(a_ptr[3]), _MM_HINT_T1);

    if (inca == 1 && incy == 1)
    {

        v4df_t ymm0, ymm1, ymm2, ymm3;
        v4df_t ymm4, ymm5, ymm6, ymm7;
        v4df_t ymm8, ymm10;
        v4df_t ymm12, ymm13, ymm14, ymm15;

        // broadcast real & imag parts of 4 elements of x
        ymm0.v = _mm256_broadcast_sd(&chi0.real); // real part of x0
        ymm1.v = _mm256_broadcast_sd(&chi0.imag); // imag part of x0
        ymm2.v = _mm256_broadcast_sd(&chi1.real); // real part of x1
        ymm3.v = _mm256_broadcast_sd(&chi1.imag); // imag part of x1
        ymm4.v = _mm256_broadcast_sd(&chi2.real); // real part of x2
        ymm5.v = _mm256_broadcast_sd(&chi2.imag); // imag part of x2
        ymm6.v = _mm256_broadcast_sd(&chi3.real); // real part of x3
        ymm7.v = _mm256_broadcast_sd(&chi3.imag); // imag part of x3

        if (bli_is_noconj(conja))
        {

            for (; (i + 3) < m; i += 4)
            {
                // load first two columns of A
                ymm8.v = _mm256_loadu_pd(a_ptr[0]);  // 2 complex values from a0
                ymm10.v = _mm256_loadu_pd(a_ptr[1]); // 2 complex values from a0
                // load 3rd and 4th columns of A
                ymm14.v = _mm256_loadu_pd(a_ptr[2]);
                ymm15.v = _mm256_loadu_pd(a_ptr[3]);

                // Multiply the loaded columns of A by X
                ymm12.v = _mm256_mul_pd(ymm8.v, ymm0.v);
                ymm13.v = _mm256_mul_pd(ymm8.v, ymm1.v);

                ymm12.v = _mm256_fmadd_pd(ymm10.v, ymm2.v, ymm12.v);
                ymm13.v = _mm256_fmadd_pd(ymm10.v, ymm3.v, ymm13.v);

                _mm_prefetch((char const*)(a_ptr[0] + distance), _MM_HINT_T1);
                _mm_prefetch((char const*)(a_ptr[1] + distance), _MM_HINT_T1);
                _mm_prefetch((char const*)(a_ptr[2] + distance), _MM_HINT_T1);
                _mm_prefetch((char const*)(a_ptr[3] + distance), _MM_HINT_T1);

                ymm12.v = _mm256_fmadd_pd(ymm14.v, ymm4.v, ymm12.v);
                ymm13.v = _mm256_fmadd_pd(ymm14.v, ymm5.v, ymm13.v);

                _mm_prefetch((char const*)(y0 + distance), _MM_HINT_T1);

                ymm12.v = _mm256_fmadd_pd(ymm15.v, ymm6.v, ymm12.v);
                ymm13.v = _mm256_fmadd_pd(ymm15.v, ymm7.v, ymm13.v);

                // load Y vector
                ymm10.v = _mm256_loadu_pd(y0);

                // Permute and reduce the complex and real parts
                ymm13.v = _mm256_permute_pd(ymm13.v, 5);
                ymm8.v = _mm256_addsub_pd(ymm12.v, ymm13.v);

                ymm12.v = _mm256_add_pd(ymm8.v, ymm10.v);

                _mm256_storeu_pd((double *)(y0), ymm12.v);

                // load first two columns of A
                ymm8.v = _mm256_loadu_pd(a_ptr[0] + 4);  // 2 complex values from a0
                ymm10.v = _mm256_loadu_pd(a_ptr[1] + 4); // 2 complex values from a0
                // load 3rd and 4th columns of A
                ymm14.v = _mm256_loadu_pd(a_ptr[2] + 4);
                ymm15.v = _mm256_loadu_pd(a_ptr[3] + 4);

                ymm12.v = _mm256_mul_pd(ymm8.v, ymm0.v);
                ymm13.v = _mm256_mul_pd(ymm8.v, ymm1.v);

                ymm12.v = _mm256_fmadd_pd(ymm10.v, ymm2.v, ymm12.v);
                ymm13.v = _mm256_fmadd_pd(ymm10.v, ymm3.v, ymm13.v);

                _mm_prefetch((char const*)(a_ptr[0] + distance * 2), _MM_HINT_T1);
                _mm_prefetch((char const*)(a_ptr[1] + distance * 2), _MM_HINT_T1);
                _mm_prefetch((char const*)(a_ptr[2] + distance * 2), _MM_HINT_T1);
                _mm_prefetch((char const*)(a_ptr[3] + distance * 2), _MM_HINT_T1);

                ymm12.v = _mm256_fmadd_pd(ymm14.v, ymm4.v, ymm12.v);
                ymm13.v = _mm256_fmadd_pd(ymm14.v, ymm5.v, ymm13.v);

                _mm_prefetch((char const*)(y0 + distance * 2), _MM_HINT_T1);

                ymm12.v = _mm256_fmadd_pd(ymm15.v, ymm6.v, ymm12.v);
                ymm13.v = _mm256_fmadd_pd(ymm15.v, ymm7.v, ymm13.v);

                // load Y vector
                ymm10.v = _mm256_loadu_pd(y0 + 4);

                ymm13.v = _mm256_permute_pd(ymm13.v, 5);
                ymm8.v = _mm256_addsub_pd(ymm12.v, ymm13.v);

                ymm12.v = _mm256_add_pd(ymm8.v, ymm10.v);

                _mm256_storeu_pd((double *)(y0 + 4), ymm12.v);

                y0 += 8;
                a_ptr[0] += 8;
                a_ptr[1] += 8;
                a_ptr[2] += 8;
                a_ptr[3] += 8;
            }

            for (; (i + 1) < m; i += 2)
            {
                // load first two columns of A
                ymm8.v = _mm256_loadu_pd(a_ptr[0]);  // 2 complex values from a0
                ymm10.v = _mm256_loadu_pd(a_ptr[1]); // 2 complex values from a0
                // load 3rd and 4th columns of A
                ymm14.v = _mm256_loadu_pd(a_ptr[2]);
                ymm15.v = _mm256_loadu_pd(a_ptr[3]);

                ymm12.v = _mm256_mul_pd(ymm8.v, ymm0.v);
                ymm13.v = _mm256_mul_pd(ymm8.v, ymm1.v);

                ymm12.v = _mm256_fmadd_pd(ymm10.v, ymm2.v, ymm12.v);
                ymm13.v = _mm256_fmadd_pd(ymm10.v, ymm3.v, ymm13.v);

                ymm12.v = _mm256_fmadd_pd(ymm14.v, ymm4.v, ymm12.v);
                ymm13.v = _mm256_fmadd_pd(ymm14.v, ymm5.v, ymm13.v);

                ymm12.v = _mm256_fmadd_pd(ymm15.v, ymm6.v, ymm12.v);
                ymm13.v = _mm256_fmadd_pd(ymm15.v, ymm7.v, ymm13.v);

                // load Y vector
                ymm10.v = _mm256_loadu_pd(y0);

                ymm13.v = _mm256_permute_pd(ymm13.v, 5);
                ymm8.v = _mm256_addsub_pd(ymm12.v, ymm13.v);

                ymm12.v = _mm256_add_pd(ymm8.v, ymm10.v);

                _mm256_storeu_pd((double *)(y0), ymm12.v);

                y0 += 4;
                a_ptr[0] += 4;
                a_ptr[1] += 4;
                a_ptr[2] += 4;
                a_ptr[3] += 4;
            }
        }
        else
        {

            for (; (i + 3) < m; i += 4)
            {
                // load first two columns of A
                ymm8.v = _mm256_loadu_pd(a_ptr[0]);  // 2 complex values from a0
                ymm10.v = _mm256_loadu_pd(a_ptr[1]); // 2 complex values from a0
                // load 3rd and 4th columns of A
                ymm14.v = _mm256_loadu_pd(a_ptr[2]);
                ymm15.v = _mm256_loadu_pd(a_ptr[3]);

                ymm12.v = _mm256_mul_pd(ymm8.v, ymm0.v);
                ymm13.v = _mm256_mul_pd(ymm8.v, ymm1.v);

                ymm12.v = _mm256_fmadd_pd(ymm10.v, ymm2.v, ymm12.v);
                ymm13.v = _mm256_fmadd_pd(ymm10.v, ymm3.v, ymm13.v);

                _mm_prefetch((char const*)(a_ptr[0] + distance), _MM_HINT_T1);
                _mm_prefetch((char const*)(a_ptr[1] + distance), _MM_HINT_T1);
                _mm_prefetch((char const*)(a_ptr[2] + distance), _MM_HINT_T1);
                _mm_prefetch((char const*)(a_ptr[3] + distance), _MM_HINT_T1);

                ymm12.v = _mm256_fmadd_pd(ymm14.v, ymm4.v, ymm12.v);
                ymm13.v = _mm256_fmadd_pd(ymm14.v, ymm5.v, ymm13.v);

                _mm_prefetch((char const*)(y0 + distance), _MM_HINT_T1);

                ymm12.v = _mm256_fmadd_pd(ymm15.v, ymm6.v, ymm12.v);
                ymm13.v = _mm256_fmadd_pd(ymm15.v, ymm7.v, ymm13.v);

                // load Y vector
                ymm10.v = _mm256_loadu_pd(y0);

                ymm12.v = _mm256_permute_pd(ymm12.v, 5);
                ymm8.v = _mm256_addsub_pd(ymm13.v, ymm12.v);
                ymm8.v = _mm256_permute_pd(ymm8.v, 5);

                ymm12.v = _mm256_add_pd(ymm8.v, ymm10.v);

                _mm256_storeu_pd((double *)(y0), ymm12.v);

                ymm8.v = _mm256_loadu_pd(a_ptr[0] + 4);  // 2 complex values from a0
                ymm10.v = _mm256_loadu_pd(a_ptr[1] + 4); // 2 complex values from a0
                // load 3rd and 4th columns of A
                ymm14.v = _mm256_loadu_pd(a_ptr[2] + 4);
                ymm15.v = _mm256_loadu_pd(a_ptr[3] + 4);

                ymm12.v = _mm256_mul_pd(ymm8.v, ymm0.v);
                ymm13.v = _mm256_mul_pd(ymm8.v, ymm1.v);

                ymm12.v = _mm256_fmadd_pd(ymm10.v, ymm2.v, ymm12.v);
                ymm13.v = _mm256_fmadd_pd(ymm10.v, ymm3.v, ymm13.v);

                _mm_prefetch((char const*)(a_ptr[0] + distance * 2), _MM_HINT_T1);
                _mm_prefetch((char const*)(a_ptr[1] + distance * 2), _MM_HINT_T1);
                _mm_prefetch((char const*)(a_ptr[2] + distance * 2), _MM_HINT_T1);
                _mm_prefetch((char const*)(a_ptr[3] + distance * 2), _MM_HINT_T1);

                ymm12.v = _mm256_fmadd_pd(ymm14.v, ymm4.v, ymm12.v);
                ymm13.v = _mm256_fmadd_pd(ymm14.v, ymm5.v, ymm13.v);

                _mm_prefetch((char const*)(y0 + distance * 2), _MM_HINT_T1);

                ymm12.v = _mm256_fmadd_pd(ymm15.v, ymm6.v, ymm12.v);
                ymm13.v = _mm256_fmadd_pd(ymm15.v, ymm7.v, ymm13.v);

                // load Y vector
                ymm10.v = _mm256_loadu_pd(y0 + 4);

                ymm12.v = _mm256_permute_pd(ymm12.v, 5);
                ymm8.v = _mm256_addsub_pd(ymm13.v, ymm12.v);
                ymm8.v = _mm256_permute_pd(ymm8.v, 5);

                ymm12.v = _mm256_add_pd(ymm8.v, ymm10.v);

                _mm256_storeu_pd((double *)(y0 + 4), ymm12.v);

                y0 += 8;
                a_ptr[0] += 8;
                a_ptr[1] += 8;
                a_ptr[2] += 8;
                a_ptr[3] += 8;
            }

            for (; (i + 1) < m; i += 2)
            {
                // load first two columns of A
                ymm8.v = _mm256_loadu_pd(a_ptr[0]);  // 2 complex values from a0
                ymm10.v = _mm256_loadu_pd(a_ptr[1]); // 2 complex values from a0
                // load 3rd and 4th columns of A
                ymm14.v = _mm256_loadu_pd(a_ptr[2]);
                ymm15.v = _mm256_loadu_pd(a_ptr[3]);

                ymm12.v = _mm256_mul_pd(ymm8.v, ymm0.v);
                ymm13.v = _mm256_mul_pd(ymm8.v, ymm1.v);

                ymm12.v = _mm256_fmadd_pd(ymm10.v, ymm2.v, ymm12.v);
                ymm13.v = _mm256_fmadd_pd(ymm10.v, ymm3.v, ymm13.v);

                //_mm_prefetch(y0, _MM_HINT_T1);

                ymm12.v = _mm256_fmadd_pd(ymm14.v, ymm4.v, ymm12.v);
                ymm13.v = _mm256_fmadd_pd(ymm14.v, ymm5.v, ymm13.v);

                ymm12.v = _mm256_fmadd_pd(ymm15.v, ymm6.v, ymm12.v);
                ymm13.v = _mm256_fmadd_pd(ymm15.v, ymm7.v, ymm13.v);

                // load Y vector
                ymm10.v = _mm256_loadu_pd(y0);

                ymm12.v = _mm256_permute_pd(ymm12.v, 5);
                ymm8.v = _mm256_addsub_pd(ymm13.v, ymm12.v);
                ymm8.v = _mm256_permute_pd(ymm8.v, 5);

                ymm12.v = _mm256_add_pd(ymm8.v, ymm10.v);

                _mm256_storeu_pd((double *)(y0), ymm12.v);

                y0 += 4;
                a_ptr[0] += 4;
                a_ptr[1] += 4;
                a_ptr[2] += 4;
                a_ptr[3] += 4;
            }
        }
    }

    // Issue vzeroupper instruction to clear upper lanes of ymm registers.
    // This avoids a performance penalty caused by false dependencies when
    // transitioning from AVX to SSE instructions (which may occur later,
    // especially if BLIS is compiled with -mfpmath=sse).
    _mm256_zeroupper();

    __m128d a_vec[4], y_vec, inter[2];

    // broadcast real & imag parts of 4 elements of x
    x_vec[0] = _mm_set1_pd(chi0.real); // real part of x0
    x_vec[1] = _mm_set1_pd(chi0.imag); // imag part of x0
    x_vec[2] = _mm_set1_pd(chi1.real); // real part of x1
    x_vec[3] = _mm_set1_pd(chi1.imag); // imag part of x1
    x_vec[4] = _mm_set1_pd(chi2.real); // real part of x2
    x_vec[5] = _mm_set1_pd(chi2.imag); // imag part of x2
    x_vec[6] = _mm_set1_pd(chi3.real); // real part of x3
    x_vec[7] = _mm_set1_pd(chi3.imag); // imag part of x3

    if (bli_is_noconj(conja))
    {
        for (; i < m; i++)
        {
            // load first two columns of A
            a_vec[0] = _mm_loadu_pd(a_ptr[0]); // 2 complex values from a0
            a_vec[1] = _mm_loadu_pd(a_ptr[1]); // 2 complex values from a0
            a_vec[2] = _mm_loadu_pd(a_ptr[2]); // 2 complex values from a0
            a_vec[3] = _mm_loadu_pd(a_ptr[3]); // 2 complex values from a0

            inter[0] = _mm_mul_pd(a_vec[0], x_vec[0]);
            inter[1] = _mm_mul_pd(a_vec[0], x_vec[1]);

            inter[0] = _mm_fmadd_pd(a_vec[1], x_vec[2], inter[0]);
            inter[1] = _mm_fmadd_pd(a_vec[1], x_vec[3], inter[1]);

            //_mm_prefetch(y0, _MM_HINT_T1);

            inter[0] = _mm_fmadd_pd(a_vec[2], x_vec[4], inter[0]);
            inter[1] = _mm_fmadd_pd(a_vec[2], x_vec[5], inter[1]);

            inter[0] = _mm_fmadd_pd(a_vec[3], x_vec[6], inter[0]);
            inter[1] = _mm_fmadd_pd(a_vec[3], x_vec[7], inter[1]);

            inter[1] = _mm_permute_pd(inter[1], 0b01);
            inter[0] = _mm_addsub_pd(inter[0], inter[1]);

            // load Y vector
            y_vec = _mm_loadu_pd(y0);

            y_vec = _mm_add_pd(y_vec, inter[0]);

            _mm_storeu_pd((double *)(y0), y_vec);

            y0 += 2 * incy;
            a_ptr[0] += 2 * inca;
            a_ptr[1] += 2 * inca;
            a_ptr[2] += 2 * inca;
            a_ptr[3] += 2 * inca;
        }
    }
    else
    {
        for (; i < m; i++)
        {
            // load first two columns of A
            a_vec[0] = _mm_loadu_pd(a_ptr[0]); // 2 complex values from a0
            a_vec[1] = _mm_loadu_pd(a_ptr[1]); // 2 complex values from a0
                                               // load 3rd and 4th columns of A
            a_vec[2] = _mm_loadu_pd(a_ptr[2]); // 2 complex values from a0
            a_vec[3] = _mm_loadu_pd(a_ptr[3]); // 2 complex values from a0

            inter[0] = _mm_mul_pd(a_vec[0], x_vec[0]);
            inter[1] = _mm_mul_pd(a_vec[0], x_vec[1]);

            inter[0] = _mm_fmadd_pd(a_vec[1], x_vec[2], inter[0]);
            inter[1] = _mm_fmadd_pd(a_vec[1], x_vec[3], inter[1]);

            // load Y vector
            y_vec = _mm_loadu_pd(y0);

            inter[0] = _mm_fmadd_pd(a_vec[2], x_vec[4], inter[0]);
            inter[1] = _mm_fmadd_pd(a_vec[2], x_vec[5], inter[1]);

            inter[0] = _mm_fmadd_pd(a_vec[3], x_vec[6], inter[0]);
            inter[1] = _mm_fmadd_pd(a_vec[3], x_vec[7], inter[1]);

            inter[0] = _mm_permute_pd(inter[0], 0b01);
            inter[0] = _mm_addsub_pd(inter[1], inter[0]);
            inter[0] = _mm_permute_pd(inter[0], 0b01);

            y_vec = _mm_add_pd(y_vec, inter[0]);

            _mm_storeu_pd((double *)(y0), y_vec);

            y0 += 2 * incy;
            a_ptr[0] += 2 * inca;
            a_ptr[1] += 2 * inca;
            a_ptr[2] += 2 * inca;
            a_ptr[3] += 2 * inca;
        }
    }

    // vzeroupper is added by the compiler at the end of the kernel
}
