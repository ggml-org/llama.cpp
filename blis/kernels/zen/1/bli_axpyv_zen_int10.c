/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2016 - 2025, Advanced Micro Devices, Inc. All rights reserved.
   Copyright (C) 2018 - 2020, The University of Texas at Austin. All rights reserved.

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

// -----------------------------------------------------------------------------

void bli_saxpyv_zen_int_10
     (
             conj_t  conjx,
             dim_t   n,
       const void*   alpha00,
       const void*   x00, inc_t incx,
             void*   y00, inc_t incy,
       const cntx_t* cntx
     )
{
	float *alpha = (float *)alpha00;
	float *x     = (float *)x00;
	float *y     = (float *)y00;

    const dim_t      n_elem_per_reg = 8;

    dim_t            i;

    float*  restrict x0;
    float*  restrict y0;

    __m256           alphav;
    __m256           xv[15];
    __m256           yv[15];
    __m256           zv[15];

    // If the vector dimension is zero, or if alpha is zero, return early.
    if ( bli_zero_dim1( n ) || PASTEMAC(s,eq0)( *alpha ) )
    {
        return;
    }

    // Initialize local pointers.
    x0 = x;
    y0 = y;

    if ( incx == 1 && incy == 1 )
    {
        // Broadcast the alpha scalar to all elements of a vector register.
        alphav = _mm256_broadcast_ss( alpha );

        for (i = 0; (i + 119) < n; i += 120)
        {
            // 120 elements will be processed per loop; 15 FMAs will run per loop.
            xv[0] = _mm256_loadu_ps(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm256_loadu_ps(x0 + 1 * n_elem_per_reg);
            xv[2] = _mm256_loadu_ps(x0 + 2 * n_elem_per_reg);
            xv[3] = _mm256_loadu_ps(x0 + 3 * n_elem_per_reg);
            xv[4] = _mm256_loadu_ps(x0 + 4 * n_elem_per_reg);
            xv[5] = _mm256_loadu_ps(x0 + 5 * n_elem_per_reg);
            xv[6] = _mm256_loadu_ps(x0 + 6 * n_elem_per_reg);
            xv[7] = _mm256_loadu_ps(x0 + 7 * n_elem_per_reg);
            xv[8] = _mm256_loadu_ps(x0 + 8 * n_elem_per_reg);
            xv[9] = _mm256_loadu_ps(x0 + 9 * n_elem_per_reg);
            xv[10] = _mm256_loadu_ps(x0 + 10 * n_elem_per_reg);
            xv[11] = _mm256_loadu_ps(x0 + 11 * n_elem_per_reg);
            xv[12] = _mm256_loadu_ps(x0 + 12 * n_elem_per_reg);
            xv[13] = _mm256_loadu_ps(x0 + 13 * n_elem_per_reg);
            xv[14] = _mm256_loadu_ps(x0 + 14 * n_elem_per_reg);

            yv[0] = _mm256_loadu_ps(y0 + 0 * n_elem_per_reg);
            yv[1] = _mm256_loadu_ps(y0 + 1 * n_elem_per_reg);
            yv[2] = _mm256_loadu_ps(y0 + 2 * n_elem_per_reg);
            yv[3] = _mm256_loadu_ps(y0 + 3 * n_elem_per_reg);
            yv[4] = _mm256_loadu_ps(y0 + 4 * n_elem_per_reg);
            yv[5] = _mm256_loadu_ps(y0 + 5 * n_elem_per_reg);
            yv[6] = _mm256_loadu_ps(y0 + 6 * n_elem_per_reg);
            yv[7] = _mm256_loadu_ps(y0 + 7 * n_elem_per_reg);
            yv[8] = _mm256_loadu_ps(y0 + 8 * n_elem_per_reg);
            yv[9] = _mm256_loadu_ps(y0 + 9 * n_elem_per_reg);
            yv[10] = _mm256_loadu_ps(y0 + 10 * n_elem_per_reg);
            yv[11] = _mm256_loadu_ps(y0 + 11 * n_elem_per_reg);
            yv[12] = _mm256_loadu_ps(y0 + 12 * n_elem_per_reg);
            yv[13] = _mm256_loadu_ps(y0 + 13 * n_elem_per_reg);
            yv[14] = _mm256_loadu_ps(y0 + 14 * n_elem_per_reg);

            zv[0] = _mm256_fmadd_ps(xv[0], alphav, yv[0]);
            zv[1] = _mm256_fmadd_ps(xv[1], alphav, yv[1]);
            zv[2] = _mm256_fmadd_ps(xv[2], alphav, yv[2]);
            zv[3] = _mm256_fmadd_ps(xv[3], alphav, yv[3]);
            zv[4] = _mm256_fmadd_ps(xv[4], alphav, yv[4]);
            zv[5] = _mm256_fmadd_ps(xv[5], alphav, yv[5]);
            zv[6] = _mm256_fmadd_ps(xv[6], alphav, yv[6]);
            zv[7] = _mm256_fmadd_ps(xv[7], alphav, yv[7]);
            zv[8] = _mm256_fmadd_ps(xv[8], alphav, yv[8]);
            zv[9] = _mm256_fmadd_ps(xv[9], alphav, yv[9]);
            zv[10] = _mm256_fmadd_ps(xv[10], alphav, yv[10]);
            zv[11] = _mm256_fmadd_ps(xv[11], alphav, yv[11]);
            zv[12] = _mm256_fmadd_ps(xv[12], alphav, yv[12]);
            zv[13] = _mm256_fmadd_ps(xv[13], alphav, yv[13]);
            zv[14] = _mm256_fmadd_ps(xv[14], alphav, yv[14]);

            _mm256_storeu_ps((y0 + 0 * n_elem_per_reg), zv[0]);
            _mm256_storeu_ps((y0 + 1 * n_elem_per_reg), zv[1]);
            _mm256_storeu_ps((y0 + 2 * n_elem_per_reg), zv[2]);
            _mm256_storeu_ps((y0 + 3 * n_elem_per_reg), zv[3]);
            _mm256_storeu_ps((y0 + 4 * n_elem_per_reg), zv[4]);
            _mm256_storeu_ps((y0 + 5 * n_elem_per_reg), zv[5]);
            _mm256_storeu_ps((y0 + 6 * n_elem_per_reg), zv[6]);
            _mm256_storeu_ps((y0 + 7 * n_elem_per_reg), zv[7]);
            _mm256_storeu_ps((y0 + 8 * n_elem_per_reg), zv[8]);
            _mm256_storeu_ps((y0 + 9 * n_elem_per_reg), zv[9]);
            _mm256_storeu_ps((y0 + 10 * n_elem_per_reg), zv[10]);
            _mm256_storeu_ps((y0 + 11 * n_elem_per_reg), zv[11]);
            _mm256_storeu_ps((y0 + 12 * n_elem_per_reg), zv[12]);
            _mm256_storeu_ps((y0 + 13 * n_elem_per_reg), zv[13]);
            _mm256_storeu_ps((y0 + 14 * n_elem_per_reg), zv[14]);

            x0 += 15 * n_elem_per_reg;
            y0 += 15 * n_elem_per_reg;
        }

        for (; (i + 79) < n; i += 80 )
        {
            // 80 elements will be processed per loop; 10 FMAs will run per loop.
            xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
            xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );
            xv[2] = _mm256_loadu_ps( x0 + 2*n_elem_per_reg );
            xv[3] = _mm256_loadu_ps( x0 + 3*n_elem_per_reg );
            xv[4] = _mm256_loadu_ps( x0 + 4*n_elem_per_reg );
            xv[5] = _mm256_loadu_ps( x0 + 5*n_elem_per_reg );
            xv[6] = _mm256_loadu_ps( x0 + 6*n_elem_per_reg );
            xv[7] = _mm256_loadu_ps( x0 + 7*n_elem_per_reg );
            xv[8] = _mm256_loadu_ps( x0 + 8*n_elem_per_reg );
            xv[9] = _mm256_loadu_ps( x0 + 9*n_elem_per_reg );

            yv[0] = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );
            yv[1] = _mm256_loadu_ps( y0 + 1*n_elem_per_reg );
            yv[2] = _mm256_loadu_ps( y0 + 2*n_elem_per_reg );
            yv[3] = _mm256_loadu_ps( y0 + 3*n_elem_per_reg );
            yv[4] = _mm256_loadu_ps( y0 + 4*n_elem_per_reg );
            yv[5] = _mm256_loadu_ps( y0 + 5*n_elem_per_reg );
            yv[6] = _mm256_loadu_ps( y0 + 6*n_elem_per_reg );
            yv[7] = _mm256_loadu_ps( y0 + 7*n_elem_per_reg );
            yv[8] = _mm256_loadu_ps( y0 + 8*n_elem_per_reg );
            yv[9] = _mm256_loadu_ps( y0 + 9*n_elem_per_reg );

            zv[0] = _mm256_fmadd_ps( xv[0], alphav, yv[0] );
            zv[1] = _mm256_fmadd_ps( xv[1], alphav, yv[1] );
            zv[2] = _mm256_fmadd_ps( xv[2], alphav, yv[2] );
            zv[3] = _mm256_fmadd_ps( xv[3], alphav, yv[3] );
            zv[4] = _mm256_fmadd_ps( xv[4], alphav, yv[4] );
            zv[5] = _mm256_fmadd_ps( xv[5], alphav, yv[5] );
            zv[6] = _mm256_fmadd_ps( xv[6], alphav, yv[6] );
            zv[7] = _mm256_fmadd_ps( xv[7], alphav, yv[7] );
            zv[8] = _mm256_fmadd_ps( xv[8], alphav, yv[8] );
            zv[9] = _mm256_fmadd_ps( xv[9], alphav, yv[9] );

            _mm256_storeu_ps( (y0 + 0*n_elem_per_reg), zv[0] );
            _mm256_storeu_ps( (y0 + 1*n_elem_per_reg), zv[1] );
            _mm256_storeu_ps( (y0 + 2*n_elem_per_reg), zv[2] );
            _mm256_storeu_ps( (y0 + 3*n_elem_per_reg), zv[3] );
            _mm256_storeu_ps( (y0 + 4*n_elem_per_reg), zv[4] );
            _mm256_storeu_ps( (y0 + 5*n_elem_per_reg), zv[5] );
            _mm256_storeu_ps( (y0 + 6*n_elem_per_reg), zv[6] );
            _mm256_storeu_ps( (y0 + 7*n_elem_per_reg), zv[7] );
            _mm256_storeu_ps( (y0 + 8*n_elem_per_reg), zv[8] );
            _mm256_storeu_ps( (y0 + 9*n_elem_per_reg), zv[9] );

            x0 += 10*n_elem_per_reg;
            y0 += 10*n_elem_per_reg;
        }

        for ( ; (i + 39) < n; i += 40 )
        {
            xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
            xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );
            xv[2] = _mm256_loadu_ps( x0 + 2*n_elem_per_reg );
            xv[3] = _mm256_loadu_ps( x0 + 3*n_elem_per_reg );
            xv[4] = _mm256_loadu_ps( x0 + 4*n_elem_per_reg );

            yv[0] = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );
            yv[1] = _mm256_loadu_ps( y0 + 1*n_elem_per_reg );
            yv[2] = _mm256_loadu_ps( y0 + 2*n_elem_per_reg );
            yv[3] = _mm256_loadu_ps( y0 + 3*n_elem_per_reg );
            yv[4] = _mm256_loadu_ps( y0 + 4*n_elem_per_reg );

            zv[0] = _mm256_fmadd_ps( xv[0], alphav, yv[0] );
            zv[1] = _mm256_fmadd_ps( xv[1], alphav, yv[1] );
            zv[2] = _mm256_fmadd_ps( xv[2], alphav, yv[2] );
            zv[3] = _mm256_fmadd_ps( xv[3], alphav, yv[3] );
            zv[4] = _mm256_fmadd_ps( xv[4], alphav, yv[4] );

            _mm256_storeu_ps( (y0 + 0*n_elem_per_reg), zv[0] );
            _mm256_storeu_ps( (y0 + 1*n_elem_per_reg), zv[1] );
            _mm256_storeu_ps( (y0 + 2*n_elem_per_reg), zv[2] );
            _mm256_storeu_ps( (y0 + 3*n_elem_per_reg), zv[3] );
            _mm256_storeu_ps( (y0 + 4*n_elem_per_reg), zv[4] );

            x0 += 5*n_elem_per_reg;
            y0 += 5*n_elem_per_reg;
        }

        for ( ; (i + 31) < n; i += 32 )
        {
            xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
            xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );
            xv[2] = _mm256_loadu_ps( x0 + 2*n_elem_per_reg );
            xv[3] = _mm256_loadu_ps( x0 + 3*n_elem_per_reg );

            yv[0] = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );
            yv[1] = _mm256_loadu_ps( y0 + 1*n_elem_per_reg );
            yv[2] = _mm256_loadu_ps( y0 + 2*n_elem_per_reg );
            yv[3] = _mm256_loadu_ps( y0 + 3*n_elem_per_reg );

            zv[0] = _mm256_fmadd_ps( xv[0], alphav, yv[0] );
            zv[1] = _mm256_fmadd_ps( xv[1], alphav, yv[1] );
            zv[2] = _mm256_fmadd_ps( xv[2], alphav, yv[2] );
            zv[3] = _mm256_fmadd_ps( xv[3], alphav, yv[3] );

            _mm256_storeu_ps( (y0 + 0*n_elem_per_reg), zv[0] );
            _mm256_storeu_ps( (y0 + 1*n_elem_per_reg), zv[1] );
            _mm256_storeu_ps( (y0 + 2*n_elem_per_reg), zv[2] );
            _mm256_storeu_ps( (y0 + 3*n_elem_per_reg), zv[3] );

            x0 += 4*n_elem_per_reg;
            y0 += 4*n_elem_per_reg;
        }

        for ( ; (i + 15) < n; i += 16 )
        {
            xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
            xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );

            yv[0] = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );
            yv[1] = _mm256_loadu_ps( y0 + 1*n_elem_per_reg );

            zv[0] = _mm256_fmadd_ps( xv[0], alphav, yv[0] );
            zv[1] = _mm256_fmadd_ps( xv[1], alphav, yv[1] );

            _mm256_storeu_ps( (y0 + 0*n_elem_per_reg), zv[0] );
            _mm256_storeu_ps( (y0 + 1*n_elem_per_reg), zv[1] );

            x0 += 2*n_elem_per_reg;
            y0 += 2*n_elem_per_reg;
        }

        for ( ; (i + 7) < n; i += 8 )
        {
            xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );

            yv[0] = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );

            zv[0] = _mm256_fmadd_ps( xv[0], alphav, yv[0] );

            _mm256_storeu_ps( (y0 + 0*n_elem_per_reg), zv[0] );

            x0 += 1*n_elem_per_reg;
            y0 += 1*n_elem_per_reg;
        }

        // Issue vzeroupper instruction to clear upper lanes of ymm registers.
        // This avoids a performance penalty caused by false dependencies when
        // transitioning from AVX to SSE instructions (which may occur as soon
        // as the n_left cleanup loop below if BLIS is compiled with
        // -mfpmath=sse).

        _mm256_zeroupper();

        for ( ; (i + 0) < n; i += 1 )
        {
            *y0 += (*alpha) * (*x0);

            x0 += 1;
            y0 += 1;
        }
    }
    else
    {
        const float alphac = *alpha;

        for ( i = 0; i < n; ++i )
        {
            const float x0c = *x0;

            *y0 += alphac * x0c;

            x0 += incx;
            y0 += incy;
        }
    }

}

// -----------------------------------------------------------------------------

BLIS_EXPORT_BLIS void bli_daxpyv_zen_int_10
     (
             conj_t  conjx,
             dim_t   n,
       const void*   alpha00,
       const void*   x00, inc_t incx,
             void*   y00, inc_t incy,
       const cntx_t* cntx
     )
{

	double *alpha = (double *)alpha00;
	double *x     = (double *)x00;
	double *y     = (double *)y00;

    const dim_t      n_elem_per_reg = 4;

    dim_t            i;

    double* restrict x0 = x;
    double* restrict y0 = y;

    __m256d          alphav;
    __m256d          xv[4];
    __m256d          yv[4];
    __m256d          zv[4];

    // If the vector dimension is zero, or if alpha is zero, return early.
    if ( bli_zero_dim1( n ) || PASTEMAC(d,eq0)( *alpha ) )
    {
        return;
    }

    // Initialize local pointers.
    x0 = x;
    y0 = y;

    if ( incx == 1 && incy == 1 )
    {
        // Broadcast the alpha scalar to all elements of a vector register.
        alphav = _mm256_broadcast_sd( alpha );

        for ( i = 0; ( i + 15 ) < n; i += 16 )
        {
            xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
            xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );
            xv[2] = _mm256_loadu_pd( x0 + 2*n_elem_per_reg );
            xv[3] = _mm256_loadu_pd( x0 + 3*n_elem_per_reg );

            yv[0] = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
            yv[1] = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );
            yv[2] = _mm256_loadu_pd( y0 + 2*n_elem_per_reg );
            yv[3] = _mm256_loadu_pd( y0 + 3*n_elem_per_reg );

            zv[0] = _mm256_fmadd_pd( xv[0], alphav, yv[0] );
            zv[1] = _mm256_fmadd_pd( xv[1], alphav, yv[1] );
            zv[2] = _mm256_fmadd_pd( xv[2], alphav, yv[2] );
            zv[3] = _mm256_fmadd_pd( xv[3], alphav, yv[3] );

            _mm256_storeu_pd( (y0 + 0*n_elem_per_reg), zv[0] );
            _mm256_storeu_pd( (y0 + 1*n_elem_per_reg), zv[1] );
            _mm256_storeu_pd( (y0 + 2*n_elem_per_reg), zv[2] );
            _mm256_storeu_pd( (y0 + 3*n_elem_per_reg), zv[3] );

            x0 += 4*n_elem_per_reg;
            y0 += 4*n_elem_per_reg;
        }

        for ( ; ( i + 7 ) < n; i += 8 )
        {
            xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
            xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );

            yv[0] = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
            yv[1] = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );

            zv[0] = _mm256_fmadd_pd( xv[0], alphav, yv[0] );
            zv[1] = _mm256_fmadd_pd( xv[1], alphav, yv[1] );

            _mm256_storeu_pd( (y0 + 0*n_elem_per_reg), zv[0] );
            _mm256_storeu_pd( (y0 + 1*n_elem_per_reg), zv[1] );

            x0 += 2*n_elem_per_reg;
            y0 += 2*n_elem_per_reg;
        }

        for ( ; ( i + 3 ) < n; i += 4 )
        {
            xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );

            yv[0] = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );

            zv[0] = _mm256_fmadd_pd( xv[0], alphav, yv[0] );

            _mm256_storeu_pd( (y0 + 0*n_elem_per_reg), zv[0] );

            x0 += 1*n_elem_per_reg;
            y0 += 1*n_elem_per_reg;
        }

        // Issue vzeroupper instruction to clear upper lanes of ymm registers.
        // This avoids a performance penalty caused by false dependencies when
        // transitioning from AVX to SSE instructions (which may occur as soon
        // as the n_left cleanup loop below if BLIS is compiled with
        // -mfpmath=sse).
        _mm256_zeroupper();

        for ( ; i < n; i += 1 )
        {
            *y0 += (*alpha) * (*x0);

            y0 += 1;
            x0 += 1;
        }
    }
    else
    {
        const double alphac = *alpha;

        for ( i = 0; i < n; ++i )
        {
            const double x0c = *x0;

            *y0 += alphac * x0c;

            x0 += incx;
            y0 += incy;
        }
    }
}

// -----------------------------------------------------------------------------

void bli_caxpyv_zen_int_5
     (
             conj_t  conjx,
             dim_t   n,
       const void*   alpha00,
       const void*   x00, inc_t incx,
             void*   y00, inc_t incy,
       const cntx_t* cntx
     )
{
	scomplex *alpha = (scomplex *)alpha00;
	scomplex *x     = (scomplex *)x00;
	scomplex *y     = (scomplex *)y00;

    const dim_t      n_elem_per_reg = 8;

    dim_t            i;

    float*  restrict x0;
    float*  restrict y0;
    float*  restrict alpha0;

    float alphaR, alphaI;

    //scomplex alpha => aR + aI i
    __m256           alphaRv;            // for broadcast vector aR (real part of alpha)
    __m256           alphaIv;            // for broadcast vector aI (imaginary part of alpha)
    __m256           xv[10];
    __m256           xShufv[10];
    __m256           yv[10];

    conj_t conjx_use = conjx;

    // If the vector dimension is zero, or if alpha is zero, return early.
    if ( bli_zero_dim1( n ) || PASTEMAC(c,eq0)( *alpha ) )
    {
        return;
    }

    // Initialize local pointers.
    x0 = (float*)x;
    y0 = (float*)y;
    alpha0 = (float*)alpha;

    alphaR = alpha->real;
    alphaI = alpha->imag;

    if ( incx == 1 && incy == 1 )
    {
        // Broadcast the alpha scalar to all elements of a vector register.
        if ( !bli_is_conj (conjx) ) // If BLIS_NO_CONJUGATE
        {
            alphaRv = _mm256_broadcast_ss( &alphaR );

            alphaIv = _mm256_set_ps(alphaI, -alphaI, alphaI, -alphaI, alphaI, -alphaI, alphaI, -alphaI);

        }
        else
        {
            alphaIv = _mm256_broadcast_ss( &alphaI );

            alphaRv = _mm256_set_ps(-alphaR, alphaR, -alphaR, alphaR, -alphaR, alphaR, -alphaR, alphaR);
        }

        //----------Scalar algorithm BLIS_NO_CONJUGATE arg-------------
        // y = alpha*x + y
        // y = (aR + aIi) * (xR + xIi) + (yR + yIi)
        // y = aR.xR + aR.xIi + aIi.xR - aIxI + (yR + yIi)
        // y = aR.xR - aIxI + yR + aR.xIi + xR.aIi + yIi
        // y = ( aR.xR - aIxI + yR ) + ( aR.xI + aI.xR + yI )i

        // SIMD algorithm
        // xv  = xR1  xI1  xR2  xI2  xR3  xI3  xR4  xI4
        // xv' = xI1  xR1  xI2  xR2  xI3  xR3  xI4  xR4 (shuffle xv)
        // arv = aR   aR   aR   aR   aR   aR   aR   aR
        // aiv = aI  -aI   aI  -aI   aI  -aI   aI  -aI
        // yv  = yR1  yI1  yR2  yI2  yR3  yI3  yR4  yI4


        //----------Scalar algorithm for BLIS_CONJUGATE arg-------------
        // y = alpha*conj(x) + y
        // y = (aR + aIi) * (xR - xIi) + (yR + yIi)
        // y = aR.xR - aR.xIi + aIi.xR + aIxI + (yR + yIi)
        // y = aR.xR + aIxI + yR - aR.xIi + xR.aIi + yIi
        // y = ( aR.xR + aIxI + yR ) + ( -aR.xI + aI.xR + yI )i
        // y = ( aR.xR + aIxI + yR ) + (aI.xR - aR.xI + yI)i

        // SIMD algorithm
        // xv  = xR1  xI1  xR2  xI2  xR3  xI3  xR4  xI4
        // xv' = xI1  xR1  xI2  xR2  xI3  xR3  xI4  xR4
        // arv = aR  -aR   aR   -aR   aR  -aR   aR  -aR
        // aiv = aI   aI   aI   aI   aI   aI   aI   aI
        // yv  = yR1  yI1  yR2  yI2  yR3  yI3  yR4  yI4

        // step 1 :  Shuffle xv vector -> xv'
        // step 2 : fma yv = arv*xv + yv
        // step 3 : fma yv = aiv*xv' + yv (old)
        //              yv = aiv*xv' + arv*xv + yv

        for ( i= 0 ; (i + 19) < n; i += 20 )
        {
            // 20 elements will be processed per loop; 10 FMAs will run per loop.

            // alphaRv = aR   aR   aR   aR   aR   aR   aR   aR
            // xv      = xR1  xI1  xR2  xI2  xR3  xI3  xR4  xI4
            xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
            xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );
            xv[2] = _mm256_loadu_ps( x0 + 2*n_elem_per_reg );
            xv[3] = _mm256_loadu_ps( x0 + 3*n_elem_per_reg );
            xv[4] = _mm256_loadu_ps( x0 + 4*n_elem_per_reg );

            // yv      = yR1  yI1  yR2  yI2  yR3  yI3  yR4  yI4
            yv[0] = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );
            yv[1] = _mm256_loadu_ps( y0 + 1*n_elem_per_reg );
            yv[2] = _mm256_loadu_ps( y0 + 2*n_elem_per_reg );
            yv[3] = _mm256_loadu_ps( y0 + 3*n_elem_per_reg );
            yv[4] = _mm256_loadu_ps( y0 + 4*n_elem_per_reg );

            // xv'     = xI1  xR1  xI2  xR2  xI3  xR3  xI4  xR4
            xShufv[0] = _mm256_permute_ps( xv[0], 0xB1);
            xShufv[1] = _mm256_permute_ps( xv[1], 0xB1);
            xShufv[2] = _mm256_permute_ps( xv[2], 0xB1);
            xShufv[3] = _mm256_permute_ps( xv[3], 0xB1);
            xShufv[4] = _mm256_permute_ps( xv[4], 0xB1);

            // alphaIv = -aI   aI  -aI   aI  -aI   aI  -aI  aI

            // yv  = ar*xv + yv
            //     = aR.xR1 + yR1, aR.xI1 + yI1, aR.xR2 + yR2, aR.xI2 + yI2, ...
            yv[0] = _mm256_fmadd_ps( xv[0], alphaRv ,yv[0]);
            yv[1] = _mm256_fmadd_ps( xv[1], alphaRv ,yv[1]);
            yv[2] = _mm256_fmadd_ps( xv[2], alphaRv ,yv[2]);
            yv[3] = _mm256_fmadd_ps( xv[3], alphaRv ,yv[3]);
            yv[4] = _mm256_fmadd_ps( xv[4], alphaRv ,yv[4]);

            // yv =  ai*xv' + yv (old)
            // yv =  ai*xv' + ar*xv + yv
            //    = -aI*xI1 + aR.xR1 + yR1, aI.xR1 + aR.xI1 + yI1, .........
            yv[0] = _mm256_fmadd_ps( xShufv[0], alphaIv, yv[0]);
            yv[1] = _mm256_fmadd_ps( xShufv[1], alphaIv, yv[1]);
            yv[2] = _mm256_fmadd_ps( xShufv[2], alphaIv, yv[2]);
            yv[3] = _mm256_fmadd_ps( xShufv[3], alphaIv, yv[3]);
            yv[4] = _mm256_fmadd_ps( xShufv[4], alphaIv, yv[4]);

            // Store back the results
            _mm256_storeu_ps( (y0 + 0*n_elem_per_reg), yv[0] );
            _mm256_storeu_ps( (y0 + 1*n_elem_per_reg), yv[1] );
            _mm256_storeu_ps( (y0 + 2*n_elem_per_reg), yv[2] );
            _mm256_storeu_ps( (y0 + 3*n_elem_per_reg), yv[3] );
            _mm256_storeu_ps( (y0 + 4*n_elem_per_reg), yv[4] );

            x0 += 5*n_elem_per_reg;
            y0 += 5*n_elem_per_reg;
        }

        for ( ; (i + 7) < n; i += 8 )
        {
            // alphaRv = aR   aR   aR   aR   aR   aR   aR   aR
            // xv      = xR1  xI1  xR2  xI2  xR3  xI3  xR4  xI4
            xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
            xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );

            // yv      = yR1  yI1  yR2  yI2  yR3  yI3  yR4  yI4
            yv[0] = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );
            yv[1] = _mm256_loadu_ps( y0 + 1*n_elem_per_reg );

            // xv'     = xI1  xR1  xI2  xR2  xI3  xR3  xI4  xR4
            xShufv[0] = _mm256_permute_ps( xv[0], 0xB1);
            xShufv[1] = _mm256_permute_ps( xv[1], 0xB1);

            // alphaIv = -aI   aI  -aI   aI  -aI   aI  -aI  aI

            // yv  = ar*xv + yv
            //     = aR.xR1 + yR1, aR.xI1 + yI1, aR.xR2 + yR2, aR.xI2 + yI2, ...
            yv[0] = _mm256_fmadd_ps( xv[0], alphaRv ,yv[0]);
            yv[1] = _mm256_fmadd_ps( xv[1], alphaRv ,yv[1]);

            // yv = ai*xv' + yv (old)
            // yv = ai*xv' + ar*xv + yv
            //    = -aI*xI1 + aR.xR1 + yR1, aI.xR1 + aR.xI1 + yI1, .........
            yv[0] = _mm256_fmadd_ps( xShufv[0], alphaIv, yv[0]);
            yv[1] = _mm256_fmadd_ps( xShufv[1], alphaIv, yv[1]);

            // Store back the result
            _mm256_storeu_ps( (y0 + 0*n_elem_per_reg), yv[0] );
            _mm256_storeu_ps( (y0 + 1*n_elem_per_reg), yv[1] );

            x0 += 2*n_elem_per_reg;
            y0 += 2*n_elem_per_reg;
        }

        for ( ; (i + 3) < n; i += 4 )
        {
            // alphaRv = aR   aR   aR   aR   aR   aR   aR   aR
            // xv      = xR1  xI1  xR2  xI2  xR3  xI3  xR4  xI4
            xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );

            // yv      = yR1  yI1  yR2  yI2  yR3  yI3  yR4  yI4
            yv[0] = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );

            // xv'     = xI1  xR1  xI2  xR2  xI3  xR3  xI4  xR4
            xShufv[0] = _mm256_permute_ps( xv[0], 0xB1);

            // alphaIv = -aI   aI  -aI   aI  -aI   aI  -aI  aI

            // yv = ar*xv + yv
            //    = aR.xR1 + yR1, aR.xI1 + yI1, aR.xR2 + yR2, aR.xI2 + yI2, ...
            yv[0] = _mm256_fmadd_ps( xv[0], alphaRv ,yv[0]);

            // yv = ai*xv' + yv (old)
            // yv = ai*xv' + ar*xv + yv
            //    = aR.xR1 - aI*xI1 + yR1, aR.xI1 + aI.xR1 + yI1
            yv[0] = _mm256_fmadd_ps( xShufv[0], alphaIv, yv[0]);

            // Store back the result
            _mm256_storeu_ps( (y0 + 0*n_elem_per_reg), yv[0] );

            x0 += 1*n_elem_per_reg;
            y0 += 1*n_elem_per_reg;
        }

        // Issue vzeroupper instruction to clear upper lanes of ymm registers.
        // This avoids a performance penalty caused by false dependencies when
        // transitioning from AVX to SSE instructions (which may occur as soon
        // as the n_left cleanup loop below if BLIS is compiled with
        // -mfpmath=sse).
        _mm256_zeroupper();

        /* Residual values are calculated here
        y0 += (alpha) * (x0); --> BLIS_NO_CONJUGATE
        y0 += ( aR.xR - aIxI + yR ) + ( aR.xI + aI.xR + yI )i

        y0 += (alpha) * conjx(x0); --> BLIS_CONJUGATE
        y0 = ( aR.xR + aIxI + yR ) + (aI.xR - aR.xI + yI)i */

        if ( !bli_is_conj(conjx_use) ) //  BLIS_NO_CONJUGATE
        {
            for ( ; (i + 0) < n; i += 1 )
            {
                // real part: ( aR.xR - aIxI + yR )
                *y0       += *alpha0 * (*x0) - (*(alpha0 + 1)) * (*(x0+1));
                // img part: ( aR.xI + aI.xR + yI )
                *(y0 + 1) += *alpha0 * (*(x0+1)) +  (*(alpha0 + 1)) * (*x0);
                x0 += 2;
                y0 += 2;
            }
        }
        else //  BLIS_CONJUGATE
        {
            for ( ; (i + 0) < n; i += 1 )
            {
                // real part: ( aR.xR + aIxI + yR )
                *y0       += *alpha0 * (*x0) + (*(alpha0 + 1)) * (*(x0+1));
                // img part: (  aI.xR - aR.xI + yI )
                *(y0 + 1) += (*(alpha0 + 1)) * (*x0) - (*alpha0) * (*(x0+1));
                x0 += 2;
                y0 += 2;
            }
        }

    }
    else
    {
        const float alphar = *alpha0;
        const float alphai = *(alpha0 + 1);

        if ( !bli_is_conj(conjx_use) )
        {
            for ( i = 0; i < n; ++i )
            {
                const float x0c = *x0;
                const float x1c = *( x0+1 );

                *y0         += alphar * x0c - alphai * x1c;
                *(y0 + 1)   += alphar * x1c + alphai * x0c;

                x0 += incx * 2;
                y0 += incy * 2;
            }
        }
        else
        {
            for ( i = 0; i < n; ++i )
            {
                const float x0c = *x0;
                const float x1c = *( x0+1 );

                *y0         += alphar * x0c + alphai * x1c;
                *(y0 + 1)   += alphai * x0c - alphar * x1c;

                x0 += incx * 2;
                y0 += incy * 2;
            }
        }

    }
}

// -----------------------------------------------------------------------------

void bli_zaxpyv_zen_int_5
     (
             conj_t  conjx,
             dim_t   n,
       const void*   alpha00,
       const void*   x00, inc_t incx,
             void*   y00, inc_t incy,
       const cntx_t* cntx
     )
{

	dcomplex *alpha = (dcomplex *)alpha00;
	dcomplex *x     = (dcomplex *)x00;
	dcomplex *y     = (dcomplex *)y00;

    // If the vector dimension is zero, or if alpha is zero, return early.
    if ( bli_zero_dim1( n ) || PASTEMAC(z,eq0)( *alpha ) )
    {
         return;
    }

    dim_t            i = 0;

    // Initialize local pointers.
    double* x0 = (double*)x;
    double* y0 = (double*)y;

    double alphaR = alpha->real;
    double alphaI = alpha->imag;

    if ( incx == 1 && incy == 1 )
    {
        const dim_t n_elem_per_reg = 4;

        __m256d alphaRv; // for broadcast vector aR (real part of alpha)
        __m256d alphaIv; // for broadcast vector aI (imaginary part of alpha)
        __m256d xv[7]; // Holds the X vector elements
        __m256d xShufv[5]; // Holds the permuted X vector elements
        __m256d yv[7]; // Holds the y vector elements

        // Prefetch distance used in the kernel based on number of cycles
        // In this case, 16 cycles
        const dim_t distance = 16;

        // Prefetch X vector to the L1 cache
        // as these elements will be need anyway
        _mm_prefetch((char const*)x0, _MM_HINT_T1);

        // Broadcast the alpha scalar to all elements of a vector register.
        if (bli_is_noconj(conjx)) // If BLIS_NO_CONJUGATE
        {
            alphaRv = _mm256_broadcast_sd(&alphaR);

            alphaIv[0] = -alphaI;
            alphaIv[1] = alphaI;
            alphaIv[2] = -alphaI;
            alphaIv[3] = alphaI;
        }
        else
        {
            alphaIv = _mm256_broadcast_sd(&alphaI);

            alphaRv[0] = alphaR;
            alphaRv[1] = -alphaR;
            alphaRv[2] = alphaR;
            alphaRv[3] = -alphaR;
        }

        // --------Scalar algorithm BLIS_NO_CONJUGATE arg-------------
        // y = alpha*x + y
        // y = (aR + aIi) * (xR + xIi) + (yR + yIi)
        // y = aR.xR + aR.xIi + aIi.xR - aIxI + (yR + yIi)
        // y = aR.xR - aIxI + yR + aR.xIi + xR.aIi + yIi
        // y = ( aR.xR - aIxI + yR ) + ( aR.xI + aI.xR + yI )i

        // SIMD algorithm
        // xv  = xR1  xI1  xR2  xI2
        // xv' = xI1  xR1  xI2  xR2
        // arv = aR   aR   aR   aR
        // aiv = -aI  aI  -aI   aI
        // yv  = yR1  yI1  yR2  yI2

        // S1 : xv' = xI1  xR1  xI2  xR2 (Shuffle)
        // S2 : reg0 = (aR.xR1  aR.xI1  aR.xR2  aR.xI2) + yv
        // S3 : reg1 = (-aI.xI1  aI.xR1 -aI.xI2  aI.xR2) + reg0
        //----------------------------------------------------------------
        // Ans : aR.xR1 -aI.xI1 + yR1, aR.xI1 + aI.xR1 + yI1, aR.xR2 -aI.xI2 + yR2, aR.xI2 + aI.xR2 + yI2

        //----------Scalar algorithm for BLIS_CONJUGATE arg-------------
        // y = alpha*conj(x) + y
        // y = (aR + aIi) * (xR - xIi) + (yR + yIi)
        // y = aR.xR - aR.xIi + aIi.xR + aIxI + (yR + yIi)
        // y = aR.xR + aIxI + yR - aR.xIi + xR.aIi + yIi
        // y = ( aR.xR + aIxI + yR ) + ( -aR.xI + aI.xR + yI )i
        // y = ( aR.xR + aIxI + yR ) + (aI.xR - aR.xI + yI)i

        // SIMD algorithm
        // xv  = xR1  xI1  xR2  xI2
        // xv' = xI1  xR1  xI2  xR2
        // arv = aR  -aR   aR   -aR
        // aiv = aI   aI   aI   aI
        // yv  = yR1  yI1  yR2  yI2

        // step 1 :  Shuffle xv vector
        // reg xv : xv' = xI1  xR1  xI2  xR2
        // step 2 : fma :yv = ar*xv + yv = ar*xv + yv
        // step 3 : fma :yv = ai*xv' + yv (old)
        //               yv = ai*xv' + ar*xv + yv

        for (i = 0; (i + 13) < n; i += 14)
        {
            // 14 elements will be processed per loop; 14 FMAs will run per loop.

            // alphaRv = aR   aR   aR   aR
            // xv      = xR1  xI1  xR2  xI2
            xv[0] = _mm256_loadu_pd(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm256_loadu_pd(x0 + 1 * n_elem_per_reg);
            xv[2] = _mm256_loadu_pd(x0 + 2 * n_elem_per_reg);
            xv[3] = _mm256_loadu_pd(x0 + 3 * n_elem_per_reg);
            xv[4] = _mm256_loadu_pd(x0 + 4 * n_elem_per_reg);
            xv[5] = _mm256_loadu_pd(x0 + 5 * n_elem_per_reg);
            xv[6] = _mm256_loadu_pd(x0 + 6 * n_elem_per_reg);

            // yv    =  yR1  yI1  yR2  yI2
            yv[0] = _mm256_loadu_pd(y0 + 0 * n_elem_per_reg);
            yv[1] = _mm256_loadu_pd(y0 + 1 * n_elem_per_reg);
            yv[2] = _mm256_loadu_pd(y0 + 2 * n_elem_per_reg);
            yv[3] = _mm256_loadu_pd(y0 + 3 * n_elem_per_reg);
            yv[4] = _mm256_loadu_pd(y0 + 4 * n_elem_per_reg);
            yv[5] = _mm256_loadu_pd(y0 + 5 * n_elem_per_reg);
            yv[6] = _mm256_loadu_pd(y0 + 6 * n_elem_per_reg);

            // yv  =  ar*xv + yv
            //     =  aR.xR1 + yR1, aR.xI1 + yI1, aR.xR2 + yR2, aR.xI2 + yI2, ...
            yv[0] = _mm256_fmadd_pd(xv[0], alphaRv, yv[0]);
            yv[1] = _mm256_fmadd_pd(xv[1], alphaRv, yv[1]);
            yv[2] = _mm256_fmadd_pd(xv[2], alphaRv, yv[2]);
            yv[3] = _mm256_fmadd_pd(xv[3], alphaRv, yv[3]);
            yv[4] = _mm256_fmadd_pd(xv[4], alphaRv, yv[4]);
            yv[5] = _mm256_fmadd_pd(xv[5], alphaRv, yv[5]);
            yv[6] = _mm256_fmadd_pd(xv[6], alphaRv, yv[6]);

            // xv'   =  xI1  xRI  xI2  xR2
            xv[0] = _mm256_permute_pd(xv[0], 5);
            xv[1] = _mm256_permute_pd(xv[1], 5);
            xv[2] = _mm256_permute_pd(xv[2], 5);
            xv[3] = _mm256_permute_pd(xv[3], 5);
            xv[4] = _mm256_permute_pd(xv[4], 5);
            xv[5] = _mm256_permute_pd(xv[5], 5);
            xv[6] = _mm256_permute_pd(xv[6], 5);

            // Prefetch X and Y vectors to the L1 cache
            _mm_prefetch((char const*)(x0 + distance), _MM_HINT_T1);
            _mm_prefetch((char const*)(y0 + distance), _MM_HINT_T1);
            // alphaIv = -aI   aI  -aI   aI

            // yv  =  ar*xv + yv
            //     =  aR.xR1 + yR1, aR.xI1 + yI1, aR.xR2 + yR2, aR.xI2 + yI2, ...
            yv[0] = _mm256_fmadd_pd(xv[0], alphaIv, yv[0]);
            yv[1] = _mm256_fmadd_pd(xv[1], alphaIv, yv[1]);
            yv[2] = _mm256_fmadd_pd(xv[2], alphaIv, yv[2]);
            yv[3] = _mm256_fmadd_pd(xv[3], alphaIv, yv[3]);
            yv[4] = _mm256_fmadd_pd(xv[4], alphaIv, yv[4]);
            yv[5] = _mm256_fmadd_pd(xv[5], alphaIv, yv[5]);
            yv[6] = _mm256_fmadd_pd(xv[6], alphaIv, yv[6]);

            // Store back the result
            _mm256_storeu_pd((y0 + 0 * n_elem_per_reg), yv[0]);
            _mm256_storeu_pd((y0 + 1 * n_elem_per_reg), yv[1]);
            _mm256_storeu_pd((y0 + 2 * n_elem_per_reg), yv[2]);
            _mm256_storeu_pd((y0 + 3 * n_elem_per_reg), yv[3]);
            _mm256_storeu_pd((y0 + 4 * n_elem_per_reg), yv[4]);
            _mm256_storeu_pd((y0 + 5 * n_elem_per_reg), yv[5]);
            _mm256_storeu_pd((y0 + 6 * n_elem_per_reg), yv[6]);

            x0 += 7 * n_elem_per_reg;
            y0 += 7 * n_elem_per_reg;
        }

        for ( ; (i + 9) < n; i += 10 )
        {
            // 10 elements will be processed per loop; 10 FMAs will run per loop.

            // alphaRv = aR   aR   aR   aR
            // xv      = xR1  xI1  xR2  xI2
            xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
            xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );
            xv[2] = _mm256_loadu_pd( x0 + 2*n_elem_per_reg );
            xv[3] = _mm256_loadu_pd( x0 + 3*n_elem_per_reg );
            xv[4] = _mm256_loadu_pd( x0 + 4*n_elem_per_reg );

            // yv    =  yR1  yI1  yR2  yI2
            yv[0] = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
            yv[1] = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );
            yv[2] = _mm256_loadu_pd( y0 + 2*n_elem_per_reg );
            yv[3] = _mm256_loadu_pd( y0 + 3*n_elem_per_reg );
            yv[4] = _mm256_loadu_pd( y0 + 4*n_elem_per_reg );

            // xv'   =  xI1  xRI  xI2  xR2
            xShufv[0] = _mm256_permute_pd( xv[0], 5);
            xShufv[1] = _mm256_permute_pd( xv[1], 5);
            xShufv[2] = _mm256_permute_pd( xv[2], 5);
            xShufv[3] = _mm256_permute_pd( xv[3], 5);
            xShufv[4] = _mm256_permute_pd( xv[4], 5);

            // alphaIv = -aI   aI  -aI   aI

            // yv  =  ar*xv + yv
            //     =  aR.xR1 + yR1, aR.xI1 + yI1, aR.xR2 + yR2, aR.xI2 + yI2, ...
            yv[0] = _mm256_fmadd_pd( xv[0], alphaRv ,yv[0]);
            yv[1] = _mm256_fmadd_pd( xv[1], alphaRv ,yv[1]);
            yv[2] = _mm256_fmadd_pd( xv[2], alphaRv ,yv[2]);
            yv[3] = _mm256_fmadd_pd( xv[3], alphaRv ,yv[3]);
            yv[4] = _mm256_fmadd_pd( xv[4], alphaRv ,yv[4]);

            // yv =  ai*xv' + yv (old)
            // yv =  ai*xv' + ar*xv + yv
            //    = -aI*xI1 + aR.xR1 + yR1, aI.xR1 + aR.xI1 + yI1, .........
            yv[0] = _mm256_fmadd_pd( xShufv[0], alphaIv, yv[0]);
            yv[1] = _mm256_fmadd_pd( xShufv[1], alphaIv, yv[1]);
            yv[2] = _mm256_fmadd_pd( xShufv[2], alphaIv, yv[2]);
            yv[3] = _mm256_fmadd_pd( xShufv[3], alphaIv, yv[3]);
            yv[4] = _mm256_fmadd_pd( xShufv[4], alphaIv, yv[4]);

            // Store back the result
            _mm256_storeu_pd( (y0 + 0*n_elem_per_reg), yv[0] );
            _mm256_storeu_pd( (y0 + 1*n_elem_per_reg), yv[1] );
            _mm256_storeu_pd( (y0 + 2*n_elem_per_reg), yv[2] );
            _mm256_storeu_pd( (y0 + 3*n_elem_per_reg), yv[3] );
            _mm256_storeu_pd( (y0 + 4*n_elem_per_reg), yv[4] );

            x0 += 5*n_elem_per_reg;
            y0 += 5*n_elem_per_reg;
        }

        for (; (i + 5) < n; i += 6)
        {
            // alphaRv = aR   aR   aR   aR
            // xv      = xR1  xI1  xR2  xI2
            xv[0] = _mm256_loadu_pd(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm256_loadu_pd(x0 + 1 * n_elem_per_reg);
            xv[2] = _mm256_loadu_pd(x0 + 2 * n_elem_per_reg);

            // yv    =  yR1  yI1  yR2  yI2
            yv[0] = _mm256_loadu_pd(y0 + 0 * n_elem_per_reg);
            yv[1] = _mm256_loadu_pd(y0 + 1 * n_elem_per_reg);
            yv[2] = _mm256_loadu_pd(y0 + 2 * n_elem_per_reg);

            // xv'   =  xI1  xRI  xI2  xR2
            xShufv[0] = _mm256_permute_pd(xv[0], 5);
            xShufv[1] = _mm256_permute_pd(xv[1], 5);
            xShufv[2] = _mm256_permute_pd(xv[2], 5);

            // alphaIv = -aI   aI  -aI   aI

            // yv  =  ar*xv + yv
            //     =  aR.xR1 + yR1, aR.xI1 + yI1, aR.xR2 + yR2, aR.xI2 + yI2, ...
            yv[0] = _mm256_fmadd_pd(xv[0], alphaRv, yv[0]);
            yv[1] = _mm256_fmadd_pd(xv[1], alphaRv, yv[1]);
            yv[2] = _mm256_fmadd_pd(xv[2], alphaRv, yv[2]);

            // yv =  ai*xv' + yv (old)
            // yv =  ai*xv' + ar*xv + yv
            //    = -aI*xI1 + aR.xR1 + yR1, aI.xR1 + aR.xI1 + yI1, .........
            yv[0] = _mm256_fmadd_pd(xShufv[0], alphaIv, yv[0]);
            yv[1] = _mm256_fmadd_pd(xShufv[1], alphaIv, yv[1]);
            yv[2] = _mm256_fmadd_pd(xShufv[2], alphaIv, yv[2]);

            // Store back the result
            _mm256_storeu_pd((y0 + 0 * n_elem_per_reg), yv[0]);
            _mm256_storeu_pd((y0 + 1 * n_elem_per_reg), yv[1]);
            _mm256_storeu_pd((y0 + 2 * n_elem_per_reg), yv[2]);

            x0 += 3 * n_elem_per_reg;
            y0 += 3 * n_elem_per_reg;
        }

        for ( ; (i + 3) < n; i += 4 )
        {
            // alphaRv = aR   aR   aR   aR
            // xv      = xR1  xI1  xR2  xI2
            xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
            xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );

            // yv    =  yR1  yI1  yR2  yI2
            yv[0] = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
            yv[1] = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );

            // xv'   =  xI1  xRI  xI2  xR2
            xShufv[0] = _mm256_permute_pd( xv[0], 5);
            xShufv[1] = _mm256_permute_pd( xv[1], 5);

            // alphaIv = -aI   aI  -aI   aI

            // yv  =  ar*xv + yv
            //     =  aR.xR1 + yR1, aR.xI1 + yI1, aR.xR2 + yR2, aR.xI2 + yI2, ...
            yv[0] = _mm256_fmadd_pd( xv[0], alphaRv ,yv[0]);
            yv[1] = _mm256_fmadd_pd( xv[1], alphaRv ,yv[1]);

            // yv =  ai*xv' + yv (old)
            // yv =  ai*xv' + ar*xv + yv
            //    = -aI*xI1 + aR.xR1 + yR1, aI.xR1 + aR.xI1 + yI1, .........
            yv[0] = _mm256_fmadd_pd( xShufv[0], alphaIv, yv[0]);
            yv[1] = _mm256_fmadd_pd( xShufv[1], alphaIv, yv[1]);

            // Store back the result
            _mm256_storeu_pd( (y0 + 0*n_elem_per_reg), yv[0] );
            _mm256_storeu_pd( (y0 + 1*n_elem_per_reg), yv[1] );

            x0 += 2*n_elem_per_reg;
            y0 += 2*n_elem_per_reg;
        }

        for (  ; (i + 1) < n; i += 2 )
        {
            // alphaRv = aR   aR   aR   aR
            // xv      = xR1  xI1  xR2  xI2
            xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );

            // yv    =  yR1  yI1  yR2  yI2
            yv[0] = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );

            // xv'   =  xI1  xRI  xI2  xR2
            xShufv[0] = _mm256_permute_pd( xv[0], 5);

            // alphaIv = -aI   aI  -aI   aI

            // yv  =  ar*xv + yv
            //     =  aR.xR1 + yR1, aR.xI1 + yI1, aR.xR2 + yR2, aR.xI2 + yI2, ...
            yv[0] = _mm256_fmadd_pd( xv[0], alphaRv ,yv[0]);

            // yv =  ai*xv' + yv (old)
            // yv =  ai*xv' + ar*xv + yv
            //    = -aI*xI1 + aR.xR1 + yR1, aI.xR1 + aR.xI1 + yI1, .........
            yv[0] = _mm256_fmadd_pd( xShufv[0], alphaIv, yv[0]);

            // Store back the result
            _mm256_storeu_pd( (y0 + 0*n_elem_per_reg), yv[0] );

            x0 += 1*n_elem_per_reg;
            y0 += 1*n_elem_per_reg;
        }

        // Issue vzeroupper instruction to clear upper lanes of ymm registers.
        // This avoids a performance penalty caused by false dependencies when
        // transitioning from AVX to SSE instructions (which may occur as soon
        // as the n_left cleanup loop below if BLIS is compiled with
        // -mfpmath=sse).
        _mm256_zeroupper();
    }

    __m128d alpha_r, alpha_i, x_vec, y_vec;

    // Broadcast the alpha scalar to all elements of a vector register.
    if (bli_is_noconj(conjx)) // If BLIS_NO_CONJUGATE
    {
        alpha_r = _mm_set1_pd(alphaR);

        alpha_i[0] = -alphaI;
        alpha_i[1] = alphaI;
    }
    else
    {
        alpha_i = _mm_set1_pd(alphaI);

        alpha_r[0] = alphaR;
        alpha_r[1] = -alphaR;
    }

    /* This loop has two functions:
        1. Acts as the the fringe case when incx == 1 and incy == 1
        2. Performs the complete computation when incx != 1 or incy != 1
    */
    for (; i < n; ++i)
    {

        x_vec = _mm_loadu_pd(x0);
        y_vec = _mm_loadu_pd(y0);

        y_vec = _mm_fmadd_pd(x_vec, alpha_r, y_vec);
        x_vec = _mm_permute_pd(x_vec, 0b01);
        y_vec = _mm_fmadd_pd(x_vec, alpha_i, y_vec);

        _mm_storeu_pd(y0, y_vec);

        x0 += incx * 2;
        y0 += incy * 2;
    }

}
