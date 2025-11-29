/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2019 - 2020, Advanced Micro Devices, Inc.

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

// -----------------------------------------------------------------------------

void bli_scopyv_zen_int
     (
             conj_t  conjx,
             dim_t   n,
       const void*   x0, inc_t incx,
             void*   y0, inc_t incy,
       const cntx_t* cntx
     )
{
	const float* x = x0;
	      float* y = y0;

	const dim_t num_elem_per_reg = 8;
	dim_t       i = 0;
	__m256      xv[16];

	// If the vector dimension is zero return early.
	if ( bli_zero_dim1( n ) ) return;

	if ( incx == 1 && incy == 1 )
	{
#if 0
	PRAGMA_SIMD
	for (i = 0; i < n; i++)
	{
		y[i] = x[i];
	}
#endif
#if 0
	memcpy(y, x, n << 2);
#endif
#if 1

		// For loop with n & ~0x7F => n & 0xFFFFFF80 masks the lower bits and results in multiples of 128
		// for example if n = 255
		// n & ~0x7F results in 128: copy from 0 to 128 happens in first loop
		// n & ~0x3F results in 192: copy from 128 to 192 happens in second loop
		// n & ~0x1F results in 224: copy from 128 to 192 happens in third loop and so on.
		for ( i = 0; i < (n & (~0x7F)); i += 128 )
		{
			xv[0] = _mm256_loadu_ps(x + num_elem_per_reg * 0);
			xv[1] = _mm256_loadu_ps(x + num_elem_per_reg * 1);
			xv[2] = _mm256_loadu_ps(x + num_elem_per_reg * 2);
			xv[3] = _mm256_loadu_ps(x + num_elem_per_reg * 3);
			xv[4] = _mm256_loadu_ps(x + num_elem_per_reg * 4);
			xv[5] = _mm256_loadu_ps(x + num_elem_per_reg * 5);
			xv[6] = _mm256_loadu_ps(x + num_elem_per_reg * 6);
			xv[7] = _mm256_loadu_ps(x + num_elem_per_reg * 7);
			xv[8] = _mm256_loadu_ps(x + num_elem_per_reg * 8);
			xv[9] = _mm256_loadu_ps(x + num_elem_per_reg * 9);
			xv[10] = _mm256_loadu_ps(x + num_elem_per_reg * 10);
			xv[11] = _mm256_loadu_ps(x + num_elem_per_reg * 11);
			xv[12] = _mm256_loadu_ps(x + num_elem_per_reg * 12);
			xv[13] = _mm256_loadu_ps(x + num_elem_per_reg * 13);
			xv[14] = _mm256_loadu_ps(x + num_elem_per_reg * 14);
			xv[15] = _mm256_loadu_ps(x + num_elem_per_reg * 15);

			_mm256_storeu_ps(y + num_elem_per_reg * 0, xv[0]);
			_mm256_storeu_ps(y + num_elem_per_reg * 1, xv[1]);
			_mm256_storeu_ps(y + num_elem_per_reg * 2, xv[2]);
			_mm256_storeu_ps(y + num_elem_per_reg * 3, xv[3]);
			_mm256_storeu_ps(y + num_elem_per_reg * 4, xv[4]);
			_mm256_storeu_ps(y + num_elem_per_reg * 5, xv[5]);
			_mm256_storeu_ps(y + num_elem_per_reg * 6, xv[6]);
			_mm256_storeu_ps(y + num_elem_per_reg * 7, xv[7]);
			_mm256_storeu_ps(y + num_elem_per_reg * 8, xv[8]);
			_mm256_storeu_ps(y + num_elem_per_reg * 9, xv[9]);
			_mm256_storeu_ps(y + num_elem_per_reg * 10, xv[10]);
			_mm256_storeu_ps(y + num_elem_per_reg * 11, xv[11]);
			_mm256_storeu_ps(y + num_elem_per_reg * 12, xv[12]);
			_mm256_storeu_ps(y + num_elem_per_reg * 13, xv[13]);
			_mm256_storeu_ps(y + num_elem_per_reg * 14, xv[14]);
			_mm256_storeu_ps(y + num_elem_per_reg * 15, xv[15]);

			y += 128;
			x += 128;
		}
		for ( ; i < (n & (~0x3F)); i += 64 )
		{
			xv[0] = _mm256_loadu_ps(x + num_elem_per_reg * 0);
			xv[1] = _mm256_loadu_ps(x + num_elem_per_reg * 1);
			xv[2] = _mm256_loadu_ps(x + num_elem_per_reg * 2);
			xv[3] = _mm256_loadu_ps(x + num_elem_per_reg * 3);
			xv[4] = _mm256_loadu_ps(x + num_elem_per_reg * 4);
			xv[5] = _mm256_loadu_ps(x + num_elem_per_reg * 5);
			xv[6] = _mm256_loadu_ps(x + num_elem_per_reg * 6);
			xv[7] = _mm256_loadu_ps(x + num_elem_per_reg * 7);

			_mm256_storeu_ps(y + num_elem_per_reg * 0, xv[0]);
			_mm256_storeu_ps(y + num_elem_per_reg * 1, xv[1]);
			_mm256_storeu_ps(y + num_elem_per_reg * 2, xv[2]);
			_mm256_storeu_ps(y + num_elem_per_reg * 3, xv[3]);
			_mm256_storeu_ps(y + num_elem_per_reg * 4, xv[4]);
			_mm256_storeu_ps(y + num_elem_per_reg * 5, xv[5]);
			_mm256_storeu_ps(y + num_elem_per_reg * 6, xv[6]);
			_mm256_storeu_ps(y + num_elem_per_reg * 7, xv[7]);

			y += 64;
			x += 64;
		}
		for ( ; i < (n & (~0x1F)); i += 32 )
		{
			xv[0] = _mm256_loadu_ps(x + num_elem_per_reg * 0);
			xv[1] = _mm256_loadu_ps(x + num_elem_per_reg * 1);
			xv[2] = _mm256_loadu_ps(x + num_elem_per_reg * 2);
			xv[3] = _mm256_loadu_ps(x + num_elem_per_reg * 3);

			_mm256_storeu_ps(y + num_elem_per_reg * 0, xv[0]);
			_mm256_storeu_ps(y + num_elem_per_reg * 1, xv[1]);
			_mm256_storeu_ps(y + num_elem_per_reg * 2, xv[2]);
			_mm256_storeu_ps(y + num_elem_per_reg * 3, xv[3]);

			y += 32;
			x += 32;
		}
		for ( ; i < (n & (~0x0F)); i += 16 )
		{
			xv[0] = _mm256_loadu_ps(x + num_elem_per_reg * 0);
			xv[1] = _mm256_loadu_ps(x + num_elem_per_reg * 1);

			_mm256_storeu_ps(y + num_elem_per_reg * 0, xv[0]);
			_mm256_storeu_ps(y + num_elem_per_reg * 1, xv[1]);

			y += 16;
			x += 16;
		}
		for ( ; i < (n & (~0x07)); i += 8 )
		{
			xv[0] = _mm256_loadu_ps(x + num_elem_per_reg * 0);
			_mm256_storeu_ps(y + num_elem_per_reg * 0, xv[0]);
			y += 8;
			x += 8;
		}
		for ( ; i < n; ++i )
		{
			*y++ = *x++;
		}
#endif
	}
	else
	{
		for ( dim_t i = 0; i < n; ++i )
		{
			*y = *x;
			x += incx;
			y += incy;
		}
	}
}

// -----------------------------------------------------------------------------

void bli_dcopyv_zen_int
     (
             conj_t  conjx,
             dim_t   n,
       const void*   x0, inc_t incx,
             void*   y0, inc_t incy,
       const cntx_t* cntx
     )
{
	const double* x = x0;
	      double* y = y0;

	const dim_t num_elem_per_reg = 4;
	dim_t       i = 0;
	__m256d     xv[16];

	// If the vector dimension is zero return early.
	if ( bli_zero_dim1( n ) ) return;

	if ( incx == 1 && incy == 1 )
	{
#if 0
	PRAGMA_SIMD
	for (i = 0; i < n; ++i)
	{
		y[i] = x[i];
	}
#endif
#if 0
	memcpy(y, x, n << 3);
#endif
#if 1
		// n & (~0x3F) = n & 0xFFFFFFC0 -> this masks the numbers less than 64,
		// the copy operation will be done for the multiples of 64
		for ( i = 0; i < (n & (~0x3F)); i += 64 )
		{
			xv[0] = _mm256_loadu_pd(x + num_elem_per_reg * 0);
			xv[1] = _mm256_loadu_pd(x + num_elem_per_reg * 1);
			xv[2] = _mm256_loadu_pd(x + num_elem_per_reg * 2);
			xv[3] = _mm256_loadu_pd(x + num_elem_per_reg * 3);
			xv[4] = _mm256_loadu_pd(x + num_elem_per_reg * 4);
			xv[5] = _mm256_loadu_pd(x + num_elem_per_reg * 5);
			xv[6] = _mm256_loadu_pd(x + num_elem_per_reg * 6);
			xv[7] = _mm256_loadu_pd(x + num_elem_per_reg * 7);
			xv[8] = _mm256_loadu_pd(x + num_elem_per_reg * 8);
			xv[9] = _mm256_loadu_pd(x + num_elem_per_reg * 9);
			xv[10] = _mm256_loadu_pd(x + num_elem_per_reg * 10);
			xv[11] = _mm256_loadu_pd(x + num_elem_per_reg * 11);
			xv[12] = _mm256_loadu_pd(x + num_elem_per_reg * 12);
			xv[13] = _mm256_loadu_pd(x + num_elem_per_reg * 13);
			xv[14] = _mm256_loadu_pd(x + num_elem_per_reg * 14);
			xv[15] = _mm256_loadu_pd(x + num_elem_per_reg * 15);
			_mm256_storeu_pd(y + num_elem_per_reg * 0, xv[0]);
			_mm256_storeu_pd(y + num_elem_per_reg * 1, xv[1]);
			_mm256_storeu_pd(y + num_elem_per_reg * 2, xv[2]);
			_mm256_storeu_pd(y + num_elem_per_reg * 3, xv[3]);
			_mm256_storeu_pd(y + num_elem_per_reg * 4, xv[4]);
			_mm256_storeu_pd(y + num_elem_per_reg * 5, xv[5]);
			_mm256_storeu_pd(y + num_elem_per_reg * 6, xv[6]);
			_mm256_storeu_pd(y + num_elem_per_reg * 7, xv[7]);
			_mm256_storeu_pd(y + num_elem_per_reg * 8, xv[8]);
			_mm256_storeu_pd(y + num_elem_per_reg * 9, xv[9]);
			_mm256_storeu_pd(y + num_elem_per_reg * 10, xv[10]);
			_mm256_storeu_pd(y + num_elem_per_reg * 11, xv[11]);
			_mm256_storeu_pd(y + num_elem_per_reg * 12, xv[12]);
			_mm256_storeu_pd(y + num_elem_per_reg * 13, xv[13]);
			_mm256_storeu_pd(y + num_elem_per_reg * 14, xv[14]);
			_mm256_storeu_pd(y + num_elem_per_reg * 15, xv[15]);
			y += num_elem_per_reg * 16;
			x += num_elem_per_reg * 16;
		}
		for ( ; i < (n & (~0x1F)); i += 32 )
		{
			xv[0] = _mm256_loadu_pd(x + num_elem_per_reg * 0);
			xv[1] = _mm256_loadu_pd(x + num_elem_per_reg * 1);
			xv[2] = _mm256_loadu_pd(x + num_elem_per_reg * 2);
			xv[3] = _mm256_loadu_pd(x + num_elem_per_reg * 3);
			xv[4] = _mm256_loadu_pd(x + num_elem_per_reg * 4);
			xv[5] = _mm256_loadu_pd(x + num_elem_per_reg * 5);
			xv[6] = _mm256_loadu_pd(x + num_elem_per_reg * 6);
			xv[7] = _mm256_loadu_pd(x + num_elem_per_reg * 7);

			_mm256_storeu_pd(y + num_elem_per_reg * 0, xv[0]);
			_mm256_storeu_pd(y + num_elem_per_reg * 1, xv[1]);
			_mm256_storeu_pd(y + num_elem_per_reg * 2, xv[2]);
			_mm256_storeu_pd(y + num_elem_per_reg * 3, xv[3]);
			_mm256_storeu_pd(y + num_elem_per_reg * 4, xv[4]);
			_mm256_storeu_pd(y + num_elem_per_reg * 5, xv[5]);
			_mm256_storeu_pd(y + num_elem_per_reg * 6, xv[6]);
			_mm256_storeu_pd(y + num_elem_per_reg * 7, xv[7]);

			y += num_elem_per_reg * 8;
			x += num_elem_per_reg * 8;
		}
		for ( ; i < (n & (~0xF)); i += 16 )
		{
			xv[0] = _mm256_loadu_pd(x + num_elem_per_reg * 0);
			xv[1] = _mm256_loadu_pd(x + num_elem_per_reg * 1);
			xv[2] = _mm256_loadu_pd(x + num_elem_per_reg * 2);
			xv[3] = _mm256_loadu_pd(x + num_elem_per_reg * 3);

			_mm256_storeu_pd(y + num_elem_per_reg * 0, xv[0]);
			_mm256_storeu_pd(y + num_elem_per_reg * 1, xv[1]);
			_mm256_storeu_pd(y + num_elem_per_reg * 2, xv[2]);
			_mm256_storeu_pd(y + num_elem_per_reg * 3, xv[3]);

			y += num_elem_per_reg * 4;
			x += num_elem_per_reg * 4;
		}
		for ( ; i < (n & (~0x07)); i += 8 )
		{
			xv[0] = _mm256_loadu_pd(x + num_elem_per_reg * 0);
			xv[1] = _mm256_loadu_pd(x + num_elem_per_reg * 1);

			_mm256_storeu_pd(y + num_elem_per_reg * 0, xv[0]);
			_mm256_storeu_pd(y + num_elem_per_reg * 1, xv[1]);

			y += num_elem_per_reg * 2;
			x += num_elem_per_reg * 2;
		}
		for ( ; i < (n & (~0x03)); i += 4 )
		{
			xv[0] = _mm256_loadu_pd(x + num_elem_per_reg * 0);
			_mm256_storeu_pd(y + num_elem_per_reg * 0, xv[0]);
			y += num_elem_per_reg;
			x += num_elem_per_reg;
		}
		for ( ; i < n; ++i )
		{
			*y++ = *x++;
		}
#endif
	}
	else
	{
		for ( i = 0; i < n; ++i )
		{
			*y = *x;

			x += incx;
			y += incy;
		}
	}
}

