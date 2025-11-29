/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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

#include "blis.h"

#if PPAPI_RELEASE >= 36
typedef float v4sf __attribute__ ((vector_size(16)));

inline v4sf v4sf_splat(float x) {
        return (v4sf) { x, x, x, x };
}

inline v4sf v4sf_load(const float* a) {
        return *((const v4sf*)a);
}

inline v4sf v4sf_cload(const scomplex* a) {
        return *((const v4sf*)a);
}

inline void v4sf_store(float* a, v4sf x) {
        *((v4sf*)a) = x;
}

inline void v4sf_cstore(scomplex* a, v4sf x) {
        *((v4sf*)a) = x;
}

inline v4sf v4sf_zero() {
        return (v4sf) { 0.0f, 0.0f, 0.0f, 0.0f };
}
#endif


void bli_saxpyv_opt(
	conj_t conjx,
	dim_t  n,
	float  alpha[restrict static 1],
	float  x[restrict static n],
	inc_t  incx,
	float  y[restrict static n],
	inc_t  incy)
{
	if (bli_zero_dim1(n)) {
		return;
	}

	if (bli_seq0(*alpha)) {
		return;
	}

#if PPAPI_RELEASE >= 36
	if (!bli_has_nonunit_inc2(incx, incy)) {
		const v4sf alphav = v4sf_splat(*alpha);
		while (n >= 4) {
			const v4sf xv = v4sf_load(x);
			v4sf yv = v4sf_load(y);
			yv += xv * alphav;
			v4sf_store(y, yv);

			x += 4;
			y += 4;
			n -= 4;
		}
		const float alphac = *alpha;
		while (n--) {
			(*y++) += (*x++) * alphac;
		}
	}
#endif
	/* Just call the reference implementation. */
	BLIS_SAXPYV_KERNEL_REF(
		conjx,
		n,
		alpha,
		x,
		incx,
		y,
		incy);
}


void bli_caxpyv_opt(
	conj_t   conjx,
	dim_t    n,
	scomplex alpha[restrict static 1],
	scomplex x[restrict static n],
	inc_t    incx,
	scomplex y[restrict static n],
	inc_t    incy)
{
	if (bli_zero_dim1(n)) {
		return;
	}

	if (bli_ceq0(*alpha)) {
		return;
	}

#if PPAPI_RELEASE >= 36
	if (!bli_has_nonunit_inc2(incx, incy)) {
		if (bli_is_noconj(conjx)) {
			const v4sf alphav0 = v4sf_splat(alpha->real);
			const v4sf alphav1 = (v4sf) { -alpha->imag, alpha->imag, -alpha->imag, alpha->imag };
			while (n >= 2) {
				const v4sf xv0 = v4sf_cload(x);
				v4sf yv = v4sf_cload(y);
				const v4sf xv1 = __builtin_shufflevector(xv0, xv0, 1, 0, 3, 2);
				yv += xv0 * alphav0 + xv1 * alphav1;
				v4sf_cstore(y, yv);

				x += 2;
				y += 2;
				n -= 2;
			}
			const float alphar = alpha->real;
			const float alphai = alpha->imag;
			while (n--) {
				const float xr = x->real;
				const float xi = x->imag;
				const float yr = y->real;
				const float yi = y->imag;

				y->real = yr + xr * alphar - xi * alphai;
				y->imag = yi + xr * alphai + xi * alphar; 

				x += 1;
				y += 1;
			}
		} else {
			const v4sf alphav0 = (v4sf) { alpha->real, -alpha->real, alpha->real, -alpha->real };
			const v4sf alphav1 = v4sf_splat(alpha->imag);
			while (n >= 2) {
				const v4sf xv0 = v4sf_cload(x);
				v4sf yv = v4sf_cload(y);
				const v4sf xv1 = __builtin_shufflevector(xv0, xv0, 1, 0, 3, 2);
				yv += xv0 * alphav0 + xv1 * alphav1;
				v4sf_cstore(y, yv);

				x += 2;
				y += 2;
				n -= 2;
			}
			const float alphar = alpha->real;
			const float alphai = alpha->imag;
			while (n--) {
				const float xr = x->real;
				const float xi = x->imag;
				const float yr = y->real;
				const float yi = y->imag;

				y->real = yr + xr * alphar + xi * alphai;
				y->imag = yi + xr * alphai - xi * alphar; 

				x += 1;
				y += 1;
			}
		}
	}
#endif


	/* Just call the reference implementation. */
	BLIS_CAXPYV_KERNEL_REF(
		conjx,
		n,
		alpha,
		x,
		incx,
		y,
		incy);
}
