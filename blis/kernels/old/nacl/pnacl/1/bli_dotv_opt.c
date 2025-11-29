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

void bli_sdotv_opt(
	conj_t conjx,
	conj_t conjy,
	dim_t  n,
	float  x[restrict static n],
	inc_t  incx,
	float  y[restrict static n],
	inc_t  incy,
	float  rho[restrict static 1])
{
#if PPAPI_RELEASE >= 36
	// If the vector lengths are zero, set rho to zero and return.
	if (bli_zero_dim1(n)) {
		*rho = 0.0f;
		return;
	}
	
	// If there is anything that would interfere with our use of aligned
	// vector loads/stores, call the reference implementation.
	if (bli_has_nonunit_inc2(incx, incy)) {
		float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f, sum5 = 0.0f;
		while (n >= 6) {
			sum0 += (*x) * (*y);
			x += incx;
			y += incy;

			sum1 += (*x) * (*y);
			x += incx;
			y += incy;

			sum2 += (*x) * (*y);
			x += incx;
			y += incy;

			sum3 += (*x) * (*y);
			x += incx;
			y += incy;

			sum4 += (*x) * (*y);
			x += incx;
			y += incy;

			sum5 += (*x) * (*y);
			x += incx;
			y += incy;

			n -= 6;
		}
		float sum = (sum0 + sum1 + sum2) + (sum3 + sum4 + sum5);
		while (n--) {
			sum += (*x) * (*y);
			x += incx;
			y += incy;
		}
		*rho = sum;
	} else {
		v4sf vsum0 = v4sf_zero(), vsum1 = v4sf_zero(), vsum2 = v4sf_zero();
		v4sf vsum3 = v4sf_zero(), vsum4 = v4sf_zero(), vsum5 = v4sf_zero();
		while (n >= 24) {
			vsum0 += v4sf_load(x) * v4sf_load(y);
			vsum1 += v4sf_load(x+4) * v4sf_load(y+4);
			vsum2 += v4sf_load(x+8) * v4sf_load(y+8);
			vsum3 += v4sf_load(x+12) * v4sf_load(y+12);
			vsum4 += v4sf_load(x+16) * v4sf_load(y+16);
			vsum5 += v4sf_load(x+20) * v4sf_load(y+20);

			x += 24;
			y += 24;
			n -= 24;
		}
		v4sf vsum = (vsum0 + vsum1 + vsum2) + (vsum3 + vsum4 + vsum5);
		while (n >= 4) {
			vsum += v4sf_load(x) * v4sf_load(y);

			x += 4;
			y += 4;
			n -= 4;
		}
		float sum = (vsum[0] + vsum[1]) + (vsum[2] + vsum[3]);
		while (n--) {
			sum += (*x++) * (*y++);
		}
		*rho = sum;
	}
#else
	float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f, sum5 = 0.0f;
	while (n >= 6) {
		sum0 += (*x) * (*y);
		x += incx;
		y += incy;

		sum1 += (*x) * (*y);
		x += incx;
		y += incy;

		sum2 += (*x) * (*y);
		x += incx;
		y += incy;

		sum3 += (*x) * (*y);
		x += incx;
		y += incy;

		sum4 += (*x) * (*y);
		x += incx;
		y += incy;

		sum5 += (*x) * (*y);
		x += incx;
		y += incy;

		n -= 6;
	}
	float sum = (sum0 + sum1 + sum2) + (sum3 + sum4 + sum5);
	while (n--) {
		sum += (*x) * (*y);
		x += incx;
		y += incy;
	}
	*rho = sum;
#endif
}

void bli_ddotv_opt(
	conj_t conjx,
	conj_t conjy,
	dim_t  n,
	double x[restrict static n],
	inc_t  incx,
	double y[restrict static n],
	inc_t  incy,
	double rho[restrict static 1])
{
	double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0, sum5 = 0.0;
	while (n >= 6) {
		sum0 += (*x) * (*y);
		x += incx;
		y += incy;

		sum1 += (*x) * (*y);
		x += incx;
		y += incy;

		sum2 += (*x) * (*y);
		x += incx;
		y += incy;

		sum3 += (*x) * (*y);
		x += incx;
		y += incy;

		sum4 += (*x) * (*y);
		x += incx;
		y += incy;

		sum5 += (*x) * (*y);
		x += incx;
		y += incy;

		n -= 6;
	}
	double sum = (sum0 + sum1 + sum2) + (sum3 + sum4 + sum5);
	while (n--) {
		sum += (*x) * (*y);
		x += incx;
		y += incy;
	}
	*rho = sum;
}

void bli_cdotv_opt(
	conj_t   conjx,
	conj_t   conjy,
	dim_t    n,
	scomplex x[restrict static n],
	inc_t    incx,
	scomplex y[restrict static n],
	inc_t    incy,
	scomplex rho[restrict static 1])
{
	if (bli_is_conj(conjy)) {
		bli_toggle_conj(&conjx);
	}

	if (bli_zero_dim1(n)) {
		rho->real = 0.0f;
		rho->imag = 0.0f;
		return;
	}

	float sumr;
	float sumi;
#if PPAPI_RELEASE >= 36
	if (bli_is_noconj(conjx)) {
		if (bli_has_nonunit_inc2(incx, incy)) {
			float sum0r = 0.0f, sum1r = 0.0f;
			float sum0i = 0.0f, sum1i = 0.0f;
			while (n >= 2) {
				const float x0r = x->real;
				const float x0i = x->imag;
				const float y0r = y->real;
				const float y0i = y->imag;

				sum0r += x0r * y0r - x0i * y0i;
				sum0i += x0r * y0i + x0i * y0r;

				x += incx;
				y += incy;

				const float x1r = x->real;
				const float x1i = x->imag;
				const float y1r = y->real;
				const float y1i = y->imag;

				sum1r += x1r * y1r - x1i * y1i;
				sum1i += x1r * y1i + x1i * y1r;

				x += incx;
				y += incy;

				n -= 2;
			}
			sumr = sum0r + sum1r;
			sumi = sum0i + sum1i;
		} else {
			v4sf sumv0r = v4sf_zero(), sumv1r = v4sf_zero();
			v4sf sumv0i = v4sf_zero(), sumv1i = v4sf_zero();
			while (n >= 8) {
				const v4sf xv0t = v4sf_cload(x);
				const v4sf xv0b = v4sf_cload(x+2);
				const v4sf yv0t = v4sf_cload(y);
				const v4sf yv0b = v4sf_cload(y+2);

				const v4sf xv0r = __builtin_shufflevector(xv0t, xv0b, 0, 2, 4, 6);
				const v4sf xv0i = __builtin_shufflevector(xv0t, xv0b, 1, 3, 5, 7);
				const v4sf yv0r = __builtin_shufflevector(yv0t, yv0b, 0, 2, 4, 6);
				const v4sf yv0i = __builtin_shufflevector(yv0t, yv0b, 1, 3, 5, 7);

				sumv0r += xv0r * yv0r - xv0i * yv0i;
				sumv0i += xv0r * yv0i + xv0i * yv0r;

				const v4sf xv1t = v4sf_cload(x+4);
				const v4sf xv1b = v4sf_cload(x+6);
				const v4sf yv1t = v4sf_cload(y+4);
				const v4sf yv1b = v4sf_cload(y+6);

				const v4sf xv1r = __builtin_shufflevector(xv1t, xv1b, 0, 2, 4, 6);
				const v4sf xv1i = __builtin_shufflevector(xv1t, xv1b, 1, 3, 5, 7);
				const v4sf yv1r = __builtin_shufflevector(yv1t, yv1b, 0, 2, 4, 6);
				const v4sf yv1i = __builtin_shufflevector(yv1t, yv1b, 1, 3, 5, 7);

				sumv1r += xv1r * yv1r - xv1i * yv1i;
				sumv1i += xv1r * yv1i + xv1i * yv1r;

				x += 8;
				y += 8;

				n -= 8;
			}
			const v4sf sumvr = sumv0r + sumv1r;
			const v4sf sumvi = sumv0i + sumv1i;
			sumr = (sumvr[0] + sumvr[1]) + (sumvr[2] + sumvr[3]);
			sumi = (sumvi[0] + sumvi[1]) + (sumvi[2] + sumvi[3]);
		}
		while (n--) {
			const float xr = x->real;
			const float xi = x->imag;
			const float yr = y->real;
			const float yi = y->imag;

			sumr += xr * yr - xi * yi;
			sumi += xr * yi + xi * yr;

			x += incx;
			y += incy;
		}
	} else {
		if (bli_has_nonunit_inc2(incx, incy)) {
			float sum0r = 0.0f, sum1r = 0.0f;
			float sum0i = 0.0f, sum1i = 0.0f;
			while (n >= 2) {
				const float x0r = x->real;
				const float x0i = x->imag;
				const float y0r = y->real;
				const float y0i = y->imag;

				sum0r += x0r * y0r + x0i * y0i;
				sum0i += x0r * y0i - x0i * y0r;

				x += incx;
				y += incy;

				const float x1r = x->real;
				const float x1i = x->imag;
				const float y1r = y->real;
				const float y1i = y->imag;

				sum1r += x1r * y1r + x1i * y1i;
				sum1i += x1r * y1i - x1i * y1r;

				x += incx;
				y += incy;

				n -= 2;
			}
			sumr = sum0r + sum1r;
			sumi = sum0i + sum1i;
		} else {
			v4sf sumv0r = v4sf_zero(), sumv1r = v4sf_zero();
			v4sf sumv0i = v4sf_zero(), sumv1i = v4sf_zero();
			while (n >= 8) {
				const v4sf xv0t = v4sf_cload(x);
				const v4sf xv0b = v4sf_cload(x+2);
				const v4sf yv0t = v4sf_cload(y);
				const v4sf yv0b = v4sf_cload(y+2);

				const v4sf xv0r = __builtin_shufflevector(xv0t, xv0b, 0, 2, 4, 6);
				const v4sf xv0i = __builtin_shufflevector(xv0t, xv0b, 1, 3, 5, 7);
				const v4sf yv0r = __builtin_shufflevector(yv0t, yv0b, 0, 2, 4, 6);
				const v4sf yv0i = __builtin_shufflevector(yv0t, yv0b, 1, 3, 5, 7);

				sumv0r += xv0r * yv0r + xv0i * yv0i;
				sumv0i += xv0r * yv0i - xv0i * yv0r;

				const v4sf xv1t = v4sf_cload(x+4);
				const v4sf xv1b = v4sf_cload(x+6);
				const v4sf yv1t = v4sf_cload(y+4);
				const v4sf yv1b = v4sf_cload(y+6);

				const v4sf xv1r = __builtin_shufflevector(xv1t, xv1b, 0, 2, 4, 6);
				const v4sf xv1i = __builtin_shufflevector(xv1t, xv1b, 1, 3, 5, 7);
				const v4sf yv1r = __builtin_shufflevector(yv1t, yv1b, 0, 2, 4, 6);
				const v4sf yv1i = __builtin_shufflevector(yv1t, yv1b, 1, 3, 5, 7);

				sumv1r += xv1r * yv1r + xv1i * yv1i;
				sumv1i += xv1r * yv1i - xv1i * yv1r;

				x += 8;
				y += 8;

				n -= 8;
			}
			const v4sf sumvr = sumv0r + sumv1r;
			const v4sf sumvi = sumv0i + sumv1i;
			sumr = (sumvr[0] + sumvr[1]) + (sumvr[2] + sumvr[3]);
			sumi = (sumvi[0] + sumvi[1]) + (sumvi[2] + sumvi[3]);
		}
		while (n--) {
			const float xr = x->real;
			const float xi = x->imag;
			const float yr = y->real;
			const float yi = y->imag;

			sumr += xr * yr + xi * yi;
			sumi += xr * yi - xi * yr;

			x += incx;
			y += incy;
		}
	}
#else
	if (bli_is_noconj(conjx)) {
		float sum0r = 0.0f, sum1r = 0.0f;
		float sum0i = 0.0f, sum1i = 0.0f;
		while (n >= 2) {
			const float x0r = x->real;
			const float x0i = x->imag;
			const float y0r = y->real;
			const float y0i = y->imag;

			sum0r += x0r * y0r - x0i * y0i;
			sum0i += x0r * y0i + x0i * y0r;

			x += incx;
			y += incy;

			const float x1r = x->real;
			const float x1i = x->imag;
			const float y1r = y->real;
			const float y1i = y->imag;

			sum1r += x1r * y1r - x1i * y1i;
			sum1i += x1r * y1i + x1i * y1r;

			x += incx;
			y += incy;

			n -= 2;
		}
		sumr = sum0r + sum1r;
		sumi = sum0i + sum1i;
		if (n != 0) {
			const float xr = x->real;
			const float xi = x->imag;
			const float yr = y->real;
			const float yi = y->imag;

			sumr += xr * yr - xi * yi;
			sumi += xr * yi + xi * yr;
		}
	} else {
		float sum0r = 0.0f, sum1r = 0.0f;
		float sum0i = 0.0f, sum1i = 0.0f;
		while (n >= 2) {
			const float x0r = x->real;
			const float x0i = x->imag;
			const float y0r = y->real;
			const float y0i = y->imag;

			sum0r += x0r * y0r + x0i * y0i;
			sum0i += x0r * y0i - x0i * y0r;

			x += incx;
			y += incy;

			const float x1r = x->real;
			const float x1i = x->imag;
			const float y1r = y->real;
			const float y1i = y->imag;

			sum1r += x1r * y1r + x1i * y1i;
			sum1i += x1r * y1i - x1i * y1r;

			x += incx;
			y += incy;

			n -= 2;
		}
		sumr = sum0r + sum1r;
		sumi = sum0i + sum1i;
		if (n != 0) {
			const float xr = x->real;
			const float xi = x->imag;
			const float yr = y->real;
			const float yi = y->imag;

			sumr += xr * yr + xi * yi;
			sumi += xr * yi - xi * yr;
		}
	}
#endif

	rho->real = sumr;
	rho->imag = bli_is_conj(conjy) ? -sumi : sumi;
}



void bli_zdotv_opt(
	conj_t   conjx,
	conj_t   conjy,
	dim_t    n,
	dcomplex x[restrict static n],
	inc_t    incx,
	dcomplex y[restrict static n],
	inc_t    incy,
	dcomplex rho[restrict static 1])
{
	if (bli_is_conj(conjy)) {
		bli_toggle_conj(&conjx);
	}

	if (bli_zero_dim1(n)) {
		rho->real = 0.0;
		rho->imag = 0.0;
		return;
	}

	double sumr;
	double sumi;
	if (bli_is_noconj(conjx)) {
		double sum0r = 0.0, sum1r = 0.0;
		double sum0i = 0.0, sum1i = 0.0;
		while (n >= 2) {
			const double x0r = x->real;
			const double x0i = x->imag;
			const double y0r = y->real;
			const double y0i = y->imag;

			sum0r += x0r * y0r - x0i * y0i;
			sum0i += x0r * y0i + x0i * y0r;

			x += incx;
			y += incy;

			const double x1r = x->real;
			const double x1i = x->imag;
			const double y1r = y->real;
			const double y1i = y->imag;

			sum1r += x1r * y1r - x1i * y1i;
			sum1i += x1r * y1i + x1i * y1r;

			x += incx;
			y += incy;

			n -= 2;
		}
		sumr = sum0r + sum1r;
		sumi = sum0i + sum1i;
		if (n != 0) {
			const double xr = x->real;
			const double xi = x->imag;
			const double yr = y->real;
			const double yi = y->imag;

			sumr += xr * yr - xi * yi;
			sumi += xr * yi + xi * yr;
		}
	} else {
		double sum0r = 0.0, sum1r = 0.0;
		double sum0i = 0.0, sum1i = 0.0;
		while (n >= 2) {
			const double x0r = x->real;
			const double x0i = x->imag;
			const double y0r = y->real;
			const double y0i = y->imag;

			sum0r += x0r * y0r + x0i * y0i;
			sum0i += x0r * y0i - x0i * y0r;

			x += incx;
			y += incy;

			const double x1r = x->real;
			const double x1i = x->imag;
			const double y1r = y->real;
			const double y1i = y->imag;

			sum1r += x1r * y1r + x1i * y1i;
			sum1i += x1r * y1i - x1i * y1r;

			x += incx;
			y += incy;

			n -= 2;
		}
		sumr = sum0r + sum1r;
		sumi = sum0i + sum1i;
		if (n != 0) {
			const double xr = x->real;
			const double xi = x->imag;
			const double yr = y->real;
			const double yi = y->imag;

			sumr += xr * yr + xi * yi;
			sumi += xr * yi - xi * yr;
		}
	}

	rho->real = sumr;
	rho->imag = bli_is_conj(conjy) ? -sumi : sumi;
}

