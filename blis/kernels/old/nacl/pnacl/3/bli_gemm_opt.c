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

	void bli_sgemm_opt
	     (
		   dim_t      k,
		   float      alpha[restrict static 1],
		   float      a[restrict static 8*k],
		   float      b[restrict static k*4],
		   float      beta[restrict static 1],
		   float      c[restrict static 8*4],
		   inc_t      rs_c,
		   inc_t      cs_c,
		   auxinfo_t* data,
		   cntx_t*    cntx
	     )
	{
		// Vectors for accummulating column 0, 1, 2, 3 (initialize to 0.0)
		v4sf abv0t = v4sf_zero(), abv1t = v4sf_zero(), abv2t = v4sf_zero(), abv3t = v4sf_zero();
		v4sf abv0b = v4sf_zero(), abv1b = v4sf_zero(), abv2b = v4sf_zero(), abv3b = v4sf_zero();
		for (dim_t i = 0; i < k; i += 1) {
			const v4sf avt = v4sf_load(a);
			const v4sf avb = v4sf_load(a+4);

			const v4sf bv_xxxx = v4sf_splat(b[0]);
			abv0t += avt * bv_xxxx;
			abv0b += avb * bv_xxxx;

			const v4sf bv_yyyy = v4sf_splat(b[1]);
			abv1t += avt * bv_yyyy;
			abv1b += avb * bv_yyyy;

			const v4sf bv_zzzz = v4sf_splat(b[2]);
			abv2t += avt * bv_zzzz;
			abv2b += avb * bv_zzzz;

			const v4sf bv_wwww = v4sf_splat(b[3]);
			abv3t += avt * bv_wwww;
			abv3b += avb * bv_wwww;

			a += 8;
			b += 4;
		}

		const v4sf alphav = v4sf_splat(*alpha);
		abv0t *= alphav;
		abv0b *= alphav;
		abv1t *= alphav;
		abv1b *= alphav;
		abv2t *= alphav;
		abv2b *= alphav;
		abv3t *= alphav;
		abv3b *= alphav;

		if (rs_c == 1) {
			v4sf cv0t = v4sf_load(&c[0*rs_c + 0*cs_c]);
			v4sf cv1t = v4sf_load(&c[0*rs_c + 1*cs_c]); 
			v4sf cv2t = v4sf_load(&c[0*rs_c + 2*cs_c]); 
			v4sf cv3t = v4sf_load(&c[0*rs_c + 3*cs_c]); 
			v4sf cv0b = v4sf_load(&c[4*rs_c + 0*cs_c]);
			v4sf cv1b = v4sf_load(&c[4*rs_c + 1*cs_c]); 
			v4sf cv2b = v4sf_load(&c[4*rs_c + 2*cs_c]); 
			v4sf cv3b = v4sf_load(&c[4*rs_c + 3*cs_c]); 

			const v4sf betav = v4sf_splat(*beta);
			cv0t = cv0t * betav + abv0t;
			cv1t = cv1t * betav + abv1t;
			cv2t = cv2t * betav + abv2t;
			cv3t = cv3t * betav + abv3t;
			cv0b = cv0b * betav + abv0b;
			cv1b = cv1b * betav + abv1b;
			cv2b = cv2b * betav + abv2b;
			cv3b = cv3b * betav + abv3b;

			v4sf_store(&c[0*rs_c + 0*cs_c], cv0t);
			v4sf_store(&c[0*rs_c + 1*cs_c], cv1t); 
			v4sf_store(&c[0*rs_c + 2*cs_c], cv2t); 
			v4sf_store(&c[0*rs_c + 3*cs_c], cv3t); 
			v4sf_store(&c[4*rs_c + 0*cs_c], cv0b);
			v4sf_store(&c[4*rs_c + 1*cs_c], cv1b); 
			v4sf_store(&c[4*rs_c + 2*cs_c], cv2b); 
			v4sf_store(&c[4*rs_c + 3*cs_c], cv3b); 
		} else {
			// Load columns 0, 1, 2, 3 (top part)
			v4sf cv0t = (v4sf){ c[0*rs_c + 0*cs_c], c[1*rs_c + 0*cs_c], c[2*rs_c + 0*cs_c], c[3*rs_c + 0*cs_c] };
			v4sf cv1t = (v4sf){ c[0*rs_c + 1*cs_c], c[1*rs_c + 1*cs_c], c[2*rs_c + 1*cs_c], c[3*rs_c + 1*cs_c] };
			v4sf cv2t = (v4sf){ c[0*rs_c + 2*cs_c], c[1*rs_c + 2*cs_c], c[2*rs_c + 2*cs_c], c[3*rs_c + 2*cs_c] };
			v4sf cv3t = (v4sf){ c[0*rs_c + 3*cs_c], c[1*rs_c + 3*cs_c], c[2*rs_c + 3*cs_c], c[3*rs_c + 3*cs_c] };
			// Load columns 0, 1, 2, 3 (bottom part)
			v4sf cv0b = (v4sf){ c[4*rs_c + 0*cs_c], c[5*rs_c + 0*cs_c], c[6*rs_c + 0*cs_c], c[7*rs_c + 0*cs_c] };
			v4sf cv1b = (v4sf){ c[4*rs_c + 1*cs_c], c[5*rs_c + 1*cs_c], c[6*rs_c + 1*cs_c], c[7*rs_c + 1*cs_c] };
			v4sf cv2b = (v4sf){ c[4*rs_c + 2*cs_c], c[5*rs_c + 2*cs_c], c[6*rs_c + 2*cs_c], c[7*rs_c + 2*cs_c] };
			v4sf cv3b = (v4sf){ c[4*rs_c + 3*cs_c], c[5*rs_c + 3*cs_c], c[6*rs_c + 3*cs_c], c[7*rs_c + 3*cs_c] };

			const v4sf betav = v4sf_splat(*beta);
			cv0t = cv0t * betav + abv0t;
			cv1t = cv1t * betav + abv1t;
			cv2t = cv2t * betav + abv2t;
			cv3t = cv3t * betav + abv3t;
			cv0b = cv0b * betav + abv0b;
			cv1b = cv1b * betav + abv1b;
			cv2b = cv2b * betav + abv2b;
			cv3b = cv3b * betav + abv3b;

			// Store column 0
			c[0*rs_c + 0*cs_c] = cv0t[0];
			c[1*rs_c + 0*cs_c] = cv0t[1];
			c[2*rs_c + 0*cs_c] = cv0t[2];
			c[3*rs_c + 0*cs_c] = cv0t[3];
			c[4*rs_c + 0*cs_c] = cv0b[0];
			c[5*rs_c + 0*cs_c] = cv0b[1];
			c[6*rs_c + 0*cs_c] = cv0b[2];
			c[7*rs_c + 0*cs_c] = cv0b[3];

			// Store column 1
			c[0*rs_c + 1*cs_c] = cv1t[0];
			c[1*rs_c + 1*cs_c] = cv1t[1];
			c[2*rs_c + 1*cs_c] = cv1t[2];
			c[3*rs_c + 1*cs_c] = cv1t[3];
			c[4*rs_c + 1*cs_c] = cv1b[0];
			c[5*rs_c + 1*cs_c] = cv1b[1];
			c[6*rs_c + 1*cs_c] = cv1b[2];
			c[7*rs_c + 1*cs_c] = cv1b[3];

			// Store column 2
			c[0*rs_c + 2*cs_c] = cv2t[0];
			c[1*rs_c + 2*cs_c] = cv2t[1];
			c[2*rs_c + 2*cs_c] = cv2t[2];
			c[3*rs_c + 2*cs_c] = cv2t[3];
			c[4*rs_c + 2*cs_c] = cv2b[0];
			c[5*rs_c + 2*cs_c] = cv2b[1];
			c[6*rs_c + 2*cs_c] = cv2b[2];
			c[7*rs_c + 2*cs_c] = cv2b[3];

			// Store column 3
			c[0*rs_c + 3*cs_c] = cv3t[0];
			c[1*rs_c + 3*cs_c] = cv3t[1];
			c[2*rs_c + 3*cs_c] = cv3t[2];
			c[3*rs_c + 3*cs_c] = cv3t[3];
			c[4*rs_c + 3*cs_c] = cv3b[0];
			c[5*rs_c + 3*cs_c] = cv3b[1];
			c[6*rs_c + 3*cs_c] = cv3b[2];
			c[7*rs_c + 3*cs_c] = cv3b[3];
		}
	}

	void bli_cgemm_opt
	     (
		   dim_t      k,
		   scomplex   alpha[restrict static 1],
		   scomplex   a[restrict static 4*k],
		   scomplex   b[restrict static k*4],
		   scomplex   beta[restrict static 1],
		   scomplex   c[restrict static 4*4],
		   inc_t      rs_c,
		   inc_t      cs_c,
		   auxinfo_t* data,
		   cntx_t*    cntx
	     )
	{
		// Vectors for accummulating column 0, 1, 2, 3 (initialize to 0.0)
		v4sf abv0r = v4sf_zero(), abv1r = v4sf_zero(), abv2r = v4sf_zero(), abv3r = v4sf_zero();
		v4sf abv0i = v4sf_zero(), abv1i = v4sf_zero(), abv2i = v4sf_zero(), abv3i = v4sf_zero();
		for (dim_t i = 0; i < k; i += 1) {
			const v4sf avt = v4sf_cload(a);
			const v4sf avb = v4sf_cload(a+2);
			const v4sf avr = __builtin_shufflevector(avt, avb, 0, 2, 4, 6);
			const v4sf avi = __builtin_shufflevector(avt, avb, 1, 3, 5, 7);

			const v4sf bv0r = v4sf_splat(b[0].real);
			const v4sf bv0i = v4sf_splat(b[0].imag);
			abv0r += avr * bv0r - avi * bv0i;
			abv0i += avr * bv0i + avi * bv0r;

			const v4sf bv1r = v4sf_splat(b[1].real);
			const v4sf bv1i = v4sf_splat(b[1].imag);
			abv1r += avr * bv1r - avi * bv1i;
			abv1i += avr * bv1i + avi * bv1r;

			const v4sf bv2r = v4sf_splat(b[2].real);
			const v4sf bv2i = v4sf_splat(b[2].imag);
			abv2r += avr * bv2r - avi * bv2i;
			abv2i += avr * bv2i + avi * bv2r;

			const v4sf bv3r = v4sf_splat(b[3].real);
			const v4sf bv3i = v4sf_splat(b[3].imag);
			abv3r += avr * bv3r - avi * bv3i;
			abv3i += avr * bv3i + avi * bv3r;

			a += 4;
			b += 4;
		}

		const v4sf alphavr = v4sf_splat(alpha->real);
		const v4sf alphavi = v4sf_splat(alpha->imag);
		v4sf temp;

		temp  = abv0r * alphavr - abv0i * alphavi;
		abv0i = abv0r * alphavi + abv0i * alphavr;
		abv0r = temp;

		temp  = abv1r * alphavr - abv1i * alphavi;
		abv1i = abv1r * alphavi + abv1i * alphavr;
		abv1r = temp;

		temp  = abv2r * alphavr - abv2i * alphavi;
		abv2i = abv2r * alphavi + abv2i * alphavr;
		abv2r = temp;

		temp  = abv3r * alphavr - abv3i * alphavi;
		abv3i = abv3r * alphavi + abv3i * alphavr;
		abv3r = temp;

		if (rs_c == 1) {
			const v4sf cv0t = v4sf_cload(&c[0*rs_c + 0*cs_c]);
			const v4sf cv1t = v4sf_cload(&c[0*rs_c + 1*cs_c]);
			const v4sf cv2t = v4sf_cload(&c[0*rs_c + 2*cs_c]);
			const v4sf cv3t = v4sf_cload(&c[0*rs_c + 3*cs_c]);
			const v4sf cv0b = v4sf_cload(&c[2*rs_c + 0*cs_c]);
			const v4sf cv1b = v4sf_cload(&c[2*rs_c + 1*cs_c]);
			const v4sf cv2b = v4sf_cload(&c[2*rs_c + 2*cs_c]);
			const v4sf cv3b = v4sf_cload(&c[2*rs_c + 3*cs_c]);

			v4sf cv0r = __builtin_shufflevector(cv0t, cv0b, 0, 2, 4, 6);
			v4sf cv0i = __builtin_shufflevector(cv0t, cv0b, 1, 3, 5, 7);
			v4sf cv1r = __builtin_shufflevector(cv1t, cv1b, 0, 2, 4, 6);
			v4sf cv1i = __builtin_shufflevector(cv1t, cv1b, 1, 3, 5, 7);
			v4sf cv2r = __builtin_shufflevector(cv2t, cv2b, 0, 2, 4, 6);
			v4sf cv2i = __builtin_shufflevector(cv2t, cv2b, 1, 3, 5, 7);
			v4sf cv3r = __builtin_shufflevector(cv3t, cv3b, 0, 2, 4, 6);
			v4sf cv3i = __builtin_shufflevector(cv3t, cv3b, 1, 3, 5, 7);

			const v4sf betavr = v4sf_splat(beta->real);
			const v4sf betavi = v4sf_splat(beta->imag);
			
			temp = abv0r + cv0r * betavr - cv0i * betavi;
			cv0i = abv0i + cv0r * betavi + cv0i * betavr;
			cv0r = temp;

			temp = abv1r + cv1r * betavr - cv1i * betavi;
			cv1i = abv1i + cv1r * betavi + cv1i * betavr;
			cv1r = temp;

			temp = abv2r + cv2r * betavr - cv2i * betavi;
			cv2i = abv2i + cv2r * betavi + cv2i * betavr;
			cv2r = temp;

			temp = abv3r + cv3r * betavr - cv3i * betavi;
			cv3i = abv3i + cv3r * betavi + cv3i * betavr;
			cv3r = temp;

			v4sf_cstore(&c[0*rs_c + 0*cs_c], __builtin_shufflevector(cv0r, cv0i, 0, 4, 1, 5));
			v4sf_cstore(&c[2*rs_c + 0*cs_c], __builtin_shufflevector(cv0r, cv0i, 2, 6, 3, 7));
			v4sf_cstore(&c[0*rs_c + 1*cs_c], __builtin_shufflevector(cv1r, cv1i, 0, 4, 1, 5));
			v4sf_cstore(&c[2*rs_c + 1*cs_c], __builtin_shufflevector(cv1r, cv1i, 2, 6, 3, 7));
			v4sf_cstore(&c[0*rs_c + 2*cs_c], __builtin_shufflevector(cv2r, cv2i, 0, 4, 1, 5));
			v4sf_cstore(&c[2*rs_c + 2*cs_c], __builtin_shufflevector(cv2r, cv2i, 2, 6, 3, 7));
			v4sf_cstore(&c[0*rs_c + 3*cs_c], __builtin_shufflevector(cv3r, cv3i, 0, 4, 1, 5));
			v4sf_cstore(&c[2*rs_c + 3*cs_c], __builtin_shufflevector(cv3r, cv3i, 2, 6, 3, 7));
		} else {
			// Load columns 0, 1, 2, 3 (real part)
			v4sf cv0r = (v4sf){ c[0*rs_c + 0*cs_c].real, c[1*rs_c + 0*cs_c].real, c[2*rs_c + 0*cs_c].real, c[3*rs_c + 0*cs_c].real };
			v4sf cv1r = (v4sf){ c[0*rs_c + 1*cs_c].real, c[1*rs_c + 1*cs_c].real, c[2*rs_c + 1*cs_c].real, c[3*rs_c + 1*cs_c].real };
			v4sf cv2r = (v4sf){ c[0*rs_c + 2*cs_c].real, c[1*rs_c + 2*cs_c].real, c[2*rs_c + 2*cs_c].real, c[3*rs_c + 2*cs_c].real };
			v4sf cv3r = (v4sf){ c[0*rs_c + 3*cs_c].real, c[1*rs_c + 3*cs_c].real, c[2*rs_c + 3*cs_c].real, c[3*rs_c + 3*cs_c].real };
			// Load columns 0, 1, 2, 3 (imaginary part)
			v4sf cv0i = (v4sf){ c[0*rs_c + 0*cs_c].imag, c[1*rs_c + 0*cs_c].imag, c[2*rs_c + 0*cs_c].imag, c[3*rs_c + 0*cs_c].imag };
			v4sf cv1i = (v4sf){ c[0*rs_c + 1*cs_c].imag, c[1*rs_c + 1*cs_c].imag, c[2*rs_c + 1*cs_c].imag, c[3*rs_c + 1*cs_c].imag };
			v4sf cv2i = (v4sf){ c[0*rs_c + 2*cs_c].imag, c[1*rs_c + 2*cs_c].imag, c[2*rs_c + 2*cs_c].imag, c[3*rs_c + 2*cs_c].imag };
			v4sf cv3i = (v4sf){ c[0*rs_c + 3*cs_c].imag, c[1*rs_c + 3*cs_c].imag, c[2*rs_c + 3*cs_c].imag, c[3*rs_c + 3*cs_c].imag };

			const v4sf betavr = v4sf_splat(beta->real);
			const v4sf betavi = v4sf_splat(beta->imag);

			temp = abv0r + cv0r * betavr - cv0i * betavi;
			cv0i = abv0i + cv0r * betavi + cv0i * betavr;
			cv0r = temp;

			temp = abv1r + cv1r * betavr - cv1i * betavi;
			cv1i = abv1i + cv1r * betavi + cv1i * betavr;
			cv1r = temp;

			temp = abv2r + cv2r * betavr - cv2i * betavi;
			cv2i = abv2i + cv2r * betavi + cv2i * betavr;
			cv2r = temp;

			temp = abv3r + cv3r * betavr - cv3i * betavi;
			cv3i = abv3i + cv3r * betavi + cv3i * betavr;
			cv3r = temp;

			// Store column 0
			c[0*rs_c + 0*cs_c].real = cv0r[0];
			c[0*rs_c + 0*cs_c].imag = cv0i[0];
			c[1*rs_c + 0*cs_c].real = cv0r[1];
			c[1*rs_c + 0*cs_c].imag = cv0i[1];
			c[2*rs_c + 0*cs_c].real = cv0r[2];
			c[2*rs_c + 0*cs_c].imag = cv0i[2];
			c[3*rs_c + 0*cs_c].real = cv0r[3];
			c[3*rs_c + 0*cs_c].imag = cv0i[3];

			// Store column 1
			c[0*rs_c + 1*cs_c].real = cv1r[0];
			c[0*rs_c + 1*cs_c].imag = cv1i[0];
			c[1*rs_c + 1*cs_c].real = cv1r[1];
			c[1*rs_c + 1*cs_c].imag = cv1i[1];
			c[2*rs_c + 1*cs_c].real = cv1r[2];
			c[2*rs_c + 1*cs_c].imag = cv1i[2];
			c[3*rs_c + 1*cs_c].real = cv1r[3];
			c[3*rs_c + 1*cs_c].imag = cv1i[3];

			// Store column 2
			c[0*rs_c + 2*cs_c].real = cv2r[0];
			c[0*rs_c + 2*cs_c].imag = cv2i[0];
			c[1*rs_c + 2*cs_c].real = cv2r[1];
			c[1*rs_c + 2*cs_c].imag = cv2i[1];
			c[2*rs_c + 2*cs_c].real = cv2r[2];
			c[2*rs_c + 2*cs_c].imag = cv2i[2];
			c[3*rs_c + 2*cs_c].real = cv2r[3];
			c[3*rs_c + 2*cs_c].imag = cv2i[3];

			// Store column 3
			c[0*rs_c + 3*cs_c].real = cv3r[0];
			c[0*rs_c + 3*cs_c].imag = cv3i[0];
			c[1*rs_c + 3*cs_c].real = cv3r[1];
			c[1*rs_c + 3*cs_c].imag = cv3i[1];
			c[2*rs_c + 3*cs_c].real = cv3r[2];
			c[2*rs_c + 3*cs_c].imag = cv3i[2];
			c[3*rs_c + 3*cs_c].real = cv3r[3];
			c[3*rs_c + 3*cs_c].imag = cv3i[3];
		}
	}
#endif
