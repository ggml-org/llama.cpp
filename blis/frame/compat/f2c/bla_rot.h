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

BLIS_EXPORT_BLAS int PASTEF77(s,rot)(const bla_integer *n, bla_real *sx, const bla_integer *incx, bla_real *sy, const bla_integer *incy, const bla_real *c__, const bla_real *s);
BLIS_EXPORT_BLAS int PASTEF77(d,rot)(const bla_integer *n, bla_double *dx, const bla_integer *incx, bla_double *dy, const bla_integer *incy, const bla_double *c__, const bla_double *s);
BLIS_EXPORT_BLAS int PASTEF77(cs,rot)(const bla_integer *n, bla_scomplex *cx, const bla_integer *incx, bla_scomplex *cy, const bla_integer *incy, const bla_real *c__, const bla_real *s);
BLIS_EXPORT_BLAS int PASTEF77(zd,rot)(const bla_integer *n, bla_dcomplex *zx, const bla_integer *incx, bla_dcomplex *zy, const bla_integer *incy, const bla_double *c__, const bla_double *s);
#ifndef BLIS_DISABLE_CROT
BLIS_EXPORT_BLAS int PASTEF77(c,rot)(const bla_integer *n, bla_scomplex *cx, const bla_integer *incx, bla_scomplex *cy, const bla_integer *incy, const bla_real *c__, const bla_scomplex *s);
#endif
#ifndef BLIS_DISABLE_ZROT
BLIS_EXPORT_BLAS int PASTEF77(z,rot)(const bla_integer *n, bla_dcomplex *cx, const bla_integer *incx, bla_dcomplex *cy, const bla_integer *incy, const bla_double *c__, const bla_dcomplex *s);
#endif

