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

#ifndef BLIS_L1V_KER_PARAMS_H
#define BLIS_L1V_KER_PARAMS_H


#define addv_params \
\
             conj_t  conjx, \
             dim_t   n, \
       const void*   x, inc_t incx, \
             void*   y, inc_t incy

#define amaxv_params \
\
             dim_t   n, \
       const void*   x, inc_t incx, \
             dim_t*  index

#define axpbyv_params \
\
             conj_t  conjx, \
             dim_t   n, \
       const void*   alpha, \
       const void*   x, inc_t incx, \
       const void*   beta, \
             void*   y, inc_t incy

#define axpyv_params \
\
             conj_t  conjx, \
             dim_t   n, \
       const void*   alpha, \
       const void*   x, inc_t incx, \
             void*   y, inc_t incy

#define copyv_params \
\
             conj_t  conjx, \
             dim_t   n, \
       const void*   x, inc_t incx, \
             void*   y, inc_t incy

#define dotv_params \
\
             conj_t  conjx, \
             conj_t  conjy, \
             dim_t   n, \
       const void*   x, inc_t incx, \
       const void*   y, inc_t incy, \
             void*   rho

#define dotxv_params \
\
             conj_t  conjx, \
             conj_t  conjy, \
             dim_t   n, \
       const void*   alpha, \
       const void*   x, inc_t incx, \
       const void*   y, inc_t incy, \
       const void*   beta, \
             void*   rho

#define invertv_params \
\
             dim_t   n, \
             void*   x, inc_t incx

#define invscalv_params \
\
             conj_t  conjalpha, \
             dim_t   n, \
       const void*   alpha, \
             void*   x, inc_t incx

#define scalv_params \
\
             conj_t  conjalpha, \
             dim_t   n, \
       const void*   alpha, \
             void*   x, inc_t incx

#define scal2v_params \
\
             conj_t  conjx, \
             dim_t   n, \
       const void*   alpha, \
       const void*   x, inc_t incx, \
             void*   y, inc_t incy

#define setv_params \
\
             conj_t  conjalpha, \
             dim_t   n, \
       const void*   alpha, \
             void*   x, inc_t incx

#define subv_params \
\
             conj_t  conjx, \
             dim_t   n, \
       const void*   x, inc_t incx, \
             void*   y, inc_t incy

#define swapv_params \
\
             dim_t   n, \
             void*   x, inc_t incx, \
             void*   y, inc_t incy

#define xpbyv_params \
\
             conj_t  conjx, \
             dim_t   n, \
       const void*   x, inc_t incx, \
       const void*   beta, \
             void*   y, inc_t incy

#endif

