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
#include "../../riscv_overloaded_intrinsics.h"
#include "blis.h"
#include <limits.h>
#include <riscv_vector.h>
#include <stdbool.h>
#include <stddef.h>

#define AMAXV_(PRECISION_CHAR, T) void bli_##PRECISION_CHAR##amaxv_sifive_rvv_intr(\
          dim_t            n,              \
    const T*      restrict x_, inc_t incx, \
          dim_t*           index,          \
    const cntx_t*          cntx            \
)

#define AMAXV(...)  AMAXV_(__VA_ARGS__)

// BLIS defines integers to be 32 or 64 bits according to BLIS_INT_TYPE_SIZE.
// If BLIS_INT_TYPE_SIZE is any other value, integers are defined to be longs.
#if BLIS_INT_TYPE_SIZE == 32 || BLIS_INT_TYPE_SIZE == 64
#define AMAXV_SIFIVE_RVV_INT_SIZE BLIS_INT_TYPE_SIZE
#elif LONG_MAX == INT32_MAX
#define AMAXV_SIFIVE_RVV_INT_SIZE 32
#elif LONG_MAX == INT64_MAX
#define AMAXV_SIFIVE_RVV_INT_SIZE 64
#else
#error "Integers must be 32- or 64-bits for bli_?amaxv_sifive_rvv_intr."
#endif

// Single precision real
#define DATATYPE float
#define PRECISION_CHAR s
#define PREC_X 32
#define PREC_I AMAXV_SIFIVE_RVV_INT_SIZE
#if PREC_I == 32
#define LMUL_X m4
#define LMUL_I m4
#define RATIO 8
#elif PREC_I == 64
#define LMUL_X m4
#define LMUL_I m8
#define RATIO 8
#endif
#define FLT_SIZE sizeof(float)

#include "./bli_amaxv_sifive_rvv_intr_real.c"

#undef DATATYPE
#undef PRECISION_CHAR
#undef PREC_X
#undef PREC_I
#undef LMUL_X
#undef LMUL_I
#undef RATIO
#undef FLT_SIZE

// Double precision real
#define DATATYPE double
#define PRECISION_CHAR d
#define PREC_X 64
#define PREC_I AMAXV_SIFIVE_RVV_INT_SIZE
#if PREC_I == 32
#define LMUL_X m8
#define LMUL_I m4
#define RATIO 8
#elif PREC_I == 64
#define LMUL_X m4
#define LMUL_I m4
#define RATIO 16
#endif
#define FLT_SIZE sizeof(double)

#include "./bli_amaxv_sifive_rvv_intr_real.c"

#undef DATATYPE
#undef PRECISION_CHAR
#undef PREC_X
#undef PREC_I
#undef LMUL_X
#undef LMUL_I
#undef RATIO
#undef FLT_SIZE

// Single precision complex
#define DATATYPE scomplex
#define BASE_DT float
#define PRECISION_CHAR c
#define PREC_X 32
#define PREC_I AMAXV_SIFIVE_RVV_INT_SIZE
#if PREC_I == 32
#define LMUL_X m4
#define LMUL_I m4
#define RATIO 8
#elif PREC_I == 64
#define LMUL_X m4
#define LMUL_I m8
#define RATIO 8
#endif
#define FLT_SIZE sizeof(float)

#include "./bli_amaxv_sifive_rvv_intr_complex.c"

#undef DATATYPE
#undef BASE_DT
#undef PRECISION_CHAR
#undef PREC_X
#undef PREC_I
#undef LMUL_X
#undef LMUL_I
#undef RATIO
#undef FLT_SIZE

// Double precision complex
#define DATATYPE dcomplex
#define BASE_DT double
#define PRECISION_CHAR z
#define PREC_X 64
#define PREC_I AMAXV_SIFIVE_RVV_INT_SIZE
#if PREC_I == 32
#define LMUL_X m8
#define LMUL_I m4
#define RATIO 8
#elif PREC_I == 64
#define LMUL_X m4
#define LMUL_I m4
#define RATIO 16
#endif
#define FLT_SIZE sizeof(double)

#include "./bli_amaxv_sifive_rvv_intr_complex.c"

#undef DATATYPE
#undef BASE_DT
#undef PRECISION_CHAR
#undef PREC_X
#undef PREC_I
#undef LMUL_X
#undef LMUL_I
#undef RATIO
#undef FLT_SIZE

#undef AMAXV_SIFIVE_RVV_INT_SIZE

#undef AMAXV
#undef AMAXV_
