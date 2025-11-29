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
#include "blis.h"
#include "../../riscv_cmul_macros_intr.h"
#include "../../bli_kernels_sifive_rvv.h"
#include <stdint.h>
#include <riscv_vector.h>

#define GEMMTRSM_L(PRECISION_CHAR, T) void bli_##PRECISION_CHAR##gemmtrsm_l_sifive_rvv_intr(\
          dim_t               m,      \
          dim_t               n,      \
          dim_t               k,      \
    const T*         restrict alpha_, \
    const T*         restrict a10_,   \
    const T*         restrict a11_,   \
    const T*         restrict b01_,   \
          T*         restrict b11_,   \
          T*         restrict c11_,   \
          inc_t               rsc,    \
          inc_t               csc,    \
    const auxinfo_t* restrict data,   \
    const cntx_t*    restrict cntx    \
    )

#define GEMMTRSM_U(PRECISION_CHAR, T) void bli_##PRECISION_CHAR##gemmtrsm_u_sifive_rvv_intr(\
          dim_t               m,      \
          dim_t               n,      \
          dim_t               k,      \
    const T*         restrict alpha_, \
    const T*         restrict a12_,   \
    const T*         restrict a11_,   \
    const T*         restrict b21_,   \
          T*         restrict b11_,   \
          T*         restrict c11_,   \
          inc_t               rsc,    \
          inc_t               csc,    \
    const auxinfo_t* restrict data,   \
    const cntx_t*    restrict cntx    \
    )

#define GEMMTRSM(macro, ...)  macro(__VA_ARGS__)

// Single precision real
#define DATATYPE float
#define PRECISION_CHAR s
#define PREC 32
#define LMUL m4
#define FLT_SIZE sizeof(float)
#define PACKMR 8
#define PACKNR ( 4 * __riscv_vlenb() / 4 )

#include "./bli_gemmtrsm_sifive_rvv_intr_real.c"

#undef DATATYPE
#undef PRECISION_CHAR
#undef PREC
#undef LMUL
#undef FLT_SIZE
#undef PACKMR
#undef PACKNR

// Double precision real
#define DATATYPE double
#define PRECISION_CHAR d
#define PREC 64
#define LMUL m4
#define FLT_SIZE sizeof(double)
#define PACKMR 8
#define PACKNR ( 4 * __riscv_vlenb() / 8 )

#include "./bli_gemmtrsm_sifive_rvv_intr_real.c"

#undef DATATYPE
#undef PRECISION_CHAR
#undef PREC
#undef LMUL
#undef FLT_SIZE
#undef PACKMR
#undef PACKNR

// Single precision complex
#define DATATYPE scomplex
#define BASE_DT float
#define PRECISION_CHAR c
#define PREC 32
#define LMUL m2
#define FLT_SIZE sizeof(float)
#define PACKMR 8
#define PACKNR ( 2 * __riscv_vlenb() / 4 )

#include "./bli_gemmtrsm_sifive_rvv_intr_complex.c"

#undef DATATYPE
#undef BASE_DT
#undef PRECISION_CHAR
#undef PREC
#undef LMUL
#undef FLT_SIZE
#undef PACKMR
#undef PACKNR

// Double precision complex
#define DATATYPE dcomplex
#define BASE_DT double
#define PRECISION_CHAR z
#define PREC 64
#define LMUL m2
#define FLT_SIZE sizeof(double)
#define PACKMR 8
#define PACKNR ( 2 * __riscv_vlenb() / 8 )

#include "./bli_gemmtrsm_sifive_rvv_intr_complex.c"

#undef DATATYPE
#undef BASE_DT
#undef PRECISION_CHAR
#undef PREC
#undef LMUL
#undef FLT_SIZE
#undef PACKMR
#undef PACKNR

#undef GEMMTRSM
#undef GEMMTRSM_L
#undef GEMMTRSM_U
