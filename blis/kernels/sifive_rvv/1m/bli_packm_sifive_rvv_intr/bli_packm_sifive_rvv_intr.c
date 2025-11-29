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

#include "../../riscv_cmul_macros_intr.h"
#include "../../riscv_overloaded_intrinsics.h"
#include "blis.h"
#include <stdint.h>
#include <riscv_vector.h>

#define PACKM_(PRECISION_CHAR, T) void bli_##PRECISION_CHAR##packm_sifive_rvv_intr(\
         conj_t           conja,                     \
         pack_t           schema,                    \
         dim_t            cdim,                      \
         dim_t            cdim_max,                  \
         dim_t            cdim_bcast,                \
         dim_t            n,                         \
         dim_t            n_max,                     \
   const T*      restrict kappa_,                    \
   const T*      restrict a_, inc_t inca, inc_t lda, \
         T*      restrict p_,             inc_t ldp, \
   const T*      restrict params,                    \
   const cntx_t*          cntx                       \
)

#define PACKM(...)  PACKM_(__VA_ARGS__)

#define BLI_SCAL2BBS_MXN_(PRECISION_CHAR) bli_##PRECISION_CHAR##scal2bbs_mxn
#define BLI_SCAL2BBS_MXN(PRECISION_CHAR) BLI_SCAL2BBS_MXN_(PRECISION_CHAR)

#define BLI_SET0S_EDGE_(PRECISION_CHAR) bli_##PRECISION_CHAR##set0s_edge
#define BLI_SET0S_EDGE(PRECISION_CHAR) BLI_SET0S_EDGE_(PRECISION_CHAR)

// LMUL is the LMUL used when a is "row major" (lda == 1). Since we use
// segment stores with more than 4 fields, this is usually m1.
// LMUL_MR is an LMUL large enough to hold MR floats (for spackm, cpackm)
// or doubles (for dpackm, zpackm). LMUL_NR is analogous.

// Single precision real
#define DATATYPE float
#define PRECISION_CHAR s
#define PREC 32
#define LMUL m1
#define LMUL_MR m1
#define LMUL_NR m4
#define FLT_SIZE sizeof(float)
#define MR 7
#define NR ( 4 * __riscv_vlenb() / 4 )

#include "./bli_packm_sifive_rvv_intr_real.c"

#undef DATATYPE
#undef PRECISION_CHAR
#undef PREC
#undef LMUL
#undef LMUL_MR
#undef LMUL_NR
#undef FLT_SIZE
#undef MR
#undef NR

// Double precision real
#define DATATYPE double
#define PRECISION_CHAR d
#define PREC 64
#define LMUL m1
#define LMUL_MR m1
#define LMUL_NR m4
#define FLT_SIZE sizeof(double)
#define MR 7
#define NR ( 4 * __riscv_vlenb() / 8 )

#include "./bli_packm_sifive_rvv_intr_real.c"

#undef DATATYPE
#undef PRECISION_CHAR
#undef PREC
#undef LMUL
#undef LMUL_MR
#undef LMUL_NR
#undef FLT_SIZE
#undef MR
#undef NR

// Single precision complex
#define DATATYPE scomplex
#define BASE_DT float
#define PRECISION_CHAR c
#define PREC 32
#define LMUL m1
#define LMUL_MR m1
#define LMUL_NR m2
#define FLT_SIZE sizeof(float)
#define MR 6
#define NR ( 2 * __riscv_vlenb() / 4 )

#include "./bli_packm_sifive_rvv_intr_complex.c"

#undef DATATYPE
#undef BASE_DT
#undef PRECISION_CHAR
#undef PREC
#undef LMUL
#undef LMUL_MR
#undef LMUL_NR
#undef FLT_SIZE
#undef MR
#undef NR

// Double precision complex
#define DATATYPE dcomplex
#define BASE_DT double
#define PRECISION_CHAR z
#define PREC 64
#define LMUL m1
#define LMUL_MR m1
#define LMUL_NR m2
#define FLT_SIZE sizeof(double)
#define MR 6
#define NR ( 2 * __riscv_vlenb() / 8 )

#include "./bli_packm_sifive_rvv_intr_complex.c"

#undef DATATYPE
#undef BASE_DT
#undef PRECISION_CHAR
#undef PREC
#undef LMUL
#undef LMUL_MR
#undef LMUL_NR
#undef FLT_SIZE
#undef MR
#undef NR

#undef BLI_SCAL2BBS_MXN_
#undef BLI_SCAL2BBS_MXN

#undef BLI_SET0S_EDGE_
#undef BLI_SET0S_EDGE

#undef PACKM
#undef PACKM_
