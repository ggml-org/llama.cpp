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

#include "riscv_overloaded_intrinsics.h"

// macros to emit complex multiplication
// caveat: the destination registers cannot overlap the source registers!

// vd = vs2 * f[rs1]
#define VCMUL_VF(PREC, LMUL, VD_R, VD_I, VS2_R, VS2_I, RS1_R, RS1_I, VL) \
    do {                                                                 \
        VD_R = VFMUL_VF(PREC, LMUL)(VS2_R, RS1_R, VL);                   \
        VD_I = VFMUL_VF(PREC, LMUL)(VS2_R, RS1_I, VL);                   \
        VD_R = VFNMSAC_VF(PREC, LMUL)(VD_R, RS1_I, VS2_I, VL);           \
        VD_I = VFMACC_VF(PREC, LMUL)(VD_I, RS1_R, VS2_I, VL);            \
    } while(0)

// vd = conj(vs2) * f[rs1]
#define VCMUL_VF_CONJ(PREC, LMUL, VD_R, VD_I, VS2_R, VS2_I, RS1_R, RS1_I, VL) \
    do {                                                                      \
        VD_R = VFMUL_VF(PREC, LMUL)(VS2_R, RS1_R, VL);                        \
        VD_I = VFMUL_VF(PREC, LMUL)(VS2_R, RS1_I, VL);                        \
        VD_R = VFMACC_VF(PREC, LMUL)(VD_R, RS1_I, VS2_I, VL);                 \
        VD_I = VFNMSAC_VF(PREC, LMUL)(VD_I, RS1_R, VS2_I, VL);                \
    } while(0)

// vd = vs2 * f[rs1]
#define VCMUL_VF_TU(PREC, LMUL, VD_R, VD_I, VS2_R, VS2_I, RS1_R, RS1_I, VL) \
    do {                                                                    \
        VD_R = VFMUL_VF_TU(PREC, LMUL)(VS2_R, VS2_R, RS1_R, VL);            \
        VD_I = VFMUL_VF_TU(PREC, LMUL)(VS2_I, VS2_I, RS1_R, VL);            \
        VD_R = VFNMSAC_VF_TU(PREC, LMUL)(VD_R, RS1_I, VS2_I, VL);           \
        VD_I = VFMACC_VF_TU(PREC, LMUL)(VD_I, RS1_I, VS2_R, VL);            \
    } while(0)

// vd = conj(vs2) * f[rs1]
#define VCMUL_VF_CONJ_TU(PREC, LMUL, VD_R, VD_I, VS2_R, VS2_I, RS1_R, RS1_I, VL) \
    do {                                                                         \
        VD_R = VFMUL_VF_TU(PREC, LMUL)(VS2_R, VS2_R, RS1_R, VL);                 \
        VD_I = VFMUL_VF_TU(PREC, LMUL)(VS2_I, VS2_I, RS1_R, VL);                 \
        VD_R = VFMACC_VF_TU(PREC, LMUL)(VD_R, RS1_I, VS2_I, VL);                 \
        VD_I = VFMSAC_VF_TU(PREC, LMUL)(VD_I, RS1_I, VS2_R, VL);                 \
    } while(0)

// vd = vs2 * vs1
#define VCMUL_VV(PREC, LMUL, VD_R, VD_I, VS2_R, VS2_I, VS1_R, VS1_I, VL) \
    do {                                                                 \
        VD_R = VFMUL_VV(PREC, LMUL)(VS2_R, VS1_R, VL);                   \
        VD_I = VFMUL_VV(PREC, LMUL)(VS2_R, VS1_I, VL);                   \
        VD_R = VFNMSAC_VV(PREC, LMUL)(VD_R, VS1_I, VS2_I, VL);           \
        VD_I = VFMACC_VV(PREC, LMUL)(VD_I, VS1_R, VS2_I, VL);            \
    } while(0)

// vd = conj(vs2) * vs1
#define VCMUL_VV_CONJ(PREC, LMUL, VD_R, VD_I, VS2_R, VS2_I, VS1_R, VS1_I, VL) \
    do {                                                                      \
        VD_R = VFMUL_VV(PREC, LMUL)(VS2_R, VS1_R, VL);                        \
        VD_I = VFMUL_VV(PREC, LMUL)(VS2_R, VS1_I, VL);                        \
        VD_R = VFMACC_VV(PREC, LMUL)(VD_R, VS1_I, VS2_I, VL);                 \
        VD_I = VFNMSAC_VV(PREC, LMUL)(VD_I, VS1_R, VS2_I, VL);                \
    } while(0)

// vd += vs2 * f[rs1]
#define VCMACC_VF(PREC, LMUL, VD_R, VD_I, RS1_R, RS1_I, VS2_R, VS2_I, VL) \
    do {                                                                  \
        VD_R = VFMACC_VF(PREC, LMUL)(VD_R, RS1_R, VS2_R, VL);             \
        VD_I = VFMACC_VF(PREC, LMUL)(VD_I, RS1_I, VS2_R, VL);             \
        VD_R = VFNMSAC_VF(PREC, LMUL)(VD_R, RS1_I, VS2_I, VL);            \
        VD_I = VFMACC_VF(PREC, LMUL)(VD_I, RS1_R, VS2_I, VL);             \
    } while(0)

// vd += conj(vs2) * f[rs1]
#define VCMACC_VF_CONJ(PREC, LMUL, VD_R, VD_I, RS1_R, RS1_I, VS2_R, VS2_I, VL) \
    do {                                                                       \
        VD_R = VFMACC_VF(PREC, LMUL)(VD_R, RS1_R, VS2_R, VL);                  \
        VD_I = VFMACC_VF(PREC, LMUL)(VD_I, RS1_I, VS2_R, VL);                  \
        VD_R = VFMACC_VF(PREC, LMUL)(VD_R, RS1_I, VS2_I, VL);                  \
        VD_I = VFNMSAC_VF(PREC, LMUL)(VD_I, RS1_R, VS2_I, VL);                 \
    } while(0)

// vd = vs2 * f[rs1] - vd
#define VCMSAC_VF(PREC, LMUL, VD_R, VD_I, RS1_R, RS1_I, VS2_R, VS2_I, VL)  \
    do {                                                                   \
        VD_R = VFMSAC_VF(PREC, LMUL)(VD_R, RS1_R, VS2_R, VL);              \
        VD_I = VFMSAC_VF(PREC, LMUL)(VD_I, RS1_I, VS2_R, VL);              \
        VD_R = VFNMSAC_VF(PREC, LMUL)(VD_R, RS1_I, VS2_I, VL);             \
        VD_I = VFMACC_VF(PREC, LMUL)(VD_I, RS1_R, VS2_I, VL);              \
    } while(0)

// vd -= vs2 * f[rs1]
#define VCNMSAC_VF(PREC, LMUL, VD_R, VD_I, RS1_R, RS1_I, VS2_R, VS2_I, VL) \
    do {                                                                   \
        VD_R = VFNMSAC_VF(PREC, LMUL)(VD_R, RS1_R, VS2_R, VL);             \
        VD_I = VFNMSAC_VF(PREC, LMUL)(VD_I, RS1_I, VS2_R, VL);             \
        VD_R = VFMACC_VF(PREC, LMUL)(VD_R, RS1_I, VS2_I, VL);              \
        VD_I = VFNMSAC_VF(PREC, LMUL)(VD_I, RS1_R, VS2_I, VL);             \
    } while(0)

// vd += vs2 * vs1
#define VCMACC_VV_TU(PREC, LMUL, VD_R, VD_I, VS1_R, VS1_I, VS2_R, VS2_I, VL) \
    do {                                                                     \
        VD_R = VFMACC_VV_TU(PREC, LMUL)(VD_R, VS1_R, VS2_R, VL);             \
        VD_I = VFMACC_VV_TU(PREC, LMUL)(VD_I, VS1_I, VS2_R, VL);             \
        VD_R = VFNMSAC_VV_TU(PREC, LMUL)(VD_R, VS1_I, VS2_I, VL);            \
        VD_I = VFMACC_VV_TU(PREC, LMUL)(VD_I, VS1_R, VS2_I, VL);             \
    } while(0)

// vd += conj(vs2) * vs1
#define VCMACC_VV_CONJ_TU(PREC, LMUL, VD_R, VD_I, VS1_R, VS1_I, VS2_R, VS2_I, VL) \
    do {                                                                          \
        VD_R = VFMACC_VV_TU(PREC, LMUL)(VD_R, VS1_R, VS2_R, VL);                  \
        VD_I = VFMACC_VV_TU(PREC, LMUL)(VD_I, VS1_I, VS2_R, VL);                  \
        VD_R = VFMACC_VV_TU(PREC, LMUL)(VD_R, VS1_I, VS2_I, VL);                  \
        VD_I = VFNMSAC_VV_TU(PREC, LMUL)(VD_I, VS1_R, VS2_I, VL);                 \
    } while(0)

