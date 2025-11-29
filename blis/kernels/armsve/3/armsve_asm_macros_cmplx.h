/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020, The University of Tokyo

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
#include "armsve_asm_macros.h"

#define FMUL_COL2(ZD0,ZD1,Z0,Z1,ZFACTOR) \
" fmul  "#ZD0"."DT", "#Z0"."DT", "#ZFACTOR"."DT" \n\t" \
" fmul  "#ZD1"."DT", "#Z1"."DT", "#ZFACTOR"."DT" \n\t" \

#define GEMM_FMLX2(CCOLFH,CCOLLH,PT,ACOLFH,ACOLLH,BV) \
" fmla  "#CCOLFH"."DT", "#PT"/m, "#ACOLFH"."DT", "#BV"."DT" \n\t" \
" fmls  "#CCOLLH"."DT", "#PT"/m, "#ACOLLH"."DT", "#BV"."DT" \n\t"

#define GEMM_FMLX2_LD1R(CCOLFH,CCOLLH,PT,ACOLFH,ACOLLH,BV,BADDR,NSHIFT) \
  GEMM_FMLX2(CCOLFH,CCOLLH,PT,ACOLFH,ACOLLH,BV) \
" "LD1R"  "#BV"."DT", "#PT"/z, ["#BADDR", #"#NSHIFT"*"SZ"]\n\t"

#define GEMM_FMULCMPLX(ZDRe,ZDIm,PT,Z0Re,Z0Im,Z1Re,Z1Im) \
  FMUL_COL2(ZDRe,ZDIm,Z0Re,Z0Im,Z1Re) \
  GEMM_FMLX2(ZDIm,ZDRe,PT,Z0Re,Z0Im,Z1Im)

#define GEMM_FMLACMPLX(ZDRe,ZDIm,PT,Z0Re,Z0Im,Z1Re,Z1Im) \
  GEMM_FMLA2(ZDRe,ZDIm,PT,Z0Re,Z0Im,Z1Re) \
  GEMM_FMLX2(ZDIm,ZDRe,PT,Z0Re,Z0Im,Z1Im)

#define GEMM_ACOLCMPLX_CONTIGUOUS_LOAD(ZRe,ZIm,PT,AAddr) \
" "LD2" {"#ZRe"."DT", "#ZIm"."DT"}, "#PT"/z, ["#AAddr"] \n\t"

#define GEMM_ACOLCMPLX_CONTIGUOUS_STORE(ZRe,ZIm,PT,AAddr) \
" "ST2" {"#ZRe"."DT", "#ZIm"."DT"}, "#PT", ["#AAddr"] \n\t"

#define GEMM_ACOLCMPLX_CONTIGUOUS_LOAD_FWD(ZRe,ZIm,PT,AAddr,ACS) \
  GEMM_ACOLCMPLX_CONTIGUOUS_LOAD(ZRe,ZIm,PT,AAddr) \
" add  "#AAddr", "#AAddr", "#ACS" \n\t" /* Forward A address (load) to next column. */

#define GEMM_CCOLCMPLX_CONTIGUOUS_LOAD_FWD(ZRe,ZIm,PT,CAddr,CCS) \
  GEMM_ACOLCMPLX_CONTIGUOUS_LOAD_FWD(ZRe,ZIm,PT,CAddr,CCS)

#define GEMM_ACOLCMPLX_CONTIGUOUS_STORE_FWD(ZRe,ZIm,PT,AAddr,ACS) \
  GEMM_ACOLCMPLX_CONTIGUOUS_STORE(ZRe,ZIm,PT,AAddr) \
" add  "#AAddr", "#AAddr", "#ACS" \n\t" /* Forward A address (load) to next column. */

#define GEMM_CCOLCMPLX_CONTIGUOUS_STORE_FWD(ZRe,ZIm,PT,CAddr,CCS) \
  GEMM_ACOLCMPLX_CONTIGUOUS_STORE_FWD(ZRe,ZIm,PT,CAddr,CCS)

#define GEMM_CCOLCMPLX_GATHER_LOAD_FWD(ZRe,ZIm,ZIndex,PRe,PIm,CAddr,CCS,CTemp) \
" add  "#CTemp", "#CAddr", #"SZ"  \n\t" /* Imaginary skip */ \
" "LD1" "#ZRe"."DT", "#PRe"/z, ["#CAddr", "#ZIndex"."DT", "OFFS"]\n\t" \
" "LD1" "#ZIm"."DT", "#PRe"/z, ["#CTemp", "#ZIndex"."DT", "OFFS"]\n\t" \
" add  "#CAddr", "#CAddr", "#CCS" \n\t"

#define GEMM_CCOLCMPLX_SCATTER_STORE_FWD(ZRe,ZIm,ZIndex,PRe,PIm,CAddr,CCS,CTemp) \
" add  "#CTemp", "#CAddr", #"SZ"  \n\t" /* Imaginary skip */ \
" "ST1" "#ZRe"."DT", "#PRe", ["#CAddr", "#ZIndex"."DT", "OFFS"]\n\t" \
" "ST1" "#ZIm"."DT", "#PRe", ["#CTemp", "#ZIndex"."DT", "OFFS"]\n\t" \
" add  "#CAddr", "#CAddr", "#CCS" \n\t"

