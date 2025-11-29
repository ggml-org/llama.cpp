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
// Clang's label requirements.
#if defined(__clang__)
#define LABEL(str) "   L" #str"%=: \n\t"
#define BEQ(str) "b.eq L" #str"%=  \n\t"
#define BNE(str) "b.ne L" #str"%=  \n\t"
#define BRANCH(str) "b L" #str"%=  \n\t"
#else
#define LABEL(str) "   ." #str": \n\t"
#define BEQ(str) "b.eq ." #str"  \n\t"
#define BNE(str) "b.ne ." #str"  \n\t"
#define BRANCH(str) "b ." #str"  \n\t"
#endif

#define CLEAR_COL2(Z0,Z1) \
" dup  "#Z0"."DT", #0 \n\t" \
" dup  "#Z1"."DT", #0 \n\t"

#define CLEAR_COL4(Z0,Z1,Z2,Z3) \
  CLEAR_COL2(Z0,Z1) \
  CLEAR_COL2(Z2,Z3)

#define SCALE_COL2(Z0,Z1,ZFACTOR) \
" fmul  "#Z0"."DT", "#Z0"."DT", "#ZFACTOR"."DT" \n\t" \
" fmul  "#Z1"."DT", "#Z1"."DT", "#ZFACTOR"."DT" \n\t" \

#define SCALE_COL4(Z0,Z1,Z2,Z3,ZFACTOR) \
  SCALE_COL2(Z0,Z1,ZFACTOR) \
  SCALE_COL2(Z2,Z3,ZFACTOR)

// Prefetch or not.
#define PREFETCH_CONTIGUOUS_noprfm(LV,PROP,ADDR,SHIFT)
#define PREFETCH_CONTIGUOUS_prfm(LV,PROP,ADDR,SHIFT) \
" prfm  PLD"#LV""#PROP", ["#ADDR", "#SHIFT"] \n\t"

#define GEMM_FMLA2(CCOLFH,CCOLLH,PT,ACOLFH,ACOLLH,BV) \
" fmla  "#CCOLFH"."DT", "#PT"/m, "#ACOLFH"."DT", "#BV"."DT" \n\t" /* A Row 0 :VL */ \
" fmla  "#CCOLLH"."DT", "#PT"/m, "#ACOLLH"."DT", "#BV"."DT" \n\t" /* A Row VL:2VL */

#define GEMM_FMLA2_LD1R(CCOLFH,CCOLLH,PT,ACOLFH,ACOLLH,BV,BADDR,NSHIFT) \
  GEMM_FMLA2(CCOLFH,CCOLLH,PT,ACOLFH,ACOLLH,BV) \
" "LD1R"  "#BV"."DT", "#PT"/z, ["#BADDR", #"#NSHIFT"*"SZ"]\n\t"

#define GEMM_FMLA2_LD1R_G_ELMFWD(CCOLFH,CCOLLH,PT,ACOLFH,ACOLLH,BV,BELMADDR,BCSBIT) \
  GEMM_FMLA2(CCOLFH,CCOLLH,PT,ACOLFH,ACOLLH,BV) \
" "LD1R"  "#BV"."DT", "#PT"/z, ["#BELMADDR"] \n\t" /* Load B */ \
" add     "#BELMADDR", "#BELMADDR", "#BCSBIT" \n\t" /* Forward B element */

#define GEMM_ACOL_CONTIGUOUS_LOAD(ZFH,ZLH,PFH,PLH,AADDR) \
" "LD1"  "#ZFH"."DT", "#PFH"/z, ["#AADDR"]            \n\t" \
" "LD1"  "#ZLH"."DT", "#PLH"/z, ["#AADDR", #1, mul vl]\n\t"

#define GEMM_ACOL_GATHER_LOAD(ZFH,ZLH,ZIDX,PFH,PLH,AADDR,AVSKIP,ATEMP) \
" "LD1"  "#ZFH"."DT", "#PFH"/z, ["#AADDR", "#ZIDX"."DT", "OFFS"]\n\t" \
" add    "#ATEMP", "#AADDR", "#AVSKIP" \n\t" \
" "LD1"  "#ZLH"."DT", "#PLH"/z, ["#ATEMP", "#ZIDX"."DT", "OFFS"]\n\t"

// Prefetch or not.
#define GEMM_ACOL_GATHER_noprfm(LV,PROP,ZIDX,PFH,PLH,AADDR,AVSKIP,ATEMP)
#define GEMM_ACOL_GATHER_prfm(LV,PROP,ZIDX,PFH,PLH,AADDR,AVSKIP,ATEMP) \
" "PRFG" PLD"#LV""#PROP", "#PFH", ["#AADDR", "#ZIDX"."DT", "OFFS"] \n\t" \
" add    "#ATEMP", "#AADDR", "#AVSKIP" \n\t" \
" "PRFG" PLD"#LV""#PROP", "#PLH", ["#ATEMP", "#ZIDX"."DT", "OFFS"] \n\t"

#define GEMMSUP_ACOL_PREFETCH_NEXT_LOAD_C(ZFH,ZLH,PFH,PLH,AADDR,A4KS,ACS,ATEMP,PREFMODE) \
" add  "#ATEMP", "#AADDR", "#A4KS" \n\t" \
" add  "#AADDR", "#AADDR", "#ACS"  \n\t" /* Forward A's address to the next column. */ \
  GEMM_ACOL_CONTIGUOUS_LOAD(ZFH,ZLH,PFH,PLH,AADDR) \
  PREFETCH_CONTIGUOUS_ ##PREFMODE(L1,STRM,ATEMP,0)

#define GEMMSUP_ACOL_PREFETCH_NEXT_LOAD_G(ZFH,ZLH,ZIDX,PFH,PLH,AADDR,A4KS,APS,ACS,AVSKIP,ATEMP,PREFMODEL1,PREFMODEL2) \
" add  "#ATEMP", "#AADDR", "#A4KS" \n\t" \
  GEMM_ACOL_GATHER_ ##PREFMODEL1(L1,STRM,ZIDX,PFH,PLH,ATEMP,AVSKIP,ATEMP) \
" add  "#ATEMP", "#AADDR", "#APS"  \n\t" \
  GEMM_ACOL_GATHER_ ##PREFMODEL2(L2,STRM,ZIDX,PFH,PLH,ATEMP,AVSKIP,ATEMP) \
" add  "#AADDR", "#AADDR", "#ACS"  \n\t" /* Forward A's address to the next column. */ \
  GEMM_ACOL_GATHER_LOAD(ZFH,ZLH,ZIDX,PFH,PLH,AADDR,AVSKIP,ATEMP)

#define GEMM_CCOL_CONTIGUOUS_LOAD_FWD(ZFH,ZLH,PFH,PLH,CADDR,CCS) \
  GEMM_ACOL_CONTIGUOUS_LOAD(ZFH,ZLH,PFH,PLH,CADDR) \
" add  "#CADDR", "#CADDR", "#CCS" \n\t" /* Forward C address (load) to next column. */

#define GEMM_CCOL_CONTIGUOUS_STORE_FWD(ZFH,ZLH,PFH,PLH,CADDR,CCS) \
" "ST1" "#ZFH"."DT", "#PFH", ["#CADDR"]             \n\t" \
" "ST1" "#ZLH"."DT", "#PLH", ["#CADDR", #1, mul vl] \n\t" \
" add    "#CADDR", "#CADDR", "#CCS" \n\t" /* Forward C address (store) to next column. */

#define GEMM_CCOL_FMAD(ZFH,ZLH,PFH,PLH,CFH,CLH,ZSCALE) \
" fmad  "#ZFH"."DT", "#PFH"/m, "#ZSCALE"."DT", "#CFH"."DT" \n\t" \
" fmad  "#ZLH"."DT", "#PLH"/m, "#ZSCALE"."DT", "#CLH"."DT" \n\t"

#define GEMM_CCOL_GATHER_LOAD_FWD(ZFH,ZLH,ZIDX,PFH,PLH,CADDR,CCS,CVSKIP,CTEMP) \
  GEMM_ACOL_GATHER_LOAD(ZFH,ZLH,ZIDX,PFH,PLH,CADDR,CVSKIP,CTEMP) \
" add  "#CADDR", "#CADDR", "#CCS"      \n\t"

#define GEMM_CCOL_SCATTER_STORE_FWD(ZFH,ZLH,ZIDX,PFH,PLH,CADDR,CCS,CVSKIP,CTEMP) \
" "ST1" "#ZFH"."DT", "#PFH", ["#CADDR", "#ZIDX"."DT", "OFFS"]\n\t" \
" add   "#CTEMP", "#CADDR", "#CVSKIP"   \n\t" \
" "ST1" "#ZLH"."DT", "#PLH", ["#CTEMP", "#ZIDX"."DT", "OFFS"]\n\t" \
" add   "#CADDR", "#CADDR", "#CCS"      \n\t"


