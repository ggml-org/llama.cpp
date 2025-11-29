/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2021, The University of Tokyo

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

// Apple/Clang's local label requirements.
#if defined(__APPLE__) || defined(__clang__)
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

// Clear vectors.
#define CLEAR1V(V) \
" dup   v"#V".2d, xzr \n\t"
#define CLEAR2V(V0,V1) \
  CLEAR1V(V0) \
  CLEAR1V(V1)
#define CLEAR4V(V0,V1,V2,V3) \
  CLEAR2V(V0,V1) \
  CLEAR2V(V2,V3)
#define CLEAR8V(V0,V1,V2,V3,V4,V5,V6,V7) \
  CLEAR4V(V0,V1,V2,V3) \
  CLEAR4V(V4,V5,V6,V7)

// Scale vectors.
#define SSCALE1V(V,A,IDX) \
" fmul  v"#V".4s, v"#V".4s, v"#A".s["#IDX"] \n\t"
#define SSCALE2V(V0,V1,A,IDX) \
  SSCALE1V(V0,A,IDX) \
  SSCALE1V(V1,A,IDX)
#define SSCALE4V(V0,V1,V2,V3,A,IDX) \
  SSCALE2V(V0,V1,A,IDX) \
  SSCALE2V(V2,V3,A,IDX)
#define SSCALE8V(V0,V1,V2,V3,V4,V5,V6,V7,A,IDX) \
  SSCALE4V(V0,V1,V2,V3,A,IDX) \
  SSCALE4V(V4,V5,V6,V7,A,IDX)

#define DSCALE1V(V,A,IDX) \
" fmul  v"#V".2d, v"#V".2d, v"#A".d["#IDX"] \n\t"
#define DSCALE2V(V0,V1,A,IDX) \
  DSCALE1V(V0,A,IDX) \
  DSCALE1V(V1,A,IDX)
#define DSCALE4V(V0,V1,V2,V3,A,IDX) \
  DSCALE2V(V0,V1,A,IDX) \
  DSCALE2V(V2,V3,A,IDX)
#define DSCALE8V(V0,V1,V2,V3,V4,V5,V6,V7,A,IDX) \
  DSCALE4V(V0,V1,V2,V3,A,IDX) \
  DSCALE4V(V4,V5,V6,V7,A,IDX)

// Scale-accumulate.
#define SSCALEA1V(D,S,A,IDX) \
" fmla  v"#D".4s, v"#S".4s, v"#A".s["#IDX"] \n\t"
#define SSCALEA2V(D0,D1,S0,S1,A,IDX) \
  SSCALEA1V(D0,S0,A,IDX) \
  SSCALEA1V(D1,S1,A,IDX)
#define SSCALEA4V(D0,D1,D2,D3,S0,S1,S2,S3,A,IDX) \
  SSCALEA2V(D0,D1,S0,S1,A,IDX) \
  SSCALEA2V(D2,D3,S2,S3,A,IDX)
#define SSCALEA8V(D0,D1,D2,D3,D4,D5,D6,D7,S0,S1,S2,S3,S4,S5,S6,S7,A,IDX) \
  SSCALEA4V(D0,D1,D2,D3,S0,S1,S2,S3,A,IDX) \
  SSCALEA4V(D4,D5,D6,D7,S4,S5,S6,S7,A,IDX)

#define DSCALEA1V(D,S,A,IDX) \
" fmla  v"#D".2d, v"#S".2d, v"#A".d["#IDX"] \n\t"
#define DSCALEA2V(D0,D1,S0,S1,A,IDX) \
  DSCALEA1V(D0,S0,A,IDX) \
  DSCALEA1V(D1,S1,A,IDX)
#define DSCALEA4V(D0,D1,D2,D3,S0,S1,S2,S3,A,IDX) \
  DSCALEA2V(D0,D1,S0,S1,A,IDX) \
  DSCALEA2V(D2,D3,S2,S3,A,IDX)
#define DSCALEA8V(D0,D1,D2,D3,D4,D5,D6,D7,S0,S1,S2,S3,S4,S5,S6,S7,A,IDX) \
  DSCALEA4V(D0,D1,D2,D3,S0,S1,S2,S3,A,IDX) \
  DSCALEA4V(D4,D5,D6,D7,S4,S5,S6,S7,A,IDX)

// Load one line.
#define DLOAD1V(V,ADDR,SHIFT) \
" ldr   q"#V", ["#ADDR", #"#SHIFT"] \n\t"
#define DLOAD2V(V0,V1,ADDR,SHIFT) \
  DLOAD1V(V0,ADDR,SHIFT) \
  DLOAD1V(V1,ADDR,SHIFT+16)
#define DLOAD4V(V0,V1,V2,V3,ADDR,SHIFT) \
  DLOAD2V(V0,V1,ADDR,SHIFT) \
  DLOAD2V(V2,V3,ADDR,SHIFT+32)
#define SLOAD1V DLOAD1V
#define SLOAD2V DLOAD2V
#define SLOAD4V DLOAD4V

// Generic: load one line.
#define SLOAD1V_GATHER_ELMFWD(V,ADDR,INC) \
" ld1   {v"#V".s}[0], ["#ADDR"], "#INC" \n\t" \
" ld1   {v"#V".s}[1], ["#ADDR"], "#INC" \n\t" \
" ld1   {v"#V".s}[2], ["#ADDR"], "#INC" \n\t" \
" ld1   {v"#V".s}[3], ["#ADDR"], "#INC" \n\t"
#define DLOAD1V_GATHER_ELMFWD(V,ADDR,INC) \
" ld1   {v"#V".d}[0], ["#ADDR"], "#INC" \n\t" \
" ld1   {v"#V".d}[1], ["#ADDR"], "#INC" \n\t"

// Store one line.
#define DSTORE1V(V,ADDR,SHIFT) \
" str   q"#V", ["#ADDR", #"#SHIFT"] \n\t"
#define DSTORE2V(V0,V1,ADDR,SHIFT) \
  DSTORE1V(V0,ADDR,SHIFT) \
  DSTORE1V(V1,ADDR,SHIFT+16)
#define DSTORE4V(V0,V1,V2,V3,ADDR,SHIFT) \
  DSTORE2V(V0,V1,ADDR,SHIFT) \
  DSTORE2V(V2,V3,ADDR,SHIFT+32)
#define SSTORE1V DSTORE1V
#define SSTORE2V DSTORE2V
#define SSTORE4V DSTORE4V

// Generic: store one line.
#define SSTORE1V_SCATTER_ELMFWD(V,ADDR,INC) \
" st1   {v"#V".s}[0], ["#ADDR"], "#INC" \n\t" \
" st1   {v"#V".s}[1], ["#ADDR"], "#INC" \n\t" \
" st1   {v"#V".s}[2], ["#ADDR"], "#INC" \n\t" \
" st1   {v"#V".s}[3], ["#ADDR"], "#INC" \n\t"
#define DSTORE1V_SCATTER_ELMFWD(V,ADDR,INC) \
" st1   {v"#V".d}[0], ["#ADDR"], "#INC" \n\t" \
" st1   {v"#V".d}[1], ["#ADDR"], "#INC" \n\t"


