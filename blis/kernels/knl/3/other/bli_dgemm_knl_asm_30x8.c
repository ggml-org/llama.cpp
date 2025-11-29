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
   AS IS AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
   OF TEXAS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
   OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"
#include <assert.h>

#include "bli_avx512_macros.h"

#define UNROLL_K 32

#define SCATTER_PREFETCH_C 1

#define PREFETCH_A_L2 0
#define PREFETCH_B_L2 0
#define L2_PREFETCH_DIST 64

#define A_L1_PREFETCH_DIST 18
#define B_L1_PREFETCH_DIST 18

#define LOOP_ALIGN ALIGN16

#define UPDATE_C_FOUR_ROWS(R1,R2,R3,R4) \
\
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VFMADD231PD(ZMM(R1), ZMM(1), MEM(RCX      )) \
    VFMADD231PD(ZMM(R2), ZMM(1), MEM(RCX,RAX,1)) \
    VFMADD231PD(ZMM(R3), ZMM(1), MEM(RCX,RAX,2)) \
    VFMADD231PD(ZMM(R4), ZMM(1), MEM(RCX,RDI,1)) \
    VMOVUPD(MEM(RCX      ), ZMM(R1)) \
    VMOVUPD(MEM(RCX,RAX,1), ZMM(R2)) \
    VMOVUPD(MEM(RCX,RAX,2), ZMM(R3)) \
    VMOVUPD(MEM(RCX,RDI,1), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RAX,4))

#define UPDATE_C_TWO_ROWS(R1,R2) \
\
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VFMADD231PD(ZMM(R1), ZMM(1), MEM(RCX      )) \
    VFMADD231PD(ZMM(R2), ZMM(1), MEM(RCX,RAX,1)) \
    VMOVUPD(MEM(RCX      ), ZMM(R1)) \
    VMOVUPD(MEM(RCX,RAX,1), ZMM(R2)) \
    LEA(RCX, MEM(RCX,RAX,2))

#define UPDATE_C_BZ_FOUR_ROWS(R1,R2,R3,R4) \
\
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VMOVUPD(MEM(RCX      ), ZMM(R1)) \
    VMOVUPD(MEM(RCX,RAX,1), ZMM(R2)) \
    VMOVUPD(MEM(RCX,RAX,2), ZMM(R3)) \
    VMOVUPD(MEM(RCX,RDI,1), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RAX,4))

#define UPDATE_C_BZ_TWO_ROWS(R1,R2) \
\
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMOVUPD(MEM(RCX      ), ZMM(R1)) \
    VMOVUPD(MEM(RCX,RAX,1), ZMM(R2)) \
    LEA(RCX, MEM(RCX,RAX,2))

#define UPDATE_C_ROW_SCATTERED(NUM) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    VGATHERDPD(ZMM(1) MASK_K(1), MEM(RCX,YMM(0),8)) \
    VFMADD231PD(ZMM(NUM), ZMM(1), MEM_1TO8(RBX)) \
    VSCATTERDPD(MEM(RCX,YMM(0),8) MASK_K(2), ZMM(NUM)) \
    ADD(RCX, RAX)

#define UPDATE_C_BZ_ROW_SCATTERED(NUM) \
\
    KXNORW(K(1), K(0), K(0)) \
    VSCATTERDPD(MEM(RCX,YMM(0),8) MASK_K(1), ZMM(NUM)) \
    ADD(RCX, RAX)

#define PREFETCH_A_L1_1(n) PREFETCH(0, MEM(RAX,(A_L1_PREFETCH_DIST+n)*32*8))
#define PREFETCH_A_L1_2(n) PREFETCH(0, MEM(RAX,(A_L1_PREFETCH_DIST+n)*32*8+64))
#define PREFETCH_A_L1_3(n) PREFETCH(0, MEM(RAX,(A_L1_PREFETCH_DIST+n)*32*8+128))
#define PREFETCH_A_L1_4(n) PREFETCH(0, MEM(RAX,(A_L1_PREFETCH_DIST+n)*32*8+192))

#if PREFETCH_A_L2
#undef PREFETCH_A_L2

#define PREFETCH_A_L2(n) \
\
    PREFETCH(1, MEM(RAX,(L2_PREFETCH_DIST+n)*32*8)) \
    PREFETCH(1, MEM(RAX,(L2_PREFETCH_DIST+n)*32*8+64)) \
    PREFETCH(1, MEM(RAX,(L2_PREFETCH_DIST+n)*32*8+128)) \
    PREFETCH(1, MEM(RAX,(L2_PREFETCH_DIST+n)*32*8+192))

#else
#undef PREFETCH_A_L2
#define PREFETCH_A_L2(...)
#endif

#define PREFETCH_B_L1(n) PREFETCH(0, MEM(RBX,(B_L1_PREFETCH_DIST+n)*8*8))

#if PREFETCH_B_L2
#undef PREFETCH_B_L2

#define PREFETCH_B_L2(n) PREFETCH(1, MEM(RBX,(L2_PREFETCH_DIST+n)*8*8))

#else
#undef PREFETCH_B_L2
#define PREFETCH_B_L2(...)
#endif

#define PREFETCH_C_L1_1
#define PREFETCH_C_L1_2
#define PREFETCH_C_L1_3
#define PREFETCH_C_L1_4

//
// n: index in unrolled loop
//
// a: ZMM register to load into
// b: ZMM register to read from
//
// ...: addressing for A, except for offset
//
#define SUBITER(n,a,b,...) \
\
        PREFETCH_A_L2(n) \
\
        VMOVAPD(ZMM(a), MEM(RBX,(n+1)*64)) \
        VFMADD231PD(ZMM( 2), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+ 0)*8)) \
        VFMADD231PD(ZMM( 3), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+ 1)*8)) \
        VFMADD231PD(ZMM( 4), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+ 2)*8)) \
        PREFETCH_A_L1_1(n) \
        VFMADD231PD(ZMM( 5), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+ 3)*8)) \
        VFMADD231PD(ZMM( 6), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+ 4)*8)) \
        VFMADD231PD(ZMM( 7), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+ 5)*8)) \
        PREFETCH_C_L1_1 \
        VFMADD231PD(ZMM( 8), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+ 6)*8)) \
        VFMADD231PD(ZMM( 9), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+ 7)*8)) \
        VFMADD231PD(ZMM(10), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+ 8)*8)) \
        PREFETCH_A_L1_2(n) \
        VFMADD231PD(ZMM(11), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+ 9)*8)) \
        VFMADD231PD(ZMM(12), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+10)*8)) \
        VFMADD231PD(ZMM(13), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+11)*8)) \
        PREFETCH_C_L1_2 \
        VFMADD231PD(ZMM(14), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+12)*8)) \
        VFMADD231PD(ZMM(15), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+13)*8)) \
        VFMADD231PD(ZMM(16), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+14)*8)) \
        PREFETCH_A_L1_3(n) \
        VFMADD231PD(ZMM(17), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+15)*8)) \
        VFMADD231PD(ZMM(18), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+16)*8)) \
        VFMADD231PD(ZMM(19), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+17)*8)) \
        PREFETCH_C_L1_3 \
        VFMADD231PD(ZMM(20), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+18)*8)) \
        VFMADD231PD(ZMM(21), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+19)*8)) \
        VFMADD231PD(ZMM(22), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+20)*8)) \
        PREFETCH_A_L1_4(n) \
        VFMADD231PD(ZMM(23), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+21)*8)) \
        VFMADD231PD(ZMM(24), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+22)*8)) \
        VFMADD231PD(ZMM(25), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+23)*8)) \
        PREFETCH_C_L1_4 \
        VFMADD231PD(ZMM(26), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+24)*8)) \
        VFMADD231PD(ZMM(27), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+25)*8)) \
        VFMADD231PD(ZMM(28), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+26)*8)) \
        PREFETCH_B_L1(n) \
        VFMADD231PD(ZMM(29), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+27)*8)) \
        VFMADD231PD(ZMM(30), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+28)*8)) \
        VFMADD231PD(ZMM(31), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*32+29)*8)) \
        PREFETCH_B_L2(n)

//This is an array used for the scatter/gather instructions.
extern int32_t offsets[32];

//#define MONITORS
//#define LOOPMON
void bli_dgemm_knl_asm_30x8
     (
       dim_t               k,
       double*    restrict alpha,
       double*    restrict a,
       double*    restrict b,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c, inc_t cs_c,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    const int32_t * offsetPtr = &offsets[0];
    uint64_t k64 = k;
    
    __asm__ volatile
    (

    VPXORD(ZMM(2), ZMM(2), ZMM(2)) //clear out registers
    VMOVAPS(ZMM( 3), ZMM(2))
    VMOVAPS(ZMM( 4), ZMM(2))
    VMOVAPS(ZMM( 5), ZMM(2))
    VMOVAPS(ZMM( 6), ZMM(2))
    VMOVAPS(ZMM( 7), ZMM(2))
    VMOVAPS(ZMM( 8), ZMM(2))
    VMOVAPS(ZMM( 9), ZMM(2))   MOV(R12, VAR(rs_c))
    VMOVAPS(ZMM(10), ZMM(2))   MOV(RSI, VAR(k)) //loop index
    VMOVAPS(ZMM(11), ZMM(2))   MOV(RAX, VAR(a)) //load address of a
    VMOVAPS(ZMM(12), ZMM(2))   MOV(RBX, VAR(b)) //load address of b
    VMOVAPS(ZMM(13), ZMM(2))   MOV(RCX, VAR(c)) //load address of c
    VMOVAPS(ZMM(14), ZMM(2))   VMOVAPD(ZMM(0), MEM(RBX)) //pre-load b
    VMOVAPS(ZMM(15), ZMM(2))   MOV(RDI, VAR(offsetPtr))
    VMOVAPS(ZMM(16), ZMM(2))
    VMOVAPS(ZMM(17), ZMM(2))
    VMOVAPS(ZMM(18), ZMM(2))
    VMOVAPS(ZMM(19), ZMM(2))   LEA(R13, MEM(R12,R12,2))
    VMOVAPS(ZMM(20), ZMM(2))   LEA(R14, MEM(R12,R12,4))
    VMOVAPS(ZMM(21), ZMM(2))   LEA(R15, MEM(R13,R12,4))
    VMOVAPS(ZMM(22), ZMM(2))
    VMOVAPS(ZMM(23), ZMM(2))
    VMOVAPS(ZMM(24), ZMM(2))
    VMOVAPS(ZMM(25), ZMM(2))   MOV(R8, IMM(4*32*8))     //offset for 4 iterations
    VMOVAPS(ZMM(26), ZMM(2))   LEA(R9, MEM(R8,R8,2))    //*3
    VMOVAPS(ZMM(27), ZMM(2))   LEA(R10, MEM(R8,R8,4))   //*5
    VMOVAPS(ZMM(28), ZMM(2))   LEA(R11, MEM(R9,R8,4))   //*7
    VMOVAPS(ZMM(29), ZMM(2))
    VMOVAPS(ZMM(30), ZMM(2))
    VMOVAPS(ZMM(31), ZMM(2))

    SUB(RSI, IMM(38))
    JLE(TAIL)

    //prefetch C into L2
#if SCATTER_PREFETCH_C
    VPBROADCASTD(ZMM(0), R12D)
    VPBROADCASTD(ZMM(1), R12D)
    VPMULLD(ZMM(0), ZMM(0), MEM(RDI))
    VPMULLD(ZMM(1), ZMM(1), MEM(RDI,64))
    ADD(RSI, IMM(30))
    KXNORW(K(1), K(0), K(0))
    KSHIFTRW(K(2), K(1), IMM(2))
    VSCATTERPFDPS(1, MEM(RCX,ZMM(0),8) MASK_K(1))
    VSCATTERPFDPS(1, MEM(RCX,ZMM(1),8) MASK_K(2))
    VMOVAPD(ZMM(0), MEM(RBX))
#else
    PREFETCHW1(MEM(RCX      ))
    SUBITER( 0,1,0,RAX      )
    PREFETCHW1(MEM(RCX,R12,1))
    SUBITER( 1,0,1,RAX      )
    PREFETCHW1(MEM(RCX,R12,2))
    SUBITER( 2,1,0,RAX      )
    PREFETCHW1(MEM(RCX,R13,1))
    SUBITER( 3,0,1,RAX      )
    PREFETCHW1(MEM(RCX,R12,4))
    SUBITER( 4,1,0,RAX,R8, 1)
    PREFETCHW1(MEM(RCX,R14,1))
    SUBITER( 5,0,1,RAX,R8, 1)
    PREFETCHW1(MEM(RCX,R13,2))
    SUBITER( 6,1,0,RAX,R8, 1)
    PREFETCHW1(MEM(RCX,R15,1))
    LEA(RDX, MEM(RCX,R12,8))
    SUBITER( 7,0,1,RAX,R8, 1)

    PREFETCHW1(MEM(RDX      ))
    SUBITER( 8,1,0,RAX,R8, 2)
    PREFETCHW1(MEM(RDX,R12,1))
    SUBITER( 9,0,1,RAX,R8, 2)
    PREFETCHW1(MEM(RDX,R12,2))
    SUBITER(10,1,0,RAX,R8, 2)
    PREFETCHW1(MEM(RDX,R13,1))
    SUBITER(11,0,1,RAX,R8, 2)
    PREFETCHW1(MEM(RDX,R12,4))
    SUBITER(12,1,0,RAX,R9, 1)
    PREFETCHW1(MEM(RDX,R14,1))
    SUBITER(13,0,1,RAX,R9, 1)
    PREFETCHW1(MEM(RDX,R13,2))
    SUBITER(14,1,0,RAX,R9, 1)
    PREFETCHW1(MEM(RDX,R15,1))
    LEA(RDX, MEM(RDX,R12,8))
    SUBITER(15,0,1,RAX,R9, 1)

    PREFETCHW1(MEM(RDX      ))
    SUBITER(16,1,0,RAX,R8, 4)
    PREFETCHW1(MEM(RDX,R12,1))
    SUBITER(17,0,1,RAX,R8, 4)
    PREFETCHW1(MEM(RDX,R12,2))
    SUBITER(18,1,0,RAX,R8, 4)
    PREFETCHW1(MEM(RDX,R13,1))
    SUBITER(19,0,1,RAX,R8, 4)
    PREFETCHW1(MEM(RDX,R12,4))
    SUBITER(20,1,0,RAX,R10,1)
    PREFETCHW1(MEM(RDX,R14,1))
    SUBITER(21,0,1,RAX,R10,1)
    PREFETCHW1(MEM(RDX,R13,2))
    SUBITER(22,1,0,RAX,R10,1)
    PREFETCHW1(MEM(RDX,R15,1))
    LEA(RDX, MEM(RDX,R12,8))
    SUBITER(23,0,1,RAX,R10,1)

    PREFETCHW1(MEM(RDX      ))
    SUBITER(24,1,0,RAX,R9, 2)
    PREFETCHW1(MEM(RDX,R12,1))
    SUBITER(25,0,1,RAX,R9, 2)
    PREFETCHW1(MEM(RDX,R12,2))
    SUBITER(26,1,0,RAX,R9, 2)
    PREFETCHW1(MEM(RDX,R13,1))
    SUBITER(27,0,1,RAX,R9, 2)
    PREFETCHW1(MEM(RDX,R12,4))
    SUBITER(28,1,0,RAX,R11,1)
    PREFETCHW1(MEM(RDX,R14,1))
    SUBITER(29,0,1,RAX,R11,1)

    ADD(RAX, IMM(30*32*8))
    ADD(RBX, IMM(30* 8*8))
#endif

    MOV(RDI, RSI)
    AND(RDI, IMM(31))
    SAR(RSI, IMM(5))
    JZ(REM_1)

    LOOP_ALIGN
    LABEL(MAIN_LOOP)

        SUBITER( 0,1,0,RAX      )
        SUBITER( 1,0,1,RAX      )
        SUBITER( 2,1,0,RAX      )
        SUBITER( 3,0,1,RAX      )
        SUBITER( 4,1,0,RAX,R8, 1)
        SUBITER( 5,0,1,RAX,R8, 1)
        SUBITER( 6,1,0,RAX,R8, 1)
        SUBITER( 7,0,1,RAX,R8, 1)
        SUBITER( 8,1,0,RAX,R8, 2)
        SUBITER( 9,0,1,RAX,R8, 2)
        SUBITER(10,1,0,RAX,R8, 2)
        SUBITER(11,0,1,RAX,R8, 2)
        SUBITER(12,1,0,RAX,R9, 1)
        SUBITER(13,0,1,RAX,R9, 1)
        SUBITER(14,1,0,RAX,R9, 1)
        SUBITER(15,0,1,RAX,R9, 1)
        SUBITER(16,1,0,RAX,R8, 4)
        SUBITER(17,0,1,RAX,R8, 4)
        SUBITER(18,1,0,RAX,R8, 4)
        SUBITER(19,0,1,RAX,R8, 4)
        SUBITER(20,1,0,RAX,R10,1)
        SUBITER(21,0,1,RAX,R10,1)
        SUBITER(22,1,0,RAX,R10,1)
        SUBITER(23,0,1,RAX,R10,1)
        SUBITER(24,1,0,RAX,R9, 2)
        SUBITER(25,0,1,RAX,R9, 2)
        SUBITER(26,1,0,RAX,R9, 2)
        SUBITER(27,0,1,RAX,R9, 2)
        SUBITER(28,1,0,RAX,R11,1)
        SUBITER(29,0,1,RAX,R11,1)
        SUBITER(30,1,0,RAX,R11,1)
        SUBITER(31,0,1,RAX,R11,1)

        ADD(RAX, IMM(32*32*8))
        ADD(RBX, IMM(32* 8*8))

        SUB(RSI, IMM(1))

    JNZ(MAIN_LOOP)

    LABEL(REM_1)
    SAR1(RDI)
    JNC(REM_2)

    SUBITER(0,1,0,RAX)
    VMOVAPD(ZMM(0), ZMM(1))
    ADD(RAX, IMM(32*8))
    ADD(RBX, IMM( 8*8))

    LABEL(REM_2)
    SAR1(RDI)
    JNC(REM_4)

    SUBITER(0,1,0,RAX)
    SUBITER(1,0,1,RAX)
    ADD(RAX, IMM(2*32*8))
    ADD(RBX, IMM(2* 8*8))

    LABEL(REM_4)
    SAR1(RDI)
    JNC(REM_8)

    SUBITER(0,1,0,RAX)
    SUBITER(1,0,1,RAX)
    SUBITER(2,1,0,RAX)
    SUBITER(3,0,1,RAX)
    ADD(RAX, IMM(4*32*8))
    ADD(RBX, IMM(4* 8*8))

    LABEL(REM_8)
    SAR1(RDI)
    JNC(REM_16)

    SUBITER(0,1,0,RAX     )
    SUBITER(1,0,1,RAX     )
    SUBITER(2,1,0,RAX     )
    SUBITER(3,0,1,RAX     )
    SUBITER(4,1,0,RAX,R8,1)
    SUBITER(5,0,1,RAX,R8,1)
    SUBITER(6,1,0,RAX,R8,1)
    SUBITER(7,0,1,RAX,R8,1)
    ADD(RAX, IMM(8*32*8))
    ADD(RBX, IMM(8* 8*8))

    LABEL(REM_16)
    SAR1(RDI)
    JNC(AFTER_LOOP)

    SUBITER( 0,1,0,RAX      )
    SUBITER( 1,0,1,RAX      )
    SUBITER( 2,1,0,RAX      )
    SUBITER( 3,0,1,RAX      )
    SUBITER( 4,1,0,RAX,R8, 1)
    SUBITER( 5,0,1,RAX,R8, 1)
    SUBITER( 6,1,0,RAX,R8, 1)
    SUBITER( 7,0,1,RAX,R8, 1)
    SUBITER( 8,1,0,RAX,R8, 2)
    SUBITER( 9,0,1,RAX,R8, 2)
    SUBITER(10,1,0,RAX,R8, 2)
    SUBITER(11,0,1,RAX,R8, 2)
    SUBITER(12,1,0,RAX,R9, 1)
    SUBITER(13,0,1,RAX,R9, 1)
    SUBITER(14,1,0,RAX,R9, 1)
    SUBITER(15,0,1,RAX,R9, 1)
    ADD(RAX, IMM(16*32*8))
    ADD(RBX, IMM(16* 8*8))

    LABEL(AFTER_LOOP)

    //prefetch C into L1
#if SCATTER_PREFETCH_C
    MOV(RDI, VAR(offsetPtr))
    VPBROADCASTD(ZMM(0), R12D)
    VPBROADCASTD(ZMM(1), R12D)
    VPMULLD(ZMM(0), ZMM(0), MEM(RDI))
    VPMULLD(ZMM(1), ZMM(1), MEM(RDI,64))
    KXNORW(K(1), K(0), K(0))
    KSHIFTRW(K(2), K(1), IMM(2))
    VSCATTERPFDPS(0, MEM(RCX,ZMM(0),8) MASK_K(1))
    VSCATTERPFDPS(0, MEM(RCX,ZMM(1),8) MASK_K(2))
    VMOVAPD(ZMM(0), MEM(RBX))

    SUBITER(0,1,0,RAX     )
    SUBITER(1,0,1,RAX     )
    SUBITER(2,1,0,RAX     )
    SUBITER(3,0,1,RAX     )
    SUBITER(4,1,0,RAX,R8,1)
    SUBITER(5,0,1,RAX,R8,1)
    SUBITER(6,1,0,RAX,R8,1)
    SUBITER(7,0,1,RAX,R8,1)
#else
#undef PREFETCH_C_L1_1
#undef PREFETCH_C_L1_2
#undef PREFETCH_C_L1_3
#undef PREFETCH_C_L1_4
#define PREFETCH_C_L1_1 PREFETCHW0(MEM(RCX      ))
#define PREFETCH_C_L1_2 PREFETCHW0(MEM(RCX,R12,1))
#define PREFETCH_C_L1_3 PREFETCHW0(MEM(RCX,R12,2))
#define PREFETCH_C_L1_4 PREFETCHW0(MEM(RCX,R13,1))
    SUBITER(0,1,0,RAX     )

#undef PREFETCH_C_L1_1
#undef PREFETCH_C_L1_2
#undef PREFETCH_C_L1_3
#undef PREFETCH_C_L1_4
#define PREFETCH_C_L1_1 PREFETCHW0(MEM(RCX,R12,4))
#define PREFETCH_C_L1_2 PREFETCHW0(MEM(RCX,R14,1))
#define PREFETCH_C_L1_3 PREFETCHW0(MEM(RCX,R13,2))
#define PREFETCH_C_L1_4 PREFETCHW0(MEM(RCX,R15,1))
    SUBITER(1,0,1,RAX     )

    LEA(RDX, MEM(RCX,R12,8))

#undef PREFETCH_C_L1_1
#undef PREFETCH_C_L1_2
#undef PREFETCH_C_L1_3
#undef PREFETCH_C_L1_4
#define PREFETCH_C_L1_1 PREFETCHW0(MEM(RDX      ))
#define PREFETCH_C_L1_2 PREFETCHW0(MEM(RDX,R12,1))
#define PREFETCH_C_L1_3 PREFETCHW0(MEM(RDX,R12,2))
#define PREFETCH_C_L1_4 PREFETCHW0(MEM(RDX,R13,1))
    SUBITER(2,1,0,RAX     )

#undef PREFETCH_C_L1_1
#undef PREFETCH_C_L1_2
#undef PREFETCH_C_L1_3
#undef PREFETCH_C_L1_4
#define PREFETCH_C_L1_1 PREFETCHW0(MEM(RDX,R12,4))
#define PREFETCH_C_L1_2 PREFETCHW0(MEM(RDX,R14,1))
#define PREFETCH_C_L1_3 PREFETCHW0(MEM(RDX,R13,2))
#define PREFETCH_C_L1_4 PREFETCHW0(MEM(RDX,R15,1))
    SUBITER(3,0,1,RAX     )

    LEA(RDX, MEM(RDX,R12,8))

#undef PREFETCH_C_L1_1
#undef PREFETCH_C_L1_2
#undef PREFETCH_C_L1_3
#undef PREFETCH_C_L1_4
#define PREFETCH_C_L1_1 PREFETCHW0(MEM(RDX      ))
#define PREFETCH_C_L1_2 PREFETCHW0(MEM(RDX,R12,1))
#define PREFETCH_C_L1_3 PREFETCHW0(MEM(RDX,R12,2))
#define PREFETCH_C_L1_4 PREFETCHW0(MEM(RDX,R13,1))
    SUBITER(4,1,0,RAX,R8,1)

#undef PREFETCH_C_L1_1
#undef PREFETCH_C_L1_2
#undef PREFETCH_C_L1_3
#undef PREFETCH_C_L1_4
#define PREFETCH_C_L1_1 PREFETCHW0(MEM(RDX,R12,4))
#define PREFETCH_C_L1_2 PREFETCHW0(MEM(RDX,R14,1))
#define PREFETCH_C_L1_3 PREFETCHW0(MEM(RDX,R13,2))
#define PREFETCH_C_L1_4 PREFETCHW0(MEM(RDX,R15,1))
    SUBITER(5,0,1,RAX,R8,1)

    LEA(RDX, MEM(RDX,R12,8))

#undef PREFETCH_C_L1_1
#undef PREFETCH_C_L1_2
#undef PREFETCH_C_L1_3
#undef PREFETCH_C_L1_4
#define PREFETCH_C_L1_1 PREFETCHW0(MEM(RDX      ))
#define PREFETCH_C_L1_2 PREFETCHW0(MEM(RDX,R12,1))
#define PREFETCH_C_L1_3 PREFETCHW0(MEM(RDX,R12,2))
#define PREFETCH_C_L1_4 PREFETCHW0(MEM(RDX,R13,1))
    SUBITER(6,1,0,RAX,R8,1)

#undef PREFETCH_C_L1_1
#undef PREFETCH_C_L1_2
#undef PREFETCH_C_L1_3
#undef PREFETCH_C_L1_4
#define PREFETCH_C_L1_1 PREFETCHW0(MEM(RDX,R12,4))
#define PREFETCH_C_L1_2 PREFETCHW0(MEM(RDX,R14,1))
#define PREFETCH_C_L1_3
#define PREFETCH_C_L1_4
    SUBITER(7,0,1,RAX,R8,1)
#endif

    JMP(POSTACCUM)

    LABEL(TAIL)

    MOV(RDX, RCX)
    ADD(RSI, IMM(38))

    LABEL(TAIL_LOOP)

        PREFETCHW0(MEM(RDX))
        ADD(RDX, R12)

        SUBITER(0,1,0,RAX)
        VMOVAPD(ZMM(0), ZMM(1))
        ADD(RAX, IMM(32*8))
        ADD(RBX, IMM( 8*8))

        SUB(RSI, IMM(1))

    JNZ(TAIL_LOOP)

    LABEL(POSTACCUM)

    MOV(RAX, VAR(alpha))
    MOV(RBX, VAR(beta))
    VBROADCASTSD(ZMM(0), MEM(RAX))
    VBROADCASTSD(ZMM(1), MEM(RBX))

    // Check if C is row stride. If not, jump to the slow scattered update
    MOV(RAX, VAR(rs_c))
    LEA(RAX, MEM(,RAX,8))
    MOV(RBX, VAR(cs_c))
    LEA(RDI, MEM(RAX,RAX,2))
    CMP(RBX, IMM(1))
    JNE(SCATTEREDUPDATE)

    VMOVQ(RDX, XMM(1))
    SAL1(RDX) //shift out sign bit
    JZ(COLSTORBZ)

    UPDATE_C_FOUR_ROWS( 2, 3, 4, 5)
    UPDATE_C_FOUR_ROWS( 6, 7, 8, 9)
    UPDATE_C_FOUR_ROWS(10,11,12,13)
    UPDATE_C_FOUR_ROWS(14,15,16,17)
    UPDATE_C_FOUR_ROWS(18,19,20,21)
    UPDATE_C_FOUR_ROWS(22,23,24,25)
    UPDATE_C_FOUR_ROWS(26,27,28,29)
    UPDATE_C_TWO_ROWS (30,31)

    JMP(END)

    LABEL(COLSTORBZ)

    UPDATE_C_BZ_FOUR_ROWS( 2, 3, 4, 5)
    UPDATE_C_BZ_FOUR_ROWS( 6, 7, 8, 9)
    UPDATE_C_BZ_FOUR_ROWS(10,11,12,13)
    UPDATE_C_BZ_FOUR_ROWS(14,15,16,17)
    UPDATE_C_BZ_FOUR_ROWS(18,19,20,21)
    UPDATE_C_BZ_FOUR_ROWS(22,23,24,25)
    UPDATE_C_BZ_FOUR_ROWS(26,27,28,29)
    UPDATE_C_BZ_TWO_ROWS (30,31)

    JMP(END)

    LABEL(SCATTEREDUPDATE)

    VMULPD(ZMM( 2), ZMM( 2), ZMM(0))
    VMULPD(ZMM( 3), ZMM( 3), ZMM(0))
    VMULPD(ZMM( 4), ZMM( 4), ZMM(0))
    VMULPD(ZMM( 5), ZMM( 5), ZMM(0))
    VMULPD(ZMM( 6), ZMM( 6), ZMM(0))
    VMULPD(ZMM( 7), ZMM( 7), ZMM(0))
    VMULPD(ZMM( 8), ZMM( 8), ZMM(0))
    VMULPD(ZMM( 9), ZMM( 9), ZMM(0))
    VMULPD(ZMM(10), ZMM(10), ZMM(0))
    VMULPD(ZMM(11), ZMM(11), ZMM(0))
    VMULPD(ZMM(12), ZMM(12), ZMM(0))
    VMULPD(ZMM(13), ZMM(13), ZMM(0))
    VMULPD(ZMM(14), ZMM(14), ZMM(0))
    VMULPD(ZMM(15), ZMM(15), ZMM(0))
    VMULPD(ZMM(16), ZMM(16), ZMM(0))
    VMULPD(ZMM(17), ZMM(17), ZMM(0))
    VMULPD(ZMM(18), ZMM(18), ZMM(0))
    VMULPD(ZMM(19), ZMM(19), ZMM(0))
    VMULPD(ZMM(20), ZMM(20), ZMM(0))
    VMULPD(ZMM(21), ZMM(21), ZMM(0))
    VMULPD(ZMM(22), ZMM(22), ZMM(0))
    VMULPD(ZMM(23), ZMM(23), ZMM(0))
    VMULPD(ZMM(24), ZMM(24), ZMM(0))
    VMULPD(ZMM(25), ZMM(25), ZMM(0))
    VMULPD(ZMM(26), ZMM(26), ZMM(0))
    VMULPD(ZMM(27), ZMM(27), ZMM(0))
    VMULPD(ZMM(28), ZMM(28), ZMM(0))
    VMULPD(ZMM(29), ZMM(29), ZMM(0))
    VMULPD(ZMM(30), ZMM(30), ZMM(0))
    VMULPD(ZMM(31), ZMM(31), ZMM(0))

    VMOVQ(RDX, XMM(1))

    /* Note that this ignores the upper 32 bits in cs_c */
    MOV(RDI, VAR(offsetPtr))
    VPBROADCASTD(ZMM(0), EBX)
    VPMULLD(ZMM(0), ZMM(0), MEM(RDI))
    MOV(RBX, VAR(beta))

    SAL1(RDX) //shift out sign bit
    JZ(SCATTERBZ)

    UPDATE_C_ROW_SCATTERED( 2)
    UPDATE_C_ROW_SCATTERED( 3)
    UPDATE_C_ROW_SCATTERED( 4)
    UPDATE_C_ROW_SCATTERED( 5)
    UPDATE_C_ROW_SCATTERED( 6)
    UPDATE_C_ROW_SCATTERED( 7)
    UPDATE_C_ROW_SCATTERED( 8)
    UPDATE_C_ROW_SCATTERED( 9)
    UPDATE_C_ROW_SCATTERED(10)
    UPDATE_C_ROW_SCATTERED(11)
    UPDATE_C_ROW_SCATTERED(12)
    UPDATE_C_ROW_SCATTERED(13)
    UPDATE_C_ROW_SCATTERED(14)
    UPDATE_C_ROW_SCATTERED(15)
    UPDATE_C_ROW_SCATTERED(16)
    UPDATE_C_ROW_SCATTERED(17)
    UPDATE_C_ROW_SCATTERED(18)
    UPDATE_C_ROW_SCATTERED(19)
    UPDATE_C_ROW_SCATTERED(20)
    UPDATE_C_ROW_SCATTERED(21)
    UPDATE_C_ROW_SCATTERED(22)
    UPDATE_C_ROW_SCATTERED(23)
    UPDATE_C_ROW_SCATTERED(24)
    UPDATE_C_ROW_SCATTERED(25)
    UPDATE_C_ROW_SCATTERED(26)
    UPDATE_C_ROW_SCATTERED(27)
    UPDATE_C_ROW_SCATTERED(28)
    UPDATE_C_ROW_SCATTERED(29)
    UPDATE_C_ROW_SCATTERED(30)
    UPDATE_C_ROW_SCATTERED(31)

    JMP(END)

    LABEL(SCATTERBZ)

    UPDATE_C_BZ_ROW_SCATTERED( 2)
    UPDATE_C_BZ_ROW_SCATTERED( 3)
    UPDATE_C_BZ_ROW_SCATTERED( 4)
    UPDATE_C_BZ_ROW_SCATTERED( 5)
    UPDATE_C_BZ_ROW_SCATTERED( 6)
    UPDATE_C_BZ_ROW_SCATTERED( 7)
    UPDATE_C_BZ_ROW_SCATTERED( 8)
    UPDATE_C_BZ_ROW_SCATTERED( 9)
    UPDATE_C_BZ_ROW_SCATTERED(10)
    UPDATE_C_BZ_ROW_SCATTERED(11)
    UPDATE_C_BZ_ROW_SCATTERED(12)
    UPDATE_C_BZ_ROW_SCATTERED(13)
    UPDATE_C_BZ_ROW_SCATTERED(14)
    UPDATE_C_BZ_ROW_SCATTERED(15)
    UPDATE_C_BZ_ROW_SCATTERED(16)
    UPDATE_C_BZ_ROW_SCATTERED(17)
    UPDATE_C_BZ_ROW_SCATTERED(18)
    UPDATE_C_BZ_ROW_SCATTERED(19)
    UPDATE_C_BZ_ROW_SCATTERED(20)
    UPDATE_C_BZ_ROW_SCATTERED(21)
    UPDATE_C_BZ_ROW_SCATTERED(22)
    UPDATE_C_BZ_ROW_SCATTERED(23)
    UPDATE_C_BZ_ROW_SCATTERED(24)
    UPDATE_C_BZ_ROW_SCATTERED(25)
    UPDATE_C_BZ_ROW_SCATTERED(26)
    UPDATE_C_BZ_ROW_SCATTERED(27)
    UPDATE_C_BZ_ROW_SCATTERED(28)
    UPDATE_C_BZ_ROW_SCATTERED(29)
    UPDATE_C_BZ_ROW_SCATTERED(30)
    UPDATE_C_BZ_ROW_SCATTERED(31)

    LABEL(END)

    : // output operands
    : // input operands
      [k]         "m" (k64),
      [a]         "m" (a),
      [b]         "m" (b),
      [alpha]     "m" (alpha),
      [beta]      "m" (beta),
      [c]         "m" (c),
      [rs_c]      "m" (rs_c),
      [cs_c]      "m" (cs_c),
      [offsetPtr] "m" (offsetPtr)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11", "r12",
      "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
      "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
      "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
      "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
      "zmm30", "zmm31", "memory"
    );
}
