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

#define BLIS_ASM_SYNTAX_INTEL
#include "bli_x86_asm_macros.h"

#define UNROLL_K 32

#define SCATTER_PREFETCH_C 1

#define PREFETCH_A_L2 0
#define PREFETCH_B_L2 0
#define L2_PREFETCH_DIST 64

#define A_L1_PREFETCH_DIST 36
#define B_L1_PREFETCH_DIST 18

#define LOOP_ALIGN ALIGN16

#define UPDATE_C_FOUR_ROWS(R1,R2,R3,R4) \
\
    VMULPS(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPS(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPS(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPS(ZMM(R4), ZMM(R4), ZMM(0)) \
    VFMADD231PS(ZMM(R1), ZMM(1), MEM(RCX      )) \
    VFMADD231PS(ZMM(R2), ZMM(1), MEM(RCX,RAX,1)) \
    VFMADD231PS(ZMM(R3), ZMM(1), MEM(RCX,RAX,2)) \
    VFMADD231PS(ZMM(R4), ZMM(1), MEM(RCX,RDI,1)) \
    VMOVUPS(MEM(RCX      ), ZMM(R1)) \
    VMOVUPS(MEM(RCX,RAX,1), ZMM(R2)) \
    VMOVUPS(MEM(RCX,RAX,2), ZMM(R3)) \
    VMOVUPS(MEM(RCX,RDI,1), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RAX,4))

#define UPDATE_C_BZ_FOUR_ROWS(R1,R2,R3,R4) \
\
    VMULPS(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPS(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPS(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPS(ZMM(R4), ZMM(R4), ZMM(0)) \
    VMOVUPS(MEM(RCX      ), ZMM(R1)) \
    VMOVUPS(MEM(RCX,RAX,1), ZMM(R2)) \
    VMOVUPS(MEM(RCX,RAX,2), ZMM(R3)) \
    VMOVUPS(MEM(RCX,RDI,1), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RAX,4))

#define UPDATE_C_ROW_SCATTERED(NUM) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    VMULPS(ZMM(NUM), ZMM(NUM), ZMM(0)) \
    VGATHERDPS(ZMM(3) MASK_K(1), MEM(RCX,ZMM(2),4)) \
    VFMADD231PS(ZMM(NUM), ZMM(3), ZMM(1)) \
    VSCATTERDPS(MEM(RCX,ZMM(2),4) MASK_K(2), ZMM(NUM)) \
    ADD(RCX, RAX)

#define UPDATE_C_BZ_ROW_SCATTERED(NUM) \
\
    KXNORW(K(1), K(0), K(0)) \
    VMULPS(ZMM(NUM), ZMM(NUM), ZMM(0)) \
    VSCATTERDPS(MEM(RCX,ZMM(2),4) MASK_K(1), ZMM(NUM)) \
    ADD(RCX, RAX)

#define PREFETCH_A_L1_1(n) PREFETCH(0, MEM(RAX,(A_L1_PREFETCH_DIST+n)*24*4))
#define PREFETCH_A_L1_2(n) PREFETCH(0, MEM(RAX,(A_L1_PREFETCH_DIST+n)*24*4+64))

#if PREFETCH_A_L2
#undef PREFETCH_A_L2

#define PREFETCH_A_L2(n) \
\
    PREFETCH(1, MEM(RAX,(L2_PREFETCH_DIST+n)*24*4)) \
    PREFETCH(1, MEM(RAX,(L2_PREFETCH_DIST+n)*24*4+64))

#else
#undef PREFETCH_A_L2
#define PREFETCH_A_L2(...)
#endif

#define PREFETCH_B_L1(n) PREFETCH(0, MEM(RBX,(B_L1_PREFETCH_DIST+n)*16*4))

#if PREFETCH_B_L2
#undef PREFETCH_B_L2

#define PREFETCH_B_L2(n) PREFETCH(1, MEM(RBX,(L2_PREFETCH_DIST+n)*16*4))

#else
#undef PREFETCH_B_L2
#define PREFETCH_B_L2(...)
#endif

#define PREFETCH_C_L1_1
#define PREFETCH_C_L1_2
#define PREFETCH_C_L1_3

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
        VMOVAPS(ZMM(a), MEM(RBX,(n+1)*64)) \
        VFMADD231PS(ZMM( 8), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+ 0)*4)) \
        VFMADD231PS(ZMM( 9), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+ 1)*4)) \
        VFMADD231PS(ZMM(10), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+ 2)*4)) \
        PREFETCH_A_L1_1(n) \
        VFMADD231PS(ZMM(11), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+ 3)*4)) \
        VFMADD231PS(ZMM(12), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+ 4)*4)) \
        VFMADD231PS(ZMM(13), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+ 5)*4)) \
        PREFETCH_C_L1_1 \
        VFMADD231PS(ZMM(14), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+ 6)*4)) \
        VFMADD231PS(ZMM(15), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+ 7)*4)) \
        VFMADD231PS(ZMM(16), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+ 8)*4)) \
        PREFETCH_A_L1_2(n) \
        VFMADD231PS(ZMM(17), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+ 9)*4)) \
        VFMADD231PS(ZMM(18), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+10)*4)) \
        VFMADD231PS(ZMM(19), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+11)*4)) \
        PREFETCH_C_L1_2 \
        VFMADD231PS(ZMM(20), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+12)*4)) \
        VFMADD231PS(ZMM(21), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+13)*4)) \
        VFMADD231PS(ZMM(22), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+14)*4)) \
        PREFETCH_C_L1_3 \
        VFMADD231PS(ZMM(23), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+15)*4)) \
        VFMADD231PS(ZMM(24), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+16)*4)) \
        VFMADD231PS(ZMM(25), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+17)*4)) \
        PREFETCH_B_L1(n) \
        VFMADD231PS(ZMM(26), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+18)*4)) \
        VFMADD231PS(ZMM(27), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+19)*4)) \
        VFMADD231PS(ZMM(28), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+20)*4)) \
        PREFETCH_B_L2(n) \
        VFMADD231PS(ZMM(29), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+21)*4)) \
        VFMADD231PS(ZMM(30), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+22)*4)) \
        VFMADD231PS(ZMM(31), ZMM(b), MEM_1TO16(__VA_ARGS__,((n%%4)*24+23)*4))

//This is an array used for the scatter/gather instructions.
static int32_t offsets[32] __attribute__((aligned(64))) =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
     16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};

//#define MONITORS
//#define LOOPMON
void bli_sgemm_knl_asm_24x16
     (
             dim_t      m,
             dim_t      n,
             dim_t      k_,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c_, inc_t cs_c_,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
    (void)data;
    (void)cntx;

    const double* a_next = bli_auxinfo_next_a( data );
    const double* b_next = bli_auxinfo_next_b( data );

    int32_t* offsetPtr = &offsets[0];
    int64_t k = k_;
    int64_t rs_c = rs_c_;
    int64_t cs_c = cs_c_;

    GEMM_UKR_SETUP_CT( s, 24, 16, true );

#ifdef MONITORS
    int toph, topl, both, botl, midl, midh, mid2l, mid2h;
#endif
#ifdef LOOPMON
    int tlooph, tloopl, blooph, bloopl;
#endif

    BEGIN_ASM()

#ifdef MONITORS
    RDTSC
    MOV(VAR(topl), EAX)
    MOV(VAR(toph), EDX)
#endif

    VPXORD(ZMM(8), ZMM(8), ZMM(8)) //clear out registers
    VMOVAPS(ZMM( 9), ZMM(8))   MOV(R12, VAR(rs_c))
    VMOVAPS(ZMM(10), ZMM(8))   MOV(RSI, VAR(k)) //loop index
    VMOVAPS(ZMM(11), ZMM(8))   MOV(RAX, VAR(a)) //load address of a
    VMOVAPS(ZMM(12), ZMM(8))   MOV(RBX, VAR(b)) //load address of b
    VMOVAPS(ZMM(13), ZMM(8))   MOV(RCX, VAR(c)) //load address of c
    VMOVAPS(ZMM(14), ZMM(8))   VMOVAPD(ZMM(0), MEM(RBX)) //pre-load b
    VMOVAPS(ZMM(15), ZMM(8))   MOV(RDI, VAR(offsetPtr))
    VMOVAPS(ZMM(16), ZMM(8))   VMOVAPS(ZMM(4), MEM(RDI))
#if SCATTER_PREFETCH_C
    VMOVAPS(ZMM(17), ZMM(8))
    VMOVAPS(ZMM(18), ZMM(8))
    VMOVAPS(ZMM(19), ZMM(8))   VBROADCASTSS(ZMM(5), VAR(rs_c))
    VMOVAPS(ZMM(20), ZMM(8))
    VMOVAPS(ZMM(21), ZMM(8))   VPMULLD(ZMM(2), ZMM(4), ZMM(5))
    VMOVAPS(ZMM(22), ZMM(8))   VMOVAPS(YMM(3), MEM(RDI,64))
    VMOVAPS(ZMM(23), ZMM(8))   VPMULLD(YMM(3), YMM(3), YMM(5))
#else
    VMOVAPS(ZMM(17), ZMM(8))
    VMOVAPS(ZMM(18), ZMM(8))   LEA(R13, MEM(R12,R12,2))
    VMOVAPS(ZMM(19), ZMM(8))   LEA(R14, MEM(R12,R12,4))
    VMOVAPS(ZMM(20), ZMM(8))   LEA(R15, MEM(R13,R12,4))
    VMOVAPS(ZMM(21), ZMM(8))
    VMOVAPS(ZMM(22), ZMM(8))
    VMOVAPS(ZMM(23), ZMM(8))
#endif
    VMOVAPS(ZMM(24), ZMM(8))   VPSLLD(ZMM(4), ZMM(4), IMM(2))
    VMOVAPS(ZMM(25), ZMM(8))   MOV(R8, IMM(4*24*4))     //offset for 4 iterations
    VMOVAPS(ZMM(26), ZMM(8))   LEA(R9, MEM(R8,R8,2))    //*3
    VMOVAPS(ZMM(27), ZMM(8))   LEA(R10, MEM(R8,R8,4))   //*5
    VMOVAPS(ZMM(28), ZMM(8))   LEA(R11, MEM(R9,R8,4))   //*7
    VMOVAPS(ZMM(29), ZMM(8))
    VMOVAPS(ZMM(30), ZMM(8))
    VMOVAPS(ZMM(31), ZMM(8))

#ifdef MONITORS
    RDTSC
    MOV(VAR(midl), EAX)
    MOV(VAR(midh), EDX)
#endif

    SUB(RSI, IMM(32))
    JLE(TAIL)

    //prefetch C into L2
#if SCATTER_PREFETCH_C
    ADD(RSI, IMM(24))
    KXNORW(K(1), K(0), K(0))
    KXNORW(K(2), K(0), K(0))
    VSCATTERPFDPS(1, MEM(RCX,ZMM(2),8) MASK_K(1))
    VSCATTERPFDPD(1, MEM(RCX,YMM(3),8) MASK_K(2))
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
    SUBITER( 7,0,1,RAX,R8, 1)

    LEA(RDX, MEM(RCX,R12,8))

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
    SUBITER(15,0,1,RAX,R9, 1)

    LEA(RDI, MEM(RDX,R12,8))

    PREFETCHW1(MEM(RDI      ))
    SUBITER(16,1,0,RAX,R8, 4)
    PREFETCHW1(MEM(RDI,R12,1))
    SUBITER(17,0,1,RAX,R8, 4)
    PREFETCHW1(MEM(RDI,R12,2))
    SUBITER(18,1,0,RAX,R8, 4)
    PREFETCHW1(MEM(RDI,R13,1))
    SUBITER(19,0,1,RAX,R8, 4)
    PREFETCHW1(MEM(RDI,R12,4))
    SUBITER(20,1,0,RAX,R10,1)
    PREFETCHW1(MEM(RDI,R14,1))
    SUBITER(21,0,1,RAX,R10,1)
    PREFETCHW1(MEM(RDI,R13,2))
    SUBITER(22,1,0,RAX,R10,1)
    PREFETCHW1(MEM(RDI,R15,1))
    SUBITER(23,0,1,RAX,R10,1)

    ADD(RAX, IMM(24*24*4))
    ADD(RBX, IMM(24*16*4))
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

        ADD(RAX, IMM(32*24*4))
        ADD(RBX, IMM(32*16*4))

        SUB(RSI, IMM(1))

    JNZ(MAIN_LOOP)

    LABEL(REM_1)
    SAR(RDI)
    JNC(REM_2)

    SUBITER(0,1,0,RAX)
    VMOVAPD(ZMM(0), ZMM(1))
    ADD(RAX, IMM(24*4))
    ADD(RBX, IMM(16*4))

    LABEL(REM_2)
    SAR(RDI)
    JNC(REM_4)

    SUBITER(0,1,0,RAX)
    SUBITER(1,0,1,RAX)
    ADD(RAX, IMM(2*24*4))
    ADD(RBX, IMM(2*16*4))

    LABEL(REM_4)
    SAR(RDI)
    JNC(REM_8)

    SUBITER(0,1,0,RAX)
    SUBITER(1,0,1,RAX)
    SUBITER(2,1,0,RAX)
    SUBITER(3,0,1,RAX)
    ADD(RAX, IMM(4*24*4))
    ADD(RBX, IMM(4*16*4))

    LABEL(REM_8)
    SAR(RDI)
    JNC(REM_16)

    SUBITER(0,1,0,RAX     )
    SUBITER(1,0,1,RAX     )
    SUBITER(2,1,0,RAX     )
    SUBITER(3,0,1,RAX     )
    SUBITER(4,1,0,RAX,R8,1)
    SUBITER(5,0,1,RAX,R8,1)
    SUBITER(6,1,0,RAX,R8,1)
    SUBITER(7,0,1,RAX,R8,1)
    ADD(RAX, IMM(8*24*4))
    ADD(RBX, IMM(8*16*4))

    LABEL(REM_16)
    SAR(RDI)
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
    ADD(RAX, IMM(16*24*4))
    ADD(RBX, IMM(16*16*4))

    LABEL(AFTER_LOOP)

    //prefetch C into L1
#if SCATTER_PREFETCH_C
    KXNORW(K(1), K(0), K(0))
    KXNORW(K(2), K(0), K(0))
    VSCATTERPFDPS(0, MEM(RCX,ZMM(2),8) MASK_K(1))
    VSCATTERPFDPD(0, MEM(RCX,YMM(3),8) MASK_K(2))

    SUBITER(0,1,0,RAX     )
    SUBITER(1,0,1,RAX     )
    SUBITER(2,1,0,RAX     )
    SUBITER(3,0,1,RAX     )
    SUBITER(4,1,0,RAX,R8,1)
    SUBITER(5,0,1,RAX,R8,1)
    SUBITER(6,1,0,RAX,R8,1)
    SUBITER(7,0,1,RAX,R8,1)
#else

    LEA(RDX, MEM(RCX,R12,8))
    LEA(RDI, MEM(RDX,R12,8))

#undef PREFETCH_C_L1_1
#undef PREFETCH_C_L1_2
#undef PREFETCH_C_L1_3
#define PREFETCH_C_L1_1 PREFETCHW0(MEM(RCX      ))
#define PREFETCH_C_L1_2 PREFETCHW0(MEM(RCX,R12,1))
#define PREFETCH_C_L1_3 PREFETCHW0(MEM(RCX,R12,2))
    SUBITER(0,1,0,RAX     )
#undef PREFETCH_C_L1_1
#undef PREFETCH_C_L1_2
#undef PREFETCH_C_L1_3
#define PREFETCH_C_L1_1 PREFETCHW0(MEM(RCX,R13,1))
#define PREFETCH_C_L1_2 PREFETCHW0(MEM(RCX,R12,4))
#define PREFETCH_C_L1_3 PREFETCHW0(MEM(RCX,R14,1))
    SUBITER(1,0,1,RAX     )
#undef PREFETCH_C_L1_1
#undef PREFETCH_C_L1_2
#undef PREFETCH_C_L1_3
#define PREFETCH_C_L1_1 PREFETCHW0(MEM(RCX,R13,2))
#define PREFETCH_C_L1_2 PREFETCHW0(MEM(RCX,R15,1))
#define PREFETCH_C_L1_3 PREFETCHW0(MEM(RDX      ))
    SUBITER(2,1,0,RAX     )
#undef PREFETCH_C_L1_1
#undef PREFETCH_C_L1_2
#undef PREFETCH_C_L1_3
#define PREFETCH_C_L1_1 PREFETCHW0(MEM(RDX,R12,1))
#define PREFETCH_C_L1_2 PREFETCHW0(MEM(RDX,R12,2))
#define PREFETCH_C_L1_3 PREFETCHW0(MEM(RDX,R13,1))
    SUBITER(3,0,1,RAX     )
#undef PREFETCH_C_L1_1
#undef PREFETCH_C_L1_2
#undef PREFETCH_C_L1_3
#define PREFETCH_C_L1_1 PREFETCHW0(MEM(RDX,R12,4))
#define PREFETCH_C_L1_2 PREFETCHW0(MEM(RDX,R14,1))
#define PREFETCH_C_L1_3 PREFETCHW0(MEM(RDX,R13,2))
    SUBITER(4,1,0,RAX,R8,1)
#undef PREFETCH_C_L1_1
#undef PREFETCH_C_L1_2
#undef PREFETCH_C_L1_3
#define PREFETCH_C_L1_1 PREFETCHW0(MEM(RDX,R15,1))
#define PREFETCH_C_L1_2 PREFETCHW0(MEM(RDI      ))
#define PREFETCH_C_L1_3 PREFETCHW0(MEM(RDI,R12,1))
    SUBITER(5,0,1,RAX,R8,1)
#undef PREFETCH_C_L1_1
#undef PREFETCH_C_L1_2
#undef PREFETCH_C_L1_3
#define PREFETCH_C_L1_1 PREFETCHW0(MEM(RDI,R12,2))
#define PREFETCH_C_L1_2 PREFETCHW0(MEM(RDI,R13,1))
#define PREFETCH_C_L1_3 PREFETCHW0(MEM(RDI,R12,4))
    SUBITER(6,1,0,RAX,R8,1)
#undef PREFETCH_C_L1_1
#undef PREFETCH_C_L1_2
#undef PREFETCH_C_L1_3
#define PREFETCH_C_L1_1 PREFETCHW0(MEM(RDI,R14,1))
#define PREFETCH_C_L1_2 PREFETCHW0(MEM(RDI,R13,2))
#define PREFETCH_C_L1_3 PREFETCHW0(MEM(RDI,R15,1))
    SUBITER(7,0,1,RAX,R8,1)
#endif

    JMP(POSTACCUM)

    LABEL(TAIL)

    MOV(RDX, RCX)
    ADD(RSI, IMM(32))
    JZ(POSTACCUM)

    LABEL(TAIL_LOOP)

        PREFETCHW0(MEM(RDX))
        ADD(RDX, R12)

        SUBITER(0,1,0,RAX)
        VMOVAPD(ZMM(0), ZMM(1))
        ADD(RAX, IMM(24*4))
        ADD(RBX, IMM(16*4))

        SUB(RSI, IMM(1))

    JNZ(TAIL_LOOP)

    LABEL(POSTACCUM)

#ifdef MONITORS
    RDTSC
    MOV(VAR(mid2l), EAX)
    MOV(VAR(mid2h), EDX)
#endif

    MOV(RAX, VAR(alpha))
    MOV(RBX, VAR(beta))
    VBROADCASTSS(ZMM(0), MEM(RAX))
    VBROADCASTSS(ZMM(1), MEM(RBX))

    // Check if C is row stride. If not, jump to the slow scattered update
    MOV(RAX, VAR(rs_c))
    LEA(RAX, MEM(,RAX,4))
    LEA(RDI, MEM(RAX,RAX,2))

    VMOVD(EDX, XMM(1))
    SAL(EDX) //shift out sign bit
    JZ(COLSTORBZ)

    UPDATE_C_FOUR_ROWS( 8, 9,10,11)
    UPDATE_C_FOUR_ROWS(12,13,14,15)
    UPDATE_C_FOUR_ROWS(16,17,18,19)
    UPDATE_C_FOUR_ROWS(20,21,22,23)
    UPDATE_C_FOUR_ROWS(24,25,26,27)
    UPDATE_C_FOUR_ROWS(28,29,30,31)

    JMP(END)

    LABEL(COLSTORBZ)

    UPDATE_C_BZ_FOUR_ROWS( 8, 9,10,11)
    UPDATE_C_BZ_FOUR_ROWS(12,13,14,15)
    UPDATE_C_BZ_FOUR_ROWS(16,17,18,19)
    UPDATE_C_BZ_FOUR_ROWS(20,21,22,23)
    UPDATE_C_BZ_FOUR_ROWS(24,25,26,27)
    UPDATE_C_BZ_FOUR_ROWS(28,29,30,31)

    LABEL(END)

#ifdef MONITORS
    RDTSC
    MOV(VAR(botl), EAX)
    MOV(VAR(both), EDX)
#endif

    END_ASM(
    : // output operands
#ifdef MONITORS
      [topl]  "=m" (topl),
      [toph]  "=m" (toph),
      [midl]  "=m" (midl),
      [midh]  "=m" (midh),
      [mid2l] "=m" (mid2l),
      [mid2h] "=m" (mid2h),
      [botl]  "=m" (botl),
      [both]  "=m" (both)
#endif
    : // input operands
      [k]         "m" (k),
      [a]         "m" (a),
      [b]         "m" (b),
      [alpha]     "m" (alpha),
      [beta]      "m" (beta),
      [c]         "m" (c),
      [rs_c]      "m" (rs_c),
      [cs_c]      "m" (cs_c),
      [a_next]    "m" (a_next),
      [b_next]    "m" (b_next),
      [offsetPtr] "m" (offsetPtr)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11", "r12",
      "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
      "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
      "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
      "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
      "zmm30", "zmm31", "memory"
    )

    GEMM_UKR_FLUSH_CT( s );

#ifdef LOOPMON
    printf("looptime = \t%d\n", bloopl - tloopl);
#endif
#ifdef MONITORS
    dim_t top = ((dim_t)toph << 32) | topl;
    dim_t mid = ((dim_t)midh << 32) | midl;
    dim_t mid2 = ((dim_t)mid2h << 32) | mid2l;
    dim_t bot = ((dim_t)both << 32) | botl;
    printf("setup =\t%u\tmain loop =\t%u\tcleanup=\t%u\ttotal=\t%u\n", mid - top, mid2 - mid, bot - mid2, bot - top);
#endif
}
