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

#define UNROLL_K 8

#define SCATTER_PREFETCH_AB 0
#define SCATTER_PREFETCH_C 1

#define PREFETCH_A_L2 0
#define PREFETCH_B_L2 0
#define L2_PREFETCH_DIST 64

#define A_L1_PREFETCH_DIST 32
#define B_L1_PREFETCH_DIST 12

#define C_MIN_L2_ITERS 64 //C is not prefetched into L2 for k <= this
#define C_L1_ITERS 8 //number of iterations before the end to prefetch C into L1
                      //make sure there is an unrolled MAIN_LOOP_X for this number

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

#define UPDATE_C_ROW_SCATTERED(NUM) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    VMULPD(ZMM(NUM), ZMM(NUM), ZMM(0)) \
    VGATHERDPD(ZMM(3) MASK_K(1), MEM(RCX,YMM(2),8)) \
    VFMADD231PD(ZMM(NUM), ZMM(3), ZMM(1)) \
    VSCATTERDPD(MEM(RCX,YMM(2),8) MASK_K(2), ZMM(NUM)) \
    ADD(RCX, RAX)

#define UPDATE_C_BZ_ROW_SCATTERED(NUM) \
\
    KXNORW(K(1), K(0), K(0)) \
    VMULPD(ZMM(NUM), ZMM(NUM), ZMM(0)) \
    VSCATTERDPD(MEM(RCX,YMM(2),8) MASK_K(1), ZMM(NUM)) \
    ADD(RCX, RAX)

#define PREFETCH_B_L1_1(n) PREFETCH(0, MEM(RBX,(B_L1_PREFETCH_DIST+n)*24*8))
#define PREFETCH_B_L1_2(n) PREFETCH(0, MEM(RBX,(B_L1_PREFETCH_DIST+n)*24*8+64))
#define PREFETCH_B_L1_3(n) PREFETCH(0, MEM(RBX,(B_L1_PREFETCH_DIST+n)*24*8+128))

#if PREFETCH_B_L2
#undef PREFETCH_B_L2

#define PREFETCH_B_L2(n) \
\
    PREFETCH(1, MEM(RBX,(L2_PREFETCH_DIST+n)*24*8)) \
    PREFETCH(1, MEM(RBX,(L2_PREFETCH_DIST+n)*24*8+64)) \
    PREFETCH(1, MEM(RBX,(L2_PREFETCH_DIST+n)*24*8+128))

#else
#undef PREFETCH_B_L2
#define PREFETCH_B_L2(...)
#endif

#define PREFETCH_A_L1(n) PREFETCH(0, MEM(RAX,(A_L1_PREFETCH_DIST+n)*8*8))

#if PREFETCH_A_L2
#undef PREFETCH_A_L2

#define PREFETCH_A_L2(n) PREFETCH(1, MEM(RAX,(L2_PREFETCH_DIST+n)*8*8))

#else
#undef PREFETCH_A_L2
#define PREFETCH_A_L2(...)
#endif

#if SCATTER_PREFETCH_AB
#undef SCATTER_PREFETCH_AB
#undef PREFETCH_B_L1_1
#undef PREFETCH_B_L1_2
#undef PREFETCH_B_L1_3
#undef PREFETCH_A_L1

#define SCATTER_PREFETCH_AB(n) \
\
    KXNORW(K(1), K(0), K(0)) \
    VGATHERPFDPS(0, MEM(RBX,ZMM(4),8,((3*n  )*16+3*B_L1_PREFETCH_DIST)*64) MASK_K(1)) \
    KXNORW(K(2), K(0), K(0)) \
    VGATHERPFDPS(0, MEM(RBX,ZMM(4),8,((3*n+1)*16+3*B_L1_PREFETCH_DIST)*64) MASK_K(2)) \
    KXNORW(K(3), K(0), K(0)) \
    VGATHERPFDPS(0, MEM(RBX,ZMM(4),8,((3*n+2)*16+3*B_L1_PREFETCH_DIST)*64) MASK_K(3)) \
    KXNORW(K(4), K(0), K(0)) \
    VGATHERPFDPS(0, MEM(RAX,ZMM(4),8,(   n   *16+  A_L1_PREFETCH_DIST)*64) MASK_K(4))

#define PREFETCH_B_L1_1(...)
#define PREFETCH_B_L1_2(...)
#define PREFETCH_B_L1_3(...)
#define PREFETCH_A_L1(...)

#else
#undef SCATTER_PREFETCH_AB

#define SCATTER_PREFETCH_AB(...)

#endif

//
// n: index in unrolled loop (for prefetching offsets)
//
// a: ZMM register to load into
// b: ZMM register to read from
//
// ...: addressing for B, except for offset
//
#define SUBITER(n,a,b,...) \
\
        PREFETCH_B_L2(n) \
\
        VMOVAPD(ZMM(a), MEM(RAX,(n+1)*64)) \
        VFMADD231PD(ZMM( 8), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 0)*8)) \
        VFMADD231PD(ZMM( 9), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 1)*8)) \
        VFMADD231PD(ZMM(10), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 2)*8)) \
        VFMADD231PD(ZMM(11), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 3)*8)) \
        PREFETCH_B_L1_1(n) \
        VFMADD231PD(ZMM(12), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 4)*8)) \
        VFMADD231PD(ZMM(13), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 5)*8)) \
        VFMADD231PD(ZMM(14), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 6)*8)) \
        VFMADD231PD(ZMM(15), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 7)*8)) \
        PREFETCH_B_L1_2(n) \
        VFMADD231PD(ZMM(16), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 8)*8)) \
        VFMADD231PD(ZMM(17), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 9)*8)) \
        VFMADD231PD(ZMM(18), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+10)*8)) \
        VFMADD231PD(ZMM(19), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+11)*8)) \
        PREFETCH_B_L1_3(n) \
        VFMADD231PD(ZMM(20), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+12)*8)) \
        VFMADD231PD(ZMM(21), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+13)*8)) \
        VFMADD231PD(ZMM(22), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+14)*8)) \
        VFMADD231PD(ZMM(23), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+15)*8)) \
        PREFETCH_A_L1(n) \
        VFMADD231PD(ZMM(24), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+16)*8)) \
        VFMADD231PD(ZMM(25), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+17)*8)) \
        VFMADD231PD(ZMM(26), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+18)*8)) \
        VFMADD231PD(ZMM(27), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+19)*8)) \
        PREFETCH_A_L2(n) \
        VFMADD231PD(ZMM(28), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+20)*8)) \
        VFMADD231PD(ZMM(29), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+21)*8)) \
        VFMADD231PD(ZMM(30), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+22)*8)) \
        VFMADD231PD(ZMM(31), ZMM(b), MEM_1TO8(__VA_ARGS__,((n%%4)*24+23)*8))

#define TAIL_LOOP(NAME) \
\
    LOOP_ALIGN \
    LABEL(NAME) \
\
        SUBITER(0,1,0,RBX) \
\
        VMOVAPD(ZMM(0), ZMM(1)) \
\
        LEA(RBX, MEM(RBX,24*8)) \
        LEA(RAX, MEM(RAX, 8*8)) \
\
        SUB(RDI, IMM(1)) \
\
    JNZ(NAME)

#define MAIN_LOOP_1(NAME) \
\
    LOOP_ALIGN \
    LABEL(NAME##_LOOP) \
\
        SUBITER(0,1,0,RBX) \
\
        VMOVAPD(ZMM(0), ZMM(1)) \
\
        LEA(RBX, MEM(RBX,24*8)) \
        LEA(RAX, MEM(RAX, 8*8)) \
\
        SUB(RSI, IMM(1)) \
\
    JNZ(NAME##_LOOP)

#define MAIN_LOOP_2(NAME) \
\
    MOV(RDI, RSI) \
    AND(RDI, IMM(1)) \
    SAR1(RSI) \
    JZ(NAME##_TAIL) \
\
    LOOP_ALIGN \
    LABEL(NAME##_LOOP) \
\
        SUBITER(0,1,0,RBX) \
        SUBITER(1,0,1,RBX) \
\
        LEA(RBX, MEM(RBX,2*24*8)) \
        LEA(RAX, MEM(RAX,2* 8*8)) \
\
        SUB(RSI, IMM(1)) \
\
    JNZ(NAME##_LOOP) \
\
    TEST(RDI, RDI) \
    JZ(NAME##_DONE) \
\
    LABEL(NAME##_TAIL) \
\
    SUBITER(0,1,0,RBX) \
\
    VMOVAPD(ZMM(0), ZMM(1)) \
\
    LEA(RBX, MEM(RBX,24*8)) \
    LEA(RAX, MEM(RAX, 8*8)) \
\
    LABEL(NAME##_DONE)

#define MAIN_LOOP_4(NAME) \
\
    MOV(RDI, RSI) \
    AND(RDI, IMM(3)) \
    SAR(RSI, IMM(2)) \
    JZ(NAME##_TAIL) \
\
    LOOP_ALIGN \
    LABEL(NAME##_LOOP) \
\
        SUBITER(0,1,0,RBX) \
        SUBITER(1,0,1,RBX) \
        SUBITER(2,1,0,RBX) \
        SUBITER(3,0,1,RBX) \
\
        LEA(RBX, MEM(RBX,4*24*8)) \
        LEA(RAX, MEM(RAX,4* 8*8)) \
\
        SUB(RSI, IMM(1)) \
\
    JNZ(NAME##_LOOP) \
\
    TEST(RDI, RDI) \
    JZ(NAME##_DONE) \
\
    TAIL_LOOP(NAME##_TAIL) \
\
    LABEL(NAME##_DONE)

#define MAIN_LOOP_8(NAME) \
\
    MOV(RDI, RSI) \
    AND(RDI, IMM(7)) \
    SAR(RSI, IMM(3)) \
    JZ(NAME##_TAIL) \
\
    LOOP_ALIGN \
    LABEL(NAME##_LOOP) \
\
        SUBITER(0,1,0,RBX) \
        SUBITER(1,0,1,RBX) \
        SUBITER(2,1,0,RBX) \
        SUBITER(3,0,1,RBX) \
        SUBITER(4,1,0,RBX,R8,1) \
        SUBITER(5,0,1,RBX,R8,1) \
        SUBITER(6,1,0,RBX,R8,1) \
        SUBITER(7,0,1,RBX,R8,1) \
\
        LEA(RBX, MEM(RBX,8*24*8)) \
        LEA(RAX, MEM(RAX,8* 8*8)) \
\
        SUB(RSI, IMM(1)) \
\
    JNZ(NAME##_LOOP) \
\
    TEST(RDI, RDI) \
    JZ(NAME##_DONE) \
\
    TAIL_LOOP(NAME##_TAIL) \
\
    LABEL(NAME##_DONE)

#define MAIN_LOOP_16(NAME) \
\
    MOV(RDI, RSI) \
    AND(RDI, IMM(15)) \
    SAR(RSI, IMM(4)) \
    JZ(NAME##_TAIL) \
\
    LOOP_ALIGN \
    LABEL(NAME##_LOOP) \
\
        SCATTER_PREFETCH_AB(0) \
\
        SUBITER( 0,1,0,RBX) \
        SUBITER( 1,0,1,RBX) \
        SUBITER( 2,1,0,RBX) \
        SUBITER( 3,0,1,RBX) \
        SUBITER( 4,1,0,RBX,R8,1) \
        SUBITER( 5,0,1,RBX,R8,1) \
        SUBITER( 6,1,0,RBX,R8,1) \
        SUBITER( 7,0,1,RBX,R8,1) \
        SUBITER( 8,1,0,RBX,R8,2) \
        SUBITER( 9,0,1,RBX,R8,2) \
        SUBITER(10,1,0,RBX,R8,2) \
        SUBITER(11,0,1,RBX,R8,2) \
        SUBITER(12,1,0,RBX,R9,1) \
        SUBITER(13,0,1,RBX,R9,1) \
        SUBITER(14,1,0,RBX,R9,1) \
        SUBITER(15,0,1,RBX,R9,1) \
\
        LEA(RBX, MEM(RBX,16*24*8)) \
        LEA(RAX, MEM(RAX,16* 8*8)) \
\
        SUB(RSI, IMM(1)) \
\
    JNZ(NAME##_LOOP) \
\
    TEST(RDI, RDI) \
    JZ(NAME##_DONE) \
\
    SCATTER_PREFETCH_AB(0) \
\
    TAIL_LOOP(NAME##_TAIL) \
\
    LABEL(NAME##_DONE)

#define MAIN_LOOP_32(NAME) \
\
    MOV(RDI, RSI) \
    AND(RDI, IMM(31)) \
    SAR(RSI, IMM(5)) \
    JZ(NAME##_TAIL) \
\
    LOOP_ALIGN \
    LABEL(NAME##_LOOP) \
\
        SCATTER_PREFETCH_AB(0) \
\
        SUBITER( 0,1,0,RBX) \
        SUBITER( 1,0,1,RBX) \
        SUBITER( 2,1,0,RBX) \
        SUBITER( 3,0,1,RBX) \
        SUBITER( 4,1,0,RBX,R8,1) \
        SUBITER( 5,0,1,RBX,R8,1) \
        SUBITER( 6,1,0,RBX,R8,1) \
        SUBITER( 7,0,1,RBX,R8,1) \
        SUBITER( 8,1,0,RBX,R8,2) \
        SUBITER( 9,0,1,RBX,R8,2) \
        SUBITER(10,1,0,RBX,R8,2) \
        SUBITER(11,0,1,RBX,R8,2) \
        SUBITER(12,1,0,RBX,R9,1) \
        SUBITER(13,0,1,RBX,R9,1) \
        SUBITER(14,1,0,RBX,R9,1) \
        SUBITER(15,0,1,RBX,R9,1) \
\
        SCATTER_PREFETCH_AB(1) \
\
        SUBITER(16,1,0,RBX,R8,4) \
        SUBITER(17,0,1,RBX,R8,4) \
        SUBITER(18,1,0,RBX,R8,4) \
        SUBITER(19,0,1,RBX,R8,4) \
        SUBITER(20,1,0,RBX,R10,1) \
        SUBITER(21,0,1,RBX,R10,1) \
        SUBITER(22,1,0,RBX,R10,1) \
        SUBITER(23,0,1,RBX,R10,1) \
        SUBITER(24,1,0,RBX,R9,2) \
        SUBITER(25,0,1,RBX,R9,2) \
        SUBITER(26,1,0,RBX,R9,2) \
        SUBITER(27,0,1,RBX,R9,2) \
        SUBITER(28,1,0,RBX,R11,1) \
        SUBITER(29,0,1,RBX,R11,1) \
        SUBITER(30,1,0,RBX,R11,1) \
        SUBITER(31,0,1,RBX,R11,1) \
\
        LEA(RBX, MEM(RBX,32*24*8)) \
        LEA(RAX, MEM(RAX,32* 8*8)) \
\
        SUB(RSI, IMM(1)) \
\
    JNZ(NAME##_LOOP) \
\
    TEST(RDI, RDI) \
    JZ(NAME##_DONE) \
\
    SCATTER_PREFETCH_AB(0) \
    SCATTER_PREFETCH_AB(1) \
\
    TAIL_LOOP(NAME##_TAIL) \
\
    LABEL(NAME##_DONE)

#define LOOP_K_(M,K) M##K
#define LOOP_K(M,K,NAME) LOOP_K_(M,K)(NAME)

#define MAIN_LOOP_L2 LOOP_K(MAIN_LOOP_,UNROLL_K,MAIN_LOOP_L2)
#define MAIN_LOOP_L1 LOOP_K(MAIN_LOOP_,C_L1_ITERS,MAIN_LOOP_L1)

//This is an array used for the scatter/gather instructions.
extern int32_t offsets[24];

//#define MONITORS
//#define LOOPMON
void bli_dgemm_knl_asm_8x24
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
    const double * a_next = bli_auxinfo_next_a( data );
    const double * b_next = bli_auxinfo_next_b( data );

    const int32_t * offsetPtr = &offsets[0];

    uint64_t k64 = k;

#ifdef MONITORS
    int toph, topl, both, botl, midl, midh, mid2l, mid2h;
#endif
#ifdef LOOPMON
    int tlooph, tloopl, blooph, bloopl;
#endif

    __asm__ volatile
    (
#ifdef MONITORS
    RDTSC
    MOV(VAR(topl), EAX)
    MOV(VAR(toph), EDX)
#endif

    VPXORD(ZMM(8), ZMM(8), ZMM(8)) //clear out registers
    VMOVAPS(ZMM( 9), ZMM(8))
    VMOVAPS(ZMM(10), ZMM(8))   MOV(RSI, VAR(k)) //loop index
    VMOVAPS(ZMM(11), ZMM(8))   MOV(RAX, VAR(a)) //load address of a
    VMOVAPS(ZMM(12), ZMM(8))   MOV(RBX, VAR(b)) //load address of b
    VMOVAPS(ZMM(13), ZMM(8))   MOV(RCX, VAR(c)) //load address of c
    VMOVAPS(ZMM(14), ZMM(8))   VMOVAPD(ZMM(0), MEM(RAX)) //pre-load a
    VMOVAPS(ZMM(15), ZMM(8))   MOV(RDI, VAR(offsetPtr))
    VMOVAPS(ZMM(16), ZMM(8))   VMOVAPS(ZMM(4), MEM(RDI))
#if SCATTER_PREFETCH_C
    VMOVAPS(ZMM(17), ZMM(8))
    VMOVAPS(ZMM(18), ZMM(8))
    VMOVAPS(ZMM(19), ZMM(8))   VBROADCASTSS(ZMM(5), VAR(cs_c))
    VMOVAPS(ZMM(20), ZMM(8))
    VMOVAPS(ZMM(21), ZMM(8))   VPMULLD(ZMM(2), ZMM(4), ZMM(5))
    VMOVAPS(ZMM(22), ZMM(8))   VMOVAPS(YMM(3), MEM(RDI,64))
    VMOVAPS(ZMM(23), ZMM(8))   VPMULLD(YMM(3), YMM(3), YMM(5))
#else
    VMOVAPS(ZMM(17), ZMM(8))   MOV(R12, VAR(cs_c))
    VMOVAPS(ZMM(18), ZMM(8))   LEA(R13, MEM(R12,R12,2))
    VMOVAPS(ZMM(19), ZMM(8))   LEA(R14, MEM(R12,R12,4))
    VMOVAPS(ZMM(20), ZMM(8))   LEA(R15, MEM(R13,R12,4))
    VMOVAPS(ZMM(21), ZMM(8))   LEA(RDX, MEM(RCX,R12,8))
    VMOVAPS(ZMM(22), ZMM(8))   LEA(RDI, MEM(RDX,R12,8))
    VMOVAPS(ZMM(23), ZMM(8))
#endif
    VMOVAPS(ZMM(24), ZMM(8))   VPSLLD(ZMM(4), ZMM(4), IMM(3))
    VMOVAPS(ZMM(25), ZMM(8))   MOV(R8, IMM(4*24*8))     //offset for 4 iterations
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

    //need 0+... to satisfy preprocessor
    CMP(RSI, IMM(0+C_MIN_L2_ITERS))
    JLE(PREFETCH_C_L1)

    SUB(RSI, IMM(0+C_L1_ITERS))

    //prefetch C into L2
#if SCATTER_PREFETCH_C
    KXNORW(K(1), K(0), K(0))
    KXNORW(K(2), K(0), K(0))
    VSCATTERPFDPS(1, MEM(RCX,ZMM(2),8) MASK_K(1))
    VSCATTERPFDPD(1, MEM(RCX,YMM(3),8) MASK_K(2))
#else
    PREFETCH(1, MEM(RCX      ))
    PREFETCH(1, MEM(RCX,R12,1))
    PREFETCH(1, MEM(RCX,R12,2))
    PREFETCH(1, MEM(RCX,R13,1))
    PREFETCH(1, MEM(RCX,R12,4))
    PREFETCH(1, MEM(RCX,R14,1))
    PREFETCH(1, MEM(RCX,R13,2))
    PREFETCH(1, MEM(RCX,R15,1))
    PREFETCH(1, MEM(RDX      ))
    PREFETCH(1, MEM(RDX,R12,1))
    PREFETCH(1, MEM(RDX,R12,2))
    PREFETCH(1, MEM(RDX,R13,1))
    PREFETCH(1, MEM(RDX,R12,4))
    PREFETCH(1, MEM(RDX,R14,1))
    PREFETCH(1, MEM(RDX,R13,2))
    PREFETCH(1, MEM(RDX,R15,1))
    PREFETCH(1, MEM(RDI      ))
    PREFETCH(1, MEM(RDI,R12,1))
    PREFETCH(1, MEM(RDI,R12,2))
    PREFETCH(1, MEM(RDI,R13,1))
    PREFETCH(1, MEM(RDI,R12,4))
    PREFETCH(1, MEM(RDI,R14,1))
    PREFETCH(1, MEM(RDI,R13,2))
    PREFETCH(1, MEM(RDI,R15,1))
#endif

    MAIN_LOOP_L2

    MOV(RSI, IMM(0+C_L1_ITERS))

    LABEL(PREFETCH_C_L1)

    //prefetch C into L1
#if SCATTER_PREFETCH_C
    KXNORW(K(1), K(0), K(0))
    KXNORW(K(2), K(0), K(0))
    VSCATTERPFDPS(0, MEM(RCX,ZMM(2),8) MASK_K(1))
    VSCATTERPFDPD(0, MEM(RCX,YMM(3),8) MASK_K(2))
#else
    PREFETCH(0, MEM(RCX      ))
    PREFETCH(0, MEM(RCX,R12,1))
    PREFETCH(0, MEM(RCX,R12,2))
    PREFETCH(0, MEM(RCX,R13,1))
    PREFETCH(0, MEM(RCX,R12,4))
    PREFETCH(0, MEM(RCX,R14,1))
    PREFETCH(0, MEM(RCX,R13,2))
    PREFETCH(0, MEM(RCX,R15,1))
    PREFETCH(0, MEM(RDX      ))
    PREFETCH(0, MEM(RDX,R12,1))
    PREFETCH(0, MEM(RDX,R12,2))
    PREFETCH(0, MEM(RDX,R13,1))
    PREFETCH(0, MEM(RDX,R12,4))
    PREFETCH(0, MEM(RDX,R14,1))
    PREFETCH(0, MEM(RDX,R13,2))
    PREFETCH(0, MEM(RDX,R15,1))
    PREFETCH(0, MEM(RDI      ))
    PREFETCH(0, MEM(RDI,R12,1))
    PREFETCH(0, MEM(RDI,R12,2))
    PREFETCH(0, MEM(RDI,R13,1))
    PREFETCH(0, MEM(RDI,R12,4))
    PREFETCH(0, MEM(RDI,R14,1))
    PREFETCH(0, MEM(RDI,R13,2))
    PREFETCH(0, MEM(RDI,R15,1))
#endif

    MAIN_LOOP_L1

    LABEL(POSTACCUM)

#ifdef MONITORS
    RDTSC
    MOV(VAR(mid2l), EAX)
    MOV(VAR(mid2h), EDX)
#endif

    MOV(RAX, VAR(alpha))
    MOV(RBX, VAR(beta))
    VBROADCASTSD(ZMM(0), MEM(RAX))
    VBROADCASTSD(ZMM(1), MEM(RBX))

    // Check if C is column stride. If not, jump to the slow scattered update
    MOV(RAX, VAR(cs_c))
    LEA(RAX, MEM(,RAX,8))
    MOV(RBX, VAR(rs_c))
    LEA(RDI, MEM(RAX,RAX,2))
    CMP(RBX, IMM(1))
    JNE(SCATTEREDUPDATE)

    VMOVQ(RDX, XMM(1))
    SAL1(RDX) //shift out sign bit
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

    JMP(END)

    LABEL(SCATTEREDUPDATE)

    MOV(RDI, VAR(offsetPtr))
    VMOVAPS(ZMM(2), MEM(RDI))
    /* Note that this ignores the upper 32 bits in rs_c */
    VPBROADCASTD(ZMM(3), EBX)
    VPMULLD(ZMM(2), ZMM(3), ZMM(2))

    VMOVQ(RDX, XMM(1))
    SAL1(RDX) //shift out sign bit
    JZ(SCATTERBZ)

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

#ifdef MONITORS
    RDTSC
    MOV(VAR(botl), EAX)
    MOV(VAR(both), EDX)
#endif
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
      [k]         "m" (k64),
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
    );

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
