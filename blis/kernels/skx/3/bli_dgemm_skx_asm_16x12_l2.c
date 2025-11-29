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

#define BLIS_ASM_SYNTAX_INTEL
#include "bli_x86_asm_macros.h"

#define A_L1_PREFETCH_DIST 4 //should be multiple of 2

/*The pointer of B is moved ahead by one iteration of k
before the loop starts.Therefore, prefetching 3 k iterations
ahead*/
#define B_L1_PREFETCH_DIST 4

#define TAIL_NITER 8

#define CACHELINE_SIZE 64 //size of cache line in bytes

/* During each subiteration, prefetching 2 cache lines of B
 * UNROLL factor ahead. 2cache lines = 16 doubles (NR).
 * */
#define PREFETCH_A_L1(n, k) \
    PREFETCH(0, MEM(RAX, A_L1_PREFETCH_DIST*16*8 + (2*n+k)  * CACHELINE_SIZE))

/* Preloading B for the first iteration of the main loop.
     * for subiter(1), subiter(2), and subiter(3) */
#define PREFETCH_B_L1_1ITER \
    PREFETCH(0, MEM(RBX                   )) \
    PREFETCH(0, MEM(RBX,    CACHELINE_SIZE)) \
    PREFETCH(0, MEM(RBX,  2*CACHELINE_SIZE)) \
    PREFETCH(0, MEM(RBX,  3*CACHELINE_SIZE)) \
    PREFETCH(0, MEM(RBX,  4*CACHELINE_SIZE)) \
    PREFETCH(0, MEM(RBX,  5*CACHELINE_SIZE))

#define LOOP_ALIGN ALIGN16

#define UPDATE_C(R1,R2,R3,R4) \
\
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VFMADD231PD(ZMM(R1), ZMM(1), MEM(RCX,0*64)) \
    VFMADD231PD(ZMM(R2), ZMM(1), MEM(RCX,1*64)) \
    VFMADD231PD(ZMM(R3), ZMM(1), MEM(RCX,RAX,1,0*64)) \
    VFMADD231PD(ZMM(R4), ZMM(1), MEM(RCX,RAX,1,1*64)) \
    VMOVUPD(MEM(RCX,0*64), ZMM(R1)) \
    VMOVUPD(MEM(RCX,1*64), ZMM(R2)) \
    VMOVUPD(MEM(RCX,RAX,1,0*64), ZMM(R3)) \
    VMOVUPD(MEM(RCX,RAX,1,1*64), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RAX,2))

#define UPDATE_C_BZ(R1,R2,R3,R4) \
\
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VMOVUPD(MEM(RCX,0*64), ZMM(R1)) \
    VMOVUPD(MEM(RCX,1*64), ZMM(R2)) \
    VMOVUPD(MEM(RCX,RAX,1,0*64), ZMM(R3)) \
    VMOVUPD(MEM(RCX,RAX,1,1*64), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RAX,2))

#define UPDATE_C_ROW_SCATTERED(R1,R2,R3,R4) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VGATHERQPD(ZMM(6) MASK_K(1), MEM(RCX,ZMM(2),1)) \
    VFMADD231PD(ZMM(R1), ZMM(6), ZMM(1)) \
    VSCATTERQPD(MEM(RCX,ZMM(2),1) MASK_K(2), ZMM(R1)) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VGATHERQPD(ZMM(6) MASK_K(1), MEM(RCX,ZMM(3),1)) \
    VFMADD231PD(ZMM(R2), ZMM(6), ZMM(1)) \
    VSCATTERQPD(MEM(RCX,ZMM(3),1) MASK_K(2), ZMM(R2)) \
\
    LEA(RCX, MEM(RCX,RAX,1)) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VGATHERQPD(ZMM(6) MASK_K(1), MEM(RCX,ZMM(2),1)) \
    VFMADD231PD(ZMM(R3), ZMM(6), ZMM(1)) \
    VSCATTERQPD(MEM(RCX,ZMM(2),1) MASK_K(2), ZMM(R3)) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VGATHERQPD(ZMM(6) MASK_K(1), MEM(RCX,ZMM(3),1)) \
    VFMADD231PD(ZMM(R4), ZMM(6), ZMM(1)) \
    VSCATTERQPD(MEM(RCX,ZMM(3),1) MASK_K(2), ZMM(R4)) \
\
    LEA(RCX, MEM(RCX,RAX,1))

#define UPDATE_C_BZ_ROW_SCATTERED(R1,R2,R3,R4) \
\
    KXNORW(K(1), K(0), K(0)) \
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VSCATTERQPD(MEM(RCX,ZMM(2),1) MASK_K(1), ZMM(R1)) \
\
    KXNORW(K(1), K(0), K(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VSCATTERQPD(MEM(RCX,ZMM(3),1) MASK_K(1), ZMM(R2)) \
\
    LEA(RCX, MEM(RCX,RAX,1)) \
\
    KXNORW(K(1), K(0), K(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VSCATTERQPD(MEM(RCX,ZMM(2),1) MASK_K(1), ZMM(R3)) \
\
    KXNORW(K(1), K(0), K(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VSCATTERQPD(MEM(RCX,ZMM(3),1) MASK_K(1), ZMM(R4)) \
\
    LEA(RCX, MEM(RCX,RAX,1))

#ifdef PREFETCH_C_L2
#undef PREFETCH_C_L2
#define PREFETCH_C_L2 \
\
    PREFETCH(1, MEM(RCX,      0*64)) \
    PREFETCH(1, MEM(RCX,      1*64)) \
    \
    PREFETCH(1, MEM(RCX,R12,1,0*64)) \
    PREFETCH(1, MEM(RCX,R12,1,1*64)) \
    \
    PREFETCH(1, MEM(RCX,R12,2,0*64)) \
    PREFETCH(1, MEM(RCX,R12,2,1*64)) \
    \
    PREFETCH(1, MEM(RCX,R13,1,0*64)) \
    PREFETCH(1, MEM(RCX,R13,1,1*64)) \
    \
    PREFETCH(1, MEM(RCX,R12,4,0*64)) \
    PREFETCH(1, MEM(RCX,R12,4,1*64)) \
    \
    PREFETCH(1, MEM(RCX,R14,1,0*64)) \
    PREFETCH(1, MEM(RCX,R14,1,1*64)) \
    \
    PREFETCH(1, MEM(RCX,R13,2,0*64)) \
    PREFETCH(1, MEM(RCX,R13,2,1*64)) \
    \
    PREFETCH(1, MEM(RCX,R15,1,0*64)) \
    PREFETCH(1, MEM(RCX,R15,1,1*64)) \
    \
    PREFETCH(1, MEM(RDX,      0*64)) \
    PREFETCH(1, MEM(RDX,      1*64)) \
    \
    PREFETCH(1, MEM(RDX,R12,1,0*64)) \
    PREFETCH(1, MEM(RDX,R12,1,1*64)) \
    \
    PREFETCH(1, MEM(RDX,R12,2,0*64)) \
    PREFETCH(1, MEM(RDX,R12,2,1*64)) \
    \
    PREFETCH(1, MEM(RDX,R13,1,0*64)) \
    PREFETCH(1, MEM(RDX,R13,1,1*64))

#else
#undef PREFETCH_C_L2
#define PREFETCH_C_L2
#endif


#define PREFETCH_C_L1 \
\
    PREFETCHW0(MEM(RCX,      0*64)) \
    PREFETCHW0(MEM(RCX,      1*64)) \
    PREFETCHW0(MEM(RCX,R12,1,0*64)) \
    PREFETCHW0(MEM(RCX,R12,1,1*64)) \
    PREFETCHW0(MEM(RCX,R12,2,0*64)) \
    PREFETCHW0(MEM(RCX,R12,2,1*64)) \
    PREFETCHW0(MEM(RCX,R13,1,0*64)) \
    PREFETCHW0(MEM(RCX,R13,1,1*64)) \
    PREFETCHW0(MEM(RCX,R12,4,0*64)) \
    PREFETCHW0(MEM(RCX,R12,4,1*64)) \
    PREFETCHW0(MEM(RCX,R14,1,0*64)) \
    PREFETCHW0(MEM(RCX,R14,1,1*64)) \
    PREFETCHW0(MEM(RCX,R13,2,0*64)) \
    PREFETCHW0(MEM(RCX,R13,2,1*64)) \
    PREFETCHW0(MEM(RCX,R15,1,0*64)) \
    PREFETCHW0(MEM(RCX,R15,1,1*64)) \
    PREFETCHW0(MEM(RDX,      0*64)) \
    PREFETCHW0(MEM(RDX,      1*64)) \
    PREFETCHW0(MEM(RDX,R12,1,0*64)) \
    PREFETCHW0(MEM(RDX,R12,1,1*64)) \
    PREFETCHW0(MEM(RDX,R12,2,0*64)) \
    PREFETCHW0(MEM(RDX,R12,2,1*64)) \
    PREFETCHW0(MEM(RDX,R13,1,0*64)) \
    PREFETCHW0(MEM(RDX,R13,1,1*64))

//
// n: index in unrolled loop
//
// a: ZMM register to load into
// b: ZMM register to read from
//
// ...: addressing for A, except for offset
//
#define SUBITER(n) \
\
    PREFETCH_A_L1(n, 0) \
    \
    VBROADCASTSD(ZMM(3), MEM(RBX,(12*n+ 0)*8)) \
    VBROADCASTSD(ZMM(4), MEM(RBX,(12*n+ 1)*8)) \
    VFMADD231PD(ZMM( 8), ZMM(0), ZMM(3)) \
    VFMADD231PD(ZMM( 9), ZMM(1), ZMM(3)) \
    VFMADD231PD(ZMM(10), ZMM(0), ZMM(4)) \
    VFMADD231PD(ZMM(11), ZMM(1), ZMM(4)) \
    \
    VBROADCASTSD(ZMM(3), MEM(RBX,(12*n+ 2)*8)) \
    VBROADCASTSD(ZMM(4), MEM(RBX,(12*n+ 3)*8)) \
    VFMADD231PD(ZMM(12), ZMM(0), ZMM(3)) \
    VFMADD231PD(ZMM(13), ZMM(1), ZMM(3)) \
    VFMADD231PD(ZMM(14), ZMM(0), ZMM(4)) \
    VFMADD231PD(ZMM(15), ZMM(1), ZMM(4)) \
    \
    VBROADCASTSD(ZMM(3), MEM(RBX,(12*n+ 4)*8)) \
    VBROADCASTSD(ZMM(4), MEM(RBX,(12*n+ 5)*8)) \
    VFMADD231PD(ZMM(16), ZMM(0), ZMM(3)) \
    VFMADD231PD(ZMM(17), ZMM(1), ZMM(3)) \
    VFMADD231PD(ZMM(18), ZMM(0), ZMM(4)) \
    VFMADD231PD(ZMM(19), ZMM(1), ZMM(4)) \
    \
    PREFETCH_A_L1(n, 1) \
    \
    VBROADCASTSD(ZMM(3), MEM(RBX,(12*n+ 6)*8)) \
    VBROADCASTSD(ZMM(4), MEM(RBX,(12*n+ 7)*8)) \
    VFMADD231PD(ZMM(20), ZMM(0), ZMM(3)) \
    VFMADD231PD(ZMM(21), ZMM(1), ZMM(3)) \
    VFMADD231PD(ZMM(22), ZMM(0), ZMM(4)) \
    VFMADD231PD(ZMM(23), ZMM(1), ZMM(4)) \
    \
    VBROADCASTSD(ZMM(3), MEM(RBX,(12*n+ 8)*8)) \
    VBROADCASTSD(ZMM(4), MEM(RBX,(12*n+ 9)*8)) \
    VFMADD231PD(ZMM(24), ZMM(0), ZMM(3)) \
    VFMADD231PD(ZMM(25), ZMM(1), ZMM(3)) \
    VFMADD231PD(ZMM(26), ZMM(0), ZMM(4)) \
    VFMADD231PD(ZMM(27), ZMM(1), ZMM(4)) \
    \
    VBROADCASTSD(ZMM(3), MEM(RBX,(12*n+10)*8)) \
    VBROADCASTSD(ZMM(4), MEM(RBX,(12*n+11)*8)) \
    VFMADD231PD(ZMM(28), ZMM(0), ZMM(3)) \
    VFMADD231PD(ZMM(29), ZMM(1), ZMM(3)) \
    VFMADD231PD(ZMM(30), ZMM(0), ZMM(4)) \
    VFMADD231PD(ZMM(31), ZMM(1), ZMM(4)) \
    \
    VMOVAPD(ZMM(0), MEM(RAX,(16*n+0)*8)) \
    VMOVAPD(ZMM(1), MEM(RAX,(16*n+8)*8))

#if 0
//This is an array used for the scatter/gather instructions.
static int64_t offsets[16] __attribute__((aligned(64))) =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15};
#endif


void bli_dgemm_skx_asm_16x12_l2
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

    int64_t k = k_;
    int64_t rs_c = rs_c_;
    int64_t cs_c = cs_c_;

    GEMM_UKR_SETUP_CT( d, 16, 12, false );

    BEGIN_ASM()

    VXORPD(YMM(8), YMM(8), YMM(8)) //clear out registers
    VMOVAPD(YMM( 7), YMM(8))
    VMOVAPD(YMM( 9), YMM(8))
    VMOVAPD(YMM(10), YMM(8))   MOV(RSI, VAR(k)) //loop index
    VMOVAPD(YMM(11), YMM(8))   MOV(RAX, VAR(a)) //load address of a
    VMOVAPD(YMM(12), YMM(8))   MOV(RBX, VAR(b)) //load address of b
    VMOVAPD(YMM(13), YMM(8))   MOV(RCX, VAR(c)) //load address of c
    VMOVAPD(YMM(14), YMM(8))
    VMOVAPD(YMM(15), YMM(8))   VMOVAPD(ZMM(0), MEM(RAX, 0*8)) //pre-load a
    VMOVAPD(YMM(16), YMM(8))   VMOVAPD(ZMM(1), MEM(RAX, 8*8)) //pre-load a
    VMOVAPD(YMM(17), YMM(8))
    VMOVAPD(YMM(18), YMM(8))
    VMOVAPD(YMM(19), YMM(8))   MOV(R12, VAR(cs_c))      //cs_c
    VMOVAPD(YMM(20), YMM(8))   LEA(R13, MEM(R12,R12,2)) //*3
    VMOVAPD(YMM(21), YMM(8))   LEA(R14, MEM(R12,R12,4)) //*5
    VMOVAPD(YMM(22), YMM(8))   LEA(R15, MEM(R14,R12,2)) //*7
    VMOVAPD(YMM(23), YMM(8))   LEA(RDX, MEM(RCX,R12,8)) //c + 8*cs_c
    VMOVAPD(YMM(24), YMM(8))
    VMOVAPD(YMM(25), YMM(8))   MOV(R8, IMM(16*8)) //mr*sizeof(double)
    VMOVAPD(YMM(26), YMM(8))   MOV(R9, IMM(12*8)) //nr*sizeof(double)
    VMOVAPD(YMM(27), YMM(8))
    VMOVAPD(YMM(28), YMM(8))   LEA(RAX, MEM(RAX,R8,1)) //adjust a for pre-load
    VMOVAPD(YMM(29), YMM(8))
    VMOVAPD(YMM(30), YMM(8))
    VMOVAPD(YMM(31), YMM(8))

    TEST(RSI, RSI)
    JZ(POSTACCUM)

#ifdef PREFETCH_A_BEFORE
    PREFETCH(0, MEM(RAX,0*64))
    PREFETCH(0, MEM(RAX,1*64))
    PREFETCH(0, MEM(RAX,2*64))
    PREFETCH(0, MEM(RAX,3*64))
    PREFETCH(0, MEM(RAX,4*64))
    PREFETCH(0, MEM(RAX,5*64))
    PREFETCH(0, MEM(RAX,6*64))
    PREFETCH(0, MEM(RAX,7*64))

#endif

#ifdef PREFETCH_B_BEFORE
    PREFETCH(0, MEM(RBX,0*64))
    PREFETCH(0, MEM(RBX,1*64))
    PREFETCH(0, MEM(RBX,2*64))
    PREFETCH(0, MEM(RBX,3*64))
    PREFETCH(0, MEM(RBX,4*64))
    PREFETCH(0, MEM(RBX,5*64))
#endif

    PREFETCH_C_L2

    MOV(RDI, RSI)
    AND(RSI, IMM(3))
    SAR(RDI, IMM(2))

    SUB(RDI, IMM(0+TAIL_NITER))
    JLE(K_SMALL)

    LOOP_ALIGN
    LABEL(MAIN_LOOP)

        PREFETCH(0, MEM(RBX,B_L1_PREFETCH_DIST*12*8))
        PREFETCH(0, MEM(RBX,B_L1_PREFETCH_DIST*12*8+64))
        SUBITER(0)
        PREFETCH(0, MEM(RBX,B_L1_PREFETCH_DIST*12*8+128))
        SUBITER(1)
        PREFETCH(0, MEM(RBX,B_L1_PREFETCH_DIST*12*8+192))
        PREFETCH(0, MEM(RBX,B_L1_PREFETCH_DIST*12*8+256))
        SUBITER(2)
        PREFETCH(0, MEM(RBX,B_L1_PREFETCH_DIST*12*8+320))
        SUBITER(3)

        LEA(RAX, MEM(RAX,R8,4))
        LEA(RBX, MEM(RBX,R9,4))

        DEC(RDI)

    JNZ(MAIN_LOOP)

    LABEL(K_SMALL)

    PREFETCH_C_L1

    ADD(RDI, IMM(0+TAIL_NITER))
    JZ(TAIL_LOOP)

    LOOP_ALIGN
    LABEL(SMALL_LOOP)

        PREFETCH(0, MEM(RBX,B_L1_PREFETCH_DIST*12*8))
        PREFETCH(0, MEM(RBX,B_L1_PREFETCH_DIST*12*8+64))
        SUBITER(0)
        PREFETCH(0, MEM(RBX,B_L1_PREFETCH_DIST*12*8+128))
        SUBITER(1)
        PREFETCH(0, MEM(RBX,B_L1_PREFETCH_DIST*12*8+192))
        PREFETCH(0, MEM(RBX,B_L1_PREFETCH_DIST*12*8+256))
        SUBITER(2)
        PREFETCH(0, MEM(RBX,B_L1_PREFETCH_DIST*12*8+320))
        SUBITER(3)

        LEA(RAX, MEM(RAX,R8,4))
        LEA(RBX, MEM(RBX,R9,4))

        DEC(RDI)

    JNZ(SMALL_LOOP)

    TEST(RSI, RSI)
    JZ(POSTACCUM)

    LOOP_ALIGN
    LABEL(TAIL_LOOP)

        PREFETCH(0, MEM(RBX,B_L1_PREFETCH_DIST*12*8))
        PREFETCH(0, MEM(RBX,B_L1_PREFETCH_DIST*12*8+64))
        SUBITER(0)

        ADD(RAX, R8)
        ADD(RBX, R9)

        DEC(RSI)

    JNZ(TAIL_LOOP)

    LABEL(POSTACCUM)

#ifdef PREFETCH_A_AFTER
    MOV(R8, VAR(a))
    PREFETCH(0, MEM(R8,0*64))
    PREFETCH(0, MEM(R8,1*64))
    PREFETCH(0, MEM(R8,2*64))
    PREFETCH(0, MEM(R8,3*64))
    PREFETCH(0, MEM(R8,4*64))
    PREFETCH(0, MEM(R8,5*64))
    PREFETCH(0, MEM(R8,6*64))
    PREFETCH(0, MEM(R8,7*64))
#endif

#ifdef PREFETCH_B_AFTER
    MOV(R9, VAR(b))
    PREFETCH(0, MEM(R9,0*64))
    PREFETCH(0, MEM(R9,1*64))
    PREFETCH(0, MEM(R9,2*64))
    PREFETCH(0, MEM(R9,3*64))
    PREFETCH(0, MEM(R9,4*64))
    PREFETCH(0, MEM(R9,5*64))
#endif

    MOV(RAX, VAR(alpha))
    MOV(RBX, VAR(beta))
    VBROADCASTSD(ZMM(0), MEM(RAX))
    VBROADCASTSD(ZMM(1), MEM(RBX))

    MOV(RAX, VAR(cs_c))
    LEA(RAX, MEM(,RAX,8))

    VCOMISD(XMM(1), XMM(7))
    JE(COLSTORBZ)

        UPDATE_C( 8, 9,10,11)
        UPDATE_C(12,13,14,15)
        UPDATE_C(16,17,18,19)
        UPDATE_C(20,21,22,23)
        UPDATE_C(24,25,26,27)
        UPDATE_C(28,29,30,31)

    JMP(END)
    LABEL(COLSTORBZ)

        UPDATE_C_BZ( 8, 9,10,11)
        UPDATE_C_BZ(12,13,14,15)
        UPDATE_C_BZ(16,17,18,19)
        UPDATE_C_BZ(20,21,22,23)
        UPDATE_C_BZ(24,25,26,27)
        UPDATE_C_BZ(28,29,30,31)

    LABEL(END)

    VZEROUPPER()

    END_ASM(
    : // output operands
    : // input operands
      [k]         "m" (k),
      [a]         "m" (a),
      [b]         "m" (b),
      [alpha]     "m" (alpha),
      [beta]      "m" (beta),
      [c]         "m" (c),
      [rs_c]      "m" (rs_c),
      [cs_c]      "m" (cs_c)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11", "r12",
      "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
      "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
      "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
      "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
      "zmm30", "zmm31", "memory"
    )

    GEMM_UKR_FLUSH_CT( d );
}
