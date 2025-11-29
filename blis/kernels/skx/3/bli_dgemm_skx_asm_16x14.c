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
#include "bli_x86_asm_macros.h"

#define A_L1_PREFETCH_DIST 4 // in units of k iterations
#define B_L1_PREFETCH_DIST 4 // e.g. 4 k iterations ~= 56 cycles
#define TAIL_NITER 5 // in units of 4x unrolled k iterations
                     // e.g. 5 -> 4*5 k iterations ~= 280 cycles

#define PREFETCH_A_L1(n, k) \
    PREFETCH(0, MEM(RAX, A_L1_PREFETCH_DIST*16*8 + (2*n+k)*64))
#define PREFETCH_B_L1(n, k) \
    PREFETCH(0, MEM(RBX, B_L1_PREFETCH_DIST*14*8 + (2*n+k)*56))

#define LOOP_ALIGN ALIGN32

#define UPDATE_C(R1,R2) \
\
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VFMADD231PD(ZMM(R1), ZMM(1), MEM(RCX)) \
    VFMADD231PD(ZMM(R2), ZMM(1), MEM(RCX,64)) \
    VMOVUPD(MEM(RCX), ZMM(R1)) \
    VMOVUPD(MEM(RCX,64), ZMM(R2)) \
    LEA(RCX, MEM(RCX,RBX,1))

#define UPDATE_C_BZ(R1,R2) \
\
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMOVUPD(MEM(RCX), ZMM(R1)) \
    VMOVUPD(MEM(RCX,64), ZMM(R2)) \
    LEA(RCX, MEM(RCX,RBX,1))

#define UPDATE_C_COL_SCATTERED(R1,R2) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    KXNORW(K(3), K(0), K(0)) \
    KXNORW(K(4), K(0), K(0)) \
    VGATHERQPD(ZMM(0) MASK_K(1), MEM(RCX,ZMM(2),1)) \
    VFMADD231PD(ZMM(R1), ZMM(0), ZMM(1)) \
    VGATHERQPD(ZMM(0) MASK_K(2), MEM(RCX,ZMM(3),1)) \
    VFMADD231PD(ZMM(R2), ZMM(0), ZMM(1)) \
    VSCATTERQPD(MEM(RCX,ZMM(2),1) MASK_K(3), ZMM(R1)) \
    VSCATTERQPD(MEM(RCX,ZMM(3),1) MASK_K(4), ZMM(R2)) \
    LEA(RCX, MEM(RCX,RBX,1))

#define UPDATE_C_BZ_COL_SCATTERED(R1,R2) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    VSCATTERQPD(MEM(RCX,ZMM(2),1) MASK_K(1), ZMM(R1)) \
    VSCATTERQPD(MEM(RCX,ZMM(3),1) MASK_K(2), ZMM(R2)) \
    LEA(RCX, MEM(RCX,RBX,1))

#define SUBITER(n) \
\
    PREFETCH_A_L1(n, 0) \
    \
    VBROADCASTSD(ZMM(2), MEM(RBX,(14*n+ 0)*8)) \
    VBROADCASTSD(ZMM(3), MEM(RBX,(14*n+ 1)*8)) \
    VFMADD231PD(ZMM( 4), ZMM(0), ZMM(2)) \
    VFMADD231PD(ZMM( 5), ZMM(1), ZMM(2)) \
    VFMADD231PD(ZMM( 6), ZMM(0), ZMM(3)) \
    VFMADD231PD(ZMM( 7), ZMM(1), ZMM(3)) \
    \
    VBROADCASTSD(ZMM(2), MEM(RBX,(14*n+ 2)*8)) \
    VBROADCASTSD(ZMM(3), MEM(RBX,(14*n+ 3)*8)) \
    VFMADD231PD(ZMM( 8), ZMM(0), ZMM(2)) \
    VFMADD231PD(ZMM( 9), ZMM(1), ZMM(2)) \
    VFMADD231PD(ZMM(10), ZMM(0), ZMM(3)) \
    VFMADD231PD(ZMM(11), ZMM(1), ZMM(3)) \
    \
    PREFETCH_B_L1(n, 0) \
    \
    VBROADCASTSD(ZMM(2), MEM(RBX,(14*n+ 4)*8)) \
    VBROADCASTSD(ZMM(3), MEM(RBX,(14*n+ 5)*8)) \
    VFMADD231PD(ZMM(12), ZMM(0), ZMM(2)) \
    VFMADD231PD(ZMM(13), ZMM(1), ZMM(2)) \
    VFMADD231PD(ZMM(14), ZMM(0), ZMM(3)) \
    VFMADD231PD(ZMM(15), ZMM(1), ZMM(3)) \
    \
    VBROADCASTSD(ZMM(2), MEM(RBX,(14*n+ 6)*8)) \
    VBROADCASTSD(ZMM(3), MEM(RBX,(14*n+ 7)*8)) \
    VFMADD231PD(ZMM(16), ZMM(0), ZMM(2)) \
    VFMADD231PD(ZMM(17), ZMM(1), ZMM(2)) \
    VFMADD231PD(ZMM(18), ZMM(0), ZMM(3)) \
    VFMADD231PD(ZMM(19), ZMM(1), ZMM(3)) \
    \
    PREFETCH_A_L1(n, 1) \
    \
    VBROADCASTSD(ZMM(2), MEM(RBX,(14*n+ 8)*8)) \
    VBROADCASTSD(ZMM(3), MEM(RBX,(14*n+ 9)*8)) \
    VFMADD231PD(ZMM(20), ZMM(0), ZMM(2)) \
    VFMADD231PD(ZMM(21), ZMM(1), ZMM(2)) \
    VFMADD231PD(ZMM(22), ZMM(0), ZMM(3)) \
    VFMADD231PD(ZMM(23), ZMM(1), ZMM(3)) \
    \
    VBROADCASTSD(ZMM(2), MEM(RBX,(14*n+10)*8)) \
    VBROADCASTSD(ZMM(3), MEM(RBX,(14*n+11)*8)) \
    VFMADD231PD(ZMM(24), ZMM(0), ZMM(2)) \
    VFMADD231PD(ZMM(25), ZMM(1), ZMM(2)) \
    VFMADD231PD(ZMM(26), ZMM(0), ZMM(3)) \
    VFMADD231PD(ZMM(27), ZMM(1), ZMM(3)) \
    \
    PREFETCH_B_L1(n, 1) \
    \
    VBROADCASTSD(ZMM(2), MEM(RBX,(14*n+12)*8)) \
    VBROADCASTSD(ZMM(3), MEM(RBX,(14*n+13)*8)) \
    VFMADD231PD(ZMM(28), ZMM(0), ZMM(2)) \
    VFMADD231PD(ZMM(29), ZMM(1), ZMM(2)) \
    VFMADD231PD(ZMM(30), ZMM(0), ZMM(3)) \
    VFMADD231PD(ZMM(31), ZMM(1), ZMM(3)) \
    \
    VMOVAPD(ZMM(0), MEM(RAX,(16*n+0)*8)) \
    VMOVAPD(ZMM(1), MEM(RAX,(16*n+8)*8))

#if 0
//This is an array used for the scatter/gather instructions.
static int64_t offsets[16] __attribute__((aligned(64))) =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15};
#endif

void bli_dgemm_skx_asm_16x14
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

    GEMM_UKR_SETUP_CT( d, 16, 14, false );

    BEGIN_ASM()

    VXORPD(YMM( 4), YMM( 4), YMM( 4)) //clear out registers
    VXORPD(YMM( 5), YMM( 5), YMM( 5))
    VXORPD(YMM( 6), YMM( 6), YMM( 6))
    VXORPD(YMM( 7), YMM( 7), YMM( 7))
    VXORPD(YMM( 8), YMM( 8), YMM( 8))
    VXORPD(YMM( 9), YMM( 9), YMM( 9))
    VXORPD(YMM(10), YMM(10), YMM(10))
    VXORPD(YMM(11), YMM(11), YMM(11))
    VXORPD(YMM(12), YMM(12), YMM(12))
    VXORPD(YMM(13), YMM(13), YMM(13))
    VXORPD(YMM(14), YMM(14), YMM(14))
    VXORPD(YMM(15), YMM(15), YMM(15))
    VXORPD(YMM(16), YMM(16), YMM(16))
    VXORPD(YMM(17), YMM(17), YMM(17))
    VXORPD(YMM(18), YMM(18), YMM(18))
    VXORPD(YMM(19), YMM(19), YMM(19))
    VXORPD(YMM(20), YMM(20), YMM(20))
    VXORPD(YMM(21), YMM(21), YMM(21))
    VXORPD(YMM(22), YMM(22), YMM(22))
    VXORPD(YMM(23), YMM(23), YMM(23))
    VXORPD(YMM(24), YMM(24), YMM(24))
    VXORPD(YMM(25), YMM(25), YMM(25))
    VXORPD(YMM(26), YMM(26), YMM(26))
    VXORPD(YMM(27), YMM(27), YMM(27))
    VXORPD(YMM(28), YMM(28), YMM(28))
    VXORPD(YMM(29), YMM(29), YMM(29))
    VXORPD(YMM(30), YMM(30), YMM(30))
    VXORPD(YMM(31), YMM(31), YMM(31))

    MOV(RSI, VAR(k)) //loop index
    MOV(RAX, VAR(a)) //load address of a
    MOV(RBX, VAR(b)) //load address of b
    MOV(RCX, VAR(c)) //load address of c

    LEA(RDX, MEM(RSI,RSI,2))
    LEA(RDX, MEM(,RDX,4))
    LEA(RDX, MEM(RDX,RSI,2)) // 14*k
    LEA(RDX, MEM(RBX,RDX,8,-128)) // b_next
    LEA(R9, MEM(RCX,63)) // c for prefetching

    VMOVAPD(ZMM(0), MEM(RAX, 0*8)) //pre-load a
    VMOVAPD(ZMM(1), MEM(RAX, 8*8)) //pre-load a
    LEA(RAX, MEM(RAX,16*8)) //adjust a for pre-load

    MOV(R12, VAR(rs_c))
    MOV(R10, VAR(cs_c))
    LEA(R12, MEM(,R12,8))
    LEA(R10, MEM(,R10,8))

    MOV(RDI, RSI)
    AND(RSI, IMM(3))
    SAR(RDI, IMM(2))

    SUB(RDI, IMM(14+TAIL_NITER))
    JLE(K_LE_80)

        LOOP_ALIGN
        LABEL(LOOP1)

            SUBITER(0)
            PREFETCH(1, MEM(RDX))
            SUBITER(1)
            SUB(RDI, IMM(1))
            SUBITER(2)
            PREFETCH(1, MEM(RDX,64))
            SUBITER(3)

            LEA(RAX, MEM(RAX,4*16*8))
            LEA(RBX, MEM(RBX,4*14*8))
            LEA(RDX, MEM(RDX,16*8))

        JNZ(LOOP1)

    LABEL(K_LE_80)

    ADD(RDI, IMM(14))
    JLE(K_LE_24)

        LOOP_ALIGN
        LABEL(LOOP2)

            PREFETCH(0, MEM(R9))
            SUBITER(0)
            PREFETCH(1, MEM(RDX))
            SUBITER(1)
            PREFETCH(0, MEM(R9,64))
            SUB(RDI, IMM(1))
            SUBITER(2)
            PREFETCH(1, MEM(RDX,64))
            SUBITER(3)

            LEA(RAX, MEM(RAX,4*16*8))
            LEA(RBX, MEM(RBX,4*14*8))
            LEA(RDX, MEM(RDX,16*8))
            LEA(R9, MEM(R9,R10,1))

        JNZ(LOOP2)

    LABEL(K_LE_24)

    ADD(RDI, IMM(0+TAIL_NITER))
    JLE(TAIL)

        LOOP_ALIGN
        LABEL(LOOP3)

            SUBITER(0)
            PREFETCH(1, MEM(RDX))
            SUBITER(1)
            SUB(RDI, IMM(1))
            SUBITER(2)
            PREFETCH(1, MEM(RDX,64))
            SUBITER(3)

            LEA(RAX, MEM(RAX,4*16*8))
            LEA(RBX, MEM(RBX,4*14*8))
            LEA(RDX, MEM(RDX,16*8))

        JNZ(LOOP3)

    LABEL(TAIL)

    TEST(RSI, RSI)
    JZ(POSTACCUM)

        LOOP_ALIGN
        LABEL(TAIL_LOOP)

            SUB(RSI, IMM(1))
            SUBITER(0)

            LEA(RAX, MEM(RAX,16*8))
            LEA(RBX, MEM(RBX,14*8))

        JNZ(TAIL_LOOP)

    LABEL(POSTACCUM)

    MOV(RAX, VAR(alpha))
    MOV(RBX, VAR(beta))
    VBROADCASTSD(ZMM(0), MEM(RAX))
    VBROADCASTSD(ZMM(1), MEM(RBX))

    VXORPD(YMM(2), YMM(2), YMM(2))

    MOV(RAX, R12)
    MOV(RBX, R10)

    VCOMISD(XMM(1), XMM(2))
    JE(COLSTORBZ)

        UPDATE_C( 4, 5)
        UPDATE_C( 6, 7)
        UPDATE_C( 8, 9)
        UPDATE_C(10,11)
        UPDATE_C(12,13)
        UPDATE_C(14,15)
        UPDATE_C(16,17)
        UPDATE_C(18,19)
        UPDATE_C(20,21)
        UPDATE_C(22,23)
        UPDATE_C(24,25)
        UPDATE_C(26,27)
        UPDATE_C(28,29)
        UPDATE_C(30,31)

    JMP(END)
    LABEL(COLSTORBZ)

        UPDATE_C_BZ( 4, 5)
        UPDATE_C_BZ( 6, 7)
        UPDATE_C_BZ( 8, 9)
        UPDATE_C_BZ(10,11)
        UPDATE_C_BZ(12,13)
        UPDATE_C_BZ(14,15)
        UPDATE_C_BZ(16,17)
        UPDATE_C_BZ(18,19)
        UPDATE_C_BZ(20,21)
        UPDATE_C_BZ(22,23)
        UPDATE_C_BZ(24,25)
        UPDATE_C_BZ(26,27)
        UPDATE_C_BZ(28,29)
        UPDATE_C_BZ(30,31)

    LABEL(END)

    VZEROUPPER()

    END_ASM
    (
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
