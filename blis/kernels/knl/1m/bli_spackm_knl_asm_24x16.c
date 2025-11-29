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

#include "blis.h"

#define BLIS_ASM_SYNTAX_INTEL
#include "bli_x86_asm_macros.h"

#define LOADMUL8x8(a,o,s1,s3,s5,s7, \
                   z0,z1,z2,z3,z4,z5,z6,z7) \
    \
    VMULPS(YMM(z0), YMM(15), MEM(a,     o)) \
    VMULPS(YMM(z1), YMM(15), MEM(a,s1,1,o)) \
    VMULPS(YMM(z2), YMM(15), MEM(a,s1,2,o)) \
    VMULPS(YMM(z3), YMM(15), MEM(a,s3,1,o)) \
    VMULPS(YMM(z4), YMM(15), MEM(a,s1,4,o)) \
    VMULPS(YMM(z5), YMM(15), MEM(a,s5,1,o)) \
    VMULPS(YMM(z6), YMM(15), MEM(a,s3,2,o)) \
    VMULPS(YMM(z7), YMM(15), MEM(a,s7,1,o))

#define STORE8x8(a,o,s, \
                 z0,z1,z2,z3,z4,z5,z6,z7) \
    \
    VMOVUPS(MEM(a,(o)+0*(s)), YMM(z0)) \
    VMOVUPS(MEM(a,(o)+1*(s)), YMM(z1)) \
    VMOVUPS(MEM(a,(o)+2*(s)), YMM(z2)) \
    VMOVUPS(MEM(a,(o)+3*(s)), YMM(z3)) \
    VMOVUPS(MEM(a,(o)+4*(s)), YMM(z4)) \
    VMOVUPS(MEM(a,(o)+5*(s)), YMM(z5)) \
    VMOVUPS(MEM(a,(o)+6*(s)), YMM(z6)) \
    VMOVUPS(MEM(a,(o)+7*(s)), YMM(z7))

#define STORETRANS8x8(a,o,s, \
                      a0,a1,a2,a3,a4,a5,a6,a7, \
                      t0,t1,t2,t3,t4,t5) \
    \
    VUNPCKLPS(YMM(t0), YMM(a0), YMM(a1)) \
    VUNPCKLPS(YMM(t2), YMM(a2), YMM(a3)) \
    VUNPCKLPS(YMM(t1), YMM(a4), YMM(a5)) \
    VUNPCKLPS(YMM(t3), YMM(a6), YMM(a7)) \
    \
    VSHUFPS(YMM(t4), YMM(t0), YMM(t2), IMM(0x44)) \
    VSHUFPS(YMM(t5), YMM(t1), YMM(t3), IMM(0x44)) \
    VMOVUPS(MEM(a,(o   )+0*(s)), XMM(t4)) \
    VMOVUPS(MEM(a,(o+16)+0*(s)), XMM(t5)) \
    VEXTRACTF128(MEM(a,(o   )+4*(s)), YMM(t4), IMM(1)) \
    VEXTRACTF128(MEM(a,(o+16)+4*(s)), YMM(t5), IMM(1)) \
    \
    VSHUFPS(YMM(t4), YMM(t0), YMM(t2), IMM(0xEE)) \
    VSHUFPS(YMM(t5), YMM(t1), YMM(t3), IMM(0xEE)) \
    VMOVUPS(MEM(a,(o   )+1*(s)), XMM(t4)) \
    VMOVUPS(MEM(a,(o+16)+1*(s)), XMM(t5)) \
    VEXTRACTF128(MEM(a,(o   )+5*(s)), YMM(t4), IMM(1)) \
    VEXTRACTF128(MEM(a,(o+16)+5*(s)), YMM(t5), IMM(1)) \
    \
    VUNPCKHPS(YMM(t0), YMM(a0), YMM(a1)) \
    VUNPCKHPS(YMM(t2), YMM(a2), YMM(a3)) \
    VUNPCKHPS(YMM(t1), YMM(a4), YMM(a5)) \
    VUNPCKHPS(YMM(t3), YMM(a6), YMM(a7)) \
    \
    VSHUFPS(YMM(t4), YMM(t0), YMM(t2), IMM(0x44)) \
    VSHUFPS(YMM(t5), YMM(t1), YMM(t3), IMM(0x44)) \
    VMOVUPS(MEM(a,(o   )+2*(s)), XMM(t4)) \
    VMOVUPS(MEM(a,(o+16)+2*(s)), XMM(t5)) \
    VEXTRACTF128(MEM(a,(o   )+6*(s)), YMM(t4), IMM(1)) \
    VEXTRACTF128(MEM(a,(o+16)+6*(s)), YMM(t5), IMM(1)) \
    \
    VSHUFPS(YMM(t4), YMM(t0), YMM(t2), IMM(0xEE)) \
    VSHUFPS(YMM(t5), YMM(t1), YMM(t3), IMM(0xEE)) \
    VMOVUPS(MEM(a,(o   )+3*(s)), XMM(t4)) \
    VMOVUPS(MEM(a,(o+16)+3*(s)), XMM(t5)) \
    VEXTRACTF128(MEM(a,(o   )+7*(s)), YMM(t4), IMM(1)) \
    VEXTRACTF128(MEM(a,(o+16)+7*(s)), YMM(t5), IMM(1))

//This is an array used for the scatter/gather instructions.
static int32_t offsets[32] __attribute__((aligned(64))) =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
     16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};

void bli_spackm_knl_asm_24x16
     (
             conj_t  conja,
             pack_t  schema,
             dim_t   cdim_,
             dim_t   cdim_max,
             dim_t   cdim_bcast,
             dim_t   n_,
             dim_t   n_max_,
       const void*   kappa_,
       const void*   a_, inc_t inca_, inc_t lda_,
             void*   p_,              inc_t ldp_,
       const void*   params,
       const cntx_t* cntx
     )
{
    const int32_t* offsetPtr = &offsets[0];

    float*        a     = ( float* )a_;
    float*        p     = ( float* )p_;
    float*        kappa = ( float* )kappa_;
    const int64_t cdim  = cdim_;
    const int64_t mr    = 24;
    const int64_t nr    = 16;
    const int64_t n     = n_;
    const int64_t n_max = n_max_;
    const int64_t inca  = inca_;
    const int64_t lda   = lda_;
    const int64_t ldp   = ldp_;


    if ( cdim == mr && cdim_bcast == 1 )
    {

    BEGIN_ASM()

        MOV(RSI, VAR(n))
        MOV(RAX, VAR(a))
        MOV(RBX, VAR(inca))
        MOV(RCX, VAR(lda))
        MOV(R14, VAR(p))
        MOV(RDI, VAR(ldp))

        TEST(RSI, RSI)
        JZ(PACK24_DONE)

        LEA(RBX, MEM(,RBX,4))    //inca in bytes
        LEA(RCX, MEM(,RCX,4))    //lda in bytes
        LEA(RDI, MEM(,RDI,4))    //ldp in bytes

        VBROADCASTSS(ZMM(15), VAR(kappa))

        CMP(RBX, IMM(4))
        JNE(PACK24_T)

        LABEL(PACK24_N)

            MOV(RDX, RSI)
            AND(RDX, IMM(7))
            SAR(RSI, IMM(3))
            JZ(PACK24_N_TAIL)

            LEA(R8,  MEM(RCX,RCX,2)) //lda*3
            LEA(R9,  MEM(RCX,RCX,4)) //lda*5
            LEA(R10, MEM(R8 ,RCX,4)) //lda*7

            LABEL(PACK24_N_LOOP)

                LOADMUL8x8(RAX,0,RCX,R8,R9,R10,0,1,2,3,4,5,6,7)
                STORE8x8(R14,0,24*4,0,1,2,3,4,5,6,7)

                LOADMUL8x8(RAX,32,RCX,R8,R9,R10,0,1,2,3,4,5,6,7)
                STORE8x8(R14,32,24*4,0,1,2,3,4,5,6,7)

                LOADMUL8x8(RAX,64,RCX,R8,R9,R10,0,1,2,3,4,5,6,7)
                STORE8x8(R14,64,24*4,0,1,2,3,4,5,6,7)

                LEA(RAX, MEM(RAX,RCX,8))
                LEA(R14, MEM(R14,RDI,8))

                SUB(RSI, IMM(1))

            JNZ(PACK24_N_LOOP)

            TEST(RDX, RDX)
            JZ(PACK24_DONE)

            LABEL(PACK24_N_TAIL)

                VMULPS(ZMM(0), ZMM(15), MEM(RAX))
                VMOVUPS(MEM(R14), ZMM(0))

                VMULPS(YMM(1), YMM(15), MEM(RAX,64))
                VMOVUPS(MEM(R14,64), YMM(1))

                LEA(RAX, MEM(RAX,RCX,1))
                LEA(R14, MEM(R14,RDI,1))

                SUB(RDX, IMM(1))

            JNZ(PACK24_N_TAIL)

            JMP(PACK24_DONE)

        LABEL(PACK24_T)

            CMP(RCX, IMM(4))
            JNE(PACK24_G)

            LEA(R8,  MEM(RBX,RBX,2)) //inca*3
            LEA(R9,  MEM(RBX,RBX,4)) //inca*5
            LEA(R10, MEM(R8 ,RBX,4)) //inca*7
            LEA(R11, MEM(RAX,RBX,8))
            LEA(R12, MEM(R11,RBX,8))

            MOV(RDX, RSI)
            AND(RDX, IMM(7))
            SAR(RSI, IMM(3))
            JZ(PACK24_T_TAIL)

            LABEL(PACK24_T_LOOP)

                LOADMUL8x8(RAX,0,RBX,R8,R9,R10,0,1,2,3,4,5,6,7)
                STORETRANS8x8(R14,0,24*4,0,1,2,3,4,5,6,7,8,9,10,11,12,13)

                LOADMUL8x8(R11,0,RBX,R8,R9,R10,0,1,2,3,4,5,6,7)
                STORETRANS8x8(R14,32,24*4,0,1,2,3,4,5,6,7,8,9,10,11,12,13)

                LOADMUL8x8(R12,0,RBX,R8,R9,R10,0,1,2,3,4,5,6,7)
                STORETRANS8x8(R14,64,24*4,0,1,2,3,4,5,6,7,8,9,10,11,12,13)

                LEA(RAX, MEM(RAX,RCX,8))
                LEA(R11, MEM(R11,RCX,8))
                LEA(R12, MEM(R12,RCX,8))
                LEA(R14, MEM(R14,RDI,8))

                SUB(RSI, IMM(1))

            JNZ(PACK24_T_LOOP)

            TEST(RDX, RDX)
            JZ(PACK24_DONE)

            LABEL(PACK24_T_TAIL)

                VMULSS(XMM(0), XMM(15), MEM(RAX))
                VMULSS(XMM(1), XMM(15), MEM(RAX,RBX,1))
                VMULSS(XMM(2), XMM(15), MEM(RAX,RBX,2))
                VMULSS(XMM(3), XMM(15), MEM(RAX,R8,1))
                VMULSS(XMM(4), XMM(15), MEM(RAX,RBX,4))
                VMULSS(XMM(5), XMM(15), MEM(RAX,R9,1))
                VMULSS(XMM(6), XMM(15), MEM(RAX,R8,2))
                VMULSS(XMM(7), XMM(15), MEM(RAX,R10,1))
                VMOVSS(MEM(R14,0*4), XMM(0))
                VMOVSS(MEM(R14,1*4), XMM(1))
                VMOVSS(MEM(R14,2*4), XMM(2))
                VMOVSS(MEM(R14,3*4), XMM(3))
                VMOVSS(MEM(R14,4*4), XMM(4))
                VMOVSS(MEM(R14,5*4), XMM(5))
                VMOVSS(MEM(R14,6*4), XMM(6))
                VMOVSS(MEM(R14,7*4), XMM(7))

                VMULSS(XMM(0), XMM(15), MEM(R11))
                VMULSS(XMM(1), XMM(15), MEM(R11,RBX,1))
                VMULSS(XMM(2), XMM(15), MEM(R11,RBX,2))
                VMULSS(XMM(3), XMM(15), MEM(R11,R8,1))
                VMULSS(XMM(4), XMM(15), MEM(R11,RBX,4))
                VMULSS(XMM(5), XMM(15), MEM(R11,R9,1))
                VMULSS(XMM(6), XMM(15), MEM(R11,R8,2))
                VMULSS(XMM(7), XMM(15), MEM(R11,R10,1))
                VMOVSS(MEM(R14, 8*4), XMM(0))
                VMOVSS(MEM(R14, 9*4), XMM(1))
                VMOVSS(MEM(R14,10*4), XMM(2))
                VMOVSS(MEM(R14,11*4), XMM(3))
                VMOVSS(MEM(R14,12*4), XMM(4))
                VMOVSS(MEM(R14,13*4), XMM(5))
                VMOVSS(MEM(R14,14*4), XMM(6))
                VMOVSS(MEM(R14,15*4), XMM(7))

                VMULSS(XMM(0), XMM(15), MEM(R12))
                VMULSS(XMM(1), XMM(15), MEM(R12,RBX,1))
                VMULSS(XMM(2), XMM(15), MEM(R12,RBX,2))
                VMULSS(XMM(3), XMM(15), MEM(R12,R8,1))
                VMULSS(XMM(4), XMM(15), MEM(R12,RBX,4))
                VMULSS(XMM(5), XMM(15), MEM(R12,R9,1))
                VMULSS(XMM(6), XMM(15), MEM(R12,R8,2))
                VMULSS(XMM(7), XMM(15), MEM(R12,R10,1))
                VMOVSS(MEM(R14,16*4), XMM(0))
                VMOVSS(MEM(R14,17*4), XMM(1))
                VMOVSS(MEM(R14,18*4), XMM(2))
                VMOVSS(MEM(R14,19*4), XMM(3))
                VMOVSS(MEM(R14,20*4), XMM(4))
                VMOVSS(MEM(R14,21*4), XMM(5))
                VMOVSS(MEM(R14,22*4), XMM(6))
                VMOVSS(MEM(R14,23*4), XMM(7))

                LEA(RAX, MEM(RAX,RCX,1))
                LEA(R11, MEM(R11,RCX,1))
                LEA(R12, MEM(R12,RCX,1))
                LEA(R14, MEM(R14,RDI,1))

                SUB(RDX, IMM(1))

            JNZ(PACK24_T_TAIL)

            JMP(PACK24_DONE)

        LABEL(PACK24_G)

            VPBROADCASTD(ZMM(3), VAR(inca))
            MOV(RBX, VAR(offsetPtr))
            VPMULLD(ZMM(0), ZMM(3), MEM(RBX))

            LEA(R11, MEM(RAX,RBX,8))
            LEA(R11, MEM(R11,RBX,8))

            LABEL(PACK24_G_LOOP)

                KXNORW(K(1), K(0), K(0))
                KSHIFTRW(K(2), K(1), IMM(8))
                VGATHERDPS(ZMM(3) MASK_K(1), MEM(RAX,ZMM(0),8))
                VGATHERDPS(ZMM(4) MASK_K(2), MEM(R11,ZMM(0),8))
                VMULPS(ZMM(3), ZMM(3), ZMM(15))
                VMULPS(YMM(4), YMM(4), YMM(15))
                VMOVUPS(MEM(R14), ZMM(3))
                VMOVUPS(MEM(R14,64), YMM(4))

                LEA(RAX, MEM(RAX,RCX,1))
                LEA(R14, MEM(R14,RDI,1))

                SUB(RSI, IMM(1))

            JNZ(PACK24_G_LOOP)

        LABEL(PACK24_DONE)

    END_ASM(
        : //output operands
        : //input operands
          [n]         "m" (n),
          [kappa]     "m" (*kappa),
          [a]         "m" (a),
          [inca]      "m" (inca),
          [lda]       "m" (lda),
          [p]         "m" (p),
          [ldp]       "m" (ldp),
          [offsetPtr] "m" (offsetPtr)
        : //clobbers
          "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
          "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17",
          "zmm18", "zmm19", "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
          "zmm30", "zmm31",
          "rax", "rbx", "rcx", "rdx", "rdi", "rsi",
          "r8", "r9", "r10", "r11", "r12", "r13", "r14", "memory"
    )

	}
    if ( cdim == nr && ldp == nr && cdim_bcast == 1 )
    {

    BEGIN_ASM()

        MOV(RSI, VAR(n))
        MOV(RAX, VAR(a))
        MOV(RBX, VAR(inca))
        MOV(RCX, VAR(lda))
        MOV(R14, VAR(p))

        TEST(RSI, RSI)
        JZ(PACK16_DONE)

        LEA(RBX, MEM(,RBX,4))    //inca in bytes
        LEA(RCX, MEM(,RCX,4))    //lda in bytes

        VBROADCASTSS(YMM(15), VAR(kappa))

        CMP(RBX, IMM(4))
        JNE(PACK16_T)

        LABEL(PACK16_N)

            MOV(RDX, RSI)
            AND(RDX, IMM(7))
            SAR(RSI, IMM(3))
            JZ(PACK16_N_TAIL)

            LEA(R8,  MEM(RCX,RCX,2)) //lda*3
            LEA(R9,  MEM(RCX,RCX,4)) //lda*5
            LEA(R10, MEM(R8 ,RCX,4)) //lda*7

            LABEL(PACK16_N_LOOP)

                LOADMUL8x8(RAX,0,RCX,R8,R9,R10,0,1,2,3,4,5,6,7)
                STORE8x8(R14,0,16*4,0,1,2,3,4,5,6,7)

                LOADMUL8x8(RAX,32,RCX,R8,R9,R10,0,1,2,3,4,5,6,7)
                STORE8x8(R14,32,16*4,0,1,2,3,4,5,6,7)

                LEA(RAX, MEM(RAX,RCX,8))
                LEA(R14, MEM(R14,16*8*4))

                SUB(RSI, IMM(1))

            JNZ(PACK16_N_LOOP)

            TEST(RDX, RDX)
            JZ(PACK16_DONE)

            LABEL(PACK16_N_TAIL)

                VMULPS(YMM(0), YMM(15), MEM(RAX   ))
                VMULPS(YMM(1), YMM(15), MEM(RAX,32))
                VMOVUPS(MEM(R14   ), YMM(0))
                VMOVUPS(MEM(R14,32), YMM(1))

                LEA(RAX, MEM(RAX,RCX,1))
                LEA(R14, MEM(R14, 16*4))

                SUB(RDX, IMM(1))

            JNZ(PACK16_N_TAIL)

            JMP(PACK16_DONE)

        LABEL(PACK16_T)

            CMP(RCX, IMM(4))
            JNE(PACK16_G)

            LEA(R8,  MEM(RBX,RBX,2)) //inca*3
            LEA(R9,  MEM(RBX,RBX,4)) //inca*5
            LEA(R10, MEM(R8 ,RBX,4)) //inca*7
            LEA(R11, MEM(RAX,RBX,8))

            MOV(RDX, RSI)
            AND(RDX, IMM(7))
            SAR(RSI, IMM(3))
            JZ(PACK16_T_TAIL)

            LABEL(PACK16_T_LOOP)

                LOADMUL8x8(RAX,0,RBX,R8,R9,R10,0,1,2,3,4,5,6,7)
                STORETRANS8x8(R14,0,16*4,0,1,2,3,4,5,6,7,8,9,10,11,12,13)

                LOADMUL8x8(R11,0,RBX,R8,R9,R10,0,1,2,3,4,5,6,7)
                STORETRANS8x8(R14,32,16*4,0,1,2,3,4,5,6,7,8,9,10,11,12,13)

                LEA(RAX, MEM(RAX,   8*4))
                LEA(R11, MEM(R11,   8*4))
                LEA(R14, MEM(R14,16*8*4))

                SUB(RSI, IMM(1))

            JNZ(PACK16_T_LOOP)

            TEST(RDX, RDX)
            JZ(PACK16_DONE)

            LABEL(PACK16_T_TAIL)

                VMULSS(XMM(0), XMM(15), MEM(RAX      ))
                VMULSS(XMM(1), XMM(15), MEM(RAX,RBX,1))
                VMULSS(XMM(2), XMM(15), MEM(RAX,RBX,2))
                VMULSS(XMM(3), XMM(15), MEM(RAX,R8 ,1))
                VMULSS(XMM(4), XMM(15), MEM(RAX,RBX,4))
                VMULSS(XMM(5), XMM(15), MEM(RAX,R9 ,1))
                VMULSS(XMM(6), XMM(15), MEM(RAX,R8 ,2))
                VMULSS(XMM(7), XMM(15), MEM(RAX,R10,1))
                VMOVSS(MEM(R14,0*4), XMM(0))
                VMOVSS(MEM(R14,1*4), XMM(1))
                VMOVSS(MEM(R14,2*4), XMM(2))
                VMOVSS(MEM(R14,3*4), XMM(3))
                VMOVSS(MEM(R14,4*4), XMM(4))
                VMOVSS(MEM(R14,5*4), XMM(5))
                VMOVSS(MEM(R14,6*4), XMM(6))
                VMOVSS(MEM(R14,7*4), XMM(7))

                VMULSS(XMM(0), XMM(15), MEM(R11      ))
                VMULSS(XMM(1), XMM(15), MEM(R11,RBX,1))
                VMULSS(XMM(2), XMM(15), MEM(R11,RBX,2))
                VMULSS(XMM(3), XMM(15), MEM(R11,R8 ,1))
                VMULSS(XMM(4), XMM(15), MEM(R11,RBX,4))
                VMULSS(XMM(5), XMM(15), MEM(R11,R9 ,1))
                VMULSS(XMM(6), XMM(15), MEM(R11,R8 ,2))
                VMULSS(XMM(7), XMM(15), MEM(R11,R10,1))
                VMOVSS(MEM(R14, 8*4), XMM(0))
                VMOVSS(MEM(R14, 9*4), XMM(1))
                VMOVSS(MEM(R14,10*4), XMM(2))
                VMOVSS(MEM(R14,11*4), XMM(3))
                VMOVSS(MEM(R14,12*4), XMM(4))
                VMOVSS(MEM(R14,13*4), XMM(5))
                VMOVSS(MEM(R14,14*4), XMM(6))
                VMOVSS(MEM(R14,15*4), XMM(7))

                LEA(RAX, MEM(RAX,   4))
                LEA(R11, MEM(R11,   4))
                LEA(R14, MEM(R14,16*4))

                SUB(RDX, IMM(1))

            JNZ(PACK16_T_TAIL)

            JMP(PACK16_DONE)

        LABEL(PACK16_G)

            VPBROADCASTD(ZMM(3), VAR(inca))
            MOV(RBX, VAR(offsetPtr))
            VPMULLD(ZMM(0), ZMM(3), MEM(RBX))

            LABEL(PACK16_G_LOOP)

                KXNORW(K(1), K(0), K(0))
                VGATHERDPS(ZMM(3) MASK_K(1), MEM(RAX,ZMM(0),8))
                VMULPS(ZMM(3), ZMM(3), ZMM(15))
                VMOVUPS(MEM(R14), ZMM(3))

                LEA(RAX, MEM(RAX,RCX,1))
                LEA(R14, MEM(R14, 16*4))

                SUB(RSI, IMM(1))

            JNZ(PACK16_G_LOOP)

        LABEL(PACK16_DONE)

    END_ASM(
        : //output operands
        : //input operands
          [n]         "m" (n),
          [kappa]     "m" (*kappa),
          [a]         "m" (a),
          [inca]      "m" (inca),
          [lda]       "m" (lda),
          [p]         "m" (p),
          [ldp]       "m" (ldp),
          [offsetPtr] "m" (offsetPtr)
        : //clobbers
          "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
          "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17",
          "zmm18", "zmm19", "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
          "zmm30", "zmm31",
          "rax", "rbx", "rcx", "rdx", "rdi", "rsi",
          "r8", "r9", "r10", "r11", "r12", "r13", "r14", "memory"
    )

	}
	else
	{
		bli_sscal2bbs_mxn
		(
		  BLIS_NO_CONJUGATE,
		  cdim,
		  n,
		  kappa,
		  a,       inca, lda,
		  p, cdim_bcast, ldp
		);
	}

	bli_sset0s_edge
	(
	  cdim*cdim_bcast, cdim_max*cdim_bcast,
	  n, n_max,
	  p, ldp
	);
}
