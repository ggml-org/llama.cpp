/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2018, The University of Texas at Austin
   Copyright (C) 2019, Advanced Micro Devices, Inc.

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

#ifndef BLIS_X86_ASM_MACROS_H
#define BLIS_X86_ASM_MACROS_H

//
// Assembly macros to make inline x86 with AT&T syntax somewhat less painful
//
// "Private" macros end with _
//

// Default syntax is Intel
#if !defined(BLIS_ASM_SYNTAX_ATT) && !defined(BLIS_ASM_SYNTAX_INTEL)
#define BLIS_ASM_SYNTAX_INTEL
#endif

#define STRINGIFY_(...) #__VA_ARGS__
#define GET_MACRO_(_1_,_2_,_3_,_4_,NAME,...) NAME

#if (defined(_WIN32) && !defined(__clang__) && !defined(__MINGW32__)) || defined(__MIC__)

// Intel-style assembly blocks

#define BEGIN_ASM __asm {
#define END_ASM(...) }

#ifdef BLIS_ASM_SYNTAX_INTEL

#define INSTR_4_(name,_0,_1,_2,_3) name _0,_1,_2,_3
#define INSTR_3_(name,_0,_1,_2) name _0,_1,_2
#define INSTR_2_(name,_0,_1) name _0,_1
#define INSTR_1_(name,_0) name _0
#define INSTR_0_(name) name

#else

#define INSTR_4_(name,_0,_1,_2,_3) name _3,_2,_1,_0
#define INSTR_3_(name,_0,_1,_2) name _2,_1,_0
#define INSTR_2_(name,_0,_1) name _1,_0
#define INSTR_1_(name,_0) name _0
#define INSTR_0_(name) name

#endif

#define LABEL(label) label:
#define REGISTER_(r) r
#define IMM(x) x
#define VAR(x) x
#define MASK_(x) {x}
#define JMP_(insn, target) insn target

#define MEM_4_(reg,off,scale,disp) [reg + off*scale + disp]
#define MEM_3_(reg,off,scale) [reg + off*scale]
#define MEM_2_(reg,disp) [reg + disp]
#define MEM_1_(reg) [reg]

#define ALIGN4 align 4
#define ALIGN8 align 8
#define ALIGN16 align 16
#define ALIGN32 align 32

#else

// GCC extended assembly with AT&T syntax

#define COMMENT_BEGIN "#"
#define COMMENT_END

#define BEGIN_ASM() __asm__ volatile (
#define END_ASM(...) __VA_ARGS__ );


#ifdef BLIS_ASM_SYNTAX_ATT

#define INSTR_4_(name,_0,_1,_2,_3) STRINGIFY_(name) " " STRINGIFY_(_0,_1,_2,_3) "\n\t"
#define INSTR_3_(name,_0,_1,_2) STRINGIFY_(name) " " STRINGIFY_(_0,_1,_2) "\n\t"
#define INSTR_2_(name,_0,_1) STRINGIFY_(name) " " STRINGIFY_(_0,_1) "\n\t"
#define INSTR_1_(name,_0) STRINGIFY_(name) " " STRINGIFY_(_0) "\n\t"
#define INSTR_0_(name) STRINGIFY_(name) "\n\t"

#else

#define INSTR_4_(name,_0,_1,_2,_3) STRINGIFY_(name) " " STRINGIFY_(_3,_2,_1,_0) "\n\t"
#define INSTR_3_(name,_0,_1,_2) STRINGIFY_(name) " " STRINGIFY_(_2,_1,_0) "\n\t"
#define INSTR_2_(name,_0,_1) STRINGIFY_(name) " " STRINGIFY_(_1,_0) "\n\t"
#define INSTR_1_(name,_0) STRINGIFY_(name) " " STRINGIFY_(_0) "\n\t"
#define INSTR_0_(name) STRINGIFY_(name) "\n\t"

#endif

#if BLIS_OS_OSX

#define LABEL_(label) "L" STRINGIFY_(label) "%="

#else

#define LABEL_(label) ".L" STRINGIFY_(label) "%="

#endif

#define REGISTER_(r) %%r
#define IMM(x) $##x
#define VAR(x) %[x]
#define MASK_(x) %{x%}
#define LABEL(target) LABEL_(target) ":\n\t"
#define JMP_(insn, target) STRINGIFY_(insn) " " LABEL_(target) "\n\t"

#define MEM_4_(reg,off,scale,disp) disp(reg,off,scale)
#define MEM_3_(reg,off,scale) (reg,off,scale)
#define MEM_2_(reg,disp) disp(reg)
#define MEM_1_(reg) (reg)

#define ALIGN4 ".p2align 2 \n\t"
#define ALIGN8 ".p2align 3 \n\t"
#define ALIGN16 ".p2align 4 \n\t"
#define ALIGN32 ".p2align 5 \n\t"

#endif

#define begin_asm() BEGIN_ASM()
#define end_asm(...) END_ASM(__VA_ARGS__)

#define label(...) LABEL(__VA_ARGS__)
#define imm(...) IMM(__VA_ARGS__)
#define var(...) VAR(__VA_ARGS__)
#define align16 ALIGN16
#define align32 ALIGN32

// General-purpose registers

#define AL REGISTER_(al)
#define AH REGISTER_(ah)
#define BL REGISTER_(bl)
#define BH REGISTER_(bh)
#define CL REGISTER_(cl)
#define CH REGISTER_(ch)
#define DL REGISTER_(dl)
#define DH REGISTER_(dh)
#define R8B REGISTER_(r8b)
#define R9B REGISTER_(r9b)
#define R10B REGISTER_(r10b)
#define R11B REGISTER_(r11b)
#define R12B REGISTER_(r12b)
#define R13B REGISTER_(r13b)
#define R14B REGISTER_(r14b)
#define R15B REGISTER_(r15b)

#define al AL
#define ah AH
#define bl BL
#define bh BH
#define cl CL
#define ch CH
#define dl DL
#define dh DH
#define r8b R8B
#define r9b R9B
#define r10b R10B
#define r11b R11B
#define r12b R12B
#define r13b R13B
#define r14b R14B
#define r15b R15B

#define AX REGISTER_(ax)
#define BX REGISTER_(bx)
#define CX REGISTER_(cx)
#define DX REGISTER_(dx)
#define SI REGISTER_(si)
#define DI REGISTER_(di)
#define BP REGISTER_(bp)
#define SP REGISTER_(sp)
#define R8W REGISTER_(r8w)
#define R9W REGISTER_(r9w)
#define R10W REGISTER_(r10w)
#define R11W REGISTER_(r11w)
#define R12W REGISTER_(r12w)
#define R13W REGISTER_(r13w)
#define R14W REGISTER_(r14w)
#define R15W REGISTER_(r15w)

#define ax AX
#define bx BX
#define cx CX
#define dx DX
#define si SI
#define di DI
#define bp BP
#define sp SP
#define r8w R8W
#define r9w R9W
#define r10w R10W
#define r11w R11W
#define r12w R12W
#define r13w R13W
#define r14w R14W
#define r15w R15W

#define EAX REGISTER_(eax)
#define EBX REGISTER_(ebx)
#define ECX REGISTER_(ecx)
#define EDX REGISTER_(edx)
#define ESP REGISTER_(esp)
#define EBP REGISTER_(ebp)
#define EDI REGISTER_(edi)
#define ESI REGISTER_(esi)
#define R8D REGISTER_(r8d)
#define R9D REGISTER_(r9d)
#define R10D REGISTER_(r10d)
#define R11D REGISTER_(r11d)
#define R12D REGISTER_(r12d)
#define R13D REGISTER_(r13d)
#define R14D REGISTER_(r14d)
#define R15D REGISTER_(r15d)

#define eax EAX
#define ebx EBX
#define ecx ECX
#define edx EDX
#define esp ESP
#define ebp EBP
#define edi EDI
#define esi ESI
#define r8d R8D
#define r9d R9D
#define r10d R10D
#define r11d R11D
#define r12d R12D
#define r13d R13D
#define r14d R14D
#define r15d R15D

#define RAX REGISTER_(rax)
#define RBX REGISTER_(rbx)
#define RCX REGISTER_(rcx)
#define RDX REGISTER_(rdx)
#define RSP REGISTER_(rsp)
#define RBP REGISTER_(rbp)
#define RDI REGISTER_(rdi)
#define RSI REGISTER_(rsi)
#define R8 REGISTER_(r8)
#define R9 REGISTER_(r9)
#define R10 REGISTER_(r10)
#define R11 REGISTER_(r11)
#define R12 REGISTER_(r12)
#define R13 REGISTER_(r13)
#define R14 REGISTER_(r14)
#define R15 REGISTER_(r15)

#define rax RAX
#define rbx RBX
#define rcx RCX
#define rdx RDX
#define rsp RSP
#define rbp RBP
#define rdi RDI
#define rsi RSI
#define r8 R8
#define r9 R9
#define r10 R10
#define r11 R11
#define r12 R12
#define r13 R13
#define r14 R14
#define r15 R15

// Vector registers

#define XMM(x) REGISTER_(Xmm##x)
#define YMM(x) REGISTER_(Ymm##x)
#define ZMM(x) REGISTER_(Zmm##x)
#define K(x) REGISTER_(k##x)
#define MASK_K(n) MASK_(K(n))
#define MASK_KZ(n) MASK_(K(n))MASK_(z)

#define xmm(x) XMM(x)
#define ymm(x) YMM(x)
#define zmm(x) ZMM(x)
#define k(x) K(x)
#define mask_k(x) MASK_K(x)
#define mask_kz(x) MASK_KZ(x)

#define XMM0 XMM(0)
#define XMM1 XMM(1)
#define XMM2 XMM(2)
#define XMM3 XMM(3)
#define XMM4 XMM(4)
#define XMM5 XMM(5)
#define XMM6 XMM(6)
#define XMM7 XMM(7)
#define XMM8 XMM(8)
#define XMM9 XMM(9)
#define XMM10 XMM(10)
#define XMM11 XMM(11)
#define XMM12 XMM(12)
#define XMM13 XMM(13)
#define XMM14 XMM(14)
#define XMM15 XMM(15)
#define XMM16 XMM(16)
#define XMM17 XMM(17)
#define XMM18 XMM(18)
#define XMM19 XMM(19)
#define XMM20 XMM(20)
#define XMM21 XMM(21)
#define XMM22 XMM(22)
#define XMM23 XMM(23)
#define XMM24 XMM(24)
#define XMM25 XMM(25)
#define XMM26 XMM(26)
#define XMM27 XMM(27)
#define XMM28 XMM(28)
#define XMM29 XMM(29)
#define XMM30 XMM(30)
#define XMM31 XMM(31)

#define YMM0 YMM(0)
#define YMM1 YMM(1)
#define YMM2 YMM(2)
#define YMM3 YMM(3)
#define YMM4 YMM(4)
#define YMM5 YMM(5)
#define YMM6 YMM(6)
#define YMM7 YMM(7)
#define YMM8 YMM(8)
#define YMM9 YMM(9)
#define YMM10 YMM(10)
#define YMM11 YMM(11)
#define YMM12 YMM(12)
#define YMM13 YMM(13)
#define YMM14 YMM(14)
#define YMM15 YMM(15)
#define YMM16 YMM(16)
#define YMM17 YMM(17)
#define YMM18 YMM(18)
#define YMM19 YMM(19)
#define YMM20 YMM(20)
#define YMM21 YMM(21)
#define YMM22 YMM(22)
#define YMM23 YMM(23)
#define YMM24 YMM(24)
#define YMM25 YMM(25)
#define YMM26 YMM(26)
#define YMM27 YMM(27)
#define YMM28 YMM(28)
#define YMM29 YMM(29)
#define YMM30 YMM(30)
#define YMM31 YMM(31)

#define ZMM0 ZMM(0)
#define ZMM1 ZMM(1)
#define ZMM2 ZMM(2)
#define ZMM3 ZMM(3)
#define ZMM4 ZMM(4)
#define ZMM5 ZMM(5)
#define ZMM6 ZMM(6)
#define ZMM7 ZMM(7)
#define ZMM8 ZMM(8)
#define ZMM9 ZMM(9)
#define ZMM10 ZMM(10)
#define ZMM11 ZMM(11)
#define ZMM12 ZMM(12)
#define ZMM13 ZMM(13)
#define ZMM14 ZMM(14)
#define ZMM15 ZMM(15)
#define ZMM16 ZMM(16)
#define ZMM17 ZMM(17)
#define ZMM18 ZMM(18)
#define ZMM19 ZMM(19)
#define ZMM20 ZMM(20)
#define ZMM21 ZMM(21)
#define ZMM22 ZMM(22)
#define ZMM23 ZMM(23)
#define ZMM24 ZMM(24)
#define ZMM25 ZMM(25)
#define ZMM26 ZMM(26)
#define ZMM27 ZMM(27)
#define ZMM28 ZMM(28)
#define ZMM29 ZMM(29)
#define ZMM30 ZMM(30)
#define ZMM31 ZMM(31)

#define xmm0 xmm(0)
#define xmm1 xmm(1)
#define xmm2 xmm(2)
#define xmm3 xmm(3)
#define xmm4 xmm(4)
#define xmm5 xmm(5)
#define xmm6 xmm(6)
#define xmm7 xmm(7)
#define xmm8 xmm(8)
#define xmm9 xmm(9)
#define xmm10 xmm(10)
#define xmm11 xmm(11)
#define xmm12 xmm(12)
#define xmm13 xmm(13)
#define xmm14 xmm(14)
#define xmm15 xmm(15)
#define xmm16 xmm(16)
#define xmm17 xmm(17)
#define xmm18 xmm(18)
#define xmm19 xmm(19)
#define xmm20 xmm(20)
#define xmm21 xmm(21)
#define xmm22 xmm(22)
#define xmm23 xmm(23)
#define xmm24 xmm(24)
#define xmm25 xmm(25)
#define xmm26 xmm(26)
#define xmm27 xmm(27)
#define xmm28 xmm(28)
#define xmm29 xmm(29)
#define xmm30 xmm(30)
#define xmm31 xmm(31)

#define ymm0 ymm(0)
#define ymm1 ymm(1)
#define ymm2 ymm(2)
#define ymm3 ymm(3)
#define ymm4 ymm(4)
#define ymm5 ymm(5)
#define ymm6 ymm(6)
#define ymm7 ymm(7)
#define ymm8 ymm(8)
#define ymm9 ymm(9)
#define ymm10 ymm(10)
#define ymm11 ymm(11)
#define ymm12 ymm(12)
#define ymm13 ymm(13)
#define ymm14 ymm(14)
#define ymm15 ymm(15)
#define ymm16 ymm(16)
#define ymm17 ymm(17)
#define ymm18 ymm(18)
#define ymm19 ymm(19)
#define ymm20 ymm(20)
#define ymm21 ymm(21)
#define ymm22 ymm(22)
#define ymm23 ymm(23)
#define ymm24 ymm(24)
#define ymm25 ymm(25)
#define ymm26 ymm(26)
#define ymm27 ymm(27)
#define ymm28 ymm(28)
#define ymm29 ymm(29)
#define ymm30 ymm(30)
#define ymm31 ymm(31)

#define zmm0 zmm(0)
#define zmm1 zmm(1)
#define zmm2 zmm(2)
#define zmm3 zmm(3)
#define zmm4 zmm(4)
#define zmm5 zmm(5)
#define zmm6 zmm(6)
#define zmm7 zmm(7)
#define zmm8 zmm(8)
#define zmm9 zmm(9)
#define zmm10 zmm(10)
#define zmm11 zmm(11)
#define zmm12 zmm(12)
#define zmm13 zmm(13)
#define zmm14 zmm(14)
#define zmm15 zmm(15)
#define zmm16 zmm(16)
#define zmm17 zmm(17)
#define zmm18 zmm(18)
#define zmm19 zmm(19)
#define zmm20 zmm(20)
#define zmm21 zmm(21)
#define zmm22 zmm(22)
#define zmm23 zmm(23)
#define zmm24 zmm(24)
#define zmm25 zmm(25)
#define zmm26 zmm(26)
#define zmm27 zmm(27)
#define zmm28 zmm(28)
#define zmm29 zmm(29)
#define zmm30 zmm(30)
#define zmm31 zmm(31)

// Memory access

// MEM(rax)            ->     (%rax)        or [rax]
// MEM(rax,0x80)       -> 0x80(%rax)        or [rax + 0x80]
// MEM(rax,rsi,4)      ->     (%rax,%rsi,4) or [rax + rsi*4]
// MEM(rax,rsi,4,0x80) -> 0x80(%rax,%rsi,4) or [rax + rsi*4 + 0x80]

#define MEM(...) GET_MACRO_(__VA_ARGS__,MEM_4_,MEM_3_,MEM_2_,MEM_1_)(__VA_ARGS__)
#define MEM_1TO8(...) MEM(__VA_ARGS__) MASK_(1to8)
#define MEM_1TO16(...) MEM(__VA_ARGS__) MASK_(1to16)
#define MEM_BCAST(...) MEM(__VA_ARGS__) MASK_(b)

#define mem(...) MEM(__VA_ARGS__)
#define mem_1to8(...) MEM_1TO8(__VA_ARGS__)
#define mem_1to16(...) MEM_1TO16(__VA_ARGS__)
#define mem_bcast(...) MEM_BCAST(__VA_ARGS__)

#define VAR_1TO8(...) VAR(__VA_ARGS__) MASK_(1to8)
#define VAR_1TO16(...) VAR(__VA_ARGS__) MASK_(1to16)
#define VAR_BCAST(...) VAR(__VA_ARGS__) MASK_(b)

#define var_1to8(...) VAR_1TO8(__VA_ARGS__)
#define var_1to16(...) VAR_1TO16(__VA_ARGS__)
#define var_bcast(...) VAR_BCAST(__VA_ARGS__)

// Instructions

#define INSTR_(name,...) GET_MACRO_(__VA_ARGS__,INSTR_4_,INSTR_3_,INSTR_2_, \
                                    INSTR_1_,INSTR_0_)(name,__VA_ARGS__)

// Jumps

#define JC(_0) JMP_(jc, _0)
#define JB(_0) JC(_0)
#define JNAE(_0) JC(_0)
#define JNC(_0) JMP_(jnc, _0)
#define JNB(_0) JNC(_0)
#define JAE(_0) JNC(_0)

#define jc(_0) JC(_0)
#define jb(_0) JB(_0)
#define jnae(_0) JNAE(_0)
#define jnc(_0) JNC(_0)
#define jnb(_0) JNB(_0)
#define jae(_0) JAE(_0)

#define JO(_0) JMP_(jo, _0)
#define JNO(_0) JMP_(jno, _0)

#define jo(_0) JO(_0)
#define jno(_0) JNO(_0)

#define JP(_0) JMP_(jp, _0)
#define JPE(_0) JP(_0)
#define JNP(_0) JMP_(jnp, _0)
#define JPO(_0) JNP(_0)

#define jp(_0) JP(_0)
#define jpe(_0) JPE(_0)
#define jnp(_0) JNP(_0)
#define jpo(_0) JPO(_0)

#define JS(_0) JMP_(js, _0)
#define JNS(_0) JMP_(jns, _0)

#define js(_0) JS(_0)
#define jns(_0) JNS(_0)

#define JA(_0) JMP_(ja, _0)
#define JNBE(_0) JA(_0)
#define JNA(_0) JMP_(jna, _0)
#define JBE(_0) JNA(_0)

#define ja(_0) JA(_0)
#define jnbe(_0) JNBE(_0)
#define jna(_0) JNA(_0)
#define jbe(_0) JBE(_0)

#define JL(_0) JMP_(jl, _0)
#define JNGE(_0) JL(_0)
#define JNL(_0) JMP_(jnl, _0)
#define JGE(_0) JNL(_0)

#define jl(_0) JL(_0)
#define jnge(_0) JNGE(_0)
#define jnl(_0) JNL(_0)
#define jge(_0) JGE(_0)

#define JG(_0) JMP_(jg, _0)
#define JNLE(_0) JG(_0)
#define JNG(_0) JMP_(jng, _0)
#define JLE(_0) JNG(_0)

#define jg(_0) JG(_0)
#define jnle(_0) JNLE(_0)
#define jng(_0) JNG(_0)
#define jle(_0) JLE(_0)

#define JE(_0) JMP_(je, _0)
#define JZ(_0) JE(_0)
#define JNE(_0) JMP_(jne, _0)
#define JNZ(_0) JNE(_0)

#define je(_0) JE(_0)
#define jz(_0) JZ(_0)
#define jne(_0) JNE(_0)
#define jnz(_0) JNZ(_0)

#define JMP(_0) JMP_(jmp, _0)

#define jmp(_0) JMP(_0)

#define SETE(_0) INSTR_(sete, _0)
#define SETZ(_0) SETE(_0)

#define sete(_0) SETE(_0)
#define setz(_0) SETZ(_0)

// Comparisons

#define CMP(_0, _1) INSTR_(cmp, _0, _1)
#define TEST(_0, _1) INSTR_(test, _0, _1)

#define cmp(_0, _1) CMP(_0, _1)
#define test(_0, _1) TEST(_0, _1)

// Integer math

#define AND(_0, _1) INSTR_(and, _0, _1)
#define OR(_0, _1) INSTR_(or, _0, _1)
#define XOR(_0, _1) INSTR_(xor, _0, _1)
#define ADD(_0, _1) INSTR_(add, _0, _1)
#define SUB(_0, _1) INSTR_(sub, _0, _1)
#define IMUL(_0, _1) INSTR_(imul, _0, _1)
#define SAL(...) INSTR_(sal, __VA_ARGS__)
#define SAR(...) INSTR_(sar, __VA_ARGS__)
#define SHLX(_0, _1, _2) INSTR_(shlx, _0, _1, _2)
#define SHRX(_0, _1, _2) INSTR_(shrx, _0, _1, _2)
#define RORX(_0, _1, _2) INSTR_(rorx, _0, _1, _2)
#define DEC(_0) INSTR_(dec, _0)
#define INC(_0) INSTR_(inc, _0)

#define and(_0, _1) AND(_0, _1)
#define or(_0, _1) OR(_0, _1)
#define xor(_0, _1) XOR(_0, _1)
#define add(_0, _1) ADD(_0, _1)
#define sub(_0, _1) SUB(_0, _1)
#define imul(_0, _1) IMUL(_0, _1)
#define sal(...) SAL(__VA_ARGS__)
#define sar(...) SAR(__VA_ARGS__)
#define shlx(_0, _1, _2) SHLX(_0, _1, _2)
#define shrx(_0, _1, _2) SHRX(_0, _1, _2)
#define rorx(_0, _1, _2) RORX(_0, _1, _2)
#define dec(_0) DEC(_0)
#define inc(_0) INC(_0)

// Memory access

#define LEA(_0, _1) INSTR_(lea, _0, _1)
#define MOV(_0, _1) INSTR_(mov, _0, _1)
#define MOVD(_0, _1) INSTR_(movd, _0, _1)
#define MOVL(_0, _1) INSTR_(movl, _0, _1)
#define MOVQ(_0, _1) INSTR_(movq, _0, _1)
#define CMOVA(_0, _1) INSTR_(cmova, _0, _1)
#define CMOVAE(_0, _1) INSTR_(cmovae, _0, _1)
#define CMOVB(_0, _1) INSTR_(cmovb, _0, _1)
#define CMOVBE(_0, _1) INSTR_(cmovbe, _0, _1)
#define CMOVC(_0, _1) INSTR_(cmovc, _0, _1)
#define CMOVP(_0, _1) INSTR_(cmovp, _0, _1)
#define CMOVO(_0, _1) INSTR_(cmovo, _0, _1)
#define CMOVS(_0, _1) INSTR_(cmovs, _0, _1)
#define CMOVE(_0, _1) INSTR_(cmove, _0, _1)
#define CMOVZ(_0, _1) INSTR_(cmovz, _0, _1)
#define CMOVG(_0, _1) INSTR_(cmovg, _0, _1)
#define CMOVGE(_0, _1) INSTR_(cmovge, _0, _1)
#define CMOVL(_0, _1) INSTR_(cmovl, _0, _1)
#define CMOVLE(_0, _1) INSTR_(cmovle, _0, _1)
#define CMOVNA(_0, _1) INSTR_(cmovna, _0, _1)
#define CMOVNAE(_0, _1) INSTR_(cmovnae, _0, _1)
#define CMOVNB(_0, _1) INSTR_(cmovnb, _0, _1)
#define CMOVNBE(_0, _1) INSTR_(cmovnbe, _0, _1)
#define CMOVNC(_0, _1) INSTR_(cmovnc, _0, _1)
#define CMOVNP(_0, _1) INSTR_(cmovnp, _0, _1)
#define CMOVNO(_0, _1) INSTR_(cmovno, _0, _1)
#define CMOVNS(_0, _1) INSTR_(cmovns, _0, _1)
#define CMOVNE(_0, _1) INSTR_(cmovne, _0, _1)
#define CMOVNZ(_0, _1) INSTR_(cmovnz, _0, _1)
#define CMOVNG(_0, _1) INSTR_(cmovng, _0, _1)
#define CMOVNGE(_0, _1) INSTR_(cmovnge, _0, _1)
#define CMOVNL(_0, _1) INSTR_(cmovnl, _0, _1)
#define CMOVNLE(_0, _1) INSTR_(cmovnle, _0, _1)

#define lea(_0, _1) LEA(_0, _1)
#define mov(_0, _1) MOV(_0, _1)
#define movd(_0, _1) MOVD(_0, _1)
#define movl(_0, _1) MOVL(_0, _1)
#define movq(_0, _1) MOVQ(_0, _1)
#define cmova(_0, _1) CMOVA(_0, _1)
#define cmovae(_0, _1) CMOVAE(_0, _1)
#define cmovb(_0, _1) CMOVB(_0, _1)
#define cmovbe(_0, _1) CMOVBE(_0, _1)
#define cmovc(_0, _1) CMOVC(_0, _1)
#define cmovp(_0, _1) CMOVP(_0, _1)
#define cmovo(_0, _1) CMOVO(_0, _1)
#define cmovs(_0, _1) CMOVS(_0, _1)
#define cmove(_0, _1) CMOVE(_0, _1)
#define cmovz(_0, _1) CMOVZ(_0, _1)
#define cmovg(_0, _1) CMOVG(_0, _1)
#define cmovge(_0, _1) CMOVGE(_0, _1)
#define cmovl(_0, _1) CMOVL(_0, _1)
#define cmovle(_0, _1) CMOVLE(_0, _1)
#define cmovna(_0, _1) CMOVNA(_0, _1)
#define cmovnae(_0, _1) CMOVNAE(_0, _1)
#define cmovnb(_0, _1) CMOVNB(_0, _1)
#define cmovnbe(_0, _1) CMOVNBE(_0, _1)
#define cmovnc(_0, _1) CMOVNC(_0, _1)
#define cmovnp(_0, _1) CMOVNP(_0, _1)
#define cmovno(_0, _1) CMOVNO(_0, _1)
#define cmovns(_0, _1) CMOVNS(_0, _1)
#define cmovne(_0, _1) CMOVNE(_0, _1)
#define cmovnz(_0, _1) CMOVNZ(_0, _1)
#define cmovng(_0, _1) CMOVNG(_0, _1)
#define cmovnge(_0, _1) CMOVNGE(_0, _1)
#define cmovnl(_0, _1) CMOVNL(_0, _1)
#define cmovnle(_0, _1) CMOVNLE(_0, _1)

// Vector moves

#define MOVSS(_0, _1) INSTR_(movss, _0, _1)
#define MOVSD(_0, _1) INSTR_(movsd, _0, _1)
#define MOVAPS(_0, _1) INSTR_(movaps, _0, _1)
#define MOVAPD(_0, _1) INSTR_(movaps, _0, _1) //use movaps because it is shorter
#define MOVDDUP(_0, _1) INSTR_(movddup, _0, _1)
#define MOVLPS(_0, _1) INSTR_(movlps, _0, _1)
#define MOVHPS(_0, _1) INSTR_(movhps, _0, _1)
#define MOVLPD(_0, _1) INSTR_(movlpd, _0, _1)
#define MOVHPD(_0, _1) INSTR_(movhpd, _0, _1)

#define movss(_0, _1) MOVSS(_0, _1)
#define movsd(_0, _1) MOVSD(_0, _1)
#define movaps(_0, _1) MOVAPS(_0, _1)
#define movapd(_0, _1) MOVAPD(_0, _1)
#define movddup(_0, _1) MOVDDUP(_0, _1)
#define movlps(_0, _1) MOVLPS(_0, _1)
#define movhps(_0, _1) MOVHPS(_0, _1)
#define movlpd(_0, _1) MOVLPD(_0, _1)
#define movhpd(_0, _1) MOVHPD(_0, _1)

#define VMOVDDUP(_0, _1) INSTR_(vmovddup, _0, _1)
#define VMOVSLDUP(_0, _1) INSTR_(vmovsldup, _0, _1)
#define VMOVSHDUP(_0, _1) INSTR_(vmovshdup, _0, _1)
#define VMOVD(_0, _1) INSTR_(vmovd, _0, _1)
#define VMOVQ(_0, _1) INSTR_(vmovq, _0, _1)
#define VMOVSS(_0, _1) INSTR_(vmovss, _0, _1)
#define VMOVSD(_0, _1) INSTR_(vmovsd, _0, _1)
#define VMOVAPS(_0, _1) INSTR_(vmovaps, _0, _1)
#define VMOVUPS(_0, _1) INSTR_(vmovups, _0, _1)
#define VMOVAPD(_0, _1) INSTR_(vmovapd, _0, _1)
#define VMOVUPD(_0, _1) INSTR_(vmovupd, _0, _1)
#define VMOVLPS(...) INSTR_(vmovlps, __VA_ARGS__)
#define VMOVHPS(...) INSTR_(vmovhps, __VA_ARGS__)
#define VMOVLPD(...) INSTR_(vmovlpd, __VA_ARGS__)
#define VMOVHPD(...) INSTR_(vmovhpd, __VA_ARGS__)
#define VMOVDQA(_0, _1) INSTR_(vmovdqa, _0, _1)
#define VMOVDQA32(_0, _1) INSTR_(vmovdqa32, _0, _1)
#define VMOVDQA64(_0, _1) INSTR_(vmovdqa64, _0, _1)
#define VMOVDQU(_0, _1) INSTR_(vmovdqu, _0, _1)
#define VBROADCASTSS(_0, _1) INSTR_(vbroadcastss, _0, _1)
#define VBROADCASTSD(_0, _1) INSTR_(vbroadcastsd, _0, _1)
#define VPBROADCASTD(_0, _1) INSTR_(vpbroadcastd, _0, _1)
#define VPBROADCASTQ(_0, _1) INSTR_(vpbroadcastq, _0, _1)
#define VBROADCASTF128(_0, _1) INSTR_(vbroadcastf128, _0, _1)
#define VBROADCASTF64X4(_0, _1) INSTR_(vbroadcastf64x4, _0, _1)
#define VGATHERDPS(...) INSTR_(vgatherdps, __VA_ARGS__)
#define VSCATTERDPS(_0, _1) INSTR_(vscatterdps, _0, _1)
#define VGATHERDPD(...) INSTR_(vgatherdpd, __VA_ARGS__)
#define VSCATTERDPD(_0, _1) INSTR_(vscatterdpd, _0, _1)
#define VGATHERQPS(...) INSTR_(vgatherqps, __VA_ARGS__)
#define VSCATTERQPS(_0, _1) INSTR_(vscatterqps, _0, _1)
#define VGATHERQPD(...) INSTR_(vgatherqpd, __VA_ARGS__)
#define VSCATTERQPD(_0, _1) INSTR_(vscatterqpd, _0, _1)

#define vmovddup(_0, _1) VMOVDDUP(_0, _1)
#define vmovsldup(_0, _1) VMOVSLDUP(_0, _1)
#define vmovshdup(_0, _1) VMOVSHDUP(_0, _1)
#define vmovd(_0, _1) VMOVD(_0, _1)
#define vmovq(_0, _1) VMOVQ(_0, _1)
#define vmovss(_0, _1) VMOVSS(_0, _1)
#define vmovsd(_0, _1) VMOVSD(_0, _1)
#define vmovaps(_0, _1) VMOVAPS(_0, _1)
#define vmovups(_0, _1) VMOVUPS(_0, _1)
#define vmovapd(_0, _1) VMOVAPD(_0, _1)
#define vmovupd(_0, _1) VMOVUPD(_0, _1)
#define vmovlps(...) VMOVLPS(__VA_ARGS__)
#define vmovhps(...) VMOVHPS(__VA_ARGS__)
#define vmovlpd(...) VMOVLPD(__VA_ARGS__)
#define vmovhpd(...) VMOVHPD(__VA_ARGS__)
#define vmovdqa(_0, _1) VMOVDQA(_0, _1)
#define vmovdqa32(_0, _1) VMOVDQA32(_0, _1)
#define vmovdqa64(_0, _1) VMOVDQA64(_0, _1)
#define vmovdqu(_0, _1) VMOVDQU(_0, _1)
#define vbroadcastss(_0, _1) VBROADCASTSS(_0, _1)
#define vbroadcastsd(_0, _1) VBROADCASTSD(_0, _1)
#define vpbroadcastd(_0, _1) VPBROADCASTD(_0, _1)
#define vpbroadcastq(_0, _1) VPBROADCASTQ(_0, _1)
#define vbroadcastf128(_0, _1) VBROADCASTF128(_0, _1)
#define vbroadcastf64x4(_0, _1) VBROADCASTF64X4(_0, _1)
#define vgatherdps(...) VGATHERDPS(__VA_ARGS__)
#define vscatterdps(_0, _1) VSCATTERDPS(_0, _1)
#define vgatherdpd(...) VGATHERDPD(__VA_ARGS__)
#define vscatterdpd(_0, _1) VSCATTERDPD(_0, _1)
#define vgatherqps(...) VGATHERQPS(__VA_ARGS__)
#define vscatterqps(_0, _1) VSCATTERQPS(_0, _1)
#define vgatherqpd(...) VGATHERQPD(__VA_ARGS__)
#define vscatterqpd(_0, _1) VSCATTERQPD(_0, _1)

// Vector comparisons

#define VPCMPEQB(_0, _1, _2) INSTR_(vpcmpeqb, _0, _1, _2)
#define VPCMPEQW(_0, _1, _2) INSTR_(vpcmpeqw, _0, _1, _2)
#define VPCMPEQD(_0, _1, _2) INSTR_(vpcmpeqd, _0, _1, _2)

#define vpcmpeqb(_0, _1, _2) VPCMPEQB(_0, _1, _2)
#define vpcmpeqw(_0, _1, _2) VPCMPEQW(_0, _1, _2)
#define vpcmpeqd(_0, _1, _2) VPCMPEQD(_0, _1, _2)

// Vector integer math

#define VPADDB(_0, _1, _2) INSTR_(vpaddb, _0, _1, _2)
#define VPADDW(_0, _1, _2) INSTR_(vpaddw, _0, _1, _2)
#define VPADDD(_0, _1, _2) INSTR_(vpaddd, _0, _1, _2)
#define VPADDQ(_0, _1, _2) INSTR_(vpaddq, _0, _1, _2)

#define vpaddb(_0, _1, _2) VPADDB(_0, _1, _2)
#define vpaddw(_0, _1, _2) VPADDW(_0, _1, _2)
#define vpaddd(_0, _1, _2) VPADDD(_0, _1, _2)
#define vpaddq(_0, _1, _2) VPADDQ(_0, _1, _2)

// Vector math

#define ADDPS(_0, _1) INSTR_(addps, _0, _1)
#define ADDPD(_0, _1) INSTR_(addpd, _0, _1)
#define SUBPS(_0, _1) INSTR_(subps, _0, _1)
#define SUBPD(_0, _1) INSTR_(subpd, _0, _1)
#define MULPS(_0, _1) INSTR_(mulps, _0, _1)
#define MULPD(_0, _1) INSTR_(mulpd, _0, _1)
#define DIVPS(_0, _1) INSTR_(divps, _0, _1)
#define DIVPD(_0, _1) INSTR_(divpd, _0, _1)
#define XORPS(_0, _1) INSTR_(xorps, _0, _1)
#define XORPD(_0, _1) INSTR_(xorpd, _0, _1)

#define UCOMISS(_0, _1) INSTR_(ucomiss, _0, _1)
#define UCOMISD(_0, _1) INSTR_(ucomisd, _0, _1)
#define COMISS(_0, _1) INSTR_(comiss, _0, _1)
#define COMISD(_0, _1) INSTR_(comisd, _0, _1)

#define addps(_0, _1) ADDPS(_0, _1)
#define addpd(_0, _1) ADDPD(_0, _1)
#define subps(_0, _1) SUBPS(_0, _1)
#define subpd(_0, _1) SUBPD(_0, _1)
#define mulps(_0, _1) MULPS(_0, _1)
#define mulpd(_0, _1) MULPD(_0, _1)
#define divps(_0, _1) DIVPS(_0, _1)
#define divpd(_0, _1) DIVPD(_0, _1)
#define xorps(_0, _1) XORPS(_0, _1)
#define xorpd(_0, _1) XORPD(_0, _1)

#define ucomiss(_0, _1) UCOMISS(_0, _1)
#define ucomisd(_0, _1) UCOMISD(_0, _1)
#define cmoiss(_0, _1) COMISS(_0, _1)
#define comisd(_0, _1) COMISD(_0, _1)

#define VADDSUBPS(_0, _1, _2) INSTR_(vaddsubps, _0, _1, _2)
#define VADDSUBPD(_0, _1, _2) INSTR_(vaddsubpd, _0, _1, _2)
#define VHADDPD(_0, _1, _2) INSTR_(vhaddpd, _0, _1, _2)
#define VHADDPS(_0, _1, _2) INSTR_(vhaddps, _0, _1, _2)
#define VHSUBPD(_0, _1, _2) INSTR_(vhsubpd, _0, _1, _2)
#define VHSUBPS(_0, _1, _2) INSTR_(vhsubps, _0, _1, _2)
#define VADDPS(_0, _1, _2) INSTR_(vaddps, _0, _1, _2)
#define VADDPD(_0, _1, _2) INSTR_(vaddpd, _0, _1, _2)
#define VSUBPS(_0, _1, _2) INSTR_(vsubps, _0, _1, _2)
#define VSUBPD(_0, _1, _2) INSTR_(vsubpd, _0, _1, _2)
#define VMULSS(_0, _1, _2) INSTR_(vmulss, _0, _1, _2)
#define VMULSD(_0, _1, _2) INSTR_(vmulsd, _0, _1, _2)
#define VMULPS(_0, _1, _2) INSTR_(vmulps, _0, _1, _2)
#define VMULPD(_0, _1, _2) INSTR_(vmulpd, _0, _1, _2)
#define VDIVSS(_0, _1, _2) INSTR_(vdivss, _0, _1, _2)
#define VDIVSD(_0, _1, _2) INSTR_(vdivsd, _0, _1, _2)
#define VDIVPS(_0, _1, _2) INSTR_(vdivps, _0, _1, _2)
#define VDIVPD(_0, _1, _2) INSTR_(vdivpd, _0, _1, _2)
#define VPMULLD(_0, _1, _2) INSTR_(vpmulld, _0, _1, _2)
#define VPMULLQ(_0, _1, _2) INSTR_(vpmullq, _0, _1, _2)
#define VPADDD(_0, _1, _2) INSTR_(vpaddd, _0, _1, _2)
#define VPSLLD(_0, _1, _2) INSTR_(vpslld, _0, _1, _2)
#define VXORPS(_0, _1, _2) INSTR_(vxorps, _0, _1, _2)
#define VXORPD(_0, _1, _2) INSTR_(vxorpd, _0, _1, _2)
#define VPXORD(_0, _1, _2) INSTR_(vpxord, _0, _1, _2)

#define VUCOMISS(_0, _1) INSTR_(vucomiss, _0, _1)
#define VUCOMISD(_0, _1) INSTR_(vucomisd, _0, _1)
#define VCOMISS(_0, _1) INSTR_(vcomiss, _0, _1)
#define VCOMISD(_0, _1) INSTR_(vcomisd, _0, _1)

#define VMASKMOVPD(_0, _1, _2) INSTR_(vmaskmovpd, _0, _1, _2)
#define VMASKMOVPS(_0, _1, _2) INSTR_(vmaskmovps, _0, _1, _2)
#define VFMADD132SS(_0, _1, _2) INSTR_(vfmadd132ss, _0, _1, _2)
#define VFMADD213SS(_0, _1, _2) INSTR_(vfmadd213ss, _0, _1, _2)
#define VFMADD231SS(_0, _1, _2) INSTR_(vfmadd231ss, _0, _1, _2)
#define VFMADD132SD(_0, _1, _2) INSTR_(vfmadd132sd, _0, _1, _2)
#define VFMADD213SD(_0, _1, _2) INSTR_(vfmadd213sd, _0, _1, _2)
#define VFMADD231SD(_0, _1, _2) INSTR_(vfmadd231sd, _0, _1, _2)
#define VFMADD132PS(_0, _1, _2) INSTR_(vfmadd132ps, _0, _1, _2)
#define VFMADD213PS(_0, _1, _2) INSTR_(vfmadd213ps, _0, _1, _2)
#define VFMADD231PS(_0, _1, _2) INSTR_(vfmadd231ps, _0, _1, _2)
#define VFMADD132PD(_0, _1, _2) INSTR_(vfmadd132pd, _0, _1, _2)
#define VFMADD213PD(_0, _1, _2) INSTR_(vfmadd213pd, _0, _1, _2)
#define VFMADD231PD(_0, _1, _2) INSTR_(vfmadd231pd, _0, _1, _2)
#define VFMSUB132SS(_0, _1, _2) INSTR_(vfmsub132ss, _0, _1, _2)
#define VFMSUB213SS(_0, _1, _2) INSTR_(vfmsub213ss, _0, _1, _2)
#define VFMSUB231SS(_0, _1, _2) INSTR_(vfmsub231ss, _0, _1, _2)
#define VFMSUB132SD(_0, _1, _2) INSTR_(vfmsub132sd, _0, _1, _2)
#define VFMSUB213SD(_0, _1, _2) INSTR_(vfmsub213sd, _0, _1, _2)
#define VFMSUB231SD(_0, _1, _2) INSTR_(vfmsub231sd, _0, _1, _2)
#define VFMSUB132PS(_0, _1, _2) INSTR_(vfmsub132ps, _0, _1, _2)
#define VFMSUB213PS(_0, _1, _2) INSTR_(vfmsub213ps, _0, _1, _2)
#define VFMSUB231PS(_0, _1, _2) INSTR_(vfmsub231ps, _0, _1, _2)
#define VFMSUB132PD(_0, _1, _2) INSTR_(vfmsub132pd, _0, _1, _2)
#define VFMSUB213PD(_0, _1, _2) INSTR_(vfmsub213pd, _0, _1, _2)
#define VFMSUB231PD(_0, _1, _2) INSTR_(vfmsub231pd, _0, _1, _2)
#define VFNMADD132SS(_0, _1, _2) INSTR_(vfnmadd132ss, _0, _1, _2)
#define VFNMADD213SS(_0, _1, _2) INSTR_(vfnmadd213ss, _0, _1, _2)
#define VFNMADD231SS(_0, _1, _2) INSTR_(vfnmadd231ss, _0, _1, _2)
#define VFNMADD132SD(_0, _1, _2) INSTR_(vfnmadd132sd, _0, _1, _2)
#define VFNMADD213SD(_0, _1, _2) INSTR_(vfnmadd213sd, _0, _1, _2)
#define VFNMADD231SD(_0, _1, _2) INSTR_(vfnmadd231sd, _0, _1, _2)
#define VFNMADD132PS(_0, _1, _2) INSTR_(vfnmadd132ps, _0, _1, _2)
#define VFNMADD213PS(_0, _1, _2) INSTR_(vfnmadd213ps, _0, _1, _2)
#define VFNMADD231PS(_0, _1, _2) INSTR_(vfnmadd231ps, _0, _1, _2)
#define VFNMADD132PD(_0, _1, _2) INSTR_(vfnmadd132pd, _0, _1, _2)
#define VFNMADD213PD(_0, _1, _2) INSTR_(vfnmadd213pd, _0, _1, _2)
#define VFNMADD231PD(_0, _1, _2) INSTR_(vfnmadd231pd, _0, _1, _2)
#define VFNMSUB132SS(_0, _1, _2) INSTR_(vfnmsub132ss, _0, _1, _2)
#define VFNMSUB213SS(_0, _1, _2) INSTR_(vfnmsub213ss, _0, _1, _2)
#define VFNMSUB231SS(_0, _1, _2) INSTR_(vfnmsub231ss, _0, _1, _2)
#define VFNMSUB132SD(_0, _1, _2) INSTR_(vfnmsub132sd, _0, _1, _2)
#define VFNMSUB213SD(_0, _1, _2) INSTR_(vfnmsub213sd, _0, _1, _2)
#define VFNMSUB231SD(_0, _1, _2) INSTR_(vfnmsub231sd, _0, _1, _2)
#define VFNMSUB132PS(_0, _1, _2) INSTR_(vfnmsub132ps, _0, _1, _2)
#define VFNMSUB213PS(_0, _1, _2) INSTR_(vfnmsub213ps, _0, _1, _2)
#define VFNMSUB231PS(_0, _1, _2) INSTR_(vfnmsub231ps, _0, _1, _2)
#define VFNMSUB132PD(_0, _1, _2) INSTR_(vfnmsub132pd, _0, _1, _2)
#define VFNMSUB213PD(_0, _1, _2) INSTR_(vfnmsub213pd, _0, _1, _2)
#define VFNMSUB231PD(_0, _1, _2) INSTR_(vfnmsub231pd, _0, _1, _2)
#define VFMADDSUB132SS(_0, _1, _2) INSTR_(vfmaddsub132ss, _0, _1, _2)
#define VFMADDSUB213SS(_0, _1, _2) INSTR_(vfmaddsub213ss, _0, _1, _2)
#define VFMADDSUB231SS(_0, _1, _2) INSTR_(vfmaddsub231ss, _0, _1, _2)
#define VFMADDSUB132SD(_0, _1, _2) INSTR_(vfmaddsub132sd, _0, _1, _2)
#define VFMADDSUB213SD(_0, _1, _2) INSTR_(vfmaddsub213sd, _0, _1, _2)
#define VFMADDSUB231SD(_0, _1, _2) INSTR_(vfmaddsub231sd, _0, _1, _2)
#define VFMADDSUB132PS(_0, _1, _2) INSTR_(vfmaddsub132ps, _0, _1, _2)
#define VFMADDSUB213PS(_0, _1, _2) INSTR_(vfmaddsub213ps, _0, _1, _2)
#define VFMADDSUB231PS(_0, _1, _2) INSTR_(vfmaddsub231ps, _0, _1, _2)
#define VFMADDSUB132PD(_0, _1, _2) INSTR_(vfmaddsub132pd, _0, _1, _2)
#define VFMADDSUB213PD(_0, _1, _2) INSTR_(vfmaddsub213pd, _0, _1, _2)
#define VFMADDSUB231PD(_0, _1, _2) INSTR_(vfmaddsub231pd, _0, _1, _2)
#define VFMSUBADD132SS(_0, _1, _2) INSTR_(vfmsubadd132ss, _0, _1, _2)
#define VFMSUBADD213SS(_0, _1, _2) INSTR_(vfmsubadd213ss, _0, _1, _2)
#define VFMSUBADD231SS(_0, _1, _2) INSTR_(vfmsubadd231ss, _0, _1, _2)
#define VFMSUBADD132SD(_0, _1, _2) INSTR_(vfmsubadd132sd, _0, _1, _2)
#define VFMSUBADD213SD(_0, _1, _2) INSTR_(vfmsubadd213sd, _0, _1, _2)
#define VFMSUBADD231SD(_0, _1, _2) INSTR_(vfmsubadd231sd, _0, _1, _2)
#define VFMSUBADD132PS(_0, _1, _2) INSTR_(vfmsubadd132ps, _0, _1, _2)
#define VFMSUBADD213PS(_0, _1, _2) INSTR_(vfmsubadd213ps, _0, _1, _2)
#define VFMSUBADD231PS(_0, _1, _2) INSTR_(vfmsubadd231ps, _0, _1, _2)
#define VFMSUBADD132PD(_0, _1, _2) INSTR_(vfmsubadd132pd, _0, _1, _2)
#define VFMSUBADD213PD(_0, _1, _2) INSTR_(vfmsubadd213pd, _0, _1, _2)
#define VFMSUBADD231PD(_0, _1, _2) INSTR_(vfmsubadd231pd, _0, _1, _2)
#define VFMADDSS(_0, _1, _2, _3) INSTR_(vfmaddss, _0, _1, _2, _3)
#define VFMADDSD(_0, _1, _2, _3) INSTR_(vfmaddsd, _0, _1, _2, _3)
#define VFMADDPS(_0, _1, _2, _3) INSTR_(vfmaddps, _0, _1, _2, _3)
#define VFMADDPD(_0, _1, _2, _3) INSTR_(vfmaddpd, _0, _1, _2, _3)
#define VFMSUBSS(_0, _1, _2, _3) INSTR_(vfmsubss, _0, _1, _2, _3)
#define VFMSUBSD(_0, _1, _2, _3) INSTR_(vfmsubsd, _0, _1, _2, _3)
#define VFMSUBPS(_0, _1, _2, _3) INSTR_(vfmsubps, _0, _1, _2, _3)
#define VFMSUBPD(_0, _1, _2, _3) INSTR_(vfmsubpd, _0, _1, _2, _3)
#define VFNMADDSS(_0, _1, _2, _3) INSTR_(vfnmaddss, _0, _1, _2, _3)
#define VFNMADDSD(_0, _1, _2, _3) INSTR_(vfnmaddsd, _0, _1, _2, _3)
#define VFNMADDPS(_0, _1, _2, _3) INSTR_(vfnmaddps, _0, _1, _2, _3)
#define VFNMADDPD(_0, _1, _2, _3) INSTR_(vfnmaddpd, _0, _1, _2, _3)
#define VFNMSUBSS(_0, _1, _2, _3) INSTR_(vfnmsubss, _0, _1, _2, _3)
#define VFNMSUBSD(_0, _1, _2, _3) INSTR_(vfnmsubsd, _0, _1, _2, _3)
#define VFNMSUBPS(_0, _1, _2, _3) INSTR_(vfnmsubps, _0, _1, _2, _3)
#define VFNMSUBPD(_0, _1, _2, _3) INSTR_(vfnmsubpd, _0, _1, _2, _3)
#define VFMADDSUBSS(_0, _1, _2, _3) INSTR_(vfmaddsubss, _0, _1, _2, _3)
#define VFMADDSUBSD(_0, _1, _2, _3) INSTR_(vfmaddsubsd, _0, _1, _2, _3)
#define VFMADDSUBPS(_0, _1, _2, _3) INSTR_(vfmaddsubps, _0, _1, _2, _3)
#define VFMADDSUBPD(_0, _1, _2, _3) INSTR_(vfmaddsubpd, _0, _1, _2, _3)
#define VFMSUBADDSS(_0, _1, _2, _3) INSTR_(vfmsubaddss, _0, _1, _2, _3)
#define VFMSUBADDSD(_0, _1, _2, _3) INSTR_(vfmsubaddsd, _0, _1, _2, _3)
#define VFMSUBADDPS(_0, _1, _2, _3) INSTR_(vfmsubaddps, _0, _1, _2, _3)
#define VFMSUBADDPD(_0, _1, _2, _3) INSTR_(vfmsubaddpd, _0, _1, _2, _3)
#define V4FMADDSS(_0, _1, _2) INSTR_(v4fmaddss, _0, _1, _2)
#define V4FMADDPS(_0, _1, _2) INSTR_(v4fmaddps, _0, _1, _2)
#define V4FNMADDSS(_0, _1, _2) INSTR_(v4fnmaddss, _0, _1, _2)
#define V4FNMADDPS(_0, _1, _2) INSTR_(v4fnmaddps, _0, _1, _2)

#define vaddsubps(_0, _1, _2) VADDSUBPS(_0, _1, _2)
#define vaddsubpd(_0, _1, _2) VADDSUBPD(_0, _1, _2)
#define vhaddpd(_0, _1, _2) VHADDPD(_0, _1, _2)
#define vhaddps(_0, _1, _2) VHADDPS(_0, _1, _2)
#define vhsubpd(_0, _1, _2) VHSUBPD(_0, _1, _2)
#define vhsubps(_0, _1, _2) VHSUBPS(_0, _1, _2)
#define vaddps(_0, _1, _2) VADDPS(_0, _1, _2)
#define vaddpd(_0, _1, _2) VADDPD(_0, _1, _2)
#define vsubps(_0, _1, _2) VSUBPS(_0, _1, _2)
#define vsubpd(_0, _1, _2) VSUBPD(_0, _1, _2)
#define vmulss(_0, _1, _2) VMULSS(_0, _1, _2)
#define vmulps(_0, _1, _2) VMULPS(_0, _1, _2)
#define vmulsd(_0, _1, _2) VMULSD(_0, _1, _2)
#define vmulpd(_0, _1, _2) VMULPD(_0, _1, _2)
#define vdivss(_0, _1, _2) VDIVSS(_0, _1, _2)
#define vdivps(_0, _1, _2) VDIVPS(_0, _1, _2)
#define vdivsd(_0, _1, _2) VDIVSD(_0, _1, _2)
#define vdivpd(_0, _1, _2) VDIVPD(_0, _1, _2)
#define vpmulld(_0, _1, _2) VPMULLD(_0, _1, _2)
#define vpmullq(_0, _1, _2) VPMULLQ(_0, _1, _2)
#define vpaddd(_0, _1, _2) VPADDD(_0, _1, _2)
#define vpslld(_0, _1, _2) VPSLLD(_0, _1, _2)
#define vxorps(_0, _1, _2) VXORPS(_0, _1, _2)
#define vxorpd(_0, _1, _2) VXORPD(_0, _1, _2)
#define vpxord(_0, _1, _2) VPXORD(_0, _1, _2)

#define vucomiss(_0, _1) VUCOMISS(_0, _1)
#define vucomisd(_0, _1) VUCOMISD(_0, _1)
#define vcomiss(_0, _1) VCOMISS(_0, _1)
#define vcomisd(_0, _1) VCOMISD(_0, _1)

#define vfmadd132ss(_0, _1, _2) VFMADD132SS(_0, _1, _2)
#define vfmadd213ss(_0, _1, _2) VFMADD213SS(_0, _1, _2)
#define vfmadd231ss(_0, _1, _2) VFMADD231SS(_0, _1, _2)
#define vfmadd132sd(_0, _1, _2) VFMADD132SD(_0, _1, _2)
#define vfmadd213sd(_0, _1, _2) VFMADD213SD(_0, _1, _2)
#define vfmadd231sd(_0, _1, _2) VFMADD231SD(_0, _1, _2)
#define vfmadd132ps(_0, _1, _2) VFMADD132PS(_0, _1, _2)
#define vfmadd213ps(_0, _1, _2) VFMADD213PS(_0, _1, _2)
#define vfmadd231ps(_0, _1, _2) VFMADD231PS(_0, _1, _2)
#define vfmadd132pd(_0, _1, _2) VFMADD132PD(_0, _1, _2)
#define vfmadd213pd(_0, _1, _2) VFMADD213PD(_0, _1, _2)
#define vfmadd231pd(_0, _1, _2) VFMADD231PD(_0, _1, _2)
#define vfmadd132ss(_0, _1, _2) VFMADD132SS(_0, _1, _2)
#define vfmsub213ss(_0, _1, _2) VFMSUB213SS(_0, _1, _2)
#define vfmsub231ss(_0, _1, _2) VFMSUB231SS(_0, _1, _2)
#define vfmsub132sd(_0, _1, _2) VFMSUB132SD(_0, _1, _2)
#define vfmsub213sd(_0, _1, _2) VFMSUB213SD(_0, _1, _2)
#define vfmsub231sd(_0, _1, _2) VFMSUB231SD(_0, _1, _2)
#define vfmsub132ps(_0, _1, _2) VFMSUB132PS(_0, _1, _2)
#define vfmsub213ps(_0, _1, _2) VFMSUB213PS(_0, _1, _2)
#define vfmsub231ps(_0, _1, _2) VFMSUB231PS(_0, _1, _2)
#define vfmsub132pd(_0, _1, _2) VFMSUB132PD(_0, _1, _2)
#define vfmsub213pd(_0, _1, _2) VFMSUB213PD(_0, _1, _2)
#define vfmsub231pd(_0, _1, _2) VFMSUB231PD(_0, _1, _2)
#define vfnmadd132ss(_0, _1, _2) VFNMADD132SS(_0, _1, _2)
#define vfnmadd213ss(_0, _1, _2) VFNMADD213SS(_0, _1, _2)
#define vfnmadd231ss(_0, _1, _2) VFNMADD231SS(_0, _1, _2)
#define vfnmadd132sd(_0, _1, _2) VFNMADD132SD(_0, _1, _2)
#define vfnmadd213sd(_0, _1, _2) VFNMADD213SD(_0, _1, _2)
#define vfnmadd231sd(_0, _1, _2) VFNMADD231SD(_0, _1, _2)
#define vfnmadd132ps(_0, _1, _2) VFNMADD132PS(_0, _1, _2)
#define vfnmadd213ps(_0, _1, _2) VFNMADD213PS(_0, _1, _2)
#define vfnmadd231ps(_0, _1, _2) VFNMADD231PS(_0, _1, _2)
#define vfnmadd132pd(_0, _1, _2) VFNMADD132PD(_0, _1, _2)
#define vfnmadd213pd(_0, _1, _2) VFNMADD213PD(_0, _1, _2)
#define vfnmadd231pd(_0, _1, _2) VFNMADD231PD(_0, _1, _2)
#define vfnmadd132ss(_0, _1, _2) VFNMADD132SS(_0, _1, _2)
#define vfnmsub213ss(_0, _1, _2) VFNMSUB213SS(_0, _1, _2)
#define vfnmsub231ss(_0, _1, _2) VFNMSUB231SS(_0, _1, _2)
#define vfnmsub132sd(_0, _1, _2) VFNMSUB132SD(_0, _1, _2)
#define vfnmsub213sd(_0, _1, _2) VFNMSUB213SD(_0, _1, _2)
#define vfnmsub231sd(_0, _1, _2) VFNMSUB231SD(_0, _1, _2)
#define vfnmsub132ps(_0, _1, _2) VFNMSUB132PS(_0, _1, _2)
#define vfnmsub213ps(_0, _1, _2) VFNMSUB213PS(_0, _1, _2)
#define vfnmsub231ps(_0, _1, _2) VFNMSUB231PS(_0, _1, _2)
#define vfnmsub132pd(_0, _1, _2) VFNMSUB132PD(_0, _1, _2)
#define vfnmsub213pd(_0, _1, _2) VFNMSUB213PD(_0, _1, _2)
#define vfnmsub231pd(_0, _1, _2) VFNMSUB231PD(_0, _1, _2)
#define vfmaddsub132ss(_0, _1, _2) VFMADDSUB132SS(_0, _1, _2)
#define vfmaddsub213ss(_0, _1, _2) VFMADDSUB213SS(_0, _1, _2)
#define vfmaddsub231ss(_0, _1, _2) VFMADDSUB231SS(_0, _1, _2)
#define vfmaddsub132sd(_0, _1, _2) VFMADDSUB132SD(_0, _1, _2)
#define vfmaddsub213sd(_0, _1, _2) VFMADDSUB213SD(_0, _1, _2)
#define vfmaddsub231sd(_0, _1, _2) VFMADDSUB231SD(_0, _1, _2)
#define vfmaddsub132ps(_0, _1, _2) VFMADDSUB132PS(_0, _1, _2)
#define vfmaddsub213ps(_0, _1, _2) VFMADDSUB213PS(_0, _1, _2)
#define vfmaddsub231ps(_0, _1, _2) VFMADDSUB231PS(_0, _1, _2)
#define vfmaddsub132pd(_0, _1, _2) VFMADDSUB132PD(_0, _1, _2)
#define vfmaddsub213pd(_0, _1, _2) VFMADDSUB213PD(_0, _1, _2)
#define vfmaddsub231pd(_0, _1, _2) VFMADDSUB231PD(_0, _1, _2)
#define vfmsubadd132ss(_0, _1, _2) VFMSUBADD132SS(_0, _1, _2)
#define vfmsubadd213ss(_0, _1, _2) VFMSUBADD213SS(_0, _1, _2)
#define vfmsubadd231ss(_0, _1, _2) VFMSUBADD231SS(_0, _1, _2)
#define vfmsubadd132sd(_0, _1, _2) VFMSUBADD132SD(_0, _1, _2)
#define vfmsubadd213sd(_0, _1, _2) VFMSUBADD213SD(_0, _1, _2)
#define vfmsubadd231sd(_0, _1, _2) VFMSUBADD231SD(_0, _1, _2)
#define vfmsubadd132ps(_0, _1, _2) VFMSUBADD132PS(_0, _1, _2)
#define vfmsubadd213ps(_0, _1, _2) VFMSUBADD213PS(_0, _1, _2)
#define vfmsubadd231ps(_0, _1, _2) VFMSUBADD231PS(_0, _1, _2)
#define vfmsubadd132pd(_0, _1, _2) VFMSUBADD132PD(_0, _1, _2)
#define vfmsubadd213pd(_0, _1, _2) VFMSUBADD213PD(_0, _1, _2)
#define vfmsubadd231pd(_0, _1, _2) VFMSUBADD231PD(_0, _1, _2)
#define vfmaddss(_0, _1, _2, _3) VFMADDSS(_0, _1, _2, _3)
#define vfmaddsd(_0, _1, _2, _3) VFMADDSD(_0, _1, _2, _3)
#define vfmaddps(_0, _1, _2, _3) VFMADDPS(_0, _1, _2, _3)
#define vfmaddpd(_0, _1, _2, _3) VFMADDPD(_0, _1, _2, _3)
#define vfmsubss(_0, _1, _2, _3) VFMSUBSS(_0, _1, _2, _3)
#define vfmsubsd(_0, _1, _2, _3) VFMSUBSD(_0, _1, _2, _3)
#define vfmsubps(_0, _1, _2, _3) VFMSUBPS(_0, _1, _2, _3)
#define vfmsubpd(_0, _1, _2, _3) VFMSUBPD(_0, _1, _2, _3)
#define vfnmaddss(_0, _1, _2, _3) VFNMADDSS(_0, _1, _2, _3)
#define vfnmaddsd(_0, _1, _2, _3) VFNMADDSD(_0, _1, _2, _3)
#define vfnmaddps(_0, _1, _2, _3) VFNMADDPS(_0, _1, _2, _3)
#define vfnmaddpd(_0, _1, _2, _3) VFNMADDPD(_0, _1, _2, _3)
#define vfnmsubss(_0, _1, _2, _3) VFNMSUBSS(_0, _1, _2, _3)
#define vfnmsubsd(_0, _1, _2, _3) VFNMSUBSD(_0, _1, _2, _3)
#define vfnmsubps(_0, _1, _2, _3) VFNMSUBPS(_0, _1, _2, _3)
#define vfnmsubpd(_0, _1, _2, _3) VFNMSUBPD(_0, _1, _2, _3)
#define vfmaddsubss(_0, _1, _2, _3) VFMADDSUBSS(_0, _1, _2, _3)
#define vfmaddsubsd(_0, _1, _2, _3) VFMADDSUBSD(_0, _1, _2, _3)
#define vfmaddsubps(_0, _1, _2, _3) VFMADDSUBPS(_0, _1, _2, _3)
#define vfmaddsubpd(_0, _1, _2, _3) VFMADDSUBPD(_0, _1, _2, _3)
#define vfmsubaddss(_0, _1, _2, _3) VFMSUBADDSS(_0, _1, _2, _3)
#define vfmsubaddsd(_0, _1, _2, _3) VFMSUBADDSD(_0, _1, _2, _3)
#define vfmsubaddps(_0, _1, _2, _3) VFMSUBADDPS(_0, _1, _2, _3)
#define vfmsubaddpd(_0, _1, _2, _3) VFMSUBADDPD(_0, _1, _2, _3)
#define v4fmaddss(_0, _1, _2) V4FMADDSS(_0, _1, _2)
#define v4fmaddps(_0, _1, _2) V4FMADDPS(_0, _1, _2)
#define v4fnmaddss(_0, _1, _2) V4FNMADDSS(_0, _1, _2)
#define v4fnmaddps(_0, _1, _2) V4FNMADDPS(_0, _1, _2)

// Conversions

#define CVTSS2SD(_0, _1) INSTR_(cvtss2sd, _0, _1)
#define CVTSD2SS(_0, _1) INSTR_(cvtsd2ss, _0, _1)
#define CVTPS2PD(_0, _1) INSTR_(cvtps2pd, _0, _1)
#define CVTPD2PS(_0, _1) INSTR_(cvtpd2ps, _0, _1)

#define cvtss2sd(_0, _1) CVTSS2SD(_0, _1)
#define cvtsd2ss(_0, _1) CVTSD2SS(_0, _1)
#define cvtps2pd(_0, _1) CVTPS2PD(_0, _1)
#define cvtpd2ps(_0, _1) CVTPD2PS(_0, _1)

#define VCVTSS2SD(_0, _1) INSTR_(vcvtss2sd, _0, _1)
#define VCVTSD2SS(_0, _1) INSTR_(vcvtsd2ss, _0, _1)
#define VCVTPS2PD(_0, _1) INSTR_(vcvtps2pd, _0, _1)
#define VCVTPD2PS(_0, _1) INSTR_(vcvtpd2ps, _0, _1)

#define vcvtss2sd(_0, _1) VCVTSS2SD(_0, _1)
#define vcvtsd2ss(_0, _1) VCVTSD2SS(_0, _1)
#define vcvtps2pd(_0, _1) VCVTPS2PD(_0, _1)
#define vcvtpd2ps(_0, _1) VCVTPD2PS(_0, _1)

// Vector shuffles

#define PSHUFD(_0, _1, _2) INSTR_(pshufd, _0, _1, _2)
#define SHUFPS(_0, _1, _2) INSTR_(shufps, _0, _1, _2)
#define SHUFPD(_0, _1, _2) INSTR_(shufpd, _0, _1, _2)
#define UNPCKLPS(_0, _1) INSTR_(unpcklps, _0, _1)
#define UNPCKHPS(_0, _1) INSTR_(unpckhps, _0, _1)
#define UNPCKLPD(_0, _1) INSTR_(unpcklpd, _0, _1)
#define UNPCKHPD(_0, _1) INSTR_(unpckhpd, _0, _1)

#define pshufd(_0, _1, _2) PSHUFD(_0, _1, _2)
#define shufps(_0, _1, _2) SHUFPS(_0, _1, _2)
#define shufpd(_0, _1, _2) SHUFPD(_0, _1, _2)
#define unpcklps(_0, _1) UNPCKLPS(_0, _1)
#define unpckhps(_0, _1) UNPCKHPS(_0, _1)
#define unpcklpd(_0, _1) UNPCKLPD(_0, _1)
#define unpckhpd(_0, _1) UNPCKHPD(_0, _1)

#define VSHUFPS(_0, _1, _2, _3) INSTR_(vshufps, _0, _1, _2, _3)
#define VSHUFPD(_0, _1, _2, _3) INSTR_(vshufpd, _0, _1, _2, _3)
#define VPERMILPS(_0, _1, _2) INSTR_(vpermilps, _0, _1, _2)
#define VPERMILPD(_0, _1, _2) INSTR_(vpermilpd, _0, _1, _2)
#define VPERM2F128(_0, _1, _2, _3) INSTR_(vperm2f128, _0, _1, _2, _3)
#define VPERMPD(_0, _1, _2) INSTR_(vpermpd, _0, _1, _2)
#define VUNPCKLPS(_0, _1, _2) INSTR_(vunpcklps, _0, _1, _2)
#define VUNPCKHPS(_0, _1, _2) INSTR_(vunpckhps, _0, _1, _2)
#define VUNPCKLPD(_0, _1, _2) INSTR_(vunpcklpd, _0, _1, _2)
#define VUNPCKHPD(_0, _1, _2) INSTR_(vunpckhpd, _0, _1, _2)
#define VSHUFF32X4(_0, _1, _2, _3) INSTR_(vshuff32x4, _0, _1, _2, _3)
#define VSHUFF64X2(_0, _1, _2, _3) INSTR_(vshuff64x2, _0, _1, _2, _3)
#define VINSERTF128(_0, _1, _2, _3) INSTR_(vinsertf128, _0, _1, _2, _3)
#define VINSERTF32X4(_0, _1, _2, _3) INSTR_(vinsertf32x4, _0, _1, _2, _3)
#define VINSERTF32X8(_0, _1, _2, _3) INSTR_(vinsertf32x8, _0, _1, _2, _3)
#define VINSERTF64X2(_0, _1, _2, _3) INSTR_(vinsertf64x2, _0, _1, _2, _3)
#define VINSERTF64X4(_0, _1, _2, _3) INSTR_(vinsertf64x4, _0, _1, _2, _3)
#define VEXTRACTF128(_0, _1, _2) INSTR_(vextractf128, _0, _1, _2)
#define VEXTRACTF32X4(_0, _1, _2) INSTR_(vextractf32x4, _0, _1, _2)
#define VEXTRACTF32X8(_0, _1, _2) INSTR_(vextractf32x8, _0, _1, _2)
#define VEXTRACTF64X2(_0, _1, _2) INSTR_(vextractf64x2, _0, _1, _2)
#define VEXTRACTF64X4(_0, _1, _2) INSTR_(vextractf64x4, _0, _1, _2)
#define VBLENDPS(_0, _1, _2, _3) INSTR_(vblendps, _0, _1, _2, _3)
#define VBLENDPD(_0, _1, _2, _3) INSTR_(vblendpd, _0, _1, _2, _3)
#define VBLENDMPS(_0, _1, _2) INSTR_(vblendmps, _0, _1, _2)
#define VBLENDMPD(_0, _1, _2) INSTR_(vblendmpd, _0, _1, _2)

#define vshufps(_0, _1, _2, _3) VSHUFPS(_0, _1, _2, _3)
#define vshufpd(_0, _1, _2, _3) VSHUFPD(_0, _1, _2, _3)
#define vpermilps(_0, _1, _2) VPERMILPS(_0, _1, _2)
#define vpermilpd(_0, _1, _2) VPERMILPD(_0, _1, _2)
#define vperm2f128(_0, _1, _2, _3) VPERM2F128(_0, _1, _2, _3)
#define vpermpd(_0, _1, _2) VPERMPD(_0, _1, _2)
#define vunpcklps(_0, _1, _2) VUNPCKLPS(_0, _1, _2)
#define vunpckhps(_0, _1, _2) VUNPCKHPS(_0, _1, _2)
#define vunpcklpd(_0, _1, _2) VUNPCKLPD(_0, _1, _2)
#define vunpckhpd(_0, _1, _2) VUNPCKHPD(_0, _1, _2)
#define vshuff32x4(_0, _1, _2, _3) VSHUFF32x4(_0, _1, _2, _3)
#define vshuff64x2(_0, _1, _2, _3) VSHUFF64x2(_0, _1, _2, _3)
#define vinsertf128(_0, _1, _2, _3) VINSERTF128(_0, _1, _2, _3)
#define vinsertf32x4(_0, _1, _2, _3) VINSERTF32x4(_0, _1, _2, _3)
#define vinsertf32x8(_0, _1, _2, _3) VINSERTF32x8(_0, _1, _2, _3)
#define vinsertf64x2(_0, _1, _2, _3) VINSERTF64x2(_0, _1, _2, _3)
#define vinsertf64x4(_0, _1, _2, _3) VINSERTF64x4(_0, _1, _2, _3)
#define vextractf128(_0, _1, _2) VEXTRACTF128(_0, _1, _2)
#define vextractf32x4(_0, _1, _2) VEXTRACTF32x4(_0, _1, _2)
#define vextractf32x8(_0, _1, _2) VEXTRACTF32x8(_0, _1, _2)
#define vextractf64x2(_0, _1, _2) VEXTRACTF64x2(_0, _1, _2)
#define vextractf64x4(_0, _1, _2) VEXTRACTF64x4(_0, _1, _2)
#define vblendps(_0, _1, _2, _3) VBLENDPS(_0, _1, _2, _3)
#define vblendpd(_0, _1, _2, _3) VBLENDPD(_0, _1, _2, _3)
#define vblendmps(_0, _1, _2) VBLENDMSD(_0, _1, _2)
#define vblendmpd(_0, _1, _2) VBLENDMPD(_0, _1, _2)
#define vmaskmovps(_0, _1, _2) VMASKMOVPS(_0, _1, _2)
#define vmaskmovpd(_0, _1, _2) VMASKMOVPD(_0, _1, _2)

// Prefetches

#define PREFETCH(_0, _1) INSTR_(prefetcht##_0, _1)
#define PREFETCHW0(_0) INSTR_(prefetchw, _0)
#define PREFETCHW1(_0) INSTR_(prefetchwt1, _0)
#define VGATHERPFDPS(_0, _1) INSTR_(vgatherpf##_0##dps, _1)
#define VSCATTERPFDPS(_0, _1) INSTR_(vscatterpf##_0##dps, _1)
#define VGATHERPFDPD(_0, _1) INSTR_(vgatherpf##_0##dpd, _1)
#define VSCATTERPFDPD(_0, _1) INSTR_(vscatterpf##_0##dpd, _1)
#define VGATHERPFQPS(_0, _1) INSTR_(vgatherpf##_0##qps, _1)
#define VSCATTERPFQPS(_0, _1) INSTR_(vscatterpf##_0##qps, _1)
#define VGATHERPFQPD(_0, _1) INSTR_(vgatherpf##_0##qpd, _1)
#define VSCATTERPFQPD(_0, _1) INSTR_(vscatterpf##_0##qpd, _1)

#define prefetch(_0, _1) PREFETCH(_0, _1)
#define prefetchw0(_0) PREFETCHW0(_0)
#define prefetchw1(_0) PREFETCHW1(_0)
#define vgatherpfdps(_0, _1) VGATHERPFDPS(_0, _1)
#define vscatterpfdps(_0, _1) VSCATTERPFDPS(_0, _1)
#define vgatherpfdpd(_0, _1) VGATHERPFDPD(_0, _1)
#define vscatterpfdpd(_0, _1) VSCATTERPFDPD(_0, _1)
#define vgatherpfqps(_0, _1) VGATHERPFQPS(_0, _1)
#define vscatterpfqps(_0, _1) VSCATTERPFQPS(_0, _1)
#define vgatherpfqpd(_0, _1) VGATHERPFQPD(_0, _1)
#define vscatterpfqpd(_0, _1) VSCATTERPFQPD(_0, _1)

// Mask operations

#ifdef __MIC__

#define KMOVW(_0, _1) INSTR_(kmov, _0, _1)
#define JKNZD(_0, _1) INSTR_(jknzd, _0, _1)

#else

#define KMOVW(_0, _1) INSTR_(kmovw, _0, _1)
#define JKNZD(_0, _1) INSTR_(kortestw, _0, _0) INSTR_(jnz, _1)

#endif

#define KXNORW(_0, _1, _2) INSTR_(kxnorw, _0, _1, _2)
#define KSHIFTRW(_0, _1, _2) INSTR_(kshiftrw, _0, _1, _2)

#define kmovw(_0, _1) KMOVW(_0, _1)
#define jknzd(_0, _1) JKNZD(_0, _1)
#define kxnorw(_0, _1, _2) KXNORW(_0, _1, _2)
#define kshiftrw(_0, _1, _2) KSHIFTRW(_0, _1, _2)

// Other

#define RDTSC() INSTR_(rdtsc)
#define VZEROALL() INSTR_(vzeroall)
#define VZEROUPPER() INSTR_(vzeroupper)

#define rdtsc() RDTSC()
#define vzeroall() VZEROALL()
#define vzeroupper() VZEROUPPER()

#endif
