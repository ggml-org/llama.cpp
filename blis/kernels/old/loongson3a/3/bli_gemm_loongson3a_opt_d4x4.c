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


void bli_dgemm_loongson3a_opt_4x4
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
    uint64_t k_iter  = k / 4;
    uint64_t k_left  = k % 4;

	__asm__ volatile
	(
	 //General purpose registers
	 //
	 //$8=k_iter, $9=k_left
	 //$10=a address, $11=b address
	 //$12=prefetch a, $13=prefetch b
	 //$14=rs_c, $15=cs_c,
	 //
	 //$16=c00 address, $17=c01 address, 
	 //$18=c02 address, $19=c03 address,
	 //
	 //Floating-point registers
	 //
	 //$f0=a0, $f1=a1, $f2=a4, $f3=a3
	 //$f4=next_a0, $f5=next_a1, $f6=next_a2, $f7=next_a3
	 //
	 //$f8=b0, $f9=b1, $f10=b2, $f11=b3
	 //$f12=next_b0, $f13=next_b1, $f14=next_b2, $f15=next_b3
	 //
	 //$f16=a0b0, $f17=a0b1, $f18=a0b2, $f19=a0b3
	 //$f20=a1b0, $f21=a1b1, $f22=a1b2, $f23=a1b3
	 //$f24=a2b0, $f25=a2b1, $f26=a2b2, $f27=a2b3
	 //$f28=a3b0, $f29=a3b1, $f30=a3b2, $f31=a3b3
	 //
	 "ld       $8,   %0                \n\t"  //load k_iter
	 "dmtc1    $0,    $f16              \n\t"  //Init

	 "ld       $9,   %1                \n\t"  //load k_left
	 "dmtc1    $0,    $f17              \n\t"  //Init

	 "ld       $14,   %7                \n\t"  //load rs_c
	 "dmtc1    $0,    $f18              \n\t"  //Init

	 "ld       $15,   %8                \n\t"  //load cs_c
	 "dmtc1    $0,    $f19              \n\t"  //Init

	 "ld       $16,   %6                \n\t"  //load c
	 "dmtc1    $0,    $f20              \n\t"  //Init

	 "ld       $10,   %2                \n\t"  //load a
	 "dmtc1    $0,    $f21              \n\t"  //Init

	 "ld       $11,   %3                \n\t"  //load b
	 "dmtc1    $0,    $f22              \n\t"  //Init

	 "dsll     $14,   $14,   3          \n\t"  //rs_c * sizeof(double)
	 "dmtc1    $0,    $f23              \n\t"  //Init

	 "dsll     $15,   $15,   3          \n\t"  //cs_c * sizeof(double)
	 "dmtc1    $0,    $f24              \n\t"  //Init

	 "dadd     $17,   $16,   $15        \n\t"  //c01 address
	 "ld       $12,   %9                \n\t"  //load kc
	 "dmtc1    $0,    $f25              \n\t"  //Init
	 "dmtc1    $0,    $f26              \n\t"  //Init

	 "dadd     $18,   $17,   $15        \n\t"  //c02 address
	 "dsll     $13,   $12,   5          \n\t"  //B prefetch distance= next panel B(nr*kc = kc*4*8bytes = kc<<5)
	 "dmtc1    $0,    $f27              \n\t"  //Init
	 "dmtc1    $0,    $f28              \n\t"  //Init

	 "dadd     $19,   $18,   $15        \n\t"  //c03 address
	 "dsll     $12,   $12,   4          \n\t"  //A prefetch distance= panel A/2(mr*kc/2 = kc*4*8bytes/2 = kc<<4)
	 "dmtc1    $0,    $f29              \n\t"  //Init
	 "dmtc1    $0,    $f30              \n\t"  //Init

	 "dadd     $13,   $11,   $13        \n\t"  //B prefetch address
	 "ld       $0,    0($16)            \n\t"  //prefetch c00
	 "dmtc1    $0,    $f31              \n\t"  //Init

	 "dadd     $12,   $10,   $12        \n\t"  //A prefetch address
	 "ld       $0,    0($17)            \n\t"  //prefetch c01

	 "gsLQC1   $f1, $f0, 0($10)         \n\t"  //load 2 values from a
	 "gsLQC1   $f9, $f8, 0($11)         \n\t"  //load 2 values from b
	 "gsLQC1   $f3, $f2, 1*16($10)      \n\t"  //load 2 values from a
	 "gsLQC1   $f11, $f10, 1*16($11)    \n\t"  //load 2 values from b

 	 "ld       $0,    0($18)            \n\t"  //prefetch c02
	 "ld       $0,    0($19)            \n\t"  //prefetch c03
	 "beqz     $8,   .Remain           \n\t"  
	 ".align 4                          \n\t"

	 ".MainLoop:                        \n\t"
	 "                                  \n\t"  //iteration 0
	 "daddiu   $8, $8, -1             \n\t"  //k_iter--
	 "gsLQC1   $f5, $f4, 2*16($10)      \n\t"  //load next 2 values from a
	 "madd.d   $f16, $f16, $f0, $f8     \n\t"  //a0b0
	 "madd.d   $f20, $f20, $f1, $f8     \n\t"  //a1b0
	 "                                  \n\t"  
	 "gsLQC1   $f13, $f12, 2*16($11)    \n\t"  //load next 2 values from b
	 "madd.d   $f17, $f17, $f0, $f9     \n\t"  //a0b1
	 "madd.d   $f21, $f21, $f1, $f9     \n\t"  //a1b1
	 "                                  \n\t"  
	 "gsLQC1   $f7, $f6, 3*16($10)      \n\t"  //load next 2 values from a
	 "madd.d   $f24, $f24, $f2, $f8     \n\t"  //a2b0
	 "madd.d   $f28, $f28, $f3, $f8     \n\t"  //a3b0
	 "                                  \n\t"  
	 "gsLQC1   $f15, $f14, 3*16($11)    \n\t"  //load next 2 values from b
	 "madd.d   $f25, $f25, $f2, $f9     \n\t"  //a2b1
	 "madd.d   $f29, $f29, $f3, $f9     \n\t"  //a3b1
	 "                                  \n\t"  
	 "ld       $0,   0($13)             \n\t"  //prefetch B
	 "madd.d   $f18, $f18, $f0, $f10    \n\t"  //a0b2
	 "madd.d   $f22, $f22, $f1, $f10    \n\t"  //a1b2
	 "                                  \n\t"  
	 "madd.d   $f19, $f19, $f0, $f11    \n\t"  //a0b3
	 "madd.d   $f23, $f23, $f1, $f11    \n\t"  //a1b3
	 "                                  \n\t"  
	 "ld       $0,   0($12)             \n\t"  //prefetch A
	 "madd.d   $f26, $f26, $f2, $f10    \n\t"  //a2b2
	 "madd.d   $f30, $f30, $f3, $f10    \n\t"  //a3b2
	 "                                  \n\t"  
	 "madd.d   $f27, $f27, $f2, $f11    \n\t"  //a2b3
	 "madd.d   $f31, $f31, $f3, $f11    \n\t"  //a3b3

	 "                                  \n\t"  //iteration 1
	 "gsLQC1   $f1, $f0, 4*16($10)      \n\t"  //load next 2 values from a
	 "madd.d   $f16, $f16, $f4, $f12    \n\t"  //a0b0
	 "madd.d   $f20, $f20, $f5, $f12    \n\t"  //a1b0
	 "                                  \n\t"  
	 "gsLQC1   $f9, $f8, 4*16($11)      \n\t"  //load next 2 values from b
	 "madd.d   $f17, $f17, $f4, $f13    \n\t"  //a0b1
	 "madd.d   $f21, $f21, $f5, $f13    \n\t"  //a1b1
	 "                                  \n\t"  
	 "gsLQC1   $f3, $f2, 5*16($10)      \n\t"  //load next 2 values from a
	 "madd.d   $f24, $f24, $f6, $f12    \n\t"  //a2b0
	 "madd.d   $f28, $f28, $f7, $f12    \n\t"  //a3b0
	 "                                  \n\t"  
	 "gsLQC1   $f11, $f10, 5*16($11)    \n\t"  //load next 2 values from b
	 "madd.d   $f25, $f25, $f6, $f13    \n\t"  //a2b1
	 "madd.d   $f29, $f29, $f7, $f13    \n\t"  //a3b1
	 "                                  \n\t"  
	 "ld       $0,   4*8($13)           \n\t"  //prefetch B
	 "madd.d   $f18, $f18, $f4, $f14    \n\t"  //a0b2
	 "madd.d   $f22, $f22, $f5, $f14    \n\t"  //a1b2
	 "                                  \n\t"  
	 "madd.d   $f19, $f19, $f4, $f15    \n\t"  //a0b3
	 "madd.d   $f23, $f23, $f5, $f15    \n\t"  //a1b3
	 "                                  \n\t"  
	 "ld       $0,   4*8($12)           \n\t"  //prefetch A
	 "madd.d   $f26, $f26, $f6, $f14    \n\t"  //a2b2
	 "madd.d   $f30, $f30, $f7, $f14    \n\t"  //a3b2
	 "                                  \n\t"  
	 "madd.d   $f27, $f27, $f6, $f15    \n\t"  //a2b3
	 "madd.d   $f31, $f31, $f7, $f15    \n\t"  //a3b3

	 "                                  \n\t"  //iteration 2
	 "gsLQC1   $f5, $f4, 6*16($10)      \n\t"  //load next 2 values from a
	 "madd.d   $f16, $f16, $f0, $f8     \n\t"  //a0b0
	 "madd.d   $f20, $f20, $f1, $f8     \n\t"  //a1b0
	 "                                  \n\t"  
	 "gsLQC1   $f13, $f12, 6*16($11)    \n\t"  //load next 2 values from b
	 "madd.d   $f17, $f17, $f0, $f9     \n\t"  //a0b1
	 "madd.d   $f21, $f21, $f1, $f9     \n\t"  //a1b1
	 "                                  \n\t"  
	 "gsLQC1   $f7, $f6, 7*16($10)      \n\t"  //load next 2 values from a
	 "madd.d   $f24, $f24, $f2, $f8     \n\t"  //a2b0
	 "madd.d   $f28, $f28, $f3, $f8     \n\t"  //a3b0
	 "daddu    $10,  $10, 16*8          \n\t"  //move A address
	 "                                  \n\t"  
	 "gsLQC1   $f15, $f14, 7*16($11)    \n\t"  //load next 2 values from b
	 "madd.d   $f25, $f25, $f2, $f9     \n\t"  //a2b1
	 "madd.d   $f29, $f29, $f3, $f9     \n\t"  //a3b1
	 "daddu    $11,  $11, 16*8          \n\t"  //move B address
	 "                                  \n\t"  
	 "ld       $0,   8*8($13)           \n\t"  //prefetch B
	 "madd.d   $f18, $f18, $f0, $f10    \n\t"  //a0b2
	 "madd.d   $f22, $f22, $f1, $f10    \n\t"  //a1b2
	 "                                  \n\t"  
	 "madd.d   $f19, $f19, $f0, $f11    \n\t"  //a0b3
	 "madd.d   $f23, $f23, $f1, $f11    \n\t"  //a1b3
	 "                                  \n\t"  
	 "ld       $0,   8*8($12)           \n\t"  //prefetch A
	 "madd.d   $f26, $f26, $f2, $f10    \n\t"  //a2b2
	 "madd.d   $f30, $f30, $f3, $f10    \n\t"  //a3b2
	 "                                  \n\t"  
	 "madd.d   $f27, $f27, $f2, $f11    \n\t"  //a2b3
	 "madd.d   $f31, $f31, $f3, $f11    \n\t"  //a3b3

	 "                                  \n\t"  //iteration 3
	 "gsLQC1   $f1, $f0,   0($10)       \n\t"  //load next 2 values from a
	 "madd.d   $f16, $f16, $f4, $f12    \n\t"  //a0b0
	 "madd.d   $f20, $f20, $f5, $f12    \n\t"  //a1b0
	 "                                  \n\t"  
	 "gsLQC1   $f9, $f8,   0($11)       \n\t"  //load next 2 values from b
	 "madd.d   $f17, $f17, $f4, $f13    \n\t"  //a0b1
	 "madd.d   $f21, $f21, $f5, $f13    \n\t"  //a1b1
	 "                                  \n\t"  
	 "gsLQC1   $f3, $f2,   1*16($10)    \n\t"  //load next 2 values from a
	 "madd.d   $f24, $f24, $f6, $f12    \n\t"  //a2b0
	 "madd.d   $f28, $f28, $f7, $f12    \n\t"  //a3b0
	 "                                  \n\t"  
	 "gsLQC1   $f11, $f10, 1*16($11)    \n\t"  //load next 2 values from b
	 "madd.d   $f25, $f25, $f6, $f13    \n\t"  //a2b1
	 "madd.d   $f29, $f29, $f7, $f13    \n\t"  //a3b1
	 "                                  \n\t"  
	 "ld       $0,   12*8($13)          \n\t"  //prefetch B
	 "madd.d   $f18, $f18, $f4, $f14    \n\t"  //a0b2
	 "madd.d   $f22, $f22, $f5, $f14    \n\t"  //a1b2
	 "daddu    $13,  $13, 16*8          \n\t"  //move prefetch B address
	 "                                  \n\t"  
	 "madd.d   $f19, $f19, $f4, $f15    \n\t"  //a0b3
	 "madd.d   $f23, $f23, $f5, $f15    \n\t"  //a1b3
	 "                                  \n\t"  
	 "ld       $0,   12*8($12)          \n\t"  //prefetch A
	 "madd.d   $f26, $f26, $f6, $f14    \n\t"  //a2b2
	 "madd.d   $f30, $f30, $f7, $f14    \n\t"  //a3b2
	 "daddu    $12,  $12, 16*8          \n\t"  //move prefetch B address
	 "                                  \n\t"  
	 "madd.d   $f27, $f27, $f6, $f15    \n\t"  //a2b3
	 "madd.d   $f31, $f31, $f7, $f15    \n\t"  //a3b3
	 "bnez     $8,  .MainLoop          \n\t"
	 ".align 4                          \n\t"

	 ".Remain:                          \n\t"  //deal with the tail. k%4
	 "beqz     $9,   .StoreC           \n\t"  
	 "andi     $8,  $9,  2            \n\t"
	 "nop                               \n\t"
	 "nop                               \n\t"

	 "beqz     $8,   .Remaink1         \n\t"
	 "nop                               \n\t"
	 "                                  \n\t"  // k%4=2
	 "gsLQC1   $f5, $f4, 2*16($10)      \n\t"  //load next 2 values from a
	 "madd.d   $f16, $f16, $f0, $f8     \n\t"  //a0b0
	 "madd.d   $f20, $f20, $f1, $f8     \n\t"  //a1b0
	 "                                  \n\t"  
	 "gsLQC1   $f13, $f12, 2*16($11)    \n\t"  //load next 2 values from b
	 "madd.d   $f17, $f17, $f0, $f9     \n\t"  //a0b1
	 "madd.d   $f21, $f21, $f1, $f9     \n\t"  //a1b1
	 "                                  \n\t"  
	 "gsLQC1   $f7, $f6, 3*16($10)      \n\t"  //load next 2 values from a
	 "madd.d   $f24, $f24, $f2, $f8     \n\t"  //a2b0
	 "madd.d   $f28, $f28, $f3, $f8     \n\t"  //a3b0
	 "daddu    $10,  $10,  8*8          \n\t"  //move A address
	 "                                  \n\t"  
	 "gsLQC1   $f15, $f14, 3*16($11)    \n\t"  //load next 2 values from b
	 "madd.d   $f25, $f25, $f2, $f9     \n\t"  //a2b1
	 "madd.d   $f29, $f29, $f3, $f9     \n\t"  //a3b1
	 "daddu    $11,  $11,  8*8          \n\t"  //move B address
	 "                                  \n\t"  
	 "ld       $0,   0($13)             \n\t"  //prefetch B
	 "madd.d   $f18, $f18, $f0, $f10    \n\t"  //a0b2
	 "madd.d   $f22, $f22, $f1, $f10    \n\t"  //a1b2
	 "                                  \n\t"  
	 "madd.d   $f19, $f19, $f0, $f11    \n\t"  //a0b3
	 "madd.d   $f23, $f23, $f1, $f11    \n\t"  //a1b3
	 "                                  \n\t"  
	 "ld       $0,   0($12)             \n\t"  //prefetch A
	 "madd.d   $f26, $f26, $f2, $f10    \n\t"  //a2b2
	 "madd.d   $f30, $f30, $f3, $f10    \n\t"  //a3b2
	 "                                  \n\t"  
	 "madd.d   $f27, $f27, $f2, $f11    \n\t"  //a2b3
	 "madd.d   $f31, $f31, $f3, $f11    \n\t"  //a3b3

	 "                                  \n\t"  
	 "gsLQC1   $f1, $f0, 0*16($10)      \n\t"  //load next 2 values from a
	 "madd.d   $f16, $f16, $f4, $f12    \n\t"  //a0b0
	 "madd.d   $f20, $f20, $f5, $f12    \n\t"  //a1b0
	 "                                  \n\t"  
	 "gsLQC1   $f9, $f8, 0*16($11)      \n\t"  //load next 2 values from b
	 "madd.d   $f17, $f17, $f4, $f13    \n\t"  //a0b1
	 "madd.d   $f21, $f21, $f5, $f13    \n\t"  //a1b1
	 "                                  \n\t"  
	 "gsLQC1   $f3, $f2, 1*16($10)      \n\t"  //load next 2 values from a
	 "madd.d   $f24, $f24, $f6, $f12    \n\t"  //a2b0
	 "madd.d   $f28, $f28, $f7, $f12    \n\t"  //a3b0
	 "                                  \n\t"  
	 "gsLQC1   $f11, $f10, 1*16($11)    \n\t"  //load next 2 values from b
	 "madd.d   $f25, $f25, $f6, $f13    \n\t"  //a2b1
	 "madd.d   $f29, $f29, $f7, $f13    \n\t"  //a3b1
	 "                                  \n\t"  
	 "ld       $0,   4*8($13)           \n\t"  //prefetch B
	 "madd.d   $f18, $f18, $f4, $f14    \n\t"  //a0b2
	 "madd.d   $f22, $f22, $f5, $f14    \n\t"  //a1b2
	 "                                  \n\t"  
	 "daddu    $13,  $13,  8*8          \n\t"  
	 "madd.d   $f19, $f19, $f4, $f15    \n\t"  //a0b3
	 "madd.d   $f23, $f23, $f5, $f15    \n\t"  //a1b3
	 "                                  \n\t"  
	 "ld       $0,   4*8($12)           \n\t"  //prefetch A
	 "madd.d   $f26, $f26, $f6, $f14    \n\t"  //a2b2
	 "madd.d   $f30, $f30, $f7, $f14    \n\t"  //a3b2
	 "                                  \n\t"  
	 "daddu    $12,  $12,  8*8          \n\t"  
	 "madd.d   $f27, $f27, $f6, $f15    \n\t"  //a2b3
	 "madd.d   $f31, $f31, $f7, $f15    \n\t"  //a3b3

	 ".align 4                          \n\t"
	 ".Remaink1:                        \n\t" // k%4=1
	 "andi     $8,  $9,  1            \n\t"
	 "beqz     $8,   .StoreC           \n\t"
	 "nop                               \n\t"
	 "                                  \n\t"  
	 "ld       $0,   0($13)             \n\t"  //prefetch B
	 "madd.d   $f16, $f16, $f0, $f8     \n\t"  //a0b0
	 "madd.d   $f20, $f20, $f1, $f8     \n\t"  //a1b0
	 "                                  \n\t"  
	 "madd.d   $f17, $f17, $f0, $f9     \n\t"  //a0b1
	 "madd.d   $f21, $f21, $f1, $f9     \n\t"  //a1b1
	 "                                  \n\t"  
	 "ld       $0,   0($12)             \n\t"  //prefetch A
	 "madd.d   $f24, $f24, $f2, $f8     \n\t"  //a2b0
	 "madd.d   $f28, $f28, $f3, $f8     \n\t"  //a3b0
	 "                                  \n\t"  
	 "madd.d   $f25, $f25, $f2, $f9     \n\t"  //a2b1
	 "madd.d   $f29, $f29, $f3, $f9     \n\t"  //a3b1
	 "                                  \n\t"  
	 "madd.d   $f18, $f18, $f0, $f10    \n\t"  //a0b2
	 "madd.d   $f22, $f22, $f1, $f10    \n\t"  //a1b2
	 "                                  \n\t"  
	 "madd.d   $f19, $f19, $f0, $f11    \n\t"  //a0b3
	 "madd.d   $f23, $f23, $f1, $f11    \n\t"  //a1b3
	 "                                  \n\t"  
	 "madd.d   $f26, $f26, $f2, $f10    \n\t"  //a2b2
	 "madd.d   $f30, $f30, $f3, $f10    \n\t"  //a3b2
	 "                                  \n\t"  
	 "madd.d   $f27, $f27, $f2, $f11    \n\t"  //a2b3
	 "madd.d   $f31, $f31, $f3, $f11    \n\t"  //a3b3

	 ".align 4                          \n\t"
	 ".StoreC:                          \n\t" //Write C
	 "                                  \n\t" //$f14=alpha, $f15=beta
	 "                                  \n\t"  
	 "ld       $8,     %4               \n\t" //load alpha address
	 "ld       $9,     %5               \n\t" //load beta address
	 "ldc1     $f14,   0($8)               \n\t" //load alpha 
	 "ldc1     $f15,   0($9)               \n\t" //load beta 
	 "                                  \n\t"  
	 "ldc1     $f0,    0($16)           \n\t" //load c00
	 "dadd     $20,    $16,   $14       \n\t" 
	 "ldc1     $f1,    0($17)           \n\t" //load c01
	 "dadd     $21,    $17,   $14       \n\t" 
	 "ldc1     $f2,    0($18)           \n\t" //load c02
	 "dadd     $22,    $18,   $14       \n\t" 
	 "ldc1     $f3,    0($19)           \n\t" //load c03
	 "dadd     $23,    $19,   $14       \n\t" 
	 "                                  \n\t"  

	 "ldc1     $f4,    0($20)           \n\t" //load c10
	 "dadd     $8,    $20,   $14       \n\t" 
	 "mul.d    $f0,    $f0,   $f15      \n\t" //c00 * beta

	 "ldc1     $f5,    0($21)           \n\t" //load c11
	 "dadd     $9,    $21,   $14       \n\t" 
	 "mul.d    $f1,    $f1,   $f15      \n\t" //c01 * beta

	 "ldc1     $f6,    0($22)           \n\t" //load c12
	 "dadd     $10,    $22,   $14       \n\t" 
	 "mul.d    $f2,    $f2,   $f15      \n\t" //c02 * beta

	 "ldc1     $f7,    0($23)           \n\t" //load c13
	 "dadd     $11,    $23,   $14       \n\t" 
	 "mul.d    $f3,    $f3,   $f15      \n\t" //c03 * beta
	 "                                  \n\t"  

	 "ldc1     $f8,    0($8)           \n\t" //load c20
	 "dadd     $12,    $8,   $14       \n\t" 
	 "mul.d    $f4,    $f4,   $f15      \n\t" //c10 * beta
	 "madd.d   $f16,   $f0,   $f16, $f14\n\t" //c00+=alpha*a0b0

	 "ldc1     $f9,    0($9)           \n\t" //load c21
	 "dadd     $13,    $9,   $14       \n\t" 
	 "mul.d    $f5,    $f5,   $f15      \n\t" //c11 * beta
	 "madd.d   $f17,   $f1,   $f17, $f14\n\t" //c01+=alpha*a0b1

	 "ldc1     $f10,    0($10)          \n\t" //load c22
	 "dadd     $24,    $10,   $14       \n\t" 
	 "mul.d    $f6,    $f6,   $f15      \n\t" //c12 * beta
	 "madd.d   $f18,   $f2,   $f18, $f14\n\t" //c02+=alpha*a0b2

	 "ldc1     $f11,    0($11)          \n\t" //load c23
	 "dadd     $25,    $11,   $14       \n\t" 
	 "mul.d    $f7,    $f7,   $f15      \n\t" //c13 * beta
	 "madd.d   $f19,   $f3,   $f19, $f14\n\t" //c03+=alpha*a0b3

	 "                                  \n\t"  
	 "ldc1     $f12,   0($12)           \n\t" //load c30
	 "mul.d    $f8,    $f8,   $f15      \n\t" //c20 * beta
	 "madd.d   $f20,   $f4,   $f20, $f14 \n\t" //c10+=alpha*a1b0

	 "ldc1     $f13,   0($13)           \n\t" //load c31
	 "mul.d    $f9,    $f9,   $f15      \n\t" //c21 * beta
	 "madd.d   $f21,   $f5,   $f21, $f14 \n\t" //c11+=alpha*a1b1

	 "ldc1     $f0,    0($24)           \n\t" //load c32
	 "mul.d    $f10,   $f10,  $f15      \n\t" //c22 * beta
	 "madd.d   $f22,   $f6,   $f22, $f14 \n\t" //c12+=alpha*a1b2

	 "ldc1     $f1,    0($25)           \n\t" //load c33
	 "mul.d    $f11,   $f11,  $f15      \n\t" //c23 * beta
	 "madd.d   $f23,   $f7,   $f23, $f14 \n\t" //c13+=alpha*a1b3
	 "                                  \n\t"  

	 "sdc1     $f16,   0($16)           \n\t" //store c00
	 "mul.d    $f12,   $f12,  $f15      \n\t" //c30 * beta
	 "madd.d   $f24,   $f8,   $f24, $f14 \n\t" //c20+=alpha*a2b0

	 "sdc1     $f17,   0($17)           \n\t" //store c01
	 "mul.d    $f13,   $f13,  $f15      \n\t" //c31 * beta
	 "madd.d   $f25,   $f9,   $f25, $f14 \n\t" //c21+=alpha*a2b1

	 "sdc1     $f18,   0($18)           \n\t" //store c02
	 "mul.d    $f0,    $f0,   $f15      \n\t" //c32 * beta
	 "madd.d   $f26,   $f10,  $f26, $f14 \n\t" //c22+=alpha*a2b2

	 "sdc1     $f19,   0($19)           \n\t" //store c03
	 "mul.d    $f1,    $f1,   $f15      \n\t" //c33 * beta
	 "madd.d   $f27,   $f11,  $f27, $f14 \n\t" //c23+=alpha*a2b3
	 "                                  \n\t"  

	 "sdc1     $f20,   0($20)           \n\t" //store c10
	 "madd.d   $f28,   $f12,  $f28, $f14 \n\t" //c30+=alpha*a3b0

	 "sdc1     $f21,   0($21)           \n\t" //store c11
	 "madd.d   $f29,   $f13,  $f29, $f14 \n\t" //c31+=alpha*a3b1

	 "sdc1     $f22,   0($22)           \n\t" //store c12
	 "madd.d   $f30,   $f0,  $f30, $f14 \n\t" //c32+=alpha*a3b2

	 "sdc1     $f23,   0($23)           \n\t" //store c13
	 "madd.d   $f31,   $f1,  $f31, $f14 \n\t" //c33+=alpha*a3b3
	 "                                  \n\t"  

	 "sdc1     $f24,   0($8)           \n\t" //store c20
	 "sdc1     $f25,   0($9)           \n\t" //store c21
	 "sdc1     $f26,   0($10)           \n\t" //store c22
	 "sdc1     $f27,   0($11)           \n\t" //store c23
	 "                                  \n\t"  
	 "sdc1     $f28,   0($12)           \n\t" //store c30
	 "sdc1     $f29,   0($13)           \n\t" //store c31
	 "sdc1     $f30,   0($24)           \n\t" //store c32
	 "sdc1     $f31,   0($25)           \n\t" //store c33
	 "                                  \n\t"  	 
	 ://output operands (none)
	 ://input operands
	  "m" (k_iter),
	  "m" (k_left),
	  "m" (a),
	  "m" (b),
	  "m" (alpha),
	  "m" (beta),
	  "m" (c),
	  "m" (rs_c),
	  "m" (cs_c),
	  "m" (k)
	 ://register clober list
	  //general purpose registers
	  "$8", "$9", "$10", "$11", 
	  "$12", "$13", "$14", "$15", 
	  "$16", "$17", "$18", "$19", 
	  "$20", "$21", "$22", "$23", 
	  "$24", "$25",
	  //floating-point registers
	  "$f0", "$f1", "$f2", "$f3",
	  "$f4", "$f5", "$f6", "$f7",
	  "$f8", "$f9", "$f10", "$f11",
	  "$f12", "$f13", "$f14", "$f15",
	  "$f16", "$f17", "$f18", "$f19",
	  "$f20", "$f21", "$f22", "$f23",
	  "$f24", "$f25", "$f26", "$f27",
	  "$f28", "$f29", "$f30", "$f31",
	  "memory"
	);

}

