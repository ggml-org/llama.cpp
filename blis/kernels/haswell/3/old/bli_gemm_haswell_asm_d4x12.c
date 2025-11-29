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


#define SGEMM_INPUT_GS_BETA_NZ \
	"vmovlps    (%%rcx        ),  %%xmm0,  %%xmm0  \n\t" \
	"vmovhps    (%%rcx,%%rsi,1),  %%xmm0,  %%xmm0  \n\t" \
	"vmovlps    (%%rcx,%%rsi,2),  %%xmm1,  %%xmm1  \n\t" \
	"vmovhps    (%%rcx,%%r13  ),  %%xmm1,  %%xmm1  \n\t" \
	"vshufps    $0x88,   %%xmm1,  %%xmm0,  %%xmm0  \n\t" \
	"vmovlps    (%%rcx,%%rsi,4),  %%xmm2,  %%xmm2  \n\t" \
	"vmovhps    (%%rcx,%%r15  ),  %%xmm2,  %%xmm2  \n\t" \
	/* We can't use vmovhps for loading the last element becauase that
	   might result in reading beyond valid memory. (vmov[lh]psd load
	   pairs of adjacent floats at a time.) So we need to use vmovss
	   instead. But since we're limited to using ymm0 through ymm2
	   (ymm3 contains beta and ymm4 through ymm15 contain the microtile)
	   and due to the way vmovss zeros out all bits above 31, we have to
	   load element 7 before element 6. */ \
	"vmovss     (%%rcx,%%r10  ),  %%xmm1           \n\t" \
	"vpermilps  $0xcf,   %%xmm1,  %%xmm1           \n\t" \
	"vmovlps    (%%rcx,%%r13,2),  %%xmm1,  %%xmm1  \n\t" \
	/*"vmovhps    (%%rcx,%%r10  ),  %%xmm1,  %%xmm1  \n\t"*/ \
	"vshufps    $0x88,   %%xmm1,  %%xmm2,  %%xmm2  \n\t" \
	"vperm2f128 $0x20,   %%ymm2,  %%ymm0,  %%ymm0  \n\t"

#define SGEMM_OUTPUT_GS_BETA_NZ \
	"vextractf128  $1, %%ymm0,  %%xmm2           \n\t" \
	"vmovss            %%xmm0, (%%rcx        )   \n\t" \
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t" \
	"vmovss            %%xmm1, (%%rcx,%%rsi,1)   \n\t" \
	"vpermilps  $0x39, %%xmm1,  %%xmm0           \n\t" \
	"vmovss            %%xmm0, (%%rcx,%%rsi,2)   \n\t" \
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t" \
	"vmovss            %%xmm1, (%%rcx,%%r13  )   \n\t" \
	"vmovss            %%xmm2, (%%rcx,%%rsi,4)   \n\t" \
	"vpermilps  $0x39, %%xmm2,  %%xmm1           \n\t" \
	"vmovss            %%xmm1, (%%rcx,%%r15  )   \n\t" \
	"vpermilps  $0x39, %%xmm1,  %%xmm2           \n\t" \
	"vmovss            %%xmm2, (%%rcx,%%r13,2)   \n\t" \
	"vpermilps  $0x39, %%xmm2,  %%xmm1           \n\t" \
	"vmovss            %%xmm1, (%%rcx,%%r10  )   \n\t"

void bli_sgemm_haswell_asm_4x24
     (
       dim_t               k0,
       float*     restrict alpha,
       float*     restrict a,
       float*     restrict b,
       float*     restrict beta,
       float*     restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	//void*   a_next = bli_auxinfo_next_a( data );
	//void*   b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	__asm__ volatile
	(
	"                                            \n\t"
	"vzeroall                                    \n\t" // zero all xmm/ymm registers.
	"                                            \n\t"
	"                                            \n\t"
	"movq                %2, %%rax               \n\t" // load address of a.
	"movq                %3, %%rbx               \n\t" // load address of b.
	//"movq                %9, %%r15               \n\t" // load address of b_next.
	"                                            \n\t"
	"                                            \n\t" // initialize loop by pre-loading
	"vmovaps            0 * 32(%%rbx), %%ymm0    \n\t"
	"vmovaps            1 * 32(%%rbx), %%ymm1    \n\t"
	"vmovaps            2 * 32(%%rbx), %%ymm2    \n\t"
	"                                            \n\t"
	"movq                %6, %%rcx               \n\t" // load address of c
	"movq                %7, %%rdi               \n\t" // load rs_c
	"leaq        (,%%rdi,4), %%rdi               \n\t" // rs_c *= sizeof(float)
	"                                            \n\t"
	"leaq   (%%rdi,%%rdi,2), %%r13               \n\t" // r13 = 3*rs_c;
	"prefetcht0   7 * 4(%%rcx)                   \n\t" // prefetch c + 0*rs_c
	"prefetcht0   7 * 4(%%rcx,%%rdi)             \n\t" // prefetch c + 1*rs_c
	"prefetcht0   7 * 4(%%rcx,%%rdi,2)           \n\t" // prefetch c + 2*rs_c
	"prefetcht0   7 * 4(%%rcx,%%r13)             \n\t" // prefetch c + 3*rs_c
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq      %0, %%rsi                         \n\t" // i = k_iter;
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
	"je     .SCONSIDKLEFT                        \n\t" // if i == 0, jump to code that
	"                                            \n\t" // contains the k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".SLOOPKITER:                                \n\t" // MAIN LOOP
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 0
	"prefetcht0  16 * 32(%%rax)                  \n\t"
	"                                            \n\t"
	"vbroadcastss       0 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm4    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm5    \n\t"
	"vfmadd231ps       %%ymm2, %%ymm3, %%ymm6    \n\t"
	"                                            \n\t"
	"vbroadcastss       1 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm7    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm8    \n\t"
	"vfmadd231ps       %%ymm2, %%ymm3, %%ymm9    \n\t"
	"                                            \n\t"
	"vbroadcastss       2 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm10   \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm11   \n\t"
	"vfmadd231ps       %%ymm2, %%ymm3, %%ymm12   \n\t"
	"                                            \n\t"
	"vbroadcastss       3 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm13   \n\t"
	"vmovaps            3 * 32(%%rbx), %%ymm0    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm14   \n\t"
	"vmovaps            4 * 32(%%rbx), %%ymm1    \n\t"
	"vfmadd231ps       %%ymm2, %%ymm3, %%ymm15   \n\t"
	"vmovaps            5 * 32(%%rbx), %%ymm2    \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 1
	"vbroadcastss       4 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm4    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm5    \n\t"
	"vfmadd231ps       %%ymm2, %%ymm3, %%ymm6    \n\t"
	"                                            \n\t"
	"vbroadcastss       5 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm7    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm8    \n\t"
	"vfmadd231ps       %%ymm2, %%ymm3, %%ymm9    \n\t"
	"                                            \n\t"
	"vbroadcastss       6 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm10   \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm11   \n\t"
	"vfmadd231ps       %%ymm2, %%ymm3, %%ymm12   \n\t"
	"                                            \n\t"
	"vbroadcastss       7 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm13   \n\t"
	"vmovaps            6 * 32(%%rbx), %%ymm0    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm14   \n\t"
	"vmovaps            7 * 32(%%rbx), %%ymm1    \n\t"
	"vfmadd231ps       %%ymm2, %%ymm3, %%ymm15   \n\t"
	"vmovaps            8 * 32(%%rbx), %%ymm2    \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 2
	"prefetcht0  22 * 32(%%rax)                  \n\t"
	"                                            \n\t"
	"vbroadcastss       8 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm4    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm5    \n\t"
	"vfmadd231ps       %%ymm2, %%ymm3, %%ymm6    \n\t"
	"                                            \n\t"
	"vbroadcastss       9 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm7    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm8    \n\t"
	"vfmadd231ps       %%ymm2, %%ymm3, %%ymm9    \n\t"
	"                                            \n\t"
	"vbroadcastss      10 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm10   \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm11   \n\t"
	"vfmadd231ps       %%ymm2, %%ymm3, %%ymm12   \n\t"
	"                                            \n\t"
	"vbroadcastss      11 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm13   \n\t"
	"vmovaps            9 * 32(%%rbx), %%ymm0    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm14   \n\t"
	"vmovaps           10 * 32(%%rbx), %%ymm1    \n\t"
	"vfmadd231ps       %%ymm2, %%ymm3, %%ymm15   \n\t"
	"vmovaps           11 * 32(%%rbx), %%ymm2    \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 3
	"vbroadcastss      12 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm4    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm5    \n\t"
	"vfmadd231ps       %%ymm2, %%ymm3, %%ymm6    \n\t"
	"                                            \n\t"
	"vbroadcastss      13 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm7    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm8    \n\t"
	"vfmadd231ps       %%ymm2, %%ymm3, %%ymm9    \n\t"
	"                                            \n\t"
	"vbroadcastss      14 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm10   \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm11   \n\t"
	"vfmadd231ps       %%ymm2, %%ymm3, %%ymm12   \n\t"
	"                                            \n\t"
	"vbroadcastss      15 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm13   \n\t"
	"vmovaps           12 * 32(%%rbx), %%ymm0    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm14   \n\t"
	"vmovaps           13 * 32(%%rbx), %%ymm1    \n\t"
	"vfmadd231ps       %%ymm2, %%ymm3, %%ymm15   \n\t"
	"vmovaps           14 * 32(%%rbx), %%ymm2    \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"addq          $4 *  4 * 4, %%rax            \n\t" // a += 4*4  (unroll x mr)
	"addq          $4 * 24 * 4, %%rbx            \n\t" // b += 4*24 (unroll x nr)
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .SLOOPKITER                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SCONSIDKLEFT:                              \n\t"
	"                                            \n\t"
	"movq      %1, %%rsi                         \n\t" // i = k_left;
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
	"je     .SPOSTACCUM                          \n\t" // if i == 0, we're done; jump to end.
	"                                            \n\t" // else, we prepare to enter k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".SLOOPKLEFT:                                \n\t" // EDGE LOOP
	"                                            \n\t"
	"prefetcht0  16 * 32(%%rax)                  \n\t"
	"                                            \n\t"
	"vbroadcastss       0 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm4    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm5    \n\t"
	"vfmadd231ps       %%ymm2, %%ymm3, %%ymm6    \n\t"
	"                                            \n\t"
	"vbroadcastss       1 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm7    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm8    \n\t"
	"vfmadd231ps       %%ymm2, %%ymm3, %%ymm9    \n\t"
	"                                            \n\t"
	"vbroadcastss       2 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm10   \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm11   \n\t"
	"vfmadd231ps       %%ymm2, %%ymm3, %%ymm12   \n\t"
	"                                            \n\t"
	"vbroadcastss       3 *  4(%%rax), %%ymm3    \n\t"
	"vfmadd231ps       %%ymm0, %%ymm3, %%ymm13   \n\t"
	"vmovaps            3 * 32(%%rbx), %%ymm0    \n\t"
	"vfmadd231ps       %%ymm1, %%ymm3, %%ymm14   \n\t"
	"vmovaps            4 * 32(%%rbx), %%ymm1    \n\t"
	"vfmadd231ps       %%ymm2, %%ymm3, %%ymm15   \n\t"
	"vmovaps            5 * 32(%%rbx), %%ymm2    \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"addq          $1 *  4 * 4, %%rax            \n\t" // a += 1*4  (unroll x mr)
	"addq          $1 * 24 * 4, %%rbx            \n\t" // b += 1*24 (unroll x nr)
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .SLOOPKLEFT                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SPOSTACCUM:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq         %4, %%rax                      \n\t" // load address of alpha
	"movq         %5, %%rbx                      \n\t" // load address of beta 
	"vbroadcastss    (%%rax), %%ymm0             \n\t" // load alpha and duplicate
	"vbroadcastss    (%%rbx), %%ymm3             \n\t" // load beta and duplicate
	"                                            \n\t"
	"vmulps           %%ymm0,  %%ymm4,  %%ymm4   \n\t" // scale by alpha
	"vmulps           %%ymm0,  %%ymm5,  %%ymm5   \n\t"
	"vmulps           %%ymm0,  %%ymm6,  %%ymm6   \n\t"
	"vmulps           %%ymm0,  %%ymm7,  %%ymm7   \n\t"
	"vmulps           %%ymm0,  %%ymm8,  %%ymm8   \n\t"
	"vmulps           %%ymm0,  %%ymm9,  %%ymm9   \n\t"
	"vmulps           %%ymm0,  %%ymm10, %%ymm10  \n\t"
	"vmulps           %%ymm0,  %%ymm11, %%ymm11  \n\t"
	"vmulps           %%ymm0,  %%ymm12, %%ymm12  \n\t"
	"vmulps           %%ymm0,  %%ymm13, %%ymm13  \n\t"
	"vmulps           %%ymm0,  %%ymm14, %%ymm14  \n\t"
	"vmulps           %%ymm0,  %%ymm15, %%ymm15  \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq                %8, %%rsi               \n\t" // load cs_c
	"leaq        (,%%rsi,4), %%rsi               \n\t" // rsi = cs_c * sizeof(float)
	"                                            \n\t"
	"leaq   (%%rcx,%%rsi,8), %%rdx               \n\t" // rdx = c +  8*cs_c;
	"leaq   (%%rdx,%%rsi,8), %%r12               \n\t" // r12 = c + 16*cs_c;
	"                                            \n\t"
	"leaq   (%%rsi,%%rsi,2), %%r13               \n\t" // r13 = 3*cs_c;
	"leaq   (%%rsi,%%rsi,4), %%r15               \n\t" // r15 = 5*cs_c;
	"leaq   (%%r13,%%rsi,4), %%r10               \n\t" // r10 = 7*cs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // now avoid loading C if beta == 0
	"                                            \n\t"
	"vxorps    %%ymm0,  %%ymm0,  %%ymm0          \n\t" // set ymm0 to zero.
	"vucomiss  %%xmm0,  %%xmm3                   \n\t" // set ZF if beta == 0.
	"je      .SBETAZERO                          \n\t" // if ZF = 1, jump to beta == 0 case
	"                                            \n\t"
	"                                            \n\t"
	"cmpq       $4, %%rsi                        \n\t" // set ZF if (4*cs_c) == 4.
	"jz      .SROWSTORED                         \n\t" // jump to row storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SGENSTORED:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	SGEMM_INPUT_GS_BETA_NZ
	"vfmadd213ps      %%ymm4,  %%ymm3,  %%ymm0   \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	SGEMM_INPUT_GS_BETA_NZ
	"vfmadd213ps      %%ymm7,  %%ymm3,  %%ymm0   \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	SGEMM_INPUT_GS_BETA_NZ
	"vfmadd213ps      %%ymm10, %%ymm3,  %%ymm0   \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	SGEMM_INPUT_GS_BETA_NZ
	"vfmadd213ps      %%ymm13, %%ymm3,  %%ymm0   \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	//"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"movq      %%rdx, %%rcx                      \n\t" // rcx = c + 8*cs_c
	"                                            \n\t"
	"                                            \n\t"
	SGEMM_INPUT_GS_BETA_NZ
	"vfmadd213ps      %%ymm5,  %%ymm3,  %%ymm0   \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	SGEMM_INPUT_GS_BETA_NZ
	"vfmadd213ps      %%ymm8,  %%ymm3,  %%ymm0   \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	SGEMM_INPUT_GS_BETA_NZ
	"vfmadd213ps      %%ymm11, %%ymm3,  %%ymm0   \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	SGEMM_INPUT_GS_BETA_NZ
	"vfmadd213ps      %%ymm14, %%ymm3,  %%ymm0   \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	//"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"movq      %%r12, %%rcx                      \n\t" // rcx = c + 16*cs_c
	"                                            \n\t"
	"                                            \n\t"
	SGEMM_INPUT_GS_BETA_NZ
	"vfmadd213ps      %%ymm6,  %%ymm3,  %%ymm0   \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	SGEMM_INPUT_GS_BETA_NZ
	"vfmadd213ps      %%ymm9,  %%ymm3,  %%ymm0   \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	SGEMM_INPUT_GS_BETA_NZ
	"vfmadd213ps      %%ymm12, %%ymm3,  %%ymm0   \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	SGEMM_INPUT_GS_BETA_NZ
	"vfmadd213ps      %%ymm15, %%ymm3,  %%ymm0   \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	//"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .SDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SROWSTORED:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vfmadd231ps      (%%rcx), %%ymm3,  %%ymm4   \n\t"
	"vmovups          %%ymm4,  (%%rcx)           \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vfmadd231ps      (%%rdx), %%ymm3,  %%ymm5   \n\t"
	"vmovups          %%ymm5,  (%%rdx)           \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"vfmadd231ps      (%%r12), %%ymm3,  %%ymm6   \n\t"
	"vmovups          %%ymm6,  (%%r12)           \n\t"
	"addq      %%rdi, %%r12                      \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vfmadd231ps      (%%rcx), %%ymm3,  %%ymm7   \n\t"
	"vmovups          %%ymm7,  (%%rcx)           \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vfmadd231ps      (%%rdx), %%ymm3,  %%ymm8   \n\t"
	"vmovups          %%ymm8,  (%%rdx)           \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"vfmadd231ps      (%%r12), %%ymm3,  %%ymm9   \n\t"
	"vmovups          %%ymm9,  (%%r12)           \n\t"
	"addq      %%rdi, %%r12                      \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vfmadd231ps      (%%rcx), %%ymm3,  %%ymm10  \n\t"
	"vmovups          %%ymm10, (%%rcx)           \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vfmadd231ps      (%%rdx), %%ymm3,  %%ymm11  \n\t"
	"vmovups          %%ymm11, (%%rdx)           \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"vfmadd231ps      (%%r12), %%ymm3,  %%ymm12  \n\t"
	"vmovups          %%ymm12, (%%r12)           \n\t"
	"addq      %%rdi, %%r12                      \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vfmadd231ps      (%%rcx), %%ymm3,  %%ymm13  \n\t"
	"vmovups          %%ymm13, (%%rcx)           \n\t"
	//"addq      %%rdi, %%rcx                      \n\t"
	"vfmadd231ps      (%%rdx), %%ymm3,  %%ymm14  \n\t"
	"vmovups          %%ymm14, (%%rdx)           \n\t"
	//"addq      %%rdi, %%rdx                      \n\t"
	"vfmadd231ps      (%%r12), %%ymm3,  %%ymm15  \n\t"
	"vmovups          %%ymm15, (%%r12)           \n\t"
	//"addq      %%rdi, %%r12                      \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .SDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SBETAZERO:                                 \n\t"
	"                                            \n\t"
	"cmpq       $4, %%rsi                        \n\t" // set ZF if (4*cs_c) == 4.
	"jz      .SROWSTORBZ                         \n\t" // jump to row storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SGENSTORBZ:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm4,  %%ymm0           \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm7,  %%ymm0           \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm10, %%ymm0           \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm13, %%ymm0           \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	//"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"movq      %%rdx, %%rcx                      \n\t" // rcx = c + 8*cs_c
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm5,  %%ymm0           \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm8,  %%ymm0           \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm11, %%ymm0           \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm14, %%ymm0           \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	//"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"movq      %%r12, %%rcx                      \n\t" // rcx = c + 16*cs_c
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm6,  %%ymm0           \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm9,  %%ymm0           \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm12, %%ymm0           \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm15, %%ymm0           \n\t"
	SGEMM_OUTPUT_GS_BETA_NZ
	//"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .SDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SROWSTORBZ:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovups          %%ymm4,  (%%rcx)           \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vmovups          %%ymm5,  (%%rdx)           \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"vmovups          %%ymm6,  (%%r12)           \n\t"
	"addq      %%rdi, %%r12                      \n\t"
	"                                            \n\t"
	"vmovups          %%ymm7,  (%%rcx)           \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vmovups          %%ymm8,  (%%rdx)           \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"vmovups          %%ymm9,  (%%r12)           \n\t"
	"addq      %%rdi, %%r12                      \n\t"
	"                                            \n\t"
	"vmovups          %%ymm10, (%%rcx)           \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vmovups          %%ymm11, (%%rdx)           \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"vmovups          %%ymm12, (%%r12)           \n\t"
	"addq      %%rdi, %%r12                      \n\t"
	"                                            \n\t"
	"vmovups          %%ymm13, (%%rcx)           \n\t"
	//"addq      %%rdi, %%rcx                      \n\t"
	"vmovups          %%ymm14, (%%rdx)           \n\t"
	//"addq      %%rdi, %%rdx                      \n\t"
	"vmovups          %%ymm15, (%%r12)           \n\t"
	//"addq      %%rdi, %%r12                      \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SDONE:                                     \n\t"
	"                                            \n\t"
	"vzeroupper                                  \n\t"
	"                                            \n\t"

	: // output operands (none)
	: // input operands
	  "m" (k_iter), // 0
	  "m" (k_left), // 1
	  "m" (a),      // 2
	  "m" (b),      // 3
	  "m" (alpha),  // 4
	  "m" (beta),   // 5
	  "m" (c),      // 6
	  "m" (rs_c),   // 7
	  "m" (cs_c)/*,   // 8
	  "m" (b_next), // 9
	  "m" (a_next)*/  // 10
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", 
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	);
}

#define DGEMM_INPUT_GS_BETA_NZ \
	"vmovlpd    (%%rcx        ),  %%xmm0,  %%xmm0  \n\t" \
	"vmovhpd    (%%rcx,%%rsi,1),  %%xmm0,  %%xmm0  \n\t" \
	"vmovlpd    (%%rcx,%%rsi,2),  %%xmm1,  %%xmm1  \n\t" \
	"vmovhpd    (%%rcx,%%r13  ),  %%xmm1,  %%xmm1  \n\t" \
	"vperm2f128 $0x20,   %%ymm1,  %%ymm0,  %%ymm0  \n\t" /*\
	"vmovlps    (%%rcx,%%rsi,4),  %%xmm2,  %%xmm2  \n\t" \
	"vmovhps    (%%rcx,%%r15  ),  %%xmm2,  %%xmm2  \n\t" \
	"vmovlps    (%%rcx,%%r13,2),  %%xmm1,  %%xmm1  \n\t" \
	"vmovhps    (%%rcx,%%r10  ),  %%xmm1,  %%xmm1  \n\t" \
	"vperm2f128 $0x20,   %%ymm1,  %%ymm2,  %%ymm2  \n\t"*/

#define DGEMM_OUTPUT_GS_BETA_NZ \
	"vextractf128  $1, %%ymm0,  %%xmm1           \n\t" \
	"vmovlpd           %%xmm0,  (%%rcx        )  \n\t" \
	"vmovhpd           %%xmm0,  (%%rcx,%%rsi  )  \n\t" \
	"vmovlpd           %%xmm1,  (%%rcx,%%rsi,2)  \n\t" \
	"vmovhpd           %%xmm1,  (%%rcx,%%r13  )  \n\t" /*\
	"vextractf128  $1, %%ymm2,  %%xmm1           \n\t" \
	"vmovlpd           %%xmm2,  (%%rcx,%%rsi,4)  \n\t" \
	"vmovhpd           %%xmm2,  (%%rcx,%%r15  )  \n\t" \
	"vmovlpd           %%xmm1,  (%%rcx,%%r13,2)  \n\t" \
	"vmovhpd           %%xmm1,  (%%rcx,%%r10  )  \n\t"*/

void bli_dgemm_haswell_asm_4x12
     (
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a,
       double*    restrict b,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	//void*   a_next = bli_auxinfo_next_a( data );
	//void*   b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	__asm__ volatile
	(
	"                                            \n\t"
	"vzeroall                                    \n\t" // zero all xmm/ymm registers.
	"                                            \n\t"
	"                                            \n\t"
	"movq                %2, %%rax               \n\t" // load address of a.
	"movq                %3, %%rbx               \n\t" // load address of b.
	//"movq                %9, %%r15               \n\t" // load address of b_next.
	"                                            \n\t"
	"                                            \n\t" // initialize loop by pre-loading
	"vmovapd            0 * 32(%%rbx), %%ymm0    \n\t"
	"vmovapd            1 * 32(%%rbx), %%ymm1    \n\t"
	"vmovapd            2 * 32(%%rbx), %%ymm2    \n\t"
	"                                            \n\t"
	"movq                %6, %%rcx               \n\t" // load address of c
	"movq                %7, %%rdi               \n\t" // load rs_c
	"leaq        (,%%rdi,8), %%rdi               \n\t" // rs_c *= sizeof(double)
	"                                            \n\t"
	"leaq   (%%rdi,%%rdi,2), %%r13               \n\t" // r13 = 3*rs_c;
	"prefetcht0   7 * 8(%%rcx)                   \n\t" // prefetch c + 0*rs_c
	"prefetcht0   7 * 8(%%rcx,%%rdi)             \n\t" // prefetch c + 1*rs_c
	"prefetcht0   7 * 8(%%rcx,%%rdi,2)           \n\t" // prefetch c + 2*rs_c
	"prefetcht0   7 * 8(%%rcx,%%r13)             \n\t" // prefetch c + 3*rs_c
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq      %0, %%rsi                         \n\t" // i = k_iter;
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
	"je     .DCONSIDKLEFT                        \n\t" // if i == 0, jump to code that
	"                                            \n\t" // contains the k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".DLOOPKITER:                                \n\t" // MAIN LOOP
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 0
	"prefetcht0  16 * 32(%%rax)                  \n\t"
	"                                            \n\t"
	"vbroadcastsd       0 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm4    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm5    \n\t"
	"vfmadd231pd       %%ymm2, %%ymm3, %%ymm6    \n\t"
	"                                            \n\t"
	"vbroadcastsd       1 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm7    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm8    \n\t"
	"vfmadd231pd       %%ymm2, %%ymm3, %%ymm9    \n\t"
	"                                            \n\t"
	"vbroadcastsd       2 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
	"vfmadd231pd       %%ymm2, %%ymm3, %%ymm12   \n\t"
	"                                            \n\t"
	"vbroadcastsd       3 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm13   \n\t"
	"vmovapd            3 * 32(%%rbx), %%ymm0    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm14   \n\t"
	"vmovapd            4 * 32(%%rbx), %%ymm1    \n\t"
	"vfmadd231pd       %%ymm2, %%ymm3, %%ymm15   \n\t"
	"vmovapd            5 * 32(%%rbx), %%ymm2    \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 1
	"vbroadcastsd       4 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm4    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm5    \n\t"
	"vfmadd231pd       %%ymm2, %%ymm3, %%ymm6    \n\t"
	"                                            \n\t"
	"vbroadcastsd       5 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm7    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm8    \n\t"
	"vfmadd231pd       %%ymm2, %%ymm3, %%ymm9    \n\t"
	"                                            \n\t"
	"vbroadcastsd       6 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
	"vfmadd231pd       %%ymm2, %%ymm3, %%ymm12   \n\t"
	"                                            \n\t"
	"vbroadcastsd       7 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm13   \n\t"
	"vmovapd            6 * 32(%%rbx), %%ymm0    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm14   \n\t"
	"vmovapd            7 * 32(%%rbx), %%ymm1    \n\t"
	"vfmadd231pd       %%ymm2, %%ymm3, %%ymm15   \n\t"
	"vmovapd            8 * 32(%%rbx), %%ymm2    \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 2
	"prefetcht0  22 * 32(%%rax)                  \n\t"
	"                                            \n\t"
	"vbroadcastsd       8 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm4    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm5    \n\t"
	"vfmadd231pd       %%ymm2, %%ymm3, %%ymm6    \n\t"
	"                                            \n\t"
	"vbroadcastsd       9 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm7    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm8    \n\t"
	"vfmadd231pd       %%ymm2, %%ymm3, %%ymm9    \n\t"
	"                                            \n\t"
	"vbroadcastsd      10 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
	"vfmadd231pd       %%ymm2, %%ymm3, %%ymm12   \n\t"
	"                                            \n\t"
	"vbroadcastsd      11 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm13   \n\t"
	"vmovapd            9 * 32(%%rbx), %%ymm0    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm14   \n\t"
	"vmovapd           10 * 32(%%rbx), %%ymm1    \n\t"
	"vfmadd231pd       %%ymm2, %%ymm3, %%ymm15   \n\t"
	"vmovapd           11 * 32(%%rbx), %%ymm2    \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 3
	"vbroadcastsd      12 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm4    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm5    \n\t"
	"vfmadd231pd       %%ymm2, %%ymm3, %%ymm6    \n\t"
	"                                            \n\t"
	"vbroadcastsd      13 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm7    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm8    \n\t"
	"vfmadd231pd       %%ymm2, %%ymm3, %%ymm9    \n\t"
	"                                            \n\t"
	"vbroadcastsd      14 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
	"vfmadd231pd       %%ymm2, %%ymm3, %%ymm12   \n\t"
	"                                            \n\t"
	"vbroadcastsd      15 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm13   \n\t"
	"vmovapd           12 * 32(%%rbx), %%ymm0    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm14   \n\t"
	"vmovapd           13 * 32(%%rbx), %%ymm1    \n\t"
	"vfmadd231pd       %%ymm2, %%ymm3, %%ymm15   \n\t"
	"vmovapd           14 * 32(%%rbx), %%ymm2    \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"addq          $4 *  4 * 8, %%rax            \n\t" // a += 4*4  (unroll x mr)
	"addq          $4 * 12 * 8, %%rbx            \n\t" // b += 4*12 (unroll x nr)
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .DLOOPKITER                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DCONSIDKLEFT:                              \n\t"
	"                                            \n\t"
	"movq      %1, %%rsi                         \n\t" // i = k_left;
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
	"je     .DPOSTACCUM                          \n\t" // if i == 0, we're done; jump to end.
	"                                            \n\t" // else, we prepare to enter k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".DLOOPKLEFT:                                \n\t" // EDGE LOOP
	"                                            \n\t"
	"prefetcht0  16 * 32(%%rax)                  \n\t"
	"                                            \n\t"
	"vbroadcastsd       0 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm4    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm5    \n\t"
	"vfmadd231pd       %%ymm2, %%ymm3, %%ymm6    \n\t"
	"                                            \n\t"
	"vbroadcastsd       1 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm7    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm8    \n\t"
	"vfmadd231pd       %%ymm2, %%ymm3, %%ymm9    \n\t"
	"                                            \n\t"
	"vbroadcastsd       2 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
	"vfmadd231pd       %%ymm2, %%ymm3, %%ymm12   \n\t"
	"                                            \n\t"
	"vbroadcastsd       3 *  8(%%rax), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm13   \n\t"
	"vmovapd            3 * 32(%%rbx), %%ymm0    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm14   \n\t"
	"vmovapd            4 * 32(%%rbx), %%ymm1    \n\t"
	"vfmadd231pd       %%ymm2, %%ymm3, %%ymm15   \n\t"
	"vmovapd            5 * 32(%%rbx), %%ymm2    \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"addq          $1 *  4 * 8, %%rax            \n\t" // a += 1*4  (unroll x mr)
	"addq          $1 * 12 * 8, %%rbx            \n\t" // b += 1*12 (unroll x nr)
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .DLOOPKLEFT                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DPOSTACCUM:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq         %4, %%rax                      \n\t" // load address of alpha
	"movq         %5, %%rbx                      \n\t" // load address of beta 
	"vbroadcastsd    (%%rax), %%ymm0             \n\t" // load alpha and duplicate
	"vbroadcastsd    (%%rbx), %%ymm3             \n\t" // load beta and duplicate
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm4   \n\t" // scale by alpha
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm5   \n\t"
	"vmulpd           %%ymm0,  %%ymm6,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm7,  %%ymm7   \n\t"
	"vmulpd           %%ymm0,  %%ymm8,  %%ymm8   \n\t"
	"vmulpd           %%ymm0,  %%ymm9,  %%ymm9   \n\t"
	"vmulpd           %%ymm0,  %%ymm10, %%ymm10  \n\t"
	"vmulpd           %%ymm0,  %%ymm11, %%ymm11  \n\t"
	"vmulpd           %%ymm0,  %%ymm12, %%ymm12  \n\t"
	"vmulpd           %%ymm0,  %%ymm13, %%ymm13  \n\t"
	"vmulpd           %%ymm0,  %%ymm14, %%ymm14  \n\t"
	"vmulpd           %%ymm0,  %%ymm15, %%ymm15  \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq                %8, %%rsi               \n\t" // load cs_c
	"leaq        (,%%rsi,8), %%rsi               \n\t" // rsi = cs_c * sizeof(double)
	"                                            \n\t"
	"leaq   (%%rcx,%%rsi,4), %%rdx               \n\t" // rdx = c +  4*cs_c;
	"leaq   (%%rcx,%%rsi,8), %%r12               \n\t" // r12 = c +  8*cs_c;
	"                                            \n\t"
	"leaq   (%%rsi,%%rsi,2), %%r13               \n\t" // r13 = 3*cs_c;
	//"leaq   (%%rsi,%%rsi,4), %%r15               \n\t" // r15 = 5*cs_c;
	//"leaq   (%%r13,%%rsi,4), %%r10               \n\t" // r10 = 7*cs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // now avoid loading C if beta == 0
	"                                            \n\t"
	"vxorpd    %%ymm0,  %%ymm0,  %%ymm0          \n\t" // set ymm0 to zero.
	"vucomisd  %%xmm0,  %%xmm3                   \n\t" // set ZF if beta == 0.
	"je      .DBETAZERO                          \n\t" // if ZF = 1, jump to beta == 0 case
	"                                            \n\t"
	"                                            \n\t"
	"cmpq       $8, %%rsi                        \n\t" // set ZF if (8*cs_c) == 8.
	"jz      .DROWSTORED                         \n\t" // jump to row storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DGENSTORED:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	DGEMM_INPUT_GS_BETA_NZ
	"vfmadd213pd      %%ymm4,  %%ymm3,  %%ymm0   \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	DGEMM_INPUT_GS_BETA_NZ
	"vfmadd213pd      %%ymm7,  %%ymm3,  %%ymm0   \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	DGEMM_INPUT_GS_BETA_NZ
	"vfmadd213pd      %%ymm10, %%ymm3,  %%ymm0   \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	DGEMM_INPUT_GS_BETA_NZ
	"vfmadd213pd      %%ymm13, %%ymm3,  %%ymm0   \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	//"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"movq      %%rdx, %%rcx                      \n\t" // rcx = c + 4*cs_c
	"                                            \n\t"
	"                                            \n\t"
	DGEMM_INPUT_GS_BETA_NZ
	"vfmadd213pd      %%ymm5,  %%ymm3,  %%ymm0   \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	DGEMM_INPUT_GS_BETA_NZ
	"vfmadd213pd      %%ymm8,  %%ymm3,  %%ymm0   \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	DGEMM_INPUT_GS_BETA_NZ
	"vfmadd213pd      %%ymm11, %%ymm3,  %%ymm0   \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	DGEMM_INPUT_GS_BETA_NZ
	"vfmadd213pd      %%ymm14, %%ymm3,  %%ymm0   \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	//"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"movq      %%r12, %%rcx                      \n\t" // rcx = c + 8*cs_c
	"                                            \n\t"
	"                                            \n\t"
	DGEMM_INPUT_GS_BETA_NZ
	"vfmadd213pd      %%ymm6,  %%ymm3,  %%ymm0   \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	DGEMM_INPUT_GS_BETA_NZ
	"vfmadd213pd      %%ymm9,  %%ymm3,  %%ymm0   \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	DGEMM_INPUT_GS_BETA_NZ
	"vfmadd213pd      %%ymm12, %%ymm3,  %%ymm0   \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	DGEMM_INPUT_GS_BETA_NZ
	"vfmadd213pd      %%ymm15, %%ymm3,  %%ymm0   \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	//"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .DDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DROWSTORED:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vfmadd231pd      (%%rcx), %%ymm3,  %%ymm4   \n\t"
	"vmovupd          %%ymm4,  (%%rcx)           \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vfmadd231pd      (%%rdx), %%ymm3,  %%ymm5   \n\t"
	"vmovupd          %%ymm5,  (%%rdx)           \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"vfmadd231pd      (%%r12), %%ymm3,  %%ymm6   \n\t"
	"vmovupd          %%ymm6,  (%%r12)           \n\t"
	"addq      %%rdi, %%r12                      \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vfmadd231pd      (%%rcx), %%ymm3,  %%ymm7   \n\t"
	"vmovupd          %%ymm7,  (%%rcx)           \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vfmadd231pd      (%%rdx), %%ymm3,  %%ymm8   \n\t"
	"vmovupd          %%ymm8,  (%%rdx)           \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"vfmadd231pd      (%%r12), %%ymm3,  %%ymm9   \n\t"
	"vmovupd          %%ymm9,  (%%r12)           \n\t"
	"addq      %%rdi, %%r12                      \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vfmadd231pd      (%%rcx), %%ymm3,  %%ymm10  \n\t"
	"vmovupd          %%ymm10, (%%rcx)           \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vfmadd231pd      (%%rdx), %%ymm3,  %%ymm11  \n\t"
	"vmovupd          %%ymm11, (%%rdx)           \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"vfmadd231pd      (%%r12), %%ymm3,  %%ymm12  \n\t"
	"vmovupd          %%ymm12, (%%r12)           \n\t"
	"addq      %%rdi, %%r12                      \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vfmadd231pd      (%%rcx), %%ymm3,  %%ymm13  \n\t"
	"vmovupd          %%ymm13, (%%rcx)           \n\t"
	//"addq      %%rdi, %%rcx                      \n\t"
	"vfmadd231pd      (%%rdx), %%ymm3,  %%ymm14  \n\t"
	"vmovupd          %%ymm14, (%%rdx)           \n\t"
	//"addq      %%rdi, %%rdx                      \n\t"
	"vfmadd231pd      (%%r12), %%ymm3,  %%ymm15  \n\t"
	"vmovupd          %%ymm15, (%%r12)           \n\t"
	//"addq      %%rdi, %%r12                      \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .DDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DBETAZERO:                                 \n\t"
	"                                            \n\t" // check if aligned/column-stored
	"cmpq       $8, %%rsi                        \n\t" // set ZF if (8*cs_c) == 8.
	"jz      .DROWSTORBZ                         \n\t" // jump to row storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DGENSTORBZ:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm4,  %%ymm0           \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm7,  %%ymm0           \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm10, %%ymm0           \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm13, %%ymm0           \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	//"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"movq      %%rdx, %%rcx                      \n\t" // rcx = c + 4*cs_c
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm5,  %%ymm0           \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm8,  %%ymm0           \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm11, %%ymm0           \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm14, %%ymm0           \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	//"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"movq      %%r12, %%rcx                      \n\t" // rcx = c + 8*cs_c
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm6,  %%ymm0           \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm9,  %%ymm0           \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm12, %%ymm0           \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm15, %%ymm0           \n\t"
	DGEMM_OUTPUT_GS_BETA_NZ
	//"addq      %%rdi, %%rcx                      \n\t" // c += rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .DDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DROWSTORBZ:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovupd          %%ymm4,  (%%rcx)           \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vmovupd          %%ymm5,  (%%rdx)           \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"vmovupd          %%ymm6,  (%%r12)           \n\t"
	"addq      %%rdi, %%r12                      \n\t"
	"                                            \n\t"
	"vmovupd          %%ymm7,  (%%rcx)           \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vmovupd          %%ymm8,  (%%rdx)           \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"vmovupd          %%ymm9,  (%%r12)           \n\t"
	"addq      %%rdi, %%r12                      \n\t"
	"                                            \n\t"
	"vmovupd          %%ymm10, (%%rcx)           \n\t"
	"addq      %%rdi, %%rcx                      \n\t"
	"vmovupd          %%ymm11, (%%rdx)           \n\t"
	"addq      %%rdi, %%rdx                      \n\t"
	"vmovupd          %%ymm12, (%%r12)           \n\t"
	"addq      %%rdi, %%r12                      \n\t"
	"                                            \n\t"
	"vmovupd          %%ymm13, (%%rcx)           \n\t"
	//"addq      %%rdi, %%rcx                      \n\t"
	"vmovupd          %%ymm14, (%%rdx)           \n\t"
	//"addq      %%rdi, %%rdx                      \n\t"
	"vmovupd          %%ymm15, (%%r12)           \n\t"
	//"addq      %%rdi, %%r12                      \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DDONE:                                     \n\t"
	"                                            \n\t"
	"vzeroupper                                  \n\t"
	"                                            \n\t"

	: // output operands (none)
	: // input operands
	  "m" (k_iter), // 0
	  "m" (k_left), // 1
	  "m" (a),      // 2
	  "m" (b),      // 3
	  "m" (alpha),  // 4
	  "m" (beta),   // 5
	  "m" (c),      // 6
	  "m" (rs_c),   // 7
	  "m" (cs_c)/*,   // 8
	  "m" (b_next), // 9
	  "m" (a_next)*/  // 10
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", 
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	);
}

#if 0

void bli_cgemm_haswell_asm_
     (
       dim_t               k0,
       scomplex*  restrict alpha,
       scomplex*  restrict a,
       scomplex*  restrict b,
       scomplex*  restrict beta,
       scomplex*  restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	//void*   a_next = bli_auxinfo_next_a( data );
	//void*   b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

}



void bli_zgemm_haswell_asm_
     (
       dim_t               k0,
       dcomplex*  restrict alpha,
       dcomplex*  restrict a,
       dcomplex*  restrict b,
       dcomplex*  restrict beta,
       dcomplex*  restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	//void*   a_next = bli_auxinfo_next_a( data );
	//void*   b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

}

#endif
