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

void bli_sgemm_opt_d4x2(
                         dim_t           k,
                         float* restrict alpha,
                         float* restrict a,
                         float* restrict b,
                         float* restrict beta,
                         float* restrict c, inc_t rs_c, inc_t cs_c,
                         float* restrict a_next,
                         float* restrict b_next
                       )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

void bli_dgemm_opt_d4x2(
                         dim_t            k,
                         double* restrict alpha,
                         double* restrict a,
                         double* restrict b,
                         double* restrict beta,
                         double* restrict c, inc_t rs_c, inc_t cs_c,
                         double* restrict a_next,
                         double* restrict b_next
                       )
{
	dim_t   k_iter;
	dim_t   k_left;

	k_iter  = k / 8;
	k_left  = k % 8;

	__asm__ volatile
	(
		"                                \n\t"
		"movl         %6, %%ecx          \n\t" // load address of c
		"                                \n\t"
		"movl         %8, %%edi          \n\t" // load cs_c
		"sall         $3, %%edi          \n\t" // cs_c *= sizeof(double)
		"                                \n\t"
		"prefetcht0      (%%ecx)         \n\t" // give a T0 prefetch hint for c00.
		"prefetcht0      (%%ecx,%%edi)   \n\t" // give a T0 prefetch hint for c01.
		"                                \n\t"
		"movl         %2, %%eax          \n\t" // load address of a.
		"movl         %3, %%ebx          \n\t" // load address of b.
		"                                \n\t"
		"addl    $8 * 16, %%eax          \n\t" // increment pointers to allow byte
		"addl    $8 * 16, %%ebx          \n\t" // offsets in the unrolled iterations.
		"                                \n\t"
		"movapd  -8 * 16(%%eax), %%xmm0  \n\t" // initialize loop by pre-loading elements
		"movapd  -4 * 16(%%eax), %%xmm3  \n\t" // of a.
		"                                \n\t"
		"pxor     %%xmm4, %%xmm4         \n\t"
		"pxor     %%xmm5, %%xmm5         \n\t"
		"pxor     %%xmm6, %%xmm6         \n\t"
		"pxor     %%xmm7, %%xmm7         \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movl      %0, %%esi             \n\t" // i = k_iter;
		"testl  %%esi, %%esi             \n\t" // check i via logical AND.
		"je     .CONSIDERKLEFT           \n\t" // if i == 0, jump to code that
		"                                \n\t" // contains the k_left loop.
		"                                \n\t"
		"                                \n\t"
		".LOOPKITER:                     \n\t" // MAIN LOOP
		"                                \n\t"
		"movapd  -8 * 16(%%ebx), %%xmm1  \n\t" // iteration 0
		"movapd   %%xmm1, %%xmm2         \n\t"
		"mulpd    %%xmm0, %%xmm1         \n\t"
		"addpd    %%xmm1, %%xmm4         \n\t"
		"movapd  -7 * 16(%%ebx), %%xmm1  \n\t"
		"mulpd    %%xmm1, %%xmm0         \n\t"
		"addpd    %%xmm0, %%xmm5         \n\t"
		"movapd  -7 * 16(%%eax), %%xmm0  \n\t"
		"mulpd    %%xmm0, %%xmm2         \n\t"
		"mulpd    %%xmm0, %%xmm1         \n\t"
		"movapd  -6 * 16(%%eax), %%xmm0  \n\t"
		"addpd    %%xmm2, %%xmm6         \n\t"
		"addpd    %%xmm1, %%xmm7         \n\t"
		"                                \n\t"
		"movapd  -6 * 16(%%ebx), %%xmm1  \n\t" // iteration 1
		"movapd   %%xmm1, %%xmm2         \n\t"
		"mulpd    %%xmm0, %%xmm1         \n\t"
		"addpd    %%xmm1, %%xmm4         \n\t"
		"movapd  -5 * 16(%%ebx), %%xmm1  \n\t"
		"mulpd    %%xmm1, %%xmm0         \n\t"
		"addpd    %%xmm0, %%xmm5         \n\t"
		"movapd  -5 * 16(%%eax), %%xmm0  \n\t"
		"mulpd    %%xmm0, %%xmm2         \n\t"
		"mulpd    %%xmm0, %%xmm1         \n\t"
		"movapd   0 * 16(%%eax), %%xmm0  \n\t"
		"addpd    %%xmm2, %%xmm6         \n\t"
		"addpd    %%xmm1, %%xmm7         \n\t"
		"                                \n\t"
		"movapd  -4 * 16(%%ebx), %%xmm1  \n\t" // iteration 2
		"movapd   %%xmm1, %%xmm2         \n\t"
		"mulpd    %%xmm3, %%xmm1         \n\t"
		"addpd    %%xmm1, %%xmm4         \n\t"
		"movapd  -3 * 16(%%ebx), %%xmm1  \n\t"
		"mulpd    %%xmm1, %%xmm3         \n\t"
		"addpd    %%xmm3, %%xmm5         \n\t"
		"movapd  -3 * 16(%%eax), %%xmm3  \n\t"
		"mulpd    %%xmm3, %%xmm2         \n\t"
		"mulpd    %%xmm3, %%xmm1         \n\t"
		"movapd  -2 * 16(%%eax), %%xmm3  \n\t"
		"addpd    %%xmm2, %%xmm6         \n\t"
		"addpd    %%xmm1, %%xmm7         \n\t"
		"                                \n\t"
		"movapd  -2 * 16(%%ebx), %%xmm1  \n\t" // iteration 3
		"movapd   %%xmm1, %%xmm2         \n\t"
		"mulpd    %%xmm3, %%xmm1         \n\t"
		"addpd    %%xmm1, %%xmm4         \n\t"
		"movapd  -1 * 16(%%ebx), %%xmm1  \n\t"
		"mulpd    %%xmm1, %%xmm3         \n\t"
		"addpd    %%xmm3, %%xmm5         \n\t"
		"movapd  -1 * 16(%%eax), %%xmm3  \n\t"
		"mulpd    %%xmm3, %%xmm2         \n\t"
		"mulpd    %%xmm3, %%xmm1         \n\t"
		"movapd   4 * 16(%%eax), %%xmm3  \n\t"
		"addpd    %%xmm2, %%xmm6         \n\t"
		"addpd    %%xmm1, %%xmm7         \n\t"
		"                                \n\t"
		"movapd   0 * 16(%%ebx), %%xmm1  \n\t" // iteration 4
		"movapd   %%xmm1, %%xmm2         \n\t"
		"mulpd    %%xmm0, %%xmm1         \n\t"
		"addpd    %%xmm1, %%xmm4         \n\t"
		"movapd   1 * 16(%%ebx), %%xmm1  \n\t"
		"mulpd    %%xmm1, %%xmm0         \n\t"
		"addpd    %%xmm0, %%xmm5         \n\t"
		"movapd   1 * 16(%%eax), %%xmm0  \n\t"
		"mulpd    %%xmm0, %%xmm2         \n\t"
		"mulpd    %%xmm0, %%xmm1         \n\t"
		"movapd   2 * 16(%%eax), %%xmm0  \n\t"
		"addpd    %%xmm2, %%xmm6         \n\t"
		"addpd    %%xmm1, %%xmm7         \n\t"
		"                                \n\t"
		"movapd   2 * 16(%%ebx), %%xmm1  \n\t" // iteration 5
		"movapd   %%xmm1, %%xmm2         \n\t"
		"mulpd    %%xmm0, %%xmm1         \n\t"
		"addpd    %%xmm1, %%xmm4         \n\t"
		"movapd   3 * 16(%%ebx), %%xmm1  \n\t"
		"mulpd    %%xmm1, %%xmm0         \n\t"
		"addpd    %%xmm0, %%xmm5         \n\t"
		"movapd   3 * 16(%%eax), %%xmm0  \n\t"
		"mulpd    %%xmm0, %%xmm2         \n\t"
		"mulpd    %%xmm0, %%xmm1         \n\t"
		"movapd   8 * 16(%%eax), %%xmm0  \n\t"
		"addpd    %%xmm2, %%xmm6         \n\t"
		"addpd    %%xmm1, %%xmm7         \n\t"
		"                                \n\t"
		"movapd   4 * 16(%%ebx), %%xmm1  \n\t" // iteration 6
		"movapd   %%xmm1, %%xmm2         \n\t"
		"mulpd    %%xmm3, %%xmm1         \n\t"
		"addpd    %%xmm1, %%xmm4         \n\t"
		"movapd   5 * 16(%%ebx), %%xmm1  \n\t"
		"mulpd    %%xmm1, %%xmm3         \n\t"
		"addpd    %%xmm3, %%xmm5         \n\t"
		"movapd   5 * 16(%%eax), %%xmm3  \n\t"
		"mulpd    %%xmm3, %%xmm2         \n\t"
		"mulpd    %%xmm3, %%xmm1         \n\t"
		"movapd   6 * 16(%%eax), %%xmm3  \n\t"
		"addl   $8 * 4 * 8, %%eax        \n\t" // a += 8*4   (unroll x mr)
		"addpd    %%xmm2, %%xmm6         \n\t"
		"addpd    %%xmm1, %%xmm7         \n\t"
		"                                \n\t"
		"movapd   6 * 16(%%ebx), %%xmm1  \n\t" // iteration 7
		"movapd   %%xmm1, %%xmm2         \n\t"
		"mulpd    %%xmm3, %%xmm1         \n\t"
		"addpd    %%xmm1, %%xmm4         \n\t"
		"movapd   7 * 16(%%ebx), %%xmm1  \n\t"
		"addl   $8 * 2 * 2 * 8, %%ebx    \n\t" // b += 8*2*2 (unroll x nr x ndup)
		"mulpd    %%xmm1, %%xmm3         \n\t"
		"addpd    %%xmm3, %%xmm5         \n\t"
		"movapd  -9 * 16(%%eax), %%xmm3  \n\t"
		"mulpd    %%xmm3, %%xmm2         \n\t"
		"mulpd    %%xmm3, %%xmm1         \n\t"
		"decl   %%esi                    \n\t" // i -= 1;
		"movapd  -4 * 16(%%eax), %%xmm3  \n\t"
		"addpd    %%xmm2, %%xmm6         \n\t"
		"addpd    %%xmm1, %%xmm7         \n\t"
		"                                \n\t"
		"jne    .LOOPKITER               \n\t" // iterate again if i != 0.
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".CONSIDERKLEFT:                 \n\t"
		"                                \n\t"
		"movl      %1, %%esi             \n\t" // i = k_left;
		"testl  %%esi, %%esi             \n\t" // check i via logical AND.
		"je     .POSTACCUM               \n\t" // if i == 0, we're done; jump to end.
		"                                \n\t" // else, we prepare to enter k_left loop.
		"                                \n\t"
		"                                \n\t"
		".LOOPKLEFT:                     \n\t" // EDGE LOOP
		"                                \n\t"
		"movapd  -8 * 16(%%ebx), %%xmm1  \n\t" // iteration i
		"movapd   %%xmm1, %%xmm2         \n\t"
		"mulpd    %%xmm0, %%xmm1         \n\t"
		"addpd    %%xmm1, %%xmm4         \n\t"
		"movapd  -7 * 16(%%ebx), %%xmm1  \n\t"
		"addl   $1 * 2 * 2 * 8, %%ebx    \n\t" // b += 2*2 (1 x nr x ndup)
		"mulpd    %%xmm1, %%xmm0         \n\t"
		"addpd    %%xmm0, %%xmm5         \n\t"
		"movapd  -7 * 16(%%eax), %%xmm0  \n\t"
		"mulpd    %%xmm0, %%xmm2         \n\t"
		"mulpd    %%xmm0, %%xmm1         \n\t"
		"movapd  -6 * 16(%%eax), %%xmm0  \n\t"
		"addl   $1 * 4 * 8, %%eax        \n\t" // a += 4 (1 x mr)
		"addpd    %%xmm2, %%xmm6         \n\t"
		"addpd    %%xmm1, %%xmm7         \n\t"
		"                                \n\t"
		"decl   %%esi                    \n\t" // i -= 1;
		"jne    .LOOPKLEFT               \n\t" // iterate again if i != 0.
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".POSTACCUM:                     \n\t"
		"                                \n\t"
		"movl    %4, %%eax               \n\t" // load address of alpha
		"movl    %5, %%ebx               \n\t" // load address of beta 
		"movddup (%%eax), %%xmm2         \n\t" // load alpha and duplicate
		"movddup (%%ebx), %%xmm3         \n\t" // load beta and duplicate
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movl    %7, %%esi               \n\t" // load rs_c
		"sall    $3, %%esi               \n\t" // rs_c *= sizeof(double)
		"                                \n\t"
		"leal   (%%ecx,%%esi,2), %%edx   \n\t" // load address of c + 2*rs_c;
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movlpd  (%%ecx),       %%xmm0   \n\t" // load c00 and c10,
		"movhpd  (%%ecx,%%esi), %%xmm0   \n\t"
		"mulpd   %%xmm2, %%xmm4          \n\t" // scale by alpha,
		"mulpd   %%xmm3, %%xmm0          \n\t" // scale by beta,
		"addpd   %%xmm4, %%xmm0          \n\t" // add the gemm result,
		"movlpd  %%xmm0, (%%ecx)         \n\t" // and store back to memory.
		"movhpd  %%xmm0, (%%ecx,%%esi)   \n\t"
		"addl     %%edi, %%ecx           \n\t"
		"                                \n\t"
		"movlpd  (%%edx),       %%xmm1   \n\t" // load c01 and c11,
		"movhpd  (%%edx,%%esi), %%xmm1   \n\t"
		"mulpd   %%xmm2, %%xmm6          \n\t" // scale by alpha,
		"mulpd   %%xmm3, %%xmm1          \n\t" // scale by beta,
		"addpd   %%xmm6, %%xmm1          \n\t" // add the gemm result,
		"movlpd  %%xmm1, (%%edx)         \n\t" // and store back to memory.
		"movhpd  %%xmm1, (%%edx,%%esi)   \n\t"
		"addl     %%edi, %%edx           \n\t"
		"                                \n\t"
		"movlpd  (%%ecx),       %%xmm0   \n\t" // load c20 and c30,
		"movhpd  (%%ecx,%%esi), %%xmm0   \n\t"
		"mulpd   %%xmm2, %%xmm5          \n\t" // scale by alpha,
		"mulpd   %%xmm3, %%xmm0          \n\t" // scale by beta,
		"addpd   %%xmm5, %%xmm0          \n\t" // add the gemm result,
		"movlpd  %%xmm0, (%%ecx)         \n\t" // and store back to memory.
		"movhpd  %%xmm0, (%%ecx,%%esi)   \n\t"
		"                                \n\t"
		"movlpd  (%%edx),       %%xmm1   \n\t" // load c21 and c31,
		"movhpd  (%%edx,%%esi), %%xmm1   \n\t"
		"mulpd   %%xmm2, %%xmm7          \n\t" // scale by alpha,
		"mulpd   %%xmm3, %%xmm1          \n\t" // scale by beta,
		"addpd   %%xmm7, %%xmm1          \n\t" // add the gemm result,
		"movlpd  %%xmm1, (%%edx)         \n\t" // and store back to memory.
		"movhpd  %%xmm1, (%%edx,%%esi)   \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"

		: // output operands (none)
		: // input operands
		  "m" (k_iter),
		  "m" (k_left),
		  "m" (a),
		  "m" (b),
		  "m" (alpha),
		  "m" (beta),
		  "m" (c),
		  "m" (rs_c),
		  "m" (cs_c)
		: // register clobber list
		  "eax", "ebx", "ecx", "edx", "esi", "edi",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7",
		  "memory"
	);

}

void bli_cgemm_opt_d4x2(
                         dim_t              k,
                         scomplex* restrict alpha,
                         scomplex* restrict a,
                         scomplex* restrict b,
                         scomplex* restrict beta,
                         scomplex* restrict c, inc_t rs_c, inc_t cs_c,
                         scomplex* restrict a_next,
                         scomplex* restrict b_next
                       )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

void bli_zgemm_opt_d4x2(
                         dim_t              k,
                         dcomplex* restrict alpha,
                         dcomplex* restrict a,
                         dcomplex* restrict b,
                         dcomplex* restrict beta,
                         dcomplex* restrict c, inc_t rs_c, inc_t cs_c,
                         dcomplex* restrict a_next,
                         dcomplex* restrict b_next
                       )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

