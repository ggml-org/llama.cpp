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

void bli_sgemmtrsm_l_opt_d4x2(
                               dim_t              k,
                               float* restrict    alpha,
                               float* restrict    a10,
                               float* restrict    a11,
                               float* restrict    bd01,
                               float* restrict    bd11,
                               float* restrict    b11,
                               float* restrict    c11, inc_t rs_c, inc_t cs_c,
                               float* restrict    a_next,
                               float* restrict    b_next
                             )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

void bli_dgemmtrsm_l_opt_d4x2(
                               dim_t              k,
                               double* restrict   alpha,
                               double* restrict   a10,
                               double* restrict   a11,
                               double* restrict   bd01,
                               double* restrict   bd11,
                               double* restrict   b11,
                               double* restrict   c11, inc_t rs_c, inc_t cs_c,
                               double* restrict   a_next,
                               double* restrict   b_next
                             )
{
	dim_t k_iter;
	dim_t k_left;

	k_iter = k / 8;
	k_left = k % 8;

	__asm__ volatile
	(
		"                                  \n\t"
		"movl         %2, %%eax            \n\t" // load address of a10.
		"movl         %4, %%ebx            \n\t" // load address of bd01.
		"                                  \n\t"
		"addl    $8 * 16, %%eax            \n\t" // increment pointers to allow byte
		"addl    $8 * 16, %%ebx            \n\t" // offsets in the unrolled iterations.
		"                                  \n\t"
		"movapd  -8 * 16(%%eax), %%xmm0    \n\t" // initialize loop by pre-loading elements
		"movapd  -4 * 16(%%eax), %%xmm3    \n\t" // and of a.
		"                                  \n\t"
		"pxor     %%xmm4, %%xmm4           \n\t"
		"pxor     %%xmm5, %%xmm5           \n\t"
		"pxor     %%xmm6, %%xmm6           \n\t"
		"pxor     %%xmm7, %%xmm7           \n\t"
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t"
		"movl      %0, %%esi               \n\t" // i = k_iter;
		"testl  %%esi, %%esi               \n\t" // check i via logical AND.
		"je     .CONSIDERKLEFT             \n\t" // if i == 0, jump to code that
		"                                  \n\t" // contains the k_left loop.
		"                                  \n\t"
		"                                  \n\t"
		".LOOPKITER:                       \n\t" // MAIN LOOP
		"                                  \n\t"
		"movapd  -8 * 16(%%ebx), %%xmm1    \n\t" // iteration 0
		"movapd   %%xmm1, %%xmm2           \n\t"
		"mulpd    %%xmm0, %%xmm1           \n\t"
		"addpd    %%xmm1, %%xmm4           \n\t"
		"movapd  -7 * 16(%%ebx), %%xmm1    \n\t"
		"mulpd    %%xmm1, %%xmm0           \n\t"
		"addpd    %%xmm0, %%xmm5           \n\t"
		"movapd  -7 * 16(%%eax), %%xmm0    \n\t"
		"mulpd    %%xmm0, %%xmm2           \n\t"
		"mulpd    %%xmm0, %%xmm1           \n\t"
		"movapd  -6 * 16(%%eax), %%xmm0    \n\t"
		"addpd    %%xmm2, %%xmm6           \n\t"
		"addpd    %%xmm1, %%xmm7           \n\t"
		"                                  \n\t"
		"movapd  -6 * 16(%%ebx), %%xmm1    \n\t" // iteration 1
		"movapd   %%xmm1, %%xmm2           \n\t"
		"mulpd    %%xmm0, %%xmm1           \n\t"
		"addpd    %%xmm1, %%xmm4           \n\t"
		"movapd  -5 * 16(%%ebx), %%xmm1    \n\t"
		"mulpd    %%xmm1, %%xmm0           \n\t"
		"addpd    %%xmm0, %%xmm5           \n\t"
		"movapd  -5 * 16(%%eax), %%xmm0    \n\t"
		"mulpd    %%xmm0, %%xmm2           \n\t"
		"mulpd    %%xmm0, %%xmm1           \n\t"
		"movapd   0 * 16(%%eax), %%xmm0    \n\t"
		"addpd    %%xmm2, %%xmm6           \n\t"
		"addpd    %%xmm1, %%xmm7           \n\t"
		"                                  \n\t"
		"movapd  -4 * 16(%%ebx), %%xmm1    \n\t" // iteration 2
		"movapd   %%xmm1, %%xmm2           \n\t"
		"mulpd    %%xmm3, %%xmm1           \n\t"
		"addpd    %%xmm1, %%xmm4           \n\t"
		"movapd  -3 * 16(%%ebx), %%xmm1    \n\t"
		"mulpd    %%xmm1, %%xmm3           \n\t"
		"addpd    %%xmm3, %%xmm5           \n\t"
		"movapd  -3 * 16(%%eax), %%xmm3    \n\t"
		"mulpd    %%xmm3, %%xmm2           \n\t"
		"mulpd    %%xmm3, %%xmm1           \n\t"
		"movapd  -2 * 16(%%eax), %%xmm3    \n\t"
		"addpd    %%xmm2, %%xmm6           \n\t"
		"addpd    %%xmm1, %%xmm7           \n\t"
		"                                  \n\t"
		"movapd  -2 * 16(%%ebx), %%xmm1    \n\t" // iteration 3
		"movapd   %%xmm1, %%xmm2           \n\t"
		"mulpd    %%xmm3, %%xmm1           \n\t"
		"addpd    %%xmm1, %%xmm4           \n\t"
		"movapd  -1 * 16(%%ebx), %%xmm1    \n\t"
		"mulpd    %%xmm1, %%xmm3           \n\t"
		"addpd    %%xmm3, %%xmm5           \n\t"
		"movapd  -1 * 16(%%eax), %%xmm3    \n\t"
		"mulpd    %%xmm3, %%xmm2           \n\t"
		"mulpd    %%xmm3, %%xmm1           \n\t"
		"movapd   4 * 16(%%eax), %%xmm3    \n\t"
		"addpd    %%xmm2, %%xmm6           \n\t"
		"addpd    %%xmm1, %%xmm7           \n\t"
		"                                  \n\t"
		"movapd   0 * 16(%%ebx), %%xmm1    \n\t" // iteration 4
		"movapd   %%xmm1, %%xmm2           \n\t"
		"mulpd    %%xmm0, %%xmm1           \n\t"
		"addpd    %%xmm1, %%xmm4           \n\t"
		"movapd   1 * 16(%%ebx), %%xmm1    \n\t"
		"mulpd    %%xmm1, %%xmm0           \n\t"
		"addpd    %%xmm0, %%xmm5           \n\t"
		"movapd   1 * 16(%%eax), %%xmm0    \n\t"
		"mulpd    %%xmm0, %%xmm2           \n\t"
		"mulpd    %%xmm0, %%xmm1           \n\t"
		"movapd   2 * 16(%%eax), %%xmm0    \n\t"
		"addpd    %%xmm2, %%xmm6           \n\t"
		"addpd    %%xmm1, %%xmm7           \n\t"
		"                                  \n\t"
		"movapd   2 * 16(%%ebx), %%xmm1    \n\t" // iteration 5
		"movapd   %%xmm1, %%xmm2           \n\t"
		"mulpd    %%xmm0, %%xmm1           \n\t"
		"addpd    %%xmm1, %%xmm4           \n\t"
		"movapd   3 * 16(%%ebx), %%xmm1    \n\t"
		"mulpd    %%xmm1, %%xmm0           \n\t"
		"addpd    %%xmm0, %%xmm5           \n\t"
		"movapd   3 * 16(%%eax), %%xmm0    \n\t"
		"mulpd    %%xmm0, %%xmm2           \n\t"
		"mulpd    %%xmm0, %%xmm1           \n\t"
		"movapd   8 * 16(%%eax), %%xmm0    \n\t"
		"addpd    %%xmm2, %%xmm6           \n\t"
		"addpd    %%xmm1, %%xmm7           \n\t"
		"                                  \n\t"
		"movapd   4 * 16(%%ebx), %%xmm1    \n\t" // iteration 6
		"movapd   %%xmm1, %%xmm2           \n\t"
		"mulpd    %%xmm3, %%xmm1           \n\t"
		"addpd    %%xmm1, %%xmm4           \n\t"
		"movapd   5 * 16(%%ebx), %%xmm1    \n\t"
		"mulpd    %%xmm1, %%xmm3           \n\t"
		"addpd    %%xmm3, %%xmm5           \n\t"
		"movapd   5 * 16(%%eax), %%xmm3    \n\t"
		"mulpd    %%xmm3, %%xmm2           \n\t"
		"mulpd    %%xmm3, %%xmm1           \n\t"
		"movapd   6 * 16(%%eax), %%xmm3    \n\t"
		"addl   $8 * 4 * 8, %%eax          \n\t" // a += 8*4   (unroll x mr)
		"addpd    %%xmm2, %%xmm6           \n\t"
		"addpd    %%xmm1, %%xmm7           \n\t"
		"                                  \n\t"
		"movapd   6 * 16(%%ebx), %%xmm1    \n\t" // iteration 7
		"movapd   %%xmm1, %%xmm2           \n\t"
		"mulpd    %%xmm3, %%xmm1           \n\t"
		"addpd    %%xmm1, %%xmm4           \n\t"
		"movapd   7 * 16(%%ebx), %%xmm1    \n\t"
		"addl   $8 * 2 * 2 * 8, %%ebx      \n\t" // b += 8*2*2 (unroll x nr x ndup)
		"mulpd    %%xmm1, %%xmm3           \n\t"
		"addpd    %%xmm3, %%xmm5           \n\t"
		"movapd  -9 * 16(%%eax), %%xmm3    \n\t"
		"mulpd    %%xmm3, %%xmm2           \n\t"
		"mulpd    %%xmm3, %%xmm1           \n\t"
		"decl   %%esi                      \n\t" // i -= 1;
		"movapd  -4 * 16(%%eax), %%xmm3    \n\t"
		"addpd    %%xmm2, %%xmm6           \n\t"
		"addpd    %%xmm1, %%xmm7           \n\t"
		"                                  \n\t"
		"jne    .LOOPKITER                 \n\t" // iterate again if i != 0.
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t"
		".CONSIDERKLEFT:                   \n\t"
		"                                  \n\t"
		"movl      %1, %%esi               \n\t" // i = k_left;
		"testl  %%esi, %%esi               \n\t" // check i via logical AND.
		"je     .POSTACCUM                 \n\t" // if i == 0, we're done; jump to end.
		"                                  \n\t" // else, we prepare to enter k_left loop.
		"                                  \n\t"
		"                                  \n\t"
		".LOOPKLEFT:                       \n\t" // EDGE LOOP
		"                                  \n\t"
		"movapd  -8 * 16(%%ebx), %%xmm1    \n\t" // iteration i
		"movapd   %%xmm1, %%xmm2           \n\t"
		"mulpd    %%xmm0, %%xmm1           \n\t"
		"addpd    %%xmm1, %%xmm4           \n\t"
		"movapd  -7 * 16(%%ebx), %%xmm1    \n\t"
		"addl   $1 * 2 * 2 * 8, %%ebx      \n\t" // b += 2*2 (1 x nr x ndup)
		"mulpd    %%xmm1, %%xmm0           \n\t"
		"addpd    %%xmm0, %%xmm5           \n\t"
		"movapd  -7 * 16(%%eax), %%xmm0    \n\t"
		"mulpd    %%xmm0, %%xmm2           \n\t"
		"mulpd    %%xmm0, %%xmm1           \n\t"
		"movapd  -6 * 16(%%eax), %%xmm0    \n\t"
		"addl   $1 * 4 * 8, %%eax          \n\t" // a += 4 (1 x mr)
		"addpd    %%xmm2, %%xmm6           \n\t"
		"addpd    %%xmm1, %%xmm7           \n\t"
		"                                  \n\t"
		"decl   %%esi                      \n\t" // i -= 1;
		"jne    .LOOPKLEFT                 \n\t" // iterate again if i != 0.
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t"
		".POSTACCUM:                       \n\t"
		"                                  \n\t"
		"                                  \n\t"
		"movl      %6, %%ebx               \n\t" // load address of b11.
		"                                  \n\t"
		"                                  \n\t" // xmm4 == ( ab00    xmm5 == ( ab01  
		"                                  \n\t" //           ab10 )            ab11 )
		"                                  \n\t" // xmm6 == ( ab20    xmm7 == ( ab21  
		"                                  \n\t" //           ab30 )            ab31 )
		"movapd    %%xmm4, %%xmm0          \n\t" 
		"unpcklpd  %%xmm5, %%xmm0          \n\t"
		"unpckhpd  %%xmm5, %%xmm4          \n\t"
		"movapd    %%xmm4, %%xmm1          \n\t"
		"                                  \n\t"
		"movapd    %%xmm6, %%xmm2          \n\t"
		"unpcklpd  %%xmm7, %%xmm2          \n\t"
		"unpckhpd  %%xmm7, %%xmm6          \n\t"
		"movapd    %%xmm6, %%xmm3          \n\t"
		"                                  \n\t" // xmm0 == ( ab00 ab01 )
		"                                  \n\t" // xmm1 == ( ab10 ab11 )
		"                                  \n\t" // xmm2 == ( ab20 ab21 )
		"                                  \n\t" // xmm3 == ( ab30 ab31 )
		"                                  \n\t"
		"movl     %10, %%eax               \n\t" // load address of alpha
		"movddup  (%%eax), %%xmm7          \n\t" // load alpha and duplicate
		"                                  \n\t"
		"movapd  0 * 16(%%ebx), %%xmm4     \n\t"
		"movapd  1 * 16(%%ebx), %%xmm5     \n\t"
		"mulpd    %%xmm7, %%xmm4           \n\t" // xmm4 = alpha * ( beta00 beta01 )
		"mulpd    %%xmm7, %%xmm5           \n\t" // xmm5 = alpha * ( beta10 beta11 )
		"movapd  2 * 16(%%ebx), %%xmm6     \n\t"
		"mulpd    %%xmm7, %%xmm6           \n\t" // xmm6 = alpha * ( beta20 beta21 )
		"mulpd   3 * 16(%%ebx), %%xmm7     \n\t" // xmm7 = alpha * ( beta30 beta31 )
		"                                  \n\t"
		"subpd    %%xmm0, %%xmm4           \n\t" // xmm4 -= xmm0
		"subpd    %%xmm1, %%xmm5           \n\t" // xmm5 -= xmm1
		"subpd    %%xmm2, %%xmm6           \n\t" // xmm6 -= xmm2
		"subpd    %%xmm3, %%xmm7           \n\t" // xmm7 -= xmm3
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t"
		".TRSM:                            \n\t"
		"                                  \n\t"
		"                                  \n\t"
		"movl     %3, %%eax                \n\t" // load address of a11
		"movl     %7, %%ecx                \n\t" // load address of c11
		"                                  \n\t"
		"movl     %8, %%edi                \n\t" // load rs_c
		"movl     %9, %%esi                \n\t" // load cs_c
		"sall     $3, %%edi                \n\t" // rs_c *= sizeof( double )
		"sall     $3, %%esi                \n\t" // cs_c *= sizeof( double )
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t" // iteration 0
		"                                  \n\t"
		"movddup (0+0*4)*8(%%eax), %%xmm0  \n\t" // load xmm0 = (1/alpha00)
		"                                  \n\t"
		"mulpd    %%xmm0, %%xmm4           \n\t" // xmm4 *= (1/alpha00);
		"                                  \n\t"
		"movapd   %%xmm4,  0 * 16(%%ebx)   \n\t" // store ( beta00 beta01 ) = xmm4
		"movlpd   %%xmm4,  (%%ecx)         \n\t" // store ( gamma00 ) = xmm4[0]
		"movhpd   %%xmm4,  (%%ecx,%%esi)   \n\t" // store ( gamma01 ) = xmm4[1]
		"addl     %%edi, %%ecx             \n\t" // c11 += rs_c
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t" // iteration 1
		"                                  \n\t"
		"movddup (1+0*4)*8(%%eax), %%xmm0  \n\t" // load xmm0 = alpha10
		"movddup (1+1*4)*8(%%eax), %%xmm1  \n\t" // load xmm1 = (1/alpha11)
		"                                  \n\t"
		"mulpd    %%xmm4, %%xmm0           \n\t" // xmm0 = alpha10 * ( beta00 beta01 )
		"subpd    %%xmm0, %%xmm5           \n\t" // xmm5 -= xmm0
		"mulpd    %%xmm1, %%xmm5           \n\t" // xmm5 *= (1/alpha11);
		"                                  \n\t"
		"movapd   %%xmm5,  1 * 16(%%ebx)   \n\t" // store ( beta10 beta11 ) = xmm5
		"movlpd   %%xmm5,  (%%ecx)         \n\t" // store ( gamma10 ) = xmm5[0]
		"movhpd   %%xmm5,  (%%ecx,%%esi)   \n\t" // store ( gamma11 ) = xmm5[1]
		"addl     %%edi, %%ecx             \n\t" // c11 += rs_c
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t" // iteration 2
		"                                  \n\t"
		"movddup (2+0*4)*8(%%eax), %%xmm0  \n\t" // load xmm0 = alpha20
		"movddup (2+1*4)*8(%%eax), %%xmm1  \n\t" // load xmm1 = alpha21
		"movddup (2+2*4)*8(%%eax), %%xmm2  \n\t" // load xmm2 = (1/alpha22)
		"                                  \n\t"
		"mulpd    %%xmm4, %%xmm0           \n\t" // xmm0 = alpha20 * ( beta00 beta01 )
		"mulpd    %%xmm5, %%xmm1           \n\t" // xmm1 = alpha21 * ( beta10 beta11 )
		"addpd    %%xmm1, %%xmm0           \n\t" // xmm0 += xmm1;
		"subpd    %%xmm0, %%xmm6           \n\t" // xmm6 -= xmm0
		"mulpd    %%xmm2, %%xmm6           \n\t" // xmm6 *= (1/alpha22);
		"                                  \n\t"
		"movapd   %%xmm6,  2 * 16(%%ebx)   \n\t" // store ( beta20 beta21 ) = xmm6
		"movlpd   %%xmm6,  (%%ecx)         \n\t" // store ( gamma20 ) = xmm6[0]
		"movhpd   %%xmm6,  (%%ecx,%%esi)   \n\t" // store ( gamma21 ) = xmm6[1]
		"addl     %%edi, %%ecx             \n\t" // c11 += rs_c
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t" // iteration 3
		"                                  \n\t"
		"movddup (3+0*4)*8(%%eax), %%xmm0  \n\t" // load xmm0 = alpha30
		"movddup (3+1*4)*8(%%eax), %%xmm1  \n\t" // load xmm1 = alpha31
		"movddup (3+2*4)*8(%%eax), %%xmm2  \n\t" // load xmm2 = alpha32
		"movddup (3+3*4)*8(%%eax), %%xmm3  \n\t" // load xmm3 = (1/alpha33)
		"                                  \n\t"
		"mulpd    %%xmm4, %%xmm0           \n\t" // xmm0 = alpha30 * ( beta00 beta01 )
		"mulpd    %%xmm5, %%xmm1           \n\t" // xmm1 = alpha31 * ( beta10 beta11 )
		"mulpd    %%xmm6, %%xmm2           \n\t" // xmm2 = alpha32 * ( beta20 beta21 )
		"addpd    %%xmm1, %%xmm0           \n\t" // xmm0 += xmm1
		"addpd    %%xmm2, %%xmm0           \n\t" // xmm0 += xmm2
		"subpd    %%xmm0, %%xmm7           \n\t" // xmm7 -= xmm0
		"mulpd    %%xmm3, %%xmm7           \n\t" // xmm7 *= (1/alpha33);
		"                                  \n\t"
		"movapd   %%xmm7,  3 * 16(%%ebx)   \n\t" // store ( beta30 beta31 ) = xmm7
		"movlpd   %%xmm7,  (%%ecx)         \n\t" // store ( gamma30 ) = xmm7[0]
		"movhpd   %%xmm7,  (%%ecx,%%esi)   \n\t" // store ( gamma31 ) = xmm7[1]
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t"
		".UPDATEBD11:                      \n\t"
		"                                  \n\t"
		"                                  \n\t"
		"movl     %5, %%edx                \n\t"
		"                                  \n\t"
		"movddup  %%xmm4, %%xmm0           \n\t"
		"movddup  %%xmm5, %%xmm1           \n\t"
		"movddup  %%xmm6, %%xmm2           \n\t"
		"movddup  %%xmm7, %%xmm3           \n\t"
		"                                  \n\t"
		"unpckhpd %%xmm4, %%xmm4           \n\t"
		"unpckhpd %%xmm5, %%xmm5           \n\t"
		"unpckhpd %%xmm6, %%xmm6           \n\t"
		"unpckhpd %%xmm7, %%xmm7           \n\t"
		"                                  \n\t"
		"movapd   %%xmm0,  0 * 16(%%edx)   \n\t"
		"movapd   %%xmm4,  1 * 16(%%edx)   \n\t"
		"movapd   %%xmm1,  2 * 16(%%edx)   \n\t"
		"movapd   %%xmm5,  3 * 16(%%edx)   \n\t"
		"movapd   %%xmm2,  4 * 16(%%edx)   \n\t"
		"movapd   %%xmm6,  5 * 16(%%edx)   \n\t"
		"movapd   %%xmm3,  6 * 16(%%edx)   \n\t"
		"movapd   %%xmm7,  7 * 16(%%edx)   \n\t"
		"                                  \n\t"
		"                                  \n\t"

		: // output operands (none)
		: // input operands
		  "m" (k_iter),
		  "m" (k_left),
		  "m" (a10),
		  "m" (a11),
		  "m" (bd01),
		  "m" (bd11),
		  "m" (b11),
		  "m" (c11),
		  "m" (rs_c),
		  "m" (cs_c),
		  "m" (alpha)
		: // register clobber list
		  "eax", "ebx", "ecx", "edx", "esi", "edi",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7",
		  "memory"
	);

}

void bli_cgemmtrsm_l_opt_d4x2(
                               dim_t              k,
                               scomplex* restrict alpha,
                               scomplex* restrict a10,
                               scomplex* restrict a11,
                               scomplex* restrict bd01,
                               scomplex* restrict bd11,
                               scomplex* restrict b11,
                               scomplex* restrict c11, inc_t rs_c, inc_t cs_c,
                               scomplex* restrict a_next,
                               scomplex* restrict b_next
                             )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

void bli_zgemmtrsm_l_opt_d4x2(
                               dim_t              k,
                               dcomplex* restrict alpha,
                               dcomplex* restrict a10,
                               dcomplex* restrict a11,
                               dcomplex* restrict bd01,
                               dcomplex* restrict bd11,
                               dcomplex* restrict b11,
                               dcomplex* restrict c11, inc_t rs_c, inc_t cs_c,
                               dcomplex* restrict a_next,
                               dcomplex* restrict b_next
                             )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

