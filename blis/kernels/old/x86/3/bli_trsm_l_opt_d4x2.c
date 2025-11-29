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

void bli_strsm_l_opt_d4x2(
                           float* restrict   a11,
                           float* restrict   b11,
                           float* restrict   bd11,
                           float* restrict   c11, inc_t rs_c, inc_t cs_c
                         )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

void bli_dtrsm_l_opt_d4x2(
                           double* restrict  a11,
                           double* restrict  b11,
                           double* restrict  bd11,
                           double* restrict  c11, inc_t rs_c, inc_t cs_c
                         )
{
	__asm__ volatile
	(
		"                                  \n\t"
		"movl      %1, %%ebx               \n\t" // load address of b11.
		"                                  \n\t"
		"movapd  0 * 16(%%ebx), %%xmm4     \n\t" // load xmm4 = ( beta00 beta01 )
		"movapd  1 * 16(%%ebx), %%xmm5     \n\t" // load xmm5 = ( beta10 beta11 )
		"movapd  2 * 16(%%ebx), %%xmm6     \n\t" // load xmm6 = ( beta20 beta21 )
		"movapd  3 * 16(%%ebx), %%xmm7     \n\t" // load xmm7 = ( beta30 beta31 )
		"                                  \n\t"
		"                                  \n\t"
		"movl     %0, %%eax                \n\t" // load address of a11
		"movl     %3, %%ecx                \n\t" // load address of c11
		"                                  \n\t"
		"movl     %4, %%edi                \n\t" // load rs_c
		"movl     %5, %%esi                \n\t" // load cs_c
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
		"movl     %2, %%edx                \n\t"
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
		  "m" (a11),
		  "m" (b11),
		  "m" (bd11),
		  "m" (c11),
		  "m" (rs_c),
		  "m" (cs_c)
		: // register clobber list
		  "eax", "ebx", "ecx", "edx", "esi", "edi",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7",
		  "memory"
	);

}

void bli_ctrsm_l_opt_d4x2(
                           scomplex* restrict a11,
                           scomplex* restrict b11,
                           scomplex* restrict bd11,
                           scomplex* restrict c11, inc_t rs_c, inc_t cs_c
                         )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

void bli_ztrsm_l_opt_d4x2(
                           dcomplex* restrict a11,
                           dcomplex* restrict b11,
                           dcomplex* restrict bd11,
                           dcomplex* restrict c11, inc_t rs_c, inc_t cs_c
                         )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

