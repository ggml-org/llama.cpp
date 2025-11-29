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

void bli_spackm_2xk(
                     conj_t  conja,
                     dim_t   n,
                     void*   beta,
                     void*   a, inc_t inca, inc_t lda,
                     void*   p
                   )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

void bli_dpackm_2xk(
                     conj_t  conja,
                     dim_t   n,
                     void*   beta,
                     void*   a, inc_t inca, inc_t lda,
                     void*   p
                   )
{
	double* restrict beta_cast = beta;
	double* restrict alpha1    = a;
	double* restrict pi1       = p;

	inc_t            off1      = 1 * inca * sizeof(double);
	inc_t            ldas      =      lda * sizeof(double);

	if ( bli_deq1( *beta_cast ) )
	{
		dim_t n_iter = n / 4;
		dim_t n_left = n % 4;

		__asm__ volatile
		(
			"                                 \n\t"
			"movl     %2, %%edi               \n\t" // load a
			"movl     %3, %%ebp               \n\t" // load p
			"                                 \n\t"
			"movl     %4, %%eax               \n\t" // load ldas
			"leal  (%%edi,%%eax), %%edx       \n\t" // load a + ldas
			"sall     $1, %%eax               \n\t" // ldas *= 2;
			"                                 \n\t"
			"movl     %5, %%ebx               \n\t" // load off1
			"                                 \n\t"
			"                                 \n\t"
			"movl      %0, %%esi              \n\t"
			"testl  %%esi, %%esi              \n\t"
			"je     .DCONSIDERKLEFT           \n\t"
			"                                 \n\t"
			"                                 \n\t"
			".DLOOPKITER:                     \n\t"
			"                                 \n\t"
			"addl     $64, %%ebp              \n\t"
			"                                 \n\t"
			"movlpd  (%%edi        ), %%xmm0  \n\t" // iteration 0
			"movhpd  (%%edi,%%ebx, ), %%xmm0  \n\t"
			"addl    %%eax, %%edi             \n\t"
			"movapd  %%xmm0,  -8 * 8(%%ebp)   \n\t"
			"                                 \n\t"
			"movlpd  (%%edx        ), %%xmm1  \n\t" // iteration 1
			"movhpd  (%%edx,%%ebx, ), %%xmm1  \n\t"
			"addl    %%eax, %%edx             \n\t"
			"movapd  %%xmm1,  -6 * 8(%%ebp)   \n\t"
			"                                 \n\t"
			"movlpd  (%%edi        ), %%xmm2  \n\t" // iteration 2
			"movhpd  (%%edi,%%ebx, ), %%xmm2  \n\t"
			"addl    %%eax, %%edi             \n\t"
			"movapd  %%xmm2,  -4 * 8(%%ebp)   \n\t"
			"                                 \n\t"
			"movlpd  (%%edx        ), %%xmm3  \n\t" // iteration 3
			"movhpd  (%%edx,%%ebx, ), %%xmm3  \n\t"
			"addl    %%eax, %%edx             \n\t"
			"movapd  %%xmm3,  -2 * 8(%%ebp)   \n\t"
			"                                 \n\t"
			"decl   %%esi                     \n\t"
			"jne    .DLOOPKITER               \n\t"
			"                                 \n\t"
			"                                 \n\t"
			"                                 \n\t"
			".DCONSIDERKLEFT:                 \n\t"
			"                                 \n\t"
			"movl      %1, %%esi              \n\t"
			"testl  %%esi, %%esi              \n\t"
			"je     .DDONE                    \n\t"
			"                                 \n\t"
			"                                 \n\t"
			"                                 \n\t"
			".DLOOPKLEFT:                     \n\t"
			"                                 \n\t"
			"addl     $16, %%ebp              \n\t"
			"                                 \n\t"
			"movlpd  (%%edi        ), %%xmm0  \n\t"
			"movhpd  (%%edi,%%ebx, ), %%xmm0  \n\t"
			"addl    %%eax, %%edi             \n\t"
			"movapd  %%xmm0,  -2 * 8(%%ebp)   \n\t"
			"                                 \n\t"
			"decl   %%esi                     \n\t"
			"jne    .DLOOPKLEFT               \n\t"
			"                                 \n\t"
			"                                 \n\t"
			"                                 \n\t"
			".DDONE:                          \n\t"
			"                                 \n\t"
			: // output operands
			: // input operands
			  "m" (n_iter),
			  "m" (n_left),
			  "m" (alpha1),
			  "m" (pi1),
			  "m" (ldas),
			  "m" (off1)
			: // register clobber list
			  "eax", "ebx", "ecx", "edx", "edi", "ebp", "esi",
			  "xmm0", "xmm1", "xmm2", "xmm3",
			  "xmm4", "xmm5", "xmm6", "xmm7",
			  "memory"
		);
	}
	else
	{
		dim_t n_iter = n / 4;
		dim_t n_left = n % 4;

		__asm__ volatile
		(
			"                                 \n\t"
			"movl     %2, %%edi               \n\t" // load a
			"movl     %3, %%ebp               \n\t" // load p
			"                                 \n\t"
			"movl     %4, %%eax               \n\t" // load ldas
			"leal  (%%edi,%%eax), %%edx       \n\t" // load a + ldas
			"sall     $1, %%eax               \n\t" // ldas *= 2;
			"                                 \n\t"
			"movl     %5, %%ebx               \n\t" // load off1
			"                                 \n\t"
			"movl          %6, %%esi          \n\t" // load beta
			"movddup  (%%esi), %%xmm7         \n\t" // load and duplicate *beta
			"                                 \n\t"
			"movl      %0, %%esi              \n\t"
			"testl  %%esi, %%esi              \n\t"
			"je     .DCONSIDERKLEFT2          \n\t"
			"                                 \n\t"
			"                                 \n\t"
			".DLOOPKITER2:                    \n\t"
			"                                 \n\t"
			"addl     $64, %%ebp              \n\t"
			"                                 \n\t"
			"movlpd  (%%edi        ), %%xmm0  \n\t" // iteration 0
			"movhpd  (%%edi,%%ebx, ), %%xmm0  \n\t"
			"mulpd   %%xmm7, %%xmm0           \n\t"
			"addl    %%eax, %%edi             \n\t"
			"movapd  %%xmm0,  -8 * 8(%%ebp)   \n\t"
			"                                 \n\t"
			"movlpd  (%%edx        ), %%xmm1  \n\t" // iteration 1
			"movhpd  (%%edx,%%ebx, ), %%xmm1  \n\t"
			"mulpd   %%xmm7, %%xmm1           \n\t"
			"addl    %%eax, %%edx             \n\t"
			"movapd  %%xmm1,  -6 * 8(%%ebp)   \n\t"
			"                                 \n\t"
			"movlpd  (%%edi        ), %%xmm2  \n\t" // iteration 2
			"movhpd  (%%edi,%%ebx, ), %%xmm2  \n\t"
			"mulpd   %%xmm7, %%xmm2           \n\t"
			"addl    %%eax, %%edi             \n\t"
			"movapd  %%xmm2,  -4 * 8(%%ebp)   \n\t"
			"                                 \n\t"
			"movlpd  (%%edx        ), %%xmm3  \n\t" // iteration 3
			"movhpd  (%%edx,%%ebx, ), %%xmm3  \n\t"
			"mulpd   %%xmm7, %%xmm3           \n\t"
			"addl    %%eax, %%edx             \n\t"
			"movapd  %%xmm3,  -2 * 8(%%ebp)   \n\t"
			"                                 \n\t"
			"decl   %%esi                     \n\t"
			"jne    .DLOOPKITER2              \n\t"
			"                                 \n\t"
			"                                 \n\t"
			"                                 \n\t"
			".DCONSIDERKLEFT2:                \n\t"
			"                                 \n\t"
			"movl      %1, %%esi              \n\t"
			"testl  %%esi, %%esi              \n\t"
			"je     .DDONE2                   \n\t"
			"                                 \n\t"
			"                                 \n\t"
			"                                 \n\t"
			".DLOOPKLEFT2:                    \n\t"
			"                                 \n\t"
			"addl     $16, %%ebp              \n\t"
			"                                 \n\t"
			"movlpd  (%%edi        ), %%xmm0  \n\t"
			"movhpd  (%%edi,%%ebx, ), %%xmm0  \n\t"
			"mulpd   %%xmm7, %%xmm0           \n\t"
			"addl    %%eax, %%edi             \n\t"
			"movapd  %%xmm0,  -2 * 8(%%ebp)   \n\t"
			"                                 \n\t"
			"decl   %%esi                     \n\t"
			"jne    .DLOOPKLEFT2              \n\t"
			"                                 \n\t"
			"                                 \n\t"
			"                                 \n\t"
			".DDONE2:                         \n\t"
			"                                 \n\t"
			: // output operands
			: // input operands
			  "m" (n_iter),
			  "m" (n_left),
			  "m" (alpha1),
			  "m" (pi1),
			  "m" (ldas),
			  "m" (off1),
			  "m" (beta)
			: // register clobber list
			  "eax", "ebx", "ecx", "edx", "edi", "ebp", "esi",
			  "xmm0", "xmm1", "xmm2", "xmm3",
			  "xmm4", "xmm5", "xmm6", "xmm7",
			  "memory"
		);
	}
}

void bli_cpackm_2xk(
                     conj_t  conja,
                     dim_t   n,
                     void*   beta,
                     void*   a, inc_t inca, inc_t lda,
                     void*   p
                   )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

void bli_zpackm_2xk(
                     conj_t  conja,
                     dim_t   n,
                     void*   beta,
                     void*   a, inc_t inca, inc_t lda,
                     void*   p
                   )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

