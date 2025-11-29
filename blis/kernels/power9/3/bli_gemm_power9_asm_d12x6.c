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
#include "bli_pwr9_asm_macros_12x6.h"

void bli_dgemm_power9_asm_12x6
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.

	uint64_t k_iter = k / 16;
	uint64_t k_left = k % 16;

	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	GEMM_UKR_SETUP_CT( d, 12, 6, false );

	__asm__ volatile
	(
	"                                               \n\t"
	"ld               %%r7,  %2                     \n\t" // load ptr of A
	"ld               %%r8,  %3                     \n\t" // load ptr of B
	"ld               %%r16, %6                     \n\t" // load ptr of C
	"                                               \n\t"
	"ld               %%r28, %4                     \n\t" // load ptr for alpha
	"ld               %%r29, %5                     \n\t" // load ptr for beta
	"                                               \n\t"
	"ld               %%r11, %0                     \n\t" // load k_iter
	"ld               %%r12, %1                     \n\t" // load k_left
	"                                               \n\t"
	"ld               %%r10, %8                     \n\t" // load cs_c
	"slwi             %%r10, %%r10, 3               \n\t" // mul by size of elem
	"                                               \n\t"
	"ld               %%r9,  %7                     \n\t" // load rs_c
	"slwi             %%r9,  %%r9, 3                \n\t" // mul by size of elem
	"                                               \n\t"
	"ld               %%r26,  0(%%r29)              \n\t" // load val of beta
	"                                               \n\t"
	"lxvdsx           %%vs62, 0, %%r28              \n\t" // splat alpha
	"lxvdsx           %%vs63, 0, %%r29              \n\t" // splat beta
	"                                               \n\t"
	"add              %%r17, %%r16, %%r10           \n\t" // addr of col 1 of C
	"add              %%r18, %%r17, %%r10           \n\t" //         col 2 of C
	"add              %%r19, %%r18, %%r10           \n\t" //         col 3 of C
	"add              %%r20, %%r19, %%r10           \n\t" //         col 4 of C
	"add              %%r21, %%r20, %%r10           \n\t" //         col 5 of C
	"                                               \n\t"
	DZERO_OUT_VREG
	"                                               \n\t"
	DPRELOAD
	"                                               \n\t"
	"addi             %%r8, %%r8, 96                \n\t" // move to next col/row of A/B
	"addi             %%r7, %%r7, 96                \n\t"
	"                                               \n\t"
	DPREFETCH
	"                                               \n\t"
	"cmpwi                  %%r11, 0                \n\t" // if k_iter == 0,
	"beq                    DCONSIDERKLEFT          \n\t" // then jmp to k_left
	"mtctr            %%r11                         \n\t" // else, do k_iter loop
	"                                               \n\t"
	"DLOOPKITER:                                    \n\t" // k_iter loop
	"                                               \n\t"
	A_B_PRODUCT_16                                        // compute A*B
	"                                               \n\t"
	"bdnz             DLOOPKITER                    \n\t"
	"                                               \n\t"
	"DCONSIDERKLEFT:                                \n\t"
	"                                               \n\t"
	"cmpwi                  %%r12, 0                \n\t" // if k_left == 0,
	"beq                    DPOSTACCUM              \n\t" // then jmp to post accum
	"mtctr            %%r12                         \n\t" // else, do k_left loop
	"                                               \n\t"
	"DLOOPKLEFT:                                    \n\t" // k_left loop
	"                                               \n\t"
	A_B_PRODUCT_1
	"                                               \n\t"
	"bdnz             DLOOPKLEFT                    \n\t"
	"                                               \n\t"
	"DPOSTACCUM:                                    \n\t"
	"                                               \n\t"
	DSCALE_ALPHA
	"                                               \n\t"
	"cmpdi                  %%r26, 0                \n\t" // if beta == 0,
	"beq                    DBETAZERO               \n\t" // then jmp to BZ
	"                                               \n\t"
	DCOL_SCALE_BETA
	"                                               \n\t"
	"DBETAZERO:                                     \n\t" // BZ case
	"                                               \n\t"
	DCOL_STORE
	"                                               \n\t"
	"DDONE:                                         \n\t"
	"                                               \n\t"
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
	/* unclobberable regs: r2, r3, r4, r5, r6, r13, r14, r15, r30, r31 */
	"r0", "r7",  "r8",  "r9",
	"r10", "r11", "r12", "r16", "r17", "r18", "r19",
	"r20", "r21", "r22", "r23", "r24", "r25", "r26", "r27", "r28", "r29"

	#if XLC
	,"f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"
	, "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19"
	, "f20" ,"f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29"
	, "f30" ,"f31"
	, "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9"
	, "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19"
	, "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29"
	, "v30", "v31"
	#else
	, "vs0", "vs1", "vs2", "vs3", "vs4", "vs5", "vs6", "vs7", "vs8", "vs9"
	, "vs10", "vs11", "vs12", "vs13", "vs14", "vs15", "vs16", "vs17", "vs18", "vs19"
	, "vs20", "vs21", "vs22", "vs23", "vs24", "vs25", "vs26", "vs27", "vs28", "vs29"
	, "vs30", "vs31", "vs32", "vs33", "vs34", "vs35", "vs36", "vs37", "vs38", "vs39"
	, "vs40", "vs41", "vs42", "vs43", "vs44", "vs45", "vs46", "vs47", "vs48", "vs49"
	, "vs50", "vs51", "vs52", "vs53"
	#endif

	);

	GEMM_UKR_FLUSH_CT( d );
}
