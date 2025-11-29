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

#include "gemm_template.h"
#include "bli_sandbox.h"


GENERIC_GEMM( 
    sb, // kernel name prefix 
    bfloat16, // input type
    float, // output type    
    (pb/2 + pb%2), // innermost loop iterations
    sb_pack_a,
    sb_pack_b, // pack kernel for B
    bli_sbgemm_power10_mma_8x16, // microkernel function name
    2, // K_MMA
    8, // MR
    16, // NR
    384, // MC
    3328, // KC
    4096, // NC
    0, // A_ALIGN
    0 // B_ALIGN
);

GENERIC_GEMM( 
    sh, // kernel name prefix 
    float16, // input type
    float, // output type    
    (pb/2 + pb%2), // innermost loop iterations
    sh_pack_a, // pack kernel for A
    sh_pack_b, // pack kernel for B
    bli_shgemm_power10_mma_8x16, // microkernel function name
    2, // K_MMA
    8, // MR
    16, // NR
    384, // MC
    3328, // KC
    4096, // NC
    0, // A_ALIGN
    0 // B_ALIGN
);

GENERIC_GEMM( 
    i16, // kernel name prefix 
    int16_t, // input type
    int, // output type    
    (pb/2 + pb%2), // innermost loop iterations
    i16_pack_a, // pack kernel for A
    i16_pack_b, // pack kernel for B
    bli_i16gemm_power10_mma_8x16, // microkernel function name
    2, // K_MMA
    8, // MR
    16, // NR
    384, // MC
    3328, // KC
    4096, // NC
    0, // A_ALIGN
    0 // B_ALIGN
);

GENERIC_GEMM( 
    i8, // kernel name prefix 
    int8_t, // input type
    int, // output type    
    (pb/4 + (pb%4>0)), // innermost loop iterations
    i8_pack_a, // pack kernel for A
    i8_pack_b, // pack kernel for B
    bli_i8gemm_power10_mma_8x16, // microkernel function name
    4, // K_MMA
    8, // MR
    16, // NR
    384, // MC
    6656, // KC
    4096, // NC
    0, // A_ALIGN
    0 // B_ALIGN
);

GENERIC_GEMM( 
    i4, // kernel name prefix 
    nibbles, // input type
    int, // output type    
    (pb/8 + (pb%8>0)), // innermost loop iterations
    i4_pack_a, // pack kernel for A
    i4_pack_b, // pack kernel for B
    bli_i4gemm_power10_mma_8x16, // microkernel function name
    8, // K_MMA
    8, // MR
    16, // NR
    384, // MC
    6656, // KC
    4096, // NC
    0, // A_ALIGN
    0 // B_ALIGN
);

