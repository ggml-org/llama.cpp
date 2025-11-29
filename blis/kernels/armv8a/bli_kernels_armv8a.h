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

PACKM_KER_PROT( float,    s, packm_armv8a_int_8x12 )
PACKM_KER_PROT( double,   d, packm_armv8a_int_6x8 )

GEMM_UKR_PROT( float,    s, gemm_armv8a_asm_8x12 )
GEMM_UKR_PROT( double,   d, gemm_armv8a_asm_6x8 )
GEMM_UKR_PROT( float,    s, gemm_armv8a_asm_12x8r )
GEMM_UKR_PROT( double,   d, gemm_armv8a_asm_8x6r )
// GEMM_UKR_PROT( double,   d, gemm_armv8a_asm_6x8r )
// GEMM_UKR_PROT( double,   d, gemm_armv8a_asm_8x4 )
// GEMM_UKR_PROT( double,   d, gemm_armv8a_asm_4x4 )

GEMMSUP_KER_PROT( double,   d, gemmsup_rd_armv8a_asm_6x8n )
GEMMSUP_KER_PROT( double,   d, gemmsup_rd_armv8a_asm_6x8m )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_armv8a_asm_6x8n )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_armv8a_asm_6x8m )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_armv8a_asm_6x7m )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_armv8a_asm_6x6m )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_armv8a_asm_6x5m )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_armv8a_asm_5x8n )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_armv8a_asm_4x8n )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_armv8a_asm_4x8m )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_armv8a_asm_8x4m )

GEMMSUP_KER_PROT( double,   d, gemmsup_rd_armv8a_int_2x8 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rd_armv8a_int_3x4 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rd_armv8a_asm_3x4 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rd_armv8a_asm_6x3 )

GEMMSUP_KER_PROT( double,   d, gemmsup_rv_armv8a_int_6x4mn )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_armv8a_int_3x8mn )

