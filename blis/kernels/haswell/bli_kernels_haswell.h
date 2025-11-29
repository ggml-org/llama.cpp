/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
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

// -- level-1m -----------------------------------------------------------------

// packm (asm)
PACKM_KER_PROT( float,    s, packm_haswell_asm_6x16 )
PACKM_KER_PROT( double,   d, packm_haswell_asm_6x8 )
PACKM_KER_PROT( scomplex, c, packm_haswell_asm_3x8 )
PACKM_KER_PROT( dcomplex, z, packm_haswell_asm_3x4 )


// -- level-3 ------------------------------------------------------------------

// gemm (asm d6x8)
GEMM_UKR_PROT( float,    s, gemm_haswell_asm_6x16 )
GEMM_UKR_PROT( double,   d, gemm_haswell_asm_6x8 )
GEMM_UKR_PROT( scomplex, c, gemm_haswell_asm_3x8 )
GEMM_UKR_PROT( dcomplex, z, gemm_haswell_asm_3x4 )

// gemm (asm d8x6)
GEMM_UKR_PROT( float,    s, gemm_haswell_asm_16x6 )
GEMM_UKR_PROT( double,   d, gemm_haswell_asm_8x6 )
GEMM_UKR_PROT( scomplex, c, gemm_haswell_asm_8x3 )
GEMM_UKR_PROT( dcomplex, z, gemm_haswell_asm_4x3 )

// gemmtrsm_l (asm d6x8)
GEMMTRSM_UKR_PROT( float,    s, gemmtrsm_l_haswell_asm_6x16 )
GEMMTRSM_UKR_PROT( double,   d, gemmtrsm_l_haswell_asm_6x8 )

// gemmtrsm_u (asm d6x8)
GEMMTRSM_UKR_PROT( float,    s, gemmtrsm_u_haswell_asm_6x16 )
GEMMTRSM_UKR_PROT( double,   d, gemmtrsm_u_haswell_asm_6x8 )


// gemm (asm d8x6)
//GEMM_UKR_PROT( float,    s, gemm_haswell_asm_16x6 )
//GEMM_UKR_PROT( double,   d, gemm_haswell_asm_8x6 )
//GEMM_UKR_PROT( scomplex, c, gemm_haswell_asm_8x3 )
//GEMM_UKR_PROT( dcomplex, z, gemm_haswell_asm_4x3 )


// -- level-3 sup --------------------------------------------------------------

// -- single real --

// gemmsup_r

GEMMSUP_KER_PROT( float,    s, gemmsup_r_haswell_ref_6x1 )
GEMMSUP_KER_PROT( float,    s, gemmsup_r_haswell_ref_5x1 )
GEMMSUP_KER_PROT( float,    s, gemmsup_r_haswell_ref_4x1 )
GEMMSUP_KER_PROT( float,    s, gemmsup_r_haswell_ref_3x1 )
GEMMSUP_KER_PROT( float,    s, gemmsup_r_haswell_ref_2x1 )
GEMMSUP_KER_PROT( float,    s, gemmsup_r_haswell_ref_1x1 )

// gemmsup_rv

GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_6x16 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_5x16 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_4x16 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_3x16 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_2x16 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_1x16 )

GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_6x12 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_5x12 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_4x12 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_3x12 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_2x12 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_1x12 )

GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_6x8 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_5x8 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_4x8 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_3x8 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_2x8 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_1x8 )

GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_6x6 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_5x6 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_4x6 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_3x6 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_2x6 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_1x6 )

GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_6x4 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_5x4 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_4x4 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_3x4 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_2x4 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_1x4 )

GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_6x2 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_5x2 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_4x2 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_3x2 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_2x2 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_1x2 )

// gemmsup_rv (mkernel in m dim)

GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_6x16m )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_6x12m )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_6x8m )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_6x6m )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_6x4m )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_6x2m )

// gemmsup_rv (mkernel in n dim)

GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_6x16n )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_5x16n )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_4x16n )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_3x16n )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_2x16n )
GEMMSUP_KER_PROT( float,    s, gemmsup_rv_haswell_asm_1x16n )

// gemmsup_rd

GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_6x16 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_2x16 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_1x16 )

GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_6x12 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_2x12 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_1x12 )

GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_6x8 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_2x8 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_1x8 )

GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_6x4 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_2x4 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_1x4 )

GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_6x2 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_3x2 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_2x2 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_1x2 )

GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_6x1 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_3x1 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_2x1 )
GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_1x1 )

// gemmsup_rd (mkernel in m dim)

GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_6x16m )
GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_6x12m )
GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_6x8m )
GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_6x4m )
GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_6x2m )

// gemmsup_rd (mkernel in n dim)

GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_6x16n )
GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_3x16n )
GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_2x16n )
GEMMSUP_KER_PROT( float,    s, gemmsup_rd_haswell_asm_1x16n )



// -- double real --

// gemmsup_r

GEMMSUP_KER_PROT( double,   d, gemmsup_r_haswell_ref_6x1 )
GEMMSUP_KER_PROT( double,   d, gemmsup_r_haswell_ref_5x1 )
GEMMSUP_KER_PROT( double,   d, gemmsup_r_haswell_ref_4x1 )
GEMMSUP_KER_PROT( double,   d, gemmsup_r_haswell_ref_3x1 )
GEMMSUP_KER_PROT( double,   d, gemmsup_r_haswell_ref_2x1 )
GEMMSUP_KER_PROT( double,   d, gemmsup_r_haswell_ref_1x1 )

// gemmsup_rv

GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_6x8 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_5x8 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_4x8 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_3x8 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_2x8 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_1x8 )

GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_6x7 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_5x7 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_4x7 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_3x7 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_2x7 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_1x7 )

GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_6x6 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_5x6 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_4x6 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_3x6 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_2x6 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_1x6 )

GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_6x5 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_5x5 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_4x5 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_3x5 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_2x5 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_1x5 )

GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_6x4 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_5x4 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_4x4 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_3x4 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_2x4 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_1x4 )

GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_6x3 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_5x3 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_4x3 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_3x3 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_2x3 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_1x3 )

GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_6x2 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_5x2 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_4x2 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_3x2 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_2x2 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_1x2 )

GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_6x1 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_5x1 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_4x1 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_3x1 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_2x1 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_1x1 )
// gemmsup_rv (mkernel in m dim)

GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_6x8m )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_6x6m )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_6x4m )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_6x2m )

// gemmsup_rv (mkernel in n dim)

GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_6x8n )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_5x8n )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_4x8n )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_3x8n )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_2x8n )
GEMMSUP_KER_PROT( double,   d, gemmsup_rv_haswell_asm_1x8n )

// gemmsup_rd

GEMMSUP_KER_PROT( double,   d, gemmsup_rd_haswell_asm_6x8 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rd_haswell_asm_2x8 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rd_haswell_asm_1x8 )

GEMMSUP_KER_PROT( double,   d, gemmsup_rd_haswell_asm_6x4 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rd_haswell_asm_2x4 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rd_haswell_asm_1x4 )

GEMMSUP_KER_PROT( double,   d, gemmsup_rd_haswell_asm_6x2 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rd_haswell_asm_3x2 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rd_haswell_asm_2x2 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rd_haswell_asm_1x2 )

GEMMSUP_KER_PROT( double,   d, gemmsup_rd_haswell_asm_6x1 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rd_haswell_asm_3x1 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rd_haswell_asm_2x1 )
GEMMSUP_KER_PROT( double,   d, gemmsup_rd_haswell_asm_1x1 )

// gemmsup_rd (mkernel in m dim)

GEMMSUP_KER_PROT( double,   d, gemmsup_rd_haswell_asm_6x8m )
GEMMSUP_KER_PROT( double,   d, gemmsup_rd_haswell_asm_6x4m )
GEMMSUP_KER_PROT( double,   d, gemmsup_rd_haswell_asm_6x2m )

// gemmsup_rd (mkernel in n dim)

GEMMSUP_KER_PROT( double,   d, gemmsup_rd_haswell_asm_6x8n )
GEMMSUP_KER_PROT( double,   d, gemmsup_rd_haswell_asm_3x8n )
GEMMSUP_KER_PROT( double,   d, gemmsup_rd_haswell_asm_2x8n )
GEMMSUP_KER_PROT( double,   d, gemmsup_rd_haswell_asm_1x8n )

