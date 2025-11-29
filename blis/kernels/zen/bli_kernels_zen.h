/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

// -- level-1m --
PACKM_KER_PROT(double, d, packm_8xk_gen_zen)
PACKM_KER_PROT(double, d, packm_6xk_gen_zen)
PACKM_KER_PROT(double, d, packm_8xk_nn_zen)
PACKM_KER_PROT(double, d, packm_6xk_nn_zen)


// -- level-1v --

// amaxv (intrinsics)
AMAXV_KER_PROT( float,    s, amaxv_zen_int )
AMAXV_KER_PROT( double,   d, amaxv_zen_int )

// axpyv (intrinsics)
AXPYV_KER_PROT( float,    s, axpyv_zen_int )
AXPYV_KER_PROT( double,   d, axpyv_zen_int )

// axpyv (intrinsics unrolled x10)
AXPYV_KER_PROT( float,    s, axpyv_zen_int_10 )
AXPYV_KER_PROT( double,   d, axpyv_zen_int_10 )
AXPYV_KER_PROT( scomplex, c, axpyv_zen_int_5 )
AXPYV_KER_PROT( dcomplex, z, axpyv_zen_int_5 )
// dotv (intrinsics)
DOTV_KER_PROT( float,    s, dotv_zen_int )
DOTV_KER_PROT( double,   d, dotv_zen_int )

// dotv (intrinsics, unrolled x10)
DOTV_KER_PROT( float,    s, dotv_zen_int10 )
DOTV_KER_PROT( double,   d, dotv_zen_int10 )

// dotxv (intrinsics)
DOTXV_KER_PROT( float,    s, dotxv_zen_int )
DOTXV_KER_PROT( double,   d, dotxv_zen_int )

// scalv (intrinsics)
SCALV_KER_PROT( float,    s, scalv_zen_int )
SCALV_KER_PROT( double,   d, scalv_zen_int )

// scalv (intrinsics unrolled x10)
SCALV_KER_PROT( float,    s, scalv_zen_int10 )
SCALV_KER_PROT( double,   d, scalv_zen_int10 )
SCALV_KER_PROT( scomplex, c, scalv_zen_int10 )

// swapv (intrinsics)
SWAPV_KER_PROT(float,    s, swapv_zen_int8 )
SWAPV_KER_PROT(double,   d, swapv_zen_int8 )

// copyv (intrinsics)
COPYV_KER_PROT( float,    s, copyv_zen_int )
COPYV_KER_PROT( double,   d, copyv_zen_int )

//
SETV_KER_PROT(float,    s, setv_zen_int)
SETV_KER_PROT(double,   d, setv_zen_int)

// swapv (intrinsics)
SWAPV_KER_PROT(float, 	s, swapv_zen_int8 )
SWAPV_KER_PROT(double,	d, swapv_zen_int8 )


// -- level-1f --

// axpyf (intrinsics)
AXPYF_KER_PROT( float,    s, axpyf_zen_int_8 )
AXPYF_KER_PROT( double,   d, axpyf_zen_int_8 )
AXPYF_KER_PROT( float,    s, axpyf_zen_int_5 )
AXPYF_KER_PROT( double,   d, axpyf_zen_int_5 )
AXPYF_KER_PROT( scomplex,    c, axpyf_zen_int_5 )
AXPYF_KER_PROT( dcomplex,   z, axpyf_zen_int_5 )

AXPYF_KER_PROT( double,   d, axpyf_zen_int_16x4 )
AXPYF_KER_PROT( double,   d, axpyf_zen_int_16x2 )
AXPYF_KER_PROT( scomplex, c, axpyf_zen_int_4 )

// dotxf (intrinsics)
DOTXF_KER_PROT( float,    s, dotxf_zen_int_8 )
DOTXF_KER_PROT( double,   d, dotxf_zen_int_8 )

// -- level-3 sup --------------------------------------------------------------

// semmsup_rv

//GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_6x16 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_5x16 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_4x16 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_3x16 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_2x16 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_1x16 )

GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_6x8 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_5x8 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_4x8 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_3x8 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_2x8 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_1x8 )

GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_6x4 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_5x4 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_4x4 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_3x4 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_2x4 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_1x4 )

GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_6x2 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_5x2 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_4x2 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_3x2 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_2x2 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_1x2 )

GEMMSUP_KER_PROT( float,   s, gemmsup_r_zen_ref_6x1 )
GEMMSUP_KER_PROT( float,   s, gemmsup_r_zen_ref_5x1 )
GEMMSUP_KER_PROT( float,   s, gemmsup_r_zen_ref_4x1 )
GEMMSUP_KER_PROT( float,   s, gemmsup_r_zen_ref_3x1 )
GEMMSUP_KER_PROT( float,   s, gemmsup_r_zen_ref_2x1 )
GEMMSUP_KER_PROT( float,   s, gemmsup_r_zen_ref_1x1 )

// gemmsup_rv (mkernel in m dim)
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_6x16m )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_6x8m )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_6x4m )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_6x2m )
//gemmsup_rv (mkernel in m dim) for mask load/store
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_6x16m_mask )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_6x8m_mask )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_6x4m_mask )
GEMMSUP_KER_PROT( float,   s, bli_sgemmsup_rv_zen_asm_6x8m )
GEMMSUP_KER_PROT( float,   s, bli_sgemmsup_rv_zen_asm_6x4m )
GEMMSUP_KER_PROT( float,   s, bli_sgemmsup_rv_zen_asm_6x2m )

//gemmsup_rv (mkernel in m dim) for fringe case
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_1x16_mask )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_2x16_mask )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_3x16_mask )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_4x16_mask )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_5x16_mask )

GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_1x8_mask )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_2x8_mask )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_3x8_mask )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_4x8_mask )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_5x8_mask )

GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_1x4_mask )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_2x4_mask )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_3x4_mask )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_4x4_mask )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_5x4_mask )

// gemmsup_rv (mkernel in n dim)

GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_6x16n )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_5x16n )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_4x16n )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_3x16n )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_2x16n )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_1x16n )

// gemmsup_rd
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_2x8)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_2x16)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_1x8)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_1x16)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_6x4)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_2x4)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_1x4)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_6x2)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_3x2)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_2x2)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_1x2)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_6x16m)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_6x8m)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_6x4m)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_6x2m)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_6x16n)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_3x16n)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_2x16n)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_1x16n)

