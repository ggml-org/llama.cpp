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

/*

    Details on bit16_dt vector data structure

    Vector X = [ X[0,0] X[0,1] X[1,0] X[1,1] X[2,0] X[2,1] X[3,0] X[3,1] ]
    Vector Y = [ Y[0,0] Y[0,1] Y[1,0] Y[1,1] Y[2,0] Y[2,1] Y[3,0] Y[3,1] ]

    These bit16_dt vectors represent a 4x2 matrix. Hence, in matrix form it 
    looks like the following:

    X = [ X[0,0] X[0,1] 
          X[1,0] X[1,1]
          X[2,0] X[2,1]
          X[3,0] X[3,1] ]

    The outer product instruction: xvbf16ger2 (bfloat16 outer product)

    Syntax: 

        xvbf16ger2 ACCUMULATOR A, VECTOR X, VECTOR Y

    Semantics:

        A = X * Y^T

    The generic packing routine would load 8 elements from the same column.
    This causes an issue since the instruction expects the vector to be a
    4x2 matrix where the data is packed in contiguous order. Thus, we must make 
    a packing routine that will interleave the matrix data. Making it so 
    that when we load the 8 contiguous elements from A, it will represent
    a 4x2 section of the matrix.

*/

#include "pack_a_templates.h"
#include "pack_b_templates.h"
#include "bli_sandbox.h"

// 16 bit routines
BIT16_PACK_A(sb, bfloat16);
BIT16_PACK_B(sb, bfloat16);
BIT16_PACK_A(sh, float16);
BIT16_PACK_B(sh, float16);
BIT16_PACK_A(i16, int16_t);
BIT16_PACK_B(i16, int16_t);

// 8 bit
BIT8_PACK_A(i8, int8_t);
BIT8_PACK_B(i8, int8_t);

// 4 bit
BIT4_PACK_A(i4, nibbles);
BIT4_PACK_B(i4, nibbles);

