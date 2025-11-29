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

// Common include/defines across microkernels

#include "blis.h"

#define PREFETCH1(x, y) __asm__ volatile ("dcbt %0, %1" : : "r" (x), "b" (y) : "memory");

#define LOAD_VECTORS \
        ca = (vec_t *) A0; \
        rb = (vec_t *) B0; 

typedef __vector float fv4sf_t;
typedef __vector double dv4sf_t;
typedef __vector int32_t iv4sf_t;
typedef __vector unsigned char vec_t;

#define SAVE_ACC(v_t, ACC, rs_c, j)                \
    __builtin_mma_disassemble_acc ( (void *) result, ACC);       \
    rowC = (v_t *) &C0[j];                        \
    rowC[0] = alpha_ * result[0] + beta_ * rowC[0];    \
    rowC = (v_t *) &C0[rs_c+j];                     \
    rowC[0] = alpha_ * result[1] + beta_ * rowC[0];    \
    rowC = (v_t *) &C0[2*rs_c+j];                   \
    rowC[0] = alpha_ * result[2] + beta_ * rowC[0] ;   \
    rowC = (v_t *) &C0[3*rs_c+j];                   \
    rowC[0] = alpha_ * result[3] + beta_ * rowC[0] ;

#define SAVE_ACC_bz(v_t, ACC, rs_c, j)                     \
    __builtin_mma_disassemble_acc ( (void *) result, ACC);     \
    rowC = (v_t *) &C0[j];                      \
    rowC[0] = alpha_ * result[0];                      \
    rowC = (v_t *) &C0[rs_c+j];                     \
    rowC[0] = alpha_ * result[1];                      \
    rowC = (v_t *) &C0[2*rs_c+j];                   \
    rowC[0] = alpha_ * result[2];                      \
    rowC = (v_t *) &C0[3*rs_c+j];                   \
    rowC[0] = alpha_ * result[3];
    