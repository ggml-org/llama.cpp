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

/* 
    Macro function template for creating BLIS GEMM kernels using the Goto method.

    This GEMM template assumes that the matrices are both not transposed.

    ch - kernel name prefix
    DTYPE_IN, DTYPE_OUT - datatypes of the input and output operands respectively
    NEW_PB - number of iterations of the innermost loop
    PACK_A, PACK_B - pack kernels names
    MICROKERNEL - microkernel function name
    K_MMA - number of outer products performed by an instruction
    MR, NR, MC, KC, NC - Cache blocking parameters
    B_ALIGN, A_ALIGN - Extra byte alignment for the pack matrix buffers
*/
#define GENERIC_GEMM( \
    ch, \
    DTYPE_IN, \
    DTYPE_OUT, \
    NEW_PB, \
    PACK_A, \
    PACK_B, \
    MICROKERNEL, \
    K_MMA, \
    MR, \
    NR, \
    MC, \
    KC, \
    NC, \
    B_ALIGN, \
    A_ALIGN \
) \
\
void GEMM_FUNC_NAME(ch) \
    ( \
        trans_t transa, \
        trans_t transb, \
        dim_t   m, \
        dim_t   n, \
        dim_t   k, \
        DTYPE_OUT*  alpha, \
        DTYPE_IN*  a, inc_t rsa, inc_t csa, \
        DTYPE_IN*  b, inc_t rsb, inc_t csb, \
        DTYPE_OUT*  beta, \
        DTYPE_OUT*  c, inc_t rsc, inc_t csc \
    ) \
{ \
    DTYPE_IN * restrict btilde_sys = ( DTYPE_IN *) aligned_alloc( P10_PG_SIZE, B_ALIGN + KC * NC * sizeof( DTYPE_IN ) ); \
    DTYPE_IN * restrict atilde_sys = ( DTYPE_IN *) aligned_alloc( P10_PG_SIZE, A_ALIGN + MC * KC * sizeof( DTYPE_IN ) ); \
    \
    DTYPE_IN * restrict btilde_usr = ( DTYPE_IN *)((char *)btilde_sys + B_ALIGN); \
    DTYPE_IN * restrict atilde_usr = ( DTYPE_IN *)((char *)atilde_sys + A_ALIGN); \
    \
    const int rstep_c = MC * rsc; \
    const int cstep_c = NC * csc; \
    \
    const int rstep_a = MC * rsa; \
    const int cstep_a = KC * csa; \
    \
    const int rstep_b = KC * rsb; \
    const int cstep_b = NC * csb; \
    \
    const int rstep_mt_c = MR * rsc; \
    const int cstep_mt_c = NR * csc; \
    \
    DTYPE_OUT * restrict cblock = c; \
    DTYPE_IN  * restrict bblock = b; \
    \
    for ( int jc=0; jc<n; jc+=NC ) \
    { \
        int jb = bli_min( NC, n-jc ); \
        DTYPE_IN * restrict apanel = a; \
        DTYPE_IN * restrict bpanel = bblock; \
        \
        for ( int pc=0; pc<k; pc+=KC ) \
        { \
            int pb = bli_min( KC, k-pc ); \
            PACK_B (NR, pb, jb, bpanel, rsb, csb, btilde_usr); \
            \
            int new_pb = NEW_PB; \
            const int a_ps = new_pb * (K_MMA * MR); \
            const int b_ps = new_pb * (K_MMA * NR); \
            \
            DTYPE_OUT * restrict cpanel = cblock; \
            DTYPE_IN  * restrict ablock = apanel; \
            \
            for ( int ic=0; ic<m; ic+=MC ) \
            { \
                int ib = bli_min( MC, m-ic ); \
                \
                /* pack_a (ib, pb, (uint32_t *) ablock, rsa, csa, (uint32_t *) atilde_usr ); */ \
                PACK_A (MR, ib, pb, ablock, rsa, csa, atilde_usr ); \
                \
                DTYPE_OUT * restrict cmicrotile_col = cpanel; \
                DTYPE_IN  * restrict bmicropanel = btilde_usr; \
                \
                for ( int jr=0; jr<jb; jr+=NR ) \
                { \
                    int jrb = bli_min( NR, jb-jr ); \
                    DTYPE_OUT * restrict cmicrotile = cmicrotile_col; \
                    DTYPE_IN  * restrict amicropanel = atilde_usr; \
                    \
                    for ( int ir=0; ir<ib; ir+=MR ) \
                    {    \
                        int irb = bli_min( MR, ib-ir ); \
                        \
                        MICROKERNEL \
                        ( \
                          irb, jrb, new_pb, alpha, amicropanel, bmicropanel, \
                          beta, cmicrotile, rsc, csc, NULL, NULL \
                        ); \
                        amicropanel += a_ps; \
                        cmicrotile += rstep_mt_c; \
                    } \
                    bmicropanel += b_ps; \
                    cmicrotile_col += cstep_mt_c; \
                } \
                ablock += rstep_a; \
                cpanel += rstep_c; \
            } \
            apanel += cstep_a; \
            bpanel += rstep_b; \
        } \
        cblock += cstep_c; \
        bblock += cstep_b; \
    } \
    \
    free(btilde_sys); \
    free(atilde_sys); \
} 

