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

// This avoids a known issue with GCC15 ("error: bp cannot be used in asm here", #845).
// Only check for version 15 since this may be fixed in 16, and also make sure the
// compiler isn't clang since it also confusingly defines __GNUC__
#if !defined(__clang__) && defined(__GNUC__) && __GNUC__ == 15
#pragma GCC optimize("-fno-tree-slp-vectorize")
#endif

#include "blis.h"

#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"

/*
   rrr:
     --------        ------        --------
     --------        ------        --------
     --------   +=   ------ ...    --------
     --------        ------        --------
     --------        ------            :
     --------        ------            :

   rcr:
     --------        | | | |       --------
     --------        | | | |       --------
     --------   +=   | | | | ...   --------
     --------        | | | |       --------
     --------        | | | |           :
     --------        | | | |           :

   Assumptions:
   - B is row-stored;
   - A is row- or column-stored;
   - m0 and n0 are at most MR and NR, respectively.
   Therefore, this (r)ow-preferential kernel is well-suited for contiguous
   (v)ector loads on B and single-element broadcasts from A.

   NOTE: These kernels explicitly support column-oriented IO, implemented
   via an in-register transpose. And thus they also support the crr and
   ccr cases, though only crr is ever utilized (because ccr is handled by
   transposing the operation and executing rcr, which does not incur the
   cost of the in-register transpose).

   crr:
     | | | | | | | |       ------        --------
     | | | | | | | |       ------        --------
     | | | | | | | |  +=   ------ ...    --------
     | | | | | | | |       ------        --------
     | | | | | | | |       ------            :
     | | | | | | | |       ------            :
*/
void bli_sgemmsup_rv_zen_asm_6x16m
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
    float *a = (float *)a0;
    float *b = (float *)b0;
    float *c = (float *)c0;

    uint64_t n_left = n0 % 16;

    /* For row storage format, kernel is re-written to */
    /* use mask load/store instruction                 */
    if ( n_left && (rs_c0 != 1))
    {
        float* restrict cij = (float *)c;
        float* restrict bj  = (float *)b;
        float* restrict ai  = (float *)a;
        /**************************************************************************/
        /* Mask load and store support is added for fringe cases                  */
        /* Fringe cases are the numbers which not multiple of xmm or ymm register */
        /* n_left : 15,14,13,11,10,9,7,6,5,3                                      */
        /* When mask register values are set, load/store is performed             */
        /* When mask register values are not set, load/store is not performed     */
        /*Elements: 1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16*/
        /*n0=16   : -----------ymm---------------  -----------ymm---------------- */
        /*n0=15   : -----------ymm---------------   -1  -1  -1  -1  -1  -1  -1  0 */
        /*n0=14   : -----------ymm---------------   -1  -1  -1  -1  -1  -1   0  0 */
        /*n0=9    : -----------ymm---------------   -1   0   0   0   0   0   0  0 */
        /*n0=8    : -----------ymm---------------   -----------Not used---------  */
        /*n0=7    : -1  -1  -1  -1  -1  -1  -1  0   -----------Not used---------  */
        /*n0=3    : -1  -1  -1   0   0   0   0  0   -----------Not used---------  */
        /*Same code can be resued for multiple n_left by just varing mask register*/
        /*We will be able to perform complete operation of tile with this approach*/
        /**************************************************************************/
        switch(n_left)
        {
          /*Fringe cases*/
            case 15: case 14: case 13:
            case 11: case 10: case 9:
            {
                const dim_t nr_cur = n_left;
                /**********************************************/
                /* These case is executed when nleft - 9 to 15*/
                /* 16 Elements in col order                   */
                /*   ---YMM REG-----    ---YMM Mask Reg---    */
                /*   0 1 2 3 4 5 6 7   8 9 10 11 12 13 14 15  */
                /*15:0 1 2 3 4 5 6 7   8 9 10 11 12 13 14 x   */
                /*14:0 1 2 3 4 5 6 7   8 9 10 11 12 13  x x   */
                /*11:0 1 2 3 4 5 6 7   8 9 10  x  x  x  x x   */
                /* and so on                                  */
                /**********************************************/
                bli_sgemmsup_rv_zen_asm_6x16m_mask
                (
                  conja, conjb, m0, nr_cur, k0,
                  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
                  beta, cij, rs_c0, cs_c0, data, cntx
                );
                break;
            }
            case 7: case 6: case 5:
            {
               /***********************************************/
                /* These case is executed when nleft - 5 to 7 */
                /* 8 Elements in col order                    */
                /*       YMM Mask REG                         */
                /*      0 1 2 3 4 5 6 7                       */
                /*7:    0 1 2 3 4 5 6 x                       */
                /*6:    0 1 2 3 4 5 x x                       */
                /*5:    0 1 2 3 4 x x x                       */
                /**********************************************/
                const dim_t nr_cur = n_left;

                bli_sgemmsup_rv_zen_asm_6x8m_mask
                (
                  conja, conjb, m0, nr_cur, k0,
                  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
                  beta, cij, rs_c0, cs_c0, data, cntx
                );
                break;
            }
            case 3: case 1:
            {
               /***********************************************/
                /* These case is executed when nleft - 3/1    */
                /* 4 Elements in col order                    */
                /*      XMM Mask REG                         */
                /*      0 1 2 3                               */
                /*3:    0 1 2 x                               */
                /*1:    0 x x x                               */
                /**********************************************/
                const dim_t nr_cur = n_left;
                bli_sgemmsup_rv_zen_asm_6x4m_mask
                (
                  conja, conjb, m0, nr_cur, k0,
                  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
                  beta, cij, rs_c0, cs_c0, data, cntx
                );
                break;
            }
            /*Non-Fringe cases*/
            case 12:
            {
              #if 0
                const dim_t nr_cur = 12;
                bli_sgemmsup_rv_haswell_asm_6x12m
                (
                  conja, conjb, m0, nr_cur, k0,
                  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
                  beta, cij, rs_c0, cs_c0, data, cntx
                );
                break;
              #endif

              dim_t nr_cur = 8;

              bli_sgemmsup_rv_zen_asm_6x8m
              (
                conja, conjb, m0, nr_cur, k0,
                alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
                beta, cij, rs_c0, cs_c0, data, cntx
              );
              cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;

              nr_cur = 4;
              bli_sgemmsup_rv_zen_asm_6x4m
              (
                conja, conjb, m0, nr_cur, k0,
                alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
                beta, cij, rs_c0, cs_c0, data, cntx
              );
              break;

            }
            case 8:
            {
                const dim_t nr_cur = 8;

                bli_sgemmsup_rv_zen_asm_6x8m
                (
                  conja, conjb, m0, nr_cur, k0,
                  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
                  beta, cij, rs_c0, cs_c0, data, cntx
                );
                break;
            }
            case 4:
            {
                const dim_t nr_cur = 4;

                bli_sgemmsup_rv_zen_asm_6x4m
                (
                  conja, conjb, m0, nr_cur, k0,
                  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
                  beta, cij, rs_c0, cs_c0, data, cntx
                );
                break;
            }
            case 2:
            {
                const dim_t nr_cur = 2;

                bli_sgemmsup_rv_zen_asm_6x2m
                (
                  conja, conjb, m0, nr_cur, k0,
                  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
                  beta, cij, rs_c0, cs_c0, data, cntx
                );
                break;
            }
        default:
        break;
        }
        return;
    }

    // First check whether this is a edge case in the n dimension. If so,
    // dispatch other 6x?m kernels, as needed.
    if (n_left )
    {
        float*  cij = (float *)c;
        float*  bj  = (float *)b;
        float*  ai  = (float *)a;

        if ( 8 <= n_left )
        {
            const dim_t nr_cur = 8;

            bli_sgemmsup_rv_zen_asm_6x8m
            (
              conja, conjb, m0, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
            cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
        }

        if ( 4 <= n_left )
        {
            const dim_t nr_cur = 4;

            bli_sgemmsup_rv_zen_asm_6x4m
            (
              conja, conjb, m0, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
            cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
        }

        if ( 2 <= n_left )
        {
            const dim_t nr_cur = 2;

            bli_sgemmsup_rv_zen_asm_6x2m
            (
              conja, conjb, m0, nr_cur, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
              beta, cij, rs_c0, cs_c0, data, cntx
            );
            cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
        }

        if ( 1 == n_left )
        {
            dim_t ps_a0 = bli_auxinfo_ps_a( data );
            if ( ps_a0 == 6 * rs_a0 )
            {
                bli_sgemv_ex
                (
                    BLIS_NO_TRANSPOSE, conjb, m0, k0,
                    alpha, ai, rs_a0, cs_a0, bj, rs_b0,
                    beta, cij, rs_c0, cntx, NULL
                );
            }
            else
            {
                const dim_t mr = 6;

                // Since A is packed into row panels, we must use a loop over
                // gemv.
                dim_t m_iter = ( m0 + mr - 1 ) / mr;
                dim_t m_left =   m0            % mr;

                float* restrict ai_ii  = ai;
                float* restrict cij_ii = cij;

                for ( dim_t ii = 0; ii < m_iter; ii += 1 )
                {
                    dim_t mr_cur = ( bli_is_not_edge_f( ii, m_iter, m_left )
                                     ? mr : m_left );

                    bli_sgemv_ex
                    (
                      BLIS_NO_TRANSPOSE, conjb, mr_cur, k0,
                      alpha, ai_ii, rs_a0, cs_a0, bj, rs_b0,
                      beta, cij_ii, rs_c0, cntx, NULL
                    );
                    cij_ii += mr*rs_c0; ai_ii += ps_a0;
                }
            }
        }

        return;
    }

    //void*    a_next = bli_auxinfo_next_a( data );
    //void*    b_next = bli_auxinfo_next_b( data );

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.


    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a( data );
    uint64_t ps_a4  = ps_a * sizeof( float );

    if ( m_iter == 0 ) goto consider_edge_cases;

    // -------------------------------------------------------------------------
    begin_asm()

    mov(var(a), r14)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(dt)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(dt)

    lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
    lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(dt)
                                       // NOTE: We cannot pre-load elements of a or b
                                       // because it could eventually, in the last
                                       // unrolled iter or the cleanup loop, result
                                       // in reading beyond the bounds allocated mem
                                       // (the likely result: a segmentation fault).

    mov(var(c), r12)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(dt)

    // During preamble and loops:
    // r12 = rcx = c
    // r14 = rax = a
    // read rbx from var(b) near beginning of loop
    // r11 = m dim index ii

    mov(var(m_iter), r11)              // ii = m_iter;

    label(.SLOOP6X16I)                 // LOOP OVER ii = [ m_iter ... 1 0 ]

    vxorps(ymm4,  ymm4,  ymm4)
    vxorps(ymm5,  ymm5,  ymm5)
    vxorps(ymm6,  ymm6,  ymm6)
    vxorps(ymm7,  ymm7,  ymm7)
    vxorps(ymm8,  ymm8,  ymm8)
    vxorps(ymm9,  ymm9,  ymm9)
    vxorps(ymm10, ymm10, ymm10)
    vxorps(ymm11, ymm11, ymm11)
    vxorps(ymm12, ymm12, ymm12)
    vxorps(ymm13, ymm13, ymm13)
    vxorps(ymm14, ymm14, ymm14)
    vxorps(ymm15, ymm15, ymm15)

    mov(var(b), rbx)                   // load address of b.
    //mov(r12, rcx)                    // reset rcx to current utile of c.
    mov(r14, rax)                      // reset rax to current upanel of a.

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored pre-fetching on c // not used

    lea(mem(r12, rdi, 2), rdx)         //
    lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
    prefetch(0, mem(r12, 7*8))         // prefetch c + 0*rs_c
    prefetch(0, mem(r12, rdi, 1, 7*8)) // prefetch c + 1*rs_c
    prefetch(0, mem(r12, rdi, 2, 7*8)) // prefetch c + 2*rs_c
    prefetch(0, mem(rdx, 7*8))         // prefetch c + 3*rs_c
    prefetch(0, mem(rdx, rdi, 1, 7*8)) // prefetch c + 4*rs_c
    prefetch(0, mem(rdx, rdi, 2, 7*8)) // prefetch c + 5*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of pre-fetching c
    label(.SCOLPFETCH)                 // column-stored pre-fetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(dt)
    lea(mem(r12, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(r12, 5*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(r12, rsi, 1, 5*8)) // prefetch c + 1*cs_c
    prefetch(0, mem(r12, rsi, 2, 5*8)) // prefetch c + 2*cs_c
    prefetch(0, mem(rdx, 5*8))        // prefetch c + 3*cs_c
    prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 4*cs_c
    prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 5*cs_c
    lea(mem(rdx, rsi, 2), rdx)         // rdx = c + 5*cs_c;
    prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 6*cs_c
    prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 7*cs_c

    label(.SPOSTPFETCH)                // done prefetching c

    lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
    lea(mem(rax, r8,  4), rdx)         // use rdx for pre-fetching lines
    lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.

    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                       // contains the k_left loop.

    label(.SLOOPKITER)                 // MAIN LOOP

    // ---------------------------------- iteration 0
    prefetch(0, mem(rdx, 5*8))

    vmovups(mem(rbx,  0*32), ymm0)
    vmovups(mem(rbx,  1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vfmadd231ps(ymm0, ymm3, ymm6)
    vfmadd231ps(ymm1, ymm3, ymm7)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm1, ymm2, ymm9)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)

    vbroadcastss(mem(rax, r8,  4), ymm2)
    vbroadcastss(mem(rax, r15, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm12)
    vfmadd231ps(ymm1, ymm2, ymm13)
    vfmadd231ps(ymm0, ymm3, ymm14)
    vfmadd231ps(ymm1, ymm3, ymm15)

    // ---------------------------------- iteration 1
    prefetch(0, mem(rdx, r9, 1, 5*8))

    vmovups(mem(rbx,  0*32), ymm0)
    vmovups(mem(rbx,  1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vfmadd231ps(ymm0, ymm3, ymm6)
    vfmadd231ps(ymm1, ymm3, ymm7)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm1, ymm2, ymm9)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)

    vbroadcastss(mem(rax, r8,  4), ymm2)
    vbroadcastss(mem(rax, r15, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm12)
    vfmadd231ps(ymm1, ymm2, ymm13)
    vfmadd231ps(ymm0, ymm3, ymm14)
    vfmadd231ps(ymm1, ymm3, ymm15)

    // ---------------------------------- iteration 2
    prefetch(0, mem(rdx, r9, 2, 5*8))

    vmovups(mem(rbx,  0*32), ymm0)
    vmovups(mem(rbx,  1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vfmadd231ps(ymm0, ymm3, ymm6)
    vfmadd231ps(ymm1, ymm3, ymm7)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm1, ymm2, ymm9)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)

    vbroadcastss(mem(rax, r8,  4), ymm2)
    vbroadcastss(mem(rax, r15, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm12)
    vfmadd231ps(ymm1, ymm2, ymm13)
    vfmadd231ps(ymm0, ymm3, ymm14)
    vfmadd231ps(ymm1, ymm3, ymm15)

    // ---------------------------------- iteration 3
    prefetch(0, mem(rdx, rcx, 1, 5*8))
    lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;

    vmovups(mem(rbx, 0*32), ymm0)
    vmovups(mem(rbx, 1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vfmadd231ps(ymm0, ymm3, ymm6)
    vfmadd231ps(ymm1, ymm3, ymm7)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm1, ymm2, ymm9)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)

    vbroadcastss(mem(rax, r8,  4), ymm2)
    vbroadcastss(mem(rax, r15, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm12)
    vfmadd231ps(ymm1, ymm2, ymm13)
    vfmadd231ps(ymm0, ymm3, ymm14)
    vfmadd231ps(ymm1, ymm3, ymm15)

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.

    label(.SCONSIDKLEFT)

    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                       // else, we prepare to enter k_left loop.

    label(.SLOOPKLEFT)                 // EDGE LOOP

    vmovups(mem(rbx,  0*32), ymm0)
    vmovups(mem(rbx,  1*32), ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vfmadd231ps(ymm0, ymm3, ymm6)
    vfmadd231ps(ymm1, ymm3, ymm7)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm1, ymm2, ymm9)
    vfmadd231ps(ymm0, ymm3, ymm10)
    vfmadd231ps(ymm1, ymm3, ymm11)

    vbroadcastss(mem(rax, r8,  4), ymm2)
    vbroadcastss(mem(rax, r15, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm12)
    vfmadd231ps(ymm1, ymm2, ymm13)
    vfmadd231ps(ymm0, ymm3, ymm14)
    vfmadd231ps(ymm1, ymm3, ymm15)

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.

    label(.SPOSTACCUM)

    mov(r12, rcx)                      // reset rcx to current utile of c.
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), ymm3)       // load beta and duplicate

    vmulps(ymm0, ymm4, ymm4)           // scale by alpha
    vmulps(ymm0, ymm5, ymm5)
    vmulps(ymm0, ymm6, ymm6)
    vmulps(ymm0, ymm7, ymm7)
    vmulps(ymm0, ymm8, ymm8)
    vmulps(ymm0, ymm9, ymm9)
    vmulps(ymm0, ymm10, ymm10)
    vmulps(ymm0, ymm11, ymm11)
    vmulps(ymm0, ymm12, ymm12)
    vmulps(ymm0, ymm13, ymm13)
    vmulps(ymm0, ymm14, ymm14)
    vmulps(ymm0, ymm15, ymm15)

    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(dt)

    lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;
    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

                                       // now avoid loading C if beta == 0
    vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORED)                    // jump to column storage case

    label(.SROWSTORED)

    vfmadd231ps(mem(rcx), ymm3, ymm4)
    vmovups(ymm4, mem(rcx))

    vfmadd231ps(mem(rcx, rsi, 8), ymm3, ymm5)
    vmovups(ymm5, mem(rcx, rsi, 8))
    add(rdi, rcx)


    vfmadd231ps(mem(rcx), ymm3, ymm6)
    vmovups(ymm6, mem(rcx))

    vfmadd231ps(mem(rcx, rsi, 8), ymm3, ymm7)
    vmovups(ymm7, mem(rcx, rsi, 8))
    add(rdi, rcx)


    vfmadd231ps(mem(rcx), ymm3, ymm8)
    vmovups(ymm8, mem(rcx))

    vfmadd231ps(mem(rcx, rsi, 8), ymm3, ymm9)
    vmovups(ymm9, mem(rcx, rsi, 8))
    add(rdi, rcx)


    vfmadd231ps(mem(rcx), ymm3, ymm10)
    vmovups(ymm10, mem(rcx))

    vfmadd231ps(mem(rcx, rsi, 8), ymm3, ymm11)
    vmovups(ymm11, mem(rcx, rsi, 8))
    add(rdi, rcx)


    vfmadd231ps(mem(rcx), ymm3, ymm12)
    vmovups(ymm12, mem(rcx))

    vfmadd231ps(mem(rcx, rsi, 8), ymm3, ymm13)
    vmovups(ymm13, mem(rcx, rsi, 8))
    add(rdi, rcx)


    vfmadd231ps(mem(rcx), ymm3, ymm14)
    vmovups(ymm14, mem(rcx))

    vfmadd231ps(mem(rcx, rsi, 8), ymm3, ymm15)
    vmovups(ymm15, mem(rcx, rsi, 8))
    //add(rdi, rcx)


    jmp(.SDONE)                        // jump to end.


    label(.SCOLSTORED)

  /*|-----------------|              |-----|----|
    |         |       |              | 8x4 | 8x2|
    |    4x8  |   4x8 |              |     |    |
    |         |       |              |-----|----|
    |-----------------|              | 8x4 | 8x2|
    |    2x8  |   2x8 |              |     |    |
    |------------------              |----------|*/

    /****6x16 tile is transposed and saved in col major as 6x16*****/
    /****top left tile 4x8 transposed to top left tile 8x4**********/
    vunpcklps(ymm6, ymm4, ymm0)//a0b0a1b1 a4b4a5b5
    vunpcklps(ymm10, ymm8, ymm1)//c0d0c1d1 c4d4c5d5
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)//a1b1c0d0 a5b5c4d4
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)//a0b0c0d0 a4b4c4d4
    vblendps(imm(0x33), ymm2, ymm1, ymm1)//a1b1c1d1 a5b5c5d5

    vextractf128(imm(0x1), ymm0, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm0)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma04..gamma34 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c

    vextractf128(imm(0x1), ymm1, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm1)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma01..gamma31 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma05..gamma35 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c

    vunpckhps(ymm6, ymm4, ymm0)
    vunpckhps(ymm10, ymm8, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)

    vextractf128(imm(0x1), ymm0, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm0)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma02..gamma32 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma06..gamma36 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c

    vextractf128(imm(0x1), ymm1, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm1)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma03..gamma33 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma07..gamma37 )

    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c
    lea(mem(rcx, rsi, 4), rcx) // rcx += 4*cs_c

    /***bottom left tile - 2x8 is transposed to top right tile 8x2**********/
    vunpcklps(ymm14, ymm12, ymm0)
    vextractf128(imm(0x1), ymm0, xmm2)
    vmovlpd(mem(rdx), xmm1, xmm1)
    vmovhpd(mem(rdx, rsi, 1), xmm1, xmm1)
    vfmadd231ps(xmm1, xmm3, xmm0)
    vmovlpd(xmm0, mem(rdx)) // store ( gamma40..gamma50 )
    vmovhpd(xmm0, mem(rdx, rsi, 1)) // store ( gamma41..gamma51 )
    lea(mem(rdx, rsi, 4), rax) // rax += 4*cs_c

    vmovlpd(mem(rax), xmm1, xmm1)
    vmovhpd(mem(rax, rsi, 1), xmm1, xmm1)
    vfmadd231ps(xmm1, xmm3, xmm2)
    vmovlpd(xmm2, mem(rax)) // store ( gamma44..gamma54 )
    vmovhpd(xmm2, mem(rax, rsi, 1)) // store ( gamma45..gamma55 )
    lea(mem(rdx, rsi, 2), rdx) // rdx += 2*cs_c

    vunpckhps(ymm14, ymm12, ymm0)
    vextractf128(imm(0x1), ymm0, xmm2)
    vmovlpd(mem(rdx), xmm1, xmm1)
    vmovhpd(mem(rdx, rsi, 1), xmm1, xmm1)
    vfmadd231ps(xmm1, xmm3, xmm0)
    vmovlpd(xmm0, mem(rdx)) // store ( gamma42..gamma52 )
    vmovhpd(xmm0, mem(rdx, rsi, 1)) // store ( gamma43..gamma53 )
    lea(mem(rdx, rsi, 4), rdx) // rdx += 4*cs_c
    vmovlpd(mem(rdx), xmm1, xmm1)
    vmovhpd(mem(rdx, rsi, 1), xmm1, xmm1)
    vfmadd231ps(xmm1, xmm3, xmm2)
    vmovlpd(xmm2, mem(rdx)) // store ( gamma46..gamma56 )
    vmovhpd(xmm2, mem(rdx, rsi, 1)) // store ( gamma47..gamma57 )

    lea(mem(rdx, rsi, 2), rdx) // rdx += 2*cs_c

    /***top right tile  4x8 is transposed to bottom left tile 8x4**********/
    vunpcklps(ymm7, ymm5, ymm0)
    vunpcklps(ymm11, ymm9, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)

    vextractf128(imm(0x1), ymm0, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm0)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma04..gamma34 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c

    vextractf128(imm(0x1), ymm1, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm1)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma01..gamma31 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma05..gamma35 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c

    vunpckhps(ymm7, ymm5, ymm0)
    vunpckhps(ymm11, ymm9, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)

    vextractf128(imm(0x1), ymm0, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm0)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma02..gamma32 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma06..gamma36 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += cs_c

    vextractf128(imm(0x1), ymm1, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm1)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma03..gamma33 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma07..gamma37 )

    //lea(mem(rcx, rsi, 8), rcx) // rcx += 8*cs_c
    /*** bottom right 2x8 is transposed to bottom right tile 8x2*******/
    vunpcklps(ymm15, ymm13, ymm0)
    vextractf128(imm(0x1), ymm0, xmm2)
    vmovlpd(mem(rdx), xmm1, xmm1)
    vmovhpd(mem(rdx, rsi, 1), xmm1, xmm1)
    vfmadd231ps(xmm1, xmm3, xmm0)
    vmovlpd(xmm0, mem(rdx)) // store ( gamma40..gamma50 )
    vmovhpd(xmm0, mem(rdx, rsi, 1)) // store ( gamma41..gamma51 )
    lea(mem(rdx, rsi, 4), rax) // rax += 4*cs_c

    vmovlpd(mem(rax), xmm1, xmm1)
    vmovhpd(mem(rax, rsi, 1), xmm1, xmm1)
    vfmadd231ps(xmm1, xmm3, xmm2)
    vmovlpd(xmm2, mem(rax)) // store ( gamma44..gamma54 )
    vmovhpd(xmm2, mem(rax, rsi, 1)) // store ( gamma45..gamma55 )
    lea(mem(rdx, rsi, 2), rdx) // rdx += 2*cs_c

    vunpckhps(ymm15, ymm13, ymm0)
    vextractf128(imm(0x1), ymm0, xmm2)
    vmovlpd(mem(rdx), xmm1, xmm1)
    vmovhpd(mem(rdx, rsi, 1), xmm1, xmm1)
    vfmadd231ps(xmm1, xmm3, xmm0)
    vmovlpd(xmm0, mem(rdx)) // store ( gamma42..gamma52 )
    vmovhpd(xmm0, mem(rdx, rsi, 1)) // store ( gamma43..gamma53 )
    lea(mem(rdx, rsi, 4), rdx) // rdx += 4*cs_c
    vmovlpd(mem(rdx), xmm1, xmm1)
    vmovhpd(mem(rdx, rsi, 1), xmm1, xmm1)
    vfmadd231ps(xmm1, xmm3, xmm2)
    vmovlpd(xmm2, mem(rdx)) // store ( gamma46..gamma56 )
    vmovhpd(xmm2, mem(rdx, rsi, 1)) // store ( gamma47..gamma57 )

    jmp(.SDONE)                        // jump to end.

    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLSTORBZ)                    // jump to column storage case


    label(.SROWSTORBZ)

    vmovups(ymm4, mem(rcx))
    vmovups(ymm5, mem(rcx, rsi, 8))
    add(rdi, rcx)


    vmovups(ymm6, mem(rcx))
    vmovups(ymm7, mem(rcx, rsi, 8))
    add(rdi, rcx)


    vmovups(ymm8, mem(rcx))
    vmovups(ymm9, mem(rcx, rsi, 8))
    add(rdi, rcx)


    vmovups(ymm10, mem(rcx))
    vmovups(ymm11, mem(rcx, rsi, 8))
    add(rdi, rcx)


    vmovups(ymm12, mem(rcx))
    vmovups(ymm13, mem(rcx, rsi, 8))
    add(rdi, rcx)


    vmovups(ymm14, mem(rcx))
    vmovups(ymm15, mem(rcx, rsi, 8))
    //add(rdi, rcx)

    jmp(.SDONE)                        // jump to end.


    label(.SCOLSTORBZ)
    /****6x16 tile going to save into 16x6 tile in C*****/
    /******************top left tile 8x4***************************/
    vunpcklps(ymm6, ymm4, ymm0)
    vunpcklps(ymm10, ymm8, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)

    vextractf128(imm(0x1), ymm0, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma04..gamma34 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c
    vextractf128(imm(0x1), ymm1, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma01..gamma31 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma05..gamma35 )

    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c
    vunpckhps(ymm6, ymm4, ymm0)
    vunpckhps(ymm10, ymm8, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)

    vextractf128(imm(0x1), ymm0, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma02..gamma32 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma06..gamma36 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c
    vextractf128(imm(0x1), ymm1, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma03..gamma33 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma07..gamma37 )

    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c
    lea(mem(rcx, rsi, 4), rcx) // rcx += 4*cs_c
    /******************top right tile 8x2***************************/
    vunpcklps(ymm14, ymm12, ymm0)
    vextractf128(imm(0x1), ymm0, xmm2)
    vmovlpd(xmm0, mem(rdx)) // store ( gamma40..gamma50 )
    vmovlpd(xmm2, mem(rdx, rsi, 4)) // store ( gamma44..gamma54 )
    lea(mem(rdx, rsi, 1), rdx)
    vmovhpd(xmm0, mem(rdx)) // store ( gamma41..gamma51 )
    vmovhpd(xmm2, mem(rdx, rsi, 4)) // store ( gamma45..gamma55 )
    lea(mem(rdx, rsi, 1), rdx)

    vunpckhps(ymm14, ymm12, ymm0)
    vextractf128(imm(0x1), ymm0, xmm2)
    vmovlpd(xmm0, mem(rdx)) // store ( gamma42..gamma52 )
    vmovlpd(xmm2, mem(rdx, rsi, 4)) // store ( gamma46..gamma56 )
    lea(mem(rdx, rsi, 1), rdx)
    vmovhpd(xmm0, mem(rdx)) // store ( gamma43..gamma53 )
    vmovhpd(xmm2, mem(rdx, rsi, 4)) // store ( gamma47..gamma57 )
    lea(mem(rdx, rsi, 1), rdx)
    lea(mem(rdx, rsi, 4), rdx) // rdx += 8*cs_c

    /******************bottom left tile 8x4***************************/
    vunpcklps(ymm7, ymm5, ymm0)
    vunpcklps(ymm11, ymm9, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)

    vextractf128(imm(0x1), ymm0, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma04..gamma34 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c
    vextractf128(imm(0x1), ymm1, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma01..gamma31 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma05..gamma35 )

    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c
    vunpckhps(ymm7, ymm5, ymm0)
    vunpckhps(ymm11, ymm9, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)

    vextractf128(imm(0x1), ymm0, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma02..gamma32 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma06..gamma36 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c
    vextractf128(imm(0x1), ymm1, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma03..gamma33 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma07..gamma37 )

    /******************bottom right tile 8x2***************************/
    vunpcklps(ymm15, ymm13, ymm0)
    vextractf128(imm(0x1), ymm0, xmm2)
    vmovlpd(xmm0, mem(rdx)) // store ( gamma40..gamma50 )
    vmovlpd(xmm2, mem(rdx, rsi, 4)) // store ( gamma44..gamma54 )
    lea(mem(rdx, rsi, 1), rdx)
    vmovhpd(xmm0, mem(rdx)) // store ( gamma41..gamma51 )
    vmovhpd(xmm2, mem(rdx, rsi, 4)) // store ( gamma45..gamma55 )
    lea(mem(rdx, rsi, 1), rdx)

    vunpckhps(ymm15, ymm13, ymm0)
    vextractf128(imm(0x1), ymm0, xmm2)
    vmovlpd(xmm0, mem(rdx)) // store ( gamma42..gamma52 )
    vmovlpd(xmm2, mem(rdx, rsi, 4)) // store ( gamma46..gamma56 )
    lea(mem(rdx, rsi, 1), rdx)
    vmovhpd(xmm0, mem(rdx)) // store ( gamma43..gamma53 )
    vmovhpd(xmm2, mem(rdx, rsi, 4)) // store ( gamma47..gamma57 )

    label(.SDONE)

    lea(mem(r12, rdi, 4), r12)         //
    lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

    //lea(mem(r14, r8,  4), r14)         //
    //lea(mem(r14, r8,  2), r14)         // a_ii = r14 += 6*rs_a
    mov(var(ps_a4), rax)               // load ps_a4
    lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a4

    dec(r11)                           // ii -= 1;
    jne(.SLOOP6X16I)                    // iterate again if ii != 0.

    label(.SRETURN)

    end_asm(
    : // output operands (none)
    : // input operands
      [m_iter] "m" (m_iter),
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a4]  "m" (ps_a4),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3",
      "xmm4", "xmm5", "xmm6", "xmm7",
      "xmm8", "xmm9", "xmm10", "xmm11",
      "xmm12", "xmm13", "xmm14", "xmm15",
      "ymm0", "ymm1", "ymm2", "ymm3",
      "ymm4", "ymm5", "ymm6", "ymm7",
      "ymm8", "ymm9", "ymm10", "ymm11",
      "ymm12", "ymm13", "ymm14", "ymm15",
      "memory"
    )

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      nr_cur = 16;
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float*  restrict cij = (float *)c + i_edge*rs_c;
        float*  restrict ai  = (float *)a + m_iter*ps_a;
        float*  restrict bj  = (float *)b;

        gemmsup_ker_ft ker_fps[6] =
        {
          NULL,
          bli_sgemmsup_rv_zen_asm_1x16,
          bli_sgemmsup_rv_zen_asm_2x16,
          bli_sgemmsup_rv_zen_asm_3x16,
          bli_sgemmsup_rv_zen_asm_4x16,
          bli_sgemmsup_rv_zen_asm_5x16
        };

        gemmsup_ker_ft ker_fp = ker_fps[ m_left ];

        ker_fp
        (
          conja, conjb, m_left, nr_cur, k0,
          alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
          beta, cij, rs_c0, cs_c0, data, cntx
        );
        return;

    }
}

void bli_sgemmsup_rv_zen_asm_6x8m
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
    //void*    a_next = bli_auxinfo_next_a( data );
    //void*    b_next = bli_auxinfo_next_b( data );

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    float *a = (float *)a0;
    float *b = (float *)b0;
    float *c = (float *)c0;

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a( data );
    uint64_t ps_a4  = ps_a * sizeof( float );

    if ( m_iter == 0 ) goto consider_edge_cases;

    // -------------------------------------------------------------------------
    begin_asm()

    mov(var(a), r14)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

    lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
    lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)

                                       // NOTE: We cannot pre-load elements of a or b
                                       // because it could eventually, in the last
                                       // unrolled iter or the cleanup loop, result
                                       // in reading beyond the bounds allocated mem
                                       // (the likely result: a segmentation fault).

    mov(var(c), r12)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


    // During preamble and loops:
    // r12 = rcx = c
    // r14 = rax = a
    // read rbx from var(b) near beginning of loop
    // r11 = m dim index ii

    mov(var(m_iter), r11)              // ii = m_iter;

    label(.SLOOP6X8I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]

                                       // skylake can execute 3 vxorpd ipc with
                                       // a latency of 1 cycle, while vzeroall
                                       // has a latency of 12 cycles.
    vxorps(ymm1,  ymm1,  ymm1)         // zero ymm1 since we only use the lower
    vxorps(ymm4,  ymm4,  ymm4)         // half (xmm1), and nans/infs may slow us down.
    vxorps(ymm6,  ymm6,  ymm6)
    vxorps(ymm8,  ymm8,  ymm8)
    vxorps(ymm10, ymm10, ymm10)
    vxorps(ymm12, ymm12, ymm12)
    vxorps(ymm14, ymm14, ymm14)

    mov(var(b), rbx)                   // load address of b.
    mov(r14, rax)                      // reset rax to current upanel of a.

    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    lea(mem(r12, rdi, 2), rdx)         //
    lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
    prefetch(0, mem(r12, 5*8))         // prefetch c + 0*rs_c
    prefetch(0, mem(r12, rdi, 1, 5*8)) // prefetch c + 1*rs_c
    prefetch(0, mem(r12, rdi, 2, 5*8)) // prefetch c + 2*rs_c
    prefetch(0, mem(rdx, 5*8))         // prefetch c + 3*rs_c
    prefetch(0, mem(rdx, rdi, 1, 5*8)) // prefetch c + 4*rs_c
    prefetch(0, mem(rdx, rdi, 2, 5*8)) // prefetch c + 5*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(r12, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(r12, 5*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(r12, rsi, 1, 5*8)) // prefetch c + 1*cs_c
    prefetch(0, mem(r12, rsi, 2, 5*8)) // prefetch c + 2*cs_c
    prefetch(0, mem(rdx, 5*8))         // prefetch c + 3*cs_c
    prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 4*cs_c
    prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 5*cs_c

    label(.SPOSTPFETCH)                // done prefetching c


    lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
    lea(mem(rax, r8,  4), rdx)         // use rdx for prefetching lines
    lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.


    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                       // contains the k_left loop.


    label(.SLOOPKITER)                 // MAIN LOOP


    // ---------------------------------- iteration 0
    prefetch(0, mem(rdx, 5*8))

    vmovups(mem(rbx), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm0, ymm3, ymm10)

    vbroadcastss(mem(rax, r8,  4), ymm2)
    vbroadcastss(mem(rax, r15, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm12)
    vfmadd231ps(ymm0, ymm3, ymm14)

    // ---------------------------------- iteration 1
    prefetch(0, mem(rdx, r9, 1, 5*8))

    vmovups(mem(rbx), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm0, ymm3, ymm10)

    vbroadcastss(mem(rax, r8,  4), ymm2)
    vbroadcastss(mem(rax, r15, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm12)
    vfmadd231ps(ymm0, ymm3, ymm14)

    // ---------------------------------- iteration 2
    prefetch(0, mem(rdx, r9, 2, 5*8))

    vmovups(mem(rbx), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm0, ymm3, ymm10)

    vbroadcastss(mem(rax, r8,  4), ymm2)
    vbroadcastss(mem(rax, r15, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm12)
    vfmadd231ps(ymm0, ymm3, ymm14)

    // ---------------------------------- iteration 3
    prefetch(0, mem(rdx, rcx, 1, 5*8))

    vmovups(mem(rbx), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm0, ymm3, ymm10)

    vbroadcastss(mem(rax, r8,  4), ymm2)
    vbroadcastss(mem(rax, r15, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm12)
    vfmadd231ps(ymm0, ymm3, ymm14)


    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.


    label(.SCONSIDKLEFT)

    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                       // else, we prepare to enter k_left loop.

    label(.SLOOPKLEFT)                 // EDGE LOOP

    vmovups(mem(rbx), ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vbroadcastss(mem(rax, r8,  1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm0, ymm3, ymm6)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vbroadcastss(mem(rax, r13, 1), ymm3)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm0, ymm3, ymm10)

    vbroadcastss(mem(rax, r8,  4), ymm2)
    vbroadcastss(mem(rax, r15, 1), ymm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm12)
    vfmadd231ps(ymm0, ymm3, ymm14)


    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.


    label(.SPOSTACCUM)

    mov(r12, rcx)                      // reset rcx to current utile of c.
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), ymm3)       // load beta and duplicate

    vmulps(ymm0, ymm4, ymm4)           // scale by alpha
    vmulps(ymm0, ymm6, ymm6)
    vmulps(ymm0, ymm8, ymm8)
    vmulps(ymm0, ymm10, ymm10)
    vmulps(ymm0, ymm12, ymm12)
    vmulps(ymm0, ymm14, ymm14)

    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

    lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;
    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

                                       // now avoid loading C if beta == 0

    vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case


    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLSTORED)                    // jump to column storage case


    label(.SROWSTORED)

    vfmadd231ps(mem(rcx), ymm3, ymm4)
    vmovups(ymm4, mem(rcx))
    add(rdi, rcx)
    vfmadd231ps(mem(rcx), ymm3, ymm6)
    vmovups(ymm6, mem(rcx))
    add(rdi, rcx)
    vfmadd231ps(mem(rcx), ymm3, ymm8)
    vmovups(ymm8, mem(rcx))
    add(rdi, rcx)
    vfmadd231ps(mem(rcx), ymm3, ymm10)
    vmovups(ymm10, mem(rcx))
    add(rdi, rcx)
    vfmadd231ps(mem(rcx), ymm3, ymm12)
    vmovups(ymm12, mem(rcx))
    add(rdi, rcx)
    vfmadd231ps(mem(rcx), ymm3, ymm14)
    vmovups(ymm14, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORED)

    /****6x8 tile is transposed and saved in col major as 8x6*****/
    vunpcklps(ymm6, ymm4, ymm0)
    vunpcklps(ymm10, ymm8, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)

    vextractf128(imm(0x1), ymm0, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm0)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma04..gamma34 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c
    vextractf128(imm(0x1), ymm1, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm1)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma01..gamma31 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma05..gamma35 )

    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c
    vunpckhps(ymm6, ymm4, ymm0)
    vunpckhps(ymm10, ymm8, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)

    vextractf128(imm(0x1), ymm0, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm0)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma02..gamma32 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma06..gamma36 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c
    vextractf128(imm(0x1), ymm1, xmm2)
    vfmadd231ps(mem(rcx), xmm3, xmm1)
    vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma03..gamma33 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma07..gamma37 )

    vunpcklps(ymm14, ymm12, ymm0)
    vextractf128(imm(0x1), ymm0, xmm2)
    vpermilps(imm(0xe),xmm0,xmm5)
    vpermilps(imm(0xe),xmm2,xmm6)
    vmovq(mem(rdx),xmm4)
    vmovq(mem(rdx, rsi, 4),xmm1)
    vfmadd231ps(xmm4, xmm3, xmm0)
    vfmadd231ps(xmm1, xmm3, xmm2)
    vmovlpd(xmm0, mem(rdx)) // store ( gamma40..gamma50 )
    vmovlpd(xmm2, mem(rdx, rsi, 4)) // store ( gamma44..gamma54 )
    lea(mem(rdx, rsi, 1), rdx)
    vmovq(mem(rdx),xmm4)
    vmovq(mem(rdx, rsi, 4),xmm1)
    vfmadd231ps(xmm4, xmm3, xmm5)
    vfmadd231ps(xmm1, xmm3, xmm6)
    vmovlpd(xmm5, mem(rdx)) // store ( gamma41..gamma51 )
    vmovlpd(xmm6, mem(rdx, rsi, 4)) // store ( gamma45..gamma55 )
    lea(mem(rdx, rsi, 1), rdx)

    vunpckhps(ymm14, ymm12, ymm0)
    vextractf128(imm(0x1), ymm0, xmm2)
    vpermilps(imm(0xe),xmm0,xmm5)
    vpermilps(imm(0xe),xmm2,xmm6)
    vmovq(mem(rdx),xmm4)
    vmovq(mem(rdx, rsi, 4),xmm1)
    vfmadd231ps(xmm4, xmm3, xmm0)
    vfmadd231ps(xmm1, xmm3, xmm2)
    vmovlpd(xmm0, mem(rdx)) // store ( gamma42..gamma52 )
    vmovlpd(xmm2, mem(rdx, rsi, 4)) // store ( gamma46..gamma56 )
    lea(mem(rdx, rsi, 1), rdx)
    vmovq(mem(rdx),xmm4)
    vmovq(mem(rdx, rsi, 4),xmm1)
    vfmadd231ps(xmm4, xmm3, xmm5)
    vfmadd231ps(xmm1, xmm3, xmm6)
    vmovlpd(xmm5, mem(rdx)) // store ( gamma43..gamma53 )
    vmovlpd(xmm6, mem(rdx, rsi, 4)) // store ( gamma47..gamma57 )

    jmp(.SDONE)                        // jump to end.

    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLSTORBZ)                    // jump to column storage case

    label(.SROWSTORBZ)

    vmovups(ymm4, mem(rcx))
    add(rdi, rcx)
    vmovups(ymm6, mem(rcx))
    add(rdi, rcx)
    vmovups(ymm8, mem(rcx))
    add(rdi, rcx)
    vmovups(ymm10, mem(rcx))
    add(rdi, rcx)
    vmovups(ymm12, mem(rcx))
    add(rdi, rcx)
    vmovups(ymm14, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)

    vunpcklps(ymm6, ymm4, ymm0)
    vunpcklps(ymm10, ymm8, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)

    vextractf128(imm(0x1), ymm0, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma04..gamma34 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c
    vextractf128(imm(0x1), ymm1, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma01..gamma31 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma05..gamma35 )

    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c
    vunpckhps(ymm6, ymm4, ymm0)
    vunpckhps(ymm10, ymm8, ymm1)
    vshufps(imm(0x4e), ymm1, ymm0, ymm2)
    vblendps(imm(0xcc), ymm2, ymm0, ymm0)
    vblendps(imm(0x33), ymm2, ymm1, ymm1)

    vextractf128(imm(0x1), ymm0, xmm2)
    vmovups(xmm0, mem(rcx)) // store ( gamma02..gamma32 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma06..gamma36 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c
    vextractf128(imm(0x1), ymm1, xmm2)
    vmovups(xmm1, mem(rcx)) // store ( gamma03..gamma33 )
    vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma07..gamma37 )
    /******************top right tile 8x2***************************/
    vunpcklps(ymm14, ymm12, ymm0)
    vextractf128(imm(0x1), ymm0, xmm2)
    vmovlpd(xmm0, mem(rdx)) // store ( gamma40..gamma50 )
    vmovlpd(xmm2, mem(rdx, rsi, 4)) // store ( gamma44..gamma54 )
    lea(mem(rdx, rsi, 1), rdx)
    vmovhpd(xmm0, mem(rdx)) // store ( gamma41..gamma51 )
    vmovhpd(xmm2, mem(rdx, rsi, 4)) // store ( gamma45..gamma55 )
    lea(mem(rdx, rsi, 1), rdx)

    vunpckhps(ymm14, ymm12, ymm0)
    vextractf128(imm(0x1), ymm0, xmm2)
    vmovlpd(xmm0, mem(rdx)) // store ( gamma42..gamma52 )
    vmovlpd(xmm2, mem(rdx, rsi, 4)) // store ( gamma46..gamma56 )
    lea(mem(rdx, rsi, 1), rdx)
    vmovhpd(xmm0, mem(rdx)) // store ( gamma43..gamma53 )
    vmovhpd(xmm2, mem(rdx, rsi, 4)) // store ( gamma47..gamma57 )


    label(.SDONE)

    lea(mem(r12, rdi, 4), r12)         //
    lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

    //lea(mem(r14, r8,  4), r14)         //
    //lea(mem(r14, r8,  2), r14)         // a_ii = r14 += 6*rs_a
    mov(var(ps_a4), rax)               // load ps_a4
    lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a4

    dec(r11)                           // ii -= 1;
    jne(.SLOOP6X8I)                    // iterate again if ii != 0.

    label(.SRETURN)


    end_asm(
    : // output operands (none)
    : // input operands
      [m_iter] "m" (m_iter),
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a4]  "m" (ps_a4),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3",
      "xmm4", "xmm5", "xmm6", "xmm7",
      "xmm8", "xmm9", "xmm10", "xmm11",
      "xmm12", "xmm13", "xmm14", "xmm15",
      "ymm0", "ymm1", "ymm2", "ymm3",
      "ymm4", "ymm6", "ymm8", "ymm10",
      "ymm12", "ymm14",
      "memory"
    )

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      nr_cur = 8;
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = (float *)c + i_edge*rs_c;
        float* restrict ai  = (float *)a + m_iter*ps_a;
        float* restrict bj  = (float *)b;

        gemmsup_ker_ft ker_fps[6] =
        {
          NULL,
          bli_sgemmsup_rv_zen_asm_1x8,
          bli_sgemmsup_rv_zen_asm_2x8,
          bli_sgemmsup_rv_zen_asm_3x8,
          bli_sgemmsup_rv_zen_asm_4x8,
          bli_sgemmsup_rv_zen_asm_5x8
        };

        gemmsup_ker_ft ker_fp = ker_fps[ m_left ];

        ker_fp
        (
          conja, conjb, m_left, nr_cur, k0,
          alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
          beta, cij, rs_c0, cs_c0, data, cntx
        );
        return;
    }
}

void bli_sgemmsup_rv_zen_asm_6x4m
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
    //void*    a_next = bli_auxinfo_next_a( data );
    //void*    b_next = bli_auxinfo_next_b( data );

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    float *a = (float *)a0;
    float *b = (float *)b0;
    float *c = (float *)c0;

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a( data );
    uint64_t ps_a4  = ps_a * sizeof( float );

    if ( m_iter == 0 ) goto consider_edge_cases;

    // -------------------------------------------------------------------------
    begin_asm()

    mov(var(a), r14)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

    lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
    lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)

                                       // NOTE: We cannot pre-load elements of a or b
                                       // because it could eventually, in the last
                                       // unrolled iter or the cleanup loop, result
                                       // in reading beyond the bounds allocated mem
                                       // (the likely result: a segmentation fault).

    mov(var(c), r12)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)

    // During preamble and loops:
    // r12 = rcx = c    // r14 = rax = a
    // read rbx from var(b) near beginning of loop
    // r11 = m dim index ii

    mov(var(m_iter), r11)              // ii = m_iter;

    label(.SLOOP6X4I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]

    vxorps(xmm1,  xmm1,  xmm1)
    vxorps(xmm4,  xmm4,  xmm4)
    vxorps(xmm6,  xmm6,  xmm6)
    vxorps(xmm8,  xmm8,  xmm8)
    vxorps(xmm10, xmm10, xmm10)
    vxorps(xmm12, xmm12, xmm12)
    vxorps(xmm14, xmm14, xmm14)

    mov(var(b), rbx)                   // load address of b.
    mov(r14, rax)                      // reset rax to current upanel of a.

    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    lea(mem(r12, rdi, 2), rdx)         //
    lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
    prefetch(0, mem(r12, 5*8))         // prefetch c + 0*rs_c
    prefetch(0, mem(r12, rdi, 1, 5*8)) // prefetch c + 1*rs_c
    prefetch(0, mem(r12, rdi, 2, 5*8)) // prefetch c + 2*rs_c
    prefetch(0, mem(rdx, 5*8))         // prefetch c + 3*rs_c
    prefetch(0, mem(rdx, rdi, 1, 5*8)) // prefetch c + 4*rs_c
    prefetch(0, mem(rdx, rdi, 2, 5*8)) // prefetch c + 5*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(r12, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(r12, 5*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(r12, rsi, 1, 5*8)) // prefetch c + 1*cs_c
    prefetch(0, mem(r12, rsi, 2, 5*8)) // prefetch c + 2*cs_c
    prefetch(0, mem(rdx, 5*8))         // prefetch c + 3*cs_c
    prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 4*cs_c
    prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 5*cs_c

    label(.SPOSTPFETCH)                // done prefetching c

    lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
    lea(mem(rax, r8,  4), rdx)         // use rdx for prefetching lines
    lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.

    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                       // contains the k_left loop.
    label(.SLOOPKITER)                 // MAIN LOOP

    // ---------------------------------- iteration 0
    prefetch(0, mem(rdx, 5*8))

    vmovups(mem(rbx), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    vbroadcastss(mem(rax, r8,  4), xmm2)
    vbroadcastss(mem(rax, r15, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)
    vfmadd231ps(xmm0, xmm3, xmm14)

    // ---------------------------------- iteration 1
    prefetch(0, mem(rdx, r9, 1, 5*8))

    vmovups(mem(rbx), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    vbroadcastss(mem(rax, r8,  4), xmm2)
    vbroadcastss(mem(rax, r15, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)
    vfmadd231ps(xmm0, xmm3, xmm14)

    // ---------------------------------- iteration 2
    prefetch(0, mem(rdx, r9, 2, 5*8))

    vmovups(mem(rbx), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    vbroadcastss(mem(rax, r8,  4), xmm2)
    vbroadcastss(mem(rax, r15, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)
    vfmadd231ps(xmm0, xmm3, xmm14)

    // ---------------------------------- iteration 3
    prefetch(0, mem(rdx, rcx, 1, 5*8))

    vmovups(mem(rbx), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    vbroadcastss(mem(rax, r8,  4), xmm2)
    vbroadcastss(mem(rax, r15, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)
    vfmadd231ps(xmm0, xmm3, xmm14)

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.

    label(.SCONSIDKLEFT)

    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                       // else, we prepare to enter k_left loop.

    label(.SLOOPKLEFT)                 // EDGE LOOP

    vmovups(mem(rbx), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    vbroadcastss(mem(rax, r8,  4), xmm2)
    vbroadcastss(mem(rax, r15, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)
    vfmadd231ps(xmm0, xmm3, xmm14)


    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.


    label(.SPOSTACCUM)

    mov(r12, rcx)                      // reset rcx to current utile of c.
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), xmm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), xmm3)       // load beta and duplicate

    vmulps(xmm0, xmm4, xmm4)           // scale by alpha
    vmulps(xmm0, xmm6, xmm6)
    vmulps(xmm0, xmm8, xmm8)
    vmulps(xmm0, xmm10, xmm10)
    vmulps(xmm0, xmm12, xmm12)
    vmulps(xmm0, xmm14, xmm14)

    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

    lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;
    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

                                       // now avoid loading C if beta == 0

    vxorps(xmm0, xmm0, xmm0)           // set xmm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case


    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLSTORED)                    // jump to column storage case


    label(.SROWSTORED)

    vfmadd231ps(mem(rcx), xmm3, xmm4)
    vmovups(xmm4, mem(rcx))
    add(rdi, rcx)
    vfmadd231ps(mem(rcx), xmm3, xmm6)
    vmovups(xmm6, mem(rcx))
    add(rdi, rcx)
    vfmadd231ps(mem(rcx), xmm3, xmm8)
    vmovups(xmm8, mem(rcx))
    add(rdi, rcx)
    vfmadd231ps(mem(rcx), xmm3, xmm10)
    vmovups(xmm10, mem(rcx))
    add(rdi, rcx)
    vfmadd231ps(mem(rcx), xmm3, xmm12)
    vmovups(xmm12, mem(rcx))
    add(rdi, rcx)
    vfmadd231ps(mem(rcx), xmm3, xmm14)
    vmovups(xmm14, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORED)

    /****6x4 tile is transposed and saved in col major as 4x6*****/
    vunpcklps(xmm6, xmm4, xmm0)
    vunpcklps(xmm10, xmm8, xmm1)
    vshufps(imm(0x4e), xmm1, xmm0, xmm2)
    vblendps(imm(0xcc), xmm2, xmm0, xmm0)
    vblendps(imm(0x33), xmm2, xmm1, xmm1)
    vfmadd231ps(mem(rcx), xmm3, xmm0)
    vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c
    vfmadd231ps(mem(rcx), xmm3, xmm1)
    vmovups(xmm1, mem(rcx)) // store ( gamma01..gamma31 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c

    vunpckhps(xmm6, xmm4, xmm0)
    vunpckhps(xmm10, xmm8, xmm1)
    vshufps(imm(0x4e), xmm1, xmm0, xmm2)
    vblendps(imm(0xcc), xmm2, xmm0, xmm0)
    vblendps(imm(0x33), xmm2, xmm1, xmm1)
    vfmadd231ps(mem(rcx), xmm3, xmm0)
    vmovups(xmm0, mem(rcx)) // store ( gamma02..gamma32 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c
    vfmadd231ps(mem(rcx), xmm3, xmm1)
    vmovups(xmm1, mem(rcx)) // store ( gamma03..gamma33 )

    vunpcklps(xmm14, xmm12, xmm0)
    vpermilps(imm(0x4e), xmm0, xmm5)
    vmovq(mem(rdx),xmm4)
    vfmadd231ps(xmm4, xmm3, xmm0)
    vmovlpd(xmm0, mem(rdx)) // store ( gamma40..gamma50 )

    lea(mem(rdx, rsi, 1), rdx)
    vmovq(mem(rdx),xmm4)
    vfmadd231ps(xmm4, xmm3, xmm5)
    vmovlpd(xmm5, mem(rdx)) // store ( gamma41..gamma51 )

    lea(mem(rdx, rsi, 1), rdx)
    vunpckhps(xmm14, xmm12, xmm0)
    vpermilps(imm(0x4e), xmm0, xmm5)
    vmovq(mem(rdx),xmm4)
    vfmadd231ps(xmm4, xmm3, xmm0)
    vmovlpd(xmm0, mem(rdx)) // store ( gamma42..gamma52 )

    lea(mem(rdx, rsi, 1), rdx)
    vmovq(mem(rdx),xmm4)
    vfmadd231ps(xmm4, xmm3, xmm5)
    vmovlpd(xmm5, mem(rdx)) // store ( gamma43..gamma53 )

    jmp(.SDONE)                        // jump to end.

    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLSTORBZ)                    // jump to column storage case

    label(.SROWSTORBZ)

    vmovups(xmm4, mem(rcx))
    add(rdi, rcx)
    vmovups(xmm6, mem(rcx))
    add(rdi, rcx)
    vmovups(xmm8, mem(rcx))
    add(rdi, rcx)
    vmovups(xmm10, mem(rcx))
    add(rdi, rcx)
    vmovups(xmm12, mem(rcx))
    add(rdi, rcx)
    vmovups(xmm14, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)

    vunpcklps(xmm6, xmm4, xmm0)
    vunpcklps(xmm10, xmm8, xmm1)
    vshufps(imm(0x4e), xmm1, xmm0, xmm2)
    vblendps(imm(0xcc), xmm2, xmm0, xmm0)
    vblendps(imm(0x33), xmm2, xmm1, xmm1)
    vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c
    vmovups(xmm1, mem(rcx)) // store ( gamma01..gamma31 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c
    vunpckhps(xmm6, xmm4, xmm0)
    vunpckhps(xmm10, xmm8, xmm1)
    vshufps(imm(0x4e), xmm1, xmm0, xmm2)
    vblendps(imm(0xcc), xmm2, xmm0, xmm0)
    vblendps(imm(0x33), xmm2, xmm1, xmm1)
    vmovups(xmm0, mem(rcx)) // store ( gamma02..gamma32 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c
    vmovups(xmm1, mem(rcx)) // store ( gamma03..gamma33 )

    vunpcklps(xmm14, xmm12, xmm0)
    vmovlpd(xmm0, mem(rdx)) // store ( gamma40..gamma50 )
    lea(mem(rdx, rsi, 1), rdx)
    vmovhpd(xmm0, mem(rdx)) // store ( gamma41..gamma51 )
    lea(mem(rdx, rsi, 1), rdx)
    vunpckhps(xmm14, xmm12, xmm0)
    vmovlpd(xmm0, mem(rdx)) // store ( gamma42..gamma52 )
    lea(mem(rdx, rsi, 1), rdx)
    vmovhpd(xmm0, mem(rdx)) // store ( gamma43..gamma53 )

    label(.SDONE)

    lea(mem(r12, rdi, 4), r12)         //
    lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

    //lea(mem(r14, r8,  4), r14)         //
    //lea(mem(r14, r8,  2), r14)         // a_ii = r14 += 6*rs_a
    mov(var(ps_a4), rax)               // load ps_a4
    lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a4

    dec(r11)                           // ii -= 1;
    jne(.SLOOP6X4I)                    // iterate again if ii != 0.

    label(.SRETURN)


    end_asm(
    : // output operands (none)
    : // input operands
      [m_iter] "m" (m_iter),
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a4]  "m" (ps_a4),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3",
      "xmm4", "xmm5", "xmm6", "xmm7",
      "xmm8", "xmm9", "xmm10", "xmm11",
      "xmm12", "xmm13", "xmm14", "xmm15",
      "memory"
    )

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      nr_cur = 4;
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = (float *)c + i_edge*rs_c;
        float* restrict ai  = (float *)a + m_iter*ps_a;
        float* restrict bj  = (float *)b;

        gemmsup_ker_ft ker_fps[6] =
        {
          NULL,
          bli_sgemmsup_rv_zen_asm_1x4,
          bli_sgemmsup_rv_zen_asm_2x4,
          bli_sgemmsup_rv_zen_asm_3x4,
          bli_sgemmsup_rv_zen_asm_4x4,
          bli_sgemmsup_rv_zen_asm_5x4
        };

        gemmsup_ker_ft ker_fp = ker_fps[ m_left ];

        ker_fp
        (
          conja, conjb, m_left, nr_cur, k0,
          alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
          beta, cij, rs_c0, cs_c0, data, cntx
        );
        return;
    }
}

void bli_sgemmsup_rv_zen_asm_6x2m
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
    //void*    a_next = bli_auxinfo_next_a( data );
    //void*    b_next = bli_auxinfo_next_b( data );

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    float *a = (float *)a0;
    float *b = (float *)b0;
    float *c = (float *)c0;

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a( data );
    uint64_t ps_a4  = ps_a * sizeof( float );

    if ( m_iter == 0 ) goto consider_edge_cases;

    // -------------------------------------------------------------------------
    begin_asm()

    mov(var(a), r14)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

    lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
    lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)

                                       // NOTE: We cannot pre-load elements of a or b
                                       // because it could eventually, in the last
                                       // unrolled iter or the cleanup loop, result
                                       // in reading beyond the bounds allocated mem
                                       // (the likely result: a segmentation fault).

    mov(var(c), r12)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)

    // During preamble and loops:
    // r12 = rcx = c    // r14 = rax = a
    // read rbx from var(b) near beginning of loop
    // r11 = m dim index ii

    mov(var(m_iter), r11)              // ii = m_iter;

    label(.SLOOP6X2I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]

    vxorps(xmm1,  xmm1,  xmm1)
    vxorps(xmm4,  xmm4,  xmm4)
    vxorps(xmm6,  xmm6,  xmm6)
    vxorps(xmm8,  xmm8,  xmm8)
    vxorps(xmm10, xmm10, xmm10)
    vxorps(xmm12, xmm12, xmm12)
    vxorps(xmm14, xmm14, xmm14)

    mov(var(b), rbx)                   // load address of b.
    mov(r14, rax)                      // reset rax to current upanel of a.

    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    lea(mem(r12, rdi, 2), rdx)         //
    lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
    prefetch(0, mem(r12, 5*8))         // prefetch c + 0*rs_c
    prefetch(0, mem(r12, rdi, 1, 5*8)) // prefetch c + 1*rs_c
    prefetch(0, mem(r12, rdi, 2, 5*8)) // prefetch c + 2*rs_c
    prefetch(0, mem(rdx, 5*8))         // prefetch c + 3*rs_c
    prefetch(0, mem(rdx, rdi, 1, 5*8)) // prefetch c + 4*rs_c
    prefetch(0, mem(rdx, rdi, 2, 5*8)) // prefetch c + 5*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(r12, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(r12, 5*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(r12, rsi, 1, 5*8)) // prefetch c + 1*cs_c
    prefetch(0, mem(r12, rsi, 2, 5*8)) // prefetch c + 2*cs_c
    prefetch(0, mem(rdx, 5*8))         // prefetch c + 3*cs_c
    prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 4*cs_c
    prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 5*cs_c

    label(.SPOSTPFETCH)                // done prefetching c

    lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
    lea(mem(rax, r8,  4), rdx)         // use rdx for prefetching lines
    lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.

    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                       // contains the k_left loop.
    label(.SLOOPKITER)                 // MAIN LOOP

    // ---------------------------------- iteration 0
    prefetch(0, mem(rdx, 5*8))
    vmovq(mem(rbx), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    vbroadcastss(mem(rax, r8,  4), xmm2)
    vbroadcastss(mem(rax, r15, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)
    vfmadd231ps(xmm0, xmm3, xmm14)

    // ---------------------------------- iteration 1
    prefetch(0, mem(rdx, r9, 1, 5*8))

    vmovq(mem(rbx), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    vbroadcastss(mem(rax, r8,  4), xmm2)
    vbroadcastss(mem(rax, r15, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)
    vfmadd231ps(xmm0, xmm3, xmm14)

    // ---------------------------------- iteration 2
    prefetch(0, mem(rdx, r9, 2, 5*8))

    vmovq(mem(rbx), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    vbroadcastss(mem(rax, r8,  4), xmm2)
    vbroadcastss(mem(rax, r15, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)
    vfmadd231ps(xmm0, xmm3, xmm14)

    // ---------------------------------- iteration 3
    prefetch(0, mem(rdx, rcx, 1, 5*8))

    vmovq(mem(rbx), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    vbroadcastss(mem(rax, r8,  4), xmm2)
    vbroadcastss(mem(rax, r15, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)
    vfmadd231ps(xmm0, xmm3, xmm14)

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.

    label(.SCONSIDKLEFT)

    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                       // else, we prepare to enter k_left loop.

    label(.SLOOPKLEFT)                 // EDGE LOOP

    vmovq(mem(rbx), xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    vbroadcastss(mem(rax, r8,  4), xmm2)
    vbroadcastss(mem(rax, r15, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)
    vfmadd231ps(xmm0, xmm3, xmm14)


    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.


    label(.SPOSTACCUM)

    mov(r12, rcx)                      // reset rcx to current utile of c.
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), xmm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), xmm3)       // load beta and duplicate

    vmulps(xmm0, xmm4, xmm4)           // scale by alpha
    vmulps(xmm0, xmm6, xmm6)
    vmulps(xmm0, xmm8, xmm8)
    vmulps(xmm0, xmm10, xmm10)
    vmulps(xmm0, xmm12, xmm12)
    vmulps(xmm0, xmm14, xmm14)

    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

    lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;
    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

                                       // now avoid loading C if beta == 0

    vxorps(xmm0, xmm0, xmm0)           // set xmm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case


    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLSTORED)                    // jump to column storage case


    label(.SROWSTORED)

    vmovsd(mem(rcx), xmm0)
    vfmadd231ps(xmm0, xmm3, xmm4)
    vmovlpd(xmm4, mem(rcx))
    add(rdi, rcx)
    vmovsd(mem(rcx), xmm0)
    vfmadd231ps(xmm0, xmm3, xmm6)
    vmovlpd(xmm6, mem(rcx))
    add(rdi, rcx)
    vmovsd(mem(rcx), xmm0)
    vfmadd231ps(xmm0, xmm3, xmm8)
    vmovlpd(xmm8, mem(rcx))
    add(rdi, rcx)
    vmovsd(mem(rcx), xmm0)
    vfmadd231ps(xmm0, xmm3, xmm10)
    vmovlpd(xmm10, mem(rcx))
    add(rdi, rcx)
    vmovsd(mem(rcx), xmm0)
    vfmadd231ps(xmm0, xmm3, xmm12)
    vmovlpd(xmm12, mem(rcx))
    add(rdi, rcx)
    vmovsd(mem(rcx), xmm0)
    vfmadd231ps(xmm0, xmm3, xmm14)
    vmovlpd(xmm14, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORED)

    /****6x2 tile is transposed and saved in col major as 2x6*****/
    vunpcklps(xmm6, xmm4, xmm0)//a0b0a1b1
    vunpcklps(xmm10, xmm8, xmm1)//c0d0c1d1
    vshufps(imm(0x44), xmm1, xmm0, xmm2) //01-00-01-00
    vshufps(imm(0xee), xmm1, xmm0, xmm4) //11-10-11-10

    vfmadd231ps(mem(rcx), xmm3, xmm2)
    vmovupd(xmm2, mem(rcx)) // store ( gamma00..gamma30 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c
    vfmadd231ps(mem(rcx), xmm3, xmm4)
    vmovupd(xmm4, mem(rcx)) // store ( gamma01..gamma31 )

    vunpcklps(xmm14, xmm12, xmm0)//eof0e1f1
    vpermilps(imm(0x4e),xmm0,xmm5)
    vmovq(mem(rdx), xmm4)
    vfmadd231ps(xmm4, xmm3, xmm0)
    vmovlpd(xmm0, mem(rdx)) // store ( gamma40..gamma50 )
    lea(mem(rdx, rsi, 1), rdx)
    vmovq(mem(rdx), xmm4)
    vfmadd231ps(xmm4, xmm3, xmm5)
    vmovlpd(xmm5, mem(rdx)) // store ( gamma41..gamma51 )

    jmp(.SDONE)                        // jump to end.

    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLSTORBZ)                    // jump to column storage case

    label(.SROWSTORBZ)

    vmovlpd(xmm4, mem(rcx))
    add(rdi, rcx)
    vmovlpd(xmm6, mem(rcx))
    add(rdi, rcx)
    vmovlpd(xmm8, mem(rcx))
    add(rdi, rcx)
    vmovlpd(xmm10, mem(rcx))
    add(rdi, rcx)
    vmovlpd(xmm12, mem(rcx))
    add(rdi, rcx)
    vmovlpd(xmm14, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)

    vunpcklps(xmm6, xmm4, xmm0)//a0b0a1b1
    vunpcklps(xmm10, xmm8, xmm1)//c0d0c1d1
    vshufps(imm(0x44), xmm1, xmm0, xmm2) //01-00-01-00
    vshufps(imm(0xee), xmm1, xmm0, xmm4) //11-10-11-10

    vmovupd(xmm2, mem(rcx)) // store ( gamma00..gamma30 )
    lea(mem(rcx, rsi, 1), rcx) // rcx += 1*cs_c
    vmovupd(xmm4, mem(rcx)) // store ( gamma01..gamma31 )

    vunpcklps(xmm14, xmm12, xmm0)//eof0e1f1
    vmovlpd(xmm0, mem(rdx)) // store ( gamma40..gamma50 )
    lea(mem(rdx, rsi, 1), rdx)
    vmovhpd(xmm0, mem(rdx)) // store ( gamma41..gamma51 )

    label(.SDONE)

    lea(mem(r12, rdi, 4), r12)         //
    lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

    //lea(mem(r14, r8,  4), r14)         //
    //lea(mem(r14, r8,  2), r14)         // a_ii = r14 += 6*rs_a
    mov(var(ps_a4), rax)               // load ps_a4
    lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a4

    dec(r11)                           // ii -= 1;
    jne(.SLOOP6X2I)                    // iterate again if ii != 0.

    label(.SRETURN)


    end_asm(
    : // output operands (none)
    : // input operands
      [m_iter] "m" (m_iter),
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a4]  "m" (ps_a4),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3",
      "xmm4", "xmm5", "xmm6", "xmm7",
      "xmm8", "xmm9", "xmm10", "xmm11",
      "xmm12", "xmm13", "xmm14", "xmm15",
      "memory"
    )

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      nr_cur = 2;
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = (float *)c + i_edge*rs_c;
        float* restrict ai  = (float *)a + m_iter*ps_a;
        float* restrict bj  = (float *)b;

        gemmsup_ker_ft ker_fps[6] =
        {
          NULL,
          bli_sgemmsup_rv_zen_asm_1x2,
          bli_sgemmsup_rv_zen_asm_2x2,
          bli_sgemmsup_rv_zen_asm_3x2,
          bli_sgemmsup_rv_zen_asm_4x2,
          bli_sgemmsup_rv_zen_asm_5x2
        };

        gemmsup_ker_ft ker_fp = ker_fps[ m_left ];

        ker_fp
        (
          conja, conjb, m_left, nr_cur, k0,
          alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
          beta, cij, rs_c0, cs_c0, data, cntx
        );
        return;
    }
}

/* Mask elements to specify how many elements to be loaded from C buffer */
static const int32_t mask[8][8] = { {0, 0, 0, 0, 0, 0, 0, 0}, //load no values, not used currently
                                    {-1, 0, 0, 0, 0, 0, 0, 0}, // load 1 value from memory
                                    {-1, -1, 0, 0, 0, 0, 0, 0}, // load 2 values from memory
                                    {-1, -1, -1, 0, 0, 0, 0, 0},
                                    {-1, -1, -1, -1, 0, 0, 0, 0},
                                    {-1, -1, -1, -1, -1, 0, 0, 0},
                                    {-1, -1, -1, -1, -1, -1, 0, 0},
                                    {-1, -1, -1, -1, -1, -1, -1, 0},
                                  };

void bli_sgemmsup_rv_zen_asm_6x16m_mask
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    float *a = (float *)a0;
    float *b = (float *)b0;
    float *c = (float *)c0;

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a( data );
    uint64_t ps_a4  = ps_a * sizeof( float );

    if ( m_iter == 0 ) goto consider_edge_cases;

    // This kernel is called when n_left is greater than 8. This kernel operates 16 columns at time.
    // First 8 elements can be loaded directly and next elements will be loaded based the mask reg
    //
    // Sets up the mask for loading relevant remainder elements in load direction
    //
    // ______ymm0______  __________ymm1_________
    // | | | | | | | | | | | |  |  |  |  |  |  |
    // |0|1|2|3|4|5|6|7| |8|9|10|11|12|13|14|15|    ----> Source vector
    // |_|_|_|_|_|_|_|_| |_|_|__|__|__|__|__|__|
    //
    // ________________  ______ymm3_______
    // | | | | | | | | | | | | | | | | | |
    // |NoMASK Required| |x|x|x|x|x|x|x|x|  ----> Mask vector[x can be -1/0]
    // |_|_|_|_|_|_|_|_| |_|_|_|_|_|_|_|_|
    //
    // For example when n_left = 13
    // ________________  ________ymm3__________
    // | | | | | | | | | |  |  |  |  |  | | | |
    // |NoMASK Required| |-1|-1|-1|-1|-1|0|0|0|  ----> Mask vector
    // |_|_|_|_|_|_|_|_| |__|__|__|__|__|_|_|_|
    //
    // ______ymm0_______  ________ymm1__________
    // | | | | | | | | | | | |  |  |  |  |  |  |
    // |0|1|2|3|4|5|6|7| |8|9|10|11|12|0 |0 |0 |  ----> Destination vector
    // |_|_|_|_|_|_|_|_| |_|_|__|__|__|__|__|__|
    //

    uint64_t n_mod8 = n0 % 8 ;
    const int32_t *mask_vec = mask[n_mod8];
    // -------------------------------------------------------------------------

    begin_asm()

    mov(var(mask_vec), rdx)
    vmovdqu(mem(rdx), ymm3)           //load mask values

    mov(var(a), r14)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

    lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
    lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)

                                       // NOTE: We cannot pre-load elements of a or b
                                       // because it could eventually, in the last
                                       // unrolled iter or the cleanup loop, result
                                       // in reading beyond the bounds allocated mem
                                       // (the likely result: a segmentation fault).

    mov(var(c), r12)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


    // During preamble and loops:
    // r12 = rcx = c
    // r14 = rax = a
    // read rbx from var(b) near beginning of loop
    // r11 = m dim index ii

    mov(var(m_iter), r11)              // ii = m_iter;

    label(.SLOOP6X15I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]

                                       // skylake can execute 3 vxorps ipc with
                                       // a latency of 1 cycle, while vzeroall
                                       // has a latency of 12 cycles.
    vxorps(ymm4,  ymm4,  ymm4)
    vxorps(ymm5,  ymm5,  ymm5)
    vxorps(ymm6,  ymm6,  ymm6)
    vxorps(ymm7,  ymm7,  ymm7)
    vxorps(ymm8,  ymm8,  ymm8)
    vxorps(ymm9,  ymm9,  ymm9)
    vxorps(ymm10, ymm10, ymm10)
    vxorps(ymm11, ymm11, ymm11)
    vxorps(ymm12, ymm12, ymm12)
    vxorps(ymm13, ymm13, ymm13)
    vxorps(ymm14, ymm14, ymm14)
    vxorps(ymm15, ymm15, ymm15)

    mov(var(b), rbx)                   // load address of b.
    mov(r14, rax)                      // reset rax to current upanel of a.

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    lea(mem(r12, rdi, 2), rdx)         //
    lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
    prefetch(0, mem(r12,         8*4)) // prefetch c + 0*rs_c
    prefetch(0, mem(r12, rdi, 1, 8*4)) // prefetch c + 1*rs_c
    prefetch(0, mem(r12, rdi, 2, 8*4)) // prefetch c + 2*rs_c
    prefetch(0, mem(rdx,         8*4)) // prefetch c + 3*rs_c
    prefetch(0, mem(rdx, rdi, 1, 8*4)) // prefetch c + 4*rs_c
    prefetch(0, mem(rdx, rdi, 2, 8*4)) // prefetch c + 5*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(rsi, rsi, 2), rcx)         // rcx = 3*cs_c;
    prefetch(0, mem(r12,         5*4)) // prefetch c + 0*cs_c
    prefetch(0, mem(r12, rsi, 1, 5*4)) // prefetch c + 1*cs_c
    prefetch(0, mem(r12, rsi, 2, 5*4)) // prefetch c + 2*cs_c
    prefetch(0, mem(r12, rcx, 1, 5*4)) // prefetch c + 3*cs_c
    prefetch(0, mem(r12, rsi, 4, 5*4)) // prefetch c + 4*cs_c
    lea(mem(r12, rsi, 4), rdx)         // rdx = c + 4*cs_c;
    prefetch(0, mem(rdx, rsi, 1, 5*4)) // prefetch c + 5*cs_c
    prefetch(0, mem(rdx, rsi, 2, 5*4)) // prefetch c + 6*cs_c
    prefetch(0, mem(rdx, rcx, 1, 5*4)) // prefetch c + 7*cs_c
    prefetch(0, mem(rdx, rsi, 4, 5*4)) // prefetch c + 8*cs_c
    lea(mem(r12, rsi, 8), rdx)         // rdx = c + 8*cs_c;
    prefetch(0, mem(rdx, rsi, 1, 5*4)) // prefetch c + 9*cs_c
    prefetch(0, mem(rdx, rsi, 2, 5*4)) // prefetch c + 10*cs_c
    prefetch(0, mem(rdx, rcx, 1, 5*4)) // prefetch c + 11*cs_c
    prefetch(0, mem(rdx, rsi, 4, 5*4)) // prefetch c + 12*cs_c
    lea(mem(r12, rcx, 4), rdx)         // rdx = c + 12*cs_c;
    prefetch(0, mem(rdx, rsi, 1, 5*4)) // prefetch c + 13*cs_c
    prefetch(0, mem(rdx, rsi, 2, 5*4)) // prefetch c + 14*cs_c

    label(.SPOSTPFETCH)                // done prefetching c

    mov(var(ps_a4), rdx)               // load ps_a4
    lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a4
    lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
                                       // use rcx, rdx for prefetching lines
                                       // from next upanel of a.

    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                       // contains the k_left loop.

    label(.SLOOPKITER)                 // MAIN LOOP

    // ---------------------------------- iteration 0

    prefetch(0, mem(rdx, 5*8))

    vmovups(mem(rbx, 0*32), ymm0)      //load first 8 elements
    vmaskmovps(mem(rbx, 1*32), ymm3, ymm1) //load next required elements
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)
    vfmadd231ps(ymm1, ymm2, ymm7)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm1, ymm2, ymm9)
    vbroadcastss(mem(rax, r13, 1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm10)
    vfmadd231ps(ymm1, ymm2, ymm11)

    vbroadcastss(mem(rax, r8,  4), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm12)
    vfmadd231ps(ymm1, ymm2, ymm13)
    vbroadcastss(mem(rax, r15, 1), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm14)
    vfmadd231ps(ymm1, ymm2, ymm15)

    // ---------------------------------- iteration 1

    prefetch(0, mem(rdx, r9, 1, 5*8))

    vmovups(mem(rbx, 0*32), ymm0)
    vmaskmovps(mem(rbx, 1*32), ymm3, ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)
    vfmadd231ps(ymm1, ymm2, ymm7)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm1, ymm2, ymm9)
    vbroadcastss(mem(rax, r13, 1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm10)
    vfmadd231ps(ymm1, ymm2, ymm11)

    vbroadcastss(mem(rax, r8,  4), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm12)
    vfmadd231ps(ymm1, ymm2, ymm13)
    vbroadcastss(mem(rax, r15, 1), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm14)
    vfmadd231ps(ymm1, ymm2, ymm15)

    // ---------------------------------- iteration 2

    prefetch(0, mem(rdx, r9, 2, 5*8))

    vmovups(mem(rbx, 0*32), ymm0)
    vmaskmovps(mem(rbx, 1*32), ymm3, ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)
    vfmadd231ps(ymm1, ymm2, ymm7)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm1, ymm2, ymm9)
    vbroadcastss(mem(rax, r13, 1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm10)
    vfmadd231ps(ymm1, ymm2, ymm11)

    vbroadcastss(mem(rax, r8,  4), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm12)
    vfmadd231ps(ymm1, ymm2, ymm13)
    vbroadcastss(mem(rax, r15, 1), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm14)
    vfmadd231ps(ymm1, ymm2, ymm15)

    // ---------------------------------- iteration 3

    prefetch(0, mem(rdx, rcx, 1, 5*8))
    lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;

    vmovups(mem(rbx, 0*32), ymm0)
    vmaskmovps(mem(rbx, 1*32), ymm3, ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)
    vfmadd231ps(ymm1, ymm2, ymm7)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm1, ymm2, ymm9)
    vbroadcastss(mem(rax, r13, 1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm10)
    vfmadd231ps(ymm1, ymm2, ymm11)

    vbroadcastss(mem(rax, r8,  4), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm12)
    vfmadd231ps(ymm1, ymm2, ymm13)
    vbroadcastss(mem(rax, r15, 1), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm14)
    vfmadd231ps(ymm1, ymm2, ymm15)

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.

    label(.SCONSIDKLEFT)

    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                       // ee, we prepare to enter k_left loop.

    label(.SLOOPKLEFT)                 // EDGE LOOP

    prefetch(0, mem(rdx, 5*8))
    add(r9, rdx)

    vmovups(mem(rbx, 0*32), ymm0)
    vmaskmovps(mem(rbx, 1*32), ymm3, ymm1)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vfmadd231ps(ymm1, ymm2, ymm5)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)
    vfmadd231ps(ymm1, ymm2, ymm7)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vfmadd231ps(ymm1, ymm2, ymm9)
    vbroadcastss(mem(rax, r13, 1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm10)
    vfmadd231ps(ymm1, ymm2, ymm11)

    vbroadcastss(mem(rax, r8,  4), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm12)
    vfmadd231ps(ymm1, ymm2, ymm13)
    vbroadcastss(mem(rax, r15, 1), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm14)
    vfmadd231ps(ymm1, ymm2, ymm15)

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.

    label(.SPOSTACCUM)

    mov(r12, rcx)                      // reset rcx to current utile of c.
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), ymm1)       // load beta and duplicate

    vmulps(ymm0, ymm4, ymm4)           // scale by alpha
    vmulps(ymm0, ymm5, ymm5)
    vmulps(ymm0, ymm6, ymm6)
    vmulps(ymm0, ymm7, ymm7)
    vmulps(ymm0, ymm8, ymm8)
    vmulps(ymm0, ymm9, ymm9)
    vmulps(ymm0, ymm10, ymm10)
    vmulps(ymm0, ymm11, ymm11)
    vmulps(ymm0, ymm12, ymm12)
    vmulps(ymm0, ymm13, ymm13)
    vmulps(ymm0, ymm14, ymm14)
    vmulps(ymm0, ymm15, ymm15)

    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

    lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
    lea(mem(rsi, rsi, 4), rbx)         // rbx = 5*cs_c;

                                       // now avoid loading C if beta == 0

    vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm1)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOTORED)                    // jump to column storage case

    label(.SROWSTORED)

    vfmadd231ps(mem(rcx, 0*32), ymm1, ymm4)
    vmovups(ymm4, mem(rcx, 0*32))

    vmaskmovps(mem(rcx, 1*32), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm1, ymm5)
    vmaskmovps(ymm5, ymm3, mem(rcx, 1*32))    //store only required elements

    add(rdi, rcx)

    vfmadd231ps(mem(rcx, 0*32), ymm1, ymm6)
    vmovups(ymm6, mem(rcx, 0*32))

    vmaskmovps(mem(rcx, 1*32), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm1, ymm7)
    vmaskmovps(ymm7, ymm3, mem(rcx, 1*32))

    add(rdi, rcx)

    vfmadd231ps(mem(rcx, 0*32), ymm1, ymm8)
    vmovups(ymm8, mem(rcx, 0*32))

    vmaskmovps(mem(rcx, 1*32), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm1, ymm9)
    vmaskmovps(ymm9, ymm3, mem(rcx, 1*32))

    add(rdi, rcx)

    vfmadd231ps(mem(rcx, 0*32), ymm1, ymm10)
    vmovups(ymm10, mem(rcx, 0*32))

    vmaskmovps(mem(rcx, 1*32), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm1, ymm11)
    vmaskmovps(ymm11, ymm3, mem(rcx, 1*32))

    add(rdi, rcx)

    vfmadd231ps(mem(rcx, 0*32), ymm1, ymm12)
    vmovups(ymm12, mem(rcx, 0*32))

    vmaskmovps(mem(rcx, 1*32), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm1, ymm13)
    vmaskmovps(ymm13, ymm3, mem(rcx, 1*32))

    add(rdi, rcx)

    vfmadd231ps(mem(rcx, 0*32), ymm1, ymm14)
    vmovups(ymm14, mem(rcx, 0*32))

    vmaskmovps(mem(rcx, 1*32), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm1, ymm15)
    vmaskmovps(ymm15, ymm3, mem(rcx, 1*32))

    jmp(.SDONE)                        // jump to end.

    label(.SCOTORED)

    /* TODO: Add column storage support */

    jmp(.SDONE)                        // jump to end.

    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOTORBZ)                    // jump to column storage case

    label(.SROWSTORBZ)

    vmovups(ymm4, mem(rcx, 0*32))
    vmaskmovps(ymm5, ymm3, mem(rcx, 1*32))    //Store only required elements
    add(rdi, rcx)

    vmovups(ymm6, mem(rcx, 0*32))
    vmaskmovps(ymm7, ymm3, mem(rcx, 1*32))
    add(rdi, rcx)

    vmovups(ymm8, mem(rcx, 0*32))
    vmaskmovps(ymm9, ymm3, mem(rcx, 1*32))
    add(rdi, rcx)

    vmovups(ymm10, mem(rcx, 0*32))
    vmaskmovps(ymm11, ymm3, mem(rcx, 1*32))
    add(rdi, rcx)

    vmovups(ymm12, mem(rcx, 0*32))
    vmaskmovps(ymm13, ymm3, mem(rcx, 1*32))
    add(rdi, rcx)

    vmovups(ymm14, mem(rcx, 0*32))
    vmaskmovps(ymm15, ymm3, mem(rcx, 1*32))

    jmp(.SDONE)                        // jump to end.

    label(.SCOTORBZ)

    /* TODO: Add column storage support*/

    label(.SDONE)

    lea(mem(r12, rdi, 4), r12)         //
    lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

    mov(var(ps_a4), rax)               // load ps_a4
    lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a4

    dec(r11)                           // ii -= 1;
    jne(.SLOOP6X15I)                    // iterate again if ii != 0.

    label(.SRETURN)

    end_asm(
    : // output operands (none)
    : // input operands
      [m_iter]     "m"   (m_iter),
      [k_iter]     "m"   (k_iter),
      [k_left]     "m"   (k_left),
      [a]          "m"   (a),
      [rs_a]       "m"   (rs_a),
      [cs_a]       "m"   (cs_a),
      [ps_a4]      "m"   (ps_a4),
      [b]          "m"   (b),
      [rs_b]       "m"   (rs_b),
      [cs_b]       "m"   (cs_b),
      [alpha]      "m"   (alpha),
      [beta]       "m"   (beta),
      [c]          "m"   (c),
      [rs_c]       "m"   (rs_c),
      [cs_c]       "m"   (cs_c),
      [mask_vec]   "m"   (mask_vec)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1",
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
      "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12",
      "ymm13", "ymm14", "ymm15",
      "memory"
    )

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      nr_cur = n0;
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = (float *)c + i_edge*rs_c;
        float* restrict ai  = (float *)a + m_iter * ps_a;
        float* restrict bj  = (float *)b;

        gemmsup_ker_ft ker_fps[6] =
            {
              NULL,
              bli_sgemmsup_rv_zen_asm_1x16_mask,
              bli_sgemmsup_rv_zen_asm_2x16_mask,
              bli_sgemmsup_rv_zen_asm_3x16_mask,
              bli_sgemmsup_rv_zen_asm_4x16_mask,
              bli_sgemmsup_rv_zen_asm_5x16_mask
            };

        gemmsup_ker_ft ker_fp = ker_fps[ m_left ];

        ker_fp
        (
          conja, conjb, m_left, nr_cur, k0,
          alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
          beta, cij, rs_c0, cs_c0, data, cntx
        );

        return;
    }
}

void bli_sgemmsup_rv_zen_asm_6x8m_mask
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    float *a = (float *)a0;
    float *b = (float *)b0;
    float *c = (float *)c0;

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a( data );
    uint64_t ps_a4  = ps_a * sizeof( float );

    if ( m_iter == 0 ) goto consider_edge_cases;

    // This kernel is called when n_left 7, 6, 5. This kernel operates 8 columns at time.
    //
    // Sets up the mask for loading relevant remainder elements in load direction
    //
    // ______ymm0_______
    // | | | | | | | | |
    // |0|1|2|3|4|5|6|7|   ----> Source vector
    // |_|_|_|_|_|_|_|_|
    //
    //______ymm3_______
    //| | | | | | | | |
    //|x|x|x|x|x|x|x|x|  ----> Mask vector[x can be -1/0]
    //|_|_|_|_|_|_|_|_|
    //
    // For example when n_left = 6
    // ________ymm3__________
    // |  |  |  |  |  |  | | |
    // |-1|-1|-1|-1|-1|-1|0|0|  ----> Mask vector
    // |__|__|__|__|__|__|_|_|
    //
    // _______ymm0______
    // | | | | | | | | |
    // |0|1|2|3|4|5|0|0|  ----> Destination vector
    // |_|_|_|_|_|_|_|_|
    //
    uint64_t n_mod8 = n0 % 8 ;
    const int32_t *mask_vec = mask[n_mod8];
    // -------------------------------------------------------------------------

    begin_asm()

    mov(var(mask_vec), rdx)
    vmovdqu(mem(rdx), ymm3)            //load mask values

    mov(var(a), r14)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

    lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
    lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)

                                       // NOTE: We cannot pre-load elements of a or b
                                       // because it could eventually, in the last
                                       // unrolled iter or the cleanup loop, result
                                       // in reading beyond the bounds allocated mem
                                       // (the likely result: a segmentation fault).

    mov(var(c), r12)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


    // During preamble and loops:
    // r12 = rcx = c
    // r14 = rax = a
    // read rbx from var(b) near beginning of loop
    // r11 = m dim index ii

    mov(var(m_iter), r11)              // ii = m_iter;

    label(.SLOOP6X7I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]

                                       // skylake can execute 3 vxorps ipc with
                                       // a latency of 1 cycle, while vzeroall
                                       // has a latency of 12 cycles.
    vxorps(ymm4,  ymm4,  ymm4)
    vxorps(ymm6,  ymm6,  ymm6)
    vxorps(ymm8,  ymm8,  ymm8)
    vxorps(ymm10, ymm10, ymm10)
    vxorps(ymm12, ymm12, ymm12)
    vxorps(ymm14, ymm14, ymm14)

    mov(var(b), rbx)                   // load address of b.
    mov(r14, rax)                      // reset rax to current upanel of a.

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    lea(mem(r12, rdi, 2), rdx)         //
    lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
    prefetch(0, mem(r12,         4*4)) // prefetch c + 0*rs_c
    prefetch(0, mem(r12, rdi, 1, 4*4)) // prefetch c + 1*rs_c
    prefetch(0, mem(r12, rdi, 2, 4*4)) // prefetch c + 2*rs_c
    prefetch(0, mem(rdx,         4*4)) // prefetch c + 3*rs_c
    prefetch(0, mem(rdx, rdi, 1, 4*4)) // prefetch c + 4*rs_c
    prefetch(0, mem(rdx, rdi, 2, 4*4)) // prefetch c + 5*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(rsi, rsi, 2), rcx)         // rcx = 3*cs_c;
    prefetch(0, mem(r12,         5*4)) // prefetch c + 0*cs_c
    prefetch(0, mem(r12, rsi, 1, 5*4)) // prefetch c + 1*cs_c
    prefetch(0, mem(r12, rsi, 2, 5*4)) // prefetch c + 2*cs_c
    prefetch(0, mem(r12, rcx, 1, 5*4)) // prefetch c + 3*cs_c
    prefetch(0, mem(r12, rsi, 4, 5*4)) // prefetch c + 4*cs_c
    lea(mem(r12, rsi, 4), rdx)         // rdx = c + 4*cs_c;
    prefetch(0, mem(rdx, rsi, 1, 5*4)) // prefetch c + 5*cs_c
    prefetch(0, mem(rdx, rsi, 2, 5*4)) // prefetch c + 6*cs_c

    label(.SPOSTPFETCH)                // done prefetching c

    mov(var(ps_a4), rdx)               // load ps_a4
    lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a4
    lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
                                       // use rcx, rdx for prefetching lines
                                       // from next upanel of a.

    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                       // contains the k_left loop.

    label(.SLOOPKITER)                 // MAIN LOOP

    // ---------------------------------- iteration 0

    prefetch(0, mem(rdx, 5*8))

    vmaskmovps(mem(rbx, 0), ymm3, ymm0) //load  required elements
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vbroadcastss(mem(rax, r13, 1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm10)

    vbroadcastss(mem(rax, r8,  4), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm12)
    vbroadcastss(mem(rax, r15, 1), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm14)

    // ---------------------------------- iteration 1

    prefetch(0, mem(rdx, r9, 1, 5*8))

    vmaskmovps(mem(rbx, 0), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vbroadcastss(mem(rax, r13, 1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm10)

    vbroadcastss(mem(rax, r8,  4), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm12)
    vbroadcastss(mem(rax, r15, 1), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm14)

    // ---------------------------------- iteration 2

    prefetch(0, mem(rdx, r9, 2, 5*8))

    vmaskmovps(mem(rbx, 0), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vbroadcastss(mem(rax, r13, 1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm10)

    vbroadcastss(mem(rax, r8,  4), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm12)
    vbroadcastss(mem(rax, r15, 1), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm14)

    // ---------------------------------- iteration 3

    prefetch(0, mem(rdx, rcx, 1, 5*8))
    lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;

    vmaskmovps(mem(rbx, 0), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vbroadcastss(mem(rax, r13, 1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm10)

    vbroadcastss(mem(rax, r8,  4), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm12)
    vbroadcastss(mem(rax, r15, 1), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm14)

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.

    label(.SCONSIDKLEFT)

    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                       // ee, we prepare to enter k_left loop.

    label(.SLOOPKLEFT)                 // EDGE LOOP

    prefetch(0, mem(rdx, 5*8))
    add(r9, rdx)

    vmaskmovps(mem(rbx, 0), ymm3, ymm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm4)
    vbroadcastss(mem(rax, r8,  1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm6)

    vbroadcastss(mem(rax, r8,  2), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm8)
    vbroadcastss(mem(rax, r13, 1), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm10)

    vbroadcastss(mem(rax, r8,  4), ymm2)
    vfmadd231ps(ymm0, ymm2, ymm12)
    vbroadcastss(mem(rax, r15, 1), ymm2)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(ymm0, ymm2, ymm14)

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.

    label(.SPOSTACCUM)

    mov(r12, rcx)                      // reset rcx to current utile of c.
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), ymm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), ymm1)       // load beta and duplicate

    vmulps(ymm0, ymm4, ymm4)           // scale by alpha
    vmulps(ymm0, ymm6, ymm6)
    vmulps(ymm0, ymm8, ymm8)
    vmulps(ymm0, ymm10, ymm10)
    vmulps(ymm0, ymm12, ymm12)
    vmulps(ymm0, ymm14, ymm14)

    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

    lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
    lea(mem(rsi, rsi, 4), rbx)         // rbx = 5*cs_c;

                                       // now avoid loading C if beta == 0

    vxorps(ymm0, ymm0, ymm0)           // set ymm0 to zero.
    vucomiss(xmm0, xmm1)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOTORED)                    // jump to column storage case

    label(.SROWSTORED)

    vmaskmovps(mem(rcx, 0), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm1, ymm4)
    vmaskmovps(ymm4, ymm3, mem(rcx, 0))

    add(rdi, rcx)

    vmaskmovps(mem(rcx, 0), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm1, ymm6)
    vmaskmovps(ymm6, ymm3, mem(rcx, 0))

    add(rdi, rcx)

    vmaskmovps(mem(rcx, 0), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm1, ymm8)
    vmaskmovps(ymm8, ymm3, mem(rcx, 0))

    add(rdi, rcx)

    vmaskmovps(mem(rcx, 0), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm1, ymm10)
    vmaskmovps(ymm10, ymm3, mem(rcx, 0))

    add(rdi, rcx)

    vmaskmovps(mem(rcx, 0), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm1, ymm12)
    vmaskmovps(ymm12, ymm3, mem(rcx, 0))
    add(rdi, rcx)

    vmaskmovps(mem(rcx, 0), ymm3, ymm2)
    vfmadd231ps(ymm2, ymm1, ymm14)
    vmaskmovps(ymm14, ymm3, mem(rcx, 0))

    jmp(.SDONE)                        // jump to end.

    label(.SCOTORED)

    /* TODO: Add column storage support*/

    jmp(.SDONE)                        // jump to end.

    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (4*rs_c) == 4.
    jz(.SCOTORBZ)                    // jump to column storage case

    label(.SROWSTORBZ)

    vmaskmovps(ymm4, ymm3, mem(rcx, 0))
    add(rdi, rcx)

    vmaskmovps(ymm6, ymm3, mem(rcx, 0))
    add(rdi, rcx)

    vmaskmovps(ymm8, ymm3, mem(rcx, 0))
    add(rdi, rcx)

    vmaskmovps(ymm10, ymm3, mem(rcx, 0))
    add(rdi, rcx)

    vmaskmovps(ymm12, ymm3, mem(rcx, 0))
    add(rdi, rcx)

    vmaskmovps(ymm14, ymm3, mem(rcx, 0))

    jmp(.SDONE)                        // jump to end.

    label(.SCOTORBZ)

    /* TODO: Add column storage support*/

    label(.SDONE)

    lea(mem(r12, rdi, 4), r12)         //
    lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

    mov(var(ps_a4), rax)               // load ps_a4
    lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a4

    dec(r11)                           // ii -= 1;
    jne(.SLOOP6X7I)                    // iterate again if ii != 0.

    label(.SRETURN)

    end_asm(
    : // output operands (none)
    : // input operands
      [m_iter]     "m"   (m_iter),
      [k_iter]     "m"   (k_iter),
      [k_left]     "m"   (k_left),
      [a]          "m"   (a),
      [rs_a]       "m"   (rs_a),
      [cs_a]       "m"   (cs_a),
      [ps_a4]      "m"   (ps_a4),
      [b]          "m"   (b),
      [rs_b]       "m"   (rs_b),
      [cs_b]       "m"   (cs_b),
      [alpha]      "m"   (alpha),
      [beta]       "m"   (beta),
      [c]          "m"   (c),
      [rs_c]       "m"   (rs_c),
      [cs_c]       "m"   (cs_c),
      [mask_vec]   "m"   (mask_vec)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1",
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm6",
      "ymm8", "ymm10", "ymm12", "ymm14",
      "memory"
    )

    consider_edge_cases:
    // Handle edge cases in the m dimension, if they exist.
    if (m_left )
    {
        const dim_t      nr_cur = n0;
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = (float *)c + i_edge*rs_c;
        float* restrict ai  = (float *)a + m_iter * ps_a;
        float* restrict bj  = (float *)b;

        gemmsup_ker_ft ker_fps[6] =
            {
              NULL,
              bli_sgemmsup_rv_zen_asm_1x8_mask,
              bli_sgemmsup_rv_zen_asm_2x8_mask,
              bli_sgemmsup_rv_zen_asm_3x8_mask,
              bli_sgemmsup_rv_zen_asm_4x8_mask,
              bli_sgemmsup_rv_zen_asm_5x8_mask
            };

        gemmsup_ker_ft ker_fp = ker_fps[ m_left ];

        ker_fp
        (
          conja, conjb, m_left, nr_cur, k0,
          alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
          beta, cij, rs_c0, cs_c0, data, cntx
        );

        return;
    }
}

void bli_sgemmsup_rv_zen_asm_6x4m_mask
     (
             conj_t     conja,
             conj_t     conjb,
             dim_t      m0,
             dim_t      n0,
             dim_t      k0,
       const void*      alpha,
       const void*      a0, inc_t rs_a0, inc_t cs_a0,
       const void*      b0, inc_t rs_b0, inc_t cs_b0,
       const void*      beta,
             void*      c0, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    float *a = (float *)a0;
    float *b = (float *)b0;
    float *c = (float *)c0;

    uint64_t k_iter = k0 / 4;
    uint64_t k_left = k0 % 4;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    // Query the panel stride of A and convert it to units of bytes.
    uint64_t ps_a   = bli_auxinfo_ps_a( data );
    uint64_t ps_a4  = ps_a * sizeof( float );

    // This kernel is called when n_left is 3/1. This kernel operates 4 columns at time.
    //
    // Sets up the mask for loading relevant remainder elements in load direction
    //
    // __xmm0___
    // | | | | |
    // |0|1|2|3|   ----> Source vector
    // |_|_|_|_|
    //
    // __xmm7___
    // | | | | |
    // |x|x|x|x|  ----> Mask vector[x can be -1/0]
    // |_|_|_|_|
    //
    // For example when n_left = 3
    // ___xmm7_____
    // |  |  |  | |
    // |-1|-1|-1|0|  ----> Mask vector
    // |__|__|__|_|
    //
    // For example when n_left = 1
    // ___xmm7___
    // |  | | | |
    // |-1|0|0|0|  ----> Mask vector
    // |__|_|_|_|
    //
    // __xmm0___
    // | | | | |
    // |0|1|2|3|  ----> Destination vector
    // |_|_|_|_|
    //
    const int32_t *mask_vec = mask[n0];

    if ( m_iter == 0 ) goto consider_edge_cases;

    // -------------------------------------------------------------------------
    begin_asm()

    mov(var(mask_vec), rdx)
    vmovdqu(mem(rdx), xmm7)           //load mask values
    mov(var(a), r14)                   // load address of a.
    mov(var(rs_a), r8)                 // load rs_a
    mov(var(cs_a), r9)                 // load cs_a
    lea(mem(, r8, 4), r8)              // rs_a *= sizeof(float)
    lea(mem(, r9, 4), r9)              // cs_a *= sizeof(float)

    lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
    lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

    mov(var(rs_b), r10)                // load rs_b
    lea(mem(, r10, 4), r10)            // rs_b *= sizeof(float)

                                       // NOTE: We cannot pre-load elements of a or b
                                       // because it could eventually, in the last
                                       // unrolled iter or the cleanup loop, result
                                       // in reading beyond the bounds allocated mem
                                       // (the likely result: a segmentation fault).

    mov(var(c), r12)                   // load address of c
    mov(var(rs_c), rdi)                // load rs_c
    lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)

    // During preamble and loops:
    // r12 = rcx = c    // r14 = rax = a
    // read rbx from var(b) near beginning of loop
    // r11 = m dim index ii

    mov(var(m_iter), r11)              // ii = m_iter;

    label(.SLOOP6X4I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]

    vxorps(xmm1,  xmm1,  xmm1)
    vxorps(xmm4,  xmm4,  xmm4)
    vxorps(xmm6,  xmm6,  xmm6)
    vxorps(xmm8,  xmm8,  xmm8)
    vxorps(xmm10, xmm10, xmm10)
    vxorps(xmm12, xmm12, xmm12)
    vxorps(xmm14, xmm14, xmm14)

    mov(var(b), rbx)                   // load address of b.
    mov(r14, rax)                      // reset rax to current upanel of a.

    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLPFETCH)                    // jump to column storage case
    label(.SROWPFETCH)                 // row-stored prefetching on c

    lea(mem(r12, rdi, 2), rdx)         //
    lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
    prefetch(0, mem(r12, 0))         // prefetch c + 0*rs_c
    prefetch(0, mem(r12, rdi, 1, 0)) // prefetch c + 1*rs_c
    prefetch(0, mem(r12, rdi, 2, 0)) // prefetch c + 2*rs_c
    prefetch(0, mem(rdx, 0))         // prefetch c + 3*rs_c
    prefetch(0, mem(rdx, rdi, 1, 0)) // prefetch c + 4*rs_c
    prefetch(0, mem(rdx, rdi, 2, 0)) // prefetch c + 5*rs_c

    jmp(.SPOSTPFETCH)                  // jump to end of prefetching c
    label(.SCOLPFETCH)                 // column-stored prefetching c

    mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
    lea(mem(, rsi, 4), rsi)            // cs_c *= sizeof(float)
    lea(mem(r12, rsi, 2), rdx)         //
    lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
    prefetch(0, mem(r12, 5*8))         // prefetch c + 0*cs_c
    prefetch(0, mem(r12, rsi, 1, 5*8)) // prefetch c + 1*cs_c
    prefetch(0, mem(r12, rsi, 2, 5*8)) // prefetch c + 2*cs_c
    prefetch(0, mem(rdx, 5*8))         // prefetch c + 3*cs_c
    prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 4*cs_c
    prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 5*cs_c

    label(.SPOSTPFETCH)                // done prefetching c

    lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
    lea(mem(rax, r8,  4), rdx)         // use rdx for prefetching lines
    lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.

    mov(var(k_iter), rsi)              // i = k_iter;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SCONSIDKLEFT)                  // if i == 0, jump to code that
                                       // contains the k_left loop.
    label(.SLOOPKITER)                 // MAIN LOOP

    // ---------------------------------- iteration 0
    prefetch(0, mem(rdx, 5*8))

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    vbroadcastss(mem(rax, r8,  4), xmm2)
    vbroadcastss(mem(rax, r15, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)
    vfmadd231ps(xmm0, xmm3, xmm14)

    // ---------------------------------- iteration 1
    prefetch(0, mem(rdx, r9, 1, 5*8))

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    vbroadcastss(mem(rax, r8,  4), xmm2)
    vbroadcastss(mem(rax, r15, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)
    vfmadd231ps(xmm0, xmm3, xmm14)

    // ---------------------------------- iteration 2
    prefetch(0, mem(rdx, r9, 2, 5*8))

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    vbroadcastss(mem(rax, r8,  4), xmm2)
    vbroadcastss(mem(rax, r15, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)
    vfmadd231ps(xmm0, xmm3, xmm14)

    // ---------------------------------- iteration 3
    prefetch(0, mem(rdx, rcx, 1, 5*8))

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    vbroadcastss(mem(rax, r8,  4), xmm2)
    vbroadcastss(mem(rax, r15, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)
    vfmadd231ps(xmm0, xmm3, xmm14)

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKITER)                   // iterate again if i != 0.

    label(.SCONSIDKLEFT)

    mov(var(k_left), rsi)              // i = k_left;
    test(rsi, rsi)                     // check i via logical AND.
    je(.SPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                       // else, we prepare to enter k_left loop.

    label(.SLOOPKLEFT)                 // EDGE LOOP

    vmaskmovps(mem(rbx), xmm7, xmm0)
    add(r10, rbx)                      // b += rs_b;

    vbroadcastss(mem(rax        ), xmm2)
    vbroadcastss(mem(rax, r8,  1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm4)
    vfmadd231ps(xmm0, xmm3, xmm6)

    vbroadcastss(mem(rax, r8,  2), xmm2)
    vbroadcastss(mem(rax, r13, 1), xmm3)
    vfmadd231ps(xmm0, xmm2, xmm8)
    vfmadd231ps(xmm0, xmm3, xmm10)

    vbroadcastss(mem(rax, r8,  4), xmm2)
    vbroadcastss(mem(rax, r15, 1), xmm3)
    add(r9, rax)                       // a += cs_a;
    vfmadd231ps(xmm0, xmm2, xmm12)
    vfmadd231ps(xmm0, xmm3, xmm14)

    dec(rsi)                           // i -= 1;
    jne(.SLOOPKLEFT)                   // iterate again if i != 0.

    label(.SPOSTACCUM)

    mov(r12, rcx)                      // reset rcx to current utile of c.
    mov(var(alpha), rax)               // load address of alpha
    mov(var(beta), rbx)                // load address of beta
    vbroadcastss(mem(rax), xmm0)       // load alpha and duplicate
    vbroadcastss(mem(rbx), xmm3)       // load beta and duplicate

    vmulps(xmm0, xmm4, xmm4)           // scale by alpha
    vmulps(xmm0, xmm6, xmm6)
    vmulps(xmm0, xmm8, xmm8)
    vmulps(xmm0, xmm10, xmm10)
    vmulps(xmm0, xmm12, xmm12)
    vmulps(xmm0, xmm14, xmm14)

    mov(var(cs_c), rsi)                // load cs_c
    lea(mem(, rsi, 4), rsi)            // rsi = cs_c * sizeof(float)

    lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;
    lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

                                       // now avoid loading C if beta == 0

    vxorps(xmm0, xmm0, xmm0)           // set xmm0 to zero.
    vucomiss(xmm0, xmm3)               // set ZF if beta == 0.
    je(.SBETAZERO)                     // if ZF = 1, jump to beta == 0 case


    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLSTORED)                    // jump to column storage case


    label(.SROWSTORED)

    vmaskmovps(mem(rcx), xmm7, xmm0)
    vfmadd231ps(xmm0, xmm3, xmm4)
    vmaskmovps(xmm4, xmm7, mem(rcx))
    add(rdi, rcx)

    vmaskmovps(mem(rcx), xmm7, xmm1)
    vfmadd231ps(xmm1, xmm3, xmm6)
    vmaskmovps(xmm6, xmm7, mem(rcx))
    add(rdi, rcx)

    vmaskmovps(mem(rcx), xmm7, xmm0)
    vfmadd231ps(xmm0, xmm3, xmm8)
    vmaskmovps(xmm8, xmm7, mem(rcx))
    add(rdi, rcx)

    vmaskmovps(mem(rcx), xmm7, xmm1)
    vfmadd231ps(xmm1, xmm3, xmm10)
    vmaskmovps(xmm10, xmm7, mem(rcx))
    add(rdi, rcx)

    vmaskmovps(mem(rcx), xmm7, xmm0)
    vfmadd231ps(xmm0, xmm3, xmm12)
    vmaskmovps(xmm12, xmm7, mem(rcx))
    add(rdi, rcx)

    vmaskmovps(mem(rcx), xmm7, xmm1)
    vfmadd231ps(xmm1, xmm3, xmm14)
    vmaskmovps(xmm14, xmm7, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORED)

    /* TODO: Add column storage support*/

    jmp(.SDONE)                        // jump to end.

    label(.SBETAZERO)

    cmp(imm(4), rdi)                   // set ZF if (8*rs_c) == 8.
    jz(.SCOLSTORBZ)                    // jump to column storage case

    label(.SROWSTORBZ)

    vmaskmovps(xmm4, xmm7, mem(rcx))
    add(rdi, rcx)
    vmaskmovps(xmm6, xmm7, mem(rcx))
    add(rdi, rcx)
    vmaskmovps(xmm8, xmm7, mem(rcx))
    add(rdi, rcx)
    vmaskmovps(xmm10, xmm7, mem(rcx))
    add(rdi, rcx)
    vmaskmovps(xmm12, xmm7, mem(rcx))
    add(rdi, rcx)
    vmaskmovps(xmm14, xmm7, mem(rcx))

    jmp(.SDONE)                        // jump to end.

    label(.SCOLSTORBZ)

    /* TODO: Add column storage support*/

    label(.SDONE)

    lea(mem(r12, rdi, 4), r12)         //
    lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

    mov(var(ps_a4), rax)               // load ps_a4
    lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a4

    dec(r11)                           // ii -= 1;
    jne(.SLOOP6X4I)                    // iterate again if ii != 0.

    label(.SRETURN)


    end_asm(
    : // output operands (none)
    : // input operands
      [m_iter]     "m"   (m_iter),
      [k_iter]     "m"   (k_iter),
      [k_left]     "m"   (k_left),
      [a]          "m"   (a),
      [rs_a]       "m"   (rs_a),
      [cs_a]       "m"   (cs_a),
      [ps_a4]      "m"   (ps_a4),
      [b]          "m"   (b),
      [rs_b]       "m"   (rs_b),
      [cs_b]       "m"   (cs_b),
      [alpha]      "m"   (alpha),
      [beta]       "m"   (beta),
      [c]          "m"   (c),
      [rs_c]       "m"   (rs_c),
      [cs_c]       "m"   (cs_c),
      [mask_vec]   "m"   (mask_vec)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "xmm0", "xmm1", "xmm2", "xmm3",
      "xmm4", "xmm6", "xmm7",
      "xmm8", "xmm10", "xmm12", "xmm14",
      "memory"
    )

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      nr_cur = n0;
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float* restrict cij = (float *)c + i_edge*rs_c;
        float* restrict ai  = (float *)a + m_iter*ps_a;
        float* restrict bj  = (float *)b;

        gemmsup_ker_ft ker_fps[6] =
        {
          NULL,
          bli_sgemmsup_rv_zen_asm_1x4_mask,
          bli_sgemmsup_rv_zen_asm_2x4_mask,
          bli_sgemmsup_rv_zen_asm_3x4_mask,
          bli_sgemmsup_rv_zen_asm_4x4_mask,
          bli_sgemmsup_rv_zen_asm_5x4_mask
        };

        gemmsup_ker_ft ker_fp = ker_fps[ m_left ];

        ker_fp
        (
          conja, conjb, m_left, nr_cur, k0,
          alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
          beta, cij, rs_c0, cs_c0, data, cntx
        );
        return;
    }
}
