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

// These are macros are used for int4 packing

// zero out 1 nibbles struct
#define zero_out_full(dest) \
    dest->v = 0; \
    dest++;

// zero out 4 nibbles struct
#define zero_out_dest(dest) \
    memset(dest, 0, 4);


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//////////////////////////// Col Major Order Macros ////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*

    The following macros handle the case when there is a full size panel 
    (ib/jb == MR/NR) and no edge case (k%8 == 0).

*/

#define col_m_order_1(dest, matrix, rs_mul, rs, cs) \
    dest->bits.nib1 = matrix[rs_mul*rs + (p_idx+0)*cs].bits.nib1; \
    dest->bits.nib2 = matrix[rs_mul*rs + (p_idx+1)*cs].bits.nib1; \
    dest++; \
    dest->bits.nib1 = matrix[rs_mul*rs + (p_idx+2)*cs].bits.nib1; \
    dest->bits.nib2 = matrix[rs_mul*rs + (p_idx+3)*cs].bits.nib1; \
    dest++; \
    dest->bits.nib1 = matrix[rs_mul*rs + (p_idx+4)*cs].bits.nib1; \
    dest->bits.nib2 = matrix[rs_mul*rs + (p_idx+5)*cs].bits.nib1; \
    dest++; \
    dest->bits.nib1 = matrix[rs_mul*rs + (p_idx+6)*cs].bits.nib1; \
    dest->bits.nib2 = matrix[rs_mul*rs + (p_idx+7)*cs].bits.nib1; \
    dest++;

#define col_m_order_2(dest, matrix, rs_mul, rs, cs) \
    dest->bits.nib1 = matrix[rs_mul*rs + (p_idx+0)*cs].bits.nib2; \
    dest->bits.nib2 = matrix[rs_mul*rs + (p_idx+1)*cs].bits.nib2; \
    dest++; \
    dest->bits.nib1 = matrix[rs_mul*rs + (p_idx+2)*cs].bits.nib2; \
    dest->bits.nib2 = matrix[rs_mul*rs + (p_idx+3)*cs].bits.nib2; \
    dest++; \
    dest->bits.nib1 = matrix[rs_mul*rs + (p_idx+4)*cs].bits.nib2; \
    dest->bits.nib2 = matrix[rs_mul*rs + (p_idx+5)*cs].bits.nib2; \
    dest++; \
    dest->bits.nib1 = matrix[rs_mul*rs + (p_idx+6)*cs].bits.nib2; \
    dest->bits.nib2 = matrix[rs_mul*rs + (p_idx+7)*cs].bits.nib2; \
    dest++;

/*

    The following macros handle the case when there is a full size panel 
    (ib/jb == MR/NR) and there is an edge case (k%8 != 0).

*/

#define col_m_order_1_kleft7(dest, matrix, rs_mul, rs, cs) \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-7)*cs].bits.nib1; \
    dest->bits.nib2 = matrix[rs_mul*rs + (k-6)*cs].bits.nib1; \
    dest++; \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-5)*cs].bits.nib1; \
    dest->bits.nib2 = matrix[rs_mul*rs + (k-4)*cs].bits.nib1; \
    dest++; \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-3)*cs].bits.nib1; \
    dest->bits.nib2 = matrix[rs_mul*rs + (k-2)*cs].bits.nib1; \
    dest++; \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-1)*cs].bits.nib1; \
    dest->bits.nib2 = 0; \
    dest++;

#define col_m_order_2_kleft7(dest, matrix, rs_mul, rs, cs) \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-7)*cs].bits.nib2; \
    dest->bits.nib2 = matrix[rs_mul*rs + (k-6)*cs].bits.nib2; \
    dest++; \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-5)*cs].bits.nib2; \
    dest->bits.nib2 = matrix[rs_mul*rs + (k-4)*cs].bits.nib2; \
    dest++; \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-3)*cs].bits.nib2; \
    dest->bits.nib2 = matrix[rs_mul*rs + (k-2)*cs].bits.nib2; \
    dest++; \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-1)*cs].bits.nib2; \
    dest->bits.nib2 = 0; \
    dest++;

#define col_m_order_1_kleft6(dest, matrix, rs_mul, rs, cs) \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-6)*cs].bits.nib1; \
    dest->bits.nib2 = matrix[rs_mul*rs + (k-5)*cs].bits.nib1; \
    dest++; \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-4)*cs].bits.nib1; \
    dest->bits.nib2 = matrix[rs_mul*rs + (k-3)*cs].bits.nib1; \
    dest++; \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-2)*cs].bits.nib1; \
    dest->bits.nib2 = matrix[rs_mul*rs + (k-1)*cs].bits.nib1; \
    dest++; \
    zero_out_full(dest);

#define col_m_order_2_kleft6(dest, matrix, rs_mul, rs, cs) \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-6)*cs].bits.nib2; \
    dest->bits.nib2 = matrix[rs_mul*rs + (k-5)*cs].bits.nib2; \
    dest++; \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-4)*cs].bits.nib2; \
    dest->bits.nib2 = matrix[rs_mul*rs + (k-3)*cs].bits.nib2; \
    dest++; \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-2)*cs].bits.nib2; \
    dest->bits.nib2 = matrix[rs_mul*rs + (k-1)*cs].bits.nib2; \
    dest++; \
    zero_out_full(dest);

#define col_m_order_1_kleft5(dest, matrix, rs_mul, rs, cs) \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-5)*cs].bits.nib1; \
    dest->bits.nib2 = matrix[rs_mul*rs + (k-4)*cs].bits.nib1; \
    dest++; \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-3)*cs].bits.nib1; \
    dest->bits.nib2 = matrix[rs_mul*rs + (k-2)*cs].bits.nib1; \
    dest++; \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-1)*cs].bits.nib1; \
    dest->bits.nib2 = 0; \
    dest++; \
    zero_out_full(dest);

#define col_m_order_2_kleft5(dest, matrix, rs_mul, rs, cs) \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-5)*cs].bits.nib2; \
    dest->bits.nib2 = matrix[rs_mul*rs + (k-4)*cs].bits.nib2; \
    dest++; \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-3)*cs].bits.nib2; \
    dest->bits.nib2 = matrix[rs_mul*rs + (k-2)*cs].bits.nib2; \
    dest++; \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-1)*cs].bits.nib2; \
    dest->bits.nib2 = 0; \
    dest++; \
    zero_out_full(dest);

#define col_m_order_1_kleft4(dest, matrix, rs_mul, rs, cs) \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-4)*cs].bits.nib1; \
    dest->bits.nib2 = matrix[rs_mul*rs + (k-3)*cs].bits.nib1; \
    dest++; \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-2)*cs].bits.nib1; \
    dest->bits.nib2 = matrix[rs_mul*rs + (k-1)*cs].bits.nib1; \
    dest++; \
    zero_out_full(dest); \
    zero_out_full(dest);

#define col_m_order_2_kleft4(dest, matrix, rs_mul, rs, cs) \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-4)*cs].bits.nib2; \
    dest->bits.nib2 = matrix[rs_mul*rs + (k-3)*cs].bits.nib2; \
    dest++; \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-2)*cs].bits.nib2; \
    dest->bits.nib2 = matrix[rs_mul*rs + (k-1)*cs].bits.nib2; \
    dest++; \
    zero_out_full(dest); \
    zero_out_full(dest);

#define col_m_order_1_kleft3(dest, matrix, rs_mul, rs, cs) \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-3)*cs].bits.nib1; \
    dest->bits.nib2 = matrix[rs_mul*rs + (k-2)*cs].bits.nib1; \
    dest++; \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-1)*cs].bits.nib1; \
    dest->bits.nib2 = 0; \
    dest++; \
    zero_out_full(dest); \
    zero_out_full(dest);

#define col_m_order_2_kleft3(dest, matrix, rs_mul, rs, cs) \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-3)*cs].bits.nib2; \
    dest->bits.nib2 = matrix[rs_mul*rs + (k-2)*cs].bits.nib2; \
    dest++; \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-1)*cs].bits.nib2; \
    dest->bits.nib2 = 0; \
    dest++; \
    zero_out_full(dest); \
    zero_out_full(dest);

#define col_m_order_1_kleft2(dest, matrix, rs_mul, rs, cs) \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-2)*cs].bits.nib1; \
    dest->bits.nib2 = matrix[rs_mul*rs + (k-1)*cs].bits.nib1; \
    dest++; \
    zero_out_full(dest); \
    zero_out_full(dest); \
    zero_out_full(dest);

#define col_m_order_2_kleft2(dest, matrix, rs_mul, rs, cs) \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-2)*cs].bits.nib2; \
    dest->bits.nib2 = matrix[rs_mul*rs + (k-1)*cs].bits.nib2; \
    dest++; \
    zero_out_full(dest); \
    zero_out_full(dest); \
    zero_out_full(dest);

#define col_m_order_1_kleft1(dest, matrix, rs_mul, rs, cs) \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-1)*cs].bits.nib1; \
    dest->bits.nib2 = 0; \
    dest++; \
    zero_out_full(dest); \
    zero_out_full(dest); \
    zero_out_full(dest);

#define col_m_order_2_kleft1(dest, matrix, rs_mul, rs, cs) \
    dest->bits.nib1 = matrix[rs_mul*rs + (k-1)*cs].bits.nib2; \
    dest->bits.nib2 = 0; \
    dest++; \
    zero_out_full(dest); \
    zero_out_full(dest); \
    zero_out_full(dest);

/*


    The following macros are used when we have a full panel (ib == MR) 
    and we need to handle an edge case (k%8 != 0).

    The MR loop is unrolled resulting in the stream of macros.

*/

#define apad_col_kleft7(dest, matrix, rs, cs) \
    col_m_order_1_kleft7(dest, matrix, (i  ), rs, cs); \
    col_m_order_2_kleft7(dest, matrix, (i  ), rs, cs); \
    col_m_order_1_kleft7(dest, matrix, (i+1), rs, cs); \
    col_m_order_2_kleft7(dest, matrix, (i+1), rs, cs); \
    col_m_order_1_kleft7(dest, matrix, (i+2), rs, cs); \
    col_m_order_2_kleft7(dest, matrix, (i+2), rs, cs); \
    col_m_order_1_kleft7(dest, matrix, (i+3), rs, cs); \
    col_m_order_2_kleft7(dest, matrix, (i+3), rs, cs); 

#define apad_col_kleft6(dest, matrix, rs, cs) \
    col_m_order_1_kleft6(dest, matrix, (i  ), rs, cs); \
    col_m_order_2_kleft6(dest, matrix, (i  ), rs, cs); \
    col_m_order_1_kleft6(dest, matrix, (i+1), rs, cs); \
    col_m_order_2_kleft6(dest, matrix, (i+1), rs, cs); \
    col_m_order_1_kleft6(dest, matrix, (i+2), rs, cs); \
    col_m_order_2_kleft6(dest, matrix, (i+2), rs, cs); \
    col_m_order_1_kleft6(dest, matrix, (i+3), rs, cs); \
    col_m_order_2_kleft6(dest, matrix, (i+3), rs, cs); 

#define apad_col_kleft5(dest, matrix, rs, cs) \
    col_m_order_1_kleft5(dest, matrix, (i  ), rs, cs); \
    col_m_order_2_kleft5(dest, matrix, (i  ), rs, cs); \
    col_m_order_1_kleft5(dest, matrix, (i+1), rs, cs); \
    col_m_order_2_kleft5(dest, matrix, (i+1), rs, cs); \
    col_m_order_1_kleft5(dest, matrix, (i+2), rs, cs); \
    col_m_order_2_kleft5(dest, matrix, (i+2), rs, cs); \
    col_m_order_1_kleft5(dest, matrix, (i+3), rs, cs); \
    col_m_order_2_kleft5(dest, matrix, (i+3), rs, cs); 

#define apad_col_kleft4(dest, matrix, rs, cs) \
    col_m_order_1_kleft4(dest, matrix, (i  ), rs, cs); \
    col_m_order_2_kleft4(dest, matrix, (i  ), rs, cs); \
    col_m_order_1_kleft4(dest, matrix, (i+1), rs, cs); \
    col_m_order_2_kleft4(dest, matrix, (i+1), rs, cs); \
    col_m_order_1_kleft4(dest, matrix, (i+2), rs, cs); \
    col_m_order_2_kleft4(dest, matrix, (i+2), rs, cs); \
    col_m_order_1_kleft4(dest, matrix, (i+3), rs, cs); \
    col_m_order_2_kleft4(dest, matrix, (i+3), rs, cs); 

#define apad_col_kleft3(dest, matrix, rs, cs) \
    col_m_order_1_kleft3(dest, matrix, (i  ), rs, cs); \
    col_m_order_2_kleft3(dest, matrix, (i  ), rs, cs); \
    col_m_order_1_kleft3(dest, matrix, (i+1), rs, cs); \
    col_m_order_2_kleft3(dest, matrix, (i+1), rs, cs); \
    col_m_order_1_kleft3(dest, matrix, (i+2), rs, cs); \
    col_m_order_2_kleft3(dest, matrix, (i+2), rs, cs); \
    col_m_order_1_kleft3(dest, matrix, (i+3), rs, cs); \
    col_m_order_2_kleft3(dest, matrix, (i+3), rs, cs); 

#define apad_col_kleft2(dest, matrix, rs, cs) \
    col_m_order_1_kleft2(dest, matrix, (i  ), rs, cs); \
    col_m_order_2_kleft2(dest, matrix, (i  ), rs, cs); \
    col_m_order_1_kleft2(dest, matrix, (i+1), rs, cs); \
    col_m_order_2_kleft2(dest, matrix, (i+1), rs, cs); \
    col_m_order_1_kleft2(dest, matrix, (i+2), rs, cs); \
    col_m_order_2_kleft2(dest, matrix, (i+2), rs, cs); \
    col_m_order_1_kleft2(dest, matrix, (i+3), rs, cs); \
    col_m_order_2_kleft2(dest, matrix, (i+3), rs, cs); 

#define apad_col_kleft1(dest, matrix, rs, cs) \
    col_m_order_1_kleft1(dest, matrix, (i  ), rs, cs); \
    col_m_order_2_kleft1(dest, matrix, (i  ), rs, cs); \
    col_m_order_1_kleft1(dest, matrix, (i+1), rs, cs); \
    col_m_order_2_kleft1(dest, matrix, (i+1), rs, cs); \
    col_m_order_1_kleft1(dest, matrix, (i+2), rs, cs); \
    col_m_order_2_kleft1(dest, matrix, (i+2), rs, cs); \
    col_m_order_1_kleft1(dest, matrix, (i+3), rs, cs); \
    col_m_order_2_kleft1(dest, matrix, (i+3), rs, cs); 

/*

    The following macros are used when we have a full panel (jb == NR) 
    and we need to handle an edge case (k%8 != 0).

    The NR loop is unrolled resulting in the stream of macros.

*/

#define bpad_col_kleft7(dest, matrix, rs, cs) \
    col_m_order_1_kleft7(dest, matrix, (j  ), rs, cs); \
    col_m_order_2_kleft7(dest, matrix, (j  ), rs, cs); \
    col_m_order_1_kleft7(dest, matrix, (j+1), rs, cs); \
    col_m_order_2_kleft7(dest, matrix, (j+1), rs, cs); \
    col_m_order_1_kleft7(dest, matrix, (j+2), rs, cs); \
    col_m_order_2_kleft7(dest, matrix, (j+2), rs, cs); \
    col_m_order_1_kleft7(dest, matrix, (j+3), rs, cs); \
    col_m_order_2_kleft7(dest, matrix, (j+3), rs, cs); \
    col_m_order_1_kleft7(dest, matrix, (j+4), rs, cs); \
    col_m_order_2_kleft7(dest, matrix, (j+4), rs, cs); \
    col_m_order_1_kleft7(dest, matrix, (j+5), rs, cs); \
    col_m_order_2_kleft7(dest, matrix, (j+5), rs, cs); \
    col_m_order_1_kleft7(dest, matrix, (j+6), rs, cs); \
    col_m_order_2_kleft7(dest, matrix, (j+6), rs, cs); \
    col_m_order_1_kleft7(dest, matrix, (j+7), rs, cs); \
    col_m_order_2_kleft7(dest, matrix, (j+7), rs, cs); 

#define bpad_col_kleft6(dest, matrix, rs, cs) \
    col_m_order_1_kleft6(dest, matrix, (j  ), rs, cs); \
    col_m_order_2_kleft6(dest, matrix, (j  ), rs, cs); \
    col_m_order_1_kleft6(dest, matrix, (j+1), rs, cs); \
    col_m_order_2_kleft6(dest, matrix, (j+1), rs, cs); \
    col_m_order_1_kleft6(dest, matrix, (j+2), rs, cs); \
    col_m_order_2_kleft6(dest, matrix, (j+2), rs, cs); \
    col_m_order_1_kleft6(dest, matrix, (j+3), rs, cs); \
    col_m_order_2_kleft6(dest, matrix, (j+3), rs, cs); \
    col_m_order_1_kleft6(dest, matrix, (j+4), rs, cs); \
    col_m_order_2_kleft6(dest, matrix, (j+4), rs, cs); \
    col_m_order_1_kleft6(dest, matrix, (j+5), rs, cs); \
    col_m_order_2_kleft6(dest, matrix, (j+5), rs, cs); \
    col_m_order_1_kleft6(dest, matrix, (j+6), rs, cs); \
    col_m_order_2_kleft6(dest, matrix, (j+6), rs, cs); \
    col_m_order_1_kleft6(dest, matrix, (j+7), rs, cs); \
    col_m_order_2_kleft6(dest, matrix, (j+7), rs, cs); 

#define bpad_col_kleft5(dest, matrix, rs, cs) \
    col_m_order_1_kleft5(dest, matrix, (j  ), rs, cs); \
    col_m_order_2_kleft5(dest, matrix, (j  ), rs, cs); \
    col_m_order_1_kleft5(dest, matrix, (j+1), rs, cs); \
    col_m_order_2_kleft5(dest, matrix, (j+1), rs, cs); \
    col_m_order_1_kleft5(dest, matrix, (j+2), rs, cs); \
    col_m_order_2_kleft5(dest, matrix, (j+2), rs, cs); \
    col_m_order_1_kleft5(dest, matrix, (j+3), rs, cs); \
    col_m_order_2_kleft5(dest, matrix, (j+3), rs, cs); \
    col_m_order_1_kleft5(dest, matrix, (j+4), rs, cs); \
    col_m_order_2_kleft5(dest, matrix, (j+4), rs, cs); \
    col_m_order_1_kleft5(dest, matrix, (j+5), rs, cs); \
    col_m_order_2_kleft5(dest, matrix, (j+5), rs, cs); \
    col_m_order_1_kleft5(dest, matrix, (j+6), rs, cs); \
    col_m_order_2_kleft5(dest, matrix, (j+6), rs, cs); \
    col_m_order_1_kleft5(dest, matrix, (j+7), rs, cs); \
    col_m_order_2_kleft5(dest, matrix, (j+7), rs, cs); 

#define bpad_col_kleft4(dest, matrix, rs, cs) \
    col_m_order_1_kleft4(dest, matrix, (j  ), rs, cs); \
    col_m_order_2_kleft4(dest, matrix, (j  ), rs, cs); \
    col_m_order_1_kleft4(dest, matrix, (j+1), rs, cs); \
    col_m_order_2_kleft4(dest, matrix, (j+1), rs, cs); \
    col_m_order_1_kleft4(dest, matrix, (j+2), rs, cs); \
    col_m_order_2_kleft4(dest, matrix, (j+2), rs, cs); \
    col_m_order_1_kleft4(dest, matrix, (j+3), rs, cs); \
    col_m_order_2_kleft4(dest, matrix, (j+3), rs, cs); \
    col_m_order_1_kleft4(dest, matrix, (j+4), rs, cs); \
    col_m_order_2_kleft4(dest, matrix, (j+4), rs, cs); \
    col_m_order_1_kleft4(dest, matrix, (j+5), rs, cs); \
    col_m_order_2_kleft4(dest, matrix, (j+5), rs, cs); \
    col_m_order_1_kleft4(dest, matrix, (j+6), rs, cs); \
    col_m_order_2_kleft4(dest, matrix, (j+6), rs, cs); \
    col_m_order_1_kleft4(dest, matrix, (j+7), rs, cs); \
    col_m_order_2_kleft4(dest, matrix, (j+7), rs, cs); 

#define bpad_col_kleft3(dest, matrix, rs, cs) \
    col_m_order_1_kleft3(dest, matrix, (j  ), rs, cs); \
    col_m_order_2_kleft3(dest, matrix, (j  ), rs, cs); \
    col_m_order_1_kleft3(dest, matrix, (j+1), rs, cs); \
    col_m_order_2_kleft3(dest, matrix, (j+1), rs, cs); \
    col_m_order_1_kleft3(dest, matrix, (j+2), rs, cs); \
    col_m_order_2_kleft3(dest, matrix, (j+2), rs, cs); \
    col_m_order_1_kleft3(dest, matrix, (j+3), rs, cs); \
    col_m_order_2_kleft3(dest, matrix, (j+3), rs, cs); \
    col_m_order_1_kleft3(dest, matrix, (j+4), rs, cs); \
    col_m_order_2_kleft3(dest, matrix, (j+4), rs, cs); \
    col_m_order_1_kleft3(dest, matrix, (j+5), rs, cs); \
    col_m_order_2_kleft3(dest, matrix, (j+5), rs, cs); \
    col_m_order_1_kleft3(dest, matrix, (j+6), rs, cs); \
    col_m_order_2_kleft3(dest, matrix, (j+6), rs, cs); \
    col_m_order_1_kleft3(dest, matrix, (j+7), rs, cs); \
    col_m_order_2_kleft3(dest, matrix, (j+7), rs, cs);

#define bpad_col_kleft2(dest, matrix, rs, cs) \
    col_m_order_1_kleft2(dest, matrix, (j  ), rs, cs); \
    col_m_order_2_kleft2(dest, matrix, (j  ), rs, cs); \
    col_m_order_1_kleft2(dest, matrix, (j+1), rs, cs); \
    col_m_order_2_kleft2(dest, matrix, (j+1), rs, cs); \
    col_m_order_1_kleft2(dest, matrix, (j+2), rs, cs); \
    col_m_order_2_kleft2(dest, matrix, (j+2), rs, cs); \
    col_m_order_1_kleft2(dest, matrix, (j+3), rs, cs); \
    col_m_order_2_kleft2(dest, matrix, (j+3), rs, cs); \
    col_m_order_1_kleft2(dest, matrix, (j+4), rs, cs); \
    col_m_order_2_kleft2(dest, matrix, (j+4), rs, cs); \
    col_m_order_1_kleft2(dest, matrix, (j+5), rs, cs); \
    col_m_order_2_kleft2(dest, matrix, (j+5), rs, cs); \
    col_m_order_1_kleft2(dest, matrix, (j+6), rs, cs); \
    col_m_order_2_kleft2(dest, matrix, (j+6), rs, cs); \
    col_m_order_1_kleft2(dest, matrix, (j+7), rs, cs); \
    col_m_order_2_kleft2(dest, matrix, (j+7), rs, cs);

#define bpad_col_kleft1(dest, matrix, rs, cs) \
    col_m_order_1_kleft1(dest, matrix, (j  ), rs, cs); \
    col_m_order_2_kleft1(dest, matrix, (j  ), rs, cs); \
    col_m_order_1_kleft1(dest, matrix, (j+1), rs, cs); \
    col_m_order_2_kleft1(dest, matrix, (j+1), rs, cs); \
    col_m_order_1_kleft1(dest, matrix, (j+2), rs, cs); \
    col_m_order_2_kleft1(dest, matrix, (j+2), rs, cs); \
    col_m_order_1_kleft1(dest, matrix, (j+3), rs, cs); \
    col_m_order_2_kleft1(dest, matrix, (j+3), rs, cs); \
    col_m_order_1_kleft1(dest, matrix, (j+4), rs, cs); \
    col_m_order_2_kleft1(dest, matrix, (j+4), rs, cs); \
    col_m_order_1_kleft1(dest, matrix, (j+5), rs, cs); \
    col_m_order_2_kleft1(dest, matrix, (j+5), rs, cs); \
    col_m_order_1_kleft1(dest, matrix, (j+6), rs, cs); \
    col_m_order_2_kleft1(dest, matrix, (j+6), rs, cs); \
    col_m_order_1_kleft1(dest, matrix, (j+7), rs, cs); \
    col_m_order_2_kleft1(dest, matrix, (j+7), rs, cs);


/*

    The following macros handle non full size panels (ib/jb != MR/NR) and 
    edge cases (k%8 != 0).

*/

#define edge(edgefun, dest, matrix, panel, left, rs, cs) \
    for (int ir=0; ir<left; ir++) { \
        if (ir%2==0) { \
            col_m_order_1_ ## edgefun ## (dest, matrix, (panel+ir/2), rs, cs); \
        } \
        else { \
            col_m_order_2_ ## edgefun ## (dest, matrix, (panel+ir/2), rs, cs); \
        } \
    } 

#define edge7(dest, matrix, panel, left, rs, cs) \
    for (int ir=0; ir<left; ir++) { \
        if (ir%2==0) { \
            col_m_order_1_kleft7(dest, matrix, (panel+ir/2), rs, cs); \
        } \
        else { \
            col_m_order_2_kleft7(dest, matrix, (panel+ir/2), rs, cs); \
        } \
    }

#define edge6(dest, matrix, panel, left, rs, cs) \
    for (int ir=0; ir<left; ir++) { \
        if (ir%2==0) { \
            col_m_order_1_kleft6(dest, matrix, (panel+ir/2), rs, cs); \
        } \
        else { \
            col_m_order_2_kleft6(dest, matrix, (panel+ir/2), rs, cs); \
        } \
    }

#define edge5(dest, matrix, panel, left, rs, cs) \
    for (int ir=0; ir<left; ir++) { \
        if (ir%2==0) { \
            col_m_order_1_kleft5(dest, matrix, (panel+ir/2), rs, cs); \
        } \
        else { \
            col_m_order_2_kleft5(dest, matrix, (panel+ir/2), rs, cs); \
        } \
    }

#define edge4(dest, matrix, panel, left, rs, cs) \
    for (int ir=0; ir<left; ir++) { \
        if (ir%2==0) { \
            col_m_order_1_kleft4(dest, matrix, (panel+ir/2), rs, cs); \
        } \
        else { \
            col_m_order_2_kleft4(dest, matrix, (panel+ir/2), rs, cs); \
        } \
    }

#define edge3(dest, matrix, panel, left, rs, cs) \
    for (int ir=0; ir<left; ir++) { \
        if (ir%2==0) { \
            col_m_order_1_kleft3(dest, matrix, (panel+ir/2), rs, cs); \
        } \
        else { \
            col_m_order_2_kleft3(dest, matrix, (panel+ir/2), rs, cs); \
        } \
    }

#define edge2(dest, matrix, panel, left, rs, cs) \
    for (int ir=0; ir<left; ir++) { \
        if (ir%2==0) { \
            col_m_order_1_kleft2(dest, matrix, (panel+ir/2), rs, cs); \
        } \
        else { \
            col_m_order_2_kleft2(dest, matrix, (panel+ir/2), rs, cs); \
        } \
    }

#define edge1(dest, matrix, panel, left, rs, cs) \
    for (int ir=0; ir<left; ir++) { \
        if (ir%2==0) { \
            col_m_order_1_kleft1(dest, matrix, (panel+ir/2), rs, cs); \
        } \
        else { \
            col_m_order_2_kleft1(dest, matrix, (panel+ir/2), rs, cs); \
        } \
    }
    