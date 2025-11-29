/*

BLIS
An object-based framework for developing high-performance BLAS-like
libraries.

Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
- Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
- Neither the name of The University of Texas at Austin nor the names
of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "immintrin.h"
#include "xmmintrin.h"
#include "blis.h"

#ifdef BLIS_ENABLE_SMALL_MATRIX

#define MR 32
#define D_MR (MR >> 1)
#define NR 3

#define BLIS_ENABLE_PREFETCH
#define F_SCRATCH_DIM (BLIS_SMALL_MATRIX_THRES * BLIS_SMALL_MATRIX_THRES)
static float A_pack[F_SCRATCH_DIM]  __attribute__((aligned(64)));
static float C_pack[F_SCRATCH_DIM]  __attribute__((aligned(64)));
#define D_BLIS_SMALL_MATRIX_THRES (BLIS_SMALL_MATRIX_THRES / 2 )
#define D_BLIS_SMALL_M_RECT_MATRIX_THRES (BLIS_SMALL_M_RECT_MATRIX_THRES / 2)
#define D_BLIS_SMALL_K_RECT_MATRIX_THRES (BLIS_SMALL_K_RECT_MATRIX_THRES / 2)
#define D_SCRATCH_DIM (D_BLIS_SMALL_MATRIX_THRES * D_BLIS_SMALL_MATRIX_THRES)
static double D_A_pack[D_SCRATCH_DIM]  __attribute__((aligned(64)));
static double D_C_pack[D_SCRATCH_DIM]  __attribute__((aligned(64)));
#define BLIS_ATBN_M_THRES 40 // Threshold value of M for/below which small matrix code is called.
#define AT_MR 4 // The kernel dimension of the A transpose GEMMT kernel.(AT_MR * NR).
static err_t bli_sgemmt_small
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       cntl_t* cntl
     );

static err_t bli_dgemmt_small
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       cntl_t* cntl
     );

static err_t bli_sgemmt_small_atbn
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       cntl_t* cntl
     );

static err_t bli_dgemmt_small_atbn
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       cntl_t* cntl
     );
/*
* The bli_gemmt_small function will use the
* custom MRxNR kernels, to perform the computation.
* The custom kernels are used if the [M * N] < 240 * 240
*/
err_t bli_gemmt_small
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       cntl_t* cntl
     )
{
	// FGVZ: This code was originally in bli_gemmt_front(). However, it really
	// fits more naturally here within the bli_gemmt_small() function. This
	// becomes a bit more obvious now that the code is here, as it contains
	// cpp macros such as BLIS_SMALL_MATRIX_A_THRES_M_GEMMT, which are specific
	// to this implementation.
	if ( bli_obj_has_trans( a ) )
	{
		// Continue with small implementation.
		;
	}
	else if ( ( bli_obj_length( a ) <= BLIS_SMALL_MATRIX_A_THRES_M_GEMMT &&
	            bli_obj_width( a )  <  BLIS_SMALL_MATRIX_A_THRES_N_GEMMT ) ||
	          ( bli_obj_length( a ) <  BLIS_SMALL_MATRIX_A_THRES_M_GEMMT &&
	            bli_obj_width( a )  <= BLIS_SMALL_MATRIX_A_THRES_N_GEMMT ) )
	{
		// Continue with small implementation.
		;
	}
	else
	{
		// Reject the problem and return to large code path.
		return BLIS_FAILURE;
	}

#ifdef BLIS_ENABLE_MULTITHREADING
    return BLIS_NOT_YET_IMPLEMENTED;
#endif
    // If alpha is zero, scale by beta and return.
    if (bli_obj_equals(alpha, &BLIS_ZERO))
    {
        return BLIS_NOT_YET_IMPLEMENTED;
    }

    // if row major format return.
    if ((bli_obj_row_stride( a ) != 1) ||
        (bli_obj_row_stride( b ) != 1) ||
        (bli_obj_row_stride( c ) != 1))
    {
        return BLIS_INVALID_ROW_STRIDE;
    }

    num_t dt = ((*c).info & (0x7 << 0));

    if (bli_obj_has_trans( a ))
    {
        if (bli_obj_has_notrans( b ))
        {
            if (dt == BLIS_FLOAT)
            {
                return bli_sgemmt_small_atbn(alpha, a, b, beta, c, cntx, cntl);
            }
            else if (dt == BLIS_DOUBLE)
            {
                return bli_dgemmt_small_atbn(alpha, a, b, beta, c, cntx, cntl);
            }
        }

        return BLIS_NOT_YET_IMPLEMENTED;
    }

    if (dt == BLIS_DOUBLE)
    {
        return bli_dgemmt_small(alpha, a, b, beta, c, cntx, cntl);
    }

    if (dt == BLIS_FLOAT)
    {
        return bli_sgemmt_small(alpha, a, b, beta, c, cntx, cntl);
    }

    return BLIS_NOT_YET_IMPLEMENTED;
};


static err_t bli_sgemmt_small
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       cntl_t* cntl
     )
{

    int M = bli_obj_length( c ); // number of rows of Matrix C
    int N = bli_obj_width( c );  // number of columns of Matrix C
    int K = bli_obj_width( a );  // number of columns of OP(A), will be updated if OP(A) is Transpose(A) .
    int L = M * N;

     if ((((L) < (BLIS_SMALL_MATRIX_THRES * BLIS_SMALL_MATRIX_THRES))
        || ((M  < BLIS_SMALL_M_RECT_MATRIX_THRES) && (K < BLIS_SMALL_K_RECT_MATRIX_THRES))) && ((L!=0) && (K!=0)))
    {

        int lda = bli_obj_col_stride(a); // column stride of matrix OP(A), where OP(A) is Transpose(A) if transA enabled.
        int ldb = bli_obj_col_stride(b); // column stride of matrix OP(B), where OP(B) is Transpose(B) if transB enabled.
        int ldc_matC = bli_obj_col_stride( c ); // column stride of matrix C
        int ldc = M;//bli_obj_col_stride( c ); // column stride of static buffer for matrix C
        int row_idx, col_idx, k;
        int rs_matC = bli_obj_row_stride( c );
        int rsc = 1;
        float *A = a->buffer; // pointer to elements of Matrix A
        float *B = b->buffer; // pointer to elements of Matrix B
        float *C = C_pack; // pointer to elements of Matrix C
        float *matCbuf = c->buffer;

        float *tA = A, *tB = B, *tC = C;//, *tA_pack;
        float *tA_packed; // temprorary pointer to hold packed A memory pointer
        int row_idx_packed; //packed A memory row index
        int lda_packed; //lda of packed A
        int col_idx_start; //starting index after A matrix is packed.
        dim_t tb_inc_row = 1; // row stride of matrix B
        dim_t tb_inc_col = ldb; // column stride of matrix B
        __m256 ymm4, ymm5, ymm6, ymm7;
        __m256 ymm8, ymm9, ymm10, ymm11;
        __m256 ymm12, ymm13, ymm14, ymm15;
        __m256 ymm0, ymm1, ymm2, ymm3;

        int n_remainder; // If the N is non multiple of 3.(N%3)
        int m_remainder; // If the M is non multiple of 32.(M%32)

        float *alpha_cast, *beta_cast; // alpha, beta multiples
        alpha_cast = (alpha->buffer);
        beta_cast = (beta->buffer);
        int required_packing_A = 1;

        // when N is equal to 1 call GEMV instead of GEMMT
        if (N == 1)
        {
            bli_gemv
            (
                alpha,
                a,
                b,
                beta,
                c
            );
            return BLIS_SUCCESS;
        }

        //update the pointer math if matrix B needs to be transposed.
        if (bli_obj_has_trans( b ))
        {
            tb_inc_col = 1; //switch row and column strides
            tb_inc_row = ldb;
        }

        if ((N <= 3) || ((MR * K) > F_SCRATCH_DIM))
        {
            required_packing_A = 0;
        }
        /*
        * The computation loop runs for MRxN columns of C matrix, thus
        * accessing the MRxK A matrix data and KxNR B matrix data.
        * The computation is organized as inner loops of dimension MRxNR.
        */
        // Process MR rows of C matrix at a time.
        for (row_idx = 0; (row_idx + (MR - 1)) < M; row_idx += MR)
        {

            col_idx_start = 0;
            tA_packed = A;
            row_idx_packed = row_idx;
            lda_packed = lda;

            // This is the part of the pack and compute optimization.
            // During the first column iteration, we store the accessed A matrix into
            // contiguous static memory. This helps to keep te A matrix in Cache and
            // aviods the TLB misses.
            if (required_packing_A)
            {
                col_idx = 0;

                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;
                tA_packed = A_pack;

#if 0//def BLIS_ENABLE_PREFETCH
                _mm_prefetch((char*)(tC + 0), _MM_HINT_T0);
                _mm_prefetch((char*)(tC + 16), _MM_HINT_T0);
                _mm_prefetch((char*)(tC + ldc), _MM_HINT_T0);
                _mm_prefetch((char*)(tC + ldc + 16), _MM_HINT_T0);
                _mm_prefetch((char*)(tC + 2 * ldc), _MM_HINT_T0);
                _mm_prefetch((char*)(tC + 2 * ldc + 16), _MM_HINT_T0);
#endif
                // clear scratch registers.
                ymm4 = _mm256_setzero_ps();
                ymm5 = _mm256_setzero_ps();
                ymm6 = _mm256_setzero_ps();
                ymm7 = _mm256_setzero_ps();
                ymm8 = _mm256_setzero_ps();
                ymm9 = _mm256_setzero_ps();
                ymm10 = _mm256_setzero_ps();
                ymm11 = _mm256_setzero_ps();
                ymm12 = _mm256_setzero_ps();
                ymm13 = _mm256_setzero_ps();
                ymm14 = _mm256_setzero_ps();
                ymm15 = _mm256_setzero_ps();

                for (k = 0; k < K; ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    // This loop is processing MR x K
                    ymm0 = _mm256_broadcast_ss(tB + tb_inc_col * 0);
                    ymm1 = _mm256_broadcast_ss(tB + tb_inc_col * 1);
                    ymm2 = _mm256_broadcast_ss(tB + tb_inc_col * 2);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_ps(tA);
                    _mm256_storeu_ps(tA_packed, ymm3); // the packing of matrix A
                                                       //                   ymm4 += ymm0 * ymm3;
                    ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
                    //                    ymm8 += ymm1 * ymm3;
                    ymm8 = _mm256_fmadd_ps(ymm1, ymm3, ymm8);
                    //                    ymm12 += ymm2 * ymm3;
                    ymm12 = _mm256_fmadd_ps(ymm2, ymm3, ymm12);

                    ymm3 = _mm256_loadu_ps(tA + 8);
                    _mm256_storeu_ps(tA_packed + 8, ymm3); // the packing of matrix A
                                                           //                    ymm5 += ymm0 * ymm3;
                    ymm5 = _mm256_fmadd_ps(ymm0, ymm3, ymm5);
                    //                    ymm9 += ymm1 * ymm3;
                    ymm9 = _mm256_fmadd_ps(ymm1, ymm3, ymm9);
                    //                    ymm13 += ymm2 * ymm3;
                    ymm13 = _mm256_fmadd_ps(ymm2, ymm3, ymm13);

                    ymm3 = _mm256_loadu_ps(tA + 16);
                    _mm256_storeu_ps(tA_packed + 16, ymm3); // the packing of matrix A
                                                            //                   ymm6 += ymm0 * ymm3;
                    ymm6 = _mm256_fmadd_ps(ymm0, ymm3, ymm6);
                    //                    ymm10 += ymm1 * ymm3;
                    ymm10 = _mm256_fmadd_ps(ymm1, ymm3, ymm10);
                    //                    ymm14 += ymm2 * ymm3;
                    ymm14 = _mm256_fmadd_ps(ymm2, ymm3, ymm14);

                    ymm3 = _mm256_loadu_ps(tA + 24);
                    _mm256_storeu_ps(tA_packed + 24, ymm3); // the packing of matrix A
                                                            //                    ymm7 += ymm0 * ymm3;
                    ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
                    //                    ymm11 += ymm1 * ymm3;
                    ymm11 = _mm256_fmadd_ps(ymm1, ymm3, ymm11);
                    //                   ymm15 += ymm2 * ymm3;
                    ymm15 = _mm256_fmadd_ps(ymm2, ymm3, ymm15);

                    tA += lda;
                    tA_packed += MR;
                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_ss(alpha_cast);
                //ymm1 = _mm256_broadcast_ss(beta_cast);

                //multiply A*B by alpha.
                ymm4 = _mm256_mul_ps(ymm4, ymm0);
                ymm5 = _mm256_mul_ps(ymm5, ymm0);
                ymm6 = _mm256_mul_ps(ymm6, ymm0);
                ymm7 = _mm256_mul_ps(ymm7, ymm0);
                ymm8 = _mm256_mul_ps(ymm8, ymm0);
                ymm9 = _mm256_mul_ps(ymm9, ymm0);
                ymm10 = _mm256_mul_ps(ymm10, ymm0);
                ymm11 = _mm256_mul_ps(ymm11, ymm0);
                ymm12 = _mm256_mul_ps(ymm12, ymm0);
                ymm13 = _mm256_mul_ps(ymm13, ymm0);
                ymm14 = _mm256_mul_ps(ymm14, ymm0);
                ymm15 = _mm256_mul_ps(ymm15, ymm0);

                // multiply C by beta and accumulate col 1.
                /*ymm2 = _mm256_loadu_ps(tC);
                ymm4 = _mm256_fmadd_ps(ymm2, ymm1, ymm4);
                ymm2 = _mm256_loadu_ps(tC + 8);
                ymm5 = _mm256_fmadd_ps(ymm2, ymm1, ymm5);
                ymm2 = _mm256_loadu_ps(tC + 16);
                ymm6 = _mm256_fmadd_ps(ymm2, ymm1, ymm6);
                ymm2 = _mm256_loadu_ps(tC + 24);
                ymm7 = _mm256_fmadd_ps(ymm2, ymm1, ymm7);*/
                _mm256_storeu_ps(tC, ymm4);
                _mm256_storeu_ps(tC + 8, ymm5);
                _mm256_storeu_ps(tC + 16, ymm6);
                _mm256_storeu_ps(tC + 24, ymm7);

                // multiply C by beta and accumulate, col 2.
                tC += ldc;
                /*ymm2 = _mm256_loadu_ps(tC);
                ymm8 = _mm256_fmadd_ps(ymm2, ymm1, ymm8);
                ymm2 = _mm256_loadu_ps(tC + 8);
                ymm9 = _mm256_fmadd_ps(ymm2, ymm1, ymm9);
                ymm2 = _mm256_loadu_ps(tC + 16);
                ymm10 = _mm256_fmadd_ps(ymm2, ymm1, ymm10);
                ymm2 = _mm256_loadu_ps(tC + 24);
                ymm11 = _mm256_fmadd_ps(ymm2, ymm1, ymm11);*/
                _mm256_storeu_ps(tC, ymm8);
                _mm256_storeu_ps(tC + 8, ymm9);
                _mm256_storeu_ps(tC + 16, ymm10);
                _mm256_storeu_ps(tC + 24, ymm11);

                // multiply C by beta and accumulate, col 3.
                tC += ldc;
                /*ymm2 = _mm256_loadu_ps(tC);
                ymm12 = _mm256_fmadd_ps(ymm2, ymm1, ymm12);
                ymm2 = _mm256_loadu_ps(tC + 8);
                ymm13 = _mm256_fmadd_ps(ymm2, ymm1, ymm13);
                ymm2 = _mm256_loadu_ps(tC + 16);
                ymm14 = _mm256_fmadd_ps(ymm2, ymm1, ymm14);
                ymm2 = _mm256_loadu_ps(tC + 24);
                ymm15 = _mm256_fmadd_ps(ymm2, ymm1, ymm15);*/
                _mm256_storeu_ps(tC, ymm12);
                _mm256_storeu_ps(tC + 8, ymm13);
                _mm256_storeu_ps(tC + 16, ymm14);
                _mm256_storeu_ps(tC + 24, ymm15);

                // modify the pointer arithematic to use packed A matrix.
                col_idx_start = NR;
                tA_packed = A_pack;
                row_idx_packed = 0;
                lda_packed = MR;
            }
            // Process NR columns of C matrix at a time.
            for (col_idx = col_idx_start; (col_idx + (NR - 1)) < N; col_idx += NR)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = tA_packed + row_idx_packed;

#if 0//def BLIS_ENABLE_PREFETCH
                _mm_prefetch((char*)(tC + 0), _MM_HINT_T0);
                _mm_prefetch((char*)(tC + 16), _MM_HINT_T0);
                _mm_prefetch((char*)(tC + ldc), _MM_HINT_T0);
                _mm_prefetch((char*)(tC + ldc + 16), _MM_HINT_T0);
                _mm_prefetch((char*)(tC + 2 * ldc), _MM_HINT_T0);
                _mm_prefetch((char*)(tC + 2 * ldc + 16), _MM_HINT_T0);
#endif
                // clear scratch registers.
                ymm4 = _mm256_setzero_ps();
                ymm5 = _mm256_setzero_ps();
                ymm6 = _mm256_setzero_ps();
                ymm7 = _mm256_setzero_ps();
                ymm8 = _mm256_setzero_ps();
                ymm9 = _mm256_setzero_ps();
                ymm10 = _mm256_setzero_ps();
                ymm11 = _mm256_setzero_ps();
                ymm12 = _mm256_setzero_ps();
                ymm13 = _mm256_setzero_ps();
                ymm14 = _mm256_setzero_ps();
                ymm15 = _mm256_setzero_ps();

                for (k = 0; k < K; ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    // This loop is processing MR x K
                    ymm0 = _mm256_broadcast_ss(tB + tb_inc_col * 0);
                    ymm1 = _mm256_broadcast_ss(tB + tb_inc_col * 1);
                    ymm2 = _mm256_broadcast_ss(tB + tb_inc_col * 2);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_ps(tA);
                    //                   ymm4 += ymm0 * ymm3;
                    ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
                    //                    ymm8 += ymm1 * ymm3;
                    ymm8 = _mm256_fmadd_ps(ymm1, ymm3, ymm8);
                    //                    ymm12 += ymm2 * ymm3;
                    ymm12 = _mm256_fmadd_ps(ymm2, ymm3, ymm12);

                    ymm3 = _mm256_loadu_ps(tA + 8);
                    //                    ymm5 += ymm0 * ymm3;
                    ymm5 = _mm256_fmadd_ps(ymm0, ymm3, ymm5);
                    //                    ymm9 += ymm1 * ymm3;
                    ymm9 = _mm256_fmadd_ps(ymm1, ymm3, ymm9);
                    //                    ymm13 += ymm2 * ymm3;
                    ymm13 = _mm256_fmadd_ps(ymm2, ymm3, ymm13);

                    ymm3 = _mm256_loadu_ps(tA + 16);
                    //                   ymm6 += ymm0 * ymm3;
                    ymm6 = _mm256_fmadd_ps(ymm0, ymm3, ymm6);
                    //                    ymm10 += ymm1 * ymm3;
                    ymm10 = _mm256_fmadd_ps(ymm1, ymm3, ymm10);
                    //                    ymm14 += ymm2 * ymm3;
                    ymm14 = _mm256_fmadd_ps(ymm2, ymm3, ymm14);

                    ymm3 = _mm256_loadu_ps(tA + 24);
                    //                    ymm7 += ymm0 * ymm3;
                    ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
                    //                    ymm11 += ymm1 * ymm3;
                    ymm11 = _mm256_fmadd_ps(ymm1, ymm3, ymm11);
                    //                   ymm15 += ymm2 * ymm3;
                    ymm15 = _mm256_fmadd_ps(ymm2, ymm3, ymm15);

                    tA += lda_packed;
                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_ss(alpha_cast);
                //ymm1 = _mm256_broadcast_ss(beta_cast);

                //multiply A*B by alpha.
                ymm4 = _mm256_mul_ps(ymm4, ymm0);
                ymm5 = _mm256_mul_ps(ymm5, ymm0);
                ymm6 = _mm256_mul_ps(ymm6, ymm0);
                ymm7 = _mm256_mul_ps(ymm7, ymm0);
                ymm8 = _mm256_mul_ps(ymm8, ymm0);
                ymm9 = _mm256_mul_ps(ymm9, ymm0);
                ymm10 = _mm256_mul_ps(ymm10, ymm0);
                ymm11 = _mm256_mul_ps(ymm11, ymm0);
                ymm12 = _mm256_mul_ps(ymm12, ymm0);
                ymm13 = _mm256_mul_ps(ymm13, ymm0);
                ymm14 = _mm256_mul_ps(ymm14, ymm0);
                ymm15 = _mm256_mul_ps(ymm15, ymm0);

                // multiply C by beta and accumulate col 1.
                /*ymm2 = _mm256_loadu_ps(tC);
                ymm4 = _mm256_fmadd_ps(ymm2, ymm1, ymm4);
                ymm2 = _mm256_loadu_ps(tC + 8);
                ymm5 = _mm256_fmadd_ps(ymm2, ymm1, ymm5);
                ymm2 = _mm256_loadu_ps(tC + 16);
                ymm6 = _mm256_fmadd_ps(ymm2, ymm1, ymm6);
                ymm2 = _mm256_loadu_ps(tC + 24);
                ymm7 = _mm256_fmadd_ps(ymm2, ymm1, ymm7);*/
                _mm256_storeu_ps(tC, ymm4);
                _mm256_storeu_ps(tC + 8, ymm5);
                _mm256_storeu_ps(tC + 16, ymm6);
                _mm256_storeu_ps(tC + 24, ymm7);

                // multiply C by beta and accumulate, col 2.
                tC += ldc;
                /*ymm2 = _mm256_loadu_ps(tC);
                ymm8 = _mm256_fmadd_ps(ymm2, ymm1, ymm8);
                ymm2 = _mm256_loadu_ps(tC + 8);
                ymm9 = _mm256_fmadd_ps(ymm2, ymm1, ymm9);
                ymm2 = _mm256_loadu_ps(tC + 16);
                ymm10 = _mm256_fmadd_ps(ymm2, ymm1, ymm10);
                ymm2 = _mm256_loadu_ps(tC + 24);
                ymm11 = _mm256_fmadd_ps(ymm2, ymm1, ymm11);*/
                _mm256_storeu_ps(tC, ymm8);
                _mm256_storeu_ps(tC + 8, ymm9);
                _mm256_storeu_ps(tC + 16, ymm10);
                _mm256_storeu_ps(tC + 24, ymm11);

                // multiply C by beta and accumulate, col 3.
                tC += ldc;
                /*ymm2 = _mm256_loadu_ps(tC);
                ymm12 = _mm256_fmadd_ps(ymm2, ymm1, ymm12);
                ymm2 = _mm256_loadu_ps(tC + 8);
                ymm13 = _mm256_fmadd_ps(ymm2, ymm1, ymm13);
                ymm2 = _mm256_loadu_ps(tC + 16);
                ymm14 = _mm256_fmadd_ps(ymm2, ymm1, ymm14);
                ymm2 = _mm256_loadu_ps(tC + 24);
                ymm15 = _mm256_fmadd_ps(ymm2, ymm1, ymm15);*/
                _mm256_storeu_ps(tC, ymm12);
                _mm256_storeu_ps(tC + 8, ymm13);
                _mm256_storeu_ps(tC + 16, ymm14);
                _mm256_storeu_ps(tC + 24, ymm15);

            }
            n_remainder = N - col_idx;

            // if the N is not multiple of 3.
            // handling edge case.
            if (n_remainder == 2)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                // clear scratch registers.
                ymm8 = _mm256_setzero_ps();
                ymm9 = _mm256_setzero_ps();
                ymm10 = _mm256_setzero_ps();
                ymm11 = _mm256_setzero_ps();
                ymm12 = _mm256_setzero_ps();
                ymm13 = _mm256_setzero_ps();
                ymm14 = _mm256_setzero_ps();
                ymm15 = _mm256_setzero_ps();

                for (k = 0; k < K; ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_ss(tB + tb_inc_col * 0);
                    ymm1 = _mm256_broadcast_ss(tB + tb_inc_col * 1);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_ps(tA);
                    ymm8 = _mm256_fmadd_ps(ymm0, ymm3, ymm8);
                    ymm12 = _mm256_fmadd_ps(ymm1, ymm3, ymm12);

                    ymm3 = _mm256_loadu_ps(tA + 8);
                    ymm9 = _mm256_fmadd_ps(ymm0, ymm3, ymm9);
                    ymm13 = _mm256_fmadd_ps(ymm1, ymm3, ymm13);

                    ymm3 = _mm256_loadu_ps(tA + 16);
                    ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
                    ymm14 = _mm256_fmadd_ps(ymm1, ymm3, ymm14);

                    ymm3 = _mm256_loadu_ps(tA + 24);
                    ymm11 = _mm256_fmadd_ps(ymm0, ymm3, ymm11);
                    ymm15 = _mm256_fmadd_ps(ymm1, ymm3, ymm15);

                    tA += lda;

                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_ss(alpha_cast);
                //ymm1 = _mm256_broadcast_ss(beta_cast);

                //multiply A*B by alpha.
                ymm8 = _mm256_mul_ps(ymm8, ymm0);
                ymm9 = _mm256_mul_ps(ymm9, ymm0);
                ymm10 = _mm256_mul_ps(ymm10, ymm0);
                ymm11 = _mm256_mul_ps(ymm11, ymm0);
                ymm12 = _mm256_mul_ps(ymm12, ymm0);
                ymm13 = _mm256_mul_ps(ymm13, ymm0);
                ymm14 = _mm256_mul_ps(ymm14, ymm0);
                ymm15 = _mm256_mul_ps(ymm15, ymm0);

                // multiply C by beta and accumulate, col 1.
                /*ymm2 = _mm256_loadu_ps(tC + 0);
                ymm8 = _mm256_fmadd_ps(ymm2, ymm1, ymm8);
                ymm2 = _mm256_loadu_ps(tC + 8);
                ymm9 = _mm256_fmadd_ps(ymm2, ymm1, ymm9);
                ymm2 = _mm256_loadu_ps(tC + 16);
                ymm10 = _mm256_fmadd_ps(ymm2, ymm1, ymm10);
                ymm2 = _mm256_loadu_ps(tC + 24);
                ymm11 = _mm256_fmadd_ps(ymm2, ymm1, ymm11);*/
                _mm256_storeu_ps(tC + 0, ymm8);
                _mm256_storeu_ps(tC + 8, ymm9);
                _mm256_storeu_ps(tC + 16, ymm10);
                _mm256_storeu_ps(tC + 24, ymm11);

                // multiply C by beta and accumulate, col 2.
                tC += ldc;
                /*ymm2 = _mm256_loadu_ps(tC);
                ymm12 = _mm256_fmadd_ps(ymm2, ymm1, ymm12);
                ymm2 = _mm256_loadu_ps(tC + 8);
                ymm13 = _mm256_fmadd_ps(ymm2, ymm1, ymm13);
                ymm2 = _mm256_loadu_ps(tC + 16);
                ymm14 = _mm256_fmadd_ps(ymm2, ymm1, ymm14);
                ymm2 = _mm256_loadu_ps(tC + 24);
                ymm15 = _mm256_fmadd_ps(ymm2, ymm1, ymm15);*/
                _mm256_storeu_ps(tC, ymm12);
                _mm256_storeu_ps(tC + 8, ymm13);
                _mm256_storeu_ps(tC + 16, ymm14);
                _mm256_storeu_ps(tC + 24, ymm15);

                col_idx += 2;
            }
            // if the N is not multiple of 3.
            // handling edge case.
            if (n_remainder == 1)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                // clear scratch registers.
                ymm12 = _mm256_setzero_ps();
                ymm13 = _mm256_setzero_ps();
                ymm14 = _mm256_setzero_ps();
                ymm15 = _mm256_setzero_ps();

                for (k = 0; k < K; ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_ss(tB + tb_inc_col * 0);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_ps(tA);
                    ymm12 = _mm256_fmadd_ps(ymm0, ymm3, ymm12);

                    ymm3 = _mm256_loadu_ps(tA + 8);
                    ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);

                    ymm3 = _mm256_loadu_ps(tA + 16);
                    ymm14 = _mm256_fmadd_ps(ymm0, ymm3, ymm14);

                    ymm3 = _mm256_loadu_ps(tA + 24);
                    ymm15 = _mm256_fmadd_ps(ymm0, ymm3, ymm15);

                    tA += lda;

                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_ss(alpha_cast);
                //ymm1 = _mm256_broadcast_ss(beta_cast);

                //multiply A*B by alpha.
                ymm12 = _mm256_mul_ps(ymm12, ymm0);
                ymm13 = _mm256_mul_ps(ymm13, ymm0);
                ymm14 = _mm256_mul_ps(ymm14, ymm0);
                ymm15 = _mm256_mul_ps(ymm15, ymm0);

                // multiply C by beta and accumulate.
                /*ymm2 = _mm256_loadu_ps(tC + 0);
                ymm12 = _mm256_fmadd_ps(ymm2, ymm1, ymm12);
                ymm2 = _mm256_loadu_ps(tC + 8);
                ymm13 = _mm256_fmadd_ps(ymm2, ymm1, ymm13);
                ymm2 = _mm256_loadu_ps(tC + 16);
                ymm14 = _mm256_fmadd_ps(ymm2, ymm1, ymm14);
                ymm2 = _mm256_loadu_ps(tC + 24);
                ymm15 = _mm256_fmadd_ps(ymm2, ymm1, ymm15);*/

                _mm256_storeu_ps(tC + 0, ymm12);
                _mm256_storeu_ps(tC + 8, ymm13);
                _mm256_storeu_ps(tC + 16, ymm14);
                _mm256_storeu_ps(tC + 24, ymm15);
            }
        }

        m_remainder = M - row_idx;

        if (m_remainder >= 24)
        {
            m_remainder -= 24;

            for (col_idx = 0; (col_idx + 2) < N; col_idx += 3)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                // clear scratch registers.
                ymm4 = _mm256_setzero_ps();
                ymm5 = _mm256_setzero_ps();
                ymm6 = _mm256_setzero_ps();
                ymm8 = _mm256_setzero_ps();
                ymm9 = _mm256_setzero_ps();
                ymm10 = _mm256_setzero_ps();
                ymm12 = _mm256_setzero_ps();
                ymm13 = _mm256_setzero_ps();
                ymm14 = _mm256_setzero_ps();

                for (k = 0; k < K; ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_ss(tB + tb_inc_col * 0);
                    ymm1 = _mm256_broadcast_ss(tB + tb_inc_col * 1);
                    ymm2 = _mm256_broadcast_ss(tB + tb_inc_col * 2);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_ps(tA);
                    //                   ymm4 += ymm0 * ymm3;
                    ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
                    //                    ymm8 += ymm1 * ymm3;
                    ymm8 = _mm256_fmadd_ps(ymm1, ymm3, ymm8);
                    //                    ymm12 += ymm2 * ymm3;
                    ymm12 = _mm256_fmadd_ps(ymm2, ymm3, ymm12);

                    ymm3 = _mm256_loadu_ps(tA + 8);
                    //                    ymm5 += ymm0 * ymm3;
                    ymm5 = _mm256_fmadd_ps(ymm0, ymm3, ymm5);
                    //                    ymm9 += ymm1 * ymm3;
                    ymm9 = _mm256_fmadd_ps(ymm1, ymm3, ymm9);
                    //                    ymm13 += ymm2 * ymm3;
                    ymm13 = _mm256_fmadd_ps(ymm2, ymm3, ymm13);

                    ymm3 = _mm256_loadu_ps(tA + 16);
                    //                   ymm6 += ymm0 * ymm3;
                    ymm6 = _mm256_fmadd_ps(ymm0, ymm3, ymm6);
                    //                    ymm10 += ymm1 * ymm3;
                    ymm10 = _mm256_fmadd_ps(ymm1, ymm3, ymm10);
                    //                    ymm14 += ymm2 * ymm3;
                    ymm14 = _mm256_fmadd_ps(ymm2, ymm3, ymm14);

                    tA += lda;
                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_ss(alpha_cast);
                //ymm1 = _mm256_broadcast_ss(beta_cast);

                //multiply A*B by alpha.
                ymm4 = _mm256_mul_ps(ymm4, ymm0);
                ymm5 = _mm256_mul_ps(ymm5, ymm0);
                ymm6 = _mm256_mul_ps(ymm6, ymm0);
                ymm8 = _mm256_mul_ps(ymm8, ymm0);
                ymm9 = _mm256_mul_ps(ymm9, ymm0);
                ymm10 = _mm256_mul_ps(ymm10, ymm0);
                ymm12 = _mm256_mul_ps(ymm12, ymm0);
                ymm13 = _mm256_mul_ps(ymm13, ymm0);
                ymm14 = _mm256_mul_ps(ymm14, ymm0);

                // multiply C by beta and accumulate.
                /*ymm2 = _mm256_loadu_ps(tC);
                ymm4 = _mm256_fmadd_ps(ymm2, ymm1, ymm4);
                ymm2 = _mm256_loadu_ps(tC + 8);
                ymm5 = _mm256_fmadd_ps(ymm2, ymm1, ymm5);
                ymm2 = _mm256_loadu_ps(tC + 16);
                ymm6 = _mm256_fmadd_ps(ymm2, ymm1, ymm6);*/
                _mm256_storeu_ps(tC, ymm4);
                _mm256_storeu_ps(tC + 8, ymm5);
                _mm256_storeu_ps(tC + 16, ymm6);

                // multiply C by beta and accumulate.
                tC += ldc;
                /*ymm2 = _mm256_loadu_ps(tC);
                ymm8 = _mm256_fmadd_ps(ymm2, ymm1, ymm8);
                ymm2 = _mm256_loadu_ps(tC + 8);
                ymm9 = _mm256_fmadd_ps(ymm2, ymm1, ymm9);
                ymm2 = _mm256_loadu_ps(tC + 16);
                ymm10 = _mm256_fmadd_ps(ymm2, ymm1, ymm10);*/
                _mm256_storeu_ps(tC, ymm8);
                _mm256_storeu_ps(tC + 8, ymm9);
                _mm256_storeu_ps(tC + 16, ymm10);

                // multiply C by beta and accumulate.
                tC += ldc;
                /*ymm2 = _mm256_loadu_ps(tC);
                ymm12 = _mm256_fmadd_ps(ymm2, ymm1, ymm12);
                ymm2 = _mm256_loadu_ps(tC + 8);
                ymm13 = _mm256_fmadd_ps(ymm2, ymm1, ymm13);
                ymm2 = _mm256_loadu_ps(tC + 16);
                ymm14 = _mm256_fmadd_ps(ymm2, ymm1, ymm14);*/
                _mm256_storeu_ps(tC, ymm12);
                _mm256_storeu_ps(tC + 8, ymm13);
                _mm256_storeu_ps(tC + 16, ymm14);

            }
            n_remainder = N - col_idx;
            // if the N is not multiple of 3.
            // handling edge case.
            if (n_remainder == 2)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                // clear scratch registers.
                ymm8 = _mm256_setzero_ps();
                ymm9 = _mm256_setzero_ps();
                ymm10 = _mm256_setzero_ps();
                ymm12 = _mm256_setzero_ps();
                ymm13 = _mm256_setzero_ps();
                ymm14 = _mm256_setzero_ps();

                for (k = 0; k < K; ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_ss(tB + tb_inc_col * 0);
                    ymm1 = _mm256_broadcast_ss(tB + tb_inc_col * 1);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_ps(tA);
                    ymm8 = _mm256_fmadd_ps(ymm0, ymm3, ymm8);
                    ymm12 = _mm256_fmadd_ps(ymm1, ymm3, ymm12);

                    ymm3 = _mm256_loadu_ps(tA + 8);
                    ymm9 = _mm256_fmadd_ps(ymm0, ymm3, ymm9);
                    ymm13 = _mm256_fmadd_ps(ymm1, ymm3, ymm13);

                    ymm3 = _mm256_loadu_ps(tA + 16);
                    ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
                    ymm14 = _mm256_fmadd_ps(ymm1, ymm3, ymm14);

                    tA += lda;

                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_ss(alpha_cast);
                //ymm1 = _mm256_broadcast_ss(beta_cast);

                //multiply A*B by alpha.
                ymm8 = _mm256_mul_ps(ymm8, ymm0);
                ymm9 = _mm256_mul_ps(ymm9, ymm0);
                ymm10 = _mm256_mul_ps(ymm10, ymm0);
                ymm12 = _mm256_mul_ps(ymm12, ymm0);
                ymm13 = _mm256_mul_ps(ymm13, ymm0);
                ymm14 = _mm256_mul_ps(ymm14, ymm0);

                // multiply C by beta and accumulate.
                /*ymm2 = _mm256_loadu_ps(tC + 0);
                ymm8 = _mm256_fmadd_ps(ymm2, ymm1, ymm8);
                ymm2 = _mm256_loadu_ps(tC + 8);
                ymm9 = _mm256_fmadd_ps(ymm2, ymm1, ymm9);
                ymm2 = _mm256_loadu_ps(tC + 16);
                ymm10 = _mm256_fmadd_ps(ymm2, ymm1, ymm10);*/
                _mm256_storeu_ps(tC + 0, ymm8);
                _mm256_storeu_ps(tC + 8, ymm9);
                _mm256_storeu_ps(tC + 16, ymm10);

                // multiply C by beta and accumulate.
                tC += ldc;
                /*ymm2 = _mm256_loadu_ps(tC);
                ymm12 = _mm256_fmadd_ps(ymm2, ymm1, ymm12);
                ymm2 = _mm256_loadu_ps(tC + 8);
                ymm13 = _mm256_fmadd_ps(ymm2, ymm1, ymm13);
                ymm2 = _mm256_loadu_ps(tC + 16);
                ymm14 = _mm256_fmadd_ps(ymm2, ymm1, ymm14);*/
                _mm256_storeu_ps(tC, ymm12);
                _mm256_storeu_ps(tC + 8, ymm13);
                _mm256_storeu_ps(tC + 16, ymm14);

                col_idx += 2;
            }
            // if the N is not multiple of 3.
            // handling edge case.
            if (n_remainder == 1)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                // clear scratch registers.
                ymm12 = _mm256_setzero_ps();
                ymm13 = _mm256_setzero_ps();
                ymm14 = _mm256_setzero_ps();

                for (k = 0; k < K; ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_ss(tB + tb_inc_col * 0);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_ps(tA);
                    ymm12 = _mm256_fmadd_ps(ymm0, ymm3, ymm12);

                    ymm3 = _mm256_loadu_ps(tA + 8);
                    ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);

                    ymm3 = _mm256_loadu_ps(tA + 16);
                    ymm14 = _mm256_fmadd_ps(ymm0, ymm3, ymm14);

                    tA += lda;

                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_ss(alpha_cast);
                //ymm1 = _mm256_broadcast_ss(beta_cast);

                //multiply A*B by alpha.
                ymm12 = _mm256_mul_ps(ymm12, ymm0);
                ymm13 = _mm256_mul_ps(ymm13, ymm0);
                ymm14 = _mm256_mul_ps(ymm14, ymm0);

                // multiply C by beta and accumulate.
                /*ymm2 = _mm256_loadu_ps(tC + 0);
                ymm12 = _mm256_fmadd_ps(ymm2, ymm1, ymm12);
                ymm2 = _mm256_loadu_ps(tC + 8);
                ymm13 = _mm256_fmadd_ps(ymm2, ymm1, ymm13);
                ymm2 = _mm256_loadu_ps(tC + 16);
                ymm14 = _mm256_fmadd_ps(ymm2, ymm1, ymm14);*/

                _mm256_storeu_ps(tC + 0, ymm12);
                _mm256_storeu_ps(tC + 8, ymm13);
                _mm256_storeu_ps(tC + 16, ymm14);
            }

            row_idx += 24;
        }

        if (m_remainder >= 16)
        {
            m_remainder -= 16;

            for (col_idx = 0; (col_idx + 2) < N; col_idx += 3)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                // clear scratch registers.
                ymm4 = _mm256_setzero_ps();
                ymm5 = _mm256_setzero_ps();
                ymm6 = _mm256_setzero_ps();
                ymm7 = _mm256_setzero_ps();
                ymm8 = _mm256_setzero_ps();
                ymm9 = _mm256_setzero_ps();

                for (k = 0; k < K; ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_ss(tB + tb_inc_col * 0);
                    ymm1 = _mm256_broadcast_ss(tB + tb_inc_col * 1);
                    ymm2 = _mm256_broadcast_ss(tB + tb_inc_col * 2);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_ps(tA);
                    ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
                    ymm6 = _mm256_fmadd_ps(ymm1, ymm3, ymm6);
                    ymm8 = _mm256_fmadd_ps(ymm2, ymm3, ymm8);

                    ymm3 = _mm256_loadu_ps(tA + 8);
                    ymm5 = _mm256_fmadd_ps(ymm0, ymm3, ymm5);
                    ymm7 = _mm256_fmadd_ps(ymm1, ymm3, ymm7);
                    ymm9 = _mm256_fmadd_ps(ymm2, ymm3, ymm9);

                    tA += lda;
                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_ss(alpha_cast);
                //ymm1 = _mm256_broadcast_ss(beta_cast);

                //multiply A*B by alpha.
                ymm4 = _mm256_mul_ps(ymm4, ymm0);
                ymm5 = _mm256_mul_ps(ymm5, ymm0);
                ymm6 = _mm256_mul_ps(ymm6, ymm0);
                ymm7 = _mm256_mul_ps(ymm7, ymm0);
                ymm8 = _mm256_mul_ps(ymm8, ymm0);
                ymm9 = _mm256_mul_ps(ymm9, ymm0);

                // multiply C by beta and accumulate.
                /*ymm2 = _mm256_loadu_ps(tC);
                ymm4 = _mm256_fmadd_ps(ymm2, ymm1, ymm4);
                ymm2 = _mm256_loadu_ps(tC + 8);
                ymm5 = _mm256_fmadd_ps(ymm2, ymm1, ymm5);*/
                _mm256_storeu_ps(tC, ymm4);
                _mm256_storeu_ps(tC + 8, ymm5);

                // multiply C by beta and accumulate.
                tC += ldc;
                /*ymm2 = _mm256_loadu_ps(tC);
                ymm6 = _mm256_fmadd_ps(ymm2, ymm1, ymm6);
                ymm2 = _mm256_loadu_ps(tC + 8);
                ymm7 = _mm256_fmadd_ps(ymm2, ymm1, ymm7);*/
                _mm256_storeu_ps(tC, ymm6);
                _mm256_storeu_ps(tC + 8, ymm7);

                // multiply C by beta and accumulate.
                tC += ldc;
                /*ymm2 = _mm256_loadu_ps(tC);
                ymm8 = _mm256_fmadd_ps(ymm2, ymm1, ymm8);
                ymm2 = _mm256_loadu_ps(tC + 8);
                ymm9 = _mm256_fmadd_ps(ymm2, ymm1, ymm9);*/
                _mm256_storeu_ps(tC, ymm8);
                _mm256_storeu_ps(tC + 8, ymm9);

            }
            n_remainder = N - col_idx;
            // if the N is not multiple of 3.
            // handling edge case.
            if (n_remainder == 2)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                // clear scratch registers.
                ymm4 = _mm256_setzero_ps();
                ymm5 = _mm256_setzero_ps();
                ymm6 = _mm256_setzero_ps();
                ymm7 = _mm256_setzero_ps();

                for (k = 0; k < K; ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_ss(tB + tb_inc_col * 0);
                    ymm1 = _mm256_broadcast_ss(tB + tb_inc_col * 1);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_ps(tA);
                    ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
                    ymm6 = _mm256_fmadd_ps(ymm1, ymm3, ymm6);

                    ymm3 = _mm256_loadu_ps(tA + 8);
                    ymm5 = _mm256_fmadd_ps(ymm0, ymm3, ymm5);
                    ymm7 = _mm256_fmadd_ps(ymm1, ymm3, ymm7);

                    tA += lda;
                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_ss(alpha_cast);
                //ymm1 = _mm256_broadcast_ss(beta_cast);

                //multiply A*B by alpha.
                ymm4 = _mm256_mul_ps(ymm4, ymm0);
                ymm5 = _mm256_mul_ps(ymm5, ymm0);
                ymm6 = _mm256_mul_ps(ymm6, ymm0);
                ymm7 = _mm256_mul_ps(ymm7, ymm0);

                // multiply C by beta and accumulate.
                /*ymm2 = _mm256_loadu_ps(tC);
                ymm4 = _mm256_fmadd_ps(ymm2, ymm1, ymm4);
                ymm2 = _mm256_loadu_ps(tC + 8);
                ymm5 = _mm256_fmadd_ps(ymm2, ymm1, ymm5);*/
                _mm256_storeu_ps(tC, ymm4);
                _mm256_storeu_ps(tC + 8, ymm5);

                // multiply C by beta and accumulate.
                tC += ldc;
                /*ymm2 = _mm256_loadu_ps(tC);
                ymm6 = _mm256_fmadd_ps(ymm2, ymm1, ymm6);
                ymm2 = _mm256_loadu_ps(tC + 8);
                ymm7 = _mm256_fmadd_ps(ymm2, ymm1, ymm7);*/
                _mm256_storeu_ps(tC, ymm6);
                _mm256_storeu_ps(tC + 8, ymm7);

                col_idx += 2;

            }
            // if the N is not multiple of 3.
            // handling edge case.
            if (n_remainder == 1)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                ymm4 = _mm256_setzero_ps();
                ymm5 = _mm256_setzero_ps();

                for (k = 0; k < K; ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_ss(tB + tb_inc_col * 0);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_ps(tA);
                    ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);

                    ymm3 = _mm256_loadu_ps(tA + 8);
                    ymm5 = _mm256_fmadd_ps(ymm0, ymm3, ymm5);

                    tA += lda;
                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_ss(alpha_cast);
                //ymm1 = _mm256_broadcast_ss(beta_cast);

                ymm4 = _mm256_mul_ps(ymm4, ymm0);
                ymm5 = _mm256_mul_ps(ymm5, ymm0);

                // multiply C by beta and accumulate.
                /*ymm2 = _mm256_loadu_ps(tC);
                ymm4 = _mm256_fmadd_ps(ymm2, ymm1, ymm4);
                ymm2 = _mm256_loadu_ps(tC + 8);
                ymm5 = _mm256_fmadd_ps(ymm2, ymm1, ymm5);*/
                _mm256_storeu_ps(tC, ymm4);
                _mm256_storeu_ps(tC + 8, ymm5);

            }

            row_idx += 16;
        }

        if (m_remainder >= 8)
        {
            m_remainder -= 8;

            for (col_idx = 0; (col_idx + 2) < N; col_idx += 3)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                // clear scratch registers.
                ymm4 = _mm256_setzero_ps();
                ymm5 = _mm256_setzero_ps();
                ymm6 = _mm256_setzero_ps();

                for (k = 0; k < K; ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_ss(tB + tb_inc_col * 0);
                    ymm1 = _mm256_broadcast_ss(tB + tb_inc_col * 1);
                    ymm2 = _mm256_broadcast_ss(tB + tb_inc_col * 2);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_ps(tA);
                    ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
                    ymm5 = _mm256_fmadd_ps(ymm1, ymm3, ymm5);
                    ymm6 = _mm256_fmadd_ps(ymm2, ymm3, ymm6);

                    tA += lda;
                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_ss(alpha_cast);
                //ymm1 = _mm256_broadcast_ss(beta_cast);

                //multiply A*B by alpha.
                ymm4 = _mm256_mul_ps(ymm4, ymm0);
                ymm5 = _mm256_mul_ps(ymm5, ymm0);
                ymm6 = _mm256_mul_ps(ymm6, ymm0);

                // multiply C by beta and accumulate.
                /*ymm2 = _mm256_loadu_ps(tC);
                ymm4 = _mm256_fmadd_ps(ymm2, ymm1, ymm4);*/
                _mm256_storeu_ps(tC, ymm4);

                // multiply C by beta and accumulate.
                tC += ldc;
                /*ymm2 = _mm256_loadu_ps(tC);
                ymm5 = _mm256_fmadd_ps(ymm2, ymm1, ymm5);*/
                _mm256_storeu_ps(tC, ymm5);

                // multiply C by beta and accumulate.
                tC += ldc;
                /*ymm2 = _mm256_loadu_ps(tC);
                ymm6 = _mm256_fmadd_ps(ymm2, ymm1, ymm6);*/
                _mm256_storeu_ps(tC, ymm6);
            }
            n_remainder = N - col_idx;
            // if the N is not multiple of 3.
            // handling edge case.
            if (n_remainder == 2)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                ymm4 = _mm256_setzero_ps();
                ymm5 = _mm256_setzero_ps();

                for (k = 0; k < K; ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_ss(tB + tb_inc_col * 0);
                    ymm1 = _mm256_broadcast_ss(tB + tb_inc_col * 1);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_ps(tA);
                    ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
                    ymm5 = _mm256_fmadd_ps(ymm1, ymm3, ymm5);

                    tA += lda;
                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_ss(alpha_cast);
                //ymm1 = _mm256_broadcast_ss(beta_cast);

                //multiply A*B by alpha.
                ymm4 = _mm256_mul_ps(ymm4, ymm0);
                ymm5 = _mm256_mul_ps(ymm5, ymm0);

                // multiply C by beta and accumulate.
                /*ymm2 = _mm256_loadu_ps(tC);
                ymm4 = _mm256_fmadd_ps(ymm2, ymm1, ymm4);*/
                _mm256_storeu_ps(tC, ymm4);

                // multiply C by beta and accumulate.
                tC += ldc;
                /*ymm2 = _mm256_loadu_ps(tC);
                ymm5 = _mm256_fmadd_ps(ymm2, ymm1, ymm5);*/
                _mm256_storeu_ps(tC, ymm5);

                col_idx += 2;

            }
            // if the N is not multiple of 3.
            // handling edge case.
            if (n_remainder == 1)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                ymm4 = _mm256_setzero_ps();

                for (k = 0; k < K; ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_ss(tB + tb_inc_col * 0);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_ps(tA);
                    ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);

                    tA += lda;
                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_ss(alpha_cast);
                //ymm1 = _mm256_broadcast_ss(beta_cast);

                ymm4 = _mm256_mul_ps(ymm4, ymm0);

                // multiply C by beta and accumulate.
                /*ymm2 = _mm256_loadu_ps(tC);
                ymm4 = _mm256_fmadd_ps(ymm2, ymm1, ymm4);*/
                _mm256_storeu_ps(tC, ymm4);

            }

            row_idx += 8;
        }
        // M is not a multiple of 32.
        // The handling of edge case where the remainder
        // dimension is less than 8. The padding takes place
        // to handle this case.
        if ((m_remainder) && (lda > 7))
        {
            float f_temp[8];

            for (col_idx = 0; (col_idx + 2) < N; col_idx += 3)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                // clear scratch registers.
                ymm5 = _mm256_setzero_ps();
                ymm7 = _mm256_setzero_ps();
                ymm9 = _mm256_setzero_ps();

                for (k = 0; k < (K - 1); ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_ss(tB + tb_inc_col * 0);
                    ymm1 = _mm256_broadcast_ss(tB + tb_inc_col * 1);
                    ymm2 = _mm256_broadcast_ss(tB + tb_inc_col * 2);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_ps(tA);
                    ymm5 = _mm256_fmadd_ps(ymm0, ymm3, ymm5);
                    ymm7 = _mm256_fmadd_ps(ymm1, ymm3, ymm7);
                    ymm9 = _mm256_fmadd_ps(ymm2, ymm3, ymm9);

                    tA += lda;
                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_ss(tB + tb_inc_col * 0);
                ymm1 = _mm256_broadcast_ss(tB + tb_inc_col * 1);
                ymm2 = _mm256_broadcast_ss(tB + tb_inc_col * 2);
                tB += tb_inc_row;

                for (int i = 0; i < m_remainder; i++)
                {
                    f_temp[i] = tA[i];
                }
                ymm3 = _mm256_loadu_ps(f_temp);
                ymm5 = _mm256_fmadd_ps(ymm0, ymm3, ymm5);
                ymm7 = _mm256_fmadd_ps(ymm1, ymm3, ymm7);
                ymm9 = _mm256_fmadd_ps(ymm2, ymm3, ymm9);

                ymm0 = _mm256_broadcast_ss(alpha_cast);
                //ymm1 = _mm256_broadcast_ss(beta_cast);

                //multiply A*B by alpha.
                ymm5 = _mm256_mul_ps(ymm5, ymm0);
                ymm7 = _mm256_mul_ps(ymm7, ymm0);
                ymm9 = _mm256_mul_ps(ymm9, ymm0);


                /*for (int i = 0; i < m_remainder; i++)
                {
                    f_temp[i] = tC[i];
                }
                ymm2 = _mm256_loadu_ps(f_temp);
                ymm5 = _mm256_fmadd_ps(ymm2, ymm1, ymm5);*/
                _mm256_storeu_ps(f_temp, ymm5);
                for (int i = 0; i < m_remainder; i++)
                {
                    tC[i] = f_temp[i];
                }

                tC += ldc;
                /*for (int i = 0; i < m_remainder; i++)
                {
                    f_temp[i] = tC[i];
                }
                ymm2 = _mm256_loadu_ps(f_temp);
                ymm7 = _mm256_fmadd_ps(ymm2, ymm1, ymm7);*/
                _mm256_storeu_ps(f_temp, ymm7);
                for (int i = 0; i < m_remainder; i++)
                {
                    tC[i] = f_temp[i];
                }

                tC += ldc;
                /*for (int i = 0; i < m_remainder; i++)
                {
                    f_temp[i] = tC[i];
                }
                ymm2 = _mm256_loadu_ps(f_temp);
                ymm9 = _mm256_fmadd_ps(ymm2, ymm1, ymm9);*/
                _mm256_storeu_ps(f_temp, ymm9);
                for (int i = 0; i < m_remainder; i++)
                {
                    tC[i] = f_temp[i];
                }
            }
            n_remainder = N - col_idx;
            // if the N is not multiple of 3.
            // handling edge case.
            if (n_remainder == 2)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                ymm5 = _mm256_setzero_ps();
                ymm7 = _mm256_setzero_ps();

                for (k = 0; k < (K - 1); ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_ss(tB + tb_inc_col * 0);
                    ymm1 = _mm256_broadcast_ss(tB + tb_inc_col * 1);
                    tB += tb_inc_row;

                    ymm3 = _mm256_loadu_ps(tA);
                    ymm5 = _mm256_fmadd_ps(ymm0, ymm3, ymm5);
                    ymm7 = _mm256_fmadd_ps(ymm1, ymm3, ymm7);

                    tA += lda;
                }

                ymm0 = _mm256_broadcast_ss(tB + tb_inc_col * 0);
                ymm1 = _mm256_broadcast_ss(tB + tb_inc_col * 1);
                tB += tb_inc_row;

                for (int i = 0; i < m_remainder; i++)
                {
                    f_temp[i] = tA[i];
                }
                ymm3 = _mm256_loadu_ps(f_temp);
                ymm5 = _mm256_fmadd_ps(ymm0, ymm3, ymm5);
                ymm7 = _mm256_fmadd_ps(ymm1, ymm3, ymm7);

                ymm0 = _mm256_broadcast_ss(alpha_cast);
                //ymm1 = _mm256_broadcast_ss(beta_cast);

                ymm5 = _mm256_mul_ps(ymm5, ymm0);
                ymm7 = _mm256_mul_ps(ymm7, ymm0);

                /*for (int i = 0; i < m_remainder; i++)
                {
                    f_temp[i] = tC[i];
                }
                ymm2 = _mm256_loadu_ps(f_temp);
                ymm5 = _mm256_fmadd_ps(ymm2, ymm1, ymm5);*/
                _mm256_storeu_ps(f_temp, ymm5);
                for (int i = 0; i < m_remainder; i++)
                {
                    tC[i] = f_temp[i];
                }

                tC += ldc;
                /*for (int i = 0; i < m_remainder; i++)
                {
                    f_temp[i] = tC[i];
                }
                ymm2 = _mm256_loadu_ps(f_temp);
                ymm7 = _mm256_fmadd_ps(ymm2, ymm1, ymm7);*/
                _mm256_storeu_ps(f_temp, ymm7);
                for (int i = 0; i < m_remainder; i++)
                {
                    tC[i] = f_temp[i];
                }
            }
            // if the N is not multiple of 3.
            // handling edge case.
            if (n_remainder == 1)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                ymm5 = _mm256_setzero_ps();

                for (k = 0; k < (K - 1); ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_ss(tB + tb_inc_col * 0);
                    tB += tb_inc_row;

                    ymm3 = _mm256_loadu_ps(tA);
                    ymm5 = _mm256_fmadd_ps(ymm0, ymm3, ymm5);

                    tA += lda;
                }

                ymm0 = _mm256_broadcast_ss(tB + tb_inc_col * 0);
                tB += tb_inc_row;

                for (int i = 0; i < m_remainder; i++)
                {
                    f_temp[i] = tA[i];
                }
                ymm3 = _mm256_loadu_ps(f_temp);
                ymm5 = _mm256_fmadd_ps(ymm0, ymm3, ymm5);

                ymm0 = _mm256_broadcast_ss(alpha_cast);
                //ymm1 = _mm256_broadcast_ss(beta_cast);

                // multiply C by beta and accumulate.
                ymm5 = _mm256_mul_ps(ymm5, ymm0);

                /*for (int i = 0; i < m_remainder; i++)
                {
                    f_temp[i] = tC[i];
                }
                ymm2 = _mm256_loadu_ps(f_temp);
                ymm5 = _mm256_fmadd_ps(ymm2, ymm1, ymm5);*/
                _mm256_storeu_ps(f_temp, ymm5);
                for (int i = 0; i < m_remainder; i++)
                {
                    tC[i] = f_temp[i];
                }
            }
            m_remainder = 0;
        }

        if (m_remainder)
        {
            float result;
            for (; row_idx < M; row_idx += 1)
            {
                for (col_idx = 0; col_idx < N; col_idx += 1)
                {
                    //pointer math to point to proper memory
                    tC = C + ldc * col_idx + row_idx;
                    tB = B + tb_inc_col * col_idx;
                    tA = A + row_idx;

                    result = 0;
                    for (k = 0; k < K; ++k)
                    {
                        result += (*tA) * (*tB);
                        tA += lda;
                        tB += tb_inc_row;
                    }

                    result *= (*alpha_cast);
                    (*tC) = /*(*tC) * (*beta_cast) + */result;
                }
            }
        }

        //copy/compute sryk values back to C using SIMD
        if ( bli_seq0( *beta_cast ) )
        {//just copy in case of beta = 0
            dim_t _i, _j, k, _l;
            if(bli_obj_is_lower(c)) // c is lower
            {
                //first column
                _j = 0;
                k = M >> 3;
                _i = 0;
                for ( _l = 0; _l < k; _l++ )
                {
                    ymm0 = _mm256_loadu_ps((C + _i*rsc));
                    _mm256_storeu_ps((matCbuf + _i*rs_matC), ymm0);
                    _i += 8;
                }
                while (_i < M )
                {
                    bli_sscopys( *(C + _i*rsc + _j*ldc),
                                 *(matCbuf + _i*rs_matC + _j*ldc_matC) );
                    _i++;
                }
                _j++;
                while ( _j < N ) //next column
                {
                    //k = (_j + (8 - (_j & 7)));
                    _l = _j & 7;
                    k = (_l != 0) ? (_j + (8 - _l)) : _j;
                    k = (k <= M) ? k : M;
                    for ( _i = _j; _i < k; ++_i )
                    {
                        bli_sscopys( *(C + _i*rsc + _j*ldc),
                                     *(matCbuf + _i*rs_matC + _j*ldc_matC) );
                    }
                    k = (M - _i) >> 3;
                    _l = 0;
                    while ( _l < k )
                    {
                        ymm0 = _mm256_loadu_ps((C + _i*rsc + _j*ldc));
                        _mm256_storeu_ps((matCbuf + _i*rs_matC + _j*ldc_matC), ymm0);

                        _i += 8;
                        _l++;
                    }
                    while (_i < M )
                    {
                        bli_sscopys( *(C + _i*rsc + _j*ldc),
                                 *(matCbuf + _i*rs_matC + _j*ldc_matC) );
                        _i++;
                    }
                    _j++;
                }
            }
            else //c is upper
            {
                for ( _j = 0; _j < N; ++_j )
                {
					k = (_j + 1) >> 3;
                    _i = 0;
                    _l = 0;
                    while ( _l < k )
                    {
                        ymm0 = _mm256_loadu_ps((C + _i*rsc + _j*ldc));
                        _mm256_storeu_ps((matCbuf + _i*rs_matC + _j*ldc_matC), ymm0);
                        _i += 8;
                        _l++;
                    }
                    while (_i <= _j )
                    {
                        bli_sscopys( *(C + _i*rsc + _j*ldc),
                                     *(matCbuf + _i*rs_matC + _j*ldc_matC) );
                        ++_i;
                    }
                }
            }
        }
        else
        {//when beta is non-zero, fmadd and store the results
            dim_t _i, _j, k, _l;
            ymm1 = _mm256_broadcast_ss(beta_cast);
            if(bli_obj_is_lower(c)) //c is lower
            {
                //first column
                _j = 0;
                k = M >> 3;
                _i = 0;
                for ( _l = 0; _l < k; _l++ )
                {
                    ymm2 = _mm256_loadu_ps((matCbuf + _i*rs_matC));
                    ymm0 = _mm256_loadu_ps((C + _i*rsc));
                    ymm0 = _mm256_fmadd_ps(ymm2, ymm1, ymm0);
                    _mm256_storeu_ps((matCbuf + _i*rs_matC), ymm0);
                    _i += 8;
                }
                while (_i < M )
                {
                    bli_sssxpbys( *(C + _i*rsc + _j*ldc),
                                  *(beta_cast),
                                  *(matCbuf + _i*rs_matC + _j*ldc_matC) );
                    _i++;
                }
                _j++;
                while ( _j < N ) //next column
                {
                    //k = (_j + (8 - (_j & 7)));
                    _l = _j & 7;
                    k = (_l != 0) ? (_j + (8 - _l)) : _j;
                    k = (k <= M) ? k : M;
                    for ( _i = _j; _i < k; ++_i )
                    {
                        bli_sssxpbys( *(C + _i*rsc + _j*ldc),
                                          *(beta_cast),
                                          *(matCbuf + _i*rs_matC + _j*ldc_matC) );
                    }
                    k = (M - _i) >> 3;
                    _l = 0;
                    while ( _l < k )
                    {
                        ymm2 = _mm256_loadu_ps((matCbuf + _i*rs_matC + _j*ldc_matC));
                        ymm0 = _mm256_loadu_ps((C + _i*rsc + _j*ldc));
                        ymm0 = _mm256_fmadd_ps(ymm2, ymm1, ymm0);
                        _mm256_storeu_ps((matCbuf + _i*rs_matC + _j*ldc_matC), ymm0);

                        _i += 8;
                        _l++;
                    }
                    while (_i < M )
                    {
                        bli_sssxpbys( *(C + _i*rsc + _j*ldc),
                                      *(beta_cast),
                                      *(matCbuf + _i*rs_matC + _j*ldc_matC) );
                        _i++;
                    }
                    _j++;
                }
            }
            else //c is upper
            {
                for ( _j = 0; _j < N; ++_j )
                {
					k = (_j + 1) >> 3;
                    _i = 0;
                    _l = 0;
                    while ( _l < k )
                    {
                        ymm2 = _mm256_loadu_ps((matCbuf + _i*rs_matC + _j*ldc_matC));
                        ymm0 = _mm256_loadu_ps((C + _i*rsc + _j*ldc));
                        ymm0 = _mm256_fmadd_ps(ymm2, ymm1, ymm0);
                        _mm256_storeu_ps((matCbuf + _i*rs_matC + _j*ldc_matC), ymm0);
                        _i += 8;
                        _l++;
                    }
                    while (_i <= _j )
                    {
                        bli_sssxpbys( *(C + _i*rsc + _j*ldc),
                                      *(beta_cast),
                                      *(matCbuf + _i*rs_matC + _j*ldc_matC) );
                        ++_i;
                    }
                }
            }
        }

        return BLIS_SUCCESS;
    }
    else
        return BLIS_NONCONFORMAL_DIMENSIONS;


};

static err_t bli_dgemmt_small
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       cntl_t* cntl
     )
{

    int M = bli_obj_length( c ); // number of rows of Matrix C
    int N = bli_obj_width( c );  // number of columns of Matrix C
    int K = bli_obj_width( a );  // number of columns of OP(A), will be updated if OP(A) is Transpose(A) .
    int L = M * N;

    // If alpha is zero, scale by beta and return.
    if ((((L) < (D_BLIS_SMALL_MATRIX_THRES * D_BLIS_SMALL_MATRIX_THRES))
        || ((M  < D_BLIS_SMALL_M_RECT_MATRIX_THRES) && (K < D_BLIS_SMALL_K_RECT_MATRIX_THRES))) && ((L!=0) && (K!=0)))
    {

        int lda = bli_obj_col_stride( a ); // column stride of matrix OP(A), where OP(A) is Transpose(A) if transA enabled.
        int ldb = bli_obj_col_stride( b ); // column stride of matrix OP(B), where OP(B) is Transpose(B) if transB enabled.
        int ldc_matC = bli_obj_col_stride( c ); // column stride of matrix C
        int ldc = M;//bli_obj_col_stride( c ); // column stride of static buffer for matrix C
        int row_idx, col_idx, k;
        int rs_matC = bli_obj_row_stride( c );
        int rsc = 1;
        double *A = a->buffer; // pointer to elements of Matrix A
        double *B = b->buffer; // pointer to elements of Matrix B
        double *C = D_C_pack; // pointer to elements of Matrix C
        double *matCbuf = c->buffer;

        double *tA = A, *tB = B, *tC = C;//, *tA_pack;
        double *tA_packed; // temprorary pointer to hold packed A memory pointer
        int row_idx_packed; //packed A memory row index
        int lda_packed; //lda of packed A
        int col_idx_start; //starting index after A matrix is packed.
        dim_t tb_inc_row = 1; // row stride of matrix B
        dim_t tb_inc_col = ldb; // column stride of matrix B
        __m256d ymm4, ymm5, ymm6, ymm7;
        __m256d ymm8, ymm9, ymm10, ymm11;
        __m256d ymm12, ymm13, ymm14, ymm15;
        __m256d ymm0, ymm1, ymm2, ymm3;

        int n_remainder; // If the N is non multiple of 3.(N%3)
        int m_remainder; // If the M is non multiple of 16.(M%16)

        double *alpha_cast, *beta_cast; // alpha, beta multiples
        alpha_cast = (alpha->buffer);
        beta_cast = (beta->buffer);
        int required_packing_A = 1;

        // when N is equal to 1 call GEMV instead of GEMMT
        if (N == 1)
        {
            bli_gemv
            (
                alpha,
                a,
                b,
                beta,
                c
            );
            return BLIS_SUCCESS;
        }

        //update the pointer math if matrix B needs to be transposed.
        if (bli_obj_has_trans( b ))
        {
            tb_inc_col = 1; //switch row and column strides
            tb_inc_row = ldb;
        }

        if ((N <= 3) || ((D_MR * K) > D_SCRATCH_DIM))
        {
            required_packing_A = 0;
        }
        /*
        * The computation loop runs for D_MRxN columns of C matrix, thus
        * accessing the D_MRxK A matrix data and KxNR B matrix data.
        * The computation is organized as inner loops of dimension D_MRxNR.
        */
        // Process D_MR rows of C matrix at a time.
        for (row_idx = 0; (row_idx + (D_MR - 1)) < M; row_idx += D_MR)
        {

            col_idx_start = 0;
            tA_packed = A;
            row_idx_packed = row_idx;
            lda_packed = lda;

            // This is the part of the pack and compute optimization.
            // During the first column iteration, we store the accessed A matrix into
            // contiguous static memory. This helps to keep te A matrix in Cache and
            // aviods the TLB misses.
            if (required_packing_A)
            {
                col_idx = 0;

                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;
                tA_packed = D_A_pack;

#if 0//def BLIS_ENABLE_PREFETCH
                _mm_prefetch((char*)(tC + 0), _MM_HINT_T0);
                _mm_prefetch((char*)(tC + 8), _MM_HINT_T0);
                _mm_prefetch((char*)(tC + ldc), _MM_HINT_T0);
                _mm_prefetch((char*)(tC + ldc + 8), _MM_HINT_T0);
                _mm_prefetch((char*)(tC + 2 * ldc), _MM_HINT_T0);
                _mm_prefetch((char*)(tC + 2 * ldc + 8), _MM_HINT_T0);
#endif
                // clear scratch registers.
                ymm4 = _mm256_setzero_pd();
                ymm5 = _mm256_setzero_pd();
                ymm6 = _mm256_setzero_pd();
                ymm7 = _mm256_setzero_pd();
                ymm8 = _mm256_setzero_pd();
                ymm9 = _mm256_setzero_pd();
                ymm10 = _mm256_setzero_pd();
                ymm11 = _mm256_setzero_pd();
                ymm12 = _mm256_setzero_pd();
                ymm13 = _mm256_setzero_pd();
                ymm14 = _mm256_setzero_pd();
                ymm15 = _mm256_setzero_pd();

                for (k = 0; k < K; ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    // This loop is processing D_MR x K
                    ymm0 = _mm256_broadcast_sd(tB + tb_inc_col * 0);
                    ymm1 = _mm256_broadcast_sd(tB + tb_inc_col * 1);
                    ymm2 = _mm256_broadcast_sd(tB + tb_inc_col * 2);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_pd(tA);
                    _mm256_storeu_pd(tA_packed, ymm3); // the packing of matrix A
                                                       //                   ymm4 += ymm0 * ymm3;
                    ymm4 = _mm256_fmadd_pd(ymm0, ymm3, ymm4);
                    //                    ymm8 += ymm1 * ymm3;
                    ymm8 = _mm256_fmadd_pd(ymm1, ymm3, ymm8);
                    //                    ymm12 += ymm2 * ymm3;
                    ymm12 = _mm256_fmadd_pd(ymm2, ymm3, ymm12);

                    ymm3 = _mm256_loadu_pd(tA + 4);
                    _mm256_storeu_pd(tA_packed + 4, ymm3); // the packing of matrix A
                                                           //                    ymm5 += ymm0 * ymm3;
                    ymm5 = _mm256_fmadd_pd(ymm0, ymm3, ymm5);
                    //                    ymm9 += ymm1 * ymm3;
                    ymm9 = _mm256_fmadd_pd(ymm1, ymm3, ymm9);
                    //                    ymm13 += ymm2 * ymm3;
                    ymm13 = _mm256_fmadd_pd(ymm2, ymm3, ymm13);

                    ymm3 = _mm256_loadu_pd(tA + 8);
                    _mm256_storeu_pd(tA_packed + 8, ymm3); // the packing of matrix A
                                                           //                   ymm6 += ymm0 * ymm3;
                    ymm6 = _mm256_fmadd_pd(ymm0, ymm3, ymm6);
                    //                    ymm10 += ymm1 * ymm3;
                    ymm10 = _mm256_fmadd_pd(ymm1, ymm3, ymm10);
                    //                    ymm14 += ymm2 * ymm3;
                    ymm14 = _mm256_fmadd_pd(ymm2, ymm3, ymm14);

                    ymm3 = _mm256_loadu_pd(tA + 12);
                    _mm256_storeu_pd(tA_packed + 12, ymm3); // the packing of matrix A
                                                            //                    ymm7 += ymm0 * ymm3;
                    ymm7 = _mm256_fmadd_pd(ymm0, ymm3, ymm7);
                    //                    ymm11 += ymm1 * ymm3;
                    ymm11 = _mm256_fmadd_pd(ymm1, ymm3, ymm11);
                    //                   ymm15 += ymm2 * ymm3;
                    ymm15 = _mm256_fmadd_pd(ymm2, ymm3, ymm15);

                    tA += lda;
                    tA_packed += D_MR;
                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_sd(alpha_cast);
                //ymm1 = _mm256_broadcast_sd(beta_cast);

                //multiply A*B by alpha.
                ymm4 = _mm256_mul_pd(ymm4, ymm0);
                ymm5 = _mm256_mul_pd(ymm5, ymm0);
                ymm6 = _mm256_mul_pd(ymm6, ymm0);
                ymm7 = _mm256_mul_pd(ymm7, ymm0);
                ymm8 = _mm256_mul_pd(ymm8, ymm0);
                ymm9 = _mm256_mul_pd(ymm9, ymm0);
                ymm10 = _mm256_mul_pd(ymm10, ymm0);
                ymm11 = _mm256_mul_pd(ymm11, ymm0);
                ymm12 = _mm256_mul_pd(ymm12, ymm0);
                ymm13 = _mm256_mul_pd(ymm13, ymm0);
                ymm14 = _mm256_mul_pd(ymm14, ymm0);
                ymm15 = _mm256_mul_pd(ymm15, ymm0);

                // multiply C by beta and accumulate col 1.
                /*ymm2 = _mm256_loadu_pd(tC);
                ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4);
                ymm2 = _mm256_loadu_pd(tC + 4);
                ymm5 = _mm256_fmadd_pd(ymm2, ymm1, ymm5);
                ymm2 = _mm256_loadu_pd(tC + 8);
                ymm6 = _mm256_fmadd_pd(ymm2, ymm1, ymm6);
                ymm2 = _mm256_loadu_pd(tC + 12);
                ymm7 = _mm256_fmadd_pd(ymm2, ymm1, ymm7);*/
                _mm256_storeu_pd(tC, ymm4);
                _mm256_storeu_pd(tC + 4, ymm5);
                _mm256_storeu_pd(tC + 8, ymm6);
                _mm256_storeu_pd(tC + 12, ymm7);

                // multiply C by beta and accumulate, col 2.
                tC += ldc;
                /*ymm2 = _mm256_loadu_pd(tC);
                ymm8 = _mm256_fmadd_pd(ymm2, ymm1, ymm8);
                ymm2 = _mm256_loadu_pd(tC + 4);
                ymm9 = _mm256_fmadd_pd(ymm2, ymm1, ymm9);
                ymm2 = _mm256_loadu_pd(tC + 8);
                ymm10 = _mm256_fmadd_pd(ymm2, ymm1, ymm10);
                ymm2 = _mm256_loadu_pd(tC + 12);
                ymm11 = _mm256_fmadd_pd(ymm2, ymm1, ymm11);*/
                _mm256_storeu_pd(tC, ymm8);
                _mm256_storeu_pd(tC + 4, ymm9);
                _mm256_storeu_pd(tC + 8, ymm10);
                _mm256_storeu_pd(tC + 12, ymm11);

                // multiply C by beta and accumulate, col 3.
                tC += ldc;
                /*ymm2 = _mm256_loadu_pd(tC);
                ymm12 = _mm256_fmadd_pd(ymm2, ymm1, ymm12);
                ymm2 = _mm256_loadu_pd(tC + 4);
                ymm13 = _mm256_fmadd_pd(ymm2, ymm1, ymm13);
                ymm2 = _mm256_loadu_pd(tC + 8);
                ymm14 = _mm256_fmadd_pd(ymm2, ymm1, ymm14);
                ymm2 = _mm256_loadu_pd(tC + 12);
                ymm15 = _mm256_fmadd_pd(ymm2, ymm1, ymm15);*/
                _mm256_storeu_pd(tC, ymm12);
                _mm256_storeu_pd(tC + 4, ymm13);
                _mm256_storeu_pd(tC + 8, ymm14);
                _mm256_storeu_pd(tC + 12, ymm15);

                // modify the pointer arithematic to use packed A matrix.
                col_idx_start = NR;
                tA_packed = D_A_pack;
                row_idx_packed = 0;
                lda_packed = D_MR;
            }
            // Process NR columns of C matrix at a time.
            for (col_idx = col_idx_start; (col_idx + (NR - 1)) < N; col_idx += NR)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = tA_packed + row_idx_packed;

#if 0//def BLIS_ENABLE_PREFETCH
                _mm_prefetch((char*)(tC + 0), _MM_HINT_T0);
                _mm_prefetch((char*)(tC + 8), _MM_HINT_T0);
                _mm_prefetch((char*)(tC + ldc), _MM_HINT_T0);
                _mm_prefetch((char*)(tC + ldc + 8), _MM_HINT_T0);
                _mm_prefetch((char*)(tC + 2 * ldc), _MM_HINT_T0);
                _mm_prefetch((char*)(tC + 2 * ldc + 8), _MM_HINT_T0);
#endif
                // clear scratch registers.
                ymm4 = _mm256_setzero_pd();
                ymm5 = _mm256_setzero_pd();
                ymm6 = _mm256_setzero_pd();
                ymm7 = _mm256_setzero_pd();
                ymm8 = _mm256_setzero_pd();
                ymm9 = _mm256_setzero_pd();
                ymm10 = _mm256_setzero_pd();
                ymm11 = _mm256_setzero_pd();
                ymm12 = _mm256_setzero_pd();
                ymm13 = _mm256_setzero_pd();
                ymm14 = _mm256_setzero_pd();
                ymm15 = _mm256_setzero_pd();

                for (k = 0; k < K; ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    // This loop is processing D_MR x K
                    ymm0 = _mm256_broadcast_sd(tB + tb_inc_col * 0);
                    ymm1 = _mm256_broadcast_sd(tB + tb_inc_col * 1);
                    ymm2 = _mm256_broadcast_sd(tB + tb_inc_col * 2);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_pd(tA);
                    //                   ymm4 += ymm0 * ymm3;
                    ymm4 = _mm256_fmadd_pd(ymm0, ymm3, ymm4);
                    //                    ymm8 += ymm1 * ymm3;
                    ymm8 = _mm256_fmadd_pd(ymm1, ymm3, ymm8);
                    //                    ymm12 += ymm2 * ymm3;
                    ymm12 = _mm256_fmadd_pd(ymm2, ymm3, ymm12);

                    ymm3 = _mm256_loadu_pd(tA + 4);
                    //                    ymm5 += ymm0 * ymm3;
                    ymm5 = _mm256_fmadd_pd(ymm0, ymm3, ymm5);
                    //                    ymm9 += ymm1 * ymm3;
                    ymm9 = _mm256_fmadd_pd(ymm1, ymm3, ymm9);
                    //                    ymm13 += ymm2 * ymm3;
                    ymm13 = _mm256_fmadd_pd(ymm2, ymm3, ymm13);

                    ymm3 = _mm256_loadu_pd(tA + 8);
                    //                   ymm6 += ymm0 * ymm3;
                    ymm6 = _mm256_fmadd_pd(ymm0, ymm3, ymm6);
                    //                    ymm10 += ymm1 * ymm3;
                    ymm10 = _mm256_fmadd_pd(ymm1, ymm3, ymm10);
                    //                    ymm14 += ymm2 * ymm3;
                    ymm14 = _mm256_fmadd_pd(ymm2, ymm3, ymm14);

                    ymm3 = _mm256_loadu_pd(tA + 12);
                    //                    ymm7 += ymm0 * ymm3;
                    ymm7 = _mm256_fmadd_pd(ymm0, ymm3, ymm7);
                    //                    ymm11 += ymm1 * ymm3;
                    ymm11 = _mm256_fmadd_pd(ymm1, ymm3, ymm11);
                    //                   ymm15 += ymm2 * ymm3;
                    ymm15 = _mm256_fmadd_pd(ymm2, ymm3, ymm15);

                    tA += lda_packed;
                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_sd(alpha_cast);
                //ymm1 = _mm256_broadcast_sd(beta_cast);

                //multiply A*B by alpha.
                ymm4 = _mm256_mul_pd(ymm4, ymm0);
                ymm5 = _mm256_mul_pd(ymm5, ymm0);
                ymm6 = _mm256_mul_pd(ymm6, ymm0);
                ymm7 = _mm256_mul_pd(ymm7, ymm0);
                ymm8 = _mm256_mul_pd(ymm8, ymm0);
                ymm9 = _mm256_mul_pd(ymm9, ymm0);
                ymm10 = _mm256_mul_pd(ymm10, ymm0);
                ymm11 = _mm256_mul_pd(ymm11, ymm0);
                ymm12 = _mm256_mul_pd(ymm12, ymm0);
                ymm13 = _mm256_mul_pd(ymm13, ymm0);
                ymm14 = _mm256_mul_pd(ymm14, ymm0);
                ymm15 = _mm256_mul_pd(ymm15, ymm0);

                // multiply C by beta and accumulate col 1.
                /*ymm2 = _mm256_loadu_pd(tC);
                ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4);
                ymm2 = _mm256_loadu_pd(tC + 4);
                ymm5 = _mm256_fmadd_pd(ymm2, ymm1, ymm5);
                ymm2 = _mm256_loadu_pd(tC + 8);
                ymm6 = _mm256_fmadd_pd(ymm2, ymm1, ymm6);
                ymm2 = _mm256_loadu_pd(tC + 12);
                ymm7 = _mm256_fmadd_pd(ymm2, ymm1, ymm7);*/
                _mm256_storeu_pd(tC, ymm4);
                _mm256_storeu_pd(tC + 4, ymm5);
                _mm256_storeu_pd(tC + 8, ymm6);
                _mm256_storeu_pd(tC + 12, ymm7);

                // multiply C by beta and accumulate, col 2.
                tC += ldc;
                /*ymm2 = _mm256_loadu_pd(tC);
                ymm8 = _mm256_fmadd_pd(ymm2, ymm1, ymm8);
                ymm2 = _mm256_loadu_pd(tC + 4);
                ymm9 = _mm256_fmadd_pd(ymm2, ymm1, ymm9);
                ymm2 = _mm256_loadu_pd(tC + 8);
                ymm10 = _mm256_fmadd_pd(ymm2, ymm1, ymm10);
                ymm2 = _mm256_loadu_pd(tC + 12);
                ymm11 = _mm256_fmadd_pd(ymm2, ymm1, ymm11);*/
                _mm256_storeu_pd(tC, ymm8);
                _mm256_storeu_pd(tC + 4, ymm9);
                _mm256_storeu_pd(tC + 8, ymm10);
                _mm256_storeu_pd(tC + 12, ymm11);

                // multiply C by beta and accumulate, col 3.
                tC += ldc;
                /*ymm2 = _mm256_loadu_pd(tC);
                ymm12 = _mm256_fmadd_pd(ymm2, ymm1, ymm12);
                ymm2 = _mm256_loadu_pd(tC + 4);
                ymm13 = _mm256_fmadd_pd(ymm2, ymm1, ymm13);
                ymm2 = _mm256_loadu_pd(tC + 8);
                ymm14 = _mm256_fmadd_pd(ymm2, ymm1, ymm14);
                ymm2 = _mm256_loadu_pd(tC + 12);
                ymm15 = _mm256_fmadd_pd(ymm2, ymm1, ymm15);*/
                _mm256_storeu_pd(tC, ymm12);
                _mm256_storeu_pd(tC + 4, ymm13);
                _mm256_storeu_pd(tC + 8, ymm14);
                _mm256_storeu_pd(tC + 12, ymm15);

            }
            n_remainder = N - col_idx;

            // if the N is not multiple of 3.
            // handling edge case.
            if (n_remainder == 2)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                // clear scratch registers.
                ymm8 = _mm256_setzero_pd();
                ymm9 = _mm256_setzero_pd();
                ymm10 = _mm256_setzero_pd();
                ymm11 = _mm256_setzero_pd();
                ymm12 = _mm256_setzero_pd();
                ymm13 = _mm256_setzero_pd();
                ymm14 = _mm256_setzero_pd();
                ymm15 = _mm256_setzero_pd();

                for (k = 0; k < K; ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_sd(tB + tb_inc_col * 0);
                    ymm1 = _mm256_broadcast_sd(tB + tb_inc_col * 1);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_pd(tA);
                    ymm8 = _mm256_fmadd_pd(ymm0, ymm3, ymm8);
                    ymm12 = _mm256_fmadd_pd(ymm1, ymm3, ymm12);

                    ymm3 = _mm256_loadu_pd(tA + 4);
                    ymm9 = _mm256_fmadd_pd(ymm0, ymm3, ymm9);
                    ymm13 = _mm256_fmadd_pd(ymm1, ymm3, ymm13);

                    ymm3 = _mm256_loadu_pd(tA + 8);
                    ymm10 = _mm256_fmadd_pd(ymm0, ymm3, ymm10);
                    ymm14 = _mm256_fmadd_pd(ymm1, ymm3, ymm14);

                    ymm3 = _mm256_loadu_pd(tA + 12);
                    ymm11 = _mm256_fmadd_pd(ymm0, ymm3, ymm11);
                    ymm15 = _mm256_fmadd_pd(ymm1, ymm3, ymm15);

                    tA += lda;

                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_sd(alpha_cast);
                //ymm1 = _mm256_broadcast_sd(beta_cast);

                //multiply A*B by alpha.
                ymm8 = _mm256_mul_pd(ymm8, ymm0);
                ymm9 = _mm256_mul_pd(ymm9, ymm0);
                ymm10 = _mm256_mul_pd(ymm10, ymm0);
                ymm11 = _mm256_mul_pd(ymm11, ymm0);
                ymm12 = _mm256_mul_pd(ymm12, ymm0);
                ymm13 = _mm256_mul_pd(ymm13, ymm0);
                ymm14 = _mm256_mul_pd(ymm14, ymm0);
                ymm15 = _mm256_mul_pd(ymm15, ymm0);

                // multiply C by beta and accumulate, col 1.
                /*ymm2 = _mm256_loadu_pd(tC + 0);
                ymm8 = _mm256_fmadd_pd(ymm2, ymm1, ymm8);
                ymm2 = _mm256_loadu_pd(tC + 4);
                ymm9 = _mm256_fmadd_pd(ymm2, ymm1, ymm9);
                ymm2 = _mm256_loadu_pd(tC + 8);
                ymm10 = _mm256_fmadd_pd(ymm2, ymm1, ymm10);
                ymm2 = _mm256_loadu_pd(tC + 12);
                ymm11 = _mm256_fmadd_pd(ymm2, ymm1, ymm11);*/
                _mm256_storeu_pd(tC + 0, ymm8);
                _mm256_storeu_pd(tC + 4, ymm9);
                _mm256_storeu_pd(tC + 8, ymm10);
                _mm256_storeu_pd(tC + 12, ymm11);

                // multiply C by beta and accumulate, col 2.
                tC += ldc;
                /*ymm2 = _mm256_loadu_pd(tC);
                ymm12 = _mm256_fmadd_pd(ymm2, ymm1, ymm12);
                ymm2 = _mm256_loadu_pd(tC + 4);
                ymm13 = _mm256_fmadd_pd(ymm2, ymm1, ymm13);
                ymm2 = _mm256_loadu_pd(tC + 8);
                ymm14 = _mm256_fmadd_pd(ymm2, ymm1, ymm14);
                ymm2 = _mm256_loadu_pd(tC + 12);
                ymm15 = _mm256_fmadd_pd(ymm2, ymm1, ymm15);*/
                _mm256_storeu_pd(tC, ymm12);
                _mm256_storeu_pd(tC + 4, ymm13);
                _mm256_storeu_pd(tC + 8, ymm14);
                _mm256_storeu_pd(tC + 12, ymm15);

                col_idx += 2;
            }
            // if the N is not multiple of 3.
            // handling edge case.
            if (n_remainder == 1)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                // clear scratch registers.
                ymm12 = _mm256_setzero_pd();
                ymm13 = _mm256_setzero_pd();
                ymm14 = _mm256_setzero_pd();
                ymm15 = _mm256_setzero_pd();

                for (k = 0; k < K; ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_sd(tB + tb_inc_col * 0);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_pd(tA);
                    ymm12 = _mm256_fmadd_pd(ymm0, ymm3, ymm12);

                    ymm3 = _mm256_loadu_pd(tA + 4);
                    ymm13 = _mm256_fmadd_pd(ymm0, ymm3, ymm13);

                    ymm3 = _mm256_loadu_pd(tA + 8);
                    ymm14 = _mm256_fmadd_pd(ymm0, ymm3, ymm14);

                    ymm3 = _mm256_loadu_pd(tA + 12);
                    ymm15 = _mm256_fmadd_pd(ymm0, ymm3, ymm15);

                    tA += lda;

                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_sd(alpha_cast);
                //ymm1 = _mm256_broadcast_sd(beta_cast);

                //multiply A*B by alpha.
                ymm12 = _mm256_mul_pd(ymm12, ymm0);
                ymm13 = _mm256_mul_pd(ymm13, ymm0);
                ymm14 = _mm256_mul_pd(ymm14, ymm0);
                ymm15 = _mm256_mul_pd(ymm15, ymm0);

                // multiply C by beta and accumulate.
                /*ymm2 = _mm256_loadu_pd(tC + 0);
                ymm12 = _mm256_fmadd_pd(ymm2, ymm1, ymm12);
                ymm2 = _mm256_loadu_pd(tC + 4);
                ymm13 = _mm256_fmadd_pd(ymm2, ymm1, ymm13);
                ymm2 = _mm256_loadu_pd(tC + 8);
                ymm14 = _mm256_fmadd_pd(ymm2, ymm1, ymm14);
                ymm2 = _mm256_loadu_pd(tC + 12);
                ymm15 = _mm256_fmadd_pd(ymm2, ymm1, ymm15);*/

                _mm256_storeu_pd(tC + 0, ymm12);
                _mm256_storeu_pd(tC + 4, ymm13);
                _mm256_storeu_pd(tC + 8, ymm14);
                _mm256_storeu_pd(tC + 12, ymm15);
            }
        }

        m_remainder = M - row_idx;

        if (m_remainder >= 12)
        {
            m_remainder -= 12;

            for (col_idx = 0; (col_idx + 2) < N; col_idx += 3)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                // clear scratch registers.
                ymm4 = _mm256_setzero_pd();
                ymm5 = _mm256_setzero_pd();
                ymm6 = _mm256_setzero_pd();
                ymm8 = _mm256_setzero_pd();
                ymm9 = _mm256_setzero_pd();
                ymm10 = _mm256_setzero_pd();
                ymm12 = _mm256_setzero_pd();
                ymm13 = _mm256_setzero_pd();
                ymm14 = _mm256_setzero_pd();

                for (k = 0; k < K; ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_sd(tB + tb_inc_col * 0);
                    ymm1 = _mm256_broadcast_sd(tB + tb_inc_col * 1);
                    ymm2 = _mm256_broadcast_sd(tB + tb_inc_col * 2);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_pd(tA);
                    //                   ymm4 += ymm0 * ymm3;
                    ymm4 = _mm256_fmadd_pd(ymm0, ymm3, ymm4);
                    //                    ymm8 += ymm1 * ymm3;
                    ymm8 = _mm256_fmadd_pd(ymm1, ymm3, ymm8);
                    //                    ymm12 += ymm2 * ymm3;
                    ymm12 = _mm256_fmadd_pd(ymm2, ymm3, ymm12);

                    ymm3 = _mm256_loadu_pd(tA + 4);
                    //                    ymm5 += ymm0 * ymm3;
                    ymm5 = _mm256_fmadd_pd(ymm0, ymm3, ymm5);
                    //                    ymm9 += ymm1 * ymm3;
                    ymm9 = _mm256_fmadd_pd(ymm1, ymm3, ymm9);
                    //                    ymm13 += ymm2 * ymm3;
                    ymm13 = _mm256_fmadd_pd(ymm2, ymm3, ymm13);

                    ymm3 = _mm256_loadu_pd(tA + 8);
                    //                   ymm6 += ymm0 * ymm3;
                    ymm6 = _mm256_fmadd_pd(ymm0, ymm3, ymm6);
                    //                    ymm10 += ymm1 * ymm3;
                    ymm10 = _mm256_fmadd_pd(ymm1, ymm3, ymm10);
                    //                    ymm14 += ymm2 * ymm3;
                    ymm14 = _mm256_fmadd_pd(ymm2, ymm3, ymm14);

                    tA += lda;
                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_sd(alpha_cast);
                //ymm1 = _mm256_broadcast_sd(beta_cast);

                //multiply A*B by alpha.
                ymm4 = _mm256_mul_pd(ymm4, ymm0);
                ymm5 = _mm256_mul_pd(ymm5, ymm0);
                ymm6 = _mm256_mul_pd(ymm6, ymm0);
                ymm8 = _mm256_mul_pd(ymm8, ymm0);
                ymm9 = _mm256_mul_pd(ymm9, ymm0);
                ymm10 = _mm256_mul_pd(ymm10, ymm0);
                ymm12 = _mm256_mul_pd(ymm12, ymm0);
                ymm13 = _mm256_mul_pd(ymm13, ymm0);
                ymm14 = _mm256_mul_pd(ymm14, ymm0);

                // multiply C by beta and accumulate.
                /*ymm2 = _mm256_loadu_pd(tC);
                ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4);
                ymm2 = _mm256_loadu_pd(tC + 4);
                ymm5 = _mm256_fmadd_pd(ymm2, ymm1, ymm5);
                ymm2 = _mm256_loadu_pd(tC + 8);
                ymm6 = _mm256_fmadd_pd(ymm2, ymm1, ymm6);*/
                _mm256_storeu_pd(tC, ymm4);
                _mm256_storeu_pd(tC + 4, ymm5);
                _mm256_storeu_pd(tC + 8, ymm6);

                // multiply C by beta and accumulate.
                tC += ldc;
                /*ymm2 = _mm256_loadu_pd(tC);
                ymm8 = _mm256_fmadd_pd(ymm2, ymm1, ymm8);
                ymm2 = _mm256_loadu_pd(tC + 4);
                ymm9 = _mm256_fmadd_pd(ymm2, ymm1, ymm9);
                ymm2 = _mm256_loadu_pd(tC + 8);
                ymm10 = _mm256_fmadd_pd(ymm2, ymm1, ymm10);*/
                _mm256_storeu_pd(tC, ymm8);
                _mm256_storeu_pd(tC + 4, ymm9);
                _mm256_storeu_pd(tC + 8, ymm10);

                // multiply C by beta and accumulate.
                tC += ldc;
                /*ymm2 = _mm256_loadu_pd(tC);
                ymm12 = _mm256_fmadd_pd(ymm2, ymm1, ymm12);
                ymm2 = _mm256_loadu_pd(tC + 4);
                ymm13 = _mm256_fmadd_pd(ymm2, ymm1, ymm13);
                ymm2 = _mm256_loadu_pd(tC + 8);
                ymm14 = _mm256_fmadd_pd(ymm2, ymm1, ymm14);*/
                _mm256_storeu_pd(tC, ymm12);
                _mm256_storeu_pd(tC + 4, ymm13);
                _mm256_storeu_pd(tC + 8, ymm14);

            }
            n_remainder = N - col_idx;
            // if the N is not multiple of 3.
            // handling edge case.
            if (n_remainder == 2)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                // clear scratch registers.
                ymm8 = _mm256_setzero_pd();
                ymm9 = _mm256_setzero_pd();
                ymm10 = _mm256_setzero_pd();
                ymm12 = _mm256_setzero_pd();
                ymm13 = _mm256_setzero_pd();
                ymm14 = _mm256_setzero_pd();

                for (k = 0; k < K; ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_sd(tB + tb_inc_col * 0);
                    ymm1 = _mm256_broadcast_sd(tB + tb_inc_col * 1);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_pd(tA);
                    ymm8 = _mm256_fmadd_pd(ymm0, ymm3, ymm8);
                    ymm12 = _mm256_fmadd_pd(ymm1, ymm3, ymm12);

                    ymm3 = _mm256_loadu_pd(tA + 4);
                    ymm9 = _mm256_fmadd_pd(ymm0, ymm3, ymm9);
                    ymm13 = _mm256_fmadd_pd(ymm1, ymm3, ymm13);

                    ymm3 = _mm256_loadu_pd(tA + 8);
                    ymm10 = _mm256_fmadd_pd(ymm0, ymm3, ymm10);
                    ymm14 = _mm256_fmadd_pd(ymm1, ymm3, ymm14);

                    tA += lda;

                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_sd(alpha_cast);
                //ymm1 = _mm256_broadcast_sd(beta_cast);

                //multiply A*B by alpha.
                ymm8 = _mm256_mul_pd(ymm8, ymm0);
                ymm9 = _mm256_mul_pd(ymm9, ymm0);
                ymm10 = _mm256_mul_pd(ymm10, ymm0);
                ymm12 = _mm256_mul_pd(ymm12, ymm0);
                ymm13 = _mm256_mul_pd(ymm13, ymm0);
                ymm14 = _mm256_mul_pd(ymm14, ymm0);

                // multiply C by beta and accumulate.
                /*ymm2 = _mm256_loadu_pd(tC + 0);
                ymm8 = _mm256_fmadd_pd(ymm2, ymm1, ymm8);
                ymm2 = _mm256_loadu_pd(tC + 4);
                ymm9 = _mm256_fmadd_pd(ymm2, ymm1, ymm9);
                ymm2 = _mm256_loadu_pd(tC + 8);
                ymm10 = _mm256_fmadd_pd(ymm2, ymm1, ymm10);*/
                _mm256_storeu_pd(tC + 0, ymm8);
                _mm256_storeu_pd(tC + 4, ymm9);
                _mm256_storeu_pd(tC + 8, ymm10);

                // multiply C by beta and accumulate.
                tC += ldc;
                /*ymm2 = _mm256_loadu_pd(tC);
                ymm12 = _mm256_fmadd_pd(ymm2, ymm1, ymm12);
                ymm2 = _mm256_loadu_pd(tC + 4);
                ymm13 = _mm256_fmadd_pd(ymm2, ymm1, ymm13);
                ymm2 = _mm256_loadu_pd(tC + 8);
                ymm14 = _mm256_fmadd_pd(ymm2, ymm1, ymm14);*/
                _mm256_storeu_pd(tC, ymm12);
                _mm256_storeu_pd(tC + 4, ymm13);
                _mm256_storeu_pd(tC + 8, ymm14);

                col_idx += 2;
            }
            // if the N is not multiple of 3.
            // handling edge case.
            if (n_remainder == 1)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                // clear scratch registers.
                ymm12 = _mm256_setzero_pd();
                ymm13 = _mm256_setzero_pd();
                ymm14 = _mm256_setzero_pd();

                for (k = 0; k < K; ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_sd(tB + tb_inc_col * 0);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_pd(tA);
                    ymm12 = _mm256_fmadd_pd(ymm0, ymm3, ymm12);

                    ymm3 = _mm256_loadu_pd(tA + 4);
                    ymm13 = _mm256_fmadd_pd(ymm0, ymm3, ymm13);

                    ymm3 = _mm256_loadu_pd(tA + 8);
                    ymm14 = _mm256_fmadd_pd(ymm0, ymm3, ymm14);

                    tA += lda;

                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_sd(alpha_cast);
                //ymm1 = _mm256_broadcast_sd(beta_cast);

                //multiply A*B by alpha.
                ymm12 = _mm256_mul_pd(ymm12, ymm0);
                ymm13 = _mm256_mul_pd(ymm13, ymm0);
                ymm14 = _mm256_mul_pd(ymm14, ymm0);

                // multiply C by beta and accumulate.
                /*ymm2 = _mm256_loadu_pd(tC + 0);
                ymm12 = _mm256_fmadd_pd(ymm2, ymm1, ymm12);
                ymm2 = _mm256_loadu_pd(tC + 4);
                ymm13 = _mm256_fmadd_pd(ymm2, ymm1, ymm13);
                ymm2 = _mm256_loadu_pd(tC + 8);
                ymm14 = _mm256_fmadd_pd(ymm2, ymm1, ymm14);*/

                _mm256_storeu_pd(tC + 0, ymm12);
                _mm256_storeu_pd(tC + 4, ymm13);
                _mm256_storeu_pd(tC + 8, ymm14);
            }

            row_idx += 12;
        }

        if (m_remainder >= 8)
        {
            m_remainder -= 8;

            for (col_idx = 0; (col_idx + 2) < N; col_idx += 3)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                // clear scratch registers.
                ymm4 = _mm256_setzero_pd();
                ymm5 = _mm256_setzero_pd();
                ymm6 = _mm256_setzero_pd();
                ymm7 = _mm256_setzero_pd();
                ymm8 = _mm256_setzero_pd();
                ymm9 = _mm256_setzero_pd();

                for (k = 0; k < K; ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_sd(tB + tb_inc_col * 0);
                    ymm1 = _mm256_broadcast_sd(tB + tb_inc_col * 1);
                    ymm2 = _mm256_broadcast_sd(tB + tb_inc_col * 2);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_pd(tA);
                    ymm4 = _mm256_fmadd_pd(ymm0, ymm3, ymm4);
                    ymm6 = _mm256_fmadd_pd(ymm1, ymm3, ymm6);
                    ymm8 = _mm256_fmadd_pd(ymm2, ymm3, ymm8);

                    ymm3 = _mm256_loadu_pd(tA + 4);
                    ymm5 = _mm256_fmadd_pd(ymm0, ymm3, ymm5);
                    ymm7 = _mm256_fmadd_pd(ymm1, ymm3, ymm7);
                    ymm9 = _mm256_fmadd_pd(ymm2, ymm3, ymm9);

                    tA += lda;
                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_sd(alpha_cast);
                //ymm1 = _mm256_broadcast_sd(beta_cast);

                //multiply A*B by alpha.
                ymm4 = _mm256_mul_pd(ymm4, ymm0);
                ymm5 = _mm256_mul_pd(ymm5, ymm0);
                ymm6 = _mm256_mul_pd(ymm6, ymm0);
                ymm7 = _mm256_mul_pd(ymm7, ymm0);
                ymm8 = _mm256_mul_pd(ymm8, ymm0);
                ymm9 = _mm256_mul_pd(ymm9, ymm0);

                // multiply C by beta and accumulate.
                /*ymm2 = _mm256_loadu_pd(tC);
                ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4);
                ymm2 = _mm256_loadu_pd(tC + 4);
                ymm5 = _mm256_fmadd_pd(ymm2, ymm1, ymm5);*/
                _mm256_storeu_pd(tC, ymm4);
                _mm256_storeu_pd(tC + 4, ymm5);

                // multiply C by beta and accumulate.
                tC += ldc;
                /*ymm2 = _mm256_loadu_pd(tC);
                ymm6 = _mm256_fmadd_pd(ymm2, ymm1, ymm6);
                ymm2 = _mm256_loadu_pd(tC + 4);
                ymm7 = _mm256_fmadd_pd(ymm2, ymm1, ymm7);*/
                _mm256_storeu_pd(tC, ymm6);
                _mm256_storeu_pd(tC + 4, ymm7);

                // multiply C by beta and accumulate.
                tC += ldc;
                /*ymm2 = _mm256_loadu_pd(tC);
                ymm8 = _mm256_fmadd_pd(ymm2, ymm1, ymm8);
                ymm2 = _mm256_loadu_pd(tC + 4);
                ymm9 = _mm256_fmadd_pd(ymm2, ymm1, ymm9);*/
                _mm256_storeu_pd(tC, ymm8);
                _mm256_storeu_pd(tC + 4, ymm9);

            }
            n_remainder = N - col_idx;
            // if the N is not multiple of 3.
            // handling edge case.
            if (n_remainder == 2)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                // clear scratch registers.
                ymm4 = _mm256_setzero_pd();
                ymm5 = _mm256_setzero_pd();
                ymm6 = _mm256_setzero_pd();
                ymm7 = _mm256_setzero_pd();

                for (k = 0; k < K; ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_sd(tB + tb_inc_col * 0);
                    ymm1 = _mm256_broadcast_sd(tB + tb_inc_col * 1);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_pd(tA);
                    ymm4 = _mm256_fmadd_pd(ymm0, ymm3, ymm4);
                    ymm6 = _mm256_fmadd_pd(ymm1, ymm3, ymm6);

                    ymm3 = _mm256_loadu_pd(tA + 4);
                    ymm5 = _mm256_fmadd_pd(ymm0, ymm3, ymm5);
                    ymm7 = _mm256_fmadd_pd(ymm1, ymm3, ymm7);

                    tA += lda;
                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_sd(alpha_cast);
                //ymm1 = _mm256_broadcast_sd(beta_cast);

                //multiply A*B by alpha.
                ymm4 = _mm256_mul_pd(ymm4, ymm0);
                ymm5 = _mm256_mul_pd(ymm5, ymm0);
                ymm6 = _mm256_mul_pd(ymm6, ymm0);
                ymm7 = _mm256_mul_pd(ymm7, ymm0);

                // multiply C by beta and accumulate.
                /*ymm2 = _mm256_loadu_pd(tC);
                ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4);
                ymm2 = _mm256_loadu_pd(tC + 4);
                ymm5 = _mm256_fmadd_pd(ymm2, ymm1, ymm5);*/
                _mm256_storeu_pd(tC, ymm4);
                _mm256_storeu_pd(tC + 4, ymm5);

                // multiply C by beta and accumulate.
                tC += ldc;
                /*ymm2 = _mm256_loadu_pd(tC);
                ymm6 = _mm256_fmadd_pd(ymm2, ymm1, ymm6);
                ymm2 = _mm256_loadu_pd(tC + 4);
                ymm7 = _mm256_fmadd_pd(ymm2, ymm1, ymm7);*/
                _mm256_storeu_pd(tC, ymm6);
                _mm256_storeu_pd(tC + 4, ymm7);

                col_idx += 2;

            }
            // if the N is not multiple of 3.
            // handling edge case.
            if (n_remainder == 1)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                ymm4 = _mm256_setzero_pd();
                ymm5 = _mm256_setzero_pd();

                for (k = 0; k < K; ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_sd(tB + tb_inc_col * 0);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_pd(tA);
                    ymm4 = _mm256_fmadd_pd(ymm0, ymm3, ymm4);

                    ymm3 = _mm256_loadu_pd(tA + 4);
                    ymm5 = _mm256_fmadd_pd(ymm0, ymm3, ymm5);

                    tA += lda;
                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_sd(alpha_cast);
                //ymm1 = _mm256_broadcast_sd(beta_cast);

                ymm4 = _mm256_mul_pd(ymm4, ymm0);
                ymm5 = _mm256_mul_pd(ymm5, ymm0);

                // multiply C by beta and accumulate.
                /*ymm2 = _mm256_loadu_pd(tC);
                ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4);
                ymm2 = _mm256_loadu_pd(tC + 4);
                ymm5 = _mm256_fmadd_pd(ymm2, ymm1, ymm5);*/
                _mm256_storeu_pd(tC, ymm4);
                _mm256_storeu_pd(tC + 4, ymm5);

            }

            row_idx += 8;
        }

        if (m_remainder >= 4)
        {
            m_remainder -= 4;

            for (col_idx = 0; (col_idx + 2) < N; col_idx += 3)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                // clear scratch registers.
                ymm4 = _mm256_setzero_pd();
                ymm5 = _mm256_setzero_pd();
                ymm6 = _mm256_setzero_pd();

                for (k = 0; k < K; ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_sd(tB + tb_inc_col * 0);
                    ymm1 = _mm256_broadcast_sd(tB + tb_inc_col * 1);
                    ymm2 = _mm256_broadcast_sd(tB + tb_inc_col * 2);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_pd(tA);
                    ymm4 = _mm256_fmadd_pd(ymm0, ymm3, ymm4);
                    ymm5 = _mm256_fmadd_pd(ymm1, ymm3, ymm5);
                    ymm6 = _mm256_fmadd_pd(ymm2, ymm3, ymm6);

                    tA += lda;
                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_sd(alpha_cast);
                //ymm1 = _mm256_broadcast_sd(beta_cast);

                //multiply A*B by alpha.
                ymm4 = _mm256_mul_pd(ymm4, ymm0);
                ymm5 = _mm256_mul_pd(ymm5, ymm0);
                ymm6 = _mm256_mul_pd(ymm6, ymm0);

                // multiply C by beta and accumulate.
                /*ymm2 = _mm256_loadu_pd(tC);
                ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4);*/
                _mm256_storeu_pd(tC, ymm4);

                // multiply C by beta and accumulate.
                tC += ldc;
                /*ymm2 = _mm256_loadu_pd(tC);
                ymm5 = _mm256_fmadd_pd(ymm2, ymm1, ymm5);*/
                _mm256_storeu_pd(tC, ymm5);

                // multiply C by beta and accumulate.
                tC += ldc;
                /*ymm2 = _mm256_loadu_pd(tC);
                ymm6 = _mm256_fmadd_pd(ymm2, ymm1, ymm6);*/
                _mm256_storeu_pd(tC, ymm6);
            }
            n_remainder = N - col_idx;
            // if the N is not multiple of 3.
            // handling edge case.
            if (n_remainder == 2)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                ymm4 = _mm256_setzero_pd();
                ymm5 = _mm256_setzero_pd();

                for (k = 0; k < K; ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_sd(tB + tb_inc_col * 0);
                    ymm1 = _mm256_broadcast_sd(tB + tb_inc_col * 1);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_pd(tA);
                    ymm4 = _mm256_fmadd_pd(ymm0, ymm3, ymm4);
                    ymm5 = _mm256_fmadd_pd(ymm1, ymm3, ymm5);

                    tA += lda;
                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_sd(alpha_cast);
                //ymm1 = _mm256_broadcast_sd(beta_cast);

                //multiply A*B by alpha.
                ymm4 = _mm256_mul_pd(ymm4, ymm0);
                ymm5 = _mm256_mul_pd(ymm5, ymm0);

                // multiply C by beta and accumulate.
                /*ymm2 = _mm256_loadu_pd(tC);
                ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4);*/
                _mm256_storeu_pd(tC, ymm4);

                // multiply C by beta and accumulate.
                tC += ldc;
                /*ymm2 = _mm256_loadu_pd(tC);
                ymm5 = _mm256_fmadd_pd(ymm2, ymm1, ymm5);*/
                _mm256_storeu_pd(tC, ymm5);

                col_idx += 2;

            }
            // if the N is not multiple of 3.
            // handling edge case.
            if (n_remainder == 1)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                ymm4 = _mm256_setzero_pd();

                for (k = 0; k < K; ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_sd(tB + tb_inc_col * 0);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_pd(tA);
                    ymm4 = _mm256_fmadd_pd(ymm0, ymm3, ymm4);

                    tA += lda;
                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_sd(alpha_cast);
                //ymm1 = _mm256_broadcast_sd(beta_cast);

                ymm4 = _mm256_mul_pd(ymm4, ymm0);

                // multiply C by beta and accumulate.
                /*ymm2 = _mm256_loadu_pd(tC);
                ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4);*/
                _mm256_storeu_pd(tC, ymm4);

            }

            row_idx += 4;
        }
        // M is not a multiple of 32.
        // The handling of edge case where the remainder
        // dimension is less than 8. The padding takes place
        // to handle this case.
        if ((m_remainder) && (lda > 3))
        {
            double f_temp[8];

            for (col_idx = 0; (col_idx + 2) < N; col_idx += 3)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                // clear scratch registers.
                ymm5 = _mm256_setzero_pd();
                ymm7 = _mm256_setzero_pd();
                ymm9 = _mm256_setzero_pd();

                for (k = 0; k < (K - 1); ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_sd(tB + tb_inc_col * 0);
                    ymm1 = _mm256_broadcast_sd(tB + tb_inc_col * 1);
                    ymm2 = _mm256_broadcast_sd(tB + tb_inc_col * 2);
                    tB += tb_inc_row;

                    //broadcasted matrix B elements are multiplied
                    //with matrix A columns.
                    ymm3 = _mm256_loadu_pd(tA);
                    ymm5 = _mm256_fmadd_pd(ymm0, ymm3, ymm5);
                    ymm7 = _mm256_fmadd_pd(ymm1, ymm3, ymm7);
                    ymm9 = _mm256_fmadd_pd(ymm2, ymm3, ymm9);

                    tA += lda;
                }
                // alpha, beta multiplication.
                ymm0 = _mm256_broadcast_sd(tB + tb_inc_col * 0);
                ymm1 = _mm256_broadcast_sd(tB + tb_inc_col * 1);
                ymm2 = _mm256_broadcast_sd(tB + tb_inc_col * 2);
                tB += tb_inc_row;

                for (int i = 0; i < m_remainder; i++)
                {
                    f_temp[i] = tA[i];
                }
                ymm3 = _mm256_loadu_pd(f_temp);
                ymm5 = _mm256_fmadd_pd(ymm0, ymm3, ymm5);
                ymm7 = _mm256_fmadd_pd(ymm1, ymm3, ymm7);
                ymm9 = _mm256_fmadd_pd(ymm2, ymm3, ymm9);

                ymm0 = _mm256_broadcast_sd(alpha_cast);
                //ymm1 = _mm256_broadcast_sd(beta_cast);

                //multiply A*B by alpha.
                ymm5 = _mm256_mul_pd(ymm5, ymm0);
                ymm7 = _mm256_mul_pd(ymm7, ymm0);
                ymm9 = _mm256_mul_pd(ymm9, ymm0);


                /*for (int i = 0; i < m_remainder; i++)
                {
                    f_temp[i] = tC[i];
                }
                ymm2 = _mm256_loadu_pd(f_temp);
                ymm5 = _mm256_fmadd_pd(ymm2, ymm1, ymm5);*/
                _mm256_storeu_pd(f_temp, ymm5);
                for (int i = 0; i < m_remainder; i++)
                {
                    tC[i] = f_temp[i];
                }

                tC += ldc;
                /*for (int i = 0; i < m_remainder; i++)
                {
                    f_temp[i] = tC[i];
                }
                ymm2 = _mm256_loadu_pd(f_temp);
                ymm7 = _mm256_fmadd_pd(ymm2, ymm1, ymm7);*/
                _mm256_storeu_pd(f_temp, ymm7);
                for (int i = 0; i < m_remainder; i++)
                {
                    tC[i] = f_temp[i];
                }

                tC += ldc;
                /*for (int i = 0; i < m_remainder; i++)
                {
                    f_temp[i] = tC[i];
                }
                ymm2 = _mm256_loadu_pd(f_temp);
                ymm9 = _mm256_fmadd_pd(ymm2, ymm1, ymm9);*/
                _mm256_storeu_pd(f_temp, ymm9);
                for (int i = 0; i < m_remainder; i++)
                {
                    tC[i] = f_temp[i];
                }
            }
            n_remainder = N - col_idx;
            // if the N is not multiple of 3.
            // handling edge case.
            if (n_remainder == 2)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                ymm5 = _mm256_setzero_pd();
                ymm7 = _mm256_setzero_pd();

                for (k = 0; k < (K - 1); ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_sd(tB + tb_inc_col * 0);
                    ymm1 = _mm256_broadcast_sd(tB + tb_inc_col * 1);
                    tB += tb_inc_row;

                    ymm3 = _mm256_loadu_pd(tA);
                    ymm5 = _mm256_fmadd_pd(ymm0, ymm3, ymm5);
                    ymm7 = _mm256_fmadd_pd(ymm1, ymm3, ymm7);

                    tA += lda;
                }

                ymm0 = _mm256_broadcast_sd(tB + tb_inc_col * 0);
                ymm1 = _mm256_broadcast_sd(tB + tb_inc_col * 1);
                tB += tb_inc_row;

                for (int i = 0; i < m_remainder; i++)
                {
                    f_temp[i] = tA[i];
                }
                ymm3 = _mm256_loadu_pd(f_temp);
                ymm5 = _mm256_fmadd_pd(ymm0, ymm3, ymm5);
                ymm7 = _mm256_fmadd_pd(ymm1, ymm3, ymm7);

                ymm0 = _mm256_broadcast_sd(alpha_cast);
                //ymm1 = _mm256_broadcast_sd(beta_cast);

                ymm5 = _mm256_mul_pd(ymm5, ymm0);
                ymm7 = _mm256_mul_pd(ymm7, ymm0);

                /*for (int i = 0; i < m_remainder; i++)
                {
                    f_temp[i] = tC[i];
                }
                ymm2 = _mm256_loadu_pd(f_temp);
                ymm5 = _mm256_fmadd_pd(ymm2, ymm1, ymm5);*/
                _mm256_storeu_pd(f_temp, ymm5);
                for (int i = 0; i < m_remainder; i++)
                {
                    tC[i] = f_temp[i];
                }

                tC += ldc;
                /*for (int i = 0; i < m_remainder; i++)
                {
                    f_temp[i] = tC[i];
                }
                ymm2 = _mm256_loadu_pd(f_temp);
                ymm7 = _mm256_fmadd_pd(ymm2, ymm1, ymm7);*/
                _mm256_storeu_pd(f_temp, ymm7);
                for (int i = 0; i < m_remainder; i++)
                {
                    tC[i] = f_temp[i];
                }
            }
            // if the N is not multiple of 3.
            // handling edge case.
            if (n_remainder == 1)
            {
                //pointer math to point to proper memory
                tC = C + ldc * col_idx + row_idx;
                tB = B + tb_inc_col * col_idx;
                tA = A + row_idx;

                ymm5 = _mm256_setzero_pd();

                for (k = 0; k < (K - 1); ++k)
                {
                    // The inner loop broadcasts the B matrix data and
                    // multiplies it with the A matrix.
                    ymm0 = _mm256_broadcast_sd(tB + tb_inc_col * 0);
                    tB += tb_inc_row;

                    ymm3 = _mm256_loadu_pd(tA);
                    ymm5 = _mm256_fmadd_pd(ymm0, ymm3, ymm5);

                    tA += lda;
                }

                ymm0 = _mm256_broadcast_sd(tB + tb_inc_col * 0);
                tB += tb_inc_row;

                for (int i = 0; i < m_remainder; i++)
                {
                    f_temp[i] = tA[i];
                }
                ymm3 = _mm256_loadu_pd(f_temp);
                ymm5 = _mm256_fmadd_pd(ymm0, ymm3, ymm5);

                ymm0 = _mm256_broadcast_sd(alpha_cast);
                //ymm1 = _mm256_broadcast_sd(beta_cast);

                // multiply C by beta and accumulate.
                ymm5 = _mm256_mul_pd(ymm5, ymm0);

                /*for (int i = 0; i < m_remainder; i++)
                {
                    f_temp[i] = tC[i];
                }
                ymm2 = _mm256_loadu_pd(f_temp);
                ymm5 = _mm256_fmadd_pd(ymm2, ymm1, ymm5);*/
                _mm256_storeu_pd(f_temp, ymm5);
                for (int i = 0; i < m_remainder; i++)
                {
                    tC[i] = f_temp[i];
                }
            }
            m_remainder = 0;
        }

        if (m_remainder)
        {
            double result;
            for (; row_idx < M; row_idx += 1)
            {
                for (col_idx = 0; col_idx < N; col_idx += 1)
                {
                    //pointer math to point to proper memory
                    tC = C + ldc * col_idx + row_idx;
                    tB = B + tb_inc_col * col_idx;
                    tA = A + row_idx;

                    result = 0;
                    for (k = 0; k < K; ++k)
                    {
                        result += (*tA) * (*tB);
                        tA += lda;
                        tB += tb_inc_row;
                    }

                    result *= (*alpha_cast);
                    (*tC) = /*(*tC) * (*beta_cast) + */result;
                }
            }
        }

        //copy/compute sryk values back to C using SIMD
        if ( bli_seq0( *beta_cast ) )
        {//just copy for beta = 0
            dim_t _i, _j, k, _l;
            if(bli_obj_is_lower(c)) //c is lower
            {
                //first column
                _j = 0;
                k = M >> 2;
                _i = 0;
                for ( _l = 0; _l < k; _l++ )
                {
                    ymm0 = _mm256_loadu_pd((C + _i*rsc));
                    _mm256_storeu_pd((matCbuf + _i*rs_matC), ymm0);
                    _i += 4;
                }
                while (_i < M )
                {
                    bli_ddcopys( *(C + _i*rsc + _j*ldc),
                                 *(matCbuf + _i*rs_matC + _j*ldc_matC) );
                    _i++;
                }
                _j++;
                while ( _j < N ) //next column
                {
                    //k = (_j + (4 - (_j & 3)));
                    _l = _j & 3;
                    k = (_l != 0) ? (_j + (4 - _l)) : _j;
                    k = (k <= M) ? k : M;
                    for ( _i = _j; _i < k; ++_i )
                    {
                        bli_ddcopys( *(C + _i*rsc + _j*ldc),
                                     *(matCbuf + _i*rs_matC + _j*ldc_matC) );
                    }
                    k = (M - _i) >> 2;
                    _l = 0;
                    while ( _l < k )
                    {
                        ymm0 = _mm256_loadu_pd((C + _i*rsc + _j*ldc));
                        _mm256_storeu_pd((matCbuf + _i*rs_matC + _j*ldc_matC), ymm0);

                        _i += 4;
                        _l++;
                    }
                    while (_i < M )
                    {
                        bli_ddcopys( *(C + _i*rsc + _j*ldc),
                                 *(matCbuf + _i*rs_matC + _j*ldc_matC) );
                        _i++;
                    }
                    _j++;
                }
            }
            else //c is upper
            {
                for ( _j = 0; _j < N; ++_j )
                {
					k = (_j + 1) >> 2;
                    _i = 0;
                    _l = 0;
                    while ( _l < k )
                    {
                        ymm0 = _mm256_loadu_pd((C + _i*rsc + _j*ldc));
                        _mm256_storeu_pd((matCbuf + _i*rs_matC + _j*ldc_matC), ymm0);
                        _i += 4;
                        _l++;
                    }
                    while (_i <= _j )
                    {
                        bli_ddcopys( *(C + _i*rsc + _j*ldc),
                                     *(matCbuf + _i*rs_matC + _j*ldc_matC) );
                        ++_i;
                    }
                }
            }
        }
        else
        {//when beta is non-zero, fmadd and store the results
            dim_t _i, _j, k, _l;
			ymm1 = _mm256_broadcast_sd(beta_cast);
            if(bli_obj_is_lower(c)) //c is lower
            {
                //first column
                _j = 0;
                k = M >> 2;
                _i = 0;
                for ( _l = 0; _l < k; _l++ )
                {
                    ymm2 = _mm256_loadu_pd((matCbuf + _i*rs_matC));
                    ymm0 = _mm256_loadu_pd((C + _i*rsc));
                    ymm0 = _mm256_fmadd_pd(ymm2, ymm1, ymm0);
                    _mm256_storeu_pd((matCbuf + _i*rs_matC), ymm0);
                    _i += 4;
                }
                while (_i < M )
                {
                    bli_dddxpbys( *(C + _i*rsc + _j*ldc),
                                  *(beta_cast),
                                  *(matCbuf + _i*rs_matC + _j*ldc_matC) );
                    _i++;
                }
                _j++;
                while ( _j < N ) //next column
                {
                    //k = (_j + (4 - (_j & 3)));
                    _l = _j & 3;
                    k = (_l != 0) ? (_j + (4 - _l)) : _j;
                    k = (k <= M) ? k : M;
                    for ( _i = _j; _i < k; ++_i )
                    {
                        bli_dddxpbys( *(C + _i*rsc + _j*ldc),
                                      *(beta_cast),
                                      *(matCbuf + _i*rs_matC + _j*ldc_matC) );
                    }
                    k = (M - _i) >> 2;
                    _l = 0;
                    while ( _l < k )
                    {
                        ymm2 = _mm256_loadu_pd((matCbuf + _i*rs_matC + _j*ldc_matC));
                        ymm0 = _mm256_loadu_pd((C + _i*rsc + _j*ldc));
                        ymm0 = _mm256_fmadd_pd(ymm2, ymm1, ymm0);
                        _mm256_storeu_pd((matCbuf + _i*rs_matC + _j*ldc_matC), ymm0);

                        _i += 4;
                        _l++;
                    }
                    while (_i < M )
                    {
                        bli_dddxpbys( *(C + _i*rsc + _j*ldc),
                                      *(beta_cast),
                                      *(matCbuf + _i*rs_matC + _j*ldc_matC) );
                        _i++;
                    }
                    _j++;
                }
            }
            else //c is upper
            {
                for ( _j = 0; _j < N; ++_j )
                {
					k = (_j + 1) >> 2;
                    _i = 0;
                    _l = 0;
                    while ( _l < k )
                    {
                        ymm2 = _mm256_loadu_pd((matCbuf + _i*rs_matC + _j*ldc_matC));
                        ymm0 = _mm256_loadu_pd((C + _i*rsc + _j*ldc));
                        ymm0 = _mm256_fmadd_pd(ymm2, ymm1, ymm0);
                        _mm256_storeu_pd((matCbuf + _i*rs_matC + _j*ldc_matC), ymm0);
                        _i += 4;
                        _l++;
                    }
                    while (_i <= _j )
                    {
                        bli_dddxpbys( *(C + _i*rsc + _j*ldc),
                                      *(beta_cast),
                                      *(matCbuf + _i*rs_matC + _j*ldc_matC) );
                        ++_i;
                    }
                }
            }
        }

        return BLIS_SUCCESS;
    }
    else
        return BLIS_NONCONFORMAL_DIMENSIONS;


};

static err_t bli_sgemmt_small_atbn
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       cntl_t* cntl
     )
{
    int M = bli_obj_length(c); // number of rows of Matrix C
    int N = bli_obj_width(c);  // number of columns of Matrix C
    int K = bli_obj_length(b); // number of rows of Matrix B
    int lda = bli_obj_col_stride(a); // column stride of matrix OP(A), where OP(A) is Transpose(A) if transA enabled.
    int ldb = bli_obj_col_stride(b); // column stride of matrix OP(B), where OP(B) is Transpose(B) if transB enabled.
    int ldc_matC = bli_obj_col_stride( c ); // column stride of matrix C
    int ldc = M;//bli_obj_col_stride( c ); // column stride of static buffer for matrix C
    int row_idx = 0, col_idx = 0, k;
    int rs_matC = bli_obj_row_stride( c );
    int rsc = 1;
    float *A = a->buffer; // pointer to matrix A elements, stored in row major format
    float *B = b->buffer; // pointer to matrix B elements, stored in column major format
    float *C = C_pack; // pointer to matrix C elements, stored in column major format
    float *matCbuf = c->buffer;

    float *tA = A, *tB = B, *tC = C;

    __m256 ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11;
    __m256 ymm12, ymm13, ymm14, ymm15;
    __m256 ymm0, ymm1, ymm2, ymm3;

    float result, scratch[8];
    float *alpha_cast, *beta_cast; // alpha, beta multiples
    alpha_cast = (alpha->buffer);
    beta_cast = (beta->buffer);

    // The non-copy version of the A^T GEMMT gives better performance for the small M cases.
    // The threshold is controlled by BLIS_ATBN_M_THRES
    if (M <= BLIS_ATBN_M_THRES)
    {
        for (col_idx = 0; (col_idx + (NR - 1)) < N; col_idx += NR)
        {
            for (row_idx = 0; (row_idx + (AT_MR - 1)) < M; row_idx += AT_MR)
            {
                tA = A + row_idx * lda;
                tB = B + col_idx * ldb;
                tC = C + col_idx * ldc + row_idx;
                // clear scratch registers.
                ymm4 = _mm256_setzero_ps();
                ymm5 = _mm256_setzero_ps();
                ymm6 = _mm256_setzero_ps();
                ymm7 = _mm256_setzero_ps();
                ymm8 = _mm256_setzero_ps();
                ymm9 = _mm256_setzero_ps();
                ymm10 = _mm256_setzero_ps();
                ymm11 = _mm256_setzero_ps();
                ymm12 = _mm256_setzero_ps();
                ymm13 = _mm256_setzero_ps();
                ymm14 = _mm256_setzero_ps();
                ymm15 = _mm256_setzero_ps();

                //The inner loop computes the 4x3 values of the matrix.
                //The computation pattern is:
                // ymm4  ymm5  ymm6
                // ymm7  ymm8  ymm9
                // ymm10 ymm11 ymm12
                // ymm13 ymm14 ymm15

                //The Dot operation is performed in the inner loop, 8 float elements fit
                //in the YMM register hence loop count incremented by 8
                for (k = 0; (k + 7) < K; k += 8)
                {
                    ymm0 = _mm256_loadu_ps(tB + 0);
                    ymm1 = _mm256_loadu_ps(tB + ldb);
                    ymm2 = _mm256_loadu_ps(tB + 2 * ldb);

                    ymm3 = _mm256_loadu_ps(tA);
                    ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
                    ymm5 = _mm256_fmadd_ps(ymm1, ymm3, ymm5);
                    ymm6 = _mm256_fmadd_ps(ymm2, ymm3, ymm6);

                    ymm3 = _mm256_loadu_ps(tA + lda);
                    ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
                    ymm8 = _mm256_fmadd_ps(ymm1, ymm3, ymm8);
                    ymm9 = _mm256_fmadd_ps(ymm2, ymm3, ymm9);

                    ymm3 = _mm256_loadu_ps(tA + 2 * lda);
                    ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
                    ymm11 = _mm256_fmadd_ps(ymm1, ymm3, ymm11);
                    ymm12 = _mm256_fmadd_ps(ymm2, ymm3, ymm12);

                    ymm3 = _mm256_loadu_ps(tA + 3 * lda);
                    ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);
                    ymm14 = _mm256_fmadd_ps(ymm1, ymm3, ymm14);
                    ymm15 = _mm256_fmadd_ps(ymm2, ymm3, ymm15);

                    tA += 8;
                    tB += 8;

                }

                // if K is not a multiple of 8, padding is done before load using temproary array.
                if (k < K)
                {
                    int iter;
                    float data_feeder[8] = { 0.0 };

                    for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tB[iter];
                    ymm0 = _mm256_loadu_ps(data_feeder);
                    for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tB[iter + ldb];
                    ymm1 = _mm256_loadu_ps(data_feeder);
                    for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tB[iter + 2 * ldb];
                    ymm2 = _mm256_loadu_ps(data_feeder);

                    for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tA[iter];
                    ymm3 = _mm256_loadu_ps(data_feeder);
                    ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);
                    ymm5 = _mm256_fmadd_ps(ymm1, ymm3, ymm5);
                    ymm6 = _mm256_fmadd_ps(ymm2, ymm3, ymm6);

                    for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tA[lda + iter];
                    ymm3 = _mm256_loadu_ps(data_feeder);
                    ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);
                    ymm8 = _mm256_fmadd_ps(ymm1, ymm3, ymm8);
                    ymm9 = _mm256_fmadd_ps(ymm2, ymm3, ymm9);

                    for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tA[2 * lda + iter];
                    ymm3 = _mm256_loadu_ps(data_feeder);
                    ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
                    ymm11 = _mm256_fmadd_ps(ymm1, ymm3, ymm11);
                    ymm12 = _mm256_fmadd_ps(ymm2, ymm3, ymm12);

                    for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tA[3 * lda + iter];
                    ymm3 = _mm256_loadu_ps(data_feeder);
                    ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);
                    ymm14 = _mm256_fmadd_ps(ymm1, ymm3, ymm14);
                    ymm15 = _mm256_fmadd_ps(ymm2, ymm3, ymm15);

                }

                //horizontal addition and storage of the data.
                //Results for 4x3 blocks of C is stored here
                ymm4 = _mm256_hadd_ps(ymm4, ymm4);
                ymm4 = _mm256_hadd_ps(ymm4, ymm4);
                _mm256_storeu_ps(scratch, ymm4);
                result = scratch[0] + scratch[4];
                result *= (*alpha_cast);
                tC[0] = result/* + tC[0] * (*beta_cast)*/;

                ymm7 = _mm256_hadd_ps(ymm7, ymm7);
                ymm7 = _mm256_hadd_ps(ymm7, ymm7);
                _mm256_storeu_ps(scratch, ymm7);
                result = scratch[0] + scratch[4];
                result *= (*alpha_cast);
                tC[1] = result/* + tC[1] * (*beta_cast)*/;

                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                _mm256_storeu_ps(scratch, ymm10);
                result = scratch[0] + scratch[4];
                result *= (*alpha_cast);
                tC[2] = result/* + tC[2] * (*beta_cast)*/;

                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                _mm256_storeu_ps(scratch, ymm13);
                result = scratch[0] + scratch[4];
                result *= (*alpha_cast);
                tC[3] = result/* + tC[3] * (*beta_cast)*/;

                tC += ldc;
                ymm5 = _mm256_hadd_ps(ymm5, ymm5);
                ymm5 = _mm256_hadd_ps(ymm5, ymm5);
                _mm256_storeu_ps(scratch, ymm5);
                result = scratch[0] + scratch[4];
                result *= (*alpha_cast);
                tC[0] = result/* + tC[0] * (*beta_cast)*/;

                ymm8 = _mm256_hadd_ps(ymm8, ymm8);
                ymm8 = _mm256_hadd_ps(ymm8, ymm8);
                _mm256_storeu_ps(scratch, ymm8);
                result = scratch[0] + scratch[4];
                result *= (*alpha_cast);
                tC[1] = result/* + tC[1] * (*beta_cast)*/;

                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                _mm256_storeu_ps(scratch, ymm11);
                result = scratch[0] + scratch[4];
                result *= (*alpha_cast);
                tC[2] = result/* + tC[2] * (*beta_cast)*/;

                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                _mm256_storeu_ps(scratch, ymm14);
                result = scratch[0] + scratch[4];
                result *= (*alpha_cast);
                tC[3] = result/* + tC[3] * (*beta_cast)*/;

                tC += ldc;
                ymm6 = _mm256_hadd_ps(ymm6, ymm6);
                ymm6 = _mm256_hadd_ps(ymm6, ymm6);
                _mm256_storeu_ps(scratch, ymm6);
                result = scratch[0] + scratch[4];
                result *= (*alpha_cast);
                tC[0] = result/* + tC[0] * (*beta_cast)*/;

                ymm9 = _mm256_hadd_ps(ymm9, ymm9);
                ymm9 = _mm256_hadd_ps(ymm9, ymm9);
                _mm256_storeu_ps(scratch, ymm9);
                result = scratch[0] + scratch[4];
                result *= (*alpha_cast);
                tC[1] = result/* + tC[1] * (*beta_cast)*/;

                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                _mm256_storeu_ps(scratch, ymm12);
                result = scratch[0] + scratch[4];
                result *= (*alpha_cast);
                tC[2] = result/* + tC[2] * (*beta_cast)*/;

                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                _mm256_storeu_ps(scratch, ymm15);
                result = scratch[0] + scratch[4];
                result *= (*alpha_cast);
                tC[3] = result/* + tC[3] * (*beta_cast)*/;
            }
        }

        int processed_col = col_idx;
        int processed_row = row_idx;

        //The edge case handling where N is not a multiple of 3
        if (processed_col < N)
        {
            for (col_idx = processed_col; col_idx < N; col_idx += 1)
            {
                for (row_idx = 0; (row_idx + (AT_MR - 1)) < M; row_idx += AT_MR)
                {
                    tA = A + row_idx * lda;
                    tB = B + col_idx * ldb;
                    tC = C + col_idx * ldc + row_idx;
                    // clear scratch registers.
                    ymm4 = _mm256_setzero_ps();
                    ymm7 = _mm256_setzero_ps();
                    ymm10 = _mm256_setzero_ps();
                    ymm13 = _mm256_setzero_ps();

                    //The inner loop computes the 4x1 values of the matrix.
                    //The computation pattern is:
                    // ymm4
                    // ymm7
                    // ymm10
                    // ymm13

                    for (k = 0; (k + 7) < K; k += 8)
                    {
                        ymm0 = _mm256_loadu_ps(tB + 0);

                        ymm3 = _mm256_loadu_ps(tA);
                        ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);

                        ymm3 = _mm256_loadu_ps(tA + lda);
                        ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);

                        ymm3 = _mm256_loadu_ps(tA + 2 * lda);
                        ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);

                        ymm3 = _mm256_loadu_ps(tA + 3 * lda);
                        ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);

                        tA += 8;
                        tB += 8;
                    }

                    // if K is not a multiple of 8, padding is done before load using temproary array.
                    if (k < K)
                    {
                        int iter;
                        float data_feeder[8] = { 0.0 };

                        for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tB[iter];
                        ymm0 = _mm256_loadu_ps(data_feeder);

                        for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tA[iter];
                        ymm3 = _mm256_loadu_ps(data_feeder);
                        ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);

                        for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tA[lda + iter];
                        ymm3 = _mm256_loadu_ps(data_feeder);
                        ymm7 = _mm256_fmadd_ps(ymm0, ymm3, ymm7);

                        for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tA[2 * lda + iter];
                        ymm3 = _mm256_loadu_ps(data_feeder);
                        ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);

                        for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tA[3 * lda + iter];
                        ymm3 = _mm256_loadu_ps(data_feeder);
                        ymm13 = _mm256_fmadd_ps(ymm0, ymm3, ymm13);

                    }

                    //horizontal addition and storage of the data.
                    //Results for 4x1 blocks of C is stored here
                    ymm4 = _mm256_hadd_ps(ymm4, ymm4);
                    ymm4 = _mm256_hadd_ps(ymm4, ymm4);
                    _mm256_storeu_ps(scratch, ymm4);
                    result = scratch[0] + scratch[4];
                    result *= (*alpha_cast);
                    tC[0] = result/* + tC[0] * (*beta_cast)*/;

                    ymm7 = _mm256_hadd_ps(ymm7, ymm7);
                    ymm7 = _mm256_hadd_ps(ymm7, ymm7);
                    _mm256_storeu_ps(scratch, ymm7);
                    result = scratch[0] + scratch[4];
                    result *= (*alpha_cast);
                    tC[1] = result/* + tC[1] * (*beta_cast)*/;

                    ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                    ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                    _mm256_storeu_ps(scratch, ymm10);
                    result = scratch[0] + scratch[4];
                    result *= (*alpha_cast);
                    tC[2] = result/* + tC[2] * (*beta_cast)*/;

                    ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                    ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                    _mm256_storeu_ps(scratch, ymm13);
                    result = scratch[0] + scratch[4];
                    result *= (*alpha_cast);
                    tC[3] = result/* + tC[3] * (*beta_cast)*/;

                }
            }
            processed_row = row_idx;
        }

        //The edge case handling where M is not a multiple of 4
        if (processed_row < M)
        {
            for (row_idx = processed_row; row_idx < M; row_idx += 1)
            {
                for (col_idx = 0; col_idx < N; col_idx += 1)
                {
                    tA = A + row_idx * lda;
                    tB = B + col_idx * ldb;
                    tC = C + col_idx * ldc + row_idx;
                    // clear scratch registers.
                    ymm4 = _mm256_setzero_ps();

                    for (k = 0; (k + 7) < K; k += 8)
                    {
                        ymm0 = _mm256_loadu_ps(tB + 0);
                        ymm3 = _mm256_loadu_ps(tA);
                        ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);

                        tA += 8;
                        tB += 8;
                    }

                    // if K is not a multiple of 8, padding is done before load using temproary array.
                    if (k < K)
                    {
                        int iter;
                        float data_feeder[8] = { 0.0 };

                        for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tB[iter];
                        ymm0 = _mm256_loadu_ps(data_feeder);

                        for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tA[iter];
                        ymm3 = _mm256_loadu_ps(data_feeder);
                        ymm4 = _mm256_fmadd_ps(ymm0, ymm3, ymm4);

                    }

                    //horizontal addition and storage of the data.
                    ymm4 = _mm256_hadd_ps(ymm4, ymm4);
                    ymm4 = _mm256_hadd_ps(ymm4, ymm4);
                    _mm256_storeu_ps(scratch, ymm4);
                    result = scratch[0] + scratch[4];
                    result *= (*alpha_cast);
                    tC[0] = result/* + tC[0] * (*beta_cast)*/;

                }
            }
        }

        //copy/compute sryk values back to C
        if ( bli_seq0( *beta_cast ) ) //when beta is 0, just copy result to C
        {
            dim_t _i, _j;
            if(bli_obj_is_lower(c)) //c is lower
            {
                for ( _j = 0; _j < N; ++_j )
                    for ( _i = 0; _i < M; ++_i )
                        if ( (doff_t)_j - (doff_t)_i <= 0 )
                        {
                            bli_sscopys( *(C + _i*rsc + _j*ldc),
                                         *(matCbuf + _i*rs_matC + _j*ldc_matC) );
                        }
            }
            else //c is upper
            {
                for ( _j = 0; _j < N; ++_j )
                    for ( _i = 0; _i < M; ++_i )
                        if ( (doff_t)_j - (doff_t)_i >= 0 )
                        {
                            bli_sscopys( *(C + _i*rsc + _j*ldc),
                                         *(matCbuf + _i*rs_matC + _j*ldc_matC) );
                        }
            }
        }
        else //when beta is non-zero, multiply and store result to C
        {
            dim_t _i, _j;
            if(bli_obj_is_lower(c)) //c is lower
            {
                for ( _j = 0; _j < N; ++_j )
                    for ( _i = 0; _i < M; ++_i )
                        if ( (doff_t)_j - (doff_t)_i <= 0 )
                        {
                            bli_sssxpbys( *(C + _i*rsc + _j*ldc),
                                          *(beta_cast),
                                          *(matCbuf + _i*rs_matC + _j*ldc_matC) );
                        }
            }
            else //c is upper
            {
                for ( _j = 0; _j < N; ++_j )
                    for ( _i = 0; _i < M; ++_i )
                        if ( (doff_t)_j - (doff_t)_i >= 0 )
                        {
                            bli_sssxpbys( *(C + _i*rsc + _j*ldc),
                                          *(beta_cast),
                                          *(matCbuf + _i*rs_matC + _j*ldc_matC) );
                }
            }
        }

        return BLIS_SUCCESS;
    }
    else
        return BLIS_NONCONFORMAL_DIMENSIONS;
}

static err_t bli_dgemmt_small_atbn
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       cntl_t* cntl
     )
{
    int M = bli_obj_length( c ); // number of rows of Matrix C
    int N = bli_obj_width( c );  // number of columns of Matrix C
    int K = bli_obj_length( b ); // number of rows of Matrix B
    int lda = bli_obj_col_stride( a ); // column stride of matrix OP(A), where OP(A) is Transpose(A) if transA enabled.
    int ldb = bli_obj_col_stride( b ); // column stride of matrix OP(B), where OP(B) is Transpose(B) if transB enabled.
    int ldc_matC = bli_obj_col_stride( c ); // column stride of matrix C
    int ldc = M;//bli_obj_col_stride( c ); // column stride of static buffer for matrix C
    int row_idx = 0, col_idx = 0, k;
    int rs_matC = bli_obj_row_stride( c );
    int rsc = 1;
    double *A = a->buffer; // pointer to matrix A elements, stored in row major format
    double *B = b->buffer; // pointer to matrix B elements, stored in column major format
    double *C = D_C_pack; // pointer to matrix C elements, stored in column major format
    double *matCbuf = c->buffer;

    double *tA = A, *tB = B, *tC = C;

    __m256d ymm4, ymm5, ymm6, ymm7;
    __m256d ymm8, ymm9, ymm10, ymm11;
    __m256d ymm12, ymm13, ymm14, ymm15;
    __m256d ymm0, ymm1, ymm2, ymm3;

    double result, scratch[8];
    double *alpha_cast, *beta_cast; // alpha, beta multiples
    alpha_cast = (alpha->buffer);
    beta_cast = (beta->buffer);

    // The non-copy version of the A^T GEMMT gives better performance for the small M cases.
    // The threshold is controlled by BLIS_ATBN_M_THRES
    if (M <= BLIS_ATBN_M_THRES)
    {
        for (col_idx = 0; (col_idx + (NR - 1)) < N; col_idx += NR)
        {
            for (row_idx = 0; (row_idx + (AT_MR - 1)) < M; row_idx += AT_MR)
            {
                tA = A + row_idx * lda;
                tB = B + col_idx * ldb;
                tC = C + col_idx * ldc + row_idx;
                // clear scratch registers.
                ymm4 = _mm256_setzero_pd();
                ymm5 = _mm256_setzero_pd();
                ymm6 = _mm256_setzero_pd();
                ymm7 = _mm256_setzero_pd();
                ymm8 = _mm256_setzero_pd();
                ymm9 = _mm256_setzero_pd();
                ymm10 = _mm256_setzero_pd();
                ymm11 = _mm256_setzero_pd();
                ymm12 = _mm256_setzero_pd();
                ymm13 = _mm256_setzero_pd();
                ymm14 = _mm256_setzero_pd();
                ymm15 = _mm256_setzero_pd();

                //The inner loop computes the 4x3 values of the matrix.
                //The computation pattern is:
                // ymm4  ymm5  ymm6
                // ymm7  ymm8  ymm9
                // ymm10 ymm11 ymm12
                // ymm13 ymm14 ymm15

                //The Dot operation is performed in the inner loop, 4 double elements fit
                //in the YMM register hence loop count incremented by 4
                for (k = 0; (k + 3) < K; k += 4)
                {
                    ymm0 = _mm256_loadu_pd(tB + 0);
                    ymm1 = _mm256_loadu_pd(tB + ldb);
                    ymm2 = _mm256_loadu_pd(tB + 2 * ldb);

                    ymm3 = _mm256_loadu_pd(tA);
                    ymm4 = _mm256_fmadd_pd(ymm0, ymm3, ymm4);
                    ymm5 = _mm256_fmadd_pd(ymm1, ymm3, ymm5);
                    ymm6 = _mm256_fmadd_pd(ymm2, ymm3, ymm6);

                    ymm3 = _mm256_loadu_pd(tA + lda);
                    ymm7 = _mm256_fmadd_pd(ymm0, ymm3, ymm7);
                    ymm8 = _mm256_fmadd_pd(ymm1, ymm3, ymm8);
                    ymm9 = _mm256_fmadd_pd(ymm2, ymm3, ymm9);

                    ymm3 = _mm256_loadu_pd(tA + 2 * lda);
                    ymm10 = _mm256_fmadd_pd(ymm0, ymm3, ymm10);
                    ymm11 = _mm256_fmadd_pd(ymm1, ymm3, ymm11);
                    ymm12 = _mm256_fmadd_pd(ymm2, ymm3, ymm12);

                    ymm3 = _mm256_loadu_pd(tA + 3 * lda);
                    ymm13 = _mm256_fmadd_pd(ymm0, ymm3, ymm13);
                    ymm14 = _mm256_fmadd_pd(ymm1, ymm3, ymm14);
                    ymm15 = _mm256_fmadd_pd(ymm2, ymm3, ymm15);

                    tA += 4;
                    tB += 4;

                }

                // if K is not a multiple of 4, padding is done before load using temproary array.
                if (k < K)
                {
                    int iter;
                    double data_feeder[4] = { 0.0 };

                    for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tB[iter];
                    ymm0 = _mm256_loadu_pd(data_feeder);
                    for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tB[iter + ldb];
                    ymm1 = _mm256_loadu_pd(data_feeder);
                    for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tB[iter + 2 * ldb];
                    ymm2 = _mm256_loadu_pd(data_feeder);

                    for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tA[iter];
                    ymm3 = _mm256_loadu_pd(data_feeder);
                    ymm4 = _mm256_fmadd_pd(ymm0, ymm3, ymm4);
                    ymm5 = _mm256_fmadd_pd(ymm1, ymm3, ymm5);
                    ymm6 = _mm256_fmadd_pd(ymm2, ymm3, ymm6);

                    for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tA[lda + iter];
                    ymm3 = _mm256_loadu_pd(data_feeder);
                    ymm7 = _mm256_fmadd_pd(ymm0, ymm3, ymm7);
                    ymm8 = _mm256_fmadd_pd(ymm1, ymm3, ymm8);
                    ymm9 = _mm256_fmadd_pd(ymm2, ymm3, ymm9);

                    for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tA[2 * lda + iter];
                    ymm3 = _mm256_loadu_pd(data_feeder);
                    ymm10 = _mm256_fmadd_pd(ymm0, ymm3, ymm10);
                    ymm11 = _mm256_fmadd_pd(ymm1, ymm3, ymm11);
                    ymm12 = _mm256_fmadd_pd(ymm2, ymm3, ymm12);

                    for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tA[3 * lda + iter];
                    ymm3 = _mm256_loadu_pd(data_feeder);
                    ymm13 = _mm256_fmadd_pd(ymm0, ymm3, ymm13);
                    ymm14 = _mm256_fmadd_pd(ymm1, ymm3, ymm14);
                    ymm15 = _mm256_fmadd_pd(ymm2, ymm3, ymm15);

                }

                //horizontal addition and storage of the data.
                //Results for 4x3 blocks of C is stored here
                ymm4 = _mm256_hadd_pd(ymm4, ymm4);
                _mm256_storeu_pd(scratch, ymm4);
                result = scratch[0] + scratch[2];
                result *= (*alpha_cast);
                tC[0] = result/* + tC[0] * (*beta_cast)*/;

                ymm7 = _mm256_hadd_pd(ymm7, ymm7);
                _mm256_storeu_pd(scratch, ymm7);
                result = scratch[0] + scratch[2];
                result *= (*alpha_cast);
                tC[1] = result/* + tC[1] * (*beta_cast)*/;

                ymm10 = _mm256_hadd_pd(ymm10, ymm10);
                _mm256_storeu_pd(scratch, ymm10);
                result = scratch[0] + scratch[2];
                result *= (*alpha_cast);
                tC[2] = result/* + tC[2] * (*beta_cast)*/;

                ymm13 = _mm256_hadd_pd(ymm13, ymm13);
                _mm256_storeu_pd(scratch, ymm13);
                result = scratch[0] + scratch[2];
                result *= (*alpha_cast);
                tC[3] = result/* + tC[3] * (*beta_cast)*/;


                tC += ldc;
                ymm5 = _mm256_hadd_pd(ymm5, ymm5);
                _mm256_storeu_pd(scratch, ymm5);
                result = scratch[0] + scratch[2];
                result *= (*alpha_cast);
                tC[0] = result/* + tC[0] * (*beta_cast)*/;

                ymm8 = _mm256_hadd_pd(ymm8, ymm8);
                _mm256_storeu_pd(scratch, ymm8);
                result = scratch[0] + scratch[2];
                result *= (*alpha_cast);
                tC[1] = result/* + tC[1] * (*beta_cast)*/;

                ymm11 = _mm256_hadd_pd(ymm11, ymm11);
                _mm256_storeu_pd(scratch, ymm11);
                result = scratch[0] + scratch[2];
                result *= (*alpha_cast);
                tC[2] = result/* + tC[2] * (*beta_cast)*/;

                ymm14 = _mm256_hadd_pd(ymm14, ymm14);
                _mm256_storeu_pd(scratch, ymm14);
                result = scratch[0] + scratch[2];
                result *= (*alpha_cast);
                tC[3] = result/* + tC[3] * (*beta_cast)*/;


                tC += ldc;
                ymm6 = _mm256_hadd_pd(ymm6, ymm6);
                _mm256_storeu_pd(scratch, ymm6);
                result = scratch[0] + scratch[2];
                result *= (*alpha_cast);
                tC[0] = result/* + tC[0] * (*beta_cast)*/;

                ymm9 = _mm256_hadd_pd(ymm9, ymm9);
                _mm256_storeu_pd(scratch, ymm9);
                result = scratch[0] + scratch[2];
                result *= (*alpha_cast);
                tC[1] = result/* + tC[1] * (*beta_cast)*/;

                ymm12 = _mm256_hadd_pd(ymm12, ymm12);
                _mm256_storeu_pd(scratch, ymm12);
                result = scratch[0] + scratch[2];
                result *= (*alpha_cast);
                tC[2] = result/* + tC[2] * (*beta_cast)*/;

                ymm15 = _mm256_hadd_pd(ymm15, ymm15);
                _mm256_storeu_pd(scratch, ymm15);
                result = scratch[0] + scratch[2];
                result *= (*alpha_cast);
                tC[3] = result/* + tC[3] * (*beta_cast)*/;
            }
        }

        int processed_col = col_idx;
        int processed_row = row_idx;

        //The edge case handling where N is not a multiple of 3
        if (processed_col < N)
        {
            for (col_idx = processed_col; col_idx < N; col_idx += 1)
            {
                for (row_idx = 0; (row_idx + (AT_MR - 1)) < M; row_idx += AT_MR)
                {
                    tA = A + row_idx * lda;
                    tB = B + col_idx * ldb;
                    tC = C + col_idx * ldc + row_idx;
                    // clear scratch registers.
                    ymm4 = _mm256_setzero_pd();
                    ymm7 = _mm256_setzero_pd();
                    ymm10 = _mm256_setzero_pd();
                    ymm13 = _mm256_setzero_pd();

                    //The inner loop computes the 4x1 values of the matrix.
                    //The computation pattern is:
                    // ymm4
                    // ymm7
                    // ymm10
                    // ymm13

                    for (k = 0; (k + 3) < K; k += 4)
                    {
                        ymm0 = _mm256_loadu_pd(tB + 0);

                        ymm3 = _mm256_loadu_pd(tA);
                        ymm4 = _mm256_fmadd_pd(ymm0, ymm3, ymm4);

                        ymm3 = _mm256_loadu_pd(tA + lda);
                        ymm7 = _mm256_fmadd_pd(ymm0, ymm3, ymm7);

                        ymm3 = _mm256_loadu_pd(tA + 2 * lda);
                        ymm10 = _mm256_fmadd_pd(ymm0, ymm3, ymm10);

                        ymm3 = _mm256_loadu_pd(tA + 3 * lda);
                        ymm13 = _mm256_fmadd_pd(ymm0, ymm3, ymm13);

                        tA += 4;
                        tB += 4;
                    }
                    // if K is not a multiple of 4, padding is done before load using temproary array.
                    if (k < K)
                    {
                        int iter;
                        double data_feeder[4] = { 0.0 };

                        for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tB[iter];
                        ymm0 = _mm256_loadu_pd(data_feeder);

                        for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tA[iter];
                        ymm3 = _mm256_loadu_pd(data_feeder);
                        ymm4 = _mm256_fmadd_pd(ymm0, ymm3, ymm4);

                        for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tA[lda + iter];
                        ymm3 = _mm256_loadu_pd(data_feeder);
                        ymm7 = _mm256_fmadd_pd(ymm0, ymm3, ymm7);

                        for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tA[2 * lda + iter];
                        ymm3 = _mm256_loadu_pd(data_feeder);
                        ymm10 = _mm256_fmadd_pd(ymm0, ymm3, ymm10);

                        for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tA[3 * lda + iter];
                        ymm3 = _mm256_loadu_pd(data_feeder);
                        ymm13 = _mm256_fmadd_pd(ymm0, ymm3, ymm13);

                    }

                    //horizontal addition and storage of the data.
                    //Results for 4x1 blocks of C is stored here
                    ymm4 = _mm256_hadd_pd(ymm4, ymm4);
                    _mm256_storeu_pd(scratch, ymm4);
                    result = scratch[0] + scratch[2];
                    result *= (*alpha_cast);
                    tC[0] = result/* + tC[0] * (*beta_cast)*/;

                    ymm7 = _mm256_hadd_pd(ymm7, ymm7);
                    _mm256_storeu_pd(scratch, ymm7);
                    result = scratch[0] + scratch[2];
                    result *= (*alpha_cast);
                    tC[1] = result/* + tC[1] * (*beta_cast)*/;

                    ymm10 = _mm256_hadd_pd(ymm10, ymm10);
                    _mm256_storeu_pd(scratch, ymm10);
                    result = scratch[0] + scratch[2];
                    result *= (*alpha_cast);
                    tC[2] = result/* + tC[2] * (*beta_cast)*/;

                    ymm13 = _mm256_hadd_pd(ymm13, ymm13);
                    _mm256_storeu_pd(scratch, ymm13);
                    result = scratch[0] + scratch[2];
                    result *= (*alpha_cast);
                    tC[3] = result/* + tC[3] * (*beta_cast)*/;

                }
            }
            processed_row = row_idx;
        }

        // The edge case handling where M is not a multiple of 4
        if (processed_row < M)
        {
            for (row_idx = processed_row; row_idx < M; row_idx += 1)
            {
                for (col_idx = 0; col_idx < N; col_idx += 1)
                {
                    tA = A + row_idx * lda;
                    tB = B + col_idx * ldb;
                    tC = C + col_idx * ldc + row_idx;
                    // clear scratch registers.
                    ymm4 = _mm256_setzero_pd();

                    for (k = 0; (k + 3) < K; k += 4)
                    {
                        ymm0 = _mm256_loadu_pd(tB + 0);
                        ymm3 = _mm256_loadu_pd(tA);
                        ymm4 = _mm256_fmadd_pd(ymm0, ymm3, ymm4);

                        tA += 4;
                        tB += 4;
                    }

                    // if K is not a multiple of 4, padding is done before load using temproary array.
                    if (k < K)
                    {
                        int iter;
                        double data_feeder[4] = { 0.0 };

                        for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tB[iter];
                        ymm0 = _mm256_loadu_pd(data_feeder);

                        for (iter = 0; iter < (K - k); iter++) data_feeder[iter] = tA[iter];
                        ymm3 = _mm256_loadu_pd(data_feeder);
                        ymm4 = _mm256_fmadd_pd(ymm0, ymm3, ymm4);

                    }

                    //horizontal addition and storage of the data.
                    ymm4 = _mm256_hadd_pd(ymm4, ymm4);
                    _mm256_storeu_pd(scratch, ymm4);
                    result = scratch[0] + scratch[2];
                    result *= (*alpha_cast);
                    tC[0] = result/* + tC[0] * (*beta_cast)*/;

                }
            }
        }

		//copy/compute sryk values back to C
        if ( bli_seq0( *beta_cast ) ) //when beta is 0, just copy result to C
        {
            dim_t _i, _j;
            if(bli_obj_is_lower(c)) //c is lower
            {
                for ( _j = 0; _j < N; ++_j )
                    for ( _i = 0; _i < M; ++_i )
                        if ( (doff_t)_j - (doff_t)_i <= 0 )
                        {
                            bli_ddcopys( *(C + _i*rsc + _j*ldc),
                                         *(matCbuf + _i*rs_matC + _j*ldc_matC) );
                        }
            }
            else //c is upper
            {
                for ( _j = 0; _j < N; ++_j )
                    for ( _i = 0; _i < M; ++_i )
                        if ( (doff_t)_j - (doff_t)_i >= 0 )
                        {
                            bli_ddcopys( *(C + _i*rsc + _j*ldc),
                                         *(matCbuf + _i*rs_matC + _j*ldc_matC) );
                        }
            }
        }
        else //when beta is non-zero, multiply and store result to C
        {
            dim_t _i, _j;
            if(bli_obj_is_lower(c)) //c is lower
            {
                for ( _j = 0; _j < N; ++_j )
                    for ( _i = 0; _i < M; ++_i )
                        if ( (doff_t)_j - (doff_t)_i <= 0 )
                        {
                            bli_dddxpbys( *(C + _i*rsc + _j*ldc),
                                          *(beta_cast),
                                          *(matCbuf + _i*rs_matC + _j*ldc_matC) );
                        }
            }
            else //c is upper
            {
                for ( _j = 0; _j < N; ++_j )
                    for ( _i = 0; _i < M; ++_i )
                        if ( (doff_t)_j - (doff_t)_i >= 0 )
                        {
                            bli_dddxpbys( *(C + _i*rsc + _j*ldc),
                                          *(beta_cast),
                                          *(matCbuf + _i*rs_matC + _j*ldc_matC) );
                }
            }
        }

        return BLIS_SUCCESS;
    }
    else
        return BLIS_NONCONFORMAL_DIMENSIONS;
}

#endif

