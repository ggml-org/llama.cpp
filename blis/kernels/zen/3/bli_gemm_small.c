/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2017 - 2020, Advanced Micro Devices, Inc.

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

#include "immintrin.h"
#include "xmmintrin.h"
#include "blis.h"

#define AOCL_DTL_TRACE_ENTRY(x)      ;
#define AOCL_DTL_TRACE_EXIT(x)       ;
#define AOCL_DTL_TRACE_EXIT_ERR(x,y) ;

#ifdef BLIS_ENABLE_SMALL_MATRIX

#define MR 32
#define D_MR (MR >> 1)
#define NR 3
#define D_BLIS_SMALL_MATRIX_K_THRES_ROME    256

#define BLIS_ENABLE_PREFETCH
#define D_BLIS_SMALL_MATRIX_THRES (BLIS_SMALL_MATRIX_THRES / 2 )
#define D_BLIS_SMALL_M_RECT_MATRIX_THRES (BLIS_SMALL_M_RECT_MATRIX_THRES / 2)
#define D_BLIS_SMALL_K_RECT_MATRIX_THRES (BLIS_SMALL_K_RECT_MATRIX_THRES / 2)
#define BLIS_ATBN_M_THRES 40 // Threshold value of M for/below which small matrix code is called. 
#define AT_MR 4 // The kernel dimension of the A transpose GEMM kernel.(AT_MR * NR).
static err_t bli_sgemm_small
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
             cntl_t* cntl
     );

static err_t bli_dgemm_small
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
             cntl_t* cntl
     );

static err_t bli_sgemm_small_atbn
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
             cntl_t* cntl
     );

static err_t bli_dgemm_small_atbn
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
             cntl_t* cntl
     );
/*
* The bli_gemm_small function will use the
* custom MRxNR kernels, to perform the computation.
* The custom kernels are used if the [M * N] < 240 * 240
*/
err_t bli_gemm_small
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
             cntl_t* cntl
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	
#ifdef BLIS_ENABLE_MULTITHREADING
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
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

    num_t dt = bli_obj_dt(c);

    if (bli_obj_has_trans( a ))
    {
        if (bli_obj_has_notrans( b ))
        {
            if (dt == BLIS_FLOAT)
            {
                return bli_sgemm_small_atbn(alpha, a, b, beta, c, cntx, cntl);
            }
            else if (dt == BLIS_DOUBLE)
            {
                return bli_dgemm_small_atbn(alpha, a, b, beta, c, cntx, cntl);
            }
        }

        return BLIS_NOT_YET_IMPLEMENTED;
    }

    if (dt == BLIS_DOUBLE)
    {
        return bli_dgemm_small(alpha, a, b, beta, c, cntx, cntl);
    }

    if (dt == BLIS_FLOAT)
    {
        return bli_sgemm_small(alpha, a, b, beta, c, cntx, cntl);
    }

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
    return BLIS_NOT_YET_IMPLEMENTED;
};


static err_t bli_sgemm_small
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
             cntl_t* cntl
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
    gint_t M = bli_obj_length( c ); // number of rows of Matrix C
    gint_t N = bli_obj_width( c );  // number of columns of Matrix C
    gint_t K = bli_obj_width( a );  // number of columns of OP(A), will be updated if OP(A) is Transpose(A) .
    gint_t L = M * N;

	// when N is equal to 1 call GEMV instead of GEMM
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
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
        return BLIS_SUCCESS;
    }


    if ((((L) < (BLIS_SMALL_MATRIX_THRES * BLIS_SMALL_MATRIX_THRES))
        || ((M  < BLIS_SMALL_M_RECT_MATRIX_THRES) && (K < BLIS_SMALL_K_RECT_MATRIX_THRES))) && ((L!=0) && (K!=0)))
    {
        guint_t lda = bli_obj_col_stride( a ); // column stride of matrix OP(A), where OP(A) is Transpose(A) if transA enabled.
        guint_t ldb = bli_obj_col_stride( b ); // column stride of matrix OP(B), where OP(B) is Transpose(B) if transB enabled.
        guint_t ldc = bli_obj_col_stride( c ); // column stride of matrix C
        guint_t row_idx, col_idx, k;

        float *A = bli_obj_buffer_at_off(a); // pointer to elements of Matrix A
        float *B = bli_obj_buffer_at_off(b); // pointer to elements of Matrix B
        float *C = bli_obj_buffer_at_off(c); // pointer to elements of Matrix C

        float *tA = A, *tB = B, *tC = C;//, *tA_pack;
        float *tA_packed; // temporary pointer to hold packed A memory pointer

        guint_t row_idx_packed; //packed A memory row index
        guint_t lda_packed; //lda of packed A
        guint_t col_idx_start; //starting index after A matrix is packed.
        dim_t tb_inc_row = 1; // row stride of matrix B
        dim_t tb_inc_col = ldb; // column stride of matrix B

	    __m256 ymm4, ymm5, ymm6, ymm7;
        __m256 ymm8, ymm9, ymm10, ymm11;
        __m256 ymm12, ymm13, ymm14, ymm15;
        __m256 ymm0, ymm1, ymm2, ymm3;

        gint_t n_remainder; // If the N is non multiple of 3.(N%3)
        gint_t m_remainder; // If the M is non multiple of 32.(M%32)
        gint_t required_packing_A = 1;
        mem_t local_mem_buf_A_s;
        float *A_pack = NULL;
        rntm_t rntm;

        const num_t    dt_exec   = bli_obj_dt( c );
        float* restrict alpha_cast = bli_obj_buffer_for_1x1( dt_exec, alpha );
        float* restrict beta_cast  = bli_obj_buffer_for_1x1( dt_exec, beta );	

        /*Beta Zero Check*/
        bool is_beta_non_zero=0;
        if ( !bli_obj_equals( beta, &BLIS_ZERO ) ){
            is_beta_non_zero = 1;
        }

	//update the pointer math if matrix B needs to be transposed.
        if (bli_obj_has_trans( b )) {
            tb_inc_col = 1; //switch row and column strides
            tb_inc_row = ldb;
        }

        /*
         * This function was using global array to pack part of A input when needed.
         * However, using this global array make the function non-reentrant.
         * Instead of using a global array we should allocate buffer for each invocation.
         * Since the buffer size is too big or stack and doing malloc every time will be too expensive,
         * better approach is to get the buffer from the pre-allocated pool and return
         * it the pool once we are doing.
         *
         * In order to get the buffer from pool, we need access to memory broker,
         * currently this function is not invoked in such a way that it can receive
         * the memory broker (via rntm). Following hack will get the global memory
         * broker that can be use it to access the pool.
         *
         * Note there will be memory allocation at least on first innovation
         * as there will not be any pool created for this size.
         * Subsequent invocations will just reuse the buffer from the pool.
         */

        bli_rntm_init_from_global( &rntm );
        bli_rntm_set_num_threads_only( 1, &rntm );
        bli_pba_rntm_set_pba( &rntm );

        // Get the current size of the buffer pool for A block packing.
        // We will use the same size to avoid pool re-initialization 
        siz_t buffer_size = bli_pool_block_size(bli_pba_pool(bli_packbuf_index(BLIS_BITVAL_BUFFER_FOR_A_BLOCK),
                                                bli_rntm_pba(&rntm)));

        // Based on the available memory in the buffer we will decide if 
        // we want to do packing or not.
        //
        // This kernel assumes that "A" will be un-packged if N <= 3.
        // Usually this range (N <= 3) is handled by SUP, however,
        // if SUP is disabled or for any other condition if we do
        // enter this kernel with N <= 3, we want to make sure that
        // "A" remains unpacked.
        //
        // If this check is removed it will result in the crash as
        // reported in CPUPL-587.
        //
        
        if ((N <= 3) || (((MR * K) << 2) > buffer_size))
        {
            required_packing_A = 0;
        }
        else 
        {
#ifdef BLIS_ENABLE_MEM_TRACING
            printf( "bli_sgemm_small: Requesting mem pool block of size %lu\n", buffer_size);
#endif
            // Get the buffer from the pool, if there is no pool with
            // required size, it will be created. 
            bli_pba_acquire_m(&rntm,
                                 buffer_size,
                                 BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
                                 &local_mem_buf_A_s);

            A_pack = bli_mem_buffer(&local_mem_buf_A_s);
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

#ifdef BLIS_ENABLE_PREFETCH
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

                if(is_beta_non_zero)
                {
                    ymm1 = _mm256_broadcast_ss(beta_cast);
                    // multiply C by beta and accumulate col 1.
                    ymm2 = _mm256_loadu_ps(tC);
                    ymm4 = _mm256_fmadd_ps(ymm2, ymm1, ymm4);
                    ymm2 = _mm256_loadu_ps(tC + 8);
                    ymm5 = _mm256_fmadd_ps(ymm2, ymm1, ymm5);
                    ymm2 = _mm256_loadu_ps(tC + 16);
                    ymm6 = _mm256_fmadd_ps(ymm2, ymm1, ymm6);
                    ymm2 = _mm256_loadu_ps(tC + 24);
                    ymm7 = _mm256_fmadd_ps(ymm2, ymm1, ymm7);

                    float* ttC = tC +ldc;
                    ymm2 = _mm256_loadu_ps(ttC);
                    ymm8 = _mm256_fmadd_ps(ymm2, ymm1, ymm8);
                    ymm2 = _mm256_loadu_ps(ttC + 8);
                    ymm9 = _mm256_fmadd_ps(ymm2, ymm1, ymm9);
                    ymm2 = _mm256_loadu_ps(ttC + 16);
                    ymm10 = _mm256_fmadd_ps(ymm2, ymm1, ymm10);
                    ymm2 = _mm256_loadu_ps(ttC + 24);
                    ymm11 = _mm256_fmadd_ps(ymm2, ymm1, ymm11);

                    ttC += ldc;
                    ymm2 = _mm256_loadu_ps(ttC);
                    ymm12 = _mm256_fmadd_ps(ymm2, ymm1, ymm12);
                    ymm2 = _mm256_loadu_ps(ttC + 8);
                    ymm13 = _mm256_fmadd_ps(ymm2, ymm1, ymm13);
                    ymm2 = _mm256_loadu_ps(ttC + 16);
                    ymm14 = _mm256_fmadd_ps(ymm2, ymm1, ymm14);
                    ymm2 = _mm256_loadu_ps(ttC + 24);
                    ymm15 = _mm256_fmadd_ps(ymm2, ymm1, ymm15);
                }
                _mm256_storeu_ps(tC, ymm4);
                _mm256_storeu_ps(tC + 8, ymm5);
                _mm256_storeu_ps(tC + 16, ymm6);
                _mm256_storeu_ps(tC + 24, ymm7);

                // multiply C by beta and accumulate, col 2.
                tC += ldc;
                _mm256_storeu_ps(tC, ymm8);
                _mm256_storeu_ps(tC + 8, ymm9);
                _mm256_storeu_ps(tC + 16, ymm10);
                _mm256_storeu_ps(tC + 24, ymm11);

                // multiply C by beta and accumulate, col 3.
                tC += ldc;
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

#ifdef BLIS_ENABLE_PREFETCH
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

                if(is_beta_non_zero)
                {
                    ymm1 = _mm256_broadcast_ss(beta_cast);
                    // multiply C by beta and accumulate col 1.
                    ymm2 = _mm256_loadu_ps(tC);
                    ymm4 = _mm256_fmadd_ps(ymm2, ymm1, ymm4);
                    ymm2 = _mm256_loadu_ps(tC + 8);
                    ymm5 = _mm256_fmadd_ps(ymm2, ymm1, ymm5);
                    ymm2 = _mm256_loadu_ps(tC + 16);
                    ymm6 = _mm256_fmadd_ps(ymm2, ymm1, ymm6);
                    ymm2 = _mm256_loadu_ps(tC + 24);
                    ymm7 = _mm256_fmadd_ps(ymm2, ymm1, ymm7);
                    float* ttC = tC +ldc;
                    ymm2 = _mm256_loadu_ps(ttC);
                    ymm8 = _mm256_fmadd_ps(ymm2, ymm1, ymm8);
                    ymm2 = _mm256_loadu_ps(ttC + 8);
                    ymm9 = _mm256_fmadd_ps(ymm2, ymm1, ymm9);
                    ymm2 = _mm256_loadu_ps(ttC + 16);
                    ymm10 = _mm256_fmadd_ps(ymm2, ymm1, ymm10);
                    ymm2 = _mm256_loadu_ps(ttC + 24);
                    ymm11 = _mm256_fmadd_ps(ymm2, ymm1, ymm11);
                    ttC = ttC +ldc;
                    ymm2 = _mm256_loadu_ps(ttC);
                    ymm12 = _mm256_fmadd_ps(ymm2, ymm1, ymm12);
                    ymm2 = _mm256_loadu_ps(ttC + 8);
                    ymm13 = _mm256_fmadd_ps(ymm2, ymm1, ymm13);
                    ymm2 = _mm256_loadu_ps(ttC + 16);
                    ymm14 = _mm256_fmadd_ps(ymm2, ymm1, ymm14);
                    ymm2 = _mm256_loadu_ps(ttC + 24);
                    ymm15 = _mm256_fmadd_ps(ymm2, ymm1, ymm15);
                }
                _mm256_storeu_ps(tC, ymm4);
                _mm256_storeu_ps(tC + 8, ymm5);
                _mm256_storeu_ps(tC + 16, ymm6);
                _mm256_storeu_ps(tC + 24, ymm7);

                // multiply C by beta and accumulate, col 2.
                tC += ldc;
                _mm256_storeu_ps(tC, ymm8);
                _mm256_storeu_ps(tC + 8, ymm9);
                _mm256_storeu_ps(tC + 16, ymm10);
                _mm256_storeu_ps(tC + 24, ymm11);

                // multiply C by beta and accumulate, col 3.
                tC += ldc;
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
                if(is_beta_non_zero)
                {
                    ymm1 = _mm256_broadcast_ss(beta_cast);
                    ymm2 = _mm256_loadu_ps(tC);
                    ymm8 = _mm256_fmadd_ps(ymm2, ymm1, ymm8);
                    ymm2 = _mm256_loadu_ps(tC + 8);
                    ymm9 = _mm256_fmadd_ps(ymm2, ymm1, ymm9);
                    ymm2 = _mm256_loadu_ps(tC + 16);
                    ymm10 = _mm256_fmadd_ps(ymm2, ymm1, ymm10);
                    ymm2 = _mm256_loadu_ps(tC + 24);
                    ymm11 = _mm256_fmadd_ps(ymm2, ymm1, ymm11);

                    float* ttC = tC +ldc;
                    // multiply C by beta and accumulate, col 2.
                    ymm2 = _mm256_loadu_ps(ttC);
                    ymm12 = _mm256_fmadd_ps(ymm2, ymm1, ymm12);
                    ymm2 = _mm256_loadu_ps(ttC + 8);
                    ymm13 = _mm256_fmadd_ps(ymm2, ymm1, ymm13);
                    ymm2 = _mm256_loadu_ps(ttC + 16);
                    ymm14 = _mm256_fmadd_ps(ymm2, ymm1, ymm14);
                    ymm2 = _mm256_loadu_ps(ttC + 24);
                    ymm15 = _mm256_fmadd_ps(ymm2, ymm1, ymm15);
                }
                _mm256_storeu_ps(tC, ymm8);
                _mm256_storeu_ps(tC + 8, ymm9);
                _mm256_storeu_ps(tC + 16, ymm10);
                _mm256_storeu_ps(tC + 24, ymm11);
                tC += ldc;
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

                //multiply A*B by alpha.
                ymm12 = _mm256_mul_ps(ymm12, ymm0);
                ymm13 = _mm256_mul_ps(ymm13, ymm0);
                ymm14 = _mm256_mul_ps(ymm14, ymm0);
                ymm15 = _mm256_mul_ps(ymm15, ymm0);

                if(is_beta_non_zero)
                {
                    ymm1 = _mm256_broadcast_ss(beta_cast);
                    // multiply C by beta and accumulate.
                    ymm2 = _mm256_loadu_ps(tC + 0);
                    ymm12 = _mm256_fmadd_ps(ymm2, ymm1, ymm12);
                    ymm2 = _mm256_loadu_ps(tC + 8);
                    ymm13 = _mm256_fmadd_ps(ymm2, ymm1, ymm13);
                    ymm2 = _mm256_loadu_ps(tC + 16);
                    ymm14 = _mm256_fmadd_ps(ymm2, ymm1, ymm14);
                    ymm2 = _mm256_loadu_ps(tC + 24);
                    ymm15 = _mm256_fmadd_ps(ymm2, ymm1, ymm15);
                }

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

                if(is_beta_non_zero)
                {
                    ymm1 = _mm256_broadcast_ss(beta_cast);
                    // multiply C by beta and accumulate.
                    ymm2 = _mm256_loadu_ps(tC);
                    ymm4 = _mm256_fmadd_ps(ymm2, ymm1, ymm4);
                    ymm2 = _mm256_loadu_ps(tC + 8);
                    ymm5 = _mm256_fmadd_ps(ymm2, ymm1, ymm5);
                    ymm2 = _mm256_loadu_ps(tC + 16);
                    ymm6 = _mm256_fmadd_ps(ymm2, ymm1, ymm6);
                    float* ttC = tC +ldc;
                    ymm2 = _mm256_loadu_ps(ttC);
                    ymm8 = _mm256_fmadd_ps(ymm2, ymm1, ymm8);
                    ymm2 = _mm256_loadu_ps(ttC + 8);
                    ymm9 = _mm256_fmadd_ps(ymm2, ymm1, ymm9);
                    ymm2 = _mm256_loadu_ps(ttC + 16);
                    ymm10 = _mm256_fmadd_ps(ymm2, ymm1, ymm10);
                    ttC += ldc;
                    ymm2 = _mm256_loadu_ps(ttC);
                    ymm12 = _mm256_fmadd_ps(ymm2, ymm1, ymm12);
                    ymm2 = _mm256_loadu_ps(ttC + 8);
                    ymm13 = _mm256_fmadd_ps(ymm2, ymm1, ymm13);
                    ymm2 = _mm256_loadu_ps(ttC + 16);
                    ymm14 = _mm256_fmadd_ps(ymm2, ymm1, ymm14);
                }
                _mm256_storeu_ps(tC, ymm4);
                _mm256_storeu_ps(tC + 8, ymm5);
                _mm256_storeu_ps(tC + 16, ymm6);

                // multiply C by beta and accumulate.
                tC += ldc;
                _mm256_storeu_ps(tC, ymm8);
                _mm256_storeu_ps(tC + 8, ymm9);
                _mm256_storeu_ps(tC + 16, ymm10);

                // multiply C by beta and accumulate.
                tC += ldc;
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

                //multiply A*B by alpha.
                ymm8 = _mm256_mul_ps(ymm8, ymm0);
                ymm9 = _mm256_mul_ps(ymm9, ymm0);
                ymm10 = _mm256_mul_ps(ymm10, ymm0);
                ymm12 = _mm256_mul_ps(ymm12, ymm0);
                ymm13 = _mm256_mul_ps(ymm13, ymm0);
                ymm14 = _mm256_mul_ps(ymm14, ymm0);

                if(is_beta_non_zero)
                {
                    ymm1 = _mm256_broadcast_ss(beta_cast);
                    // multiply C by beta and accumulate.
                    ymm2 = _mm256_loadu_ps(tC);
                    ymm8 = _mm256_fmadd_ps(ymm2, ymm1, ymm8);
                    ymm2 = _mm256_loadu_ps(tC + 8);
                    ymm9 = _mm256_fmadd_ps(ymm2, ymm1, ymm9);
                    ymm2 = _mm256_loadu_ps(tC + 16);
                    ymm10 = _mm256_fmadd_ps(ymm2, ymm1, ymm10);

                    float* ttC = tC +ldc;
                    // multiply C by beta and accumulate.
                    ymm2 = _mm256_loadu_ps(ttC);
                    ymm12 = _mm256_fmadd_ps(ymm2, ymm1, ymm12);
                    ymm2 = _mm256_loadu_ps(ttC + 8);
                    ymm13 = _mm256_fmadd_ps(ymm2, ymm1, ymm13);
                    ymm2 = _mm256_loadu_ps(ttC + 16);
                    ymm14 = _mm256_fmadd_ps(ymm2, ymm1, ymm14);
                }

                _mm256_storeu_ps(tC, ymm8);
                _mm256_storeu_ps(tC + 8, ymm9);
                _mm256_storeu_ps(tC + 16, ymm10);

                tC += ldc;

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

                //multiply A*B by alpha.
                ymm12 = _mm256_mul_ps(ymm12, ymm0);
                ymm13 = _mm256_mul_ps(ymm13, ymm0);
                ymm14 = _mm256_mul_ps(ymm14, ymm0);

                if(is_beta_non_zero)
                {
                    ymm1 = _mm256_broadcast_ss(beta_cast);
                    // multiply C by beta and accumulate.
                    ymm2 = _mm256_loadu_ps(tC + 0);
                    ymm12 = _mm256_fmadd_ps(ymm2, ymm1, ymm12);
                    ymm2 = _mm256_loadu_ps(tC + 8);
                    ymm13 = _mm256_fmadd_ps(ymm2, ymm1, ymm13);
                    ymm2 = _mm256_loadu_ps(tC + 16);
                    ymm14 = _mm256_fmadd_ps(ymm2, ymm1, ymm14);
                }
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

                //multiply A*B by alpha.
                ymm4 = _mm256_mul_ps(ymm4, ymm0);
                ymm5 = _mm256_mul_ps(ymm5, ymm0);
                ymm6 = _mm256_mul_ps(ymm6, ymm0);
                ymm7 = _mm256_mul_ps(ymm7, ymm0);
                ymm8 = _mm256_mul_ps(ymm8, ymm0);
                ymm9 = _mm256_mul_ps(ymm9, ymm0);

                if(is_beta_non_zero)
                {
                    ymm1 = _mm256_broadcast_ss(beta_cast);
                    // multiply C by beta and accumulate.
                    ymm2 = _mm256_loadu_ps(tC);
                    ymm4 = _mm256_fmadd_ps(ymm2, ymm1, ymm4);
                    ymm2 = _mm256_loadu_ps(tC + 8);
                    ymm5 = _mm256_fmadd_ps(ymm2, ymm1, ymm5);
                    float* ttC = tC + ldc;
                    ymm2 = _mm256_loadu_ps(ttC);
                    ymm6 = _mm256_fmadd_ps(ymm2, ymm1, ymm6);
                    ymm2 = _mm256_loadu_ps(ttC + 8);
                    ymm7 = _mm256_fmadd_ps(ymm2, ymm1, ymm7);
                    ttC += ldc;
                    ymm2 = _mm256_loadu_ps(ttC);
                    ymm8 = _mm256_fmadd_ps(ymm2, ymm1, ymm8);
                    ymm2 = _mm256_loadu_ps(ttC + 8);
                    ymm9 = _mm256_fmadd_ps(ymm2, ymm1, ymm9);
                }
                _mm256_storeu_ps(tC, ymm4);
                _mm256_storeu_ps(tC + 8, ymm5);

                // multiply C by beta and accumulate.
                tC += ldc;
                _mm256_storeu_ps(tC, ymm6);
                _mm256_storeu_ps(tC + 8, ymm7);

                // multiply C by beta and accumulate.
                tC += ldc;
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

                //multiply A*B by alpha.
                ymm4 = _mm256_mul_ps(ymm4, ymm0);
                ymm5 = _mm256_mul_ps(ymm5, ymm0);
                ymm6 = _mm256_mul_ps(ymm6, ymm0);
                ymm7 = _mm256_mul_ps(ymm7, ymm0);

                if(is_beta_non_zero)
                {
                    ymm1 = _mm256_broadcast_ss(beta_cast);
                    // multiply C by beta and accumulate.
                    ymm2 = _mm256_loadu_ps(tC);
                    ymm4 = _mm256_fmadd_ps(ymm2, ymm1, ymm4);
                    ymm2 = _mm256_loadu_ps(tC + 8);
                    ymm5 = _mm256_fmadd_ps(ymm2, ymm1, ymm5);
                    float* ttC = tC + ldc;
                    ymm2 = _mm256_loadu_ps(ttC);
                    ymm6 = _mm256_fmadd_ps(ymm2, ymm1, ymm6);
                    ymm2 = _mm256_loadu_ps(ttC + 8);
                    ymm7 = _mm256_fmadd_ps(ymm2, ymm1, ymm7);
                }
                _mm256_storeu_ps(tC, ymm4);
                _mm256_storeu_ps(tC + 8, ymm5);

                // multiply C by beta and accumulate.
                tC += ldc;
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

                ymm4 = _mm256_mul_ps(ymm4, ymm0);
                ymm5 = _mm256_mul_ps(ymm5, ymm0);

                // multiply C by beta and accumulate.
                if(is_beta_non_zero)
                {
                    ymm1 = _mm256_broadcast_ss(beta_cast);
                    ymm2 = _mm256_loadu_ps(tC);
                    ymm4 = _mm256_fmadd_ps(ymm2, ymm1, ymm4);
                    ymm2 = _mm256_loadu_ps(tC + 8);
                    ymm5 = _mm256_fmadd_ps(ymm2, ymm1, ymm5);
                }
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

                //multiply A*B by alpha.
                ymm4 = _mm256_mul_ps(ymm4, ymm0);
                ymm5 = _mm256_mul_ps(ymm5, ymm0);
                ymm6 = _mm256_mul_ps(ymm6, ymm0);

                if(is_beta_non_zero)
                {
                    ymm1 = _mm256_broadcast_ss(beta_cast);
                    ymm2 = _mm256_loadu_ps(tC);
                    ymm4 = _mm256_fmadd_ps(ymm2, ymm1, ymm4);
                    ymm2 = _mm256_loadu_ps(tC + ldc);
                    ymm5 = _mm256_fmadd_ps(ymm2, ymm1, ymm5);
                    ymm2 = _mm256_loadu_ps(tC + 2*ldc);
                    ymm6 = _mm256_fmadd_ps(ymm2, ymm1, ymm6);
                }
                _mm256_storeu_ps(tC, ymm4);

                // multiply C by beta and accumulate.
                tC += ldc;
                _mm256_storeu_ps(tC, ymm5);

                // multiply C by beta and accumulate.
                tC += ldc;
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

                //multiply A*B by alpha.
                ymm4 = _mm256_mul_ps(ymm4, ymm0);
                ymm5 = _mm256_mul_ps(ymm5, ymm0);

                if(is_beta_non_zero)
                {
                    ymm1 = _mm256_broadcast_ss(beta_cast);
                    // multiply C by beta and accumulate.
                    ymm2 = _mm256_loadu_ps(tC);
                    ymm4 = _mm256_fmadd_ps(ymm2, ymm1, ymm4);
                    ymm2 = _mm256_loadu_ps(tC + ldc);
                    ymm5 = _mm256_fmadd_ps(ymm2, ymm1, ymm5);
                }
                _mm256_storeu_ps(tC, ymm4);
                // multiply C by beta and accumulate.
                tC += ldc;
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
                ymm4 = _mm256_mul_ps(ymm4, ymm0);

                if(is_beta_non_zero)
                {
                    ymm1 = _mm256_broadcast_ss(beta_cast);
                    // multiply C by beta and accumulate.
                    ymm2 = _mm256_loadu_ps(tC);
                    ymm4 = _mm256_fmadd_ps(ymm2, ymm1, ymm4);
                }
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
            float f_temp[8] = {0.0};

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
                ymm1 = _mm256_broadcast_ss(beta_cast);

                //multiply A*B by alpha.
                ymm5 = _mm256_mul_ps(ymm5, ymm0);
                ymm7 = _mm256_mul_ps(ymm7, ymm0);
                ymm9 = _mm256_mul_ps(ymm9, ymm0);


                for (int i = 0; i < m_remainder; i++)
                {
                    f_temp[i] = tC[i];
                }
                ymm2 = _mm256_loadu_ps(f_temp);
                if(is_beta_non_zero){
                    ymm5 = _mm256_fmadd_ps(ymm2, ymm1, ymm5);
                }
                _mm256_storeu_ps(f_temp, ymm5);
                for (int i = 0; i < m_remainder; i++)
                {
                    tC[i] = f_temp[i];
                }

                tC += ldc;
                for (int i = 0; i < m_remainder; i++)
                {
                    f_temp[i] = tC[i];
                }
                ymm2 = _mm256_loadu_ps(f_temp);
                if(is_beta_non_zero){
                    ymm7 = _mm256_fmadd_ps(ymm2, ymm1, ymm7);
                }
                _mm256_storeu_ps(f_temp, ymm7);
                for (int i = 0; i < m_remainder; i++)
                {
                    tC[i] = f_temp[i];
                }

                tC += ldc;
                for (int i = 0; i < m_remainder; i++)
                {
                    f_temp[i] = tC[i];
                }
                ymm2 = _mm256_loadu_ps(f_temp);
                if(is_beta_non_zero){
                    ymm9 = _mm256_fmadd_ps(ymm2, ymm1, ymm9);
                }
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
                ymm1 = _mm256_broadcast_ss(beta_cast);

                ymm5 = _mm256_mul_ps(ymm5, ymm0);
                ymm7 = _mm256_mul_ps(ymm7, ymm0);

                for (int i = 0; i < m_remainder; i++)
                {
                    f_temp[i] = tC[i];
                }
                ymm2 = _mm256_loadu_ps(f_temp);
                if(is_beta_non_zero){
                    ymm5 = _mm256_fmadd_ps(ymm2, ymm1, ymm5);
                }
                _mm256_storeu_ps(f_temp, ymm5);
                for (int i = 0; i < m_remainder; i++)
                {
                    tC[i] = f_temp[i];
                }

                tC += ldc;
                for (int i = 0; i < m_remainder; i++)
                {
                    f_temp[i] = tC[i];
                }
                ymm2 = _mm256_loadu_ps(f_temp);
                if(is_beta_non_zero){
                    ymm7 = _mm256_fmadd_ps(ymm2, ymm1, ymm7);
                }
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

                // multiply C by beta and accumulate.
                ymm5 = _mm256_mul_ps(ymm5, ymm0);

                for (int i = 0; i < m_remainder; i++)
                {
                    f_temp[i] = tC[i];
                }
                ymm2 = _mm256_loadu_ps(f_temp);
                if(is_beta_non_zero){
                    ymm1 = _mm256_broadcast_ss(beta_cast);
                    ymm5 = _mm256_fmadd_ps(ymm2, ymm1, ymm5);
                }
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
                    if(is_beta_non_zero){
                        (*tC) = (*tC) * (*beta_cast) + result;
                    }else{
                        (*tC) = result;
                    }
                }
            }
        }

        // Return the buffer to pool
        if ((required_packing_A == 1) && bli_mem_is_alloc( &local_mem_buf_A_s) ) {

#ifdef BLIS_ENABLE_MEM_TRACING
        printf( "bli_sgemm_small(): releasing mem pool block\n" );
#endif
            bli_pba_release(&rntm,
                               &local_mem_buf_A_s);
        }
		
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
        return BLIS_SUCCESS;
    }
    else
	{
		AOCL_DTL_TRACE_EXIT_ERR(
			AOCL_DTL_LEVEL_INFO,
			"Invalid dimesions for small gemm."
			);
		return BLIS_NONCONFORMAL_DIMENSIONS;
	}

};

static err_t bli_dgemm_small
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
             cntl_t* cntl
     )
{

	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_INFO);
	
    gint_t M = bli_obj_length( c ); // number of rows of Matrix C
    gint_t N = bli_obj_width( c );  // number of columns of Matrix C
    gint_t K = bli_obj_width( a );  // number of columns of OP(A), will be updated if OP(A) is Transpose(A) .
    gint_t L = M * N;

    // when N is equal to 1 call GEMV instead of GEMM
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
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
        return BLIS_SUCCESS;
    }

    if (N<3) //Implemenation assumes that N is atleast 3.
	{
		AOCL_DTL_TRACE_EXIT_ERR(
			AOCL_DTL_LEVEL_INFO,
			"N < 3, cannot be processed by small gemm"
			);
        return BLIS_NOT_YET_IMPLEMENTED;
	}

#ifdef BLIS_ENABLE_SMALL_MATRIX_ROME
    if( (L && K) && ((K < D_BLIS_SMALL_MATRIX_K_THRES_ROME) || ((N < BLIS_SMALL_MATRIX_THRES_ROME) && (K < BLIS_SMALL_MATRIX_THRES_ROME))))
#else
    if ((((L) < (D_BLIS_SMALL_MATRIX_THRES * D_BLIS_SMALL_MATRIX_THRES))
        || ((M  < D_BLIS_SMALL_M_RECT_MATRIX_THRES) && (K < D_BLIS_SMALL_K_RECT_MATRIX_THRES))) && ((L!=0) && (K!=0)))
#endif   
    {
        guint_t lda = bli_obj_col_stride( a ); // column stride of matrix OP(A), where OP(A) is Transpose(A) if transA enabled.
        guint_t ldb = bli_obj_col_stride( b ); // column stride of matrix OP(B), where OP(B) is Transpose(B) if transB enabled.
        guint_t ldc = bli_obj_col_stride( c ); // column stride of matrix C
        guint_t row_idx, col_idx, k;
        double *A = bli_obj_buffer_at_off(a); // pointer to elements of Matrix A
        double *B = bli_obj_buffer_at_off(b); // pointer to elements of Matrix B
        double *C = bli_obj_buffer_at_off(c); // pointer to elements of Matrix C

        double *tA = A, *tB = B, *tC = C;//, *tA_pack;
        double *tA_packed; // temprorary pointer to hold packed A memory pointer
        guint_t row_idx_packed; //packed A memory row index
        guint_t lda_packed; //lda of packed A
        guint_t col_idx_start; //starting index after A matrix is packed.
        dim_t tb_inc_row = 1; // row stride of matrix B
        dim_t tb_inc_col = ldb; // column stride of matrix B
        __m256d ymm4, ymm5, ymm6, ymm7;
        __m256d ymm8, ymm9, ymm10, ymm11;
        __m256d ymm12, ymm13, ymm14, ymm15;
        __m256d ymm0, ymm1, ymm2, ymm3;

        gint_t n_remainder; // If the N is non multiple of 3.(N%3)
        gint_t m_remainder; // If the M is non multiple of 16.(M%16)

        double *alpha_cast, *beta_cast; // alpha, beta multiples
        alpha_cast = bli_obj_buffer_for_1x1(BLIS_DOUBLE, alpha);
        beta_cast = bli_obj_buffer_for_1x1(BLIS_DOUBLE, beta);

        gint_t required_packing_A = 1;
        mem_t local_mem_buf_A_s;
        double *D_A_pack = NULL;
        rntm_t rntm;

        //update the pointer math if matrix B needs to be transposed.
        if (bli_obj_has_trans( b ))
        {
            tb_inc_col = 1; //switch row and column strides
            tb_inc_row = ldb;
        }

        //checking whether beta value is zero.
        //if true, we should perform C=alpha * A*B operation
        //instead of C = beta * C + alpha * (A * B)
        bool is_beta_non_zero = 0;
        if(!bli_obj_equals(beta, &BLIS_ZERO))
                is_beta_non_zero = 1;

        /*
         * This function was using global array to pack part of A input when needed.
         * However, using this global array make the function non-reentrant.
         * Instead of using a global array we should allocate buffer for each invocation.
         * Since the buffer size is too big or stack and doing malloc every time will be too expensive,
         * better approach is to get the buffer from the pre-allocated pool and return
         * it the pool once we are doing.
         *
         * In order to get the buffer from pool, we need access to memory broker,
         * currently this function is not invoked in such a way that it can receive
         * the memory broker (via rntm). Following hack will get the global memory
         * broker that can be use it to access the pool.
         *
         * Note there will be memory allocation at least on first innovation
         * as there will not be any pool created for this size.
         * Subsequent invocations will just reuse the buffer from the pool.
         */

        bli_rntm_init_from_global( &rntm );
        bli_rntm_set_num_threads_only( 1, &rntm );
        bli_pba_rntm_set_pba( &rntm );

        // Get the current size of the buffer pool for A block packing.
        // We will use the same size to avoid pool re-initliazaton 
        siz_t buffer_size = bli_pool_block_size(
            bli_pba_pool(bli_packbuf_index(BLIS_BITVAL_BUFFER_FOR_A_BLOCK),
                            bli_rntm_pba(&rntm)));

        //
        // This kernel assumes that "A" will be unpackged if N <= 3.
        // Usually this range (N <= 3) is handled by SUP, however,
        // if SUP is disabled or for any other condition if we do
        // enter this kernel with N <= 3, we want to make sure that
        // "A" remains unpacked.
        //
        // If this check is removed it will result in the crash as
        // reported in CPUPL-587.
        //

        if ((N <= 3) || ((D_MR * K) << 3) > buffer_size)
        {
            required_packing_A = 0;
        }
        
        if (required_packing_A == 1)
        {
#ifdef BLIS_ENABLE_MEM_TRACING
            printf( "bli_dgemm_small: Requesting mem pool block of size %lu\n", buffer_size);
#endif
            // Get the buffer from the pool.
            bli_pba_acquire_m(&rntm,
                                 buffer_size,
                                 BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
                                 &local_mem_buf_A_s);

            D_A_pack = bli_mem_buffer(&local_mem_buf_A_s);
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

#ifdef BLIS_ENABLE_PREFETCH
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
                ymm1 = _mm256_broadcast_sd(beta_cast);

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

                if(is_beta_non_zero)
                {
                    // multiply C by beta and accumulate col 1.
                    ymm2 = _mm256_loadu_pd(tC);
                    ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4);
                    ymm2 = _mm256_loadu_pd(tC + 4);
                    ymm5 = _mm256_fmadd_pd(ymm2, ymm1, ymm5);
                    ymm2 = _mm256_loadu_pd(tC + 8);
                    ymm6 = _mm256_fmadd_pd(ymm2, ymm1, ymm6);
                    ymm2 = _mm256_loadu_pd(tC + 12);
                    ymm7 = _mm256_fmadd_pd(ymm2, ymm1, ymm7);

                    double* ttC = tC + ldc;

                    // multiply C by beta and accumulate, col 2.
                    ymm2 = _mm256_loadu_pd(ttC);
                    ymm8 = _mm256_fmadd_pd(ymm2, ymm1, ymm8);
                    ymm2 = _mm256_loadu_pd(ttC + 4);
                    ymm9 = _mm256_fmadd_pd(ymm2, ymm1, ymm9);
                    ymm2 = _mm256_loadu_pd(ttC + 8);
                    ymm10 = _mm256_fmadd_pd(ymm2, ymm1, ymm10);
                    ymm2 = _mm256_loadu_pd(ttC + 12);
                    ymm11 = _mm256_fmadd_pd(ymm2, ymm1, ymm11);

                    ttC += ldc;

                    // multiply C by beta and accumulate, col 3.
                    ymm2 = _mm256_loadu_pd(ttC);
                    ymm12 = _mm256_fmadd_pd(ymm2, ymm1, ymm12);
                    ymm2 = _mm256_loadu_pd(ttC + 4);
                    ymm13 = _mm256_fmadd_pd(ymm2, ymm1, ymm13);
                    ymm2 = _mm256_loadu_pd(ttC + 8);
                    ymm14 = _mm256_fmadd_pd(ymm2, ymm1, ymm14);
                    ymm2 = _mm256_loadu_pd(ttC + 12);
                    ymm15 = _mm256_fmadd_pd(ymm2, ymm1, ymm15);
                }
                _mm256_storeu_pd(tC, ymm4);
                _mm256_storeu_pd(tC + 4, ymm5);
                _mm256_storeu_pd(tC + 8, ymm6);
                _mm256_storeu_pd(tC + 12, ymm7);

                tC += ldc;

                _mm256_storeu_pd(tC, ymm8);
                _mm256_storeu_pd(tC + 4, ymm9);
                _mm256_storeu_pd(tC + 8, ymm10);
                _mm256_storeu_pd(tC + 12, ymm11);

                tC += ldc;

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

#ifdef BLIS_ENABLE_PREFETCH
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
                ymm1 = _mm256_broadcast_sd(beta_cast);

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

                if(is_beta_non_zero)
                {
                    // multiply C by beta and accumulate col 1.
                    ymm2 = _mm256_loadu_pd(tC);
                    ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4);
                    ymm2 = _mm256_loadu_pd(tC + 4);
                    ymm5 = _mm256_fmadd_pd(ymm2, ymm1, ymm5);
                    ymm2 = _mm256_loadu_pd(tC + 8);
                    ymm6 = _mm256_fmadd_pd(ymm2, ymm1, ymm6);
                    ymm2 = _mm256_loadu_pd(tC + 12);
                    ymm7 = _mm256_fmadd_pd(ymm2, ymm1, ymm7);

                    // multiply C by beta and accumulate, col 2.
                    double* ttC  = tC + ldc;
                    ymm2 = _mm256_loadu_pd(ttC);
                    ymm8 = _mm256_fmadd_pd(ymm2, ymm1, ymm8);
                    ymm2 = _mm256_loadu_pd(ttC + 4);
                    ymm9 = _mm256_fmadd_pd(ymm2, ymm1, ymm9);
                    ymm2 = _mm256_loadu_pd(ttC + 8);
                    ymm10 = _mm256_fmadd_pd(ymm2, ymm1, ymm10);
                    ymm2 = _mm256_loadu_pd(ttC + 12);
                    ymm11 = _mm256_fmadd_pd(ymm2, ymm1, ymm11);

                    // multiply C by beta and accumulate, col 3.
                    ttC += ldc;
                    ymm2 = _mm256_loadu_pd(ttC);
                    ymm12 = _mm256_fmadd_pd(ymm2, ymm1, ymm12);
                    ymm2 = _mm256_loadu_pd(ttC + 4);
                    ymm13 = _mm256_fmadd_pd(ymm2, ymm1, ymm13);
                    ymm2 = _mm256_loadu_pd(ttC + 8);
                    ymm14 = _mm256_fmadd_pd(ymm2, ymm1, ymm14);
                    ymm2 = _mm256_loadu_pd(ttC + 12);
                    ymm15 = _mm256_fmadd_pd(ymm2, ymm1, ymm15);
                }
                _mm256_storeu_pd(tC, ymm4);
                _mm256_storeu_pd(tC + 4, ymm5);
                _mm256_storeu_pd(tC + 8, ymm6);
                _mm256_storeu_pd(tC + 12, ymm7);

                tC += ldc;

                _mm256_storeu_pd(tC, ymm8);
                _mm256_storeu_pd(tC + 4, ymm9);
                _mm256_storeu_pd(tC + 8, ymm10);
                _mm256_storeu_pd(tC + 12, ymm11);

                tC += ldc;

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
                ymm1 = _mm256_broadcast_sd(beta_cast);

                //multiply A*B by alpha.
                ymm8 = _mm256_mul_pd(ymm8, ymm0);
                ymm9 = _mm256_mul_pd(ymm9, ymm0);
                ymm10 = _mm256_mul_pd(ymm10, ymm0);
                ymm11 = _mm256_mul_pd(ymm11, ymm0);
                ymm12 = _mm256_mul_pd(ymm12, ymm0);
                ymm13 = _mm256_mul_pd(ymm13, ymm0);
                ymm14 = _mm256_mul_pd(ymm14, ymm0);
                ymm15 = _mm256_mul_pd(ymm15, ymm0);

                if(is_beta_non_zero)
                {
                    // multiply C by beta and accumulate, col 1.
                    ymm2 = _mm256_loadu_pd(tC + 0);
                    ymm8 = _mm256_fmadd_pd(ymm2, ymm1, ymm8);
                    ymm2 = _mm256_loadu_pd(tC + 4);
                    ymm9 = _mm256_fmadd_pd(ymm2, ymm1, ymm9);
                    ymm2 = _mm256_loadu_pd(tC + 8);
                    ymm10 = _mm256_fmadd_pd(ymm2, ymm1, ymm10);
                    ymm2 = _mm256_loadu_pd(tC + 12);
                    ymm11 = _mm256_fmadd_pd(ymm2, ymm1, ymm11);

                    // multiply C by beta and accumulate, col 2.
                    double *ttC = tC + ldc;

                    ymm2 = _mm256_loadu_pd(ttC);
                    ymm12 = _mm256_fmadd_pd(ymm2, ymm1, ymm12);
                    ymm2 = _mm256_loadu_pd(ttC + 4);
                    ymm13 = _mm256_fmadd_pd(ymm2, ymm1, ymm13);
                    ymm2 = _mm256_loadu_pd(ttC + 8);
                    ymm14 = _mm256_fmadd_pd(ymm2, ymm1, ymm14);
                    ymm2 = _mm256_loadu_pd(ttC + 12);
                    ymm15 = _mm256_fmadd_pd(ymm2, ymm1, ymm15);
                }

                _mm256_storeu_pd(tC + 0, ymm8);
                _mm256_storeu_pd(tC + 4, ymm9);
                _mm256_storeu_pd(tC + 8, ymm10);
                _mm256_storeu_pd(tC + 12, ymm11);

                tC += ldc;

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
                ymm1 = _mm256_broadcast_sd(beta_cast);

                //multiply A*B by alpha.
                ymm12 = _mm256_mul_pd(ymm12, ymm0);
                ymm13 = _mm256_mul_pd(ymm13, ymm0);
                ymm14 = _mm256_mul_pd(ymm14, ymm0);
                ymm15 = _mm256_mul_pd(ymm15, ymm0);

                if(is_beta_non_zero)
                {
                    // multiply C by beta and accumulate.
                    ymm2 = _mm256_loadu_pd(tC + 0);
                    ymm12 = _mm256_fmadd_pd(ymm2, ymm1, ymm12);
                    ymm2 = _mm256_loadu_pd(tC + 4);
                    ymm13 = _mm256_fmadd_pd(ymm2, ymm1, ymm13);
                    ymm2 = _mm256_loadu_pd(tC + 8);
                    ymm14 = _mm256_fmadd_pd(ymm2, ymm1, ymm14);
                    ymm2 = _mm256_loadu_pd(tC + 12);
                    ymm15 = _mm256_fmadd_pd(ymm2, ymm1, ymm15);
                }

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
                ymm1 = _mm256_broadcast_sd(beta_cast);

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

                if(is_beta_non_zero)
                {
                    // multiply C by beta and accumulate.
                    ymm2 = _mm256_loadu_pd(tC);
                    ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4);
                    ymm2 = _mm256_loadu_pd(tC + 4);
                    ymm5 = _mm256_fmadd_pd(ymm2, ymm1, ymm5);
                    ymm2 = _mm256_loadu_pd(tC + 8);
                    ymm6 = _mm256_fmadd_pd(ymm2, ymm1, ymm6);

                    // multiply C by beta and accumulate.
                    double *ttC = tC +ldc;
                    ymm2 = _mm256_loadu_pd(ttC);
                    ymm8 = _mm256_fmadd_pd(ymm2, ymm1, ymm8);
                    ymm2 = _mm256_loadu_pd(ttC + 4);
                    ymm9 = _mm256_fmadd_pd(ymm2, ymm1, ymm9);
                    ymm2 = _mm256_loadu_pd(ttC + 8);
                    ymm10 = _mm256_fmadd_pd(ymm2, ymm1, ymm10);

                    // multiply C by beta and accumulate.
                    ttC += ldc;
                    ymm2 = _mm256_loadu_pd(ttC);
                    ymm12 = _mm256_fmadd_pd(ymm2, ymm1, ymm12);
                    ymm2 = _mm256_loadu_pd(ttC + 4);
                    ymm13 = _mm256_fmadd_pd(ymm2, ymm1, ymm13);
                    ymm2 = _mm256_loadu_pd(ttC + 8);
                    ymm14 = _mm256_fmadd_pd(ymm2, ymm1, ymm14);

                }
                _mm256_storeu_pd(tC, ymm4);
                _mm256_storeu_pd(tC + 4, ymm5);
                _mm256_storeu_pd(tC + 8, ymm6);

                tC += ldc;

                _mm256_storeu_pd(tC, ymm8);
                _mm256_storeu_pd(tC + 4, ymm9);
                _mm256_storeu_pd(tC + 8, ymm10);

                tC += ldc;

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
                ymm1 = _mm256_broadcast_sd(beta_cast);

                //multiply A*B by alpha.
                ymm8 = _mm256_mul_pd(ymm8, ymm0);
                ymm9 = _mm256_mul_pd(ymm9, ymm0);
                ymm10 = _mm256_mul_pd(ymm10, ymm0);
                ymm12 = _mm256_mul_pd(ymm12, ymm0);
                ymm13 = _mm256_mul_pd(ymm13, ymm0);
                ymm14 = _mm256_mul_pd(ymm14, ymm0);


                if(is_beta_non_zero)
                {
                    // multiply C by beta and accumulate.
                    ymm2 = _mm256_loadu_pd(tC + 0);
                    ymm8 = _mm256_fmadd_pd(ymm2, ymm1, ymm8);
                    ymm2 = _mm256_loadu_pd(tC + 4);
                    ymm9 = _mm256_fmadd_pd(ymm2, ymm1, ymm9);
                    ymm2 = _mm256_loadu_pd(tC + 8);
                    ymm10 = _mm256_fmadd_pd(ymm2, ymm1, ymm10);

                    double *ttC = tC + ldc;

                    // multiply C by beta and accumulate.
                    ymm2 = _mm256_loadu_pd(ttC);
                    ymm12 = _mm256_fmadd_pd(ymm2, ymm1, ymm12);
                    ymm2 = _mm256_loadu_pd(ttC + 4);
                    ymm13 = _mm256_fmadd_pd(ymm2, ymm1, ymm13);
                    ymm2 = _mm256_loadu_pd(ttC + 8);
                    ymm14 = _mm256_fmadd_pd(ymm2, ymm1, ymm14);

                }
                _mm256_storeu_pd(tC + 0, ymm8);
                _mm256_storeu_pd(tC + 4, ymm9);
                _mm256_storeu_pd(tC + 8, ymm10);

                tC += ldc;

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
                ymm1 = _mm256_broadcast_sd(beta_cast);

                //multiply A*B by alpha.
                ymm12 = _mm256_mul_pd(ymm12, ymm0);
                ymm13 = _mm256_mul_pd(ymm13, ymm0);
                ymm14 = _mm256_mul_pd(ymm14, ymm0);


                if(is_beta_non_zero)
                {
                    // multiply C by beta and accumulate.
                    ymm2 = _mm256_loadu_pd(tC + 0);
                    ymm12 = _mm256_fmadd_pd(ymm2, ymm1, ymm12);
                    ymm2 = _mm256_loadu_pd(tC + 4);
                    ymm13 = _mm256_fmadd_pd(ymm2, ymm1, ymm13);
                    ymm2 = _mm256_loadu_pd(tC + 8);
                    ymm14 = _mm256_fmadd_pd(ymm2, ymm1, ymm14);

                }
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
                ymm1 = _mm256_broadcast_sd(beta_cast);

                //multiply A*B by alpha.
                ymm4 = _mm256_mul_pd(ymm4, ymm0);
                ymm5 = _mm256_mul_pd(ymm5, ymm0);
                ymm6 = _mm256_mul_pd(ymm6, ymm0);
                ymm7 = _mm256_mul_pd(ymm7, ymm0);
                ymm8 = _mm256_mul_pd(ymm8, ymm0);
                ymm9 = _mm256_mul_pd(ymm9, ymm0);

                if(is_beta_non_zero)
                {
                    // multiply C by beta and accumulate.
                    ymm2 = _mm256_loadu_pd(tC);
                    ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4);
                    ymm2 = _mm256_loadu_pd(tC + 4);
                    ymm5 = _mm256_fmadd_pd(ymm2, ymm1, ymm5);

                    double* ttC = tC + ldc;

                    // multiply C by beta and accumulate.
                    ymm2 = _mm256_loadu_pd(ttC);
                    ymm6 = _mm256_fmadd_pd(ymm2, ymm1, ymm6);
                    ymm2 = _mm256_loadu_pd(ttC + 4);
                    ymm7 = _mm256_fmadd_pd(ymm2, ymm1, ymm7);

                    ttC += ldc;

                    // multiply C by beta and accumulate.
                    ymm2 = _mm256_loadu_pd(ttC);
                    ymm8 = _mm256_fmadd_pd(ymm2, ymm1, ymm8);
                    ymm2 = _mm256_loadu_pd(ttC + 4);
                    ymm9 = _mm256_fmadd_pd(ymm2, ymm1, ymm9);
                }

                _mm256_storeu_pd(tC, ymm4);
                _mm256_storeu_pd(tC + 4, ymm5);

                tC += ldc;
                _mm256_storeu_pd(tC, ymm6);
                _mm256_storeu_pd(tC + 4, ymm7);

                tC += ldc;
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
                ymm1 = _mm256_broadcast_sd(beta_cast);

                //multiply A*B by alpha.
                ymm4 = _mm256_mul_pd(ymm4, ymm0);
                ymm5 = _mm256_mul_pd(ymm5, ymm0);
                ymm6 = _mm256_mul_pd(ymm6, ymm0);
                ymm7 = _mm256_mul_pd(ymm7, ymm0);

                if(is_beta_non_zero)
                {
                // multiply C by beta and accumulate.
                ymm2 = _mm256_loadu_pd(tC);
                ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4);
                ymm2 = _mm256_loadu_pd(tC + 4);
                ymm5 = _mm256_fmadd_pd(ymm2, ymm1, ymm5);

                double* ttC = tC + ldc;

                // multiply C by beta and accumulate.
                ymm2 = _mm256_loadu_pd(ttC);
                ymm6 = _mm256_fmadd_pd(ymm2, ymm1, ymm6);
                ymm2 = _mm256_loadu_pd(ttC + 4);
                ymm7 = _mm256_fmadd_pd(ymm2, ymm1, ymm7);
                }
                _mm256_storeu_pd(tC, ymm4);
                _mm256_storeu_pd(tC + 4, ymm5);

                tC += ldc;
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
                ymm1 = _mm256_broadcast_sd(beta_cast);

                ymm4 = _mm256_mul_pd(ymm4, ymm0);
                ymm5 = _mm256_mul_pd(ymm5, ymm0);

                if(is_beta_non_zero)
                {
                    // multiply C by beta and accumulate.
                    ymm2 = _mm256_loadu_pd(tC);
                    ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4);
                    ymm2 = _mm256_loadu_pd(tC + 4);
                    ymm5 = _mm256_fmadd_pd(ymm2, ymm1, ymm5);
                }
                _mm256_storeu_pd(tC, ymm4);
                _mm256_storeu_pd(tC + 4, ymm5);

            }

            row_idx += 8;
        }

        if (m_remainder >= 4)
        {
            //printf("HERE\n");
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
                ymm1 = _mm256_broadcast_sd(beta_cast);

                //multiply A*B by alpha.
                ymm4 = _mm256_mul_pd(ymm4, ymm0);
                ymm5 = _mm256_mul_pd(ymm5, ymm0);
                ymm6 = _mm256_mul_pd(ymm6, ymm0);

                if(is_beta_non_zero)
                {
                    // multiply C by beta and accumulate.
                    ymm2 = _mm256_loadu_pd(tC);
                    ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4);

                    double* ttC = tC + ldc;

                    // multiply C by beta and accumulate.
                    ymm2 = _mm256_loadu_pd(ttC);
                    ymm5 = _mm256_fmadd_pd(ymm2, ymm1, ymm5);

                    ttC += ldc;

                    // multiply C by beta and accumulate.
                    ymm2 = _mm256_loadu_pd(ttC);
                    ymm6 = _mm256_fmadd_pd(ymm2, ymm1, ymm6);
                }
                _mm256_storeu_pd(tC, ymm4);

                tC += ldc;
                _mm256_storeu_pd(tC, ymm5);

                tC += ldc;
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
                ymm1 = _mm256_broadcast_sd(beta_cast);

                //multiply A*B by alpha.
                ymm4 = _mm256_mul_pd(ymm4, ymm0);
                ymm5 = _mm256_mul_pd(ymm5, ymm0);

                if(is_beta_non_zero)
                {
                    // multiply C by beta and accumulate.
                    ymm2 = _mm256_loadu_pd(tC);
                    ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4);

                    double* ttC = tC + ldc;

                    // multiply C by beta and accumulate.
                    ymm2 = _mm256_loadu_pd(ttC);
                    ymm5 = _mm256_fmadd_pd(ymm2, ymm1, ymm5);
                }
                _mm256_storeu_pd(tC, ymm4);

                tC += ldc;
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
                ymm1 = _mm256_broadcast_sd(beta_cast);

                ymm4 = _mm256_mul_pd(ymm4, ymm0);

                if(is_beta_non_zero)
                {
                    // multiply C by beta and accumulate.
                    ymm2 = _mm256_loadu_pd(tC);
                    ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4);

                }
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
            double f_temp[8] = {0.0};

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
                ymm1 = _mm256_broadcast_sd(beta_cast);

                //multiply A*B by alpha.
                ymm5 = _mm256_mul_pd(ymm5, ymm0);
                ymm7 = _mm256_mul_pd(ymm7, ymm0);
                ymm9 = _mm256_mul_pd(ymm9, ymm0);

                if(is_beta_non_zero)
                {
                    for (int i = 0; i < m_remainder; i++)
                    {
                        f_temp[i] = tC[i];
                    }
                    ymm2 = _mm256_loadu_pd(f_temp);
                    ymm5 = _mm256_fmadd_pd(ymm2, ymm1, ymm5);


                    double* ttC = tC + ldc;

                    for (int i = 0; i < m_remainder; i++)
                    {
                        f_temp[i] = ttC[i];
                    }
                    ymm2 = _mm256_loadu_pd(f_temp);
                    ymm7 = _mm256_fmadd_pd(ymm2, ymm1, ymm7);

                    ttC += ldc;
                    for (int i = 0; i < m_remainder; i++)
                    {
                        f_temp[i] = ttC[i];
                    }
                    ymm2 = _mm256_loadu_pd(f_temp);
                    ymm9 = _mm256_fmadd_pd(ymm2, ymm1, ymm9);
                }
                    _mm256_storeu_pd(f_temp, ymm5);
                    for (int i = 0; i < m_remainder; i++)
                    {
                        tC[i] = f_temp[i];
                    }

                    tC += ldc;
                    _mm256_storeu_pd(f_temp, ymm7);
                    for (int i = 0; i < m_remainder; i++)
                    {
                        tC[i] = f_temp[i];
                    }

                    tC += ldc;
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
                ymm1 = _mm256_broadcast_sd(beta_cast);

                ymm5 = _mm256_mul_pd(ymm5, ymm0);
                ymm7 = _mm256_mul_pd(ymm7, ymm0);

                if(is_beta_non_zero)
                {
                    for (int i = 0; i < m_remainder; i++)
                    {
                        f_temp[i] = tC[i];
                    }
                    ymm2 = _mm256_loadu_pd(f_temp);
                    ymm5 = _mm256_fmadd_pd(ymm2, ymm1, ymm5);

                    double* ttC = tC + ldc;

                    for (int i = 0; i < m_remainder; i++)
                    {
                        f_temp[i] = ttC[i];
                    }
                    ymm2 = _mm256_loadu_pd(f_temp);
                    ymm7 = _mm256_fmadd_pd(ymm2, ymm1, ymm7);

                }
                _mm256_storeu_pd(f_temp, ymm5);
                for (int i = 0; i < m_remainder; i++)
                {
                    tC[i] = f_temp[i];
                }

                tC += ldc;
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
                ymm1 = _mm256_broadcast_sd(beta_cast);

                // multiply C by beta and accumulate.
                ymm5 = _mm256_mul_pd(ymm5, ymm0);

                if(is_beta_non_zero)
                {

                    for (int i = 0; i < m_remainder; i++)
                    {
                        f_temp[i] = tC[i];
                    }
                    ymm2 = _mm256_loadu_pd(f_temp);
                    ymm5 = _mm256_fmadd_pd(ymm2, ymm1, ymm5);
                }
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
                    if(is_beta_non_zero)
                        (*tC) = (*tC) * (*beta_cast) + result;
                    else
                    (*tC) = result;
                }
            }
        }

    // Return the buffer to pool
        if ((required_packing_A == 1) && bli_mem_is_alloc( &local_mem_buf_A_s )) {
#ifdef BLIS_ENABLE_MEM_TRACING
        printf( "bli_dgemm_small(): releasing mem pool block\n" );
#endif
        bli_pba_release(&rntm,
                           &local_mem_buf_A_s);
        }
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
        return BLIS_SUCCESS;
    }
    else
	{
		AOCL_DTL_TRACE_EXIT_ERR(
			AOCL_DTL_LEVEL_INFO,
			"Invalid dimesions for small gemm."
			);
        return BLIS_NONCONFORMAL_DIMENSIONS;
	}
};

static err_t bli_sgemm_small_atbn
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
             cntl_t* cntl
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_INFO);
	
	gint_t M = bli_obj_length( c ); // number of rows of Matrix C
    gint_t N = bli_obj_width( c );  // number of columns of Matrix C
    gint_t K = bli_obj_length( b ); // number of rows of Matrix B

    guint_t lda = bli_obj_col_stride( a ); // column stride of matrix OP(A), where OP(A) is Transpose(A) if transA enabled.
    guint_t ldb = bli_obj_col_stride( b ); // column stride of matrix OP(B), where OP(B) is Transpose(B) if transB enabled.
    guint_t ldc = bli_obj_col_stride( c ); // column stride of matrix C

    int row_idx = 0, col_idx = 0, k;

    float *A = bli_obj_buffer_at_off(a); // pointer to matrix A elements, stored in row major format
    float *B = bli_obj_buffer_at_off(b); // pointer to matrix B elements, stored in column major format
    float *C = bli_obj_buffer_at_off(c); // pointer to matrix C elements, stored in column major format

    float *tA = A, *tB = B, *tC = C;

    __m256 ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11;
    __m256 ymm12, ymm13, ymm14, ymm15;
    __m256 ymm0, ymm1, ymm2, ymm3;

    float result;
    float scratch[8] = {0.0};
    const num_t    dt_exec   = bli_obj_dt( c );
    float* restrict alpha_cast = bli_obj_buffer_for_1x1( dt_exec, alpha );
    float* restrict beta_cast  = bli_obj_buffer_for_1x1( dt_exec, beta );	

    /*Beta Zero Check*/
    bool is_beta_non_zero=0;
    if ( !bli_obj_equals( beta, &BLIS_ZERO ) ){
        is_beta_non_zero = 1;
    }

    // The non-copy version of the A^T GEMM gives better performance for the small M cases.
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
                if(is_beta_non_zero){
                    tC[0] = result + tC[0] * (*beta_cast);
                }else{
                    tC[0] = result;
                }

                ymm7 = _mm256_hadd_ps(ymm7, ymm7);
                ymm7 = _mm256_hadd_ps(ymm7, ymm7);
                _mm256_storeu_ps(scratch, ymm7);
                result = scratch[0] + scratch[4];
                result *= (*alpha_cast);
                if(is_beta_non_zero){
                    tC[1] = result + tC[1] * (*beta_cast);
                }else{
                    tC[1] = result;
                }

                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                _mm256_storeu_ps(scratch, ymm10);
                result = scratch[0] + scratch[4];
                result *= (*alpha_cast);
                if(is_beta_non_zero){
                    tC[2] = result + tC[2] * (*beta_cast);
                }else{
                    tC[2] = result;
                }

                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                _mm256_storeu_ps(scratch, ymm13);
                result = scratch[0] + scratch[4];
                result *= (*alpha_cast);
                if(is_beta_non_zero){
                    tC[3] = result + tC[3] * (*beta_cast);
                }else{
                    tC[3] = result;
                }

                tC += ldc;
                ymm5 = _mm256_hadd_ps(ymm5, ymm5);
                ymm5 = _mm256_hadd_ps(ymm5, ymm5);
                _mm256_storeu_ps(scratch, ymm5);
                result = scratch[0] + scratch[4];
                result *= (*alpha_cast);
                if(is_beta_non_zero){
                    tC[0] = result + tC[0] * (*beta_cast);
                }else{
                    tC[0] = result;
                }

                ymm8 = _mm256_hadd_ps(ymm8, ymm8);
                ymm8 = _mm256_hadd_ps(ymm8, ymm8);
                _mm256_storeu_ps(scratch, ymm8);
                result = scratch[0] + scratch[4];
                result *= (*alpha_cast);
                if(is_beta_non_zero){
                    tC[1] = result + tC[1] * (*beta_cast);
                }else{
                    tC[1] = result;
                }

                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                ymm11 = _mm256_hadd_ps(ymm11, ymm11);
                _mm256_storeu_ps(scratch, ymm11);
                result = scratch[0] + scratch[4];
                result *= (*alpha_cast);
                if(is_beta_non_zero){
                    tC[2] = result + tC[2] * (*beta_cast);
                }else{
                    tC[2] = result;
                }

                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                ymm14 = _mm256_hadd_ps(ymm14, ymm14);
                _mm256_storeu_ps(scratch, ymm14);
                result = scratch[0] + scratch[4];
                result *= (*alpha_cast);
                if(is_beta_non_zero){
                    tC[3] = result + tC[3] * (*beta_cast);
                }else{
                    tC[3] = result;
                }

                tC += ldc;
                ymm6 = _mm256_hadd_ps(ymm6, ymm6);
                ymm6 = _mm256_hadd_ps(ymm6, ymm6);
                _mm256_storeu_ps(scratch, ymm6);
                result = scratch[0] + scratch[4];
                result *= (*alpha_cast);
                if(is_beta_non_zero){
                    tC[0] = result + tC[0] * (*beta_cast);
                }else{
                    tC[0] = result;
                }

                ymm9 = _mm256_hadd_ps(ymm9, ymm9);
                ymm9 = _mm256_hadd_ps(ymm9, ymm9);
                _mm256_storeu_ps(scratch, ymm9);
                result = scratch[0] + scratch[4];
                result *= (*alpha_cast);
                if(is_beta_non_zero){
                    tC[1] = result + tC[1] * (*beta_cast);
                }else{
                    tC[1] = result;
                }

                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                ymm12 = _mm256_hadd_ps(ymm12, ymm12);
                _mm256_storeu_ps(scratch, ymm12);
                result = scratch[0] + scratch[4];
                result *= (*alpha_cast);
                if(is_beta_non_zero){
                    tC[2] = result + tC[2] * (*beta_cast);
                }else{
                    tC[2] = result;
                }

                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                ymm15 = _mm256_hadd_ps(ymm15, ymm15);
                _mm256_storeu_ps(scratch, ymm15);
                result = scratch[0] + scratch[4];
                result *= (*alpha_cast);
                if(is_beta_non_zero){
                    tC[3] = result + tC[3] * (*beta_cast);
                }else{
                    tC[3] = result;
                }
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
                    if(is_beta_non_zero){
                        tC[0] = result + tC[0] * (*beta_cast);
                    }else{
                        tC[0] = result;
                    }

                    ymm7 = _mm256_hadd_ps(ymm7, ymm7);
                    ymm7 = _mm256_hadd_ps(ymm7, ymm7);
                    _mm256_storeu_ps(scratch, ymm7);
                    result = scratch[0] + scratch[4];
                    result *= (*alpha_cast);
                    if(is_beta_non_zero){
                        tC[1] = result + tC[1] * (*beta_cast);
                    }else{
                        tC[1] = result;
                    }

                    ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                    ymm10 = _mm256_hadd_ps(ymm10, ymm10);
                    _mm256_storeu_ps(scratch, ymm10);
                    result = scratch[0] + scratch[4];
                    result *= (*alpha_cast);
                    if(is_beta_non_zero){
                        tC[2] = result + tC[2] * (*beta_cast);
                    }else{
                        tC[2] = result;
                    }

                    ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                    ymm13 = _mm256_hadd_ps(ymm13, ymm13);
                    _mm256_storeu_ps(scratch, ymm13);
                    result = scratch[0] + scratch[4];
                    result *= (*alpha_cast);
                    if(is_beta_non_zero){
                        tC[3] = result + tC[3] * (*beta_cast);
                    }else{
                        tC[3] = result;
                    }
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
                    if(is_beta_non_zero){
                        tC[0] = result + tC[0] * (*beta_cast);
                    }else{
                        tC[0] = result;
                    }

                }
            }
        }
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
        return BLIS_SUCCESS;
    }
    else
	{
		AOCL_DTL_TRACE_EXIT_ERR(
			AOCL_DTL_LEVEL_INFO,
			"Invalid dimesions for small gemm."
			);
        return BLIS_NONCONFORMAL_DIMENSIONS;
	}
}

static err_t bli_dgemm_small_atbn
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
             cntl_t* cntl
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_INFO);
	
    gint_t M = bli_obj_length( c ); // number of rows of Matrix C
    gint_t N = bli_obj_width( c );  // number of columns of Matrix C
    gint_t K = bli_obj_length( b ); // number of rows of Matrix B

    // The non-copy version of the A^T GEMM gives better performance for the small M cases.
    // The threshold is controlled by BLIS_ATBN_M_THRES
    if (M <= BLIS_ATBN_M_THRES)
    {
    guint_t lda = bli_obj_col_stride( a ); // column stride of matrix OP(A), where OP(A) is Transpose(A) if transA enabled.
    guint_t ldb = bli_obj_col_stride( b ); // column stride of matrix OP(B), where OP(B) is Transpose(B) if transB enabled.
    guint_t ldc = bli_obj_col_stride( c ); // column stride of matrix C
    guint_t row_idx = 0, col_idx = 0, k;
    double *A = bli_obj_buffer_at_off(a); // pointer to matrix A elements, stored in row major format
    double *B = bli_obj_buffer_at_off(b); // pointer to matrix B elements, stored in column major format
    double *C = bli_obj_buffer_at_off(c); // pointer to matrix C elements, stored in column major format

    double *tA = A, *tB = B, *tC = C;

    __m256d ymm4, ymm5, ymm6, ymm7;
    __m256d ymm8, ymm9, ymm10, ymm11;
    __m256d ymm12, ymm13, ymm14, ymm15;
    __m256d ymm0, ymm1, ymm2, ymm3;

    double result;
    double scratch[8] = {0.0};
    double *alpha_cast, *beta_cast; // alpha, beta multiples
    alpha_cast = bli_obj_buffer_for_1x1(BLIS_DOUBLE, alpha);
    beta_cast = bli_obj_buffer_for_1x1(BLIS_DOUBLE, beta);

    //check if beta is zero
    //if true, we need to perform C = alpha * (A * B)
    //instead of C = beta * C  + alpha * (A * B)
    bool is_beta_non_zero = 0;
    if(!bli_obj_equals(beta,&BLIS_ZERO))
        is_beta_non_zero = 1;

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
                if(is_beta_non_zero)
                    tC[0] = result + tC[0] * (*beta_cast);
                else
                    tC[0] = result;

                ymm7 = _mm256_hadd_pd(ymm7, ymm7);
                _mm256_storeu_pd(scratch, ymm7);
                result = scratch[0] + scratch[2];
                result *= (*alpha_cast);
                if(is_beta_non_zero)
                    tC[1] = result + tC[1] * (*beta_cast);
                else
                    tC[1] = result;

                ymm10 = _mm256_hadd_pd(ymm10, ymm10);
                _mm256_storeu_pd(scratch, ymm10);
                result = scratch[0] + scratch[2];
                result *= (*alpha_cast);
                if(is_beta_non_zero)
                    tC[2] = result + tC[2] * (*beta_cast);
                else
                    tC[2] = result;

                ymm13 = _mm256_hadd_pd(ymm13, ymm13);
                _mm256_storeu_pd(scratch, ymm13);
                result = scratch[0] + scratch[2];
                result *= (*alpha_cast);
                if(is_beta_non_zero)
                    tC[3] = result + tC[3] * (*beta_cast);
                else
                    tC[3] = result;

                tC += ldc;
                ymm5 = _mm256_hadd_pd(ymm5, ymm5);
                _mm256_storeu_pd(scratch, ymm5);
                result = scratch[0] + scratch[2];
                result *= (*alpha_cast);
                if(is_beta_non_zero)
                    tC[0] = result + tC[0] * (*beta_cast);
                else
                    tC[0] = result;

                ymm8 = _mm256_hadd_pd(ymm8, ymm8);
                _mm256_storeu_pd(scratch, ymm8);
                result = scratch[0] + scratch[2];
                result *= (*alpha_cast);
                if(is_beta_non_zero)
                    tC[1] = result + tC[1] * (*beta_cast);
                else
                    tC[1] = result;

                ymm11 = _mm256_hadd_pd(ymm11, ymm11);
                _mm256_storeu_pd(scratch, ymm11);
                result = scratch[0] + scratch[2];
                result *= (*alpha_cast);
                if(is_beta_non_zero)
                    tC[2] = result + tC[2] * (*beta_cast);
                else
                    tC[2] = result;

                ymm14 = _mm256_hadd_pd(ymm14, ymm14);
                _mm256_storeu_pd(scratch, ymm14);
                result = scratch[0] + scratch[2];
                result *= (*alpha_cast);
                if(is_beta_non_zero)
                    tC[3] = result + tC[3] * (*beta_cast);
                else
                    tC[3] = result;

                tC += ldc;
                ymm6 = _mm256_hadd_pd(ymm6, ymm6);
                _mm256_storeu_pd(scratch, ymm6);
                result = scratch[0] + scratch[2];
                result *= (*alpha_cast);
                if(is_beta_non_zero)
                    tC[0] = result + tC[0] * (*beta_cast);
                else
                    tC[0] = result;

                ymm9 = _mm256_hadd_pd(ymm9, ymm9);
                _mm256_storeu_pd(scratch, ymm9);
                result = scratch[0] + scratch[2];
                result *= (*alpha_cast);
                if(is_beta_non_zero)
                    tC[1] = result + tC[1] * (*beta_cast);
                else
                    tC[1] = result;

                ymm12 = _mm256_hadd_pd(ymm12, ymm12);
                _mm256_storeu_pd(scratch, ymm12);
                result = scratch[0] + scratch[2];
                result *= (*alpha_cast);
                if(is_beta_non_zero)
                    tC[2] = result + tC[2] * (*beta_cast);
                else
                    tC[2] = result;

                ymm15 = _mm256_hadd_pd(ymm15, ymm15);
                _mm256_storeu_pd(scratch, ymm15);
                result = scratch[0] + scratch[2];
                result *= (*alpha_cast);
                if(is_beta_non_zero)
                    tC[3] = result + tC[3] * (*beta_cast);
                else
                    tC[3] = result;
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
                    if(is_beta_non_zero)
                        tC[0] = result + tC[0] * (*beta_cast);
                    else
                        tC[0] = result;

                    ymm7 = _mm256_hadd_pd(ymm7, ymm7);
                    _mm256_storeu_pd(scratch, ymm7);
                    result = scratch[0] + scratch[2];
                    result *= (*alpha_cast);
                    if(is_beta_non_zero)
                        tC[1] = result + tC[1] * (*beta_cast);
                    else
                        tC[1] = result;

                    ymm10 = _mm256_hadd_pd(ymm10, ymm10);
                    _mm256_storeu_pd(scratch, ymm10);
                    result = scratch[0] + scratch[2];
                    result *= (*alpha_cast);
                    if(is_beta_non_zero)
                        tC[2] = result + tC[2] * (*beta_cast);
                    else
                        tC[2] = result;

                    ymm13 = _mm256_hadd_pd(ymm13, ymm13);
                    _mm256_storeu_pd(scratch, ymm13);
                    result = scratch[0] + scratch[2];
                    result *= (*alpha_cast);
                    if(is_beta_non_zero)
                        tC[3] = result + tC[3] * (*beta_cast);
                    else
                        tC[3] = result;
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
                    if(is_beta_non_zero)
                        tC[0] = result + tC[0] * (*beta_cast);
                    else
                        tC[0] = result;
                }
            }
        }
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO);
        return BLIS_SUCCESS;
    }
    else
	{
		AOCL_DTL_TRACE_EXIT_ERR(
			AOCL_DTL_LEVEL_INFO,
			"Invalid dimesions for small gemm."
			);
		return BLIS_NONCONFORMAL_DIMENSIONS;
	}
}
#endif

