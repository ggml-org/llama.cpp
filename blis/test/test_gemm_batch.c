/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.

   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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

#ifdef WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#include "blis.h"

//#define CHECK_CBLAS
#ifdef CHECK_CBLAS
#include "cblas.h"
#endif

/* Format for FILE input
 * For each input set, first line contains 'storage scheme'
 *  and 'group count' seperated by space.
 * Following 'group_count' number of lines contains all the parameters of
 * each group separated by space in each line in the following order:
 *  tA tB m n k lda ldb ldc alpha_r alpha_i beta_r beta_i group_size
 *
 * Example:
 * c 2
 * n n 4 8 4 4 4 4 1.1 0.0 0.9 0.0 2
 * n n 3 3 6 3 6 3 1.0 0.0 2.0 0.0 2
 *
 */

//#define FILE_IN_OUT
#ifndef FILE_IN_OUT
#define GRP_COUNT 2
#endif

//#define PRINT

int main( int argc, char** argv )
{
    num_t dt;

    char stor_scheme;
    dim_t i, j, idx;
    dim_t r, n_repeats;

    double dtime;
    double dtime_save;
    double gflops;

    dim_t total_count = 0;

#if 1
    dt = BLIS_FLOAT;
    //dt = BLIS_DOUBLE;
#else
    dt = BLIS_SCOMPLEX;
    //dt = BLIS_DCOMPLEX;
#endif

    n_repeats = 1;

#ifdef FILE_IN_OUT
    FILE* fin = NULL;
    FILE* fout = NULL;

    if(argc < 3)
    {
        printf("Usage: ./test_gemm_batch_XX.x input.csv output.csv\n");
        exit(1);
    }

    fin = fopen(argv[1], "r");
    if( fin == NULL )
    {
        printf("Error opening input file %s \n", argv[1]);
        exit(1);
    }

    fout = fopen(argv[2], "w");
    if(fout == NULL)
    {
        printf("Error opening output file %s\n",argv[2]);
        exit(1);
    }

    dim_t GRP_COUNT;

    fprintf(fout, "m\t n\t k\t lda\t ldb\t ldc\t transa\t transb\t grp_size\n");

    while(fscanf(fin, "%c %ld\n", &stor_scheme, &GRP_COUNT) == 2)
    {
        char transa[GRP_COUNT];
        char transb[GRP_COUNT];

        dim_t m[GRP_COUNT];
        dim_t n[GRP_COUNT];
        dim_t k[GRP_COUNT];

        dim_t lda[GRP_COUNT];
        dim_t ldb[GRP_COUNT];
        dim_t ldc[GRP_COUNT];

        double alpha_real[GRP_COUNT];
        double alpha_imag[GRP_COUNT];
        double beta_real[GRP_COUNT];
        double beta_imag[GRP_COUNT];

        dim_t group_size[GRP_COUNT];
        obj_t alpha[GRP_COUNT], beta[GRP_COUNT];

        total_count = 0;
        for(i = 0; i < GRP_COUNT; i++)
        {
            fscanf(fin, "%c %c %ld %ld %ld %ld %ld %ld %lf %lf %lf %lf %ld\n", &transa[i], &transb[i], &m[i], &n[i], &k[i], &lda[i], &ldb[i], &ldc[i], &alpha_real[i], &alpha_imag[i], &beta_real[i], &beta_imag[i], &group_size[i]);

            total_count += group_size[i];
        }
#else
        printf("m\t n\t k\t lda\t ldb\t ldc\t transa\t transb\t grp_size\n");

        stor_scheme = 'c';

        dim_t m[GRP_COUNT] = {4, 3};
        dim_t n[GRP_COUNT] = {8, 3};
        dim_t k[GRP_COUNT] = {4, 6};

        dim_t lda[GRP_COUNT] = {4, 3};
        dim_t ldb[GRP_COUNT] = {4, 6};
        dim_t ldc[GRP_COUNT] = {4, 3};

        char transa[GRP_COUNT] = {'N', 'N'};
        char transb[GRP_COUNT] = {'N', 'N'};

        double alpha_real[GRP_COUNT] = {1.1, 1.0};
        double alpha_imag[GRP_COUNT] = {0.0, 0.0};

        double beta_real[GRP_COUNT] = {0.9, 2.0};
        double beta_imag[GRP_COUNT] = {0.0, 0.0};

        dim_t group_size[GRP_COUNT] = {2,2};

        obj_t alpha[GRP_COUNT], beta[GRP_COUNT];

        total_count = 0;
        for(i = 0; i < GRP_COUNT; i++)
            total_count += group_size[i];

#endif
        obj_t a[total_count], b[total_count];
        obj_t c[total_count], c_save[total_count];
        f77_int f77_m[GRP_COUNT], f77_n[GRP_COUNT], f77_k[GRP_COUNT];
        f77_int f77_lda[GRP_COUNT], f77_ldb[GRP_COUNT], f77_ldc[GRP_COUNT];
        f77_int f77_group_size[GRP_COUNT];
        f77_int f77_group_count = GRP_COUNT;
#ifdef CHECK_CBLAS
        enum CBLAS_ORDER cblas_order;
        enum CBLAS_TRANSPOSE cblas_transa[GRP_COUNT];
        enum CBLAS_TRANSPOSE cblas_transb[GRP_COUNT];

        if(stor_scheme == 'R' || stor_scheme == 'r')
            cblas_order = CblasRowMajor;
        else
            cblas_order = CblasColMajor;

#else
        f77_char f77_transa[GRP_COUNT];
        f77_char f77_transb[GRP_COUNT];

        if(stor_scheme == 'r' || stor_scheme == 'R' )
        {
            printf("BLAS Interface doesn't support row-major order\n");
#ifdef FILE_IN_OUT
            continue;
#else
            exit(1);
#endif
        }
#endif

        idx = 0;
        for(i = 0; i < GRP_COUNT; i++)
        {
            bli_obj_create(dt, 1, 1, 0, 0, &alpha[i]);
            bli_obj_create(dt, 1, 1, 0, 0, &beta[i] );

            bli_setsc(alpha_real[i], alpha_imag[i], &alpha[i]);
            bli_setsc(beta_real[i],  beta_imag[i],  &beta[i] );

            trans_t blis_transa, blis_transb;
            if(transa[i] == 't' || transa[i] == 'T')
                blis_transa = BLIS_TRANSPOSE;
            else if (transa[i] == 'c' || transa[i] == 'C')
                blis_transa = BLIS_CONJ_TRANSPOSE;
            else if ( transa[i] == 'n' || transa[i] == 'N')
                    blis_transa = BLIS_NO_TRANSPOSE;
            else
            {
                printf("Illegal transA setting %c for group %ld\n", transa[i], i);
                exit(1);
            }

            if(transb[i] == 't' || transb[i] == 'T')
                blis_transb = BLIS_TRANSPOSE;
            else if (transb[i] == 'c' || transb[i] == 'C')
                blis_transb = BLIS_CONJ_TRANSPOSE;
            else if (transb[i] == 'n' || transb[i] == 'N')
                blis_transb = BLIS_NO_TRANSPOSE;
            else
            {
                printf("Illegal transB setting %c for group %ld\n", transb[i], i);
                exit(1);
            }
#ifdef CHECK_CBLAS
            if(bli_is_trans( blis_transa ))
                cblas_transa[i] = CblasTrans;
            else if (bli_is_conjtrans( blis_transa ))
                cblas_transa[i] = CblasConjTrans;
            else
                cblas_transa[i] = CblasNoTrans;

            if(bli_is_trans( blis_transb ))
                cblas_transb[i] = CblasTrans;
            else if (bli_is_conjtrans( blis_transb ))
                cblas_transb[i] = CblasConjTrans;
            else
                cblas_transb[i] = CblasNoTrans;
#else
            bli_param_map_blis_to_netlib_trans( blis_transa, &f77_transa[i]);
            bli_param_map_blis_to_netlib_trans( blis_transb, &f77_transb[i]);

#endif
            dim_t m0_a, n0_a;
            dim_t m0_b, n0_b;
            bli_set_dims_with_trans( blis_transa, m[i], k[i], &m0_a, &n0_a );
            bli_set_dims_with_trans( blis_transb, k[i], n[i], &m0_b, &n0_b );
            if(stor_scheme == 'C' || stor_scheme == 'c')
            {
                for(j = 0; j < group_size[i]; j++)
                {
                    bli_obj_create(dt, m0_a, n0_a, 1, lda[i], &a[idx]);
                    bli_obj_create(dt, m0_b, n0_b, 1, ldb[i], &b[idx]);
                    bli_obj_create(dt, m[i], n[i], 1, ldc[i], &c[idx]);
                    bli_obj_create(dt, m[i], n[i], 1, ldc[i], &c_save[idx]);

                    bli_randm( &a[idx] );
                    bli_randm( &b[idx] );
                    bli_randm( &c[idx] );

                    bli_obj_set_conjtrans(blis_transa, &a[idx]);
                    bli_obj_set_conjtrans(blis_transb, &b[idx]);
                    idx++;
                }
            }
            else if(stor_scheme == 'R' || stor_scheme == 'r')
            {
                for(j = 0; j < group_size[i]; j++)
                {
                    bli_obj_create(dt, m0_a, n0_a, lda[i], 1, &a[idx]);
                    bli_obj_create(dt, m0_b, n0_b, ldb[i], 1, &b[idx]);
                    bli_obj_create(dt, m[i], n[i], ldc[i], 1, &c[idx]);
                    bli_obj_create(dt, m[i], n[i], ldc[i], 1, &c_save[idx]);

                    bli_randm( &a[idx] );
                    bli_randm( &b[idx] );
                    bli_randm( &c[idx] );

                    bli_obj_set_conjtrans(blis_transa, &a[idx]);
                    bli_obj_set_conjtrans(blis_transb, &b[idx]);
                    idx++;
                }
            }
            f77_m[i] = m[i];
            f77_n[i] = n[i];
            f77_k[i] = k[i];
            f77_lda[i] = lda[i];
            f77_ldb[i] = ldb[i];
            f77_ldc[i] = ldc[i];
            f77_group_size[i] = group_size[i];

        }

        idx = 0;
         for(i = 0; i < GRP_COUNT; i++)
            for(j = 0; j < group_size[i]; j++)
            {
                bli_copym(&c[idx], &c_save[idx]);
                idx++;
            }

        dtime_save = DBL_MAX;

        for( r = 0; r < n_repeats; ++r )
        {
            idx = 0;
            for(i = 0; i < GRP_COUNT; i++)
                for(j = 0; j < group_size[i]; j++)
                {
                    bli_copym( &c_save[idx], &c[idx]);
                    idx++;
                }

            dtime = bli_clock();

#ifdef PRINT
        idx = 0;
        for(i = 0; i < GRP_COUNT; i++)
            for(j = 0; j < group_size[i]; j++)
                {
                    printf("Group: %ld Member: %ld\n", i, j);

                    bli_printm("a", &a[idx], "%4.1f", "");
                    bli_printm("b", &b[idx], "%4.1f", "");
                    bli_printm("c", &c[idx], "%4.1f", "");

                    idx++;
                }
#endif

        if(bli_is_float(dt))
        {
            const float *ap[total_count], *bp[total_count];
            float *cp[total_count];
            float alphap[GRP_COUNT], betap[GRP_COUNT];

            idx = 0;
            for(i = 0; i < GRP_COUNT; i++)
            {
                for(j = 0; j < group_size[i]; j++)
                {
                    ap[idx] = bli_obj_buffer( &a[idx] );
                    bp[idx] = bli_obj_buffer( &b[idx] );
                    cp[idx] = bli_obj_buffer( &c[idx] );

                    idx++;
                }
                alphap[i] = *(float*)bli_obj_buffer_for_1x1(dt, &alpha[i]);
                betap[i]  = *(float*)bli_obj_buffer_for_1x1(dt, &beta[i] );
            }

#ifdef CHECK_CBLAS
            cblas_sgemm_batch( cblas_order,
                   cblas_transa,
                   cblas_transb,
                   f77_m, f77_n, f77_k,
                   alphap, ap, f77_lda,
                   bp, f77_ldb,
                   betap, cp, f77_ldc,
                   f77_group_count,
                   f77_group_size
                );
#else
            sgemm_batch_( f77_transa,
                  f77_transb,
                  f77_m, f77_n, f77_k,
                  alphap, ap, f77_lda,
                  bp, f77_ldb,
                  betap, cp, f77_ldc,
                  &f77_group_count,
                  f77_group_size
                );
#endif

        }
        else if(bli_is_double(dt))
        {
            const double *ap[total_count], *bp[total_count];
            double *cp[total_count];
            double alphap[GRP_COUNT], betap[GRP_COUNT];

            idx = 0;
            for(i = 0; i < GRP_COUNT; i++)
            {
                for(j = 0; j < group_size[i]; j++)
                {
                    ap[idx] = bli_obj_buffer( &a[idx] );
                    bp[idx] = bli_obj_buffer( &b[idx] );
                    cp[idx] = bli_obj_buffer( &c[idx] );

                    idx++;
                }
                alphap[i] = *(double*)bli_obj_buffer_for_1x1(dt, &alpha[i]);
                betap[i]  = *(double*)bli_obj_buffer_for_1x1(dt, &beta[i] );
            }
#ifdef CHECK_CBLAS
            cblas_dgemm_batch( cblas_order,
                   cblas_transa,
                   cblas_transb,
                   f77_m, f77_n, f77_k,
                   alphap, ap, f77_lda,
                   bp, f77_ldb,
                   betap, cp, f77_ldc,
                   f77_group_count,
                   f77_group_size
                );
#else
            dgemm_batch_( f77_transa,
                  f77_transb,
                  f77_m, f77_n, f77_k,
                  alphap, ap, f77_lda,
                  bp, f77_ldb,
                  betap, cp, f77_ldc,
                  &f77_group_count,
                  f77_group_size
                );
#endif

        }
        else if(bli_is_scomplex(dt))
        {
            const scomplex *ap[total_count], *bp[total_count];
            scomplex *cp[total_count];
            scomplex alphap[GRP_COUNT], betap[GRP_COUNT];

            idx = 0;
            for(i = 0; i < GRP_COUNT; i++)
            {
                for(j = 0; j < group_size[i]; j++)
                {
                    ap[idx] = bli_obj_buffer( &a[idx] );
                    bp[idx] = bli_obj_buffer( &b[idx] );
                    cp[idx] = bli_obj_buffer( &c[idx] );

                    idx++;
                }
                alphap[i] = *(scomplex*)bli_obj_buffer_for_1x1(dt, &alpha[i]);
                betap[i]  = *(scomplex*)bli_obj_buffer_for_1x1(dt, &beta[i] );
            }
#ifdef CHECK_CBLAS
            cblas_cgemm_batch( cblas_order,
                   cblas_transa,
                   cblas_transb,
                   f77_m, f77_n, f77_k,
                   (const void*)alphap,
                   (const void**)ap, f77_lda,
                   (const void**)bp, f77_ldb,
                   (const void*)betap, (void**)cp, f77_ldc,
                   f77_group_count,
                   f77_group_size
                );
#else
            cgemm_batch_( f77_transa,
                  f77_transb,
                  f77_m, f77_n, f77_k,
                  alphap, ap, f77_lda,
                  bp, f77_ldb,
                  betap, cp, f77_ldc,
                  &f77_group_count,
                  f77_group_size
                );
#endif
        }
        else if(bli_is_dcomplex(dt))
        {
            const dcomplex *ap[total_count], *bp[total_count];
            dcomplex *cp[total_count];
            dcomplex alphap[GRP_COUNT], betap[GRP_COUNT];

            idx = 0;
            for(i = 0; i < GRP_COUNT; i++)
            {
                for(j = 0; j < group_size[i]; j++)
                {
                    ap[idx] = bli_obj_buffer( &a[idx] );
                    bp[idx] = bli_obj_buffer( &b[idx] );
                    cp[idx] = bli_obj_buffer( &c[idx] );

                    idx++;
                }
                alphap[i] = *(dcomplex*)bli_obj_buffer_for_1x1(dt, &alpha[i]);
                betap[i]  = *(dcomplex*)bli_obj_buffer_for_1x1(dt, &beta[i] );
            }

#ifdef CHECK_CBLAS
            cblas_zgemm_batch( cblas_order,
                   cblas_transa,
                   cblas_transb,
                   f77_m, f77_n, f77_k,
                   (const void*)alphap,
                   (const void**)ap, f77_lda,
                   (const void**)bp, f77_ldb,
                   (const void*)betap, (void**)cp, f77_ldc,
                   f77_group_count,
                   f77_group_size
                );
#else
            zgemm_batch_( f77_transa,
                  f77_transb,
                  f77_m, f77_n, f77_k,
                  alphap, ap, f77_lda,
                  bp, f77_ldb,
                  betap, cp, f77_ldc,
                  &f77_group_count,
                  f77_group_size
                );
#endif
        }
#ifdef PRINT
        idx = 0;
        for(i = 0; i < GRP_COUNT; i++)
            for(j = 0; j < group_size[i]; j++)
            {
                printf("Group: %ld Member: %ld\n", i, j);
                bli_printm("c after", &c[idx], "%4.1f", "");

                idx++;
            }
#endif
            dtime_save = bli_clock_min_diff( dtime_save, dtime );
        }

        dim_t fp_ops = 0;
        for(i = 0; i < GRP_COUNT; i++)
                fp_ops += 2.0 * m[i] * k[i] * n[i] * group_size[i];

        gflops = fp_ops / (dtime_save * 1.0e9 );

        if(bli_is_complex( dt ) ) gflops *= 4.0;

#ifdef FILE_IN_OUT
        fprintf(fout, "Stor_scheme = %c, group_count = %lu, gflops = %7.2f\n", stor_scheme, GRP_COUNT, gflops);
        for(i = 0; i < GRP_COUNT; i++)
            fprintf(fout, "%4lu \t %4lu\t %4lu\t %4lu\t %4lu\t %4lu\t %c\t %c\t %4lu\n", m[i], n[i], k[i], lda[i], ldb[i], ldc[i], transa[i], transb[i], group_size[i]);

        fflush(fout);
#else
        printf( "Stor_scheme = %c, group_count = %d, gflops = %7.2f\n", stor_scheme, GRP_COUNT, gflops);
        for(i = 0; i < GRP_COUNT; i++)
            printf("%4lu \t %4lu\t %4lu\t %4lu\t %4lu\t %4lu\t %c\t %c\t %4lu\n", m[i], n[i], k[i], lda[i], ldb[i], ldc[i], transa[i], transb[i], group_size[i]);

#endif

    idx = 0;
    for(i = 0; i < GRP_COUNT; i++)
    {
        bli_obj_free( &alpha[i]);
        bli_obj_free( &beta[i] );

        for(j = 0; j < group_size[i]; j++ )
        {
            bli_obj_free( &a[idx]);
            bli_obj_free( &b[idx]);
            bli_obj_free( &c[idx]);
            bli_obj_free( &c_save[idx]);

            idx++;
        }
    }
#ifdef FILE_IN_OUT
    }
    fclose(fin);
    fclose(fout);
#endif
    return 0;
}

