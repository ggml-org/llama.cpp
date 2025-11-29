#ifdef UTEST

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>           /* fabs */

#include "blis_utest.h"
#include "bli_gemm_power7_opt_8x4.h"

#define COLMAJ_INDEX(row,col,ld) ((col*ld)+row)
#define ROWMAJ_INDEX(row,col,ld) ((row*ld)+col)
#define BLIS_INDEX(row,col,rs,cs) ((row*rs)+(col*cs))

#define MR    BLIS_DEFAULT_MR_D
#define NR    BLIS_DEFAULT_NR_D
#define LDA   MR
#define LDB   NR   

#define EPSILON		0.0000001

/*
 * Perform
 *   c = beta * c + alpha * a * b
 * where
 *   alpha & beta are scalars
 *   c is mr x nr in blis-format, (col-stride & row-stride)
 *   a is mr x k in packed col-maj format (leading dim is mr)
 *   b is k x nr in packed row-maj format (leading dim is nr)
 */
void bli_dgemm_check(
                      dim_t              k,
                      double*   restrict alpha,
                      double*   restrict a,
                      double*   restrict b,
                      double*   restrict beta,
                      double*   restrict c, inc_t rs_c, inc_t cs_c,
                      auxinfo_t*         data
                    )
{
    int i, j, kk;
    double c00;

    for (i=0; i < MR; i++) {
        for (j=0; j < NR; j++) {
            c00 = c[BLIS_INDEX(i,j,rs_c,cs_c)] * *beta;
            for (kk=0; kk < k; kk++) 
                c00 += *alpha * (a[COLMAJ_INDEX(i,kk,LDA)] * b[ROWMAJ_INDEX(kk,j,LDB)]);
            c[BLIS_INDEX(i,j,rs_c,cs_c)] = c00;
        }
    }
}

int main(int argc, char *argv[])
{
    double         *A, *B, *C, *C2;
    double         alpha = 1.0, beta = 1.0;
    long           i, j;
    long           k = 128;
    int            iters = 10;
    int            errors;

    struct timeval tv_start, tv_end;

    switch (argc) {
        case 2:
            k = atoi(argv[1]);
        case 1:
            break;
        default:
            printf("Usage: %s [k]\n", argv[0]);
            return 1;
            break;
    }

    //long rs_c = 1, cs_c = MR;     // Column major
    long rs_c = NR, cs_c = 1;     // Row major

    A = (double*)malloc(LDA * k * sizeof(double));
    B = (double*)malloc(LDB * k * sizeof(double));

    C = (double*)malloc(MR * NR * sizeof(double));
    C2 = (double*)malloc(MR * NR * sizeof(double));
 
    /* Initialize C matrix in blis format */
    for (j=0; j<NR; j++)
        for (i=0; i<MR; i++)
            C2[BLIS_INDEX(i,j,rs_c,cs_c)] = C[BLIS_INDEX(i,j,rs_c,cs_c)] = drand48();

    /* Initialize A matrix in column major format */
    for (j=0; j<k; j++)
        for (i=0; i<MR; i++)
            A[COLMAJ_INDEX(i,j,LDA)] = drand48();

    /* Initialize B matrix in row major format */
    for (j=0; j<NR; j++)
        for (i=0; i<k; i++)
            B[ROWMAJ_INDEX(i,j,LDB)] = drand48();

    /* First check the results */

    bli_dgemm_opt_8x4(k, &alpha, A, B, &beta, C, rs_c, cs_c, NULL);

    bli_dgemm_check(k, &alpha, A, B, &beta, C2, rs_c, cs_c, NULL);

    for (i=0, errors=0; i<MR*NR-1; i++) {
        if (fabs(C[i] - C2[i]) > EPSILON) { 
            if (errors<20) printf(" %ld expected=%f got=%f\n", i, C2[i], C[i]);
            errors++;
        }
    }
    printf("Errors = %d\n", errors);

    if (errors) {
        return -1;
    }

    /* Now get the performance */

    gettimeofday(&tv_start, NULL);

    for (i=0; i<iters; i++) {
        bli_dgemm_opt_8x4(k, &alpha, A, B, &beta, C, rs_c, cs_c, NULL);
    }

    gettimeofday(&tv_end, NULL);

    float secs = (tv_end.tv_sec - tv_start.tv_sec) + (double)(tv_end.tv_usec - tv_start.tv_usec)/1E6;

    {
        float gflops = ((2.0*MR*NR*k*iters)/1E9)/secs;
        printf("%d %d %ld : GFLOPS = %6.3f\n", MR, NR, k, gflops);
    }

    return 0;
}

#endif
