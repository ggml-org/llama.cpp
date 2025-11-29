#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 *
 * cblas_zgemm_batch.c
 * This program is a C interface to zgemm_batch.
 *
 * Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.
 *
 */

#include "cblas.h"
#include "cblas_f77.h"
void cblas_zgemm_batch(enum CBLAS_ORDER Order,
                       enum CBLAS_TRANSPOSE *TransA_array,
                       enum CBLAS_TRANSPOSE *TransB_array,
                       f77_int *M_array, f77_int *N_array,
                       f77_int *K_array, const void *alpha_array,
                       const void  **A_array, f77_int *lda_array,
                       const void  **B_array, f77_int *ldb_array,
                       const void *beta_array,
                       void **C_array, f77_int *ldc_array,
                       f77_int group_count, f77_int *group_size)
{
    char TA[group_count], TB[group_count];
#ifdef F77_CHAR
    F77_CHAR F77_TA[group_count], F77_TB[group_count];
#else
    #define F77_TA TA
    #define F77_TB TB
#endif

#ifdef F77_INT
    F77_INT F77_GRP_COUNT = group_count;
    F77_INT F77_M[F77_GRP_COUNT], F77_N[F77_GRP_COUNT], F77_K[F77_GRP_COUNT];
    F77_INT F77_lda[F77_GRP_COUNT], F77_ldb[F77_GRP_COUNT], F77_ldc[F77_GRP_COUNT];
    F77_INT F77_GRP_SIZE[F77_GRP_COUNT];
#else
    #define F77_GRP_COUNT group_count
    #define F77_M M_array
    #define F77_N N_array
    #define F77_K K_array
    #define F77_lda lda_array
    #define F77_ldb ldb_array
    #define F77_ldc ldc_array
    #define F77_GRP_SIZE group_size
#endif

    extern int CBLAS_CallFromC;
    extern int RowMajorStrg;
    RowMajorStrg = 0;
    CBLAS_CallFromC = 1;

    dim_t i;
    if( Order == CblasColMajor )
    {
        for(i = 0; i < group_count; i++)
        {
            if(TransA_array[i] == CblasTrans) TA[i]='T';
            else if ( TransA_array[i] == CblasConjTrans ) TA[i]='C';
            else if ( TransA_array[i] == CblasNoTrans )   TA[i]='N';
            else
            {
                cblas_xerbla(2, "cblas_zgemm_batch",
                       "Illegal TransA setting %d for group %d\n", TransA_array[i], i);
                CBLAS_CallFromC = 0;
                RowMajorStrg = 0;
                return;
            }

            if(TransB_array[i] == CblasTrans) TB[i]='T';
            else if ( TransB_array[i] == CblasConjTrans ) TB[i]='C';
            else if ( TransB_array[i] == CblasNoTrans )   TB[i]='N';
            else
            {
                cblas_xerbla(3, "cblas_zgemm_batch",
                       "Illegal TransB setting %d for group %d\n", TransB_array[i], i);
                CBLAS_CallFromC = 0;
                RowMajorStrg = 0;
                return;
            }

#ifdef F77_CHAR
            F77_TA[i] = C2F_CHAR(TA+i);
            F77_TB[i] = C2F_CHAR(TB+i);
#endif

#ifdef F77_INT
            F77_M[i] = M_array[i];
            F77_N[i] = N_array[i];
            F77_K[i] = K_array[i];
            F77_lda[i] = lda_array[i];
            F77_ldb[i] = ldb_array[i];
            F77_ldc[i] = ldc_array[i];
            F77_GRP_SIZE[i] = group_size[i];
#endif
    }

        F77_zgemm_batch(F77_TA, F77_TB,
                        F77_M, F77_N, F77_K,
                        (const dcomplex*)alpha_array,
                        (const dcomplex**)A_array, F77_lda,
                        (const dcomplex**)B_array, F77_ldb,
                        (const dcomplex*)beta_array,
                        (dcomplex**)C_array, F77_ldc,
                        &F77_GRP_COUNT, F77_GRP_SIZE);
    }
    else if (Order == CblasRowMajor)
    {
        RowMajorStrg = 1;
        dim_t i;

        for(i = 0; i < group_count; i++)
        {
            if(TransA_array[i] == CblasTrans) TB[i]='T';
            else if ( TransA_array[i] == CblasConjTrans ) TB[i]='C';
            else if ( TransA_array[i] == CblasNoTrans )   TB[i]='N';
            else
            {
                cblas_xerbla(2, "cblas_zgemm_batch",
                       "Illegal TransA setting %d for group %d\n", TransA_array[i], i);
                CBLAS_CallFromC = 0;
                RowMajorStrg = 0;
                return;
            }
            if(TransB_array[i] == CblasTrans) TA[i]='T';
            else if ( TransB_array[i] == CblasConjTrans ) TA[i]='C';
            else if ( TransB_array[i] == CblasNoTrans )   TA[i]='N';
            else
            {
                cblas_xerbla(2, "cblas_zgemm_batch",
                       "Illegal TransB setting %d for group %d\n", TransB_array[i], i);
                CBLAS_CallFromC = 0;
                RowMajorStrg = 0;
                return;
            }

#ifdef F77_CHAR
            F77_TA = C2F_CHAR(&TA);
            F77_TB = C2F_CHAR(&TB);
#endif

#ifdef F77_INT
            F77_M[i] = M_array[i];
            F77_N[i] = N_array[i];
            F77_K[i] = K_array[i];
            F77_lda[i] = lda_array[i];
            F77_ldb[i] = ldb_array[i];
            F77_ldc[i] = ldc_array[i];
            F77_GRP_SIZE = group_size[i];
#endif
        }

        F77_zgemm_batch(F77_TA, F77_TB,
                        F77_N, F77_M, F77_K,
                        (const dcomplex*)alpha_array,
                        (const dcomplex**)B_array, F77_ldb,
                        (const dcomplex**)A_array, F77_lda,
                        (const dcomplex*)beta_array,
                        (dcomplex**)C_array, F77_ldc,
                        &F77_GRP_COUNT, F77_GRP_SIZE);
   } else
     cblas_xerbla(1, "cblas_zgemm_batch",
                     "Illegal Order setting, %d\n", Order);
   CBLAS_CallFromC = 0;
   RowMajorStrg = 0;
}
#endif
