#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_caxpby.c
 *
 * The program is a C interface to caxpby.
 *
 * Copyright (C) 2020, Advanced Micro Devices, Inc
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
void cblas_caxpby( f77_int N, const void *alpha,
               const void *X, f77_int incX,
               const void *beta,
               void *Y, f77_int incY)
{
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX, F77_incY=incY;
#else 
   #define F77_N N
   #define F77_incX incX
   #define F77_incY incY
#endif
   F77_caxpby( &F77_N, (scomplex*)alpha, (scomplex*)X, &F77_incX, (scomplex*)beta, (scomplex*)Y, &F77_incY);
} 
#endif
