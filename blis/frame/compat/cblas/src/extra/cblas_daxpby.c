#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_daxpby.c
 *
 * The program is a C interface to daxpby.
 *
 * Copyright (C) 2020, Advanced Micro Devices, Inc.
 */
#include "cblas.h"
#include "cblas_f77.h"
void cblas_daxpby( f77_int N, double alpha,
               const double *X, f77_int incX,
               double beta,
               double *Y, f77_int incY)
{
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX, F77_incY=incY;
#else
   #define F77_N N
   #define F77_incX incX
   #define F77_incY incY
#endif
   F77_daxpby( &F77_N, &alpha, X, &F77_incX, &beta, Y, &F77_incY);
}
#endif
