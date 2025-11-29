#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_saxpby.c
 *
 * The program is a C interface to saxpby.
 * It calls the fortran wrapper before calling saxpby.
 *
 * Copyright (C) 2020, Advanced Micro Devices, Inc.
 */

#include "cblas.h"
#include "cblas_f77.h"
void cblas_saxpby( f77_int N, float alpha, 
               const float *X, f77_int incX,
               float beta,
               float *Y, f77_int incY)
{
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX, F77_incY=incY;
#else
   #define F77_N N
   #define F77_incX incX
   #define F77_incY incY
#endif
   F77_saxpby( &F77_N, &alpha, X, &F77_incX, &beta, Y, &F77_incY);
}
#endif
