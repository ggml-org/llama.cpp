#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_isamax.c
 *
 * The program is a C interface to isamax.
 * It calls the fortran wrapper before calling isamax.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
f77_int cblas_isamax( f77_int N, const float *X, f77_int incX)
{
   f77_int iamax;
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX;
#else 
   #define F77_N N
   #define F77_incX incX
#endif
   F77_isamax_sub( &F77_N, X, &F77_incX, &iamax);
   return iamax ? iamax-1 : 0;
}
#endif
