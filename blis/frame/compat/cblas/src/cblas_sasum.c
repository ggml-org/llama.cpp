#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_sasum.c
 *
 * The program is a C interface to sasum.
 * It calls the fortran wrapper before calling sasum.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
float cblas_sasum( f77_int N, const float *X, f77_int incX) 
{
   float asum;
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX;
#else 
   #define F77_N N
   #define F77_incX incX
#endif
   F77_sasum_sub( &F77_N, X, &F77_incX, &asum);
   return asum;
}
#endif
