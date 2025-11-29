#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_dasum.c
 *
 * The program is a C interface to dasum.
 * It calls the fortran wrapper before calling dasum.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
double cblas_dasum( f77_int N, const double *X, f77_int incX) 
{
   double asum;
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX;
#else 
   #define F77_N N
   #define F77_incX incX
#endif
   F77_dasum_sub( &F77_N, X, &F77_incX, &asum);
   return asum;
}
#endif
