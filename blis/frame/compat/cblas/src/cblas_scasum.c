#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_scasum.c
 *
 * The program is a C interface to scasum.
 * It calls the fortran wrapper before calling scasum.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
float cblas_scasum( f77_int N, const void *X, f77_int incX) 
{
   float asum;
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX;
#else 
   #define F77_N N
   #define F77_incX incX
#endif
   F77_scasum_sub( &F77_N, X, &F77_incX, &asum);
   return asum;
}
#endif
