#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_icamax.c
 *
 * The program is a C interface to icamax.
 * It calls the fortran wrapper before calling icamax.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
f77_int cblas_icamax( f77_int N, const void *X, f77_int incX)
{
   f77_int iamax;
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX;
#else 
   #define F77_N N
   #define F77_incX incX
#endif
   F77_icamax_sub( &F77_N, (scomplex*)X, &F77_incX, &iamax);
   return iamax ? iamax-1 : 0;
}
#endif
