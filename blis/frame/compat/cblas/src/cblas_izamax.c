#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_izamax.c
 *
 * The program is a C interface to izamax.
 * It calls the fortran wrapper before calling izamax.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
f77_int cblas_izamax( f77_int N, const void *X, f77_int incX)
{
   f77_int iamax;
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX;
#else 
   #define F77_N N
   #define F77_incX incX
#endif
   F77_izamax_sub( &F77_N, (dcomplex*)X, &F77_incX, &iamax);
   return (iamax ? iamax-1 : 0);
}
#endif
