#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_dnrm2.c
 *
 * The program is a C interface to dnrm2.
 * It calls the fortranwrapper before calling dnrm2.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
double cblas_dnrm2( f77_int N, const double *X, f77_int incX) 
{
   double nrm2;
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX;
#else 
   #define F77_N N
   #define F77_incX incX
#endif
   F77_dnrm2_sub( &F77_N, X, &F77_incX, &nrm2);
   return nrm2;
}
#endif
