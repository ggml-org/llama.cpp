#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_drot.c
 *
 * The program is a C interface to drot.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
void cblas_drot(f77_int N, double *X, f77_int incX,
   double *Y, f77_int incY, const double c, const double s)
{
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX, F77_incY=incY;
#else
   #define F77_N N 
   #define F77_incX incX 
   #define F77_incY incY 
#endif
   F77_drot(&F77_N, X, &F77_incX, Y, &F77_incY, &c, &s);
   return;
}
#endif
