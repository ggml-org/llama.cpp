#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_scnrm2.c
 *
 * The program is a C interface to scnrm2.
 * It calls the fortran wrapper before calling scnrm2.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
float cblas_scnrm2( f77_int N, const void *X, f77_int incX) 
{
   float nrm2;
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX;
#else 
   #define F77_N N
   #define F77_incX incX
#endif
   F77_scnrm2_sub( &F77_N, X, &F77_incX, &nrm2);
   return nrm2;
}
#endif
