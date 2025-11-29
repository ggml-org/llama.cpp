#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_zcopy.c
 *
 * The program is a C interface to zcopy.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
void cblas_zcopy( f77_int N, const void *X,
                      f77_int incX, void *Y, f77_int incY)
{
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX, F77_incY=incY;
#else 
   #define F77_N N
   #define F77_incX incX
   #define F77_incY incY
#endif
   F77_zcopy( &F77_N, (dcomplex*)X, &F77_incX, (dcomplex*)Y, &F77_incY);
}
#endif
