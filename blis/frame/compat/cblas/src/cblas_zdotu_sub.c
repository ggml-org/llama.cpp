#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_zdotu_sub.c
 *
 * The program is a C interface to zdotu.
 * It calls the fortran wrapper before calling zdotu.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
void cblas_zdotu_sub( f77_int N, const void *X, f77_int incX,
                      const void *Y, f77_int incY, void *dotu)
{
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX, F77_incY=incY;
#else 
   #define F77_N N
   #define F77_incX incX
   #define F77_incY incY
#endif
   F77_zdotu_sub( &F77_N, (dcomplex*)X, &F77_incX, (dcomplex*)Y, &F77_incY, (dcomplex*)dotu);
   return;
}
#endif
