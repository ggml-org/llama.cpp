#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_sdsdot.c
 *
 * The program is a C interface to sdsdot.
 * It calls the fortran wrapper before calling sdsdot.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
float cblas_sdsdot( f77_int N, float alpha, const float *X,
                      f77_int incX, const float *Y, f77_int incY)
{
   float dot;
#ifdef F77_INT
   F77_INT F77_N=N, F77_incX=incX, F77_incY=incY;
#else 
   #define F77_N N
   #define F77_incX incX
   #define F77_incY incY
#endif
   F77_sdsdot_sub( &F77_N, &alpha, X, &F77_incX, Y, &F77_incY, &dot);
   return dot;
}   
#endif
