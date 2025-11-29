#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_drotmg.c
 *
 * The program is a C interface to drotmg.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
void cblas_drotmg( double *d1, double *d2, double *b1, 
                        const double b2, double *p)
{
   F77_drotmg(d1,d2,b1,&b2,p);
}
#endif
