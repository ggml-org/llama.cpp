/****************************************************************
Copyright 1990 - 1997 by AT&T, Lucent Technologies and Bellcore.

Permission to use, copy, modify, and distribute this software
and its documentation for any purpose and without fee is hereby
granted, provided that the above copyright notice appear in all
copies and that both that the copyright notice and this
permission notice and warranty disclaimer appear in supporting
documentation, and that the names of AT&T, Bell Laboratories,
Lucent or Bellcore or any of their entities not be used in
advertising or publicity pertaining to distribution of the
software without specific, written prior permission.

AT&T, Lucent and Bellcore disclaim all warranties with regard to
this software, including all implied warranties of
merchantability and fitness.  In no event shall AT&T, Lucent or
Bellcore be liable for any special, indirect or consequential
damages or any damages whatsoever resulting from loss of use,
data or profits, whether in an action of contract, negligence or
other tortious action, arising out of or in connection with the
use or performance of this software.
****************************************************************/

#include "f2c.h"

#ifdef __cplusplus
extern "C" {
#endif
    /* Integer */
    shortint h_abs(const shortint *x)
    {
        return ( shortint )( *x >= 0 ? (*x) : (- *x) );
        //return ( shortint )abs( ( int )*x );
    }
    integer i_abs(const integer *x)
    {
        return ( integer )( *x >= 0 ? (*x) : (- *x) );
        //return ( integer )abs( ( int )*x );
    }

    /* Double */
    double r_abs(real *x)
    {
        return ( double )( *x >= 0 ? (*x) : (- *x) );
        //return ( double )fabsf( ( float )*x );
    }
    double d_abs(const doublereal *x)
    {
        return ( double )( *x >= 0 ? (*x) : (- *x) );
        //return ( double )fabs( ( double )*x );
    }

    /* Complex */
    double c_abs(const complex *z)
    {
        return ( double )hypot(z->r, z->i);
    }
    double z_abs(const doublecomplex *z)
    {
        return ( double )hypot(z->r, z->i);
    }

#ifdef __cplusplus
}
#endif
